"""Agent graph definition module.

This module defines the data analysis graph using LangGraph. It implements a 
multi-step process that:

1. Retrieves database schema information
2. Generates a SQL query from natural language
3. Reflects on whether we have enough information
4. Either generates more queries or formats the answer
5. Returns the final analysis

The graph includes error handling, retry mechanisms, and a controlled execution
flow that prevents common failure modes in LLM-based systems.
"""

#==============================================================================
# IMPORTS
#==============================================================================
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing import Literal

from .utils.state import DataAnalysisState
from .utils.nodes import (
    get_schema_node,
    query_node,
    format_answer_node,
    submit_final_answer_node,
    save_node,
    reflect_node,
    MAX_ITERATIONS,
    retrieve_similar_selections_hybrid_search_node,
    rerank_node,
    relevant_selections_node,
    print__debug,
    rewrite_query_node,
    summarize_messages_node,
    retrieve_similar_chunks_hybrid_search_node,
    rerank_chunks_node,
    relevant_chunks_node
)

# Load environment variables
load_dotenv()

#==============================================================================
# GRAPH CREATION
#==============================================================================
def create_graph(checkpointer=None):
    """Create the graph for data analysis.
    
    This function constructs a directed graph representing the graph for
    data analysis tasks. The graph design follows several important principles:
    
    1. Clear separation of concerns between nodes
    2. Explicit error handling and recovery paths
    3. Controlled iteration with cycle prevention
    4. Checkpointing for execution resumption
    
    The resulting graph manages the complete process from natural language understanding
    to query execution and result formatting, with built-in safeguards against
    common failure modes.
    
    Args:
        checkpointer: Optional checkpointer instance. If None, defaults to InMemorySaver
                     for backward compatibility. In production, should use AsyncPostgresSaver.
    
    Returns:
        A compiled StateGraph ready for execution
    """
    # Initialize with our custom state type to track conversation and results
    graph = StateGraph(DataAnalysisState)

    #--------------------------------------------------------------------------
    # Add nodes - each handling a specific step in the process
    #--------------------------------------------------------------------------
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_similar_selections_hybrid_search", retrieve_similar_selections_hybrid_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("relevant_selections", relevant_selections_node)
    # New PDF chunk nodes - run in parallel with selection nodes
    graph.add_node("retrieve_similar_chunks_hybrid_search", retrieve_similar_chunks_hybrid_search_node)
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("relevant_chunks", relevant_chunks_node)
    graph.add_node("get_schema", get_schema_node)
    graph.add_node("query_gen", query_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("format_answer", format_answer_node)
    graph.add_node("submit_final_answer", submit_final_answer_node)
    graph.add_node("save", save_node)
    graph.add_node("summarize_messages_rewrite", summarize_messages_node)
    graph.add_node("summarize_messages_query", summarize_messages_node)
    graph.add_node("summarize_messages_reflect", summarize_messages_node)
    graph.add_node("summarize_messages_format", summarize_messages_node)

    #--------------------------------------------------------------------------
    # Define the graph execution path
    #--------------------------------------------------------------------------
    # Start: prompt -> rewrite_query -> summarize_messages -> retrieve (both selections and chunks in parallel)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "summarize_messages_rewrite")
    # After summarize_messages_rewrite, branch to both selection and chunk retrieval (parallel execution)
    graph.add_edge("summarize_messages_rewrite", "retrieve_similar_selections_hybrid_search")
    graph.add_edge("summarize_messages_rewrite", "retrieve_similar_chunks_hybrid_search")
    
    # Selection path: retrieve -> rerank -> relevant
    graph.add_edge("retrieve_similar_selections_hybrid_search", "rerank")
    graph.add_edge("rerank", "relevant_selections")
    
    # PDF chunk path: retrieve -> rerank -> relevant (runs in parallel)
    graph.add_edge("retrieve_similar_chunks_hybrid_search", "rerank_chunks")
    graph.add_edge("rerank_chunks", "relevant_chunks")

    # Add a synchronization node that both branches feed into
    def route_decision_node(state: DataAnalysisState) -> DataAnalysisState:
        """Synchronization node that waits for both selection and chunk processing to complete."""
        print__debug("SYNC: Both selection and chunk branches completed")
        return state  # Pass through state unchanged
    
    graph.add_node("route_decision", route_decision_node)
    
    # Both branches feed into the synchronization node
    graph.add_edge("relevant_selections", "route_decision")  
    graph.add_edge("relevant_chunks", "route_decision")
    
    # Single routing logic from the synchronization node
    def route_after_sync(state: DataAnalysisState):
        print__debug("ROUTING: Making decision after synchronization")
        # Check if we have selection codes to proceed with database queries
        if state.get("top_selection_codes") and len(state["top_selection_codes"]) > 0:
            print__debug("ROUTING: Found selections, proceeding to database schema")
            return "get_schema"
        elif state.get("chromadb_missing"):
            print("ERROR: ChromaDB directory is missing. Please unzip or create the ChromaDB at 'metadata/czsu_chromadb'.")
            return END
        else:
            # No database selections found - proceed directly to answer with available PDF chunks
            print("No relevant dataset selections found, proceeding with PDF chunks only")
            chunks_available = len(state.get("top_chunks", []))
            print__debug(f"ROUTING: Available PDF chunks: {chunks_available}")
            return "format_answer"
    
    graph.add_conditional_edges(
        "route_decision",
        route_after_sync,
        {
            "get_schema": "get_schema",  
            "format_answer": "format_answer",
            END: END
        }
    )

    # get_schema -> query_gen (no summarize_messages_schema)
    graph.add_edge("get_schema", "query_gen")

    # query_gen -> summarize_messages -> reflect/format_answer
    graph.add_edge("query_gen", "summarize_messages_query")
    def route_after_query(state: DataAnalysisState) -> Literal["reflect", "format_answer"]:
        print(f"Routing decision, iteration={state.get('iteration', 0)}")
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "format_answer"
        else:
            return "reflect"
    graph.add_conditional_edges(
        "summarize_messages_query",
        route_after_query,
        {
            "reflect": "reflect",
            "format_answer": "format_answer"
        }
    )

    # reflect -> summarize_messages -> query_gen/format_answer
    graph.add_edge("reflect", "summarize_messages_reflect")
    def route_after_reflect(state: DataAnalysisState) -> Literal["query_gen", "format_answer"]:
        decision = state.get("reflection_decision", "improve")
        if decision == "answer":
            return "format_answer"
        else:
            return "query_gen"
    graph.add_conditional_edges(
        "summarize_messages_reflect",
        route_after_reflect,
        {
            "query_gen": "query_gen",
            "format_answer": "format_answer"
        }
    )

    # format_answer -> summarize_messages -> submit_final_answer
    graph.add_edge("format_answer", "summarize_messages_format")
    graph.add_edge("summarize_messages_format", "submit_final_answer")
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", END)

    # Compile with checkpointing for execution persistence
    # This enables resuming interrupted runs and improves reliability
    if checkpointer is None:
        # Import here to avoid circular imports and provide fallback
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
        print("⚠ Using InMemorySaver fallback - consider using AsyncPostgresSaver for production")
    return graph.compile(checkpointer=checkpointer) 
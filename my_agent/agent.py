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
    rewrite_query_node,
    summarize_messages_node,
    retrieve_similar_chunks_hybrid_search_node,
    rerank_chunks_node,
    relevant_chunks_node
)

# Load environment variables
load_dotenv()

import os

def print__analysis_tracing_debug(msg: str) -> None:
    """Print analysis tracing debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    analysis_tracing_debug_mode = os.environ.get('print__analysis_tracing_debug', '0')
    if analysis_tracing_debug_mode == '1':
        print(f"[print__analysis_tracing_debug] 🔍 {msg}")
        import sys
        sys.stdout.flush()

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
    print__analysis_tracing_debug("84 - GRAPH CREATION START: Starting graph creation")
    
    # Initialize with our custom state type to track conversation and results
    graph = StateGraph(DataAnalysisState)
    print__analysis_tracing_debug("85 - STATE GRAPH INIT: StateGraph initialized with DataAnalysisState")

    #--------------------------------------------------------------------------
    # Add nodes - each handling a specific step in the process
    #--------------------------------------------------------------------------
    print__analysis_tracing_debug("86 - ADDING NODES: Adding all graph nodes")
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
    print__analysis_tracing_debug("87 - NODES ADDED: All 17 graph nodes added successfully")

    #--------------------------------------------------------------------------
    # Define the graph execution path
    #--------------------------------------------------------------------------
    print__analysis_tracing_debug("88 - ADDING EDGES: Defining graph execution path")
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
    print__analysis_tracing_debug("89 - PARALLEL EDGES: Added parallel processing edges for selections and chunks")

    # Add a synchronization node that both branches feed into
    def route_decision_node(state: DataAnalysisState) -> DataAnalysisState:
        """Synchronization node that waits for both selection and chunk processing to complete."""
        print__analysis_tracing_debug("90 - SYNC NODE: Both selection and chunk branches completed")
        return state  # Pass through state unchanged
    
    graph.add_node("route_decision", route_decision_node)
    print__analysis_tracing_debug("91 - SYNC NODE ADDED: Route decision synchronization node added")
    
    # Both branches feed into the synchronization node
    graph.add_edge("relevant_selections", "route_decision")  
    graph.add_edge("relevant_chunks", "route_decision")
    print__analysis_tracing_debug("92 - SYNC EDGES: Added edges to synchronization node")
    
    # Single routing logic from the synchronization node
    def route_after_sync(state: DataAnalysisState):
        print__analysis_tracing_debug("93 - ROUTING DECISION: Making routing decision after synchronization")
        # Check if we have selection codes to proceed with database queries
        if state.get("top_selection_codes") and len(state["top_selection_codes"]) > 0:
            print__analysis_tracing_debug(f"94 - SCHEMA ROUTE: Found {len(state['top_selection_codes'])} selections, proceeding to database schema")
            return "get_schema"
        elif state.get("chromadb_missing"):
            print("❌ ERROR: ChromaDB directory is missing. Please unzip or create the ChromaDB at 'metadata/czsu_chromadb'.")
            print__analysis_tracing_debug("95 - CHROMADB ERROR: ChromaDB directory missing, ending execution")
            return END
        else:
            # No database selections found - proceed directly to answer with available PDF chunks
            print("⚠️ No relevant dataset selections found, proceeding with PDF chunks only")
            chunks_available = len(state.get("top_chunks", []))
            print__analysis_tracing_debug(f"96 - CHUNKS ONLY ROUTE: No selections found, proceeding with {chunks_available} PDF chunks")
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
    print__analysis_tracing_debug("97 - CONDITIONAL EDGES: Added conditional routing edges after synchronization")

    # get_schema -> query_gen (no summarize_messages_schema)
    graph.add_edge("get_schema", "query_gen")

    # query_gen -> summarize_messages -> reflect/format_answer
    graph.add_edge("query_gen", "summarize_messages_query")
    def route_after_query(state: DataAnalysisState) -> Literal["reflect", "format_answer"]:
        iteration = state.get('iteration', 0)
        print(f"🔀 Routing decision, iteration={iteration}")
        print__analysis_tracing_debug(f"98 - QUERY ROUTING: Making routing decision after query, iteration={iteration}")
        if iteration >= MAX_ITERATIONS:
            print__analysis_tracing_debug(f"99 - MAX ITERATIONS: Reached max iterations ({MAX_ITERATIONS}), proceeding to format answer")
            return "format_answer"
        else:
            print__analysis_tracing_debug(f"100 - REFLECT ROUTE: Iteration {iteration} < {MAX_ITERATIONS}, proceeding to reflect")
            return "reflect"
    graph.add_conditional_edges(
        "summarize_messages_query",
        route_after_query,
        {
            "reflect": "reflect",
            "format_answer": "format_answer"
        }
    )
    print__analysis_tracing_debug("101 - QUERY CONDITIONAL: Added conditional edges after query generation")

    # reflect -> summarize_messages -> query_gen/format_answer
    graph.add_edge("reflect", "summarize_messages_reflect")
    def route_after_reflect(state: DataAnalysisState) -> Literal["query_gen", "format_answer"]:
        decision = state.get("reflection_decision", "improve")
        print__analysis_tracing_debug(f"102 - REFLECT ROUTING: Reflection decision is '{decision}'")
        if decision == "answer":
            print__analysis_tracing_debug("103 - ANSWER ROUTE: Reflection says answer is ready, proceeding to format")
            return "format_answer"
        else:
            print__analysis_tracing_debug("104 - IMPROVE ROUTE: Reflection says improve needed, going back to query generation")
            return "query_gen"
    graph.add_conditional_edges(
        "summarize_messages_reflect",
        route_after_reflect,
        {
            "query_gen": "query_gen",
            "format_answer": "format_answer"
        }
    )
    print__analysis_tracing_debug("105 - REFLECT CONDITIONAL: Added conditional edges after reflection")

    # format_answer -> summarize_messages -> submit_final_answer
    graph.add_edge("format_answer", "summarize_messages_format")
    graph.add_edge("summarize_messages_format", "submit_final_answer")
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", END)
    print__analysis_tracing_debug("106 - FINAL EDGES: Added final edges to completion")

    print__analysis_tracing_debug("107 - GRAPH COMPILATION: Starting graph compilation with checkpointer")
    # Compile with SELECTIVE checkpointing - only checkpoint after 'save' node
    # This dramatically reduces database storage from 15+ records to just 2 records per analysis
    if checkpointer is None:
        print__analysis_tracing_debug("108 - INMEMORY SAVER: No checkpointer provided, using InMemorySaver")
        # Import here to avoid circular imports and provide fallback
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
        print("⚠️ Using InMemorySaver fallback - consider using AsyncPostgresSaver for production")
        print__analysis_tracing_debug("109 - INMEMORY CREATED: InMemorySaver fallback created")
    else:
        print__analysis_tracing_debug(f"110 - CHECKPOINTER PROVIDED: Using provided checkpointer ({type(checkpointer).__name__})")
    
    # OPTIMAL SOLUTION: Use interrupt_after to only checkpoint when analysis is complete
    # This reduces checkpoint storage from 15+ records to just 2 records per analysis
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["submit_final_answer"]  # Only checkpoint after the submit_final_answer node completes
    )
    print__analysis_tracing_debug("111 - GRAPH COMPILED: Graph successfully compiled with selective checkpointing (interrupt_after=['save'])")
    return compiled_graph 
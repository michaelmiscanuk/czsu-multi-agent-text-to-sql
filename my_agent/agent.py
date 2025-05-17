"""Agent graph graph definition module.

This module defines the data analysis graph using LangGraph. It implements a 
multi-step process that:

1. Retrieves database schema information
2. Generates a pandas query from natural language
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing import Literal

from .utils.state import DataAnalysisState
from .utils.nodes import (
    get_schema_node,
    query_node,
    format_answer_node,
    submit_final_answer_node,
    save_node,
    reflect_node,
    increment_iteration_node,
    MAX_ITERATIONS
)
from .utils.tools import PandasQueryTool

# Load environment variables
load_dotenv()

#==============================================================================
# GRAPH CREATION
#==============================================================================
def create_graph():
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
    
    Returns:
        A compiled StateGraph ready for execution
    """
    # Initialize with our custom state type to track conversation and results
    graph = StateGraph(DataAnalysisState)

    #--------------------------------------------------------------------------
    # Add nodes - each handling a specific step in the process
    #--------------------------------------------------------------------------
    # Schema retrieval provides context about available data
    graph.add_node("get_schema", get_schema_node)
    
    # Query generation converts natural language to executable code
    graph.add_node("query_gen", query_node)
    
    # Reflection on current state and feedback
    graph.add_node("reflect", reflect_node)
    
    # Iteration management
    graph.add_node("increment_iteration", increment_iteration_node)
    
    # Natural language formatting of query results
    graph.add_node("format_answer", format_answer_node)
    
    # Final answer formatting creates user-friendly responses
    graph.add_node("submit_final_answer", submit_final_answer_node)
    
    # Result persistence ensures we don't lose completed analyses
    graph.add_node("save", save_node)

    #--------------------------------------------------------------------------
    # Define the graph execution path
    #--------------------------------------------------------------------------
    # Start by loading the schema to understand available data
    graph.add_edge(START, "get_schema")
    graph.add_edge("get_schema", "query_gen")
    
    # After query generation, decide whether to reflect or format answer
    def route_after_query(state: DataAnalysisState) -> Literal["reflect", "format_answer"]:
        print(f"Routing decision, iteration={state.get('iteration', 0)}")
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            print("Routing to format_answer")
            return "format_answer"
        else:
            print("Routing to reflect")
            return "reflect"

    
    graph.add_conditional_edges(
        "query_gen",
        route_after_query,
        {
            "reflect": "reflect",
            "format_answer": "format_answer"
        }
    )

    
    # After reflection, always increment iteration and go back to query generation
    graph.add_edge("reflect", "increment_iteration")
    graph.add_edge("increment_iteration", "query_gen")
    
    # Continue with final answer formatting
    graph.add_edge("format_answer", "submit_final_answer")
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", END)

    # Compile with memory-based checkpointing for execution persistence
    # This enables resuming interrupted runs and improves reliability
    return graph.compile(checkpointer=MemorySaver())

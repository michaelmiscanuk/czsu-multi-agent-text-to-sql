"""Agent graph graph definition module.

This module defines the data analysis graph using LangGraph. It implements a 
multi-step process that:

1. Retrieves database schema information
2. Generates a pandas query from natural language
3. Validates and corrects the query
4. Executes the query against the dataset
5. Formats and returns the results

The graph includes error handling, retry mechanisms, and a controlled execution
flow that prevents common failure modes in LLM-based systems.
"""

#==============================================================================
# IMPORTS
#==============================================================================
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .utils import DataAnalysisState, save_node
from .utils.nodes import (
    get_schema_node,
    query_gen_node,
    check_query_node,
    execute_query_node,
    submit_final_answer_node,
    should_continue,
)

# Load environment variables
load_dotenv()

#==============================================================================
# GRAPH CREATION
#==============================================================================
def create_graph():
    """Create the graph graph for data analysis.
    
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
    graph.add_node("query_gen", query_gen_node)
    
    # Query correction handles validation and fixes common errors
    graph.add_node("correct_query", check_query_node)
    
    # Query execution runs the generated code against the dataset
    graph.add_node("execute_query", execute_query_node)
    
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

    # After generating a query, decide whether to execute it or get more info
    # This conditional routing is crucial for handling complex queries
    graph.add_conditional_edges("query_gen", should_continue)

    # After correction, always proceed to execution
    graph.add_edge("correct_query", "execute_query")

    # After execution, either submit the answer or fix errors
    # This creates a correction loop with built-in cycle prevention
    graph.add_conditional_edges("execute_query", should_continue)

    # Final steps to save the result and complete the graph
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", END)

    # Compile with memory-based checkpointing for execution persistence
    # This enables resuming interrupted runs and improves reliability
    return graph.compile(checkpointer=MemorySaver())

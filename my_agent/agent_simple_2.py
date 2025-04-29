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
from .utils.nodes_2 import (
    get_schema_node,
    query_gen_node,
    execute_query_node,
    submit_final_answer_node,
)

# Load environment variables
load_dotenv()

#==============================================================================
# NODE CONSTANTS
#==============================================================================
GET_SCHEMA = "get_schema"
QUERY_GEN = "query_gen"
CORRECT_QUERY = "correct_query"
EXECUTE_QUERY = "execute_query"
SUBMIT_FINAL_ANSWER = "submit_final_answer"
SAVE = "save"

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
    graph.add_node(GET_SCHEMA, get_schema_node)
    
    # Query generation converts natural language to executable code
    graph.add_node(QUERY_GEN, query_gen_node)
       
    # Query execution runs the generated code against the dataset
    graph.add_node(EXECUTE_QUERY, execute_query_node)
    
    # Final answer formatting creates user-friendly responses
    graph.add_node(SUBMIT_FINAL_ANSWER, submit_final_answer_node)
    
    # Result persistence ensures we don't lose completed analyses
    graph.add_node(SAVE, save_node)

    #--------------------------------------------------------------------------
    # Define the graph execution path - simplified linear flow
    #--------------------------------------------------------------------------
    # Create a simple linear flow between all nodes
    graph.add_edge(START, GET_SCHEMA)
    graph.add_edge(GET_SCHEMA, QUERY_GEN)
    graph.add_edge(QUERY_GEN, EXECUTE_QUERY)
    graph.add_edge(EXECUTE_QUERY, SUBMIT_FINAL_ANSWER)
    graph.add_edge(SUBMIT_FINAL_ANSWER, SAVE)
    graph.add_edge(SAVE, END)

    # Compile with memory-based checkpointing for execution persistence
    # This enables resuming interrupted runs and improves reliability
    return graph.compile(checkpointer=MemorySaver())

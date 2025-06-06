"""Main entry point for the data analysis application.

This module provides the main execution logic for the data analysis
application, handling command line arguments and invoking the LangGraph workflow.

The system is designed to support both interactive use (as a library) and 
command-line execution with configurable prompts, making it versatile for 
different deployment scenarios.
"""

#==============================================================================
# IMPORTS
#==============================================================================
import uuid
import asyncio
import argparse
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
import os
import sys

# Load environment variables
load_dotenv()

from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.nodes import MAX_ITERATIONS

# Robust BASE_DIR logic for project root
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
# Default prompt if none provided
# DEFAULT_PROMPT = "Did Prague have more residents than Central Bohemia at the start of 2024?"
# DEFAULT_PROMPT = "Can you compare number of man and number of woman in prague and in plzen? Create me a bar chart with this data."
# DEFAULT_PROMPT = "How much did Prague's population grow from start to end of Q3?"
# DEFAULT_PROMPT = "What was South Bohemia's population change rate per month?"
# DEFAULT_PROMPT = "Tell me a joke"
# DEFAULT_PROMPT = "Is there some very interesting trend in my data?"
# DEFAULT_PROMPT = "tell me about how many people were in prague at 2024 and compare it with whole republic data? Pak mi dej distribuci kazdeho regionu, v procentech."
# DEFAULT_PROMPT = "tell me about people in prague, compare, contrast, what is interesting, provide trends."
# DEFAULT_PROMPT = "What was the maximum female population recorded in any region?"
# DEFAULT_PROMPT = "List regions where the absolute difference between male and female population changes was greater than 3000, and indicate whether men or women changed more"
# DEFAULT_PROMPT = "What is the average population rate of change for regions with more than 1 million residents?"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy v Praze"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy?"
# DEFAULT_PROMPT = """
# This table contains information about wages and salaries across different industries. It includes data on average wages categorized by economic sectors or industries.

# Available columns:

# industry (odvětví): distinct values include manufacturing, IT, construction, healthcare, education, etc.

# average_wage (průměrná mzda): numerical values representing monthly or annual averages

# year: distinct values may include 2020, 2021, 2022, etc.

# measurement_unit: e.g., CZK, EUR, USD per month/year

# The table allows comparison of wage levels across different economic sectors.
# """
DEFAULT_PROMPT = "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?"
# DEFAULT_PROMPT = "Jaký byl podíl osob používajících internet v Česku ve věku 16 a vice v roce 2023?"

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def extract_table_names_from_sql(sql_query: str) -> List[str]:
    """Extract table names from SQL query FROM clauses.
    
    Args:
        sql_query: The SQL query string
        
    Returns:
        List of table names found in FROM clauses
    """
    # Remove comments and normalize whitespace
    sql_clean = re.sub(r'--.*?(?=\n|$)', '', sql_query, flags=re.MULTILINE)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    sql_clean = ' '.join(sql_clean.split())
    
    # Pattern to match FROM clause with table names
    # This handles: FROM table_name, FROM schema.table_name, FROM "table_name", etc.
    from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'
    
    table_names = []
    matches = re.finditer(from_pattern, sql_clean, re.IGNORECASE)
    
    for match in matches:
        # Extract the main table name (group 2)
        if match.group(2):
            table_names.append(match.group(2).upper())
        # Extract additional table names if comma-separated (group 4)
        if match.group(4):
            table_names.append(match.group(4).upper())
    
    # Also handle JOIN clauses
    join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
    join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)
    
    for match in join_matches:
        if match.group(2):
            table_names.append(match.group(2).upper())
    
    return list(set(table_names))  # Remove duplicates

def get_used_selection_codes(queries_and_results: list, top_selection_codes: List[str]) -> List[str]:
    """Filter top_selection_codes to only include those actually used in queries.
    
    Args:
        queries_and_results: List of (query, result) tuples
        top_selection_codes: List of all candidate selection codes
        
    Returns:
        List of selection codes that were actually used in the queries
    """
    if not queries_and_results or not top_selection_codes:
        return []
    
    # Extract all table names used in queries
    used_table_names = set()
    for query, _ in queries_and_results:
        if query:
            table_names = extract_table_names_from_sql(query)
            used_table_names.update(table_names)
    
    # Filter selection codes to only include those that match used table names
    used_selection_codes = []
    for selection_code in top_selection_codes:
        if selection_code.upper() in used_table_names:
            used_selection_codes.append(selection_code)
    
    return used_selection_codes

#==============================================================================
# MAIN FUNCTION
#==============================================================================
async def main(prompt=None, thread_id=None, checkpointer=None):
    """Main entry point for the application.
    
    This async function serves as the central coordinator for the data analysis process.
    It handles prompt acquisition from different sources (function parameter,
    command line, or default), initializes tracing for observability, and
    executes the LangGraph workflow. A thread ID is generated to allow
    tracking of each analysis run independently.
    
    Args:
        prompt (str, optional): The analysis prompt to process. If None and script is run
                               directly, will attempt to get from command line args.
        thread_id (str, optional): The conversation thread ID for memory. If None and script is run
                                   directly, a new thread ID will be generated.
        checkpointer (optional): External checkpointer instance for shared memory. If None,
                                creates a new InMemorySaver instance.
    
    Returns:
        dict: A dictionary containing the prompt, result, and thread_id for downstream
              processing or API responses.
    """
    # Handle prompt sourcing - command line args have priority over defaults
    # This allows flexibility in how the application is used
    if prompt is None and __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run data analysis with LangGraph')
        parser.add_argument('prompt', nargs='?', default=DEFAULT_PROMPT,
                           help=f'Analysis prompt (default: "{DEFAULT_PROMPT}")')
        parser.add_argument('--thread_id', type=str, default=None, help='Conversation thread ID for memory')
        args = parser.parse_args()
        prompt = args.prompt
        thread_id = args.thread_id
    
    # Ensure we always have a valid prompt to avoid None-type errors downstream
    if prompt is None:
        prompt = DEFAULT_PROMPT
        
    # Use a thread_id for short-term memory (thread-level persistence)
    if thread_id is None:
        thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
    
    # Initialize tracing for debugging and performance monitoring
    # This is crucial for production deployments to track execution paths
    # instrument(project_name="LangGraph_czsu-multi-agent-text-to-sql", framework=Framework.LANGGRAPH)
    
    # Create the LangGraph execution graph with InMemorySaver for persistent short-term memory
    if checkpointer is None:
        checkpointer = InMemorySaver()
    graph = create_graph(checkpointer=checkpointer)
        
    print(f"Processing prompt: {prompt} (thread_id={thread_id})")
    
    # Configuration for thread-level persistence
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if there's existing state for this thread to determine if this is a new or continuing conversation
    try:
        existing_state = graph.get_state(config)
        is_continuing_conversation = (
            existing_state and 
            existing_state.values and 
            existing_state.values.get("messages") and 
            len(existing_state.values.get("messages", [])) > 0
        )
    except Exception:
        is_continuing_conversation = False
    
    # Prepare input state based on whether this is a new or continuing conversation
    if is_continuing_conversation:
        # For continuing conversations, pass only the fields that need to be updated
        # The checkpointer will merge this with the existing state
        input_state = {
            "prompt": prompt,
            "rewritten_prompt": None,
            "iteration": 0,  # Reset for new question
        }
    else:
        # For new conversations, initialize with complete state
        input_state = {
            "prompt": prompt,
            "rewritten_prompt": None,
            "rewritten_prompt_history": [],
            "messages": [SystemMessage(content=""), AIMessage(content="")],  # Initialize for new conversation
            "iteration": 0,
            "queries_and_results": [],
            "chromadb_missing": False
        }
    
    # Execute the graph with checkpoint configuration asynchronously
    # Checkpoints allow resuming execution if interrupted and maintaining conversation memory
    result = await graph.ainvoke(
        input_state,
        config=config
    )
        
    # Extract values from the graph result dictionary         
    # The graph now uses a messages list: [summary (SystemMessage), last_message (AIMessage)]
    queries_and_results = result["queries_and_results"]
    final_answer = result["messages"][-1].content if result.get("messages") and len(result["messages"]) > 1 else ""

    # Use top_selection_codes for dataset reference (use first if available)
    top_selection_codes = result.get("top_selection_codes", [])
    sql_query = queries_and_results[-1][0] if queries_and_results else None
    
    # Filter to only include selection codes actually used in queries
    used_selection_codes = get_used_selection_codes(queries_and_results, top_selection_codes)
    
    dataset_url = None
    if used_selection_codes:
        dataset_url = f"/datasets/{used_selection_codes[0]}"

    # Convert the result to a JSON-serializable format
    serializable_result = {
        "prompt": prompt,
        "result": final_answer,
        "queries_and_results": queries_and_results,
        "thread_id": thread_id,
        "top_selection_codes": used_selection_codes,  # Return only codes actually used in queries
        "iteration": result.get("iteration", 0),
        "max_iterations": MAX_ITERATIONS,
        "sql": sql_query,
        "datasetUrl": dataset_url
    }
        
    print(f"Result: {final_answer}")
    return serializable_result

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":
    asyncio.run(main())

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
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from my_agent.utils.instrument import instrument, Framework
import os
import sys
from urllib.parse import quote

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
# MAIN FUNCTION
#==============================================================================
async def main(prompt=None, thread_id=None):
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
    
    # Create the LangGraph execution graph with AsyncSqliteSaver for persistent short-term memory
    print("[DEBUG] BASE_DIR:", BASE_DIR)
    DB_PATH = BASE_DIR / "data" / "langgraph_checkpoints.db"
    print("[DEBUG] DB_PATH:", DB_PATH)
    print("[DEBUG] data dir exists:", (BASE_DIR / "data").exists())
    print("[DEBUG] DB file exists:", DB_PATH.exists())
    conn_str = str(DB_PATH)
    print("[DEBUG] DB file path (not URI):", conn_str)
    async with AsyncSqliteSaver.from_conn_string(conn_str) as checkpointer:
        await checkpointer.setup()
        graph = create_graph(checkpointer=checkpointer)
        
        print(f"Processing prompt: {prompt} (thread_id={thread_id})")
        
        # Retrieve previous messages for this thread (short-term memory)
        # For InMemorySaver, this is handled by LangGraph automatically when using the same thread_id
        # So we just need to pass the thread_id in config

        # Initial state: messages is always a two-item list: [SystemMessage (summary), AIMessage (last_message)].
        # This is a placeholder; the workflow will update it to always keep only the summary and the latest message.
        initial_state: DataAnalysisState = {
            "prompt": prompt,
            "rewritten_prompt": None,
            "rewritten_prompt_history": [],
            "messages": [SystemMessage(content=""), AIMessage(content="")],
            "iteration": 0,
            "queries_and_results": [],
            "chromadb_missing": False
        }
        
        # Execute the graph with checkpoint configuration asynchronously
        # Checkpoints allow resuming execution if interrupted
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract values from the graph result dictionary         
        # The graph now uses a messages list: [summary (SystemMessage), last_message (AIMessage)]
        queries_and_results = result["queries_and_results"]
        final_answer = result["messages"][-1].content if result.get("messages") and len(result["messages"]) > 1 else ""
        selection_with_possible_answer = result.get("selection_with_possible_answer")

        # Extract SQL from the last query, if available
        sql_query = queries_and_results[-1][0] if queries_and_results else None
        # Construct dataset URL (customize as needed)
        dataset_url = None
        if selection_with_possible_answer:
            dataset_url = f"/datasets/{selection_with_possible_answer}"

        # Convert the result to a JSON-serializable format
        serializable_result = {
            "prompt": prompt,
            "result": final_answer,
            "queries_and_results": queries_and_results,
            "thread_id": thread_id,
            "selection_with_possible_answer": selection_with_possible_answer,
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

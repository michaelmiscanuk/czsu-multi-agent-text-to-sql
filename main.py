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
from langchain_core.messages import BaseMessage
# from my_agent.utils.instrument import instrument, Framework


# Load environment variables
load_dotenv()

from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState

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
async def main(prompt=None):
    """Main entry point for the application.
    
    This async function serves as the central coordinator for the data analysis process.
    It handles prompt acquisition from different sources (function parameter,
    command line, or default), initializes tracing for observability, and
    executes the LangGraph workflow. A thread ID is generated to allow
    tracking of each analysis run independently.
    
    Args:
        prompt (str, optional): The analysis prompt to process. If None and script is run
                               directly, will attempt to get from command line args.
    
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
        args = parser.parse_args()
        prompt = args.prompt
    
    # Ensure we always have a valid prompt to avoid None-type errors downstream
    if prompt is None:
        prompt = DEFAULT_PROMPT
        
    # Initialize tracing for debugging and performance monitoring
    # This is crucial for production deployments to track execution paths
    # instrument(project_name="LangGraph_CZSU_Multi_Agent_App", framework=Framework.LANGGRAPH)
    
    # Create the LangGraph execution graph - this defines our workflow steps
    graph = create_graph()
    
    print(f"Processing prompt: {prompt}")
    
    # Create initial state with the user's prompt
    initial_state: DataAnalysisState = {
        "prompt": prompt,
        "messages": [],
        "iteration": 0,
        "queries_and_results": [],
        "chromadb_missing": False
    }
    
    # Generate a unique thread ID to track this specific analysis run
    # This is important for concurrent executions and audit trails
    thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
    
    # Execute the graph with checkpoint configuration asynchronously
    # Checkpoints allow resuming execution if interrupted
    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Extract values from the graph result dictionary         
    messages = result["messages"]
    queries_and_results = result["queries_and_results"]
    final_answer = messages[-1].content if messages else ""
    
    # Convert the result to a JSON-serializable format
    serializable_result = {
        "prompt": prompt,
        "result": final_answer,
        "queries_and_results": queries_and_results,
        "thread_id": thread_id
    }
    
    print(f"Result: {final_answer}")
    return serializable_result

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":
    asyncio.run(main())

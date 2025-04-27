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
import argparse
from pathlib import Path
from dotenv import load_dotenv
from my_agent.utils.instrument import instrument, Framework

# Load environment variables
load_dotenv()

from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
# Default prompt if none provided
DEFAULT_PROMPT = "What is the amount of men in Prague at the end of Q3 2024?"

#==============================================================================
# MAIN FUNCTION
#==============================================================================
def main(prompt=None):
    """Main entry point for the application.
    
    This function serves as the central coordinator for the data analysis process.
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
    instrument(project_name="LangGraph_Prototype4", framework=Framework.LANGGRAPH)
    
    # Create the LangGraph execution graph - this defines our workflow steps
    graph = create_graph()
    
    print(f"Processing prompt: {prompt}")
    
    # Create initial state with the user's prompt
    # The DataAnalysisState holds all contextual information during processing
    initial_state = DataAnalysisState(prompt=prompt)
    
    # Generate a unique thread ID to track this specific analysis run
    # This is important for concurrent executions and audit trails
    thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
    
    # Execute the graph with checkpoint configuration
    # Checkpoints allow resuming execution if interrupted
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Convert the result to a JSON-serializable format for API responses
    # This ensures consistent output structure regardless of internal processing
    serializable_result = {
        "prompt": prompt,
        "result": result['result'],
        "thread_id": thread_id
    }
    
    print(f"Result: {serializable_result['result']}")
    return serializable_result

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":
    main()

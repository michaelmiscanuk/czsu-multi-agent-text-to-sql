import os
import uuid
import argparse
import json
from dotenv import load_dotenv
from utils.instrument import instrument, Framework

# Load environment variables
load_dotenv()

# Import after setting up environment
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState

def main(prompt=None):
    """Main entry point for the application.
    
    Args:
        prompt (str, optional): The analysis prompt to process. If None and script is run
                               directly, will attempt to get from command line args.
    """
    # If no prompt provided and script is run directly, parse command line arguments
    if prompt is None and __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run data analysis with LangGraph')
        parser.add_argument('prompt_pos', nargs='?', type=str, 
                            help='The analysis prompt as positional argument')
        parser.add_argument('--prompt', type=str, 
                            help='The analysis prompt with flag')
        args = parser.parse_args()
        
        # Prioritize flag over positional argument
        prompt = args.prompt or args.prompt_pos
    
    # Use default if prompt is still None
    if prompt is None:
        prompt = "What is the amount of men in Prague at the end of Q3 2024?"
        
    # Initialize tracing
    instrument(project_name="LangGraph_Prototype4", framework=Framework.LANGGRAPH)
    
    # Create the graph
    graph = create_graph()
    
    print(f"Prompt: {prompt}")
    
    # Create initial state
    initial_state = DataAnalysisState(prompt=prompt)
    
    # Create a unique thread ID for this analysis run
    thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
    
    # Run the graph with checkpoint configuration
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # Ensure the result is JSON serializable
    serializable_result = {
        "prompt": prompt,
        "result": result['result'],
        "thread_id": thread_id
    }
    
    print(f"Result: {serializable_result['result']}")
    return serializable_result

if __name__ == "__main__":
    main()

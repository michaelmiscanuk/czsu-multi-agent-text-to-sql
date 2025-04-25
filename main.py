import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Phoenix tracing as shown in the documentation
from phoenix.otel import register

# Configure Phoenix tracer
print("Initializing Phoenix tracing...")
try:
    tracer_provider = register(
        project_name="LangGraph_Prototype4",
        auto_instrument=True
    )
    print("✅ Phoenix tracing initialized")
except Exception as e:
    print(f"⚠️ Phoenix tracing initialization failed: {str(e)}")

# Import after tracing is configured
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState

def main():
    """Main entry point for the application."""
    # Create the graph
    graph = create_graph()
    
    # Set the query from environment variables or use default
    query = os.getenv("ANALYSIS_PROMPT", "What is the amount of men in Prague at the end of Q3 2024?")
    print(f"Query: {query}")
    
    # Create an empty initial state
    initial_state = DataAnalysisState()
    
    # Run the graph with checkpoint configuration
    result = graph.invoke(
        initial_state, 
        config={"configurable": {"thread_id": "data_analysis_thread"}}
    )
    
    # Print the result - properly accessing the dictionary-like object
    print(f"Result: {result['result']}")

if __name__ == "__main__":
    main()

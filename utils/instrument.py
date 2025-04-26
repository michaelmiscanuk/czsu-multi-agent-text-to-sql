import os
from enum import Enum
from dotenv import load_dotenv
from opentelemetry import trace

# Load environment variables for API keys
load_dotenv()

class Framework(Enum):
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    CUSTOM = "custom"

def instrument(project_name="LangGraph_Prototype4", framework=Framework.LANGGRAPH):
    """Configure Phoenix instrumentation."""
    print(f"Initializing Phoenix tracing for {framework.value}...")
    
    # Try to import phoenix and configure tracing
    try:
        # Using the phoenix.otel package
        from phoenix.otel import register
        
        # Register with phoenix - using batch=True to use BatchSpanProcessor
        # This is the correct parameter according to the documentation
        register(
            project_name=project_name,
            auto_instrument=True,
            batch=True,  # Use BatchSpanProcessor instead of SimpleSpanProcessor
            verbose=True
        )
        print("✅ Phoenix tracing initialized with BatchSpanProcessor")
            
    except ImportError as e:
        print(f"⚠️ Phoenix tracing not available: {str(e)}")
    
    return trace.get_tracer(__name__)

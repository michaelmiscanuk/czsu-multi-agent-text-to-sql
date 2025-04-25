import os
from enum import Enum
from dotenv import load_dotenv
from opentelemetry import trace

# Load environment variables
load_dotenv()

class Framework(Enum):
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    CUSTOM = "custom"

def instrument(project_name="LangGraph_Prototype4", framework=Framework.LANGGRAPH):
    """Configure Phoenix instrumentation based on the framework."""
    print(f"Initializing Phoenix tracing for {framework.value}...")
    
    try:
        from arize.phoenix.opentelemetry import configure_opentelemetry
        
        # Configure OpenTelemetry with Phoenix
        configure_opentelemetry(
            service_name=project_name,
            collector_endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"),
            headers={"api_key": os.getenv("PHOENIX_API_KEY")}
        )
        
        # Framework-specific instrumentation
        if framework == Framework.LANGCHAIN:
            from langchain.callbacks.opentelemetry_callback import OpenTelemetryCallback
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            return OpenTelemetryCallback()
        
        print("✅ Phoenix tracing initialized")
        return trace.get_tracer(__name__)
        
    except ImportError as e:
        print(f"⚠️ Phoenix tracing not available: {str(e)}. Using default tracer.")
        return trace.get_tracer(__name__)

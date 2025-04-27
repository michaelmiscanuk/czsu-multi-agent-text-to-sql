"""Instrumentation utilities for tracing and monitoring.

This module provides functionality for setting up telemetry and tracing
with Phoenix for monitoring application performance and behavior.

The instrumentation is critical for production deployments as it enables:
1. Visibility into execution paths and bottlenecks
2. Error tracking and diagnosis
3. Performance monitoring and optimization
4. Usage analytics for improvement
"""

#==============================================================================
# IMPORTS
#==============================================================================
import os
from enum import Enum
from dotenv import load_dotenv
from opentelemetry import trace

# Load environment variables for API keys
load_dotenv()

#==============================================================================
# CLASSES
#==============================================================================
class Framework(Enum):
    """Enum defining supported frameworks for instrumentation."""
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    CUSTOM = "custom"

#==============================================================================
# FUNCTIONS
#==============================================================================
def instrument(project_name="LangGraph_Prototype4", framework=Framework.LANGGRAPH):
    """Configure Phoenix instrumentation for tracing and monitoring.
    
    This function sets up the OpenTelemetry-based tracing infrastructure that
    provides visibility into the application's execution. The instrumentation is
    designed to be non-intrusive and fail gracefully if Phoenix is not available,
    ensuring the application can run in development environments without dependencies.
    
    In production, the tracing provides crucial insights for:
    - Identifying performance bottlenecks in the workflow
    - Debugging execution paths when errors occur
    - Monitoring LLM usage and response times
    - Tracking conversation flows for improvement
    
    Args:
        project_name (str): Name of the project for Phoenix tracing - used to group
                          related traces in the Phoenix UI
        framework (Framework): The framework being used - affects which components
                             are instrumented automatically
        
    Returns:
        A tracer object configured for the specified project that can be used
        for additional manual instrumentation if needed
    """
    print(f"Initializing Phoenix tracing for {framework.value}...")
    
    # Attempt to set up Phoenix tracing but gracefully handle its absence
    # This makes the code work in both development and production environments
    try:
        # Using the phoenix.otel package for OpenTelemetry integration
        from phoenix.otel import register
        
        # Register with phoenix using batch processing for efficiency
        # BatchSpanProcessor reduces overhead by sending traces in batches
        # rather than one at a time, which is important for performance
        register(
            project_name=project_name,
            auto_instrument=True,  # Automatically instrument supported libraries
            batch=True,            # Use efficient batch processing
            verbose=True           # Enable detailed logs for debugging
        )
        print("✅ Phoenix tracing initialized with BatchSpanProcessor")
            
    except ImportError as e:
        # Provide a clear message but continue execution
        # This ensures the application works even without Phoenix
        print(f"⚠️ Phoenix tracing not available: {str(e)}")
    
    # Always return a valid tracer regardless of Phoenix availability
    # This allows code to use the tracer without checking if Phoenix is installed
    return trace.get_tracer(__name__)

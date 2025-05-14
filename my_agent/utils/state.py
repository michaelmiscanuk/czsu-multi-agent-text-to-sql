"""State definitions for the data analysis workflow.

This module defines the Pydantic models used to represent state
in the LangGraph-based data analysis application.
"""

#==============================================================================
# IMPORTS
#==============================================================================
from typing import List, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

#==============================================================================
# STATE CLASSES
#==============================================================================
class DataAnalysisState(BaseModel):
    """State for the data analysis graph.
    
    This model tracks the state of the data analysis workflow,
    including the user prompt, conversation messages, results,
    and iteration counter for loop prevention.
    """
    prompt: str = Field(default="", description="User query to analyze")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation history")
    iteration: int = Field(default=0, description="Iteration counter for workflow loop prevention")
    queries_and_results: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Collection of executed queries and their corresponding results"
    )

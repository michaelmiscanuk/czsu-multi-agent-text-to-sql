from typing import Dict, List, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class DataAnalysisState(BaseModel):
    """State for the data analysis graph"""
    prompt: str = Field(default="", description="User query to analyze")
    messages: List[BaseMessage] = Field(default_factory=list, description="Messages in the conversation")
    result: str = Field(default="", description="Result of the data analysis")

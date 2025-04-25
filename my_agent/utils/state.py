from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class DataAnalysisState(BaseModel):
    """State for the data analysis graph"""
    
    # Input data
    prompt: str = Field(default="", description="User query to analyze")
    data_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema metadata of the dataset")
    
    # Agent information
    messages: List[BaseMessage] = Field(default_factory=list, description="Messages in the conversation")
    
    # Output data
    result: str = Field(default="", description="Result of the data analysis")

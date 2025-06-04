"""State definitions for the data analysis workflow.

This module defines the state schema used to represent state
in the LangGraph-based data analysis application.
"""

#==============================================================================
# IMPORTS
#==============================================================================
from typing import List, Tuple, TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage

#==============================================================================
# STATE CLASSES
#==============================================================================
class DataAnalysisState(TypedDict):
    """State for the data analysis graph.
    
    This model tracks the state of the data analysis workflow,
    including the user prompt, conversation messages, results,
    and iteration counter for loop prevention.
    
    The state uses Annotated types with reducers to properly handle
    state updates:
    - messages: Uses add reducer to append new messages
    - queries_and_results: Uses add reducer to append new query results
    - iteration: Uses default override behavior
    - prompt: Uses default override behavior
    - reflection_decision: Stores the last decision from the reflection node ("improve" or "answer")
    """
    prompt: str  # User query to analyze
    rewritten_prompt: str  # Rewritten user query for downstream nodes
    rewritten_prompt_history: Annotated[List[str], add]  # History of rewritten prompts for conversational context
    messages: Annotated[List[BaseMessage], add]  # Conversation history with add reducer
    iteration: int  # Iteration counter for workflow loop prevention
    queries_and_results: Annotated[List[Tuple[str, str]], add]  # Collection of executed queries and their results with add reducer
    reflection_decision: str  # Last decision from the reflection node: "improve" or "answer"
    most_similar_selections: Annotated[List[Tuple[str, float]], add]  # List of (selection_code, cosine_similarity)
    selection_with_possible_answer: str  # Name of selection_code with possible answer, or None
    chromadb_missing: bool  # True if ChromaDB directory is missing, else False or not present

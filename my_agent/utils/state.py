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
    
    This tracks the state of the data analysis workflow, including the user prompt,
    conversation messages, query results, and iteration counter for loop prevention.
    
    Key features:
    - messages: Always [summary (SystemMessage), last_message (AIMessage/HumanMessage)]
    - queries_and_results: Uses add reducer to append new query results
    - iteration: Loop prevention counter
    """
    prompt: str  # User query to analyze
    rewritten_prompt: str  # Rewritten user query for downstream nodes
    rewritten_prompt_history: List[str]  # History of rewritten prompts for conversational context
    messages: List[BaseMessage]  # Always [summary (SystemMessage), last_message (AIMessage or HumanMessage)]
    iteration: int  # Iteration counter for workflow loop prevention
    queries_and_results: Annotated[List[Tuple[str, str]], add]  # Collection of executed queries and their results with add reducer
    reflection_decision: str  # Last decision from the reflection node: "improve" or "answer"
    most_similar_selections: Annotated[List[Tuple[str, float]], add]  # List of (selection_code, cosine_similarity
    top_selection_codes: List[str]  # List of top N selection codes (e.g., top 3)
    chromadb_missing: bool  # True if ChromaDB directory is missing, else False or not present

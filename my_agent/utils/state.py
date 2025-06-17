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
from langchain_core.documents import Document

#==============================================================================
# CUSTOM REDUCERS
#==============================================================================
def limited_queries_reducer(left: List[Tuple[str, str]], right: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Custom reducer that limits queries_and_results to the latest MAX_QUERIES_LIMIT_FOR_REFLECT entries.
    
    This prevents excessive memory usage and token consumption when there are loops
    in the workflow that generate many similar queries.
    
    Args:
        left: Existing queries and results
        right: New queries and results to add
        
    Returns:
        Combined list limited to MAX_QUERIES_LIMIT_FOR_REFLECT most recent entries
    """
    # Configurable limit - can be overridden by environment variable
    import os
    MAX_QUERIES_LIMIT_FOR_REFLECT = int(os.environ.get('MAX_QUERIES_LIMIT_FOR_REFLECT', '10'))
    
    # Combine existing and new queries
    combined = (left or []) + (right or [])
    
    # Return only the latest entries up to the limit
    return combined[-MAX_QUERIES_LIMIT_FOR_REFLECT:]

#==============================================================================
# STATE CLASSES
#==============================================================================
class DataAnalysisState(TypedDict):
    """State for the data analysis graph.
    
    This tracks the state of the data analysis workflow, including the user prompt,
    conversation messages, query results, and iteration counter for loop prevention.
    
    Key features:
    - messages: Always [summary (SystemMessage), last_message (AIMessage/HumanMessage)]
    - queries_and_results: Uses limited_queries_reducer to keep only recent queries
    - iteration: Loop prevention counter
    - final_answer: Explicitly tracked final formatted answer string
    """
    prompt: str  # User query to analyze
    rewritten_prompt: str  # Rewritten user query for downstream nodes
    rewritten_prompt_history: List[str]  # History of rewritten prompts for conversational context
    messages: List[BaseMessage]  # Always [summary (SystemMessage), last_message (AIMessage or HumanMessage)]
    iteration: int  # Iteration counter for workflow loop prevention
    queries_and_results: Annotated[List[Tuple[str, str]], limited_queries_reducer]  # Collection of executed queries and their results with limited reducer
    reflection_decision: str  # Last decision from the reflection node: "improve" or "answer"
    hybrid_search_results: List[Document]  # Intermediate hybrid search results before reranking (uses default replacement behavior)
    most_similar_selections: List[Tuple[str, float]]  # List of (selection_code, cohere_rerank_score) after reranking
    top_selection_codes: List[str]  # List of top N selection codes (e.g., top 3)
    chromadb_missing: bool  # True if ChromaDB directory is missing, else False or not present
    # New PDF chunk functionality states
    hybrid_search_chunks: List[Document]  # Intermediate hybrid search results for PDF chunks before reranking
    most_similar_chunks: List[Tuple[Document, float]]  # List of (document, cohere_rerank_score) after reranking PDF chunks
    top_chunks: List[Document]  # List of top N PDF chunks that passed relevance threshold
    final_answer: str  # Explicitly tracked final formatted answer string

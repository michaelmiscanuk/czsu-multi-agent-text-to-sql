module_description = r"""State Management for LangGraph Multi-Agent Text-to-SQL Workflow

This module defines the state schema and custom reducers for the LangGraph-based data analysis
workflow. It implements a TypedDict-based state structure that tracks all data flowing through
the multi-agent system, from initial user query to final formatted answer.

The state design emphasizes memory efficiency, token limit management, and workflow control
through specialized reducers and carefully structured data fields.

Architecture Overview:
=====================
The state management system consists of:
1. DataAnalysisState TypedDict (main state container with 15 fields)
2. Custom Reducers (limited_queries_reducer for memory management)
3. Field-Level Documentation (inline comments for each state field)
4. Type Safety (strict typing with Annotated types for custom reducers)

Key Features:
============
1. Memory-Efficient Message Management:
   - Always maintains exactly 2 messages: [summary (SystemMessage), last_message]
   - Prevents token overflow through automatic summarization
   - Preserves conversational context without full history
   - Enables multi-turn conversations with bounded memory

2. Limited Query History Tracking:
   - Custom reducer limits queries_and_results to latest N entries
   - Configurable limit via MAX_QUERIES_LIMIT_FOR_REFLECT (default: 10)
   - Prevents excessive memory usage in reflection loops
   - Maintains relevant query context without overflow

3. Dual-Source Retrieval State:
   - Separate tracking for database selections and PDF chunks
   - Parallel processing state fields (hybrid_search_results, hybrid_search_chunks)
   - Reranked results with scores (most_similar_selections, most_similar_chunks)
   - Final filtered results (top_selection_codes, top_chunks)

4. Workflow Control Fields:
   - iteration: Loop counter for reflection cycle prevention
   - reflection_decision: Controls "improve" vs "answer" routing
   - chromadb_missing: Error flag for missing vector database

5. Multi-Stage Query Processing:
   - prompt: Original user question (unchanged)
   - rewritten_prompt: Search-optimized standalone question
   - rewritten_prompt_history: Full conversation context preservation

6. Explicit Final Answer Tracking:
   - final_answer: Dedicated field for formatted response
   - Prevents ambiguity in multi-stage answer generation
   - Enables easy extraction of final result

State Field Categories:
======================

Input Fields (User-Provided):
-----------------------------
- prompt: Original natural language question from user

Query Processing Fields:
-----------------------
- rewritten_prompt: Standalone question optimized for retrieval
- rewritten_prompt_history: List of all rewritten prompts in conversation

Conversation Management:
-----------------------
- messages: Always [summary (SystemMessage), last_message (BaseMessage)]
  * Uses default replacement behavior (new messages replace old)
  * Summarization happens in summarize_messages_node before replacement

Workflow Control:
----------------
- iteration: int (loop counter, default: 0)
- reflection_decision: "improve" | "answer" (from reflect_node)
- chromadb_missing: bool (error flag for missing database)

Database Selection Retrieval:
-----------------------------
- hybrid_search_results: List[Document] (raw hybrid search results)
- most_similar_selections: List[Tuple[str, float]] (reranked selection codes with scores)
- top_selection_codes: List[str] (final top-k selection codes for schema loading)

PDF Chunk Retrieval:
-------------------
- hybrid_search_chunks: List[Document] (raw PDF hybrid search results)
- most_similar_chunks: List[Tuple[Document, float]] (reranked chunks with scores)
- top_chunks: List[Document] (final filtered chunks above relevance threshold)

Query Execution:
---------------
- queries_and_results: Annotated[List[Tuple[str, str]], limited_queries_reducer]
  * List of (SQL_query, result_string) tuples
  * Uses custom reducer to limit to most recent N entries
  * Prevents token overflow in reflection loops

Output:
------
- final_answer: str (formatted answer synthesizing all sources)

Custom Reducers:
===============

limited_queries_reducer:
-----------------------
Purpose: Prevents memory/token overflow by limiting query history

Behavior:
- Combines existing (left) and new (right) queries
- Keeps only the latest MAX_QUERIES_LIMIT_FOR_REFLECT entries
- Default limit: 10 (configurable via environment variable)

Usage:
```python
queries_and_results: Annotated[
    List[Tuple[str, str]], 
    limited_queries_reducer
]
```

Why needed:
- Reflection loops can generate many similar queries
- Full query history can exceed token limits
- Recent queries are most relevant for reflection
- Prevents state growth over long conversations

Default Behavior (Other Fields):
================================
Fields without custom reducers use LangGraph's default behavior:
- Replacement: New value replaces old value (e.g., messages, prompt)
- Append: Not used in this state (would use operator.add if needed)

Message Management Strategy:
===========================
The messages field uses a two-stage approach:

Stage 1 - Summarization (in summarize_messages_node):
- Takes current messages list (may be long)
- Generates summary of conversation using LLM
- Creates new [summary (SystemMessage), last_message] pair
- Returns this condensed structure

Stage 2 - Replacement (LangGraph default):
- New 2-message list replaces old messages
- State always contains exactly 2 messages
- Token-efficient without custom reducer

This design choice:
- Keeps reducer logic simple (just replacement)
- Moves intelligence to node (summarize_messages_node)
- Maintains clean separation of concerns

State Lifecycle Example:
=======================

Initial State:
-------------
```python
{
    "prompt": "How many teachers in Prague?",
    "messages": [],
    "iteration": 0
}
```

After rewrite_query:
-------------------
```python
{
    "prompt": "How many teachers in Prague?",
    "rewritten_prompt": "What is the total number of teachers employed in Prague?",
    "messages": [SystemMessage(summary), AIMessage(rewritten)],
    "iteration": 0
}
```

After parallel retrieval:
------------------------
```python
{
    # ... previous fields ...
    "hybrid_search_results": [Doc1, Doc2, ...],  # Database selections
    "hybrid_search_chunks": [Chunk1, Chunk2, ...],  # PDF chunks
}
```

After reranking:
---------------
```python
{
    # ... previous fields ...
    "most_similar_selections": [("sel_123", 0.95), ("sel_456", 0.87)],
    "most_similar_chunks": [(Chunk1, 0.92), (Chunk2, 0.81)],
}
```

After filtering:
---------------
```python
{
    # ... previous fields ...
    "top_selection_codes": ["sel_123", "sel_456", "sel_789"],
    "top_chunks": [Chunk1, Chunk2],  # Above relevance threshold
}
```

After query execution:
---------------------
```python
{
    # ... previous fields ...
    "queries_and_results": [
        ("SELECT COUNT(*) FROM teachers WHERE city='Prague'", "Result: 1500")
    ],
    "iteration": 0,
}
```

After reflection:
----------------
```python
{
    # ... previous fields ...
    "reflection_decision": "answer",  # or "improve"
    "iteration": 1,
}
```

After final formatting:
----------------------
```python
{
    # ... previous fields ...
    "final_answer": "Based on CZSU database, there are 1,500 teachers in Prague.",
}
```

Type Safety & Validation:
=========================
- All fields use explicit Python type hints
- TypedDict provides IDE autocomplete and type checking
- Annotated types enable custom reducer attachment
- Document and BaseMessage types from langchain-core
- Tuple types specify exact structure (e.g., Tuple[str, float])

Configuration:
=============
Environment Variables:
- MAX_QUERIES_LIMIT_FOR_REFLECT: Maximum query history entries (default: 10)

Usage in Graph Nodes:
====================

Reading State:
-------------
```python
async def my_node(state: DataAnalysisState) -> DataAnalysisState:
    prompt = state["prompt"]
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])
    # ... process ...
```

Updating State:
--------------
```python
async def my_node(state: DataAnalysisState) -> DataAnalysisState:
    # Return dict with fields to update
    return {
        "rewritten_prompt": "optimized question",
        "iteration": state.get("iteration", 0) + 1,
    }
```

Using Custom Reducer Fields:
---------------------------
```python
# Append to queries_and_results (reducer handles limiting)
return {
    "queries_and_results": [("SELECT ...", "Result: ...")]
}
# Reducer automatically combines with existing and limits to 10
```

Best Practices:
==============
1. Always check for field existence with .get() for optional fields
2. Use explicit types for better IDE support and error catching
3. Document any new fields with inline comments
4. Consider memory implications before adding large data structures
5. Use custom reducers for fields that accumulate over iterations
6. Keep state flat (avoid nested structures) for simplicity
7. Use descriptive field names that indicate data stage (e.g., hybrid_search_results vs most_similar_selections)

Integration with Graph:
======================
This state is used by:
1. Graph initialization: StateGraph(DataAnalysisState)
2. Node functions: All nodes receive and return DataAnalysisState
3. Checkpointing: Full state serialized to PostgreSQL/memory
4. API responses: State fields extracted for user responses

See my_agent/agent.py for graph structure using this state.
See my_agent/utils/nodes.py for node implementations manipulating this state.
"""

"""State definitions for the data analysis workflow.

This module defines the state schema used to represent state
in the LangGraph-based data analysis application.
"""

from operator import add

# ==============================================================================
# IMPORTS
# ==============================================================================
from typing import Annotated, List, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


# ==============================================================================
# CUSTOM REDUCERS
# ==============================================================================
def limited_queries_reducer(
    left: List[Tuple[str, str]], right: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
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

    MAX_QUERIES_LIMIT_FOR_REFLECT = int(
        os.environ.get("MAX_QUERIES_LIMIT_FOR_REFLECT", "10")
    )

    # Combine existing and new queries
    combined = (left or []) + (right or [])

    # Return only the latest entries up to the limit
    return combined[-MAX_QUERIES_LIMIT_FOR_REFLECT:]


# ==============================================================================
# STATE CLASSES
# ==============================================================================
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
    rewritten_prompt_history: List[
        str
    ]  # History of rewritten prompts for conversational context
    messages: List[
        BaseMessage
    ]  # Always [summary (SystemMessage), last_message (AIMessage or HumanMessage)]
    iteration: int  # Iteration counter for workflow loop prevention
    queries_and_results: Annotated[
        List[Tuple[str, str]], limited_queries_reducer
    ]  # Collection of executed queries and their results with limited reducer
    reflection_decision: (
        str  # Last decision from the reflection node: "improve" or "answer"
    )
    hybrid_search_results: List[
        Document
    ]  # Intermediate hybrid search results before reranking (uses default replacement behavior)
    most_similar_selections: List[
        Tuple[str, float]
    ]  # List of (selection_code, cohere_rerank_score) after reranking
    top_selection_codes: List[str]  # List of top N selection codes (e.g., top 3)
    chromadb_missing: (
        bool  # True if ChromaDB directory is missing, else False or not present
    )
    hybrid_search_chunks: List[
        Document
    ]  # Intermediate hybrid search results for PDF chunks before reranking
    most_similar_chunks: List[
        Tuple[Document, float]
    ]  # List of (document, cohere_rerank_score) after reranking PDF chunks
    top_chunks: List[
        Document
    ]  # List of top N PDF chunks that passed relevance threshold
    final_answer: str  # Explicitly tracked final formatted answer string

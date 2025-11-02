module_description = r"""LangGraph Workflow Definition for Multi-Agent Text-to-SQL Analysis

This module defines the LangGraph StateGraph structure and execution flow for a multi-agent
text-to-SQL analysis system. It orchestrates the complete workflow from natural language
question input to formatted answer output, managing state transitions, parallel execution
branches, conditional routing logic, and checkpointing.

Designed for Czech statistical data (CZSU) with support for dual retrieval sources
(database selections + PDF documentation) and iterative query improvement through reflection.

Graph Architecture:
==================
The workflow implements a directed acyclic graph (with controlled cycles) containing:
1. Parallel Retrieval Phase (2 branches: database selections + PDF chunks)
2. Synchronization & Routing Logic (conditional paths based on available data)
3. Query Generation & Execution Loop (with optional reflection for improvement)
4. Answer Synthesis & Finalization (multi-source information formatting)

Graph Structure & Execution Flow:
================================

Phase 1: Query Preprocessing
----------------------------
START → rewrite_prompt → summarize_messages_rewrite

- Converts conversational questions into standalone search queries
- Summarizes conversation history to manage token limits
- Prepares optimized query for parallel retrieval

Phase 2: Parallel Retrieval (Dual Branches)
-------------------------------------------
summarize_messages_rewrite splits into TWO parallel branches:

Branch A (Database Selections):
  → retrieve_similar_selections_hybrid_search
  → rerank_table_descriptions (Cohere reranking)
  → relevant_selections (top-k filtering)

Branch B (PDF Documentation):
  → retrieve_similar_chunks_hybrid_search
  → rerank_chunks (Cohere reranking)
  → relevant_chunks (threshold filtering)

Both branches use hybrid search (semantic + BM25) with configurable weighting.

Phase 3: Synchronization & Conditional Routing
----------------------------------------------
[relevant_selections, relevant_chunks] → post_retrieval_sync

Routing logic (handled by my_agent.utils.routers.route_after_sync):
- IF top_selection_codes found → get_schema (proceed with database queries)
- ELIF chromadb_missing → END (error: no ChromaDB available)
- ELSE → format_answer (PDF-only response, no database data)

Phase 4: Query Loop (Optional Reflection)
-----------------------------------------
get_schema → generate_query → summarize_messages_query

Conditional routing based on iteration count (handled by my_agent.utils.routers.route_after_query):
- IF iteration < MAX_ITERATIONS → reflect
- ELSE → format_answer (force answer at iteration limit)

Reflection cycle (optional):
reflect → summarize_messages_reflect

Reflection decision (handled by my_agent.utils.routers.route_after_reflect):
- IF decision == "improve" → generate_query (loop back for better query)
- ELIF decision == "answer" → format_answer (sufficient data collected)

Phase 5: Answer Finalization
----------------------------
format_answer → summarize_messages_format → generate_followup_prompts → submit_final_answer → save → cleanup_resources → END

- Synthesizes information from all sources (SQL results, PDF chunks, selection descriptions)
- Generates followup prompts for continued conversation
- Submits formatted answer to user
- Optionally saves results to file
- Cleans up resources and connections

State Management:
================
Uses DataAnalysisState TypedDict with key fields:
- prompt: Original user question
- rewritten_prompt: Search-optimized standalone question
- messages: [summary (SystemMessage), last_message] - token-efficient structure
- iteration: Loop counter for cycle prevention (default max: 1)
- queries_and_results: Limited list of (SQL_query, result) tuples
- reflection_decision: "improve" or "answer" from reflect node
- top_selection_codes: Database table identifiers for schema loading
- top_chunks: Relevant PDF documentation chunks
- final_answer: Formatted answer string

Node Summary:
=======================
Preprocessing: rewrite_prompt
Retrieval: retrieve_similar_selections_hybrid_search, retrieve_similar_chunks_hybrid_search
Reranking: rerank_table_descriptions, rerank_chunks
Filtering: relevant_selections, relevant_chunks
Routing: post_retrieval_sync (inline function)
Query: get_schema, generate_query
Reflection: reflect
Formatting: format_answer, generate_followup_prompts, submit_final_answer
Persistence: save, cleanup_resources
Memory: summarize_messages_rewrite/query/reflect/format (4 instances of same node)

(Detailed node documentation available in my_agent.utils.nodes)
(Routing functions documented in my_agent.utils.routers)

Key Design Principles:
=====================
1. Parallel Execution: Dual retrieval branches run simultaneously for efficiency
2. Conditional Routing: Smart decision points based on available data (externalized to routers module)
3. Controlled Iteration: MAX_ITERATIONS prevents infinite reflection loops
4. State Checkpointing: PostgreSQL persistence for workflow resumption
5. Token Management: Automatic message summarization at key points
6. Resource Cleanup: Explicit cleanup node for connection management
7. Modular Architecture: Routing logic separated into dedicated utils.routers module
8. Absolute Imports: All imports use absolute paths with my_agent prefix for clarity

Configuration Constants:
=======================
Workflow Control:
- MAX_ITERATIONS: 1 (configurable via environment variable)

Debug Tracing:
- Graph-level construction tracing (present for development/debugging)

Usage Example:
=============
```python
from my_agent.agent import create_graph

# Create graph with checkpointer
graph = create_graph(checkpointer=await get_async_postgres_checkpointer())

# Execute workflow
result = await graph.ainvoke({
    "prompt": "Your question here",
    "messages": [],
    "iteration": 0,
    "followup_prompts": []
}, config={"configurable": {"thread_id": "conv-123"}})

# Access final answer
print(result["final_answer"])
```

Checkpointer Behavior:
======================
- If checkpointer=None: Uses InMemorySaver fallback (development/testing)
- Production: Should provide AsyncPostgresSaver for persistent state
- Enables workflow interruption, resumption, and conversation history

Module Organization:
==================
- my_agent.utils.nodes: All node function implementations
- my_agent.utils.state: DataAnalysisState schema and reducers
- my_agent.utils.routers: Conditional routing functions (route_after_sync, route_after_query, route_after_reflect)
- api.utils.debug: Debug tracing utilities

See my_agent/utils/nodes.py for detailed node implementation documentation.
See my_agent/utils/state.py for complete state schema with reducers.
See my_agent/utils/routers.py for routing logic documentation.
"""

from typing import Literal

# ==============================================================================
# IMPORTS
# ==============================================================================
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import (
    cleanup_resources_node,
    followup_prompts_node,
    format_answer_node,
    generate_query_node,
    get_schema_node,
    reflect_node,
    relevant_chunks_node,
    relevant_selections_node,
    rerank_chunks_node,
    rerank_table_descriptions_node,
    retrieve_similar_chunks_hybrid_search_node,
    retrieve_similar_selections_hybrid_search_node,
    rewrite_prompt_node,
    post_retrieval_sync_node,
    save_node,
    submit_final_answer_node,
    summarize_messages_node,
)
from my_agent.utils.state import DataAnalysisState

# Import routing functions
from my_agent.utils.routers import (
    route_after_query,
    route_after_reflect,
    route_after_sync,
)

# Load environment variables
load_dotenv()

import os

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Import debug functions from utils
from api.utils.debug import print__analysis_tracing_debug


# ==============================================================================
# GRAPH CREATION
# ==============================================================================
def create_graph(checkpointer=None):
    """Create the graph for data analysis.

    This function constructs a directed graph representing the graph for
    data analysis tasks. The graph design follows several important principles:

    1. Clear separation of concerns between nodes
    2. Explicit error handling and recovery paths
    3. Controlled iteration with cycle prevention
    4. Checkpointing for execution resumption

    The resulting graph manages the complete process from natural language understanding
    to query execution and result formatting, with built-in safeguards against
    common failure modes.

    Args:
        checkpointer: Optional checkpointer instance. If None, defaults to InMemorySaver
                     for backward compatibility. In production, should use AsyncPostgresSaver.

    Returns:
        A compiled StateGraph ready for execution
    """

    # --------------------------------------------------------------------------
    # INITIALIZE GRAPH WITH CUSTOM STATE
    # --------------------------------------------------------------------------
    # Initialize with our custom state type to track conversation and results
    graph = StateGraph(DataAnalysisState)

    # --------------------------------------------------------------------------
    # ADD NODES: each node is handling a specific step in the process
    # --------------------------------------------------------------------------

    graph.add_node("rewrite_prompt", rewrite_prompt_node)
    graph.add_node("summarize_messages_rewrite", summarize_messages_node)
    graph.add_node(
        "retrieve_similar_selections_hybrid_search",
        retrieve_similar_selections_hybrid_search_node,
    )
    graph.add_node("rerank_table_descriptions", rerank_table_descriptions_node)
    graph.add_node("relevant_selections", relevant_selections_node)
    graph.add_node(
        "retrieve_similar_chunks_hybrid_search",
        retrieve_similar_chunks_hybrid_search_node,
    )
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("relevant_chunks", relevant_chunks_node)
    graph.add_node("post_retrieval_sync", post_retrieval_sync_node)
    graph.add_node("get_schema", get_schema_node)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("summarize_messages_query", summarize_messages_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("summarize_messages_reflect", summarize_messages_node)
    graph.add_node("format_answer", format_answer_node)
    graph.add_node("summarize_messages_format", summarize_messages_node)
    graph.add_node("generate_followup_prompts", followup_prompts_node)
    graph.add_node("submit_final_answer", submit_final_answer_node)
    graph.add_node("save", save_node)
    graph.add_node("cleanup_resources", cleanup_resources_node)

    # --------------------------------------------------------------------------
    # EDGES: Define the graph execution path
    # --------------------------------------------------------------------------

    # Start: prompt -> rewrite_prompt -> summarize_messages -> retrieve (both selections and chunks in parallel)
    graph.add_edge(START, "rewrite_prompt")
    graph.add_edge("rewrite_prompt", "summarize_messages_rewrite")

    # After summarize_messages_rewrite, branch to both selection and chunk retrieval (parallel execution)
    graph.add_edge(
        "summarize_messages_rewrite", "retrieve_similar_selections_hybrid_search"
    )
    graph.add_edge(
        "summarize_messages_rewrite", "retrieve_similar_chunks_hybrid_search"
    )

    # Selection path: retrieve -> rerank_table_descriptions -> relevant table descriptions
    graph.add_edge(
        "retrieve_similar_selections_hybrid_search", "rerank_table_descriptions"
    )
    graph.add_edge("rerank_table_descriptions", "relevant_selections")

    # PDF chunk path: retrieve -> rerank_chunks -> relevant (runs in parallel)
    graph.add_edge("retrieve_similar_chunks_hybrid_search", "rerank_chunks")
    graph.add_edge("rerank_chunks", "relevant_chunks")

    # Both branches feed into the synchronization node
    graph.add_edge("relevant_selections", "post_retrieval_sync")
    graph.add_edge("relevant_chunks", "post_retrieval_sync")

    graph.add_conditional_edges(
        "post_retrieval_sync",
        route_after_sync,
        {"get_schema": "get_schema", "format_answer": "format_answer", END: END},
    )

    # get_schema -> generate_query (no summarize_messages_schema)
    graph.add_edge("get_schema", "generate_query")

    # generate_query -> summarize_messages -> reflect/format_answer
    graph.add_edge("generate_query", "summarize_messages_query")

    graph.add_conditional_edges(
        "summarize_messages_query",
        route_after_query,
        {"reflect": "reflect", "format_answer": "format_answer"},
    )

    # reflect -> summarize_messages -> generate_query/format_answer
    graph.add_edge("reflect", "summarize_messages_reflect")

    graph.add_conditional_edges(
        "summarize_messages_reflect",
        route_after_reflect,
        {"generate_query": "generate_query", "format_answer": "format_answer"},
    )

    # format_answer -> summarize_messages -> generate_followup_prompts -> submit_final_answer
    graph.add_edge("format_answer", "summarize_messages_format")
    graph.add_edge("summarize_messages_format", "generate_followup_prompts")
    graph.add_edge("generate_followup_prompts", "submit_final_answer")
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", "cleanup_resources")
    graph.add_edge("cleanup_resources", END)

    # --------------------------------------------------------------------------
    # CHECKPOINTER: Use provided or default to InMemorySaver
    # --------------------------------------------------------------------------
    if checkpointer is None:
        print__analysis_tracing_debug(
            "108 - INMEMORY SAVER: No checkpointer provided, using InMemorySaver"
        )
        # Import here to avoid circular imports and provide fallback
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        print(
            "⚠️ Using InMemorySaver fallback - consider using AsyncPostgresSaver for production"
        )
        print__analysis_tracing_debug(
            "109 - INMEMORY CREATED: InMemorySaver fallback created"
        )
    else:
        print__analysis_tracing_debug(
            f"110 - CHECKPOINTER PROVIDED: Using provided checkpointer ({type(checkpointer).__name__})"
        )

    # --------------------------------------------------------------------------
    # COMPILATION
    # --------------------------------------------------------------------------
    # Compile the graph with the checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph

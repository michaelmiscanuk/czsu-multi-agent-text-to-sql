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
START → rewrite_query → summarize_messages_rewrite

- Converts conversational questions into standalone search queries
- Summarizes conversation history to manage token limits
- Prepares optimized query for parallel retrieval

Phase 2: Parallel Retrieval (Dual Branches)
-------------------------------------------
summarize_messages_rewrite splits into TWO parallel branches:

Branch A (Database Selections):
  → retrieve_similar_selections_hybrid_search
  → rerank (Cohere reranking)
  → relevant_selections (top-k filtering)

Branch B (PDF Documentation):
  → retrieve_similar_chunks_hybrid_search
  → rerank_chunks (Cohere reranking)
  → relevant_chunks (threshold filtering)

Both branches use hybrid search (semantic + BM25) with configurable weighting.

Phase 3: Synchronization & Conditional Routing
----------------------------------------------
[relevant_selections, relevant_chunks] → route_decision

Routing logic:
- IF top_selection_codes found → get_schema (proceed with database queries)
- ELIF chromadb_missing → END (error: no ChromaDB available)
- ELSE → format_answer (PDF-only response, no database data)

Phase 4: Query Loop (Optional Reflection)
-----------------------------------------
get_schema → query_gen → summarize_messages_query

Conditional routing based on iteration count:
- IF iteration < MAX_ITERATIONS → reflect
- ELSE → format_answer (force answer at iteration limit)

Reflection cycle (optional):
reflect → summarize_messages_reflect

Reflection decision:
- IF decision == "improve" → query_gen (loop back for better query)
- ELIF decision == "answer" → format_answer (sufficient data collected)

Phase 5: Answer Finalization
----------------------------
format_answer → summarize_messages_format → submit_final_answer → save → cleanup_resources → END

- Synthesizes information from all sources (SQL results, PDF chunks, selection descriptions)
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

Node Summary (18 total):
=======================
Preprocessing: rewrite_query
Retrieval: retrieve_similar_selections_hybrid_search, retrieve_similar_chunks_hybrid_search
Reranking: rerank, rerank_chunks
Filtering: relevant_selections, relevant_chunks
Routing: route_decision (inline function)
Query: get_schema, query_gen
Reflection: reflect
Formatting: format_answer, submit_final_answer
Persistence: save, cleanup_resources
Memory: summarize_messages_rewrite/query/reflect/format (4 instances of same node)

(Detailed node documentation available in my_agent/utils/nodes.py)

Key Design Principles:
=====================
1. Parallel Execution: Dual retrieval branches run simultaneously for efficiency
2. Conditional Routing: Smart decision points based on available data
3. Controlled Iteration: MAX_ITERATIONS prevents infinite reflection loops
4. State Checkpointing: PostgreSQL persistence for workflow resumption
5. Token Management: Automatic message summarization at key points
6. Resource Cleanup: Explicit cleanup node for connection management
7. Debug Tracing: Unique IDs (84-111) for each graph construction step

Configuration Constants:
=======================
Workflow Control:
- MAX_ITERATIONS: 1 (configurable via environment variable)

Debug Tracing:
- print__analysis_tracing_debug: Graph-level construction tracing (IDs 84-111)

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

See my_agent/utils/nodes.py for detailed node implementation documentation.
See my_agent/utils/state.py for complete state schema with reducers.
"""

"""Agent graph definition module.

This module defines the data analysis graph using LangGraph. It implements a
multi-step process that:

1. Retrieves database schema information
2. Generates a SQL query from natural language
3. Reflects on whether we have enough information
4. Either generates more queries or formats the answer
5. Returns the final analysis

The graph includes error handling, retry mechanisms, and a controlled execution
flow that prevents common failure modes in LLM-based systems.
"""

from typing import Literal

# ==============================================================================
# IMPORTS
# ==============================================================================
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from .utils.nodes import (
    MAX_ITERATIONS,
    cleanup_resources_node,
    followup_prompts_node,
    format_answer_node,
    get_schema_node,
    query_node,
    reflect_node,
    relevant_chunks_node,
    relevant_selections_node,
    rerank_chunks_node,
    rerank_node,
    retrieve_similar_chunks_hybrid_search_node,
    retrieve_similar_selections_hybrid_search_node,
    rewrite_query_node,
    route_decision_node,
    save_node,
    submit_final_answer_node,
    summarize_messages_node,
)
from .utils.state import DataAnalysisState

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
    print__analysis_tracing_debug("84 - GRAPH CREATION START: Starting graph creation")

    # Initialize with our custom state type to track conversation and results
    graph = StateGraph(DataAnalysisState)
    print__analysis_tracing_debug(
        "85 - STATE GRAPH INIT: StateGraph initialized with DataAnalysisState"
    )

    # --------------------------------------------------------------------------
    # Add nodes - each handling a specific step in the process
    # --------------------------------------------------------------------------
    print__analysis_tracing_debug("86 - ADDING NODES: Adding all graph nodes")
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_followup_prompts", followup_prompts_node)
    graph.add_node(
        "retrieve_similar_selections_hybrid_search",
        retrieve_similar_selections_hybrid_search_node,
    )
    graph.add_node("rerank", rerank_node)
    graph.add_node("relevant_selections", relevant_selections_node)
    # New PDF chunk nodes - run in parallel with selection nodes
    graph.add_node(
        "retrieve_similar_chunks_hybrid_search",
        retrieve_similar_chunks_hybrid_search_node,
    )
    graph.add_node("rerank_chunks", rerank_chunks_node)
    graph.add_node("relevant_chunks", relevant_chunks_node)
    graph.add_node("get_schema", get_schema_node)
    graph.add_node("query_gen", query_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("format_answer", format_answer_node)
    graph.add_node("submit_final_answer", submit_final_answer_node)
    graph.add_node("save", save_node)
    graph.add_node("cleanup_resources", cleanup_resources_node)
    graph.add_node("summarize_messages_rewrite", summarize_messages_node)
    graph.add_node("summarize_messages_query", summarize_messages_node)
    graph.add_node("summarize_messages_reflect", summarize_messages_node)
    graph.add_node("summarize_messages_format", summarize_messages_node)
    print__analysis_tracing_debug(
        "87 - NODES ADDED: All 19 graph nodes added successfully (including cleanup and generate_followup_prompts)"
    )

    # --------------------------------------------------------------------------
    # Define the graph execution path
    # --------------------------------------------------------------------------
    print__analysis_tracing_debug("88 - ADDING EDGES: Defining graph execution path")
    # Start: prompt -> rewrite_query -> generate_followup_prompts -> summarize_messages -> retrieve (both selections and chunks in parallel)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "generate_followup_prompts")
    graph.add_edge("generate_followup_prompts", "summarize_messages_rewrite")
    # After summarize_messages_rewrite, branch to both selection and chunk retrieval (parallel execution)
    graph.add_edge(
        "summarize_messages_rewrite", "retrieve_similar_selections_hybrid_search"
    )
    graph.add_edge(
        "summarize_messages_rewrite", "retrieve_similar_chunks_hybrid_search"
    )

    # Selection path: retrieve -> rerank -> relevant
    graph.add_edge("retrieve_similar_selections_hybrid_search", "rerank")
    graph.add_edge("rerank", "relevant_selections")

    # PDF chunk path: retrieve -> rerank -> relevant (runs in parallel)
    graph.add_edge("retrieve_similar_chunks_hybrid_search", "rerank_chunks")
    graph.add_edge("rerank_chunks", "relevant_chunks")
    print__analysis_tracing_debug(
        "89 - PARALLEL EDGES: Added parallel processing edges for selections and chunks"
    )

    # Add the synchronization node that both branches feed into
    graph.add_node("route_decision", route_decision_node)
    print__analysis_tracing_debug(
        "91 - SYNC NODE ADDED: Route decision synchronization node added"
    )

    # Both branches feed into the synchronization node
    graph.add_edge("relevant_selections", "route_decision")
    graph.add_edge("relevant_chunks", "route_decision")
    print__analysis_tracing_debug(
        "92 - SYNC EDGES: Added edges to synchronization node"
    )

    # Single routing logic from the synchronization node
    def route_after_sync(state: DataAnalysisState):
        print__analysis_tracing_debug(
            "93 - ROUTING DECISION: Making routing decision after synchronization"
        )
        # Check if we have selection codes to proceed with database queries
        if state.get("top_selection_codes") and len(state["top_selection_codes"]) > 0:
            print__analysis_tracing_debug(
                f"94 - SCHEMA ROUTE: Found {len(state['top_selection_codes'])} selections, proceeding to database schema"
            )
            return "get_schema"
        elif state.get("chromadb_missing"):
            print(
                "❌ ERROR: ChromaDB directory is missing. Please unzip or create the ChromaDB at 'metadata/czsu_chromadb'."
            )
            print__analysis_tracing_debug(
                "95 - CHROMADB ERROR: ChromaDB directory missing, ending execution"
            )
            return END
        else:
            # No database selections found - proceed directly to answer with available PDF chunks
            print(
                "⚠️ No relevant dataset selections found, proceeding with PDF chunks only"
            )
            chunks_available = len(state.get("top_chunks", []))
            print__analysis_tracing_debug(
                f"96 - CHUNKS ONLY ROUTE: No selections found, proceeding with {chunks_available} PDF chunks"
            )
            return "format_answer"

    graph.add_conditional_edges(
        "route_decision",
        route_after_sync,
        {"get_schema": "get_schema", "format_answer": "format_answer", END: END},
    )
    print__analysis_tracing_debug(
        "97 - CONDITIONAL EDGES: Added conditional routing edges after synchronization"
    )

    # get_schema -> query_gen (no summarize_messages_schema)
    graph.add_edge("get_schema", "query_gen")

    # query_gen -> summarize_messages -> reflect/format_answer
    graph.add_edge("query_gen", "summarize_messages_query")

    def route_after_query(
        state: DataAnalysisState,
    ) -> Literal["reflect", "format_answer"]:
        iteration = state.get("iteration", 0)
        print(f"🔀 Routing decision, iteration={iteration}")
        print__analysis_tracing_debug(
            f"98 - QUERY ROUTING: Making routing decision after query, iteration={iteration}"
        )
        if iteration >= MAX_ITERATIONS:
            print__analysis_tracing_debug(
                f"99 - MAX ITERATIONS: Reached max iterations ({MAX_ITERATIONS}), proceeding to format answer"
            )
            return "format_answer"
        else:
            print__analysis_tracing_debug(
                f"100 - REFLECT ROUTE: Iteration {iteration} < {MAX_ITERATIONS}, proceeding to reflect"
            )
            return "reflect"

    graph.add_conditional_edges(
        "summarize_messages_query",
        route_after_query,
        {"reflect": "reflect", "format_answer": "format_answer"},
    )
    print__analysis_tracing_debug(
        "101 - QUERY CONDITIONAL: Added conditional edges after query generation"
    )

    # reflect -> summarize_messages -> query_gen/format_answer
    graph.add_edge("reflect", "summarize_messages_reflect")

    def route_after_reflect(
        state: DataAnalysisState,
    ) -> Literal["query_gen", "format_answer"]:
        decision = state.get("reflection_decision", "improve")
        print__analysis_tracing_debug(
            f"102 - REFLECT ROUTING: Reflection decision is '{decision}'"
        )
        if decision == "answer":
            print__analysis_tracing_debug(
                "103 - ANSWER ROUTE: Reflection says answer is ready, proceeding to format"
            )
            return "format_answer"
        else:
            print__analysis_tracing_debug(
                "104 - IMPROVE ROUTE: Reflection says improve needed, going back to query generation"
            )
            return "query_gen"

    graph.add_conditional_edges(
        "summarize_messages_reflect",
        route_after_reflect,
        {"query_gen": "query_gen", "format_answer": "format_answer"},
    )
    print__analysis_tracing_debug(
        "105 - REFLECT CONDITIONAL: Added conditional edges after reflection"
    )

    # format_answer -> summarize_messages -> submit_final_answer
    graph.add_edge("format_answer", "summarize_messages_format")
    graph.add_edge("summarize_messages_format", "submit_final_answer")
    graph.add_edge("submit_final_answer", "save")
    graph.add_edge("save", "cleanup_resources")
    graph.add_edge("cleanup_resources", END)
    print__analysis_tracing_debug(
        "106 - FINAL EDGES: Added final edges to completion with cleanup"
    )

    print__analysis_tracing_debug(
        "107 - GRAPH COMPILATION: Starting graph compilation with checkpointer"
    )
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

    # Compile the graph with the checkpointer
    compiled_graph = graph.compile(checkpointer=checkpointer)
    print__analysis_tracing_debug(
        "111 - GRAPH COMPILED: Graph successfully compiled with checkpointer"
    )
    return compiled_graph

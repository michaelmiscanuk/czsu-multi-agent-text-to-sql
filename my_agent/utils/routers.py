"""Routing functions for the LangGraph workflow.

This module contains all routing logic used in the multi-agent text-to-SQL analysis system.
These functions determine the next node to execute based on the current state of the workflow.
"""

from typing import Literal
import sys
import os
from pathlib import Path
from langgraph.graph import END

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

from my_agent.utils.nodes import MAX_ITERATIONS
from my_agent.utils.state import DataAnalysisState

# Import debug functions from utils
from api.utils.debug import print__analysis_tracing_debug


def route_after_sync(state: DataAnalysisState):
    """Route after synchronization based on available data sources.

    Determines the next step after parallel retrieval branches complete.
    Routes to database schema retrieval if selections found, or directly
    to answer formatting if only PDF chunks are available.

    Args:
        state: Current workflow state

    Returns:
        Next node name or END
    """
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
            """
            ❌ ERROR: ChromaDB directory is missing. Please unzip or create the ChromaDB at 'metadata/czsu_chromadb'.
            ❌ Or use ChromaDB on the Cloud - trychroma.com, but need to set .env CHROMA_USE_CLOUD="true" and other ChromaDB vars.
            """
        )
        print__analysis_tracing_debug(
            "95 - CHROMADB ERROR: ChromaDB directory missing, ending execution"
        )
        return END
    else:
        # No database selections found - proceed directly to answer with available PDF chunks
        print("⚠️ No relevant dataset selections found, proceeding with PDF chunks only")
        chunks_available = len(state.get("top_chunks", []))
        print__analysis_tracing_debug(
            f"96 - CHUNKS ONLY ROUTE: No selections found, proceeding with {chunks_available} PDF chunks"
        )
        return "format_answer"


def route_after_query(
    state: DataAnalysisState,
) -> Literal["reflect", "format_answer"]:
    """Route after query generation based on iteration count.

    Controls the reflection loop by checking if maximum iterations
    have been reached. Routes to reflection for improvement or
    directly to answer formatting.

    Args:
        state: Current workflow state

    Returns:
        "reflect" to continue improving, or "format_answer" to finalize
    """
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


def route_after_reflect(
    state: DataAnalysisState,
) -> Literal["generate_query", "format_answer"]:
    """Route after reflection based on reflection decision.

    Analyzes the reflection node's decision to either continue
    improving the query or proceed with answer formatting.

    Args:
        state: Current workflow state

    Returns:
        "generate_query" to generate a better query, or "format_answer" to finalize
    """
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
        return "generate_query"

"""
This script runs an evaluation of the complete selection retrieval functionality using the full pipeline.
It evaluates the two-node sequence: retrieve_similar_selections_hybrid_search_node -> rerank_node
and checks if the correct selection code is retrieved after reranking.

This tests the complete pipeline:
1. Hybrid search (semantic + BM25) retrieval
2. Cohere rerank model reordering
3. Final selection code extraction

Key components:
- example_to_state: converts dataset inputs to the agent's state format
- retry_node_sequence: executes the hybrid search -> rerank sequence with retries
- selection_correct: evaluator function that checks if the top-1 reranked selection matches expected
- selection_in_top_n: evaluator function that checks if expected selection is in top-N reranked results
- aevaluate: runs the evaluation over the dataset with concurrency and experiment tracking
"""

import asyncio
import os

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys
from pathlib import Path

# Handle base directory path
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Remove any existing paths that might conflict
sys.path = [p for p in sys.path if "testing_prototypes_home" not in p]

# Add the parent directory to the Python path so we can import the my_agent module
if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0, str(BASE_DIR)
    )  # Insert at beginning to ensure it's checked first

print(f"BASE_DIR: {BASE_DIR}")
print(f"Python path: {sys.path}")

from my_agent import create_graph

# Temporarily override DEBUG for this notebook execution
original_debug_value = os.environ.get("DEBUG")
os.environ["DEBUG"] = "0"
print(
    f"Temporarily setting DEBUG=0 for this notebook execution (was: {original_debug_value})"
)

import uuid

from langsmith import aevaluate

from my_agent.utils.nodes import (
    relevant_selections_node,
    rerank_node,
    retrieve_similar_selections_hybrid_search_node,
)
from my_agent.utils.state import DataAnalysisState

# ==============================================================================
# INITIALIZATION
# ==============================================================================
# Experiment configuration
EXPERIMENT_CONFIG = {
    "dataset_name": "czsu agent selection retrieval",  # Name of the LangSmith dataset to use
    "experiment_prefix": "full-pipeline-hybrid-rerank",  # Prefix for the experiment run
    "max_concurrency": 4,  # Maximum number of concurrent evaluations
    "evaluators": [
        "selection_correct",
        "selection_in_top_n",
    ],  # List of evaluator functions to use
}

# Create the LangGraph app/graph
app = create_graph()


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
async def selection_correct(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if the top-1 reranked selection matches the expected answer.
    Tests the complete pipeline output after reranking.
    """
    print(f"[Evaluator: correct] outputs: {outputs}")
    print(f"[Evaluator: correct] reference_outputs: {reference_outputs}")

    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    most_similar = outputs.get("most_similar_selections", [])
    expected_selection = reference_outputs.get("answers")
    if expected_selection is not None:
        expected_selection = str(expected_selection).strip().upper()

    actual_selection = None
    if most_similar and len(most_similar[0]) > 0:
        actual_selection = str(most_similar[0][0]).strip().upper()

    result = actual_selection == expected_selection
    print(
        f"[Evaluator: correct] return: {result} (actual: {actual_selection}, expected: {expected_selection})"
    )
    return result


async def selection_in_top_n(outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluator function that checks if the expected selection code is in the top-N reranked selections.
    Tests the complete pipeline output after reranking.
    """
    print(f"[Evaluator: in_top_n] outputs: {outputs}")
    print(f"[Evaluator: in_top_n] reference_outputs: {reference_outputs}")

    # Handle possible wrapping (LangSmith may wrap outputs under 'outputs' key)
    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    most_similar = outputs.get("most_similar_selections", [])
    expected_selection = reference_outputs.get("answers")

    # Normalize expected_selection
    if expected_selection is not None:
        expected_selection = str(expected_selection).strip().upper()

    # Check if expected_selection is in any of the tuples (as the first element)
    found = False
    for tup in most_similar:
        if tup and len(tup) > 0:
            candidate = str(tup[0]).strip().upper()
            if candidate == expected_selection:
                found = True
                break
    print(
        f"[Evaluator: in_top_n] return: {found} (expected: {expected_selection}, candidates: {[t[0] for t in most_similar]})"
    )
    return found


# ==============================================================================
# STATE CONVERSION FUNCTIONS
# ==============================================================================
def example_to_state(inputs: dict) -> dict:
    """
    Converts dataset example inputs into the agent's internal DataAnalysisState format.

    Args:
        inputs (dict): Input dictionary from the dataset, expected to have 'question' key

    Returns:
        DataAnalysisState: Initialized state object for the agent
    """
    return {
        "prompt": inputs["question"],
        "rewritten_prompt": inputs[
            "question"
        ],  # Use the question directly for the pipeline
        "messages": [],
        "iteration": 0,
        "queries_and_results": [],
        "reflection_decision": "",
        "hybrid_search_results": [],
        "most_similar_selections": [],
        "top_selection_codes": [],
    }


# ==============================================================================
# PIPELINE EXECUTION FUNCTIONS
# ==============================================================================
async def retry_node_sequence(inputs, max_attempts=6, wait_seconds=10):
    """Execute the complete two-node sequence: hybrid_search -> rerank

    This function tests the full pipeline:
    1. retrieve_similar_selections_hybrid_search_node: Gets initial candidates via hybrid search
    2. rerank_node: Reranks candidates using Cohere model

    Args:
        inputs: Dataset inputs containing the question
        max_attempts: Maximum retry attempts if empty results
        wait_seconds: Wait time between retries

    Returns:
        dict: Final output containing most_similar_selections with Cohere scores
    """
    state = example_to_state(inputs)
    prompt = state["prompt"]
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"[retry_node_sequence] Attempt {attempt} for prompt: {prompt}")

        # Execute hybrid search first
        hybrid_output = await retrieve_similar_selections_hybrid_search_node(state)

        # Check if hybrid search succeeded
        hybrid_results = hybrid_output.get("hybrid_search_results", [])
        if not hybrid_results:
            print(
                f"[retry_node_sequence] Empty hybrid search output for prompt: {prompt} (attempt {attempt}), waiting {wait_seconds}s before retry..."
            )
            await asyncio.sleep(wait_seconds)
            continue

        print(
            f"[retry_node_sequence] Hybrid search returned {len(hybrid_results)} results, proceeding to rerank..."
        )

        # Execute rerank with hybrid results
        rerank_input = {**state, **hybrid_output}
        final_output = await rerank_node(rerank_input)

        # Check if we got final results
        most_similar = final_output.get("most_similar_selections", [])
        if most_similar:
            print(
                f"[retry_node_sequence] Got {len(most_similar)} reranked results for prompt: {prompt} on attempt {attempt}"
            )
            print(f"[retry_node_sequence] Top 3 results: {most_similar[:3]}")
            return final_output
        else:
            print(
                f"[retry_node_sequence] Empty rerank output for prompt: {prompt} (attempt {attempt}), waiting {wait_seconds}s before retry..."
            )
            await asyncio.sleep(wait_seconds)

    print(
        f"[retry_node_sequence] WARNING: No result for prompt: {prompt} after {max_attempts} attempts!"
    )
    return final_output


async def retry_node(inputs, node, max_attempts=6, wait_seconds=10):
    """Legacy retry function - kept for backward compatibility but now uses new two-node sequence"""
    return await retry_node_sequence(inputs, max_attempts, wait_seconds)


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
if __name__ == "__main__":
    # For demonstration, import or define app and node as needed
    import sys

    print("[INFO] Starting full pipeline evaluation (hybrid search + rerank)...")

    async def node_target(inputs):
        return await retry_node_sequence(inputs)

    async def main():
        experiment_results = await aevaluate(
            node_target,
            data="czsu agent selection retrieval",  # Replace with your dataset name
            evaluators=[selection_correct, selection_in_top_n],
            max_concurrency=4,
            experiment_prefix="full-pipeline-hybrid-rerank",
        )
        print(f"Evaluation results: {experiment_results}")

    asyncio.run(main())

    # Restore original DEBUG value
    print(f"Finished evaluation. Restoring DEBUG environment variable...")
    if original_debug_value is not None:
        os.environ["DEBUG"] = original_debug_value
        print(f"Restored DEBUG to original value: {original_debug_value}")
    else:
        # If it was not set before, remove it
        if "DEBUG" in os.environ:
            del os.environ["DEBUG"]
            print("Removed temporary DEBUG environment variable")

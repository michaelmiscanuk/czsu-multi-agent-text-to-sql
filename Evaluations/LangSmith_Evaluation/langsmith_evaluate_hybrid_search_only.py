"""
This script runs an evaluation of the hybrid search functionality only.
It evaluates only the retrieve_similar_selections_hybrid_search_node
and checks if the correct selection code is retrieved in the hybrid search results.

Key components:
- example_to_state: converts dataset inputs to the agent's state format
- selection_correct_hybrid: evaluator function that checks if the retrieved selection matches expected in hybrid results
- selection_in_top_n_hybrid: evaluator function that checks if expected selection is in top-N hybrid results
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

from my_agent.utils.nodes import retrieve_similar_selections_hybrid_search_node
from my_agent.utils.state import DataAnalysisState

# ==============================================================================
# INITIALIZATION
# ==============================================================================
# Experiment configuration
EXPERIMENT_CONFIG = {
    "dataset_name": "czsu agent selection retrieval",  # Name of the LangSmith dataset to use
    "experiment_prefix": "hybrid-search-only",  # Prefix for the experiment run
    "max_concurrency": 4,  # Maximum number of concurrent evaluations
    "evaluators": [
        "selection_correct_hybrid",
        "selection_in_top_n_hybrid",
    ],  # List of evaluator functions to use
}

# Create the LangGraph app/graph
app = create_graph()


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
async def selection_correct_hybrid(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if the top-1 retrieved selection in hybrid search results matches the expected answer.
    """
    print(f"[Evaluator: correct_hybrid] outputs: {outputs}")
    print(f"[Evaluator: correct_hybrid] reference_outputs: {reference_outputs}")

    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    hybrid_results = outputs.get("hybrid_search_results", [])
    expected_selection = reference_outputs.get("answers")
    if expected_selection is not None:
        expected_selection = str(expected_selection).strip().upper()

    actual_selection = None
    if hybrid_results and len(hybrid_results) > 0:
        # Get the first document's selection from metadata
        first_doc = hybrid_results[0]
        if hasattr(first_doc, "metadata") and first_doc.metadata:
            actual_selection = (
                str(first_doc.metadata.get("selection", "")).strip().upper()
            )

    result = actual_selection == expected_selection
    print(
        f"[Evaluator: correct_hybrid] return: {result} (actual: {actual_selection}, expected: {expected_selection})"
    )
    return result


async def selection_in_top_n_hybrid(outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluator function that checks if the expected selection code is in the top-N hybrid search results.
    """
    print(f"[Evaluator: in_top_n_hybrid] outputs: {outputs}")
    print(f"[Evaluator: in_top_n_hybrid] reference_outputs: {reference_outputs}")

    # Handle possible wrapping (LangSmith may wrap outputs under 'outputs' key)
    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    hybrid_results = outputs.get("hybrid_search_results", [])
    expected_selection = reference_outputs.get("answers")

    # Normalize expected_selection
    if expected_selection is not None:
        expected_selection = str(expected_selection).strip().upper()

    # Check if expected_selection is in any of the hybrid results
    found = False
    candidates = []
    for doc in hybrid_results:
        if hasattr(doc, "metadata") and doc.metadata:
            candidate = str(doc.metadata.get("selection", "")).strip().upper()
            candidates.append(candidate)
            if candidate == expected_selection:
                found = True
                break

    print(
        f"[Evaluator: in_top_n_hybrid] return: {found} (expected: {expected_selection}, candidates: {candidates[:10]})"
    )  # Show first 10 candidates
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
        ],  # For hybrid search, use the question directly
        "messages": [],
        "iteration": 0,
        "queries_and_results": [],
        "reflection_decision": "",
        "hybrid_search_results": [],
        "most_similar_selections": [],
        "top_selection_codes": [],
    }


# --- Retry wrapper for hybrid search evaluation ---
async def retry_hybrid_search(inputs, max_attempts=6, wait_seconds=10):
    """Execute only the hybrid search node"""
    state = example_to_state(inputs)
    prompt = state["prompt"]
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"[retry_hybrid_search] Attempt {attempt} for prompt: {prompt}")

        # Execute hybrid search only
        output = await retrieve_similar_selections_hybrid_search_node(state)

        # Check if hybrid search succeeded
        hybrid_results = output.get("hybrid_search_results", [])
        if hybrid_results:
            print(
                f"[retry_hybrid_search] Got {len(hybrid_results)} results for prompt: {prompt} on attempt {attempt}"
            )
            return output
        else:
            print(
                f"[retry_hybrid_search] Empty hybrid search output for prompt: {prompt} (attempt {attempt}), waiting {wait_seconds}s before retry..."
            )
            await asyncio.sleep(wait_seconds)

    print(
        f"[retry_hybrid_search] WARNING: No result for prompt: {prompt} after {max_attempts} attempts!"
    )
    return output


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
if __name__ == "__main__":
    # For demonstration, import or define app and node as needed
    import sys

    print("[INFO] Starting hybrid search only evaluation...")

    async def node_target(inputs):
        return await retry_hybrid_search(inputs)

    async def main():
        experiment_results = await aevaluate(
            node_target,
            data="czsu agent selection retrieval",  # Replace with your dataset name
            evaluators=[selection_correct_hybrid, selection_in_top_n_hybrid],
            max_concurrency=4,
            experiment_prefix="hybrid-search-only",
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

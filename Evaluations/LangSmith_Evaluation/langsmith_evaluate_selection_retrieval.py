"""
This script runs an evaluation of the selection retrieval functionality of the LangGraph agent.
It evaluates only the first two nodes (retrieve_similar_selections and relevant_selections)
and checks if the correct selection code is retrieved.

Key components:
- example_to_state: converts dataset inputs to the agent's state format
- target_with_config: async wrapper to invoke the agent graph with checkpoint config
- selection_correct: evaluator function that checks if the retrieved selection matches expected
- aevaluate: runs the evaluation over the dataset with concurrency and experiment tracking
"""

#==============================================================================
# IMPORTS
#==============================================================================
import sys
import os
from pathlib import Path

# Handle base directory path
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Remove any existing paths that might conflict
sys.path = [p for p in sys.path if 'testing_prototypes_home' not in p]

# Add the parent directory to the Python path so we can import the my_agent module
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # Insert at beginning to ensure it's checked first

print(f"BASE_DIR: {BASE_DIR}")
print(f"Python path: {sys.path}")

# Temporarily override MY_AGENT_DEBUG for this notebook execution
original_debug_value = os.environ.get('MY_AGENT_DEBUG')
os.environ['MY_AGENT_DEBUG'] = '0'
print(f"Temporarily setting MY_AGENT_DEBUG=0 for this notebook execution (was: {original_debug_value})")

import uuid
from langsmith import aevaluate
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.nodes import retrieve_similar_selections_node, relevant_selections_node

#==============================================================================
# INITIALIZATION
#==============================================================================
# Experiment configuration
EXPERIMENT_CONFIG = {
    "dataset_name": "czsu agent selection retrieval",   # Name of the LangSmith dataset to use
    "experiment_prefix": "selection-retrieval",   # Prefix for the experiment run
    "max_concurrency": 4,            # Maximum number of concurrent evaluations
    "evaluators": ["selection_correct"],   # List of evaluator functions to use
}

#==============================================================================
# EVALUATION FUNCTIONS
#==============================================================================
async def selection_correct(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if the top-1 retrieved selection matches the expected answer.
    """
    print(f"[Evaluator: correct] outputs: {outputs}")
    print(f"[Evaluator: correct] reference_outputs: {reference_outputs}")

    if isinstance(outputs, dict) and 'outputs' in outputs:
        outputs = outputs['outputs']

    most_similar = outputs.get("most_similar_selections", [])
    expected_selection = reference_outputs.get("answers")
    if expected_selection is not None:
        expected_selection = str(expected_selection).strip().upper()

    actual_selection = None
    if most_similar and len(most_similar[0]) > 0:
        actual_selection = str(most_similar[0][0]).strip().upper()

    result = (actual_selection == expected_selection)
    print(f"[Evaluator: correct] return: {result} (actual: {actual_selection}, expected: {expected_selection})")
    return result

async def selection_in_top_n(outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluator function that checks if the expected selection code is in the top-N retrieved selections.
    """
    print(f"[Evaluator: in_top_n] outputs: {outputs}")
    print(f"[Evaluator: in_top_n] reference_outputs: {reference_outputs}")

    # Handle possible wrapping (LangSmith may wrap outputs under 'outputs' key)
    if isinstance(outputs, dict) and 'outputs' in outputs:
        outputs = outputs['outputs']

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
    print(f"[Evaluator: in_top_n] return: {found} (expected: {expected_selection}, candidates: {[t[0] for t in most_similar]})")
    return found

#==============================================================================
# STATE CONVERSION FUNCTIONS
#==============================================================================
def example_to_state(inputs: dict) -> dict:
    """
    Converts dataset example inputs into the agent's internal DataAnalysisState format.

    Args:
        inputs (dict): Input dictionary from the dataset, expected to have 'question' key

    Returns:
        DataAnalysisState: Initialized state object for the agent
    """
    return DataAnalysisState(
        prompt=inputs['question'],
        messages=[],
        result="",
        iteration=0
    )

async def target_with_config(inputs: dict):
    """
    Async wrapper to invoke the agent node with a unique checkpoint thread_id in the config.
    This version only runs the first node and stops.
    """
    state = example_to_state(inputs)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = await retrieve_similar_selections_node(state)
    print("[target_with_config] state after retrieve_similar_selections_node:", state)
    return state

#==============================================================================
# MAIN EVALUATION
#==============================================================================
async def main():
    experiment_results = await aevaluate(
        target_with_config,
        data=EXPERIMENT_CONFIG["dataset_name"],
        evaluators=[selection_correct, selection_in_top_n],
        max_concurrency=EXPERIMENT_CONFIG["max_concurrency"],
        experiment_prefix=EXPERIMENT_CONFIG["experiment_prefix"],
    )

    print(f"Evaluation results: {experiment_results}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
    # Restore original MY_AGENT_DEBUG value
    print(f"Finished evaluation. Restoring MY_AGENT_DEBUG environment variable...")
    if original_debug_value is not None:
        os.environ['MY_AGENT_DEBUG'] = original_debug_value
        print(f"Restored MY_AGENT_DEBUG to original value: {original_debug_value}")
    else:
        # If it was not set before, remove it
        if 'MY_AGENT_DEBUG' in os.environ:
            del os.environ['MY_AGENT_DEBUG']
            print("Removed temporary MY_AGENT_DEBUG environment variable")

"""
This script runs an evaluation of a LangGraph agent on a dataset using LangSmith's async evaluation API.
It converts dataset examples into the agent's internal state format, invokes the agent graph with unique checkpointing,
and uses a custom evaluator that leverages an LLM judge to compare the agent's output against reference answers.

Key components:
- example_to_state: converts dataset inputs to the agent's state format.
- target_with_config: async wrapper to invoke the agent graph with checkpoint config.
- correct: async evaluator function that prompts an LLM to judge correctness of the agent's answer.
- aevaluate: runs the evaluation over the dataset with concurrency and experiment tracking.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys
import os
from pathlib import Path

# Temporarily override DEBUG for this notebook execution
original_debug_value = os.environ.get("DEBUG")
os.environ["DEBUG"] = "0"
print(
    f"Temporarily setting DEBUG=0 for this notebook execution (was: {original_debug_value})"
)

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

import uuid
from langsmith import aevaluate
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.models import get_azure_openai_chat_llm

# ==============================================================================
# INITIALIZATION
# ==============================================================================
# Experiment configuration
EXPERIMENT_CONFIG = {
    "dataset_name": "001_golden_dataset__output_correctness__simple_QA_from_SQL",  # Name of the LangSmith dataset to use
    "experiment_prefix": "test1_judge_4_0__query_gen_gpt-4.1",  # Prefix for the experiment run
    "max_concurrency": 3,  # Maximum number of concurrent evaluations
    "evaluators": ["correctness"],  # List of evaluator functions to use
}

# Get Model
judge_llm = get_azure_openai_chat_llm(
    deployment_name="gpt-4o__test1",
    model_name="gpt-4o",
    openai_api_version="2024-05-01-preview",
    temperature=0.0,
)

# Create the agent graph
graph = create_graph()


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
async def correctness(outputs: dict, reference_outputs: dict) -> bool:
    """
    Evaluator function that uses an LLM to determine if the actual answer contains all information
    from the expected answer.

    Args:
        outputs (dict): The output dictionary from the agent, expected to contain a 'messages' list.
        reference_outputs (dict): The reference outputs from the dataset, expected to have 'answers' key.

    Returns:
        bool: True if the LLM judge determines the answer is correct, False otherwise.
    """
    if not outputs or "messages" not in outputs or not outputs["messages"]:
        return False

    actual_answer = outputs["messages"][-1].content
    expected_answer = reference_outputs.get("answers", "[NO EXPECTED ANSWER PROVIDED]")

    instructions = (
        "Given an actual answer and an expected answer, determine whether"
        " the actual answer contains the information in the"
        " expected answer (). Respond with 'CORRECT' if the actual answer (can be rounded)"
        " does contain the expected information and 'INCORRECT'"
        " otherwise. Do not include anything else in your response."
    )
    user_msg = f"ACTUAL ANSWER: {actual_answer}\n\nEXPECTED ANSWER: {expected_answer}"

    response = await judge_llm.ainvoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg},
        ]
    )
    return response.content.upper() == "CORRECT"


# ==============================================================================
# STATE CONVERSION FUNCTIONS
# ==============================================================================
def example_to_state(inputs: dict) -> dict:
    """
    Converts dataset example inputs into the agent's internal DataAnalysisState format.

    Args:
        inputs (dict): Input dictionary from the dataset, expected to have 'question' key.

    Returns:
        DataAnalysisState: Initialized state object for the agent.
    """
    return DataAnalysisState(
        prompt=inputs["question"], messages=[], result="", iteration=0
    )


async def target_with_config(inputs: dict):
    """
    Async wrapper to invoke the agent graph with a unique checkpoint thread_id in the config.

    Args:
        inputs (dict): Input dictionary from the dataset.

    Returns:
        dict: The output from the agent graph invocation, including 'messages'.
    """
    state = example_to_state(inputs)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return await graph.ainvoke(state, config=config)


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
async def main():
    """Run the evaluation."""
    experiment_results = await aevaluate(
        target_with_config,
        data=EXPERIMENT_CONFIG["dataset_name"],
        evaluators=[correctness],
        max_concurrency=EXPERIMENT_CONFIG["max_concurrency"],
        experiment_prefix=EXPERIMENT_CONFIG["experiment_prefix"],
    )

    print(f"Evaluation results: {experiment_results}")
    return experiment_results


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

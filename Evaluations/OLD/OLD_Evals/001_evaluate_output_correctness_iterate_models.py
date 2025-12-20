"""
This script runs an evaluation of a LangGraph agent on a dataset using LangSmith's async evaluation API.
It iterates through multiple models for a specific node, testing each model configuration one by one.

The script:
1. Loads model configurations from model_configs_all.py
2. Temporarily updates node_models_config.py for each model to test
3. Recreates the graph with the new configuration
4. Runs evaluation for that specific model
5. Dynamically generates experiment names based on judge model, node name, and model id

Key components:
- example_to_state: converts dataset inputs to the agent's state format.
- target_with_config: async wrapper to invoke the agent graph with checkpoint config.
- correct: async evaluator function that prompts an LLM to judge correctness of the agent's answer.
- run_single_experiment: runs evaluation for a single model configuration.
- main: orchestrates iteration through all models for the specified node.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys
import os
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# Temporarily override DEBUG for this notebook execution
original_debug_value = os.environ.get("DEBUG")
os.environ["DEBUG"] = "0"
print(
    f"Temporarily setting DEBUG=0 for this notebook execution (was: {original_debug_value})"
)

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

import uuid
from langsmith import aevaluate
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.models import get_azure_openai_chat_llm
from my_agent.utils.model_configs_all import MODEL_CONFIGS_ALL

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
# Models to evaluate for each node
# Format: {"node_name": ["model_id_1", "model_id_2", ...]}
MODELS_TO_EVALUATE = {
    "generate_query_node": [
        "github_openai_gpt-4.1",
        "github_openai_gpt-4.1-mini",
        # "anthropic_claude-sonnet-4-5-20250929",
        # Add more model IDs from model_configs_all.py as needed
    ],
    # Add more nodes as needed:
    # "format_answer_node": ["azureopenai_gpt-4o-mini", "anthropic_claude-sonnet-4-5-20250929"],
}

# Base experiment configuration
EXPERIMENT_CONFIG = {
    "dataset_name": "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen",  # Name of the LangSmith dataset to use
    "max_concurrency": 2,  # Maximum number of concurrent evaluations
    "evaluators": ["correctness"],  # List of evaluator functions to use
}

# Judge model configuration
JUDGE_CONFIG = {
    "deployment_name": "gpt-4o__test1",
    "model_name": "gpt-4o",
    "model_id": "gpt-4o",  # Simplified ID for judge (used in experiment name)
    "openai_api_version": "2024-05-01-preview",
    "temperature": 0.0,
}

# Get Judge Model
judge_llm = get_azure_openai_chat_llm(
    deployment_name=JUDGE_CONFIG["deployment_name"],
    model_name=JUDGE_CONFIG["model_name"],
    openai_api_version=JUDGE_CONFIG["openai_api_version"],
    temperature=JUDGE_CONFIG["temperature"],
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_model_config_by_id(model_id: str) -> dict:
    """Get model configuration from MODEL_CONFIGS_ALL by id.

    Args:
        model_id: Model identifier (e.g., "mistral_open-mistral-nemo")

    Returns:
        dict: Model configuration dictionary

    Raises:
        ValueError: If model_id not found
    """
    for config in MODEL_CONFIGS_ALL:
        if config.get("id") == model_id:
            return config
    raise ValueError(f"Model ID '{model_id}' not found in MODEL_CONFIGS_ALL")


def create_graph_with_node_config(node_name: str, model_config: dict):
    """Create a graph with a specific node configured to use a specific model.

    This function creates an isolated graph instance with the specified node
    configured to use the given model, without modifying global state.

    Args:
        node_name: Name of the node to configure (e.g., "generate_query_node")
        model_config: Model configuration dictionary from model_configs_all.py

    Returns:
        A compiled StateGraph with the specified configuration
    """
    import importlib
    from my_agent.utils import node_models_config
    import my_agent.agent

    # Store the original configuration
    original_config = node_models_config.NODE_MODELS_CONFIG["nodes"][node_name].copy()

    try:
        # Update the node configuration
        node_models_config.NODE_MODELS_CONFIG["nodes"][node_name] = {
            "model_provider": model_config["model_provider"],
            "model_name": model_config["model_name"],
            "deployment_name": model_config.get("deployment_name", ""),
            "temperature": model_config.get("temperature", 0.0),
            "streaming": model_config.get("streaming", False),
            "openai_api_version": model_config.get(
                "openai_api_version", "2024-05-01-preview"
            ),
            "base_url": model_config.get("base_url", "http://localhost:11434"),
        }

        print(f"✓ Configured {node_name} to use model: {model_config['id']}")

        # Reload the agent module to pick up the new configuration
        importlib.reload(my_agent.agent)

        # Create graph with the reloaded configuration
        graph = create_graph()
        print(f"✓ Graph created with configuration for {model_config['id']}")

        return graph

    finally:
        # Restore original configuration
        node_models_config.NODE_MODELS_CONFIG["nodes"][node_name] = original_config


def generate_experiment_name(judge_id: str, node_name: str, model_id: str) -> str:
    """Generate experiment name in format: judge_[judge_id]__[node_name]__[model_id]

    Args:
        judge_id: Judge model identifier
        node_name: Node being evaluated
        model_id: Model identifier being tested

    Returns:
        str: Formatted experiment name
    """
    return f"judge_{judge_id}__Node_{node_name}__Model_{model_id}"


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


async def target_with_config(inputs: dict, graph):
    """
    Async wrapper to invoke the agent graph with a unique checkpoint thread_id in the config.

    Args:
        inputs (dict): Input dictionary from the dataset.
        graph: The LangGraph graph instance to invoke.

    Returns:
        dict: The output from the agent graph invocation, including 'messages'.
    """
    state = example_to_state(inputs)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return await graph.ainvoke(state, config=config)


# ==============================================================================
# EXPERIMENT EXECUTION
# ==============================================================================
# Use a lock to serialize graph creation while allowing parallel evaluation
graph_creation_lock = asyncio.Lock()


async def run_single_experiment(
    node_name: str, model_id: str, pbar: tqdm = None
) -> dict:
    """Run evaluation for a single model configuration.

    Args:
        node_name: Node being evaluated
        model_id: Model identifier to test
        pbar: Optional tqdm progress bar to update

    Returns:
        dict: Evaluation results
    """
    try:
        if pbar:
            pbar.set_description(f"Starting {node_name}/{model_id}")

        print(f"\n{'='*80}")
        print(f"Starting evaluation: {node_name} with {model_id}")
        print(f"{'='*80}")

        # Get model configuration
        model_config = get_model_config_by_id(model_id)
        print(f"✓ Retrieved config for model: {model_config['id']}")

        # Use lock to serialize graph creation (prevents module reload conflicts)
        async with graph_creation_lock:
            # Create graph with specific node configuration
            # This happens inside a lock but is fast (< 1 second)
            graph = create_graph_with_node_config(node_name, model_config)

        # Generate experiment name
        experiment_name = generate_experiment_name(
            JUDGE_CONFIG["model_id"], node_name, model_id
        )
        print(f"✓ Experiment name: {experiment_name}")

        # Create target function for this specific graph
        async def target_fn(inputs: dict):
            """Target function for evaluation with the current graph."""
            return await target_with_config(inputs, graph)

        # Run evaluation (this is the slow part and runs in parallel)
        if pbar:
            pbar.set_description(f"Evaluating {node_name}/{model_id}")

        print(f"✓ Starting evaluation...")
        experiment_results = await aevaluate(
            target_fn,
            data=EXPERIMENT_CONFIG["dataset_name"],
            evaluators=[correctness],
            max_concurrency=EXPERIMENT_CONFIG["max_concurrency"],
            experiment_prefix=experiment_name,
        )

        print(f"✓ Evaluation completed for {model_id}")
        print(f"  Results: {experiment_results}")

        if pbar:
            pbar.set_description(f"Completed {node_name}/{model_id}")

        return experiment_results
    finally:
        if pbar:
            pbar.update(1)


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
async def main():
    """Run evaluations for all configured models in parallel with isolated configs."""
    print(f"\n{'#'*80}")
    print(f"# EVALUATION SUITE: Model Comparison (PARALLEL EXECUTION)")
    print(f"# Judge Model: {JUDGE_CONFIG['model_id']}")
    print(f"# Dataset: {EXPERIMENT_CONFIG['dataset_name']}")
    print(f"# Note: Graph creation serialized, evaluations run in parallel")
    print(f"{'#'*80}\n")

    all_results = {}
    all_tasks = []

    # Iterate through each node and its models to create task list
    for node_name, model_ids in MODELS_TO_EVALUATE.items():
        print(f"\n{'*'*80}")
        print(f"* Evaluating node: {node_name}")
        print(f"* Models to test: {len(model_ids)}")
        print(f"* Execution mode: PARALLEL")
        print(f"{'*'*80}")

        # Add tasks to the list
        for idx, model_id in enumerate(model_ids, 1):
            print(f"[{idx}/{len(model_ids)}] Queuing model: {model_id}")
            all_tasks.append((node_name, model_id))

    print(
        f"\n🚀 Starting parallel evaluation of {len(all_tasks)} model configurations..."
    )

    # Create progress bar
    with tqdm(total=len(all_tasks), desc="Overall Progress", unit="model") as pbar:
        # Create all experiment tasks - they will run in parallel
        tasks = [
            run_single_experiment(node_name, model_id, pbar)
            for node_name, model_id in all_tasks
        ]

        # Execute all evaluations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process and organize results by node
    for (node_name, model_id), result in zip(all_tasks, results):
        if node_name not in all_results:
            all_results[node_name] = {}

        if isinstance(result, Exception):
            print(f"✗ Error evaluating {node_name}/{model_id}: {result}")
            all_results[node_name][model_id] = {"error": str(result)}
        else:
            print(f"✓ Successfully evaluated {node_name}/{model_id}")
            all_results[node_name][model_id] = result

    print(f"\n{'#'*80}")
    print(f"# EVALUATION SUITE COMPLETED")
    print(f"{'#'*80}\n")
    print(f"Summary:")
    for node_name, node_results in all_results.items():
        success_count = sum(
            1
            for r in node_results.values()
            if not isinstance(r, dict) or "error" not in r
        )
        error_count = sum(
            1 for r in node_results.values() if isinstance(r, dict) and "error" in r
        )
        print(
            f"  {node_name}: {len(node_results)} models evaluated ({success_count} success, {error_count} errors)"
        )

    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())

"""Helper utilities for evaluation scripts."""

import sys
import uuid
import importlib.util
from pathlib import Path
from typing import Any, List
from langsmith import Client
from langsmith.schemas import Example
from httpx import HTTPError
from Evaluations.utils.retry_utils import retry_with_exponential_backoff


def load_module_directly(module_name: str, file_path: Path):
    """Load a Python module directly from file path.

    Useful for bypassing package __init__.py files.

    Args:
        module_name: Name to register the module under in sys.modules
        file_path: Path to the .py file to load

    Returns:
        The loaded module object
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_model_config_by_id(model_id: str, model_configs: List[dict]) -> dict:
    """Get model configuration by ID.

    Args:
        model_id: The ID of the model to find
        model_configs: List of model configuration dictionaries

    Returns:
        dict: The model configuration

    Raises:
        ValueError: If model ID is not found in configs
    """
    for config in model_configs:
        if config.get("id") == model_id:
            return config
    raise ValueError(f"Model ID '{model_id}' not found")


async def get_unevaluated_examples(
    client: Client, experiment_identifier: str, dataset_name: str
) -> List[Example]:
    """Get examples that haven't been evaluated yet in the experiment.

    This function queries the experiment for existing runs and filters
    the dataset to only include examples that haven't been evaluated yet.
    This enables true resume capability without re-evaluating completed examples.

    Args:
        client: LangSmith client instance
        experiment_identifier: Name or ID (UUID) of the experiment
        dataset_name: Name of the dataset

    Returns:
        List of Example objects that haven't been evaluated yet
    """
    # Get all examples from the dataset
    all_examples = list(client.list_examples(dataset_name=dataset_name))

    # Get all runs from the experiment
    # Check if experiment_identifier is a UUID
    import re

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    try:
        if uuid_pattern.match(experiment_identifier):
            # Use project_id for UUID
            existing_runs = list(client.list_runs(project_id=experiment_identifier))
        else:
            # Use project_name for name
            existing_runs = list(client.list_runs(project_name=experiment_identifier))
    except (HTTPError, ValueError, RuntimeError) as e:
        print(f"Could not fetch existing runs: {e}", file=sys.stderr, flush=True)
        # If we can't fetch runs, assume no examples evaluated yet
        return all_examples

    # Get set of example IDs that have already been evaluated
    evaluated_example_ids = {
        run.reference_example_id for run in existing_runs if run.reference_example_id
    }

    # Filter to only unevaluated examples
    unevaluated_examples = [
        ex for ex in all_examples if ex.id not in evaluated_example_ids
    ]

    print(f"Dataset: {len(all_examples)} total examples", file=sys.stderr, flush=True)
    print(
        f"Already evaluated: {len(evaluated_example_ids)} examples",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Remaining to evaluate: {len(unevaluated_examples)} examples",
        file=sys.stderr,
        flush=True,
    )

    return unevaluated_examples


def example_to_agent_state(
    inputs: dict, state_class: Any, input_key: str = "question"
) -> dict:
    """Convert dataset example inputs to agent state format.

    Generic helper for converting LangSmith dataset examples to agent state objects.

    Args:
        inputs: The example inputs dictionary
        state_class: The state class to instantiate (e.g., DataAnalysisState)
        input_key: The key in inputs dict to use as the prompt (default: "question")

    Returns:
        dict: Agent state initialized with the input prompt
    """
    return state_class(prompt=inputs[input_key], messages=[], result="", iteration=0)


@retry_with_exponential_backoff(max_attempts=30, base_delay=1.0, max_delay=300.0)
async def invoke_graph_with_retry(
    inputs: dict, graph: Any, state_class: Any, input_key: str = "question"
) -> Any:
    """Invoke graph with retry logic for rate limiting.

    Generic wrapper for invoking LangGraph graphs with automatic retry on rate limit errors.

    Args:
        inputs: The example inputs dictionary
        graph: The compiled LangGraph graph
        state_class: The state class to use for initialization
        input_key: The key in inputs dict to use as the prompt

    Returns:
        The graph invocation result
    """
    state = example_to_agent_state(inputs, state_class, input_key)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return await graph.ainvoke(state, config=config)

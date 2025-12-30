"""Helper utilities for evaluation scripts."""

import os
import sys
import threading
import time
import uuid
import importlib.util
from pathlib import Path
from typing import Any, List
from langsmith import Client
from langsmith.schemas import Example
from httpx import HTTPError
from tqdm import tqdm
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


@retry_with_exponential_backoff(max_attempts=15, base_delay=2.0, max_delay=60.0)
async def get_unevaluated_examples(
    client: Client, experiment_identifier: str, dataset_name: str
) -> List[Example]:
    """Get examples that haven't been FULLY evaluated yet in the experiment.

    This function queries the experiment for existing runs and filters
    the dataset to only include examples that haven't been fully evaluated yet.
    An example is considered "fully evaluated" only if:
    1. It has a run that completed (has end_time)
    2. AND that run has feedback (was scored by evaluators)

    This enables true resume capability, re-running:
    - Examples that were never started
    - Examples that started but didn't finish (no end_time)
    - Examples that finished but have no feedback (evaluation failed)

    Args:
        client: LangSmith client instance
        experiment_identifier: Name or ID (UUID) of the experiment
        dataset_name: Name of the dataset

    Returns:
        List of Example objects that need to be (re)evaluated
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

    # Get set of example IDs that have been FULLY evaluated:
    # - Run completed (has end_time)
    # - Run has feedback (was scored by evaluators)
    fully_evaluated_example_ids = set()

    for run in existing_runs:
        if not run.reference_example_id:
            continue

        # Check if run completed
        if run.end_time is None:
            continue

        # Check if run has feedback (evaluator ran successfully)
        # feedback_stats is a dict like {"correctness": {"n": 1, "avg": 1.0}}
        has_feedback = (
            hasattr(run, "feedback_stats")
            and run.feedback_stats
            and len(run.feedback_stats) > 0
        )

        if has_feedback:
            fully_evaluated_example_ids.add(run.reference_example_id)

    # Filter to only unevaluated examples
    unevaluated_examples = [
        ex for ex in all_examples if ex.id not in fully_evaluated_example_ids
    ]

    print(f"Dataset: {len(all_examples)} total examples", file=sys.stderr, flush=True)
    print(
        f"Fully evaluated: {len(fully_evaluated_example_ids)} examples (completed + has feedback)",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Need (re)evaluation: {len(unevaluated_examples)} examples",
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


def monitor_progress(
    progress_file: str,
    total_examples: int,
    stop_event: threading.Event,
    num_models: int = 1,
):
    """Monitor progress file and update tqdm progress bar.

    This function runs in a background thread to monitor a progress file
    and display real-time progress using tqdm.

    Args:
        progress_file: Path to the progress tracking file
        total_examples: Total number of examples in the dataset
        stop_event: Event to signal thread shutdown
        num_models: Number of models being evaluated (default: 1)
    """
    with tqdm(
        total=total_examples * num_models,
        desc="Dataset Examples",
        unit="example",
        position=0,
    ) as pbar:
        last_count = 0
        while not stop_event.is_set():
            try:
                if os.path.exists(progress_file):
                    with open(progress_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        current_count = len(lines)
                        if current_count > last_count:
                            pbar.update(current_count - last_count)
                            last_count = current_count
            except (OSError, IOError):
                pass
            time.sleep(0.5)  # Check every 0.5 seconds

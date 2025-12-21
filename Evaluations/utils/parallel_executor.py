"""Parallel execution orchestration utilities for evaluations.

This module provides utilities for running multiple evaluations in parallel
with progress monitoring and result collection.
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Callable
from tqdm import tqdm

from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,
    is_uuid,
    monitor_langsmith_progress,
)
from Evaluations.utils.helpers import monitor_progress


def run_parallel_evaluations(
    model_experiment_map: Dict[str, str],
    evaluation_function: Callable,
    execution_id: str,
    tracker: ExperimentTracker,
    dataset_size: int,
    **kwargs,
) -> List[dict]:
    """Run evaluations in parallel with progress monitoring.

    Args:
        model_experiment_map: Dict mapping model_id -> experiment_identifier
        evaluation_function: Function to call for each evaluation
        execution_id: Execution ID for tracking
        tracker: ExperimentTracker instance
        dataset_size: Approximate dataset size for progress display
        **kwargs: Additional arguments to pass to evaluation_function

    Returns:
        List[dict]: List of evaluation results
    """
    # Create temporary progress file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
        progress_file = tf.name

    # Start progress monitoring threads
    stop_event = threading.Event()

    # Thread 1: Monitor progress file for tqdm updates
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(progress_file, dataset_size, stop_event, len(model_experiment_map)),
        daemon=True,
    )
    monitor_thread.start()

    # Thread 2: Monitor LangSmith for examples_completed updates
    langsmith_monitor_thread = threading.Thread(
        target=monitor_langsmith_progress,
        args=(execution_id, tracker, stop_event),
        daemon=True,
    )
    langsmith_monitor_thread.start()

    # Run evaluations in parallel
    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(model_experiment_map)) as executor:
            # Determine is_resume per model based on identifier format
            futures = [
                executor.submit(
                    evaluation_function,
                    model_id=model_id,
                    experiment_identifier=experiment_identifier,
                    progress_file=progress_file,
                    execution_id=execution_id,
                    tracker=tracker,
                    is_resume=is_uuid(experiment_identifier),
                    **kwargs,
                )
                for model_id, experiment_identifier in model_experiment_map.items()
            ]

            # Collect results with progress bar
            with tqdm(
                total=len(futures), desc="Models Completed", unit="model", position=1
            ) as pbar:
                for future in futures:
                    result = future.result()
                    results.append(result)

                    status = "✓" if result["success"] else "✗"
                    pbar.set_description(f"{status} {result['model_id'][:30]}")
                    pbar.update(1)
    finally:
        # Stop monitoring threads
        stop_event.set()
        monitor_thread.join(timeout=2)
        langsmith_monitor_thread.join(timeout=2)

        # Cleanup progress file
        try:
            os.unlink(progress_file)
        except (OSError, IOError):
            pass

    return results

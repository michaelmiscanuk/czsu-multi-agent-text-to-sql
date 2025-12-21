"""Subprocess execution utilities for evaluation scripts.

This module provides robust subprocess execution with real-time output capture,
metadata parsing, and timeout handling. It's designed for running evaluation
scripts in isolated subprocess environments.
"""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable
from langsmith import Client

from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,
    get_examples_completed_from_langsmith,
    find_experiment_by_prefix,
)


def create_evaluation_environment(
    base_env: dict,
    node_name: str,
    model_id: str,
    dataset_name: str,
    max_concurrency: int,
    judge_model_id: str,
    progress_file: str,
    experiment_identifier: str,
    is_resume: bool,
) -> dict:
    """Create environment variables for evaluation subprocess.

    Args:
        base_env: Base environment dictionary (typically os.environ.copy())
        node_name: Name of the node being evaluated
        model_id: Model ID to evaluate
        dataset_name: Dataset name
        max_concurrency: Maximum concurrent evaluations
        judge_model_id: Judge model ID
        progress_file: Path to progress tracking file
        experiment_identifier: Either experiment UUID (resume) or prefix (new)
        is_resume: True if resuming, False if creating new

    Returns:
        dict: Environment variables for subprocess
    """
    env = base_env.copy()
    env["EVAL_NODE_NAME"] = node_name
    env["EVAL_MODEL_ID"] = model_id
    env["EVAL_DATASET_NAME"] = dataset_name
    env["EVAL_MAX_CONCURRENCY"] = str(max_concurrency)
    env["EVAL_JUDGE_MODEL_ID"] = judge_model_id
    env["EVAL_PROGRESS_FILE"] = progress_file

    # Set experiment identifier based on mode
    if is_resume:
        env["EVAL_EXPERIMENT_NAME"] = experiment_identifier  # Resume existing
    else:
        env["EVAL_EXPERIMENT_PREFIX"] = experiment_identifier  # Create new

    return env


def parse_subprocess_stderr_for_metadata(
    line: str,
) -> Optional[tuple[str, str]]:
    """Parse subprocess stderr line for experiment metadata.

    Args:
        line: Single line from stderr

    Returns:
        tuple: ("name", value) or ("id", value) or None
    """
    line_stripped = line.strip()

    if line_stripped.startswith("EXPERIMENT_NAME: "):
        name = line_stripped.replace("EXPERIMENT_NAME: ", "").strip()
        return ("name", name)
    elif line_stripped.startswith("EXPERIMENT_ID: "):
        exp_id = line_stripped.replace("EXPERIMENT_ID: ", "").strip()
        return ("id", exp_id)

    return None


def create_stderr_reader_thread(
    pipe,
    model_id: str,
    execution_id: str,
    tracker: ExperimentTracker,
    metadata_container: dict,
    output_lines: list,
) -> Callable:
    """Create a stderr reader function for threading.

    Args:
        pipe: Subprocess stderr pipe
        model_id: Model ID being evaluated
        execution_id: Execution ID for tracker
        tracker: ExperimentTracker instance
        metadata_container: Dict to store experiment_name and experiment_id
        output_lines: List to collect stderr lines

    Returns:
        Callable: Function to run in thread
    """

    def read_stderr():
        """Read stderr line by line and update tracker immediately when metadata appears."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                output_lines.append(line)

                # Debug: Show key lines
                line_stripped = line.strip()
                if "EXPERIMENT" in line_stripped or "CREATING" in line_stripped:
                    print(f"[DEBUG {model_id[:15]}] {line_stripped[:80]}", flush=True)

                # Parse metadata
                metadata = parse_subprocess_stderr_for_metadata(line)
                if metadata:
                    key, value = metadata
                    if key == "name":
                        metadata_container["experiment_name"] = value
                        print(f"🔍 Captured NAME: {value}", flush=True)
                    elif key == "id":
                        metadata_container["experiment_id"] = value
                        print(f"🔍 Captured ID: {value}", flush=True)

                # Update tracker as soon as both are available
                if metadata_container.get("experiment_name") and metadata_container.get(
                    "experiment_id"
                ):
                    print("💾 Saving to JSON...", flush=True)
                    tracker.update_model_experiment_metadata(
                        execution_id,
                        model_id,
                        metadata_container["experiment_name"],
                        metadata_container["experiment_id"],
                    )
                    exp_display = metadata_container["experiment_name"] or ""
                    print(f"✓ {model_id[:25]} -> {exp_display}", flush=True)

        except (OSError, IOError, ValueError) as e:
            print(f"❌ Error in read_stderr: {e}", flush=True)
        finally:
            pipe.close()

    return read_stderr


def create_stdout_reader_thread(pipe, output_lines: list) -> Callable:
    """Create a stdout reader function for threading.

    Args:
        pipe: Subprocess stdout pipe
        output_lines: List to collect stdout lines

    Returns:
        Callable: Function to run in thread
    """

    def read_stdout():
        """Read stdout to avoid pipe buffer filling."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                output_lines.append(line)
        except (OSError, IOError, ValueError):
            pass
        finally:
            pipe.close()

    return read_stdout


def fetch_experiment_metadata_for_resume(
    experiment_id: str,
) -> tuple[Optional[str], str]:
    """Fetch experiment name from LangSmith using experiment UUID.

    Args:
        experiment_id: Experiment UUID

    Returns:
        tuple: (experiment_name, experiment_id) - name may be None on error
    """
    try:
        ls_client = Client()
        project = ls_client.read_project(project_id=experiment_id)
        experiment_name = project.name
        exp_display = experiment_name or ""
        print(
            f"🔄 Resuming experiment: {exp_display} (ID: {experiment_id})",
            flush=True,
        )
        return experiment_name, experiment_id
    except Exception as e:
        print(
            f"⚠️ Could not fetch experiment name for UUID {experiment_id}: {e}",
            flush=True,
        )
        return experiment_id, experiment_id  # Fallback to using UUID as name


def wait_and_poll_for_new_experiment(
    experiment_prefix: str,
    initial_wait_seconds: int = 5,
    max_wait_seconds: int = 60,
) -> Optional[dict]:
    """Wait and poll LangSmith API for newly created experiment.

    Args:
        experiment_prefix: Experiment prefix used in aevaluate
        initial_wait_seconds: Seconds to wait before first poll
        max_wait_seconds: Maximum seconds to poll

    Returns:
        dict: {"name": str, "id": str} or None if not found
    """
    print(
        f"🔍 Polling LangSmith for new experiment with prefix: {experiment_prefix[:50]}",
        flush=True,
    )
    time.sleep(initial_wait_seconds)

    experiment_metadata = find_experiment_by_prefix(
        experiment_prefix, max_wait_seconds=max_wait_seconds
    )

    if experiment_metadata:
        print(f"✓ Found experiment via API: {experiment_metadata['name']}", flush=True)
        return experiment_metadata
    else:
        print(
            "⚠️ API polling failed. Waiting for subprocess to report metadata...",
            flush=True,
        )
        return None


def finalize_evaluation_result(
    metadata_container: dict,
    returncode: int,
    stdout: str,
    stderr: str,
    model_id: str,
    experiment_identifier: str,
    execution_id: str,
    tracker: ExperimentTracker,
) -> dict:
    """Finalize evaluation result with LangSmith data and tracker updates.

    Args:
        metadata_container: Dict with experiment_name and experiment_id
        returncode: Subprocess return code
        stdout: Subprocess stdout
        stderr: Subprocess stderr
        model_id: Model ID evaluated
        experiment_identifier: Original experiment identifier
        execution_id: Execution ID for tracker
        tracker: ExperimentTracker instance

    Returns:
        dict: Complete evaluation result
    """
    experiment_name = metadata_container.get("experiment_name")
    experiment_id = metadata_container.get("experiment_id")
    examples_completed = 0

    # Get final examples_completed count from LangSmith
    if experiment_name or experiment_id:
        identifier = experiment_name or experiment_id
        examples_completed = get_examples_completed_from_langsmith(identifier)

        # Update tracker with final count
        status = "completed" if returncode == 0 and "SUCCESS" in stdout else "failed"
        tracker.update_model_status(
            execution_id,
            model_id,
            status,
            examples_completed=examples_completed,
            error=None if returncode == 0 else stderr[:500],
        )

    return {
        "model_id": model_id,
        "experiment_identifier": experiment_identifier,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "success": returncode == 0 and "SUCCESS" in stdout,
        "examples_completed": examples_completed,
    }


def run_evaluation_subprocess(
    python_exe: Path,
    eval_script: Path,
    base_dir: Path,
    model_id: str,
    experiment_identifier: str,
    progress_file: str,
    execution_id: str,
    tracker: ExperimentTracker,
    is_resume: bool,
    node_name: str,
    dataset_name: str,
    max_concurrency: int,
    judge_model_id: str,
    timeout_seconds: int = None,
) -> dict:
    """Run evaluation in subprocess with real-time progress tracking.

    This is the main entry point for subprocess execution. It handles:
    - Environment setup
    - Process spawning with output capture
    - Real-time metadata parsing and tracker updates
    - Timeout handling
    - Final result assembly

    Args:
        python_exe: Path to Python executable
        eval_script: Path to evaluation script
        base_dir: Base directory for subprocess cwd
        model_id: Model ID to evaluate
        experiment_identifier: Either experiment UUID (resume) or prefix (new)
        progress_file: Path to progress tracking file
        execution_id: Current execution ID for tracker updates
        tracker: ExperimentTracker instance for real-time updates
        is_resume: True if resuming existing experiment, False if creating new
        node_name: Name of the node being evaluated
        dataset_name: Dataset name
        max_concurrency: Maximum concurrent evaluations
        judge_model_id: Judge model ID
        timeout_seconds: Subprocess timeout in seconds (default: None = no timeout)

    Returns:
        dict: Evaluation result with status and LangSmith metadata
    """
    # Create environment
    env = create_evaluation_environment(
        os.environ,
        node_name,
        model_id,
        dataset_name,
        max_concurrency,
        judge_model_id,
        progress_file,
        experiment_identifier,
        is_resume,
    )

    # Metadata container and output collectors
    metadata_container = {"experiment_name": None, "experiment_id": None}
    stderr_lines = []
    stdout_lines = []

    try:
        # Mark as in-progress before starting subprocess
        tracker.update_model_status(execution_id, model_id, "in_progress")

        # Spawn subprocess
        process = subprocess.Popen(
            [str(python_exe), str(eval_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(base_dir),
        )

        # Start reader threads
        stderr_reader = create_stderr_reader_thread(
            process.stderr,
            model_id,
            execution_id,
            tracker,
            metadata_container,
            stderr_lines,
        )
        stdout_reader = create_stdout_reader_thread(process.stdout, stdout_lines)

        stderr_thread = threading.Thread(target=stderr_reader)
        stdout_thread = threading.Thread(target=stdout_reader)
        stderr_thread.start()
        stdout_thread.start()

        # Handle metadata discovery based on mode
        if not is_resume:
            # NEW mode: Poll LangSmith API to find newly created experiment
            experiment_metadata = wait_and_poll_for_new_experiment(
                experiment_identifier
            )
            if experiment_metadata:
                metadata_container["experiment_name"] = experiment_metadata["name"]
                metadata_container["experiment_id"] = experiment_metadata["id"]
                tracker.update_model_experiment_metadata(
                    execution_id,
                    model_id,
                    experiment_metadata["name"],
                    experiment_metadata["id"],
                )
        else:
            # RESUME mode: Fetch experiment name from UUID
            experiment_name, experiment_id = fetch_experiment_metadata_for_resume(
                experiment_identifier
            )
            metadata_container["experiment_name"] = experiment_name
            metadata_container["experiment_id"] = experiment_id
            tracker.update_model_experiment_metadata(
                execution_id, model_id, experiment_name, experiment_id
            )

        # Wait for process completion
        returncode = process.wait(timeout=timeout_seconds)

        # Wait for reader threads
        stderr_thread.join(timeout=5)
        stdout_thread.join(timeout=5)

        # Assemble output
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        # Finalize result
        return finalize_evaluation_result(
            metadata_container,
            returncode,
            stdout,
            stderr,
            model_id,
            experiment_identifier,
            execution_id,
            tracker,
        )

    except subprocess.TimeoutExpired:
        tracker.update_model_status(
            execution_id,
            model_id,
            "failed",
            error=f"Evaluation timed out after {timeout_seconds} seconds",
        )
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": metadata_container.get("experiment_name"),
            "experiment_id": metadata_container.get("experiment_id"),
            "returncode": -1,
            "stdout": "",
            "stderr": f"Evaluation timed out after {timeout_seconds} seconds",
            "success": False,
            "examples_completed": 0,
        }

    except (OSError, subprocess.SubprocessError) as e:
        tracker.update_model_status(
            execution_id, model_id, "failed", error=str(e)[:500]
        )
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": metadata_container.get("experiment_name"),
            "experiment_id": metadata_container.get("experiment_id"),
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "examples_completed": 0,
        }

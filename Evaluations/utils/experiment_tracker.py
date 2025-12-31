"""Helper module for tracking experiment executions and resuming evaluations."""

import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langsmith import Client


def generate_short_uuid() -> str:
    """Generate a short UUID (8 characters) for experiment identification.

    Returns:
        str: 8-character lowercase hex string
    """
    return uuid.uuid4().hex[:8]


def generate_execution_id() -> str:
    """Generate unique execution ID with timestamp and short UUID.

    Returns:
        str: Execution ID in format: exec_YYYY-MM-DD_HHMMSS_<short_uuid>
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short_id = generate_short_uuid()
    return f"exec_{timestamp}_{short_id}"


def generate_experiment_prefix(judge_id: str, node_name: str, model_id: str) -> str:
    """Generate experiment prefix for LangSmith (without timestamp/UUID).

    LangSmith automatically appends timestamp and UUID when creating experiments,
    so we only provide the meaningful prefix.

    Args:
        judge_id: ID of the judge model (e.g., "azureopenai_gpt-4o")
        node_name: Name of the node being evaluated (e.g., "format_answer_node")
        model_id: ID of the model being evaluated (e.g., "mistral_mistral-large-2512")

    Returns:
        str: Clean experiment prefix: judge_{judge_id}__node_{node_name}__model_{model_id}
    """
    # Clean up IDs by replacing slashes with underscores
    clean_judge = judge_id.replace("/", "-").replace("_", "-")
    clean_model = model_id.replace("/", "-").replace("_", "-")
    return f"judge-{clean_judge}__node-{node_name}__model-{clean_model}"


class ExperimentTracker:
    """Manages experiment execution tracking and persistence to JSON."""

    def __init__(self, config_file: Path):
        """Initialize tracker with config file path.

        Args:
            config_file: Path to JSON file for storing execution data
        """
        self.config_file = config_file
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing config or create new structure."""
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    # File exists but is empty
                    return {"executions": {}, "latest_execution_id": None}
                return json.loads(content)
        return {"executions": {}, "latest_execution_id": None}

    def save(self):
        """Persist current state to JSON file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def create_execution(
        self,
        node_name: str,
        dataset_name: str,
        judge_model_id: str,
        max_concurrency: int,
        model_ids: List[str],
    ) -> str:
        """Create new execution entry.

        Args:
            node_name: Node being evaluated
            dataset_name: Dataset name
            judge_model_id: Judge model ID
            max_concurrency: Max concurrency setting
            model_ids: List of model IDs to evaluate

        Returns:
            str: Generated execution ID
        """
        execution_id = generate_execution_id()

        self.data["executions"][execution_id] = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "config": {
                "node_name": node_name,
                "dataset_name": dataset_name,
                "judge_model_id": judge_model_id,
                "max_concurrency": max_concurrency,
            },
            "models": {},
        }

        # Initialize each model entry
        for model_id in model_ids:
            experiment_prefix = generate_experiment_prefix(
                judge_model_id, node_name, model_id
            )
            self.data["executions"][execution_id]["models"][model_id] = {
                "status": "pending",
                "experiment_prefix": experiment_prefix,  # Prefix we send to LangSmith
                "experiment_name": None,  # Full name returned by LangSmith
                "experiment_id": None,  # UUID returned by LangSmith
                "examples_completed": 0,
                "error": None,
            }

        self.data["latest_execution_id"] = execution_id
        self.save()
        return execution_id

    def get_latest_execution_id(self) -> Optional[str]:
        """Get the latest execution ID.

        Returns:
            str or None: Latest execution ID if exists
        """
        return self.data.get("latest_execution_id")

    def get_execution(self, execution_id: str) -> Optional[dict]:
        """Get execution data by ID.

        Args:
            execution_id: Execution ID to retrieve

        Returns:
            dict or None: Execution data if exists
        """
        return self.data["executions"].get(execution_id)

    def get_incomplete_models(self, execution_id: str) -> Dict[str, str]:
        """Get models that are not completed for given execution.

        Args:
            execution_id: Execution ID to check

        Returns:
            dict: Mapping of model_id -> experiment_prefix
                  Always returns experiment_prefix so we can search for the actual experiment
        """
        execution = self.get_execution(execution_id)
        if not execution:
            return {}

        incomplete = {}
        for model_id, model_data in execution["models"].items():
            if model_data["status"] != "completed":
                # Always return the prefix - parent will search LangSmith to find the actual experiment
                incomplete[model_id] = model_data["experiment_prefix"]

        return incomplete

    def update_model_status(
        self,
        execution_id: str,
        model_id: str,
        status: str,
        examples_completed: int = 0,
        error: Optional[str] = None,
    ):
        """Update status of a specific model in an execution.

        Args:
            execution_id: Execution ID
            model_id: Model ID to update
            status: New status ("pending", "in_progress", "completed", "failed")
            examples_completed: Number of examples completed
            error: Error message if failed
        """
        if execution_id in self.data["executions"]:
            if model_id in self.data["executions"][execution_id]["models"]:
                self.data["executions"][execution_id]["models"][model_id].update(
                    {
                        "status": status,
                        "examples_completed": examples_completed,
                        "error": error,
                    }
                )
                self.save()

    def update_model_experiment_metadata(
        self,
        execution_id: str,
        model_id: str,
        experiment_name: str,
        experiment_id: str,
    ):
        """Update experiment metadata after LangSmith creates it.

        Args:
            execution_id: Execution ID
            model_id: Model ID to update
            experiment_name: The human-readable experiment name from LangSmith
            experiment_id: The UUID experiment ID from LangSmith
        """
        if execution_id in self.data["executions"]:
            if model_id in self.data["executions"][execution_id]["models"]:
                model_data = self.data["executions"][execution_id]["models"][model_id]
                model_data["experiment_name"] = experiment_name
                model_data["experiment_id"] = experiment_id
                self.save()

    def update_execution_status(self, execution_id: str, status: str):
        """Update overall execution status.

        Args:
            execution_id: Execution ID
            status: New status ("in_progress", "completed", "partial")
        """
        if execution_id in self.data["executions"]:
            self.data["executions"][execution_id]["status"] = status
            self.save()

    def get_all_executions(self) -> List[dict]:
        """Get list of all executions sorted by timestamp (newest first).

        Returns:
            list: List of execution data dictionaries
        """
        executions = list(self.data["executions"].values())
        executions.sort(key=lambda x: x["timestamp"], reverse=True)
        return executions


def get_examples_completed_from_langsmith(experiment_identifier: str) -> int:
    """Query LangSmith to get actual number of examples completed in experiment.

    Implements retry logic with exponential backoff to handle race conditions where
    experiments are queried immediately after creation but not yet fully indexed.

    Args:
        experiment_identifier: Name or ID (UUID) of the experiment

    Returns:
        int: Number of examples with completed runs (finished, not pending)
    """
    import time

    if not experiment_identifier:
        print(
            "[get_examples_completed] No experiment identifier provided",
            file=sys.stderr,
        )
        return 0

    # Retry configuration for newly created experiments
    max_attempts = 5
    base_delay = 2.0  # Start with 2 seconds
    max_delay = 30.0

    for attempt in range(max_attempts):
        try:
            # Check if experiment_identifier is a UUID
            is_experiment_uuid = is_uuid(experiment_identifier)

            client = Client()

            if is_experiment_uuid:
                runs = list(client.list_runs(project_id=experiment_identifier))
            else:
                runs = list(client.list_runs(project_name=experiment_identifier))

            # Count unique example IDs where run has finished (has end_time)
            # A run is complete when it has an end_time set
            unique_examples = {
                run.reference_example_id
                for run in runs
                if run.reference_example_id and run.end_time is not None
            }
            completed_count = len(unique_examples)

            print(
                f"[get_examples_completed] {experiment_identifier[:50]}: {completed_count} examples completed",
                file=sys.stderr,
            )

            return completed_count

        except Exception as e:
            error_name = type(e).__name__
            is_not_found_error = (
                "NotFound" in error_name or "not found" in str(e).lower()
            )

            if is_not_found_error and attempt < max_attempts - 1:
                # Exponential backoff with jitter for "not found" errors
                delay = min(base_delay * (2**attempt), max_delay)
                jitter = delay * 0.1 * (0.5 - time.time() % 1)  # ±10% jitter
                wait_time = delay + jitter

                print(
                    f"[get_examples_completed] Attempt {attempt + 1}/{max_attempts}: "
                    f"Project not found yet, retrying in {wait_time:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue

            # Non-retryable error or max attempts reached
            print(
                f"[get_examples_completed] Error querying {experiment_identifier[:50]}: {error_name}: {e}",
                file=sys.stderr,
            )
            return 0

    # Should not reach here, but fallback
    return 0


def find_experiment_by_prefix(
    experiment_prefix: str, max_wait_seconds: int = 60
) -> Optional[dict]:
    """Poll LangSmith API to find newly created experiment by prefix.

    When aevaluate() is called with experiment_prefix, LangSmith creates a new
    experiment with that prefix plus a timestamp and UUID. This function polls
    the API to find that experiment shortly after creation.

    Args:
        experiment_prefix: The prefix used when creating the experiment
        max_wait_seconds: Maximum time to wait for experiment to appear

    Returns:
        dict: Experiment metadata with keys {"name": str, "id": str} or None
    """
    client = Client()
    start_time = time.time()
    poll_interval = 2

    print(
        f"[find_experiment] Searching for experiments starting with: {experiment_prefix}",
        file=sys.stderr,
    )

    while time.time() - start_time < max_wait_seconds:
        try:
            # List all projects (experiments) and find matches
            # Get more projects to increase chance of finding recent ones
            projects = list(client.list_projects(limit=200))

            print(
                f"[find_experiment] Retrieved {len(projects)} projects from LangSmith",
                file=sys.stderr,
            )

            for project in projects:
                # Check if project name starts with our prefix
                if project.name and project.name.startswith(experiment_prefix):
                    # Found a match! Return metadata
                    print(
                        f"[find_experiment] ✓ FOUND: {project.name} (ID: {project.id})",
                        file=sys.stderr,
                    )
                    return {"name": project.name, "id": str(project.id)}

            # If not found, show a sample of recent project names for debugging
            if projects:
                sample_names = [p.name[:80] for p in projects[:5]]
                print(
                    f"[find_experiment] Sample of recent projects: {sample_names}",
                    file=sys.stderr,
                )

            # Wait before retrying
            time.sleep(poll_interval)

        except Exception as e:
            print(f"[find_experiment] Error querying LangSmith: {e}", file=sys.stderr)
            time.sleep(poll_interval)

    print(
        f"[find_experiment] ✗ Not found after {max_wait_seconds}s: {experiment_prefix}",
        file=sys.stderr,
    )
    return None  # Not found within timeout


def monitor_langsmith_progress(
    execution_id: str, tracker: "ExperimentTracker", stop_event: threading.Event
):
    """Periodically query LangSmith for examples_completed and update tracker.

    This function runs in a background thread to monitor the progress of
    in-progress evaluations by querying LangSmith for completion counts.

    Args:
        execution_id: Current execution ID
        tracker: ExperimentTracker instance
        stop_event: Event to signal thread shutdown
    """
    while not stop_event.is_set():
        try:
            execution = tracker.get_execution(execution_id)
            if not execution:
                break

            # Check each model's experiment for progress
            for model_id, model_data in execution["models"].items():
                # Try experiment_name first, fallback to experiment_id
                experiment_identifier = model_data.get(
                    "experiment_name"
                ) or model_data.get("experiment_id")
                if experiment_identifier and model_data["status"] == "in_progress":
                    # Query LangSmith for current count
                    examples_completed = get_examples_completed_from_langsmith(
                        experiment_identifier
                    )
                    # Only update if count changed
                    if examples_completed != model_data.get("examples_completed", 0):
                        tracker.update_model_status(
                            execution_id,
                            model_id,
                            "in_progress",
                            examples_completed=examples_completed,
                        )

        except (OSError, IOError, KeyError, ValueError):
            # Silently continue on any tracker or network error
            pass

        # Check every 30 seconds (increased from 20 to reduce API calls and avoid rate limiting)
        time.sleep(30)


def is_uuid(text: str) -> bool:
    """Check if text is a UUID.

    Args:
        text: Text to check

    Returns:
        bool: True if text matches UUID format
    """
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    return bool(uuid_pattern.match(text))


def prepare_resume_execution(
    tracker: "ExperimentTracker",
    execution_id: Optional[str] = None,
) -> Optional[Tuple[str, Dict[str, str], dict]]:
    """Prepare execution for resume mode.

    This function:
    1. Gets the execution to resume (latest or specific ID)
    2. Retrieves incomplete models
    3. Searches LangSmith for existing experiments
    4. Updates model_experiment_map with UUIDs for resuming

    Args:
        tracker: ExperimentTracker instance
        execution_id: Specific execution ID to resume, or None for latest

    Returns:
        tuple: (execution_id, model_experiment_map, execution_data) or None if no execution to resume
    """
    # Get execution ID
    if not execution_id:
        execution_id = tracker.get_latest_execution_id()

    if not execution_id:
        return None

    # Get execution data
    execution = tracker.get_execution(execution_id)
    if not execution:
        return None

    # Get incomplete models
    model_experiment_map = tracker.get_incomplete_models(execution_id)

    if not model_experiment_map:
        return execution_id, {}, execution  # All completed

    # Search LangSmith for existing experiments
    for model_id, experiment_prefix in list(model_experiment_map.items()):
        # Search LangSmith for experiment matching this prefix
        experiment_metadata = find_experiment_by_prefix(
            experiment_prefix, max_wait_seconds=10
        )

        if experiment_metadata:
            # Found it! Update the map to use the UUID for resume
            model_experiment_map[model_id] = experiment_metadata["id"]

            # Update tracker with correct metadata
            tracker.update_model_experiment_metadata(
                execution_id,
                model_id,
                experiment_metadata["name"],
                experiment_metadata["id"],
            )
        # else: Keep the prefix - will create new experiment

    return execution_id, model_experiment_map, execution


def prepare_new_execution(
    tracker: "ExperimentTracker",
    node_name: str,
    dataset_name: str,
    judge_model_id: str,
    max_concurrency: int,
    model_ids: List[str],
) -> Tuple[str, Dict[str, str], dict]:
    """Prepare new execution.

    Args:
        tracker: ExperimentTracker instance
        node_name: Node being evaluated
        dataset_name: Dataset name
        judge_model_id: Judge model ID
        max_concurrency: Max concurrency setting
        model_ids: List of model IDs to evaluate

    Returns:
        tuple: (execution_id, model_experiment_map, execution_data)
    """
    # Create new execution
    execution_id = tracker.create_execution(
        node_name=node_name,
        dataset_name=dataset_name,
        judge_model_id=judge_model_id,
        max_concurrency=max_concurrency,
        model_ids=model_ids,
    )

    # Get execution data
    execution = tracker.get_execution(execution_id)

    # Build model_experiment_map with prefixes
    model_experiment_map = {
        model_id: model_data["experiment_prefix"]
        for model_id, model_data in execution["models"].items()
    }

    return execution_id, model_experiment_map, execution

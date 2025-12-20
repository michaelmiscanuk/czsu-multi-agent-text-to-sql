"""Helper module for tracking experiment executions and resuming evaluations."""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
                return json.load(f)
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
            dict: Mapping of model_id -> experiment_name (or prefix if name not set)
                  For resuming: use experiment_name if exists (continues that experiment)
                  Otherwise use experiment_prefix (creates new experiment)
        """
        execution = self.get_execution(execution_id)
        if not execution:
            return {}

        incomplete = {}
        for model_id, model_data in execution["models"].items():
            if model_data["status"] != "completed":
                # If experiment_name exists, return it to RESUME that experiment
                # Otherwise return prefix to CREATE new experiment
                incomplete[model_id] = (
                    model_data.get("experiment_name") or model_data["experiment_prefix"]
                )

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


def get_examples_completed_from_langsmith(experiment_name: str) -> int:
    """Query LangSmith to get actual number of examples completed in experiment.

    Args:
        experiment_name: Name or ID of the experiment

    Returns:
        int: Number of examples with completed runs
    """
    if not experiment_name:
        return 0

    try:
        client = Client()
        runs = list(client.list_runs(project_name=experiment_name))
        # Count unique example IDs (reference_example_id)
        unique_examples = {
            run.reference_example_id for run in runs if run.reference_example_id
        }
        return len(unique_examples)
    except Exception as e:
        print(
            f"Warning: Could not query LangSmith for examples count: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 0

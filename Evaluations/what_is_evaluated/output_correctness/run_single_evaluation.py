"""Subprocess runner for single model evaluation with isolated configuration."""

import sys
import os
import io

# Windows console encoding fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
from pathlib import Path
import functools
from langsmith import aevaluate, Client
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["DEBUG"] = "0"

# Configure logging - suppress INFO logs to keep stderr clean
# Only show WARNING and above so we can see our EXPERIMENT_NAME/ERROR messages
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Project root
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

sys.path.insert(0, str(BASE_DIR))

# Import utilities
from Evaluations.utils.helpers import (
    load_module_directly,
    get_model_config_by_id,
    get_unevaluated_examples,
    invoke_graph_with_retry,
)
from Evaluations.utils.evaluators_custom import correctness_evaluator

# Environment config
NODE_NAME = os.environ.get("EVAL_NODE_NAME", "generate_query_node")
MODEL_ID = os.environ.get("EVAL_MODEL_ID", "")
DATASET_NAME = os.environ.get(
    "EVAL_DATASET_NAME",
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen",
)
MAX_CONCURRENCY = int(os.environ.get("EVAL_MAX_CONCURRENCY", "2"))
JUDGE_MODEL_ID = os.environ.get("EVAL_JUDGE_MODEL_ID", "azureopenai_gpt-4o")
EXPERIMENT_PREFIX = os.environ.get(
    "EVAL_EXPERIMENT_PREFIX", None
)  # Prefix for new experiments
EXPERIMENT_NAME = os.environ.get(
    "EVAL_EXPERIMENT_NAME", None
)  # Existing experiment name/ID to resume

if not MODEL_ID:
    sys.exit(1)


# Load config modules directly (bypass package __init__.py)
model_configs_path = BASE_DIR / "my_agent" / "utils" / "model_configs_all.py"
model_configs_module = load_module_directly(
    "my_agent.utils.model_configs_all", model_configs_path
)
MODEL_CONFIGS_ALL = model_configs_module.MODEL_CONFIGS_ALL

node_models_config_path = BASE_DIR / "my_agent" / "utils" / "node_models_config.py"
node_models_config = load_module_directly(
    "my_agent.utils.node_models_config", node_models_config_path
)

# Set model config before importing my_agent
model_config = get_model_config_by_id(MODEL_ID, MODEL_CONFIGS_ALL)

new_config = {
    "model_provider": model_config["model_provider"],
    "model_name": model_config["model_name"],
    "deployment_name": model_config.get("deployment_name", ""),
    "temperature": model_config.get("temperature", 0.0),
    "streaming": model_config.get("streaming", False),
    "openai_api_version": model_config.get("openai_api_version", "2024-05-01-preview"),
    "base_url": model_config.get("base_url", "http://localhost:11434"),
}
node_models_config.NODE_MODELS_CONFIG["nodes"][NODE_NAME] = new_config

# Import my_agent modules after config is set
from my_agent import create_graph
from my_agent.utils.state import DataAnalysisState
from my_agent.utils.models import get_azure_openai_chat_llm


def generate_experiment_name(judge_id: str, node_name: str, model_id: str) -> str:
    """Generate experiment name with short UUID suffix.

    This function is kept for backward compatibility but should use the
    EXPERIMENT_NAME from environment when available.

    Args:
        judge_id: Judge model ID
        node_name: Node name
        model_id: Model ID

    Returns:
        str: Experiment name in format: judge_{judge_id}__Node_{node_name}__Model_{model_id}
    """
    # Note: UUID suffix is now added by parent script
    return f"judge_{judge_id}__Node_{node_name}__Model_{model_id}"


# Judge model - lookup config from JUDGE_MODEL_ID
judge_config = get_model_config_by_id(JUDGE_MODEL_ID, MODEL_CONFIGS_ALL)
judge_llm = get_azure_openai_chat_llm(
    deployment_name=judge_config["deployment_name"],
    model_name=judge_config["model_name"],
    openai_api_version=judge_config.get("openai_api_version", "2024-05-01-preview"),
    temperature=judge_config.get("temperature", 0.0),
)
# Note: judge_llm already has .with_retry(stop_after_attempt=30) from get_azure_openai_chat_llm

# Create correctness evaluator with judge_llm bound
correctness = functools.partial(correctness_evaluator, judge_llm=judge_llm)
# Preserve function metadata for LangSmith
correctness.__name__ = "Correctness"
correctness.__doc__ = "LLM judge evaluator for answer correctness"


async def run_evaluation():
    """Run evaluation silently (output captured by parent process)."""
    # Create graph
    graph = create_graph()

    # Get progress file path from env (if provided by parent)
    progress_file = os.environ.get("EVAL_PROGRESS_FILE")

    # Create target function with progress tracking
    async def target_fn(inputs: dict):
        result = await invoke_graph_with_retry(
            inputs, graph, DataAnalysisState, "question"
        )
        # Signal completion to parent process via file
        if progress_file and os.path.exists(progress_file):
            try:
                with open(progress_file, "a", encoding="utf-8") as f:
                    f.write("1\n")
            except (OSError, IOError):
                pass  # Silently fail if file write fails
        return result

    # Create client
    client = Client()

    # Check if resuming existing experiment or creating new one
    if EXPERIMENT_NAME:
        # RESUME MODE: Continue existing experiment, filtering already-evaluated examples
        # EXPERIMENT_NAME can be either:
        # 1. Experiment UUID (preferred for resume)
        # 2. Experiment name (fallback - we'll need to query LangSmith)

        print(f"RESUMING: {EXPERIMENT_NAME}", file=sys.stderr, flush=True)

        # Check if EXPERIMENT_NAME is a UUID (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        # If UUID, fetch the TracerSession object; otherwise use name as-is
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        if uuid_pattern.match(EXPERIMENT_NAME):
            # It's a UUID - fetch the TracerSession object
            print(
                f"Fetching experiment by UUID: {EXPERIMENT_NAME}",
                file=sys.stderr,
                flush=True,
            )
            existing_experiment = client.read_project(project_id=EXPERIMENT_NAME)
            experiment_identifier_for_filter = EXPERIMENT_NAME
            print(
                f"Found experiment: {existing_experiment.name}",
                file=sys.stderr,
                flush=True,
            )
        else:
            # It's a name - fetch by name
            print(
                f"Fetching experiment by name: {EXPERIMENT_NAME}",
                file=sys.stderr,
                flush=True,
            )
            existing_experiment = client.read_project(project_name=EXPERIMENT_NAME)
            experiment_identifier_for_filter = EXPERIMENT_NAME

        # Get only unevaluated examples
        unevaluated_examples = await get_unevaluated_examples(
            client, experiment_identifier_for_filter, DATASET_NAME
        )

        if not unevaluated_examples:
            print("All examples already evaluated!", file=sys.stderr, flush=True)
            print(
                f"EXPERIMENT_NAME: {existing_experiment.name}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"EXPERIMENT_ID: {existing_experiment.id}",
                file=sys.stderr,
                flush=True,
            )
            print("SUCCESS", flush=True)
            return

        # Pass the TracerSession object (not string) to continue existing experiment
        experiment_results = await aevaluate(
            target_fn,
            data=unevaluated_examples,  # Only pass unevaluated examples
            evaluators=[correctness],
            max_concurrency=MAX_CONCURRENCY,
            experiment=existing_experiment,  # Pass TracerSession object to continue existing experiment
        )

        # Print metadata IMMEDIATELY for resume mode too
        experiment_name = experiment_results.experiment_name
        print(f"EXPERIMENT_NAME: {experiment_name}", file=sys.stderr, flush=True)

        experiment_id = str(experiment_results._manager._experiment.id)
        print(f"EXPERIMENT_ID: {experiment_id}", file=sys.stderr, flush=True)

    else:
        # NEW MODE: Create new experiment with prefix
        if not EXPERIMENT_PREFIX:
            from Evaluations.utils.experiment_tracker import (
                generate_experiment_prefix,
            )

            experiment_prefix = generate_experiment_prefix(
                JUDGE_MODEL_ID, NODE_NAME, MODEL_ID
            )
        else:
            experiment_prefix = EXPERIMENT_PREFIX

        print(f"CREATING: {experiment_prefix}", file=sys.stderr, flush=True)
        print(f"DEBUG: Calling aevaluate...", file=sys.stderr, flush=True)
        experiment_results = await aevaluate(
            target_fn,
            data=DATASET_NAME,  # Use full dataset for new experiments
            evaluators=[correctness],
            max_concurrency=MAX_CONCURRENCY,
            experiment_prefix=experiment_prefix,  # LangSmith appends timestamp/UUID
        )
        print(f"DEBUG: aevaluate returned successfully", file=sys.stderr, flush=True)

    # For NEW experiments only: Print metadata IMMEDIATELY after aevaluate returns
    # For RESUME: Parent already has the metadata, don't print anything to avoid confusion
    if not EXPERIMENT_NAME:  # NEW mode
        print(
            f"DEBUG: About to access experiment_results attributes",
            file=sys.stderr,
            flush=True,
        )
        try:
            print(
                f"DEBUG: Getting experiment_name property...",
                file=sys.stderr,
                flush=True,
            )
            experiment_name = experiment_results.experiment_name
            print(f"EXPERIMENT_NAME: {experiment_name}", file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f"DEBUG: Error accessing experiment_name: {e}",
                file=sys.stderr,
                flush=True,
            )
            raise

        try:
            print(
                f"DEBUG: Getting experiment_id property...",
                file=sys.stderr,
                flush=True,
            )
            experiment_id = str(experiment_results._manager._experiment.id)
            print(f"EXPERIMENT_ID: {experiment_id}", file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f"DEBUG: Error accessing experiment_id: {e}",
                file=sys.stderr,
                flush=True,
            )
            raise
    else:  # RESUME mode - don't print, parent already knows
        experiment_name = experiment_results.experiment_name
        experiment_id = str(experiment_results._manager._experiment.id)

    # Wrap everything in try/except to capture metadata even on failure
    try:
        print(f"DEBUG: Starting result consumption...", file=sys.stderr, flush=True)
        # Consume all results to ensure evaluation completes
        result_list = [r async for r in experiment_results]
        print(f"EVALUATED: {len(result_list)} examples", file=sys.stderr, flush=True)

        # Signal successful completion to parent
        print("SUCCESS", flush=True)

    except Exception as e:
        # Even on failure, try to output what we have
        print(
            f"ERROR: {type(e).__name__}: {str(e)}",
            file=sys.stderr,
            flush=True,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

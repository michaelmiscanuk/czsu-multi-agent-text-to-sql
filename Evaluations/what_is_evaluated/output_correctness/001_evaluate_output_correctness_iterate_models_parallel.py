"""
This script runs evaluations of a LangGraph agent using SUBPROCESS ISOLATION.

Each model evaluation runs in a completely separate Python process, ensuring
complete isolation of model configurations. This prevents any interference
between concurrent evaluations.

The script:
1. Spawns a separate subprocess for each model configuration
2. Each subprocess has its own Python interpreter and isolated global state
3. Subprocesses run in parallel for speed
4. Results are collected from LangSmith after completion

Key components:
- run_single_evaluation.py: The subprocess worker that runs one evaluation
- main: Orchestrates parallel subprocess execution
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tempfile
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

from langsmith import Client
from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,
    get_examples_completed_from_langsmith,
)

# Subprocess script and Python executable
SINGLE_EVAL_SCRIPT = Path(__file__).parent / "run_single_evaluation.py"
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

# Evaluation configuration
NODE_NAME = "format_answer_node"
DATASET_NAME = "001d_golden_dataset__output_correctness__simple_QA_from_SQL_reduced"
MAX_CONCURRENCY = 1
JUDGE_MODEL_ID = "azureopenai_gpt-4.1"

# ============================================================================
# PROGRESS MONITORING CONFIGURATION
# ============================================================================
# Approximate dataset size for progress bar display only
# Note: This is a dummy value since parent script only manages parallel execution and doesn't authenticate with LangSmith
DATASET_SIZE = 3  # Dummy value for monitoring visualization
# ============================================================================

# ============================================================================
# EXECUTION MODE CONFIGURATION
# ============================================================================
# Mode: "new" - Start new evaluation run
#       "resume" - Resume from previous execution
EXECUTION_MODE = "new"  # Change to "resume" to continue previous run

# Execution ID to resume (None = auto-resume latest)
# Set to specific ID like "exec_2025-12-20_103045_a1b2c3d4" to resume that run
RESUME_EXECUTION_ID = None

# Config file for tracking executions
CONFIG_FILE = Path(__file__).with_suffix(".json")
# ============================================================================

# Models to evaluate
MODELS_TO_EVALUATE = [
    "mistral_mistral-large-2512",
    "mistral_devstral-2512",
    # "mistral_codestral-2508",
    # "azureopenai_gpt-4o",
    # "azureopenai_gpt-4o-mini",
    # "azureopenai_gpt-4.1",
    # "azureopenai_gpt-5-nano",
    # "azureopenai_gpt-5.2-chat",
    # "xai_grok-4-1-fast-reasoning",
    # "xai_grok-4-1-fast-non-reasoning",
    # "gemini_gemini-3-pro-preview",
]

# # Models to evaluate
# MODELS_TO_EVALUATE = [
#     # "mistral_mistral-large-2512",
#     "mistral_devstral-2512",
#     "mistral_codestral-2508",
#     "azureopenai_gpt-4o",
#     "azureopenai_gpt-4o-mini",
#     # "azureopenai_gpt-4.1",
#     # "azureopenai_gpt-5-nano",
#     # "azureopenai_gpt-5.2-chat",
#     "xai_grok-4-1-fast-reasoning",
#     # "xai_grok-4-1-fast-non-reasoning",
#     "gemini_gemini-3-pro-preview",
# ]

# Models to evaluate
# MODELS_TO_EVALUATE = [
#     "mistral_mistral-large-2512",
#     # "mistral_devstral-2512",
#     # "mistral_codestral-2508",
#     # "azureopenai_gpt-4o",
#     # "azureopenai_gpt-4o-mini",
#     "azureopenai_gpt-4.1",
#     "azureopenai_gpt-5-nano",
#     "azureopenai_gpt-5.2-chat",
#     # "xai_grok-4-1-fast-reasoning",
#     "xai_grok-4-1-fast-non-reasoning",
#     # "gemini_gemini-3-pro-preview",
# ]


def run_subprocess_evaluation(
    model_id: str, experiment_identifier: str, progress_file: str
) -> dict:
    """Run single evaluation in subprocess.

    Args:
        model_id: Model ID to evaluate
        experiment_identifier: Either experiment_name (resume) or experiment_prefix (new)
        progress_file: Path to progress tracking file

    Returns:
        dict: Evaluation result with status and LangSmith metadata
    """
    # Environment variables for subprocess
    env = os.environ.copy()
    env["EVAL_NODE_NAME"] = NODE_NAME
    env["EVAL_MODEL_ID"] = model_id
    env["EVAL_DATASET_NAME"] = DATASET_NAME
    env["EVAL_MAX_CONCURRENCY"] = str(MAX_CONCURRENCY)
    env["EVAL_JUDGE_MODEL_ID"] = JUDGE_MODEL_ID
    env["EVAL_PROGRESS_FILE"] = progress_file

    # Determine if this is resume (has timestamp in name) or new (clean prefix)
    if any(ts in experiment_identifier for ts in ["-202", "-203"]):  # Has timestamp
        env["EVAL_EXPERIMENT_NAME"] = experiment_identifier  # Resume existing
    else:
        env["EVAL_EXPERIMENT_PREFIX"] = experiment_identifier  # Create new

    try:
        result = subprocess.run(
            [str(PYTHON_EXE), str(SINGLE_EVAL_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid characters instead of crashing
            cwd=str(BASE_DIR),
            check=False,
            timeout=600,  # 10 minute timeout per model
        )

        # Safely handle stdout/stderr which might be None
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Extract LangSmith-generated metadata from stderr
        experiment_name = None
        experiment_id = None
        examples_completed = 0

        # Parse stderr line by line
        for line in stderr.split("\n"):
            line = line.strip()
            if line.startswith("EXPERIMENT_NAME: "):
                experiment_name = line.replace("EXPERIMENT_NAME: ", "").strip()
            elif line.startswith("EXPERIMENT_ID: "):
                experiment_id = line.replace("EXPERIMENT_ID: ", "").strip()
            elif "Remaining to evaluate: " in line:
                # Extract number of examples to be evaluated (informational only)
                try:
                    parts = line.split("Remaining to evaluate: ")
                    if len(parts) > 1:
                        # Parse remaining count (not currently used, kept for future tracking)
                        _ = int(parts[1].split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": experiment_name,  # LangSmith-generated human name
            "experiment_id": experiment_id,  # LangSmith-generated UUID
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": result.returncode == 0 and "SUCCESS" in stdout,
            "examples_completed": examples_completed,  # Will be updated from LangSmith
        }
    except subprocess.TimeoutExpired:
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": None,
            "experiment_id": None,
            "returncode": -1,
            "stdout": "",
            "stderr": "Evaluation timed out after 10 minutes",
            "success": False,
            "examples_completed": 0,
        }
    except (OSError, subprocess.SubprocessError) as e:
        # Catch subprocess execution errors (file not found, permission denied, etc.)
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": None,
            "experiment_id": None,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "examples_completed": 0,
        }


def monitor_progress(
    progress_file: str, total_examples: int, stop_event: threading.Event
):
    """Monitor progress file and update tqdm."""
    with tqdm(
        total=total_examples * len(MODELS_TO_EVALUATE),
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


def main():
    """Run all evaluations in parallel."""
    print(f"\n{'='*80}")
    print("EVALUATION SUITE - Subprocess Isolation")
    print(f"Mode: {EXECUTION_MODE.upper()}")
    print(f"Node: {NODE_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Judge: {JUDGE_MODEL_ID}")
    print(f"Config: {CONFIG_FILE.name}")
    print(f"{'='*80}\n")

    # Initialize tracker
    tracker = ExperimentTracker(CONFIG_FILE)

    # Determine execution ID and models to evaluate
    if EXECUTION_MODE == "resume":
        # Resume mode
        execution_id = RESUME_EXECUTION_ID or tracker.get_latest_execution_id()

        if not execution_id:
            print("❌ No execution to resume. Run with EXECUTION_MODE='new' first.")
            return []

        execution = tracker.get_execution(execution_id)
        if not execution:
            print(f"❌ Execution '{execution_id}' not found in config.")
            return []

        print(f"📂 Resuming execution: {execution_id}")
        print(f"   Timestamp: {execution['timestamp']}")
        print(f"   Status: {execution['status']}\n")

        # Get incomplete models
        model_experiment_map = tracker.get_incomplete_models(execution_id)

        if not model_experiment_map:
            print("✅ All models already completed!")
            return []

        print(f"🔄 Models to evaluate/resume: {len(model_experiment_map)}")
        for model_id in model_experiment_map.keys():
            status = execution["models"][model_id]["status"]
            completed = execution["models"][model_id]["examples_completed"]
            print(f"  • {model_id} [{status}, {completed} examples]")

    else:
        # New mode
        model_experiment_map = {}

        print(f"🆕 Creating new execution...")
        execution_id = tracker.create_execution(
            node_name=NODE_NAME,
            dataset_name=DATASET_NAME,
            judge_model_id=JUDGE_MODEL_ID,
            max_concurrency=MAX_CONCURRENCY,
            model_ids=MODELS_TO_EVALUATE,
        )

        print(f"📂 Execution ID: {execution_id}\n")
        print(f"📋 Models to evaluate: {len(MODELS_TO_EVALUATE)}")

        execution = tracker.get_execution(execution_id)
        for model_id, model_data in execution["models"].items():
            model_experiment_map[model_id] = model_data["experiment_prefix"]
            print(f"  • {model_id}")
            print(f"    Prefix: {model_data['experiment_prefix']}")

    print(f"\n🚀 Starting {len(model_experiment_map)} parallel evaluations...\n")

    # Create temporary progress file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
        progress_file = tf.name

    # Start progress monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(progress_file, DATASET_SIZE, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    # Run evaluations in parallel (ThreadPoolExecutor for blocking subprocess calls)
    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(model_experiment_map)) as executor:
            # Submit evaluations with model_id -> experiment_identifier mapping
            # identifier is either experiment_name (resume) or experiment_prefix (new)
            futures = [
                executor.submit(
                    run_subprocess_evaluation,
                    model_id,
                    experiment_identifier,
                    progress_file,
                )
                for model_id, experiment_identifier in model_experiment_map.items()
            ]

            with tqdm(
                total=len(futures), desc="Models Completed", unit="model", position=1
            ) as pbar:
                for future in futures:
                    result = future.result()
                    results.append(result)

                    # Update tracker with result
                    model_id = result["model_id"]

                    # Update experiment metadata from LangSmith if available
                    if result["experiment_name"] and result["experiment_id"]:
                        tracker.update_model_experiment_metadata(
                            execution_id,
                            model_id,
                            result["experiment_name"],
                            result["experiment_id"],
                        )

                        # Query LangSmith for actual examples completed
                        examples_completed = get_examples_completed_from_langsmith(
                            result["experiment_name"]
                        )
                    else:
                        examples_completed = 0

                    if result["success"]:
                        tracker.update_model_status(
                            execution_id,
                            model_id,
                            "completed",
                            examples_completed=examples_completed,
                        )
                    else:
                        error_msg = (
                            result["stderr"][
                                :2000000
                            ]  # Capture more error context to debug issues
                            if result["stderr"]
                            else "Unknown error"
                        )
                        tracker.update_model_status(
                            execution_id,
                            model_id,
                            "failed",
                            error=error_msg,
                            examples_completed=examples_completed,
                        )

                    status = "✓" if result["success"] else "✗"
                    pbar.set_description(f"{status} {result['model_id'][:30]}")
                    pbar.update(1)
    finally:
        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join(timeout=2)
        # Cleanup progress file
        try:
            os.unlink(progress_file)
        except (OSError, IOError):
            pass

    # Update overall execution status
    all_completed = all(r["success"] for r in results)
    execution_status = "completed" if all_completed else "partial"
    tracker.update_execution_status(execution_id, execution_status)

    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUITE COMPLETED")
    print(f"Execution ID: {execution_id}")
    print(f"{'='*80}\n")

    success_count = 0
    error_count = 0

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['model_id']}")
        print(f"  Experiment: {result['experiment_name']}")

        if result["success"]:
            success_count += 1
        else:
            error_count += 1
            print(f"  RC: {result['returncode']}")
            if result["stderr"]:
                print(f"  Error: {result['stderr'][:200]}")

    print(f"\n{'='*80}")
    print(f"Total: {len(results)} evaluations")
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Status: {execution_status}")
    print(f"{'='*80}\n")

    if error_count > 0:
        print(f"💡 To resume failed evaluations, set:")
        print(f"   EXECUTION_MODE = 'resume'")
        print(f"   RESUME_EXECUTION_ID = '{execution_id}'  # or None for latest\n")

    return results


if __name__ == "__main__":
    evaluation_results = main()

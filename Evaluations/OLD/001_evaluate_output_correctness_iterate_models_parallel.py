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
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,
    get_examples_completed_from_langsmith,
    find_experiment_by_prefix,
    monitor_langsmith_progress,
)
from Evaluations.utils.helpers import monitor_progress

# Subprocess script and Python executable
SINGLE_EVAL_SCRIPT = Path(__file__).parent / "run_single_evaluation.py"
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

# Evaluation configuration
NODE_NAME = "format_answer_node"
DATASET_NAME = "001d_golden_dataset__output_correctness__simple_QA_from_SQL_reduced4"
MAX_CONCURRENCY = 1
JUDGE_MODEL_ID = "azureopenai_gpt-4.1"

# ============================================================================
# PROGRESS MONITORING CONFIGURATION
# ============================================================================
# Approximate dataset size for progress bar display only
# Note: This is a dummy value since parent script only manages parallel execution and doesn't authenticate with LangSmith
DATASET_SIZE = 30  # Dummy value for monitoring visualization
# ============================================================================

# ============================================================================
# EXECUTION MODE CONFIGURATION
# ============================================================================
# Mode: "new" - Start new evaluation run
#       "resume" - Resume from previous execution
EXECUTION_MODE = "resume"  # Change to "resume" to continue previous run
# EXECUTION_MODE = "new"  # Change to "resume" to continue previous run

# Execution ID to resume (None = auto-resume latest)
# Set to specific ID like "exec_2025-12-20_103045_a1b2c3d4" to resume that run
RESUME_EXECUTION_ID = "exec_2025-12-21_020551_227e7c7c"
# RESUME_EXECUTION_ID = None

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
    model_id: str,
    experiment_identifier: str,
    progress_file: str,
    execution_id: str,
    tracker: ExperimentTracker,
    is_resume: bool = False,
) -> dict:
    """Run single evaluation in subprocess with real-time progress tracking.

    Args:
        model_id: Model ID to evaluate
        experiment_identifier: Either experiment_name (resume) or experiment_prefix (new)
        progress_file: Path to progress tracking file
        execution_id: Current execution ID for tracker updates
        tracker: ExperimentTracker instance for real-time updates
        is_resume: True if resuming existing experiment, False if creating new

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

    # Determine if this is resume or new based on is_resume flag
    if is_resume:
        env["EVAL_EXPERIMENT_NAME"] = experiment_identifier  # Resume existing
    else:
        env["EVAL_EXPERIMENT_PREFIX"] = experiment_identifier  # Create new

    # Metadata to capture from subprocess output
    experiment_name = None
    experiment_id = None
    examples_completed = 0

    # Thread-safe containers for capturing output
    stderr_lines = []
    stdout_lines = []

    def read_stderr(pipe):
        """Read stderr line by line and update tracker immediately when metadata appears."""
        nonlocal experiment_name, experiment_id
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stderr_lines.append(line)
                line_stripped = line.strip()

                # Debug: Show key lines
                if "EXPERIMENT" in line_stripped or "CREATING" in line_stripped:
                    print(f"[DEBUG {model_id[:15]}] {line_stripped[:80]}", flush=True)

                # Parse and update metadata immediately
                if line_stripped.startswith("EXPERIMENT_NAME: "):
                    experiment_name = line_stripped.replace(
                        "EXPERIMENT_NAME: ", ""
                    ).strip()
                    print(f"🔍 Captured NAME: {experiment_name}", flush=True)
                elif line_stripped.startswith("EXPERIMENT_ID: "):
                    experiment_id = line_stripped.replace("EXPERIMENT_ID: ", "").strip()
                    print(f"🔍 Captured ID: {experiment_id}", flush=True)

                # Update tracker as soon as both are available
                if experiment_name and experiment_id:
                    print("💾 Saving to JSON...", flush=True)
                    tracker.update_model_experiment_metadata(
                        execution_id, model_id, experiment_name, experiment_id
                    )
                    exp_display = experiment_name or ""
                    print(f"✓ {model_id[:25]} -> {exp_display}", flush=True)
                    # Stop parsing once we have metadata (continue reading to EOF)
        except (OSError, IOError, ValueError) as e:
            print(f"❌ Error in read_stderr: {e}", flush=True)
        finally:
            pipe.close()

    def read_stdout(pipe):
        """Read stdout to avoid pipe buffer filling."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stdout_lines.append(line)
        except (OSError, IOError, ValueError):
            pass
        finally:
            pipe.close()

    try:
        # Mark as in-progress before starting subprocess
        tracker.update_model_status(execution_id, model_id, "in_progress")

        # Use Popen to read stderr in real-time
        process = subprocess.Popen(
            [str(PYTHON_EXE), str(SINGLE_EVAL_SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
        )

        # Start threads to read stdout/stderr without blocking
        stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))
        stdout_thread = threading.Thread(target=read_stdout, args=(process.stdout,))
        stderr_thread.start()
        stdout_thread.start()

        # For NEW experiments only, poll LangSmith API to find the experiment shortly after creation
        # For RESUME mode, we already have the experiment name/ID
        if not is_resume:
            print(
                f"🔍 Polling LangSmith for new experiment with prefix: {experiment_identifier[:50]}",
                flush=True,
            )
            # Wait a few seconds for aevaluate() to create the experiment
            time.sleep(5)

            experiment_metadata = find_experiment_by_prefix(
                experiment_identifier, max_wait_seconds=60
            )
            if experiment_metadata:
                experiment_name = experiment_metadata["name"]
                experiment_id = experiment_metadata["id"]
                print(f"✓ Found experiment via API: {experiment_name}", flush=True)
                # Update tracker immediately
                tracker.update_model_experiment_metadata(
                    execution_id, model_id, experiment_name, experiment_id
                )
            else:
                print(
                    "⚠️ API polling failed. Waiting for subprocess to report metadata...",
                    flush=True,
                )
                # Fallback: metadata will be captured from subprocess stderr prints
                # The read_stderr thread will update tracker when it sees EXPERIMENT_NAME/ID
        else:
            # Resume mode: experiment_identifier IS the experiment_id (UUID)
            experiment_id = experiment_identifier
            # Query LangSmith to get the actual experiment name from the UUID
            try:
                ls_client = Client()
                project = ls_client.read_project(project_id=experiment_id)
                experiment_name = project.name
                exp_display = experiment_name or ""
                print(
                    f"🔄 Resuming experiment: {exp_display} (ID: {experiment_id})",
                    flush=True,
                )
                # Update tracker with the actual experiment name we just fetched
                tracker.update_model_experiment_metadata(
                    execution_id, model_id, experiment_name, experiment_id
                )
            except Exception as e:
                print(
                    f"⚠️ Could not fetch experiment name for UUID {experiment_id}: {e}",
                    flush=True,
                )
                experiment_name = experiment_id  # Fallback to using UUID

        # Wait for process to complete
        returncode = process.wait(timeout=600)

        # Wait for reader threads to finish
        stderr_thread.join(timeout=5)
        stdout_thread.join(timeout=5)

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        # Get final examples_completed count from LangSmith
        # Try with experiment_name first, fall back to experiment_id if name not available
        if experiment_name or experiment_id:
            identifier = experiment_name or experiment_id
            examples_completed = get_examples_completed_from_langsmith(identifier)
            # Update tracker with final count
            status = (
                "completed" if returncode == 0 and "SUCCESS" in stdout else "failed"
            )
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
            "experiment_name": experiment_name,  # LangSmith-generated human name
            "experiment_id": experiment_id,  # LangSmith-generated UUID
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": returncode == 0 and "SUCCESS" in stdout,
            "examples_completed": examples_completed,
        }
    except subprocess.TimeoutExpired:
        tracker.update_model_status(
            execution_id,
            model_id,
            "failed",
            error="Evaluation timed out after 10 minutes",
        )
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "returncode": -1,
            "stdout": "",
            "stderr": "Evaluation timed out after 10 minutes",
            "success": False,
            "examples_completed": examples_completed,
        }
    except (OSError, subprocess.SubprocessError) as e:
        # Catch subprocess execution errors (file not found, permission denied, etc.)
        tracker.update_model_status(
            execution_id, model_id, "failed", error=str(e)[:500]
        )
        return {
            "model_id": model_id,
            "experiment_identifier": experiment_identifier,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "examples_completed": examples_completed,
        }


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

        # Get incomplete models - returns experiment_prefix for each
        model_experiment_map = tracker.get_incomplete_models(execution_id)

        if not model_experiment_map:
            print("✅ All models already completed!")
            return []

        print(f"🔄 Models to evaluate/resume: {len(model_experiment_map)}")

        # For RESUME mode: Search LangSmith to find the actual experiment for each model
        print("\n🔍 Searching LangSmith for existing experiments...\n")
        for model_id, experiment_prefix in list(model_experiment_map.items()):
            status = execution["models"][model_id]["status"]
            completed = execution["models"][model_id]["examples_completed"]
            print(f"  • {model_id} [{status}, {completed} examples]")
            print(f"    Searching for: {experiment_prefix[:60]}...")

            # Search LangSmith for experiment matching this prefix
            experiment_metadata = find_experiment_by_prefix(
                experiment_prefix, max_wait_seconds=10
            )

            if experiment_metadata:
                # Found it! Update the map to use the UUID for resume
                model_experiment_map[model_id] = experiment_metadata["id"]
                print(f"    ✓ Found: {experiment_metadata['name'][:60]}")
                print(f"    ID: {experiment_metadata['id']}")

                # Update tracker with correct metadata if it's wrong
                tracker.update_model_experiment_metadata(
                    execution_id,
                    model_id,
                    experiment_metadata["name"],
                    experiment_metadata["id"],
                )
            else:
                print(f"    ⚠️ Not found - will create new experiment")
                # Keep the prefix - will create new experiment

        print()

    else:
        # New mode
        model_experiment_map = {}

        print("🆕 Creating new execution...")
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

    # Start progress monitoring threads
    stop_event = threading.Event()

    # Thread 1: Monitor progress file for tqdm updates
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(progress_file, DATASET_SIZE, stop_event, len(model_experiment_map)),
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

    # Run evaluations in parallel (ThreadPoolExecutor for blocking subprocess calls)
    results = []
    try:
        with ThreadPoolExecutor(max_workers=len(model_experiment_map)) as executor:
            # Submit evaluations with model_id -> experiment_identifier mapping
            # In RESUME mode with found experiments: identifier is UUID (resume existing)
            # In NEW mode or RESUME with not-found: identifier is prefix (create new)
            # Determine is_resume per model based on identifier format (UUID = resume)
            import re

            uuid_pattern = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )

            futures = [
                executor.submit(
                    run_subprocess_evaluation,
                    model_id,
                    experiment_identifier,
                    progress_file,
                    execution_id,
                    tracker,
                    bool(
                        uuid_pattern.match(experiment_identifier)
                    ),  # is_resume = True if UUID
                )
                for model_id, experiment_identifier in model_experiment_map.items()
            ]

            with tqdm(
                total=len(futures), desc="Models Completed", unit="model", position=1
            ) as pbar:
                for future in futures:
                    result = future.result()
                    results.append(result)

                    # Note: Tracker is already updated in real-time by run_subprocess_evaluation
                    # This section only handles final progress bar update

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
        print("💡 To resume failed evaluations, set:")
        print("   EXECUTION_MODE = 'resume'")
        print(f"   RESUME_EXECUTION_ID = '{execution_id}'  # or None for latest\n")

    return results


if __name__ == "__main__":
    evaluation_results = main()

"""
Parallel model evaluation script using subprocess isolation.

This script orchestrates parallel evaluation of multiple models by:
1. Spawning separate subprocesses for each model (ensuring isolation)
2. Managing execution tracking with resume capability
3. Coordinating progress monitoring across parallel evaluations
4. Collecting and displaying results

Key components:
- run_single_evaluation.py: Subprocess worker that runs one evaluation
- Utilities handle: subprocess execution, progress monitoring, tracking, output
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup project root
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path.cwd().parents[0]

sys.path.insert(0, str(BASE_DIR))

# Import utilities
from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,
    prepare_resume_execution,
    prepare_new_execution,
)
from Evaluations.utils.subprocess_runner import run_evaluation_subprocess
from Evaluations.utils.parallel_executor import run_parallel_evaluations
from Evaluations.utils.console_output import (
    print_configuration_summary,
    print_resume_info,
    print_no_execution_to_resume,
    print_execution_not_found,
    print_all_completed,
    print_models_to_resume,
    print_searching_experiments_header,
    print_model_search_status,
    print_experiment_found,
    print_experiment_not_found,
    print_new_execution_info,
    print_model_prefix,
    print_start_evaluations,
    print_completion_summary,
    print_resume_instructions,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SINGLE_EVAL_SCRIPT = Path(__file__).parent / "run_single_evaluation.py"
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"
CONFIG_FILE = Path(__file__).with_suffix(".json")

# Evaluation settings
NODE_NAME = "generate_query_node"
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_custom_manual"
)
MAX_CONCURRENCY = 1
JUDGE_MODEL_ID = "mistral_mistral-large-2512"
DATASET_SIZE = 30  # Approximate size for progress display

# Execution mode
# EXECUTION_MODE = "resume"  # "new" or "resume"
EXECUTION_MODE = "new"  # "new" or "resume"
# RESUME_EXECUTION_ID = "exec_2025-12-21_153323_ed989867"  # or None for latest
RESUME_EXECUTION_ID = None  # or None for latest

# Models to evaluate
MODELS_TO_EVALUATE = [
    "mistral_mistral-large-2512",
    "mistral_devstral-2512",
    "mistral_codestral-2508",
    "azureopenai_gpt-4o",
    "azureopenai_gpt-4o-mini",
    "azureopenai_gpt-4.1",
    "azureopenai_gpt-5-nano",
    "azureopenai_gpt-5.2-chat",
    "xai_grok-4-1-fast-reasoning",
    "xai_grok-4-1-fast-non-reasoning",
    "gemini_gemini-3-pro-preview",
]

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run all evaluations in parallel."""
    # Print configuration
    print_configuration_summary(
        EXECUTION_MODE, NODE_NAME, DATASET_NAME, JUDGE_MODEL_ID, CONFIG_FILE.name
    )

    # Initialize tracker
    tracker = ExperimentTracker(CONFIG_FILE)

    # Prepare execution based on mode
    if EXECUTION_MODE == "resume":
        result = prepare_resume_execution(tracker, RESUME_EXECUTION_ID)

        if not result:
            print_no_execution_to_resume()
            return []

        execution_id, model_experiment_map, execution = result

        if not execution:
            print_execution_not_found(execution_id)
            return []

        # Print resume info
        print_resume_info(execution_id, execution["timestamp"], execution["status"])

        if not model_experiment_map:
            print_all_completed()
            return []

        print_models_to_resume(len(model_experiment_map))

        # Print search status for each model
        print_searching_experiments_header()
        for model_id, experiment_identifier in model_experiment_map.items():
            status = execution["models"][model_id]["status"]
            completed = execution["models"][model_id]["examples_completed"]
            experiment_prefix = execution["models"][model_id]["experiment_prefix"]

            print_model_search_status(model_id, status, completed, experiment_prefix)

            # Check if we found it (identifier is UUID) or will create new (prefix)
            from Evaluations.utils.experiment_tracker import is_uuid

            if is_uuid(experiment_identifier):
                exp_name = execution["models"][model_id].get("experiment_name", "")
                print_experiment_found(exp_name, experiment_identifier)
            else:
                print_experiment_not_found()

        print()

    else:  # "new" mode
        execution_id, model_experiment_map, execution = prepare_new_execution(
            tracker,
            NODE_NAME,
            DATASET_NAME,
            JUDGE_MODEL_ID,
            MAX_CONCURRENCY,
            MODELS_TO_EVALUATE,
        )

        print_new_execution_info(execution_id, len(MODELS_TO_EVALUATE))

        for model_id, model_data in execution["models"].items():
            print_model_prefix(model_id, model_data["experiment_prefix"])

    print_start_evaluations(len(model_experiment_map))

    # Run parallel evaluations
    results = run_parallel_evaluations(
        model_experiment_map,
        run_evaluation_subprocess,
        execution_id,
        tracker,
        DATASET_SIZE,
        # Additional kwargs for evaluation_function
        python_exe=PYTHON_EXE,
        eval_script=SINGLE_EVAL_SCRIPT,
        base_dir=BASE_DIR,
        node_name=NODE_NAME,
        dataset_name=DATASET_NAME,
        max_concurrency=MAX_CONCURRENCY,
        judge_model_id=JUDGE_MODEL_ID,
        timeout_seconds=None,
    )

    # Update execution status
    all_completed = all(r["success"] for r in results)
    execution_status = "completed" if all_completed else "partial"
    tracker.update_execution_status(execution_id, execution_status)

    # Print summary
    print_completion_summary(execution_id, results, execution_status)

    # Print resume instructions if needed
    error_count = sum(1 for r in results if not r["success"])
    if error_count > 0:
        print_resume_instructions(execution_id)

    return results


if __name__ == "__main__":
    evaluation_results = main()

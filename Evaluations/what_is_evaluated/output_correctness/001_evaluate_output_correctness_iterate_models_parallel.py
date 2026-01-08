"""Parallel Model Evaluation Orchestrator with Subprocess Isolation and Resume Capability

This module provides a sophisticated orchestration system for running LangSmith evaluations
across multiple language models in parallel, with complete process isolation, execution
tracking, and resume capability for interrupted or failed runs.

MODULE DESCRIPTION
==================

This orchestrator script is the main entry point for large-scale model evaluation campaigns.
It manages the complexity of evaluating multiple language models against a golden dataset,
ensuring each model is evaluated in complete isolation to prevent configuration leakage,
while providing robust progress tracking, error recovery, and execution resumption.

KEY FEATURES
============

1. Parallel Subprocess Execution:
   - Spawns isolated subprocesses for each model evaluation
   - Prevents configuration bleeding between models
   - Each subprocess loads its own model configuration independently
   - Subprocess-level error handling prevents cascading failures
   - Configurable concurrency limits to prevent resource exhaustion
   - Real-time progress monitoring via file-based communication

2. Execution Tracking and State Management:
   - Persistent execution state stored in JSON configuration file
   - Unique execution ID generation with timestamp and short UUID
   - Model-level status tracking (pending, in_progress, completed, failed)
   - Experiment prefix generation and UUID association
   - Progress tracking via examples_completed counters
   - Automatic execution history with metadata preservation

3. Resume Capability:
   - Intelligent resume from last execution or specific execution ID
   - Automatic detection of incomplete/failed model evaluations
   - LangSmith experiment discovery by prefix pattern matching
   - Fallback to new experiment creation if previous not found
   - Preservation of experiment continuity across interruptions
   - Resume mode validation and status reporting

4. LangSmith Integration:
   - Dataset-based evaluation using LangSmith platform
   - Experiment naming with standardized format: judge_<judge>__Node_<node>__Model_<model>
   - Automatic experiment UUID retrieval and storage
   - Support for both new and resume evaluation modes
   - Example-level progress tracking across experiments
   - Configurable judge model for correctness evaluation

5. Progress Monitoring and Reporting:
   - Real-time progress bars with dataset size estimation
   - Per-model success/failure status tracking
   - Experiment metadata display (name, UUID, status)
   - Execution summary with completion statistics
   - Detailed error reporting with resume instructions
   - Color-coded console output via utility functions

6. Configuration Management:
   - Centralized configuration section for easy modification
   - Environment variable loading via python-dotenv
   - Flexible path resolution (works from any directory)
   - Virtual environment detection and Python executable resolution
   - Node-specific model assignment configuration
   - Judge model specification for evaluation

7. Error Handling and Recovery:
   - Graceful handling of subprocess failures
   - Continuation of remaining evaluations on individual failures
   - Comprehensive error logging with context information
   - Automatic partial execution status on incomplete runs
   - Resume instructions provided for failed evaluations
   - Execution tracker state consistency maintenance

PROCESSING FLOW
===============

1. Initialization Phase:
   - Load environment variables from .env file
   - Determine project base directory and setup paths
   - Import evaluation utilities and console output functions
   - Validate configuration parameters

2. Configuration Display:
   - Print execution mode (new vs. resume)
   - Display target node name and dataset name
   - Show judge model configuration
   - Report tracking configuration file path

3. Execution Preparation:

   A. NEW MODE:
      - Generate new execution ID with timestamp and UUID
      - Create experiment prefixes for each model
      - Initialize tracking state for all models
      - Store execution metadata in configuration file

   B. RESUME MODE:
      - Load execution tracker from configuration file
      - Identify target execution (latest or specific ID)
      - Query LangSmith for existing experiments by prefix
      - Build model-to-experiment mapping (UUID if found, prefix if not)
      - Filter out completed models from processing queue
      - Display resume status with model completion info

4. Parallel Evaluation Execution:
   - Spawn subprocess for each model in parallel (controlled concurrency)
   - Pass configuration via environment variables to subprocess
   - Monitor subprocess execution with real-time progress
   - Track examples_completed via shared progress files
   - Collect subprocess results (success/failure, metadata)
   - Update execution tracker with experiment UUIDs and status

5. Post-Processing:
   - Update execution status (completed or partial)
   - Save final state to tracking configuration file
   - Display completion summary with statistics
   - Print experiment details for each model
   - Provide resume instructions if any failures occurred

6. Result Reporting:
   - Return list of evaluation results
   - Each result contains: model_id, success, experiment_name, experiment_id
   - Console output summary includes success/failure counts
   - Error details logged for debugging

ARCHITECTURE
============

Component Hierarchy:
--------------------
1. This orchestrator script (001_evaluate_output_correctness_iterate_models_parallel.py)
   ├── Spawns: run_single_evaluation.py subprocesses (one per model)
   │   └── Each subprocess: Configures agent, runs aevaluate, reports metadata
   │
   ├── Uses: ExperimentTracker (utils/experiment_tracker.py)
   │   └── Manages: Execution state, experiment mapping, resume logic
   │
   ├── Uses: subprocess_runner (utils/subprocess_runner.py)
   │   └── Handles: Process spawning, environment setup, output capture
   │
   ├── Uses: parallel_executor (utils/parallel_executor.py)
   │   └── Manages: Concurrent execution, progress monitoring, result collection
   │
   └── Uses: console_output (utils/console_output.py)
       └── Provides: Formatted console messages, progress display, status reporting

Subprocess Communication:
-------------------------
- Environment variables: Configuration passed to subprocess
- STDOUT: Success marker ("SUCCESS") captured by parent
- STDERR: Metadata output (EXPERIMENT_NAME, EXPERIMENT_ID, progress info)
- Progress files: File-based counters for real-time progress tracking
- Exit codes: 0 for success, 1 for failure

Execution State Schema:
-----------------------
{
    "execution_id": "exec_2025-01-06_153000_abc123",
    "timestamp": "2025-01-06T15:30:00",
    "status": "completed" | "partial" | "in_progress",
    "node_name": "generate_query_node",
    "dataset_name": "001d_golden_dataset__output_correctness__simple_QA_from_SQL_custom_manual",
    "judge_model_id": "mistral_mistral-large-2512",
    "max_concurrency": 1,
    "models": {
        "model_id_1": {
            "experiment_prefix": "judge_mistral-large-2512__Node_generate_query_node__Model_model_id_1",
            "experiment_name": "judge_mistral-large-2512__Node_generate_query_node__Model_model_id_1-abc123",
            "experiment_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "status": "pending" | "in_progress" | "completed" | "failed",
            "examples_completed": 25,
            "error": "Optional error message if failed"
        },
        ...
    }
}

CONFIGURATION
=============

Required Configuration (edit in CONFIGURATION section):
--------------------------------------------------------
- SINGLE_EVAL_SCRIPT: Path to run_single_evaluation.py worker script
- PYTHON_EXE: Path to Python executable in virtual environment
- CONFIG_FILE: Path to JSON file for execution tracking
- NODE_NAME: Name of graph node to evaluate
- DATASET_NAME: LangSmith dataset name for evaluation
- MAX_CONCURRENCY: Number of concurrent evaluations per model
- JUDGE_MODEL_ID: Model ID for correctness evaluation
- DATASET_SIZE: Approximate dataset size for progress display
- EXECUTION_MODE: "new" or "resume"
- RESUME_EXECUTION_ID: Specific execution to resume (or None for latest)
- MODELS_TO_EVALUATE: List of model IDs to evaluate

Environment Variables Required:
--------------------------------
- LangSmith API credentials (loaded from .env)
- Model provider API keys (Azure OpenAI, Mistral, Gemini, etc.)
- Virtual environment must be activated (.venv/Scripts/python.exe)

USAGE EXAMPLES
==============

Basic Usage - New Evaluation:
------------------------------
# Configure in script:
EXECUTION_MODE = "new"
MODELS_TO_EVALUATE = ["mistral_mistral-large-2512", "azureopenai_gpt-4o"]

# Run:
python 001_evaluate_output_correctness_iterate_models_parallel.py

Resume After Interruption:
---------------------------
# Configure in script:
EXECUTION_MODE = "resume"
RESUME_EXECUTION_ID = None  # Will resume latest

# Run:
python 001_evaluate_output_correctness_iterate_models_parallel.py

Resume Specific Execution:
---------------------------
# Configure in script:
EXECUTION_MODE = "resume"
RESUME_EXECUTION_ID = "exec_2025-01-06_153000_abc123"

# Run:
python 001_evaluate_output_correctness_iterate_models_parallel.py

DEPENDENCIES
============

Core Dependencies:
------------------
- sys, pathlib: Path and system operations
- dotenv: Environment variable loading
- json: Configuration file handling

Project Utilities (Evaluations/utils/):
----------------------------------------
- experiment_tracker: Execution state management, resume logic
- subprocess_runner: Subprocess spawning and management
- parallel_executor: Concurrent execution coordination
- console_output: Formatted terminal output

Worker Script:
--------------
- run_single_evaluation.py: Subprocess worker for single model evaluation

OUTPUT
======

Console Output:
---------------
- Configuration summary (mode, node, dataset, judge model)
- Execution ID and timestamp
- Model experiment prefix assignments
- Real-time progress bars with completion percentages
- Per-model status updates (searching, found, created)
- Final completion summary with statistics
- Resume instructions if needed

File Artifacts:
---------------
- Tracking configuration file (e.g., 001_evaluate_output_correctness_iterate_models_parallel.json)
- LangSmith experiment runs (stored in LangSmith platform)
- Progress tracking files (temporary, used during execution)

Return Value:
-------------
List of dictionaries with evaluation results:
[
    {
        "model_id": "model_id_1",
        "success": True,
        "experiment_name": "judge_mistral-large-2512__Node_generate_query_node__Model_model_id_1-abc123",
        "experiment_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "error": None
    },
    ...
]

ERROR HANDLING
==============

Subprocess Failures:
--------------------
- Individual subprocess failures don't crash entire execution
- Error captured and logged with model context
- Remaining models continue evaluation
- Execution marked as "partial" if any failures

Resume Logic:
-------------
- Validates execution ID exists in tracking file
- Checks for LangSmith experiment existence
- Falls back to new experiment if previous not found
- Handles missing or corrupted tracking files gracefully

Progress Tracking:
------------------
- Safe handling of missing progress files
- Concurrent write protection for progress counters
- Graceful degradation if progress tracking fails

NOTES
=====

Performance Considerations:
---------------------------
- Each subprocess is fully isolated (separate Python interpreter)
- Overhead of ~2-3 seconds per subprocess startup
- Memory usage: ~1GB per concurrent subprocess (depends on model)
- Recommended MAX_CONCURRENCY: 1-2 for large models, 2-4 for small models

Resume Behavior:
----------------
- Resume always checks LangSmith for existing experiments
- If experiment UUID found: Continues from last evaluated example
- If experiment not found: Creates new experiment with same prefix
- Model status determines inclusion in resume queue

Experiment Naming:
------------------
- Format: judge_<judge>__Node_<node>__Model_<model>-<timestamp>-<uuid_short>
- LangSmith automatically appends timestamp and UUID suffix
- Prefix used for experiment discovery during resume

Best Practices:
---------------
- Always run with activated virtual environment
- Set DATASET_SIZE to approximate size for accurate progress
- Use resume mode liberally - it's safe and efficient
- Keep MAX_CONCURRENCY low to prevent resource exhaustion
- Monitor first evaluation to ensure configuration is correct

Troubleshooting:
----------------
- Check .env file for required API keys
- Verify virtual environment is activated
- Ensure run_single_evaluation.py is in same directory
- Check tracking JSON file for execution state
- Review subprocess stderr output for detailed errors
- Use resume mode to retry failed models

RELATED FILES
=============

- run_single_evaluation.py: Subprocess worker script
- Evaluations/utils/experiment_tracker.py: Execution state management
- Evaluations/utils/subprocess_runner.py: Subprocess execution
- Evaluations/utils/parallel_executor.py: Parallel coordination
- Evaluations/utils/console_output.py: Terminal output formatting
- my_agent/: Agent graph and model configuration
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
import sys  # System-specific parameters and functions
from pathlib import Path  # Object-oriented filesystem paths
from dotenv import load_dotenv  # Load environment variables from .env file

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Load environment variables early (LangSmith API keys, model provider keys)
load_dotenv()

# ============================================================================
# PROJECT ROOT DETECTION
# ============================================================================
# Resolve project base directory (3 levels up from this file)
# This allows the script to work regardless of where it's executed from
try:
    # Standard case: Running as a script file
    # Goes up: output_correctness/ -> what_is_evaluated/ -> Evaluations/ -> project_root/
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    # Fallback: Running in interactive mode (Jupyter, REPL)
    BASE_DIR = Path.cwd().parents[0]

# Add project root to Python path for absolute imports
sys.path.insert(0, str(BASE_DIR))

# ============================================================================
# PROJECT UTILITY IMPORTS
# ============================================================================

# Experiment Tracker: Manages execution state, persistence, and resume logic
from Evaluations.utils.experiment_tracker import (
    ExperimentTracker,  # Main class for tracking execution state in JSON file
    prepare_resume_execution,  # Prepares model-experiment mapping for resume mode
    prepare_new_execution,  # Creates new execution with unique ID and prefixes
)

# Subprocess Runner: Handles subprocess spawning and management
from Evaluations.utils.subprocess_runner import run_evaluation_subprocess

# Parallel Executor: Coordinates concurrent evaluation execution
from Evaluations.utils.parallel_executor import run_parallel_evaluations

# Console Output: Provides formatted terminal output functions
from Evaluations.utils.console_output import (
    print_configuration_summary,  # Display config at start
    print_resume_info,  # Show resume execution details
    print_no_execution_to_resume,  # Warning when no executions exist
    print_execution_not_found,  # Error when specific execution not found
    print_all_completed,  # Info when all models already done
    print_models_to_resume,  # Count of models to process
    print_searching_experiments_header,  # Section header for search status
    print_model_search_status,  # Per-model search results
    print_experiment_found,  # Success message for found experiment
    print_experiment_not_found,  # Info when experiment needs creation
    print_new_execution_info,  # Display new execution details
    print_model_prefix,  # Show experiment prefix for each model
    print_start_evaluations,  # Begin evaluation message
    print_completion_summary,  # Final results and statistics
    print_resume_instructions,  # How to resume failed runs
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Worker script that runs evaluation for a single model in isolated subprocess
SINGLE_EVAL_SCRIPT = Path(__file__).parent / "run_single_evaluation.py"

# Python executable from virtual environment (ensures consistent environment)
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

# JSON file for execution tracking (same name as this script, .json extension)
# Stores: execution history, experiment mappings, progress, resume state
CONFIG_FILE = Path(__file__).with_suffix(".json")

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
# Node name in the agent graph to evaluate (must match graph node definition)
NODE_NAME = "generate_query_node"

# LangSmith dataset name containing golden examples for evaluation
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_custom_manual"
)

# Maximum number of concurrent evaluations per model (1 = sequential, higher = parallel)
# Lower values reduce memory usage but increase total runtime
MAX_CONCURRENCY = 1

# Model ID for judge model (evaluates correctness of model outputs)
# Should be a strong, reliable model for accurate evaluation
JUDGE_MODEL_ID = "mistral_mistral-large-2512"

# Approximate dataset size for progress bar accuracy (doesn't need to be exact)
DATASET_SIZE = 30  # Approximate size for progress display

# ============================================================================
# EXECUTION MODE CONFIGURATION
# ============================================================================
# Execution mode: "new" creates fresh evaluation, "resume" continues previous run
# EXECUTION_MODE = "resume"  # "new" or "resume"
EXECUTION_MODE = "new"  # "new" or "resume"

# Specific execution ID to resume (format: exec_YYYY-MM-DD_HHMMSS_shortUUID)
# Set to None to resume the most recent execution
# RESUME_EXECUTION_ID = "exec_2025-12-21_153323_ed989867"  # or None for latest
RESUME_EXECUTION_ID = None  # or None for latest

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# List of model IDs to evaluate in parallel
# Each model ID must have a corresponding configuration in MODEL_CONFIGS_ALL
# Format: <provider>_<model_name> (e.g., "mistral_mistral-large-2512")
MODELS_TO_EVALUATE = [
    # Mistral models
    "mistral_mistral-large-2512",
    "mistral_devstral-2512",
    "mistral_codestral-2508",
    # Azure OpenAI models
    "azureopenai_gpt-4o",
    "azureopenai_gpt-4o-mini",
    "azureopenai_gpt-4.1",
    "azureopenai_gpt-5-nano",
    "azureopenai_gpt-5.2-chat",
    # xAI Grok models
    "xai_grok-4-1-fast-reasoning",
    "xai_grok-4-1-fast-non-reasoning",
    # Google Gemini models
    "gemini_gemini-3-pro-preview",
]

# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run all evaluations in parallel with execution tracking and resume capability.

    This is the main orchestration function that:
    1. Displays configuration summary
    2. Initializes execution tracker
    3. Prepares execution (new or resume mode)
    4. Spawns parallel subprocesses for model evaluation
    5. Collects results and updates tracking state
    6. Reports completion summary and errors

    Returns:
        list: List of evaluation results, one dict per model containing:
              - model_id: Model identifier
              - success: Boolean success status
              - experiment_name: LangSmith experiment name
              - experiment_id: LangSmith experiment UUID
              - error: Error message if failed (or None)
    """
    # ========================================================================
    # CONFIGURATION DISPLAY
    # ========================================================================
    # Display all configuration settings to user for verification
    print_configuration_summary(
        EXECUTION_MODE, NODE_NAME, DATASET_NAME, JUDGE_MODEL_ID, CONFIG_FILE.name
    )

    # ========================================================================
    # TRACKER INITIALIZATION
    # ========================================================================
    # Initialize execution tracker (loads existing state or creates new file)
    tracker = ExperimentTracker(CONFIG_FILE)

    # ========================================================================
    # EXECUTION PREPARATION
    # ========================================================================
    # Prepare execution based on mode (resume existing or create new)
    if EXECUTION_MODE == "resume":
        # ====================================================================
        # RESUME MODE: Continue previous execution
        # ====================================================================
        # Load execution state and build model-to-experiment mapping
        result = prepare_resume_execution(tracker, RESUME_EXECUTION_ID)

        # Handle case where no executions exist to resume
        if not result:
            print_no_execution_to_resume()
            return []

        # Unpack resume preparation results
        execution_id, model_experiment_map, execution = result

        # Handle case where specific execution ID not found
        if not execution:
            print_execution_not_found(execution_id)
            return []

        # Display resume information (execution ID, timestamp, status)
        print_resume_info(execution_id, execution["timestamp"], execution["status"])

        # Check if all models already completed
        if not model_experiment_map:
            print_all_completed()
            return []

        # Display count of models that need processing
        print_models_to_resume(len(model_experiment_map))

        # ====================================================================
        # EXPERIMENT DISCOVERY STATUS
        # ====================================================================
        # Display search status for each model (found existing vs. will create new)
        print_searching_experiments_header()
        for model_id, experiment_identifier in model_experiment_map.items():
            # Get model status from execution state
            status = execution["models"][model_id]["status"]
            completed = execution["models"][model_id]["examples_completed"]
            experiment_prefix = execution["models"][model_id]["experiment_prefix"]

            # Display model status and completion progress
            print_model_search_status(model_id, status, completed, experiment_prefix)

            # Check if we found existing experiment (UUID) or will create new (prefix)
            from Evaluations.utils.experiment_tracker import is_uuid

            if is_uuid(experiment_identifier):
                # Existing experiment found in LangSmith - will resume
                exp_name = execution["models"][model_id].get("experiment_name", "")
                print_experiment_found(exp_name, experiment_identifier)
            else:
                # No existing experiment found - will create new with same prefix
                print_experiment_not_found()

        print()  # Blank line for readability

    else:  # "new" mode
        # ====================================================================
        # NEW MODE: Create fresh execution
        # ====================================================================
        # Generate new execution ID, create experiment prefixes, initialize state
        execution_id, model_experiment_map, execution = prepare_new_execution(
            tracker,
            NODE_NAME,
            DATASET_NAME,
            JUDGE_MODEL_ID,
            MAX_CONCURRENCY,
            MODELS_TO_EVALUATE,
        )

        # Display new execution information (ID, model count)
        print_new_execution_info(execution_id, len(MODELS_TO_EVALUATE))

        # Display experiment prefix for each model (for LangSmith experiment naming)
        for model_id, model_data in execution["models"].items():
            print_model_prefix(model_id, model_data["experiment_prefix"])

    # ========================================================================
    # PARALLEL EVALUATION EXECUTION
    # ========================================================================
    # Display start message with count of models to process
    print_start_evaluations(len(model_experiment_map))

    # Execute evaluations in parallel with progress monitoring
    # This spawns a subprocess for each model and coordinates their execution
    results = run_parallel_evaluations(
        model_experiment_map,  # Dict mapping model_id -> experiment_identifier
        run_evaluation_subprocess,  # Function to spawn subprocess for each model
        execution_id,  # Current execution ID for tracking
        tracker,  # ExperimentTracker instance for state updates
        DATASET_SIZE,  # Approximate dataset size for progress calculation
        # Additional kwargs passed to evaluation_function (run_evaluation_subprocess)
        python_exe=PYTHON_EXE,  # Python executable path
        eval_script=SINGLE_EVAL_SCRIPT,  # Worker script path
        base_dir=BASE_DIR,  # Project root directory
        node_name=NODE_NAME,  # Node name for agent configuration
        dataset_name=DATASET_NAME,  # LangSmith dataset name
        max_concurrency=MAX_CONCURRENCY,  # Concurrent evaluations per model
        judge_model_id=JUDGE_MODEL_ID,  # Judge model for correctness evaluation
        timeout_seconds=None,  # Subprocess timeout (None = no timeout)
    )

    # ========================================================================
    # POST-PROCESSING AND STATUS UPDATE
    # ========================================================================
    # Determine final execution status based on results
    all_completed = all(r["success"] for r in results)
    execution_status = "completed" if all_completed else "partial"

    # Update execution status in tracking file
    tracker.update_execution_status(execution_id, execution_status)

    # ========================================================================
    # RESULT REPORTING
    # ========================================================================
    # Display completion summary with statistics
    print_completion_summary(execution_id, results, execution_status)

    # If any failures occurred, provide resume instructions
    error_count = sum(1 for r in results if not r["success"])
    if error_count > 0:
        print_resume_instructions(execution_id)

    # Return results list for programmatic access
    return results


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Execute main function when script is run directly (not imported)
    # Returns list of evaluation results for all models
    evaluation_results = main()

"""Console output formatting utilities for evaluation scripts.

This module provides consistent, clean formatting for console output
across all evaluation scripts.
"""

from typing import List, Dict


def print_header(title: str, width: int = 80):
    """Print a formatted header.

    Args:
        title: Header title text
        width: Header width in characters
    """
    print(f"\n{'='*width}")
    print(title)
    print(f"{'='*width}\n")


def print_configuration_summary(
    mode: str,
    node_name: str,
    dataset_name: str,
    judge_model_id: str,
    config_file_name: str,
    width: int = 80,
):
    """Print evaluation configuration summary.

    Args:
        mode: Execution mode ("new" or "resume")
        node_name: Node being evaluated
        dataset_name: Dataset name
        judge_model_id: Judge model ID
        config_file_name: Config file name
        width: Header width
    """
    print(f"\n{'='*width}")
    print("EVALUATION SUITE - Subprocess Isolation")
    print(f"Mode: {mode.upper()}")
    print(f"Node: {node_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {judge_model_id}")
    print(f"Config: {config_file_name}")
    print(f"{'='*width}\n")


def print_resume_info(execution_id: str, timestamp: str, status: str):
    """Print information about resumed execution.

    Args:
        execution_id: Execution ID
        timestamp: Execution timestamp
        status: Execution status
    """
    print(f"📂 Resuming execution: {execution_id}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Status: {status}\n")


def print_model_search_status(
    model_id: str, status: str, completed: int, experiment_prefix: str
):
    """Print model search status when resuming.

    Args:
        model_id: Model ID
        status: Current status
        completed: Examples completed
        experiment_prefix: Experiment prefix being searched
    """
    print(f"  • {model_id} [{status}, {completed} examples]")
    print(f"    Searching for: {experiment_prefix[:60]}...")


def print_experiment_found(experiment_name: str, experiment_id: str):
    """Print message when experiment is found.

    Args:
        experiment_name: Experiment name
        experiment_id: Experiment ID
    """
    print(f"    ✓ Found: {experiment_name[:60]}")
    print(f"    ID: {experiment_id}")


def print_experiment_not_found():
    """Print message when experiment is not found."""
    print("    ⚠️ Not found - will create new experiment")


def print_new_execution_info(execution_id: str, num_models: int):
    """Print information about new execution.

    Args:
        execution_id: New execution ID
        num_models: Number of models to evaluate
    """
    print("🆕 Creating new execution...")
    print(f"📂 Execution ID: {execution_id}\n")
    print(f"📋 Models to evaluate: {num_models}")


def print_model_prefix(model_id: str, experiment_prefix: str):
    """Print model and its experiment prefix.

    Args:
        model_id: Model ID
        experiment_prefix: Experiment prefix
    """
    print(f"  • {model_id}")
    print(f"    Prefix: {experiment_prefix}")


def print_start_evaluations(num_evaluations: int):
    """Print message about starting evaluations.

    Args:
        num_evaluations: Number of parallel evaluations
    """
    print(f"\n🚀 Starting {num_evaluations} parallel evaluations...\n")


def print_completion_summary(
    execution_id: str, results: List[Dict], execution_status: str, width: int = 80
):
    """Print evaluation completion summary.

    Args:
        execution_id: Execution ID
        results: List of evaluation results
        execution_status: Overall execution status
        width: Header width
    """
    print(f"\n{'='*width}")
    print("EVALUATION SUITE COMPLETED")
    print(f"Execution ID: {execution_id}")
    print(f"{'='*width}\n")

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

    print(f"\n{'='*width}")
    print(f"Total: {len(results)} evaluations")
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Status: {execution_status}")
    print(f"{'='*width}\n")


def print_resume_instructions(execution_id: str):
    """Print instructions for resuming failed evaluations.

    Args:
        execution_id: Execution ID to resume
    """
    print("💡 To resume failed evaluations, set:")
    print("   EXECUTION_MODE = 'resume'")
    print(f"   RESUME_EXECUTION_ID = '{execution_id}'  # or None for latest\n")


def print_no_execution_to_resume():
    """Print error when no execution is available to resume."""
    print("❌ No execution to resume. Run with EXECUTION_MODE='new' first.")


def print_execution_not_found(execution_id: str):
    """Print error when execution ID is not found.

    Args:
        execution_id: Execution ID that wasn't found
    """
    print(f"❌ Execution '{execution_id}' not found in config.")


def print_all_completed():
    """Print message when all models are already completed."""
    print("✅ All models already completed!")


def print_models_to_resume(num_models: int):
    """Print number of models to evaluate/resume.

    Args:
        num_models: Number of models
    """
    print(f"🔄 Models to evaluate/resume: {num_models}")


def print_searching_experiments_header():
    """Print header for searching experiments section."""
    print("\n🔍 Searching LangSmith for existing experiments...\n")

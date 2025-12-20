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
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Project root
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Subprocess script and Python executable
SINGLE_EVAL_SCRIPT = Path(__file__).parent / "run_single_evaluation.py"
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

# Evaluation configuration
NODE_NAME = "rewrite_prompt_node"
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen"
)
MAX_CONCURRENCY = 2
JUDGE_MODEL_ID = "azureopenai_gpt-4o"

# Models to evaluate
MODELS_TO_EVALUATE = [
    # "azureopenai_gpt-4o",
    # "azureopenai_gpt-4o-mini",
    # "azureopenai_gpt-4.1",
    # "azureopenai_gpt-5-nano",
    # "azureopenai_gpt-5.2-chat",
    # "xai_grok-4-1-fast-reasoning",
    # "xai_grok-4-1-fast-non-reasoning",
    # "gemini_gemini-3-pro-preview",
    # "mistral_mistral-large-2512",
    "mistral_devstral-2512",
    "mistral_codestral-2508",
]


def run_subprocess_evaluation(model_id: str) -> dict:
    """Run single evaluation in subprocess."""
    # Environment variables for subprocess
    env = os.environ.copy()
    env["EVAL_NODE_NAME"] = NODE_NAME
    env["EVAL_MODEL_ID"] = model_id
    env["EVAL_DATASET_NAME"] = DATASET_NAME
    env["EVAL_MAX_CONCURRENCY"] = str(MAX_CONCURRENCY)
    env["EVAL_JUDGE_MODEL_ID"] = JUDGE_MODEL_ID
    env["DEBUG"] = "0"

    try:
        result = subprocess.run(
            [str(PYTHON_EXE), str(SINGLE_EVAL_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            check=False,
        )

        return {
            "model_id": model_id,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0 and "SUCCESS" in result.stdout,
        }
    except (OSError, subprocess.SubprocessError) as e:
        # Catch subprocess execution errors (file not found, permission denied, etc.)
        return {
            "model_id": model_id,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def main():
    """Run all evaluations in parallel."""
    print(f"\n{'='*80}")
    print("EVALUATION SUITE - Subprocess Isolation")
    print(f"Node: {NODE_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Judge: {JUDGE_MODEL_ID}")
    print(f"Models: {len(MODELS_TO_EVALUATE)}")
    print(f"{'='*80}\n")

    for model_id in MODELS_TO_EVALUATE:
        print(f"  • {model_id}")

    print(f"\n🚀 Starting {len(MODELS_TO_EVALUATE)} parallel evaluations...\n")

    # Run evaluations in parallel (ThreadPoolExecutor for blocking subprocess calls)
    results = []
    with ThreadPoolExecutor(max_workers=len(MODELS_TO_EVALUATE)) as executor:
        futures = [
            executor.submit(run_subprocess_evaluation, model_id)
            for model_id in MODELS_TO_EVALUATE
        ]

        with tqdm(total=len(futures), desc="Evaluations", unit="eval") as pbar:
            for future in futures:
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                pbar.set_description(f"{status} {result['model_id']}")
                pbar.update(1)

    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUITE COMPLETED")
    print(f"{'='*80}\n")

    success_count = 0
    error_count = 0

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['model_id']}")

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
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    evaluation_results = main()

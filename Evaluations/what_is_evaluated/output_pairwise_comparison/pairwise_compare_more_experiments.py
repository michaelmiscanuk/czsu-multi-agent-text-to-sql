"""
Generalized pairwise comparison script for multiple LangSmith experiments.
Compares all possible pairs of experiments (including reversals) using an LLM judge.

Configuration modes:
1. Manual list: Provide experiment IDs directly in MANUAL_EXPERIMENT_IDS
2. JSON extraction: Load experiments from JSON config file by execution_id

Switch between modes by setting MANUAL_EXPERIMENT_IDS to empty list.

Results are saved to a CSV file with experiment names and preference scores.
"""

import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate_comparative
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Global lock for printing to avoid mixed output in parallel processing
print_lock = Lock()

# ============================================================================
# PROJECT SETUP
# ============================================================================

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
)
from my_agent.utils.models import get_mistral_llm
from Evaluations.utils.retry_utils import retry_with_exponential_backoff

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Load config modules directly (bypass package __init__.py)
model_configs_path = BASE_DIR / "my_agent" / "utils" / "model_configs_all.py"
model_configs_module = load_module_directly(
    "my_agent.utils.model_configs_all", model_configs_path
)
MODEL_CONFIGS_ALL = model_configs_module.MODEL_CONFIGS_ALL

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# ---- MODE 1: Manual list of experiment IDs ----
# If this list is empty, will use MODE 2 (JSON extraction)
MANUAL_EXPERIMENT_IDS: List[str] = [
    # "5c5c7f12-8020-4183-a04d-2bb364ed252e",
    # "e6655735-b51a-4da5-9ea4-d486767fb9e6",
    # "c2f6d60d-1b1e-4d00-b8a4-3196fa72e5f2",
]

# ---- MODE 2: JSON extraction configuration ----
# Used only if MANUAL_EXPERIMENT_IDS is empty
JSON_CONFIG_PATH = "Evaluations/what_is_evaluated/output_correctness/001_evaluate_output_correctness_iterate_models_parallel.json"
EXECUTION_ID = (
    "exec_2025-12-21_204506_0d147d42"  # Extract experiments from this execution
)

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Evaluation settings
MAX_CONCURRENCY = 10  # Concurrency within each pair comparison
RANDOMIZE_ORDER = True  # Mitigate positional bias

# Parallel processing settings
MAX_PARALLEL_PAIRS = (
    20  # Number of pairs to process in parallel (set to 1 for sequential)
)

# Judge model configuration
JUDGE_MODEL_ID = (
    "mistral_mistral-large-2512"  # Change this to use different judge model
)

# ============================================================================
# OUTPUT EXTRACTION CONFIGURATION
# ============================================================================

# Which node (step) to get output from
TARGET_NODE_NAME = "format_answer"

# Which state field to extract from that node
TARGET_STATE_KEY = "final_answer"

# Debug prints
DEBUG = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_all_pairs_with_reversal(experiment_ids: List[str]) -> List[Tuple[str, str]]:
    """
    Create all possible pairs with reversal (AB and BA are both included).

    Args:
        experiment_ids: List of experiment IDs

    Returns:
        List of tuples, where each tuple is (experiment_a, experiment_b)

    Example:
        Input: ["A", "B", "C"]
        Output: [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")]
    """
    pairs = []

    # Get all unique combinations
    for exp_a, exp_b in combinations(experiment_ids, 2):
        # Add both orderings (AB and BA)
        pairs.append((exp_a, exp_b))
        pairs.append((exp_b, exp_a))

    print("Number of pairs:", len(pairs))
    return pairs


def extract_experiments_from_json(json_path: str, execution_id: str) -> List[str]:
    """
    Extract experiment IDs from JSON config file for a specific execution.

    Args:
        json_path: Relative path to JSON config file
        execution_id: Execution ID to extract experiments from

    Returns:
        List of experiment IDs
    """
    full_path = BASE_DIR / json_path

    if not full_path.exists():
        raise FileNotFoundError(f"JSON config file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Navigate to the specific execution
    if "executions" not in data:
        raise ValueError("JSON must have 'executions' key")

    if execution_id not in data["executions"]:
        raise ValueError(
            f"Execution ID '{execution_id}' not found in JSON. "
            f"Available: {list(data['executions'].keys())}"
        )

    execution_data = data["executions"][execution_id]

    # Extract all experiment IDs from models
    if "models" not in execution_data:
        raise ValueError(f"No 'models' key found in execution '{execution_id}'")

    experiment_ids = []
    for model_data in execution_data["models"].values():
        if "experiment_id" in model_data:
            exp_id = model_data["experiment_id"]
            # Avoid duplicates
            if exp_id not in experiment_ids:
                experiment_ids.append(exp_id)

    if not experiment_ids:
        raise ValueError(f"No experiment IDs found in execution '{execution_id}'")

    return experiment_ids


def get_experiment_ids() -> List[str]:
    """
    Get experiment IDs based on configuration mode.

    Returns:
        List of experiment IDs
    """
    # MODE 1: Use manual list if provided
    if MANUAL_EXPERIMENT_IDS:
        print("📋 Using MANUAL experiment IDs")
        return MANUAL_EXPERIMENT_IDS

    # MODE 2: Extract from JSON
    print("📋 Extracting experiment IDs from JSON config")
    print(f"   Config: {JSON_CONFIG_PATH}")
    print(f"   Execution: {EXECUTION_ID}")

    experiment_ids = extract_experiments_from_json(JSON_CONFIG_PATH, EXECUTION_ID)
    print(f"   Found {len(experiment_ids)} experiments")

    return experiment_ids


# ============================================================================
# CSV OUTPUT FUNCTIONS
# ============================================================================


def get_csv_output_path() -> Path:
    """Get the path for CSV output file with timestamp."""
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{script_name}_{timestamp}.csv"
    return Path(__file__).parent / csv_filename


def get_experiment_name(experiment_id: str, client: Client) -> str:
    """Get experiment name from ID."""
    try:
        project = client.read_project(project_id=experiment_id)
        return project.name if project.name else experiment_id
    except Exception as e:
        print(f"⚠️ Could not get name for {experiment_id}: {e}")
        return experiment_id


def extract_winner_from_results(results: list) -> Tuple[str, str, str]:
    """Extract winner statistics from comparative results.

    Returns:
        Tuple of (winner, a_wins, b_wins) where winner is 'A', 'B', or 'TIE'
    """
    a_total = 0
    b_total = 0

    for result in results:
        if "evaluation_results" in result:
            eval_results = result["evaluation_results"]
            if "feedback.preference" in eval_results:
                feedback = eval_results["feedback.preference"]
                # Check scores in the feedback
                if hasattr(feedback, "scores") and feedback.scores:
                    scores = list(feedback.scores.values())
                    if len(scores) >= 2:
                        a_total += scores[0]
                        b_total += scores[1]

    # Determine winner
    if a_total > b_total:
        winner = "A"
    elif b_total > a_total:
        winner = "B"
    else:
        winner = "TIE"

    return winner, str(int(a_total)), str(int(b_total))


def save_results_to_csv(
    csv_path: Path, results_data: List[Dict[str, Any]], client: Client
) -> None:
    """Save comparison results to CSV file."""

    # Prepare CSV data
    csv_rows = []

    for result in results_data:
        if not result:
            continue

        exp_a_id = result["exp_a"]
        exp_b_id = result["exp_b"]

        # Get experiment names
        exp_a_name = get_experiment_name(exp_a_id, client)
        exp_b_name = get_experiment_name(exp_b_id, client)

        # Extract winner
        winner, a_wins, b_wins = extract_winner_from_results(result.get("results", []))

        csv_rows.append(
            {
                "pair_number": result["pair_idx"],
                "experiment_a_name": exp_a_name,
                "experiment_b_name": exp_b_name,
                "winner": winner,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "total_comparisons": len(result.get("results", [])),
            }
        )

    # Write to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_number",
            "experiment_a_name",
            "experiment_b_name",
            "winner",
            "a_wins",
            "b_wins",
            "total_comparisons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n💾 Results saved to CSV: {csv_path}")


# ============================================================================
# JUDGE LLM SETUP
# ============================================================================

# Judge model - lookup config from JUDGE_MODEL_ID
judge_config = get_model_config_by_id(JUDGE_MODEL_ID, MODEL_CONFIGS_ALL)
judge_llm = get_mistral_llm(
    model_name=judge_config["model_name"],
)

print(
    f"\n🔍 Judge model: {JUDGE_MODEL_ID} "
    f"({judge_config['model_provider']}/{judge_config['model_name']})"
)

# ============================================================================
# OUTPUT EXTRACTION
# ============================================================================


@retry_with_exponential_backoff(max_attempts=30, base_delay=2.0, max_delay=30.0)
def get_output(run: Run, client: Client) -> str:
    """Get output from specified node and state."""
    # Get all runs in trace
    all_runs = list(client.list_runs(trace_id=run.trace_id))

    # Find the target node
    for r in all_runs:
        if r.name == TARGET_NODE_NAME and r.outputs:
            if TARGET_STATE_KEY in r.outputs:
                return str(r.outputs[TARGET_STATE_KEY])

    return ""


# ============================================================================
# EVALUATOR FUNCTION
# ============================================================================


@retry_with_exponential_backoff(max_attempts=30, base_delay=2.0, max_delay=30.0)
def pairwise_preference_evaluator(runs: list[Run], example: Example) -> dict:
    """
    Evaluate which run is better using an LLM judge.

    Args:
        runs: List of 2 runs to compare
        example: Reference example

    Returns:
        Dictionary with scores for each run and explanation
    """

    # Use configured judge LLM
    llm = judge_llm

    # Create pairwise comparison prompt with reasoning
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert evaluator comparing two AI responses.
        
Given an input question and reference answer, determine which response (A or B) is better.
If one of the responses is empty, just make it a TIE.

Evaluate based on these criteria:

1. **Answer Relevancy**: Does the answer fully address all aspects of the question with sufficient detail?

2. **Coherence**: Is the answer well-organized, logically structured, and easy to understand with proper formatting (markdown, tables, bullets)?

3. **Informativeness**: Does the answer provide concrete details and specific values rather than vague or generic statements?

4. **Analytical Quality**: Are comparisons between values meaningful? Are trends, patterns, or changes over time clearly identified when relevant?

5. **Fluency**: Correct language matching the question (Czech/English), proper grammar, and appropriate domain terminology.

Provide your response in this exact format:
VERDICT: [A/B/TIE]
REASONING: [Your explanation of why you chose this verdict]""",
            ),
            (
                "user",
                """Input Question: {input}

Reference Answer: 

{reference}

******************************

Response A: 

{response_a}

*******************************

Response B: 

{response_b}

******************************

Which response is better? Provide verdict and reasoning:""",
            ),
        ]
    )

    client = Client()

    pred_a = get_output(runs[0], client)
    pred_b = get_output(runs[1], client)
    reference = example.outputs.get("answer", "") if example.outputs else ""
    input_text = str(example.inputs)

    if DEBUG:
        with print_lock:
            print(f"\nA: {len(pred_a)} chars, B: {len(pred_b)} chars")

    # Get LLM judgment
    chain = prompt | llm
    response = chain.invoke(
        {
            "input": input_text,
            "reference": reference,
            "response_a": pred_a,
            "response_b": pred_b,
        }
    )

    # Parse response to extract verdict and reasoning
    response_text = response.content.strip()

    # Extract verdict and reasoning
    verdict = "TIE"
    reasoning = "No reasoning provided"

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().upper()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    # Fallback: if parsing fails, try to find A, B, or TIE in the response
    if verdict not in ["A", "B", "TIE"]:
        if "A" in response_text.upper().split()[0:3]:
            verdict = "A"
        elif "B" in response_text.upper().split()[0:3]:
            verdict = "B"
        else:
            verdict = "TIE"
        reasoning = response_text

    # Convert to score format with reasoning
    if verdict == "A":
        return {
            "key": "preference",
            "scores": {runs[0].id: 1, runs[1].id: 0},
            "comment": f"✅ Response A is better. Reasoning: {reasoning}",
        }
    elif verdict == "B":
        return {
            "key": "preference",
            "scores": {runs[0].id: 0, runs[1].id: 1},
            "comment": f"✅ Response B is better. Reasoning: {reasoning}",
        }
    else:
        # Tie
        return {
            "key": "preference",
            "scores": {runs[0].id: 0.0, runs[1].id: 0.0},
            "comment": f"🤝 Tie. Reasoning: {reasoning}",
        }


def process_pair(pair_data: Tuple[int, str, str, int]) -> Optional[Dict[str, Any]]:
    """Process a single pair comparison.

    Args:
        pair_data: Tuple of (pair_idx, exp_a, exp_b, total_pairs)

    Returns:
        Dictionary with results or None on error
    """
    pair_idx, exp_a, exp_b, total_pairs = pair_data

    # Use lock for clean printing in parallel execution
    with print_lock:
        print("\n" + "=" * 80)
        print(f"📍 PAIR {pair_idx}/{total_pairs}")
        print(f"   Experiment A: {exp_a}")
        print(f"   Experiment B: {exp_b}")
        print("=" * 80)

    client = Client()

    # Run comparative evaluation with retry logic
    # Wrap the entire evaluate_comparative call to handle rate limits
    # Note: This may create duplicate experiments on retry, but it's better than failing
    @retry_with_exponential_backoff(max_attempts=30, base_delay=5.0, max_delay=120.0)
    def run_comparative_evaluation():
        results = evaluate_comparative(
            (exp_a, exp_b),
            evaluators=[pairwise_preference_evaluator],
            experiment_prefix=f"pairwise_comparison_pair_{pair_idx}",
            description=f"Pairwise comparison {pair_idx}/{total_pairs}: A vs B",
            max_concurrency=MAX_CONCURRENCY,
            client=client,
            randomize_order=RANDOMIZE_ORDER,
        )
        # Collect all results
        eval_results = list(results)
        return eval_results

    try:
        eval_results = run_comparative_evaluation()

    except Exception as e:
        with print_lock:
            print(f"\n❌ Failed pair {pair_idx}: {str(e)}")
        return None

    result_dict = {
        "pair_idx": pair_idx,
        "exp_a": exp_a,
        "exp_b": exp_b,
        "results": eval_results,
    }

    with print_lock:
        print(f"\n✅ Completed pair {pair_idx}: {len(eval_results)} comparisons")

    return result_dict


def main():
    """Run pairwise comparison for all experiment pairs."""

    print("\n" + "=" * 80)
    print("🚀 GENERALIZED PAIRWISE COMPARISON")
    print("=" * 80)

    # Get experiment IDs
    experiment_ids = get_experiment_ids()

    print(f"\n📊 Found {len(experiment_ids)} experiments")
    for i, exp_id in enumerate(experiment_ids, 1):
        print(f"   {i}. {exp_id}")

    # DEBUG: Check for duplicates
    unique_count = len(set(experiment_ids))
    if unique_count != len(experiment_ids):
        print(
            f"\n⚠️  WARNING: Found {len(experiment_ids) - unique_count} duplicate experiment IDs!"
        )
        print(f"   Unique experiments: {unique_count}")

    # Create all pairs with reversal
    pairs = create_all_pairs_with_reversal(experiment_ids)

    print(f"\n🔄 Generated {len(pairs)} comparison pairs (including reversals)")
    print(
        f"   = {len(experiment_ids)} experiments × {len(experiment_ids)-1} comparisons"
    )

    # Parallel processing configuration
    if MAX_PARALLEL_PAIRS > 1:
        print(f"\n⚡ Parallel processing: {MAX_PARALLEL_PAIRS} pairs at a time")
    else:
        print("\n🔄 Sequential processing (1 pair at a time)")

    # Prepare data for processing
    pair_data_list = [
        (idx, exp_a, exp_b, len(pairs)) for idx, (exp_a, exp_b) in enumerate(pairs, 1)
    ]

    # Process pairs (parallel or sequential)
    if MAX_PARALLEL_PAIRS == 1:
        # Sequential: simple map
        all_results = [process_pair(data) for data in pair_data_list]
    else:
        # Parallel: ThreadPoolExecutor.map
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PAIRS) as executor:
            all_results = list(executor.map(process_pair, pair_data_list))

    # Filter out None results
    all_results = [r for r in all_results if r is not None]

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 ALL COMPARISONS COMPLETED")
    print("=" * 80)
    print(f"✅ Successfully compared {len(all_results)}/{len(pairs)} pairs")
    print(f"🔗 View results at: https://smith.langchain.com")
    print("=" * 80)

    # Save results to CSV
    client = Client()
    csv_path = get_csv_output_path()
    save_results_to_csv(csv_path, all_results, client)

    print(f"\n📊 View results in: {csv_path}")


if __name__ == "__main__":
    main()

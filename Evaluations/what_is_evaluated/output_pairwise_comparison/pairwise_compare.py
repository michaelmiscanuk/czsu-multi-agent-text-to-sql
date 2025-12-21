"""
Simple pairwise comparison script for LangSmith experiments.
Compares two experiments using an LLM judge.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate_comparative
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

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
# EVALUATION CONFIGURATION
# ============================================================================

# Experiment names to compare
EXPERIMENT_A = (
    "5c5c7f12-8020-4183-a04d-2bb364ed252e"  # Replace with your first experiment ID
)
EXPERIMENT_B = (
    "e6655735-b51a-4da5-9ea4-d486767fb9e6"  # Replace with your second experiment ID
)

# Evaluation settings
MAX_CONCURRENCY = 10
RANDOMIZE_ORDER = True  # Mitigate positional bias

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
# JUDGE LLM SETUP
# ============================================================================

# Judge model - lookup config from JUDGE_MODEL_ID
judge_config = get_model_config_by_id(JUDGE_MODEL_ID, MODEL_CONFIGS_ALL)
judge_llm = get_mistral_llm(
    model_name=judge_config["model_name"],
)

print(
    f"Using judge model: {JUDGE_MODEL_ID} ({judge_config['model_provider']}/{judge_config['model_name']})"
)
print(f"Comparing experiments:")
print(f"  A: {EXPERIMENT_A}")
print(f"  B: {EXPERIMENT_B}")
print()

# ============================================================================
# OUTPUT EXTRACTION
# ============================================================================


@retry_with_exponential_backoff(max_attempts=30, base_delay=2.0, max_delay=60.0)
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


@retry_with_exponential_backoff(max_attempts=30, base_delay=2.0, max_delay=60.0)
def pairwise_preference_evaluator(runs: list[Run], example: Example) -> dict:
    """
    Evaluate which run is better using an LLM judge.

    Args:
        runs: List of 2 runs to compare
        example: Reference example

    Returns:
        Dictionary with scores for each run
    """

    # Use configured judge LLM
    llm = judge_llm

    # Create pairwise comparison prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert evaluator comparing two AI responses.
        
Given an input question and reference answer, determine which response (A or B) is better.
If one of the responses is empty, just make it a TIE.

Consider:
- CONCISENESS: Is the response clear and to the point?

Respond with exactly one word: "A", "B", or "TIE".""",
            ),
            (
                "user",
                """Input Question: {input}

Reference Answer: {reference}

Response A: {response_a}

Response B: {response_b}

Which response is better? (A/B/TIE):""",
            ),
        ]
    )

    client = Client()

    pred_a = get_output(runs[0], client)
    pred_b = get_output(runs[1], client)
    reference = example.outputs.get("answer", "") if example.outputs else ""
    input_text = str(example.inputs)

    if DEBUG:
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

    verdict = response.content.strip().upper()

    # Convert to score format
    if verdict == "A":
        return {
            "key": "preference",
            "scores": {runs[0].id: 1, runs[1].id: 0},
            "comment": f"Response A is better",
        }
    elif verdict == "B":
        return {
            "key": "preference",
            "scores": {runs[0].id: 0, runs[1].id: 1},
            "comment": f"Response B is better",
        }
    else:
        # Tie
        return {
            "key": "preference",
            "scores": {runs[0].id: 0.5, runs[1].id: 0.5},
            "comment": "Tie",
        }


def main():
    """Run pairwise comparison."""
    client = Client()

    print("\n" + "=" * 80)
    print("Starting pairwise evaluation...")
    print("=" * 80)

    # Run comparative evaluation
    results = evaluate_comparative(
        (EXPERIMENT_A, EXPERIMENT_B),  # First positional argument
        evaluators=[pairwise_preference_evaluator],
        experiment_prefix="pairwise_comparison",
        description="Pairwise comparison of two experiments",
        max_concurrency=MAX_CONCURRENCY,
        client=client,
        randomize_order=RANDOMIZE_ORDER,
    )

    print("\n" + "=" * 80)
    print("Evaluating comparisons (this may take a while)...")
    print("=" * 80 + "\n")

    # Collect results (this is where actual evaluation happens)
    eval_results = list(results)

    print("\n" + "=" * 80)
    print(f"✅ Completed {len(eval_results)} comparisons")
    print(f"View results at: https://smith.langchain.com")
    print("=" * 80)


if __name__ == "__main__":
    main()

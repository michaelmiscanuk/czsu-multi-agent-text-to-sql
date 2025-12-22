"""
Extract Pairwise Comparison Results from LangSmith

This script extracts pairwise comparison results directly from LangSmith
by querying the dataset's pairwise experiments. It reads feedback scores
from comparative evaluation runs and generates a CSV with correct scores
matching what you see in the LangSmith UI.

How it works:
1. Finds all pairwise/comparative experiments linked to a dataset
2. For each pairwise experiment, extracts feedback from all runs
3. Aggregates preference scores (how many times each experiment won)
4. Outputs CSV with the same format as the original but with correct numbers

Configuration:
- DATASET_NAME: Name of the dataset in LangSmith
- Alternative: DATASET_ID for direct UUID access
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv
from langsmith import Client

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Specify dataset by name (recommended) OR by ID
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen"
)

# Alternative: Use Dataset UUID from LangSmith URL if you prefer
# Example URL: https://smith.langchain.com/o/.../datasets/<UUID>
DATASET_ID = None  # Set to None to use DATASET_NAME instead

# Output file configuration
OUTPUT_FILENAME_PREFIX = "pairwise_results_extracted_from_langsmith"

# Debug mode
DEBUG = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_dataset_id(
    client: Client, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None
) -> str:
    """
    Get dataset ID from name or validate the provided ID.

    Args:
        client: LangSmith client
        dataset_name: Optional dataset name
        dataset_id: Optional dataset ID

    Returns:
        Dataset ID string
    """
    if dataset_id and not dataset_name:
        # Validate the ID exists
        try:
            dataset = client.read_dataset(dataset_id=dataset_id)
            if DEBUG:
                print(f"✓ Found dataset: {dataset.name} ({dataset_id})")
            return str(dataset.id)
        except Exception as e:
            raise ValueError(f"Dataset ID '{dataset_id}' not found: {e}")

    elif dataset_name:
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            if DEBUG:
                print(f"✓ Found dataset: {dataset.name} ({dataset.id})")
            return str(dataset.id)
        except Exception as e:
            raise ValueError(f"Dataset '{dataset_name}' not found: {e}")

    else:
        raise ValueError("Must provide either dataset_name or dataset_id")


def get_pairwise_experiments_for_dataset(client: Client, dataset_id: str) -> List[str]:
    """
    Find all pairwise comparison experiments for a dataset.

    Pairwise experiments are identified by:
    1. Having runs that reference examples from this dataset
    2. Having exactly 2 runs per example (pairwise comparison)
    3. Having feedback with preference/ranked scores

    Args:
        client: LangSmith client
        dataset_id: Dataset UUID

    Returns:
        List of experiment (project) IDs
    """
    if DEBUG:
        print(
            f"\n🔍 Searching for pairwise experiments linked to dataset {dataset_id}..."
        )

    # Get all examples from the dataset
    examples = list(client.list_examples(dataset_id=dataset_id))
    if not examples:
        print("⚠️  No examples found in dataset")
        return []

    if DEBUG:
        print(f"   Dataset has {len(examples)} examples")

    # Collect all experiments that have runs referencing this dataset
    all_experiments = set()

    # Sample a few examples to find experiments (checking all would be slow)
    sample_size = min(3, len(examples))
    for example in examples[:sample_size]:
        runs = list(
            client.list_runs(
                reference_example_id=example.id,
                is_root=True,
                limit=100,  # API maximum is 100
            )
        )

        for run in runs:
            if run.session_id:
                all_experiments.add(str(run.session_id))

    if DEBUG:
        print(f"   Found {len(all_experiments)} unique experiments with dataset runs")

    # Now check which experiments are pairwise comparisons
    # Pairwise experiments have exactly 2 runs per example
    pairwise_experiments = []

    if DEBUG:
        print(f"\n   Checking {len(all_experiments)} experiments...")

    for idx, exp_id in enumerate(all_experiments, 1):
        if DEBUG:
            print(f"   [{idx}/{len(all_experiments)}] Checking experiment {exp_id}...")

        try:
            # Get runs for this experiment (just sample to check pattern)
            exp_runs = list(
                client.list_runs(
                    project_id=exp_id,
                    is_root=True,
                    limit=20,  # Small sample to check pattern quickly
                )
            )

            if not exp_runs:
                continue

            # Check if runs have reference_example_id (indicates evaluation on dataset)
            runs_with_examples = [r for r in exp_runs if r.reference_example_id]

            if not runs_with_examples:
                continue

            # Group by example to check if it's pairwise
            runs_by_example = defaultdict(list)
            for run in runs_with_examples[:10]:  # Sample first 10 to check pattern
                runs_by_example[str(run.reference_example_id)].append(run)

            # Check if most examples have 2 runs (pairwise pattern)
            runs_per_example = [len(runs) for runs in runs_by_example.values()]
            avg_runs = (
                sum(runs_per_example) / len(runs_per_example) if runs_per_example else 0
            )

            # Pairwise experiments typically have 2 runs per example
            # Allow some flexibility (between 1.5 and 2.5) for incomplete experiments
            if 1.5 <= avg_runs <= 2.5:
                # Read project info first
                project = client.read_project(project_id=exp_id)
                project_name = project.name if project else ""

                # Check if project references our dataset
                has_correct_dataset = False
                if (
                    hasattr(project, "reference_dataset_id")
                    and project.reference_dataset_id
                ):
                    has_correct_dataset = str(project.reference_dataset_id) == str(
                        dataset_id
                    )

                if DEBUG:
                    print(f"      Project: {project_name}")
                    print(f"      Avg runs per example: {avg_runs:.1f}")
                    print(f"      References target dataset: {has_correct_dataset}")

                # Additional check: look for preference feedback
                has_preference_feedback = False
                for run in runs_with_examples[:5]:  # Check first 5 runs
                    feedbacks = list(client.list_feedback(run_ids=[run.id], limit=5))
                    for fb in feedbacks:
                        if any(
                            kw in fb.key.lower()
                            for kw in ["preference", "ranked", "pairwise"]
                        ):
                            has_preference_feedback = True
                            break
                    if has_preference_feedback:
                        break

                if DEBUG:
                    print(f"      Has preference feedback: {has_preference_feedback}")

                # If references correct dataset AND has preference feedback, it's pairwise
                if has_correct_dataset and has_preference_feedback:
                    pairwise_experiments.append(exp_id)
                    if DEBUG:
                        print(f"   ✓ Found pairwise experiment: {project_name}")

        except Exception as e:
            if DEBUG:
                print(f"   ⚠️  Error checking experiment {exp_id}: {e}")
            continue

    return pairwise_experiments


def extract_preference_scores_from_experiment(
    client: Client, experiment_id: str
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """

    Extract preference scores from a single pairwise comparison experiment.

    The pairwise experiment contains runs from two different experiments being compared.
    Each run has feedback indicating which one was preferred.

    Args:
        client: LangSmith client
        experiment_id: Pairwise experiment (project) ID

    Returns:
        Tuple of (experiment_a_name, experiment_b_name, a_wins, b_wins, total_comparisons)
    """
    project = client.read_project(project_id=experiment_id)
    experiment_name = project.name if project else experiment_id

    if DEBUG:
        print(f"\n📊 Processing pairwise experiment: {experiment_name}")
        print(f"   Project ID: {experiment_id}")

    # Try to extract the compared experiment names from metadata or description
    exp_a_name = "Unknown_A"
    exp_b_name = "Unknown_B"

    # Parse from project metadata if available
    if hasattr(project, "metadata") and project.metadata:
        exp_a_name = project.metadata.get("experiment_a", exp_a_name)
        exp_b_name = project.metadata.get("experiment_b", exp_b_name)
        if DEBUG:
            print(f"   Metadata: {project.metadata}")

    # KEY IMPROVEMENT: Try to use feedback_stats from project first
    # This is what the LangSmith UI displays as aggregated scores
    a_wins = 0
    b_wins = 0
    total_comparisons = 0

    if hasattr(project, "feedback_stats") and project.feedback_stats:
        if DEBUG:
            print(f"   ✅ Feedback stats found: {project.feedback_stats}")

        # feedback_stats structure varies but typically contains aggregated feedback
        # Look for preference-related feedback keys
        for key, stats in project.feedback_stats.items():
            if any(kw in key.lower() for kw in ["preference", "ranked", "pairwise"]):
                if DEBUG:
                    print(f"   Found preference feedback '{key}': {stats}")

                # The stats might have 'n' (count) and value distributions
                if isinstance(stats, dict):
                    total_comparisons = stats.get("n", 0)

                    # Check for score distribution in stats
                    # Common patterns: 'counts', 'scores', 'values', 'distribution'
                    for score_key in ["counts", "scores", "values", "distribution"]:
                        if score_key in stats:
                            score_data = stats[score_key]
                            if isinstance(score_data, dict) and len(score_data) >= 2:
                                score_values = list(score_data.values())
                                a_wins = (
                                    int(score_values[0]) if len(score_values) > 0 else 0
                                )
                                b_wins = (
                                    int(score_values[1]) if len(score_values) > 1 else 0
                                )
                                break

        if total_comparisons > 0 and (a_wins > 0 or b_wins > 0):
            if DEBUG:
                print(
                    f"   ✅ Used feedback_stats: A={a_wins}, B={b_wins}, Total={total_comparisons}"
                )
            return exp_a_name, exp_b_name, a_wins, b_wins, total_comparisons

    # Fallback: Count from runs if feedback_stats doesn't have the data
    if DEBUG:
        print("   ⚠️ Feedback stats incomplete, counting from runs...")

    # Get all runs in this pairwise experiment
    runs = list(client.list_runs(project_id=experiment_id, is_root=True))

    if DEBUG:
        print(f"   Found {len(runs)} root runs")

    # Group runs by example to match pairs
    runs_by_example = defaultdict(list)
    for run in runs:
        if run.reference_example_id:
            runs_by_example[str(run.reference_example_id)].append(run)

    # Count wins for each experiment
    a_wins = 0
    b_wins = 0
    total_comparisons = 0

    # For each example, check which run won based on feedback
    for example_id, example_runs in runs_by_example.items():
        if len(example_runs) != 2:
            if DEBUG:
                print(
                    f"   ⚠️ Example {example_id} has {len(example_runs)} runs, expected 2"
                )
            continue

        total_comparisons += 1

        # Get feedback for both runs
        # The feedback with key containing "preference" or "ranked_preference" indicates winner
        run_scores = {}

        for run in example_runs:
            feedbacks = list(client.list_feedback(run_ids=[run.id]))

            for feedback in feedbacks:
                # Look for preference/ranked feedback
                if (
                    "preference" in feedback.key.lower()
                    or "ranked" in feedback.key.lower()
                ):
                    # Score of 1 means this run won, 0 means it lost
                    if feedback.score is not None:
                        run_scores[str(run.id)] = feedback.score

                        if (
                            DEBUG and total_comparisons <= 3
                        ):  # Show details for first few
                            print(f"      Run {run.id}: score={feedback.score}")

        # Determine winner for this example
        if len(run_scores) == 2:
            scores = list(run_scores.values())
            if scores[0] > scores[1]:
                a_wins += 1
            elif scores[1] > scores[0]:
                b_wins += 1
            # If tied (both 0 or equal), neither gets a win
        elif len(run_scores) == 1:
            # Only one run has feedback (unusual case)
            if list(run_scores.values())[0] > 0:
                a_wins += 1  # Assume this is the winner

    if DEBUG:
        print(
            f"   Results from run counting: A={a_wins}, B={b_wins}, Total={total_comparisons}"
        )

    return exp_a_name, exp_b_name, a_wins, b_wins, total_comparisons


def calculate_pairwise_statistics(
    a_wins: int, b_wins: int, total: int
) -> Tuple[str, int]:
    """
    Calculate winner and ties from win counts.

    Args:
        a_wins: Number of wins for experiment A
        b_wins: Number of wins for experiment B
        total: Total number of comparisons

    Returns:
        Tuple of (winner, ties_count)
    """
    ties = total - a_wins - b_wins

    if a_wins > b_wins:
        winner = "A"
    elif b_wins > a_wins:
        winner = "B"
    else:
        winner = "TIE"

    return winner, ties


def get_csv_output_path() -> Path:
    """Get the path for CSV output file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{OUTPUT_FILENAME_PREFIX}_{timestamp}.csv"
    return Path(__file__).parent / csv_filename


def save_results_to_csv(csv_path: Path, results_data: List[Dict[str, Any]]) -> None:
    """
    Save extracted results to CSV file.

    Args:
        csv_path: Path to output CSV file
        results_data: List of result dictionaries
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_number",
            "pairwise_experiment_id",
            "pairwise_experiment_name",
            "experiment_a_name",
            "experiment_b_name",
            "winner",
            "a_wins",
            "b_wins",
            "ties",
            "total_comparisons",
            "a_preference_score",
            "b_preference_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n💾 Results saved to CSV: {csv_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Extract pairwise comparison results from LangSmith."""

    print("\n" + "=" * 80)
    print("📥 EXTRACT PAIRWISE COMPARISON RESULTS FROM LANGSMITH")
    print("=" * 80)

    # Initialize client
    client = Client()

    # Get dataset ID
    try:
        if DATASET_NAME and not DATASET_ID:
            dataset_id = get_dataset_id(client, dataset_name=DATASET_NAME)
        elif DATASET_ID:
            dataset_id = get_dataset_id(client, dataset_id=DATASET_ID)
        else:
            raise ValueError("Must set either DATASET_NAME or DATASET_ID")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease configure DATASET_ID or DATASET_NAME in the script.")
        return

    print(f"\n📦 Dataset ID: {dataset_id}")

    # Find all pairwise experiments for this dataset
    pairwise_experiments = get_pairwise_experiments_for_dataset(client, dataset_id)

    if not pairwise_experiments:
        print("\n⚠️  No pairwise comparison experiments found for this dataset.")
        print(
            "\nMake sure you have run pairwise comparisons and they are linked to this dataset."
        )
        return

    print(f"\n✓ Found {len(pairwise_experiments)} pairwise experiments")

    # Extract results from each experiment
    results_data = []

    for idx, exp_id in enumerate(pairwise_experiments, 1):
        try:
            # Extract scores from this pairwise experiment
            exp_a, exp_b, a_wins, b_wins, total = (
                extract_preference_scores_from_experiment(client, exp_id)
            )

            # Calculate winner and ties
            winner, ties = calculate_pairwise_statistics(a_wins, b_wins, total)

            # Calculate preference scores (percentage)
            a_pref = (a_wins / total * 100) if total > 0 else 0
            b_pref = (b_wins / total * 100) if total > 0 else 0

            # Get experiment name
            project = client.read_project(project_id=exp_id)
            exp_name = project.name if project else exp_id

            results_data.append(
                {
                    "pair_number": idx,
                    "pairwise_experiment_id": exp_id,
                    "pairwise_experiment_name": exp_name,
                    "experiment_a_name": exp_a,
                    "experiment_b_name": exp_b,
                    "winner": winner,
                    "a_wins": a_wins,
                    "b_wins": b_wins,
                    "ties": ties,
                    "total_comparisons": total,
                    "a_preference_score": f"{a_pref:.1f}",
                    "b_preference_score": f"{b_pref:.1f}",
                }
            )

            print(
                f"\n   ✓ Pair {idx}: A={a_wins}, B={b_wins}, Ties={ties}, Total={total}"
            )

        except Exception as e:
            print(f"\n   ❌ Error processing experiment {exp_id}: {e}")
            if DEBUG:
                import traceback

                traceback.print_exc()

    # Save to CSV
    if results_data:
        csv_path = get_csv_output_path()
        save_results_to_csv(csv_path, results_data)

        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n📊 Extracted {len(results_data)} pairwise comparison results")
        print(f"💾 Results saved to: {csv_path}")
    else:
        print(
            "\n⚠️  No results extracted. Please check the configuration and try again."
        )


if __name__ == "__main__":
    main()

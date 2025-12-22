"""
Extract pairwise comparison results from LangSmith - Version 2

This script extracts pairwise experiment results directly from LangSmith
by querying projects that reference the dataset.
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DEBUG = True  # Set to False to reduce output

# Dataset configuration
DATASET_ID = "fc824afc-f08a-488b-90ba-4cda545afbca"
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen"
)

# Output CSV file
OUTPUT_CSV = "pairwise_comparison_results.csv"


# ============================================================================
# FUNCTIONS
# ============================================================================


def get_pairwise_experiments_for_dataset(
    client: Client, dataset_id: str
) -> List[Dict[str, Any]]:
    """
    Find all pairwise comparison experiments (projects) linked to a dataset.

    Args:
        client: LangSmith client
        dataset_id: Dataset UUID

    Returns:
        List of project dictionaries with metadata
    """
    if DEBUG:
        print(f"\n🔍 Finding pairwise experiments for dataset {dataset_id}...")

    # List ALL projects - list_projects returns a generator that handles pagination
    all_projects = list(client.list_projects(limit=None))  # None = get all

    if DEBUG:
        print(f"   Total projects in workspace: {len(all_projects)}")

    # Strategy 1: Look for projects with " vs " in name (pairwise naming pattern)
    pairwise_by_name = [p for p in all_projects if " vs " in p.name.lower()]
    
    if DEBUG:
        print(f"   Projects with ' vs ' in name: {len(pairwise_by_name)}")
    
    # Strategy 2: Filter for projects referencing our dataset
    dataset_projects = [
        p for p in all_projects
        if hasattr(p, "reference_dataset_id")
        and p.reference_dataset_id
        and str(p.reference_dataset_id) == str(dataset_id)
    ]
    
    if DEBUG:
        print(f"   Projects with reference_dataset_id: {len(dataset_projects)}")
    
    # Combine both strategies - prefer " vs " projects but also check dataset refs
    candidate_projects = list(set(pairwise_by_name + dataset_projects))

    if DEBUG:
        print(f"   Total candidate projects: {len(candidate_projects)}")
            # Sample runs to check pattern
            runs = list(client.list_runs(project_id=project.id, is_root=True, limit=20))

            if not runs:
                if DEBUG:
                    print(f"      ⚠️ No runs found")
                continue

            # Check for example references
            runs_with_examples = [r for r in runs if r.reference_example_id]

            if len(runs_with_examples) < 2:
                if DEBUG:
                    print(f"      ⚠️ Not enough runs with examples")
                continue

            # Check pairwise pattern (2 runs per example)
            runs_by_example = defaultdict(list)
            for run in runs_with_examples:
                runs_by_example[str(run.reference_example_id)].append(run)

            runs_per_example = [len(r) for r in runs_by_example.values()]
            avg_runs = sum(runs_per_example) / len(runs_per_example)

            if DEBUG:
                print(
                    f"      Avg runs/example: {avg_runs:.1f} (sample: {len(runs_by_example)})"
                )

            # Check if exactly 2 runs per example (pairwise)
            if not (1.8 <= avg_runs <= 2.2):
                if DEBUG:
                    print(f"      ⚠️ Not pairwise pattern (expected ~2)")
                continue

            # Check for preference feedback
            has_preference = False
            feedbacks = list(
                client.list_feedback(run_ids=[runs_with_examples[0].id], limit=5)
            )

            for fb in feedbacks:
                if "preference" in fb.key.lower():
                    has_preference = True
                    break

            if DEBUG:
                print(f"      Preference feedback: {has_preference}")

            if has_preference:
                confirmed_pairwise.append(project)
                if DEBUG:
                    print(f"      ✅ PAIRWISE CONFIRMED")
            else:
                if DEBUG:
                    print(f"      ⚠️ No preference feedback")

        except Exception as e:
            if DEBUG:
                print(f"      ❌ Error: {e}")
            continue

    if DEBUG:
        print(f"\n✅ Found {len(confirmed_pairwise)} pairwise experiments")

    return confirmed_pairwise


def extract_pairwise_results(client: Client, project) -> Tuple[str, str, int, int, int]:
    """
    Extract preference scores from a pairwise experiment.

    Args:
        client: LangSmith client
        project: Project object

    Returns:
        (experiment_a_name, experiment_b_name, a_wins, b_wins, total)
    """
    if DEBUG:
        print(f"\n📊 Extracting results from: {project.name}")

    # Try to get experiment names from metadata
    exp_a = "Experiment_A"
    exp_b = "Experiment_B"

    if hasattr(project, "metadata") and project.metadata:
        exp_a = project.metadata.get("experiment_a", exp_a)
        exp_b = project.metadata.get("experiment_b", exp_b)

    # Try to parse from project name (format: "A vs B")
    if " vs " in project.name:
        parts = project.name.split(" vs ")
        if len(parts) == 2:
            exp_a = parts[0].strip()
            exp_b = parts[1].strip()

    # Count wins by checking all runs
    runs = list(client.list_runs(project_id=project.id, is_root=True, limit=100))

    runs_by_example = defaultdict(list)
    for run in runs:
        if run.reference_example_id:
            runs_by_example[str(run.reference_example_id)].append(run)

    a_wins = 0
    b_wins = 0
    total = 0

    for example_id, example_runs in runs_by_example.items():
        if len(example_runs) != 2:
            continue

        total += 1

        # Get preference scores for both runs
        run_scores = {}
        for run in example_runs:
            feedbacks = list(client.list_feedback(run_ids=[run.id]))
            for fb in feedbacks:
                if "preference" in fb.key.lower() and fb.score is not None:
                    run_scores[str(run.id)] = fb.score
                    break

        # Determine winner
        if len(run_scores) == 2:
            scores = list(run_scores.values())
            if scores[0] > scores[1]:
                a_wins += 1
            elif scores[1] > scores[0]:
                b_wins += 1

    if DEBUG:
        print(f"   Results: {exp_a}={a_wins}, {exp_b}={b_wins}, Total={total}")

    return exp_a, exp_b, a_wins, b_wins, total


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function."""
    print("=" * 80)
    print("📥 EXTRACT PAIRWISE COMPARISON RESULTS FROM LANGSMITH - V2")
    print("=" * 80)

    # Initialize client
    client = Client()

    # Verify dataset exists
    try:
        dataset = client.read_dataset(dataset_id=DATASET_ID)
        print(f"✓ Found dataset: {dataset.name} ({dataset.id})")
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return

    # Find pairwise experiments
    pairwise_projects = get_pairwise_experiments_for_dataset(client, DATASET_ID)

    if not pairwise_projects:
        print("\n⚠️  No pairwise experiments found!")
        return

    print(f"\n📊 Extracting results from {len(pairwise_projects)} experiments...")

    # Extract results
    results = []
    for project in pairwise_projects:
        try:
            exp_a, exp_b, a_wins, b_wins, total = extract_pairwise_results(
                client, project
            )

            results.append(
                {
                    "experiment_pair": project.name,
                    "experiment_a": exp_a,
                    "experiment_b": exp_b,
                    "a_wins": a_wins,
                    "b_wins": b_wins,
                    "ties": total - a_wins - b_wins,
                    "total_comparisons": total,
                    "project_id": project.id,
                }
            )
        except Exception as e:
            print(f"❌ Error processing {project.name}: {e}")
            continue

    # Save to CSV
    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Saved {len(results)} results to {OUTPUT_CSV}")
    else:
        print("\n⚠️  No results extracted")


if __name__ == "__main__":
    main()

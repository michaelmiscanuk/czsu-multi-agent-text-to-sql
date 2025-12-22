"""
Extract pairwise comparison results from LangSmith - Version 2

Finds pairwise experiments by looking for projects with " vs " in their names.
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

DEBUG = True

DATASET_ID = "fc824afc-f08a-488b-90ba-4cda545afbca"
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen"
)
OUTPUT_CSV = "pairwise_comparison_results.csv"


# ============================================================================
# FUNCTIONS
# ============================================================================


def get_pairwise_experiments_for_dataset(client: Client, dataset_id: str) -> List[Any]:
    """Find pairwise comparison experiments."""
    if DEBUG:
        print(f"\n🔍 Finding ALL pairwise experiments (not filtered by dataset yet)...")

    # Get all projects
    all_projects = list(client.list_projects(limit=None))

    if DEBUG:
        print(f"   Total projects: {len(all_projects)}")

    # Look for projects with " vs " in name (pairwise pattern)
    # These are the actual pairwise comparison experiments
    pairwise_candidates = [p for p in all_projects if " vs " in p.name]

    if DEBUG:
        print(f"   Projects with ' vs ' in name: {len(pairwise_candidates)}")

    # Now filter to only those using our dataset's examples
    # Check if runs reference examples from our dataset
    confirmed = []

    for idx, project in enumerate(pairwise_candidates, 1):
        if DEBUG:
            print(f"\n   [{idx}/{len(pairwise_candidates)}] {project.name}")

        try:
            # Get sample runs to verify this uses our dataset
            runs = list(client.list_runs(project_id=project.id, is_root=True, limit=10))

            if not runs:
                if DEBUG:
                    print(f"      ⚠️ No runs")
                continue

            # Check if runs reference examples from our dataset
            with_examples = [r for r in runs if r.reference_example_id]

            if not with_examples:
                if DEBUG:
                    print(f"      ⚠️ No example references")
                continue

            # Get first example and check if it belongs to our dataset
            sample_example_id = str(with_examples[0].reference_example_id)

            try:
                example = client.read_example(example_id=sample_example_id)
                if str(example.dataset_id) == str(dataset_id):
                    confirmed.append(project)
                    if DEBUG:
                        print(f"      ✅ Uses our dataset!")
                else:
                    if DEBUG:
                        print(f"      ⚠️ Different dataset: {example.dataset_id}")
            except:
                if DEBUG:
                    print(f"      ⚠️ Could not verify dataset")

        except Exception as e:
            if DEBUG:
                print(f"      ❌ Error: {e}")

    if DEBUG:
        print(f"\n✅ Found {len(confirmed)} pairwise experiments")

    return confirmed


def extract_pairwise_results(client: Client, project) -> Tuple[str, str, int, int, int]:
    """Extract preference scores from pairwise experiment."""
    if DEBUG:
        print(f"\n📊 Extracting: {project.name}")

    # Parse experiment names from project name
    exp_a = "Experiment_A"
    exp_b = "Experiment_B"

    if " vs " in project.name:
        parts = project.name.split(" vs ")
        if len(parts) == 2:
            exp_a = parts[0].strip()
            exp_b = parts[1].strip()

    # Get all runs and count wins
    runs = list(client.list_runs(project_id=project.id, is_root=True, limit=100))

    by_example = defaultdict(list)
    for run in runs:
        if run.reference_example_id:
            by_example[str(run.reference_example_id)].append(run)

    a_wins = 0
    b_wins = 0
    total = 0

    for example_id, example_runs in by_example.items():
        if len(example_runs) != 2:
            continue

        total += 1

        # Get preference scores
        scores = {}
        for run in example_runs:
            feedbacks = list(client.list_feedback(run_ids=[run.id]))
            for fb in feedbacks:
                if "preference" in fb.key.lower() and fb.score is not None:
                    scores[str(run.id)] = fb.score
                    break

        # Determine winner
        if len(scores) == 2:
            score_list = list(scores.values())
            if score_list[0] > score_list[1]:
                a_wins += 1
            elif score_list[1] > score_list[0]:
                b_wins += 1

    if DEBUG:
        print(f"   {exp_a}={a_wins}, {exp_b}={b_wins}, Total={total}")

    return exp_a, exp_b, a_wins, b_wins, total


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution."""
    print("=" * 80)
    print("📥 EXTRACT PAIRWISE RESULTS - V2 (vs-based search)")
    print("=" * 80)

    client = Client()

    # Verify dataset
    try:
        dataset = client.read_dataset(dataset_id=DATASET_ID)
        print(f"✓ Dataset: {dataset.name}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Find pairwise experiments
    pairwise_projects = get_pairwise_experiments_for_dataset(client, DATASET_ID)

    if not pairwise_projects:
        print("\n⚠️  No pairwise experiments found!")
        print("\nTip: Pairwise experiments should have ' vs ' in their names.")
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

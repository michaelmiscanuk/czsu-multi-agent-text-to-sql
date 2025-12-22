"""
Debug script to find pairwise experiments in LangSmith.
"""

import os
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = Client()

# Dataset ID from your config
dataset_id = "fc824afc-f08a-488b-90ba-4cda545afbca"

print("=" * 80)
print("🔍 DEBUG: Finding Pairwise Experiments")
print("=" * 80)

# Get dataset
dataset = client.read_dataset(dataset_id=dataset_id)
print(f"\n✓ Dataset: {dataset.name}")
print(f"  ID: {dataset.id}")

# Strategy 1: List ALL projects and look for metadata
print("\n" + "=" * 80)
print("Strategy 1: List all projects and examine metadata")
print("=" * 80)

projects = list(client.list_projects(limit=100))
print(f"\nFound {len(projects)} total projects")

# Look for projects with "judge" or "vs" in name (from your screenshot)
pairwise_candidates = []
for project in projects:
    # Check name patterns from screenshot
    if any(
        kw in project.name.lower() for kw in ["vs", "judge", "pairwise", "comparative"]
    ):
        pairwise_candidates.append(project)

print(f"\nFound {len(pairwise_candidates)} projects matching pairwise name patterns")

# Examine first few candidates
for i, project in enumerate(pairwise_candidates[:3], 1):
    print(f"\n--- Candidate {i} ---")
    print(f"Name: {project.name}")
    print(f"ID: {project.id}")
    print(f"Metadata: {project.metadata if hasattr(project, 'metadata') else 'N/A'}")

    # Check for experiment_type or other identifying fields
    project_dict = project.dict() if hasattr(project, "dict") else vars(project)
    print(f"All fields: {list(project_dict.keys())}")

    # Check for special fields
    for key in [
        "experiment_type",
        "type",
        "comparison_type",
        "is_comparative",
        "is_pairwise",
    ]:
        if hasattr(project, key):
            print(f"  {key}: {getattr(project, key)}")

    # Check runs pattern
    runs = list(client.list_runs(project_id=project.id, is_root=True, limit=10))
    print(f"  Number of runs (first 10): {len(runs)}")

    if runs:
        # Check reference_example_id pattern
        runs_with_examples = [r for r in runs if r.reference_example_id]
        print(f"  Runs with reference_example_id: {len(runs_with_examples)}")

        # Check feedback
        if runs_with_examples:
            sample_run = runs_with_examples[0]
            feedbacks = list(client.list_feedback(run_ids=[sample_run.id], limit=5))
            print(f"  Feedback on sample run: {len(feedbacks)}")
            for fb in feedbacks:
                print(f"    - {fb.key}: score={fb.score}, value={fb.value}")

# Strategy 2: Check if dataset has linked experiments in metadata
print("\n" + "=" * 80)
print("Strategy 2: Check dataset metadata for linked experiments")
print("=" * 80)

dataset_dict = dataset.dict() if hasattr(dataset, "dict") else vars(dataset)
print(f"\nDataset fields: {list(dataset_dict.keys())}")

if hasattr(dataset, "metadata") and dataset.metadata:
    print(f"\nDataset metadata: {dataset.metadata}")

# Check for experiments field
if hasattr(dataset, "experiments"):
    print(f"\nDataset experiments: {dataset.experiments}")

# Strategy 3: Look for sessions with specific reference patterns
print("\n" + "=" * 80)
print("Strategy 3: Find experiments by checking reference_example_id from dataset")
print("=" * 80)

examples = list(client.list_examples(dataset_id=dataset_id, limit=5))
print(f"\nChecking first {len(examples)} examples...")

for i, example in enumerate(examples[:2], 1):
    print(f"\n--- Example {i} ---")
    print(f"ID: {example.id}")

    # Get all runs for this example
    runs = list(
        client.list_runs(reference_example_id=example.id, is_root=True, limit=50)
    )
    print(f"Total runs referencing this example: {len(runs)}")

    # Group by session
    from collections import defaultdict

    runs_by_session = defaultdict(list)
    for run in runs:
        if run.session_id:
            runs_by_session[run.session_id].append(run)

    print(f"Unique sessions: {len(runs_by_session)}")

    # Show sessions with exactly 2 runs (pairwise pattern)
    for session_id, session_runs in runs_by_session.items():
        if len(session_runs) == 2:
            project = client.read_project(project_id=session_id)
            print(f"\n  Pairwise session found: {project.name}")
            print(f"    Session ID: {session_id}")
            print(
                f"    Metadata: {project.metadata if hasattr(project, 'metadata') else 'N/A'}"
            )

print("\n" + "=" * 80)
print("✓ Debug complete")
print("=" * 80)

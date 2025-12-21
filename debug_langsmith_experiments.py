"""Debug script to list LangSmith experiments and find the correct ones."""

import sys
from pathlib import Path
from langsmith import Client

# Setup path
BASE_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(BASE_DIR))


def main():
    client = Client()

    print("\n" + "=" * 80)
    print("LISTING ALL LANGSMITH EXPERIMENTS (PROJECTS)")
    print("=" * 80 + "\n")

    # List all projects
    projects = list(client.list_projects(limit=50))

    print(f"Found {len(projects)} experiments:\n")

    for i, project in enumerate(projects, 1):
        print(f"{i}. Name: {project.name}")
        print(f"   ID: {project.id}")
        print(f"   Created: {project.created_at}")
        if hasattr(project, "run_count"):
            print(f"   Runs: {project.run_count}")
        print()

    # Search for our specific experiments
    print("\n" + "=" * 80)
    print("SEARCHING FOR FORMAT_ANSWER_NODE EXPERIMENTS")
    print("=" * 80 + "\n")

    matching = [
        p
        for p in projects
        if "format_answer_node" in p.name.lower()
        or "format-answer-node" in p.name.lower()
    ]

    if matching:
        print(f"Found {len(matching)} matching experiments:\n")
        for project in matching:
            print(f"Name: {project.name}")
            print(f"ID: {project.id}")

            # Count runs in this experiment
            runs = list(client.list_runs(project_id=project.id))
            completed_runs = [r for r in runs if r.end_time is not None]

            print(f"Total runs: {len(runs)}")
            print(f"Completed runs: {len(completed_runs)}")

            # Count unique examples
            unique_examples = {
                r.reference_example_id for r in completed_runs if r.reference_example_id
            }
            print(f"Unique examples evaluated: {len(unique_examples)}")
            print()
    else:
        print("No matching experiments found!")
        print("\nTrying broader search...")

        for keyword in ["mistral", "gpt", "judge"]:
            matches = [p for p in projects if keyword in p.name.lower()]
            if matches:
                print(f"\nFound {len(matches)} projects containing '{keyword}':")
                for p in matches[:5]:
                    print(f"  - {p.name}")


if __name__ == "__main__":
    main()

"""
Quick script to explore LangSmith API for comparative experiments
"""

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

# Check for comparative experiment methods
print("=== Methods with 'experiment' or 'project' in name ===")
methods = [
    m
    for m in dir(client)
    if not m.startswith("_")
    and (
        "experiment" in m.lower()
        or "project" in m.lower()
        or "comparative" in m.lower()
    )
]
for m in sorted(methods):
    print(f"  {m}")

print("\n=== Checking read_project for comparative experiment ===")
# Try reading a project (experiment) and checking its attributes
dataset_id = "fc824afc-f08a-488b-90ba-4cda545afbca"

# Get a sample run from the dataset
examples = list(client.list_examples(dataset_id=dataset_id, limit=1))
if examples:
    example_id = examples[0].id
    runs = list(
        client.list_runs(reference_example_id=example_id, limit=5, is_root=True)
    )

    if runs:
        project_id = runs[0].session_id
        print(f"\nSample project ID: {project_id}")

        # Read the project
        project = client.read_project(project_id=project_id)

        print(f"\nProject attributes:")
        for attr in dir(project):
            if not attr.startswith("_"):
                print(f"  {attr}: {getattr(project, attr, 'N/A')}")

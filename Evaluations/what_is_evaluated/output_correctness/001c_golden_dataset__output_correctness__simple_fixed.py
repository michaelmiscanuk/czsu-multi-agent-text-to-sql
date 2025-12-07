import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Config
DATASET_NAME = "001c_golden_dataset__output_correctness__simple_fixed"

# Load environment variables from .env file at project root
project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

# Your questions and answers
# Define questions and answers for the golden dataset
# Each question is preceded by a comment showing the filename and exact source line from the CSV data
question_answers = {
    # STA01T1.csv; Česko;"Stavební práce ""S"" celkem (mil. Kč, b.c.)";2024;695799.256
    # changed 'construction work to 'construction production' because of better translation to match with data
    "What was the total value of construction production in Czechia in 2024?": "695799.256 million CZK",
}

ls_client = Client()


# Get or create dataset
try:
    dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)
    print(f"Dataset '{dataset.name}' found with ID: {dataset.id}")
except LangSmithNotFoundError:
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Dataset of Czech Statistical Office agent questions and answers.",
    )
    print(f"Dataset '{dataset.name}' created with ID: {dataset.id}")

# Fetch existing examples' questions to avoid duplicates
existing_examples = ls_client.list_examples(dataset_id=dataset.id)
existing_questions = set()
for ex in existing_examples:
    q = ex.inputs.get("question")
    if q:
        existing_questions.add(q)

# Prepare only new examples (filter out duplicates)
new_examples = []
for question, answer in question_answers.items():
    if question not in existing_questions:
        new_examples.append(
            {"inputs": {"question": question}, "outputs": {"answers": answer}}
        )

if new_examples:
    ls_client.create_examples(dataset_id=dataset.id, examples=new_examples)
    print(f"Added {len(new_examples)} new examples to dataset '{dataset.name}'.")
else:
    print("No new examples to add; all questions already exist in the dataset.")

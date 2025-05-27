from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Configuration
CONFIG = {
    "dataset_name": "czsu agent selection retrieval",
    "dataset_description": "Dataset of Czech Statistical Office agent For Evaluation of step to find the right selection with ChromaDB"
}

# Your questions and answers
# Define questions and answers for the golden dataset
question_answers = {
    "Jake odvetvi ma nejvyssi prumerne mzdy?": "MZDQ1T2",
    # Answer: J - Informační a komunikační činnosti.
    
    "Jaký je meziměsíční index spotřebitelských cen za služby v dubnu 2025?": "CEN0101LT01",
    # Answer: 100.2
    
    "Kolik cizinců z Ukrajiny žilo v Česku v roce 2023?": "CIZ01D",
    # Answer: 574447
    
    "Kolik hromadných ubytovacích zařízení bylo v Česku v roce 2024?": "CRU01RT3",
    # Answer: 10104
}

ls_client = Client()

# Get or create dataset
try:
    dataset = ls_client.read_dataset(dataset_name=CONFIG["dataset_name"])
    print(f"Dataset '{dataset.name}' found with ID: {dataset.id}")
except LangSmithNotFoundError:
    dataset = ls_client.create_dataset(
        dataset_name=CONFIG["dataset_name"],
        description=CONFIG["dataset_description"]
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
        new_examples.append({"inputs": {"question": question}, "outputs": {"answers": answer}})

if new_examples:
    ls_client.create_examples(dataset_id=dataset.id, examples=new_examples)
    print(f"Added {len(new_examples)} new examples to dataset '{dataset.name}'.")
else:
    print("No new examples to add; all questions already exist in the dataset.")
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
    "Jake odvetvi ma nejvyssi prumerne mzdy?": "J - Informační a komunikační činnosti.",
    
    "Jaký je meziměsíční index spotřebitelských cen za služby v dubnu 2025?": "100.2",
    
    "Kolik cizinců z Ukrajiny žilo v Česku v roce 2023?": "574447",
    
    "Kolik hromadných ubytovacích zařízení bylo v Česku v roce 2024?": "10104",
    
    "Jaká byla výroba elektřiny celkem v Česku v roce 2023?": "77245.787",
    
    "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?": "330962.822",
    
    "Jaké byly daňové příjmy v Česku v roce 2023?": "447346687.505279",
    
    "Jaké byly úvěry a půjčky poskytnuté bankami v Česku v roce 2022?": "5786069.256",
    
    "Jaké byly přijaté dividendy a podíly na zisku investičními společnostmi a fondy v Česku v roce 2022?": "2644.083",
    
    "Kolik aktivních podniků bylo v ICT sektoru v Česku v roce 2023?": "60043",
    
    "Jaký byl podíl osob používajících internet v Česku ve věku 16 a vice v roce 2023?": "85.9941%",
    
    "Jaká byla rozloha státních lesů v Česku k 31. 12. 2023?": "1443991.67",
    
    "Jaký byl průměrný počet zaměstnanců v odvětví F - Stavebnictví v Česku v roce 2021?": "214.2 tis. osob",
    
    "Kolik uchazečů o zaměstnání s vysokoškolským vzděláním bylo v evidenci ÚP v Česku v roce 2023?": "25004.0",
    
    "Kolik bylo uchazečů o zaměstnání na uradu prace dele nez 24 mesice roku 2023?": "44168.0",
    
    "Jaká byla průměrná kupní cena bytů v Česku za období 2017-2019 v Kč/m2?": "26514.0",
    
    "Kolik bylo ekonomických subjektů v Česku v roce 2024?": "2836017"
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
        new_examples.append({
            "inputs": {"question": question},
            "outputs": {"answer": answer},
            "metadata": {"source": "CZSU"}
        })

if new_examples:
    ls_client.create_examples(dataset_id=dataset.id, examples=new_examples)
    print(f"Added {len(new_examples)} new examples to dataset '{dataset.name}'.")
else:
    print("No new examples to add; all questions already exist in the dataset.")
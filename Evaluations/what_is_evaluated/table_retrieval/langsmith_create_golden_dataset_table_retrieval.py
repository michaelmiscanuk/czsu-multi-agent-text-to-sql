"""Golden dataset for table retrieval evaluation.

This script creates or updates a LangSmith dataset with question-table pairs
for evaluating the Czech Statistical Office agent's ability to retrieve the
correct table based on the question.

The dataset contains curated questions with their expected table names.
"""

from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset name matches filename for easy identification
DATASET_NAME = Path(__file__).stem

# Load environment variables from project root
project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

# ============================================================================
# QUESTION-TABLE PAIRS
# ============================================================================

# Define questions and table names for the golden dataset
# Each entry includes:
# - question: Natural language query about Czech statistical data
# - answer: Expected table name (CSV file without .csv extension)
question_answers = [
    {
        "question": "What was the average consumer price of oranges in Czechia in March 2025?",
        "answer": "CEN0101DT01",
    },
    {
        "question": "What was the average consumer price of honey in Czechia in October 2024?",
        "answer": "CEN0101DT01",
    },
    {
        "question": "What was the average annual inflation rate in Czechia in 2022?",
        "answer": "CEN0101HT01",
    },
    {
        "question": "What was the average consumer price of gasoline in Czechia in May 2025?",
        "answer": "CEN0101JT01",
    },
    {
        "question": "What was the index of construction material input prices in Czechia in Q4 2024?",
        "answer": "CEN0202CT01",
    },
    {
        "question": "What was the number of foreigners in Czechia from Ukraine in 2023?",
        "answer": "CIZ01D",
    },
    {
        "question": "What was the number of camping and caravan spots in collective accommodation facilities in Zlín Region - Luhačovice in 2024?",
        "answer": "CRU01RDMOT1",
    },
    {
        "question": "What was the number of guests in hotels, motels, and botels in Czechia in August 2025?",
        "answer": "CRU02MHOTT1",
    },
    {
        "question": "What were the costs of longer trips for leisure and recreation to Greece from Czechia in 2023?",
        "answer": "CRU06AT5",
    },
    {
        "question": "What was the number of transported persons (thousands) by air transport in Czechia in 2024?",
        "answer": "DOP01BT1",
    },
    {
        "question": "What was the number of registered vehicles in Czechia in 2024?",
        "answer": "DOP02BT1",
    },
    {
        "question": "What was the total electricity production in Czechia in 2024?",
        "answer": "ENE01WENET1",
    },
    {
        "question": "What was the number of ATMs and terminals in Czechia in 2023?",
        "answer": "FIN03BANKDUK",
    },
    {
        "question": "What was the number of active enterprises in the ICT sector in Czechia in 2023?",
        "answer": "ICT01T01",
    },
    {
        "question": "What was the average gross monthly wage per adjusted employee in Prague in Q1-Q2 2025?",
        "answer": "MZDKQT5",
    },
    {
        "question": "What was the average gross monthly wage per individual in Education in Czechia in 2024?",
        "answer": "MZDRT2",
    },
    {
        "question": "What was the number of job applicants registered at the Labor Office in Prague in 2023?",
        "answer": "NEZ01T1",
    },
    {
        "question": "What was the gross domestic product in Czechia in 2024?",
        "answer": "NUC06RT01",
    },
    {
        "question": "What was the number of divorces in Czechia in 2024?",
        "answer": "OBY01AT01",
    },
    {
        "question": "What was the age median for women in Czechia in 2024?",
        "answer": "OBY02FGEN",
    },
    {
        "question": "What is the forecast of the number of live births in Czechia in 2044?",
        "answer": "OBY02PT02",
    },
    {
        "question": "What was the number of private entrepreneurs operating under the Trade Licensing Act in Czechia in Q3 2025?",
        "answer": "RES01QT1",
    },
    {
        "question": "What was the total number of economic entities in Prague in Q3 2025?",
        "answer": "RES02QT4",
    },
    {
        "question": "What was the number of uninhabited apartments in Prague in 2021?",
        "answer": "SLD023T02",
    },
    {
        "question": "What was the number of inhabited houses in Prague in 1980?",
        "answer": "SLD053T02",
    },
    {
        "question": "What was the number of building permits in Czechia in April 2025?",
        "answer": "STA08A1T1",
    },
    {
        "question": "What was the average value of the paid monthly pension in Prague in 2023?",
        "answer": "SZB01A",
    },
    {
        "question": "What was the value of paid state social support benefits in Czechia in 2024?",
        "answer": "SZB02",
    },
    {
        "question": "What was the number of social service facilities in Czechia in 2023?",
        "answer": "SZB07A1",
    },
    {
        "question": "What was the voter turnout percentage in Czechia in 2025?",
        "answer": "VOLPST2",
    },
]

# ============================================================================
# DATASET MANAGEMENT
# ============================================================================


def get_or_create_dataset(client: Client, name: str, description: str):
    """Get existing dataset or create new one.

    Args:
        client: LangSmith client instance
        name: Dataset name
        description: Dataset description

    Returns:
        Dataset object
    """
    try:
        dataset = client.read_dataset(dataset_name=name)
        print(f"Dataset '{dataset.name}' found with ID: {dataset.id}")
    except LangSmithNotFoundError:
        dataset = client.create_dataset(dataset_name=name, description=description)
        print(f"Dataset '{dataset.name}' created with ID: {dataset.id}")
    return dataset


def get_existing_questions(client: Client, dataset_id: str) -> set:
    """Fetch all existing questions from dataset to avoid duplicates.

    Args:
        client: LangSmith client instance
        dataset_id: Dataset ID

    Returns:
        Set of existing questions
    """
    existing_examples = client.list_examples(dataset_id=dataset_id)
    return {
        ex.inputs.get("question")
        for ex in existing_examples
        if ex.inputs.get("question")
    }


def prepare_new_examples(qa_pairs: list, existing_questions: set) -> list:
    """Prepare new examples by filtering out duplicates.

    Args:
        qa_pairs: List of QA dictionaries
        existing_questions: Set of already existing questions

    Returns:
        List of new example dictionaries ready for LangSmith
    """
    return [
        {
            "inputs": {"question": qa["question"]},
            "outputs": {"answer": qa["answer"]},
        }
        for qa in qa_pairs
        if qa["question"] not in existing_questions
    ]


def main():
    """Create or update the golden dataset in LangSmith."""
    ls_client = Client()

    # Get or create dataset
    dataset = get_or_create_dataset(
        ls_client,
        DATASET_NAME,
        "Dataset for evaluating table retrieval - matching questions to correct CSV tables.",
    )

    # Get existing questions to avoid duplicates
    existing_questions = get_existing_questions(ls_client, dataset.id)

    # Prepare new examples
    new_examples = prepare_new_examples(question_answers, existing_questions)

    # Add new examples to dataset
    if new_examples:
        ls_client.create_examples(dataset_id=dataset.id, examples=new_examples)
        print(f"Added {len(new_examples)} new examples to dataset '{dataset.name}'.")
    else:
        print("No new examples to add; all questions already exist in the dataset.")


if __name__ == "__main__":
    main()

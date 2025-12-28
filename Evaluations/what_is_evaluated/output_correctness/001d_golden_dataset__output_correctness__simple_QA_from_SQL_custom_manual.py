"""Golden dataset for output correctness evaluation.

This script creates or updates a LangSmith dataset with question-answer pairs
for evaluating the Czech Statistical Office agent's ability to answer questions
correctly based on SQL query results.

The dataset contains curated questions with their expected answers and source
information from CSV files in the CZSU database.
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
# QUESTION-ANSWER PAIRS
# ============================================================================

# Define questions and answers for the golden dataset
# Each entry includes:
# - question: Natural language query about Czech statistical data
# - answer: Expected answer from the database
# - source: CSV file and exact data line for verification
question_answers = [
    {
        "question": "What was the average consumer price of oranges in Czechia in March 2025?",
        "answer": "34.27 CZK per kg",
        "source": "CEN0101DT01.csv; Průměrné spotřebitelské ceny zboží a služeb;Česko;Pomeranče [1 kg];březen 2025;34.27",
    },
    {
        "question": "What was the average consumer price of honey in Czechia in October 2024?",
        "answer": "178.28 CZK per kg",
        "source": "CEN0101DT01.csv; Průměrné spotřebitelské ceny zboží a služeb;Česko;Pravý včelí med [1 kg];říjen 2024;178.28",
    },
    {
        "question": "What was the average annual inflation rate in Czechia in 2022?",
        "answer": "15.1%",
        "source": "CEN0101HT01.csv; Česko;2022;Průměrná roční míra inflace;15.1",
    },
    {
        "question": "What was the average consumer price of gasoline in Czechia in May 2025?",
        "answer": "33.94 CZK per liter",
        "source": "CEN0101JT01.csv; Průměrná cena pohonných hmot;Česko;květen 2025;Benzin automobilový bezolovnatý Natural 95 [Kč/l];33.94",
    },
    {
        "question": "What was the index of construction material input prices in Czechia in Q4 2024?",
        "answer": "149.9",
        "source": "CEN0202CT01.csv; Index cen materiálových vstupů do stavebnictví;Česko;Stavební díla;4. čtvrtletí 2024;149.9",
    },
    {
        "question": "What was the number of foreigners in Czechia from Ukraine in 2023?",
        "answer": "574447",
        "source": "CIZ01D.csv; Cizinci podle státního občanství a pohlaví;Celkem;Česko;Ukrajina;2023;574447.0",
    },
    {
        "question": "What was the number of camping and caravan spots in collective accommodation facilities in Zlín Region - Luhačovice in 2024?",
        "answer": "280",
        "source": "CRU01RDMOT1.csv; 2024;HUZ celkem;Zlínsko - Luhačovicko;Počet míst pro stany a karavany v hromadných ubytovacích zařízeních 	;280.0",
    },
    {
        "question": "What was the number of guests in hotels, motels, and botels in Czechia in August 2025?",
        "answer": "618316",
        "source": "CRU02MHOTT1.csv; Česko;srpen 2025;Hotel, motel, botel ***;Počet hostů;Celkem;618316.0",
    },
    {
        "question": "What were the costs of longer trips for leisure and recreation to Greece from Czechia in 2023?",
        "answer": "7885.7160123 million CZK",
        "source": "CRU06AT5.csv; Česko;2023;Delší cesty za účelem trávení volného času a rekreace;Řecko;Náklady (mil. Kč);7885.7160123",
    },
    {
        "question": "What was the number of transported persons (thousands) by air transport in Czechia in 2024?",
        "answer": "5100.4",
        "source": "DOP01BT1.csv; Česko;2024;Přepravené osoby (tis.);Letecká doprava;5100.4",
    },
    {
        "question": "What was the number of registered vehicles in Czechia in 2024?",
        "answer": "780571",
        "source": "DOP02BT1.csv; Česko;Počet registrovaných dopravních prostředků;2024;Nákladní vozidla;780571.0",
    },
    {
        "question": "What was the total electricity production in Czechia in 2024?",
        "answer": "73881.775 GWh",
        "source": "ENE01WENET1.csv; Česko;Výroba elektřiny celkem (GWh);2024;73881.775",
    },
    {
        "question": "What was the number of ATMs and terminals in Czechia in 2023?",
        "answer": "4963",
        "source": "FIN03BANKDUK.csv; Česko;Bankovní peněžní instituce;Počet bankomatů a terminálů, k poslednímu dni sledovaného roku, ks;2023;4963.0",
    },
    {
        "question": "What was the number of active enterprises in the ICT sector in Czechia in 2023?",
        "answer": "60043",
        "source": "ICT01T01.csv; Počet aktivních podniků;Česko;ICT sektor;2023;60043",
    },
    {
        "question": "What was the average gross monthly wage per adjusted employee in Prague in Q1-Q2 2025?",
        "answer": "62395.036985578 CZK",
        "source": "MZDKQT5.csv; Q1-Q2 2025;Pracovištní metoda;Praha;Průměrná hrubá měsíční mzda na přepočtené počty zaměstnanců (Kč);62395.036985578",
    },
    {
        "question": "What was the average gross monthly wage per individual in Education in Czechia in 2024?",
        "answer": "38999.1122333656 CZK",
        "source": "MZDRT2.csv; Česko;2024;P - Vzdělávání;Průměrná hrubá měsíční mzda na fyzické osoby (Kč);38999.1122333656",
    },
    {
        "question": "What was the number of job applicants registered at the Labor Office in Prague in 2023?",
        "answer": "27425",
        "source": "NEZ01T1.csv; 2023;Hlavní město Praha;Uchazeči o zaměstnání v evidenci ÚP - celkem;27425.0",
    },
    {
        "question": "What was the gross domestic product in Czechia in 2024?",
        "answer": "8057032.0 million CZK",
        "source": "NUC06RT01.csv; Česko;Hrubý domácí produkt (mil. Kč, běžné ceny);2024;8057032.0",
    },
    {
        "question": "What was the number of divorces in Czechia in 2024?",
        "answer": "20796",
        "source": "OBY01AT01.csv; Q1-Q4 2024;Česko;Rozvody;20796",
    },
    {
        "question": "What was the age median for women in Czechia in 2024?",
        "answer": "45.66620224 years",
        "source": "OBY02FGEN.csv; 2024;Česko;Věkový medián (roky);Ženy;45.66620224",
    },
    {
        "question": "What is the forecast of the number of live births in Czechia in 2044?",
        "answer": "93746",
        "source": "OBY02PT02.csv; Česko;Živě narození;2044;93746.0",
    },
    {
        "question": "What was the number of private entrepreneurs operating under the Trade Licensing Act in Czechia in Q3 2025?",
        "answer": "1797952",
        "source": "RES01QT1.csv; Česko;Počet ekonomických subjektů v registru;Soukromí podnikatelé podnikající podle živnostenského zákona;3. čtvrtletí 2025;1797952",
    },
    {
        "question": "What was the total number of economic entities in Prague in Q3 2025?",
        "answer": "687339",
        "source": "RES02QT4.csv; Počet ekonomických subjektů celkem;Celkem;Hlavní město Praha;3. čtvrtletí 2025;687339",
    },
    {
        "question": "What was the number of uninhabited apartments in Prague in 2021?",
        "answer": "93627",
        "source": "SLD023T02.csv; 2021;Počet bytů;Hlavní město Praha;Neobydlené byty;93627",
    },
    {
        "question": "What was the number of inhabited houses in Prague in 1980?",
        "answer": "75794",
        "source": "SLD053T02.csv; Obydlené domy celkem;Hlavní město Praha;1980;75794",
    },
    {
        "question": "What was the number of building permits in Czechia in April 2025?",
        "answer": "18568",
        "source": "STA08A1T1.csv; Česko;duben 2025;Počet stavebních povolení v ČR;18568.0",
    },
    {
        "question": "What was the average value of the paid monthly pension in Prague in 2023?",
        "answer": "20897 CZK",
        "source": "SZB01A.csv; Průměrná hodnota vyplaceného (měsíčního) důchodu (Kč);Důchody celkem;Celkem;Hlavní město Praha;2023;20897",
    },
    {
        "question": "What was the value of paid state social support benefits in Czechia in 2024?",
        "answer": "5411668.986 thousand CZK",
        "source": "SZB02.csv; Česko;Hodnota vyplacených dávek státní sociální podpory (tis. Kč);Přídavek na dítě;2024;5411668.986",
    },
    {
        "question": "What was the number of social service facilities in Czechia in 2023?",
        "answer": "526",
        "source": "SZB07A1.csv; Počet zařízení sociálních služeb;2023;Česko;Domovy pro seniory;526",
    },
    {
        "question": "What was the voter turnout percentage in Czechia in 2025?",
        "answer": "68.95%",
        "source": "VOLPST2.csv; 2025;Česko;Volební účast (%);68.95",
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
            "metadata": {"source": qa["source"]},
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
        "Dataset of Czech Statistical Office agent questions and answers.",
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

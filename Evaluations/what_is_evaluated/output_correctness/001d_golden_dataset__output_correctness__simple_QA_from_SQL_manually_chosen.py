import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Config
DATASET_NAME = (
    "001d_golden_dataset__output_correctness__simple_QA_from_SQL_manually_chosen"
)

# Load environment variables from .env file at project root
project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

# Your questions and answers
# Define questions and answers for the golden dataset
# Each question is preceded by a comment showing the filename and exact source line from the CSV data
question_answers = {
    # CEN0101DT01.csv; Průměrné spotřebitelské ceny zboží a služeb;Česko;Pomeranče [1 kg];březen 2025;34.27
    "What was the average consumer price of oranges in Czechia in March 2025?": "34.27 CZK per kg",
    # CEN0101DT01.csv; Průměrné spotřebitelské ceny zboží a služeb;Česko;Pravý včelí med [1 kg];říjen 2024;178.28
    "What was the average consumer price of honey in Czechia in October 2024?": "178.28 CZK per kg",
    # CEN0101HT01.csv; Česko;2022;Průměrná roční míra inflace;15.1
    "What was the average annual inflation rate in Czechia in 2022?": "15.1%",
    # CEN0101JT01.csv; Průměrná cena pohonných hmot;Česko;květen 2025;Benzin automobilový bezolovnatý Natural 95 [Kč/l];33.94
    "What was the average consumer price of gasoline in Czechia in May 2025?": "33.94 CZK per liter",
    # CEN0202CT01.csv; Index cen materiálových vstupů do stavebnictví;Česko;Stavební díla;4. čtvrtletí 2024;149.9
    "What was the index of construction material input prices in Czechia in Q4 2024?": "149.9",
    # CIZ01D.csv; Cizinci podle státního občanství a pohlaví;Celkem;Česko;Ukrajina;2023;574447.0
    "What was the number of foreigners in Czechia from Ukraine in 2023?": "574447",
    # CRU01RDMOT1.csv; 2024;HUZ celkem;Zlínsko - Luhačovicko;Počet míst pro stany a karavany v hromadných ubytovacích zařízeních 	;280.0
    "What was the number of camping and caravan spots in collective accommodation facilities in Zlín Region - Luhačovice in 2024?": "280",
    # CRU02MHOTT1.csv; Česko;srpen 2025;Hotel, motel, botel ***;Počet hostů;Celkem;618316.0
    "What was the number of guests in hotels, motels, and botels in Czechia in August 2025?": "618316",
    # CRU06AT5.csv; Česko;2023;Delší cesty za účelem trávení volného času a rekreace;Řecko;Náklady (mil. Kč);7885.7160123
    "What were the costs of longer trips for leisure and recreation to Greece from Czechia in 2023?": "7885.7160123 million CZK",
    # DOP01BT1.csv; Česko;2024;Přepravené osoby (tis.);Letecká doprava;5100.4
    "What was the number of transported persons (thousands) by air transport in Czechia in 2024?": "5100.4",
    # DOP02BT1.csv; Česko;Počet registrovaných dopravních prostředků;2024;Nákladní vozidla;780571.0
    "What was the number of registered vehicles in Czechia in 2024?": "780571",
    # ENE01WENET1.csv; Česko;Výroba elektřiny celkem (GWh);2024;73881.775
    "What was the total electricity production in Czechia in 2024?": "73881.775 GWh",
    # FIN03BANKDUK.csv; Česko;Bankovní peněžní instituce;Počet bankomatů a terminálů, k poslednímu dni sledovaného roku, ks;2023;4963.0
    "What was the number of ATMs and terminals in Czechia in 2023?": "4963",
    # ICT01T01.csv; Počet aktivních podniků;Česko;ICT sektor;2023;60043
    "What was the number of active enterprises in the ICT sector in Czechia in 2023?": "60043",
    # MZDKQT5.csv; Q1-Q2 2025;Pracovištní metoda;Praha;Průměrná hrubá měsíční mzda na přepočtené počty zaměstnanců (Kč);62395.036985578
    "What was the average gross monthly wage per adjusted employee in Prague in Q1-Q2 2025?": "62395.036985578 CZK",
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

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Config
DATASET_NAME = "czsu agent simple 2"

# Load environment variables from .env file at project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# Your questions and answers
# Define questions and answers for the golden dataset
# Each question is preceded by a comment showing the filename and exact source line from the CSV data
question_answers = {
    # LES0101T01.csv; Česko;Lesní pozemky k 31. 12. (ha);2024;2683138.3181
    "What was the total forest land area in Czechia as of December 31, 2024 in hectares?": "2683138.3181",
    # # LES0101T01.csv; Česko;Lesní pozemky k 31. 12. - státní lesy (ha);2024;1443909.6
    # "How many hectares of state-owned forests were there in Czechia as of December 31, 2024?": "1443909.6",
    # # KRI01T01.csv; Česko;2024;Usmrcené osoby;22.0
    # "How many fatalities were recorded in Czechia in 2024?": "22.0",
    # # KRI01T01.csv; Česko;2020;Věcná škoda (tis. Kč);293027.835
    # "What was the property damage in Czechia in 2020 in thousands of CZK?": "293027.835",
    # # DOP01T1.csv; Česko;3. čtvrtletí 2024;Přepravené osoby (tis.);Železniční doprava;50122.384
    # "How many passengers (in thousands) used railway transport in Czechia in Q3 2024?": "50122.384",
    # # DOP01T1.csv; Česko;4. čtvrtletí 2024;Přepravené osoby (tis.);Letecká doprava;2506.36
    # "How many passengers (in thousands) used air transport in Czechia in Q4 2024?": "2506.36",
    # # STA01T1.csv; Česko;Počet nových stavebních zakázek v tuzemsku;2024;82584.0
    # "How many new construction orders were placed in Czechia in 2024?": "82584.0",
    # # STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku (mil. Kč, b.c.);2024;387400.305
    # "What was the value of new construction orders in Czechia in 2024 in million CZK?": "387400.305",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2022;597168.6159
    # "What were the total healthcare expenditures in Czechia in 2022 in million CZK?": "597168.6159",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Léčebná péče;2020;254044.1055
    # "What was the expenditure on curative care in Czechia in 2020 in million CZK?": "254044.1055",
    # # OBY01AT01.csv; Q1-Q4 2024;Česko;Počet obyvatel - stav na konci období;10909500
    # "What was the population of Czechia at the end of the period in Q1-Q4 2024?": "10909500",
    # # OBY01AT01.csv; Q1-Q4 2024;Česko;Sňatky;44486
    # "How many marriages were registered in Czechia in Q1-Q4 2024?": "44486",
    # # ENE01WENET1.csv; Česko;Výroba elektřiny celkem (GWh);2024;73881.775
    # "What was the total electricity production in Czechia in 2024 in GWh?": "73881.775",
    # # ENE01WENET1.csv; Česko;Výroba elektřiny - jaderné elektrárny (GWh);2024;29696.397
    # "How much electricity was produced by nuclear power plants in Czechia in 2024 in GWh?": "29696.397",
    # # CIZ01D.csv; Cizinci;Celkem;Česko;Celkem;2023;1065740.0
    # "How many foreigners were living in Czechia in 2023?": "1065740.0",
    # # CIZ01D.csv; Cizinci;Celkem;Česko;Slovensko;2023;119182.0
    # "How many Slovak citizens were living in Czechia in 2023?": "119182.0",
    # # VAV0101T01.csv; Počet aktivních podniků;Česko;Celkem;2023;64785
    # "How many active enterprises were there in Czechia in 2023?": "64785",
    # # VAV0101T01.csv; Počet aktivních podniků;Česko;Činnosti v oblasti ICT;2023;49523
    # "How many active enterprises in the ICT sector were there in Czechia in 2023?": "49523",
    # # ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Česko;Zaměstnaní (tis. osob);5243.4576818
    # "How many employed persons (in thousands) were there in Czechia in Q2 2025?": "5243.4576818",
    # # ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Česko;Obecná míra nezaměstnanosti (%);2.7100827254
    # "What was the general unemployment rate in Czechia in Q2 2025 (in %)?": "2.7100827254",
    # # CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;HUZ celkem;10293.0
    # "How many collective accommodation establishments were there in Czechia in 2023?": "10293.0",
    # # CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;Penzion;4373.0
    # "How many pensions (guesthouses) were there in Czechia in 2023?": "4373.0",
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

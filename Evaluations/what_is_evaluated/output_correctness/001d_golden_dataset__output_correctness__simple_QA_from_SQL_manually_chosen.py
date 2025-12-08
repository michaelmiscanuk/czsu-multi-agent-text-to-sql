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
    # MZDRT2.csv; Česko;2024;P - Vzdělávání;Průměrná hrubá měsíční mzda na fyzické osoby (Kč);38999.1122333656
    "What was the average gross monthly wage per individual in Education in Czechia in 2024?": "38999.1122333656 CZK",
    # NEZ01T1.csv; 2023;Hlavní město Praha;Uchazeči o zaměstnání v evidenci ÚP - celkem;27425.0
    "What was the number of job applicants registered at the Labor Office in Prague in 2023?": "27425",
    # NUC06RT01.csv; Česko;Hrubý domácí produkt (mil. Kč, běžné ceny);2024;8057032.0
    "What was the gross domestic product in Czechia in 2024?": "8057032.0 million CZK",
    # OBY01AT01.csv; Q1-Q4 2024;Česko;Rozvody;20796
    "What was the number of divorces in Czechia in 2024?": "20796",
    # OBY02FGEN.csv; 2024;Česko;Věkový medián (roky);Ženy;45.66620224
    "What was the age median for women in Czechia in 2024?": "45.66620224 years",
    # OBY02PT02.csv; Česko;Živě narození;2044;93746.0
    "What is the forecast of the number of live births in Czechia in 2044?": "93746",
    # RES01QT1.csv; Česko;Počet ekonomických subjektů v registru;Soukromí podnikatelé podnikající podle živnostenského zákona;3. čtvrtletí 2025;1797952
    "What was the number of private entrepreneurs operating under the Trade Licensing Act in Czechia in Q3 2025?": "1797952",
    # RES02QT4.csv; Počet ekonomických subjektů celkem;Celkem;Hlavní město Praha;3. čtvrtletí 2025;687339
    "What was the total number of economic entities in Prague in Q3 2025?": "687339",
    # SLD023T02.csv; 2021;Počet bytů;Hlavní město Praha;Neobydlené byty;93627
    "What was the number of uninhabited apartments in Prague in 2021?": "93627",
    # SLD053T02.csv; Obydlené domy celkem;Hlavní město Praha;1980;75794
    "What was the number of inhabited houses in Prague in 1980?": "75794",
    # STA08A1T1.csv; Česko;duben 2025;Počet stavebních povolení v ČR;18568.0
    "What was the number of building permits in Czechia in April 2025?": "18568",
    # SZB01B.csv; Průměrná hodnota vyplaceného (měsíčního) důchodu (Kč);Důchody celkem;Celkem;Hlavní město Praha;2023;20897
    "What was the average value of the paid monthly pension in Prague in 2023?": "20897 CZK",
    # SZB02.csv; Česko;Hodnota vyplacených dávek státní sociální podpory (tis. Kč);Přídavek na dítě;2024;5411668.986
    "What was the value of paid state social support benefits in Czechia in 2024?": "5411668.986 thousand CZK",
    # SZB07A1.csv; Počet zařízení sociálních služeb;2023;Česko;Domovy pro seniory;526
    "What was the number of social service facilities in Czechia in 2023?": "526",
    # VOLPST2.csv; 2025;Česko;Volební účast (%);68.95
    "What was the voter turnout percentage in Czechia in 2025?": "68.95%",
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

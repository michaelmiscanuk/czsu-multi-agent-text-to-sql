import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Config
DATASET_NAME = "001b_golden_dataset__output_correctness__simple_QA_from_SQL"

# Load environment variables from .env file at project root
project_root = Path(__file__).resolve().parents[3]
load_dotenv(project_root / ".env")

# Your questions and answers
# Define questions and answers for the golden dataset
# Each question is preceded by a comment showing the filename and exact source line from the CSV data
question_answers = [
    {
        "question": "What was the total forest land area in Czechia as of December 31, 2024?",
        "answer": "2683138.3181 hectares",
        "source": "LES0101T01.csv; Česko;Lesní pozemky k 31. 12. (ha);2024;2683138.3181",
    },
    {
        "question": "What was the state forest land area in Czechia as of December 31, 2024?",
        "answer": "1445388.67 hectares",
        "source": "LES0101T01.csv; Česko;Lesní pozemky k 31. 12. - státní lesy (ha);2024;1445388.67",
    },
    {
        "question": "What was the total stocking land area in Czechia as of December 31, 2024?",
        "answer": "2617907.25 hectares",
        "source": "LES0101T01.csv; Česko;Porostní půda k 31. 12. - celkem (ha);2024;2617907.25",
    },
    {
        "question": "How many employed persons were there in Czechia in Q2 2025?",
        "answer": "5243.4576818 thousands",
        "source": "ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Česko;Zaměstnaní (tis. osob);5243.4576818",
    },
    {
        "question": "What was the employment rate in Prague in Q2 2025?",
        "answer": "65.26723058%",
        "source": "ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Hlavní město Praha;Míra zaměstnanosti (%);65.26723058",
    },
    {
        "question": "What was the general unemployment rate in the Moravian-Silesian Region in Q2 2025?",
        "answer": "4.5478411545%",
        "source": "ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Moravskoslezský kraj;Obecná míra nezaměstnanosti (%);4.5478411545",
    },
    {
        "question": "What was the total number of collective accommodation establishments in Czechia in 2023?",
        "answer": "10293.0",
        "source": "CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;HUZ celkem;10293.0",
    },
    {
        "question": "How many five-star hotels were there in Czechia in 2023?",
        "answer": "75.0",
        "source": "CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;Hotel *****;75.0",
    },
    {
        "question": "How many pensions (guesthouses) were there in Czechia in 2023?",
        "answer": "4373.0",
        "source": "CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;Penzion;4373.0",
    },
    {
        "question": "How many passengers were transported by railway in Czechia in Q2 2025?",
        "answer": "47072.33 thousands",
        "source": "DOP01T1.csv; Česko;2. čtvrtletí 2025;Přepravené osoby (tis.);Železniční doprava;47072.33",
    },
    {
        "question": "How many passengers were transported by air in Czechia in Q3 2024?",
        "answer": "2661.5 thousands",
        "source": "DOP01T1.csv; Česko;3. čtvrtletí 2024;Přepravené osoby (tis.);Letecká doprava;2661.5",
    },
    {
        "question": "How many passengers used urban public transport in Czechia in Q1 2020?",
        "answer": "489459.0 thousands",
        "source": "DOP01T1.csv; Česko;1. čtvrtletí 2020;Přepravené osoby (tis.);Městská hromadná doprava;489459.0",
    },
    {
        "question": "What were the total revenues of regions, municipalities and associations in Czechia in 2024?",
        "answer": "867882737.363661",
        "source": "FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Česko;Příjmy celkem;867882737.363661",
    },
    {
        "question": "What were the tax revenues for Prague in 2024?",
        "answer": "97210769.13452",
        "source": "FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Hlavní město Praha;Daňové příjmy ;97210769.13452",
    },
    {
        "question": "What were the total healthcare expenditures in Czechia in 2022?",
        "answer": "597168.6159 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2022;597168.6159",
    },
    {
        "question": "What were the expenditures for curative care in Czechia in 2022?",
        "answer": "317237.8962 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Léčebná péče;2022;317237.8962",
    },
    {
        "question": "What were the expenditures for rehabilitation care in Czechia in 2022?",
        "answer": "30595.41736 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Rehabilitační péče;2022;30595.41736",
    },
    {
        "question": "What were the expenditures for long-term healthcare in Czechia in 2022?",
        "answer": "75916.50296 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Dlouhodobá zdravotní péče;2022;75916.50296",
    },
    {
        "question": "What were the expenditures for medicines and medical devices in Czechia in 2022?",
        "answer": "96316.8245 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Léky a zdravotnické prostředky;2022;96316.8245",
    },
    {
        "question": "What were the expenditures for preventive care in Czechia in 2021?",
        "answer": "48575.0076 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Preventivní péče;2021;48575.0076",
    },
    {
        "question": "What was the year-on-year construction production index in Czechia in 2024?",
        "answer": "98.578503576%",
        "source": "STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2024;98.578503576",
    },
    {
        "question": "What was the year-on-year construction production index for building construction in Czechia in 2024?",
        "answer": "97.3045074658%",
        "source": "STA01T1.csv; Česko;Meziroční index stavební produkce - pozemní stavitelství (%);2024;97.3045074658",
    },
    {
        "question": "What was the year-on-year construction production index for civil engineering in Czechia in 2024?",
        "answer": "100.9961318622%",
        "source": "STA01T1.csv; Česko;Meziroční index stavební produkce - inženýrské stavby (%);2024;100.9961318622",
    },
    {
        "question": "What was the total value of construction production in Czechia in 2024?",
        "answer": "695799.256 million CZK",
        "source": 'STA01T1.csv; Česko;"Stavební práce ""S"" celkem (mil. Kč, b.c.)";2024;695799.256',
    },
    {
        "question": "How many new domestic construction orders were there in Czechia in 2024?",
        "answer": "82584.0",
        "source": "STA01T1.csv; Česko;Počet nových stavebních zakázek v tuzemsku;2024;82584.0",
    },
    {
        "question": "What was the value of new domestic construction orders in Czechia in 2024?",
        "answer": "387400.305 million CZK",
        "source": "STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku (mil. Kč, b.c.);2024;387400.305",
    },
    {
        "question": "What was the value of new domestic building construction orders in Czechia in 2024?",
        "answer": "176996.681 million CZK",
        "source": "STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku-pozemní stavitelství (mil. Kč, b.c.);2024;176996.681",
    },
    {
        "question": "What was the value of new domestic civil engineering orders in Czechia in 2024?",
        "answer": "210403.624 million CZK",
        "source": "STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku-inženýrské stavitelství (mil. Kč, b.c.);2024;210403.624",
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in August 2025?",
        "answer": "95.8100451617",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;srpen 2025;95.8100451617",
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in July 2025?",
        "answer": "104.8848778126",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;červenec 2025;104.8848778126",
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in April 2021?",
        "answer": "148.9912538947",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;duben 2021;148.9912538947",
    },
    {
        "question": "What was the industrial production index for mining and quarrying in Czechia in July 2021?",
        "answer": "127.3538078822",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Těžba a dobývání;červenec 2021;127.3538078822",
    },
    {
        "question": "What was the industrial production index for manufacturing in Czechia in April 2021?",
        "answer": "154.1152520125",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;duben 2021;154.1152520125",
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in April 2020?",
        "answer": "67.7670728017",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;duben 2020;67.7670728017",
    },
    {
        "question": "What was the industrial production index for manufacturing in Czechia in May 2021?",
        "answer": "135.5927296893",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;květen 2021;135.5927296893",
    },
    {
        "question": "What was the industrial production index for electricity, gas, steam and air conditioning supply in Czechia in February 2025?",
        "answer": "117.4876789237",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Výroba a rozvod elektřiny, plynu, tepla a klimatizovaného vzduchu;únor 2025;117.4876789237",
    },
    {
        "question": "What were the total healthcare expenditures in Czechia in 2020?",
        "answer": "522797.6729 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2020;522797.6729",
    },
    {
        "question": "What were the total healthcare expenditures in Czechia in 2010?",
        "answer": "301598.984 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2010;301598.984",
    },
    {
        "question": "What was the year-on-year construction production index in Czechia in 2020?",
        "answer": "92.9563354325%",
        "source": "STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2020;92.9563354325",
    },
    {
        "question": "What was the year-on-year construction production index in Czechia in 2018?",
        "answer": "109.1510381946%",
        "source": "STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2018;109.1510381946",
    },
    {
        "question": "What was the value of construction work abroad by Czech companies in 2024?",
        "answer": "14565.166 million CZK",
        "source": 'STA01T1.csv; Česko;"Stavební práce ""S"" v zahraničí (mil. Kč, b.c.)";2024;14565.166',
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in May 2020?",
        "answer": "72.6902421609",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;květen 2020;72.6902421609",
    },
    {
        "question": "What were the expenditures for healthcare system administration in Czechia in 2022?",
        "answer": "11863.24328 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Správa systému zdravotní péče;2022;11863.24328",
    },
    {
        "question": "What were the expenditures for ancillary healthcare services in Czechia in 2022?",
        "answer": "28801.72143 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Doplňkové služby;2022;28801.72143",
    },
    {
        "question": "What was the industrial production index for total industry in Czechia in January 2001?",
        "answer": "119.1851959979",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;leden 2001;119.1851959979",
    },
    {
        "question": "What was the industrial production index for manufacturing in Czechia in January 2001?",
        "answer": "126.2458803827",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;leden 2001;126.2458803827",
    },
    {
        "question": "What was the value of domestic construction work in Czechia in 2024?",
        "answer": "681234.09 million CZK",
        "source": 'STA01T1.csv; Česko;"Stavební práce ""S"" v tuzemsku (mil. Kč, b.c.)";2024;681234.0900000001',
    },
    {
        "question": "What were the total healthcare expenditures in Czechia in 2021?",
        "answer": "577424.9378 million CZK",
        "source": "ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2021;577424.9378",
    },
    {
        "question": "What was the industrial production index for mining and quarrying in Czechia in December 2023?",
        "answer": "85.4808346576",
        "source": "PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Těžba a dobývání;prosinec 2023;85.4808346576",
    },
    {
        "question": "What were the total expenditures of regions, municipalities and associations in Czechia in 2024?",
        "answer": "815740301.872018",
        "source": "FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Česko;Výdaje celkem;815740301.872018",
    },
]

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
for item in question_answers:
    if item["question"] not in existing_questions:
        new_examples.append(
            {
                "inputs": {"question": item["question"]},
                "outputs": {"answer": item["answer"]},
                "metadata": {"source": item["source"]},
            }
        )

if new_examples:
    ls_client.create_examples(dataset_id=dataset.id, examples=new_examples)
    print(f"Added {len(new_examples)} new examples to dataset '{dataset.name}'.")
else:
    print("No new examples to add; all questions already exist in the dataset.")

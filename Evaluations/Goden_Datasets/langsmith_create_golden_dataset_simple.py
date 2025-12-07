import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

# Config
DATASET_NAME = "czsu agent simple 3"

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

load_dotenv()

# Your questions and answers
# Define questions and answers for the golden dataset
# Each question is preceded by a comment showing the filename and exact source line from the CSV data
question_answers = {
    # LES0101T01.csv; Česko;Lesní pozemky k 31. 12. (ha);2024;2683138.3181
    "What was the total forest land area in Czechia as of December 31, 2024 in hectares?": "2683138.3181",
    # # LES0101T01.csv; Česko;Lesní pozemky k 31. 12. - státní lesy (ha);2024;1445388.67
    # "What was the state forest land area in Czechia as of December 31, 2024 in hectares?": "1445388.67",
    # # LES0101T01.csv; Česko;Porostní půda k 31. 12. - celkem (ha);2024;2617907.25
    # "What was the total stocking land area in Czechia as of December 31, 2024 in hectares?": "2617907.25",
    # # ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Česko;Zaměstnaní (tis. osob);5243.4576818
    # "How many employed persons (in thousands) were there in Czechia in Q2 2025?": "5243.4576818",
    # # ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Hlavní město Praha;Míra zaměstnanosti (%);65.26723058
    # "What was the employment rate in Prague in Q2 2025?": "65.26723058",
    # # ZAM01T1.csv; 2. čtvrtletí 2025;Celkem;Moravskoslezský kraj;Obecná míra nezaměstnanosti (%);4.5478411545
    # "What was the general unemployment rate in the Moravian-Silesian Region in Q2 2025?": "4.5478411545",
    # # CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;HUZ celkem;10293.0
    # "What was the total number of collective accommodation establishments in Czechia in 2023?": "10293.0",
    # # CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;Hotel *****;75.0
    # "How many five-star hotels were there in Czechia in 2023?": "75.0",
    # # CRU01T1.csv; Počet hromadných ubytovacích zařízení;2023;Česko;Penzion;4373.0
    # "How many pensions (guesthouses) were there in Czechia in 2023?": "4373.0",
    # # DOP01T1.csv; Česko;2. čtvrtletí 2025;Přepravené osoby (tis.);Železniční doprava;47072.33
    # "How many passengers (in thousands) were transported by railway in Czechia in Q2 2025?": "47072.33",
    # # DOP01T1.csv; Česko;3. čtvrtletí 2024;Přepravené osoby (tis.);Letecká doprava;2661.5
    # "How many passengers (in thousands) were transported by air in Czechia in Q3 2024?": "2661.5",
    # # DOP01T1.csv; Česko;1. čtvrtletí 2020;Přepravené osoby (tis.);Městská hromadná doprava;489459.0
    # "How many passengers (in thousands) used urban public transport in Czechia in Q1 2020?": "489459.0",
    # # FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Česko;Příjmy celkem;867882737.363661
    # "What were the total revenues of regions, municipalities and associations in Czechia in 2024?": "867882737.363661",
    # # FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Hlavní město Praha;Daňové příjmy ;97210769.13452
    # "What were the tax revenues for Prague in 2024?": "97210769.13452",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2022;597168.6159
    # "What were the total healthcare expenditures in Czechia in 2022 in million CZK?": "597168.6159",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Léčebná péče;2022;317237.8962
    # "What were the expenditures for curative care in Czechia in 2022 in million CZK?": "317237.8962",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Rehabilitační péče;2022;30595.41736
    # "What were the expenditures for rehabilitation care in Czechia in 2022 in million CZK?": "30595.41736",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Dlouhodobá zdravotní péče;2022;75916.50296
    # "What were the expenditures for long-term healthcare in Czechia in 2022 in million CZK?": "75916.50296",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Léky a zdravotnické prostředky;2022;96316.8245
    # "What were the expenditures for medicines and medical devices in Czechia in 2022 in million CZK?": "96316.8245",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Preventivní péče;2021;48575.0076
    # "What were the expenditures for preventive care in Czechia in 2021 in million CZK?": "48575.0076",
    # # STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2024;98.578503576
    # "What was the year-on-year construction production index in Czechia in 2024?": "98.578503576",
    # # STA01T1.csv; Česko;Meziroční index stavební produkce - pozemní stavitelství (%);2024;97.3045074658
    # "What was the year-on-year construction production index for building construction in Czechia in 2024?": "97.3045074658",
    # # STA01T1.csv; Česko;Meziroční index stavební produkce - inženýrské stavby (%);2024;100.9961318622
    # "What was the year-on-year construction production index for civil engineering in Czechia in 2024?": "100.9961318622",
    # # STA01T1.csv; Česko;"Stavební práce ""S"" celkem (mil. Kč, b.c.)";2024;695799.256
    # "What was the total value of construction work in Czechia in 2024 in million CZK at current prices?": "695799.256",
    # # STA01T1.csv; Česko;Počet nových stavebních zakázek v tuzemsku;2024;82584.0
    # "How many new domestic construction orders were there in Czechia in 2024?": "82584.0",
    # # STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku (mil. Kč, b.c.);2024;387400.305
    # "What was the value of new domestic construction orders in Czechia in 2024 in million CZK?": "387400.305",
    # # STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku-pozemní stavitelství (mil. Kč, b.c.);2024;176996.681
    # "What was the value of new domestic building construction orders in Czechia in 2024 in million CZK?": "176996.681",
    # # STA01T1.csv; Česko;Hodnota nových stavebních zakázek v tuzemsku-inženýrské stavitelství (mil. Kč, b.c.);2024;210403.624
    # "What was the value of new domestic civil engineering orders in Czechia in 2024 in million CZK?": "210403.624",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;srpen 2025;95.8100451617
    # "What was the industrial production index for total industry in Czechia in August 2025?": "95.8100451617",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;červenec 2025;104.8848778126
    # "What was the industrial production index for total industry in Czechia in July 2025?": "104.8848778126",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;duben 2021;148.9912538947
    # "What was the industrial production index for total industry in Czechia in April 2021?": "148.9912538947",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Těžba a dobývání;červenec 2021;127.3538078822
    # "What was the industrial production index for mining and quarrying in Czechia in July 2021?": "127.3538078822",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;duben 2021;154.1152520125
    # "What was the industrial production index for manufacturing in Czechia in April 2021?": "154.1152520125",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;duben 2020;67.7670728017
    # "What was the industrial production index for total industry in Czechia in April 2020?": "67.7670728017",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;květen 2021;135.5927296893
    # "What was the industrial production index for manufacturing in Czechia in May 2021?": "135.5927296893",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Výroba a rozvod elektřiny, plynu, tepla a klimatizovaného vzduchu;únor 2025;117.4876789237
    # "What was the industrial production index for electricity, gas, steam and air conditioning supply in Czechia in February 2025?": "117.4876789237",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2020;522797.6729
    # "What were the total healthcare expenditures in Czechia in 2020 in million CZK?": "522797.6729",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2010;301598.984
    # "What were the total healthcare expenditures in Czechia in 2010 in million CZK?": "301598.984",
    # # STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2020;92.9563354325
    # "What was the year-on-year construction production index in Czechia in 2020?": "92.9563354325",
    # # STA01T1.csv; Česko;Meziroční index stavební produkce celkem (%);2018;109.1510381946
    # "What was the year-on-year construction production index in Czechia in 2018?": "109.1510381946",
    # # STA01T1.csv; Česko;"Stavební práce ""S"" v zahraničí (mil. Kč, b.c.)";2024;14565.166
    # "What was the value of construction work abroad by Czech companies in 2024 in million CZK?": "14565.166",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;květen 2020;72.6902421609
    # "What was the industrial production index for total industry in Czechia in May 2020?": "72.6902421609",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Správa systému zdravotní péče;2022;11863.24328
    # "What were the expenditures for healthcare system administration in Czechia in 2022 in million CZK?": "11863.24328",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Doplňkové služby;2022;28801.72143
    # "What were the expenditures for ancillary healthcare services in Czechia in 2022 in million CZK?": "28801.72143",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Průmysl celkem;leden 2001;119.1851959979
    # "What was the industrial production index for total industry in Czechia in January 2001?": "119.1851959979",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Zpracovatelský průmysl;leden 2001;126.2458803827
    # "What was the industrial production index for manufacturing in Czechia in January 2001?": "126.2458803827",
    # # STA01T1.csv; Česko;"Stavební práce ""S"" v tuzemsku (mil. Kč, b.c.)";2024;681234.0900000001
    # "What was the value of domestic construction work in Czechia in 2024 in million CZK at current prices?": "681234.09",
    # # ZDR01T1.csv; Výdaje za zdravotní péči celkem (mil. Kč);Česko;Celkem;Celkem;2021;577424.9378
    # "What were the total healthcare expenditures in Czechia in 2021 in million CZK?": "577424.9378",
    # # PRU01T1.csv; Česko;Index průmyslové produkce;Meziroční index;Bez očištění;Těžba a dobývání;prosinec 2023;85.4808346576
    # "What was the industrial production index for mining and quarrying in Czechia in December 2023?": "85.4808346576",
    # # FIN01T1.csv; 2024;Souhrn rozpočtů krajů, obcí a svazků obcí;Česko;Výdaje celkem;815740301.872018
    # "What were the total expenditures of regions, municipalities and associations in Czechia in 2024?": "815740301.872018",
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

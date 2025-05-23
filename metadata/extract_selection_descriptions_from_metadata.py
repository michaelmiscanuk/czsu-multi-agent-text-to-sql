import json
import requests
import csv
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
PROCESS_ALL_DATASETS = 1  # Set to 1 to process all datasets, 0 to process specific dataset
SPECIFIC_DATASET_ID = "CRU01ROBCE"  # Only used when PROCESS_ALL_DATASETS is 0
PROCESS_ALL_SELECTIONS = 1  # Set to 1 to process all selections, 0 to process specific selection
SPECIFIC_SELECTION_ID = "CRU01ROBCET1"  # Only used when PROCESS_ALL_SELECTIONS is 0

def fetch_json(url):
    """
    Fetch JSON data from a given URL with English language preference.
    Returns the parsed JSON or None if the request fails.
    """
    try:
        headers = {"Accept-Language": "en"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        time.sleep(0.1)  # Be nice to the API (rate limiting)
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    """
    Main function to extract selection descriptions from all datasets (or a specific one),
    and write them to a CSV file. Each description includes the selection name, time granularity,
    and territory types, all in English if available.
    """
    output_file = Path('metadata/selection_descriptions.csv')
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        # Write CSV header
        writer.writerow(['selection_code', 'description'])
        
        # 1. Fetch all datasets from the API
        datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
        datasets = fetch_json(datasets_url)
        if not datasets:
            print("No datasets found.")
            return
        
        # 2. Filter datasets if needed (all or specific)
        datasets_to_process = datasets if PROCESS_ALL_DATASETS else [d for d in datasets if d.get('kod') == SPECIFIC_DATASET_ID]
        
        # 3. Process each dataset with a progress bar
        for dataset in tqdm(datasets_to_process, desc="Datasets", unit="dataset"):
            dataset_id = dataset.get('kod')
            if not dataset_id:
                continue
            
            # 4. Fetch all selections for this dataset
            selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
            selections = fetch_json(selections_url)
            if not selections:
                continue
            
            # 5. Filter selections if needed (all or specific)
            selections_to_process = selections if PROCESS_ALL_SELECTIONS else [s for s in selections if s.get('kod') == SPECIFIC_SELECTION_ID]
            
            # 6. Process each selection
            for selection in selections_to_process:
                selection_id = selection.get('kod')
                nazev = selection.get('nazev', '')
                # Extract all time granularity names (e.g., 'Year')
                time_levels = ', '.join([
                    lvl.get('nazevUrovne', '')
                    for lvl in selection.get('urovneTypObdobi', [])
                    if lvl.get('nazevUrovne')
                ])
                # Extract all territory type names (e.g., 'Region, Municipality')
                territory_levels = ', '.join([
                    lvl.get('nazevUrovne', '')
                    for lvl in selection.get('urovneTypUzemi', [])
                    if lvl.get('nazevUrovne')
                ])
                # Build the description string
                description = nazev
                if time_levels:
                    description += f". Time granularity: {time_levels}"
                if territory_levels:
                    description += f". Territory types: {territory_levels}"
                # Write the selection code and description to the CSV
                writer.writerow([selection_id, description])
    print(f"\nDescriptions have been saved to {output_file}")

if __name__ == "__main__":
    main()

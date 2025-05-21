import requests
import pandas as pd
from pyjstat import pyjstat
import os
import time
from pathlib import Path
from tqdm import tqdm
import json

# Create data/CSVs directory if it doesn't exist
csv_dir = Path("data/CSVs")
csv_dir.mkdir(parents=True, exist_ok=True)

def fetch_json(url):
    """Helper function to fetch JSON data with error handling and rate limiting"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(0.1)  # Rate limiting to be nice to the API
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_to_csv(df, filename):
    """Helper function to save DataFrame to CSV"""
    try:
        output_path = csv_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved: {filename} to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def main():
    # 1. Get list of all datasets
    print("Fetching list of datasets...")
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    datasets = fetch_json(datasets_url)
    
    if not datasets:
        print("Failed to fetch datasets list")
        return

    # Let's examine the structure of the first dataset
    if datasets:
        print("\nExample dataset structure:")
        print(json.dumps(datasets[0], indent=2, ensure_ascii=False))

    print(f"\nFound {len(datasets)} datasets to process")
    successful_saves = 0
    failed_datasets = []
    failed_selections = []

    # Process each dataset with progress bar
    for dataset in tqdm(datasets, desc="Processing datasets", unit="dataset"):
        dataset_id = dataset.get('kod')  # Using 'kod' instead of 'id'
        if not dataset_id:
            print(f"Warning: Could not find kod in dataset: {dataset}")
            continue

        # 2. Get dataset details
        dataset_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}"
        dataset_details = fetch_json(dataset_url)
        
        if not dataset_details:
            failed_datasets.append(dataset_id)
            continue

        # 3. Get available selections for this dataset
        selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
        selections = fetch_json(selections_url)
        
        if not selections:
            failed_datasets.append(dataset_id)
            continue

        # Let's examine the structure of the first selection
        if selections:
            print(f"\nExample selection structure for dataset {dataset_id}:")
            print(json.dumps(selections[0], indent=2, ensure_ascii=False))

        print(f"\nProcessing {len(selections)} selections for dataset {dataset_id}")

        # 4. Process each selection with nested progress bar
        for selection in tqdm(selections, desc=f"Processing selections for {dataset_id}", 
                            leave=False, unit="selection"):
            selection_id = selection.get('kod')  # Using 'kod' instead of 'id'
            if not selection_id:
                print(f"Warning: Could not find kod in selection: {selection}")
                continue
            
            # Fetch the actual data
            data_url = f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}"
            data = fetch_json(data_url)
            
            if not data:
                failed_selections.append(selection_id)
                continue

            try:
                # Convert JSON-stat to pandas DataFrame
                df = pyjstat.from_json_stat(data)[0]
                
                if df.empty:
                    print(f"Warning: Empty DataFrame for {selection_id}")
                    failed_selections.append(selection_id)
                    continue
                
                # Generate filename
                filename = f"{selection_id}.csv"
                
                # Save to CSV
                if save_to_csv(df, filename):
                    successful_saves += 1
                else:
                    failed_selections.append(selection_id)
                
            except Exception as e:
                print(f"Error processing {selection_id}: {e}")
                failed_selections.append(selection_id)

    print(f"\nProcessing complete:")
    print(f"Successfully saved {successful_saves} files to {csv_dir}")
    if failed_datasets:
        print(f"\nFailed to process {len(failed_datasets)} datasets:")
        print(json.dumps(failed_datasets, indent=2))
    if failed_selections:
        print(f"\nFailed to process {len(failed_selections)} selections:")
        print(json.dumps(failed_selections, indent=2))

if __name__ == "__main__":
    main() 
import json
import requests
import csv
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
PROCESS_ALL_DATASETS = 0  # Set to 1 to process all datasets, 0 to process specific dataset
SPECIFIC_DATASET_ID = "CRU01ROBCE"  # Only used when PROCESS_ALL_DATASETS is 0
PROCESS_ALL_SELECTIONS = 0  # Set to 1 to process all selections, 0 to process specific selection
SPECIFIC_SELECTION_ID = "CRU01ROBCET1"  # Only used when PROCESS_ALL_SELECTIONS is 0

def fetch_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(0.1)
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_dimension_values(dimension_data):
    """Extract all possible values for a dimension."""
    if not dimension_data or 'category' not in dimension_data:
        return []
    
    # Get all labels, excluding any None values
    labels = dimension_data['category'].get('label', {}).values()
    return [label for label in labels if label]

def build_description(selection, detailed_metadata):
    parts = []
    # Selection name (always from selection dict, not from detailed_metadata)
    if isinstance(selection, dict) and 'nazev' in selection and selection['nazev']:
        parts.append(selection['nazev'])
    # Time period type
    time_levels = []
    if isinstance(selection, dict) and 'urovneTypObdobi' in selection:
        for lvl in selection['urovneTypObdobi']:
            if isinstance(lvl, dict) and 'nazevUrovne' in lvl and lvl['nazevUrovne']:
                time_levels.append(lvl['nazevUrovne'])
    if time_levels:
        parts.append(f"Time granularity: {', '.join(time_levels)}")
        
    # Territory type
    territory_levels = []
    if isinstance(selection, dict) and 'urovneTypUzemi' in selection:
        for lvl in selection['urovneTypUzemi']:
            if isinstance(lvl, dict) and 'nazevUrovne' in lvl and lvl['nazevUrovne']:
                territory_levels.append(lvl['nazevUrovne'])
    if territory_levels:
        parts.append(f"Territory types: {', '.join(territory_levels)}")
        
    # Notes from detailed metadata
    if detailed_metadata and isinstance(detailed_metadata, dict):
        notes = detailed_metadata.get('note', [])
        valid_notes = [note for note in notes if note]
        if valid_notes:
            parts.append(f"Notes: {' '.join(valid_notes)}")
            
        # Dimensions and their possible values
        dimensions = detailed_metadata.get('dimension', {})
        if isinstance(dimensions, dict):
            for dim_name, dim_data in dimensions.items():
                dim_label = dim_data.get('label', dim_name) if isinstance(dim_data, dict) else dim_name
                values = extract_dimension_values(dim_data) if isinstance(dim_data, dict) else []
                if values:
                    parts.append(f"{dim_label} options: {', '.join(values)}")
    return '. '.join(parts)

def main():
    output_file = Path('metadata/selection_descriptions.csv')
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(['selection_code', 'description'])
        
        # 1. Fetch all datasets
        datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
        datasets = fetch_json(datasets_url)
        if not datasets:
            print("No datasets found.")
            return
        # Filter datasets if needed
        datasets_to_process = datasets if PROCESS_ALL_DATASETS else [d for d in datasets if d.get('kod') == SPECIFIC_DATASET_ID]
        for dataset in tqdm(datasets_to_process, desc="Datasets", unit="dataset"):
            dataset_id = dataset.get('kod')
            if not dataset_id:
                continue
            # 2. Fetch selections for this dataset
            selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
            selections = fetch_json(selections_url)
            if not selections:
                continue
            
            # Filter selections if needed
            selections_to_process = selections if PROCESS_ALL_SELECTIONS else [s for s in selections if s.get('kod') == SPECIFIC_SELECTION_ID]
            for selection in selections_to_process:
                selection_id = selection.get('kod')
                if not selection_id:
                    continue
                print(f"Processing selection {selection_id}...")
                # 3. Fetch detailed metadata for the selection
                detailed_metadata = fetch_json(f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}")
                
                # 4. Build description
                description = build_description(selection, detailed_metadata)
                
                # 5. Write to CSV
                writer.writerow([selection_id, description])
    print(f"\nDescriptions have been saved to {output_file}")

if __name__ == "__main__":
    main()

import json
import requests
import time
from pathlib import Path
from tqdm import tqdm

def fetch_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(0.1)  # Rate limiting
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_metadata(data):
    """Extract metadata schema from the data, excluding 'value' and 'status' arrays."""
    return {
        "version": data["version"],
        "class": data["class"],
        "href": data["href"],
        "label": data["label"],
        "source": data["source"],
        "note": data["note"],
        "updated": data["updated"],
        "id": data["id"],
        "size": data["size"],
        "role": data["role"],
        "dimension": data["dimension"]
    }

def main():
    # Create schemas directory if it doesn't exist
    schemas_dir = Path('metadata/schemas')
    schemas_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch all datasets
    datasets_url = "https://data.csu.gov.cz/api/katalog/v1/sady"
    datasets = fetch_json(datasets_url)
    if not datasets:
        print("No datasets found.")
        return

    # Process each dataset
    for dataset in tqdm(datasets, desc="Processing datasets", unit="dataset"):
        dataset_id = dataset.get('kod')
        if not dataset_id:
            continue

        # 2. Fetch selections for this dataset
        selections_url = f"https://data.csu.gov.cz/api/katalog/v1/sady/{dataset_id}/vybery"
        selections = fetch_json(selections_url)
        if not selections:
            continue

        # Process each selection
        for selection in selections:
            selection_id = selection.get('kod')
            if not selection_id:
                continue

            # 3. Fetch detailed metadata for the selection
            metadata_url = f"https://data.csu.gov.cz/api/dotaz/v1/data/vybery/{selection_id}"
            detailed_metadata = fetch_json(metadata_url)
            
            if detailed_metadata:
                # 4. Extract metadata schema
                metadata_schema = extract_metadata(detailed_metadata)
                
                # 5. Save to file
                output_file = schemas_dir / f"{selection_id}_schema.json"
                with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(metadata_schema, f, ensure_ascii=False, indent=4)
                
                print(f"Saved schema for {selection_id}")

if __name__ == "__main__":
    main()
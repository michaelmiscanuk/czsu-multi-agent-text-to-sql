import json

def extract_metadata(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a copy of the data without the 'value' and 'status' arrays
    metadata = {
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
    
    # Write the metadata to a new JSON file
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

# Example usage
input_file = "metadata/4_https__data.csu.gov.cz_api_dotaz_v1_data_vybery_CRU01ROBCET1.json"
output_file = "metadata/CRU01ROBCET1_metadata_.json"

try:
    extract_metadata(input_file, output_file)
    print(f"Metadata has been successfully extracted to {output_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
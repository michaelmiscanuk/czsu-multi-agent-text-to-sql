import json
import os

def get_json_structure(data, placeholder="."):
    if isinstance(data, dict):
        return {k: get_json_structure(v, placeholder) for k, v in data.items()}
    elif isinstance(data, list):
        return [get_json_structure(item, placeholder) for item in data]
    else:
        return placeholder  # Replace values with a placeholder

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "1.json")  # Absolute path
output_file = os.path.join(script_dir, "1_structure.json")

# Read JSON from file
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        original_json = json.load(f)
except FileNotFoundError:
    print(f"Error: '{input_file}' not found in the script's directory.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: '{input_file}' contains invalid JSON.")
    exit(1)

# Extract structure
structure = get_json_structure(original_json)

# Save structure to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(structure, f, indent=2)

print(f"JSON structure extracted and saved to '{output_file}'.")
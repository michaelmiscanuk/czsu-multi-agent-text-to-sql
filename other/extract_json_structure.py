"""Extract JSON structure by replacing values with placeholders.

This utility script reads a JSON file and creates a structure-only version
where all values are replaced with a placeholder.
"""

import json
import os


def get_json_structure(data, placeholder="."):
    """Extract the structure of JSON data by replacing values with a placeholder.

    Args:
        data: JSON data (dict, list, or primitive value)
        placeholder: String to replace values with (default: ".")

    Returns:
        JSON structure with same shape but placeholder values
    """
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
    with open(input_file, "r", encoding="utf-8") as file_handle:
        original_json = json.load(file_handle)
except FileNotFoundError:
    print(f"Error: '{input_file}' not found in the script's directory.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: '{input_file}' contains invalid JSON.")
    exit(1)

# Extract structure
structure = get_json_structure(original_json)

# Save structure to a new file
with open(output_file, "w", encoding="utf-8") as file_handle:
    json.dump(structure, file_handle, indent=2)

print(f"JSON structure extracted and saved to '{output_file}'.")

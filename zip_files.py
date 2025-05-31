import os
import zipfile
from pathlib import Path
import sys

# Get the base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Configuration of paths to zip
PATHS_TO_ZIP = [
    r"metadata\czsu_chromadb",
    r"data\czsu_data.db",
    r"data\CSVs",
    r"metadata\schemas"
    
]

def zip_path(path_to_zip: str):
    """Zip a file or folder at the specified path."""
    abs_path = BASE_DIR / path_to_zip
    
    if not abs_path.exists():
        print(f"Warning: Path does not exist: {path_to_zip}")
        return
    
    # Create zip file path (same location as original)
    zip_path = abs_path.with_suffix('.zip')
    
    print(f"Zipping: {path_to_zip}")
    print(f"Output: {zip_path}")
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if abs_path.is_file():
            # If it's a file, just add it
            zipf.write(abs_path, abs_path.name)
        else:
            # If it's a directory, add all files recursively
            for root, _, files in os.walk(abs_path):
                for file in files:
                    file_path = Path(root) / file
                    # Calculate relative path for the file in the zip
                    rel_path = file_path.relative_to(abs_path.parent)
                    zipf.write(file_path, rel_path)
    
    print(f"Successfully zipped: {path_to_zip}")

def main():
    print(f"Base directory: {BASE_DIR}")
    print("Starting zip process...")
    
    for path in PATHS_TO_ZIP:
        zip_path(path)
    
    print("\nZip process completed!")

if __name__ == "__main__":
    main() 
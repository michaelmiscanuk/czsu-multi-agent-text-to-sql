import os
import shutil
from pathlib import Path
import sys

# Get the base directory (same as in the other script)
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Configuration of paths to unzip (relative to BASE_DIR)
PATHS_TO_UNZIP = [
    r"metadata\czsu_chromadb.zip",  # Use r prefix for raw string
    r"data\czsu_data.zip"
    # Add more paths here as needed
]

def confirm_overwrite(path: Path) -> bool:
    """Ask user for confirmation before overwriting."""
    while True:
        response = input(f"Path already exists: {path}\nDo you want to overwrite? (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")

def unzip_path(path_to_unzip: str):
    """Unzip a file at the specified path."""
    # Convert to absolute path
    abs_path = BASE_DIR / path_to_unzip
    
    if not abs_path.exists():
        print(f"Warning: Zip file does not exist: {path_to_unzip}")
        return
    
    if not abs_path.suffix == '.zip':
        print(f"Warning: Not a zip file: {path_to_unzip}")
        return
    
    # Get the target path (same as zip file but without .zip extension)
    target_path = abs_path.with_suffix('')
    
    print(f"Unzipping: {path_to_unzip}")
    print(f"Output: {target_path}")
    
    # Check if target exists and ask for confirmation
    if target_path.exists():
        if not confirm_overwrite(target_path):
            print(f"Skipping: {path_to_unzip}")
            return
        # Remove existing path if confirmed
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    
    # Extract the archive
    shutil.unpack_archive(abs_path, target_path.parent, 'zip')
    
    # If it's a single file, rename it to match the original name
    if not target_path.is_dir() and target_path.parent.exists():
        extracted_files = list(target_path.parent.glob('*'))
        if len(extracted_files) == 1 and extracted_files[0] != target_path:
            extracted_files[0].rename(target_path)
    
    print(f"Successfully unzipped: {path_to_unzip}")

def main():
    print(f"Base directory: {BASE_DIR}")
    print("Starting unzip process...")
    
    for path in PATHS_TO_UNZIP:
        unzip_path(path)
    
    print("\nUnzip process completed!")

if __name__ == "__main__":
    main() 
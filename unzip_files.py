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
    BASE_DIR / "metadata" / "czsu_chromadb.zip",
    BASE_DIR / "data" / "czsu_data.zip",
    BASE_DIR / "data" / "CSVs.zip",
    BASE_DIR / "metadata" / "schemas.zip"
    # Add more paths here as needed
]

def unzip_path(path_to_unzip: Path):
    """Unzip a file at the specified path."""
    abs_path = path_to_unzip
    if not abs_path.exists():
        print(f"Warning: Zip file does not exist: {abs_path}")
        return
    if not abs_path.suffix == '.zip':
        print(f"Warning: Not a zip file: {abs_path}")
        return
    target_path = abs_path.with_suffix('')
    print(f"Unzipping: {abs_path}")
    print(f"Output: {target_path}")
    # Always overwrite existing target
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    shutil.unpack_archive(abs_path, target_path.parent, 'zip')
    if not target_path.is_dir() and target_path.parent.exists():
        extracted_files = list(target_path.parent.glob('*'))
        if len(extracted_files) == 1 and extracted_files[0] != target_path:
            extracted_files[0].rename(target_path)
    print(f"Successfully unzipped: {abs_path}")

def main():
    print(f"Base directory: {BASE_DIR}")
    print("Starting unzip process...")
    
    for path in PATHS_TO_UNZIP:
        unzip_path(path)
    
    print("\nUnzip process completed!")

if __name__ == "__main__":
    main() 
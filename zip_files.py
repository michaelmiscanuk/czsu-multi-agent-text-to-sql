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
    BASE_DIR / "metadata" / "czsu_chromadb",
    BASE_DIR / "data" / "czsu_data.db",
    BASE_DIR / "data" / "CSVs",
    BASE_DIR / "metadata" / "schemas",
    BASE_DIR / "data" / "pdf_chromadb_llamaparse"
]

def zip_path(path_to_zip: Path):
    """Zip a file or folder at the specified path with better compression."""
    abs_path = path_to_zip
    if not abs_path.exists():
        print(f"Warning: Path does not exist: {abs_path}")
        return
    # Create zip file path (same location as original)
    zip_path = abs_path.with_suffix('.zip')
    print(f"Zipping: {abs_path}")
    print(f"Output: {zip_path}")
    print("Using LZMA compression for better compression ratio...")
    
    # Create zip file with better compression
    # ZIP_LZMA provides the best compression ratio (but is slower)
    # compresslevel=9 gives maximum compression for DEFLATE
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_LZMA, compresslevel=9) as zipf:
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
    except (OSError, RuntimeError) as e:
        # Fallback to DEFLATE with maximum compression if LZMA fails
        print(f"LZMA compression failed ({e}), falling back to DEFLATE with max compression...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            if abs_path.is_file():
                zipf.write(abs_path, abs_path.name)
            else:
                for root, _, files in os.walk(abs_path):
                    for file in files:
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(abs_path.parent)
                        zipf.write(file_path, rel_path)
    
    print(f"Successfully zipped: {abs_path}")
    # Show file size info
    original_size = get_size(abs_path)
    compressed_size = zip_path.stat().st_size
    if original_size > 0:
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        print(f"Original size: {format_size(original_size)}")
        print(f"Compressed size: {format_size(compressed_size)}")
        print(f"Compression ratio: {compression_ratio:.1f}%")

def get_size(path: Path):
    """Get total size of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    else:
        total_size = 0
        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        return total_size

def format_size(size_bytes):
    """Format size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def main():
    print(f"Base directory: {BASE_DIR}")
    print("Starting zip process with improved compression...")
    
    for path in PATHS_TO_ZIP:
        zip_path(path)
    
    print("\nZip process completed!")

if __name__ == "__main__":
    main() 
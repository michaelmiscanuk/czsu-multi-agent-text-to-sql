import os
import shutil
from pathlib import Path
import sys
import zipfile

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
    BASE_DIR / "metadata" / "schemas.zip",
    BASE_DIR / "data" / "pdf_chromadb_llamaparse.zip"
    # Add more paths here as needed
]

def get_compression_info(zip_path: Path):
    """Get compression information from a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            total_compressed = 0
            total_uncompressed = 0
            compression_methods = set()
            
            for info in zipf.infolist():
                total_compressed += info.compress_size
                total_uncompressed += info.file_size
                if info.compress_type == zipfile.ZIP_STORED:
                    compression_methods.add("STORED")
                elif info.compress_type == zipfile.ZIP_DEFLATED:
                    compression_methods.add("DEFLATED")
                elif info.compress_type == zipfile.ZIP_BZIP2:
                    compression_methods.add("BZIP2")
                elif info.compress_type == zipfile.ZIP_LZMA:
                    compression_methods.add("LZMA")
                else:
                    compression_methods.add("UNKNOWN")
            
            compression_ratio = 0
            if total_uncompressed > 0:
                compression_ratio = ((total_uncompressed - total_compressed) / total_uncompressed) * 100
            
            return {
                'methods': list(compression_methods), 
                'ratio': compression_ratio,
                'compressed_size': total_compressed,
                'uncompressed_size': total_uncompressed
            }
    except Exception:
        return None

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

def unzip_path(path_to_unzip: Path):
    """Unzip a file at the specified path with improved handling."""
    abs_path = path_to_unzip
    if not abs_path.exists():
        print(f"Warning: Zip file does not exist: {abs_path}")
        return
    if not abs_path.suffix == '.zip':
        print(f"Warning: Not a zip file: {abs_path}")
        return
    
    # Get compression info
    comp_info = get_compression_info(abs_path)
    if comp_info:
        print(f"Compression methods: {', '.join(comp_info['methods'])}")
        print(f"Compressed size: {format_size(comp_info['compressed_size'])}")
        print(f"Uncompressed size: {format_size(comp_info['uncompressed_size'])}")
        print(f"Compression ratio: {comp_info['ratio']:.1f}%")
    
    target_path = abs_path.with_suffix('')
    print(f"Unzipping: {abs_path}")
    print(f"Output: {target_path}")
    
    # Always overwrite existing target
    if target_path.exists():
        print(f"Removing existing target: {target_path}")
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    
    try:
        # Use zipfile module directly for better control and error handling
        with zipfile.ZipFile(abs_path, 'r') as zipf:
            zipf.extractall(target_path.parent)
        
        # Handle potential naming mismatches
        if not target_path.exists():
            # Look for extracted content that might have a different name
            extracted_items = [item for item in target_path.parent.iterdir() 
                             if item.name != abs_path.name and 
                             item.stat().st_mtime > abs_path.stat().st_mtime - 1]
            
            if len(extracted_items) == 1 and extracted_items[0] != target_path:
                print(f"Renaming extracted content: {extracted_items[0]} -> {target_path}")
                extracted_items[0].rename(target_path)
                
    except zipfile.BadZipFile:
        print(f"Error: {abs_path} is not a valid zip file or is corrupted")
        return
    except Exception as e:
        print(f"Error extracting {abs_path}: {e}")
        # Fallback to shutil if zipfile fails
        try:
            print("Attempting fallback extraction with shutil...")
            shutil.unpack_archive(abs_path, target_path.parent, 'zip')
        except Exception as fallback_error:
            print(f"Fallback extraction also failed: {fallback_error}")
            return
    
    print(f"Successfully unzipped: {abs_path}")

def main():
    print(f"Base directory: {BASE_DIR}")
    print("Starting unzip process with improved handling...")
    
    for path in PATHS_TO_UNZIP:
        print(f"\n{'='*60}")
        unzip_path(path)
    
    print(f"\n{'='*60}")
    print("Unzip process completed!")

if __name__ == "__main__":
    main() 
import os
from pathlib import Path
import sys
import chromadb
from collections import defaultdict
from typing import Dict, List, Tuple
import hashlib
from datetime import datetime

# Ensure project root is in sys.path for local imports
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Constants
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
COLLECTION_NAME = "czsu_selections_chromadb"

def get_document_hash(text: str) -> str:
    """Generate MD5 hash for a document text.
    
    Args:
        text (str): The document text to hash
        
    Returns:
        str: MD5 hash of the document text
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def find_duplicates() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Find duplicate documents in the ChromaDB collection based on content hash.
    
    Returns:
        Dict[str, List[Tuple[str, str, str]]]: Dictionary mapping content hash to list of (id, selection_code, content) tuples
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        # Get the collection
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Get all documents with their metadata and content
        results = collection.get(
            include=["metadatas", "documents"],
            limit=10000  # Adjust this if you have more documents
        )
        
        # Dictionary to store hash -> (id, selection_code, content) mappings
        hash_map = defaultdict(list)
        
        # Process results
        if results and "metadatas" in results and results["metadatas"]:
            for idx, (metadata, document) in enumerate(zip(results["metadatas"], results["documents"])):
                if isinstance(metadata, dict) and metadata is not None and document:
                    selection_code = metadata.get('selection')
                    if selection_code:
                        # Generate hash from the actual document content
                        content_hash = get_document_hash(document)
                        hash_map[content_hash].append((results["ids"][idx], selection_code, document))
        
        # Filter only duplicates (entries with more than one document)
        duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}
        
        return duplicates
        
    except Exception as e:
        print(f"Error accessing ChromaDB: {str(e)}")
        return {}

def deduplicate_collection(duplicates: Dict[str, List[Tuple[str, str, str]]]) -> None:
    """
    Remove duplicate documents from the collection, keeping only the latest record for each duplicate group.
    
    Args:
        duplicates (Dict[str, List[Tuple[str, str, str]]]): Dictionary of duplicate documents
    """
    if not duplicates:
        print("No duplicates to remove.")
        return

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Get all documents with their metadata to find timestamps
    results = collection.get(
        include=["metadatas"],
        limit=10000
    )
    
    # Create a mapping of document IDs to their timestamps
    id_to_timestamp = {}
    if results and "metadatas" in results and results["metadatas"]:
        for idx, metadata in enumerate(results["metadatas"]):
            if isinstance(metadata, dict) and metadata is not None:
                # Try to get timestamp from metadata, default to current time if not found
                timestamp_str = metadata.get('timestamp', datetime.now().isoformat())
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp = datetime.now()
                id_to_timestamp[results["ids"][idx]] = timestamp
    
    # For each group of duplicates, keep only the latest one
    ids_to_delete = []
    for content_hash, entries in duplicates.items():
        # Sort entries by timestamp (latest first)
        sorted_entries = sorted(
            entries,
            key=lambda x: id_to_timestamp.get(x[0], datetime.min),
            reverse=True
        )
        
        # Keep the first (latest) entry, mark others for deletion
        ids_to_delete.extend(entry[0] for entry in sorted_entries[1:])
    
    if ids_to_delete:
        print(f"\nRemoving {len(ids_to_delete)} duplicate documents...")
        try:
            collection.delete(ids=ids_to_delete)
            print("Successfully removed duplicate documents.")
        except Exception as e:
            print(f"Error removing duplicates: {str(e)}")
    else:
        print("No documents to remove.")

def main():
    print("Checking for duplicates in ChromaDB collection based on content hash...")
    duplicates = find_duplicates()
    
    if not duplicates:
        print("No duplicates found in the collection.")
        return
    
    print(f"\nFound {len(duplicates)} duplicate document groups:")
    print("=" * 80)
    
    for content_hash, entries in duplicates.items():
        print(f"\nContent Hash: {content_hash}")
        print(f"Number of duplicates: {len(entries)}")
        print("Documents:")
        for doc_id, selection_code, content in entries:
            print(f"  - ID: {doc_id}")
            print(f"    Selection Code: {selection_code}")
            print(f"    Content Preview: {content[:100]}...")  # Show first 100 chars of content
        print("-" * 80)
    
    # Automatically deduplicate
    print("\nProceeding with deduplication...")
    deduplicate_collection(duplicates)

if __name__ == "__main__":
    main() 
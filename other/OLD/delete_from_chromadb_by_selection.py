import os
import sys
from pathlib import Path
import chromadb

# --- Path setup (same as your code) ---
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
COLLECTION_NAME = "czsu_selections_chromadb"
DEFAULT_SELECTION = "PRODCOM2"


def main():
    if len(sys.argv) < 2:
        print(f"No selection code provided. Defaulting to: {DEFAULT_SELECTION}")
        target_selection = DEFAULT_SELECTION
    else:
        target_selection = sys.argv[1]

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Fetch all metadatas and ids
    print("Fetching all documents from ChromaDB collection...")
    results = collection.get(include=["metadatas"], limit=100_000)
    ids_to_delete = []
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        if isinstance(meta, dict) and meta.get("selection") == target_selection:
            ids_to_delete.append(doc_id)

    print(f"Found {len(ids_to_delete)} documents with selection_code = '{target_selection}'.")

    if ids_to_delete:
        # Delete in batches if needed (ChromaDB may have a limit per call)
        BATCH_SIZE = 1000
        for i in range(0, len(ids_to_delete), BATCH_SIZE):
            batch = ids_to_delete[i:i+BATCH_SIZE]
            collection.delete(ids=batch)
            print(f"Deleted batch {i//BATCH_SIZE + 1}: {len(batch)} documents.")
        print("Deletion complete.")
    else:
        print("No documents found with the specified selection_code.")

if __name__ == "__main__":
    main() 
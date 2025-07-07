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

    print(f"Checking for documents with selection_code = '{target_selection}'...")
    results = collection.get(include=["metadatas"], limit=100_000)
    found = 0
    for meta in results["metadatas"]:
        if isinstance(meta, dict) and meta.get("selection") == target_selection:
            found += 1
    if found:
        print(f"Found {found} documents with selection_code = '{target_selection}'.")
    else:
        print(f"No documents with selection_code = '{target_selection}' found in the collection.")

if __name__ == "__main__":
    main() 
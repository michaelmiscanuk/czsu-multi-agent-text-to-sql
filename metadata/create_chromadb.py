#==============================================================================
# IMPORTS
#==============================================================================
import chromadb
from uuid import uuid4
import os
from pathlib import Path
import sys

# --- Ensure project root is in sys.path for local imports ---
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
from my_agent.utils.models import get_azure_embedding_model

CREATE_CHROMADB_ID = 30

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def debug_print(msg: str) -> None:
    if os.environ.get('MY_AGENT_DEBUG', '0') == '1':
        print(f"{CREATE_CHROMADB_ID}: {msg}")

#==============================================================================
# MAIN LOGIC
#==============================================================================
def upsert_documents_to_chromadb(document_tuples, deployment="text-embedding-3-large__test1", collection_name="aaaaaaaaaab"):
    """
    Add new documents to a ChromaDB collection using UUIDs as IDs. Checks for existence by document text.
    Stores 'selection' as metadata. Only documents not already present (by text) are added.

    Args:
        document_tuples (list[tuple]): List of (text, selection) tuples
        deployment (str): Azure embedding deployment name
        collection_name (str): ChromaDB collection name
    Returns:
        collection: The ChromaDB collection object
    """
    embedding_client = get_azure_embedding_model()

    # Use persistent ChromaDB storage
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=collection_name)

    # Get all existing documents in the collection (fetch all, not just first page)
    existing = collection.get(include=["documents"], limit=10000)
    existing_docs = set()
    if existing and "documents" in existing and existing["documents"]:
        for doc_list in existing["documents"]:
            if isinstance(doc_list, str):
                existing_docs.add(doc_list)
            else:
                existing_docs.update(doc_list)
    debug_print(f"Existing docs in collection: {existing_docs}")

    # Only add documents whose text is not already present
    new_tuples = [t for t in document_tuples if t[0] not in existing_docs]
    new_texts = [t[0] for t in new_tuples]
    debug_print(f"New texts to add: {new_texts}")
    new_selections = [t[1] for t in new_tuples]
    new_ids = [str(uuid4()) for _ in new_texts]
    new_metadatas = [{"selection": sel} for sel in new_selections]
    if not new_texts:
        print("No new documents to add.")
        return collection

    # Generate embeddings for new documents
    response = embedding_client.embeddings.create(
        input=new_texts,
        model=deployment
    )
    embeddings = [item.embedding for item in response.data]

    # Add new documents with UUIDs as IDs and selection as metadata
    collection.add(
        documents=new_texts,
        embeddings=embeddings,
        ids=new_ids,
        metadatas=new_metadatas
    )
    print(f"Added {len(new_texts)} new documents.")
    return collection

#==============================================================================
# SCRIPT ENTRY POINT (EXAMPLE USAGE)
#==============================================================================
if __name__ == "__main__":
    # Example usage: tuples of (text, selection)
    DOCUMENTS = [
        ("The quick brown fox jumps over the lazy dog.", "A"),
        ("Artificial intelligence is transforming the world.", "B"),
        ("Python is a popular programming language.", "C"),
        ("ChromaDB enables vector search for embeddings.", "D"),
        ("OpenAI provides powerful language models.", "E")
    ]
    collection = upsert_documents_to_chromadb(DOCUMENTS)

    # --- Usage Example: Similarity Search ---
    embedding_client = get_azure_embedding_model()
    QUERY = "ChromaDB enables vector search for embeddings.."
    query_embedding = embedding_client.embeddings.create(
        input=[QUERY],
        model="text-embedding-3-large__test1"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\nQuery:", QUERY)
    print("Most similar documents (with selection):")
    for doc, meta, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        selection = meta.get('selection') if isinstance(meta, dict) and meta is not None else 'N/A'
        print(f"  - {doc} (selection: {selection}, distance: {score:.4f})") 
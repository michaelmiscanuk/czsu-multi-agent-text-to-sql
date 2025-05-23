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
CHROMA_DB_PATH = BASE_DIR / "data" / "czsu_chromadb"

from my_agent.utils.models import get_azure_embedding_model

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

    # Get all existing documents in the collection
    existing = collection.get(include=["documents"])
    existing_docs = set(existing["documents"][0]) if existing and "documents" in existing and existing["documents"] else set()

    # Prepare new documents to add (skip if already present)
    new_tuples = [t for t in document_tuples if t[0] not in existing_docs]
    new_texts = [t[0] for t in new_tuples]
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
    QUERY = "What programming languages are popular?"
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
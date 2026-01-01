"""Evaluation script for semantic search retrieval.

This script evaluates the simple semantic search functionality for table retrieval.
It tests ChromaDB's semantic similarity search using embeddings only (no BM25).

Evaluators:
- top_1_correct: Checks if the correct table is ranked first
- table_in_top_n: Checks if the correct table is in top N results (configurable)
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import aevaluate

# Setup paths and environment
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from metadata.chromadb_client_factory import get_chromadb_client, should_use_cloud
from metadata.create_and_load_chromadb__04 import (
    similarity_search_chromadb,
    get_azure_embedding_model,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
# ChromaDB configuration (matching nodes.py)
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

# Evaluation configuration
TOP_N = 3  # Number of top results to check
DATASET_NAME = "langsmith_create_golden_dataset_table_retrieval"
EXPERIMENT_PREFIX = "semantic-search"
MAX_CONCURRENCY = 4
N_RESULTS = 60  # Number of results to retrieve


# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================
async def semantic_search_retrieval(inputs: dict) -> dict:
    """Perform semantic search using the existing similarity_search_chromadb function.

    This uses the same function that's used throughout the codebase for consistency.

    Args:
        inputs: Dict containing 'question' key

    Returns:
        Dict with 'retrieved_tables' containing list of (table_code, score) tuples
    """
    question = inputs["question"]

    try:
        # Get ChromaDB client and collection (matching nodes.py pattern)
        client = get_chromadb_client(
            local_path=CHROMA_DB_PATH, collection_name=CHROMA_COLLECTION_NAME
        )
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        # Get embedding model
        embedding_client = get_azure_embedding_model()

        # Perform semantic search using existing function
        results = similarity_search_chromadb(
            collection=collection,
            embedding_client=embedding_client,
            query=question,
            embedding_model_name=EMBEDDING_DEPLOYMENT,
            k=N_RESULTS,
        )

        # Extract table codes and convert distances to similarity scores
        retrieved_tables = []
        if results and results["metadatas"] and results["distances"]:
            for metadata, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                table_code = metadata.get("selection", "").upper()
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = max(0, 1 - (distance / 2))
                if table_code:
                    retrieved_tables.append((table_code, similarity_score))

        print(f"[Semantic Search] Question: {question[:50]}...")
        print(f"[Semantic Search] Retrieved {len(retrieved_tables)} tables")
        if retrieved_tables:
            print(f"[Semantic Search] Top 3: {retrieved_tables[:3]}")

        return {"retrieved_tables": retrieved_tables}

    except Exception as e:
        print(f"[Semantic Search] Error: {e}")
        return {"retrieved_tables": []}


# ============================================================================
# EVALUATORS
# ============================================================================
async def top_1_correct(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the correct table is ranked first."""
    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    retrieved_tables = outputs.get("retrieved_tables", [])
    expected_table = reference_outputs.get("answer", "").strip().upper()

    if not retrieved_tables or not expected_table:
        return False

    actual_top = retrieved_tables[0][0].strip().upper()
    result = actual_top == expected_table

    print(
        f"[Evaluator: top_1] Expected: {expected_table}, Got: {actual_top}, Match: {result}"
    )
    return result


async def table_in_top_n(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the correct table is in top N results."""
    if isinstance(outputs, dict) and "outputs" in outputs:
        outputs = outputs["outputs"]

    retrieved_tables = outputs.get("retrieved_tables", [])
    expected_table = reference_outputs.get("answer", "").strip().upper()

    if not expected_table:
        return False

    # Check if expected table is in top N
    top_n_tables = [table.strip().upper() for table, score in retrieved_tables[:TOP_N]]
    result = expected_table in top_n_tables

    print(
        f"[Evaluator: top_{TOP_N}] Expected: {expected_table}, Top {TOP_N}: {top_n_tables}, Match: {result}"
    )
    return result


# ============================================================================
# MAIN EVALUATION
# ============================================================================
async def main():
    print(f"Starting semantic search evaluation...")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Top N: {TOP_N}")
    print(f"N Results: {N_RESULTS}")
    print(f"\n🌐 ChromaDB Configuration:")
    print(f"   CHROMA_USE_CLOUD: {os.getenv('CHROMA_USE_CLOUD', 'not set')}")
    print(f"   Using Cloud: {should_use_cloud()}")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")
    print(f"   Local Path: {CHROMA_DB_PATH}\n")

    experiment_results = await aevaluate(
        semantic_search_retrieval,
        data=DATASET_NAME,
        evaluators=[top_1_correct, table_in_top_n],
        max_concurrency=MAX_CONCURRENCY,
        experiment_prefix=EXPERIMENT_PREFIX,
    )

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_results}")


if __name__ == "__main__":
    asyncio.run(main())

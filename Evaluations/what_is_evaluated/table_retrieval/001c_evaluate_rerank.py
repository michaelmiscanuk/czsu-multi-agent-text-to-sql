"""Evaluation script for rerank retrieval.

This script evaluates the full pipeline: hybrid search + Cohere reranking for table retrieval.
It tests the complete workflow that uses both retrieval and reranking.

Evaluators:
- top_1_correct: Checks if the correct table is ranked first after reranking
- table_in_top_n: Checks if the correct table is in top N results after reranking (configurable)
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
from metadata.create_and_load_chromadb__04 import hybrid_search, cohere_rerank

# ============================================================================
# CONFIGURATION
# ============================================================================
# ChromaDB configuration (matching nodes.py)
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"

# Evaluation configuration
TOP_N = 10  # Number of top results to check
DATASET_NAME = "langsmith_create_golden_dataset_table_retrieval"
EXPERIMENT_PREFIX = "rerank"
MAX_CONCURRENCY = 4
N_RESULTS = 20  # Number of results to retrieve from hybrid search
TOP_K_RERANK = 20  # Number of results to rerank


# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================
async def rerank_retrieval(inputs: dict) -> dict:
    """Perform hybrid search followed by Cohere reranking.

    Args:
        inputs: Dict containing 'question' key

    Returns:
        Dict with 'retrieved_tables' containing list of (table_code, score) tuples
    """
    question = inputs["question"]

    try:
        # Step 1: Get ChromaDB client and collection (matching nodes.py pattern)
        client = get_chromadb_client(
            local_path=CHROMA_DB_PATH, collection_name=CHROMA_COLLECTION_NAME
        )
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

        # Step 2: Perform hybrid search
        hybrid_results = hybrid_search(
            collection, query_text=question, n_results=N_RESULTS
        )

        if not hybrid_results:
            print(f"[Rerank] No hybrid search results for: {question[:50]}...")
            return {"retrieved_tables": []}

        print(f"[Rerank] Hybrid search returned {len(hybrid_results)} results")

        # Step 3: Convert dict results to Document objects (matching nodes.py pattern)
        from langchain_core.documents import Document

        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(page_content=result["document"], metadata=result["metadata"])
            hybrid_docs.append(doc)

        if not hybrid_docs:
            print(f"[Rerank] No valid documents to rerank")
            return {"retrieved_tables": []}

        # Step 4: Perform Cohere reranking (matching nodes.py call signature)
        reranked_results = cohere_rerank(
            query=question, docs=hybrid_docs, top_n=TOP_K_RERANK
        )

        # Step 5: Extract table codes and relevance scores
        # reranked_results is a list of (Document, CohereResult) tuples
        retrieved_tables = []
        for doc, cohere_result in reranked_results:
            table_code = doc.metadata.get("selection", "").upper()
            relevance_score = cohere_result.relevance_score
            if table_code:
                retrieved_tables.append((table_code, relevance_score))

        print(f"[Rerank] Question: {question[:50]}...")
        print(f"[Rerank] Reranked {len(retrieved_tables)} tables")
        if retrieved_tables:
            print(f"[Rerank] Top 3: {retrieved_tables[:3]}")

        return {"retrieved_tables": retrieved_tables}

    except Exception as e:
        print(f"[Rerank] Error: {e}")
        import traceback

        traceback.print_exc()
        return {"retrieved_tables": []}


# ============================================================================
# EVALUATORS
# ============================================================================
async def top_1_correct(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the correct table is ranked first after reranking."""
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
    """Check if the correct table is in top N results after reranking."""
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
    print(f"Starting rerank evaluation...")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Top N: {TOP_N}")
    print(f"N Results (hybrid): {N_RESULTS}")
    print(f"Top K (rerank): {TOP_K_RERANK}")
    print(f"\n🌐 ChromaDB Configuration:")
    print(f"   CHROMA_USE_CLOUD: {os.getenv('CHROMA_USE_CLOUD', 'not set')}")
    print(f"   Using Cloud: {should_use_cloud()}")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")
    print(f"   Local Path: {CHROMA_DB_PATH}\n")

    experiment_results = await aevaluate(
        rerank_retrieval,
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

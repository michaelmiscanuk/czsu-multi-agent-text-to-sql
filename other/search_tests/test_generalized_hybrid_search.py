#!/usr/bin/env python3
"""
Test script to verify the improved hybrid search approach works with dynamic Czech term handling.
This tests multiple search optimization strategies for Czech language.
"""

import sys
import os
from pathlib import Path

# Setup paths
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

sys.path.append(str(BASE_DIR / 'metadata'))

import chromadb
from create_and_load_chromadb import (
    hybrid_search, get_langchain_chroma_vectorstore, cohere_rerank
)

# Constants 
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

def test_improved_hybrid_search():
    """Test the improved hybrid search approach with multiple strategies."""
    
    print("=== Testing Improved Hybrid Search ===\n")
    
    # Use the correct ChromaDB path
    try:
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        print(f"✓ Connected to ChromaDB vectorstore at {CHROMA_DB_PATH}")
        
        # Get the underlying collection for hybrid search
        collection = chroma_vectorstore._collection  
        print(f"✓ Obtained underlying ChromaDB collection")
        
    except Exception as e:
        print(f"✗ Failed to connect to ChromaDB: {e}")
        return
    
    # Test queries with different characteristics
    test_queries = [
        "jaky je pocet obytnych domu vlastneny bytovymi drzustvy?",  # Original query (no diacritics)
        "Jaký je počet obytných domů vlastněných bytovými družstvy?",  # With diacritics
        "kolik je obyvatel v praze?",  # Different query about Prague population
        "průměrná mzda v české republice",  # Query about average salary
        "nezaměstnanost podle krajů",  # Query about unemployment by regions
        "stavebnictvi a bydleni",  # Construction and housing
        "doprava a automobilovy prumysl",  # Transport and automotive
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        
        # Strategy 1: Direct hybrid search  
        try:
            print("🔍 Direct Hybrid Search:")
            results = hybrid_search(collection, query, n_results=10)
            print(f"  Retrieved {len(results)} results")
            
            # Show top 5 results
            for j, result in enumerate(results[:5]):
                selection = result['metadata'].get('selection', 'unknown')
                score = result.get('score', 0)
                semantic_score = result.get('semantic_score', 0)
                bm25_score = result.get('bm25_score', 0)
                source = result.get('source', 'unknown')
                
                print(f"    {j+1}. {selection}: {score:.6f} "
                      f"(sem: {semantic_score:.3f}, bm25: {bm25_score:.3f}, src: {source})")
            
            # Convert results to Document objects for reranking
            if results:
                print("🎯 + Cohere Rerank:")
                # Convert dict results to Document objects
                from langchain_core.documents import Document
                docs = []
                for result in results:
                    doc = Document(
                        page_content=result['document'],
                        metadata=result['metadata']
                    )
                    docs.append(doc)
                
                reranked = cohere_rerank(query, docs, top_n=5)
                print(f"  Reranked to {len(reranked)} results")
                
                for j, (doc, res) in enumerate(reranked, 1):
                    selection = doc.metadata.get('selection') if doc.metadata else 'unknown'
                    score = res.relevance_score
                    print(f"    {j}. {selection}: Score {score:.6f}")
            
        except Exception as e:
            print(f"✗ Hybrid search failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_improved_hybrid_search() 
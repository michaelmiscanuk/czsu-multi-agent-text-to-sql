#!/usr/bin/env python3
"""
Simple test to verify semantic-focused hybrid search functionality.
Testing with the exact user query about Prague population.
"""

import os
import sys
from pathlib import Path
import asyncio

# Setup paths
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Enable debug mode
os.environ['DEBUG'] = '1'

# Direct imports to avoid circular import
sys.path.insert(0, str(BASE_DIR / "metadata"))

from create_and_load_chromadb import (
    hybrid_search,
    get_langchain_chroma_vectorstore,
    cohere_rerank
)

# Constants
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_data"
K = 20

def main():
    print("🔍 Testing Semantic-Focused Hybrid Search")
    print("=" * 60)
    
    try:
        # Check what collections exist
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collections = client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        if not collections:
            print("❌ No collections found in ChromaDB")
            return
        
        # Check each collection's document count
        for collection in collections:
            count = collection.count()
            print(f"Collection '{collection.name}': {count} documents")
        
        # Use the collection with the most documents
        target_collection = max(collections, key=lambda c: c.count())
        print(f"Using collection '{target_collection.name}' with {target_collection.count()} documents")
        
        # Test queries - focusing on the user's exact query
        test_queries = [
            "Kolik lidí žije v Praze?",  # User's exact query
            "kolik je obyvatel v praze?",  # Variant without diacritics  
            "population of Prague",  # English equivalent
            "jaky je pocet obytnych domu vlastneny bytovymi drzustvy?",  # Housing query
            "nezaměstnanost podle krajů"  # Employment query
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{query}'")
            print("-" * 50)
            
            # Test semantic-focused hybrid search directly with the collection
            hybrid_results = hybrid_search(target_collection, query, n_results=K)
            print(f"Hybrid search returned {len(hybrid_results)} results")
            
            if len(hybrid_results) == 0:
                print("⚠️  No results from hybrid search")
                continue
            
            # Show top 5 hybrid results with detailed scoring
            print(f"\nTop 5 semantic-focused hybrid search results:")
            for j, result in enumerate(hybrid_results[:5], 1):
                selection = result['metadata'].get('selection', 'N/A') if result.get('metadata') else 'N/A'
                score = result.get('score', 0)
                semantic_score = result.get('semantic_score', 0)
                bm25_score = result.get('bm25_score', 0)
                source = result.get('source', 'unknown')
                
                print(f"  #{j}: {selection}")
                print(f"      Final Score: {score:.4f} | Semantic: {semantic_score:.4f} | BM25: {bm25_score:.4f} | Source: {source}")
            
            print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
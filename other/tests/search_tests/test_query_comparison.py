#!/usr/bin/env python3
"""
Test to compare hybrid search results between original and rewritten queries.
"""

import os
import sys
from pathlib import Path

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
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

# The two queries to compare
ORIGINAL_QUERY = "jaky je pocet obytnych domu vlastneny bytovymi drzustvy?"
REWRITTEN_QUERY = "Jaký je počet obytných domů vlastněných bytovými družstvy?"
K = 60  # Increased to match new default

def test_query(query_name, query_text):
    """Test hybrid search and rerank for a specific query"""
    print(f"\n{'='*80}")
    print(f"🔍 TESTING {query_name}")
    print(f"Query: {query_text}")
    print('='*80)
    
    try:
        # Same approach as in script
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        
        # Hybrid search
        hybrid_results = hybrid_search(chroma_vectorstore, query_text, k=K)
        print(f"Hybrid search returned {len(hybrid_results)} results")
        
        # Show all selection codes
        selection_codes = [doc.metadata.get('selection') for doc in hybrid_results]
        print(f"All selection codes: {selection_codes}")
        
        # Check for key selections
        key_selections = ['SLD044T02', 'SLD036T02', 'SLD036T01', 'STAV799AT1', 'SLD025T02A', 'CRU01T1']
        print(f"\nKey selection presence:")
        for sel in key_selections:
            if sel in selection_codes:
                pos = selection_codes.index(sel) + 1
                print(f"  ✅ {sel}: Position #{pos}")
            else:
                print(f"  ❌ {sel}: NOT FOUND")
        
        # Rerank
        reranked = cohere_rerank(query_text, hybrid_results, top_n=15)
        print(f"\nCohere rerank returned {len(reranked)} results")
        
        # Show top 5 reranked results
        print("\nTop 5 reranked results:")
        for i, (doc, res) in enumerate(reranked[:5], 1):
            selection = doc.metadata.get('selection') if doc.metadata else 'N/A'
            score = res.relevance_score
            print(f"  #{i}: {selection} | Score: {score:.6f}")
            
        return reranked
        
    except Exception as e:
        print(f"Error testing {query_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main comparison function"""
    print("🧪 QUERY COMPARISON TEST")
    print("Comparing hybrid search results between original and rewritten queries")
    
    # Test both queries
    original_results = test_query("ORIGINAL QUERY", ORIGINAL_QUERY)
    rewritten_results = test_query("REWRITTEN QUERY", REWRITTEN_QUERY)
    
    # Compare results
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    if original_results and rewritten_results:
        print("\nTop 3 comparison:")
        print("Rank | Original Selection | Original Score | Rewritten Selection | Rewritten Score | Match?")
        print("-" * 90)
        
        for i in range(min(3, len(original_results), len(rewritten_results))):
            orig_sel = original_results[i][0].metadata.get('selection') if i < len(original_results) else 'N/A'
            orig_score = original_results[i][1].relevance_score if i < len(original_results) else 0
            
            rewr_sel = rewritten_results[i][0].metadata.get('selection') if i < len(rewritten_results) else 'N/A'
            rewr_score = rewritten_results[i][1].relevance_score if i < len(rewritten_results) else 0
            
            match = "✅" if orig_sel == rewr_sel else "❌"
            
            print(f"{i+1:4d} | {orig_sel:17s} | {orig_score:13.6f} | {rewr_sel:18s} | {rewr_score:14.6f} | {match}")
    
    else:
        print("❌ One or both queries failed to return results")
        
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 
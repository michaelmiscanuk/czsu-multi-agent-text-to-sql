#!/usr/bin/env python3
"""
Test script to compare rerank results between script approach and node approach.
This will help identify where the discrepancy comes from.
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
os.environ['MY_AGENT_DEBUG'] = '1'

from metadata.create_and_load_chromadb import (
    hybrid_search,
    get_langchain_chroma_vectorstore,
    cohere_rerank
)
from my_agent.utils.nodes import retrieve_similar_selections_hybrid_search_node, rerank_node
from my_agent.utils.state import DataAnalysisState

# Constants
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"
TEST_QUERY = "jaky je pocet obytnych domu vlastneny bytovymi drzustvy?"
K = 15

async def test_script_approach():
    """Test the approach used in the script"""
    print("\n" + "="*80)
    print("🔵 TESTING SCRIPT APPROACH")
    print("="*80)
    
    try:
        # Same approach as in script
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        
        print(f"Query: {TEST_QUERY}")
        print(f"K: {K}")
        
        # Hybrid search
        hybrid_results = hybrid_search(chroma_vectorstore, TEST_QUERY, k=K)
        print(f"Hybrid search returned {len(hybrid_results)} results")
        
        # Show top 5 hybrid results
        print("\nTop 5 hybrid search results:")
        for i, doc in enumerate(hybrid_results[:5], 1):
            selection = doc.metadata.get('selection') if doc.metadata else 'N/A'
            print(f"  #{i}: {selection}")
        
        # Rerank
        reranked = cohere_rerank(TEST_QUERY, hybrid_results, top_n=K)
        print(f"Cohere rerank returned {len(reranked)} results")
        
        # Show top 5 reranked results
        print("\nTop 5 reranked results:")
        for i, (doc, res) in enumerate(reranked[:5], 1):
            selection = doc.metadata.get('selection') if doc.metadata else 'N/A'
            score = res.relevance_score
            print(f"  #{i}: {selection} | Score: {score:.6f}")
            
        return reranked
        
    except Exception as e:
        print(f"Error in script approach: {e}")
        import traceback
        traceback.print_exc()
        return []

async def test_node_approach():
    """Test the approach used in our nodes"""
    print("\n" + "="*80)
    print("🟢 TESTING NODE APPROACH")
    print("="*80)
    
    try:
        # Create initial state
        state = {
            "prompt": TEST_QUERY,
            "rewritten_prompt": TEST_QUERY,
            "messages": [],
            "iteration": 0,
            "queries_and_results": [],
            "reflection_decision": "",
            "hybrid_search_results": [],
            "most_similar_selections": [],
            "top_selection_codes": [],
            "n_results": K
        }
        
        print(f"Query: {TEST_QUERY}")
        print(f"n_results: {K}")
        
        # Step 1: Hybrid search node
        print("\n--- Step 1: Hybrid Search Node ---")
        hybrid_output = await retrieve_similar_selections_hybrid_search_node(state)
        
        # Update state with hybrid results
        state.update(hybrid_output)
        
        # Step 2: Rerank node
        print("\n--- Step 2: Rerank Node ---")
        rerank_output = await rerank_node(state)
        
        print(f"\nFinal rerank output: {rerank_output}")
        
        return rerank_output.get("most_similar_selections", [])
        
    except Exception as e:
        print(f"Error in node approach: {e}")
        import traceback
        traceback.print_exc()
        return []

async def main():
    """Main comparison function"""
    print("🧪 RERANK COMPARISON TEST")
    print("Testing the same query with both approaches to identify discrepancies")
    
    # Test both approaches
    script_results = await test_script_approach()
    node_results = await test_node_approach()
    
    # Compare results
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    
    print(f"Script approach returned {len(script_results)} results")
    print(f"Node approach returned {len(node_results)} results")
    
    if script_results and node_results:
        print("\nTop 5 comparison:")
        print("Rank | Script Selection | Script Score | Node Selection | Node Score | Match?")
        print("-" * 70)
        
        max_len = max(len(script_results), len(node_results))
        for i in range(min(5, max_len)):
            script_sel = script_results[i][0].metadata.get('selection') if i < len(script_results) else 'N/A'
            script_score = script_results[i][1].relevance_score if i < len(script_results) else 0
            
            node_sel = node_results[i][0] if i < len(node_results) else 'N/A'
            node_score = node_results[i][1] if i < len(node_results) else 0
            
            match = "✅" if script_sel == node_sel else "❌"
            
            print(f"{i+1:4d} | {script_sel:15s} | {script_score:11.6f} | {node_sel:14s} | {node_score:10.6f} | {match}")
    
    else:
        print("❌ One or both approaches failed to return results")
        
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main()) 
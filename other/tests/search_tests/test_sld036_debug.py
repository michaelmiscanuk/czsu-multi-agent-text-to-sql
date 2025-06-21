#!/usr/bin/env python3
"""
Debug test specifically for SLD036T01 and SLD036T02 documents.
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
    get_langchain_chroma_vectorstore,
    hybrid_search
)

# Constants
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

REWRITTEN_QUERY = "Jaký je počet obytných domů vlastněných bytovými družstvy?"
TARGET_DOCS = ['SLD036T01', 'SLD036T02']

def debug_sld036():
    """Debug why SLD036T01 and SLD036T02 are missing"""
    print("🔍 DEBUGGING SLD036T01 and SLD036T02")
    print("="*80)
    
    try:
        # Get ChromaDB vectorstore
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        
        # Test with a much larger k to see if they appear at all
        print(f"Testing with k=100 to see if SLD036 documents appear...")
        
        hybrid_results = hybrid_search(chroma_vectorstore, REWRITTEN_QUERY, k=100)
        
        print(f"Got {len(hybrid_results)} results")
        
        # Check if our target documents are in the results
        found_docs = {}
        for i, doc in enumerate(hybrid_results):
            selection = doc.metadata.get('selection') if doc.metadata else None
            if selection in TARGET_DOCS:
                found_docs[selection] = {
                    'position': i + 1,
                    'content_preview': doc.page_content[:200]
                }
        
        print(f"\nTarget documents found:")
        for doc_id in TARGET_DOCS:
            if doc_id in found_docs:
                info = found_docs[doc_id]
                print(f"✅ {doc_id}: Position #{info['position']}")
                print(f"   Content: {info['content_preview']}...")
            else:
                print(f"❌ {doc_id}: NOT FOUND in top 100 results")
        
        # Show all selection codes to see what we're getting
        all_selections = [doc.metadata.get('selection') for doc in hybrid_results]
        print(f"\nAll selection codes in results:")
        print(all_selections)
        
        # Check if they're in the raw ChromaDB
        print(f"\nChecking raw ChromaDB for target documents...")
        raw_docs = chroma_vectorstore.get(include=["documents", "metadatas"])
        
        target_in_db = {}
        for i, (doc, meta) in enumerate(zip(raw_docs["documents"], raw_docs["metadatas"])):
            selection = meta.get('selection') if meta else None
            if selection in TARGET_DOCS:
                target_in_db[selection] = {
                    'index': i,
                    'content': doc[:200]
                }
        
        print(f"Target documents in ChromaDB:")
        for doc_id in TARGET_DOCS:
            if doc_id in target_in_db:
                info = target_in_db[doc_id]
                print(f"✅ {doc_id}: Index #{info['index']}")
                print(f"   Content: {info['content']}...")
            else:
                print(f"❌ {doc_id}: NOT FOUND in ChromaDB")
        
        return found_docs, target_in_db
        
    except Exception as e:
        print(f"Error in SLD036 debug: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}

def main():
    """Main debug function"""
    found_docs, target_in_db = debug_sld036()
    
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    if target_in_db:
        print(f"✅ Target documents ARE in ChromaDB: {list(target_in_db.keys())}")
    else:
        print("❌ Target documents NOT FOUND in ChromaDB")
    
    if found_docs:
        print(f"✅ Target documents found in hybrid search: {list(found_docs.keys())}")
        for doc_id, info in found_docs.items():
            print(f"   {doc_id}: Position #{info['position']}")
    else:
        print("❌ Target documents NOT FOUND in hybrid search results (even with k=100)")
        print("💡 This suggests they're being filtered out by the ensemble retriever")

if __name__ == "__main__":
    main() 
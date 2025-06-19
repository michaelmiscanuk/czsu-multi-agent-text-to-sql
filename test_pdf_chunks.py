#!/usr/bin/env python3
"""Test script to debug PDF chunk functionality"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path('.').resolve()
sys.path.insert(0, str(BASE_DIR))

from data.pdf_to_chromadb import CHROMA_DB_PATH, COLLECTION_NAME, hybrid_search
import chromadb

def test_pdf_chunks():
    print(f'PDF ChromaDB Path: {CHROMA_DB_PATH}')
    print(f'Collection Name: {COLLECTION_NAME}')
    print(f'ChromaDB exists: {CHROMA_DB_PATH.exists()}')
    
    if not CHROMA_DB_PATH.exists():
        print('❌ ChromaDB directory not found!')
        return False
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f'✅ Collection document count: {count}')
        
        # Test search with the current default prompt
        test_query = 'výroba kapalných paliv z ropy'
        print(f'🔍 Testing search for: "{test_query}"')
        results = hybrid_search(collection, test_query, n_results=3)
        print(f'📊 Search results: {len(results)} results')
        
        if results:
            print('📄 First result preview:')
            for i, result in enumerate(results[:2], 1):
                content = result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"]
                print(f'   Result {i}: {content}')
                if "metadata" in result:
                    print(f'   Metadata: {result["metadata"]}')
            return True
        else:
            print('❌ No search results found')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pdf_chunks() 
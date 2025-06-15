#!/usr/bin/env python3
"""
Test to analyze document content and understand why certain documents aren't found.
"""

import os
import sys
from pathlib import Path
import unicodedata

# Setup paths
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Enable debug mode
os.environ['MY_AGENT_DEBUG'] = '1'

# Direct imports to avoid circular import
sys.path.insert(0, str(BASE_DIR / "metadata"))

from create_and_load_chromadb import (
    get_langchain_chroma_vectorstore,
    normalize_czech_text
)

# Constants
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

# Key documents we're looking for
KEY_SELECTIONS = ['SLD044T02', 'SLD036T02', 'SLD036T01', 'STAV799AT1', 'SLD025T02A']

# Queries to analyze
ORIGINAL_QUERY = "jaky je pocet obytnych domu vlastneny bytovymi drzustvy?"
REWRITTEN_QUERY = "Jaký je počet obytných domů vlastněných bytovými družstvy?"

def analyze_documents():
    """Analyze document content to understand search behavior"""
    print("🔍 DOCUMENT CONTENT ANALYSIS")
    print("="*80)
    
    try:
        # Get ChromaDB vectorstore
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        
        # Get all documents
        raw_docs = chroma_vectorstore.get(include=["documents", "metadatas"])
        
        print(f"Total documents in ChromaDB: {len(raw_docs['documents'])}")
        
        # Find and analyze key documents
        key_docs = {}
        for i, (doc, meta) in enumerate(zip(raw_docs["documents"], raw_docs["metadatas"])):
            selection = meta.get('selection') if meta else None
            if selection in KEY_SELECTIONS:
                key_docs[selection] = {
                    'content': doc,
                    'metadata': meta,
                    'index': i
                }
        
        print(f"\nFound {len(key_docs)} key documents:")
        for selection, info in key_docs.items():
            print(f"\n📄 {selection}:")
            content = info['content']
            print(f"   Length: {len(content)} characters")
            print(f"   Preview: {content[:200]}...")
            
            # Check for key terms
            key_terms = ['bytov', 'družstv', 'obytný', 'dům', 'dom']
            print(f"   Key terms found:")
            for term in key_terms:
                if term.lower() in content.lower():
                    print(f"     ✅ '{term}' found")
                else:
                    print(f"     ❌ '{term}' not found")
            
            # Check normalized versions
            normalized = normalize_czech_text(content)
            print(f"   Normalized preview: {normalized[:200]}...")
        
        # Analyze query terms
        print(f"\n🔍 QUERY ANALYSIS:")
        print(f"Original query: {ORIGINAL_QUERY}")
        print(f"Rewritten query: {REWRITTEN_QUERY}")
        print(f"Original normalized: {normalize_czech_text(ORIGINAL_QUERY)}")
        print(f"Rewritten normalized: {normalize_czech_text(REWRITTEN_QUERY)}")
        
        # Check which documents contain query-related terms
        query_terms = ['bytov', 'družstv', 'obytný', 'dům', 'dom', 'pocet', 'počet']
        print(f"\n📊 DOCUMENT TERM ANALYSIS:")
        
        term_matches = {}
        for term in query_terms:
            term_matches[term] = []
            for i, (doc, meta) in enumerate(zip(raw_docs["documents"], raw_docs["metadatas"])):
                selection = meta.get('selection') if meta else f"doc_{i}"
                if term.lower() in doc.lower():
                    term_matches[term].append(selection)
        
        for term, matches in term_matches.items():
            print(f"   '{term}': {len(matches)} documents")
            if matches:
                # Show first 10 matches
                print(f"     Examples: {matches[:10]}")
                # Check if any key selections are in matches
                key_in_matches = [sel for sel in KEY_SELECTIONS if sel in matches]
                if key_in_matches:
                    print(f"     Key selections: {key_in_matches}")
        
        return key_docs
        
    except Exception as e:
        print(f"Error in document analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Main analysis function"""
    key_docs = analyze_documents()
    
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    if key_docs:
        print(f"✅ Found {len(key_docs)} key documents in ChromaDB")
        print("Key documents:", list(key_docs.keys()))
    else:
        print("❌ No key documents found in ChromaDB")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check if key documents contain the right terms")
    print("2. Verify text normalization is working correctly")
    print("3. Consider adjusting search strategy based on term analysis")

if __name__ == "__main__":
    main() 
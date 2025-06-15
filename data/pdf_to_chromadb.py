#!/usr/bin/env python3
"""PDF to ChromaDB Document Processing and Search

This module provides functionality to process large PDF documents (including those with tables),
chunk them appropriately, generate embeddings, store in ChromaDB, and perform hybrid search
with Cohere reranking.

Key Features:
-------------
1. PDF Processing:
   - Simple and reliable PyMuPDF-based PDF parsing
   - Handles tables and complex layouts
   - Smart text chunking with token awareness
   - Preserves document structure information
   - Page-based metadata tracking

2. Document Management:
   - ChromaDB vector storage
   - Document deduplication using MD5 hashing
   - UUID-based document identification
   - Metadata preservation (page numbers, chunks)
   - Automatic document splitting for long texts

3. Embedding Generation:
   - Azure OpenAI embedding model integration
   - Batch embedding generation
   - Token limit handling (8190 tokens max)
   - Smart text chunking for long documents

4. Hybrid Search:
   - Combines semantic search (Azure embeddings) with BM25
   - Semantic-focused weighting (85% semantic, 15% BM25)
   - Cohere reranking for final results
   - Configurable result count and thresholds

5. Error Handling:
   - Comprehensive error handling
   - Progress tracking with tqdm
   - Debug logging support
   - Fallback mechanisms

Usage Example:
-------------
# Process PDF and create ChromaDB collection
collection = process_pdf_to_chromadb(
    pdf_path="large_document.pdf",
    collection_name="my_pdf_collection"
)

# Perform hybrid search with reranking
results = search_pdf_documents(
    collection=collection,
    query="your search query",
    top_k=2
)
"""

#==============================================================================
# IMPORTS
#==============================================================================
import os
import sys
from pathlib import Path
import sqlite3
import hashlib
from uuid import uuid4
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set

# Third-party imports
import chromadb
import tqdm as tqdm_module
import tiktoken
import cohere
import numpy as np
import logging
import fitz  # PyMuPDF
from langchain_core.documents import Document

# Add this import after the other imports
try:
    from rank_bm25 import BM25Okapi
    print("rank_bm25 is available. BM25 search will be enabled.")
except ImportError:
    print("Warning: rank_bm25 not available. BM25 search will be disabled.")
    BM25Okapi = None

# --- Ensure project root is in sys.path for local imports ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent  # Directory containing this script (data/)
    BASE_DIR = Path(__file__).resolve().parents[1]  # Project root (one level up from data/)
except NameError:
    SCRIPT_DIR = Path(os.getcwd())
    BASE_DIR = Path(os.getcwd())
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local Imports - adjust path as needed for your project structure
try:
    from my_agent.utils.models import get_azure_embedding_model
except ImportError:
    print("Warning: Could not import get_azure_embedding_model. You'll need to adjust the import path.")
    get_azure_embedding_model = None

#==============================================================================
# CONFIGURATION
#==============================================================================
# =====================================================================
# MAIN CONFIGURATION - MODIFY THESE SETTINGS
# =====================================================================
# Processing Mode
PROCESS_PDF = 0  # Set to 1 to process PDF and create/update ChromaDB
                 # Set to 0 to test search on existing ChromaDB only

# PDF Parsing Method Selection
PDF_PARSING_METHOD = "llamaparse"  # Options: "pymupdf", "pymupdf4llm", "llamaparse"
                                # pymupdf: Basic PyMuPDF (current method)
                                # pymupdf4llm: Enhanced PyMuPDF for LLMs with better table handling
                                # llamaparse: LlamaIndex's premium parser (requires API key)

# File and Collection Settings
PDF_FILENAME = "32019824.pdf"  # Just the filename (PDF should be in same folder as script)
COLLECTION_NAME = "pdf_document_collection"  # ChromaDB collection name

# Test Query Settings
# TEST_QUERY = "What is the main topic discussed in this document?"
TEST_QUERY = "Kolik byla prumerna teplotav meteorologicke stanici Praha-Karlov roku 2010?"

# Azure OpenAI Settings
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"  # Your Azure deployment name

# LlamaParse Settings (only needed if using llamaparse method)
LLAMAPARSE_API_KEY = os.environ.get("LLAMAPARSE_API_KEY", "")  # Read from .env file

# Token and Chunking Settings
MAX_TOKENS = 8190          # Token limit for Azure OpenAI
MIN_CHUNK_SIZE = 100       # Minimum chunk size in characters
MAX_CHUNK_SIZE = 2000      # Maximum chunk size in characters

# Search Settings
HYBRID_SEARCH_RESULTS = 20      # Number of results from hybrid search
SEMANTIC_WEIGHT = 0.85          # Weight for semantic search (0.0-1.0)
BM25_WEIGHT = 0.15              # Weight for BM25 search (0.0-1.0)
FINAL_RESULTS_COUNT = 2         # Number of final results to return

# Debug and Processing Settings
PDF_ID = 40                     # Unique identifier for debug messages
CHUNK_OVERLAP = 200             # Character overlap between chunks

# =====================================================================
# PATH CONFIGURATION - AUTOMATICALLY SET
# =====================================================================
PDF_PATH = SCRIPT_DIR / PDF_FILENAME  # Full path to PDF file

# ChromaDB storage location - method-specific folders
if PDF_PARSING_METHOD == "llamaparse":
    CHROMA_DB_PATH = SCRIPT_DIR / "pdf_chromadb_llamaparse"
elif PDF_PARSING_METHOD == "pymupdf4llm":
    CHROMA_DB_PATH = SCRIPT_DIR / "pdf_chromadb_pymupdf4llm"
else:  # pymupdf
    CHROMA_DB_PATH = SCRIPT_DIR / "pdf_chromadb"

#==============================================================================
# CONSTANTS & DERIVED SETTINGS
#==============================================================================

#==============================================================================
# MONITORING AND METRICS
#==============================================================================
@dataclass
class PDFMetrics:
    """Metrics collection for PDF processing statistics."""
    start_time: float = field(default_factory=time.time)
    total_pages: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_processing_time: float = 0
    failed_records: list = field(default_factory=list)
    
    def update_processing_time(self) -> None:
        """Update the total processing time based on the current time."""
        self.total_processing_time = time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_pages": self.total_pages,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "total_processing_time": self.total_processing_time,
            "average_time_per_chunk": self.total_processing_time / max(1, self.processed_chunks),
            "success_rate": (self.processed_chunks / max(1, self.processed_chunks + self.failed_chunks)) * 100,
            "failed_records": self.failed_records
        }

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled."""
    if os.environ.get('MY_AGENT_DEBUG', '0') == '1':
        print(f"[PDF_CHROMADB] {msg}")

def get_document_hash(text: str) -> str:
    """Generate MD5 hash for a document text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a string using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def split_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into chunks that don't exceed the token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    
    if total_tokens <= max_tokens:
        return [text]
    
    num_chunks = (total_tokens + max_tokens - 1) // max_tokens  # ceil division
    chunks = []
    for i in range(num_chunks):
        start = i * max_tokens
        end = min((i + 1) * max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def smart_text_chunking(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, 
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Smart text chunking that respects sentence boundaries and token limits.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a good breaking point (sentence end, paragraph, etc.)
        chunk_text = text[start:end]
        
        # Look for sentence endings near the end of the chunk
        for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 200), -1):
            if chunk_text[i] in '.!?':
                end = start + i + 1
                break
        
        chunk = text[start:end]
        
        # Ensure chunk meets minimum size requirements
        if len(chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk.strip())
        
        start = end - overlap if end > overlap else end
    
    return [chunk for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_SIZE]

def normalize_czech_text(text: str) -> str:
    """Advanced Czech text normalization for better search matching."""
    if not text:
        return text
    
    # Convert to lowercase first
    text = text.lower()
    
    # Advanced Czech diacritics mapping for normalization
    czech_diacritics_map = {
        # Primary Czech diacritics
        'á': 'a', 'č': 'c', 'ď': 'd', 'é': 'e', 'ě': 'e', 'í': 'i', 'ň': 'n',
        'ó': 'o', 'ř': 'r', 'š': 's', 'ť': 't', 'ú': 'u', 'ů': 'u', 'ý': 'y', 'ž': 'z',
    }
    
    # Create ASCII version  
    ascii_text = text
    for diacritic, ascii_char in czech_diacritics_map.items():
        ascii_text = ascii_text.replace(diacritic, ascii_char)
    
    # Return both versions separated by space for broader indexing
    if ascii_text != text:
        return f"{text} {ascii_text}"
    return text

#==============================================================================
# PDF PROCESSING
#==============================================================================
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF with page-level metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    debug_print(f"Opening PDF: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            text = page.get_text()
            
            # Clean and normalize text
            text = text.strip()
            if not text:
                debug_print(f"Page {page_num + 1} contains no text")
                continue
            
            # Get page metadata
            page_data = {
                'text': text,
                'page_number': page_num + 1,
                'char_count': len(text),
                'word_count': len(text.split()),
                'source_file': os.path.basename(pdf_path),
                'parsing_method': 'pymupdf'
            }
            
            pages_data.append(page_data)
            debug_print(f"Page {page_num + 1}: {len(text)} characters, {len(text.split())} words")
        
        doc.close()
        debug_print(f"Successfully extracted text from {len(pages_data)} pages")
        return pages_data
        
    except Exception as e:
        debug_print(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_with_pymupdf4llm(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF4LLM for better table handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    debug_print(f"Opening PDF with PyMuPDF4LLM: {pdf_path}")
    
    try:
        # Try to import pymupdf4llm
        try:
            import pymupdf4llm
        except ImportError:
            debug_print("PyMuPDF4LLM not installed. Install with: pip install pymupdf4llm")
            debug_print("Falling back to basic PyMuPDF...")
            return extract_text_from_pdf(pdf_path)
        
        # Extract with page chunks for better structure
        md_data = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,  # Get page-by-page data
            extract_words=False,  # Don't need word-level extraction
            write_images=False  # Don't extract images for now
        )
        
        pages_data = []
        
        for page_idx, page_data in enumerate(md_data):
            # PyMuPDF4LLM returns markdown text
            text = page_data.get('text', '') if isinstance(page_data, dict) else str(page_data)
            
            if not text.strip():
                debug_print(f"Page {page_idx + 1} contains no text")
                continue
            
            # Get page metadata
            page_info = {
                'text': text,
                'page_number': page_idx + 1,
                'char_count': len(text),
                'word_count': len(text.split()),
                'source_file': os.path.basename(pdf_path),
                'parsing_method': 'pymupdf4llm'
            }
            
            pages_data.append(page_info)
            debug_print(f"Page {page_idx + 1}: {len(text)} characters, {len(text.split())} words (PyMuPDF4LLM)")
        
        debug_print(f"Successfully extracted text from {len(pages_data)} pages using PyMuPDF4LLM")
        return pages_data
        
    except Exception as e:
        debug_print(f"Error extracting text with PyMuPDF4LLM: {str(e)}")
        debug_print("Falling back to basic PyMuPDF...")
        return extract_text_from_pdf(pdf_path)

def extract_text_with_llamaparse(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using LlamaParse for superior table handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    debug_print(f"Opening PDF with LlamaParse: {pdf_path}")
    
    try:
        # Try to import llama_parse
        try:
            from llama_parse import LlamaParse
        except ImportError:
            debug_print("LlamaParse not installed. Install with: pip install llama-parse")
            debug_print("Falling back to PyMuPDF4LLM...")
            return extract_text_with_pymupdf4llm(pdf_path)
        
        # Get API key from .env file or environment
        api_key = LLAMAPARSE_API_KEY or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            debug_print("LlamaParse API key not found. Set LLAMAPARSE_API_KEY in .env file or LLAMA_CLOUD_API_KEY environment variable")
            debug_print("Falling back to PyMuPDF4LLM...")
            return extract_text_with_pymupdf4llm(pdf_path)
        
        # Initialize LlamaParse with table-focused instructions
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction="""
            This document contains meteorological data with tables.
            Please preserve table structure and ensure that headers remain connected to their data.
            Pay special attention to tabular data with years, temperatures, and measurements.
            Convert tables to well-structured markdown format.
            """,
            verbose=True
        )
        
        # Parse the document
        documents = parser.load_data(pdf_path)
        
        pages_data = []
        
        for doc_idx, document in enumerate(documents):
            text = document.text if hasattr(document, 'text') else str(document)
            
            if not text.strip():
                debug_print(f"Document {doc_idx + 1} contains no text")
                continue
            
            # Get page metadata
            page_info = {
                'text': text,
                'page_number': doc_idx + 1,  # LlamaParse may not preserve exact page numbers
                'char_count': len(text),
                'word_count': len(text.split()),
                'source_file': os.path.basename(pdf_path),
                'parsing_method': 'llamaparse'
            }
            
            pages_data.append(page_info)
            debug_print(f"Document {doc_idx + 1}: {len(text)} characters, {len(text.split())} words (LlamaParse)")
        
        debug_print(f"Successfully extracted text from {len(pages_data)} documents using LlamaParse")
        return pages_data
        
    except Exception as e:
        debug_print(f"Error extracting text with LlamaParse: {str(e)}")
        debug_print("Falling back to PyMuPDF4LLM...")
        return extract_text_with_pymupdf4llm(pdf_path)

def process_pdf_pages_to_chunks(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process PDF pages into chunks suitable for embedding.
    
    Args:
        pages_data: List of page data from extract_text_from_pdf
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    all_chunks = []
    chunk_id = 0
    
    for page_data in pages_data:
        text = page_data['text']
        page_num = page_data['page_number']
        
        # Smart chunking of the page text
        page_chunks = smart_text_chunking(text)
        
        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Further split by tokens if needed
            token_chunks = split_text_by_tokens(chunk_text)
            
            for token_chunk_idx, token_chunk in enumerate(token_chunks):
                token_count = num_tokens_from_string(token_chunk)
                
                chunk_data = {
                    'id': chunk_id,
                    'text': token_chunk,
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'token_chunk_index': token_chunk_idx,
                    'total_page_chunks': len(page_chunks),
                    'total_token_chunks': len(token_chunks),
                    'char_count': len(token_chunk),
                    'token_count': token_count,
                    'source_file': page_data['source_file'],
                    'doc_hash': get_document_hash(token_chunk)
                }
                
                all_chunks.append(chunk_data)
                chunk_id += 1
    
    debug_print(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")
    return all_chunks

#==============================================================================
# CHROMADB OPERATIONS
#==============================================================================
def process_pdf_to_chromadb(
    pdf_path: str,
    collection_name: str = COLLECTION_NAME,
    deployment: str = AZURE_EMBEDDING_DEPLOYMENT
) -> chromadb.Collection:
    """
    Process a PDF file and store it in ChromaDB with embeddings.
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the ChromaDB collection
        deployment: Azure embedding deployment name
        
    Returns:
        ChromaDB collection object
    """
    metrics = PDFMetrics()
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if get_azure_embedding_model is None:
        raise ImportError("Azure embedding model not available. Check your imports.")
    
    try:
        # Extract text from PDF using selected method
        debug_print(f"Processing PDF: {pdf_path}")
        debug_print(f"Using parsing method: {PDF_PARSING_METHOD}")
        
        if PDF_PARSING_METHOD == "llamaparse":
            pages_data = extract_text_with_llamaparse(pdf_path)
        elif PDF_PARSING_METHOD == "pymupdf4llm":
            pages_data = extract_text_with_pymupdf4llm(pdf_path)
        else:  # default to pymupdf
            pages_data = extract_text_from_pdf(pdf_path)
            
        metrics.total_pages = len(pages_data)
        
        if not pages_data:
            raise ValueError("No text found in PDF")
        
        # Process pages into chunks
        chunks_data = process_pdf_pages_to_chunks(pages_data)
        debug_print(f"Created {len(chunks_data)} chunks for processing")
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            debug_print(f"Created new ChromaDB collection: {collection_name}")
        except Exception:
            collection = client.get_collection(name=collection_name)
            debug_print(f"Using existing ChromaDB collection: {collection_name}")
        
        # Check for existing documents
        existing = collection.get(include=["metadatas"], limit=10000)
        existing_hashes = set()
        if existing and "metadatas" in existing and existing["metadatas"]:
            for metadata in existing["metadatas"]:
                if isinstance(metadata, dict) and metadata is not None:
                    doc_hash = metadata.get('doc_hash')
                    if doc_hash:
                        existing_hashes.add(doc_hash)
        
        debug_print(f"Found {len(existing_hashes)} existing documents in ChromaDB")
        
        # Filter out existing chunks
        new_chunks = [chunk for chunk in chunks_data if chunk['doc_hash'] not in existing_hashes]
        debug_print(f"Processing {len(new_chunks)} new chunks")
        
        if not new_chunks:
            debug_print("No new chunks to process")
            return collection
        
        # Initialize embedding client
        embedding_client = get_azure_embedding_model()
        
        # Process chunks with progress bar
        with tqdm_module.tqdm(
            total=len(new_chunks),
            desc="Processing chunks",
            leave=True,
            ncols=100
        ) as pbar:
            
            for chunk_data in new_chunks:
                try:
                    # Generate embedding
                    response = embedding_client.embeddings.create(
                        input=[chunk_data['text']],
                        model=deployment
                    )
                    embedding = response.data[0].embedding
                    
                    # Create metadata for ChromaDB
                    metadata = {
                        'page_number': chunk_data['page_number'],
                        'chunk_index': chunk_data['chunk_index'],
                        'token_chunk_index': chunk_data['token_chunk_index'],
                        'char_count': chunk_data['char_count'],
                        'token_count': chunk_data['token_count'],
                        'source_file': chunk_data['source_file'],
                        'doc_hash': chunk_data['doc_hash'],
                        'chunk_id': chunk_data['id']
                    }
                    
                    # Add to ChromaDB
                    collection.add(
                        documents=[chunk_data['text']],
                        embeddings=[embedding],
                        ids=[str(uuid4())],
                        metadatas=[metadata]
                    )
                    
                    metrics.processed_chunks += 1
                    pbar.update(1)
                    
                except Exception as e:
                    debug_print(f"Error processing chunk {chunk_data['id']}: {str(e)}")
                    metrics.failed_chunks += 1
                    metrics.failed_records.append((chunk_data['id'], str(e)))
                    pbar.update(1)
                    continue
        
        # Print final statistics
        metrics.update_processing_time()
        debug_print(f"\nProcessing completed:")
        debug_print(f"- Total pages: {metrics.total_pages}")
        debug_print(f"- Total chunks: {len(chunks_data)}")
        debug_print(f"- Successfully processed: {metrics.processed_chunks}")
        debug_print(f"- Failed: {metrics.failed_chunks}")
        debug_print(f"- Processing time: {metrics.total_processing_time:.2f} seconds")
        debug_print(f"- Success rate: {metrics.to_dict()['success_rate']:.1f}%")
        
        return collection
        
    except Exception as e:
        debug_print(f"Error in process_pdf_to_chromadb: {str(e)}")
        raise

#==============================================================================
# SEARCH FUNCTIONS
#==============================================================================
def similarity_search_chromadb(collection, embedding_client, query: str, 
                              embedding_model_name: str = AZURE_EMBEDDING_DEPLOYMENT, 
                              k: int = 10):
    """Perform similarity search using ChromaDB."""
    query_embedding = embedding_client.embeddings.create(
        input=[query],
        model=embedding_model_name
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def hybrid_search(collection, query_text: str, n_results: int = HYBRID_SEARCH_RESULTS) -> List[Dict]:
    """
    Hybrid search combining semantic and BM25 approaches.
    """
    debug_print(f"Hybrid search for query: '{query_text}'")
    
    try:
        # Normalize query
        normalized_query = normalize_czech_text(query_text)
        
        # Semantic search
        semantic_results = []
        try:
            embedding_client = get_azure_embedding_model()
            semantic_raw = similarity_search_chromadb(
                collection=collection,
                embedding_client=embedding_client,
                query=normalized_query,
                k=n_results
            )
            
            for i, (doc, meta, distance) in enumerate(zip(
                semantic_raw["documents"][0], 
                semantic_raw["metadatas"][0], 
                semantic_raw["distances"][0]
            )):
                similarity_score = max(0, 1 - (distance / 2))
                semantic_results.append({
                    'id': f"semantic_{i}",
                    'document': doc,
                    'metadata': meta,
                    'semantic_score': similarity_score,
                    'source': 'semantic'
                })
                
        except Exception as e:
            debug_print(f"Semantic search failed: {e}")
            semantic_results = []
        
        # BM25 search
        bm25_results = []
        try:
            all_data = collection.get(include=['documents', 'metadatas'])
            
            if all_data and 'documents' in all_data and all_data['documents']:
                documents = all_data['documents']
                metadatas = all_data['metadatas']
                
                processed_docs = [normalize_czech_text(doc) for doc in documents]
                
                if BM25Okapi:
                    tokenized_docs = [doc.split() for doc in processed_docs]
                    bm25 = BM25Okapi(tokenized_docs)
                    
                    tokenized_query = normalized_query.split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    top_indices = np.argsort(bm25_scores)[::-1][:n_results]
                    
                    for i, idx in enumerate(top_indices):
                        if bm25_scores[idx] > 0:
                            bm25_results.append({
                                'id': f"bm25_{i}",
                                'document': documents[idx],
                                'metadata': metadatas[idx] if idx < len(metadatas) else {},
                                'bm25_score': float(bm25_scores[idx]),
                                'source': 'bm25'
                            })
                
        except Exception as e:
            debug_print(f"BM25 search failed: {e}")
            bm25_results = []
        
        # Combine results with semantic focus
        combined_results = {}
        
        # Process semantic results (primary)
        for result in semantic_results:
            doc_id = result['metadata'].get('chunk_id', result['document'][:50])
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['bm25_score'] = 0.0
        
        # Process BM25 results (secondary)
        for result in bm25_results:
            doc_id = result['metadata'].get('chunk_id', result['document'][:50])
            if doc_id in combined_results:
                combined_results[doc_id]['bm25_score'] = result['bm25_score']
                combined_results[doc_id]['source'] = 'hybrid'
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['semantic_score'] = 0.0
        
        # Calculate final scores with semantic focus
        final_results = []
        max_semantic = max((r.get('semantic_score', 0) for r in combined_results.values()), default=1)
        max_bm25 = max((r.get('bm25_score', 0) for r in combined_results.values()), default=1)
        
        semantic_weight = SEMANTIC_WEIGHT
        bm25_weight = BM25_WEIGHT
        
        for doc_id, result in combined_results.items():
            semantic_score = result.get('semantic_score', 0.0) / max_semantic if max_semantic > 0 else 0.0
            bm25_score = result.get('bm25_score', 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            
            final_score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
            
            result['score'] = final_score
            result['semantic_score'] = semantic_score
            result['bm25_score'] = bm25_score
            
            final_results.append(result)
        
        # Sort by final score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:n_results]
        
    except Exception as e:
        debug_print(f"Hybrid search failed: {e}")
        return []

def cohere_rerank(query, docs, top_n):
    """Rerank documents using Cohere's rerank model."""
    cohere_api_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_api_key:
        debug_print("Warning: COHERE_API_KEY not found. Skipping reranking.")
        return [(doc, type('obj', (object,), {'relevance_score': 0.5, 'index': i})()) for i, doc in enumerate(docs)]
    
    co = cohere.Client(cohere_api_key)
    texts = [doc.page_content for doc in docs]
    docs_for_cohere = [{"text": t} for t in texts]
    
    try:
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=docs_for_cohere,
            top_n=top_n
        )
        
        reranked = []
        for res in rerank_response.results:
            doc = docs[res.index]
            reranked.append((doc, res))
        return reranked
        
    except Exception as e:
        debug_print(f"Cohere reranking failed: {e}")
        # Return original docs with dummy scores
        return [(doc, type('obj', (object,), {'relevance_score': 0.5, 'index': i})()) for i, doc in enumerate(docs)]

def search_pdf_documents(collection, query: str, top_k: int = FINAL_RESULTS_COUNT) -> List[Dict[str, Any]]:
    """
    Search PDF documents using hybrid search and Cohere reranking.
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of top results to return
        
    Returns:
        List of search results with metadata
    """
    debug_print(f"Searching for: '{query}' (returning top {top_k} results)")
    
    try:
        # Step 1: Hybrid search
        hybrid_results = hybrid_search(collection, query, n_results=HYBRID_SEARCH_RESULTS)
        
        if not hybrid_results:
            debug_print("No results from hybrid search")
            return []
        
        # Step 2: Convert to Document objects for reranking
        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(
                page_content=result['document'],
                metadata=result['metadata']
            )
            hybrid_docs.append(doc)
        
        # Step 3: Cohere reranking
        reranked = cohere_rerank(query, hybrid_docs, top_n=min(top_k * 2, len(hybrid_docs)))
        
        # Step 4: Format final results
        final_results = []
        for i, (doc, res) in enumerate(reranked[:top_k]):
            result = {
                'rank': i + 1,
                'text': doc.page_content,
                'metadata': doc.metadata,
                'cohere_score': res.relevance_score,
                'page_number': doc.metadata.get('page_number', 'N/A'),
                'source_file': doc.metadata.get('source_file', 'N/A'),
                'char_count': doc.metadata.get('char_count', len(doc.page_content))
            }
            final_results.append(result)
        
        debug_print(f"Returning {len(final_results)} final results")
        return final_results
        
    except Exception as e:
        debug_print(f"Error in search_pdf_documents: {str(e)}")
        return []

#==============================================================================
# MAIN EXECUTION
#==============================================================================
def main():
    """Main function to demonstrate PDF to ChromaDB processing and search."""
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    print("🔍 PDF to ChromaDB Processing and Search")
    print("=" * 50)
    print(f"📁 Script location: {SCRIPT_DIR}")
    print(f"📄 PDF file: {PDF_PATH}")
    print(f"🔧 Mode: {'PDF Processing + Search' if PROCESS_PDF else 'Search Testing Only'}")
    print(f"🔬 Parsing method: {PDF_PARSING_METHOD}")
    print(f"🗄️  ChromaDB location: {CHROMA_DB_PATH}")
    
    if PROCESS_PDF:
        # =====================================================================
        # STEP 1: PDF PROCESSING MODE
        # =====================================================================
        print(f"\n📄 Processing PDF: {PDF_PATH}")
        
        # Check if PDF exists
        if not PDF_PATH.exists():
            print(f"❌ Error: PDF file not found at {PDF_PATH}")
            print(f"Please make sure your PDF file '{PDF_FILENAME}' is in the same directory as this script: {SCRIPT_DIR}")
            return
        
        try:
            # Process PDF to ChromaDB
            collection = process_pdf_to_chromadb(
                pdf_path=str(PDF_PATH),
                collection_name=COLLECTION_NAME
            )
            print(f"✅ Successfully processed PDF and stored in ChromaDB collection: {COLLECTION_NAME}")
            
        except Exception as e:
            print(f"❌ Error during PDF processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    else:
        # =====================================================================
        # STEP 2: TESTING MODE - Load existing ChromaDB
        # =====================================================================
        print(f"\n📚 Loading existing ChromaDB collection: {COLLECTION_NAME}")
        
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Successfully loaded collection: {COLLECTION_NAME}")
            
            # Check collection contents
            collection_info = collection.get(limit=1, include=["metadatas"])
            if collection_info and collection_info["metadatas"]:
                total_count = collection.count()
                print(f"📊 Collection contains {total_count} documents")
                sample_metadata = collection_info["metadatas"][0]
                if sample_metadata:
                    print(f"📄 Sample document from: {sample_metadata.get('source_file', 'Unknown file')}")
            else:
                print("⚠️  Collection appears to be empty")
                
        except Exception as e:
            print(f"❌ Error: Could not load collection '{COLLECTION_NAME}'")
            print(f"💡 Make sure you have processed a PDF first by setting PROCESS_PDF = 1")
            print(f"Error details: {str(e)}")
            return
    
    # =====================================================================
    # SEARCH TESTING (Common for both modes)
    # =====================================================================
    try:
        print(f"\n🔍 Testing search with query: '{TEST_QUERY}'")
        
        # Get more results for detailed analysis
        detailed_results = search_pdf_documents(
            collection=collection,
            query=TEST_QUERY,
            top_k=10  # Get top 10 for detailed analysis
        )
        
        if detailed_results:
            print(f"\n📊 Found {len(detailed_results)} relevant chunks:")
            print("=" * 100)
            print("🏆 DETAILED RESULTS WITH COHERE RERANKING SCORES")
            print("=" * 100)
            
            # Filter results by score threshold
            SCORE_THRESHOLD = 0.01
            filtered_results = [r for r in detailed_results if r['cohere_score'] > SCORE_THRESHOLD]
            
            print(f"📋 Showing {len(filtered_results)} chunks with Cohere score > {SCORE_THRESHOLD}")
            print(f"📊 Total chunks analyzed: {len(detailed_results)}")
            
            for result in filtered_results:
                print(f"\n🏆 Rank {result['rank']} | 🎯 COHERE RERANK SCORE: {result['cohere_score']:.6f}")
                print(f"📄 Page: {result['page_number']} | File: {result['source_file']}")
                print(f"📏 Length: {result['char_count']} characters")
                print("📝 Content Preview:")
                print("-" * 80)
                # Show first 200 characters for detailed view
                content = result['text'][:200]
                if len(result['text']) > 200:
                    content += "..."
                print(content)
                print("-" * 80)
            
            # Summary of ALL scores (including filtered out ones)
            print(f"\n📈 SCORE ANALYSIS (ALL 10 CHUNKS - COHERE RERANKING SCORES):")
            print("=" * 60)
            all_scores = [r['cohere_score'] for r in detailed_results]
            filtered_scores = [r['cohere_score'] for r in filtered_results]
            
            print(f"🔝 Highest Score: {max(all_scores):.6f}")
            print(f"🔻 Lowest Score:  {min(all_scores):.6f}")
            print(f"📊 Average Score: {sum(all_scores)/len(all_scores):.6f}")
            print(f"📏 Score Range:   {max(all_scores) - min(all_scores):.6f}")
            print(f"🎯 Above Threshold ({SCORE_THRESHOLD}): {len(filtered_scores)}/{len(all_scores)} chunks")
            
            # Show all scores for reference
            print(f"\n📋 ALL COHERE SCORES (Rank: Score):")
            print("-" * 40)
            for i, score in enumerate(all_scores, 1):
                status = "✅" if score > SCORE_THRESHOLD else "❌"
                print(f"  Rank {i:2d}: {score:.6f} {status}")
            print("-" * 40)
            
            # Show top 2 results in compact format (original behavior)
            print(f"\n🎯 TOP 2 RESULTS (FINAL SELECTION):")
            print("=" * 80)
            top_2_results = search_pdf_documents(
                collection=collection,
                query=TEST_QUERY,
                top_k=2
            )
            
            for result in top_2_results:
                print(f"\n🏆 Rank {result['rank']} (Cohere Score: {result['cohere_score']:.4f})")
                print(f"📄 Page: {result['page_number']} | File: {result['source_file']}")
                print(f"📏 Length: {result['char_count']} characters")
                print("📝 Content:")
                print("-" * 60)
                # Show first 300 characters for final results
                content = result['text'][:300]
                if len(result['text']) > 300:
                    content += "..."
                print(content)
                print("-" * 60)
        else:
            print("❌ No results found for the query")
        
        print(f"\n🎉 {'Processing and search' if PROCESS_PDF else 'Search testing'} completed successfully!")
        print(f"📁 ChromaDB location: {CHROMA_DB_PATH}")
        
        # Usage hints
        if PROCESS_PDF:
            print(f"\n💡 To test different queries without reprocessing, set PROCESS_PDF = 0")
        else:
            print(f"\n💡 You can modify the TEST_QUERY variable to test different searches")
            print(f"💡 To process a new PDF, set PROCESS_PDF = 1")
        
    except Exception as e:
        print(f"❌ Error during search testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Enable debug mode for detailed logging
    os.environ['MY_AGENT_DEBUG'] = '1'
    
    main() 
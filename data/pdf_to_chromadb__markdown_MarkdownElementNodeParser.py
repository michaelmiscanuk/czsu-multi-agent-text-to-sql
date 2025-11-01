module_description = r"""PDF to ChromaDB Document Processing and Search System

This module provides a comprehensive pipeline for processing complex PDF documents (especially 
government/statistical reports with tables), extracting structured content using LlamaParse, 
applying markdown-based intelligent chunking, generating embeddings, storing in ChromaDB, 
and performing advanced hybrid search with multilingual support and Cohere reranking.

Designed for Czech government statistical reports with complex table structures, but adaptable 
to any PDF document processing workflow requiring high-quality semantic search capabilities.

Architecture Overview:
=====================
Three independent, configurable operations:
1. PDF Parsing & Content Extraction (LlamaParse integration with markdown output)
2. Intelligent Chunking & ChromaDB Storage (Markdown structural chunking strategy) 
3. Hybrid Search & Testing (Semantic + BM25 + Cohere reranking)

Key Features:
============
1. Advanced PDF Processing & Parsing:
   - LlamaParse integration with custom parsing instructions
   - Specialized table processing with MARKDOWN TABLE output (pipe-delimited format)
   - Complex layout handling (hierarchical tables, charts, mixed content)
   - Progress monitoring with job status tracking
   - Parallel processing support for multiple PDFs
   - Automatic retry logic and timeout handling
   - Natural markdown format output for seamless chunking

2. Intelligent Markdown-Based Structural Chunking:
   - Primary strategy: Parse markdown structure (tables, headers, paragraphs)
   - Markdown table detection using pipe (|) delimiters
   - Header-based section splitting (# ## ### markdown headers)
   - Preserves complete tables and text sections without splitting
   - Token-aware processing (8190 token limit compliance)
   - Content type preservation (tables vs text vs headers)
   - Quality validation with numerical data preservation checks
   - Configurable chunk sizes (MIN: 100, MAX: 4000 chars, 0 overlap)
   - Smart handling of complex document structures

3. Robust ChromaDB Document Management:
   - Persistent vector storage with cosine similarity indexing
   - Document deduplication using MD5 content hashing
   - Comprehensive metadata preservation (pages, chunks, source files, tokens)
   - Collection management with optional clearing/recreation
   - Batch processing with tqdm progress tracking
   - Error recovery with detailed failed record tracking
   - UUID-based document identification

4. Azure OpenAI Embedding Generation:
   - text-embedding-3-large model integration (1536-dimensional vectors)
   - Batch embedding generation for processing efficiency
   - Token limit validation and automatic text splitting
   - Configurable deployment selection via AZURE_EMBEDDING_DEPLOYMENT
   - Comprehensive error handling with retry logic
   - Progress tracking with visual indicators

5. Advanced Hybrid Search System:
   - Multi-modal search: Semantic (Azure OpenAI) + Keyword (BM25)
   - Configurable weighting (default: 85% semantic, 15% BM25)
   - Czech text normalization (diacritics handling) for better matching
   - Score normalization and combination algorithms
   - Cohere multilingual reranking (rerank-multilingual-v3.0)
   - Configurable result thresholds and filtering

6. Content Type Intelligence & Preservation:
   - Markdown table content extraction with structure preservation
   - Header-based text organization and sectioning
   - Chart and image descriptions in natural language paragraphs
   - Regular text section management with language preservation
   - Mixed content type processing and intelligent categorization
   - Context-rich chunk generation for optimal semantic search

7. Multilingual & Localization Support:
   - Czech language processing with diacritics normalization
   - English output for better LLM processing
   - English descriptions for tables and charts
   - Bilingual search term expansion for better matching
   - Cultural context preservation (Czech place names, terms)

Markdown Format System:
========================
LlamaParse outputs standard markdown with:
- Markdown tables using pipe (|) delimiters:
  | Header1 | Header2 | Header3 |
  | ------- | ------- | ------- |
  | Data1   | Data2   | Data3   |
- Headers using # ## ### for section titles
- Paragraphs for regular text and chart descriptions
- Bullet points and numbered lists for structured content
- Page separators: --- Page {page_number} ---

The custom_structural_chunking() function parses this markdown structure
to create intelligent chunks that preserve tables and maintain context.

Advanced Processing Flow:
========================
1. Configuration & Environment Setup:
   - Validates API keys (LlamaParse, Azure OpenAI, Cohere)
   - Configures three independent operations via flags
   - Sets up comprehensive debug logging system
   - Initializes persistent ChromaDB storage
   - Establishes Azure OpenAI embedding client connection

2. PDF Parsing & Content Extraction (Operation 1):
   - Processes multiple PDFs in parallel using ThreadPoolExecutor
   - Applies comprehensive LlamaParse instructions for markdown table output
   - Monitors parsing progress with real-time status updates
   - Implements timeout handling and error recovery
   - Saves parsed markdown text to .txt files
   - Validates content structure and markdown formatting
   - Provides detailed parsing statistics and quality metrics

3. Intelligent Content Chunking:
   - Applies markdown structural chunking strategy via custom_structural_chunking()
   - Detects markdown tables by pipe (|) delimiters
   - Splits content by headers (# ## ###) and structural elements
   - Preserves complete tables and text sections without mid-content splits
   - Token-aware processing ensures chunks stay within limits
   - Validates chunk quality with comprehensive metrics
   - Tracks numerical data preservation for statistical content

4. ChromaDB Storage & Management (Operation 2):
   - Processes chunks and generates embeddings in batches
   - Implements document deduplication using content hashes
   - Stores documents with comprehensive metadata structure
   - Tracks processing statistics and failure recovery
   - Provides detailed storage metrics and success rates

5. Hybrid Search & Testing (Operation 3):
   - Executes configurable test queries (Czech language support)
   - Performs hybrid search combining multiple approaches
   - Applies Cohere reranking for result optimization
   - Returns ranked results with detailed scoring metrics
   - Provides comprehensive debug information and analytics
   - Supports configurable result counts and score thresholds

6. Quality Assurance & Validation:
   - Validates chunk quality using multiple metrics
   - Checks numerical data preservation for statistical content
   - Monitors token limits and character count distributions
   - Tracks content type distribution across chunks
   - Provides comprehensive debug reporting and analytics
   - Implements automated quality scoring algorithms

Configuration System:
=====================
Tunable hyperparameters for optimal performance:

Core Processing:
- MAX_TOKENS: 8190 (Azure OpenAI embedding model limit)
- MIN_CHUNK_SIZE: 100 (minimum viable chunk size)
- MAX_CHUNK_SIZE: 5000 (optimal for semantic coherence)
- CHUNK_OVERLAP: 0 (disabled for separator-based chunking)

Search Configuration:
- HYBRID_SEARCH_RESULTS: 40 (initial result pool size)
- SEMANTIC_WEIGHT: 0.85 (semantic search importance)
- BM25_WEIGHT: 0.15 (keyword search importance)
- FINAL_RESULTS_COUNT: 2 (final filtered results)

API Integration:
- AZURE_EMBEDDING_DEPLOYMENT: "text-embedding-3-large__test1"
- LlamaParse: Premium parsing with table optimization
- Cohere: rerank-multilingual-v3.0 for final ranking

Usage Examples:
==============

Full Pipeline Execution:
-------------------------
# Configure all three operations
PARSE_WITH_LLAMAPARSE = 1       # Parse PDFs with advanced table handling
CHUNK_AND_STORE = 1             # Apply intelligent chunking and store
DO_TESTING = 1                  # Test search with sample queries

# Define document set
PDF_FILENAMES = ["report_2023.pdf", "statistics_2024.pdf"]

# Execute complete pipeline
python pdf_to_chromadb.py

Incremental Processing:
-----------------------
# Parse new documents only
PARSE_WITH_LLAMAPARSE = 1
CHUNK_AND_STORE = 0
DO_TESTING = 0

# Add to existing collection
CHUNK_AND_STORE = 1

# Test with new queries
DO_TESTING = 1
TEST_QUERY = "kolik učitelů pracovalo v roce 2023?"

Required Environment:
====================
Dependencies:
- Python 3.8+
- chromadb (vector storage)
- tiktoken (token counting)
- tqdm (progress tracking)
- requests (API communication)
- numpy (numerical operations)
- rank-bm25 (keyword search)
- cohere (reranking)
- langchain-core (document handling)

API Requirements:
- LlamaParse API key (LLAMAPARSE_API_KEY)
- Azure OpenAI API access (embedding model)
- Cohere API key (COHERE_API_KEY, optional)
- Azure embedding deployment configured

File System:
- Write permissions for ChromaDB directory
- PDF files in script directory
- Sufficient storage for parsed text files

Expected Output:
===============
File Structure:
- {pdf_name}_llamaparse_parsed.txt (parsed content with separators)
- pdf_chromadb_llamaparse_v2/ (ChromaDB collection directory)
  ├── chroma.sqlite3 (metadata database)
  └── collection_data/ (vector storage)

Processing Results:
- Document embeddings (1536-dimensional vectors)
- Comprehensive chunk metadata (pages, indices, tokens, hashes)
- Content type classification and statistics
- Processing metrics and quality scores
- Search results with hybrid scoring and reranking

Search Output Format:
- Ranked results with Cohere reranking scores
- Source file and page number attribution
- Character count and content preview
- Score analysis and threshold filtering
- Detailed debug information for optimization

Error Handling & Recovery:
==========================
Comprehensive error management:
- PDF parsing failures with detailed diagnostics and retry logic
- Embedding generation errors with automatic recovery
- ChromaDB connection and storage error handling
- Token limit violations with automatic text splitting
- Content validation and quality check failures
- API timeout and rate limiting management
- Progress tracking with graceful interruption handling

Performance Optimization:
========================
- Parallel PDF processing using ThreadPoolExecutor
- Batch embedding generation for API efficiency
- Efficient document deduplication using MD5 hashing
- Persistent ChromaDB storage with optimized indexing
- Memory-efficient text processing with streaming
- Smart caching of parsed content and embeddings
- Configurable batch sizes and processing limits

Quality Metrics & Analytics:
===========================
Built-in quality assurance:
- Chunk quality scoring (empty, size, context, numerical data)
- Content type distribution analysis
- Token count validation and statistics
- Processing success rates and failure tracking
- Search relevance scoring and threshold analysis
- Numerical data preservation verification
- Debug logging and comprehensive error reporting

# Run the script:
# python pdf_to_chromadb.py
"""

import hashlib
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from uuid import uuid4

# Third-party imports
import chromadb
import cohere
import numpy as np
import tiktoken
import tqdm as tqdm_module
from langchain_core.documents import Document
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import Settings

# Add this import after the other imports
try:
    from rank_bm25 import BM25Okapi

    # print("rank_bm25 is available. BM25 search will be enabled.")
except ImportError:
    # print("Warning: rank_bm25 not available. BM25 search will be disabled.")
    BM25Okapi = None

# --- Ensure project root is in sys.path for local imports ---
try:
    SCRIPT_DIR = (
        Path(__file__).resolve().parent
    )  # Directory containing this script (data/)
    BASE_DIR = (
        Path(__file__).resolve().parents[1]
    )  # Project root (one level up from data/)
except NameError:
    SCRIPT_DIR = Path(os.getcwd())
    BASE_DIR = Path(os.getcwd())
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local Imports - Azure OpenAI client setup
from api.utils.debug import print__chromadb_debug

try:
    from openai import AzureOpenAI
    import os

    def get_azure_embedding_model():
        """Create and return Azure OpenAI client for embeddings"""
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

except ImportError:
    print(
        "Warning: Could not import AzureOpenAI. You'll need to install openai package."
    )
    get_azure_embedding_model = None

# Import LLM function from models.py
try:
    from my_agent.utils.models import get_azure_llm_gpt_4o
except ImportError:
    print("Warning: Could not import get_azure_llm_gpt_4o from models.py")
    get_azure_llm_gpt_4o = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# =====================================================================
# MAIN CONFIGURATION - MODIFY THESE SETTINGS
# =====================================================================
# Processing Mode - Three independent operations
PARSE_WITH_LLAMAPARSE = 1  # Set to 1 to parse PDF with LlamaParse and save to txt file
# Set to 0 to skip parsing (use existing txt file)

CHUNK_AND_STORE = 1  # Set to 1 to chunk text and create/update ChromaDB
# Set to 0 to skip chunking (use existing ChromaDB)

DO_TESTING = 1  # Set to 1 to test search on existing ChromaDB
# Set to 0 to skip testing

# PDF Parsing Method Selection - LLAMAPARSE ONLY
PDF_PARSING_METHOD = (
    "llamaparse"  # Using LlamaIndex's premium parser (requires API key)
)

# LlamaParse Progress Monitoring Options
LLAMAPARSE_ENHANCED_MONITORING = (
    True  # Set to True for enhanced progress tracking with job status
)
# Set to False for standard LlamaParse with basic progress

# File and Collection Settings
# PDF_FILENAME = "33012024.pdf"  # Just the filename (PDF should be in same folder as script)
# PDF_FILENAME = "1_PDFsam_271_PDFsam_33012024.pdf"  # Just the filename (PDF should be in same folder as script)

# Multiple PDF Files Configuration - Add your PDF filenames here
PDF_FILENAMES = [
    # "684_PDFsam_32019824.pdf",
    # "PDFsam_merge__673_684.pdf",
    # "101_PDFsam_32019824.pdf",
    # "201_PDFsam_32019824.pdf",
    # "301_PDFsam_32019824.pdf",
    # "401_PDFsam_32019824.pdf",
    # "501_PDFsam_32019824.pdf",
    # "601_PDFsam_32019824.pdf",
    # "701_PDFsam_32019824.pdf",
    # "801_PDFsam_32019824.pdf",
    # "666_PDFsam_66632019824.pdf",
    # "32019824.pdf",
    # "1_PDFsam_32019824.pdf",
    # "501_PDFsam_32019824.pdf",
    # "661_PDFsam_32019824.pdf",
    # "453_461_524.pdf",
    "96_97_141.pdf"
]

COLLECTION_NAME = "pdf_document_collection"  # ChromaDB collection name

# Parsed Text File Settings - Will be auto-generated for each PDF
# PARSED_TEXT_FILENAME = f"{PDF_FILENAME}_{PDF_PARSING_METHOD}_parsed.txt"  # Auto-generated filename for parsed text

# Test Query Settings
# TEST_QUERY = "What is the main topic discussed in this document?"
# TEST_QUERY = "Kolik byla prumerna teplotav meteorologicke stanici Praha-Karlov roku 2010?"
# TEST_QUERY = "Kolik bylo hektaru zahrad roku 2023 v Praze?"
# TEST_QUERY = "Jake vydaje byly vladou na Obranu v roce 2023 nebo 2024"
# TEST_QUERY = "Jaka je cista mesicni mzda v Praze"
# TEST_QUERY = "Kolik je ve meste Most soukromych podnikatelu?"
# TEST_QUERY = "Kolik je centralnich bank je v Ceske republice jako ekonomickych subjektu?"
# TEST_QUERY = "Jake metody se pouzili pro ziskavani dat v zemedelstvi?"
# TEST_QUERY = "jak se zmenili zasoby v milionech korun v oboru cinnosti - vzdelani? Porovnej posleni roky"
# TEST_QUERY = "Kolik hektarů zahrad (gardens) se nachází v Praze?"
# TEST_QUERY = "Kolik je osobnich automobilu ve Varsave?"
# TEST_QUERY = "How many passenger cars there in Warsaw (total, not per 1000 inhabitants)?"
# TEST_QUERY = "Kolik pracovniku ve vyzkumu je z Akademie Ved?"
# TEST_QUERY = "Japan Imports for 2023"
# TEST_QUERY = "Dej mi data poctu stavebnich povoleni bytu za posledni roky k dispozici."
# TEST_QUERY = "Give me the data on the number of building permits for apartments in recent years available."
TEST_QUERY = "Jaky je prutok reky Metuje?"

# Azure OpenAI Settings
AZURE_EMBEDDING_DEPLOYMENT = (
    "text-embedding-3-large__test1"  # Azure deployment name (3072 dimensions)
)
AZURE_LLM_DEPLOYMENT = (
    "gpt-4"  # Azure deployment name for LLM (used by MarkdownElementNodeParser)
)


# LlamaParse Settings (only needed if using llamaparse method)
LLAMAPARSE_API_KEY = os.environ.get("LLAMAPARSE_API_KEY", "")  # Read from .env file

# =====================================================================
# TUNING HYPERPARAMETERS
# =====================================================================
# Token and Chunking Settings
MAX_TOKENS = 8190  # Token limit for Azure OpenAI
MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid very small chunks
MAX_CHUNK_SIZE = 4000  # Optimized chunk size for better semantic boundaries
CHUNK_OVERLAP = 0  # Overlap for better context preservation

# Search Settings
HYBRID_SEARCH_RESULTS = 40  # Number of results from hybrid search
SEMANTIC_WEIGHT = 0.85  # Weight for semantic search (0.0-1.0)
BM25_WEIGHT = 0.15  # Weight for BM25 search (0.0-1.0)
FINAL_RESULTS_COUNT = 10  # Number of final results to return

# =====================================================================
# PATH CONFIGURATION - AUTOMATICALLY SET
# =====================================================================
# PDF_PATH = SCRIPT_DIR / PDF_FILENAME  # Full path to PDF file
# PARSED_TEXT_PATH = SCRIPT_DIR / PARSED_TEXT_FILENAME  # Full path to parsed text file

# ChromaDB storage location
CHROMA_DB_PATH = SCRIPT_DIR / "pdf_chromadb_llamaparse_v2"


# ==============================================================================
# MONITORING AND METRICS
# ==============================================================================
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
            "average_time_per_chunk": self.total_processing_time
            / max(1, self.processed_chunks),
            "success_rate": (
                self.processed_chunks
                / max(1, self.processed_chunks + self.failed_chunks)
            )
            * 100,
            "failed_records": self.failed_records,
        }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_llamaparse_instructions() -> str:
    """
    Generate comprehensive LlamaParse instructions for consistent parsing across all methods.
    Centralizes instructions to avoid duplication and ensure consistency.

    Instructions specifically request MARKDOWN TABLES (pipe-delimited) to work with
    the custom_structural_chunking function that parses markdown table syntax.

    Returns:
        Comprehensive instruction string for LlamaParse processing
    """
    return f"""
🚨 CRITICAL: OUTPUT NATURAL MARKDOWN FORMAT! 🚨
🚨 EXTRACT EVERY VALUE - NO MISSING DATA! 🚨


MOST IMPORTANT RULES:
- Output tables as MARKDOWN TABLES using pipe-delimited format: | Header1 | Header2 |
- Use standard markdown formatting for all other content
- Numbers without separators: "49621" not "49,621"
- Keep all original data intact - do not summarize or omit any values

=== TABLE PROCESSING ===

STEP 1: Identify tables and output them as MARKDOWN TABLES
- Convert all tables to markdown table format using pipes (|) and hyphens (-)
- Example format:
  | Header1 | Header2 | Header3 |
  | ------- | ------- | ------- |
  | Data1   | Data2   | Data3   |
  | Data4   | Data5   | Data6   |
- Preserve all numerical data exactly as shown
- Include all years, categories, and values present in the original table
- Do not convert to sentences - keep tabular structure in markdown

STEP 2: For complex multi-year tables
- Use markdown table format with proper headers and data rows
- Ensure all temporal data (years, months, quarters) are preserved
- Maintain hierarchical relationships between rows and columns
- Add additional header rows if needed for complex hierarchies

STEP 3: MULTI-PAGE TABLE HANDLING
- When tables span multiple pages, consolidate them into single complete markdown tables
- Identify table fragments that belong together by matching headers, column structures, and content continuity
- Merge table parts across page breaks to create unified tables
- Preserve all data from each page segment without duplication or omission
- Maintain proper markdown table structure in the consolidated result
- If table headers appear on the first page but data continues on subsequent pages, include the header only once at the beginning
- Ensure numerical sequences and data relationships are preserved across page boundaries

=== CHART/IMAGE PROCESSING ===

🚨 CRITICAL: READ ACTUAL VALUES FROM CHART AXES - DO NOT MAKE UP LINEAR PROGRESSIONS! 🚨

- Look at Y-axis scale and read where each data point actually falls
- Use natural language description (NOT markdown tables)
- Example: "At 5 years: ~35-40%, at 10 years: ~45-50%"
- ❌ FORBIDDEN: Do NOT create fake linear data like "Year 0: 0%, Year 1: 4%, Year 2: 8%"

=== TEXT PROCESSING ===

- Output normal text in standard markdown format
- Use headers (# ## ###) for section titles
- Use bullet points and numbered lists where appropriate
- Preserve all important information and context

OTHER RULES:
- SKIP all Czech text completely EVERYWHERE IN THE PDF - use ONLY English text. 
- Always place the unit right after the value: “254891 million CZK in 2015”.
- Never say 'first/second chart'; only use the chart title.
- When English labels are available anywhere (e.g., an Indicator column), always use those; ignore Czech completely.
- List all available years in ascending order; do not skip any year or value present in the table.
- For long time series (>12 years), keep all data in the table but you can add a brief summary paragraph after the table.
- Do not mention colors or visual styles.

🚨 OUTPUT EVERYTHING IN STANDARD MARKDOWN FORMAT WITH MARKDOWN TABLES! 🚨
"""


def get_document_hash(text: str) -> str:
    """Generate MD5 hash for a document text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


def normalize_czech_text(text: str) -> str:
    """Advanced Czech text normalization for better search matching."""
    if not text:
        return text

    # Convert to lowercase first
    text = text.lower()

    # Advanced Czech diacritics mapping for normalization
    czech_diacritics_map = {
        # Primary Czech diacritics
        "á": "a",
        "č": "c",
        "ď": "d",
        "é": "e",
        "ě": "e",
        "í": "i",
        "ň": "n",
        "ó": "o",
        "ř": "r",
        "š": "s",
        "ť": "t",
        "ú": "u",
        "ů": "u",
        "ý": "y",
        "ž": "z",
    }

    # Create ASCII version
    ascii_text = text
    for diacritic, ascii_char in czech_diacritics_map.items():
        ascii_text = ascii_text.replace(diacritic, ascii_char)

    # Return both versions separated by space for broader indexing
    if ascii_text != text:
        return f"{text} {ascii_text}"
    return text


# Add after the _split_long_sentence function and before normalize_czech_text


def validate_chunk_quality(chunks: List[str]) -> Dict[str, Any]:
    """
    Validate the quality of generated chunks for debugging and optimization.

    Args:
        chunks: List of text chunks to validate

    Returns:
        Dictionary with validation metrics
    """
    if not chunks:
        return {"error": "No chunks provided"}

    metrics = {
        "total_chunks": len(chunks),
        "empty_chunks": 0,
        "too_small_chunks": 0,
        "too_large_chunks": 0,
        "avg_length": 0,
        "min_length": float("inf"),
        "max_length": 0,
        "chunks_with_context": 0,
        "chunks_with_numbers": 0,
        "potential_issues": [],
    }

    total_length = 0

    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk.strip())
        total_length += chunk_len

        # Update length metrics
        metrics["min_length"] = min(metrics["min_length"], chunk_len)
        metrics["max_length"] = max(metrics["max_length"], chunk_len)

        # Check for issues
        if not chunk.strip():
            metrics["empty_chunks"] += 1
            metrics["potential_issues"].append(f"Chunk {i}: Empty chunk")
        elif chunk_len < MIN_CHUNK_SIZE:
            metrics["too_small_chunks"] += 1
            metrics["potential_issues"].append(
                f"Chunk {i}: Too small ({chunk_len} chars)"
            )
        elif chunk_len > MAX_CHUNK_SIZE * 1.5:  # Allow some flexibility
            metrics["too_large_chunks"] += 1
            metrics["potential_issues"].append(
                f"Chunk {i}: Very large ({chunk_len} chars)"
            )

        # Check for context preservation
        context_indicators = [
            "table",
            "chart",
            "graph",
            "measured",
            "recorded",
            "hectares",
            "mm",
            "2023",
            "2022",
            "2021",
        ]
        if any(indicator in chunk.lower() for indicator in context_indicators):
            metrics["chunks_with_context"] += 1

        # Check for numerical data
        import re

        if re.search(r"\d+", chunk):
            metrics["chunks_with_numbers"] += 1

    metrics["avg_length"] = total_length / len(chunks) if chunks else 0
    metrics["min_length"] = (
        metrics["min_length"] if metrics["min_length"] != float("inf") else 0
    )

    # Calculate quality score
    quality_score = 100
    if metrics["empty_chunks"] > 0:
        quality_score -= metrics["empty_chunks"] * 10
    if metrics["too_small_chunks"] > len(chunks) * 0.1:  # More than 10% too small
        quality_score -= 20
    if metrics["too_large_chunks"] > len(chunks) * 0.1:  # More than 10% too large
        quality_score -= 15

    metrics["quality_score"] = max(0, quality_score)

    return metrics


# Add after the _split_by_character_limit function and before normalize_czech_text


def debug_chunks_for_numerical_data(
    chunks: List[str], target_numbers: List[str] = None
) -> Dict[str, Any]:
    """
    Debug function to verify that numerical data is preserved in chunks.

    Args:
        chunks: List of text chunks to analyze
        target_numbers: Specific numbers to look for (optional)

    Returns:
        Dictionary with detailed analysis of numerical data preservation
    """
    if target_numbers is None:
        target_numbers = [
            "1042843",
            "356687",
            "1574854",
            "712555",
            "730947",
        ]  # Warsaw data from example

    analysis = {
        "total_chunks": len(chunks),
        "chunks_with_any_numbers": 0,
        "target_numbers_found": {},
        "chunk_details": [],
    }

    # Initialize target number tracking
    for num in target_numbers:
        analysis["target_numbers_found"][num] = []

    for i, chunk in enumerate(chunks):
        chunk_info = {
            "chunk_index": i,
            "length": len(chunk),
            "contains_numbers": bool(re.search(r"\d+", chunk)),
            "target_numbers_in_chunk": [],
            "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
        }

        # Check for any numbers
        if chunk_info["contains_numbers"]:
            analysis["chunks_with_any_numbers"] += 1

        # Check for specific target numbers
        for target_num in target_numbers:
            if target_num in chunk:
                chunk_info["target_numbers_in_chunk"].append(target_num)
                analysis["target_numbers_found"][target_num].append(i)

        analysis["chunk_details"].append(chunk_info)

    # Summary statistics
    analysis["target_numbers_summary"] = {}
    for num, chunk_indices in analysis["target_numbers_found"].items():
        analysis["target_numbers_summary"][num] = {
            "found": len(chunk_indices) > 0,
            "count": len(chunk_indices),
            "chunk_indices": chunk_indices,
        }

    return analysis


def print_numerical_debug_report(
    chunks: List[str], target_numbers: List[str] = None
) -> None:
    """Print a friendly debug report about numerical data preservation."""
    analysis = debug_chunks_for_numerical_data(chunks, target_numbers)

    print(f"\n🔍 NUMERICAL DATA PRESERVATION ANALYSIS")
    print("=" * 60)
    print(f"📊 Total chunks: {analysis['total_chunks']}")
    print(f"🔢 Chunks with any numbers: {analysis['chunks_with_any_numbers']}")
    print()

    print("🎯 TARGET NUMBERS ANALYSIS:")
    print("-" * 40)
    for num, summary in analysis["target_numbers_summary"].items():
        status = "✅ FOUND" if summary["found"] else "❌ MISSING"
        count_info = f"(in {summary['count']} chunks)" if summary["count"] > 1 else ""
        print(f"  {num}: {status} {count_info}")
        if summary["chunk_indices"]:
            print(f"    └─ Chunk indices: {summary['chunk_indices']}")

    print()
    print("📋 CHUNKS WITH TARGET NUMBERS:")
    print("-" * 40)
    for chunk_info in analysis["chunk_details"]:
        if chunk_info["target_numbers_in_chunk"]:
            print(
                f"  Chunk {chunk_info['chunk_index']} ({chunk_info['length']} chars):"
            )
            print(f"    Numbers: {', '.join(chunk_info['target_numbers_in_chunk'])}")
            print(f"    Preview: {chunk_info['preview']}")
            print()


# ==============================================================================
# FILE I/O FUNCTIONS
# ==============================================================================
def load_parsed_text_from_file(file_path: str) -> str:
    """
    Load parsed text from a file.

    Args:
        file_path: Path to the text file

    Returns:
        The text content from the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print__chromadb_debug(f"💾 Successfully loaded parsed text from: {file_path}")
        print(f"✅ Loaded parsed text from: {file_path}")
        return text
    except FileNotFoundError:
        print__chromadb_debug(f"❌ Parsed text file not found: {file_path}")
        return None
    except Exception as e:
        print__chromadb_debug(f"❌ Error loading parsed text: {str(e)}")
        raise


def load_parsed_text_from_file(file_path: str) -> str:
    """
    Load parsed text from a file.

    Args:
        file_path: Path to the parsed text file

    Returns:
        The parsed text content
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print__chromadb_debug(f"Successfully loaded parsed text from {file_path}")
        return text
    except FileNotFoundError:
        print__chromadb_debug(f"Parsed text file not found: {file_path}")
        raise
    except Exception as e:
        print__chromadb_debug(f"Error loading parsed text: {str(e)}")
        raise


# ==============================================================================
# PDF PROCESSING - LLAMAPARSE ONLY
# ==============================================================================
def extract_text_with_llamaparse(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using LlamaParse for superior table handling.
    Now includes comprehensive progress tracking and timeout handling.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing text and metadata for each page
    """
    print__chromadb_debug(f"📄 Opening PDF with LlamaParse: {pdf_path}")

    try:
        # Try to import llama_parse
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError(
                "LlamaParse not installed. Install with: pip install llama-parse"
            )

        # Get API key from .env file or environment
        api_key = LLAMAPARSE_API_KEY or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "LlamaParse API key not found. Set LLAMAPARSE_API_KEY in .env file or LLAMA_CLOUD_API_KEY environment variable"
            )

        # Check PDF file size for time estimation
        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        estimated_time = max(
            30, pdf_size * 5
        )  # Rough estimate: 5 seconds per MB, minimum 30 seconds

        print(f"\n🚀 Starting LlamaParse processing...")
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📊 Size: {pdf_size:.1f} MB")
        print(f"⏱️  Estimated time: {estimated_time:.0f} seconds")
        print(
            f"🌐 API Status: LlamaParse is experiencing performance issues - this may take longer than usual"
        )
        print(
            f"💡 Tip: You can monitor progress at https://cloud.llamaindex.ai/parse (History tab)"
        )

        # Initialize LlamaParse with the newer parameter system
        # Note: There are currently known issues with these parameters (GitHub issue #620)
        # If the new parameters don't work, fallback to parsing_instruction may be needed

        comprehensive_instructions = get_llamaparse_instructions()

        try:
            # Try using the newer parameter system first
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                system_prompt="🚨 CRITICAL: When you see charts/graphs, READ THE ACTUAL VALUES from the Y-axis scale where data points fall. DO NOT create linear progressions (0,2,4,6...) or fake data. Look at the visual and extract real values.",
                user_prompt=comprehensive_instructions,
                page_separator="\n\n--- Page {page_number} ---\n\n",
                verbose=True,
                parse_mode="parse_page_with_lvm",  # Use vision model for accurate chart reading
                vendor_multimodal_model_name="anthropic-sonnet-4.0",
                continuous_mode=True,  # Enable multi-page table merging (beta feature)
                preserve_layout_alignment_across_pages=True,  # Keep text alignment across pages
            )
            print__chromadb_debug(
                "🔧 Using newer LlamaParse parameter system with multi-page table support (parse_document_with_llm + continuous_mode)"
            )

        except TypeError as e:
            print__chromadb_debug(
                f"⚠️ Newer parameters not available ({e}), falling back to parsing_instruction"
            )
            # Fallback to the older parameter if new ones aren't available
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                parsing_instruction=comprehensive_instructions,
                page_separator="\n\n--- Page {page_number} ---\n\n",
                verbose=True,
                parse_mode="parse_page_with_layout_agent",
            )

        # Add progress tracking with visual indicators
        print(f"\n⏳ Sending document to LlamaParse API...")
        start_time = time.time()

        # Create a progress indicator function
        def show_progress_update(elapsed_time):
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            if elapsed_time < 30:
                status = "🟡 Initializing..."
            elif elapsed_time < 120:
                status = "🟠 Processing pages..."
            elif elapsed_time < 300:
                status = "🔴 Complex document - please wait..."
            else:
                status = "🔴 This is taking longer than expected - API may be slow"

            print(
                f"\r⏱️  {status} Elapsed: {minutes:02d}:{seconds:02d}",
                end="",
                flush=True,
            )

        # Start periodic progress updates
        import signal
        import threading

        def progress_updater():
            while not getattr(progress_updater, "stop", False):
                elapsed = time.time() - start_time
                show_progress_update(elapsed)
                time.sleep(2)  # Update every 2 seconds

        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()

        try:
            # Parse the document with timeout handling
            print(f"\n📤 Document uploaded to LlamaParse. Processing...")
            print(
                f"🔍 You can monitor detailed progress at: https://cloud.llamaindex.ai/parse"
            )

            documents = parser.load_data(pdf_path)

            # Stop progress updates
            progress_updater.stop = True
            elapsed_time = time.time() - start_time
            print(
                f"\r✅ LlamaParse completed successfully! Duration: {elapsed_time:.1f} seconds"
            )

        except Exception as parse_error:
            # Stop progress updates
            progress_updater.stop = True
            elapsed_time = time.time() - start_time
            print(f"\r❌ LlamaParse failed after {elapsed_time:.1f} seconds")

            # Provide detailed error information
            error_msg = str(parse_error)
            print(f"\n🚨 LlamaParse Error Details:")
            print(f"   Error: {error_msg}")
            print(f"   Duration: {elapsed_time:.1f} seconds")

            if "timeout" in error_msg.lower():
                print(f"   💡 This appears to be a timeout. Try:")
                print(f"      - Reducing PDF size or splitting into smaller files")
                print(f"      - Trying again later (API performance varies)")
                print(f"      - Using Fast mode instead of Premium/Balanced")
            elif "rate limit" in error_msg.lower():
                print(f"   💡 Rate limit hit. Please wait and try again.")
            elif "api key" in error_msg.lower():
                print(f"   💡 Check your LLAMAPARSE_API_KEY in .env file")
            else:
                print(f"   💡 Check API status at: https://status.llamaindex.ai/")
                print(f"   💡 Recent issues: Many users reporting 10-100x slowdowns")
                print(f"   💡 Monitor your job at: https://cloud.llamaindex.ai/parse")

            raise parse_error

        pages_data = []

        print(f"\n📊 Processing {len(documents)} document sections...")

        for doc_idx, document in enumerate(documents):
            text = document.text if hasattr(document, "text") else str(document)

            if not text.strip():
                print__chromadb_debug(f"⚠️ Document {doc_idx + 1} contains no text")
                continue

            # Get page metadata
            page_info = {
                "text": text,
                "page_number": doc_idx
                + 1,  # LlamaParse may not preserve exact page numbers
                "char_count": len(text),
                "word_count": len(text.split()),
                "source_file": os.path.basename(pdf_path),
                "parsing_method": "llamaparse",
            }

            pages_data.append(page_info)
            print(
                f"   ✅ Section {doc_idx + 1}: {len(text):,} characters, {len(text.split()):,} words"
            )
            print__chromadb_debug(
                f"📄 Document {doc_idx + 1}: {len(text)} characters, {len(text.split())} words (LlamaParse)"
            )

        # Final summary
        total_chars = sum(p["char_count"] for p in pages_data)
        total_words = sum(p["word_count"] for p in pages_data)

        print(f"\n🎉 LlamaParse Processing Complete!")
        print(f"   📄 Sections: {len(pages_data)}")
        print(f"   📝 Total characters: {total_chars:,}")
        print(f"   🔤 Total words: {total_words:,}")
        print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"   📊 Processing speed: {total_chars/elapsed_time:.0f} chars/sec")

        print__chromadb_debug(
            f"🏗️ Successfully extracted text from {len(pages_data)} documents using LlamaParse"
        )
        print__chromadb_debug(
            f"Successfully extracted text from {len(pages_data)} documents using LlamaParse"
        )
        return pages_data

    except Exception as e:
        print__chromadb_debug(f"Error extracting text with LlamaParse: {str(e)}")
        print(f"\n❌ LlamaParse failed completely. Error: {str(e)}")
        # No fallback - just raise the error
        raise


def extract_text_with_llamaparse_async_monitoring(
    pdf_path: str,
) -> List[Dict[str, Any]]:
    """
    Alternative LlamaParse extraction with enhanced progress monitoring using the raw API.
    This version can check job status and provide better feedback on parsing progress.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing text and metadata for each page
    """
    import json

    import requests

    print__chromadb_debug(f"Opening PDF with LlamaParse (async monitoring): {pdf_path}")

    try:
        # Get API key
        api_key = LLAMAPARSE_API_KEY or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LlamaParse API key not found")

        # Check PDF file size for time estimation
        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        estimated_time = max(
            30, pdf_size * 5
        )  # Rough estimate: 5 seconds per MB, minimum 30 seconds

        print(f"\n🚀 Starting LlamaParse processing (Enhanced Monitoring)...")
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📊 Size: {pdf_size:.1f} MB")
        print(f"⏱️  Estimated time: {estimated_time:.0f} seconds")

        # Step 1: Upload file and start parsing job
        print(f"\n📤 Uploading file to LlamaParse...")

        headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}

        # Prepare comprehensive parsing instructions
        parsing_instructions = get_llamaparse_instructions()

        # Prepare form data for upload
        with open(pdf_path, "rb") as file:
            files = {"file": (os.path.basename(pdf_path), file, "application/pdf")}

            data = {
                "parsing_instruction": parsing_instructions,
                "result_type": "markdown",
                "page_separator": "\n\n--- Page {page_number} ---\n\n",
                "verbose": "true",
            }

            response = requests.post(
                "https://api.cloud.llamaindex.ai/api/v1/parsing/upload",
                headers=headers,
                files=files,
                data=data,
                timeout=60,  # 60 second timeout for upload
            )

        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")

        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            raise Exception(f"No job ID returned from upload: {job_data}")

        print(f"✅ File uploaded successfully. Job ID: {job_id}")
        print(f"🔍 Monitor at: https://cloud.llamaindex.ai/parse")

        # Step 2: Monitor job progress
        print(f"\n⏳ Monitoring parsing progress...")
        start_time = time.time()
        last_status = None
        status_check_interval = 5  # Check every 5 seconds

        while True:
            elapsed_time = time.time() - start_time

            try:
                # Check job status
                status_response = requests.get(
                    f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}",
                    headers=headers,
                    timeout=10,
                )

                if status_response.status_code != 200:
                    print(f"\n⚠️  Status check failed: {status_response.status_code}")
                    time.sleep(status_check_interval)
                    continue

                job_status = status_response.json()
                current_status = job_status.get("status", "unknown")

                # Update progress display
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)

                if current_status != last_status:
                    print(f"\n📊 Status changed: {current_status}")
                    last_status = current_status

                # Status-specific progress messages
                if current_status == "PENDING":
                    status_msg = "🟡 Queued - waiting to start processing..."
                elif current_status == "PROCESSING":
                    status_msg = "🟠 Processing document pages..."
                elif current_status == "SUCCESS":
                    print(
                        f"\n✅ Parsing completed successfully! Duration: {elapsed_time:.1f} seconds"
                    )
                    break
                elif current_status == "ERROR":
                    error_msg = job_status.get("error", "Unknown error")
                    raise Exception(f"Parsing job failed: {error_msg}")
                else:
                    status_msg = f"🔵 Status: {current_status}"

                print(
                    f"\r⏱️  {status_msg} Elapsed: {minutes:02d}:{seconds:02d}",
                    end="",
                    flush=True,
                )

                # Timeout check
                if elapsed_time > 600:  # 10 minute timeout
                    raise Exception(f"Job timeout after {elapsed_time:.1f} seconds")

                time.sleep(status_check_interval)

            except requests.RequestException as e:
                print(f"\n⚠️  Network error checking status: {e}")
                time.sleep(status_check_interval)
                continue

        # Step 3: Retrieve results
        print(f"\n📥 Retrieving parsed results...")

        result_response = requests.get(
            f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/markdown",
            headers=headers,
            timeout=30,
        )

        if result_response.status_code != 200:
            raise Exception(
                f"Failed to retrieve results: {result_response.status_code} - {result_response.text}"
            )

        result_data = result_response.json()

        # Process results into our expected format
        pages_data = []
        combined_text_parts = []

        # Extract text from result data
        if "markdown" in result_data:
            text_content = result_data["markdown"]
        elif "text" in result_data:
            text_content = result_data["text"]
        else:
            # Try to extract from pages if available
            pages = result_data.get("pages", [])
            if pages:
                text_content = "\n\n--- Page {page_number} ---\n\n".join(
                    [page.get("md", page.get("text", "")) for page in pages]
                )
            else:
                raise Exception(
                    f"No usable content found in results: {list(result_data.keys())}"
                )

        print(f"📊 Processing retrieved content...")

        # Split content by pages if available
        if "\n\n--- Page {page_number} ---\n\n" in text_content:
            page_texts = text_content.split("\n\n--- Page {page_number} ---\n\n")
        else:
            page_texts = [text_content]

        for page_num, page_text in enumerate(page_texts, 1):
            if page_text.strip():
                # Keep original text with separators intact
                cleaned_text = page_text.strip()

                page_info = {
                    "text": cleaned_text,
                    "page_number": page_num,
                    "char_count": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                    "source_file": os.path.basename(pdf_path),
                    "parsing_method": "llamaparse_async",
                }

                pages_data.append(page_info)
                combined_text_parts.append(cleaned_text)
                print(
                    f"   ✅ Page {page_num}: {len(cleaned_text):,} characters, {len(cleaned_text.split()):,} words"
                )

        # Save combined text to file
        if combined_text_parts:
            combined_text = "\n\n--- Page {page_number} ---\n\n".join(
                combined_text_parts
            )
            # Generate parsed text filename dynamically
            parsed_text_filename = f"{os.path.basename(pdf_path)}_llamaparse_parsed.txt"
            parsed_text_path = SCRIPT_DIR / parsed_text_filename
            save_parsed_text_to_file(combined_text, str(parsed_text_path))

        # Final summary
        total_chars = sum(p["char_count"] for p in pages_data)
        total_words = sum(p["word_count"] for p in pages_data)
        final_elapsed = time.time() - start_time

        print(f"\n🎉 LlamaParse Async Processing Complete!")
        print(f"   📄 Pages: {len(pages_data)}")
        print(f"   📝 Total characters: {total_chars:,}")
        print(f"   🔤 Total words: {total_words:,}")
        print(f"   ⏱️  Total time: {final_elapsed:.1f} seconds")
        print(f"   📊 Processing speed: {total_chars/final_elapsed:.0f} chars/sec")
        print(f"   🆔 Job ID: {job_id}")

        print__chromadb_debug(
            f"Successfully extracted text from {len(pages_data)} pages using LlamaParse async"
        )
        return pages_data

    except Exception as e:
        print__chromadb_debug(f"Error with LlamaParse async monitoring: {str(e)}")
        print(f"\n❌ LlamaParse async monitoring failed. Error: {str(e)}")
        # No fallback - just raise the error
        raise


def process_parsed_text_to_chunks(
    pages_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Process parsed text (from LlamaParse or other parsers) into chunks suitable for embedding.
    Uses MarkdownElementNodeParser for intelligent chunking that preserves table structure.

    Args:
        pages_data: List of page/section data from parsed text

    Returns:
        List of chunk dictionaries with text and metadata
    """
    from llama_index.core import Document as LlamaDocument

    all_chunks = []
    chunk_id = 0

    print__chromadb_debug(
        f"Processing {len(pages_data)} sections/pages for MarkdownElementNodeParser chunking"
    )

    # Combine all pages into a single markdown document for better chunking
    combined_markdown = ""
    source_files = set()
    page_numbers = []

    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data.get("page_number", 1)
        source_file = page_data.get("source_file", "unknown")

        # Add page separator if not the first page
        if combined_markdown:
            combined_markdown += f"\n\n--- Page {page_num} ---\n\n"

        combined_markdown += text
        source_files.add(source_file)
        page_numbers.append(page_num)

    print__chromadb_debug(
        f"Combined {len(pages_data)} pages into single markdown document ({len(combined_markdown)} characters)"
    )

    # Use LlamaIndex's built-in MarkdownElementNodeParser for intelligent chunking
    # This parser keeps tables with their headers and creates semantic chunks
    print__chromadb_debug(
        "Using MarkdownElementNodeParser for intelligent semantic chunking"
    )

    # Configure Azure OpenAI LLM for MarkdownElementNodeParser
    # This prevents the parser from defaulting to OpenAI and causing API key errors
    if get_azure_llm_gpt_4o is not None:
        llm = get_azure_llm_gpt_4o(temperature=0.0)
    else:
        # Fallback if import failed
        from llama_index.llms.azure_openai import AzureOpenAI

        llm = AzureOpenAI(
            deployment_name=AZURE_LLM_DEPLOYMENT,
            model="gpt-4",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    Settings.llm = llm

    # Create a single Document from the combined markdown
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import MarkdownElementNodeParser

    document = LlamaDocument(text=combined_markdown)

    # Initialize MarkdownElementNodeParser - it intelligently preserves table structure
    parser = MarkdownElementNodeParser(
        num_workers=1,  # Single worker for consistent chunking
        show_progress=False,  # Don't show progress bar
    )

    # Parse the document into nodes
    nodes = parser.get_nodes_from_documents([document])

    print__chromadb_debug(
        f"MarkdownElementNodeParser created {len(nodes)} intelligent nodes"
    )

    # Convert nodes to our chunk format
    all_chunks = []
    chunk_id = 0

    for node in nodes:
        # Extract text content from LlamaIndex node
        text_content = node.get_content() if hasattr(node, "get_content") else node.text

        # Skip empty nodes
        if not text_content or not text_content.strip():
            continue

        # Calculate token count
        token_count = num_tokens_from_string(text_content)

        # Create chunk data structure
        chunk_data = {
            "id": chunk_id,
            "text": text_content,
            "page_number": (
                page_numbers[0] if page_numbers else 1
            ),  # Use first page as reference
            "chunk_index": chunk_id,
            "token_chunk_index": 0,
            "total_page_chunks": len(nodes),
            "total_token_chunks": 1,
            "char_count": len(text_content),
            "token_count": token_count,
            "source_file": list(source_files)[0] if source_files else "unknown",
            "parsing_method": "llamaparse",
            "doc_hash": get_document_hash(text_content),
        }
        all_chunks.append(chunk_data)
        chunk_id += 1

    print__chromadb_debug(f"Converted {len(all_chunks)} nodes to chunks")

    # Log chunking statistics
    if all_chunks:
        token_counts = [chunk["token_count"] for chunk in all_chunks]
        char_counts = [chunk["char_count"] for chunk in all_chunks]

        print__chromadb_debug(f"Chunk statistics:")
        print__chromadb_debug(
            f"  - Average tokens: {sum(token_counts)/len(token_counts):.1f}"
        )
        print__chromadb_debug(
            f"  - Average characters: {sum(char_counts)/len(char_counts):.1f}"
        )
        print__chromadb_debug(
            f"  - Token range: {min(token_counts)} - {max(token_counts)}"
        )
        print__chromadb_debug(
            f"  - Character range: {min(char_counts)} - {max(char_counts)}"
        )

        # Validate chunk quality
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        quality_metrics = validate_chunk_quality(chunk_texts)

        print__chromadb_debug(f"Chunk quality metrics:")
        print__chromadb_debug(
            f"  - Quality score: {quality_metrics.get('quality_score', 0)}/100"
        )
        print__chromadb_debug(
            f"  - Chunks with context: {quality_metrics.get('chunks_with_context', 0)}/{len(all_chunks)}"
        )
        print__chromadb_debug(
            f"  - Chunks with numbers: {quality_metrics.get('chunks_with_numbers', 0)}/{len(all_chunks)}"
        )

        # Add numerical data preservation debugging
        print("\n🔍 DEBUGGING NUMERICAL DATA PRESERVATION:")
        print_numerical_debug_report(chunk_texts)

        if quality_metrics.get("potential_issues"):
            print__chromadb_debug(
                f"  - Potential issues found: {len(quality_metrics['potential_issues'])}"
            )
            for issue in quality_metrics["potential_issues"][:5]:  # Show first 5 issues
                print__chromadb_debug(f"    * {issue}")
            if len(quality_metrics["potential_issues"]) > 5:
                print__chromadb_debug(
                    f"    * ... and {len(quality_metrics['potential_issues']) - 5} more issues"
                )

    return all_chunks


# Keep the old function name as an alias for backward compatibility
def process_pdf_pages_to_chunks(
    pages_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use process_parsed_text_to_chunks() instead.
    This function is kept for backward compatibility.
    """
    print__chromadb_debug(
        "Warning: process_pdf_pages_to_chunks() is deprecated. Use process_parsed_text_to_chunks() instead."
    )
    return process_parsed_text_to_chunks(pages_data)


def save_parsed_text_to_file(text: str, file_path: str) -> None:
    """
    Save parsed text to a file.

    Args:
        text: The parsed text content
        file_path: Path where to save the text file
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print__chromadb_debug(f"💾 Successfully saved parsed text to: {file_path}")
        print(f"✅ Parsed text saved to: {file_path}")
        print(f"📊 Content length: {len(text):,} characters")

    except Exception as e:
        print__chromadb_debug(f"❌ Error saving parsed text: {str(e)}")
        raise


def create_documents_from_text(
    text: str, source_file: str, parsing_method: str
) -> List[Dict[str, Any]]:
    """
    Create document-like structure from parsed text for chunking.
    Simplified version for MarkdownElementNodeParser that doesn't use CONTENT_SEPARATORS.

    Args:
        text: The parsed text content (from .txt file)
        source_file: Original source filename
        parsing_method: Method used for parsing (llamaparse, pymupdf, etc.)

    Returns:
        List of document-like dictionaries representing sections/pages
    """
    print__chromadb_debug(
        f"🏗️ Creating document structure from parsed text ({len(text)} characters)"
    )
    print__chromadb_debug(f"📄 Source: {source_file}, Method: {parsing_method}")

    pages = []

    # Split by page separator (LlamaParse default)
    page_separator = "\n\n--- Page {page_number} ---\n\n"
    if page_separator in text:
        print__chromadb_debug(f"📄 Found LlamaParse page separators")
        page_texts = text.split(page_separator)
    else:
        # Fallback separators for other methods
        if "\n---\n" in text:
            print__chromadb_debug(f"📄 Found standard page separators (---)")
            page_texts = text.split("\n---\n")
        elif "\n=================\n" in text:
            print__chromadb_debug(
                f"📄 Found extended page separators (=================)"
            )
            page_texts = text.split("\n=================\n")
        else:
            # If no clear separators, treat as one large document
            print__chromadb_debug(
                f"� No page separators found, treating as single document"
            )
            page_texts = [text]

    print__chromadb_debug(f"✂️ Split text into {len(page_texts)} sections")

    for page_num, page_text in enumerate(page_texts, 1):
        if page_text.strip():  # Only add non-empty pages
            # Keep original text intact for MarkdownElementNodeParser
            cleaned_text = page_text.strip()

            page_data = {
                "text": cleaned_text,
                "page_number": page_num,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "source_file": source_file,
                "parsing_method": parsing_method,
            }
            pages.append(page_data)

            print__chromadb_debug(
                f"📄 Section {page_num}: {len(cleaned_text)} chars, {len(cleaned_text.split())} words"
            )

    print__chromadb_debug(f"✅ Created {len(pages)} document sections from parsed text")

    # Summary statistics
    total_chars = sum(p["char_count"] for p in pages)
    print__chromadb_debug(f"📊 Document summary:")
    print__chromadb_debug(f"📊   - Total characters: {total_chars}")
    print__chromadb_debug(
        f"📊   - Average section size: {total_chars/len(pages):.0f} characters"
    )

    return pages


def process_pdf_to_chromadb(
    pdf_path: str,
    collection_name: str = COLLECTION_NAME,
    deployment: str = AZURE_EMBEDDING_DEPLOYMENT,
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
        print__chromadb_debug(f"Processing PDF: {pdf_path}")
        print__chromadb_debug(f"Using parsing method: {PDF_PARSING_METHOD}")

        if PDF_PARSING_METHOD == "llamaparse":
            if LLAMAPARSE_ENHANCED_MONITORING:
                pages_data = extract_text_with_llamaparse_async_monitoring(pdf_path)
            else:
                pages_data = extract_text_with_llamaparse(pdf_path)
        else:
            raise ValueError(
                f"Only 'llamaparse' parsing method is supported. Current setting: {PDF_PARSING_METHOD}"
            )

        metrics.total_pages = len(pages_data)

        if not pages_data:
            raise ValueError("No text found in PDF")

        # Process pages into chunks
        chunks_data = process_parsed_text_to_chunks(pages_data)
        print__chromadb_debug(f"Created {len(chunks_data)} chunks for processing")

        # Initialize ChromaDB with cloud/local support
        from metadata.chromadb_client_factory import get_chromadb_client

        client = get_chromadb_client(
            local_path=CHROMA_DB_PATH, collection_name=collection_name
        )
        try:
            collection = client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            print__chromadb_debug(f"Created new ChromaDB collection: {collection_name}")
        except Exception:
            collection = client.get_collection(name=collection_name)
            print__chromadb_debug(
                f"Using existing ChromaDB collection: {collection_name}"
            )

        # Check for existing documents
        existing = collection.get(include=["metadatas"], limit=10000)
        existing_hashes = set()
        if existing and "metadatas" in existing and existing["metadatas"]:
            for metadata in existing["metadatas"]:
                if isinstance(metadata, dict) and metadata is not None:
                    doc_hash = metadata.get("doc_hash")
                    if doc_hash:
                        existing_hashes.add(doc_hash)

        print__chromadb_debug(
            f"Found {len(existing_hashes)} existing documents in ChromaDB"
        )

        # Filter out existing chunks
        new_chunks = [
            chunk for chunk in chunks_data if chunk["doc_hash"] not in existing_hashes
        ]
        print__chromadb_debug(f"Processing {len(new_chunks)} new chunks")

        if not new_chunks:
            print__chromadb_debug("No new chunks to process")
            return collection

        # Initialize embedding client
        embedding_client = get_azure_embedding_model()

        # Process chunks with progress bar
        with tqdm_module.tqdm(
            total=len(new_chunks), desc="Processing chunks", leave=True, ncols=100
        ) as pbar:

            for chunk_data in new_chunks:
                try:
                    # Generate embedding
                    response = embedding_client.embeddings.create(
                        input=[chunk_data["text"]], model=deployment
                    )
                    embedding = response.data[0].embedding

                    # Create metadata for ChromaDB
                    metadata = {
                        "page_number": chunk_data["page_number"],
                        "chunk_index": chunk_data["chunk_index"],
                        "token_chunk_index": chunk_data["token_chunk_index"],
                        "char_count": chunk_data["char_count"],
                        "token_count": chunk_data["token_count"],
                        "source_file": chunk_data["source_file"],
                        "doc_hash": chunk_data["doc_hash"],
                        "chunk_id": chunk_data["id"],
                    }

                    # Add to ChromaDB
                    collection.add(
                        documents=[chunk_data["text"]],
                        embeddings=[embedding],
                        ids=[str(uuid4())],
                        metadatas=[metadata],
                    )

                    metrics.processed_chunks += 1
                    pbar.update(1)

                except Exception as e:
                    print__chromadb_debug(
                        f"Error processing chunk {chunk_data['id']}: {str(e)}"
                    )
                    metrics.failed_chunks += 1
                    metrics.failed_records.append((chunk_data["id"], str(e)))
                    pbar.update(1)
                    continue

        # Print final statistics
        metrics.update_processing_time()
        print__chromadb_debug(f"\nProcessing completed:")
        print__chromadb_debug(f"- Total pages: {metrics.total_pages}")
        print__chromadb_debug(f"- Total chunks: {len(chunks_data)}")
        print__chromadb_debug(f"- Successfully processed: {metrics.processed_chunks}")
        print__chromadb_debug(f"- Failed: {metrics.failed_chunks}")
        print__chromadb_debug(
            f"- Processing time: {metrics.total_processing_time:.2f} seconds"
        )
        print__chromadb_debug(
            f"- Success rate: {metrics.to_dict()['success_rate']:.1f}%"
        )

        return collection

    except Exception as e:
        print__chromadb_debug(f"Error in process_pdf_to_chromadb: {str(e)}")
        raise


# ==============================================================================
# SEARCH FUNCTIONS
# ==============================================================================
def similarity_search_chromadb(
    collection,
    embedding_client,
    query: str,
    embedding_model_name: str = AZURE_EMBEDDING_DEPLOYMENT,
    k: int = 10,
):
    """Perform similarity search using ChromaDB."""
    query_embedding = (
        embedding_client.embeddings.create(input=[query], model=embedding_model_name)
        .data[0]
        .embedding
    )
    print__chromadb_debug(
        f"Generated query embedding with {len(query_embedding)} dimensions for model {embedding_model_name}"
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def hybrid_search(
    collection, query_text: str, n_results: int = HYBRID_SEARCH_RESULTS
) -> List[Dict]:
    """
    Advanced hybrid search combining semantic and BM25 approaches with RRF fusion.
    Implements best practices: BM25 pruning, RRF fusion, and proper score normalization.
    """
    print__chromadb_debug(f"Advanced hybrid search for query: '{query_text}'")

    try:
        # Normalize query
        normalized_query = normalize_czech_text(query_text)

        # Step 1: BM25 search for candidate pruning (get more candidates)
        bm25_candidates = []
        try:
            all_data = collection.get(include=["documents", "metadatas"])

            if all_data and "documents" in all_data and all_data["documents"]:
                documents = all_data["documents"]
                metadatas = all_data["metadatas"]

                processed_docs = [normalize_czech_text(doc) for doc in documents]

                if BM25Okapi:
                    tokenized_docs = [doc.split() for doc in processed_docs]
                    bm25 = BM25Okapi(tokenized_docs)

                    tokenized_query = normalized_query.split()
                    bm25_scores = bm25.get_scores(tokenized_query)

                    # Get top candidates (more than final results for better fusion)
                    top_k_bm25 = min(len(documents), n_results * 3)  # Get 3x candidates
                    top_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

                    for i, idx in enumerate(top_indices):
                        if (
                            bm25_scores[idx] > 0
                        ):  # Only include documents with BM25 score > 0
                            bm25_candidates.append(
                                {
                                    "id": f"bm25_{i}",
                                    "document": documents[idx],
                                    "metadata": (
                                        metadatas[idx] if idx < len(metadatas) else {}
                                    ),
                                    "bm25_score": float(bm25_scores[idx]),
                                    "bm25_rank": i + 1,
                                    "source": "bm25",
                                }
                            )

        except Exception as e:
            print__chromadb_debug(f"BM25 search failed: {e}")
            bm25_candidates = []

        # Step 2: Semantic search on BM25 candidates (if available) or all documents
        semantic_results = []
        try:
            embedding_client = get_azure_embedding_model()

            if bm25_candidates:
                # Search only within BM25 candidates for efficiency
                candidate_docs = [cand["document"] for cand in bm25_candidates]
                candidate_metadatas = [cand["metadata"] for cand in bm25_candidates]

                # Create temporary collection-like structure for candidates
                candidate_embeddings = []
                for doc in candidate_docs:
                    response = embedding_client.embeddings.create(
                        input=[doc], model=AZURE_EMBEDDING_DEPLOYMENT
                    )
                    candidate_embeddings.append(response.data[0].embedding)

                # Compute similarities with query
                query_embedding = (
                    embedding_client.embeddings.create(
                        input=[normalized_query], model=AZURE_EMBEDDING_DEPLOYMENT
                    )
                    .data[0]
                    .embedding
                )

                similarities = []
                for emb in candidate_embeddings:
                    # Cosine similarity
                    dot_product = np.dot(query_embedding, emb)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_emb = np.linalg.norm(emb)
                    similarity = (
                        dot_product / (norm_query * norm_emb)
                        if norm_query * norm_emb > 0
                        else 0
                    )
                    similarities.append(max(0, similarity))  # Ensure non-negative

                # Sort by similarity
                sorted_indices = np.argsort(similarities)[::-1]

                for rank, idx in enumerate(sorted_indices):
                    semantic_results.append(
                        {
                            "id": f"semantic_{rank}",
                            "document": candidate_docs[idx],
                            "metadata": candidate_metadatas[idx],
                            "semantic_score": float(similarities[idx]),
                            "semantic_rank": rank + 1,
                            "source": "semantic",
                        }
                    )
            else:
                # Fallback to full semantic search if no BM25 candidates
                semantic_raw = similarity_search_chromadb(
                    collection=collection,
                    embedding_client=embedding_client,
                    query=normalized_query,
                    k=n_results,
                )

                for i, (doc, meta, distance) in enumerate(
                    zip(
                        semantic_raw["documents"][0],
                        semantic_raw["metadatas"][0],
                        semantic_raw["distances"][0],
                    )
                ):
                    similarity_score = max(
                        0, 1 - (distance / 2)
                    )  # Convert distance to similarity
                    semantic_results.append(
                        {
                            "id": f"semantic_{i}",
                            "document": doc,
                            "metadata": meta,
                            "semantic_score": similarity_score,
                            "semantic_rank": i + 1,
                            "source": "semantic",
                        }
                    )

        except Exception as e:
            print__chromadb_debug(f"Semantic search failed: {e}")
            semantic_results = []

        # Step 3: Reciprocal Rank Fusion (RRF)
        # RRF combines rankings from different sources using reciprocal ranks
        # Formula: RRF score = Σ(1/(k + r_i)) for each retrieval method i
        # where k is a constant (typically 60), r_i is the rank from method i

        k = 60  # Standard RRF constant
        fused_results = {}

        # Process semantic results
        for result in semantic_results:
            doc_id = result["metadata"].get("chunk_id", result["document"][:50])
            if doc_id not in fused_results:
                fused_results[doc_id] = result.copy()
                fused_results[doc_id]["rrf_score"] = 0.0
                fused_results[doc_id]["methods_used"] = []

            # Add semantic RRF contribution
            semantic_rrf = 1.0 / (k + result["semantic_rank"])
            fused_results[doc_id]["rrf_score"] += semantic_rrf
            fused_results[doc_id]["methods_used"].append("semantic")
            fused_results[doc_id]["semantic_score"] = result["semantic_score"]
            fused_results[doc_id]["semantic_rank"] = result["semantic_rank"]

        # Process BM25 results
        for result in bm25_candidates:
            doc_id = result["metadata"].get("chunk_id", result["document"][:50])
            if doc_id not in fused_results:
                fused_results[doc_id] = result.copy()
                fused_results[doc_id]["rrf_score"] = 0.0
                fused_results[doc_id]["methods_used"] = []

            # Add BM25 RRF contribution
            bm25_rrf = 1.0 / (k + result["bm25_rank"])
            fused_results[doc_id]["rrf_score"] += bm25_rrf
            fused_results[doc_id]["methods_used"].append("bm25")
            fused_results[doc_id]["bm25_score"] = result["bm25_score"]
            fused_results[doc_id]["bm25_rank"] = result["bm25_rank"]

        # Convert to final results list
        final_results = list(fused_results.values())

        # Sort by RRF score (higher is better)
        final_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Add final ranking and normalize scores for display
        for i, result in enumerate(final_results):
            result["final_rank"] = i + 1
            result["score"] = result["rrf_score"]  # Keep RRF score as final score

            # Ensure all expected fields exist
            if "semantic_score" not in result:
                result["semantic_score"] = 0.0
            if "bm25_score" not in result:
                result["bm25_score"] = 0.0
            if "source" not in result:
                result["source"] = "hybrid"

        print__chromadb_debug(
            f"RRF fusion completed: {len(final_results)} results from {len(semantic_results)} semantic + {len(bm25_candidates)} BM25 candidates"
        )

        return final_results[:n_results]

    except Exception as e:
        print__chromadb_debug(f"Advanced hybrid search failed: {e}")
        return []


def format_table_for_reranking(text: str) -> str:
    """
    Format tabular data as YAML for better Cohere reranking performance.
    Converts markdown tables to structured YAML format that Cohere can better understand.
    """
    import re

    # Check if text contains markdown tables
    if not re.search(r"^\|.*\|.*\|", text, re.MULTILINE):
        return text  # Return original text if no tables found

    try:
        # Split text into sections (tables and non-table content)
        lines = text.split("\n")
        formatted_sections = []
        current_table = []
        in_table = False

        for line in lines:
            if line.strip().startswith("|") and "|" in line:
                # Table row
                if not in_table:
                    # Start new table
                    if current_table:
                        formatted_sections.append("\n".join(current_table))
                        current_table = []
                    in_table = True
                current_table.append(line)
            else:
                # Non-table content
                if in_table:
                    # End current table and format it
                    if current_table:
                        yaml_table = convert_markdown_table_to_yaml(current_table)
                        formatted_sections.append(yaml_table)
                        current_table = []
                    in_table = False
                formatted_sections.append(line)

        # Handle remaining table content
        if current_table:
            yaml_table = convert_markdown_table_to_yaml(current_table)
            formatted_sections.append(yaml_table)

        return "\n".join(formatted_sections)

    except Exception as e:
        print__chromadb_debug(f"Table formatting failed: {e}")
        return text  # Return original text on error


def convert_markdown_table_to_yaml(table_lines: List[str]) -> str:
    """
    Convert markdown table lines to YAML format for better reranking.
    """
    if not table_lines:
        return ""

    try:
        # Parse table structure
        header_line = None
        data_lines = []

        for line in table_lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("|") and not header_line:
                header_line = line
            elif line.startswith("|") and header_line:
                data_lines.append(line)

        if not header_line or not data_lines:
            return "\n".join(table_lines)  # Return original if parsing fails

        # Extract headers
        headers = [col.strip() for col in header_line.split("|")[1:-1]]

        # Extract data rows
        table_data = []
        for data_line in data_lines:
            cols = [col.strip() for col in data_line.split("|")[1:-1]]
            if len(cols) == len(headers):
                row_dict = {}
                for header, value in zip(headers, cols):
                    # Clean up header names for YAML
                    clean_header = (
                        re.sub(r"[^\w\s-]", "", header)
                        .strip()
                        .lower()
                        .replace(" ", "_")
                    )
                    row_dict[clean_header] = value
                table_data.append(row_dict)

        # Format as YAML
        yaml_output = ["table_data:"]
        for i, row in enumerate(table_data):
            yaml_output.append(f"  row_{i+1}:")
            for key, value in row.items():
                # Escape special characters and format for YAML
                clean_value = value.replace('"', '\\"').replace("\n", " ")
                yaml_output.append(f'    {key}: "{clean_value}"')

        return "\n".join(yaml_output)

    except Exception as e:
        print__chromadb_debug(f"Markdown to YAML conversion failed: {e}")
        return "\n".join(table_lines)  # Return original on error


def preprocess_query_for_reranking(query: str) -> str:
    """
    Preprocess query for better Cohere reranking performance.
    - Truncate to 256 tokens
    - Normalize Czech text
    - Remove unnecessary punctuation
    """
    try:
        # Normalize Czech text
        processed_query = normalize_czech_text(query)

        # Tokenize and truncate to 256 tokens (Cohere recommendation)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(processed_query)

        if len(tokens) > 256:
            tokens = tokens[:256]
            processed_query = encoding.decode(tokens)

        # Clean up punctuation while preserving Czech characters
        processed_query = re.sub(
            r"[^\w\sáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]", " ", processed_query
        )
        processed_query = re.sub(r"\s+", " ", processed_query).strip()

        return processed_query

    except Exception as e:
        print__chromadb_debug(f"Query preprocessing failed: {e}")
        return query  # Return original on error


def chunk_document_for_reranking(text: str, max_chunk_length: int = 1000) -> List[str]:
    """
    Chunk long documents for better reranking performance.
    Cohere performs better with smaller, focused chunks.
    """
    if len(text) <= max_chunk_length:
        return [text]

    try:
        # Split by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If still too long, split by words
        if any(len(chunk) > max_chunk_length for chunk in chunks):
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_length:
                    words = chunk.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_chunk_length:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk:
                                final_chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                else:
                    final_chunks.append(chunk)
            chunks = final_chunks

        return chunks if chunks else [text]

    except Exception as e:
        print__chromadb_debug(f"Document chunking failed: {e}")
        return [text]  # Return original on error


def cohere_rerank(query, docs, top_n):
    """Rerank documents using Cohere's rerank model with optimizations for tabular data."""
    cohere_api_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_api_key:
        print__chromadb_debug("Warning: COHERE_API_KEY not found. Skipping reranking.")
        return [
            (doc, type("obj", (object,), {"relevance_score": 0.5, "index": i})())
            for i, doc in enumerate(docs)
        ]

    co = cohere.Client(cohere_api_key)

    # Preprocess query
    processed_query = preprocess_query_for_reranking(query)
    print__chromadb_debug(f"Preprocessed query for reranking: '{processed_query}'")

    # Process documents with table formatting and chunking
    processed_docs = []
    for doc in docs:
        content = doc.page_content

        # Format tables as YAML for better understanding
        formatted_content = format_table_for_reranking(content)

        # Chunk long documents
        chunks = chunk_document_for_reranking(formatted_content)

        # Use the first chunk (or combine small chunks)
        if len(chunks) == 1:
            processed_content = chunks[0]
        else:
            # Combine first few chunks if they're small
            combined = ""
            for chunk in chunks[:3]:  # Take up to 3 chunks
                if len(combined) + len(chunk) <= 2000:  # Reasonable limit
                    combined += chunk + "\n\n"
                else:
                    break
            processed_content = combined.strip() or chunks[0]

        processed_docs.append(processed_content)

    # Prepare for Cohere
    docs_for_cohere = [{"text": content} for content in processed_docs]

    try:
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=processed_query,
            documents=docs_for_cohere,
            top_n=top_n,
        )

        reranked = []
        for res in rerank_response.results:
            doc = docs[res.index]
            reranked.append((doc, res))
        return reranked

    except Exception as e:
        print__chromadb_debug(f"Cohere reranking failed: {e}")
        # Return original docs with dummy scores
        return [
            (doc, type("obj", (object,), {"relevance_score": 0.5, "index": i})())
            for i, doc in enumerate(docs)
        ]


def search_pdf_documents(
    collection, query: str, top_k: int = FINAL_RESULTS_COUNT
) -> List[Dict[str, Any]]:
    """
    Search PDF documents using hybrid search and Cohere reranking.

    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of top results to return

    Returns:
        List of search results with metadata
    """
    print__chromadb_debug(f"Searching for: '{query}' (returning top {top_k} results)")

    try:
        # Step 1: Hybrid search
        hybrid_results = hybrid_search(
            collection, query, n_results=HYBRID_SEARCH_RESULTS
        )

        if not hybrid_results:
            print__chromadb_debug("No results from hybrid search")
            return []

        # Step 2: Convert to Document objects for reranking
        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(page_content=result["document"], metadata=result["metadata"])
            hybrid_docs.append(doc)

        # Step 3: Cohere reranking
        reranked = cohere_rerank(
            query, hybrid_docs, top_n=min(top_k * 2, len(hybrid_docs))
        )

        # Step 4: Format final results
        final_results = []
        for i, (doc, res) in enumerate(reranked[:top_k]):
            result = {
                "rank": i + 1,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "cohere_score": res.relevance_score,
                "page_number": doc.metadata.get("page_number", "N/A"),
                "source_file": doc.metadata.get("source_file", "N/A"),
                "char_count": doc.metadata.get("char_count", len(doc.page_content)),
            }
            final_results.append(result)

        print__chromadb_debug(f"Returning {len(final_results)} final results")
        return final_results

    except Exception as e:
        print__chromadb_debug(f"Error in search_pdf_documents: {str(e)}")
        return []


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def process_single_pdf(pdf_filename: str) -> Tuple[str, bool, str]:
    """
    Process a single PDF file through the LlamaParse pipeline.

    Args:
        pdf_filename: Name of the PDF file (should be in the same directory as this script)

    Returns:
        Tuple of (pdf_filename, success, message)
    """
    try:
        # Generate paths for this specific PDF
        pdf_path = SCRIPT_DIR / pdf_filename
        parsed_text_filename = f"{pdf_filename}_{PDF_PARSING_METHOD}_parsed.txt"
        parsed_text_path = SCRIPT_DIR / parsed_text_filename

        print(f"\n📄 Processing: {pdf_filename}")
        print(f"   📁 PDF path: {pdf_path}")
        print(f"   📝 Output text: {parsed_text_path}")

        # Check if PDF exists
        if not pdf_path.exists():
            error_msg = f"PDF file not found at {pdf_path}"
            print(f"   ❌ Error: {error_msg}")
            return pdf_filename, False, error_msg

        try:
            # Parse PDF using LlamaParse
            pages_data = extract_text_with_llamaparse(str(pdf_path))

            # Combine text and save to file
            if pages_data:
                combined_text_parts = []
                for page_data in pages_data:
                    combined_text_parts.append(page_data["text"])

                combined_text = "\n\n--- Page {page_number} ---\n\n".join(
                    combined_text_parts
                )
                save_parsed_text_to_file(combined_text, str(parsed_text_path))

                success_msg = f"Successfully parsed {len(pages_data)} pages"
                print(f"   ✅ {success_msg}")
                return pdf_filename, True, success_msg
            else:
                error_msg = "No pages extracted from PDF"
                print(f"   ❌ Error: {error_msg}")
                return pdf_filename, False, error_msg

        except Exception as e:
            error_msg = f"Error during PDF parsing: {str(e)}"
            print(f"   ❌ Error: {error_msg}")
            return pdf_filename, False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error processing {pdf_filename}: {str(e)}"
        print(f"   ❌ Error: {error_msg}")
        return pdf_filename, False, error_msg


def main():
    """Main function with 3 independent operations: Parse, Chunk & Store, Test."""

    # =====================================================================
    # CONFIGURATION DISPLAY
    # =====================================================================
    print("🔍 PDF to ChromaDB Processing and Search")
    print("=" * 50)
    print(f"📁 Script location: {SCRIPT_DIR}")
    print(f"📄 PDF files to process: {len(PDF_FILENAMES)} files")
    for i, pdf_name in enumerate(PDF_FILENAMES, 1):
        print(f"   {i}. {pdf_name}")
    print(f"🔬 Parsing method: {PDF_PARSING_METHOD}")
    if PDF_PARSING_METHOD == "llamaparse":
        monitoring_type = (
            "Enhanced (API monitoring)"
            if LLAMAPARSE_ENHANCED_MONITORING
            else "Standard (thread monitoring)"
        )
        print(f"📊 LlamaParse monitoring: {monitoring_type}")
    print(f"🗄️  ChromaDB location: {CHROMA_DB_PATH}")
    print()
    print("🔧 Operations to perform:")
    print(
        f"   1. Parse with LlamaParse: {'✅ Yes' if PARSE_WITH_LLAMAPARSE else '❌ No'}"
    )
    print(f"   2. Chunk and Store:      {'✅ Yes' if CHUNK_AND_STORE else '❌ No'}")
    print(f"   3. Testing/Search:       {'✅ Yes' if DO_TESTING else '❌ No'}")
    print("=" * 50)

    # =====================================================================
    # OPERATION 1: PARSE WITH LLAMAPARSE (PARALLEL PROCESSING)
    # =====================================================================
    if PARSE_WITH_LLAMAPARSE:
        print(
            f"\n📄 OPERATION 1: Parsing {len(PDF_FILENAMES)} PDFs with {PDF_PARSING_METHOD} (PARALLEL)"
        )
        print("=" * 70)

        # Validate that all PDF files exist before starting
        missing_files = []
        for pdf_filename in PDF_FILENAMES:
            pdf_path = SCRIPT_DIR / pdf_filename
            if not pdf_path.exists():
                missing_files.append(pdf_filename)

        if missing_files:
            print(f"❌ Error: The following PDF files were not found:")
            for missing_file in missing_files:
                print(f"   - {missing_file}")
            print(f"Please make sure all PDF files are in the directory: {SCRIPT_DIR}")
            return

        print(
            f"✅ All {len(PDF_FILENAMES)} PDF files found. Starting parallel processing..."
        )

        # Process PDFs in parallel using ThreadPoolExecutor
        max_workers = min(
            len(PDF_FILENAMES), 4
        )  # Limit to 4 concurrent LlamaParse requests
        print(f"🚀 Using {max_workers} parallel workers")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_filename): pdf_filename
                for pdf_filename in PDF_FILENAMES
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_pdf):
                pdf_filename = future_to_pdf[future]
                try:
                    pdf_name, success, message = future.result()
                    results.append((pdf_name, success, message))
                except Exception as e:
                    error_msg = f"Exception in processing: {str(e)}"
                    results.append((pdf_filename, False, error_msg))
                    print(f"❌ Exception processing {pdf_filename}: {str(e)}")

        # Print summary of all results
        print(f"\n🎯 PARALLEL PROCESSING SUMMARY")
        print("=" * 60)
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        print(f"✅ Successfully processed: {len(successful)}/{len(PDF_FILENAMES)} PDFs")
        for pdf_name, success, message in successful:
            print(f"   ✅ {pdf_name}: {message}")

        if failed:
            print(f"\n❌ Failed to process: {len(failed)}/{len(PDF_FILENAMES)} PDFs")
            for pdf_name, success, message in failed:
                print(f"   ❌ {pdf_name}: {message}")

        if not successful:
            print(
                f"❌ No PDFs were successfully processed. Cannot continue to next operations."
            )
            return

    else:
        print(
            f"\n⏭️  OPERATION 1: Skipping PDF parsing (using existing parsed text files)"
        )

    # =====================================================================
    # OPERATION 2: CHUNK AND STORE TO CHROMADB
    # =====================================================================
    if CHUNK_AND_STORE:
        print(f"\n🗄️  OPERATION 2: Chunking and storing to ChromaDB")

        try:
            all_pages_data = []

            # Load text from files for all PDFs
            for pdf_filename in PDF_FILENAMES:
                parsed_text_filename = f"{pdf_filename}_{PDF_PARSING_METHOD}_parsed.txt"
                parsed_text_path = SCRIPT_DIR / parsed_text_filename

                if not parsed_text_path.exists():
                    print(
                        f"⚠️  Warning: Parsed text file not found for {pdf_filename}: {parsed_text_path}"
                    )
                    print(
                        f"💡 Skipping {pdf_filename} - run with PARSE_WITH_LLAMAPARSE = 1 first"
                    )
                    continue

                # Load parsed text and create document structure
                parsed_text = load_parsed_text_from_file(str(parsed_text_path))
                pages_data = create_documents_from_text(
                    parsed_text, pdf_filename, PDF_PARSING_METHOD
                )
                all_pages_data.extend(pages_data)
                print(f"📊 Loaded {len(pages_data)} pages from {pdf_filename}")

            if not all_pages_data:
                print(
                    f"❌ No parsed text files found. Please run with PARSE_WITH_LLAMAPARSE = 1 first."
                )
                return

            print(f"📊 Total pages from all PDFs: {len(all_pages_data)}")

            # Process pages into chunks
            chunks_data = process_parsed_text_to_chunks(all_pages_data)
            print__chromadb_debug(f"Created {len(chunks_data)} chunks for processing")

            # Initialize ChromaDB with local PersistentClient (not using cloud)
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            print__chromadb_debug(f"📂 Using local ChromaDB at: {CHROMA_DB_PATH}")

            try:
                collection = client.create_collection(
                    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
                print__chromadb_debug(
                    f"✅ Created new ChromaDB collection: {COLLECTION_NAME}"
                )
            except Exception:
                collection = client.get_collection(name=COLLECTION_NAME)
                print__chromadb_debug(
                    f"📂 Using existing ChromaDB collection: {COLLECTION_NAME}"
                )

            # Check for existing documents (use count() to get total, then paginate if needed)
            try:
                total_count = collection.count()
                print__chromadb_debug(
                    f"📊 Collection has {total_count} total documents"
                )

                existing_hashes = set()
                # ChromaDB has a limit of 300 items per GET request, so we need to paginate
                BATCH_SIZE = 300
                offset = 0

                while offset < total_count:
                    existing = collection.get(
                        include=["metadatas"], limit=BATCH_SIZE, offset=offset
                    )

                    if existing and "metadatas" in existing and existing["metadatas"]:
                        for metadata in existing["metadatas"]:
                            if isinstance(metadata, dict) and metadata is not None:
                                doc_hash = metadata.get("doc_hash")
                                if doc_hash:
                                    existing_hashes.add(doc_hash)

                    offset += BATCH_SIZE

            except Exception as e:
                print__chromadb_debug(
                    f"⚠️ Warning: Could not retrieve existing documents: {e}"
                )
                existing_hashes = set()

            print__chromadb_debug(
                f"Found {len(existing_hashes)} existing documents in ChromaDB"
            )

            # Filter out existing chunks
            new_chunks = [
                chunk
                for chunk in chunks_data
                if chunk["doc_hash"] not in existing_hashes
            ]
            print__chromadb_debug(f"Processing {len(new_chunks)} new chunks")

            if new_chunks:
                # Initialize embedding client
                if get_azure_embedding_model is None:
                    raise ImportError(
                        "Azure embedding model not available. Check your imports."
                    )

                embedding_client = get_azure_embedding_model()

                # Process chunks with progress bar
                processed_chunks = 0
                failed_chunks = 0

                with tqdm_module.tqdm(
                    total=len(new_chunks),
                    desc="Processing chunks",
                    leave=True,
                    ncols=100,
                ) as pbar:

                    for chunk_data in new_chunks:
                        try:
                            # Generate embedding
                            response = embedding_client.embeddings.create(
                                input=[chunk_data["text"]],
                                model=AZURE_EMBEDDING_DEPLOYMENT,
                            )
                            embedding = response.data[0].embedding

                            # Create metadata for ChromaDB
                            metadata = {
                                "page_number": chunk_data["page_number"],
                                "chunk_index": chunk_data["chunk_index"],
                                "token_chunk_index": chunk_data["token_chunk_index"],
                                "char_count": chunk_data["char_count"],
                                "token_count": chunk_data["token_count"],
                                "source_file": chunk_data["source_file"],
                                "doc_hash": chunk_data["doc_hash"],
                                "chunk_id": chunk_data["id"],
                            }

                            # Add to ChromaDB
                            collection.add(
                                documents=[chunk_data["text"]],
                                embeddings=[embedding],
                                ids=[str(uuid4())],
                                metadatas=[metadata],
                            )

                            processed_chunks += 1
                            pbar.update(1)

                        except Exception as e:
                            print__chromadb_debug(
                                f"Error processing chunk {chunk_data['id']}: {str(e)}"
                            )
                            failed_chunks += 1
                            pbar.update(1)
                            continue

                print(f"✅ Successfully processed and stored chunks:")
                print(f"   📊 Processed: {processed_chunks}")
                print(f"   ❌ Failed: {failed_chunks}")
                print(
                    f"   📈 Success rate: {(processed_chunks/(processed_chunks+failed_chunks)*100):.1f}%"
                )
            else:
                print(
                    f"ℹ️  No new chunks to process (all chunks already exist in ChromaDB)"
                )

        except Exception as e:
            print(f"❌ Error during chunking and storage: {str(e)}")
            import traceback

            traceback.print_exc()
            return
    else:
        print(
            f"\n⏭️  OPERATION 2: Skipping chunking and storage (using existing ChromaDB)"
        )

    # =====================================================================
    # OPERATION 3: TESTING/SEARCH
    # =====================================================================
    if DO_TESTING:
        print(f"\n🔍 OPERATION 3: Testing search functionality")

        try:
            # Load ChromaDB collection using local PersistentClient (not using cloud)
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            print__chromadb_debug(f"📂 Using local ChromaDB at: {CHROMA_DB_PATH}")
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Successfully loaded collection: {COLLECTION_NAME}")

            # Check collection contents
            collection_info = collection.get(limit=1, include=["metadatas"])
            if collection_info and collection_info["metadatas"]:
                total_count = collection.count()
                print(f"📊 Collection contains {total_count} documents")
                sample_metadata = collection_info["metadatas"][0]
                if sample_metadata:
                    print(
                        f"📄 Sample document from: {sample_metadata.get('source_file', 'Unknown file')}"
                    )
            else:
                print("⚠️  Collection appears to be empty")
                return

            print(f"\n🔍 Testing search with query: '{TEST_QUERY}'")

            # Get more results for detailed analysis
            detailed_results = search_pdf_documents(
                collection=collection,
                query=TEST_QUERY,
                top_k=10,  # Get top 10 for detailed analysis
            )

            if detailed_results:
                print(f"\n📊 Found {len(detailed_results)} relevant chunks:")
                print("=" * 100)
                print("🏆 DETAILED RESULTS WITH COHERE RERANKING SCORES")
                print("=" * 100)

                # Filter results by score threshold
                SCORE_THRESHOLD = 0.01
                filtered_results = [
                    r for r in detailed_results if r["cohere_score"] > SCORE_THRESHOLD
                ]

                print(
                    f"📋 Showing {len(filtered_results)} chunks with Cohere score > {SCORE_THRESHOLD}"
                )
                print(f"📊 Total chunks analyzed: {len(detailed_results)}")

                for result in filtered_results:
                    print(
                        f"\n🏆 Rank {result['rank']} | 🎯 COHERE RERANK SCORE: {result['cohere_score']:.6f}"
                    )
                    print(
                        f"📄 Page: {result['page_number']} | File: {result['source_file']}"
                    )
                    print(f"📏 Length: {result['char_count']} characters")
                    print("📝 Content Preview:")
                    print("-" * 80)
                    # Show first 60000 characters for detailed view
                    content = result["text"][:60000]
                    # if len(result["text"]) > 60000:
                    #     content += "..."
                    print(content)
                    print("-" * 80)

                # Summary of ALL scores (including filtered out ones)
                print(f"\n📈 SCORE ANALYSIS (ALL 10 CHUNKS - COHERE RERANKING SCORES):")
                print("=" * 60)
                all_scores = [r["cohere_score"] for r in detailed_results]
                filtered_scores = [r["cohere_score"] for r in filtered_results]

                print(f"🔝 Highest Score: {max(all_scores):.6f}")
                print(f"🔻 Lowest Score:  {min(all_scores):.6f}")
                print(f"📊 Average Score: {sum(all_scores)/len(all_scores):.6f}")
                print(f"📏 Score Range:   {max(all_scores) - min(all_scores):.6f}")
                print(
                    f"🎯 Above Threshold ({SCORE_THRESHOLD}): {len(filtered_scores)}/{len(all_scores)} chunks"
                )

                # Show all scores for reference
                print(f"\n📋 ALL COHERE SCORES (Rank: Score):")
                print("-" * 40)
                for i, score in enumerate(all_scores, 1):
                    status = "✅" if score > SCORE_THRESHOLD else "❌"
                    print(f"  Rank {i:2d}: {score:.6f} {status}")
                print("-" * 40)

                # Show top 2 results in compact format (original behavior)
                print(f"\n🎯 TOP 4 RESULTS (FINAL SELECTION):")
                print("=" * 80)
                top_n_results = search_pdf_documents(
                    collection=collection, query=TEST_QUERY, top_k=4
                )

                for result in top_n_results:
                    print(
                        f"\n🏆 Rank {result['rank']} (Cohere Score: {result['cohere_score']:.4f})"
                    )
                    print(
                        f"📄 Page: {result['page_number']} | File: {result['source_file']}"
                    )
                    print(f"📏 Length: {result['char_count']} characters")
                    print("📝 Content:")
                    print("-" * 60)
                    # Show first 300 characters for final results
                    content = result["text"][:2000]
                    if len(result["text"]) > 2000:
                        content += "..."
                    print(content)
                    print("-" * 60)
            else:
                print("❌ No results found for the query")

        except Exception as e:
            print(f"❌ Error during search testing: {str(e)}")
            print(f"💡 Make sure you have processed and stored documents first")
            print(f"   Set PARSE_WITH_LLAMAPARSE = 1 and CHUNK_AND_STORE = 1")
            import traceback

            traceback.print_exc()
            return
    else:
        print(f"\n⏭️  OPERATION 3: Skipping testing/search")

    # =====================================================================
    # COMPLETION SUMMARY
    # =====================================================================
    print(f"\n🎉 OPERATIONS COMPLETED!")
    print("=" * 50)
    operations_performed = []
    if PARSE_WITH_LLAMAPARSE:
        operations_performed.append("✅ PDF Parsing (Parallel)")
    if CHUNK_AND_STORE:
        operations_performed.append("✅ Chunking & Storage")
    if DO_TESTING:
        operations_performed.append("✅ Testing & Search")

    if operations_performed:
        print("Performed operations:")
        for op in operations_performed:
            print(f"   {op}")
    else:
        print("❌ No operations were performed (all configs set to 0)")

    print(f"\n📁 Files and locations:")
    print(f"   📄 PDF files processed: {len(PDF_FILENAMES)}")
    for pdf_name in PDF_FILENAMES:
        parsed_text_filename = f"{pdf_name}_{PDF_PARSING_METHOD}_parsed.txt"
        print(f"      - {pdf_name} → {parsed_text_filename}")
    print(f"   🗄️  ChromaDB: {CHROMA_DB_PATH}")

    print(f"\n💡 Configuration tips:")
    print(f"   - To parse new PDFs: PARSE_WITH_LLAMAPARSE = 1")
    print(f"   - To update ChromaDB: CHUNK_AND_STORE = 1")
    print(f"   - To test different queries: DO_TESTING = 1 (and modify TEST_QUERY)")
    print(f"   - For full pipeline: Set all three to 1")
    print(f"   - For testing only: Set only DO_TESTING = 1")
    print(f"   - Add more PDFs to PDF_FILENAMES list and rerun")


if __name__ == "__main__":
    main()

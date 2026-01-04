module_description = r"""PDF to ChromaDB Document Processing and Search System

This module provides a comprehensive pipeline for processing complex PDF documents (especially 
government/statistical reports with tables), extracting structured content using LlamaParse, 
applying separator-based intelligent chunking, generating embeddings, storing in ChromaDB, 
and performing advanced hybrid search with multilingual support and Cohere reranking.

Designed for Czech government statistical reports with complex table structures, but adaptable 
to any PDF document processing workflow requiring high-quality semantic search capabilities.

Architecture Overview:
=====================
Three independent, configurable operations:
1. PDF Parsing & Content Extraction (LlamaParse integration)
2. Intelligent Chunking & ChromaDB Storage (Separator-based strategy) 
3. Hybrid Search & Testing (Semantic + BM25 + Cohere reranking)

Key Features:
============
1. Advanced PDF Processing & Parsing:
   - LlamaParse integration with custom parsing instructions
   - Specialized table processing (column-by-column extraction)
   - Complex layout handling (hierarchical tables, charts, mixed content)
   - Progress monitoring with job status tracking
   - Parallel processing support for multiple PDFs
   - Automatic retry logic and timeout handling
   - Content type separation using custom markers

2. Intelligent Separator-Based Chunking:
   - Primary strategy: Extract content between separator pairs ([R]...[/R], [T]...[/T], [C]...[/C])
   - Secondary strategy: Sentence-boundary chunking with ceiling division for oversized content
   - NEVER splits mid-word or mid-sentence - only at complete sentence boundaries
   - Uses ceiling division to calculate optimal chunk count, then distributes sentences evenly
   - Token-aware processing (8190 token limit compliance)
   - Content type preservation and intelligent merging
   - Quality validation with numerical data preservation checks
   - Configurable chunk sizes (MIN: 100, MAX: 4000 chars, 0 overlap)
   - Smart handling of ungrouped content with sentence preservation

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
   - Cohere multilingual reranking (rerank-v4.0-fast)
   - Configurable result thresholds and filtering

6. Content Type Intelligence & Preservation:
   - Table content extraction with context preservation
   - Row-based data organization with time-series condensation
   - Intelligent year detection (in columns, rows, or absent)
   - Image/chart content description handling with trend analysis
   - Regular text section management with language preservation
   - Mixed content type processing and intelligent categorization
   - Context-rich chunk generation for optimal semantic search

7. Multilingual & Localization Support:
   - Czech language processing with diacritics normalization
   - Preservation of original Czech text content
   - English descriptions for tables and charts
   - Bilingual search term expansion for better matching
   - Cultural context preservation (Czech place names, terms)

Content Separator System:
=========================
Custom marker system for precise content organization:
- [T]...[/T]: Table content with detailed row descriptions
- [R]...[/R]: Row content within tables (primary chunking unit for time-series data)
- [C]...[/C]: Column content within tables (alternative chunking unit)
- [X]...[/X]: Regular text content (preserved in original language)
- [I]...[/I]: Image and chart descriptions with trend analysis
- [P]: Page separators between document sections
- [.P]: Page break markers for layout preservation

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
   - Applies comprehensive LlamaParse instructions for table preservation
   - Monitors parsing progress with real-time status updates
   - Implements timeout handling and error recovery
   - Saves parsed text with separator markers to .txt files
   - Validates content structure and separator usage
   - Provides detailed parsing statistics and quality metrics

3. Intelligent Content Chunking:
   - Applies separator-based primary chunking strategy
   - Extracts content between separator pairs using regex patterns
   - Handles oversized content with sentence-boundary chunking using ceiling division
   - Calculates optimal number of chunks needed and distributes sentences evenly
   - NEVER cuts mid-word or mid-sentence - preserves complete sentence integrity
   - Removes separator artifacts from final chunks
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
- Cohere: rerank-v4.0-fast for final ranking

Usage Examples:
==============

Full Pipeline Execution:
-------------------------
# Configure all three operations
PARSE_WITH_LLAMAPARSE = 0       # Parse PDFs with advanced table handling
CHUNK_AND_STORE = 0             # Apply intelligent chunking and store
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
TEST_QUERY = "How many teachers worked in 2023?"

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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# =====================================================================
# MAIN CONFIGURATION - MODIFY THESE SETTINGS
# =====================================================================
# Processing Mode - Three independent operations
PARSE_WITH_LLAMAPARSE = 0  # Set to 1 to parse PDF with LlamaParse and save to txt file
# Set to 0 to skip parsing (use existing txt file)

CHUNK_AND_STORE = 0  # Set to 1 to chunk text and create/update ChromaDB
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
    "96_97.pdf"
]

COLLECTION_NAME = (
    "pdf_document_collection"  # ChromaDB collection name for PDF documents
)

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
TEST_QUERY = "Give me the data on the number of building permits for apartments in recent years available."

# Azure OpenAI Settings
AZURE_EMBEDDING_DEPLOYMENT = (
    "text-embedding-3-large__test1"  # Azure deployment name (3072 dimensions)
)


# LlamaParse Settings (only needed if using llamaparse method)
LLAMAPARSE_API_KEY = os.environ.get("LLAMAPARSE_API_KEY", "")  # Read from .env file

# Content Separators - unique strings unlikely to appear in normal text
CONTENT_SEPARATORS = {
    "table_start": "[T]",
    "table_end": "[/T]",
    "column_start": "[C]",
    "column_end": "[/C]",
    "row_start": "[R]",
    "row_end": "[/R]",
    "image_start": "[I]",
    "image_end": "[/I]",
    "text_start": "[X]",
    "text_end": "[/X]",
    "page_separator": "[P]",
}

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

    Returns:
        Comprehensive instruction string for LlamaParse processing
    """
    return f"""
🚨 CRITICAL: NO MARKDOWN TABLES! ONLY DESCRIPTIVE SENTENCES! 🚨
🚨 EXTRACT EVERY VALUE - NO MISSING DATA! 🚨


MOST IMPORTANT RULES:
- NEVER create markdown tables with pipes (|...|) - ONLY complete sentences
- Numbers without separators: "49621" not "49,621"
- Use Separators: {CONTENT_SEPARATORS['row_start']}...{CONTENT_SEPARATORS['row_end']}, {CONTENT_SEPARATORS['column_start']}...{CONTENT_SEPARATORS['column_end']}, {CONTENT_SEPARATORS['image_start']}...{CONTENT_SEPARATORS['image_end']}, {CONTENT_SEPARATORS['text_start']}...{CONTENT_SEPARATORS['text_end']}

=== TABLE PROCESSING ===

STEP 1: Identify if years are in columns/rows/absent

STEP 2: Convert to sentences (NO MARKDOWN!)

A) YEARS IN COLUMNS: One sentence per row with all years
Pattern: {CONTENT_SEPARATORS['row_start']}In table '[English name]', under '[hierarchy]', the [metric] was [val1] [units] in [yr1], [val2] [units] in [yr2]...{CONTENT_SEPARATORS['row_end']}
Example: {CONTENT_SEPARATORS['row_start']}In table 'Building permits granted', under 'Approximate value total', the total was 254891 million CZK in 2015, 389752 million CZK in 2020...{CONTENT_SEPARATORS['row_end']}

B) YEARS IN ROWS: One sentence per year with all columns
Pattern: {CONTENT_SEPARATORS['column_start']}In table '[English name]', for [year], [metric1] was [val1] [units], [metric2] was [val2] [units]...{CONTENT_SEPARATORS['column_end']}

C) NO YEARS: One sentence per row with all columns
Pattern: {CONTENT_SEPARATORS['row_start']}In table '[English name]', for [row_label], [metric1] was [val1] [units], [metric2] was [val2] [units]...{CONTENT_SEPARATORS['row_end']}

=== CHART/IMAGE PROCESSING ===

🚨 DO NOT write chart title/subtitle as separate line - put it INSIDE {CONTENT_SEPARATORS['image_start']}...{CONTENT_SEPARATORS['image_end']} block! 🚨
Pattern: {CONTENT_SEPARATORS['image_start']}In chart '[FULL_TITLE + SUBTITLE in English]', [category] was [val1] in [yr1], [val2] in [yr2]... Trend: [analysis].{CONTENT_SEPARATORS['image_end']}

Example: {CONTENT_SEPARATORS['image_start']}In chart 'Basic indicators of industry by economic activity in 2022', for employed persons, the total was 1403 thousand persons, with mining representing 4.0%, manufacturing 91.7%, electricity/gas supply 2.9%, and water supply 1.4%.{CONTENT_SEPARATORS['image_end']}

=== TEXT ===

Wrap normal text in paragraphs: {CONTENT_SEPARATORS['text_start']}...{CONTENT_SEPARATORS['text_end']}

OTHER RULES:
- SKIP all Czech text completely EVERYWHERE IN THE PDF - use ONLY English text. 
- Always place the unit right after the value: “254891 million CZK in 2015”.
- Never say 'first/second chart'; only use the chart title.
- When English labels are available anywhere (e.g., an Indicator column), always use those; ignore Czech completely.
- List all available years in ascending order; do not skip any year or value present in the table.
- Long time series: if >12 years, split into 2-3 sentences within the same [R] block.
- Do not mention colors or visual styles.

🚨 NEVER CREATE MARKDOWN TABLES - ONLY SENTENCES WITH SEPARATORS! 🚨
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


def smart_text_chunking(
    text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Separator-based text chunking optimized for LlamaParse formatted content.
    PRIMARY STRATEGY: Extract content between separator pairs as chunks.
    SECONDARY STRATEGY: For content exceeding MAX_CHUNK_SIZE, apply sentence-boundary chunking with ceiling division.
    NEVER splits mid-word or mid-sentence - only at complete sentence boundaries.

    All content types ([R]...[/R], [C]...[/C], [T]...[/T], [I]...[/I], [X]...[/X]) are treated equally:
    - If ≤ MAX_CHUNK_SIZE: kept as single semantic unit
    - If > MAX_CHUNK_SIZE: split intelligently using sentence boundaries and ceiling division

    Args:
        text: Text to chunk (LlamaParse formatted content)
        max_chunk_size: Maximum characters per chunk (default: 4000)
        overlap: Character overlap between chunks (used for sentence-based overlap)

    Returns:
        List of text chunks based on separator boundaries and sentence completion
    """
    import re

    chunks = []

    # PRIMARY STRATEGY: Extract content between separator pairs
    # Define separator pairs using ONLY original separators
    # Format: (start_sep, end_sep, content_type)
    separator_pairs = [
        # Table content
        (CONTENT_SEPARATORS["table_start"], CONTENT_SEPARATORS["table_end"], "table"),
        # Column content - SPECIAL: Keep as single unit for semantic coherence
        (
            CONTENT_SEPARATORS["column_start"],
            CONTENT_SEPARATORS["column_end"],
            "column",
        ),
        # Row content - Rows within tables
        (
            CONTENT_SEPARATORS["row_start"],
            CONTENT_SEPARATORS["row_end"],
            "row",
        ),
        # Image content
        (CONTENT_SEPARATORS["image_start"], CONTENT_SEPARATORS["image_end"], "image"),
        # Text content
        (CONTENT_SEPARATORS["text_start"], CONTENT_SEPARATORS["text_end"], "text"),
    ]

    print__chromadb_debug(
        f"📝 Starting separator-based chunking on {len(text)} characters"
    )

    # Extract all content between separator pairs
    for start_sep, end_sep, content_type in separator_pairs:
        # Create regex pattern to find content between separators
        # Use re.DOTALL to match newlines
        pattern = re.escape(start_sep) + r"(.*?)" + re.escape(end_sep)
        matches = re.findall(pattern, text, re.DOTALL)

        print__chromadb_debug(
            f"🔍 Looking for {start_sep} ... {end_sep} pairs ({content_type}): found {len(matches)} matches"
        )

        for i, match in enumerate(matches):
            content = match.strip()
            if content and len(content) >= MIN_CHUNK_SIZE:
                print__chromadb_debug(
                    f"📦 Found {content_type} content #{i+1}: {len(content)} chars"
                )

                # Content fits within chunk size limit
                if len(content) <= max_chunk_size:
                    chunks.append(content)
                    print__chromadb_debug(
                        f"✅ Added chunk: {content[:100]}..."
                        if len(content) > 100
                        else f"✅ Added chunk: {content}"
                    )
                else:
                    # Content exceeds size limit - split using sentence boundaries with ceiling division
                    # This applies to ALL content types (column, table, image, text) equally
                    print__chromadb_debug(
                        f"🔄 {content_type.capitalize()} content too large ({len(content)} > {max_chunk_size}), splitting at sentence boundaries"
                    )
                    large_chunks = _split_large_separator_content(
                        content, max_chunk_size, overlap
                    )
                    chunks.extend(large_chunks)
                    print__chromadb_debug(
                        f"➕ Added {len(large_chunks)} sentence-boundary chunks"
                    )
            else:
                print__chromadb_debug(
                    f"⚠️ Skipping {content_type} content #{i+1}: too small ({len(content)} < {MIN_CHUNK_SIZE})"
                )

    # Handle any remaining content not captured by separators (fallback)
    remaining_text = text
    for start_sep, end_sep, _ in separator_pairs:
        pattern = re.escape(start_sep) + r".*?" + re.escape(end_sep)
        remaining_text = re.sub(pattern, "", remaining_text, flags=re.DOTALL)

    # Clean up remaining text and check if there's anything significant
    remaining_text = remaining_text.strip()
    # Remove page separators and other noise
    remaining_text = remaining_text.replace(CONTENT_SEPARATORS["page_separator"], "")
    remaining_text = re.sub(r"\n+", "\n", remaining_text).strip()

    if remaining_text and len(remaining_text) >= MIN_CHUNK_SIZE:
        print__chromadb_debug(
            f"📄 Found ungrouped content: {len(remaining_text)} chars"
        )
        if len(remaining_text) <= max_chunk_size:
            chunks.append(remaining_text)
        else:
            fallback_chunks = _split_large_separator_content(
                remaining_text, max_chunk_size, overlap
            )
            chunks.extend(fallback_chunks)

    print__chromadb_debug(f"✅ Separator-based chunking created {len(chunks)} chunks")
    return chunks


def _split_large_separator_content(
    content: str, max_size: int, overlap: int
) -> List[str]:
    """
    Split large content by calculating optimal number of chunks and finding sentence boundaries.
    Uses ceiling division to determine chunk count, then distributes sentences evenly.
    NEVER splits mid-word or mid-sentence - only at complete sentence boundaries.

    This function is applied equally to ALL content types (column, table, image, text) when they
    exceed MAX_CHUNK_SIZE, ensuring consistent handling while preserving sentence integrity.
    """
    import math

    if len(content) <= max_size:
        return [content]

    # Calculate how many chunks we need using ceiling division
    num_chunks_needed = math.ceil(len(content) / max_size)

    print__chromadb_debug(
        f"📏 Content {len(content)} chars needs {num_chunks_needed} chunks (max {max_size})"
    )

    # Split into sentences first
    sentences = _extract_sentences(content)

    if len(sentences) <= 1:
        # If we can't find sentence boundaries, return as single chunk
        # This prevents mid-word splitting
        print__chromadb_debug(
            f"⚠️ No sentence boundaries found, keeping as single chunk"
        )
        return [content]

    # Distribute sentences across chunks
    chunks = _distribute_sentences_across_chunks(
        sentences, num_chunks_needed, max_size, overlap
    )

    print__chromadb_debug(
        f"✅ Split into {len(chunks)} chunks using sentence boundaries"
    )
    return chunks


def _extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text using LlamaParse-specific patterns.
    Returns list of complete sentences.
    """
    # Multiple sentence boundary patterns for LlamaParse format
    sentence_patterns = [
        r"(?<=\.)\s+(?=[A-Z][a-z])",  # Period + space + capital+lowercase
        r"(?<=\!)\s+(?=[A-Z][a-z])",  # Exclamation + space + capital+lowercase
        r"(?<=\?)\s+(?=[A-Z][a-z])",  # Question + space + capital+lowercase
    ]

    # Try each pattern to find sentence boundaries
    sentences = [text]  # Start with whole text
    for pattern in sentence_patterns:
        new_sentences = []
        for sentence in sentences:
            new_sentences.extend(re.split(pattern, sentence))
        sentences = new_sentences

    # Clean up sentences and remove empty ones
    sentences = [s.strip() for s in sentences if s.strip()]

    print__chromadb_debug(
        f"📝 Extracted {len(sentences)} sentences from {len(text)} chars"
    )
    return sentences


def _distribute_sentences_across_chunks(
    sentences: List[str], num_chunks: int, max_size: int, overlap: int
) -> List[str]:
    """
    Distribute sentences across the calculated number of chunks.
    Ensures complete sentences and handles overlap properly.
    """
    if len(sentences) <= num_chunks:
        # If we have fewer sentences than chunks needed, each sentence becomes a chunk
        return [s for s in sentences if len(s.strip()) >= MIN_CHUNK_SIZE]

    chunks = []
    sentences_per_chunk = len(sentences) // num_chunks
    remainder = len(sentences) % num_chunks

    start_idx = 0

    for chunk_idx in range(num_chunks):
        # Calculate how many sentences for this chunk
        chunk_sentence_count = sentences_per_chunk + (1 if chunk_idx < remainder else 0)
        end_idx = start_idx + chunk_sentence_count

        # Get sentences for this chunk
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = " ".join(chunk_sentences)

        # If chunk is too large, try to reduce by moving last sentence to next chunk
        while len(chunk_text) > max_size and len(chunk_sentences) > 1:
            chunk_sentences = chunk_sentences[:-1]
            chunk_text = " ".join(chunk_sentences)
            end_idx -= 1

        # Add overlap from previous chunk if needed
        if overlap > 0 and chunks and chunk_text:
            overlap_text = _get_sentence_overlap(chunks[-1], overlap)
            if overlap_text:
                chunk_text = overlap_text + " " + chunk_text

        if chunk_text and len(chunk_text.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text.strip())
            print__chromadb_debug(
                f"📄 Chunk {chunk_idx + 1}: {len(chunk_text)} chars, {len(chunk_sentences)} sentences"
            )

        start_idx = end_idx

    return chunks


def _get_sentence_overlap(previous_chunk: str, overlap_chars: int) -> str:
    """
    Extract overlap from previous chunk, ensuring it ends at sentence boundary.
    """
    if len(previous_chunk) <= overlap_chars:
        return previous_chunk

    # Get the last overlap_chars characters
    overlap_text = previous_chunk[-overlap_chars:]

    # Find the last complete sentence in the overlap
    sentences = _extract_sentences(overlap_text)

    if sentences:
        # Return complete sentences that fit in overlap
        return " ".join(sentences)
    else:
        # If no complete sentences found, try to find sentence ending
        # Look for the first sentence boundary in the overlap
        for i in range(len(overlap_text)):
            if overlap_text[i] in ".!?" and i < len(overlap_text) - 1:
                if overlap_text[i + 1] == " " and len(overlap_text) > i + 2:
                    if overlap_text[i + 2].isupper():
                        return overlap_text[i + 2 :].strip()

        # No sentence boundary found, return empty to avoid partial sentences
        return ""


def _get_sentence_overlap(previous_chunk: str, overlap_chars: int) -> str:
    """
    Extract overlap from previous chunk, ensuring it ends at sentence boundary.
    """
    if len(previous_chunk) <= overlap_chars:
        return previous_chunk

    # Get the last overlap_chars characters
    overlap_text = previous_chunk[-overlap_chars:]

    # Find the last complete sentence in the overlap
    sentences = _extract_sentences(overlap_text)

    if sentences:
        # Return complete sentences that fit in overlap
        return " ".join(sentences)
    else:
        # If no complete sentences found, try to find sentence ending
        # Look for the first sentence boundary in the overlap
        for i in range(len(overlap_text)):
            if overlap_text[i] in ".!?" and i < len(overlap_text) - 1:
                if overlap_text[i + 1] == " " and len(overlap_text) > i + 2:
                    if overlap_text[i + 2].isupper():
                        return overlap_text[i + 2 :].strip()

        # No sentence boundary found, return empty to avoid partial sentences
        return ""


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
def save_parsed_text_to_file(text: str, file_path: str) -> None:
    """
    Save parsed text to a file and provide content analysis.

    Args:
        text: The parsed text content with separators
        file_path: Path where to save the text file
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Analyze content structure
        content_types = extract_content_by_type(text)

        print__chromadb_debug(f"💾 Successfully saved parsed text to: {file_path}")
        print(f"✅ Parsed text saved to: {file_path}")
        print(f"📊 Content analysis:")
        print(f"   📋 Tables: {len(content_types['tables'])}")
        print(f"   🖼️  Images/Graphs: {len(content_types['images'])}")
        print(f"   📝 Text sections: {len(content_types['text'])}")

        # Check for proper separator usage
        separator_count = text.count(CONTENT_SEPARATORS["page_separator"])
        print(f"   📄 Pages: {separator_count + 1}")

        if any(content_types.values()):
            print(f"✅ Content properly separated with custom markers")
        else:
            print(f"⚠️  No content separators found - check LlamaParse instructions")

    except Exception as e:
        print__chromadb_debug(f"❌ Error saving parsed text: {str(e)}")
        raise


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


def create_documents_from_text(
    text: str, source_file: str, parsing_method: str
) -> List[Dict[str, Any]]:
    """
    Create document-like structure from parsed text for chunking.
    Optimized for LlamaParse output with content separators and page boundaries.

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

    # Analyze content structure before splitting
    content_analysis = extract_content_by_type(text)
    print__chromadb_debug(
        f"📊 Content analysis: {len(content_analysis['tables'])} tables, {len(content_analysis['columns'])} columns, {len(content_analysis['images'])} images, {len(content_analysis['text'])} text sections"
    )

    pages = []

    # Split by our custom page separator first (LlamaParse)
    if CONTENT_SEPARATORS["page_separator"] in text:
        print__chromadb_debug(f"📄 Found LlamaParse page separators")
        page_texts = text.split(CONTENT_SEPARATORS["page_separator"])
    # Fallback separators for other methods
    elif "\n---\n" in text:
        print__chromadb_debug(f"📄 Found standard page separators (---)")
        page_texts = text.split("\n---\n")
    elif "\n=================\n" in text:
        print__chromadb_debug(f"📄 Found extended page separators (=================)")
        page_texts = text.split("\n=================\n")
    else:
        # If no clear separators, treat as one large document
        print__chromadb_debug(
            f"📄 No page separators found, treating as single document"
        )
        page_texts = [text]

    print__chromadb_debug(f"✂️ Split text into {len(page_texts)} sections")

    for page_num, page_text in enumerate(page_texts, 1):
        if page_text.strip():  # Only add non-empty pages
            # Keep original text with separators intact for chunking process
            cleaned_text = page_text.strip()

            # Analyze this section's content using original separators
            section_has_tables = CONTENT_SEPARATORS["table_start"] in cleaned_text
            section_has_columns = CONTENT_SEPARATORS["column_start"] in cleaned_text
            section_has_images = CONTENT_SEPARATORS["image_start"] in cleaned_text

            page_data = {
                "text": cleaned_text,
                "page_number": page_num,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "source_file": source_file,
                "parsing_method": parsing_method,
                "has_tables": section_has_tables,
                "has_columns": section_has_columns,
                "has_images": section_has_images,
            }
            pages.append(page_data)

            print__chromadb_debug(
                f"📄 Section {page_num}: {len(cleaned_text)} chars, tables: {section_has_tables}, columns: {section_has_columns}, images: {section_has_images}"
            )

    print__chromadb_debug(f"✅ Created {len(pages)} document sections from parsed text")

    # Summary statistics
    total_chars = sum(p["char_count"] for p in pages)
    sections_with_tables = sum(1 for p in pages if p["has_tables"])
    sections_with_columns = sum(1 for p in pages if p["has_columns"])
    sections_with_images = sum(1 for p in pages if p["has_images"])

    print__chromadb_debug(f"📊 Document summary:")
    print__chromadb_debug(f"📊   - Total characters: {total_chars}")
    print__chromadb_debug(
        f"📊   - Sections with tables: {sections_with_tables}/{len(pages)}"
    )
    print__chromadb_debug(
        f"📊   - Sections with columns: {sections_with_columns}/{len(pages)}"
    )
    print__chromadb_debug(
        f"📊   - Sections with images: {sections_with_images}/{len(pages)}"
    )
    print__chromadb_debug(
        f"📊   - Average section size: {total_chars/len(pages):.0f} characters"
    )

    return pages


def clean_separator_artifacts(text: str, remove_completely: bool = True) -> str:
    """
    Intelligently clean text of separator artifacts while preserving content structure.

    Args:
        text: Text that may contain content separators
        remove_completely: If True, completely removes separators. If False, replaces with content type markers.

    Returns:
        Cleaned text with separators removed or replaced
    """
    cleaned_text = text

    if remove_completely:
        # Completely remove separator markers for final chunks
        separators_to_remove = [
            # Original separators only
            CONTENT_SEPARATORS["table_start"],
            CONTENT_SEPARATORS["table_end"],
            CONTENT_SEPARATORS["column_start"],
            CONTENT_SEPARATORS["column_end"],
            CONTENT_SEPARATORS["row_start"],
            CONTENT_SEPARATORS["row_end"],
            CONTENT_SEPARATORS["image_start"],
            CONTENT_SEPARATORS["image_end"],
            CONTENT_SEPARATORS["text_start"],
            CONTENT_SEPARATORS["text_end"],
            CONTENT_SEPARATORS["page_separator"],
            # Also remove the page break marker
            "[.P]",
        ]

        for separator in separators_to_remove:
            cleaned_text = cleaned_text.replace(separator, " ")
    # When remove_completely=False, don't modify the text at all
    # Keep original separators for chunking process

    # Clean up excessive whitespace while preserving paragraph breaks
    import re

    cleaned_text = re.sub(r"\n\n\n+", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"^\s+", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Normalize multiple spaces

    return cleaned_text.strip()


def extract_content_by_type(text: str) -> Dict[str, List[str]]:
    """
    Extract different content types from parsed text based on separators.
    Useful for debugging and content analysis.

    Args:
        text: The parsed text with content separators

    Returns:
        Dictionary with content types as keys and content lists as values
    """
    content_types = {"tables": [], "columns": [], "rows": [], "images": [], "text": []}

    # Extract tables
    table_pattern = (
        f"{CONTENT_SEPARATORS['table_start']}(.*?){CONTENT_SEPARATORS['table_end']}"
    )
    # Note: re is already imported at the top of the file

    tables = re.findall(table_pattern, text, re.DOTALL)
    content_types["tables"] = [table.strip() for table in tables]

    # Extract columns
    column_pattern = (
        f"{CONTENT_SEPARATORS['column_start']}(.*?){CONTENT_SEPARATORS['column_end']}"
    )
    columns = re.findall(column_pattern, text, re.DOTALL)
    content_types["columns"] = [column.strip() for column in columns]

    # Extract rows
    row_pattern = (
        f"{CONTENT_SEPARATORS['row_start']}(.*?){CONTENT_SEPARATORS['row_end']}"
    )
    rows = re.findall(row_pattern, text, re.DOTALL)
    content_types["rows"] = [row.strip() for row in rows]

    # Extract images
    image_pattern = (
        f"{CONTENT_SEPARATORS['image_start']}(.*?){CONTENT_SEPARATORS['image_end']}"
    )
    images = re.findall(image_pattern, text, re.DOTALL)
    content_types["images"] = [image.strip() for image in images]

    # Extract text
    text_pattern = (
        f"{CONTENT_SEPARATORS['text_start']}(.*?){CONTENT_SEPARATORS['text_end']}"
    )
    texts = re.findall(text_pattern, text, re.DOTALL)
    content_types["text"] = [text_content.strip() for text_content in texts]

    print__chromadb_debug(
        f"📊 Extracted content: {len(content_types['tables'])} tables, {len(content_types['columns'])} columns, {len(content_types['rows'])} rows, {len(content_types['images'])} images, {len(content_types['text'])} text sections"
    )

    return content_types


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
                system_prompt="You are an expert document parser specializing in converting complex tabular and chart data into chunking-friendly formats for semantic search systems and you are also an expert in describing graphs and charts in details. But also You are able to parse back normal text as is.",
                user_prompt=comprehensive_instructions,
                page_separator=CONTENT_SEPARATORS["page_separator"],
                verbose=True,
                parse_mode="parse_page_with_lvm",
                vendor_multimodal_model_name="anthropic-sonnet-4.0",
            )
            print__chromadb_debug(
                "🔧 Using newer LlamaParse parameter system (system_prompt + user_prompt)"
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
                page_separator=CONTENT_SEPARATORS["page_separator"],
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
                "page_separator": CONTENT_SEPARATORS["page_separator"],
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
                text_content = CONTENT_SEPARATORS["page_separator"].join(
                    [page.get("md", page.get("text", "")) for page in pages]
                )
            else:
                raise Exception(
                    f"No usable content found in results: {list(result_data.keys())}"
                )

        print(f"📊 Processing retrieved content...")

        # Split content by pages if available
        if CONTENT_SEPARATORS["page_separator"] in text_content:
            page_texts = text_content.split(CONTENT_SEPARATORS["page_separator"])
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
            combined_text = CONTENT_SEPARATORS["page_separator"].join(
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
    Handles both page-based data and continuous text from parsed files.
    Uses semantic-aware chunking that preserves LlamaParse sentence structure.

    Args:
        pages_data: List of page/section data from parsed text

    Returns:
        List of chunk dictionaries with text and metadata
    """
    all_chunks = []
    chunk_id = 0

    print__chromadb_debug(f"Processing {len(pages_data)} sections/pages for chunking")

    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data.get("page_number", 1)
        parsing_method = page_data.get("parsing_method", "unknown")

        print__chromadb_debug(f"Processing section {page_num}: {len(text)} characters")

        # Use semantic-aware chunking that respects LlamaParse structure
        page_chunks = smart_text_chunking(text)

        print__chromadb_debug(
            f"Section {page_num} split into {len(page_chunks)} chunks"
        )

        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Clean separator artifacts from chunk text (completely remove them)
            cleaned_chunk_text = clean_separator_artifacts(
                chunk_text, remove_completely=True
            )

            # Check token count and split only if absolutely necessary
            token_count = num_tokens_from_string(cleaned_chunk_text)

            if token_count <= MAX_TOKENS:
                # Chunk is within token limits, use as-is
                chunk_data = {
                    "id": chunk_id,
                    "text": cleaned_chunk_text,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "token_chunk_index": 0,
                    "total_page_chunks": len(page_chunks),
                    "total_token_chunks": 1,
                    "char_count": len(cleaned_chunk_text),
                    "token_count": token_count,
                    "source_file": page_data["source_file"],
                    "parsing_method": parsing_method,
                    "doc_hash": get_document_hash(cleaned_chunk_text),
                }

                all_chunks.append(chunk_data)
                chunk_id += 1
            else:
                # Only split by tokens if chunk exceeds token limit
                print__chromadb_debug(
                    f"Chunk {chunk_id} exceeds token limit ({token_count} > {MAX_TOKENS}), splitting..."
                )
                token_chunks = split_text_by_tokens(cleaned_chunk_text)

                for token_chunk_idx, token_chunk in enumerate(token_chunks):
                    # Clean each token chunk as well (completely remove separators)
                    cleaned_token_chunk = clean_separator_artifacts(
                        token_chunk, remove_completely=True
                    )
                    token_count = num_tokens_from_string(cleaned_token_chunk)

                    chunk_data = {
                        "id": chunk_id,
                        "text": cleaned_token_chunk,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "token_chunk_index": token_chunk_idx,
                        "total_page_chunks": len(page_chunks),
                        "total_token_chunks": len(token_chunks),
                        "char_count": len(cleaned_token_chunk),
                        "token_count": token_count,
                        "source_file": page_data["source_file"],
                        "parsing_method": parsing_method,
                        "doc_hash": get_document_hash(cleaned_token_chunk),
                    }

                    all_chunks.append(chunk_data)
                    chunk_id += 1

    print__chromadb_debug(
        f"Created {len(all_chunks)} chunks from {len(pages_data)} sections"
    )

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


# ==============================================================================
# CHROMADB OPERATIONS
# ==============================================================================
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
    Hybrid search combining semantic and BM25 approaches.
    """
    print__chromadb_debug(f"Hybrid search for query: '{query_text}'")

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
                k=n_results,
            )

            for i, (doc, meta, distance) in enumerate(
                zip(
                    semantic_raw["documents"][0],
                    semantic_raw["metadatas"][0],
                    semantic_raw["distances"][0],
                )
            ):
                similarity_score = max(0, 1 - (distance / 2))
                semantic_results.append(
                    {
                        "id": f"semantic_{i}",
                        "document": doc,
                        "metadata": meta,
                        "semantic_score": similarity_score,
                        "source": "semantic",
                    }
                )

        except Exception as e:
            print__chromadb_debug(f"Semantic search failed: {e}")
            semantic_results = []

        # BM25 search
        bm25_results = []
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

                    top_indices = np.argsort(bm25_scores)[::-1][:n_results]

                    for i, idx in enumerate(top_indices):
                        if bm25_scores[idx] > 0:
                            bm25_results.append(
                                {
                                    "id": f"bm25_{i}",
                                    "document": documents[idx],
                                    "metadata": (
                                        metadatas[idx] if idx < len(metadatas) else {}
                                    ),
                                    "bm25_score": float(bm25_scores[idx]),
                                    "source": "bm25",
                                }
                            )

        except Exception as e:
            print__chromadb_debug(f"BM25 search failed: {e}")
            bm25_results = []

        # Combine results with semantic focus
        combined_results = {}

        # Process semantic results (primary)
        for result in semantic_results:
            doc_id = result["metadata"].get("chunk_id", result["document"][:50])
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["bm25_score"] = 0.0

        # Process BM25 results (secondary)
        for result in bm25_results:
            doc_id = result["metadata"].get("chunk_id", result["document"][:50])
            if doc_id in combined_results:
                combined_results[doc_id]["bm25_score"] = result["bm25_score"]
                combined_results[doc_id]["source"] = "hybrid"
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["semantic_score"] = 0.0

        # Calculate final scores with semantic focus
        final_results = []
        max_semantic = max(
            (r.get("semantic_score", 0) for r in combined_results.values()), default=1
        )
        max_bm25 = max(
            (r.get("bm25_score", 0) for r in combined_results.values()), default=1
        )

        semantic_weight = SEMANTIC_WEIGHT
        bm25_weight = BM25_WEIGHT

        for doc_id, result in combined_results.items():
            semantic_score = (
                result.get("semantic_score", 0.0) / max_semantic
                if max_semantic > 0
                else 0.0
            )
            bm25_score = (
                result.get("bm25_score", 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            )

            final_score = (semantic_weight * semantic_score) + (
                bm25_weight * bm25_score
            )

            result["score"] = final_score
            result["semantic_score"] = semantic_score
            result["bm25_score"] = bm25_score

            final_results.append(result)

        # Sort by final score
        final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results[:n_results]

    except Exception as e:
        print__chromadb_debug(f"Hybrid search failed: {e}")
        return []


def cohere_rerank(query, docs, top_n):
    """Rerank documents using Cohere's rerank model.

    Automatically falls back from trial key to production key if rate limit is hit.
    Uses .index field for correct mapping back to original documents.
    """

    def _attempt_rerank(api_key, key_type="Trial"):
        """Helper function to attempt reranking with a specific API key."""
        print(f"  Initializing Cohere client with {key_type} key...")
        co = cohere.Client(api_key)

        texts = [doc.page_content for doc in docs]
        docs_for_cohere = [{"text": t} for t in texts]

        print(f"  Sending {len(docs_for_cohere)} documents to Cohere for reranking...")
        rerank_response = co.rerank(
            model="rerank-v4.0-fast",
            query=query,
            documents=docs_for_cohere,
            top_n=top_n,
        )

        print(
            f"✅ Successfully reranked {len(docs)} documents using Cohere model: rerank-v4.0-fast ({key_type} key)"
        )

        # Use .index field to map back to the original doc
        reranked = []
        for res in rerank_response.results:
            doc = docs[res.index]
            reranked.append((doc, res))

        print(f"  Returning {len(reranked)} reranked results")
        return reranked

    # Try with primary key first
    try:
        cohere_api_key = os.environ.get("COHERE_API_KEY", "")
        if not cohere_api_key:
            print__chromadb_debug(
                "Warning: COHERE_API_KEY not found. Skipping reranking."
            )
            return [
                (doc, type("obj", (object,), {"relevance_score": 0.5, "index": i})())
                for i, doc in enumerate(docs)
            ]

        return _attempt_rerank(cohere_api_key, "Trial")

    except Exception as e:
        error_str = str(e)

        # Check if it's a rate limit error (429)
        if (
            "429" in error_str
            or "Trial key" in error_str
            or "rate limit" in error_str.lower()
        ):
            print(
                f"⚠️ Trial key rate limit reached. Attempting fallback to Production key..."
            )

            # Try with production key
            try:
                cohere_api_key_prod = os.environ.get("COHERE_API_KEY_PROD", "")
                if not cohere_api_key_prod:
                    print__chromadb_debug(
                        "Warning: COHERE_API_KEY_PROD not found. Returning dummy scores."
                    )
                    print(f"❌ Original error: {e}")
                    return [
                        (
                            doc,
                            type(
                                "obj", (object,), {"relevance_score": 0.5, "index": i}
                            )(),
                        )
                        for i, doc in enumerate(docs)
                    ]

                return _attempt_rerank(cohere_api_key_prod, "Production")

            except Exception as prod_error:
                print__chromadb_debug(f"Production key also failed: {prod_error}")
                # Return original docs with dummy scores
                return [
                    (
                        doc,
                        type("obj", (object,), {"relevance_score": 0.5, "index": i})(),
                    )
                    for i, doc in enumerate(docs)
                ]
        else:
            # For non-rate-limit errors, return dummy scores
            print__chromadb_debug(f"Cohere reranking failed: {e}")
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

                combined_text = CONTENT_SEPARATORS["page_separator"].join(
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
            # Import the ChromaDB client factory to support cloud/local switching
            from metadata.chromadb_client_factory import (
                get_chromadb_client,
                should_use_cloud,
            )

            # Check if we should use cloud ChromaDB
            use_cloud = should_use_cloud()

            if use_cloud:
                print("\n" + "=" * 80)
                print(
                    "🌐🌐🌐 USING CHROMADB CLOUD - CONNECTING TO REMOTE DATABASE 🌐🌐🌐"
                )
                print("=" * 80)
                print(
                    f"   Environment: CHROMA_USE_CLOUD={os.getenv('CHROMA_USE_CLOUD')}"
                )
                print(f"   Tenant: {os.getenv('CHROMA_API_TENANT', 'not set')}")
                print(f"   Database: {os.getenv('CHROMA_API_DATABASE', 'not set')}")
                print("=" * 80 + "\n")
            else:
                print("\n" + "=" * 80)
                print("📂 Using Local ChromaDB")
                print("=" * 80)
                print(f"   Path: {CHROMA_DB_PATH}")
                print("=" * 80 + "\n")

            # Get ChromaDB client (automatically chooses cloud or local based on env)
            client = get_chromadb_client(
                local_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME
            )

            # Get the collection
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

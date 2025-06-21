#!/usr/bin/env python3
"""PDF to ChromaDB Document Processing and Search

This module provides functionality to process large PDF documents (including those with tables),
chunk them appropriately, generate embeddings, store in ChromaDB, and perform hybrid search
with Cohere reranking.

Key Features:
-------------
1. PDF Processing:
   - LlamaParse for superior table and complex layout handling
   - Smart semantic-aware text chunking with token awareness
   - Preserves document structure information
   - Content type separation (tables, images, text)
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
   - Semantic-aware chunking for optimal embeddings

4. Hybrid Search:
   - Combines semantic search (Azure embeddings) with BM25
   - Semantic-focused weighting (85% semantic, 15% BM25)
   - Cohere reranking for final results
   - Configurable result count and thresholds

5. Error Handling:
   - Comprehensive error handling
   - Progress tracking with tqdm
   - Debug logging support
   - Quality validation for chunks

Usage Example:
-------------
# Three-step workflow:
# 1. Parse PDF with LlamaParse
# 2. Chunk and store in ChromaDB
# 3. Test search functionality

# Configure operations in the script:
PARSE_WITH_LLAMAPARSE = 1
CHUNK_AND_STORE = 1
DO_TESTING = 1

# Run the script:
python pdf_to_chromadb.py
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
import re
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Processing Mode - Three independent operations
PARSE_WITH_LLAMAPARSE = 0    # Set to 1 to parse PDF with LlamaParse and save to txt file
                             # Set to 0 to skip parsing (use existing txt file)

CHUNK_AND_STORE = 0          # Set to 1 to chunk text and create/update ChromaDB
                             # Set to 0 to skip chunking (use existing ChromaDB)

DO_TESTING = 1               # Set to 1 to test search on existing ChromaDB
                             # Set to 0 to skip testing

# PDF Parsing Method Selection - LLAMAPARSE ONLY
PDF_PARSING_METHOD = "llamaparse"  # Using LlamaIndex's premium parser (requires API key)

# LlamaParse Progress Monitoring Options
LLAMAPARSE_ENHANCED_MONITORING = True  # Set to True for enhanced progress tracking with job status
                                       # Set to False for standard LlamaParse with basic progress

# File and Collection Settings
# PDF_FILENAME = "33012024.pdf"  # Just the filename (PDF should be in same folder as script)
# PDF_FILENAME = "1_PDFsam_271_PDFsam_33012024.pdf"  # Just the filename (PDF should be in same folder as script)

# Multiple PDF Files Configuration - Add your PDF filenames here
PDF_FILENAMES = [
    "1_PDFsam_32019824.pdf",
    "101_PDFsam_32019824.pdf",
    "201_PDFsam_32019824.pdf",
    "301_PDFsam_32019824.pdf",
    "401_PDFsam_32019824.pdf",
    "501_PDFsam_32019824.pdf",
    "601_PDFsam_32019824.pdf",
    "701_PDFsam_32019824.pdf",
    "801_PDFsam_32019824.pdf",
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
TEST_QUERY = "jak se zmenili zasoby v milionech korun v oboru cinnosti - vzdelani? Porovnej posleni roky"
# TEST_QUERY = "Kolik hektarů zahrad (gardens) se nachází v Praze?"
# TEST_QUERY = "Kolik je osobnich automobilu ve Varsave?"
# TEST_QUERY = "How many passenger cars there in Warsaw (total, not per 1000 inhabitants)?"

# Azure OpenAI Settings
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"  # Your Azure deployment name

# LlamaParse Settings (only needed if using llamaparse method)
LLAMAPARSE_API_KEY = os.environ.get("LLAMAPARSE_API_KEY", "")  # Read from .env file

# Content Separators - unique strings unlikely to appear in normal text
CONTENT_SEPARATORS = {
    'table_start': '[T]',
    'table_end': '[/T]',
    'image_start': '[I]',
    'image_end': '[/I]',
    'text_start': '[X]',
    'text_end': '[/X]',
    'page_separator': '[P]'
}

# Token and Chunking Settings
MAX_TOKENS = 8190          # Token limit for Azure OpenAI
MIN_CHUNK_SIZE = 400       # Optimized for statistical content
MAX_CHUNK_SIZE = 800       # Optimized for text-embedding-3-large
CHUNK_OVERLAP = 100        # Reduced for better semantic boundaries

# Search Settings
HYBRID_SEARCH_RESULTS = 20      # Number of results from hybrid search
SEMANTIC_WEIGHT = 0.85          # Weight for semantic search (0.0-1.0)
BM25_WEIGHT = 0.15              # Weight for BM25 search (0.0-1.0)
FINAL_RESULTS_COUNT = 2         # Number of final results to return

# Debug and Processing Settings
PDF_ID = 40                     # Unique identifier for debug messages

# =====================================================================
# PATH CONFIGURATION - AUTOMATICALLY SET
# =====================================================================
# PDF_PATH = SCRIPT_DIR / PDF_FILENAME  # Full path to PDF file
# PARSED_TEXT_PATH = SCRIPT_DIR / PARSED_TEXT_FILENAME  # Full path to parsed text file

# ChromaDB storage location
CHROMA_DB_PATH = SCRIPT_DIR / "pdf_chromadb_llamaparse"

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
def get_llamaparse_instructions() -> str:
    """
    Generate comprehensive LlamaParse instructions for consistent parsing across all methods.
    Centralizes instructions to avoid duplication and ensure consistency.
    
    Returns:
        Comprehensive instruction string for LlamaParse processing
    """
    return f"""
CRITICAL REQUIREMENTS FOR DOCUMENT PROCESSING:

LANGUAGE PROCESSING:
- Describe tables, graphs, and maps in English
- Preserve all Czech names, locations, and technical terms in parentheses for reference
- Use original language as-is for regular text content
- Maintain proper nouns and technical terms exactly as written

NUMBER FORMATTING:
- Format all numbers as plain integers or decimals without thousand separators
- Example: write "49621" not "49,621" or "49 621"

CONTENT SEPARATION (MANDATORY):
You MUST clearly separate different content types using these EXACT separators:
- Tables: {CONTENT_SEPARATORS['table_start']} ... content ... {CONTENT_SEPARATORS['table_end']}
- Images/Graphs: {CONTENT_SEPARATORS['image_start']} ... content ... {CONTENT_SEPARATORS['image_end']}
- Regular Text: {CONTENT_SEPARATORS['text_start']} ... content ... {CONTENT_SEPARATORS['text_end']}
- Page Breaks: {CONTENT_SEPARATORS['page_separator']}

NEVER use these separator strings within actual content - they are ONLY for marking boundaries.

=== TABLE PROCESSING ===

CONTEXT-RICH TABLE CONVERSION:
- Do NOT create traditional markdown tables that lose context when chunked
- Convert each table row into SELF-CONTAINED DESCRIPTIVE SENTENCES
- Each data point must include: table title, row label, column header, value, and units
- Repeat all contextual information for every single value
- Preserve hierarchical relationships through descriptive language

MANDATORY INFORMATION FOR EACH VALUE:
- Table title or section name
- Row identifier (location, entity, category)
- Column identifier (year, month, measurement type)
- Actual numerical value with units
- Parent category context for hierarchical tables
- Time periods and measurement context

HIERARCHY PRESERVATION:
- Always specify parent categories (e.g., "under agricultural land category")
- Include table title context in every sentence
- Preserve units with every value
- Maintain nested structure through descriptive language

SPECIAL HANDLING FOR DATA PRESERVATION:
- Preserve all location names and identifiers exactly as written
- Include codes, abbreviations, or reference numbers when present
- Maintain all units of measurement exactly as shown
- Convert abbreviations to full terms when context allows, but include abbreviations in parentheses
- Include measurement periods and time ranges
- Preserve hierarchical relationships and categorizations

TABLE PROCESSING EXAMPLE:
When you see a hierarchical table like "Land Use Balance for Prague (Bilance půdy v hl. m. Praze)" with main categories and subcategories:

{CONTENT_SEPARATORS['table_start']}
In the land use balance table for Prague (Bilance půdy v hl. m. Praze) as of December 31st, the total area recorded 49621 hectares in 2022.
In the land use balance table for Prague (Bilance půdy v hl. m. Praze) as of December 31st, the total area recorded 49621 hectares in 2023.
In the land use balance table for Prague, agricultural land (Zemědělská půda) totaled 19473 hectares in 2022.
In the land use balance table for Prague, agricultural land (Zemědělská půda) totaled 19410 hectares in 2023.
In the land use balance table under agricultural land category, arable land (orná půda) measured 13708 hectares in 2022.
In the land use balance table under agricultural land category, arable land (orná půda) measured 13585 hectares in 2023.
In the land use balance table under agricultural land category, gardens (zahrady) measured 4001 hectares in 2022.
In the land use balance table under agricultural land category, gardens (zahrady) measured 4051 hectares in 2023.
In the land use balance table under agricultural land category, orchards (ovocné sady) measured 580 hectares in 2022.
In the land use balance table under agricultural land category, orchards (ovocné sady) measured 569 hectares in 2023.
In the land use balance table for Prague, non-agricultural land totaled 30148 hectares in 2022.
In the land use balance table for Prague, non-agricultural land totaled 30211 hectares in 2023.
{CONTENT_SEPARATORS['table_end']}

=== GRAPH AND IMAGE PROCESSING ===

COMPREHENSIVE VISUAL DESCRIPTIONS:
- Extract ALL visible numerical values from charts and graphs
- Provide specific values for each data category/series shown
- Describe trends, patterns, and comparisons between data series
- Include complete axis labels, units, scales, and legend information

CRITICAL REQUIREMENTS FOR CHARTS AND GRAPHS:
- DO NOT provide generic descriptions like "varying precipitation levels"
- DO extract specific numerical values visible on the chart (even if approximate)
- DO describe trends like "increasing", "decreasing", "peak in month X", "lowest in month Y"
- DO compare different data series when multiple are shown
- DO include seasonal patterns, peaks, valleys, and notable differences

TREND ANALYSIS REQUIREMENTS:
- Identify trends: increasing, decreasing, stable, cyclical
- Point out seasonal patterns and peaks
- Compare multiple data series when present
- Describe value ranges (minimum to maximum)
- Include station details, elevations, or other contextual information

GRAPH PROCESSING EXAMPLE:
For bar charts like "Monthly Precipitation for Prague Territory 2023":

{CONTENT_SEPARATORS['image_start']}
The monthly precipitation chart for Prague territory (Úhrn srážek na území hl. m. Prahy) shows 2023 data in millimeters for two measurement stations.
Praha Karlov station (261m elevation) recorded January 15mm, March 35mm, June 80mm, August 105mm (annual peak), December 65mm.
Praha Ruzyně station (364m elevation) recorded January 10mm, March 25mm, June 75mm, August 85mm (annual peak), December 70mm.
August had highest precipitation for both stations, with Praha Karlov reaching 105mm and Praha Ruzyně 85mm.
Praha Karlov consistently recorded higher precipitation than Praha Ruzyně throughout most months.
Both stations show clear seasonal patterns with summer peaks (June-August) and winter lows (January-February).
The chart includes long-term average reference lines from 1991-2020 for comparison.
{CONTENT_SEPARATORS['image_end']}

=== TEXT PROCESSING ===

REGULAR TEXT PRESERVATION:
- Preserve original text content exactly as written
- Maintain original language (Czech text stays in Czech)
- Keep paragraph structure and formatting
- Preserve headings, subheadings, and document structure
- Do NOT translate regular text content to English
- Only use English for describing tables and graphs

TEXT ORGANIZATION:
- Wrap all regular text content with {CONTENT_SEPARATORS['text_start']} and {CONTENT_SEPARATORS['text_end']}
- Maintain paragraph breaks within text blocks
- Preserve bullet points, numbered lists, and formatting
- Keep contextual information for all statements
- Maintain document hierarchy and section organization

SECTION IDENTIFICATION:
- Clearly identify different sections and subsections
- Use descriptive headers that will be preserved in chunks
- Group related information logically
- Use {CONTENT_SEPARATORS['page_separator']} between pages

TEXT PROCESSING EXAMPLE:
For regular document text content:

{CONTENT_SEPARATORS['text_start']}
Praha je hlavní město České republiky a největší město v zemi. Leží na řece Vltavě ve středních Čechách.

Město má rozlohu 496 km² a počet obyvatel přesahuje 1,3 milionu. Praha je významným kulturním, politickým a ekonomickým centrem střední Evropy.

Historické centrum Prahy je zapsáno na seznamu světového dědictví UNESCO.
{CONTENT_SEPARATORS['text_end']}

=== CONTENT ORGANIZATION ===

CHUNKING-FRIENDLY STRUCTURE:
- Write in complete sentences that can stand alone
- Avoid references like "the table above" or "as shown in the graph"
- Include all necessary context within each paragraph
- Ensure every data point is traceable to its source and meaning

MEASUREMENT PRESERVATION:
- Always include full measurement context
- Specify what is being measured (quantities, percentages, rates)
- Include time periods (daily, monthly, annual)
- Preserve statistical terms (average, maximum, minimum, per capita, total)

ENTITY AND GEOGRAPHIC CONTEXT:
- Always specify the entity, region, or organization being discussed
- Include both full names and codes/abbreviations
- Maintain hierarchical relationships (parent → child entities)
- Preserve organizational or geographic classifications

VERBOSE DESCRIPTIONS:
- Err on the side of being overly descriptive rather than concise
- Each sentence should be understandable without additional context
- Include redundant information to ensure context preservation

CRITICAL SUCCESS CRITERIA:
- Any 500-character chunk should contain enough context to be meaningful
- No orphaned numbers without their measurement context
- Every data point traceable to its source table/graph and meaning
- All entity and temporal context preserved throughout
- Content types clearly separated with specified markers

Process this document with extreme attention to context preservation for optimal performance in chunked semantic search scenarios.
"""

def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('DEBUG', '0')
    if debug_mode == '1':
        print(f"[PDF-CHROMADB-DEBUG] {msg}")
        import sys
        sys.stdout.flush()

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
    Robust text chunking optimized for LlamaParse formatted content.
    Preserves complete sentences and numerical data integrity.
    
    Args:
        text: Text to chunk (LlamaParse formatted content)
        max_chunk_size: Maximum characters per chunk (default: 800)
        overlap: Character overlap between chunks (default: 100)
        
    Returns:
        List of text chunks that preserve content integrity
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    # First, try to split by LlamaParse content type markers
    # These represent natural content boundaries
    content_boundaries = ['\n[TABLE]\n', '\n[/TABLE]\n', '\n[IMAGE]\n', '\n[/IMAGE]\n']
    
    # If we have clear content boundaries, split there first
    sections = [text]  # Start with whole text
    for boundary in content_boundaries:
        new_sections = []
        for section in sections:
            if boundary in section:
                new_sections.extend(section.split(boundary))
            else:
                new_sections.append(section)
        sections = new_sections
    
    # Process each section
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if len(section) <= max_chunk_size:
            # Section fits in one chunk
            if len(section) >= MIN_CHUNK_SIZE:
                chunks.append(section)
        else:
            # Section needs to be split further
            section_chunks = _split_large_section(section, max_chunk_size, overlap)
            chunks.extend(section_chunks)
    
    # Filter out chunks that are too small
    final_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_SIZE]
    
    return final_chunks

def _split_large_section(text: str, max_size: int, overlap: int) -> List[str]:
    """
    Split a large section of text while preserving LlamaParse sentence integrity.
    This version is much more conservative and preserves numerical data.
    """
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    
    # First try: Split by double newlines (paragraph boundaries)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed size limit
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if substantial
                if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current paragraph
                if len(paragraph) <= max_size:
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split it conservatively
                    para_chunks = _split_paragraph_conservatively(paragraph, max_size)
                    chunks.extend(para_chunks[:-1])  # Add all but last
                    current_chunk = para_chunks[-1] if para_chunks else ""
        
        # Add final chunk
        if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
        
        if chunks:
            return chunks
    
    # Fallback: Conservative sentence splitting that preserves numerical context
    return _split_paragraph_conservatively(text, max_size)

def _split_paragraph_conservatively(text: str, max_size: int) -> List[str]:
    """
    Very conservative splitting that preserves LlamaParse formatted sentences.
    Prioritizes keeping numerical data with its full context.
    """
    if len(text) <= max_size:
        return [text]
    
    # Split only at sentence boundaries that are clearly safe
    # Look for patterns like ". In the" which indicate new sentence starts in LlamaParse
    import re
    
    # Find sentence boundaries in LlamaParse format
    # Pattern: period + space + capital letter (but not in middle of numbers)
    sentence_pattern = r'\. (?=[A-Z][a-z])'  # Period followed by space and capital+lowercase
    
    sentences = re.split(sentence_pattern, text)
    
    if len(sentences) <= 1:
        # No clear sentence boundaries found, split by character limit as last resort
        return _split_by_character_limit(text, max_size)
    
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back except for last sentence
        if i < len(sentences) - 1 and not sentence.endswith('.'):
            sentence = sentence + "."
        
        # Check if adding this sentence would exceed limit
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= max_size:
            current_chunk = test_chunk
        else:
            # Save current chunk if substantial
            if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with current sentence
            if len(sentence) <= max_size:
                current_chunk = sentence
            else:
                # Single sentence too long - split by character limit
                sentence_chunks = _split_by_character_limit(sentence, max_size)
                chunks.extend(sentence_chunks[:-1])
                current_chunk = sentence_chunks[-1] if sentence_chunks else ""
    
    # Add final chunk
    if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def _split_by_character_limit(text: str, max_size: int) -> List[str]:
    """
    Last resort: split by character limit while trying to break at word boundaries.
    """
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to break at a word boundary near the end
        break_pos = text.rfind(' ', start, end)
        if break_pos > start:
            end = break_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
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
        "min_length": float('inf'),
        "max_length": 0,
        "chunks_with_context": 0,
        "chunks_with_numbers": 0,
        "potential_issues": []
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
            metrics["potential_issues"].append(f"Chunk {i}: Too small ({chunk_len} chars)")
        elif chunk_len > MAX_CHUNK_SIZE * 1.5:  # Allow some flexibility
            metrics["too_large_chunks"] += 1
            metrics["potential_issues"].append(f"Chunk {i}: Very large ({chunk_len} chars)")
        
        # Check for context preservation
        context_indicators = ["table", "chart", "graph", "measured", "recorded", "hectares", "mm", "2023", "2022", "2021"]
        if any(indicator in chunk.lower() for indicator in context_indicators):
            metrics["chunks_with_context"] += 1
        
        # Check for numerical data
        import re
        if re.search(r'\d+', chunk):
            metrics["chunks_with_numbers"] += 1
    
    metrics["avg_length"] = total_length / len(chunks) if chunks else 0
    metrics["min_length"] = metrics["min_length"] if metrics["min_length"] != float('inf') else 0
    
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

def debug_chunks_for_numerical_data(chunks: List[str], target_numbers: List[str] = None) -> Dict[str, Any]:
    """
    Debug function to verify that numerical data is preserved in chunks.
    
    Args:
        chunks: List of text chunks to analyze
        target_numbers: Specific numbers to look for (optional)
        
    Returns:
        Dictionary with detailed analysis of numerical data preservation
    """
    if target_numbers is None:
        target_numbers = ["1042843", "356687", "1574854", "712555", "730947"]  # Warsaw data from example
    
    analysis = {
        "total_chunks": len(chunks),
        "chunks_with_any_numbers": 0,
        "target_numbers_found": {},
        "chunk_details": []
    }
    
    # Initialize target number tracking
    for num in target_numbers:
        analysis["target_numbers_found"][num] = []
    
    for i, chunk in enumerate(chunks):
        chunk_info = {
            "chunk_index": i,
            "length": len(chunk),
            "contains_numbers": bool(re.search(r'\d+', chunk)),
            "target_numbers_in_chunk": [],
            "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
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
            "chunk_indices": chunk_indices
        }
    
    return analysis

def print_numerical_debug_report(chunks: List[str], target_numbers: List[str] = None) -> None:
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
        count_info = f"(in {summary['count']} chunks)" if summary['count'] > 1 else ""
        print(f"  {num}: {status} {count_info}")
        if summary["chunk_indices"]:
            print(f"    └─ Chunk indices: {summary['chunk_indices']}")
    
    print()
    print("📋 CHUNKS WITH TARGET NUMBERS:")
    print("-" * 40)
    for chunk_info in analysis["chunk_details"]:
        if chunk_info["target_numbers_in_chunk"]:
            print(f"  Chunk {chunk_info['chunk_index']} ({chunk_info['length']} chars):")
            print(f"    Numbers: {', '.join(chunk_info['target_numbers_in_chunk'])}")
            print(f"    Preview: {chunk_info['preview']}")
            print()

#==============================================================================
# FILE I/O FUNCTIONS
#==============================================================================
def save_parsed_text_to_file(text: str, file_path: str) -> None:
    """
    Save parsed text to a file and provide content analysis.
    
    Args:
        text: The parsed text content with separators
        file_path: Path where to save the text file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Analyze content structure
        content_types = extract_content_by_type(text)
        
        debug_print(f"💾 Successfully saved parsed text to: {file_path}")
        print(f"✅ Parsed text saved to: {file_path}")
        print(f"📊 Content analysis:")
        print(f"   📋 Tables: {len(content_types['tables'])}")
        print(f"   🖼️  Images/Graphs: {len(content_types['images'])}")
        print(f"   📝 Text sections: {len(content_types['text'])}")
        
        # Check for proper separator usage
        separator_count = text.count(CONTENT_SEPARATORS['page_separator'])
        print(f"   📄 Pages: {separator_count + 1}")
        
        if any(content_types.values()):
            print(f"✅ Content properly separated with custom markers")
        else:
            print(f"⚠️  No content separators found - check LlamaParse instructions")
            
    except Exception as e:
        debug_print(f"❌ Error saving parsed text: {str(e)}")
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
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        debug_print(f"💾 Successfully loaded parsed text from: {file_path}")
        print(f"✅ Loaded parsed text from: {file_path}")
        return text
    except FileNotFoundError:
        debug_print(f"❌ Parsed text file not found: {file_path}")
        return None
    except Exception as e:
        debug_print(f"❌ Error loading parsed text: {str(e)}")
        raise

def create_documents_from_text(text: str, source_file: str, parsing_method: str) -> List[Dict[str, Any]]:
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
    debug_print(f"🏗️ Creating document structure from parsed text ({len(text)} characters)")
    debug_print(f"📄 Source: {source_file}, Method: {parsing_method}")
    
    # Analyze content structure before splitting
    content_analysis = extract_content_by_type(text)
    debug_print(f"📊 Content analysis: {len(content_analysis['tables'])} tables, {len(content_analysis['images'])} images, {len(content_analysis['text'])} text sections")
    
    pages = []
    
    # Split by our custom page separator first (LlamaParse)
    if CONTENT_SEPARATORS['page_separator'] in text:
        debug_print(f"📄 Found LlamaParse page separators")
        page_texts = text.split(CONTENT_SEPARATORS['page_separator'])
    # Fallback separators for other methods
    elif "\n---\n" in text:
        debug_print(f"📄 Found standard page separators (---)")
        page_texts = text.split("\n---\n")
    elif "\n=================\n" in text:
        debug_print(f"📄 Found extended page separators (=================)")
        page_texts = text.split("\n=================\n")
    else:
        # If no clear separators, treat as one large document
        debug_print(f"📄 No page separators found, treating as single document")
        page_texts = [text]
    
    debug_print(f"✂️ Split text into {len(page_texts)} sections")
    
    for page_num, page_text in enumerate(page_texts, 1):
        if page_text.strip():  # Only add non-empty pages
            # Clean the page text of separator artifacts that might interfere with chunking
            cleaned_text = clean_separator_artifacts(page_text.strip())
            
            # Analyze this section's content
            section_has_tables = '[TABLE CONTENT BEGINS]' in cleaned_text
            section_has_images = '[IMAGE/CHART CONTENT BEGINS]' in cleaned_text
            
            page_data = {
                'text': cleaned_text,
                'page_number': page_num,
                'char_count': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'source_file': source_file,
                'parsing_method': parsing_method,
                'has_tables': section_has_tables,
                'has_images': section_has_images
            }
            pages.append(page_data)
            
            debug_print(f"📄 Section {page_num}: {len(cleaned_text)} chars, tables: {section_has_tables}, images: {section_has_images}")
    
    debug_print(f"✅ Created {len(pages)} document sections from parsed text")
    
    # Summary statistics
    total_chars = sum(p['char_count'] for p in pages)
    sections_with_tables = sum(1 for p in pages if p['has_tables'])
    sections_with_images = sum(1 for p in pages if p['has_images'])
    
    debug_print(f"📊 Document summary:")
    debug_print(f"📊   - Total characters: {total_chars}")
    debug_print(f"📊   - Sections with tables: {sections_with_tables}/{len(pages)}")
    debug_print(f"📊   - Sections with images: {sections_with_images}/{len(pages)}")
    debug_print(f"📊   - Average section size: {total_chars/len(pages):.0f} characters")
    
    return pages

def clean_separator_artifacts(text: str) -> str:
    """
    Intelligently clean text of separator artifacts while preserving content structure.
    Maintains content type boundaries for better chunking.
    
    Args:
        text: Text that may contain content separators
        
    Returns:
        Cleaned text with separators replaced by content type markers
    """
    cleaned_text = text
    
    # Replace separator markers with minimal content type indicators
    separator_replacements = {
        CONTENT_SEPARATORS['table_start']: '\n[TABLE]\n',
        CONTENT_SEPARATORS['table_end']: '\n[/TABLE]\n',
        CONTENT_SEPARATORS['image_start']: '\n[IMAGE]\n',
        CONTENT_SEPARATORS['image_end']: '\n[/IMAGE]\n',
        CONTENT_SEPARATORS['text_start']: '\n',  # Replace with newline to preserve content
        CONTENT_SEPARATORS['text_end']: '\n'     # Replace with newline to preserve content
    }
    
    for separator, replacement in separator_replacements.items():
        cleaned_text = cleaned_text.replace(separator, replacement)
    
    # Clean up excessive whitespace while preserving paragraph breaks
    import re
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)
    
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
    content_types = {
        'tables': [],
        'images': [],
        'text': []
    }
    
    # Extract tables
    table_pattern = f"{CONTENT_SEPARATORS['table_start']}(.*?){CONTENT_SEPARATORS['table_end']}"
    import re
    tables = re.findall(table_pattern, text, re.DOTALL)
    content_types['tables'] = [table.strip() for table in tables]
    
    # Extract images
    image_pattern = f"{CONTENT_SEPARATORS['image_start']}(.*?){CONTENT_SEPARATORS['image_end']}"
    images = re.findall(image_pattern, text, re.DOTALL)
    content_types['images'] = [image.strip() for image in images]
    
    # Extract text
    text_pattern = f"{CONTENT_SEPARATORS['text_start']}(.*?){CONTENT_SEPARATORS['text_end']}"
    texts = re.findall(text_pattern, text, re.DOTALL)
    content_types['text'] = [text_content.strip() for text_content in texts]
    
    debug_print(f"📊 Extracted content: {len(content_types['tables'])} tables, {len(content_types['images'])} images, {len(content_types['text'])} text sections")
    
    return content_types

#==============================================================================
# PDF PROCESSING - LLAMAPARSE ONLY
#==============================================================================
def extract_text_with_llamaparse(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using LlamaParse for superior table handling.
    Now includes comprehensive progress tracking and timeout handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    debug_print(f"📄 Opening PDF with LlamaParse: {pdf_path}")
    
    try:
        # Try to import llama_parse
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError("LlamaParse not installed. Install with: pip install llama-parse")
        
        # Get API key from .env file or environment
        api_key = LLAMAPARSE_API_KEY or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LlamaParse API key not found. Set LLAMAPARSE_API_KEY in .env file or LLAMA_CLOUD_API_KEY environment variable")
        
        # Check PDF file size for time estimation
        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        estimated_time = max(30, pdf_size * 5)  # Rough estimate: 5 seconds per MB, minimum 30 seconds
        
        print(f"\n🚀 Starting LlamaParse processing...")
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📊 Size: {pdf_size:.1f} MB")
        print(f"⏱️  Estimated time: {estimated_time:.0f} seconds")
        print(f"🌐 API Status: LlamaParse is experiencing performance issues - this may take longer than usual")
        print(f"💡 Tip: You can monitor progress at https://cloud.llamaindex.ai/parse (History tab)")
        
        # Initialize LlamaParse with the newer parameter system
        # Note: There are currently known issues with these parameters (GitHub issue #620)
        # If the new parameters don't work, fallback to parsing_instruction may be needed
        
        comprehensive_instructions = get_llamaparse_instructions()
        
        try:
            # Try using the newer parameter system first
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                system_prompt="You are an expert document parser specializing in converting complex tabular data into chunking-friendly formats for semantic search systems and you are also an expert in describing graphs and charts in details. But also You are able to parse back normal text as is.",
                user_prompt=comprehensive_instructions,
                page_separator=CONTENT_SEPARATORS['page_separator'],
                verbose=True
            )
            debug_print("🔧 Using newer LlamaParse parameter system (system_prompt + user_prompt)")
            
        except TypeError as e:
            debug_print(f"⚠️ Newer parameters not available ({e}), falling back to parsing_instruction")
            # Fallback to the older parameter if new ones aren't available
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                parsing_instruction=comprehensive_instructions,
                page_separator=CONTENT_SEPARATORS['page_separator'],
                verbose=True
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
            
            print(f"\r⏱️  {status} Elapsed: {minutes:02d}:{seconds:02d}", end="", flush=True)
        
        # Start periodic progress updates
        import threading
        import signal
        
        def progress_updater():
            while not getattr(progress_updater, 'stop', False):
                elapsed = time.time() - start_time
                show_progress_update(elapsed)
                time.sleep(2)  # Update every 2 seconds
        
        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()
        
        try:
            # Parse the document with timeout handling
            print(f"\n📤 Document uploaded to LlamaParse. Processing...")
            print(f"🔍 You can monitor detailed progress at: https://cloud.llamaindex.ai/parse")
            
            documents = parser.load_data(pdf_path)
            
            # Stop progress updates
            progress_updater.stop = True
            elapsed_time = time.time() - start_time
            print(f"\r✅ LlamaParse completed successfully! Duration: {elapsed_time:.1f} seconds")
            
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
            text = document.text if hasattr(document, 'text') else str(document)
            
            if not text.strip():
                debug_print(f"⚠️ Document {doc_idx + 1} contains no text")
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
            print(f"   ✅ Section {doc_idx + 1}: {len(text):,} characters, {len(text.split()):,} words")
            debug_print(f"📄 Document {doc_idx + 1}: {len(text)} characters, {len(text.split())} words (LlamaParse)")
        
        # Final summary
        total_chars = sum(p['char_count'] for p in pages_data)
        total_words = sum(p['word_count'] for p in pages_data)
        
        print(f"\n🎉 LlamaParse Processing Complete!")
        print(f"   📄 Sections: {len(pages_data)}")
        print(f"   📝 Total characters: {total_chars:,}")
        print(f"   🔤 Total words: {total_words:,}")
        print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"   📊 Processing speed: {total_chars/elapsed_time:.0f} chars/sec")
        
        debug_print(f"🏗️ Successfully extracted text from {len(pages_data)} documents using LlamaParse")
        debug_print(f"Successfully extracted text from {len(pages_data)} documents using LlamaParse")
        return pages_data
        
    except Exception as e:
        debug_print(f"Error extracting text with LlamaParse: {str(e)}")
        print(f"\n❌ LlamaParse failed completely. Error: {str(e)}")
        # No fallback - just raise the error
        raise

def extract_text_with_llamaparse_async_monitoring(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Alternative LlamaParse extraction with enhanced progress monitoring using the raw API.
    This version can check job status and provide better feedback on parsing progress.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    import requests
    import json
    
    debug_print(f"Opening PDF with LlamaParse (async monitoring): {pdf_path}")
    
    try:
        # Get API key
        api_key = LLAMAPARSE_API_KEY or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError("LlamaParse API key not found")
        
        # Check PDF file size for time estimation
        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        estimated_time = max(30, pdf_size * 5)  # Rough estimate: 5 seconds per MB, minimum 30 seconds
        
        print(f"\n🚀 Starting LlamaParse processing (Enhanced Monitoring)...")
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📊 Size: {pdf_size:.1f} MB")
        print(f"⏱️  Estimated time: {estimated_time:.0f} seconds")
        
        # Step 1: Upload file and start parsing job
        print(f"\n📤 Uploading file to LlamaParse...")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'accept': 'application/json'
        }
        
        # Prepare comprehensive parsing instructions
        parsing_instructions = get_llamaparse_instructions()
        
        # Prepare form data for upload
        with open(pdf_path, 'rb') as file:
            files = {
                'file': (os.path.basename(pdf_path), file, 'application/pdf')
            }
            
            data = {
                'parsing_instruction': parsing_instructions,
                'result_type': 'markdown',
                'page_separator': CONTENT_SEPARATORS['page_separator'],
                'verbose': 'true'
            }
            
            response = requests.post(
                'https://api.cloud.llamaindex.ai/api/v1/parsing/upload',
                headers=headers,
                files=files,
                data=data,
                timeout=60  # 60 second timeout for upload
            )
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
        
        job_data = response.json()
        job_id = job_data.get('id')
        
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
                    f'https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}',
                    headers=headers,
                    timeout=10
                )
                
                if status_response.status_code != 200:
                    print(f"\n⚠️  Status check failed: {status_response.status_code}")
                    time.sleep(status_check_interval)
                    continue
                
                job_status = status_response.json()
                current_status = job_status.get('status', 'unknown')
                
                # Update progress display
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                
                if current_status != last_status:
                    print(f"\n📊 Status changed: {current_status}")
                    last_status = current_status
                
                # Status-specific progress messages
                if current_status == 'PENDING':
                    status_msg = "🟡 Queued - waiting to start processing..."
                elif current_status == 'PROCESSING':
                    status_msg = "🟠 Processing document pages..."
                elif current_status == 'SUCCESS':
                    print(f"\n✅ Parsing completed successfully! Duration: {elapsed_time:.1f} seconds")
                    break
                elif current_status == 'ERROR':
                    error_msg = job_status.get('error', 'Unknown error')
                    raise Exception(f"Parsing job failed: {error_msg}")
                else:
                    status_msg = f"🔵 Status: {current_status}"
                
                print(f"\r⏱️  {status_msg} Elapsed: {minutes:02d}:{seconds:02d}", end="", flush=True)
                
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
            f'https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/markdown',
            headers=headers,
            timeout=30
        )
        
        if result_response.status_code != 200:
            raise Exception(f"Failed to retrieve results: {result_response.status_code} - {result_response.text}")
        
        result_data = result_response.json()
        
        # Process results into our expected format
        pages_data = []
        combined_text_parts = []
        
        # Extract text from result data
        if 'markdown' in result_data:
            text_content = result_data['markdown']
        elif 'text' in result_data:
            text_content = result_data['text']
        else:
            # Try to extract from pages if available
            pages = result_data.get('pages', [])
            if pages:
                text_content = CONTENT_SEPARATORS['page_separator'].join([
                    page.get('md', page.get('text', '')) for page in pages
                ])
            else:
                raise Exception(f"No usable content found in results: {list(result_data.keys())}")
        
        print(f"📊 Processing retrieved content...")
        
        # Split content by pages if available
        if CONTENT_SEPARATORS['page_separator'] in text_content:
            page_texts = text_content.split(CONTENT_SEPARATORS['page_separator'])
        else:
            page_texts = [text_content]
        
        for page_num, page_text in enumerate(page_texts, 1):
            if page_text.strip():
                cleaned_text = clean_separator_artifacts(page_text.strip())
                
                page_info = {
                    'text': cleaned_text,
                    'page_number': page_num,
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'source_file': os.path.basename(pdf_path),
                    'parsing_method': 'llamaparse_async'
                }
                
                pages_data.append(page_info)
                combined_text_parts.append(cleaned_text)
                print(f"   ✅ Page {page_num}: {len(cleaned_text):,} characters, {len(cleaned_text.split()):,} words")
        
        # Save combined text to file
        if combined_text_parts:
            combined_text = CONTENT_SEPARATORS['page_separator'].join(combined_text_parts)
            save_parsed_text_to_file(combined_text, str(PARSED_TEXT_PATH))
        
        # Final summary
        total_chars = sum(p['char_count'] for p in pages_data)
        total_words = sum(p['word_count'] for p in pages_data)
        final_elapsed = time.time() - start_time
        
        print(f"\n🎉 LlamaParse Async Processing Complete!")
        print(f"   📄 Pages: {len(pages_data)}")
        print(f"   📝 Total characters: {total_chars:,}")
        print(f"   🔤 Total words: {total_words:,}")
        print(f"   ⏱️  Total time: {final_elapsed:.1f} seconds")
        print(f"   📊 Processing speed: {total_chars/final_elapsed:.0f} chars/sec")
        print(f"   🆔 Job ID: {job_id}")
        
        debug_print(f"Successfully extracted text from {len(pages_data)} pages using LlamaParse async")
        return pages_data
        
    except Exception as e:
        debug_print(f"Error with LlamaParse async monitoring: {str(e)}")
        print(f"\n❌ LlamaParse async monitoring failed. Error: {str(e)}")
        # No fallback - just raise the error
        raise

def process_parsed_text_to_chunks(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    
    debug_print(f"Processing {len(pages_data)} sections/pages for chunking")
    
    for page_data in pages_data:
        text = page_data['text']
        page_num = page_data.get('page_number', 1)
        parsing_method = page_data.get('parsing_method', 'unknown')
        
        debug_print(f"Processing section {page_num}: {len(text)} characters")
        
        # Use semantic-aware chunking that respects LlamaParse structure
        page_chunks = smart_text_chunking(text)
        
        debug_print(f"Section {page_num} split into {len(page_chunks)} chunks")
        
        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Check token count and split only if absolutely necessary
            token_count = num_tokens_from_string(chunk_text)
            
            if token_count <= MAX_TOKENS:
                # Chunk is within token limits, use as-is
                chunk_data = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'page_number': page_num,
                    'chunk_index': chunk_idx,
                    'token_chunk_index': 0,
                    'total_page_chunks': len(page_chunks),
                    'total_token_chunks': 1,
                    'char_count': len(chunk_text),
                    'token_count': token_count,
                    'source_file': page_data['source_file'],
                    'parsing_method': parsing_method,
                    'doc_hash': get_document_hash(chunk_text)
                }
                
                all_chunks.append(chunk_data)
                chunk_id += 1
            else:
                # Only split by tokens if chunk exceeds token limit
                debug_print(f"Chunk {chunk_id} exceeds token limit ({token_count} > {MAX_TOKENS}), splitting...")
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
                        'parsing_method': parsing_method,
                        'doc_hash': get_document_hash(token_chunk)
                    }
                    
                    all_chunks.append(chunk_data)
                    chunk_id += 1
    
    debug_print(f"Created {len(all_chunks)} chunks from {len(pages_data)} sections")
    
    # Log chunking statistics
    if all_chunks:
        token_counts = [chunk['token_count'] for chunk in all_chunks]
        char_counts = [chunk['char_count'] for chunk in all_chunks]
        
        debug_print(f"Chunk statistics:")
        debug_print(f"  - Average tokens: {sum(token_counts)/len(token_counts):.1f}")
        debug_print(f"  - Average characters: {sum(char_counts)/len(char_counts):.1f}")
        debug_print(f"  - Token range: {min(token_counts)} - {max(token_counts)}")
        debug_print(f"  - Character range: {min(char_counts)} - {max(char_counts)}")
        
        # Validate chunk quality
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        quality_metrics = validate_chunk_quality(chunk_texts)
        
        debug_print(f"Chunk quality metrics:")
        debug_print(f"  - Quality score: {quality_metrics.get('quality_score', 0)}/100")
        debug_print(f"  - Chunks with context: {quality_metrics.get('chunks_with_context', 0)}/{len(all_chunks)}")
        debug_print(f"  - Chunks with numbers: {quality_metrics.get('chunks_with_numbers', 0)}/{len(all_chunks)}")
        
        # Add numerical data preservation debugging
        print("\n🔍 DEBUGGING NUMERICAL DATA PRESERVATION:")
        print_numerical_debug_report(chunk_texts)
        
        if quality_metrics.get('potential_issues'):
            debug_print(f"  - Potential issues found: {len(quality_metrics['potential_issues'])}")
            for issue in quality_metrics['potential_issues'][:5]:  # Show first 5 issues
                debug_print(f"    * {issue}")
            if len(quality_metrics['potential_issues']) > 5:
                debug_print(f"    * ... and {len(quality_metrics['potential_issues']) - 5} more issues")
    
    return all_chunks

# Keep the old function name as an alias for backward compatibility
def process_pdf_pages_to_chunks(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use process_parsed_text_to_chunks() instead.
    This function is kept for backward compatibility.
    """
    debug_print("Warning: process_pdf_pages_to_chunks() is deprecated. Use process_parsed_text_to_chunks() instead.")
    return process_parsed_text_to_chunks(pages_data)

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
            if LLAMAPARSE_ENHANCED_MONITORING:
                pages_data = extract_text_with_llamaparse_async_monitoring(pdf_path)
            else:
                pages_data = extract_text_with_llamaparse(pdf_path)
        else:
            raise ValueError(f"Only 'llamaparse' parsing method is supported. Current setting: {PDF_PARSING_METHOD}")
            
        metrics.total_pages = len(pages_data)
        
        if not pages_data:
            raise ValueError("No text found in PDF")
        
        # Process pages into chunks
        chunks_data = process_parsed_text_to_chunks(pages_data)
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
                    combined_text_parts.append(page_data['text'])
                
                combined_text = CONTENT_SEPARATORS['page_separator'].join(combined_text_parts)
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
        monitoring_type = "Enhanced (API monitoring)" if LLAMAPARSE_ENHANCED_MONITORING else "Standard (thread monitoring)"
        print(f"📊 LlamaParse monitoring: {monitoring_type}")
    print(f"🗄️  ChromaDB location: {CHROMA_DB_PATH}")
    print()
    print("🔧 Operations to perform:")
    print(f"   1. Parse with LlamaParse: {'✅ Yes' if PARSE_WITH_LLAMAPARSE else '❌ No'}")
    print(f"   2. Chunk and Store:      {'✅ Yes' if CHUNK_AND_STORE else '❌ No'}")
    print(f"   3. Testing/Search:       {'✅ Yes' if DO_TESTING else '❌ No'}")
    print("=" * 50)
    
    # =====================================================================
    # OPERATION 1: PARSE WITH LLAMAPARSE (PARALLEL PROCESSING)
    # =====================================================================
    if PARSE_WITH_LLAMAPARSE:
        print(f"\n📄 OPERATION 1: Parsing {len(PDF_FILENAMES)} PDFs with {PDF_PARSING_METHOD} (PARALLEL)")
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
        
        print(f"✅ All {len(PDF_FILENAMES)} PDF files found. Starting parallel processing...")
        
        # Process PDFs in parallel using ThreadPoolExecutor
        max_workers = min(len(PDF_FILENAMES), 4)  # Limit to 4 concurrent LlamaParse requests
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
            print(f"❌ No PDFs were successfully processed. Cannot continue to next operations.")
            return
            
    else:
        print(f"\n⏭️  OPERATION 1: Skipping PDF parsing (using existing parsed text files)")
    
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
                    print(f"⚠️  Warning: Parsed text file not found for {pdf_filename}: {parsed_text_path}")
                    print(f"💡 Skipping {pdf_filename} - run with PARSE_WITH_LLAMAPARSE = 1 first")
                    continue
                
                # Load parsed text and create document structure
                parsed_text = load_parsed_text_from_file(str(parsed_text_path))
                pages_data = create_documents_from_text(parsed_text, pdf_filename, PDF_PARSING_METHOD)
                all_pages_data.extend(pages_data)
                print(f"📊 Loaded {len(pages_data)} pages from {pdf_filename}")
            
            if not all_pages_data:
                print(f"❌ No parsed text files found. Please run with PARSE_WITH_LLAMAPARSE = 1 first.")
                return
            
            print(f"📊 Total pages from all PDFs: {len(all_pages_data)}")
            
            # Process pages into chunks
            chunks_data = process_parsed_text_to_chunks(all_pages_data)
            debug_print(f"Created {len(chunks_data)} chunks for processing")
            
            # Initialize ChromaDB
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            try:
                collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                debug_print(f"Created new ChromaDB collection: {COLLECTION_NAME}")
            except Exception:
                collection = client.get_collection(name=COLLECTION_NAME)
                debug_print(f"Using existing ChromaDB collection: {COLLECTION_NAME}")
            
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
            
            if new_chunks:
                # Initialize embedding client
                if get_azure_embedding_model is None:
                    raise ImportError("Azure embedding model not available. Check your imports.")
                
                embedding_client = get_azure_embedding_model()
                
                # Process chunks with progress bar
                processed_chunks = 0
                failed_chunks = 0
                
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
                                model=AZURE_EMBEDDING_DEPLOYMENT
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
                            
                            processed_chunks += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            debug_print(f"Error processing chunk {chunk_data['id']}: {str(e)}")
                            failed_chunks += 1
                            pbar.update(1)
                            continue
                
                print(f"✅ Successfully processed and stored chunks:")
                print(f"   📊 Processed: {processed_chunks}")
                print(f"   ❌ Failed: {failed_chunks}")
                print(f"   📈 Success rate: {(processed_chunks/(processed_chunks+failed_chunks)*100):.1f}%")
            else:
                print(f"ℹ️  No new chunks to process (all chunks already exist in ChromaDB)")
            
        except Exception as e:
            print(f"❌ Error during chunking and storage: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"\n⏭️  OPERATION 2: Skipping chunking and storage (using existing ChromaDB)")
    
    # =====================================================================
    # OPERATION 3: TESTING/SEARCH
    # =====================================================================
    if DO_TESTING:
        print(f"\n🔍 OPERATION 3: Testing search functionality")
        
        try:
            # Load ChromaDB collection
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
                return
            
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
                    content = result['text'][:2000]
                    if len(result['text']) > 2000:
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
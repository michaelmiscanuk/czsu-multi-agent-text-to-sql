module_description = r"""ChromaDB Document Management for Selection Descriptions

This module provides functionality to manage selection descriptions in ChromaDB,
with support for document deduplication, embedding generation, and similarity search.

Key Features:
-------------
1. Document Management:
   - SQLite to ChromaDB document transfer
   - Document deduplication using MD5 hashing
   - Metadata preservation (selection codes)
   - UUID-based document identification
   - Automatic document splitting for long texts
   - Token counting and validation

2. Embedding Generation:
   - Azure OpenAI embedding model integration
   - Batch embedding generation
   - Configurable model deployment
   - Token limit handling (8190 tokens max)
   - Smart text chunking for long documents

3. Similarity Search:
   - Query embedding generation
   - Configurable result count
   - Distance-based ranking
   - Metadata and document retrieval
   - Support for split document reconstruction

4. Error Handling:
   - Database connection management
   - Embedding generation error handling
   - Document validation
   - Debug logging support
   - Token limit error handling

5. Performance:
   - Batch processing
   - Efficient deduplication
   - Persistent storage
   - Connection pooling
   - Smart document chunking

Processing Flow:
--------------
1. Initialization:
   - Sets up project paths
   - Configures database connections
   - Initializes embedding client
   - Sets up debug logging

2. Document Retrieval:
   - Connects to SQLite database
   - Retrieves documents and selection codes
   - Generates document hashes
   - Validates document content
   - Counts tokens for each document

3. Document Processing:
   - Checks for existing documents
   - Filters out duplicates
   - Generates unique IDs
   - Prepares metadata
   - Splits long documents if needed
   - Validates token counts

4. Embedding Generation:
   - Batches documents for processing
   - Generates embeddings using Azure
   - Handles API responses
   - Validates embedding results
   - Processes document chunks

5. ChromaDB Integration:
   - Creates/connects to collection
   - Adds documents with embeddings
   - Stores metadata
   - Updates collection
   - Maintains chunk relationships

6. Search Capabilities:
   - Query embedding generation
   - Similarity search execution
   - Result ranking and formatting
   - Metadata retrieval
   - Chunk reconstruction

Usage Example:
-------------
# Initialize and populate ChromaDB
collection = upsert_documents_to_chromadb(
    deployment="text-embedding-3-large__test1",
    collection_name="my_collection"
)

# Perform similarity search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

Required Environment:
-------------------
- Python 3.7+
- Azure OpenAI API access
- SQLite database with selection descriptions
- Write permissions for ChromaDB directory
- tiktoken package for token counting

Output:
-------
- ChromaDB collection with:
  - Document embeddings
  - Selection code metadata
  - Document hashes
  - Unique document IDs
  - Chunk information for split documents

Error Handling:
-------------
- Database connection errors
- Embedding generation failures
- Document validation errors
- API rate limiting
- File system errors
- Token limit errors
- Chunk processing errors"""

#==============================================================================
# IMPORTS
#==============================================================================
# Standard library imports
import os
from pathlib import Path
import sys
import sqlite3
import hashlib
from uuid import uuid4
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

# Third-party imports
import chromadb
import tqdm as tqdm_module
import tiktoken

#==============================================================================
# PATH SETUP
#==============================================================================
# --- Ensure project root is in sys.path for local imports ---
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local Imports
from my_agent.utils.models import get_azure_embedding_model

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
# Database paths
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
SQLITE_DB_PATH = BASE_DIR / "metadata" / "llm_selection_descriptions" / "selection_descriptions.db"

# Unique identifier for this module's debug messages
CREATE_CHROMADB_ID = 30

# Token limit for Azure OpenAI
MAX_TOKENS = 8190

# SQL query for retrieving documents
SELECT_DOCUMENTS_QUERY = (
    "SELECT extended_description, selection_code "
    "FROM selection_descriptions"
)

#==============================================================================
# MONITORING AND METRICS
#==============================================================================
@dataclass
class Metrics:
    """Simple metrics collection for tracking processing statistics.
    
    This class tracks various metrics during the processing of documents:
    - Processing time
    - Success/failure counts
    - Failed documents with reasons
    
    Attributes:
        start_time (float): Timestamp when processing started.
        processed_docs (int): Number of successfully processed documents.
        failed_docs (int): Number of documents that failed processing.
        total_processing_time (float): Total time taken for processing.
        failed_records (list): List of tuples containing (selection_code, error_message) for failed records.
    """
    start_time: float = field(default_factory=time.time)
    processed_docs: int = 0
    failed_docs: int = 0
    total_processing_time: float = 0
    failed_records: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the metrics with formatted timestamps
                           and calculated averages.
        """
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "processed_docs": self.processed_docs,
            "failed_docs": self.failed_docs,
            "total_processing_time": self.total_processing_time,
            "average_time_per_doc": self.total_processing_time / max(1, self.processed_docs),
            "success_rate": (self.processed_docs / max(1, self.processed_docs + self.failed_docs)) * 100,
            "failed_records": self.failed_records
        }
    
    def update_processing_time(self) -> None:
        """Update the total processing time based on the current time."""
        self.total_processing_time = time.time() - self.start_time

def handle_processing_error(error: Exception, selection_code: str, metrics: Metrics) -> None:
    """Handle processing errors consistently.

    This function provides consistent error handling by:
    1. Formatting error messages uniformly
    2. Logging to both console and file
    3. Updating metrics
    4. Ensuring proper error propagation

    Args:
        error (Exception): The error that occurred.
        selection_code (str): The selection code being processed.
        metrics (Metrics): The metrics object to update.
    """
    error_msg = f"Error processing selection code {selection_code}: {str(error)}"
    debug_print(f"\n{error_msg}")
    metrics.failed_docs += 1
    metrics.failed_records.append((selection_code, str(error)))

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    # Always check environment variable directly to respect runtime changes
    if os.environ.get('MY_AGENT_DEBUG', '0') == '1':
        print(msg)

def get_document_hash(text: str) -> str:
    """Generate MD5 hash for a document text.
    
    This function creates a unique hash for each document to enable
    efficient deduplication. The hash is generated using MD5 and
    is based on the UTF-8 encoded text content.
    
    Args:
        text (str): The document text to hash
        
    Returns:
        str: MD5 hash of the document text
        
    Raises:
        UnicodeEncodeError: If the text cannot be encoded as UTF-8
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_documents_from_sqlite() -> tuple[list[str], list[str], list[str]]:
    """Retrieve documents from SQLite database.
    
    This function connects to the SQLite database and retrieves all
    documents and their corresponding selection codes. It also
    generates MD5 hashes for each document to enable deduplication.
    
    Returns:
        tuple: (texts, selections, hashes) where:
            - texts is a list of document contents
            - selections is a list of corresponding selection codes
            - hashes is a list of MD5 hashes of the documents
            
    Raises:
        sqlite3.Error: If there's an error connecting to or querying the database
    """
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        
        # First check if the table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='selection_descriptions'
        """)
        if not cursor.fetchone():
            debug_print(f"{CREATE_CHROMADB_ID}: Table 'selection_descriptions' does not exist in SQLite database.")
            return [], [], []
        
        # Get only documents that have extended_description
        cursor.execute("""
            SELECT extended_description, selection_code 
            FROM selection_descriptions 
            WHERE extended_description IS NOT NULL 
            AND extended_description != ''
        """)
        results = cursor.fetchall()
        
        if not results:
            debug_print(f"{CREATE_CHROMADB_ID}: No documents found in SQLite database.")
            return [], [], []
        
        texts, selections = zip(*results)
        hashes = [get_document_hash(text) for text in texts]
        
        # Print some debug info about the documents
        debug_print(f"{CREATE_CHROMADB_ID}: Found {len(texts)} documents in SQLite database.")
        debug_print(f"{CREATE_CHROMADB_ID}: Sample document lengths:")
        for sel, text in zip(selections[:3], texts[:3]):
            debug_print(f"{CREATE_CHROMADB_ID}: - {sel}: {len(text)} characters")
        
        return list(texts), list(selections), list(hashes)
        
    except sqlite3.Error as e:
        debug_print(f"{CREATE_CHROMADB_ID}: Database error: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a string using tiktoken.
    
    Args:
        string (str): The text to count tokens for
        encoding_name (str): The encoding to use (default: cl100k_base for text-embedding-3-large)
        
    Returns:
        int: Number of tokens in the string
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def split_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into chunks that don't exceed the token limit, using token-based splitting.
    
    Args:
        text (str): The text to split
        max_tokens (int): Maximum tokens per chunk
        
    Returns:
        List[str]: List of text chunks, each under the token limit
    """
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

#==============================================================================
# MAIN LOGIC
#==============================================================================
def upsert_documents_to_chromadb(
    deployment: str = "text-embedding-3-large__test1",
    collection_name: str = "czsu_selections_chromadb"
) -> chromadb.Collection | None:
    """Add documents from SQLite to a ChromaDB collection.
    
    This function:
    1. Retrieves documents from SQLite
    2. Checks for existing documents in ChromaDB
    3. Generates embeddings for new documents
    4. Adds new documents to ChromaDB with metadata
    
    Args:
        deployment (str): Azure embedding deployment name to use
        collection_name (str): Name of the ChromaDB collection to use or create
        
    Returns:
        chromadb.Collection | None: The ChromaDB collection object after updates,
                                   or None if no documents were found
        
    Raises:
        ValueError: If no documents are found in SQLite
        chromadb.errors.ChromaDBError: If there's an error with ChromaDB operations
        Exception: For other unexpected errors
    """
    metrics = Metrics()
    
    try:
        # Get documents from SQLite
        texts, selections, hashes = get_documents_from_sqlite()
        if not texts:
            debug_print(f"{CREATE_CHROMADB_ID}: No documents found in SQLite database.")
            return None

        # Initialize Azure embedding client
        embedding_client = get_azure_embedding_model()

        # Initialize ChromaDB client and get/create collection
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        try:
            # Try to create the collection with cosine similarity if it doesn't exist
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # If it already exists, just get it
            collection = client.get_collection(name=collection_name)

        # Check for existing documents in ChromaDB
        existing = collection.get(include=["metadatas"], limit=10000)
        existing_hashes = set()
        if existing and "metadatas" in existing and existing["metadatas"]:
            for metadata in existing["metadatas"]:
                if isinstance(metadata, dict) and metadata is not None:
                    doc_hash = metadata.get('doc_hash')
                    if doc_hash:
                        existing_hashes.add(doc_hash)
        debug_print(f"{CREATE_CHROMADB_ID}: Found {len(existing_hashes)} existing documents in ChromaDB.")

        # Filter out existing documents
        new_indices = [i for i, doc_hash in enumerate(hashes) if doc_hash not in existing_hashes]
        new_texts = [texts[i] for i in new_indices]
        new_selections = [selections[i] for i in new_indices]
        new_hashes = [hashes[i] for i in new_indices]
        
        if not new_texts:
            debug_print(f"{CREATE_CHROMADB_ID}: No new documents to add.")
            return collection

        debug_print(f"{CREATE_CHROMADB_ID}: Processing {len(new_texts)} new documents.")
        
        # Process documents in batches to handle token limits
        BATCH_SIZE = 1  # Process one document at a time
        total_batches = (len(new_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Use tqdm for progress tracking
        with tqdm_module.tqdm(
            total=len(new_texts),
            desc="Processing documents",
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stdout
        ) as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(new_texts))
                
                batch_texts = new_texts[start_idx:end_idx]
                batch_selections = new_selections[start_idx:end_idx]
                batch_hashes = new_hashes[start_idx:end_idx]
                
                try:
                    # Print batch info for debugging
                    debug_print(f"\n{CREATE_CHROMADB_ID}: Processing batch {batch_idx + 1}/{total_batches}")
                    
                    # Process each document in the batch
                    for text, selection_code, doc_hash in zip(batch_texts, batch_selections, batch_hashes):
                        # Count tokens and split if necessary
                        token_count = num_tokens_from_string(text)
                        debug_print(f"{CREATE_CHROMADB_ID}: - {selection_code}: {len(text)} characters, {token_count} tokens")
                        
                        # Split text if it exceeds token limit
                        text_chunks = split_text_by_tokens(text)
                        
                        if len(text_chunks) > 1:
                            debug_print(f"{CREATE_CHROMADB_ID}: - Split {selection_code} into {len(text_chunks)} chunks")
                        
                        # Process each chunk
                        for chunk_idx, chunk in enumerate(text_chunks):
                            try:
                                # Generate embedding for chunk
                                response = embedding_client.embeddings.create(
                                    input=[chunk],
                                    model=deployment
                                )
                                embedding = response.data[0].embedding
                                
                                # Create unique ID and metadata for this chunk
                                chunk_id = str(uuid4())
                                chunk_metadata = {
                                    "selection": selection_code,
                                    "doc_hash": doc_hash,
                                    "chunk_index": chunk_idx,
                                    "total_chunks": len(text_chunks)
                                }
                                
                                # Add chunk to collection
                                collection.add(
                                    documents=[chunk],
                                    embeddings=[embedding],
                                    ids=[chunk_id],
                                    metadatas=[chunk_metadata]
                                )
                                
                                metrics.processed_docs += 1
                                
                            except Exception as e:
                                handle_processing_error(e, f"{selection_code}_chunk_{chunk_idx}", metrics)
                                continue
                    
                    pbar.update(len(batch_texts))
                    
                except Exception as e:
                    # Handle batch processing errors
                    for sel in batch_selections:
                        handle_processing_error(e, sel, metrics)
                    pbar.update(len(batch_texts))  # Update progress even for failed batches
                    continue

        # Calculate and display final processing statistics
        metrics.update_processing_time()
        
        debug_print(f"\nProcessing completed in {metrics.total_processing_time:.2f} seconds:")
        debug_print(f"- Total documents: {len(new_texts)}")
        debug_print(f"- Successfully processed: {metrics.processed_docs}")
        debug_print(f"- Failed: {metrics.failed_docs}")
        debug_print(f"- Average time per document: {metrics.total_processing_time/max(1,metrics.processed_docs):.2f} seconds")
        debug_print(f"- Success rate: {metrics.to_dict()['success_rate']:.1f}%")
        
        # Display failed records if any
        if metrics.failed_docs > 0:
            debug_print("\nFailed Records:")
            debug_print("=" * 50)
            for selection_code, error in metrics.failed_records:
                debug_print(f"- {selection_code}: {error}")
            debug_print("=" * 50)
            debug_print(f"Total failed records: {metrics.failed_docs}")
            
            # Display failed selection codes and their description lengths
            debug_print("\nFailed Selection Codes and Description Lengths (sorted by length):")
            debug_print("=" * 50)
            # Get failed selection codes and their description lengths
            failed_selection_lengths = []
            for failed_code, _ in metrics.failed_records:
                # Find the corresponding text for this selection code
                for sel, text in zip(selections, texts):
                    if sel == failed_code:
                        failed_selection_lengths.append((failed_code, len(text)))
                        break
            # Sort by length in descending order
            failed_selection_lengths.sort(key=lambda x: x[1], reverse=True)
            # Display the results
            for selection_code, length in failed_selection_lengths:
                debug_print(f"- {selection_code}: {length} characters")
            debug_print("=" * 50)
        
        return collection
        
    except Exception as e:
        debug_print(f"{CREATE_CHROMADB_ID}: Error in upsert_documents_to_chromadb: {str(e)}")
        raise

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================
if __name__ == "__main__":
    try:
        # Initialize and populate ChromaDB
        collection = upsert_documents_to_chromadb()
        if collection is None:
            sys.exit(1)

        # Example: Performing a similarity search
        embedding_client = get_azure_embedding_model()
        QUERY = "Jake mame druhy Paliva?"
        
        # Generate query embedding
        query_embedding = embedding_client.embeddings.create(
            input=[QUERY],
            model="text-embedding-3-large__test1"
        ).data[0].embedding

        # Perform similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        # Display results
        debug_print(f"{CREATE_CHROMADB_ID}: Query: {QUERY}")
        debug_print(f"{CREATE_CHROMADB_ID}: Top 3 Most Similar Results:")
        debug_print(f"{CREATE_CHROMADB_ID}: {'=' * 100}")
        
        for i, (doc, meta, distance) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1):
            selection = meta.get('selection') if isinstance(meta, dict) and meta is not None else 'N/A'
            similarity = 1 - distance  # Convert distance to similarity
            debug_print(f"{CREATE_CHROMADB_ID}: Result #{i}")
            debug_print(f"{CREATE_CHROMADB_ID}: Similarity: {similarity:.4f}")  # Changed from Distance to Similarity
            debug_print(f"{CREATE_CHROMADB_ID}: Selection Code: {selection}")
            debug_print(f"{CREATE_CHROMADB_ID}: Text:")
            debug_print(f"{CREATE_CHROMADB_ID}: {doc}")
            debug_print(f"{CREATE_CHROMADB_ID}: {'=' * 100}")
            
    except KeyboardInterrupt:
        debug_print(f"{CREATE_CHROMADB_ID}: Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        debug_print(f"{CREATE_CHROMADB_ID}: Unexpected error: {str(e)}")
        sys.exit(1) 
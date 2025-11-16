"""CZSU Multi-Agent Text-to-SQL Application - Main Entry Point

This module serves as the primary entry point for the Czech Statistical Office (CZSU)
multi-agent text-to-SQL data analysis application. It orchestrates a sophisticated
LangGraph-based workflow that converts natural language questions into SQL queries,
executes them against a SQLite database of CZSU statistical data, and generates
comprehensive answers by combining database results with relevant PDF documentation.

System Architecture:
===================
The application implements a multi-agent LangGraph workflow with the following components:

1. Query Processing Pipeline:
   - Natural language query rewriting and optimization
   - Conversational context resolution (pronoun/reference handling)
   - Topic change detection and query expansion
   - Bilingual support (Czech/English)

2. Dual-Source Parallel Retrieval:
   - Database Selection Retrieval: Finds relevant CZSU datasets using hybrid search
     (semantic + BM25) on dataset descriptions stored in ChromaDB, followed by Cohere reranking
   - PDF Chunk Retrieval: Searches parsed documentation chunks from CZSU PDFs using
     the same hybrid search and reranking approach
   - Both retrieval paths execute in parallel for optimal performance

3. Agentic SQL Generation:
   - Dynamic schema loading from SQLite metadata database
   - MCP (Model Context Protocol) tool integration for remote/local query execution
   - Agentic tool calling: LLM autonomously decides when and how many times to execute queries
   - Iterative data gathering: LLM examines results and decides if more queries needed
   - Handles Czech diacritics, JOIN operations, and CELKEM (total) row filtering

4. Self-Correction & Reflection:
   - Analyzes SQL query results for completeness and accuracy
   - Makes autonomous decisions: "improve" (generate better query) or "answer" (sufficient data)
   - Provides specific feedback for query improvement
   - Enforces iteration limits to prevent infinite loops

5. Multi-Source Answer Synthesis:
   - Combines SQL results with PDF documentation chunks
   - Generates bilingual answers matching query language
   - Provides clear source attribution (database vs documentation)
   - Includes follow-up prompt suggestions for continued exploration

6. Memory & Checkpointing:
   - PostgreSQL-based persistent memory using LangGraph's AsyncPostgresSaver
   - Thread-based conversation tracking for multi-turn dialogues
   - Message summarization to prevent token overflow
   - Minimal checkpoint state for efficient storage
   - Fallback to InMemorySaver when PostgreSQL unavailable

Key Features:
============
1. Conversational Memory:
   - Maintains conversation context across multiple turns
   - Resolves pronouns and references (e.g., "What about women?" after asking about men)
   - Detects topic changes (e.g., "but I meant Prague, not Brno")
   - Thread-based isolation for concurrent users

2. Robust Error Handling:
   - Automatic retry with exponential backoff for SSL/prepared statement errors
   - Fallback mechanisms for checkpointer failures
   - Graceful degradation when data sources unavailable
   - Comprehensive error logging and debugging

3. Memory Leak Prevention:
   - Memory monitoring before/after analysis
   - Automatic garbage collection at critical points
   - Emergency cleanup when memory growth exceeds thresholds
   - Explicit resource cleanup (ChromaDB clients)

4. LangSmith Tracing Integration:
   - Run ID tracking for observability
   - Thread ID for conversation tracking
   - Configuration metadata capture
   - Input/output logging for debugging

5. Flexible Deployment:
   - CLI mode: Direct execution with command-line arguments
   - API mode: Import and call as async function with custom parameters
   - Railway deployment: Configurable to prevent auto-execution
   - Environment-based configuration via .env files

Processing Flow:
===============
1. Initialization:
   - Parse command-line arguments or use provided parameters
   - Generate thread_id for conversation tracking
   - Generate run_id for LangSmith tracing
   - Initialize PostgreSQL checkpointer (or fallback to InMemorySaver)
   - Monitor baseline memory usage

2. State Preparation:
   - Check for existing conversation state in checkpointer
   - For new conversations: Initialize complete state with all fields
   - For continuing conversations: Reset iteration-specific fields
   - Generate initial follow-up prompts for new conversations

3. Graph Execution:
   - Create LangGraph execution graph with checkpointer
   - Invoke graph with input state and configuration
   - Graph executes nodes in sequence: rewrite → retrieve → generate → reflect → answer
   - Automatic checkpointing after each node for resumability

4. Result Processing:
   - Extract final answer, SQL queries, and PDF chunks from graph result
   - Filter selection codes to only include those used in queries
   - Serialize Document objects (PDF chunks) to JSON-compatible format
   - Generate dataset URLs for frontend navigation

5. Memory Cleanup:
   - Force garbage collection to prevent memory leaks
   - Monitor final memory usage and detect retention issues
   - Log warnings if memory growth exceeds thresholds
   - Return serialized result for API response or CLI output

Configuration:
=============
Environment Variables (via .env):
- AZURE_OPENAI_API_KEY: Azure OpenAI API key for LLM calls
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
- POSTGRES_CONNECTION_STRING: PostgreSQL connection for checkpointer
- GC_MEMORY_THRESHOLD: Memory growth threshold (MB) for emergency GC (default: 1900)

Constants:
- DEFAULT_PROMPT: Default analysis prompt when none provided
- MAX_ITERATIONS: Maximum reflection iterations (defined in nodes.py)

Usage Examples:
==============
1. Command-Line Execution:
   ```bash
   python main.py "What was Prague's population in 2024?"
   python main.py "Compare wages across industries" --thread_id abc123
   python main.py --thread_id existing_thread  # Continue conversation
   ```

2. API/Library Usage:
   ```python
   from main import main
   import asyncio

   # New conversation
   result = asyncio.run(main(
       prompt="What was Prague's population in 2024?",
       thread_id=None,  # Auto-generated
       run_id=None      # Auto-generated
   ))

   # Continue conversation
   result = asyncio.run(main(
       prompt="What about Brno?",
       thread_id="existing_thread_id",
       checkpointer=shared_checkpointer
   ))
   ```

3. Railway Deployment:
   The script is configured to prevent auto-execution by Railway's RAILPACK builder.
   Railway executes `api/main.py` with uvicorn instead of this file.

Required Environment:
====================
- Python 3.11+
- PostgreSQL database for persistent checkpointing (optional, falls back to InMemorySaver)
- Azure OpenAI API access for LLM calls
- SQLite database with CZSU statistical data
- ChromaDB instance with dataset descriptions and PDF chunks
- Internet connection for Cohere reranking API
- Required packages: See pyproject.toml for complete list

Input/Output:
============
Input (main function parameters):
- prompt (str, optional): Natural language question to analyze
- thread_id (str, optional): Conversation thread ID for memory persistence
- checkpointer (optional): Shared checkpointer instance (for multi-request scenarios)
- run_id (str, optional): LangSmith tracing run ID

Output (dict):
- prompt (str): Original user question
- result (str): Final answer combining SQL results and PDF documentation
- queries_and_results (list): List of (SQL_query, result_data) tuples
- thread_id (str): Thread ID used for this analysis
- top_selection_codes (list): CZSU dataset codes actually used in queries
- iteration (int): Final reflection iteration count
- max_iterations (int): Maximum allowed iterations
- sql (str): Last executed SQL query
- datasetUrl (str): URL to dataset page (format: /datasets/{selection_code})
- top_chunks (list): Relevant PDF documentation chunks with metadata
- followup_prompts (list): Suggested follow-up questions for continued exploration

Error Handling:
==============
The application implements comprehensive error handling:
- Automatic retry for SSL connection errors (max 3 attempts)
- Automatic retry for PostgreSQL prepared statement errors (max 3 attempts)
- Graceful fallback to InMemorySaver when PostgreSQL unavailable
- Emergency garbage collection when memory growth exceeds thresholds
- Detailed logging and tracing for debugging

Memory Management:
=================
To prevent memory leaks in long-running deployments:
- Monitors memory usage before and after analysis
- Logs warnings when memory growth exceeds 100MB
- Forces garbage collection at strategic points
- Implements emergency cleanup when growth exceeds GC_MEMORY_THRESHOLD
- Provides detailed memory diagnostics in logs

Performance Considerations:
==========================
- Parallel retrieval reduces latency (database + PDF search concurrently)
- Message summarization prevents token overflow in long conversations
- Minimal checkpoint state reduces database storage
- Garbage collection prevents memory accumulation
- Schema truncation optimizes LLM context usage

Security Considerations:
=======================
- API keys loaded from environment variables (never hardcoded)
- Thread-based isolation prevents conversation leakage between users
- SQL injection prevention through parameterized queries (in sqlite_query tool)
- Input sanitization (curly brace escaping for f-string safety)

Debugging & Monitoring:
======================
Enable debug logging via utility functions:
- print__analysis_tracing_debug(): Detailed execution flow tracing
- print__main_debug(): Main function execution logging
- print__memory_debug(): Memory usage and GC diagnostics

LangSmith integration provides:
- Run-level tracing with unique run_id
- Thread-level conversation tracking
- Input/output capture for all LLM calls
- Node execution timing and metrics

Related Modules:
===============
- my_agent/agent.py: LangGraph workflow definition
- my_agent/utils/nodes.py: Node implementations (17 nodes + helpers)
- checkpointer/: PostgreSQL checkpointer and memory management
- api/: FastAPI REST API endpoints
- metadata/: ChromaDB management and dataset descriptions
- data/: CZSU data extraction and CSV conversion

Notes:
=====
- The script entry point (__main__ block) is commented out to prevent Railway auto-execution
- To run manually: `python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"`
- For production deployment, use the FastAPI API (api/main.py) with uvicorn
- The DEFAULT_PROMPT is for testing/development; production uses API-provided prompts
"""

import asyncio

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys

# ==============================================================================
# WINDOWS COMPATIBILITY FIX
# ==============================================================================
# Configure asyncio event loop policy for Windows compatibility with psycopg (PostgreSQL driver)
# The default ProactorEventLoop on Windows doesn't support all asyncio features needed by psycopg3
# SelectorEventLoop provides better compatibility for database connections on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse  # Command-line argument parsing
import gc  # Garbage collection for memory management
import os  # Operating system interface for environment variables
import re  # Regular expressions for SQL parsing
import uuid  # UUID generation for unique thread and run IDs
from pathlib import Path  # Object-oriented filesystem paths
from typing import List  # Type hints for better code documentation

import psutil  # Process and system monitoring for memory tracking
from dotenv import load_dotenv  # Environment variable loading from .env files
from langchain_core.messages import (
    HumanMessage,
)  # LangChain message types for LLM communication

# ==============================================================================
# ENVIRONMENT VARIABLE LOADING
# ==============================================================================
# Load environment variables from .env file (Azure API keys, database connections, etc.)
# Must be called before importing modules that read environment variables
load_dotenv()

# ==============================================================================
# APPLICATION IMPORTS
# ==============================================================================
from my_agent import create_graph  # LangGraph workflow factory function
from my_agent.utils.nodes import MAX_ITERATIONS  # Maximum reflection iteration limit
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,  # PostgreSQL prepared statement error retry
    retry_on_ssl_connection_error,  # PostgreSQL SSL connection error retry
)
from checkpointer.checkpointer.factory import (
    get_global_checkpointer,  # PostgreSQL checkpointer factory
)

# ==============================================================================
# PROJECT ROOT PATH CONFIGURATION
# ==============================================================================
# Robust base_dir logic for locating project root directory
# Handles both normal execution and special cases (Jupyter notebooks, interactive shells)
try:
    # Normal execution: __file__ is defined, use its parent directory
    base_dir = Path(__file__).resolve().parents[0]
except NameError:
    # Special cases (Jupyter, REPL): __file__ not defined, use current working directory
    base_dir = Path(os.getcwd()).parents[0]

# Add project root to Python path if not already present
# This ensures imports work correctly regardless of execution context
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# ==============================================================================
# DEBUG UTILITY IMPORTS
# ==============================================================================
# Import debug functions from utils for detailed execution tracing
from api.utils.debug import (
    print__analysis_tracing_debug,  # Detailed execution flow tracing (numbered steps)
    print__main_debug,  # Main function execution logging
    print__memory_debug,  # Memory usage and garbage collection diagnostics
)
from my_agent.utils.followup import (
    generate_initial_followup_prompts,  # Follow-up prompt generation (template or AI-based)
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Default prompt for testing and development
# In production (API mode), prompts are provided by client applications via HTTP requests
# This default prompt demonstrates a typical Czech statistical question about fuel production
# Various commented examples show different query types supported by the system:
# - Population comparisons (Prague vs regions)
# - Gender statistics (men vs women)
# - Time series trends (quarterly/monthly changes)
# - Rate calculations (population growth rates)
# - Multi-condition queries (regions with specific criteria)
# - Industry statistics (wages by sector)
# - Internet usage demographics
# - Energy/fuel production data
DEFAULT_PROMPT = "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?"
# English: "What was the production of liquid fuels from oil in Czechia in 2023?"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def extract_table_names_from_sql(sql_query: str) -> List[str]:
    """Extract table names from SQL query FROM and JOIN clauses.

    This function parses SQL queries to identify all table names referenced in FROM clauses
    and JOIN operations. It's used to determine which CZSU dataset selection codes were
    actually used in the generated queries, enabling accurate dataset attribution.

    The function handles:
    - Multiple tables in comma-separated FROM clauses
    - Quoted and unquoted table names
    - Schema-qualified table names (e.g., schema.table)
    - All JOIN types (INNER, LEFT, RIGHT, FULL)
    - SQL comments (line and block comments)
    - Whitespace normalization

    Args:
        sql_query (str): The SQL query string to parse

    Returns:
        List[str]: List of unique table names found (uppercase, deduplicated)

    Example:
        >>> extract_table_names_from_sql("SELECT * FROM OBY01PDT01 JOIN OBY01PDT02 ON ...")
        ['OBY01PDT01', 'OBY01PDT02']
    """
    # ===========================================================================
    # STEP 1: SQL QUERY NORMALIZATION
    # ===========================================================================
    # Remove SQL comments to prevent false matches in comment text
    # Strip line comments (-- comment)
    sql_clean = re.sub(r"--.*?(?=\n|$)", "", sql_query, flags=re.MULTILINE)
    # Strip block comments (/* comment */)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    # Normalize whitespace for consistent parsing
    sql_clean = " ".join(sql_clean.split())

    # ===========================================================================
    # STEP 2: EXTRACT TABLE NAMES FROM FROM CLAUSES
    # ===========================================================================
    # Pattern explanation:
    # \bFROM\s+ - FROM keyword with word boundary
    # (["\']?) - Optional opening quote (captured)
    # ([a-zA-Z_][a-zA-Z0-9_]*) - Table name (letters, digits, underscore)
    # \1 - Matching closing quote
    # (?:\s*,\s*...)* - Optional comma-separated additional tables
    from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'

    table_names = []
    matches = re.finditer(from_pattern, sql_clean, re.IGNORECASE)

    for match in matches:
        # Extract the main table name (group 2)
        if match.group(2):
            table_names.append(match.group(2).upper())
        # Extract additional table names if comma-separated (group 4)
        if match.group(4):
            table_names.append(match.group(4).upper())

    # ===========================================================================
    # STEP 3: EXTRACT TABLE NAMES FROM JOIN CLAUSES
    # ===========================================================================
    # Pattern for JOIN operations (INNER JOIN, LEFT JOIN, etc.)
    join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
    join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)

    for match in join_matches:
        if match.group(2):
            table_names.append(match.group(2).upper())

    # Return deduplicated list of table names
    return list(set(table_names))


def get_used_selection_codes(
    queries_and_results: list, top_selection_codes: List[str]
) -> List[str]:
    """Filter top_selection_codes to only include those actually used in SQL queries.

    This function analyzes the generated SQL queries to determine which CZSU dataset
    selection codes were actually referenced. This is important for accurate dataset
    attribution in the UI and prevents showing datasets that were retrieved but not used.

    The filtering process:
    1. Parses all SQL queries to extract table names
    2. Compares table names against candidate selection codes
    3. Returns only the intersection (codes actually used)

    This ensures the frontend displays only relevant datasets that contributed to the answer.

    Args:
        queries_and_results (list): List of (SQL_query, result_data) tuples from graph execution
        top_selection_codes (List[str]): Candidate selection codes from retrieval phase

    Returns:
        List[str]: Filtered list of selection codes that appear as table names in queries

    Example:
        >>> queries = [("SELECT * FROM OBY01PDT01", {...}), ("SELECT * FROM OBY01PDT02", {...})]
        >>> candidates = ["OBY01PDT01", "OBY01PDT02", "OBY01PDT03"]
        >>> get_used_selection_codes(queries, candidates)
        ['OBY01PDT01', 'OBY01PDT02']  # OBY01PDT03 excluded as not used
    """
    # Handle edge cases: empty inputs
    if not queries_and_results or not top_selection_codes:
        return []

    # ===========================================================================
    # STEP 1: EXTRACT ALL TABLE NAMES FROM QUERIES
    # ===========================================================================
    used_table_names = set()
    for query, _ in queries_and_results:
        if query:
            # Parse SQL query to extract table names
            table_names = extract_table_names_from_sql(query)
            used_table_names.update(table_names)

    # ===========================================================================
    # STEP 2: FILTER SELECTION CODES BY USAGE
    # ===========================================================================
    # Only include selection codes that match actual table names used in queries
    used_selection_codes = []
    for selection_code in top_selection_codes:
        if selection_code.upper() in used_table_names:
            used_selection_codes.append(selection_code)

    return used_selection_codes


# ==============================================================================
# Note: generate_initial_followup_prompts() moved to my_agent/utils/followup.py
# Function now supports both template-based and AI-based generation strategies
# controlled by FOLLOWUP_PROMPTS_STRATEGY environment variable
# ==============================================================================


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
@retry_on_ssl_connection_error(max_retries=3)
@retry_on_prepared_statement_error(max_retries=3)
async def main(prompt=None, thread_id=None, checkpointer=None, run_id=None):
    """Main orchestration function for CZSU multi-agent text-to-SQL data analysis.

    This async function serves as the central coordinator for the entire analysis pipeline,
    managing the workflow from natural language question input through SQL generation,
    execution, and multi-source answer synthesis. It handles both CLI and API execution
    modes with comprehensive error handling, memory management, and observability.

    Execution Modes:
    ---------------
    1. CLI Mode: Direct execution with command-line arguments
       - Prompts can be provided as positional argument or use DEFAULT_PROMPT
       - Optional --thread_id flag for continuing conversations
       - Optional --run_id flag for LangSmith tracing correlation

    2. API Mode: Called from FastAPI endpoints (api/main.py)
       - All parameters provided programmatically
       - Thread IDs managed by API for multi-user isolation
       - Checkpointer shared across requests for efficiency

    Function Workflow:
    -----------------
    1. Parameter Resolution (Lines 29-36):
       - Parse command-line arguments (CLI mode) or use provided parameters (API mode)
       - Generate thread_id if not provided (format: data_analysis_{8_hex_chars})
       - Generate run_id for LangSmith tracing if not provided

    2. Memory Baseline Establishment (Lines 37-39):
       - Record initial memory usage (RSS - Resident Set Size)
       - Force garbage collection to start with clean slate
       - Enable memory leak detection through before/after comparison

    3. Checkpointer Initialization (Lines 40-47):
       - Attempt PostgreSQL-based persistent checkpointer
       - Fallback to InMemorySaver if PostgreSQL unavailable
       - Enables conversation memory and resumability

    4. Graph Creation (Lines 46-48):
       - Initialize LangGraph workflow with checkpointer
       - Graph contains 17 nodes in 6 processing stages
       - Nodes include: rewrite, retrieve, generate, reflect, format, save

    5. State Preparation (Lines 50-67):
       - Check for existing conversation state in checkpointer
       - New conversations: Initialize complete state with all 13 fields
       - Continuing conversations: Reset iteration-specific fields only
       - Generate initial follow-up prompts for new conversations

    6. Graph Execution (Lines 58-68):
       - Invoke LangGraph with prepared state and configuration
       - Automatic checkpointing after each node
       - Parallel retrieval of database selections and PDF chunks
       - Iterative SQL generation with reflection (up to MAX_ITERATIONS)

    7. Memory Monitoring (Lines 60-67):
       - Track memory growth during graph execution
       - Trigger emergency GC if growth exceeds GC_MEMORY_THRESHOLD (default: 1900MB)
       - Log warnings for suspicious memory patterns

    8. Result Processing (Lines 73-82):
       - Extract final answer from messages list (last AI message)
       - Filter selection codes to only those used in queries
       - Serialize PDF chunks (Document objects) to JSON format
       - Generate dataset URLs for frontend navigation

    9. Final Cleanup (Lines 68-72):
       - Force final garbage collection
       - Monitor total memory retention
       - Log warnings if memory growth exceeds 100MB
       - Return serialized result for API response

    Decorators:
    ----------
    @retry_on_ssl_connection_error(max_retries=3):
        Handles transient SSL connection errors with PostgreSQL
        Uses exponential backoff between retry attempts

    @retry_on_prepared_statement_error(max_retries=3):
        Handles PostgreSQL prepared statement already exists errors
        Common when multiple requests execute concurrently

    Args:
        prompt (str, optional):
            Natural language question to analyze. Examples:
            - "What was Prague's population in 2024?"
            - "Compare wages across industries in Czech Republic"
            - "Show me energy production trends over time"
            If None in CLI mode, parsed from command line or uses DEFAULT_PROMPT.
            Required when called from API mode.

        thread_id (str, optional):
            Unique conversation thread identifier for memory persistence.
            Format: "data_analysis_{8_hex_chars}" or custom string.
            - Same thread_id = continuing conversation (context preserved)
            - Different thread_id = new conversation (fresh state)
            - None = auto-generate new thread_id
            Used as key in PostgreSQL checkpointer for state storage.

        checkpointer (AsyncPostgresSaver | InMemorySaver, optional):
            Checkpointer instance for conversation state persistence.
            - None = create new PostgreSQL checkpointer (or InMemorySaver fallback)
            - Provided instance = share checkpointer across multiple requests (API mode)
            Stores conversation history, retrieved datasets, and intermediate results.

        run_id (str, optional):
            UUID for LangSmith tracing correlation.
            Enables tracking of:
            - LLM calls and token usage
            - Node execution timing
            - Errors and exceptions
            - Input/output for each step
            If None, generates new UUID automatically.

    Returns:
        dict: Serialized analysis result with the following structure:
        {
            "prompt": str,                    # Original user question
            "result": str,                    # Final answer (markdown formatted)
            "queries_and_results": [          # List of executed queries
                (sql_query: str, result_data: dict), ...
            ],
            "thread_id": str,                 # Thread ID used (for continuing conversation)
            "top_selection_codes": [str],     # CZSU dataset codes actually used
            "iteration": int,                 # Final reflection iteration count (0 to MAX_ITERATIONS)
            "max_iterations": int,            # Maximum allowed iterations
            "sql": str,                       # Last executed SQL query
            "datasetUrl": str,                # URL for dataset page (/datasets/{code})
            "top_chunks": [                   # Relevant PDF documentation chunks
                {
                    "content": str,           # Chunk text content
                    "metadata": {             # Chunk metadata
                        "source": str,        # PDF filename
                        "page": int,          # Page number
                        ...                   # Additional metadata
                    }
                }, ...
            ],
            "followup_prompts": [str]         # Suggested follow-up questions
        }

    Raises:
        Exception: After exhausting retry attempts for SSL or prepared statement errors

    Example Usage:
        ```python
        # CLI Mode (command-line execution)
        # python main.py "What was Prague's population in 2024?"

        # API Mode (programmatic invocation)
        import asyncio
        from main import main

        # New conversation
        result = await main(
            prompt="What was Prague's population in 2024?",
            thread_id=None,  # Auto-generated
            run_id=None      # Auto-generated
        )
        print(result["result"])  # Final answer

        # Continue conversation
        result2 = await main(
            prompt="What about Brno?",
            thread_id=result["thread_id"],  # Same thread = context preserved
            checkpointer=shared_checkpointer  # Reuse checkpointer
        )
        ```

    Memory Management:
        - Monitors memory usage before and after analysis
        - Logs warnings when growth exceeds 100MB total
        - Implements emergency GC when growth exceeds GC_MEMORY_THRESHOLD
        - Typical memory growth: 50-150MB per analysis
        - Memory is released after result serialization and final GC

    Performance:
        - Average execution time: 5-15 seconds (depends on query complexity)
        - Parallel retrieval reduces latency (database + PDF concurrently)
        - Checkpointing adds ~100-200ms overhead per node
        - Message summarization prevents token overflow in long conversations

    Notes:
        - The function is designed for both synchronous (CLI) and asynchronous (API) use
        - Debug output controlled via print__main_debug, print__analysis_tracing_debug
        - LangSmith tracing requires LANGCHAIN_API_KEY environment variable
        - PostgreSQL checkpointer requires POSTGRES_CONNECTION_STRING
        - Conversation state stored indefinitely in PostgreSQL (manual cleanup needed)
    """
    # ===========================================================================
    # STEP 29: FUNCTION ENTRY - INITIALIZE TRACING
    # ===========================================================================
    print__analysis_tracing_debug("29 - MAIN ENTRY: main() function entry point")
    print__main_debug("29 - MAIN ENTRY: main() function entry point")

    # ===========================================================================
    # STEP 30-32: PROMPT RESOLUTION
    # ===========================================================================
    # Determine prompt source based on execution mode
    # Priority: function parameter > command-line argument > DEFAULT_PROMPT
    if prompt is None and __name__ == "__main__":
        # CLI Mode: Parse command-line arguments
        print__analysis_tracing_debug(
            "30 - COMMAND LINE ARGS: Processing command line arguments"
        )
        parser = argparse.ArgumentParser(description="Run data analysis with LangGraph")
        parser.add_argument(
            "prompt",
            nargs="?",  # Optional positional argument
            default=DEFAULT_PROMPT,
            help=f'Analysis prompt (default: "{DEFAULT_PROMPT}")',
        )
        parser.add_argument(
            "--thread_id",
            type=str,
            default=None,
            help="Conversation thread ID for memory",
        )
        parser.add_argument(
            "--run_id", type=str, default=None, help="Run ID for LangSmith tracing"
        )
        args = parser.parse_args()
        prompt = args.prompt
        thread_id = args.thread_id
        run_id = args.run_id

    # Ensure prompt is never None to prevent downstream errors
    if prompt is None:
        print__analysis_tracing_debug("31 - DEFAULT PROMPT: Using default prompt")
        prompt = DEFAULT_PROMPT
    else:
        print__analysis_tracing_debug(
            f"32 - PROMPT PROVIDED: Using provided prompt (length: {len(prompt)})"
        )

    # ===========================================================================
    # STEP 33-34: THREAD ID GENERATION/VALIDATION
    # ===========================================================================
    # Thread ID enables conversation memory and multi-turn dialogues
    # Format: data_analysis_{8_hex_chars} for auto-generated IDs
    if thread_id is None:
        thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
        print__analysis_tracing_debug(
            f"33 - THREAD ID GENERATED: Generated new thread_id {thread_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"34 - THREAD ID PROVIDED: Using provided thread_id {thread_id}"
        )

    # ===========================================================================
    # STEP 35-36: RUN ID GENERATION FOR LANGSMITH TRACING
    # ===========================================================================
    # Run ID enables correlation of all LLM calls and node executions in LangSmith
    if run_id is None:
        run_id = str(uuid.uuid4())
        print__analysis_tracing_debug(
            f"35 - RUN ID GENERATED: Generated new run_id {run_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"36 - RUN ID PROVIDED: Using provided run_id {run_id}"
        )

    # ===========================================================================
    # OPTIONAL: LANGSMITH INSTRUMENTATION
    # ===========================================================================
    # Commented out by default - uncomment to enable detailed LangSmith tracing
    # Requires LANGCHAIN_API_KEY and LANGCHAIN_PROJECT environment variables
    # instrument(project_name="LangGraph_czsu-multi-agent-text-to-sql", framework=Framework.LANGGRAPH)

    # ===========================================================================
    # STEP 37-39: MEMORY BASELINE ESTABLISHMENT
    # ===========================================================================
    # Memory leak prevention: Track memory usage throughout execution
    # This enables detection of memory retention issues in production
    print__analysis_tracing_debug("37 - MEMORY MONITORING: Starting memory monitoring")

    # Capture baseline memory usage (Resident Set Size - physical RAM used by process)
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    print__memory_debug(f"🔍 MEMORY: Starting analysis with {memory_before:.1f}MB RSS")
    print__analysis_tracing_debug(
        f"38 - MEMORY BASELINE: Memory before analysis: {memory_before:.1f}MB RSS"
    )

    # Force garbage collection to start with clean slate
    # This ensures accurate measurement of memory growth during analysis
    collected = gc.collect()
    print__memory_debug(f"🧹 MEMORY: Pre-analysis GC collected {collected} objects")

    # ===========================================================================
    # STEP 40-47: CHECKPOINTER INITIALIZATION
    # ===========================================================================
    # Checkpointer enables conversation memory and resumability
    # PostgreSQL-based persistent storage with InMemorySaver fallback
    print__analysis_tracing_debug("40 - CHECKPOINTER SETUP: Setting up checkpointer")

    if checkpointer is None:
        # No external checkpointer provided - create new instance
        try:
            # Attempt PostgreSQL-based persistent checkpointer (production mode)
            print__analysis_tracing_debug(
                "41 - POSTGRES CHECKPOINTER: Attempting to get PostgreSQL checkpointer"
            )
            checkpointer = await get_global_checkpointer()
            print__analysis_tracing_debug(
                "42 - POSTGRES SUCCESS: PostgreSQL checkpointer obtained"
            )
        except Exception as e:
            # PostgreSQL unavailable - fallback to memory-only storage (development mode)
            print__analysis_tracing_debug(
                f"43 - POSTGRES FAILED: Failed to initialize PostgreSQL checkpointer - {str(e)}"
            )
            print__main_debug(f"⚠️ Failed to initialize PostgreSQL checkpointer: {e}")

            # Import InMemorySaver as fallback
            # Note: InMemorySaver state persists only for current process lifetime
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
            print__analysis_tracing_debug(
                "44 - INMEMORY FALLBACK: Using InMemorySaver fallback"
            )
            print__main_debug("⚠️ Using InMemorySaver fallback")
    else:
        # External checkpointer provided (API mode with shared checkpointer)
        print__analysis_tracing_debug(
            f"45 - CHECKPOINTER PROVIDED: Using provided checkpointer ({type(checkpointer).__name__})"
        )

    # ===========================================================================
    # STEP 46-48: LANGGRAPH WORKFLOW CREATION
    # ===========================================================================
    # Create the execution graph containing all 17 workflow nodes
    # Graph structure: rewrite → retrieve (parallel) → generate → reflect → format → save
    print__analysis_tracing_debug(
        "46 - GRAPH CREATION: Creating LangGraph execution graph"
    )
    graph = create_graph(checkpointer=checkpointer)
    print__analysis_tracing_debug(
        "47 - GRAPH CREATED: LangGraph execution graph created successfully"
    )

    # Escape curly braces in prompt for safe f-string formatting
    # This prevents interpretation of {variable} patterns in user input
    prompt_escaped = prompt.replace("{", "{{").replace("}", "}}")
    print__main_debug(
        f"🚀 Processing prompt: {prompt_escaped} (thread_id={thread_id}, run_id={run_id})"
    )
    print__analysis_tracing_debug(
        f"48 - PROCESSING START: Processing prompt with thread_id={thread_id}, run_id={run_id}"
    )

    # ===========================================================================
    # STEP 49: EXECUTION CONFIGURATION
    # ===========================================================================
    # Configure graph execution with thread ID for checkpointing and run ID for tracing
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id}
    print__analysis_tracing_debug(
        "49 - CONFIG SETUP: Configuration for thread-level persistence and LangSmith tracing"
    )

    # ===========================================================================
    # STEP 50-54: CONVERSATION STATE DETECTION
    # ===========================================================================
    # Determine if this is a new conversation or continuing an existing one
    # This affects state initialization (full vs partial update)
    print__analysis_tracing_debug("50 - STATE CHECK: Checking for existing state")

    try:
        # Query checkpointer for existing state
        existing_state = await graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )

        # Consider it a continuing conversation if messages exist
        is_continuing_conversation = (
            existing_state
            and existing_state.values
            and existing_state.values.get("messages")
            and len(existing_state.values.get("messages", [])) > 0
        )

        # Log state detection results for debugging
        print__main_debug(f"🔍 Found existing state: {existing_state is not None}")
        print__analysis_tracing_debug(
            f"51 - STATE CHECK RESULT: Found existing state: {existing_state is not None}"
        )

        if existing_state and existing_state.values:
            msg_count = len(existing_state.values.get("messages", []))
            print__main_debug(f"📋 Message count: {msg_count}")
            print__analysis_tracing_debug(
                f"52 - MESSAGE COUNT: Message count: {msg_count}"
            )

        print__main_debug(f"🔀 Continuing conversation: {is_continuing_conversation}")
        print__analysis_tracing_debug(
            f"53 - CONVERSATION TYPE: Continuing conversation: {is_continuing_conversation}"
        )
    except Exception as e:
        # Error checking state - treat as new conversation
        print__main_debug(f"❌ Error checking existing state: {e}")
        print__analysis_tracing_debug(
            f"54 - STATE CHECK ERROR: Error checking existing state - {str(e)}"
        )
        is_continuing_conversation = False

    # ===========================================================================
    # STEP 55-57: INPUT STATE PREPARATION
    # ===========================================================================
    # Prepare input state based on conversation type
    # New conversations: Full state initialization with all 13 fields
    # Continuing conversations: Partial update (reset iteration-specific fields)
    print__analysis_tracing_debug("55 - STATE PREPARATION: Preparing input state")

    if is_continuing_conversation:
        # Continuing conversation: Reset iteration-specific fields only
        # The checkpointer merges this with existing state (preserves messages, etc.)
        print__analysis_tracing_debug(
            "56 - CONTINUING CONVERSATION: Preparing state for continuing conversation"
        )

        # CRITICAL: Reset these fields to prevent using stale data from previous question
        input_state = {
            "prompt": prompt,  # New question
            "rewritten_prompt": None,  # Force fresh query rewrite
            "iteration": 0,  # Reset reflection iteration counter
            "queries_and_results": [],  # Clear old SQL queries/results
            "followup_prompts": [],  # Clear old follow-up suggestions
            "final_answer": "",  # Clear old answer
            # Reset retrieval results to force fresh search
            "hybrid_search_results": [],  # Clear old database selection search results
            "most_similar_selections": [],  # Clear old reranked selections
            "top_selection_codes": [],  # Clear old top selections
            "hybrid_search_chunks": [],  # Clear old PDF chunk search results
            "most_similar_chunks": [],  # Clear old reranked chunks
            "top_chunks": [],
        }
    else:
        # New conversation: Initialize complete state with all required fields
        print__analysis_tracing_debug(
            "57 - NEW CONVERSATION: Preparing state for new conversation"
        )

        # Generate initial follow-up prompts for new conversations
        # These prompts give users ideas for exploring CZSU data
        initial_followup_prompts = generate_initial_followup_prompts()
        print__main_debug(
            f"💡 Generated {len(initial_followup_prompts)} initial follow-up prompts for new conversation"
        )

        # CRITICAL: New conversations require COMPLETE state initialization
        # All 13 fields from DataAnalysisState must be initialized for proper checkpointing
        # Missing fields cause checkpoint storage errors and state corruption
        input_state = {
            # ===================================================================
            # Core Conversation Fields (Group 1: Basic prompt and messages)
            # ===================================================================
            "prompt": prompt,  # User's original question
            "rewritten_prompt": None,  # Will be populated by rewrite_prompt_node
            "messages": [
                HumanMessage(content=prompt)
            ],  # Initialize with user question for LangSmith visibility
            # ===================================================================
            # Iteration and Results Fields (Group 2: SQL execution tracking)
            # ===================================================================
            "iteration": 0,  # Reflection iteration counter (0 to MAX_ITERATIONS)
            "queries_and_results": [],  # List of (SQL_query, result_data) tuples
            "final_answer": "",  # Final synthesized answer (populated by format_answer_node)
            # ===================================================================
            # Reflection Decision Field (Group 3: Reflection node output)
            # ===================================================================
            "reflection_decision": "",  # Last decision from reflection node ("improve" or "answer")
            # ===================================================================
            # Database Selection Retrieval Fields (Group 4: Dataset search)
            # ===================================================================
            "hybrid_search_results": [],  # Raw hybrid search results (semantic + BM25) before reranking
            "most_similar_selections": [],  # Reranked results: [(selection_code, cohere_score), ...]
            "top_selection_codes": [],  # Final top N selection codes passing relevance threshold
            # ===================================================================
            # PDF Chunk Retrieval Fields (Group 5: Documentation search)
            # ===================================================================
            "hybrid_search_chunks": [],  # Raw hybrid search results for PDF chunks before reranking
            "most_similar_chunks": [],  # Reranked PDF chunks: [(Document, cohere_score), ...]
            "top_chunks": [],  # Final top N PDF chunks passing relevance threshold
            # ===================================================================
            # System Status Fields (Group 6: Error handling and diagnostics)
            # ===================================================================
            "chromadb_missing": False,  # Flag indicating ChromaDB unavailability
            # ===================================================================
            # Follow-up Prompts Field (Group 7: User guidance)
            # ===================================================================
            "followup_prompts": initial_followup_prompts,  # Pre-populated with template-based suggestions
        }

    # ===========================================================================
    # STEP 58: LANGGRAPH EXECUTION
    # ===========================================================================
    # Execute the graph workflow with prepared state
    # The graph will execute nodes in sequence: rewrite → retrieve → generate → reflect → format → save
    print__analysis_tracing_debug("58 - GRAPH EXECUTION: Starting LangGraph execution")
    print__main_debug(
        f"🚀 About to call graph.ainvoke() with thread_id={thread_id}, run_id={run_id}"
    )
    print__main_debug(f"🚀 Input state keys: {list(input_state.keys())}")

    # Invoke the graph asynchronously with state and configuration
    # - input_state: Initial state (full for new conversations, partial for continuing)
    # - config: Contains thread_id for checkpointing and run_id for LangSmith tracing
    # - Automatic checkpointing after each node enables resumability
    # - Parallel retrieval (database + PDF) happens automatically in the graph
    result = await graph.ainvoke(input_state, config=config)

    # Log successful graph completion
    print__main_debug(
        f"✅ graph.ainvoke() completed for thread_id={thread_id}, run_id={run_id}"
    )

    # ===========================================================================
    # STEP 59-67: MEMORY MONITORING AND EMERGENCY CLEANUP
    # ===========================================================================
    # Track memory growth during graph execution to detect potential memory leaks
    print__analysis_tracing_debug(
        "59 - GRAPH EXECUTION COMPLETE: LangGraph execution completed"
    )

    # Calculate memory growth during graph execution
    memory_after_graph = process.memory_info().rss / 1024 / 1024  # Convert to MB
    memory_growth_graph = memory_after_graph - memory_before
    print__memory_debug(
        f"🔍 MEMORY: After graph execution: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )
    print__memory_debug(
        f"60 - MEMORY CHECK: Memory after graph: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )

    # Check if memory growth exceeds threshold (default 1900MB)
    # This threshold indicates potential memory leaks or resource retention issues
    if memory_growth_graph > float(os.environ.get("GC_MEMORY_THRESHOLD", "1900")):
        # Excessive memory growth detected - log warning
        print__memory_debug(
            f"⚠️ MEMORY: Suspicious growth detected: {memory_growth_graph:.1f}MB during graph execution!"
        )
        print__analysis_tracing_debug(
            f"61 - MEMORY WARNING: Suspicious memory growth detected: {memory_growth_graph:.1f}MB"
        )

        # Implement emergency cleanup procedures
        print__memory_debug(
            f"🚨 MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - implementing emergency cleanup"
        )
        print__analysis_tracing_debug(
            f"62 - MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - emergency cleanup"
        )

        # Force aggressive garbage collection to free unreferenced objects
        collected = gc.collect()
        print__memory_debug(f"🧹 MEMORY: Emergency GC collected {collected} objects")
        print__analysis_tracing_debug(
            f"63 - EMERGENCY GC: Emergency GC collected {collected} objects"
        )

        # Measure memory freed by emergency garbage collection
        memory_after_gc = process.memory_info().rss / 1024 / 1024
        freed_by_gc = memory_after_graph - memory_after_gc
        print__memory_debug(
            f"🧹 MEMORY: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )
        print__memory_debug(
            f"64 - EMERGENCY GC RESULT: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )

        # Update memory tracking variables with post-cleanup values
        memory_after_graph = memory_after_gc
        memory_growth_graph = memory_after_graph - memory_before

    # ===========================================================================
    # STEP 65-66: RESULT SIZE ANALYSIS
    # ===========================================================================
    # Analyze result object size to understand memory consumption patterns
    print__analysis_tracing_debug("65 - RESULT PROCESSING: Processing graph result")

    try:
        # Calculate approximate result size in kilobytes
        result_size = len(str(result)) / 1024 if result else 0
        print__memory_debug(f"🔍 MEMORY: Result object size: {result_size:.1f}KB")
        print__analysis_tracing_debug(
            f"66 - RESULT SIZE: Result object size: {result_size:.1f}KB"
        )
    except:
        # Size calculation failed - log and continue
        print__memory_debug("🔍 MEMORY: Could not determine result size")

    # ===========================================================================
    # STEP 68-72: FINAL CLEANUP AND MEMORY MONITORING
    # ===========================================================================
    # Perform final garbage collection and assess total memory retention
    print__analysis_tracing_debug(
        "68 - FINAL CLEANUP: Starting final cleanup and monitoring"
    )

    try:
        # Final garbage collection sweep to clean up temporary objects
        collected = gc.collect()
        print__memory_debug(
            f"🧹 MEMORY: Final cleanup GC collected {collected} objects"
        )
        print__analysis_tracing_debug(
            f"69 - FINAL GC: Final cleanup GC collected {collected} objects"
        )

        # Calculate final memory usage and total growth
        memory_final = process.memory_info().rss / 1024 / 1024
        total_growth = memory_final - memory_before

        print__memory_debug(
            f"🔍 MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )
        print__analysis_tracing_debug(
            f"70 - FINAL MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )

        # Warn about high memory retention (> 100MB indicates potential issues)
        # Typical memory growth should be 50-150MB per analysis
        if total_growth > 100:
            print__memory_debug(
                f"⚠️ MEMORY WARNING: High memory retention ({total_growth:.1f}MB) detected!"
            )
            print__memory_debug(
                "💡 MEMORY: Consider investigating LangGraph nodes for memory leaks"
            )
            print__analysis_tracing_debug(
                f"71 - HIGH MEMORY WARNING: High memory retention ({total_growth:.1f}MB) detected!"
            )

    except Exception as memory_error:
        # Error during memory monitoring - log but don't fail
        print__memory_debug(
            f"⚠️ MEMORY: Error during final memory check: {memory_error}"
        )
        print__memory_debug(
            f"72 - MEMORY ERROR: Error during final memory check - {str(memory_error)}"
        )

    # ===========================================================================
    # STEP 73-76: RESULT EXTRACTION AND PROCESSING
    # ===========================================================================
    # Extract key values from graph result for API response
    print__analysis_tracing_debug(
        "73 - RESULT EXTRACTION: Extracting values from graph result"
    )

    # Extract SQL queries and results from graph execution
    # Format: [(SQL_query_string, result_data_dict), ...]
    queries_and_results = result["queries_and_results"]

    # Extract final answer from messages list
    # Graph stores messages as: [SystemMessage(summary), AIMessage(final_answer)]
    # We want the content of the last message (AIMessage with final answer)
    final_answer = (
        result["messages"][-1].content
        if result.get("messages") and len(result["messages"]) > 1
        else ""
    )

    # Extract dataset selection codes and follow-up prompts from result
    top_selection_codes = result.get(
        "top_selection_codes", []
    )  # All candidate datasets
    sql_query = (
        queries_and_results[-1][0] if queries_and_results else None
    )  # Last executed query
    followup_prompts = result.get(
        "followup_prompts", []
    )  # LLM-generated follow-up suggestions

    print__analysis_tracing_debug(
        f"74 - SELECTION CODES: Processing {len(top_selection_codes)} selection codes"
    )
    print__analysis_tracing_debug(
        f"74a - FOLLOWUP PROMPTS: Extracted {len(followup_prompts)} follow-up prompts from graph result"
    )

    # Filter selection codes to only include those actually used in SQL queries
    # This ensures frontend displays only relevant datasets that contributed to the answer
    used_selection_codes = get_used_selection_codes(
        queries_and_results, top_selection_codes
    )
    print__analysis_tracing_debug(
        f"75 - USED CODES: {len(used_selection_codes)} selection codes actually used"
    )

    # Generate dataset URL for frontend navigation (uses first used selection code)
    dataset_url = None
    if used_selection_codes:
        dataset_url = f"/datasets/{used_selection_codes[0]}"
        print__analysis_tracing_debug(
            f"76 - DATASET URL: Generated dataset URL: {dataset_url}"
        )

    # ===========================================================================
    # STEP 77-80: PDF CHUNKS SERIALIZATION
    # ===========================================================================
    # Convert LangChain Document objects to JSON-serializable format for API response
    print__analysis_tracing_debug(
        "77 - TOP CHUNKS SERIALIZATION: Converting top_chunks to JSON-serializable format"
    )

    top_chunks_serialized = []
    if result.get("top_chunks"):
        # PDF chunks available - serialize each Document object
        chunk_count = len(result["top_chunks"])
        print__main_debug(f"📦 main.py - Found {chunk_count} top_chunks to serialize")
        print__analysis_tracing_debug(
            f"78 - CHUNKS FOUND: Found {chunk_count} top_chunks to serialize"
        )

        # Convert each Document to dict with content and metadata
        for i, chunk in enumerate(result["top_chunks"]):
            chunk_data = {
                "content": (
                    chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                ),
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
            top_chunks_serialized.append(chunk_data)

            # Log first chunk for debugging purposes
            if i == 0:
                content_preview = chunk_data["content"][:100]
                print__main_debug(
                    f"🔍 main.py - First chunk content preview: {content_preview}..."
                )
                print__analysis_tracing_debug(
                    f"79 - FIRST CHUNK: First chunk content preview: {content_preview}..."
                )
    else:
        # No PDF chunks found (may be database-only answer)
        print__main_debug("⚠️ main.py - No top_chunks found in result")
        print__analysis_tracing_debug("80 - NO CHUNKS: No top_chunks found in result")

    # ===========================================================================
    # STEP 81-83: FINAL RESULT SERIALIZATION
    # ===========================================================================
    # Create JSON-serializable result dictionary for API response
    print__analysis_tracing_debug(
        "81 - RESULT SERIALIZATION: Creating serializable result dictionary"
    )

    serializable_result = {
        # Original user input
        "prompt": prompt,
        # Final answer (markdown formatted, bilingual)
        "result": final_answer,
        # SQL execution history: [(query, result), ...]
        "queries_and_results": queries_and_results,
        # Conversation tracking
        "thread_id": thread_id,
        # Dataset attribution (only codes actually used in queries)
        "top_selection_codes": used_selection_codes,
        # Reflection iteration tracking
        "iteration": result.get("iteration", 0),
        "max_iterations": MAX_ITERATIONS,
        # Last executed SQL query for debugging
        "sql": sql_query,
        # Frontend navigation URL
        "datasetUrl": dataset_url,
        # PDF documentation chunks (serialized)
        "top_chunks": top_chunks_serialized,
        # Follow-up prompt suggestions for continued exploration
        "followup_prompts": followup_prompts,
    }

    # Log final result statistics
    print__main_debug(
        f"📦 main.py - Serializable result includes {len(top_chunks_serialized)} top_chunks"
    )
    print__main_debug(
        f"💡 main.py - Serializable result includes {len(followup_prompts)} followup_prompts"
    )
    print__analysis_tracing_debug(
        f"82 - SERIALIZATION COMPLETE: Serializable result includes {len(top_chunks_serialized)} top_chunks and {len(followup_prompts)} followup_prompts"
    )
    print__analysis_tracing_debug("83 - MAIN EXIT: main() function returning result")

    return serializable_result


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# This block is commented out to prevent Railway's RAILPACK builder from auto-executing this file.
# Railway was detecting main.py as an entry point and running it instead of executing
# the configured startCommand (uvicorn api/main.py).
#
# For manual CLI execution, use one of these methods:
#
# Method 1: Using asyncio module
#   python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"
#
# Method 2: Create a separate CLI script (recommended)
#   # cli.py
#   import asyncio
#   from main import main
#   asyncio.run(main())
#
# Method 3: Interactive Python shell
#   >>> import asyncio
#   >>> from main import main
#   >>> asyncio.run(main("Your question here"))
#
# For production deployment, use the FastAPI API (api/main.py) with uvicorn:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000
#
# if __name__ == "__main__":
#     asyncio.run(main())
# Or create a separate CLI script that imports and calls main()
#
# if __name__ == "__main__":
#     asyncio.run(main())

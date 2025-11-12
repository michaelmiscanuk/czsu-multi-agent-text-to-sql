"""Czech Statistical Office (CZSU) Multi-Agent Text-to-SQL Data Analysis System

This module serves as the main entry point and orchestrator for an advanced multi-agent
data analysis system that processes natural language queries about Czech Statistical Office
data using LangGraph workflows, LLM agents, and SQL generation.

The system combines semantic search, retrieval-augmented generation (RAG), SQL query
generation, and conversational memory to provide intelligent responses to statistical
data questions in both Czech and English languages.
"""

MODULE_DESCRIPTION = r"""Czech Statistical Office (CZSU) Multi-Agent Text-to-SQL Data Analysis System

This module serves as the main entry point and orchestrator for an advanced multi-agent
data analysis system that processes natural language queries about Czech Statistical Office
data using LangGraph workflows, LLM agents, and SQL generation.

Key Features:
-------------
1. Multi-Agent LangGraph Workflow:
   - State machine orchestration with configurable checkpointing
   - Multiple specialized agent nodes for different tasks
   - Iterative refinement with reflection and retry logic
   - Maximum iteration controls to prevent infinite loops
   - Interrupt-based workflow control for optimization

2. Natural Language Query Processing:
   - Prompt rewriting and optimization
   - Multi-language support (Czech and English)
   - Query intent understanding and context extraction
   - Follow-up question generation for user guidance
   - Thread-based conversation memory for multi-turn dialogues

3. Hybrid Semantic Search & Retrieval:
   - ChromaDB vector database integration for similarity search
   - Cohere reranking for improved relevance scoring
   - Selection code extraction from metadata schemas
   - PDF document chunk retrieval for context enrichment
   - Relevance threshold filtering for quality control
   - Fallback mechanisms when ChromaDB is unavailable

4. SQL Query Generation & Execution:
   - Dynamic SQL generation from natural language
   - SQLite database querying (local and cloud via Turso)
   - Query validation and error handling
   - Result parsing and formatting
   - Table name extraction from generated queries
   - Query history tracking for iterative refinement

5. Memory & State Management:
   - Thread-based conversation persistence
   - PostgreSQL checkpointing for production deployments
   - In-memory fallback for development
   - State serialization and deserialization
   - Message history tracking with LangChain message types
   - Complete state initialization for new conversations

6. Observability & Monitoring:
   - LangSmith tracing integration for debugging
   - Memory leak detection and prevention
   - Garbage collection monitoring and optimization
   - Performance metrics tracking
   - Detailed debug logging with analysis tracing
   - Memory growth alerts and emergency cleanup

7. API Integration & Deployment:
   - FastAPI backend integration ready
   - Async/await pattern for concurrent operations
   - Windows compatibility with asyncio event loop policy
   - Railway deployment configuration
   - Environment variable configuration
   - Error recovery with retry decorators

8. Result Processing & Output:
   - JSON-serializable response formatting
   - PDF chunk serialization for frontend display
   - Dataset URL generation for reference linking
   - Query result aggregation and presentation
   - Follow-up prompt suggestions for continued exploration
   - Metadata extraction and enrichment

Processing Flow:
--------------
1. Initialization:
   - Load environment variables and configuration
   - Set up asyncio event loop policy for Windows
   - Initialize checkpointer (PostgreSQL or InMemory fallback)
   - Generate thread_id and run_id for tracking
   - Monitor baseline memory usage

2. Prompt Handling:
   - Accept prompt from function parameter, command line, or default
   - Validate and sanitize user input
   - Escape special characters to prevent parsing errors
   - Determine conversation type (new vs. continuing)

3. State Preparation:
   - Check for existing conversation state in checkpointer
   - Initialize complete state for new conversations
   - Generate initial follow-up prompts for new sessions
   - Prepare minimal state update for continuing conversations
   - Reset critical fields to prevent stale data usage

4. LangGraph Execution:
   - Create graph instance with checkpointer
   - Configure thread-level persistence and LangSmith tracing
   - Execute graph.ainvoke() with input state and config
   - Monitor memory growth during execution
   - Handle interrupts and checkpoints appropriately

5. Hybrid Search & Retrieval:
   - Query ChromaDB for similar selection codes
   - Retrieve relevant PDF documentation chunks
   - Apply Cohere reranking for relevance scoring
   - Filter results by relevance threshold
   - Extract top selection codes and chunks

6. Query Generation & Execution:
   - Rewrite user prompt for SQL generation
   - Generate SQL queries using LLM agents
   - Execute queries against SQLite database
   - Parse and validate query results
   - Store query-result pairs for iteration

7. Reflection & Iteration:
   - Evaluate query results for completeness
   - Determine if refinement is needed
   - Iterate up to MAX_ITERATIONS times
   - Apply reflection decision logic
   - Generate improved queries based on feedback

8. Answer Generation:
   - Synthesize final answer from query results
   - Incorporate PDF context for enrichment
   - Generate follow-up prompt suggestions
   - Format response for user consumption

9. Result Serialization:
   - Convert LangChain messages to serializable format
   - Extract final answer from message list
   - Filter selection codes to only used tables
   - Serialize PDF chunks with metadata
   - Generate dataset reference URLs

10. Cleanup & Monitoring:
    - Force garbage collection to free memory
    - Monitor memory growth and detect leaks
    - Log performance metrics and warnings
    - Return serialized result dictionary

Usage Examples:
--------------
# Command-line execution (direct)
python main.py "What was Prague's population in 2024?"
python main.py --thread_id abc123 --run_id xyz789 "Compare Prague and Brno"

# Programmatic usage (from API)
result = await main(
    prompt="What was Prague's population in 2024?",
    thread_id="session_123",
    checkpointer=shared_checkpointer,
    run_id="trace_456"
)

# Continuing a conversation
result = await main(
    prompt="How did it change from 2023?",
    thread_id="session_123",  # Same thread_id to continue
    checkpointer=shared_checkpointer
)

Required Environment:
-------------------
- Python 3.10+
- PostgreSQL database for production checkpointing
- ChromaDB instance for vector search (optional, has fallback)
- Cohere API key for reranking
- OpenAI/Anthropic API key for LLM agents
- SQLite database with CZSU statistical data
- LangSmith API key for tracing (optional)
- Environment variables configured in .env file

Environment Variables:
--------------------
- POSTGRES_CONNECTION_STRING: PostgreSQL connection for checkpointing
- COHERE_API_KEY: Cohere API key for reranking
- OPENAI_API_KEY or ANTHROPIC_API_KEY: LLM provider keys
- LANGCHAIN_TRACING_V2: Enable LangSmith tracing (true/false)
- LANGCHAIN_API_KEY: LangSmith API key
- LANGCHAIN_PROJECT: LangSmith project name
- GC_MEMORY_THRESHOLD: Memory growth threshold for GC warnings (MB)
- CZSU_SQLITE_URL: Turso SQLite cloud database URL (optional)

Dependencies:
------------
- langgraph: State machine and workflow orchestration
- langchain: LLM framework and message abstractions
- psutil: Memory monitoring and process management
- asyncio: Asynchronous operation support
- psycopg: PostgreSQL adapter for checkpointing
- chromadb: Vector database for semantic search
- cohere: Reranking API client
- openai/anthropic: LLM provider SDKs
- pandas: Data manipulation and analysis
- sqlalchemy: Database abstraction layer

Output Format:
-------------
Returns a dictionary with the following structure:
{
    "prompt": str,                    # Original user prompt
    "result": str,                    # Final answer text
    "queries_and_results": [          # SQL queries and their results
        (query: str, result: str),
        ...
    ],
    "thread_id": str,                 # Thread ID for conversation
    "top_selection_codes": [str],     # Used dataset selection codes
    "iteration": int,                 # Final iteration count
    "max_iterations": int,            # Maximum allowed iterations
    "sql": str,                       # Last SQL query executed
    "datasetUrl": str,                # Reference URL to dataset
    "top_chunks": [                   # Relevant PDF documentation chunks
        {
            "content": str,           # Chunk text content
            "metadata": dict          # Chunk metadata (source, page, etc.)
        },
        ...
    ],
    "followup_prompts": [str]         # Suggested follow-up questions
}

Error Handling:
-------------
- SSL connection errors with automatic retry
- Prepared statement errors with automatic retry
- PostgreSQL checkpointer initialization failures (fallback to InMemory)
- ChromaDB unavailability detection and graceful degradation
- Memory growth monitoring with emergency garbage collection
- Graph execution errors with detailed tracing
- State deserialization errors with logging
- Query execution failures with iteration retry

Memory Management:
-----------------
- Pre-execution garbage collection
- Post-graph execution memory monitoring
- Emergency cleanup on excessive growth (>1900MB threshold)
- Final cleanup before return
- Memory leak warnings for high retention (>100MB)
- psutil-based RSS memory tracking
- Configurable GC_MEMORY_THRESHOLD via environment

Performance Considerations:
-------------------------
- Async/await for I/O-bound operations
- Checkpointing interrupts to optimize state storage
- Minimal state updates for continuing conversations
- Efficient message history management
- Query result caching in conversation state
- Rerank score thresholding to limit processing
- Selection code filtering to only used tables

Architecture Notes:
-----------------
- Follows separation of concerns with dedicated agent nodes
- Uses LangGraph for deterministic state machine workflow
- Implements checkpoint-based memory for conversation persistence
- Separates retrieval (ChromaDB) from reasoning (LLM agents)
- Uses message abstractions for LangChain compatibility
- Supports both development (InMemory) and production (PostgreSQL) modes
- Designed for Railway cloud deployment with uvicorn startup

Deployment Notes:
----------------
- Main script execution is commented out to prevent Railway auto-run
- Railway detects main.py as entry point, conflicts with uvicorn
- Use `python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"` for CLI
- API route in FastAPI backend imports and calls main() function
- Checkpointer is shared across API requests for memory efficiency
- Thread IDs are generated per session, run IDs per request
- LangSmith tracing is enabled via environment variables

Related Modules:
---------------
- my_agent/: LangGraph agent definitions and node implementations
- checkpointer/: PostgreSQL checkpointer factory and configuration
- api/: FastAPI backend routes and dependencies
- data/: Data extraction and SQLite database management
- metadata/: ChromaDB management and schema extraction
- Evaluations/: Evaluation datasets and testing frameworks"""

import asyncio

# ==============================================================================
# IMPORTS AND SYSTEM CONFIGURATION
# ==============================================================================
import sys

# Configure asyncio event loop policy for Windows compatibility
# Required for psycopg (PostgreSQL async driver) on Windows platform
# WindowsSelectorEventLoopPolicy ensures proper async I/O operations
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse  # Command-line argument parsing
import gc  # Garbage collection for memory management
import os  # Operating system interface
import re  # Regular expressions for SQL parsing
import uuid  # UUID generation for unique identifiers
from pathlib import Path  # Object-oriented filesystem paths
from typing import List  # Type hinting for better code clarity

import psutil  # Process and system monitoring (memory tracking)
from dotenv import load_dotenv  # Environment variable loading from .env file
from langchain_core.messages import HumanMessage  # LangChain message abstraction

# Load environment variables from .env file
# Must be loaded before importing modules that use environment variables
load_dotenv()

# Import LangGraph workflow components
from my_agent import create_graph  # Main graph factory function
from my_agent.utils.nodes import MAX_ITERATIONS  # Maximum iteration limit constant

# Import error handling decorators for PostgreSQL connection robustness
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,  # Retry on prepared statement errors
    retry_on_ssl_connection_error,  # Retry on SSL/TLS connection errors
)

# Import checkpointer factory for conversation state persistence
from checkpointer.checkpointer.factory import get_global_checkpointer

# Robust base_dir logic for project root
# Handles both normal execution and edge cases (interactive shells, notebooks)
try:
    # Standard case: __file__ is available
    base_dir = Path(__file__).resolve().parents[0]
except NameError:
    # Fallback for interactive environments without __file__
    base_dir = Path(os.getcwd()).parents[0]

# Add base_dir to sys.path for absolute imports if not already present
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Import debug utility functions for detailed logging
# These functions respect debug flags and provide structured debug output
from api.utils.debug import (
    print__analysis_tracing_debug,  # Traces execution flow through analysis steps
    print__main_debug,  # Logs main execution events
    print__memory_debug,  # Tracks memory usage and leaks
)
# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Default prompt for testing and development
# Multiple examples are commented out to show variety of supported queries
# Current default is Czech language query about liquid fuel production
# To change default, uncomment desired line or modify DEFAULT_PROMPT value

# English examples:
# DEFAULT_PROMPT = "Did Prague have more residents than Central Bohemia at the start of 2024?"
# DEFAULT_PROMPT = "Can you compare number of man and number of woman in prague and in plzen? Create me a bar chart with this data."
# DEFAULT_PROMPT = "How much did Prague's population grow from start to end of Q3?"
# DEFAULT_PROMPT = "What was South Bohemia's population change rate per month?"
# DEFAULT_PROMPT = "Tell me a joke"  # Edge case testing
# DEFAULT_PROMPT = "Is there some very interesting trend in my data?"
# DEFAULT_PROMPT = "What was the maximum female population recorded in any region?"
# DEFAULT_PROMPT = "List regions where the absolute difference between male and female population changes was greater than 3000, and indicate whether men or women changed more"
# DEFAULT_PROMPT = "What is the average population rate of change for regions with more than 1 million residents?"

# Czech examples:
# DEFAULT_PROMPT = "tell me about how many people were in prague at 2024 and compare it with whole republic data? Pak mi dej distribuci kazdeho regionu, v procentech."
# DEFAULT_PROMPT = "tell me about people in prague, compare, contrast, what is interesting, provide trends."
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy v Praze"  # Which field has highest average wages in Prague
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy?"  # Which field has highest average wages
DEFAULT_PROMPT = "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?"  # What was liquid fuel production from oil in Czechia in 2023
# DEFAULT_PROMPT = "Jaký byl podíl osob používajících internet v Česku ve věku 16 a vice v roce 2023?"  # What was the share of internet users in Czechia aged 16+ in 2023

# Multi-line schema description example (commented out):
# DEFAULT_PROMPT = """
# This table contains information about wages and salaries across different industries. It includes data on average wages categorized by economic sectors or industries.
#
# Available columns:
# industry (odvětví): distinct values include manufacturing, IT, construction, healthcare, education, etc.
# average_wage (průměrná mzda): numerical values representing monthly or annual averages
# year: distinct values may include 2020, 2021, 2022, etc.
def extract_table_names_from_sql(sql_query: str) -> List[str]:
    """Extract table names from SQL query FROM and JOIN clauses.

    This function parses SQL queries to identify all referenced table names,
    supporting various SQL formats including quoted identifiers, schema-qualified
    names, and comma-separated table lists. It normalizes table names to uppercase
    for consistent comparison with selection codes.

    Supports:
    - Simple FROM clauses: FROM table_name
    - Schema-qualified: FROM schema.table_name
    - Quoted identifiers: FROM "table_name", FROM 'table_name'
    - Comma-separated: FROM table1, table2, table3
    - JOIN clauses: INNER JOIN, LEFT JOIN, RIGHT JOIN, etc.

    Args:
        sql_query (str): The SQL query string to parse

    Returns:
        List[str]: Unique list of uppercase table names found in the query

    Note:
        - Removes SQL comments before parsing (both -- and /* */ styles)
        - Normalizes whitespace for robust pattern matching
        - Returns uppercase names for case-insensitive comparison
        - Deduplicates table names before returning
    """
    # Remove SQL comments to avoid false matches
    # Single-line comments: -- comment
    sql_clean = re.sub(r"--.*?(?=\n|$)", "", sql_query, flags=re.MULTILINE)
    # Multi-line comments: /* comment */
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    # Normalize whitespace for consistent pattern matching
    sql_clean = " ".join(sql_clean.split())

    # Pattern to match FROM clause with table names
    # Captures: FROM table_name, FROM "table_name", FROM table1, table2
    # Group 2 captures main table, group 4 captures comma-separated tables
    from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'

def get_used_selection_codes(
    queries_and_results: list, top_selection_codes: List[str]
) -> List[str]:
    """Filter selection codes to only include those actually referenced in SQL queries.

    This function analyzes executed SQL queries to determine which dataset selection
    codes (table names) were actually used, filtering out retrieved but unused codes.
    This provides accurate dataset references for the frontend and prevents showing
    irrelevant datasets to users.

    The function is essential for:
    - Generating accurate dataset URLs for reference
    - Showing only relevant datasets in UI
    - Understanding which data sources contributed to the answer
    - Tracking dataset usage analytics

    Args:
        queries_and_results (list): List of (query, result) tuples from graph execution
                                   Each tuple contains an SQL query string and its result
        top_selection_codes (List[str]): List of candidate selection codes from retrieval
                                         These are potential tables but not all may be used

    Returns:
        List[str]: Filtered list of selection codes that appear in actual SQL queries
                  Empty list if no queries or selection codes provided

    Example:
        queries = [
            ("SELECT * FROM OBY01PDT01 WHERE region='Prague'", "...results..."),
            ("SELECT * FROM OBY01PDT02 WHERE year=2023", "...results...")
        ]
        candidates = ["OBY01PDT01", "OBY01PDT02", "OBY01PDT03"]
        used = get_used_selection_codes(queries, candidates)
        # Returns: ["OBY01PDT01", "OBY01PDT02"]
        # OBY01PDT03 is excluded as it wasn't used in any query
    """
    # Early return for empty inputs
    if not queries_and_results or not top_selection_codes:
        return []

    # Extract all table names used in queries
    # Use set for O(1) lookup performance when filtering
    used_table_names = set()
    for query, _ in queries_and_results:
        if query:
            # Parse SQL to extract table names from FROM and JOIN clauses
            table_names = extract_table_names_from_sql(query)
            used_table_names.update(table_names)

    # Filter selection codes to only include those that match used table names
    # Maintains original order of selection codes for consistency
    used_selection_codes = []
    for selection_code in top_selection_codes:
        # Case-insensitive comparison (extract_table_names_from_sql returns uppercase)
        if selection_code.upper() in used_table_names:
            used_selection_codes.append(selection_code)

    return used_selection_codess and chunks

6. Query Generation & Execution:
   - Rewrite user prompt for SQL generation
   - Generate SQL queries using LLM agents
   - Execute queries against SQLite database
   - Parse and validate query results
   - Store query-result pairs for iteration

7. Reflection & Iteration:
   - Evaluate query results for completeness
   - Determine if refinement is needed
   - Iterate up to MAX_ITERATIONS times
   - Apply reflection decision logic
   - Generate improved queries based on feedback

8. Answer Generation:
   - Synthesize final answer from query results
   - Incorporate PDF context for enrichment
   - Generate follow-up prompt suggestions
   - Format response for user consumption

9. Result Serialization:
   - Convert LangChain messages to serializable format
   - Extract final answer from message list
   - Filter selection codes to only used tables
   - Serialize PDF chunks with metadata
   - Generate dataset reference URLs

10. Cleanup & Monitoring:
    - Force garbage collection to free memory
    - Monitor memory growth and detect leaks
    - Log performance metrics and warnings
    - Return serialized result dictionary

Usage Examples:
--------------
# Command-line execution (direct)
python main.py "What was Prague's population in 2024?"
python main.py --thread_id abc123 --run_id xyz789 "Compare Prague and Brno"

# Programmatic usage (from API)
result = await main(
    prompt="What was Prague's population in 2024?",
    thread_id="session_123",
    checkpointer=shared_checkpointer,
    run_id="trace_456"
)

# Continuing a conversation
result = await main(
    prompt="How did it change from 2023?",
    thread_id="session_123",  # Same thread_id to continue
    checkpointer=shared_checkpointer
)

Required Environment:
-------------------
- Python 3.10+
- PostgreSQL database for production checkpointing
- ChromaDB instance for vector search (optional, has fallback)
- Cohere API key for reranking
- OpenAI/Anthropic API key for LLM agents
- SQLite database with CZSU statistical data
- LangSmith API key for tracing (optional)
- Environment variables configured in .env file

Environment Variables:
--------------------
- POSTGRES_CONNECTION_STRING: PostgreSQL connection for checkpointing
- COHERE_API_KEY: Cohere API key for reranking
- OPENAI_API_KEY or ANTHROPIC_API_KEY: LLM provider keys
- LANGCHAIN_TRACING_V2: Enable LangSmith tracing (true/false)
- LANGCHAIN_API_KEY: LangSmith API key
- LANGCHAIN_PROJECT: LangSmith project name
- GC_MEMORY_THRESHOLD: Memory growth threshold for GC warnings (MB)
- CZSU_SQLITE_URL: Turso SQLite cloud database URL (optional)

Dependencies:
------------
- langgraph: State machine and workflow orchestration
- langchain: LLM framework and message abstractions
- psutil: Memory monitoring and process management
- asyncio: Asynchronous operation support
- psycopg: PostgreSQL adapter for checkpointing
- chromadb: Vector database for semantic search
- cohere: Reranking API client
- openai/anthropic: LLM provider SDKs
- pandas: Data manipulation and analysis
- sqlalchemy: Database abstraction layer

Output Format:
-------------
Returns a dictionary with the following structure:
{
    "prompt": str,                    # Original user prompt
    "result": str,                    # Final answer text
    "queries_and_results": [          # SQL queries and their results
        (query: str, result: str),
        ...
    ],
    "thread_id": str,                 # Thread ID for conversation
    "top_selection_codes": [str],     # Used dataset selection codes
    "iteration": int,                 # Final iteration count
    "max_iterations": int,            # Maximum allowed iterations
    "sql": str,                       # Last SQL query executed
    "datasetUrl": str,                # Reference URL to dataset
    "top_chunks": [                   # Relevant PDF documentation chunks
        {
            "content": str,           # Chunk text content
            "metadata": dict          # Chunk metadata (source, page, etc.)
        },
        ...
    ],
    "followup_prompts": [str]         # Suggested follow-up questions
}

Error Handling:
-------------
- SSL connection errors with automatic retry
- Prepared statement errors with automatic retry
- PostgreSQL checkpointer initialization failures (fallback to InMemory)
- ChromaDB unavailability detection and graceful degradation
- Memory growth monitoring with emergency garbage collection
- Graph execution errors with detailed tracing
- State deserialization errors with logging
- Query execution failures with iteration retry

Memory Management:
-----------------
- Pre-execution garbage collection
- Post-graph execution memory monitoring
- Emergency cleanup on excessive growth (>1900MB threshold)
- Final cleanup before return
- Memory leak warnings for high retention (>100MB)
- psutil-based RSS memory tracking
- Configurable GC_MEMORY_THRESHOLD via environment

Performance Considerations:
-------------------------
- Async/await for I/O-bound operations
- Checkpointing interrupts to optimize state storage
- Minimal state updates for continuing conversations
- Efficient message history management
- Query result caching in conversation state
- Rerank score thresholding to limit processing
- Selection code filtering to only used tables

Architecture Notes:
-----------------
- Follows separation of concerns with dedicated agent nodes
- Uses LangGraph for deterministic state machine workflow
- Implements checkpoint-based memory for conversation persistence
- Separates retrieval (ChromaDB) from reasoning (LLM agents)
- Uses message abstractions for LangChain compatibility
- Supports both development (InMemory) and production (PostgreSQL) modes
- Designed for Railway cloud deployment with uvicorn startup

Deployment Notes:
----------------
- Main script execution is commented out to prevent Railway auto-run
- Railway detects main.py as entry point, conflicts with uvicorn
- Use `python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"` for CLI
- API route in FastAPI backend imports and calls main() function
- Checkpointer is shared across API requests for memory efficiency
- Thread IDs are generated per session, run IDs per request
- LangSmith tracing is enabled via environment variables

Related Modules:
---------------
- my_agent/: LangGraph agent definitions and node implementations
- checkpointer/: PostgreSQL checkpointer factory and configuration
- api/: FastAPI backend routes and dependencies
- data/: Data extraction and SQLite database management
- metadata/: ChromaDB management and schema extraction
- Evaluations/: Evaluation datasets and testing frameworks"""

import asyncio

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys

# Configure asyncio event loop policy for Windows compatibility with psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import gc
import os
import re
import uuid
from pathlib import Path
from typing import List

import psutil
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

from my_agent import create_graph
from my_agent.utils.nodes import MAX_ITERATIONS
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
    retry_on_ssl_connection_error,
)
from checkpointer.checkpointer.factory import get_global_checkpointer

# Robust base_dir logic for project root
try:
    base_dir = Path(__file__).resolve().parents[0]
except NameError:
    base_dir = Path(os.getcwd()).parents[0]
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Import debug functions from utils
from api.utils.debug import (
    print__analysis_tracing_debug,
    print__main_debug,
    print__memory_debug,
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Default prompt if none provided
# DEFAULT_PROMPT = "Did Prague have more residents than Central Bohemia at the start of 2024?"
# DEFAULT_PROMPT = "Can you compare number of man and number of woman in prague and in plzen? Create me a bar chart with this data."
# DEFAULT_PROMPT = "How much did Prague's population grow from start to end of Q3?"
# DEFAULT_PROMPT = "What was South Bohemia's population change rate per month?"
# DEFAULT_PROMPT = "Tell me a joke"
# DEFAULT_PROMPT = "Is there some very interesting trend in my data?"
# DEFAULT_PROMPT = "tell me about how many people were in prague at 2024 and compare it with whole republic data? Pak mi dej distribuci kazdeho regionu, v procentech."
# DEFAULT_PROMPT = "tell me about people in prague, compare, contrast, what is interesting, provide trends."
# DEFAULT_PROMPT = "What was the maximum female population recorded in any region?"
# DEFAULT_PROMPT = "List regions where the absolute difference between male and female population changes was greater than 3000, and indicate whether men or women changed more"
# DEFAULT_PROMPT = "What is the average population rate of change for regions with more than 1 million residents?"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy v Praze"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy?"
# DEFAULT_PROMPT = """
# This table contains information about wages and salaries across different industries. It includes data on average wages categorized by economic sectors or industries.

# Available columns:
# industry (odvětví): distinct values include manufacturing, IT, construction, healthcare, education, etc.
# average_wage (průměrná mzda): numerical values representing monthly or annual averages
# year: distinct values may include 2020, 2021, 2022, etc.
# measurement_unit: e.g., CZK, EUR, USD per month/year
# The table allows comparison of wage levels across different economic sectors.
# """
DEFAULT_PROMPT = "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?"
# DEFAULT_PROMPT = "Jaký byl podíl osob používajících internet v Česku ve věku 16 a vice v roce 2023?"

def generate_initial_followup_prompts() -> List[str]:
    """Generate initial follow-up prompt suggestions for new conversations using dynamic templates.

    This function generates diverse, contextually relevant starter suggestions for new
    users by filling pre-defined templates with random selections from topic categories.
    These prompts are displayed when users start a new chat session, providing inspiration
    and demonstrating the system's capabilities.

    The function uses a pseudo-random seed based on current timestamp to ensure variety
    across different user sessions while maintaining reproducibility within a session.
    This prevents users from seeing the same suggestions repeatedly.

    Template categories include:
    - Population and demographics
    - Employment and labor market
    - Economic indicators (GDP, wages, trade)
    - Social statistics (education, healthcare, migration)
    - Regional comparisons and trends

    Returns:
        List[str]: A list of 5 dynamically generated follow-up prompts
                  Each prompt is filled with context-appropriate terms
                  Examples are diverse across different statistical domains

    Note:
        - Uses timestamp-based seeding for variety across sessions
        - Ensures no duplicate templates in a single generation
        - Prompts are designed to be natural and conversational
        - Covers broad range of CZSU data topics
        - Suitable for both Czech and international users
    """
    Returns:
        List of table names found in FROM clauses
    """
    # Remove comments and normalize whitespace
    sql_clean = re.sub(r"--.*?(?=\n|$)", "", sql_query, flags=re.MULTILINE)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    sql_clean = " ".join(sql_clean.split())

    # Pattern to match FROM clause with table names
    # This handles: FROM table_name, FROM schema.table_name, FROM "table_name", etc.
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

    # Also handle JOIN clauses
    join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
    join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)

    for match in join_matches:
        if match.group(2):
            table_names.append(match.group(2).upper())

    return list(set(table_names))  # Remove duplicates


def get_used_selection_codes(
    queries_and_results: list, top_selection_codes: List[str]
) -> List[str]:
    """Filter top_selection_codes to only include those actually used in queries.

    Args:
        queries_and_results: List of (query, result) tuples
        top_selection_codes: List of all candidate selection codes

    Returns:
        List of selection codes that were actually used in the queries
    """
    if not queries_and_results or not top_selection_codes:
        return []

    # Extract all table names used in queries
    used_table_names = set()
    for query, _ in queries_and_results:
        if query:
            table_names = extract_table_names_from_sql(query)
            used_table_names.update(table_names)

    # Filter selection codes to only include those that match used table names
    used_selection_codes = []
    for selection_code in top_selection_codes:
        if selection_code.upper() in used_table_names:
            used_selection_codes.append(selection_code)

    return used_selection_codes


def generate_initial_followup_prompts() -> List[str]:
    """Generate initial follow-up prompt suggestions for new conversations using dynamic templates.

    This function generates diverse starter suggestions using pre-defined templates
    filled with random selections to ensure variety. These prompts will be displayed
    to users when they start a new chat, giving them ideas for questions they can ask
    about Czech Statistical Office data.

    Returns:
        List[str]: A list of dynamically generated suggested follow-up prompts for the user
    """
    print__main_debug(
        "🎯 PROMPT GEN: Starting dynamic template-based prompt generation"
    )

    # Generate dynamic prompts based on current timestamp to ensure variety
    import random
    import time

    # Use timestamp as seed for pseudo-randomness
    seed = int(time.time() * 1000) % 1000000
    random.seed(seed)

    # Pool of diverse prompt templates
    prompt_templates = [
        "What are the population trends in {region}?",
        "Show me employment statistics by {category}.",
        "Compare {metric} growth across different years.",
        "What are the latest statistics on {topic}?",
        "How has {indicator} changed in recent {period}?",
        "What are the {type} rates in {location}?",
        "Show me data about {subject} from {source}.",
        "What trends can you see in {area} statistics?",
        "Compare {metric} between {group1} and {group2}.",
        "What are the current {indicator} figures for {region}?",
        "Tell me about {topic} in {location}.",
        "Show me {subject} statistics for {period}.",
        "What is the {indicator} situation in {region}?",
        "Compare {metric} across {group1} and {group2}.",
        "What are the trends in {area} data?",
    ]

    # Fill in the templates with random selections
    regions = [
        "Prague",
        "Czech Republic",
        "major cities",
        "different regions",
        "Brno",
    ]
    categories = [
        "region",
        "industry",
        "age group",
        "education level",
        "sector",
    ]
    metrics = [
        "GDP",
        "employment",
        "population",
        "export",
        "import",
        "wage",
    ]
    topics = [
        "crime rates",
        "healthcare spending",
        "education levels",
        "housing prices",
        "migration",
        "birth rates",
    ]
    periods = [
        "years",
        "quarters",
        "months",
        "decades",
        "recent years",
    ]
    types = [
        "unemployment",
        "inflation",
        "birth",
        "migration",
        "divorce",
    ]
    locations = [
        "Prague",
        "Brno",
        "Czech Republic",
        "major regions",
    ]
    subjects = [
        "agricultural production",
        "industrial output",
        "tourism numbers",
        "energy consumption",
        "trade balance",
    ]
    sources = [
        "government reports",
        "statistical surveys",
        "economic indicators",
        "census data",
        "official statistics",
    ]
    areas = [
        "labor market",
        "demographic",
        "economic",
        "environmental",
        "social",
        "health",
    ]
    indicators = [
        "unemployment",
        "inflation",
        "GDP growth",
        "population",
        "wage growth",
        "export growth",
    ]
    group1_group2 = [
        ("urban and rural areas", "rural areas"),
        ("men and women", "women"),
        ("young and old", "older population"),
        ("public and private sector", "private companies"),
        ("domestic and foreign", "foreign companies"),
        ("large and small enterprises", "small businesses"),
    ]

    # Generate 5 unique prompts
    generated_prompts = []
    used_templates = set()

    while len(generated_prompts) < 5:
        template = random.choice(prompt_templates)
        if template in used_templates:
            continue
        used_templates.add(template)

        # Fill in template variables
        prompt = template
        if "{region}" in prompt:
            prompt = prompt.replace("{region}", random.choice(regions))
        if "{category}" in prompt:
            prompt = prompt.replace("{category}", random.choice(categories))
        if "{metric}" in prompt:
            prompt = prompt.replace("{metric}", random.choice(metrics))
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(topics))
        if "{period}" in prompt:
            prompt = prompt.replace("{period}", random.choice(periods))
        if "{type}" in prompt:
            prompt = prompt.replace("{type}", random.choice(types))
        if "{location}" in prompt:
            prompt = prompt.replace("{location}", random.choice(locations))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(subjects))
        if "{source}" in prompt:
            prompt = prompt.replace("{source}", random.choice(sources))
        if "{area}" in prompt:
            prompt = prompt.replace("{area}", random.choice(areas))
        if "{indicator}" in prompt:
            prompt = prompt.replace("{indicator}", random.choice(indicators))
        if "{group1} and {group2}" in prompt:
            g1, g2 = random.choice(group1_group2)
            prompt = prompt.replace("{group1}", g1).replace("{group2}", g2)

        generated_prompts.append(prompt)

    final_prompts = generated_prompts
    print__main_debug(f"🎲 Generated {len(final_prompts)} dynamic prompts")
    for i, p in enumerate(final_prompts, 1):
        print__main_debug(f"   {i}. {p}")

    return final_prompts


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
@retry_on_ssl_connection_error(max_retries=3)
@retry_on_prepared_statement_error(max_retries=3)
async def main(prompt=None, thread_id=None, checkpointer=None, run_id=None):
    """Main entry point and orchestrator for the CZSU multi-agent text-to-SQL analysis system.

    This async function serves as the central coordinator for the data analysis workflow,
    managing the complete lifecycle from prompt acquisition through result generation
    and memory persistence. It orchestrates LangGraph execution, handles state management,
    monitors memory usage, and ensures robust error recovery.

    The function is designed to support three execution modes:
    1. API mode: Called from FastAPI routes with all parameters provided
    2. CLI mode: Called from command line with argument parsing
    3. Library mode: Imported and called programmatically

    Key Responsibilities:
    -------------------
    - Prompt sourcing and validation (CLI args, function params, or defaults)
    - Thread ID management for conversation persistence
    - Run ID generation for LangSmith tracing
    - Checkpointer initialization with PostgreSQL/InMemory fallback
    - State initialization (new vs. continuing conversations)
    - LangGraph workflow execution and monitoring
    - Memory leak detection and garbage collection
    - Result serialization and dataset reference generation
    - Follow-up prompt generation for user guidance
    - Error handling with automatic retry for network issues

    Conversation State Management:
    ----------------------------
    New Conversations:
    - Initialize complete DataAnalysisState with all fields
    - Generate initial follow-up prompt suggestions
    - Create new message history with HumanMessage
    - Set all retrieval and query fields to empty/None
    
    Continuing Conversations:
    - Fetch existing state from checkpointer
    - Reset only fields that need refresh (prompt, queries, retrieval results)
    - Preserve message history and thread context
    - Maintain conversation memory across turns

    Memory Management Strategy:
    -------------------------
    1. Pre-execution baseline: Capture initial RSS memory
    2. Pre-execution GC: Force garbage collection before workflow
    3. Post-graph monitoring: Check memory growth after LangGraph
    4. Emergency cleanup: Trigger GC if growth exceeds threshold (default 1900MB)
    5. Final cleanup: Force GC before return
    6. Leak warnings: Alert if total growth exceeds 100MB

    Error Recovery:
    --------------
    - SSL connection errors: Automatic retry up to 3 attempts
    - Prepared statement errors: Automatic retry up to 3 attempts
    - PostgreSQL checkpointer failure: Fallback to InMemorySaver
    - State deserialization errors: Logged with detailed context
    - Graph execution errors: Caught and traced with LangSmith

    Args:
        prompt (str, optional): The natural language query to process.
                               If None and __name__ == "__main__", reads from command line args.
                               If still None, uses DEFAULT_PROMPT constant.
                               
        thread_id (str, optional): Unique identifier for conversation thread.
                                  Enables multi-turn conversations with memory.
                                  If None, generates new ID: f"data_analysis_{uuid}"
                                  Same thread_id continues existing conversation.
                                  
        checkpointer (optional): Checkpointer instance for state persistence.
                                AsyncPostgresSaver for production (shared across API requests).
                                InMemorySaver for development (not persisted).
                                If None, calls get_global_checkpointer() with fallback.
                                
        run_id (str, optional): Unique identifier for LangSmith trace.
                               Enables tracking individual request execution.
                               If None, generates new UUID.
                               Visible in LangSmith dashboard for debugging.

    Returns:
        dict: Serialized result dictionary with the following structure:
        {
            "prompt": str,                     # Original user query
            "result": str,                     # Final answer (from last AIMessage)
            "queries_and_results": [           # History of SQL queries and results
                (sql_query: str, result: str),
                ...
            ],
            "thread_id": str,                  # Thread ID for conversation
            "top_selection_codes": [str],      # Dataset codes actually used in queries
            "iteration": int,                  # Number of iterations performed
            "max_iterations": int,             # Maximum allowed iterations (from config)
            "sql": str | None,                 # Last SQL query executed (if any)
            "datasetUrl": str | None,          # Reference URL to first used dataset
            "top_chunks": [                    # Relevant PDF documentation chunks
                {
                    "content": str,            # Chunk text
                    "metadata": dict           # Source, page, etc.
                },
                ...
            ],
            "followup_prompts": [str]          # Suggested follow-up questions
        }

    Raises:
        Exception: Re-raised after max retries for SSL/prepared statement errors
        
    Side Effects:
        - Creates/updates conversation state in checkpointer
        - Logs debug messages via print__* functions
        - Forces garbage collection multiple times
        - Queries ChromaDB and SQLite databases
        - Makes API calls to LLM providers (OpenAI/Anthropic)
        - Records traces in LangSmith (if enabled)

    Examples:
        # API mode (typical FastAPI usage)
        result = await main(
            prompt="What was Prague's population?",
            thread_id=session.thread_id,
            checkpointer=app_checkpointer,
            run_id=str(uuid.uuid4())
        )
        
        # CLI mode (automatic argument parsing)
        if __name__ == "__main__":
            result = await main()  # Reads from sys.argv
            
        # Continuing conversation
        result1 = await main("What was Prague's population?", thread_id="abc")
        result2 = await main("How about Brno?", thread_id="abc")  # Same thread

    Note:
        - Function is decorated with retry logic for PostgreSQL connection issues
        - Windows requires WindowsSelectorEventLoopPolicy (set at module level)
        - Railway deployment requires commenting out __name__ == "__main__" block
        - Memory warnings at >100MB growth suggest potential leaks in LangGraph nodes
        - Empty top_chunks indicates no relevant PDF documentation found
        - Empty followup_prompts for continuing conversations (only in new chats)
    """
    print__analysis_tracing_debug("29 - MAIN ENTRY: main() function entry point")
    print__main_debug("29 - MAIN ENTRY: main() function entry point")

    # Handle prompt sourcing - command line args have priority over defaults
    # This allows flexibility in how the application is used
    if prompt is None and __name__ == "__main__":
        print__analysis_tracing_debug(
            "30 - COMMAND LINE ARGS: Processing command line arguments"
        )
        parser = argparse.ArgumentParser(description="Run data analysis with LangGraph")
        parser.add_argument(
            "prompt",
            nargs="?",
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

    # Ensure we always have a valid prompt to avoid None-type errors downstream
    if prompt is None:
        print__analysis_tracing_debug("31 - DEFAULT PROMPT: Using default prompt")
        prompt = DEFAULT_PROMPT
    else:
        print__analysis_tracing_debug(
            f"32 - PROMPT PROVIDED: Using provided prompt (length: {len(prompt)})"
        )

    # Use a thread_id for short-term memory (thread-level persistence)
    if thread_id is None:
        thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
        print__analysis_tracing_debug(
            f"33 - THREAD ID GENERATED: Generated new thread_id {thread_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"34 - THREAD ID PROVIDED: Using provided thread_id {thread_id}"
        )

    # Generate run_id if not provided (for command-line usage)
    if run_id is None:
        run_id = str(uuid.uuid4())
        print__analysis_tracing_debug(
            f"35 - RUN ID GENERATED: Generated new run_id {run_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"36 - RUN ID PROVIDED: Using provided run_id {run_id}"
        )

    # Initialize tracing for debugging and performance monitoring
    # This is crucial for production deployments to track execution paths
    # instrument(project_name="LangGraph_czsu-multi-agent-text-to-sql", framework=Framework.LANGGRAPH)

    print__analysis_tracing_debug("37 - MEMORY MONITORING: Starting memory monitoring")
    # MEMORY LEAK PREVENTION: Track memory before and after analysis
    # Memory monitoring before analysis
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    print__memory_debug(f"🔍 MEMORY: Starting analysis with {memory_before:.1f}MB RSS")
    print__analysis_tracing_debug(
        f"38 - MEMORY BASELINE: Memory before analysis: {memory_before:.1f}MB RSS"
    )

    # Force garbage collection before starting
    collected = gc.collect()
    print__memory_debug(f"🧹 MEMORY: Pre-analysis GC collected {collected} objects")

    print__analysis_tracing_debug("40 - CHECKPOINTER SETUP: Setting up checkpointer")
    # Create the LangGraph execution graph with standard AsyncPostgresSaver
    # We use interrupt_after=['save'] and minimal state in save_node to optimize storage
    if checkpointer is None:
        try:
            print__analysis_tracing_debug(
                "41 - POSTGRES CHECKPOINTER: Attempting to get PostgreSQL checkpointer"
            )
            checkpointer = await get_global_checkpointer()
            print__analysis_tracing_debug(
                "42 - POSTGRES SUCCESS: PostgreSQL checkpointer obtained"
            )
        except Exception as e:
            print__analysis_tracing_debug(
                f"43 - POSTGRES FAILED: Failed to initialize PostgreSQL checkpointer - {str(e)}"
            )
            print__main_debug(f"⚠️ Failed to initialize PostgreSQL checkpointer: {e}")
            # Fallback to InMemorySaver to ensure application still works
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
            print__analysis_tracing_debug(
                "44 - INMEMORY FALLBACK: Using InMemorySaver fallback"
            )
            print__main_debug("⚠️ Using InMemorySaver fallback")
    else:
        print__analysis_tracing_debug(
            f"45 - CHECKPOINTER PROVIDED: Using provided checkpointer ({type(checkpointer).__name__})"
        )

    print__analysis_tracing_debug(
        "46 - GRAPH CREATION: Creating LangGraph execution graph"
    )
    graph = create_graph(checkpointer=checkpointer)
    print__analysis_tracing_debug(
        "47 - GRAPH CREATED: LangGraph execution graph created successfully"
    )

    # FIX: Escape curly braces in prompt to prevent f-string parsing errors
    prompt_escaped = prompt.replace("{", "{{").replace("}", "}}")
    print__main_debug(
        f"🚀 Processing prompt: {prompt_escaped} (thread_id={thread_id}, run_id={run_id})"
    )
    print__analysis_tracing_debug(
        f"48 - PROCESSING START: Processing prompt with thread_id={thread_id}, run_id={run_id}"
    )

    # Configuration for thread-level persistence and LangSmith tracing
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id}
    print__analysis_tracing_debug(
        "49 - CONFIG SETUP: Configuration for thread-level persistence and LangSmith tracing"
    )

    print__analysis_tracing_debug("50 - STATE CHECK: Checking for existing state")
    # Check if there's existing state for this thread to determine if this is a new or continuing conversation
    try:
        existing_state = await graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        is_continuing_conversation = (
            existing_state
            and existing_state.values
            and existing_state.values.get("messages")
            and len(existing_state.values.get("messages", [])) > 0
        )
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
        print__main_debug(f"❌ Error checking existing state: {e}")
        print__analysis_tracing_debug(
            f"54 - STATE CHECK ERROR: Error checking existing state - {str(e)}"
        )
        is_continuing_conversation = False

    print__analysis_tracing_debug("55 - STATE PREPARATION: Preparing input state")
    # Prepare input state based on whether this is a new or continuing conversation
    if is_continuing_conversation:
        print__analysis_tracing_debug(
            "56 - CONTINUING CONVERSATION: Preparing state for continuing conversation"
        )
        # For continuing conversations, pass only the fields that need to be updated
        # The checkpointer will merge this with the existing state
        # CRITICAL FIX: Also reset rewritten_prompt and queries to prevent double execution
        input_state = {
            "prompt": prompt,
            "rewritten_prompt": None,  # Critical: reset to force fresh rewrite
            "iteration": 0,  # Reset for new question
            "queries_and_results": [],  # Critical: reset queries to prevent using old ones
            "followup_prompts": [],  # Reset follow-up prompts for new question
            "final_answer": "",  # Reset final answer
            # Reset retrieval results to force fresh search
            "hybrid_search_results": [],
            "most_similar_selections": [],
            "top_selection_codes": [],
            "hybrid_search_chunks": [],
            "most_similar_chunks": [],
            "top_chunks": [],
        }
    else:
        print__analysis_tracing_debug(
            "57 - NEW CONVERSATION: Preparing state for new conversation"
        )
        # Generate initial follow-up prompts for new conversations
        initial_followup_prompts = generate_initial_followup_prompts()
        print__main_debug(
            f"💡 Generated {len(initial_followup_prompts)} initial follow-up prompts for new conversation"
        )

        # For new conversations, initialize with COMPLETE state including ALL fields from DataAnalysisState
        # CRITICAL FIX: All state fields must be initialized for checkpointing to work properly
        input_state = {
            # Basic conversation fields
            "prompt": prompt,
            "rewritten_prompt": None,
            # Initialize messages with the prompt as a HumanMessage for LangSmith Input visibility
            "messages": [HumanMessage(content=prompt)],
            "iteration": 0,
            "queries_and_results": [],
            "chromadb_missing": False,
            "final_answer": "",  # Initialize final_answer field
            # MISSING FIELDS - These were causing checkpoint storage issues
            "reflection_decision": "",  # Last decision from reflection node
            "hybrid_search_results": [],  # Intermediate hybrid search results before reranking
            "most_similar_selections": [],  # List of (selection_code, cohere_rerank_score) after reranking
            "top_selection_codes": [],  # List of top N selection codes
            # PDF chunk functionality states
            "hybrid_search_chunks": [],  # Intermediate hybrid search results for PDF chunks
            "most_similar_chunks": [],  # List of (document, cohere_rerank_score) after reranking PDF chunks
            "top_chunks": [],  # List of top N PDF chunks that passed relevance threshold
            # Follow-up prompts functionality
            "followup_prompts": initial_followup_prompts,  # Pre-populated with initial suggestions
        }

    print__analysis_tracing_debug("58 - GRAPH EXECUTION: Starting LangGraph execution")
    print__main_debug(
        f"🚀 About to call graph.ainvoke() with thread_id={thread_id}, run_id={run_id}"
    )
    print__main_debug(f"🚀 Input state keys: {list(input_state.keys())}")

    # Execute the graph with checkpoint configuration and run_id for LangSmith tracing
    # Checkpoints allow resuming execution if interrupted and maintaining conversation memory
    result = await graph.ainvoke(input_state, config=config)

    print__main_debug(
        f"✅ graph.ainvoke() completed for thread_id={thread_id}, run_id={run_id}"
    )

    print__analysis_tracing_debug(
        "59 - GRAPH EXECUTION COMPLETE: LangGraph execution completed"
    )
    # MEMORY LEAK PREVENTION: Monitor memory after graph execution
    memory_after_graph = process.memory_info().rss / 1024 / 1024
    memory_growth_graph = memory_after_graph - memory_before
    print__memory_debug(
        f"🔍 MEMORY: After graph execution: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )
    print__memory_debug(
        f"60 - MEMORY CHECK: Memory after graph: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )

    if memory_growth_graph > float(os.environ.get("GC_MEMORY_THRESHOLD", "1900")):
        print__memory_debug(
            f"⚠️ MEMORY: Suspicious growth detected: {memory_growth_graph:.1f}MB during graph execution!"
        )
        print__analysis_tracing_debug(
            f"61 - MEMORY WARNING: Suspicious memory growth detected: {memory_growth_graph:.1f}MB"
        )

        print__memory_debug(
            f"🚨 MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - implementing emergency cleanup"
        )
        print__analysis_tracing_debug(
            f"62 - MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - emergency cleanup"
        )

        # Emergency garbage collection
        collected = gc.collect()
        print__memory_debug(f"🧹 MEMORY: Emergency GC collected {collected} objects")
        print__analysis_tracing_debug(
            f"63 - EMERGENCY GC: Emergency GC collected {collected} objects"
        )

        # Check memory after emergency GC
        memory_after_gc = process.memory_info().rss / 1024 / 1024
        freed_by_gc = memory_after_graph - memory_after_gc
        print__memory_debug(
            f"🧹 MEMORY: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )
        print__memory_debug(
            f"64 - EMERGENCY GC RESULT: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )

        # Update memory tracking
        memory_after_graph = memory_after_gc
        memory_growth_graph = memory_after_graph - memory_before

    print__analysis_tracing_debug("65 - RESULT PROCESSING: Processing graph result")
    # Log details about the result to understand memory usage
    try:
        result_size = len(str(result)) / 1024 if result else 0  # Size in KB
        print__memory_debug(f"🔍 MEMORY: Result object size: {result_size:.1f}KB")
        print__analysis_tracing_debug(
            f"66 - RESULT SIZE: Result object size: {result_size:.1f}KB"
        )
    except:
        print__memory_debug("🔍 MEMORY: Could not determine result size")

    print__analysis_tracing_debug(
        "68 - FINAL CLEANUP: Starting final cleanup and monitoring"
    )
    # MEMORY LEAK PREVENTION: Final cleanup and monitoring before return
    try:
        # Final garbage collection to clean up any temporary objects from graph execution
        collected = gc.collect()
        print__memory_debug(
            f"🧹 MEMORY: Final cleanup GC collected {collected} objects"
        )
        print__analysis_tracing_debug(
            f"69 - FINAL GC: Final cleanup GC collected {collected} objects"
        )

        # Final memory check
        memory_final = process.memory_info().rss / 1024 / 1024
        total_growth = memory_final - memory_before

        print__memory_debug(
            f"🔍 MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )
        print__analysis_tracing_debug(
            f"70 - FINAL MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )

        # Warn about high memory retention patterns
        if total_growth > 100:  # More than 100MB total growth
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
        print__memory_debug(
            f"⚠️ MEMORY: Error during final memory check: {memory_error}"
        )
        print__memory_debug(
            f"72 - MEMORY ERROR: Error during final memory check - {str(memory_error)}"
        )

    print__analysis_tracing_debug(
        "73 - RESULT EXTRACTION: Extracting values from graph result"
    )
    # Extract values from the graph result dictionary
    # The graph now uses a messages list: [summary (SystemMessage), last_message (AIMessage)]
    queries_and_results = result["queries_and_results"]
    final_answer = (
        result["messages"][-1].content
        if result.get("messages") and len(result["messages"]) > 1
        else ""
    )

    # Use top_selection_codes for dataset reference (use first if available)
    top_selection_codes = result.get("top_selection_codes", [])
    sql_query = queries_and_results[-1][0] if queries_and_results else None
    followup_prompts = result.get("followup_prompts", [])

    print__analysis_tracing_debug(
        f"74 - SELECTION CODES: Processing {len(top_selection_codes)} selection codes"
    )
    print__analysis_tracing_debug(
        f"74a - FOLLOWUP PROMPTS: Extracted {len(followup_prompts)} follow-up prompts from graph result"
    )
    # Filter to only include selection codes actually used in queries
    used_selection_codes = get_used_selection_codes(
        queries_and_results, top_selection_codes
    )
    print__analysis_tracing_debug(
        f"75 - USED CODES: {len(used_selection_codes)} selection codes actually used"
    )

    dataset_url = None
    if used_selection_codes:
        dataset_url = f"/datasets/{used_selection_codes[0]}"
        print__analysis_tracing_debug(
            f"76 - DATASET URL: Generated dataset URL: {dataset_url}"
        )

    print__analysis_tracing_debug(
        "77 - TOP CHUNKS SERIALIZATION: Converting top_chunks to JSON-serializable format"
    )
    # Convert the result to a JSON-serializable format
    # Convert top_chunks (Document objects) to JSON-serializable format
    top_chunks_serialized = []
    if result.get("top_chunks"):
        chunk_count = len(result["top_chunks"])
        print__main_debug(f"📦 main.py - Found {chunk_count} top_chunks to serialize")
        print__analysis_tracing_debug(
            f"78 - CHUNKS FOUND: Found {chunk_count} top_chunks to serialize"
        )
        for i, chunk in enumerate(result["top_chunks"]):
            chunk_data = {
                "content": (
                    chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                ),
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
            top_chunks_serialized.append(chunk_data)
            if i == 0:  # Log first chunk for debugging
                content_preview = chunk_data["content"][:100]
                print__main_debug(
                    f"🔍 main.py - First chunk content preview: {content_preview}..."
                )
                print__analysis_tracing_debug(
                    f"79 - FIRST CHUNK: First chunk content preview: {content_preview}..."
                )
    else:
        print__main_debug("⚠️ main.py - No top_chunks found in result")
        print__analysis_tracing_debug("80 - NO CHUNKS: No top_chunks found in result")

    print__analysis_tracing_debug(
        "81 - RESULT SERIALIZATION: Creating serializable result dictionary"
    )
    serializable_result = {
        "prompt": prompt,
        "result": final_answer,
        "queries_and_results": queries_and_results,
        "thread_id": thread_id,
        "top_selection_codes": used_selection_codes,  # Return only codes actually used in queries
        "iteration": result.get("iteration", 0),
        "max_iterations": MAX_ITERATIONS,
        "sql": sql_query,
        "datasetUrl": dataset_url,
        "top_chunks": top_chunks_serialized,  # Add serialized PDF chunks for frontend
        "followup_prompts": followup_prompts,  # Add follow-up prompts from graph state
    }

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
# Note: This block is commented out to prevent Railway from auto-executing this file.
# Railway's RAILPACK builder was detecting main.py as an entry point and running it
# instead of executing the startCommand (uvicorn).
#
# To run the analysis CLI manually, use:
#   python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"
# Or create a separate CLI script that imports and calls main()
#
# if __name__ == "__main__":
#     asyncio.run(main())

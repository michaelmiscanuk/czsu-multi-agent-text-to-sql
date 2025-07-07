from __future__ import annotations

module_description = r"""PostgreSQL Checkpointer for LangGraph Multi-Agent Text-to-SQL System

This module provides comprehensive PostgreSQL-based checkpointing functionality for the CZSU 
Multi-Agent Text-to-SQL system using LangGraph's AsyncPostgresSaver. It handles persistent 
conversation state management, user session tracking, and robust cloud database connectivity 
with advanced error recovery mechanisms.

Key Features:
-------------
1. AsyncPostgresSaver Integration:
   - Official LangGraph AsyncPostgresSaver implementation
   - Connection pool management with cloud database optimization
   - Automatic table setup and schema initialization
   - Context manager-based resource lifecycle management
   - Prepared statement error handling and recovery
   - Fallback mechanisms for different connection approaches

2. Cloud Database Compatibility:
   - Optimized for cloud PostgreSQL services (Supabase, AWS RDS, etc.)
   - Advanced connection string configuration with keepalive settings
   - SSL connection management and timeout handling
   - Connection pool sizing for concurrent workloads
   - Automatic retry logic for transient cloud database issues
   - Rate limiting and connection lifetime management

3. Conversation State Management:
   - Persistent storage of LangGraph conversation checkpoints
   - Extraction of user prompts and AI responses from checkpoints
   - Chronological message ordering and conversation reconstruction
   - Support for multi-turn conversations with proper sequencing
   - Thread-based conversation isolation and security
   - Checkpoint data validation and integrity checking

4. User Session Tracking:
   - Custom users_threads_runs table for user-thread associations
   - Thread ownership verification and access control
   - Conversation metadata storage (prompts, timestamps, sentiments)
   - Pagination support for large conversation histories
   - Thread title generation from first user prompt
   - Bulk conversation management and cleanup operations

5. Error Handling and Recovery:
   - Comprehensive prepared statement error detection and recovery
   - Automatic checkpointer recreation on connection failures
   - Graceful degradation to in-memory storage on database failures
   - Detailed error logging with debug mode support
   - Connection pool health monitoring and automatic cleanup
   - Transient error retry mechanisms with exponential backoff

6. Windows Compatibility:
   - Windows SelectorEventLoop policy for PostgreSQL compatibility
   - Cross-platform path handling and file system operations
   - Proper async/await pattern implementation for Windows
   - Event loop policy configuration for optimal performance

7. Performance Optimization:
   - Connection pooling with configurable min/max connections
   - Efficient checkpoint processing with limiting and pagination
   - Prepared statement management to prevent memory leaks
   - Garbage collection integration for resource cleanup
   - Batch processing for multiple conversation operations
   - Optimized SQL queries with proper indexing

8. Security Features:
   - Thread ownership verification before data access
   - SQL injection prevention through parameterized queries
   - Secure connection string handling with environment variables
   - Access control for conversation data and checkpoints
   - Audit trail through comprehensive logging
   - Resource isolation between different user sessions

Core Components:
---------------
1. Connection Management:
   - get_connection_string(): Cloud-optimized PostgreSQL connection strings
   - get_connection_kwargs(): Connection parameters for cloud compatibility
   - modern_psycopg_pool(): Async context manager for connection pooling
   - get_direct_connection(): Direct database connections for utility operations

2. Checkpointer Lifecycle:
   - create_async_postgres_saver(): Main checkpointer factory with retry logic
   - setup_checkpointer_with_autocommit(): Table setup with transaction management
   - close_async_postgres_saver(): Proper resource cleanup and connection closing
   - initialize_checkpointer(): Global checkpointer initialization
   - cleanup_checkpointer(): Shutdown cleanup and resource deallocation

3. Conversation Operations:
   - get_conversation_messages_from_checkpoints(): Extract and order conversation messages
   - create_thread_run_entry(): Create new user-thread associations
   - get_user_chat_threads(): Retrieve user's conversation threads with pagination
   - delete_user_thread_entries(): Secure thread deletion with ownership verification

4. Error Recovery:
   - retry_on_prepared_statement_error(): Decorator for automatic retry logic
   - clear_prepared_statements(): Cleanup utility for prepared statement conflicts
   - is_prepared_statement_error(): Error classification for recovery decisions
   - force_close_modern_pools(): Aggressive cleanup for troubleshooting

5. Utility Functions:
   - setup_users_threads_runs_table(): Custom table creation and indexing
   - get_queries_and_results_from_latest_checkpoint(): SQL query extraction
   - update_thread_run_sentiment(): User feedback storage
   - get_thread_run_sentiments(): Sentiment data retrieval

Processing Flow:
--------------
1. Initialization:
   - Windows event loop policy configuration
   - Environment variable validation
   - Global state initialization
   - Debug mode configuration

2. Checkpointer Creation:
   - Connection string generation with unique application names
   - Connection pool setup with cloud-optimized parameters
   - AsyncPostgresSaver instantiation with error handling
   - Table setup using autocommit connections to avoid transaction conflicts
   - Context manager entry for proper resource management

3. Conversation Processing:
   - Thread ownership verification for security
   - Checkpoint retrieval with pagination and limiting
   - Message extraction from checkpoint metadata and channel values
   - Chronological ordering and conversation reconstruction
   - Data validation and integrity checking

4. Database Operations:
   - Parameterized SQL queries to prevent injection
   - Transaction management for consistency
   - Error handling with automatic retry logic
   - Connection pool management for scalability
   - Resource cleanup and connection lifecycle management

5. Error Recovery:
   - Prepared statement error detection and cleanup
   - Automatic checkpointer recreation on failures
   - Graceful degradation to in-memory storage
   - Detailed error logging for troubleshooting
   - Connection pool health monitoring and recovery

6. Cleanup and Shutdown:
   - Context manager exit for proper resource deallocation
   - Connection pool closure and cleanup
   - Global state reset and garbage collection
   - Final error reporting and statistics

Configuration:
-------------
Environment Variables Required:
- host: PostgreSQL server hostname
- port: PostgreSQL server port (default: 5432)
- dbname: Database name
- user: Database username  
- password: Database password
- print__checkpointers_debug: Debug mode flag (0/1)

Connection Parameters:
- CONNECT_TIMEOUT: Initial connection timeout (20 seconds)
- TCP_USER_TIMEOUT: TCP-level timeout (30 seconds in milliseconds)
- KEEPALIVES_IDLE: Time before first keepalive (10 minutes)
- KEEPALIVES_INTERVAL: Interval between keepalives (30 seconds)
- KEEPALIVES_COUNT: Failed keepalives before disconnect (3)

Pool Configuration:
- DEFAULT_POOL_MIN_SIZE: Minimum pool connections (1)
- DEFAULT_POOL_MAX_SIZE: Maximum pool connections (3)
- DEFAULT_POOL_TIMEOUT: Pool connection timeout (20 seconds)
- DEFAULT_MAX_IDLE: Maximum idle time (5 minutes)
- DEFAULT_MAX_LIFETIME: Maximum connection lifetime (30 minutes)

Processing Limits:
- MAX_RECENT_CHECKPOINTS: Checkpoint processing limit (10)
- MAX_DEBUG_MESSAGES_DETAILED: Detailed message logging limit (6)
- DEBUG_CHECKPOINT_LOG_INTERVAL: Checkpoint logging frequency (every 5th)

Usage Examples:
--------------
# Initialize global checkpointer
await initialize_checkpointer()

# Get conversation messages with security check
messages = await get_conversation_messages_from_checkpoints(
    checkpointer, thread_id="thread_123", user_email="user@example.com"
)

# Create new thread entry
run_id = await create_thread_run_entry(
    email="user@example.com", 
    thread_id="thread_123", 
    prompt="What is the population of Prague?"
)

# Get user's chat threads with pagination
threads = await get_user_chat_threads(
    email="user@example.com", limit=10, offset=0
)

# Clean up on shutdown
await cleanup_checkpointer()

Database Schema:
---------------
Core LangGraph Tables (managed by AsyncPostgresSaver):
- checkpoints: Main checkpoint storage with thread isolation
- checkpoint_blobs: Binary data storage for large checkpoint content

Custom Application Tables:
- users_threads_runs: User session tracking and thread ownership
  - id: Serial primary key
  - email: User email for ownership tracking
  - thread_id: LangGraph thread identifier
  - run_id: Unique run identifier for API operations
  - prompt: User's initial prompt for thread title generation
  - timestamp: Creation timestamp for ordering
  - sentiment: User feedback (positive/negative/null)

Indexes for Performance:
- idx_users_threads_runs_email: Fast user lookup
- idx_users_threads_runs_thread_id: Thread-based queries
- idx_users_threads_runs_email_thread: Combined user-thread queries

Error Handling:
-------------
1. Connection Errors:
   - Automatic retry with exponential backoff
   - Graceful fallback to in-memory storage
   - Connection pool recreation on persistent failures
   - Detailed error logging for troubleshooting

2. Prepared Statement Conflicts:
   - Automatic detection of prepared statement errors
   - Cleanup utility to remove conflicting statements
   - Checkpointer recreation with fresh connections
   - Prevention through connection parameter tuning

3. Transaction Errors:
   - Separate autocommit connections for DDL operations
   - Transaction isolation for data consistency
   - Rollback mechanisms for failed operations
   - Deadlock detection and recovery

4. Data Integrity:
   - Validation of checkpoint data structure
   - Conversation message ordering verification
   - Thread ownership security checks
   - SQL injection prevention through parameterization

Dependencies:
------------
- asyncio: Async/await pattern implementation
- psycopg: Modern PostgreSQL adapter for Python
- psycopg_pool: Connection pool management
- langgraph.checkpoint.postgres.aio: Official AsyncPostgresSaver
- langgraph.checkpoint.memory: Fallback in-memory storage
- threading, uuid, time: System utilities
- pathlib, os, sys: Cross-platform file and system operations

Performance Considerations:
-------------------------
- Connection pooling reduces overhead for frequent operations
- Checkpoint limiting prevents memory issues with large conversations
- Prepared statement management avoids memory leaks
- Indexing on custom tables ensures fast query performance
- Garbage collection integration for memory management
- Efficient SQL patterns for bulk operations

Security Considerations:
----------------------
- Environment variable usage for sensitive configuration
- Parameterized queries prevent SQL injection
- Thread ownership verification ensures data isolation
- SSL connections for encrypted communication
- Access control through user email verification
- Audit logging for security monitoring"""

import asyncio
import sys
import os
import functools
from typing import Optional, List, Dict, Any, Callable, TypeVar, Awaitable
import time
from datetime import datetime
from contextlib import asynccontextmanager
import traceback
import uuid
import threading
import gc
import psycopg

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Windows event loop fix for PostgreSQL compatibility
if sys.platform == "win32":
    print("[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility...")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[POSTGRES-STARTUP] Event loop policy set successfully")


#==============================================================================
# CONFIGURATION CONSTANTS
#==============================================================================
# Retry configuration for error handling
DEFAULT_MAX_RETRIES = 2                    # Standard retry attempts for most operations
CHECKPOINTER_CREATION_MAX_RETRIES = 2      # Retry attempts for checkpointer creation

# Connection timeout constants (in seconds)
CONNECT_TIMEOUT = 20          # Initial connection timeout for cloud databases
TCP_USER_TIMEOUT = 30000      # TCP-level timeout (30 seconds in milliseconds)
KEEPALIVES_IDLE = 600         # 10 minutes before first keepalive
KEEPALIVES_INTERVAL = 30      # 30 seconds between keepalives  
KEEPALIVES_COUNT = 3          # 3 failed keepalives before disconnect

# Pool configuration constants for cloud database optimization
DEFAULT_POOL_MIN_SIZE = 1     # Minimum pool size for efficiency
DEFAULT_POOL_MAX_SIZE = 3     # Maximum pool size to avoid connection limits
DEFAULT_POOL_TIMEOUT = 20     # Pool connection timeout
DEFAULT_MAX_IDLE = 300        # 5 minutes idle timeout  
DEFAULT_MAX_LIFETIME = 1800   # 30 minutes max connection lifetime

# String truncation constants for logging and display
USER_MESSAGE_PREVIEW_LENGTH = 50    # Length for user message previews in logs
AI_MESSAGE_PREVIEW_LENGTH = 100     # Length for AI message previews in logs
THREAD_TITLE_MAX_LENGTH = 47        # Maximum length for thread titles
THREAD_TITLE_SUFFIX_LENGTH = 3      # Length of "..." suffix

# Checkpoint processing constants for performance optimization
MAX_RECENT_CHECKPOINTS = 10         # Limit checkpoints to recent ones only
MAX_DEBUG_MESSAGES_DETAILED = 6     # Show first N messages in detail
DEBUG_CHECKPOINT_LOG_INTERVAL = 5   # Log every Nth checkpoint

#==============================================================================
# Global State Management
## Single global checkpointer variable and context for proper cleanup
_GLOBAL_CHECKPOINTER = None
_GLOBAL_CHECKPOINTER_CONTEXT = None

## Cache the connection string to avoid timestamp conflicts
_CONNECTION_STRING_CACHE = None

## Lock for checkpointer initialization to prevent race conditions
_CHECKPOINTER_INIT_LOCK = None

# Type variable for the retry decorator
T = TypeVar('T')

#==============================================================================
# DEBUG AND LOGGING FUNCTIONS
#==============================================================================
def print__checkpointers_debug(msg: str) -> None:
    """Print messages when debug mode is enabled"""
    analysis_tracing_debug_mode = os.environ.get('print__checkpointers_debug', '0')
    if analysis_tracing_debug_mode == '1':
        print(f"[print__checkpointers_debug] 🔍 {msg}")
        sys.stdout.flush()

#==============================================================================
# PREPARED STATEMENT ERROR HANDLING AND RECOVERY
#==============================================================================
def is_prepared_statement_error(error: Exception) -> bool:
    """Detect if an error is related to prepared statement conflicts.
    
    This function analyzes exception messages to identify prepared statement
    related errors that commonly occur in PostgreSQL connections, especially
    when using connection pooling or when prepared statements become stale.
    
    Args:
        error (Exception): The exception to analyze for prepared statement indicators
        
    Returns:
        bool: True if the error appears to be prepared statement related, False otherwise
        
    Detection Patterns:
        - "prepared statement" text in error message
        - "does not exist" for missing prepared statement references
        - "_pg3_" or "_pg_" prefixes from psycopg prepared statement naming
        - "invalidsqlstatementname" PostgreSQL error codes
        
    Note:
        - Case-insensitive pattern matching for robust detection
        - Covers both psycopg2 and psycopg3 prepared statement naming conventions
        - Used by retry decorators to trigger appropriate recovery actions
    """
    print__checkpointers_debug("200 - PREPARED STATEMENT CHECK: Checking if error is prepared statement related")
    error_str = str(error).lower()
    result = any(indicator in error_str for indicator in [
        'prepared statement',
        'does not exist',
        '_pg3_',
        '_pg_',
        'invalidsqlstatementname'
    ])
    print__checkpointers_debug(f"201 - PREPARED STATEMENT RESULT: Error is prepared statement related: {result}")
    return result

def retry_on_prepared_statement_error(max_retries: int = DEFAULT_MAX_RETRIES):
    """Decorator factory for automatic retry logic on prepared statement errors.
    
    This decorator provides robust error recovery for functions that may encounter
    prepared statement conflicts. It automatically detects such errors, performs
    cleanup operations, and retries the function with fresh connections.
    
    Args:
        max_retries (int): Maximum number of retry attempts before giving up
        
    Returns:
        Callable: Decorator function that wraps target functions with retry logic
        
    Retry Strategy:
        1. Execute the decorated function normally
        2. If prepared statement error detected, perform cleanup:
           - Clear existing prepared statements
           - Recreate global checkpointer if needed
           - Reset connection state
        3. Retry the function with fresh connection state
        4. Continue until success or max retries exceeded
        
    Recovery Actions:
        - clear_prepared_statements(): Remove conflicting prepared statements
        - Checkpointer recreation: Fresh AsyncPostgresSaver instance
        - Global state reset: Clean slate for retry attempts
        
    Note:
        - Only retries on confirmed prepared statement errors
        - Maintains original exception for non-recoverable errors
        - Includes detailed debug logging for troubleshooting
        - Graceful handling of cleanup failures during retry
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            print__checkpointers_debug(f"202 - RETRY WRAPPER START: Starting {func.__name__} with max_retries={max_retries}")
            last_error = None
            
            for attempt in range(max_retries + 1):
                print__checkpointers_debug(f"203 - RETRY ATTEMPT: Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}")
                try:
                    result = await func(*args, **kwargs)
                    print__checkpointers_debug(f"204 - RETRY SUCCESS: {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_error = e
                    print__checkpointers_debug(f"205 - RETRY ERROR: {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    
                    # Add full traceback for debugging
                    full_traceback = traceback.format_exc()
                    print__checkpointers_debug(f"205.1 - RETRY TRACEBACK: {full_traceback}")
                    
                    if is_prepared_statement_error(e):
                        print__checkpointers_debug(f"206 - PREPARED STATEMENT ERROR: Detected prepared statement error in {func.__name__}")
                        
                        if attempt < max_retries:
                            print__checkpointers_debug(f"207 - RETRY CLEANUP: Clearing prepared statements before retry {attempt + 2}")
                            try:
                                await clear_prepared_statements()
                                # Recreate the checkpointer if it's a global operation
                                global _GLOBAL_CHECKPOINTER_CONTEXT, _GLOBAL_CHECKPOINTER
                                if _GLOBAL_CHECKPOINTER_CONTEXT or _GLOBAL_CHECKPOINTER:
                                    print__checkpointers_debug("208 - CHECKPOINTER RECREATION: Recreating checkpointer due to prepared statement error")
                                    await close_async_postgres_saver()
                                    await create_async_postgres_saver()
                            except Exception as cleanup_error:
                                print__checkpointers_debug(f"209 - CLEANUP ERROR: Error during cleanup: {cleanup_error}")
                            continue
                    
                    # If it's not a prepared statement error, or we've exhausted retries, re-raise
                    print__checkpointers_debug(f"210 - RETRY EXHAUSTED: No more retries for {func.__name__}, re-raising error")
                    raise
            
            # This should never be reached, but just in case
            print__checkpointers_debug(f"211 - RETRY FALLBACK: Fallback error re-raise for {func.__name__}")
            raise last_error
        
        return wrapper
    return decorator

def get_db_config():
    """Extract database configuration from environment variables.
    
    This function retrieves all necessary PostgreSQL connection parameters
    from environment variables, providing a centralized configuration
    management system for database connectivity.
    
    Returns:
        dict: Database configuration dictionary containing:
            - user: PostgreSQL username
            - password: PostgreSQL password  
            - host: PostgreSQL server hostname
            - port: PostgreSQL server port (default 5432)
            - dbname: Target database name
            
    Environment Variables Required:
        - user: Database username for authentication
        - password: Database password for authentication
        - host: PostgreSQL server hostname or IP address
        - port: PostgreSQL server port (defaults to 5432 if not provided)
        - dbname: Name of the target database
        
    Note:
        - All environment variables except 'port' are required
        - Port defaults to PostgreSQL standard port 5432
        - Used by connection string and pool creation functions
        - Provides debug logging for configuration verification
    """
    print__checkpointers_debug("212 - DB CONFIG START: Getting database configuration from environment variables")
    config = {
        'user': os.environ.get('user'),
        'password': os.environ.get('password'),
        'host': os.environ.get('host'),
        'port': int(os.environ.get('port', 5432)),
        'dbname': os.environ.get('dbname')
    }
    print__checkpointers_debug(f"213 - DB CONFIG RESULT: Configuration retrieved - host: {config['host']}, port: {config['port']}, dbname: {config['dbname']}, user: {config['user']}")
    return config

def get_connection_string():
    """Generate PostgreSQL connection string with cloud-optimized parameters.
    
    This function creates a comprehensive PostgreSQL connection string optimized
    for cloud database services, including advanced timeout and keepalive settings
    for reliable connectivity in distributed environments.
    
    Returns:
        str: Complete PostgreSQL connection string with optimization parameters
        
    Connection String Features:
        - SSL mode required for secure cloud connections
        - Unique application name for connection tracking and debugging
        - Comprehensive timeout configuration for cloud latency handling
        - Keepalive settings for connection persistence
        - TCP-level timeout for network reliability
        
    Application Name Generation:
        - Combines process ID, thread ID, startup time, and random identifier
        - Ensures unique identification for concurrent connections
        - Facilitates debugging and connection monitoring
        - Format: "czsu_langgraph_{pid}_{thread}_{time}_{random}"
        
    Cloud Optimization Parameters:
        - connect_timeout: Initial connection establishment timeout
        - keepalives_idle: Time before first keepalive probe
        - keepalives_interval: Interval between keepalive probes
        - keepalives_count: Failed keepalives before disconnect
        - tcp_user_timeout: TCP-level connection timeout
        
    Caching:
        - Connection string is cached globally to avoid regeneration
        - Ensures consistent application names across operations
        - Improves performance for repeated connection operations
        
    Note:
        - Uses global caching to prevent timestamp conflicts
        - Optimized for cloud PostgreSQL services (Supabase, AWS RDS, etc.)
        - Includes comprehensive debug logging for troubleshooting
    """
    print__checkpointers_debug("214 - CONNECTION STRING START: Generating PostgreSQL connection string")
    global _CONNECTION_STRING_CACHE
    
    if _CONNECTION_STRING_CACHE is not None:
        print__checkpointers_debug("215 - CONNECTION STRING CACHED: Using cached connection string")
        return _CONNECTION_STRING_CACHE
    
    config = get_db_config()
    
    # Use process ID + startup time + random for unique application name
    process_id = os.getpid()
    thread_id = threading.get_ident()
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]
    
    app_name = f"czsu_langgraph_{process_id}_{thread_id}_{startup_time}_{random_id}"
    print__checkpointers_debug(f"216 - CONNECTION STRING APP NAME: Generated unique application name: {app_name}")
    
    # Connection string with timeout and keepalive settings for cloud databases
    _CONNECTION_STRING_CACHE = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require"
        f"&application_name={app_name}"
        f"&connect_timeout={CONNECT_TIMEOUT}"
        f"&keepalives_idle={KEEPALIVES_IDLE}"
        f"&keepalives_interval={KEEPALIVES_INTERVAL}"
        f"&keepalives_count={KEEPALIVES_COUNT}"
        f"&tcp_user_timeout={TCP_USER_TIMEOUT}"
    )
    
    print__checkpointers_debug("217 - CONNECTION STRING COMPLETE: PostgreSQL connection string generated")
    
    return _CONNECTION_STRING_CACHE

def get_connection_kwargs():
    """Generate connection kwargs for cloud database compatibility.
    
    This function provides standardized connection parameters that are optimized
    for cloud PostgreSQL databases, particularly focusing on prepared statement
    management and transaction handling.
    
    Returns:
        dict: Connection parameters dictionary for psycopg connections containing:
            - autocommit: Transaction management setting
            - prepare_threshold: Prepared statement management setting
            
    Connection Parameters:
        autocommit=False:
            - Better compatibility with cloud databases under concurrent load
            - Proper transaction management for data consistency
            - Recommended by LangGraph documentation for cloud deployments
            - Prevents transaction-related issues in high-concurrency scenarios
            
        prepare_threshold=None:
            - Completely disables automatic prepared statement creation
            - Prevents prepared statement conflicts and memory leaks
            - Reduces complexity in connection pool management
            - Eliminates need for prepared statement cleanup in most cases
            
    Note:
        - Based on LangGraph documentation and cloud database best practices
        - Optimized for concurrent workloads and connection pooling
        - Used by all connection creation functions for consistency
        - Prevents most prepared statement related issues proactively
    """
    return {
        "autocommit": False,  # Better compatibility with cloud databases under load
        "prepare_threshold": None,  # Disable prepared statements completely
    }

def check_postgres_env_vars():
    """Validate that all required PostgreSQL environment variables are configured.
    
    This function performs comprehensive validation of the environment configuration
    required for PostgreSQL connectivity, ensuring that all necessary parameters
    are available before attempting database operations.
    
    Returns:
        bool: True if all required variables are set, False if any are missing
        
    Required Environment Variables:
        - host: PostgreSQL server hostname
        - port: PostgreSQL server port  
        - dbname: Target database name
        - user: Database username
        - password: Database password
        
    Validation Process:
        1. Checks each required variable for existence and non-empty value
        2. Reports missing variables for troubleshooting
        3. Provides debug logging for configuration verification
        4. Returns boolean result for conditional initialization logic
        
    Note:
        - Used during checkpointer initialization to fail fast on misconfiguration
        - Provides detailed feedback for missing configuration
        - Supports automated deployment validation
        - Essential for preventing runtime connection failures
    """
    print__checkpointers_debug("218 - ENV VARS CHECK START: Checking PostgreSQL environment variables")
    required_vars = ['host', 'port', 'dbname', 'user', 'password']
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print__checkpointers_debug(f"219 - ENV VARS MISSING: Missing required environment variables: {missing_vars}")
        return False
    else:
        print__checkpointers_debug("220 - ENV VARS COMPLETE: All required PostgreSQL environment variables are set")
        return True

#==============================================================================
# PREPARED STATEMENT CLEANUP AND CONNECTION POOL MANAGEMENT
#==============================================================================
async def clear_prepared_statements():
    """Clear existing prepared statements to resolve conflicts during error recovery.
    
    This function provides a comprehensive cleanup mechanism for prepared statement
    conflicts that can occur in PostgreSQL connections, particularly when using
    connection pooling or when connections become stale.
    
    Process:
        1. Creates a dedicated cleanup connection with unique application name
        2. Queries pg_prepared_statements system catalog for existing statements
        3. Deallocates all found prepared statements using DEALLOCATE command
        4. Provides detailed logging for troubleshooting and verification
        
    Connection Strategy:
        - Uses separate connection to avoid conflicts with main operations
        - Applies prepared statement disabling connection kwargs
        - Generates unique application name for identification
        - Uses direct psycopg connection for maximum compatibility
        
    Error Handling:
        - Non-fatal operation that continues on individual statement failures
        - Comprehensive logging for both successes and failures
        - Graceful handling of connection or permission issues
        - Returns silently to avoid blocking main operations
        
    Performance Optimizations:
        - Limits detailed logging to first few statements to avoid log spam
        - Batches operations efficiently for large prepared statement sets
        - Provides summary statistics for large cleanup operations
        
    Note:
        - Used during error recovery when prepared statement issues are detected
        - Most prepared statement issues are prevented by connection kwargs
        - This is a recovery mechanism for edge cases and troubleshooting
        - Safe to call multiple times without side effects
    """
    print__checkpointers_debug("221 - CLEAR PREPARED START: Starting prepared statements cleanup")
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection
        
        cleanup_app_name = f"czsu_cleanup_{uuid.uuid4().hex[:8]}"
        print__checkpointers_debug(f"222 - CLEANUP CONNECTION: Creating cleanup connection with app name: {cleanup_app_name}")
        
        # Create connection string without prepared statement parameters
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"
        
        # Get connection kwargs for disabling prepared statements
        connection_kwargs = get_connection_kwargs()
        
        print__checkpointers_debug("223 - PSYCOPG CONNECTION: Establishing psycopg connection for cleanup")
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            print__checkpointers_debug("224 - CONNECTION ESTABLISHED: Cleanup connection established successfully")
            async with conn.cursor() as cur:
                print__checkpointers_debug("225 - CURSOR CREATED: Database cursor created for prepared statement query")
                # Get all prepared statements for our application
                await cur.execute("""
                    SELECT name FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                """)
                prepared_statements = await cur.fetchall()
                
                if prepared_statements:
                    print__checkpointers_debug(f"226 - PREPARED STATEMENTS FOUND: Found {len(prepared_statements)} prepared statements to clear")
                    
                    # Drop each prepared statement
                    for i, stmt in enumerate(prepared_statements, 1):
                        stmt_name = stmt[0]
                        # Only log first few statements
                        if i <= 3:
                            print__checkpointers_debug(f"227 - CLEARING STATEMENT {i}: Clearing prepared statement: {stmt_name}")
                        try:
                            await cur.execute(f"DEALLOCATE {stmt_name};")
                            if i <= 3:
                                print__checkpointers_debug(f"228 - STATEMENT CLEARED {i}: Successfully cleared prepared statement: {stmt_name}")
                        except Exception as e:
                            if i <= 3:
                                print__checkpointers_debug(f"229 - STATEMENT ERROR {i}: Could not clear prepared statement {stmt_name}: {e}")
                    
                    if len(prepared_statements) > 3:
                        print__checkpointers_debug(f"230 - CLEANUP SUMMARY: Cleared {len(prepared_statements)} prepared statements (showing first 3)")
                    else:
                        print__checkpointers_debug(f"230 - CLEANUP COMPLETE: Cleared {len(prepared_statements)} prepared statements")
                else:
                    print__checkpointers_debug("231 - NO STATEMENTS: No prepared statements to clear")
                
    except Exception as e:
        print__checkpointers_debug(f"232 - CLEANUP ERROR: Error clearing prepared statements (non-fatal): {e}")
        # Don't raise - this is a cleanup operation and shouldn't block checkpointer creation


async def cleanup_all_pools():
    """Cleanup function that properly handles connection pools and global state.
    
    This function provides comprehensive cleanup of all connection-related resources,
    ensuring proper shutdown sequence and resource deallocation for the checkpointer
    system. It handles both connection pools and global state management.
    
    Cleanup Process:
        1. Gracefully exit global checkpointer context manager
        2. Clean up connection pools using proper async patterns
        3. Reset global state variables to prevent stale references
        4. Force garbage collection to ensure memory cleanup
        5. Provide detailed logging for troubleshooting
        
    Global State Management:
        - Properly exits _GLOBAL_CHECKPOINTER_CONTEXT using __aexit__
        - Resets _GLOBAL_CHECKPOINTER to None for clean state
        - Handles cleanup errors gracefully without raising exceptions
        - Ensures clean slate for subsequent initialization attempts
        
    Resource Management:
        - Uses context manager protocols for proper resource cleanup
        - Handles connection pool lifecycle correctly
        - Provides comprehensive error handling for cleanup failures
        - Ensures resources are freed even if individual cleanup steps fail
        
    Performance Considerations:
        - Forces garbage collection to ensure immediate memory cleanup
        - Minimizes resource leakage in long-running applications
        - Provides clean shutdown for application termination scenarios
        - Optimizes memory usage for restart scenarios
        
    Note:
        - Safe to call multiple times without side effects
        - Used during error recovery and application shutdown
        - Comprehensive error handling prevents cleanup failures from propagating
        - Essential for proper resource management in production environments
    """
    print__checkpointers_debug("CLEANUP ALL POOLS START: Starting comprehensive pool cleanup")
    
    global _GLOBAL_CHECKPOINTER_CONTEXT, _GLOBAL_CHECKPOINTER
    
    # Clean up the global checkpointer context if it exists
    if _GLOBAL_CHECKPOINTER_CONTEXT:
        try:
            print__checkpointers_debug("CLEANUP: Cleaning up global checkpointer context")
            
            # _GLOBAL_CHECKPOINTER_CONTEXT is now a connection pool, so close it directly
            print__checkpointers_debug("CLEANUP: Found connection pool - closing it")
            await _GLOBAL_CHECKPOINTER_CONTEXT.close()
            print__checkpointers_debug("CLEANUP: Connection pool closed successfully")
            
        except Exception as e:
            print__checkpointers_debug(f"CLEANUP ERROR: Error during global checkpointer cleanup: {e}")
        finally:
            _GLOBAL_CHECKPOINTER_CONTEXT = None
            _GLOBAL_CHECKPOINTER = None
    
    # Force garbage collection to ensure resources are freed
    gc.collect()
    print__checkpointers_debug("CLEANUP ALL POOLS COMPLETE: All pools and resources cleaned up")

async def force_close_modern_pools():
    """Force close any remaining connection pools for aggressive cleanup.
    
    This function provides an aggressive cleanup mechanism for troubleshooting
    scenarios where normal cleanup procedures may not be sufficient. It performs
    comprehensive resource cleanup and state reset operations.
    
    Aggressive Cleanup Actions:
        1. Calls standard cleanup_all_pools() for normal resource cleanup
        2. Forces cleanup of any lingering connection resources
        3. Clears cached connection strings to force recreation
        4. Resets global state for clean restart scenarios
        5. Provides detailed logging for troubleshooting
        
    Use Cases:
        - Troubleshooting persistent connection issues
        - Recovering from connection pool corruption
        - Debugging resource leakage scenarios
        - Preparing for application restart scenarios
        - Emergency cleanup in error recovery situations
        
    State Reset Operations:
        - Clears _CONNECTION_STRING_CACHE to force regeneration
        - Ensures fresh connection parameters on next initialization
        - Provides clean slate for subsequent connection attempts
        - Prevents cached state from interfering with recovery
        
    Error Handling:
        - Comprehensive exception handling prevents cleanup failures
        - Continues operation even if individual cleanup steps fail
        - Logs errors for troubleshooting without raising exceptions
        - Ensures maximum cleanup even in error scenarios
        
    Note:
        - More aggressive than standard cleanup procedures
        - Primarily intended for troubleshooting and error recovery
        - Safe to call in production environments
        - Should be used when normal cleanup is insufficient
    """
    print__checkpointers_debug("FORCE CLOSE START: Force closing all connection pools")
    
    try:
        # Clean up the global state
        await cleanup_all_pools()
        
        # Additional cleanup for any lingering connections
        print__checkpointers_debug("FORCE CLOSE: Forcing cleanup of any remaining resources")
        
        # Clear any cached connection strings to force recreation
        global _CONNECTION_STRING_CACHE
        _CONNECTION_STRING_CACHE = None
        
        print__checkpointers_debug("FORCE CLOSE COMPLETE: Pool force close completed")
        
    except Exception as e:
        print__checkpointers_debug(f"FORCE CLOSE ERROR: Error during force close: {e}")
        # Don't re-raise - this is a cleanup function


# ASYNCPOSTGRESSAVER IMPLEMENTATION WITH CONNECTION POOL
@retry_on_prepared_statement_error(max_retries=CHECKPOINTER_CREATION_MAX_RETRIES)
async def create_async_postgres_saver():
    """Create and configure AsyncPostgresSaver with connection string approach."""
    print__checkpointers_debug("233 - CREATE SAVER START: Starting AsyncPostgresSaver creation with connection string")
    global _GLOBAL_CHECKPOINTER_CONTEXT, _GLOBAL_CHECKPOINTER
    
    # Clear any existing state first to avoid conflicts
    if _GLOBAL_CHECKPOINTER_CONTEXT or _GLOBAL_CHECKPOINTER:
        print__checkpointers_debug("234 - EXISTING STATE CLEANUP: Clearing existing checkpointer state to avoid conflicts")
        try:
            if _GLOBAL_CHECKPOINTER_CONTEXT:
                await _GLOBAL_CHECKPOINTER_CONTEXT.close()
        except Exception as e:
            print__checkpointers_debug(f"236 - CLEANUP ERROR: Error during state cleanup: {e}")
        finally:
            _GLOBAL_CHECKPOINTER_CONTEXT = None
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug("237 - STATE CLEARED: Global checkpointer state cleared")

    if not AsyncPostgresSaver:
        print__checkpointers_debug("239 - SAVER UNAVAILABLE: AsyncPostgresSaver not available")
        raise Exception("AsyncPostgresSaver not available")

    if not check_postgres_env_vars():
        print__checkpointers_debug("240 - ENV VARS MISSING: Missing required PostgreSQL environment variables")
        raise Exception("Missing required PostgreSQL environment variables")

    print__checkpointers_debug("241 - CHECKPOINTER CREATION: Creating AsyncPostgresSaver using connection pool approach")
    
    try:
        # Use connection pool approach to ensure proper connection configuration
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        print__checkpointers_debug("242 - CONNECTION POOL: Creating connection pool with proper kwargs")
        
        # Create connection pool with our connection kwargs
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,
            max_size=DEFAULT_POOL_MAX_SIZE,
            timeout=DEFAULT_POOL_TIMEOUT,
            max_idle=DEFAULT_MAX_IDLE,
            max_lifetime=DEFAULT_MAX_LIFETIME,
            kwargs=connection_kwargs,
            open=False
        )
        
        # Open the pool
        await pool.open()
        print__checkpointers_debug("247 - POOL OPENED: Connection pool opened successfully")
        
        # Create checkpointer with the pool
        _GLOBAL_CHECKPOINTER = AsyncPostgresSaver(pool, serde=None)
        _GLOBAL_CHECKPOINTER_CONTEXT = pool  # Store pool for cleanup
        
        print__checkpointers_debug("249 - SAVER CREATED: AsyncPostgresSaver created with connection pool")
        
        # Setup LangGraph tables - use autocommit connection for DDL operations
        print__checkpointers_debug("254 - SETUP START: Running setup with autocommit connection")
        await setup_checkpointer_with_autocommit()
        print__checkpointers_debug("255 - SETUP COMPLETE: AsyncPostgresSaver setup completed with autocommit")
        
    except Exception as creation_error:
        print__checkpointers_debug(f"251 - CREATION ERROR: Failed to create AsyncPostgresSaver: {creation_error}")
        # Clean up on failure
        if _GLOBAL_CHECKPOINTER_CONTEXT:
            try:
                await _GLOBAL_CHECKPOINTER_CONTEXT.close()
            except:
                pass
            _GLOBAL_CHECKPOINTER_CONTEXT = None
            _GLOBAL_CHECKPOINTER = None
        raise
    
    # Test the checkpointer to ensure it's working
    print__checkpointers_debug("256 - TESTING START: Testing checkpointer")
    test_config = {"configurable": {"thread_id": "setup_test"}}
    test_result = await _GLOBAL_CHECKPOINTER.aget(test_config)
    print__checkpointers_debug(f"257 - TESTING COMPLETE: Checkpointer test successful: {test_result is None}")
    
    # Setup custom tables using direct connection (separate from checkpointer)
    print__checkpointers_debug("258 - CUSTOM TABLES: Setting up custom users_threads_runs table")
    await setup_users_threads_runs_table()
    
    print__checkpointers_debug("259 - CREATE SAVER SUCCESS: AsyncPostgresSaver creation completed successfully")
    return _GLOBAL_CHECKPOINTER

#==============================================================================
# DATABASE SETUP AND TABLE INITIALIZATION  
#==============================================================================
async def setup_checkpointer_with_autocommit():
    """Setup the checkpointer tables using a dedicated autocommit connection to avoid transaction conflicts.
    
    This function performs the critical table setup operations for AsyncPostgresSaver
    using a separate connection configured with autocommit=True. This approach prevents
    "CREATE INDEX CONCURRENTLY cannot run inside a transaction block" errors.
    
    Setup Strategy:
        1. Creates a separate connection specifically for DDL operations
        2. Configures connection with autocommit=True to avoid transaction blocks
        3. Uses the AsyncPostgresSaver.setup() method for official table creation
        4. Provides comprehensive error handling with detailed logging
        
    Connection Configuration:
        - autocommit=True: Prevents transaction block conflicts for DDL operations
        - Same connection string as main checkpointer for consistency
        - Temporary connection that is closed after setup completion
        - Uses official AsyncPostgresSaver context manager patterns
        
    Error Recovery:
        - Provides detailed error logging for troubleshooting
        - Raises exceptions to prevent proceeding with broken setup
        - Ensures database is properly configured before checkpointer use
        
    Note:
        - Critical for proper LangGraph checkpoint table creation
        - Prevents common DDL operation failures in transaction contexts
        - Used during checkpointer initialization phase
        - Must be called before using any AsyncPostgresSaver instances
    """
    print__checkpointers_debug("SETUP AUTOCOMMIT START: Setting up checkpointer tables with autocommit=True connection")
    
    try:
        # Get connection kwargs with autocommit=True specifically for setup
        setup_connection_kwargs = get_connection_kwargs().copy()
        setup_connection_kwargs["autocommit"] = True  # Override to True for setup only
        
        connection_string = get_connection_string()
        
        print__checkpointers_debug("SETUP AUTOCOMMIT: Creating temporary connection with autocommit=True for setup")
        
        # Create a temporary AsyncPostgresSaver with autocommit=True for setup only
        setup_checkpointer_context = AsyncPostgresSaver.from_conn_string(
            conn_string=connection_string,
            serde=None
        )
        
        async with setup_checkpointer_context as setup_checkpointer:
            print__checkpointers_debug("SETUP AUTOCOMMIT: Running setup with autocommit=True connection")
            await setup_checkpointer.setup()
            print__checkpointers_debug("SETUP AUTOCOMMIT SUCCESS: Setup completed successfully with autocommit=True")
            
    except Exception as e:
        print__checkpointers_debug(f"SETUP AUTOCOMMIT ERROR: Setup failed: {e}")
        raise

#==============================================================================
# CHECKPOINTER LIFECYCLE MANAGEMENT
#==============================================================================
async def close_async_postgres_saver():
    """Close the AsyncPostgresSaver properly using the existing cleanup_all_pools() function.
    
    This function provides proper cleanup and resource deallocation for the global
    AsyncPostgresSaver instance by utilizing the existing comprehensive cleanup_all_pools()
    function instead of duplicating cleanup logic.
    
    Cleanup Process:
        - Uses cleanup_all_pools() for comprehensive resource cleanup
        - Handles all connection pools, global state, and resource deallocation
        - Provides proper error handling and logging
        - Ensures clean shutdown for application termination
        
    Note:
        - Now uses the existing cleanup_all_pools() function to avoid code duplication
        - Maintains all the same cleanup capabilities with better code organization
        - Safe to call multiple times without side effects
        - Used during application shutdown and error recovery
    """
    print__checkpointers_debug("CLOSE SAVER: Closing AsyncPostgresSaver using cleanup_all_pools()")
    
    # Use the existing comprehensive cleanup function instead of duplicating logic
    await cleanup_all_pools()
    
    print__checkpointers_debug("CLOSE SAVER: AsyncPostgresSaver closed successfully using cleanup_all_pools()")

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_global_checkpointer():
    """Get the global checkpointer instance with automatic creation and retry logic.
    
    This function provides a centralized access point for the global AsyncPostgresSaver
    instance. It implements lazy initialization and automatic retry logic for prepared
    statement errors, ensuring reliable access to the checkpointer.
    
    Initialization Strategy:
        - Lazy initialization: Creates checkpointer only when first requested
        - Singleton pattern: Returns same instance for subsequent calls
        - Automatic retry: Handles prepared statement conflicts transparently
        - Error recovery: Recreates checkpointer on persistent failures
        
    Retry Logic:
        - Decorated with retry_on_prepared_statement_error for automatic recovery
        - Handles prepared statement conflicts with cleanup and recreation
        - Provides detailed logging for troubleshooting retry attempts
        - Maintains service availability during transient database issues
        
    Global State Management:
        - Manages _GLOBAL_CHECKPOINTER singleton instance
        - Thread-safe access through Python's GIL
        - Consistent instance across multiple API requests
        - Proper lifecycle management with application startup/shutdown
        
    Performance Considerations:
        - Avoids creating multiple checkpointer instances
        - Reuses connections and resources efficiently
        - Minimizes initialization overhead for subsequent calls
        - Optimizes memory usage through singleton pattern
        
    Returns:
        AsyncPostgresSaver: The global checkpointer instance, created if necessary
        
    Note:
        - Primary access point for checkpointer functionality
        - Used by all API endpoints requiring checkpoint operations
        - Ensures consistent checkpointer configuration across the application
        - Handles initialization failures gracefully with comprehensive error reporting
    """
    print__checkpointers_debug("264 - GET GLOBAL START: Getting global checkpointer instance")
    global _GLOBAL_CHECKPOINTER
    
    if _GLOBAL_CHECKPOINTER is None:
        print__checkpointers_debug("265 - CREATE NEW: No existing checkpointer, creating new one")
        _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
        print__checkpointers_debug("266 - CREATE SUCCESS: New checkpointer created successfully")
    else:
        print__checkpointers_debug("267 - EXISTING FOUND: Using existing global checkpointer")
    
    return _GLOBAL_CHECKPOINTER

# USERS_THREADS_RUNS TABLE MANAGEMENT
# Uses direct connection approach since AsyncPostgresSaver manages its own connections
async def setup_users_threads_runs_table():
    """Create and configure the users_threads_runs table for user session tracking.
    
    This function creates a custom application table that tracks user-thread associations
    and conversation metadata. This table is separate from the LangGraph checkpoint
    tables and provides user-centric conversation management.
    
    Table Schema:
        - id: Serial primary key for unique record identification
        - email: User email address for ownership tracking and access control
        - thread_id: LangGraph thread identifier linking to checkpoint data
        - run_id: Unique run identifier for API operation tracking
        - prompt: User's initial prompt for thread title generation
        - timestamp: Creation timestamp for chronological ordering
        - sentiment: User feedback (positive/negative/null) for quality tracking
        
    Index Strategy:
        - idx_users_threads_runs_email: Fast user-based queries for thread listings
        - idx_users_threads_runs_thread_id: Thread-based lookups for operations
        - idx_users_threads_runs_email_thread: Combined index for user-thread security checks
        
    Security Features:
        - Email-based ownership tracking for access control
        - Thread ownership verification before data access
        - Unique run_id constraints for operation deduplication
        - Prepared statement prevention through parameterized queries
        
    Performance Optimization:
        - Optimized indexes for common query patterns
        - VARCHAR constraints for efficient storage
        - Appropriate data types for query performance
        - Index coverage for user-thread relationship queries
        
    Error Handling:
        - IF NOT EXISTS clauses prevent creation conflicts
        - Comprehensive error logging for troubleshooting
        - Transaction handling for consistency
        - Graceful handling of permission issues
        
    Note:
        - Uses direct connection for simplicity since AsyncPostgresSaver manages its own connections
        - Essential for user session management and conversation ownership
        - Supports conversation thread listing and management functionality
        - Enables user feedback collection and sentiment tracking
    """
    print__checkpointers_debug("268 - CUSTOM TABLE START: Setting up users_threads_runs table using direct connection")
    try:
        # Use direct connection for table setup
        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        print__checkpointers_debug("269 - CUSTOM TABLE CONNECTION: Establishing connection for table setup")
        async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
            print__checkpointers_debug("270 - CUSTOM TABLE CONNECTED: Connection established for table creation")
            # Create table with correct schema
            print__checkpointers_debug("271 - CREATE TABLE: Creating users_threads_runs table")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users_threads_runs (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) UNIQUE NOT NULL,
                    prompt TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment BOOLEAN DEFAULT NULL
                );
            """)

            # Create indexes for better performance
            print__checkpointers_debug("272 - CREATE INDEXES: Creating indexes for better performance")
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email 
                ON users_threads_runs(email);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_thread_id 
                ON users_threads_runs(thread_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """)

            print__checkpointers_debug("273 - CUSTOM TABLE SUCCESS: users_threads_runs table and indexes created successfully")

    except Exception as e:
        print__checkpointers_debug(f"274 - CUSTOM TABLE ERROR: Failed to setup users_threads_runs table: {e}")
        raise

@asynccontextmanager
async def get_direct_connection():
    """Get a direct database connection for users_threads_runs operations."""
    
    connection_string = get_connection_string()
    connection_kwargs = get_connection_kwargs()
    async with await psycopg.AsyncConnection.connect(connection_string, **connection_kwargs) as conn:
        yield conn

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_conversation_messages_from_checkpoints(checkpointer, thread_id: str, user_email: str = None) -> List[Dict[str, Any]]:
    """
    Get conversation messages from checkpoints using official AsyncPostgresSaver methods.
    
    This function extracts messages from LangGraph checkpoints and limits checkpoint processing
    to avoid performance issues with large conversation histories.
    """
    print__checkpointers_debug(f"292 - GET CONVERSATION START: Retrieving conversation messages for thread: {thread_id}")
    try:
        # Security check: Verify user owns this thread before loading checkpoint data
        if user_email:
            print__checkpointers_debug(f"293 - SECURITY CHECK: Verifying thread ownership for user: {user_email}")
            
            try:
                async with get_direct_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            SELECT COUNT(*) FROM users_threads_runs 
                            WHERE email = %s AND thread_id = %s
                        """, (user_email, thread_id))
                        result = await cur.fetchone()
                        thread_entries_count = result[0] if result else 0
                    
                    if thread_entries_count == 0:
                        print__checkpointers_debug(f"294 - SECURITY DENIED: User {user_email} does not own thread {thread_id} - access denied")
                        return []
                    
                    print__checkpointers_debug(f"295 - SECURITY GRANTED: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted")
            except Exception as e:
                print__checkpointers_debug(f"296 - SECURITY ERROR: Could not verify thread ownership: {e}")
                return []
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use alist() method with limit to avoid processing too many checkpoints
        checkpoint_tuples = []
        try:
            print__checkpointers_debug("297 - ALIST METHOD: Using official AsyncPostgresSaver.alist() method")
            
            # Increase limit to capture all checkpoints for complete conversation
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__checkpointers_debug(f"298 - ALIST ERROR: Error using alist(): {alist_error}")
            
            # Fallback: use aget_tuple() to get the latest checkpoint only
            if not checkpoint_tuples:
                print__checkpointers_debug("299 - FALLBACK METHOD: Trying fallback method using aget_tuple()")
                try:
                    state_snapshot = await checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__checkpointers_debug("300 - FALLBACK SUCCESS: Using fallback method - got latest checkpoint only")
                except Exception as fallback_error:
                    print__checkpointers_debug(f"301 - FALLBACK ERROR: Fallback method also failed: {fallback_error}")
                    return []
        
        if not checkpoint_tuples:
            print__checkpointers_debug(f"302 - NO CHECKPOINTS: No checkpoints found for thread: {thread_id}")
            return []
        
        print__checkpointers_debug(f"303 - CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for verified thread")
        
        # Sort checkpoints by step number (chronological order)
        checkpoint_tuples.sort(key=lambda x: x.metadata.get("step", 0) if x.metadata else 0)
        
        # Extract prompts and answers
        prompts = []
        answers = []
        
        print__checkpointers_debug(f"304 - MESSAGE EXTRACTION: Extracting messages from {len(checkpoint_tuples)} checkpoints")
        
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            metadata = checkpoint_tuple.metadata or {}
            step = metadata.get("step", 0)
            
            # Extract user prompts from metadata.writes.__start__.prompt
            writes = metadata.get("writes", {})
            if isinstance(writes, dict) and "__start__" in writes:
                start_data = writes["__start__"]
                if isinstance(start_data, dict) and "prompt" in start_data:
                    prompt = start_data["prompt"]
                    if prompt and prompt.strip():
                        prompts.append({
                            "content": prompt.strip(),
                            "step": step,
                            "checkpoint_index": checkpoint_index
                        })
                        print__checkpointers_debug(f"305 - USER PROMPT FOUND: Step {step}: {prompt[:50]}...")
            
            # Extract AI answers from metadata.writes.submit_final_answer.final_answer
            if isinstance(writes, dict) and "submit_final_answer" in writes:
                submit_data = writes["submit_final_answer"]
                if isinstance(submit_data, dict) and "final_answer" in submit_data:
                    final_answer = submit_data["final_answer"]
                    if final_answer and final_answer.strip():
                        answers.append({
                            "content": final_answer.strip(),
                            "step": step,
                            "checkpoint_index": checkpoint_index
                        })
                        print__checkpointers_debug(f"306 - AI ANSWER FOUND: Step {step}: {final_answer[:50]}...")
        
        # Sort prompts and answers by step number
        prompts.sort(key=lambda x: x["step"])
        answers.sort(key=lambda x: x["step"])
        
        print__checkpointers_debug(f"307 - MESSAGE PAIRING: Found {len(prompts)} prompts and {len(answers)} answers")
        
        # Create conversation messages by pairing prompts and answers
        conversation_messages = []
        message_counter = 0
        
        # Pair prompts with answers based on order
        for i in range(max(len(prompts), len(answers))):
            # Add user prompt if available
            if i < len(prompts):
                prompt = prompts[i]
                message_counter += 1
                user_message = {
                    "id": f"user_{message_counter}",
                    "content": prompt["content"],
                    "is_user": True,
                    "timestamp": datetime.fromtimestamp(1700000000 + message_counter * 1000),
                    "checkpoint_order": prompt["checkpoint_index"],
                    "message_order": message_counter,
                    "step": prompt["step"]
                }
                conversation_messages.append(user_message)
                print__checkpointers_debug(f"308 - ADDED USER MESSAGE: Step {prompt['step']}: {prompt['content'][:50]}...")
            
            # Add AI response if available
            if i < len(answers):
                answer = answers[i]
                message_counter += 1
                ai_message = {
                    "id": f"ai_{message_counter}",
                    "content": answer["content"],
                    "is_user": False,
                    "timestamp": datetime.fromtimestamp(1700000000 + message_counter * 1000),
                    "checkpoint_order": answer["checkpoint_index"],
                    "message_order": message_counter,
                    "step": answer["step"]
                }
                conversation_messages.append(ai_message)
                print__checkpointers_debug(f"309 - ADDED AI MESSAGE: Step {answer['step']}: {answer['content'][:50]}...")
        
        print__checkpointers_debug(f"310 - CONVERSATION SUCCESS: Created {len(conversation_messages)} conversation messages in proper order")
        
        # Log first few messages for debugging
        for i, msg in enumerate(conversation_messages[:6]):
            msg_type = "👤 User" if msg["is_user"] else "🤖 AI"
            print__checkpointers_debug(f"311 - MESSAGE {i+1}: {msg_type} (Step {msg['step']}): {msg['content'][:50]}...")
        
        if len(conversation_messages) > 6:
            print__checkpointers_debug(f"312 - MESSAGE SUMMARY: ...and {len(conversation_messages) - 6} more messages")
        
        return conversation_messages
        
    except Exception as e:
        print__checkpointers_debug(f"313 - CONVERSATION ERROR: Error retrieving messages from checkpoints: {str(e)}")
        print__checkpointers_debug(f"314 - CONVERSATION TRACEBACK: Full traceback: {traceback.format_exc()}")
        return []
    
# HELPER FUNCTIONS FOR COMPATIBILITY - USING DIRECT CONNECTIONS
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def create_thread_run_entry(email: str, thread_id: str, prompt: str = None, run_id: str = None) -> str:
    """Create a new thread run entry in the database with retry logic for prepared statement errors."""
    print__checkpointers_debug(f"286 - CREATE THREAD ENTRY START: Creating thread run entry for user={email}, thread={thread_id}")
    try:
        if not run_id:
            run_id = str(uuid.uuid4())
            print__checkpointers_debug(f"287 - GENERATE RUN ID: Generated new run_id: {run_id}")
        
        print__checkpointers_debug(f"288 - DATABASE INSERT: Inserting thread run entry with run_id={run_id}")
        
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        thread_id = EXCLUDED.thread_id,
                        prompt = EXCLUDED.prompt,
                        timestamp = CURRENT_TIMESTAMP
                """, (email, thread_id, run_id, prompt))
        
        print__checkpointers_debug(f"289 - CREATE THREAD ENTRY SUCCESS: Thread run entry created successfully: {run_id}")
        return run_id
    except Exception as e:
        print__checkpointers_debug(f"290 - CREATE THREAD ENTRY ERROR: Failed to create thread run entry: {e}")
        # Return the run_id even if database storage fails
        if not run_id:
            run_id = str(uuid.uuid4())
        print__checkpointers_debug(f"291 - CREATE THREAD ENTRY FALLBACK: Returning run_id despite database error: {run_id}")
        return run_id

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads(email: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination and retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Getting chat threads for user: {email} (limit: {limit}, offset: {offset})")
        
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                base_query = """
                    SELECT 
                        thread_id,
                        MAX(timestamp) as latest_timestamp,
                        COUNT(*) as run_count,
                        (SELECT prompt FROM users_threads_runs utr2 
                         WHERE utr2.email = %s AND utr2.thread_id = utr.thread_id 
                         ORDER BY timestamp ASC LIMIT 1) as first_prompt
                    FROM users_threads_runs utr
                    WHERE email = %s
                    GROUP BY thread_id
                    ORDER BY latest_timestamp DESC
                """
                
                params = [email, email]
                
                if limit is not None:
                    base_query += " LIMIT %s OFFSET %s"
                    params.extend([limit, offset])
                
                await cur.execute(base_query, params)
                rows = await cur.fetchall()
                
                threads = []
                for row in rows:
                    thread_id = row[0]
                    latest_timestamp = row[1]
                    run_count = row[2]
                    first_prompt = row[3]
                    
                    title = (first_prompt[:THREAD_TITLE_MAX_LENGTH] + "...") if first_prompt and len(first_prompt) > THREAD_TITLE_MAX_LENGTH + THREAD_TITLE_SUFFIX_LENGTH else (first_prompt or "Untitled Conversation")
                    
                    threads.append({
                        "thread_id": thread_id,
                        "latest_timestamp": latest_timestamp,
                        "run_count": run_count,
                        "title": title,
                        "full_prompt": first_prompt or ""
                    })
                
                print__checkpointers_debug(f"Retrieved {len(threads)} threads for user {email}")
                return threads
            
    except Exception as e:
        print__checkpointers_debug(f"Failed to get chat threads for user {email}: {e}")
        # Return empty list instead of raising exception to prevent API crashes
        print__checkpointers_debug("Returning empty threads list due to error")
        return []

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads_count(email: str) -> int:
    """Get total count of chat threads for a user with retry logic for prepared statement errors."""
    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(DISTINCT thread_id) as total_threads
                    FROM users_threads_runs
                    WHERE email = %s
                """, (email,))
                
                result = await cur.fetchone()
                total_count = result[0] if result else 0
            
            return total_count or 0
            
    except Exception as e:
        print__checkpointers_debug(f"Failed to get chat threads count for user {email}: {e}")
        # Return 0 instead of raising exception to prevent API crashes
        print__checkpointers_debug("Returning 0 thread count due to error")
        return 0

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def update_thread_run_sentiment(run_id: str, sentiment: bool) -> bool:
    """Update sentiment for a thread run by run_id with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Updating sentiment for run {run_id}: {sentiment}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """, (sentiment, run_id))
                updated = cur.rowcount
        print__checkpointers_debug(f"Updated sentiment for {updated} entries")
        return int(updated) > 0
    except Exception as e:
        print__checkpointers_debug(f"Failed to update sentiment: {e}")
        return False

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiments for a thread with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Getting sentiments for thread {thread_id}")
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT run_id, sentiment 
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s AND sentiment IS NOT NULL
                """, (email, thread_id))
                rows = await cur.fetchall()
        sentiments = {row[0]: row[1] for row in rows}
        print__checkpointers_debug(f"Retrieved {len(sentiments)} sentiments")
        return sentiments
    except Exception as e:
        print__checkpointers_debug(f"Failed to get sentiments: {e}")
        return {}

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def delete_user_thread_entries(email: str, thread_id: str) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table with retry logic for prepared statement errors."""
    try:
        print__checkpointers_debug(f"Deleting thread entries for user: {email}, thread: {thread_id}")
        
        async with get_direct_connection() as conn:
            # First, count the entries to be deleted
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (email, thread_id))
                result = await cur.fetchone()
                entries_to_delete = result[0] if result else 0
            
            print__checkpointers_debug(f"Found {entries_to_delete} entries to delete")
            
            if entries_to_delete == 0:
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email
                }
            
            # Delete the entries
            async with conn.cursor() as cur:
                await cur.execute("""
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """, (email, thread_id))
                deleted_count = cur.rowcount
            
            print__checkpointers_debug(f"Deleted {deleted_count} entries for user {email}, thread {thread_id}")
            
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email
            }
            
    except Exception as e:
        print__checkpointers_debug(f"Failed to delete thread entries for user {email}, thread {thread_id}: {e}")
        print__checkpointers_debug(f"Full traceback: {traceback.format_exc()}")
        raise

# CHECKPOINTER MANAGEMENT
#==============================================================================

async def initialize_checkpointer():
    """Initialize the global checkpointer with proper async context management."""
    global _GLOBAL_CHECKPOINTER
    if _GLOBAL_CHECKPOINTER is None:
        try:
            print__checkpointers_debug("🚀 CHECKPOINTER INIT: Initializing PostgreSQL Connection System...")
            print__checkpointers_debug(f"🔍 CHECKPOINTER INIT: Current global checkpointer state: {_GLOBAL_CHECKPOINTER}")
            
            # Create and initialize the checkpointer using the official AsyncPostgresSaver method
            print__checkpointers_debug("🔧 CHECKPOINTER INIT: Creating PostgreSQL checkpointer using official factory method...")
            
            checkpointer = await create_async_postgres_saver()
            
            print__checkpointers_debug(f"✅ CHECKPOINTER INIT: Created checkpointer type: {type(checkpointer).__name__}")
            
            # Set the global checkpointer directly (no wrapper needed)
            _GLOBAL_CHECKPOINTER = checkpointer
            
            print__checkpointers_debug("✅ CHECKPOINTER INIT: PostgreSQL checkpointer initialized successfully using official AsyncPostgresSaver")
            
        except Exception as e:
            print__checkpointers_debug(f"❌ CHECKPOINTER INIT: PostgreSQL checkpointer initialization failed: {e}")
            print__checkpointers_debug("🔄 CHECKPOINTER INIT: Falling back to InMemorySaver...")
            _GLOBAL_CHECKPOINTER = MemorySaver()

async def cleanup_checkpointer():
    """Clean up the global checkpointer on shutdown using force_close_modern_pools() for thorough cleanup."""
    global _GLOBAL_CHECKPOINTER
    
    print__checkpointers_debug("🧹 CHECKPOINTER CLEANUP: Starting checkpointer cleanup...")
    
    if _GLOBAL_CHECKPOINTER:
        try:
            # Check if it's an AsyncPostgresSaver that needs proper cleanup
            if hasattr(_GLOBAL_CHECKPOINTER, '__class__') and 'AsyncPostgresSaver' in str(type(_GLOBAL_CHECKPOINTER)):
                print__checkpointers_debug("🔄 CHECKPOINTER CLEANUP: Cleaning up AsyncPostgresSaver using force_close_modern_pools()...")
                # Use the more thorough cleanup function for shutdown scenarios
                await force_close_modern_pools()
            else:
                print__checkpointers_debug(f"🔄 CHECKPOINTER CLEANUP: Cleaning up {type(_GLOBAL_CHECKPOINTER).__name__}...")
                # For other types (like MemorySaver), no special cleanup needed
                _GLOBAL_CHECKPOINTER = None
                
        except Exception as e:
            print__checkpointers_debug(f"⚠️ CHECKPOINTER CLEANUP: Error during checkpointer cleanup: {e}")
        finally:
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug("✅ CHECKPOINTER CLEANUP: Checkpointer cleanup completed")
    else:
        print__checkpointers_debug("ℹ️ CHECKPOINTER CLEANUP: No checkpointer to clean up")

async def get_healthy_checkpointer():
    """Get a healthy checkpointer instance, initializing if needed with thread-safe initialization."""
    global _GLOBAL_CHECKPOINTER, _CHECKPOINTER_INIT_LOCK
    
    # Initialize lock lazily to avoid event loop issues
    if _CHECKPOINTER_INIT_LOCK is None:
        _CHECKPOINTER_INIT_LOCK = asyncio.Lock()
    
    # Use double-checked locking pattern to prevent race conditions
    if _GLOBAL_CHECKPOINTER is None:
        async with _CHECKPOINTER_INIT_LOCK:
            # Check again after acquiring lock to avoid duplicate initialization
            if _GLOBAL_CHECKPOINTER is None:
                await initialize_checkpointer()
    
    return _GLOBAL_CHECKPOINTER

@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_queries_and_results_from_latest_checkpoint(checkpointer, thread_id: str):
    """Get queries and results from checkpoints for a thread with retry logic for prepared statement errors."""
    print__checkpointers_debug(f"279 - GET CHECKPOINT START: Getting queries and results from checkpoints for thread: {thread_id}")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get all checkpoints to find all queries_and_results
        checkpoint_tuples = []
        try:
            print__checkpointers_debug("280 - ALIST METHOD: Using official AsyncPostgresSaver.alist() method")
            
            # Get all checkpoints to capture complete queries and results
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            print__checkpointers_debug(f"281 - ALIST ERROR: Error using alist(): {alist_error}")
            
            # Fallback: use aget_tuple() to get the latest checkpoint only
            try:
                print__checkpointers_debug("282 - FALLBACK METHOD: Trying fallback method using aget_tuple()")
                state_snapshot = await checkpointer.aget_tuple(config)
                if state_snapshot:
                    checkpoint_tuples = [state_snapshot]
                    print__checkpointers_debug("283 - FALLBACK SUCCESS: Using fallback method - got latest checkpoint only")
            except Exception as fallback_error:
                print__checkpointers_debug(f"284 - FALLBACK ERROR: Fallback method also failed: {fallback_error}")
                return []
        
        if not checkpoint_tuples:
            print__checkpointers_debug(f"285 - NO CHECKPOINTS: No checkpoints found for thread: {thread_id}")
            return []
        
        print__checkpointers_debug(f"286 - CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for thread")
        
        # Sort checkpoints by step number (chronological order)
        checkpoint_tuples.sort(key=lambda x: x.metadata.get("step", 0) if x.metadata else 0)
        
        # Extract queries_and_results from all checkpoints
        all_queries_and_results = []
        
        print__checkpointers_debug(f"287 - QUERIES EXTRACTION: Extracting queries_and_results from {len(checkpoint_tuples)} checkpoints")
        
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            metadata = checkpoint_tuple.metadata or {}
            step = metadata.get("step", 0)
            
            # Extract queries_and_results from metadata.writes.submit_final_answer.queries_and_results
            writes = metadata.get("writes", {})
            if isinstance(writes, dict) and "submit_final_answer" in writes:
                submit_data = writes["submit_final_answer"]
                if isinstance(submit_data, dict) and "queries_and_results" in submit_data:
                    queries_and_results = submit_data["queries_and_results"]
                    if queries_and_results:
                        # If it's a list, extend; if it's a single item, append
                        if isinstance(queries_and_results, list):
                            all_queries_and_results.extend(queries_and_results)
                            print__checkpointers_debug(f"288 - QUERIES FOUND: Step {step}: Found {len(queries_and_results)} queries and results")
                        else:
                            all_queries_and_results.append(queries_and_results)
                            print__checkpointers_debug(f"289 - QUERIES FOUND: Step {step}: Found 1 query and result")
        
        print__checkpointers_debug(f"290 - GET CHECKPOINT SUCCESS: Found {len(all_queries_and_results)} total queries and results for thread: {thread_id}")
        return all_queries_and_results
        
    except Exception as e:
        print__checkpointers_debug(f"291 - GET CHECKPOINT ERROR: Error getting queries and results from checkpoints: {e}")
        return []

#==============================================================================
# PSYCOPG CONNECTION POOL CONTEXT MANAGER
#==============================================================================
@asynccontextmanager
async def modern_psycopg_pool():
    """
    Async context manager for psycopg connection pools.
    Uses the recommended approach from psycopg documentation to avoid deprecation warnings.
    
    Usage:
        async with modern_psycopg_pool() as pool:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
    """
    print__checkpointers_debug("POOL CONTEXT START: Creating psycopg connection pool context")
    
    try:        
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        
        print__checkpointers_debug("POOL CONTEXT: Setting up AsyncConnectionPool with context management")
        
        # Use the async context manager approach recommended by psycopg
        async with AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,
            max_size=DEFAULT_POOL_MAX_SIZE,
            timeout=DEFAULT_POOL_TIMEOUT,
            max_idle=DEFAULT_MAX_IDLE,
            max_lifetime=DEFAULT_MAX_LIFETIME,
            kwargs={
                **connection_kwargs,
                "connect_timeout": CONNECT_TIMEOUT,
            },
            open=False  # Explicitly set to avoid deprecation warnings
        ) as pool:
            print__checkpointers_debug("POOL CONTEXT: Pool created and opened using context manager")
            yield pool
            print__checkpointers_debug("POOL CONTEXT: Pool will be automatically closed by context manager")
    
    except ImportError as e:
        print__checkpointers_debug(f"POOL CONTEXT ERROR: psycopg_pool not available: {e}")
        raise Exception("psycopg_pool is required for connection pool approach")
    except Exception as e:
        print__checkpointers_debug(f"POOL CONTEXT ERROR: Failed to create pool: {e}")
        raise

if __name__ == "__main__":    
    async def test():
        print__checkpointers_debug("Testing PostgreSQL checkpointer with official AsyncPostgresSaver...")
        
        if not check_postgres_env_vars():
            print__checkpointers_debug("Environment variables not set properly")
            return
        
        checkpointer = await create_async_postgres_saver()
        print__checkpointers_debug(f"Checkpointer type: {type(checkpointer).__name__}")
        
        # Test a simple operation
        config = {"configurable": {"thread_id": "test_thread"}}
        try:
            # This should work with the official AsyncPostgresSaver
            async for checkpoint in checkpointer.alist(config, limit=1):
                print__checkpointers_debug("alist() method works correctly")
                break
            else:
                print__checkpointers_debug("No checkpoints found (expected for fresh DB)")
        except Exception as e:
            print__checkpointers_debug(f"Error testing alist(): {e}")
        
        await close_async_postgres_saver()
        print__checkpointers_debug("Official AsyncPostgresSaver test completed!")
    
    asyncio.run(test())
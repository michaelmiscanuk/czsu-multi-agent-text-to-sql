"""Database Table Creation and Schema Management

This module provides comprehensive database schema setup functionality for both
LangGraph checkpoint tables and custom application tables. It handles table creation,
index setup, schema validation, and DDL operations with proper transaction management,
autocommit handling, and comprehensive error recovery.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""Database Table Creation and Schema Management

This module provides comprehensive database schema setup functionality for both
LangGraph checkpoint tables and custom application tables. It handles table creation,
index setup, schema validation, and DDL operations with proper transaction management,
autocommit handling, and comprehensive error recovery.

Key Features:
-------------
1. LangGraph Checkpoint Table Setup:
   - Automated creation of LangGraph checkpoint tables
   - Uses AsyncPostgresSaver.setup() for official table creation
   - Autocommit connection handling for DDL operations
   - Concurrent index creation support (CREATE INDEX CONCURRENTLY)
   - Prevention of "CREATE INDEX CONCURRENTLY cannot run inside transaction block" errors
   - Temporary connection pool for setup isolation
   - Idempotent setup (safe to call multiple times)
   - Comprehensive error handling and logging

2. Custom Application Table Setup:
   - users_threads_runs table creation and management
   - User session and conversation tracking schema
   - Email-based thread ownership for access control
   - Run ID tracking for API operation monitoring
   - Sentiment tracking for user feedback collection
   - Comprehensive index strategy for query performance
   - Direct connection usage for simplified setup

3. Schema Validation:
   - Table existence checking via information_schema
   - Schema validation before operations
   - Safe table creation with IF NOT EXISTS clauses
   - Error handling for permission issues
   - Boolean return values for conditional logic

4. Index Management:
   - Performance-optimized index creation
   - Email-based user query optimization
   - Thread ID lookup optimization
   - Composite indexes for security checks
   - Concurrent index creation where supported
   - IF NOT EXISTS for idempotent operations

LangGraph Checkpoint Table Setup:
-------------------------------
setup_checkpointer_with_autocommit() function:

1. Purpose and Scope:
   - Creates all LangGraph checkpoint tables and indexes
   - Uses official AsyncPostgresSaver.setup() method
   - Handles complex DDL requirements automatically
   - Ensures proper table structure for LangGraph operations
   - Essential for LangGraph state persistence

2. Autocommit Strategy:
   - Creates dedicated connection pool with autocommit=True
   - Prevents "CREATE INDEX CONCURRENTLY cannot run inside transaction block" errors
   - Follows LangGraph issue #5327 resolution
   - Essential for concurrent index creation
   - Temporary pool isolated from application pool

3. Setup Process Flow:
   a. Get connection string from get_connection_string()
   b. Get base connection kwargs from get_connection_kwargs()
   c. Override autocommit to True for DDL operations
   d. Create temporary AsyncConnectionPool for setup
   e. Open pool and create AsyncPostgresSaver instance
   f. Call AsyncPostgresSaver.setup() to create tables
   g. Close temporary pool after setup completes
   h. Log all steps for monitoring and debugging

4. Tables Created:
   - checkpoints: Main checkpoint storage for LangGraph state
   - checkpoint_writes: Checkpoint write operations and history
   - checkpoint_blobs: Binary data storage (if applicable)
   - Plus associated indexes for performance optimization

5. Error Handling:
   - Comprehensive exception catching and logging
   - Detailed error messages for troubleshooting
   - Re-raises exceptions for caller handling
   - Debug logging at each step for monitoring

6. Idempotency:
   - Safe to call multiple times without errors
   - AsyncPostgresSaver.setup() uses IF NOT EXISTS internally
   - Existing tables and indexes are not affected
   - No data loss on repeated calls

Custom Application Table Setup:
-----------------------------
setup_users_threads_runs_table() function:

1. Table Purpose and Design:
   - Tracks user-thread associations for conversation management
   - Stores conversation metadata and ownership information
   - Enables user-specific thread listing and access control
   - Supports sentiment tracking for user feedback collection
   - Links to LangGraph checkpoints via thread_id

2. Table Schema Details:
   ```sql
   CREATE TABLE users_threads_runs (
       id SERIAL PRIMARY KEY,              -- Auto-increment unique identifier
       email VARCHAR(255) NOT NULL,        -- User email for ownership
       thread_id VARCHAR(255) NOT NULL,    -- LangGraph thread identifier
       run_id VARCHAR(255) UNIQUE NOT NULL, -- Unique run identifier
       prompt TEXT,                        -- User's initial prompt
       timestamp TIMESTAMP DEFAULT NOW,    -- Creation timestamp
       sentiment BOOLEAN DEFAULT NULL      -- User feedback (pos/neg/null)
   );
   ```

3. Column Descriptions:
   - id: Serial primary key for unique record identification and joins
   - email: User email address for ownership tracking and access control
   - thread_id: LangGraph thread identifier linking to checkpoint data
   - run_id: Unique run identifier for API operation tracking and deduplication
   - prompt: User's initial prompt for thread title generation and display
   - timestamp: Creation timestamp for chronological ordering and sorting
   - sentiment: User feedback (TRUE=positive, FALSE=negative, NULL=no feedback)

4. Index Strategy:
   a. idx_users_threads_runs_email:
      - Purpose: Fast user-based queries (list all user threads)
      - Query pattern: WHERE email = 'user@example.com'
      - Use case: Dashboard thread listing, user-specific operations
      - Performance: O(log n) lookup for user threads

   b. idx_users_threads_runs_thread_id:
      - Purpose: Fast thread-based lookups and metadata retrieval
      - Query pattern: WHERE thread_id = 'thread-uuid'
      - Use case: Thread metadata retrieval, thread operations
      - Performance: O(log n) lookup for thread data

   c. idx_users_threads_runs_email_thread:
      - Purpose: Composite index for ownership verification and security
      - Query pattern: WHERE email = 'user@example.com' AND thread_id = 'thread-uuid'
      - Use case: Verify user owns thread before allowing access
      - Performance: Covers query without table lookup (covering index)
      - Security: Essential for access control enforcement

5. Security Features:
   - Email-based ownership tracking for access control
   - Thread ownership verification before data access
   - Unique run_id constraint prevents operation duplication
   - Parameterized queries prevent SQL injection
   - Composite index supports fast security checks

6. Connection Management:
   - Uses direct connection for simplicity (one-time operation)
   - AsyncPostgresSaver manages its own connections separately
   - Context manager ensures proper connection cleanup
   - Standard connection kwargs applied (autocommit=False default)

Table Existence Checking:
-----------------------
table_exists() function:

1. Purpose and Use Cases:
   - Checks if a table exists in the public schema
   - Uses information_schema for reliable detection
   - Supports conditional logic based on table presence
   - Useful for migrations and upgrades
   - Essential for testing and validation

2. Implementation Details:
   - Queries information_schema.tables system catalog
   - Filters by table_schema='public' and table_name
   - Returns boolean result (True if exists, False otherwise)
   - Uses parameterized query to prevent SQL injection
   - Efficient EXISTS query (stops at first match)

3. Usage Pattern:
   ```python
   async with get_direct_connection() as conn:
       if await table_exists(conn, 'users_threads_runs'):
           # Table exists, proceed with operations
       else:
           # Create table first
           await setup_users_threads_runs_table()
   ```

Connection Management:
--------------------
1. LangGraph Setup:
   - Uses temporary AsyncConnectionPool
   - Pool created specifically for setup operation
   - autocommit=True for DDL compatibility
   - Pool closed after setup completes
   - Isolated from application connection pool

2. Custom Table Setup:
   - Uses direct connection via get_connection_string()
   - Simpler than pool for one-time setup operation
   - Context manager ensures cleanup
   - Standard connection kwargs applied

3. Transaction Handling:
   - LangGraph setup: autocommit=True (required for DDL)
   - Custom table setup: autocommit=False (default)
   - Proper transaction isolation for data consistency
   - Error rollback handled by psycopg

Usage Examples:
--------------
1. Initial Application Setup:
   ```python
   from checkpointer.database.table_setup import (
       setup_checkpointer_with_autocommit,
       setup_users_threads_runs_table
   )
   
   # Setup all tables during app initialization
   await setup_checkpointer_with_autocommit()  # LangGraph tables
   await setup_users_threads_runs_table()      # Custom tables
   ```

2. Check Before Setup:
   ```python
   from checkpointer.database.table_setup import table_exists
   from checkpointer.database.connection import get_direct_connection
   
   async with get_direct_connection() as conn:
       if not await table_exists(conn, 'checkpoints'):
           await setup_checkpointer_with_autocommit()
   ```

3. Testing Setup:
   ```python
   # In test fixtures
   async def setup_test_db():
       await setup_checkpointer_with_autocommit()
       await setup_users_threads_runs_table()
   ```

Error Handling:
--------------
1. Table Creation Errors:
   - Permission denied: Database user lacks CREATE TABLE privilege
   - Schema not found: Public schema doesn't exist
   - Connection errors: Database unreachable
   - Transaction block errors: Autocommit not set correctly

2. Index Creation Errors:
   - Duplicate index: Index already exists (ignored with IF NOT EXISTS)
   - Permission denied: Database user lacks CREATE INDEX privilege
   - Concurrent creation: Handled by autocommit mode

3. Recovery Strategies:
   - Detailed error logging for troubleshooting
   - Re-raise exceptions for caller handling
   - Safe to retry on transient failures
   - Idempotent operations (IF NOT EXISTS clauses)

Debug Logging:
-------------
LangGraph Setup:
- "SETUP CHECKPOINTER START": Begin setup
- "SETUP CHECKPOINTER: Creating temporary pool": Pool creation
- "SETUP CHECKPOINTER: Temporary pool opened": Pool ready
- "SETUP CHECKPOINTER: Running AsyncPostgresSaver.setup()": Table creation
- "SETUP CHECKPOINTER SUCCESS": Setup complete
- "SETUP CHECKPOINTER: Temporary pool closed": Cleanup done
- "SETUP CHECKPOINTER ERROR": Setup failure details

Custom Table Setup:
- "268 - CUSTOM TABLE START": Begin table setup
- "269 - CUSTOM TABLE CONNECTION": Connection establishment
- "270 - CUSTOM TABLE CONNECTED": Connection ready
- "271 - CREATE TABLE": Table creation
- "272 - CREATE INDEXES": Index creation
- "273 - CUSTOM TABLE SUCCESS": Setup complete
- "274 - CUSTOM TABLE ERROR": Setup failure details

Security Considerations:
-----------------------
1. SQL Injection Prevention:
   - No string concatenation for SQL
   - Parameterized queries in table_exists()
   - DDL statements use constants only
   - psycopg handles escaping automatically

2. Access Control:
   - Email-based thread ownership in schema
   - Composite indexes support ownership checks
   - Application enforces access control logic

3. Data Integrity:
   - NOT NULL constraints on critical fields
   - UNIQUE constraint on run_id
   - PRIMARY KEY for unique record identification
   - Foreign key relationships (via application logic)

Performance Considerations:
-------------------------
1. Index Strategy:
   - Single-column indexes for simple queries
   - Composite index for common join pattern
   - Covering indexes reduce table lookups
   - Appropriate index types (B-tree default)

2. Table Design:
   - VARCHAR sizing appropriate for data (255 chars)
   - SERIAL for efficient primary key
   - TEXT for variable-length content
   - TIMESTAMP for time-based queries

3. Setup Performance:
   - Temporary pool for isolated setup
   - Minimal overhead on application pool
   - Fast IF NOT EXISTS checks
   - Idempotent operations avoid duplication

Configuration Integration:
------------------------
- DEFAULT_POOL_MIN_SIZE: Setup pool minimum size
- DEFAULT_POOL_MAX_SIZE: Setup pool maximum size
- DEFAULT_POOL_TIMEOUT: Setup pool timeout
- CONNECT_TIMEOUT: Connection establishment timeout

Troubleshooting:
---------------
1. "CREATE INDEX CONCURRENTLY" Error:
   - Ensure autocommit=True in setup_checkpointer_with_autocommit()
   - Verify AsyncConnectionPool kwargs include autocommit
   - Check LangGraph version compatibility

2. Permission Errors:
   - Verify database user has CREATE TABLE privilege
   - Check CREATE INDEX permissions
   - Ensure access to public schema

3. Table Already Exists:
   - Normal if setup called multiple times
   - IF NOT EXISTS prevents errors
   - Check logs to confirm idempotent behavior

4. Connection Timeouts:
   - Check CONNECT_TIMEOUT configuration
   - Verify database accessibility
   - Review network connectivity

Dependencies:
------------
- psycopg: PostgreSQL adapter (async support)
- psycopg_pool: Connection pool implementation
- langgraph.checkpoint.postgres.aio: AsyncPostgresSaver
- checkpointer.database.connection: Connection utilities
- checkpointer.config: Configuration management
- api.utils.debug: Debug logging utilities

Future Enhancements:
------------------
- Schema migration utilities
- Table versioning support
- Automatic index optimization
- Schema validation and health checks
- Performance profiling for indexes
- Custom table creation hooks
- Schema upgrade procedures
- Backup table creation
"""

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import (
    get_connection_string,
    get_connection_kwargs,
)
from checkpointer.config import (
    DEFAULT_POOL_MIN_SIZE,
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_TIMEOUT,
)


# ==============================================================================
# MODULE FUNCTIONS
# ==============================================================================
# This module provides three core functions for table management:
# 1. setup_checkpointer_with_autocommit() - Create LangGraph checkpoint tables
# 2. setup_users_threads_runs_table() - Create custom application table
# 3. table_exists() - Check if a table exists in the database schema
# ==============================================================================


async def setup_checkpointer_with_autocommit():
    """Setup the checkpointer tables using a dedicated autocommit connection.

    This function performs the critical table setup operations for AsyncPostgresSaver
    using a separate connection configured with autocommit=True. This approach prevents
    "CREATE INDEX CONCURRENTLY cannot run inside a transaction block" errors that
    occur when using standard transaction-based connections for DDL operations.

    Setup Strategy:
        1. Creates a separate connection pool specifically for DDL operations
        2. Configures connection with autocommit=True to avoid transaction blocks
        3. Uses the AsyncPostgresSaver.setup() method for official table creation
        4. Closes temporary pool after setup completes
        5. Provides comprehensive error handling with detailed logging

    Tables Created:
        - checkpoints: Main checkpoint storage for LangGraph state
        - checkpoint_writes: Checkpoint write operations and history
        - checkpoint_blobs: Binary data storage (if applicable to version)
        - Associated indexes for query performance optimization

    Autocommit Requirement:
        - CRITICAL: Must set autocommit=True to allow CREATE INDEX CONCURRENTLY
        - Standard transaction blocks prevent concurrent index creation
        - Autocommit allows DDL statements to execute outside transactions
        - See LangGraph issue #5327: https://github.com/langchain-ai/langgraph/issues/5327
        - Required by AsyncPostgresSaver.setup() implementation

    Temporary Pool Configuration:
        - Uses same min/max size as application pool for consistency
        - Configured with DEFAULT_POOL_TIMEOUT for operations
        - Includes autocommit=True override for DDL compatibility
        - Isolated from main application connection pool
        - Closed after setup to free resources

    Process Flow:
        1. Get connection string from get_connection_string()
        2. Get base connection kwargs from get_connection_kwargs()
        3. Override autocommit to True for DDL operations
        4. Create temporary AsyncConnectionPool for setup
        5. Open pool for connection availability
        6. Create AsyncPostgresSaver instance with pool
        7. Call AsyncPostgresSaver.setup() to create tables
        8. Close temporary pool after completion

    Error Handling:
        - Comprehensive exception catching and logging
        - Detailed error messages for troubleshooting
        - Re-raises exceptions for caller handling
        - Debug logging at each step for monitoring
        - Graceful handling of permission and connection errors

    Idempotency:
        - Safe to call multiple times without errors
        - AsyncPostgresSaver.setup() uses IF NOT EXISTS internally
        - Existing tables and indexes are not affected
        - No data loss on repeated calls

    Note:
        - Uses autocommit connection to avoid transaction block conflicts
        - Creates all LangGraph checkpoint tables and indexes
        - Safe to call multiple times (idempotent operation)
        - Used during application initialization
        - Essential for LangGraph state persistence functionality
        - Temporary pool ensures isolation from application pool
    """
    print__checkpointers_debug(
        "SETUP CHECKPOINTER START: Creating LangGraph checkpoint tables with autocommit"
    )

    try:
        # Get connection configuration for pool creation
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Creating temporary connection pool for table setup"
        )

        # Create a temporary connection pool for setup using config values
        # CRITICAL: Must set autocommit=True to allow CREATE INDEX CONCURRENTLY
        # See: https://github.com/langchain-ai/langgraph/issues/5327
        setup_kwargs = {**connection_kwargs, "autocommit": True}

        # Create temporary AsyncConnectionPool with autocommit enabled
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,  # Same as application pool
            max_size=DEFAULT_POOL_MAX_SIZE,  # Same as application pool
            timeout=DEFAULT_POOL_TIMEOUT,  # Standard pool timeout
            kwargs=setup_kwargs,  # Connection kwargs with autocommit=True
            open=False,  # Will open manually
        )

        # Open the pool to make connections available
        await pool.open()
        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Temporary connection pool opened successfully"
        )

        # Create a temporary checkpointer instance just for setup
        # serde=None is standard for setup operations
        temp_checkpointer = AsyncPostgresSaver(pool, serde=None)

        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Running AsyncPostgresSaver.setup() to create tables"
        )

        # Use the official setup method - it manages its own connection
        # Creates all necessary tables, indexes, and schema objects
        await temp_checkpointer.setup()

        print__checkpointers_debug(
            "SETUP CHECKPOINTER SUCCESS: LangGraph checkpoint tables created successfully"
        )

        # Clean up the temporary pool to free resources
        await pool.close()
        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Temporary connection pool closed"
        )

    except Exception as setup_error:
        # Log detailed error information for troubleshooting
        print__checkpointers_debug(
            f"SETUP CHECKPOINTER ERROR: Failed to setup checkpoint tables: {setup_error}"
        )
        # Re-raise for caller to handle
        raise


async def setup_users_threads_runs_table():
    """Create and configure the users_threads_runs table for user session tracking.

    This function creates a custom application table that tracks user-thread associations
    and conversation metadata. This table is separate from the LangGraph checkpoint
    tables and provides user-centric conversation management, thread ownership tracking,
    and sentiment feedback collection.

    Table Purpose:
        - Tracks user-thread associations for conversation management
        - Stores conversation metadata and ownership information
        - Enables user-specific thread listing and access control
        - Supports sentiment tracking for user feedback collection
        - Links to LangGraph checkpoints via thread_id foreign key relationship

    Table Schema:
        ```sql
        CREATE TABLE users_threads_runs (
            id SERIAL PRIMARY KEY,              -- Auto-increment unique identifier
            email VARCHAR(255) NOT NULL,        -- User email for ownership
            thread_id VARCHAR(255) NOT NULL,    -- LangGraph thread identifier
            run_id VARCHAR(255) UNIQUE NOT NULL, -- Unique run identifier (API operation)
            prompt TEXT,                        -- User's initial prompt text
            timestamp TIMESTAMP DEFAULT NOW,    -- Creation timestamp
            sentiment BOOLEAN DEFAULT NULL      -- User feedback (pos/neg/null)
        );
        ```

    Column Descriptions:
        - id: Serial primary key for unique record identification and joins
        - email: User email address for ownership tracking and access control
        - thread_id: LangGraph thread identifier linking to checkpoint data
        - run_id: Unique run identifier for API operation tracking and deduplication
        - prompt: User's initial prompt for thread title generation and display
        - timestamp: Creation timestamp for chronological ordering and sorting
        - sentiment: User feedback (TRUE=positive, FALSE=negative, NULL=no feedback yet)

    Index Strategy:
        1. idx_users_threads_runs_email:
           - Purpose: Fast user-based queries (list all user threads)
           - Query pattern: WHERE email = 'user@example.com'
           - Use case: Dashboard thread listing, user-specific operations
           - Performance: O(log n) lookup for user threads

        2. idx_users_threads_runs_thread_id:
           - Purpose: Fast thread-based lookups and metadata retrieval
           - Query pattern: WHERE thread_id = 'thread-uuid'
           - Use case: Thread metadata retrieval, thread operations
           - Performance: O(log n) lookup for thread data

        3. idx_users_threads_runs_email_thread:
           - Purpose: Composite index for ownership verification and security checks
           - Query pattern: WHERE email = 'user@example.com' AND thread_id = 'thread-uuid'
           - Use case: Verify user owns thread before allowing access
           - Performance: Covers query without table lookup (covering index)
           - Security: Essential for access control enforcement

    Security Features:
        - Email-based ownership tracking for access control
        - Thread ownership verification before data access
        - Unique run_id constraint prevents operation duplication
        - Parameterized queries prevent SQL injection
        - Composite index supports fast security checks

    Performance Optimization:
        - Optimized indexes for common query patterns
        - VARCHAR constraints for efficient storage (255 chars for identifiers)
        - Appropriate data types for query performance
        - Index coverage for user-thread relationship queries
        - SERIAL primary key for fast joins

    Data Integrity:
        - NOT NULL constraints on critical fields (email, thread_id, run_id)
        - UNIQUE constraint on run_id prevents duplicates
        - PRIMARY KEY ensures record uniqueness
        - DEFAULT values for timestamp and sentiment
        - Foreign key relationships enforced at application level

    Connection Management:
        - Uses direct connection for simplicity (one-time operation)
        - AsyncPostgresSaver manages its own connections separately
        - Context manager ensures proper connection cleanup
        - Standard connection kwargs applied (autocommit=False default)

    Error Handling:
        - IF NOT EXISTS clauses prevent creation conflicts
        - Comprehensive error logging for troubleshooting
        - Transaction handling for consistency (default autocommit=False)
        - Graceful handling of permission issues
        - Detailed debug output at each step

    Idempotency:
        - Safe to call multiple times without errors
        - IF NOT EXISTS prevents duplicate creation
        - Existing table and indexes are not affected
        - No data loss on repeated calls

    Usage Example:
        ```python
        # During application initialization
        await setup_users_threads_runs_table()

        # Insert new user-thread association
        async with get_direct_connection() as conn:
            await conn.execute(
                \"\"\"
                INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                VALUES (%s, %s, %s, %s)
                \"\"\",
                (user_email, thread_id, run_id, user_prompt)
            )
        ```

    Note:
        - Uses direct connection for simplicity since AsyncPostgresSaver manages its own connections
        - Essential for user session management and conversation ownership
        - Supports conversation thread listing and management functionality
        - Enables user feedback collection and sentiment tracking
        - Separate from LangGraph checkpoint tables for clean separation of concerns
    """
    print__checkpointers_debug(
        "268 - CUSTOM TABLE START: Setting up users_threads_runs table using direct connection"
    )
    try:
        # Get connection string and standard connection kwargs
        # Use direct connection for table setup
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        print__checkpointers_debug(
            "269 - CUSTOM TABLE CONNECTION: Establishing connection for table setup"
        )

        # Establish direct async connection using context manager for proper cleanup
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            print__checkpointers_debug(
                "270 - CUSTOM TABLE CONNECTED: Connection established for table creation"
            )

            # Create table with complete schema including all columns and constraints
            print__checkpointers_debug(
                "271 - CREATE TABLE: Creating users_threads_runs table"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users_threads_runs (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    thread_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) UNIQUE NOT NULL,
                    prompt TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment BOOLEAN DEFAULT NULL
                );
            """
            )

            # Create performance-optimized indexes for common query patterns
            print__checkpointers_debug(
                "272 - CREATE INDEXES: Creating indexes for better performance"
            )

            # Index for user-based queries (list all threads for a user)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email 
                ON users_threads_runs(email);
            """
            )

            # Index for thread-based lookups (get metadata for a thread)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_thread_id 
                ON users_threads_runs(thread_id);
            """
            )

            # Composite index for ownership verification (security checks)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """
            )

            print__checkpointers_debug(
                "273 - CUSTOM TABLE SUCCESS: users_threads_runs table and indexes created successfully"
            )

    except Exception as exc:
        # Log detailed error information for troubleshooting
        print__checkpointers_debug(
            f"274 - CUSTOM TABLE ERROR: Failed to setup users_threads_runs table: {exc}"
        )
        # Re-raise for caller to handle
        raise


async def table_exists(conn, table_name):
    """Check if a table exists in the public schema.

    This function queries the PostgreSQL information_schema to determine if a
    specific table exists in the public schema. Useful for conditional logic,
    schema validation, and ensuring tables are created before operations.

    Args:
        conn: psycopg async connection object to use for the query
        table_name (str): Name of the table to check for existence

    Returns:
        bool: True if the table exists in the public schema,
              False if the table does not exist

    Query Details:
        - Uses information_schema.tables system catalog
        - Filters by table_schema='public' (standard PostgreSQL schema)
        - Filters by table_name parameter (exact match)
        - Returns EXISTS boolean result (efficient query)

    SQL Injection Prevention:
        - Uses parameterized query (%s placeholder)
        - table_name passed as query parameter, not concatenated
        - psycopg handles proper escaping automatically
        - Safe for user input (if validated)

    Usage Example:
        ```python
        async with get_direct_connection() as conn:
            if await table_exists(conn, 'users_threads_runs'):
                print("Table exists, ready for operations")
            else:
                print("Table does not exist, creating...")
                await setup_users_threads_runs_table()
        ```

    Common Use Cases:
        - Conditional table creation in initialization
        - Schema validation before operations
        - Migration script checks
        - Test setup verification
        - Health check endpoints

    Note:
        - Only checks public schema (standard for application tables)
        - Case-sensitive table name comparison
        - Efficient query using EXISTS (stops at first match)
        - Does not check table structure, only existence
        - Safe for concurrent use (read-only operation)
    """
    # Query information_schema for table existence in public schema
    async with conn.cursor() as cur:
        # Use parameterized query to prevent SQL injection
        await cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            );
            """,
            (table_name,),  # Parameterized table name for safety
        )
        # Fetch the boolean result from the EXISTS query
        result = await cur.fetchone()
        # Return boolean indicating table existence
        return result[0]

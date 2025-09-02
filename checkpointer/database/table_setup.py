"""Database table creation and schema management.

This module handles database table creation, schema setup, and table
utilities for both LangGraph and custom application tables.
"""
from __future__ import annotations

import psycopg

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import get_connection_string, get_connection_kwargs
from checkpointer.database.pool_manager import cleanup_all_pools


# This file will contain:
# - setup_checkpointer_with_autocommit() function
# - setup_users_threads_runs_table() function
# - table_exists() function
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


    try:
        # Use connection pool approach to ensure proper connection configuration
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "242 - CONNECTION POOL: Creating connection pool with proper kwargs"
        )

        # Create connection pool with our connection kwargs
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,
            max_size=DEFAULT_POOL_MAX_SIZE,
            timeout=DEFAULT_POOL_TIMEOUT,
            max_idle=DEFAULT_MAX_IDLE,
            max_lifetime=DEFAULT_MAX_LIFETIME,
            kwargs=connection_kwargs,
            open=False,
        )

        # Open the pool
        await pool.open()
        print__checkpointers_debug(
            "247 - POOL OPENED: Connection pool opened successfully"
        )

        # Create checkpointer with the pool
        _GLOBAL_CHECKPOINTER = AsyncPostgresSaver(pool, serde=None)

        print__checkpointers_debug(
            "249 - SAVER CREATED: AsyncPostgresSaver created with connection pool"
        )

        # Setup LangGraph tables - use autocommit connection for DDL operations
        print__checkpointers_debug(
            "254 - SETUP START: Checking if 'public.checkpoints' table exists before running setup"
        )
        # Use a direct connection from the pool to check for table existence
        async with await psycopg.AsyncConnection.connect(
            connection_string, autocommit=True
        ) as conn:
            exists = await table_exists(conn, "checkpoints")
        if exists:
            print__checkpointers_debug(
                "SKIP SETUP: Table 'public.checkpoints' already exists, skipping setup_checkpointer_with_autocommit()"
            )
        else:
            await setup_checkpointer_with_autocommit()
            print__checkpointers_debug(
                "255 - SETUP COMPLETE: AsyncPostgresSaver setup completed with autocommit"
            )

    except Exception as creation_error:
        print__checkpointers_debug(
            f"251 - CREATION ERROR: Failed to create AsyncPostgresSaver: {creation_error}"
        )
        # Clean up on failure
        if _GLOBAL_CHECKPOINTER:
            try:
                if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
                    await _GLOBAL_CHECKPOINTER.pool.close()
            except Exception:
                pass
            _GLOBAL_CHECKPOINTER = None
        raise

    Note:
        - Now uses the existing cleanup_all_pools() function to avoid code duplication
        - Maintains all the same cleanup capabilities with better code organization
        - Safe to call multiple times without side effects
        - Used during application shutdown and error recovery
    """
    print__checkpointers_debug(
        "CLOSE SAVER: Closing AsyncPostgresSaver using cleanup_all_pools()"
    )

    # Use the existing comprehensive cleanup function instead of duplicating logic
    await cleanup_all_pools()

    print__checkpointers_debug(
        "CLOSE SAVER: AsyncPostgresSaver closed successfully using cleanup_all_pools()"
    )


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
    print__checkpointers_debug(
        "268 - CUSTOM TABLE START: Setting up users_threads_runs table using direct connection"
    )
    try:
        # Use direct connection for table setup

        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        print__checkpointers_debug(
            "269 - CUSTOM TABLE CONNECTION: Establishing connection for table setup"
        )
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            print__checkpointers_debug(
                "270 - CUSTOM TABLE CONNECTED: Connection established for table creation"
            )
            # Create table with correct schema
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

            # Create indexes for better performance
            print__checkpointers_debug(
                "272 - CREATE INDEXES: Creating indexes for better performance"
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email 
                ON users_threads_runs(email);
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_thread_id 
                ON users_threads_runs(thread_id);
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_threads_runs_email_thread 
                ON users_threads_runs(email, thread_id);
            """
            )

            print__checkpointers_debug(
                "273 - CUSTOM TABLE SUCCESS: users_threads_runs table and indexes created successfully"
            )

    except Exception as e:
        print__checkpointers_debug(
            f"274 - CUSTOM TABLE ERROR: Failed to setup users_threads_runs table: {e}"
        )
        raise


async def table_exists(conn, table_name):
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            );
            """,
            (table_name,),
        )
        result = await cur.fetchone()
        return result[0]

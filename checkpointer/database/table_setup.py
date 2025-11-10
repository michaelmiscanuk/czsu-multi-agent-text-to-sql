"""Database table creation and schema management.

This module handles database table creation, schema setup, and table
utilities for both LangGraph and custom application tables.
"""

from __future__ import annotations

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

    Note:
        - Uses autocommit connection to avoid transaction block conflicts
        - Creates all LangGraph checkpoint tables and indexes
        - Safe to call multiple times (idempotent)
        - Used during application initialization
    """
    print__checkpointers_debug(
        "SETUP CHECKPOINTER START: Creating LangGraph checkpoint tables with autocommit"
    )

    try:
        # Get connection configuration
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Creating temporary connection pool for table setup"
        )

        # Create a temporary connection pool for setup using config values
        # CRITICAL: Must set autocommit=True to allow CREATE INDEX CONCURRENTLY
        # See: https://github.com/langchain-ai/langgraph/issues/5327
        setup_kwargs = {**connection_kwargs, "autocommit": True}

        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,
            max_size=DEFAULT_POOL_MAX_SIZE,
            timeout=DEFAULT_POOL_TIMEOUT,
            kwargs=setup_kwargs,
            open=False,
        )

        # Open the pool
        await pool.open()
        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Temporary connection pool opened successfully"
        )

        # Create a temporary checkpointer instance just for setup
        temp_checkpointer = AsyncPostgresSaver(pool, serde=None)

        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Running AsyncPostgresSaver.setup() to create tables"
        )

        # Use the official setup method - it manages its own connection
        await temp_checkpointer.setup()

        print__checkpointers_debug(
            "SETUP CHECKPOINTER SUCCESS: LangGraph checkpoint tables created successfully"
        )

        # Clean up the temporary pool
        await pool.close()
        print__checkpointers_debug(
            "SETUP CHECKPOINTER: Temporary connection pool closed"
        )

    except Exception as setup_error:
        print__checkpointers_debug(
            f"SETUP CHECKPOINTER ERROR: Failed to setup checkpoint tables: {setup_error}"
        )
        raise


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

    except Exception as exc:
        print__checkpointers_debug(
            f"274 - CUSTOM TABLE ERROR: Failed to setup users_threads_runs table: {exc}"
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

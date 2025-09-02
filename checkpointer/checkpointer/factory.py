"""Checkpointer creation and lifecycle management.

This module handles checkpointer creation, initialization, and lifecycle
management for the PostgreSQL checkpointer system.
"""
from __future__ import annotations

import asyncio

import psycopg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from api.utils.debug import print__checkpointers_debug
from checkpointer.config import DEFAULT_MAX_RETRIES, CHECKPOINTER_CREATION_MAX_RETRIES, DEFAULT_POOL_MIN_SIZE, \
    DEFAULT_POOL_MAX_SIZE, DEFAULT_POOL_TIMEOUT, DEFAULT_MAX_IDLE, DEFAULT_MAX_LIFETIME, check_postgres_env_vars
from checkpointer.error_handling.retry_decorators import retry_on_prepared_statement_error
from checkpointer.database.table_setup import setup_checkpointer_with_autocommit, setup_users_threads_runs_table, \
    table_exists
from checkpointer.database.pool_manager import force_close_modern_pools
from checkpointer.database.connection import get_connection_string, get_connection_kwargs
from checkpointer.checkpointer.health import check_pool_health_and_recreate


# This file will contain:
# - create_async_postgres_saver() function
# - close_async_postgres_saver() function ## NEED TO ADD
# - get_global_checkpointer() function
# - initialize_checkpointer() function
# - cleanup_checkpointer() function

@retry_on_prepared_statement_error(max_retries=CHECKPOINTER_CREATION_MAX_RETRIES)
async def create_async_postgres_saver():
    """Create and configure AsyncPostgresSaver with connection string approach."""
    print("AAA, create_async_postgres_saver() called")
    print__checkpointers_debug(
        "233 - CREATE SAVER START: Starting AsyncPostgresSaver creation with connection string"
    )

    global _GLOBAL_CHECKPOINTER

    # Clear any existing state first to avoid conflicts
    if _GLOBAL_CHECKPOINTER:
        print__checkpointers_debug(
            "234 - EXISTING STATE CLEANUP: Clearing existing checkpointer state to avoid conflicts"
        )
        try:
            if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
                await _GLOBAL_CHECKPOINTER.pool.close()
        except Exception as e:
            print__checkpointers_debug(
                f"236 - CLEANUP ERROR: Error during state cleanup: {e}"
            )
        finally:
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug(
                "237 - STATE CLEARED: Global checkpointer state cleared"
            )

    if not AsyncPostgresSaver:
        print__checkpointers_debug(
            "239 - SAVER UNAVAILABLE: AsyncPostgresSaver not available"
        )
        raise Exception("AsyncPostgresSaver not available")

    if not check_postgres_env_vars():
        print__checkpointers_debug(
            "240 - ENV VARS MISSING: Missing required PostgreSQL environment variables"
        )
        raise Exception("Missing required PostgreSQL environment variables")

    print__checkpointers_debug(
        "241 - CHECKPOINTER CREATION: Creating AsyncPostgresSaver using connection pool approach"
    )

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

    # Test the checkpointer to ensure it's working
    print__checkpointers_debug("256 - TESTING START: Testing checkpointer")
    test_config = {"configurable": {"thread_id": "setup_test"}}
    test_result = await _GLOBAL_CHECKPOINTER.aget(test_config)
    print__checkpointers_debug(
        f"257 - TESTING COMPLETE: Checkpointer test successful: {test_result is None}"
    )

    # Setup custom tables using direct connection (separate from checkpointer)
    print__checkpointers_debug(
        "258 - CUSTOM TABLES: Setting up custom users_threads_runs table"
    )
    await setup_users_threads_runs_table()

    print__checkpointers_debug(
        "259 - CREATE SAVER SUCCESS: AsyncPostgresSaver creation completed successfully"
    )
    return _GLOBAL_CHECKPOINTER


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_global_checkpointer():
    """
    Unified access point for the global checkpointer instance.
    Handles lazy initialization, health check, and retry logic.
    """
    print__checkpointers_debug(
        "264 - GET GLOBAL START: Getting global checkpointer instance"
    )
    global _GLOBAL_CHECKPOINTER, _CHECKPOINTER_INIT_LOCK

    # Initialize lock lazily to avoid event loop issues
    if _CHECKPOINTER_INIT_LOCK is None:
        _CHECKPOINTER_INIT_LOCK = asyncio.Lock()

    # Use double-checked locking pattern to prevent race conditions
    if _GLOBAL_CHECKPOINTER is None:
        async with _CHECKPOINTER_INIT_LOCK:
            if _GLOBAL_CHECKPOINTER is None:
                _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
                print__checkpointers_debug(
                    "266 - CREATE SUCCESS: New checkpointer created successfully"
                )
    else:
        print__checkpointers_debug(
            "267 - EXISTING FOUND: Using existing global checkpointer"
        )

    # Add frequent health check before returning checkpointer
    await check_pool_health_and_recreate()
    return _GLOBAL_CHECKPOINTER


async def initialize_checkpointer():
    """Initialize the global checkpointer with proper async context management."""
    global _GLOBAL_CHECKPOINTER
    if _GLOBAL_CHECKPOINTER is None:
        try:
            print__checkpointers_debug(
                "🚀 CHECKPOINTER INIT: Initializing PostgreSQL Connection System..."
            )
            print__checkpointers_debug(
                f"🔍 CHECKPOINTER INIT: Current global checkpointer state: {_GLOBAL_CHECKPOINTER}"
            )

            # Create and initialize the checkpointer using the official AsyncPostgresSaver method
            print__checkpointers_debug(
                "🔧 CHECKPOINTER INIT: Creating PostgreSQL checkpointer using official factory method..."
            )

            checkpointer = await create_async_postgres_saver()

            print__checkpointers_debug(
                f"✅ CHECKPOINTER INIT: Created checkpointer type: {type(checkpointer).__name__}"
            )

            # Set the global checkpointer directly (no wrapper needed)
            _GLOBAL_CHECKPOINTER = checkpointer

            print__checkpointers_debug(
                "✅ CHECKPOINTER INIT: PostgreSQL checkpointer initialized successfully using official AsyncPostgresSaver"
            )

        except Exception as e:
            print__checkpointers_debug(
                f"❌ CHECKPOINTER INIT: PostgreSQL checkpointer initialization failed: {e}"
            )
            print__checkpointers_debug(
                "🔄 CHECKPOINTER INIT: Falling back to InMemorySaver..."
            )
            _GLOBAL_CHECKPOINTER = MemorySaver()


async def cleanup_checkpointer():
    """Clean up the global checkpointer on shutdown using force_close_modern_pools() for thorough cleanup."""
    global _GLOBAL_CHECKPOINTER

    print__checkpointers_debug(
        "🧹 CHECKPOINTER CLEANUP: Starting checkpointer cleanup..."
    )

    if _GLOBAL_CHECKPOINTER:
        try:
            # Check if it's an AsyncPostgresSaver that needs proper cleanup
            if hasattr(
                _GLOBAL_CHECKPOINTER, "__class__"
            ) and "AsyncPostgresSaver" in str(type(_GLOBAL_CHECKPOINTER)):
                print__checkpointers_debug(
                    "🔄 CHECKPOINTER CLEANUP: Cleaning up AsyncPostgresSaver using force_close_modern_pools()..."
                )
                # Use the more thorough cleanup function for shutdown scenarios
                await force_close_modern_pools()
            else:
                print__checkpointers_debug(
                    f"🔄 CHECKPOINTER CLEANUP: Cleaning up {type(_GLOBAL_CHECKPOINTER).__name__}..."
                )
                # For other types (like MemorySaver), no special cleanup needed
                _GLOBAL_CHECKPOINTER = None

        except Exception as e:
            print__checkpointers_debug(
                f"⚠️ CHECKPOINTER CLEANUP: Error during checkpointer cleanup: {e}"
            )
        finally:
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug(
                "✅ CHECKPOINTER CLEANUP: Checkpointer cleanup completed"
            )
    else:
        print__checkpointers_debug(
            "ℹ️ CHECKPOINTER CLEANUP: No checkpointer to clean up"
        )

    # ...existing code...

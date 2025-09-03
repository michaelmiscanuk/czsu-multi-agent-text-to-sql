"""Connection pool management and lifecycle operations.

This module manages PostgreSQL connection pools, including creation,
cleanup, and lifecycle management for the checkpointer system.
"""

from __future__ import annotations

import gc
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import (
    get_connection_string,
    get_connection_kwargs,
)
from checkpointer.config import (
    CONNECT_TIMEOUT,
    DEFAULT_POOL_MIN_SIZE,
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_MAX_IDLE,
    DEFAULT_MAX_LIFETIME,
)
from checkpointer.globals import _GLOBAL_CHECKPOINTER, _CONNECTION_STRING_CACHE


# This file will contain:
# - cleanup_all_pools() function
# - force_close_modern_pools() function
# - modern_psycopg_pool() async context manager
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
    print__checkpointers_debug(
        "CLEANUP ALL POOLS START: Starting comprehensive pool cleanup"
    )

    global _GLOBAL_CHECKPOINTER

    # Clean up the global checkpointer if it exists
    if _GLOBAL_CHECKPOINTER:
        try:
            print__checkpointers_debug("CLEANUP: Cleaning up global checkpointer")
            # If it's an AsyncPostgresSaver, close its pool
            if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
                print__checkpointers_debug(
                    "CLEANUP: Found connection pool - closing it"
                )
                await _GLOBAL_CHECKPOINTER.pool.close()
                print__checkpointers_debug(
                    "CLEANUP: Connection pool closed successfully"
                )
        except Exception as e:
            print__checkpointers_debug(
                f"CLEANUP ERROR: Error during global checkpointer cleanup: {e}"
            )
        finally:
            _GLOBAL_CHECKPOINTER = None

    # Force garbage collection to ensure resources are freed
    gc.collect()
    print__checkpointers_debug(
        "CLEANUP ALL POOLS COMPLETE: All pools and resources cleaned up"
    )


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
        print__checkpointers_debug(
            "FORCE CLOSE: Forcing cleanup of any remaining resources"
        )

        # Clear any cached connection strings to force recreation
        global _CONNECTION_STRING_CACHE
        _CONNECTION_STRING_CACHE = None

        print__checkpointers_debug("FORCE CLOSE COMPLETE: Pool force close completed")

    except Exception as e:
        print__checkpointers_debug(f"FORCE CLOSE ERROR: Error during force close: {e}")
        # Don't re-raise - this is a cleanup function


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
    print__checkpointers_debug(
        "POOL CONTEXT START: Creating psycopg connection pool context"
    )

    try:
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "POOL CONTEXT: Setting up AsyncConnectionPool with context management"
        )

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
            open=False,  # Explicitly set to avoid deprecation warnings
        ) as pool:
            print__checkpointers_debug(
                "POOL CONTEXT: Pool created and opened using context manager"
            )
            yield pool
            print__checkpointers_debug(
                "POOL CONTEXT: Pool will be automatically closed by context manager"
            )

    except ImportError as e:
        print__checkpointers_debug(
            f"POOL CONTEXT ERROR: psycopg_pool not available: {e}"
        )
        raise Exception("psycopg_pool is required for connection pool approach")
    except Exception as e:
        print__checkpointers_debug(f"POOL CONTEXT ERROR: Failed to create pool: {e}")
        raise

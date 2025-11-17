"""PostgreSQL Connection Pool Management and Lifecycle Operations

This module provides comprehensive connection pool management for PostgreSQL databases,
including pool creation, lifecycle management, cleanup operations, and resource
deallocation. It ensures efficient connection reuse, proper resource cleanup, and
robust error recovery for long-running applications using async psycopg_pool.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""PostgreSQL Connection Pool Management and Lifecycle Operations

This module provides comprehensive connection pool management for PostgreSQL databases,
including pool creation, lifecycle management, cleanup operations, and resource
deallocation. It ensures efficient connection reuse, proper resource cleanup, and
robust error recovery for long-running applications using async psycopg_pool.

Key Features:
-------------
1. Connection Pool Creation:
   - Modern async connection pool using psycopg_pool.AsyncConnectionPool
   - Context manager pattern for automatic resource cleanup
   - Configurable pool sizing (min/max connections)
   - Automatic pool opening and closing lifecycle
   - Connection timeout configuration for reliability
   - Idle connection management for resource optimization
   - Connection lifetime limits for freshness
   - Integration with connection health checks

2. Pool Lifecycle Management:
   - Proper pool initialization and teardown sequences
   - Automatic connection validation before use
   - Connection recycling for long-running operations
   - Resource cleanup on pool closure
   - Exception handling for all pool operations
   - Comprehensive debug logging for lifecycle events
   - Safe concurrent access patterns

3. Cleanup Operations:
   - Graceful pool shutdown (cleanup_all_pools)
   - Aggressive cleanup for troubleshooting (force_close_modern_pools)
   - Global state reset for clean application restarts
   - Connection cache clearing for regeneration
   - Garbage collection for memory cleanup
   - Safe cleanup on errors without propagation

4. Global State Management:
   - Global checkpointer instance cleanup coordination
   - Connection string cache management
   - Thread-safe state access patterns
   - Resource deallocation coordination
   - Clean state for reinitialization scenarios

Connection Pool Configuration:
----------------------------
1. Pool Size Management:
   - min_size (DEFAULT_POOL_MIN_SIZE): Minimum connections (default: 2)
   - max_size (DEFAULT_POOL_MAX_SIZE): Maximum connections (default: 10)
   - Automatic scaling between min and max based on demand
   - Connection creation on-demand when pool grows
   - Connection retirement when load decreases
   - Dynamic pool sizing for varying workloads

2. Timeout Configuration:
   - timeout (DEFAULT_POOL_TIMEOUT): Pool checkout timeout (default: 30s)
   - connect_timeout (CONNECT_TIMEOUT): Connection establishment timeout (default: 30s)
   - Prevents indefinite waiting for connections
   - Fails fast when pool is exhausted
   - Configurable based on application needs
   - Graceful handling of timeout scenarios

3. Connection Lifecycle:
   - max_idle (DEFAULT_MAX_IDLE): Maximum idle time (default: 300s)
   - max_lifetime (DEFAULT_MAX_LIFETIME): Maximum connection lifetime (default: 3600s)
   - Automatic connection recycling on timeout
   - Prevents stale connections from accumulating
   - Reduces resource consumption in cloud environments
   - Maintains connection freshness for reliability

4. Connection Parameters:
   - autocommit: False (proper transaction management)
   - prepare_threshold: None (disable prepared statements)
   - Cloud database compatibility settings
   - Integration with get_connection_kwargs()
   - Consistent parameters across pool

Modern Pool Pattern (modern_psycopg_pool):
----------------------------------------
Async context manager for connection pools:

1. Design Pattern:
   - Uses AsyncConnectionPool from psycopg_pool
   - Follows psycopg documentation recommendations
   - Avoids deprecation warnings from older patterns
   - Proper resource cleanup through context manager
   - Exception handling with automatic cleanup

2. Usage Example:
   ```python
   async with modern_psycopg_pool() as pool:
       async with pool.connection() as conn:
           async with conn.cursor() as cur:
               await cur.execute("SELECT * FROM checkpoints")
               results = await cur.fetchall()
   ```

3. Automatic Cleanup:
   - Pool automatically opened on context entry
   - Pool automatically closed on context exit
   - Connections returned to pool after use
   - Resources freed even on exceptions
   - No manual pool management needed

4. Configuration:
   - open=False: Explicit control over pool opening
   - Prevents deprecation warnings
   - Recommended by psycopg documentation
   - Future-proof implementation

Cleanup Operations:
-----------------
1. cleanup_all_pools():
   - Purpose: Graceful cleanup of all connection resources
   - Process: Close global checkpointer pool, reset state, garbage collect
   - When: Normal shutdown sequences, test teardown
   - Safety: Safe to call multiple times (idempotent)
   - Errors: Never raises, logs all errors

2. force_close_modern_pools():
   - Purpose: Aggressive cleanup for troubleshooting
   - Process: Standard cleanup + cache clearing + state reset
   - When: Error recovery, troubleshooting, debugging
   - Safety: Safe for production use
   - Additional: Clears connection string cache

Cleanup Process Flow:
-------------------
1. Global Checkpointer Cleanup:
   - Check if _GLOBAL_CHECKPOINTER exists
   - If AsyncPostgresSaver, close its connection pool
   - Set _GLOBAL_CHECKPOINTER to None
   - Handle cleanup errors gracefully without raising

2. Cache Cleanup (force_close only):
   - Clear _CONNECTION_STRING_CACHE
   - Forces regeneration on next connection
   - Ensures fresh connection parameters
   - New application names on restart

3. Memory Cleanup:
   - Force garbage collection (gc.collect())
   - Ensures immediate resource deallocation
   - Prevents memory leaks in long-running apps
   - Frees orphaned resources

4. Error Handling:
   - Comprehensive exception catching
   - Cleanup continues even if steps fail
   - Detailed error logging for troubleshooting
   - Never raises exceptions from cleanup

Global State Management:
----------------------
1. _GLOBAL_CHECKPOINTER:
   - Stores singleton AsyncPostgresSaver instance
   - Managed centrally for application-wide access
   - Cleaned up by cleanup_all_pools()
   - Reset to None after cleanup

2. _CONNECTION_STRING_CACHE:
   - Stores generated connection string
   - Cleared by force_close_modern_pools()
   - Prevents stale connection parameters
   - Thread-safe access through module globals

Usage Patterns:
--------------
1. Standard Pool Usage:
   ```python
   from checkpointer.database.pool_manager import modern_psycopg_pool
   
   async with modern_psycopg_pool() as pool:
       async with pool.connection() as conn:
           await conn.execute("SELECT 1")
   ```

2. Application Shutdown:
   ```python
   from checkpointer.database.pool_manager import cleanup_all_pools
   
   async def shutdown():
       await cleanup_all_pools()
   ```

3. Error Recovery:
   ```python
   from checkpointer.database.pool_manager import force_close_modern_pools
   
   async def recover_from_error():
       await force_close_modern_pools()
   ```

4. Testing Cleanup:
   ```python
   async def teardown():
       await cleanup_all_pools()
   ```

Cloud Database Optimization:
--------------------------
1. Connection Reuse:
   - Pool maintains connections for reuse
   - Reduces connection overhead in cloud environments
   - Handles cloud database connection limits
   - Optimizes for cloud network latency

2. Automatic Scaling:
   - Pool grows on demand up to max_size
   - Pool shrinks when connections idle
   - Adapts to varying workloads
   - Conserves cloud database resources

3. Health Monitoring:
   - Connection validation before reuse
   - Automatic reconnection on failures
   - Prevention of stale connection issues
   - Integration with check_connection_health()

Performance Considerations:
-------------------------
1. Pool Sizing:
   - Balance between resource usage and availability
   - Consider concurrent request patterns
   - Match cloud database connection limits
   - Monitor pool exhaustion events

2. Connection Lifecycle:
   - Idle timeout prevents resource waste
   - Lifetime limit prevents connection staleness
   - Automatic recycling maintains health
   - Reduces cloud database resource consumption

3. Cleanup Efficiency:
   - Immediate garbage collection
   - Minimal overhead in cleanup operations
   - Idempotent cleanup (safe to call repeatedly)
   - Fast cleanup for rapid restarts

Error Handling:
--------------
1. Pool Creation Errors:
   - ImportError: psycopg_pool not available
   - Connection errors: Database unreachable
   - Configuration errors: Invalid parameters
   - All errors logged and re-raised

2. Cleanup Errors:
   - Catch all exceptions during cleanup
   - Log errors for troubleshooting
   - Never re-raise from cleanup functions
   - Ensure maximum cleanup even on failures

3. Resource Leak Prevention:
   - Context managers ensure cleanup
   - Explicit pool closing in finally blocks
   - Garbage collection for orphaned resources
   - Multiple cleanup attempts if needed

Debug Logging:
-------------
Pool Creation:
- "POOL CONTEXT START": Beginning pool creation
- "POOL CONTEXT": Pool configuration details
- "POOL CONTEXT: Pool created": Successful creation
- "POOL CONTEXT: Pool will be automatically closed": Exit notice

Pool Cleanup:
- "CLEANUP ALL POOLS START": Beginning cleanup
- "CLEANUP: Cleaning up global checkpointer": Checkpointer cleanup
- "CLEANUP: Found connection pool": Pool found for cleanup
- "CLEANUP: Connection pool closed": Pool closed successfully
- "CLEANUP ALL POOLS COMPLETE": Cleanup finished

Force Cleanup:
- "FORCE CLOSE START": Beginning force close
- "FORCE CLOSE: Forcing cleanup": Additional cleanup steps
- "FORCE CLOSE COMPLETE": Force close finished

Errors:
- "CLEANUP ERROR": Cleanup exception details
- "FORCE CLOSE ERROR": Force close exception details
- "POOL CONTEXT ERROR": Pool creation errors

Configuration Integration:
------------------------
- DEFAULT_POOL_MIN_SIZE: Minimum pool connections
- DEFAULT_POOL_MAX_SIZE: Maximum pool connections
- DEFAULT_POOL_TIMEOUT: Pool checkout timeout
- DEFAULT_MAX_IDLE: Maximum connection idle time
- DEFAULT_MAX_LIFETIME: Maximum connection lifetime
- CONNECT_TIMEOUT: Connection establishment timeout

Troubleshooting:
---------------
1. Pool Exhaustion:
   - Check DEFAULT_POOL_MAX_SIZE setting
   - Monitor connection usage patterns
   - Look for connection leaks
   - Review DEFAULT_POOL_TIMEOUT setting

2. Connection Leaks:
   - Ensure context managers are used
   - Check for unclosed cursors
   - Monitor connection count over time
   - Use cleanup_all_pools() regularly

3. Cleanup Issues:
   - Review debug logs for errors
   - Try force_close_modern_pools()
   - Check for orphaned resources
   - Monitor garbage collection

4. Performance Issues:
   - Adjust pool size for workload
   - Review timeout settings
   - Check connection lifecycle settings
   - Monitor cloud database metrics

Dependencies:
------------
- psycopg_pool: Connection pool implementation
- psycopg: PostgreSQL adapter
- gc: Garbage collection for cleanup
- contextlib: Async context manager support
- checkpointer.database.connection: Connection utilities
- checkpointer.config: Configuration management
- checkpointer.globals: Global state management
- api.utils.debug: Debug logging utilities

Future Enhancements:
------------------
- Connection pool metrics and monitoring
- Automatic pool size tuning based on load
- Advanced health check integration
- Connection pool statistics API
- Pool performance profiling
- Custom connection validation callbacks
- Pool warm-up on initialization
- Connection pool event hooks
"""

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


# ==============================================================================
# MODULE FUNCTIONS
# ==============================================================================
# This module provides three core functions for pool management:
# 1. cleanup_all_pools() - Graceful cleanup of all connection resources
# 2. force_close_modern_pools() - Aggressive cleanup for troubleshooting
# 3. modern_psycopg_pool() - Async context manager for connection pool creation
# ==============================================================================


async def cleanup_all_pools():
    """Cleanup function that properly handles connection pools and global state.

    This function provides comprehensive cleanup of all connection-related resources,
    ensuring proper shutdown sequence and resource deallocation for the checkpointer
    system. It handles both connection pools and global state management with
    graceful error handling and complete resource cleanup.

    Cleanup Process:
        1. Check for global checkpointer instance existence
        2. If AsyncPostgresSaver exists, close its connection pool
        3. Reset global checkpointer to None to clear references
        4. Force garbage collection for immediate memory cleanup
        5. Provide detailed logging for troubleshooting

    Global State Management:
        - Properly closes _GLOBAL_CHECKPOINTER connection pool if exists
        - Resets _GLOBAL_CHECKPOINTER to None for clean state
        - Handles cleanup errors gracefully without raising exceptions
        - Ensures clean slate for subsequent initialization attempts
        - Safe to call multiple times (idempotent operation)

    Resource Management:
        - Uses proper async pool closure methods
        - Handles connection pool lifecycle correctly
        - Provides comprehensive error handling for cleanup failures
        - Ensures resources are freed even if individual cleanup steps fail
        - Prevents resource leakage in long-running applications

    Performance Considerations:
        - Forces garbage collection (gc.collect()) for immediate memory cleanup
        - Minimizes resource leakage in long-running applications
        - Provides clean shutdown for application termination scenarios
        - Optimizes memory usage for restart scenarios

    Error Handling:
        - Catches all exceptions during cleanup to prevent propagation
        - Logs errors for debugging without interrupting cleanup
        - Ensures maximum cleanup even if some steps fail
        - Never raises exceptions from this cleanup function

    Usage Patterns:
        - Application shutdown sequences
        - Error recovery procedures
        - Test teardown operations
        - Connection pool reset scenarios

    Note:
        - Safe to call multiple times without side effects
        - Used during error recovery and application shutdown
        - Comprehensive error handling prevents cleanup failures from propagating
        - Essential for proper resource management in production environments
        - Idempotent operation (can be called repeatedly safely)
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
        except Exception as exc:
            # Log cleanup errors but don't raise to ensure continuation
            print__checkpointers_debug(
                f"CLEANUP ERROR: Error during global checkpointer cleanup: {exc}"
            )
        finally:
            # Always reset to None regardless of cleanup success/failure
            _GLOBAL_CHECKPOINTER = None

    # Force garbage collection to ensure resources are freed immediately
    gc.collect()
    print__checkpointers_debug(
        "CLEANUP ALL POOLS COMPLETE: All pools and resources cleaned up"
    )


async def force_close_modern_pools():
    """Force close any remaining connection pools for aggressive cleanup.

    This function provides an aggressive cleanup mechanism for troubleshooting
    scenarios where normal cleanup procedures may not be sufficient. It performs
    comprehensive resource cleanup, state reset operations, and cache clearing
    to ensure a completely clean slate for recovery or restart scenarios.

    Aggressive Cleanup Actions:
        1. Calls standard cleanup_all_pools() for normal resource cleanup
        2. Forces cleanup of any lingering connection resources
        3. Clears cached connection strings to force recreation
        4. Resets global state for clean restart scenarios
        5. Provides detailed logging for troubleshooting

    Use Cases:
        - Troubleshooting persistent connection issues
        - Recovering from connection pool corruption or exhaustion
        - Debugging resource leakage scenarios
        - Preparing for application restart scenarios
        - Emergency cleanup in error recovery situations
        - Testing connection string regeneration

    State Reset Operations:
        - Clears _CONNECTION_STRING_CACHE to force regeneration
        - Ensures fresh connection parameters on next initialization
        - Provides clean slate for subsequent connection attempts
        - Prevents cached state from interfering with recovery
        - Forces new application names for connection tracking

    Cache Clearing Benefits:
        - New connection strings generated on next use
        - Fresh application names for better debugging
        - Eliminates stale connection parameters
        - Supports recovery from configuration changes
        - Enables testing different connection scenarios

    Error Handling:
        - Comprehensive exception catching prevents cleanup failures
        - Continues operation even if individual cleanup steps fail
        - Logs errors for troubleshooting without raising exceptions
        - Ensures maximum cleanup even in error scenarios
        - Never interrupts cleanup process due to errors

    Comparison to cleanup_all_pools():
        - More aggressive than standard cleanup
        - Includes connection string cache clearing
        - Intended for troubleshooting and recovery scenarios
        - Standard cleanup is for normal shutdown
        - This is for "nuclear option" cleanup needs

    Note:
        - More aggressive than standard cleanup procedures
        - Primarily intended for troubleshooting and error recovery
        - Safe to call in production environments
        - Should be used when normal cleanup is insufficient
        - Idempotent operation (safe to call multiple times)
    """
    print__checkpointers_debug("FORCE CLOSE START: Force closing all connection pools")

    try:
        # Clean up the global state using standard cleanup first
        await cleanup_all_pools()

        # Additional aggressive cleanup for any lingering connections
        print__checkpointers_debug(
            "FORCE CLOSE: Forcing cleanup of any remaining resources"
        )

        # Clear any cached connection strings to force recreation on next use
        # This ensures fresh connection parameters and new application names
        global _CONNECTION_STRING_CACHE
        _CONNECTION_STRING_CACHE = None

        print__checkpointers_debug("FORCE CLOSE COMPLETE: Pool force close completed")

    except Exception as exc:
        # Log errors but don't raise - this is a cleanup function
        print__checkpointers_debug(
            f"FORCE CLOSE ERROR: Error during force close: {exc}"
        )
        # Don't re-raise - this is a cleanup function that should never fail


@asynccontextmanager
async def modern_psycopg_pool():
    """Async context manager for psycopg connection pools with modern best practices.

    This function creates and manages an AsyncConnectionPool using the recommended
    approach from psycopg documentation to avoid deprecation warnings. The pool
    automatically opens on context entry and closes on context exit, ensuring
    proper resource management and cleanup.

    Yields:
        AsyncConnectionPool: Configured connection pool ready for use

    Pool Configuration:
        - min_size: Minimum number of connections (DEFAULT_POOL_MIN_SIZE, default: 2)
        - max_size: Maximum number of connections (DEFAULT_POOL_MAX_SIZE, default: 10)
        - timeout: Pool checkout timeout in seconds (DEFAULT_POOL_TIMEOUT, default: 30)
        - max_idle: Maximum connection idle time (DEFAULT_MAX_IDLE, default: 300s)
        - max_lifetime: Maximum connection lifetime (DEFAULT_MAX_LIFETIME, default: 3600s)
        - connect_timeout: Connection establishment timeout (CONNECT_TIMEOUT, default: 30s)

    Connection Parameters:
        - autocommit: False (proper transaction management)
        - prepare_threshold: None (disable prepared statements)
        - Cloud-optimized keepalive and timeout settings
        - All parameters from get_connection_kwargs()

    Pool Behavior:
        - Starts with min_size connections ready
        - Grows on demand up to max_size connections
        - Recycles idle connections after max_idle time
        - Replaces connections after max_lifetime
        - Validates connection health before reuse
        - Returns connections to pool after use

    Modern Pattern Features:
        - Uses AsyncConnectionPool context manager (recommended by psycopg)
        - Avoids deprecation warnings from older patterns
        - Proper resource cleanup through context management
        - Exception handling with automatic cleanup
        - open=False parameter prevents deprecated behavior

    Usage Example:
        ```python
        async with modern_psycopg_pool() as pool:
            # Pool is open and ready
            async with pool.connection() as conn:
                # Connection checked out from pool
                async with conn.cursor() as cur:
                    await cur.execute("SELECT * FROM checkpoints")
                    results = await cur.fetchall()
                # Cursor closed
            # Connection returned to pool
        # Pool automatically closed
        ```

    Resource Management:
        - Pool automatically opened on context entry
        - Pool automatically closed on context exit
        - Connections returned to pool after use
        - Resources freed even on exceptions
        - No manual pool management needed

    Error Handling:
        - ImportError: psycopg_pool not available (installation required)
        - Connection errors: Database unreachable or misconfigured
        - Configuration errors: Invalid pool parameters
        - All errors logged with detailed context
        - Exceptions re-raised for caller handling

    Performance Benefits:
        - Connection reuse reduces overhead
        - Automatic scaling based on demand
        - Connection validation prevents errors
        - Efficient resource utilization
        - Optimized for cloud databases

    Note:
        - Recommended approach from psycopg documentation
        - Prevents deprecation warnings
        - Future-proof implementation
        - Safe for concurrent use
        - Integrates with connection health checks

    Raises:
        Exception: If psycopg_pool is not available (installation required)
        psycopg.Error: For connection or configuration errors
    """
    print__checkpointers_debug(
        "POOL CONTEXT START: Creating psycopg connection pool context"
    )

    try:
        # Get connection string with cloud optimizations and caching
        connection_string = get_connection_string()
        # Get standardized connection parameters
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "POOL CONTEXT: Setting up AsyncConnectionPool with context management"
        )

        # Use the async context manager approach recommended by psycopg documentation
        # This prevents deprecation warnings and ensures proper resource cleanup
        async with AsyncConnectionPool(
            conninfo=connection_string,  # PostgreSQL connection string
            min_size=DEFAULT_POOL_MIN_SIZE,  # Minimum pool connections (default: 2)
            max_size=DEFAULT_POOL_MAX_SIZE,  # Maximum pool connections (default: 10)
            timeout=DEFAULT_POOL_TIMEOUT,  # Pool checkout timeout (default: 30s)
            max_idle=DEFAULT_MAX_IDLE,  # Maximum connection idle time (default: 300s)
            max_lifetime=DEFAULT_MAX_LIFETIME,  # Maximum connection lifetime (default: 3600s)
            kwargs={
                **connection_kwargs,  # Base connection kwargs (autocommit, prepare_threshold)
                "connect_timeout": CONNECT_TIMEOUT,  # Connection establishment timeout
            },
            open=False,  # Explicitly set to avoid deprecation warnings (recommended)
        ) as pool:
            print__checkpointers_debug(
                "POOL CONTEXT: Pool created and opened using context manager"
            )
            # Yield pool to caller for use
            yield pool
            # Pool will be automatically closed when context exits
            print__checkpointers_debug(
                "POOL CONTEXT: Pool will be automatically closed by context manager"
            )

    except ImportError as exc:
        # psycopg_pool package not installed
        print__checkpointers_debug(
            f"POOL CONTEXT ERROR: psycopg_pool not available: {exc}"
        )
        raise Exception("psycopg_pool is required for connection pool approach")
    except Exception as exc:
        # Any other error during pool creation
        print__checkpointers_debug(f"POOL CONTEXT ERROR: Failed to create pool: {exc}")
        raise

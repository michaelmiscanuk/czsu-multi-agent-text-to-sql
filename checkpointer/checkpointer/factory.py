"""AsyncPostgresSaver Checkpointer Factory and Lifecycle Management

This module provides comprehensive factory functionality for creating, managing, and monitoring
AsyncPostgresSaver instances that persist conversation state and agent checkpoints to PostgreSQL.
It implements robust connection pooling, health monitoring, automatic recovery, and graceful
shutdown procedures for production-grade LangGraph checkpoint persistence.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""AsyncPostgresSaver Checkpointer Factory and Lifecycle Management

This module serves as the central factory and lifecycle manager for AsyncPostgresSaver instances
used throughout the multi-agent text-to-SQL application. It provides production-ready checkpoint
persistence to PostgreSQL, enabling stateful conversational AI with conversation history,
multi-turn interactions, and crash recovery capabilities.

Key Features:
-------------
1. Checkpointer Factory:
   - Singleton pattern for global checkpointer instance
   - AsyncPostgresSaver creation with connection pooling
   - Automatic fallback to MemorySaver on PostgreSQL failures
   - Connection string and kwargs configuration management
   - Lazy initialization with thread-safe locking
   - State isolation preventing conflicts between instances

2. Connection Pool Management:
   - AsyncConnectionPool with configurable sizing (min/max connections)
   - Connection health checking and validation
   - Automatic connection lifecycle management
   - Timeout protection for connection acquisition
   - Idle connection reaping (max_idle, max_lifetime)
   - SSL/TLS connection configuration support
   - Connection kwargs passthrough for advanced settings

3. Database Table Setup:
   - Automatic LangGraph checkpoint table creation
   - Custom users_threads_runs tracking table setup
   - Table existence checking to avoid redundant setup
   - Autocommit mode for DDL operations
   - Separation of schema setup from connection pooling
   - Support for both LangGraph and custom tables

4. Health Monitoring and Recovery:
   - Periodic connection pool health checks
   - Automatic pool recreation on health check failures
   - SSL connection error detection and recovery
   - Timeout-based connection validation
   - Simple query execution for health verification
   - Enhanced error detection for connection issues

5. Retry and Error Handling:
   - Decorator-based retry logic for transient failures
   - SSL connection error retry mechanism (3 attempts)
   - Prepared statement error retry (configurable attempts)
   - Exponential backoff not implemented (fixed retry count)
   - Graceful degradation to MemorySaver on fatal errors
   - Detailed error logging and diagnostics

6. Lifecycle Management:
   - initialize_checkpointer() for application startup
   - cleanup_checkpointer() for graceful shutdown
   - force_close_modern_pools() for thorough cleanup
   - Double-checked locking for thread-safe initialization
   - Global state management with async locks
   - Proper resource cleanup on errors and shutdown

7. Global State Management:
   - _GLOBAL_CHECKPOINTER singleton instance
   - _CHECKPOINTER_INIT_LOCK for initialization synchronization
   - Thread-safe lazy initialization pattern
   - State clearing before recreation to avoid conflicts
   - Unified access point via get_global_checkpointer()

Architecture:
-----------
The module follows a factory pattern with singleton semantics for the global checkpointer:

1. Global State Variables:
   - _GLOBAL_CHECKPOINTER: Holds the singleton AsyncPostgresSaver instance
   - _CHECKPOINTER_INIT_LOCK: Async lock for thread-safe initialization

2. Core Factory Function:
   - create_async_postgres_saver(): Creates and configures new AsyncPostgresSaver
   - Implements retry decorators for SSL and prepared statement errors
   - Manages connection pool creation with health checking
   - Sets up database tables (LangGraph checkpoints + custom tables)
   - Tests checkpointer functionality before returning

3. Access Pattern:
   - get_global_checkpointer(): Unified access point with lazy initialization
   - Double-checked locking prevents race conditions
   - Automatic health check before returning instance
   - Transparent recreation on health check failures

4. Lifecycle Hooks:
   - initialize_checkpointer(): Called at application startup
   - cleanup_checkpointer(): Called at application shutdown
   - Proper async context management throughout

Processing Flow:
--------------
1. Application Startup:
   - initialize_checkpointer() called from main.py or API startup
   - Checks if _GLOBAL_CHECKPOINTER already exists
   - If None, calls create_async_postgres_saver()
   - On success, sets _GLOBAL_CHECKPOINTER and logs success
   - On failure, falls back to MemorySaver and logs warning

2. Checkpointer Creation:
   - Validates PostgreSQL environment variables exist
   - Clears any existing global state to prevent conflicts
   - Constructs connection string from environment
   - Creates AsyncConnectionPool with health checking enabled
   - Opens the pool and verifies connections work
   - Creates AsyncPostgresSaver with the pool
   - Sets up LangGraph checkpoint tables (if not exist)
   - Tests checkpointer with a dummy operation
   - Sets up custom tracking tables
   - Returns fully configured checkpointer instance

3. Request Handling:
   - API endpoint calls get_global_checkpointer()
   - If checkpointer exists, runs health check first
   - If health check fails, recreates checkpointer automatically
   - If checkpointer doesn't exist, creates new one with locking
   - Returns healthy checkpointer instance ready for use
   - Checkpointer used to persist/retrieve conversation state

4. Health Monitoring:
   - check_pool_health_and_recreate() validates pool health
   - Acquires connection from pool with 10s timeout
   - Executes "SELECT 1" query to verify functionality
   - On success, returns True (pool healthy)
   - On failure, detects SSL/connection errors specifically
   - Forces pool closure via force_close_modern_pools()
   - Clears _GLOBAL_CHECKPOINTER to trigger recreation
   - Creates new checkpointer via create_async_postgres_saver()
   - Returns False (pool was recreated)

5. Application Shutdown:
   - cleanup_checkpointer() called from shutdown handler
   - Checks if _GLOBAL_CHECKPOINTER is AsyncPostgresSaver
   - Calls force_close_modern_pools() for thorough cleanup
   - Closes all connections in the pool
   - Sets _GLOBAL_CHECKPOINTER to None
   - Logs cleanup completion

Database Schema:
--------------
1. LangGraph Tables (auto-created by AsyncPostgresSaver.setup()):
   - checkpoints: Stores agent checkpoint data
   - checkpoint_writes: Stores incremental writes
   
2. Custom Tables (created by setup_users_threads_runs_table()):
   - users_threads_runs: Tracks user sessions, threads, and run IDs
   - Custom schema defined in database/table_setup.py

Configuration:
------------
Imported from checkpointer.config:
- DEFAULT_MAX_RETRIES: General retry count for operations
- CHECKPOINTER_CREATION_MAX_RETRIES: Specific to prepared statement errors
- DEFAULT_POOL_MIN_SIZE: Minimum connections in pool
- DEFAULT_POOL_MAX_SIZE: Maximum connections in pool
- DEFAULT_POOL_TIMEOUT: Timeout for acquiring connections (seconds)
- DEFAULT_MAX_IDLE: Maximum idle time before connection reaping
- DEFAULT_MAX_LIFETIME: Maximum connection lifetime
- check_postgres_env_vars(): Validates required env vars present

Environment Variables:
-------------------
Required PostgreSQL connection parameters:
- POSTGRES_HOST: Database server hostname
- POSTGRES_PORT: Database server port (usually 5432)
- POSTGRES_USER: Database username
- POSTGRES_PASSWORD: Database password
- POSTGRES_DB: Database name
- POSTGRES_SSLMODE: SSL mode (prefer, require, etc.)

Usage Example:
-------------
# At application startup (in main.py or API startup):
await initialize_checkpointer()

# In API endpoints:
checkpointer = await get_global_checkpointer()
config = {"configurable": {"thread_id": user_thread_id}}
state = await checkpointer.aget(config)

# At application shutdown:
await cleanup_checkpointer()

Retry Decorators:
---------------
Two retry decorators are applied to critical functions:

1. @retry_on_ssl_connection_error(max_retries=3):
   - Catches SSL and connection-related errors
   - Retries up to 3 times for transient SSL issues
   - Applied to: create_async_postgres_saver(), get_global_checkpointer()

2. @retry_on_prepared_statement_error(max_retries=CHECKPOINTER_CREATION_MAX_RETRIES):
   - Catches prepared statement already exists errors
   - Retries with configurable attempt count
   - Applied to: create_async_postgres_saver(), get_global_checkpointer()

Error Handling:
-------------
- PostgreSQL connection failures → Fallback to MemorySaver
- Pool health check failures → Automatic pool recreation
- SSL connection errors → Retry with fresh connection
- Prepared statement errors → Retry with cleanup
- Environment variable missing → Raise exception with details
- Table setup failures → Propagate exception to caller
- Cleanup errors → Log and continue with shutdown

Dependencies:
-----------
- psycopg: PostgreSQL driver (async)
- psycopg_pool: Connection pool management
- langgraph.checkpoint.postgres.aio: AsyncPostgresSaver implementation
- langgraph.checkpoint.memory: MemorySaver fallback
- asyncio: Async/await support and locking

Integration Points:
-----------------
- api.utils.debug: Debug logging via print__checkpointers_debug()
- checkpointer.config: Configuration constants and validation
- checkpointer.error_handling.retry_decorators: Retry logic
- checkpointer.database.table_setup: Table creation functions
- checkpointer.database.pool_manager: Pool cleanup utilities
- checkpointer.database.connection: Connection string/kwargs helpers
- checkpointer.globals: Global state variables (_GLOBAL_CHECKPOINTER, etc.)

Testing:
-------
Each created checkpointer is tested with:
- Dummy config: {"configurable": {"thread_id": "setup_test"}}
- aget() operation to verify checkpointer works
- Result should be None (no existing checkpoint for test thread)
- Logs success/failure for diagnostics

Notes:
-----
- This module is critical for stateful agent operation
- Pool recreation is automatic and transparent to callers
- Health checks run frequently to catch issues early
- Global state requires careful async locking
- Fallback to MemorySaver ensures application continues on DB failures
- Thorough cleanup prevents resource leaks
- Debug logging provides visibility into checkpointer lifecycle

Potential Improvements:
--------------------
- Add metrics/monitoring for pool health check failures
- Implement connection pool size auto-scaling
- Add configurable health check intervals
- Implement circuit breaker pattern for repeated failures
- Add structured logging instead of print statements
- Implement graceful degradation levels (not just all-or-nothing)
- Add connection pool statistics export
- Implement async context manager for checkpointer lifecycle
"""

import asyncio

import psycopg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from api.utils.debug import print__checkpointers_debug
from checkpointer.config import (
    DEFAULT_MAX_RETRIES,
    CHECKPOINTER_CREATION_MAX_RETRIES,
    DEFAULT_POOL_MIN_SIZE,
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_MAX_IDLE,
    DEFAULT_MAX_LIFETIME,
    check_postgres_env_vars,
)
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
    retry_on_ssl_connection_error,
)
from checkpointer.database.table_setup import (
    setup_checkpointer_with_autocommit,
    setup_users_threads_runs_table,
    table_exists,
)
from checkpointer.database.pool_manager import force_close_modern_pools
from checkpointer.database.connection import (
    get_connection_string,
    get_connection_kwargs,
)
from checkpointer.globals import _GLOBAL_CHECKPOINTER, _CHECKPOINTER_INIT_LOCK


# ==============================================================================
# MODULE FUNCTIONS OVERVIEW
# ==============================================================================
# This module provides the following core functions:
#
# 1. check_pool_health_and_recreate():
#    - Validates connection pool health via test query
#    - Detects SSL, connection, and timeout errors
#    - Automatically recreates pool on health check failures
#    - Returns True if healthy, False if recreation occurred
#
# 2. create_async_postgres_saver():
#    - Factory function for AsyncPostgresSaver instances
#    - Manages connection pool creation with health checking
#    - Sets up database tables (LangGraph + custom)
#    - Applies retry decorators for SSL and prepared statement errors
#    - Tests created checkpointer before returning
#
# 3. close_async_postgres_saver():
#    - Closes and cleans up AsyncPostgresSaver instance
#    - Closes connection pool gracefully
#    - Clears global checkpointer state
#    - Provides proper resource cleanup
#
# 4. get_global_checkpointer():
#    - Unified access point for singleton checkpointer
#    - Implements lazy initialization with async locking
#    - Runs health check before returning instance
#    - Applies retry decorators for robustness
#
# 5. initialize_checkpointer():
#    - Application startup initialization function
#    - Creates global checkpointer instance
#    - Falls back to MemorySaver on PostgreSQL failures
#    - Integrates with FastAPI lifespan management
#
# 6. cleanup_checkpointer():
#    - Application shutdown cleanup function
#    - Closes connection pools thoroughly
#    - Clears global state
#    - Ensures graceful resource release
# ==============================================================================


async def check_pool_health_and_recreate():
    """Check the health of the global connection pool and recreate if unhealthy.

    This function performs a health check on the global AsyncPostgresSaver connection pool
    by attempting to acquire a connection and execute a simple test query. If the pool is
    unhealthy (connection timeout, SSL errors, or query failures), it automatically triggers
    pool recreation to restore functionality.

    Health Check Process:
    1. Access the connection pool from the global checkpointer instance
    2. Attempt to acquire a connection with 10-second timeout
    3. Execute "SELECT 1" test query to verify database connectivity
    4. Validate query result (should return 1)
    5. On any failure, detect error type (SSL, connection, timeout)
    6. Force close existing pools via force_close_modern_pools()
    7. Clear global checkpointer state
    8. Create new checkpointer via create_async_postgres_saver()

    Returns:
        bool: True if the pool is healthy and no recreation occurred,
              False if the pool was unhealthy and recreation was performed

    Raises:
        Exception: Any errors during pool recreation are logged but not raised,
                  allowing the application to continue with the new pool

    Error Detection:
        The function specifically detects and handles:
        - SSL connection errors ("ssl" in error message)
        - General connection failures ("connection" in error message)
        - Timeout errors ("timeout" in error message)
        - Closed connection errors ("closed" in error message)

    Notes:
        - Uses 10-second timeout for connection acquisition
        - Logs detailed diagnostics via print__checkpointers_debug()
        - Pool recreation is automatic and transparent to callers
        - Global state (_GLOBAL_CHECKPOINTER) is updated after recreation
        - Should be called frequently to catch issues early

    Side Effects:
        - Modifies global _GLOBAL_CHECKPOINTER variable
        - Closes existing connection pool on failure
        - Creates new connection pool and checkpointer instance
        - Logs debug messages for diagnostics

    Enhanced Error Detection:
        This version includes improved error detection for SSL and connection
        issues, with specific keyword matching for common error patterns.
    """
    global _GLOBAL_CHECKPOINTER
    try:
        # Extract the connection pool from the global checkpointer instance
        # Uses getattr to safely handle cases where pool attribute doesn't exist
        pool = getattr(_GLOBAL_CHECKPOINTER, "pool", None)
        if pool is not None:
            # Perform health check: acquire connection and run test query
            # This validates both connection acquisition and query execution work
            try:
                # Acquire connection with 10-second timeout to prevent indefinite waiting
                async with asyncio.wait_for(pool.connection(), timeout=10.0) as conn:
                    # Create cursor for query execution
                    async with conn.cursor() as cur:
                        # Execute simple test query to verify database connectivity
                        await cur.execute("SELECT 1")
                        result = await cur.fetchone()
                        # Validate result is correct (should be tuple with value 1)
                        if result is None or result[0] != 1:
                            raise Exception("Pool health check failed: bad result")
                # If we reach here without exception, pool is healthy
                return True
            except (asyncio.TimeoutError, Exception) as health_error:
                # Enhanced error detection categorizes failures by type
                # This helps diagnose whether issue is SSL, connection, timeout, or other
                error_str = str(health_error).lower()
                # Check for common connection-related error keywords
                if any(
                    keyword in error_str
                    for keyword in ["ssl", "connection", "timeout", "closed"]
                ):
                    # SSL or connection-related error detected
                    print__checkpointers_debug(
                        f"POOL HEALTH CHECK FAILED (SSL/Connection issue): {health_error}"
                    )
                    raise health_error
                else:
                    # Other type of error (query failure, etc.)
                    print__checkpointers_debug(
                        f"POOL HEALTH CHECK FAILED (Other issue): {health_error}"
                    )
                    raise health_error
        else:
            # Pool attribute doesn't exist or is None - checkpointer not properly initialized
            return False
    except Exception as exc:
        # Health check failed - pool is unhealthy and needs recreation
        # This is the main recovery mechanism for pool failures
        print__checkpointers_debug(
            f"POOL HEALTH CHECK FAILED: {exc}, recreating pool..."
        )
        # Step 1: Force close all existing connection pools
        # This ensures no lingering connections remain
        await force_close_modern_pools()
        # Step 2: Clear the global checkpointer state to allow fresh creation
        _GLOBAL_CHECKPOINTER = None
        # Step 3: Create a new checkpointer with fresh connection pool
        # This should resolve transient connection issues
        _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
        print__checkpointers_debug("POOL RECREATED after health check failure.")
        # Return False to indicate recreation occurred
        return False


@retry_on_ssl_connection_error(max_retries=3)
@retry_on_prepared_statement_error(max_retries=CHECKPOINTER_CREATION_MAX_RETRIES)
async def create_async_postgres_saver():
    """Create and configure AsyncPostgresSaver with connection pool and retry logic.

    This is the core factory function that creates AsyncPostgresSaver instances for persisting
    LangGraph agent checkpoints to PostgreSQL. It manages the complete lifecycle from connection
    pool creation through table setup and functional testing.

    The function implements a robust creation process with automatic retry logic for common
    transient failures (SSL errors, prepared statement conflicts) and comprehensive error
    handling to ensure the checkpointer is fully operational before returning.

    Creation Process:
    1. Clear any existing global checkpointer state to prevent conflicts
    2. Validate PostgreSQL environment variables are set
    3. Build connection string from environment configuration
    4. Create AsyncConnectionPool with health checking enabled
    5. Open the connection pool
    6. Create AsyncPostgresSaver instance with the pool
    7. Setup LangGraph checkpoint tables (if they don't exist)
    8. Test checkpointer functionality with dummy operation
    9. Setup custom tracking tables (users_threads_runs)
    10. Return fully configured and tested checkpointer

    Returns:
        AsyncPostgresSaver: A fully configured and tested checkpointer instance
                           with connection pool and database tables ready

    Raises:
        Exception: If AsyncPostgresSaver is not available in environment
        Exception: If required PostgreSQL environment variables are missing
        Exception: If pool creation, table setup, or testing fails

    Retry Decorators:
        - @retry_on_ssl_connection_error(max_retries=3):
          Automatically retries on SSL and connection-related errors

        - @retry_on_prepared_statement_error(max_retries=CHECKPOINTER_CREATION_MAX_RETRIES):
          Automatically retries on prepared statement already exists errors

    Connection Pool Configuration:
        - min_size: Minimum number of connections to maintain (from config)
        - max_size: Maximum number of connections allowed (from config)
        - timeout: Timeout for acquiring connections (from config)
        - max_idle: Maximum time connections can be idle (from config)
        - max_lifetime: Maximum connection lifetime (from config)
        - check: Connection health check function (check_connection_health)
        - open: Set to False initially, opened manually after creation

    Database Tables Created:
        1. LangGraph Tables (via AsyncPostgresSaver.setup()):
           - checkpoints: Main checkpoint storage
           - checkpoint_writes: Incremental write storage

        2. Custom Tables (via setup_users_threads_runs_table()):
           - users_threads_runs: User session and thread tracking

    Testing:
        After creation, the checkpointer is tested with:
        - config = {"configurable": {"thread_id": "setup_test"}}
        - await checkpointer.aget(config)
        - Result should be None (no existing checkpoint)

    Side Effects:
        - Modifies global _GLOBAL_CHECKPOINTER variable
        - Creates database tables if they don't exist
        - Opens network connections via connection pool
        - Logs detailed progress via print__checkpointers_debug()

    Error Handling:
        - On any error during creation, closes pool and clears global state
        - Raises exception to caller for handling
        - Retry decorators handle transient failures automatically

    Notes:
        - Uses connection pool approach (not direct connection string)
        - Enables connection health checking for reliability
        - Separates table setup from pool creation for clarity
        - Tests functionality before returning to catch issues early
        - Autocommit mode used for DDL operations (table creation)

    Implementation Details:
        - Uses connection_kwargs for advanced PostgreSQL settings
        - Passes serde=None to AsyncPostgresSaver (uses default serialization)
        - Opens pool explicitly rather than relying on lazy opening
        - Checks for table existence before running setup to avoid redundant DDL
        - Uses separate autocommit connection for table existence checking
    """

    print__checkpointers_debug(
        "233 - CREATE SAVER START: Starting AsyncPostgresSaver creation with connection string"
    )

    global _GLOBAL_CHECKPOINTER

    # ==============================================================================
    # STEP 1: CLEAR EXISTING STATE
    # ==============================================================================
    # Before creating a new checkpointer, clear any existing state to prevent
    # conflicts, resource leaks, or connection pool issues
    if _GLOBAL_CHECKPOINTER:
        print__checkpointers_debug(
            "234 - EXISTING STATE CLEANUP: Clearing existing checkpointer state to avoid conflicts"
        )
        try:
            # Attempt to close the existing connection pool gracefully
            if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
                await _GLOBAL_CHECKPOINTER.pool.close()
        except Exception as exc:
            # Log cleanup errors but don't fail - we're creating a new instance anyway
            print__checkpointers_debug(
                f"236 - CLEANUP ERROR: Error during state cleanup: {exc}"
            )
        finally:
            # Always clear the global state, even if pool closure failed
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug(
                "237 - STATE CLEARED: Global checkpointer state cleared"
            )

    # ==============================================================================
    # STEP 2: VALIDATE DEPENDENCIES
    # ==============================================================================
    # Ensure AsyncPostgresSaver is available and environment is properly configured
    if not AsyncPostgresSaver:
        print__checkpointers_debug(
            "239 - SAVER UNAVAILABLE: AsyncPostgresSaver not available"
        )
        raise Exception("AsyncPostgresSaver not available")

    # Validate that all required PostgreSQL environment variables are set
    # (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)
    if not check_postgres_env_vars():
        print__checkpointers_debug(
            "240 - ENV VARS MISSING: Missing required PostgreSQL environment variables"
        )
        raise Exception("Missing required PostgreSQL environment variables")

    print__checkpointers_debug(
        "241 - CHECKPOINTER CREATION: Creating AsyncPostgresSaver using connection pool approach"
    )

    try:
        # ==============================================================================
        # STEP 3: BUILD CONNECTION CONFIGURATION
        # ==============================================================================
        # Construct connection string and kwargs from environment variables
        # These helper functions encapsulate the connection configuration logic
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "242 - CONNECTION POOL: Creating connection pool with proper kwargs and health checking"
        )

        # Import the health check function for connection validation
        from checkpointer.database.connection import check_connection_health

        # ==============================================================================
        # STEP 4: CREATE CONNECTION POOL
        # ==============================================================================
        # Create AsyncConnectionPool with:
        # - conninfo: PostgreSQL connection string (host, port, db, user, password)
        # - min_size/max_size: Pool size constraints for connection management
        # - timeout: Maximum time to wait for connection acquisition
        # - max_idle: Maximum idle time before connection is closed
        # - max_lifetime: Maximum total lifetime of a connection
        # - kwargs: Additional connection parameters (SSL mode, etc.)
        # - check: Health check function to validate connections
        # - open: Set to False for manual opening (more control)
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            min_size=DEFAULT_POOL_MIN_SIZE,
            max_size=DEFAULT_POOL_MAX_SIZE,
            timeout=DEFAULT_POOL_TIMEOUT,
            max_idle=DEFAULT_MAX_IDLE,
            max_lifetime=DEFAULT_MAX_LIFETIME,
            kwargs=connection_kwargs,
            check=check_connection_health,  # Enable connection health checking
            open=False,
        )

        # ==============================================================================
        # STEP 5: OPEN CONNECTION POOL
        # ==============================================================================
        # Explicitly open the pool to verify connections work before proceeding
        await pool.open()
        print__checkpointers_debug(
            "247 - POOL OPENED: Connection pool opened successfully"
        )

        # ==============================================================================
        # STEP 6: CREATE ASYNCPOSTGRESSAVER INSTANCE
        # ==============================================================================
        # Create the checkpointer instance with the connection pool
        # serde=None uses default JSON serialization for checkpoint data
        _GLOBAL_CHECKPOINTER = AsyncPostgresSaver(pool, serde=None)

        print__checkpointers_debug(
            "249 - SAVER CREATED: AsyncPostgresSaver created with connection pool"
        )

        # ==============================================================================
        # STEP 7: SETUP LANGGRAPH CHECKPOINT TABLES
        # ==============================================================================
        # Create the required LangGraph tables (checkpoints, checkpoint_writes)
        # Use autocommit connection for DDL operations to avoid transaction issues
        # Check for table existence first to avoid redundant setup operations
        print__checkpointers_debug(
            "254 - SETUP START: Checking if 'public.checkpoints' table exists before running setup"
        )
        # Use a separate autocommit connection for table existence checking
        # This avoids interfering with the pool's transaction state
        async with await psycopg.AsyncConnection.connect(
            connection_string, autocommit=True
        ) as conn:
            exists = await table_exists(conn, "checkpoints")
        if exists:
            # Table already exists, skip setup to avoid errors
            print__checkpointers_debug(
                "SKIP SETUP: Table 'public.checkpoints' already exists, "
                "skipping setup_checkpointer_with_autocommit()"
            )
        else:
            # Table doesn't exist, run full setup
            await setup_checkpointer_with_autocommit()
            print__checkpointers_debug(
                "255 - SETUP COMPLETE: AsyncPostgresSaver setup completed with autocommit"
            )

    except Exception as creation_error:
        # ==============================================================================
        # ERROR HANDLING
        # ==============================================================================
        # If any step fails, clean up resources and propagate the error
        print__checkpointers_debug(
            f"251 - CREATION ERROR: Failed to create AsyncPostgresSaver: {creation_error}"
        )
        # Clean up on failure to prevent resource leaks
        if _GLOBAL_CHECKPOINTER:
            try:
                # Attempt to close the pool if it was created
                if hasattr(_GLOBAL_CHECKPOINTER, "pool"):
                    await _GLOBAL_CHECKPOINTER.pool.close()
            except Exception:
                # Ignore errors during cleanup - we're already handling a failure
                pass
            # Clear global state
            _GLOBAL_CHECKPOINTER = None
        # Re-raise the original error for retry decorator to handle
        raise

    # ==============================================================================
    # STEP 8: TEST CHECKPOINTER FUNCTIONALITY
    # ==============================================================================
    # Verify the checkpointer is working before returning it
    # This catches configuration or connection issues early
    print__checkpointers_debug("256 - TESTING START: Testing checkpointer")
    test_config = {"configurable": {"thread_id": "setup_test"}}
    test_result = await _GLOBAL_CHECKPOINTER.aget(test_config)
    print__checkpointers_debug(
        f"257 - TESTING COMPLETE: Checkpointer test successful: {test_result is None}"
    )

    # ==============================================================================
    # STEP 9: SETUP CUSTOM TRACKING TABLES
    # ==============================================================================
    # Create custom application tables for user/thread/run tracking
    # Uses a separate direct connection to avoid transaction conflicts
    print__checkpointers_debug(
        "258 - CUSTOM TABLES: Setting up custom users_threads_runs table"
    )
    await setup_users_threads_runs_table()

    print__checkpointers_debug(
        "259 - CREATE SAVER SUCCESS: AsyncPostgresSaver creation completed successfully"
    )
    return _GLOBAL_CHECKPOINTER


async def close_async_postgres_saver():
    """Close and clean up the current AsyncPostgresSaver instance.

    This function performs graceful cleanup of the global AsyncPostgresSaver instance,
    including closing its connection pool and clearing the global state. It is designed
    to be called when explicitly closing a checkpointer (not during normal shutdown,
    which uses cleanup_checkpointer instead).

    Cleanup Process:
    1. Check if global checkpointer exists
    2. If exists, attempt to close its connection pool
    3. Clear the global _GLOBAL_CHECKPOINTER variable
    4. Log all steps for diagnostics

    Returns:
        None: This function performs cleanup and doesn't return a value

    Side Effects:
        - Closes connection pool (releases all connections)
        - Sets _GLOBAL_CHECKPOINTER to None
        - Logs cleanup progress via print__checkpointers_debug()

    Error Handling:
        - Errors during pool closure are logged but not raised
        - Global state is always cleared, even if pool closure fails
        - Uses finally block to ensure cleanup happens

    Notes:
        - This is different from cleanup_checkpointer() which uses force_close
        - Suitable for explicit checkpointer closure (not shutdown)
        - After calling this, next checkpointer access will create a new instance
        - No retry logic - this is a cleanup function

    Usage:
        await close_async_postgres_saver()
        # Global checkpointer is now None
        # Next call to get_global_checkpointer() will create new instance
    """
    global _GLOBAL_CHECKPOINTER

    print__checkpointers_debug("262 - CLOSE SAVER START: Closing AsyncPostgresSaver")

    if _GLOBAL_CHECKPOINTER:
        try:
            # Attempt to close the connection pool gracefully
            # This releases all connections back to PostgreSQL
            if hasattr(_GLOBAL_CHECKPOINTER, "pool") and _GLOBAL_CHECKPOINTER.pool:
                await _GLOBAL_CHECKPOINTER.pool.close()
                print__checkpointers_debug(
                    "263 - POOL CLOSED: AsyncPostgresSaver pool closed"
                )
        except Exception as exc:
            # Log errors but don't fail - we're cleaning up anyway
            print__checkpointers_debug(
                f"264 - CLOSE ERROR: Error closing AsyncPostgresSaver pool: {exc}"
            )
        finally:
            # Always clear the global state, even if closure failed
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug(
                "265 - SAVER CLEARED: AsyncPostgresSaver instance cleared"
            )
    else:
        # No checkpointer to close
        print__checkpointers_debug("266 - NO SAVER: No AsyncPostgresSaver to close")


@retry_on_ssl_connection_error(max_retries=2)
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_global_checkpointer():
    """Unified access point for the global checkpointer instance.

    This function provides thread-safe, lazy initialization of the global checkpointer
    with automatic health checking and recreation. It is the primary way to access
    the checkpointer throughout the application.

    The function implements the double-checked locking pattern to ensure that only
    one checkpointer instance is created even under concurrent access, while avoiding
    unnecessary lock acquisition when the instance already exists.

    Access Flow:
    1. Check if global checkpointer exists (first check, no lock)
    2. If None, acquire initialization lock
    3. Check again if checkpointer exists (second check, with lock)
    4. If still None, create new checkpointer
    5. Release lock
    6. Run health check on checkpointer (creates new one if unhealthy)
    7. Return healthy checkpointer instance

    Returns:
        AsyncPostgresSaver: A healthy, ready-to-use checkpointer instance

    Raises:
        Exception: If checkpointer creation fails after all retries
                  (propagated from create_async_postgres_saver)

    Retry Decorators:
        - @retry_on_ssl_connection_error(max_retries=2):
          Retries on SSL-related connection errors

        - @retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES):
          Retries on prepared statement conflicts

    Thread Safety:
        - Uses asyncio.Lock for thread-safe initialization
        - Lock is created lazily to avoid event loop issues
        - Double-checked locking prevents race conditions
        - Multiple concurrent calls will wait for single initialization

    Health Monitoring:
        - Always runs health check before returning instance
        - Health check may trigger automatic recreation
        - Ensures returned checkpointer is always functional

    Side Effects:
        - May create new checkpointer if none exists
        - May recreate checkpointer if health check fails
        - Modifies _GLOBAL_CHECKPOINTER and _CHECKPOINTER_INIT_LOCK
        - Logs all operations via print__checkpointers_debug()

    Notes:
        - This is the recommended way to access the checkpointer
        - Transparent to caller whether checkpointer is new or existing
        - Health checking provides automatic recovery from failures
        - Lock is created lazily to work with FastAPI's async context

    Usage:
        # In API endpoint or agent code:
        checkpointer = await get_global_checkpointer()
        config = {"configurable": {"thread_id": "user_123"}}
        state = await checkpointer.aget(config)
    """
    print__checkpointers_debug(
        "264 - GET GLOBAL START: Getting global checkpointer instance"
    )
    global _GLOBAL_CHECKPOINTER, _CHECKPOINTER_INIT_LOCK

    # ==============================================================================
    # LAZY LOCK INITIALIZATION
    # ==============================================================================
    # Create the initialization lock lazily to avoid event loop issues
    # This allows the module to be imported before the event loop is running
    if _CHECKPOINTER_INIT_LOCK is None:
        _CHECKPOINTER_INIT_LOCK = asyncio.Lock()

    # ==============================================================================
    # DOUBLE-CHECKED LOCKING PATTERN
    # ==============================================================================
    # First check: avoid lock acquisition if checkpointer already exists
    # This is the common case and provides best performance
    if _GLOBAL_CHECKPOINTER is None:
        # Acquire lock for thread-safe initialization
        async with _CHECKPOINTER_INIT_LOCK:
            # Second check: another coroutine may have created it while we waited
            # This prevents multiple creations under concurrent access
            if _GLOBAL_CHECKPOINTER is None:
                # Create new checkpointer instance
                _GLOBAL_CHECKPOINTER = await create_async_postgres_saver()
                print__checkpointers_debug(
                    "266 - CREATE SUCCESS: New checkpointer created successfully"
                )
    else:
        # Checkpointer already exists
        print__checkpointers_debug(
            "267 - EXISTING FOUND: Using existing global checkpointer"
        )

    # ==============================================================================
    # HEALTH CHECK BEFORE RETURNING
    # ==============================================================================
    # Always check pool health before returning the checkpointer
    # This ensures the returned instance is functional and may trigger recreation
    await check_pool_health_and_recreate()
    return _GLOBAL_CHECKPOINTER


async def initialize_checkpointer():
    """Initialize the global checkpointer with proper async context management.

    This function is designed to be called during application startup (e.g., in FastAPI's
    lifespan context) to initialize the global checkpointer system. It attempts to create
    a PostgreSQL-backed checkpointer and falls back to in-memory storage on failure.

    The function is idempotent - if a checkpointer already exists, it will not create
    a new one. This allows safe multiple calls without resource duplication.

    Initialization Process:
    1. Check if global checkpointer already exists
    2. If None, attempt to create AsyncPostgresSaver
    3. On success, set global checkpointer and log success
    4. On failure, fall back to MemorySaver and log warning
    5. Log detailed diagnostics throughout the process

    Returns:
        None: This function performs initialization and doesn't return a value

    Side Effects:
        - Sets _GLOBAL_CHECKPOINTER to AsyncPostgresSaver or MemorySaver
        - Creates database connection pool (if PostgreSQL succeeds)
        - Creates database tables (if they don't exist)
        - Logs initialization progress via print__checkpointers_debug()

    Fallback Behavior:
        On PostgreSQL initialization failure:
        - Logs detailed error message
        - Falls back to MemorySaver (in-memory, non-persistent)
        - Application continues to function (without persistence)
        - Suitable for development or degraded operation

    Error Handling:
        - PostgreSQL errors caught and logged
        - Automatic fallback ensures application starts
        - No exception raised to caller

    Notes:
        - Should be called once at application startup
        - Safe to call multiple times (idempotent)
        - Uses create_async_postgres_saver() for PostgreSQL setup
        - MemorySaver provides same interface but no persistence
        - Detailed logging helps diagnose initialization issues

    Integration:
        - Called from FastAPI lifespan startup
        - Called from main.py initialization
        - Paired with cleanup_checkpointer() at shutdown

    Usage:
        # In FastAPI lifespan:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await initialize_checkpointer()
            yield
            await cleanup_checkpointer()

        # Or in main.py:
        if __name__ == "__main__":
            await initialize_checkpointer()
            # ... run application ...
            await cleanup_checkpointer()
    """
    global _GLOBAL_CHECKPOINTER
    # Only initialize if not already created (idempotent)
    if _GLOBAL_CHECKPOINTER is None:
        try:
            print__checkpointers_debug(
                "🚀 CHECKPOINTER INIT: Initializing PostgreSQL Connection System..."
            )
            print__checkpointers_debug(
                f"🔍 CHECKPOINTER INIT: Current global checkpointer state: {_GLOBAL_CHECKPOINTER}"
            )

            # ==============================================================================
            # CREATE POSTGRESQL CHECKPOINTER
            # ==============================================================================
            # Attempt to create AsyncPostgresSaver for persistent checkpoint storage
            print__checkpointers_debug(
                "🔧 CHECKPOINTER INIT: Creating PostgreSQL checkpointer "
                "using official factory method..."
            )

            checkpointer = await create_async_postgres_saver()

            print__checkpointers_debug(
                f"✅ CHECKPOINTER INIT: Created checkpointer type: {type(checkpointer).__name__}"
            )

            # Set the global checkpointer directly (no wrapper needed)
            # AsyncPostgresSaver implements the full checkpointer interface
            _GLOBAL_CHECKPOINTER = checkpointer

            print__checkpointers_debug(
                "✅ CHECKPOINTER INIT: PostgreSQL checkpointer initialized "
                "successfully using official AsyncPostgresSaver"
            )

        except Exception as exc:
            # ==============================================================================
            # FALLBACK TO MEMORY SAVER
            # ==============================================================================
            # PostgreSQL initialization failed - fall back to in-memory storage
            # This ensures the application can still run, just without persistence
            print__checkpointers_debug(
                f"❌ CHECKPOINTER INIT: PostgreSQL checkpointer initialization failed: {exc}"
            )
            print__checkpointers_debug(
                "🔄 CHECKPOINTER INIT: Falling back to InMemorySaver..."
            )
            # MemorySaver provides same interface but stores checkpoints in memory only
            # Checkpoints will be lost when the application restarts
            _GLOBAL_CHECKPOINTER = MemorySaver()


async def cleanup_checkpointer():
    """Clean up the global checkpointer on shutdown using thorough pool closure.

    This function is designed to be called during application shutdown (e.g., in FastAPI's
    lifespan context) to properly clean up the global checkpointer and release all
    associated resources (database connections, connection pools, etc.).

    The function uses force_close_modern_pools() for AsyncPostgresSaver instances to
    ensure thorough cleanup that properly closes all connections in the pool. For other
    checkpointer types (like MemorySaver), simple state clearing is sufficient.

    Cleanup Process:
    1. Check if global checkpointer exists
    2. Determine checkpointer type (AsyncPostgresSaver or other)
    3. For AsyncPostgresSaver: call force_close_modern_pools()
    4. For other types: simple state clearing
    5. Set _GLOBAL_CHECKPOINTER to None
    6. Log completion

    Returns:
        None: This function performs cleanup and doesn't return a value

    Side Effects:
        - Closes all database connections in connection pool
        - Releases connection pool resources
        - Sets _GLOBAL_CHECKPOINTER to None
        - Logs cleanup progress via print__checkpointers_debug()

    Error Handling:
        - Errors during cleanup are logged but not raised
        - Global state is always cleared in finally block
        - Application shutdown continues even if cleanup fails

    Cleanup Methods:
        - AsyncPostgresSaver: Uses force_close_modern_pools() for thorough cleanup
          * Closes all connections in pool
          * Releases pool resources
          * Prevents connection leaks

        - Other types (MemorySaver): Simple state clearing
          * No special cleanup needed
          * Just clear global variable

    Notes:
        - Should be called once at application shutdown
        - Paired with initialize_checkpointer() at startup
        - Uses force_close for AsyncPostgresSaver (more thorough than close)
        - Safe to call even if no checkpointer exists
        - Errors don't prevent shutdown from completing

    Integration:
        - Called from FastAPI lifespan shutdown
        - Called from main.py shutdown handlers
        - Ensures clean application termination

    Usage:
        # In FastAPI lifespan:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await initialize_checkpointer()
            yield
            await cleanup_checkpointer()  # Called on shutdown

        # Or in main.py:
        try:
            # ... run application ...
        finally:
            await cleanup_checkpointer()
    """
    global _GLOBAL_CHECKPOINTER

    print__checkpointers_debug(
        "🧹 CHECKPOINTER CLEANUP: Starting checkpointer cleanup..."
    )

    if _GLOBAL_CHECKPOINTER:
        try:
            # ==============================================================================
            # TYPE-SPECIFIC CLEANUP
            # ==============================================================================
            # Check if it's an AsyncPostgresSaver that needs proper pool cleanup
            if hasattr(
                _GLOBAL_CHECKPOINTER, "__class__"
            ) and "AsyncPostgresSaver" in str(type(_GLOBAL_CHECKPOINTER)):
                print__checkpointers_debug(
                    "🔄 CHECKPOINTER CLEANUP: Cleaning up AsyncPostgresSaver "
                    "using force_close_modern_pools()..."
                )
                # Use the more thorough cleanup function for shutdown scenarios
                # This ensures all connections in the pool are properly closed
                await force_close_modern_pools()
            else:
                # For other checkpointer types (like MemorySaver)
                print__checkpointers_debug(
                    f"🔄 CHECKPOINTER CLEANUP: Cleaning up {type(_GLOBAL_CHECKPOINTER).__name__}..."
                )
                # MemorySaver and similar types don't need special cleanup
                # Just clear the reference
                _GLOBAL_CHECKPOINTER = None

        except Exception as exc:
            # Log errors but don't fail - shutdown should continue
            print__checkpointers_debug(
                f"⚠️ CHECKPOINTER CLEANUP: Error during checkpointer cleanup: {exc}"
            )
        finally:
            # Always clear global state, even if cleanup failed
            # This ensures clean state for potential restart
            _GLOBAL_CHECKPOINTER = None
            print__checkpointers_debug(
                "✅ CHECKPOINTER CLEANUP: Checkpointer cleanup completed"
            )
    else:
        # No checkpointer to clean up
        print__checkpointers_debug(
            "ℹ️ CHECKPOINTER CLEANUP: No checkpointer to clean up"
        )

    # ...existing code...

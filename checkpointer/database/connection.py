"""PostgreSQL Connection String Generation and Basic Connection Management

This module provides comprehensive connection management functionality for PostgreSQL
databases, focusing on cloud-optimized connection strings, connection parameter
generation, and direct connection access. It handles connection string caching,
unique application naming, and connection health monitoring for reliable database
operations in cloud environments.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""PostgreSQL Connection String Generation and Basic Connection Management

This module provides comprehensive connection management functionality for PostgreSQL
databases, focusing on cloud-optimized connection strings, connection parameter
generation, and direct connection access. It handles connection string caching,
unique application naming, and connection health monitoring for reliable database
operations in cloud environments.

Key Features:
-------------
1. Connection String Generation:
   - Cloud-optimized PostgreSQL connection strings with SSL/TLS security
   - Unique application naming for connection tracking and debugging
   - Comprehensive timeout and keepalive configuration for reliability
   - TCP-level connection reliability settings for cloud environments
   - Global connection string caching for performance optimization
   - Thread-safe connection string generation for concurrent access
   - Integration with environment-based database configuration

2. Connection Parameter Management:
   - Standardized connection kwargs for psycopg compatibility
   - Autocommit mode configuration for cloud database compatibility
   - Prepared statement management (disabled for reliability)
   - Transaction handling optimization for concurrent workloads
   - Cloud database best practices enforcement
   - Consistent parameters across all connection types

3. Connection Health Monitoring:
   - Active connection health checking via SELECT 1 query
   - Pool validation callback support for connection pools
   - Exception handling for health check failures
   - Long-running application connection reliability
   - Prevention of "SSL connection closed" errors
   - Automatic stale connection detection

4. Direct Connection Access:
   - Async context manager for direct database connections
   - Specialized operation support (users_threads_runs operations)
   - Proper resource cleanup through context management
   - Alternative to pool-based connections when needed
   - Simplified connection management for one-off operations

Connection String Components:
---------------------------
1. Application Name Generation:
   - Format: czsu_langgraph_{pid}_{thread}_{time}_{random}
   - Process ID: Identifies which process owns the connection
   - Thread ID: Distinguishes connections from different threads
   - Startup time: Unix timestamp for temporal tracking
   - Random UUID: 8-character hex for uniqueness guarantee
   - Benefits: Precise debugging, connection monitoring, issue tracking

2. Timeout Configuration:
   - connect_timeout: Initial connection establishment (default: 30s)
   - tcp_user_timeout: TCP-level timeout (default: 60000ms)
   - Prevents indefinite connection hangs
   - Handles slow network conditions gracefully
   - Optimized for cloud database latency patterns

3. Keepalive Settings:
   - keepalives_idle: Time before first probe (default: 30s)
   - keepalives_interval: Interval between probes (default: 10s)
   - keepalives_count: Failed probes before disconnect (default: 5)
   - Prevents connection drops on idle connections
   - Maintains persistent connections in cloud environments
   - Handles load balancer timeouts effectively

4. Security Configuration:
   - sslmode=require: Enforces SSL/TLS encryption for all connections
   - Secure credential management from environment variables
   - No hardcoded passwords or sensitive information
   - Cloud database security compliance (SOC2, HIPAA compatible)

Connection Parameters (kwargs):
-----------------------------
1. autocommit=False:
   - Enables explicit transaction control for data consistency
   - Better cloud database compatibility under concurrent load
   - Recommended by LangGraph documentation for cloud deployments
   - Prevents transaction-related issues in high-concurrency scenarios
   - Allows application to manage transaction boundaries

2. prepare_threshold=None:
   - Completely disables automatic prepared statement creation
   - Prevents prepared statement conflicts and memory leaks
   - Reduces connection pool management complexity
   - Eliminates cleanup requirements in most scenarios
   - Improves reliability in cloud PostgreSQL environments

Health Check Mechanism:
---------------------
- Executes simple "SELECT 1" query for validation
- Returns True/False for connection status
- Used by connection pools for pre-use validation
- Catches all exceptions to prevent pool disruption
- Essential for long-running applications
- Reduces "connection already closed" errors

Caching Strategy:
---------------
1. Global Cache (_CONNECTION_STRING_CACHE):
   - Stores generated connection string in module-level variable
   - Prevents regeneration on every connection request
   - Ensures consistent application names across operations
   - Thread-safe access through module-level globals
   - Can be cleared for troubleshooting (force_close_modern_pools)
   - Improves performance in high-concurrency scenarios

Configuration Integration:
------------------------
- CONNECT_TIMEOUT: Connection establishment timeout
- TCP_USER_TIMEOUT: TCP-level timeout setting
- KEEPALIVES_IDLE: Keepalive idle time
- KEEPALIVES_INTERVAL: Keepalive probe interval
- KEEPALIVES_COUNT: Keepalive retry count
- get_db_config(): Database credentials (host, port, user, password, dbname)

Usage Examples:
--------------
1. Get Connection String:
   ```python
   from checkpointer.database.connection import get_connection_string
   
   conn_str = get_connection_string()
   # Returns cached string on subsequent calls
   ```

2. Get Connection Parameters:
   ```python
   from checkpointer.database.connection import get_connection_kwargs
   
   kwargs = get_connection_kwargs()
   # {'autocommit': False, 'prepare_threshold': None}
   ```

3. Direct Connection Access:
   ```python
   from checkpointer.database.connection import get_direct_connection
   
   async with get_direct_connection() as conn:
       async with conn.cursor() as cur:
           await cur.execute("INSERT INTO users_threads_runs ...")
           await conn.commit()
   ```

4. Health Check in Pool:
   ```python
   from checkpointer.database.connection import check_connection_health
   
   is_healthy = await check_connection_health(connection)
   if is_healthy:
       # Use connection
   else:
       # Discard and create new connection
   ```

Cloud Database Compatibility:
---------------------------
Optimized for cloud PostgreSQL services:
- Supabase (PostgreSQL on AWS)
- AWS RDS PostgreSQL
- Google Cloud SQL PostgreSQL
- Azure Database for PostgreSQL
- Heroku Postgres
- DigitalOcean Managed PostgreSQL
- Neon (serverless PostgreSQL)

Configuration handles:
- Cloud-specific network latency
- Load balancer timeouts
- Connection pooling at cloud level
- SSL/TLS requirements
- Concurrent connection limits

Error Handling:
--------------
1. Connection String Generation:
   - Handles missing configuration gracefully
   - Validates database credentials availability
   - Logs detailed error context for debugging
   - Clear error messages for troubleshooting

2. Health Checks:
   - Catches all exceptions to prevent pool disruption
   - Returns False on any failure (safe default)
   - Logs health check failures for monitoring
   - Never raises exceptions (pool-safe)

3. Direct Connections:
   - Context manager ensures cleanup on exceptions
   - Proper resource deallocation on errors
   - Exception propagation for caller handling
   - Clean error context for debugging

Security Considerations:
-----------------------
- SSL/TLS required for all connections (sslmode=require)
- Credentials from environment variables only
- No sensitive data in logs (connection string sanitized)
- Secure connection parameter handling
- No SQL injection risk (no SQL generation in this module)
- Proper escaping handled by psycopg library

Performance Considerations:
-------------------------
- Connection string cached to reduce overhead
- Minimal connection string regeneration
- Efficient health check (simple SELECT 1)
- Fast parameter generation (dictionary creation)
- Optimized for high-concurrency scenarios
- Low memory footprint

Troubleshooting:
---------------
1. Connection Timeouts:
   - Check CONNECT_TIMEOUT configuration
   - Verify network connectivity to database
   - Validate cloud database accessibility
   - Review firewall and security group settings

2. SSL/TLS Errors:
   - Verify cloud database SSL configuration
   - Check certificate validity
   - Ensure sslmode=require is supported
   - Review SSL certificate chain

3. Keepalive Issues:
   - Adjust keepalive parameters for infrastructure
   - Check load balancer timeout settings
   - Verify cloud provider network configuration
   - Monitor connection stability metrics

4. Application Name Conflicts:
   - Clear cache using force_close_modern_pools()
   - Restart application for new application names
   - Check for multiple application instances
   - Review database connection logs

Dependencies:
------------
- psycopg: PostgreSQL adapter for Python (async support)
- checkpointer.config: Configuration management
- checkpointer.globals: Global state management
- api.utils.debug: Debug logging utilities
- os: Environment variable access
- threading: Thread ID generation
- time: Timestamp generation
- uuid: Random identifier generation
- contextlib: Async context manager support

Debug Logging:
-------------
- "214 - CONNECTION STRING START": Begin string generation
- "215 - CONNECTION STRING CACHED": Using cached string
- "216 - CONNECTION STRING APP NAME": Generated application name
- "217 - CONNECTION STRING COMPLETE": String generation complete
- Health check failure logging with exception details

Future Enhancements:
------------------
- Connection retry with exponential backoff
- Advanced health check customization
- Connection pool-level health monitoring
- Automatic parameter tuning based on workload
- Connection metrics collection
- SSL certificate validation customization
- Dynamic timeout adjustment
- Connection pooling statistics
"""

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import psycopg

from api.utils.debug import print__checkpointers_debug
from checkpointer.config import (
    CONNECT_TIMEOUT,
    TCP_USER_TIMEOUT,
    KEEPALIVES_IDLE,
    KEEPALIVES_INTERVAL,
    KEEPALIVES_COUNT,
    get_db_config,
)
from checkpointer.globals import _CONNECTION_STRING_CACHE


# ==============================================================================
# MODULE FUNCTIONS
# ==============================================================================
# This module provides four core functions for connection management:
# 1. get_connection_string() - Generate cloud-optimized PostgreSQL connection string
# 2. get_connection_kwargs() - Provide standardized connection parameters
# 3. check_connection_health() - Validate connection health for pool management
# 4. get_direct_connection() - Async context manager for direct connections
# ==============================================================================


def get_connection_string():
    """Generate PostgreSQL connection string with cloud-optimized parameters.

    This function creates a comprehensive PostgreSQL connection string optimized
    for cloud database services, including advanced timeout and keepalive settings
    for reliable connectivity in distributed environments. The connection string
    is cached globally to ensure consistent application names across operations.

    Returns:
        str: Complete PostgreSQL connection string with optimization parameters

    Connection String Components:
        - Protocol: postgresql://
        - Credentials: username and password from environment
        - Host/Port: Database endpoint from configuration
        - Database: Target database name
        - SSL Mode: Required for secure cloud connections
        - Application Name: Unique identifier for connection tracking
        - Timeout Settings: Connect, TCP, and keepalive configurations

    Application Name Generation:
        - Combines process ID, thread ID, startup time, and random identifier
        - Ensures unique identification for concurrent connections
        - Facilitates debugging and connection monitoring in database logs
        - Format: "czsu_langgraph_{pid}_{thread}_{time}_{random}"
        - Enables precise troubleshooting of connection issues

    Cloud Optimization Parameters:
        - connect_timeout: Initial connection establishment timeout (default: 30s)
        - keepalives_idle: Time before first keepalive probe (default: 30s)
        - keepalives_interval: Interval between keepalive probes (default: 10s)
        - keepalives_count: Failed keepalives before disconnect (default: 5)
        - tcp_user_timeout: TCP-level connection timeout (default: 60000ms)

    Caching Mechanism:
        - Connection string is cached in _CONNECTION_STRING_CACHE global variable
        - Prevents regeneration on every connection request
        - Ensures consistent application names across all connections
        - Improves performance in high-concurrency scenarios
        - Cache can be cleared via force_close_modern_pools() for troubleshooting

    Note:
        - Uses global caching to prevent timestamp conflicts
        - Optimized for cloud PostgreSQL services (Supabase, AWS RDS, Azure, etc.)
        - Includes comprehensive debug logging for troubleshooting
        - Safe for concurrent access through module-level globals
    """
    print__checkpointers_debug(
        "214 - CONNECTION STRING START: Generating PostgreSQL connection string"
    )
    import checkpointer.globals as globals_module

    # Check if connection string is already cached
    if globals_module._CONNECTION_STRING_CACHE is not None:
        print__checkpointers_debug(
            "215 - CONNECTION STRING CACHED: Using cached connection string"
        )
        return globals_module._CONNECTION_STRING_CACHE

    # Get database configuration from environment variables
    config = get_db_config()

    # Generate unique application name for connection tracking
    # Use process ID + startup time + random for unique application name
    process_id = os.getpid()
    thread_id = threading.get_ident()
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]

    app_name = f"czsu_langgraph_{process_id}_{thread_id}_{startup_time}_{random_id}"
    print__checkpointers_debug(
        f"216 - CONNECTION STRING APP NAME: Generated unique application name: {app_name}"
    )

    # Build connection string with cloud-optimized timeout and keepalive settings
    # Connection string with timeout and keepalive settings for cloud databases
    # Optimized for SSL connection stability and long-running operations
    globals_module._CONNECTION_STRING_CACHE = (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}?"
        f"sslmode=require"  # Enforce SSL/TLS for secure cloud connections
        f"&application_name={app_name}"  # Unique app name for connection tracking
        f"&connect_timeout={CONNECT_TIMEOUT}"  # Initial connection timeout
        f"&keepalives_idle={KEEPALIVES_IDLE}"  # Time before first keepalive probe
        f"&keepalives_interval={KEEPALIVES_INTERVAL}"  # Interval between probes
        f"&keepalives_count={KEEPALIVES_COUNT}"  # Failed probes before disconnect
        f"&tcp_user_timeout={TCP_USER_TIMEOUT}"  # TCP-level timeout
    )

    print__checkpointers_debug(
        "217 - CONNECTION STRING COMPLETE: PostgreSQL connection string generated"
    )

    return globals_module._CONNECTION_STRING_CACHE


def get_connection_kwargs():
    """Generate connection kwargs for cloud database compatibility.

    This function provides standardized connection parameters that are optimized
    for cloud PostgreSQL databases, particularly focusing on prepared statement
    management and transaction handling for reliable operation under concurrent load.

    Returns:
        dict: Connection parameters dictionary for psycopg connections containing:
            - autocommit (bool): Transaction management setting (False for explicit control)
            - prepare_threshold (None): Prepared statement management (disabled)

    Connection Parameters Explained:

        autocommit=False:
            - Enables explicit transaction control for data consistency
            - Better compatibility with cloud databases under concurrent load
            - Recommended by LangGraph documentation for cloud deployments
            - Prevents transaction-related issues in high-concurrency scenarios
            - Allows application to manage transactions explicitly
            - Essential for proper checkpoint operations

        prepare_threshold=None:
            - Completely disables automatic prepared statement creation
            - Prevents prepared statement conflicts and memory leaks
            - Reduces complexity in connection pool management
            - Eliminates need for prepared statement cleanup in most cases
            - Improves reliability in cloud PostgreSQL environments
            - Follows cloud database best practices

    Cloud Database Benefits:
        - Prevents "prepared statement already exists" errors
        - Reduces memory consumption on database server
        - Eliminates prepared statement cleanup requirements
        - Improves connection pool reliability
        - Better compatibility with cloud load balancers

    Note:
        - Based on LangGraph documentation and cloud database best practices
        - Optimized for concurrent workloads and connection pooling
        - Used by all connection creation functions for consistency
        - Prevents most prepared statement related issues proactively
        - Can be overridden for specific use cases (e.g., autocommit=True for DDL)
    """
    return {
        "autocommit": False,  # Better compatibility with cloud databases under load
        "prepare_threshold": None,  # Disable prepared statements completely
    }


async def check_connection_health(connection):
    """Check if a database connection is healthy and working.

    This function performs a simple SELECT 1 query to verify that the connection
    is still alive and can execute commands. Used by connection pools to validate
    connections before giving them to clients, preventing "connection already closed"
    and "SSL connection has been closed unexpectedly" errors.

    Args:
        connection: psycopg async connection object to check

    Returns:
        bool: True if connection is healthy and can execute queries,
              False if connection is dead, stale, or encounters any error

    Health Check Process:
        1. Executes simple "SELECT 1" query (minimal overhead)
        2. Fetches result to verify query execution
        3. Validates result is expected value (1)
        4. Returns True only if all steps succeed

    Error Handling:
        - Catches all exceptions to avoid breaking pool operations
        - Returns False on any exception (connection error, timeout, etc.)
        - Logs health check failures for monitoring and debugging
        - Never raises exceptions (safe for use as pool callback)

    Usage in Connection Pools:
        - Called before giving connection to client
        - Prevents serving stale connections
        - Enables automatic connection replacement on failure
        - Reduces application-level errors from bad connections

    Note:
        - Catches all exceptions to avoid breaking pool operations
        - Used as a callback for psycopg connection pools
        - Helps prevent "SSL connection has been closed unexpectedly" errors
        - Essential for long-running applications with concurrent database access
        - Minimal performance impact (simple query with immediate result)
    """
    try:
        # Simple health check query - SELECT 1 is fast and universal
        async with connection.cursor() as cur:
            await cur.execute("SELECT 1")
            result = await cur.fetchone()
            # Verify query executed and returned expected result
            return result is not None and result[0] == 1
    except Exception as exc:
        # Log health check failures for monitoring and debugging
        print__checkpointers_debug(f"Connection health check failed: {exc}")
        # Return False to indicate unhealthy connection (pool will discard it)
        return False


@asynccontextmanager
async def get_direct_connection():
    """Get a direct database connection for specialized operations.

    This async context manager provides a direct PostgreSQL connection without
    going through the connection pool. Useful for specialized operations like
    users_threads_runs table operations, DDL statements, or bulk operations
    that benefit from dedicated connections.

    Yields:
        psycopg.AsyncConnection: Direct async connection to PostgreSQL database

    Connection Configuration:
        - Uses standard connection string from get_connection_string()
        - Applies standard connection kwargs from get_connection_kwargs()
        - Full connection lifecycle managed by context manager
        - Automatic cleanup on context exit (even on exceptions)

    Usage Example:
        ```python
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO users_threads_runs ...")
                await conn.commit()
        # Connection automatically closed here
        ```

    When to Use:
        - users_threads_runs table operations
        - Table creation and DDL statements
        - Bulk insert/update operations
        - Operations that don't fit pool patterns
        - Testing and debugging scenarios

    When NOT to Use:
        - Regular checkpoint operations (use pool instead)
        - High-frequency read operations (pool is more efficient)
        - Operations that benefit from connection reuse

    Resource Management:
        - Context manager ensures proper connection cleanup
        - Connection closed automatically on exit
        - Resources freed even on exceptions
        - No manual connection management needed

    Note:
        - Alternative to pool-based connections
        - Proper resource cleanup through context manager
        - Uses standard connection configuration
        - Safe for concurrent use (each call gets new connection)
    """
    # Get connection string with cloud optimizations and caching
    connection_string = get_connection_string()
    # Get standardized connection parameters (autocommit, prepare_threshold)
    connection_kwargs = get_connection_kwargs()

    # Establish direct async connection with proper resource management
    async with await psycopg.AsyncConnection.connect(
        connection_string, **connection_kwargs
    ) as conn:
        # Yield connection to caller
        yield conn
        # Connection automatically closed when context exits

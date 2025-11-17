"""PostgreSQL Database Management Package for LangGraph Checkpointer System

This package provides comprehensive database management functionality for the LangGraph
checkpointer system, including connection management, connection pooling, and table schema
setup. It ensures reliable, efficient, and secure PostgreSQL database operations optimized
for cloud-based deployments.
"""

MODULE_DESCRIPTION = r"""PostgreSQL Database Management Package for LangGraph Checkpointer System

This package provides comprehensive database management functionality for the LangGraph
checkpointer system, including connection management, connection pooling, and table schema
setup. It ensures reliable, efficient, and secure PostgreSQL database operations optimized
for cloud-based deployments.

Key Features:
-------------
1. Connection Management (connection.py):
   - PostgreSQL connection string generation with cloud optimizations
   - SSL-enabled secure connections for cloud databases
   - Unique application naming for connection tracking and debugging
   - Comprehensive timeout and keepalive configuration
   - TCP-level connection reliability settings
   - Connection health checking for pool validation
   - Direct connection access for specialized operations
   - Global connection string caching for performance
   - Thread-safe connection parameter generation

2. Connection Pool Management (pool_manager.py):
   - Async connection pool lifecycle management
   - Configurable pool sizing (min/max connections)
   - Automatic connection health monitoring
   - Graceful pool cleanup and resource deallocation
   - Force cleanup for troubleshooting scenarios
   - Context manager patterns for proper resource handling
   - Connection timeout and idle connection management
   - Prepared statement management and cleanup
   - Integration with psycopg_pool for modern async patterns

3. Table Schema Setup (table_setup.py):
   - LangGraph checkpoint table creation and initialization
   - Custom application table setup (users_threads_runs)
   - Autocommit connection handling for DDL operations
   - Index creation for query performance optimization
   - Idempotent table creation (safe to run multiple times)
   - Transaction-aware table existence checking
   - Comprehensive error handling for setup failures
   - Support for concurrent index creation

4. Global State Management:
   - Centralized connection string caching
   - Global checkpointer instance management
   - Thread-safe state access patterns
   - Clean state reset for error recovery
   - Resource cleanup coordination

Architectural Design:
-------------------
The package follows a modular design pattern where each module has a specific
responsibility:

- connection.py: Low-level connection primitives and configuration
- pool_manager.py: High-level pool lifecycle and cleanup operations
- table_setup.py: Database schema initialization and maintenance
- __init__.py: Package-level documentation and module exports

This separation ensures:
- Clear separation of concerns
- Easy testability and maintenance
- Reusable components across the application
- Consistent error handling patterns
- Predictable resource lifecycle management

Cloud Database Optimization:
--------------------------
1. Connection Reliability:
   - Keepalive probes prevent connection drops
   - TCP user timeout handles network issues
   - SSL required for secure cloud connections
   - Connection health checks prevent stale connections
   - Automatic reconnection through pool management

2. Performance Tuning:
   - Connection pooling reduces overhead
   - Prepared statement management prevents leaks
   - Configurable pool sizes for workload optimization
   - Idle connection cleanup conserves resources
   - Connection lifetime limits prevent resource exhaustion

3. Concurrent Access:
   - Thread-safe connection management
   - Unique application names prevent conflicts
   - Autocommit mode for DDL operations
   - Transaction isolation for data consistency
   - Pool timeout handling for high concurrency

Usage Patterns:
--------------
1. Standard Pool Usage:
   ```python
   from checkpointer.database.pool_manager import modern_psycopg_pool
   
   async with modern_psycopg_pool() as pool:
       async with pool.connection() as conn:
           async with conn.cursor() as cur:
               await cur.execute("SELECT * FROM checkpoints")
               results = await cur.fetchall()
   ```

2. Direct Connection for Specific Operations:
   ```python
   from checkpointer.database.connection import get_direct_connection
   
   async with get_direct_connection() as conn:
       # Perform specialized operations
       await conn.execute("INSERT INTO users_threads_runs ...")
   ```

3. Initial Table Setup:
   ```python
   from checkpointer.database.table_setup import (
       setup_checkpointer_with_autocommit,
       setup_users_threads_runs_table
   )
   
   # Setup LangGraph tables
   await setup_checkpointer_with_autocommit()
   
   # Setup custom application tables
   await setup_users_threads_runs_table()
   ```

4. Cleanup and Resource Management:
   ```python
   from checkpointer.database.pool_manager import cleanup_all_pools
   
   # Normal cleanup
   await cleanup_all_pools()
   
   # Or aggressive cleanup for troubleshooting
   from checkpointer.database.pool_manager import force_close_modern_pools
   await force_close_modern_pools()
   ```

Configuration:
-------------
The package uses centralized configuration from checkpointer.config:

- CONNECT_TIMEOUT: Initial connection timeout (default: 30s)
- DEFAULT_POOL_MIN_SIZE: Minimum pool connections (default: 2)
- DEFAULT_POOL_MAX_SIZE: Maximum pool connections (default: 10)
- DEFAULT_POOL_TIMEOUT: Pool checkout timeout (default: 30s)
- DEFAULT_MAX_IDLE: Maximum connection idle time (default: 300s)
- DEFAULT_MAX_LIFETIME: Maximum connection lifetime (default: 3600s)
- TCP_USER_TIMEOUT: TCP-level timeout (default: 60000ms)
- KEEPALIVES_IDLE: Keepalive idle time (default: 30s)
- KEEPALIVES_INTERVAL: Keepalive interval (default: 10s)
- KEEPALIVES_COUNT: Keepalive retry count (default: 5)

Database credentials are loaded from environment variables:
- PGHOST: PostgreSQL host address
- PGPORT: PostgreSQL port (default: 5432)
- PGDATABASE: Database name
- PGUSER: Database username
- PGPASSWORD: Database password

Error Handling:
--------------
The package implements comprehensive error handling:

1. Connection Errors:
   - Timeout detection and reporting
   - SSL/TLS connection failures
   - Authentication errors
   - Network connectivity issues

2. Pool Errors:
   - Pool exhaustion (all connections in use)
   - Connection leak detection
   - Health check failures
   - Resource cleanup failures

3. Schema Errors:
   - Table creation conflicts
   - Index creation failures
   - Permission denied errors
   - Transaction block conflicts

4. Recovery Strategies:
   - Automatic retry for transient failures
   - Graceful degradation when possible
   - Detailed error logging for debugging
   - Clean state reset for recovery scenarios

Debugging:
---------
The package integrates with the application's debug system:

- Detailed logging via print__checkpointers_debug()
- Connection lifecycle tracking
- Pool state monitoring
- Table creation progress reporting
- Error context and stack traces
- Debug codes for quick issue identification

Debug output can be enabled through the application's debug configuration.

Dependencies:
------------
- psycopg (3.x): Modern PostgreSQL adapter for Python
- psycopg_pool: Connection pool implementation
- langgraph.checkpoint.postgres.aio: LangGraph checkpoint integration
- checkpointer.config: Configuration management
- checkpointer.globals: Global state management
- api.utils.debug: Debug logging utilities

Security Considerations:
-----------------------
1. Connection Security:
   - SSL/TLS required for all connections
   - Secure credential handling from environment
   - No hardcoded passwords or sensitive data
   - Connection string not logged in production

2. SQL Injection Prevention:
   - Parameterized queries throughout
   - No string concatenation for SQL
   - Proper escaping handled by psycopg
   - Prepared statement management

3. Access Control:
   - Email-based thread ownership in users_threads_runs
   - User isolation through application logic
   - Proper index coverage for security queries

Performance Considerations:
-------------------------
1. Connection Pooling:
   - Reuses connections to reduce overhead
   - Configurable pool size for workload matching
   - Automatic scaling within min/max bounds
   - Connection health checks prevent failures

2. Index Strategy:
   - Email index for user-based queries
   - Thread ID index for conversation lookup
   - Composite index for security checks
   - Covering indexes for common queries

3. Resource Management:
   - Automatic connection cleanup
   - Idle connection termination
   - Connection lifetime limits
   - Prepared statement cleanup

Testing:
-------
The package supports testing through:

- Isolated connection creation for unit tests
- Cleanup functions for test teardown
- Table existence checking for test setup
- Direct connection access for test fixtures
- Health check utilities for integration tests

Production Deployment:
--------------------
For production deployments:

1. Set appropriate pool sizes based on workload
2. Configure timeout values for your infrastructure
3. Enable SSL/TLS for security
4. Monitor connection pool metrics
5. Set up proper error alerting
6. Test cleanup procedures regularly
7. Document connection string format
8. Implement connection retry logic
9. Monitor database performance metrics
10. Plan for connection pool scaling

Troubleshooting:
---------------
Common issues and solutions:

1. "SSL connection has been closed unexpectedly":
   - Check keepalive settings
   - Verify cloud database SSL configuration
   - Increase TCP_USER_TIMEOUT if needed
   - Enable connection health checks

2. "CREATE INDEX CONCURRENTLY cannot run inside a transaction block":
   - Use setup_checkpointer_with_autocommit()
   - Ensure autocommit=True for DDL operations
   - Check transaction isolation settings

3. Pool exhaustion errors:
   - Increase DEFAULT_POOL_MAX_SIZE
   - Reduce DEFAULT_POOL_TIMEOUT
   - Check for connection leaks
   - Monitor connection usage patterns

4. Prepared statement warnings:
   - Set prepare_threshold=None
   - Use cleanup_all_pools() regularly
   - Monitor prepared statement count

Future Enhancements:
------------------
- Connection pool metrics and monitoring
- Automatic pool size scaling based on load
- Advanced prepared statement management
- Connection retry with exponential backoff
- Health check customization
- Connection pool statistics API
- Advanced error recovery strategies
- Performance profiling utilities
"""

"""Connection string generation and basic connection management.

This module handles PostgreSQL connection string generation and basic
connection operations for the checkpointer system.
"""

from __future__ import annotations

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


# This file will contain:
# - get_connection_string() function
# - get_connection_kwargs() function
# - get_direct_connection() async context manager
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
    print__checkpointers_debug(
        "214 - CONNECTION STRING START: Generating PostgreSQL connection string"
    )
    import checkpointer.globals as globals_module

    if globals_module._CONNECTION_STRING_CACHE is not None:
        print__checkpointers_debug(
            "215 - CONNECTION STRING CACHED: Using cached connection string"
        )
        return globals_module._CONNECTION_STRING_CACHE

    config = get_db_config()

    # Use process ID + startup time + random for unique application name
    process_id = os.getpid()
    thread_id = threading.get_ident()
    startup_time = int(time.time())
    random_id = uuid.uuid4().hex[:8]

    app_name = f"czsu_langgraph_{process_id}_{thread_id}_{startup_time}_{random_id}"
    print__checkpointers_debug(
        f"216 - CONNECTION STRING APP NAME: Generated unique application name: {app_name}"
    )

    # Connection string with timeout and keepalive settings for cloud databases
    # Optimized for SSL connection stability and long-running operations
    globals_module._CONNECTION_STRING_CACHE = (
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

    print__checkpointers_debug(
        "217 - CONNECTION STRING COMPLETE: PostgreSQL connection string generated"
    )

    return globals_module._CONNECTION_STRING_CACHE


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


async def check_connection_health(connection):
    """Check if a database connection is healthy and working.

    This function performs a simple SELECT 1 query to verify that the connection
    is still alive and can execute commands. Used by the connection pool to
    validate connections before giving them to clients.

    Args:
        connection: psycopg connection object to check

    Returns:
        bool: True if connection is healthy, False otherwise

    Note:
        - Catches all exceptions to avoid breaking pool operations
        - Used as a callback for psycopg connection pools
        - Helps prevent "SSL connection has been closed unexpectedly" errors
        - Essential for long-running applications with concurrent database access
    """
    try:
        # Simple health check query
        async with connection.cursor() as cur:
            await cur.execute("SELECT 1")
            result = await cur.fetchone()
            return result is not None and result[0] == 1
    except Exception as exc:
        print__checkpointers_debug(f"Connection health check failed: {exc}")
        return False


@asynccontextmanager
async def get_direct_connection():
    """Get a direct database connection for users_threads_runs operations."""

    connection_string = get_connection_string()
    connection_kwargs = get_connection_kwargs()
    async with await psycopg.AsyncConnection.connect(
        connection_string, **connection_kwargs
    ) as conn:
        yield conn

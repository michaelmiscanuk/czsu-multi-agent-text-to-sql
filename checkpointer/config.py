"""PostgreSQL Checkpointer Configuration Management

This module provides centralized configuration management for the PostgreSQL checkpointer
system, including connection parameters, retry logic, timeout settings, pool configuration,
and environment variable validation for robust database connectivity.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""PostgreSQL Checkpointer Configuration Management

This module serves as the central configuration hub for the PostgreSQL checkpointer system,
providing comprehensive management of database connection parameters, retry mechanisms,
timeout settings, connection pool configurations, and environment variable handling.

Key Features:
-------------
1. Connection Configuration Management:
   - PostgreSQL connection parameter extraction from environment variables
   - Centralized database configuration with validation
   - Support for custom host, port, database name, user, and password
   - Environment variable validation with missing parameter detection
   - Comprehensive debug logging for configuration verification

2. Retry and Timeout Configuration:
   - Configurable retry attempts for standard operations (DEFAULT_MAX_RETRIES)
   - Specialized retry logic for checkpointer creation (CHECKPOINTER_CREATION_MAX_RETRIES)
   - Connection timeout settings optimized for cloud database deployments
   - TCP timeout configuration for handling network delays (TCP_USER_TIMEOUT)
   - Keepalive settings for maintaining healthy long-lived connections
   - Prevents connection failures in distributed and cloud environments

3. Connection Pool Management:
   - Minimum and maximum pool size configuration for concurrency control
   - Pool timeout settings for request handling (DEFAULT_POOL_TIMEOUT)
   - Idle connection timeout management (DEFAULT_MAX_IDLE)
   - Connection lifetime limits for stability (DEFAULT_MAX_LIFETIME)
   - Optimized for high-concurrency multi-agent operations
   - Supports concurrent checkpoint operations with minimal contention

4. String Truncation and Display Settings:
   - User message preview length for log readability (USER_MESSAGE_PREVIEW_LENGTH)
   - AI message preview length for detailed diagnostics (AI_MESSAGE_PREVIEW_LENGTH)
   - Thread title length constraints for UI display (THREAD_TITLE_MAX_LENGTH)
   - Consistent formatting across logging and debugging outputs

5. Checkpoint Processing Configuration:
   - Maximum recent checkpoint limits to control memory usage (MAX_RECENT_CHECKPOINTS)
   - Debug message detail level control (MAX_DEBUG_MESSAGES_DETAILED)
   - Checkpoint logging interval for performance monitoring (DEBUG_CHECKPOINT_LOG_INTERVAL)
   - Balances verbosity with performance for production environments

6. Environment Variable Validation:
   - Comprehensive validation of required PostgreSQL environment variables
   - Missing variable detection and reporting
   - Fail-fast validation to prevent runtime connection errors
   - Detailed debug logging for troubleshooting configuration issues

Configuration Constants:
-----------------------
Connection and Retry Settings:
- DEFAULT_MAX_RETRIES: 2 - Standard retry attempts for most database operations
- CHECKPOINTER_CREATION_MAX_RETRIES: 2 - Retry attempts specifically for checkpointer initialization
- CONNECT_TIMEOUT: 90 seconds - Initial connection timeout for cloud databases supporting concurrent operations
- TCP_USER_TIMEOUT: 240000 ms (240 seconds) - TCP-level timeout for handling network interruptions

Connection Keepalive Settings:
- KEEPALIVES_IDLE: 300 seconds (5 minutes) - Time before first keepalive probe is sent
- KEEPALIVES_INTERVAL: 30 seconds - Interval between keepalive probes
- KEEPALIVES_COUNT: 3 - Number of failed keepalives before connection is considered dead

Connection Pool Configuration:
- DEFAULT_POOL_MIN_SIZE: 5 - Minimum number of connections maintained in pool (increased for concurrency)
- DEFAULT_POOL_MAX_SIZE: 25 - Maximum number of connections allowed in pool (supports high concurrency)
- DEFAULT_POOL_TIMEOUT: 180 seconds (3 minutes) - Maximum time to wait for connection from pool
- DEFAULT_MAX_IDLE: 600 seconds (10 minutes) - Time before idle connection is closed
- DEFAULT_MAX_LIFETIME: 3600 seconds (60 minutes) - Maximum lifetime of a connection before renewal

Display and Logging Settings:
- USER_MESSAGE_PREVIEW_LENGTH: 50 - Characters shown in user message previews
- AI_MESSAGE_PREVIEW_LENGTH: 100 - Characters shown in AI message previews
- THREAD_TITLE_MAX_LENGTH: 47 - Maximum characters for thread titles
- THREAD_TITLE_SUFFIX_LENGTH: 3 - Length of "..." suffix for truncated titles

Checkpoint Processing Settings:
- MAX_RECENT_CHECKPOINTS: 10 - Number of recent checkpoints to retain in memory
- MAX_DEBUG_MESSAGES_DETAILED: 6 - Number of messages to show in detailed debug output
- DEBUG_CHECKPOINT_LOG_INTERVAL: 5 - Log every Nth checkpoint for performance monitoring

Core Functions:
--------------
get_db_config():
    Extracts and returns PostgreSQL connection configuration from environment variables.
    Returns a dictionary with keys: user, password, host, port, dbname.
    Provides debug logging for configuration verification.
    Essential for creating database connection strings and connection pools.

check_postgres_env_vars():
    Validates that all required PostgreSQL environment variables are properly configured.
    Checks for: host, port, dbname, user, password.
    Returns True if all variables are set, False if any are missing.
    Provides detailed reporting of missing variables for troubleshooting.
    Used for fail-fast initialization to prevent runtime connection errors.

Usage Example:
-------------
# Import configuration module
from checkpointer.config import get_db_config, check_postgres_env_vars

# Validate environment before initialization
if not check_postgres_env_vars():
    raise RuntimeError("Missing required PostgreSQL environment variables")

# Get database configuration
db_config = get_db_config()
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# Use configuration constants for pool creation
from checkpointer.config import DEFAULT_POOL_MIN_SIZE, DEFAULT_POOL_MAX_SIZE
pool = create_pool(
    connection_string,
    min_size=DEFAULT_POOL_MIN_SIZE,
    max_size=DEFAULT_POOL_MAX_SIZE
)

Required Environment:
-------------------
- Python 3.8+ with type hints support
- PostgreSQL 12+ database server
- Environment variables properly configured for database access
- api.utils.debug module for debug logging (print__checkpointers_debug)

Required Environment Variables:
------------------------------
- host: PostgreSQL server hostname or IP address (e.g., "localhost", "db.example.com")
- port: PostgreSQL server port (default: 5432)
- dbname: Target database name (e.g., "checkpoints_db")
- user: Database username for authentication
- password: Database password for authentication

Architecture Context:
-------------------
This module is part of a multi-agent checkpointing system that uses PostgreSQL
for persistent storage of agent states, conversation threads, and checkpoints.
The configuration values are optimized for:
- Cloud-based PostgreSQL deployments (e.g., AWS RDS, Azure Database for PostgreSQL)
- High-concurrency multi-agent operations
- Long-running agent sessions with stable connections
- Distributed deployments with network latency considerations
- Production environments requiring reliability and performance

Performance Considerations:
--------------------------
- Connection pool sizes optimized for 20+ concurrent agent operations
- Timeout values balanced between responsiveness and stability
- Keepalive settings prevent premature connection drops in cloud environments
- Retry logic handles transient network failures gracefully
- Checkpoint limiting prevents unbounded memory growth
- Debug logging controlled to minimize performance impact

Error Handling:
--------------
- Environment variable validation prevents startup with incomplete configuration
- Missing variable reporting aids in deployment troubleshooting
- Debug logging provides visibility into configuration loading
- Type conversion for port number with fallback to default (5432)
- Graceful handling of missing optional parameters

Integration Points:
------------------
- Used by checkpointer initialization modules
- Integrated with connection pool management
- Referenced by checkpoint database operations
- Supports user management and thread tracking
- Enables concurrent multi-agent checkpoint operations

Design Philosophy:
-----------------
This module follows the principle of centralized configuration management,
providing a single source of truth for all database-related settings.
This approach:
- Reduces configuration duplication across modules
- Simplifies configuration updates and tuning
- Enables consistent retry and timeout behavior
- Facilitates testing with different configuration profiles
- Supports deployment-specific optimization through environment variables"""

import os
from pathlib import Path
from typing import TypeVar

from api.utils.debug import print__checkpointers_debug

# ==============================================================================
# MODULE ORGANIZATION
# ==============================================================================
# This configuration module is organized into the following sections:
# 1. Retry Configuration Constants - Control retry behavior for operations
# 2. Connection Timeout Constants - Manage database connection timeouts
# 3. Pool Configuration Constants - Define connection pool parameters
# 4. String Truncation Constants - Control text display and logging
# 5. Checkpoint Processing Constants - Configure checkpoint behavior
# 6. Utility Functions - Database configuration and validation helpers

# ==============================================================================
# RETRY CONFIGURATION CONSTANTS
# ==============================================================================
# These constants control retry behavior for database operations, providing
# resilience against transient failures in cloud and distributed environments.

DEFAULT_MAX_RETRIES = (
    2  # Standard retry attempts for most operations (queries, updates)
)
CHECKPOINTER_CREATION_MAX_RETRIES = (
    2  # Specialized retry attempts for checkpointer initialization
)

# ==============================================================================
# CONNECTION TIMEOUT CONFIGURATION
# ==============================================================================
# These constants manage connection timeouts at various levels (application,
# TCP, and keepalive) to ensure reliable connections in cloud deployments.

CONNECT_TIMEOUT = 90  # Initial connection timeout (seconds) for cloud databases supporting concurrent operations
TCP_USER_TIMEOUT = 240000  # TCP-level timeout in milliseconds (240 seconds) for handling network interruptions

# Connection keepalive settings maintain long-lived connections and detect dead connections
KEEPALIVES_IDLE = 300  # Time (seconds) before first keepalive probe (5 minutes) - optimized for connection health
KEEPALIVES_INTERVAL = (
    30  # Interval (seconds) between keepalive probes - detects issues quickly
)
KEEPALIVES_COUNT = (
    3  # Number of failed keepalive probes before declaring connection dead
)

# ==============================================================================
# CONNECTION POOL CONFIGURATION
# ==============================================================================
# These constants define connection pool behavior for managing database connections
# efficiently in high-concurrency multi-agent scenarios.

DEFAULT_POOL_MIN_SIZE = 5  # Minimum connections maintained in pool (increased from 1 for better concurrency)
DEFAULT_POOL_MAX_SIZE = (
    25  # Maximum connections allowed in pool (supports 20+ concurrent agent operations)
)
DEFAULT_POOL_TIMEOUT = (
    180  # Maximum wait time (seconds) for acquiring connection from pool (3 minutes)
)
DEFAULT_MAX_IDLE = 600  # Idle connection timeout (seconds) before closure (10 minutes for long-running ops)
DEFAULT_MAX_LIFETIME = 3600  # Maximum connection lifetime (seconds) before renewal (60 minutes for stability)

# ==============================================================================
# STRING TRUNCATION AND DISPLAY SETTINGS
# ==============================================================================
# These constants control text truncation for logging, UI display, and debugging
# to maintain readability while providing sufficient context.

USER_MESSAGE_PREVIEW_LENGTH = (
    50  # Character limit for user message previews in logs (balance brevity/context)
)
AI_MESSAGE_PREVIEW_LENGTH = (
    100  # Character limit for AI message previews (more detail for debugging)
)
THREAD_TITLE_MAX_LENGTH = 47  # Maximum characters for thread titles in UI and database
THREAD_TITLE_SUFFIX_LENGTH = (
    3  # Length of truncation suffix ("...") for shortened titles
)

# ==============================================================================
# CHECKPOINT PROCESSING CONFIGURATION
# ==============================================================================
# These constants control checkpoint storage, retrieval, and debug logging
# to balance performance, memory usage, and diagnostic visibility.

MAX_RECENT_CHECKPOINTS = (
    10  # Maximum number of recent checkpoints to retain (prevents memory bloat)
)
MAX_DEBUG_MESSAGES_DETAILED = (
    6  # Number of messages to display in detailed debug output
)
DEBUG_CHECKPOINT_LOG_INTERVAL = (
    5  # Frequency of checkpoint logging (log every Nth checkpoint)
)

# ==============================================================================
# TYPE VARIABLES
# ==============================================================================
# Generic type variable for type hints in configuration-related functions
T = TypeVar("T")

# ==============================================================================
# BASE DIRECTORY RESOLUTION
# ==============================================================================
# Determine the base directory of the project for path resolution.
# Falls back to current working directory if __file__ is not available.

try:
    # Standard case: resolve base directory from this file's location (parent of checkpointer/)
    base_dir = Path(__file__).resolve().parents[1]
except NameError:
    # Fallback for interactive environments where __file__ may not be defined
    base_dir = Path(os.getcwd()).parents[0]

# ==============================================================================
# DATABASE CONFIGURATION FUNCTIONS
# ==============================================================================


def get_db_config():
    """Extract database configuration from environment variables.

    This function retrieves all necessary PostgreSQL connection parameters
    from environment variables, providing a centralized configuration
    management system for database connectivity.

    Returns:
        dict: Database configuration dictionary containing:
            - user: PostgreSQL username
            - password: PostgreSQL password
            - host: PostgreSQL server hostname
            - port: PostgreSQL server port (default 5432)
            - dbname: Target database name

    Environment Variables Required:
        - user: Database username for authentication
        - password: Database password for authentication
        - host: PostgreSQL server hostname or IP address
        - port: PostgreSQL server port (defaults to 5432 if not provided)
        - dbname: Name of the target database

    Note:
        - All environment variables except 'port' are required
        - Port defaults to PostgreSQL standard port 5432
        - Used by connection string and pool creation functions
        - Provides debug logging for configuration verification
    """
    # Log configuration retrieval start for debugging and audit purposes
    print__checkpointers_debug(
        "212 - DB CONFIG START: Getting database configuration from environment variables"
    )

    # Extract all database connection parameters from environment variables
    # Port is converted to integer with default fallback to standard PostgreSQL port
    config = {
        "user": os.environ.get("user"),  # Database username for authentication
        "password": os.environ.get("password"),  # Database password (sensitive)
        "host": os.environ.get("host"),  # PostgreSQL server hostname or IP
        "port": int(os.environ.get("port", 5432)),  # Server port with default fallback
        "dbname": os.environ.get("dbname"),  # Target database name
    }

    # Log successful configuration retrieval with non-sensitive details
    # Password is intentionally excluded from debug output for security
    print__checkpointers_debug(
        f"213 - DB CONFIG RESULT: Configuration retrieved - host: {config['host']}, "
        f"port: {config['port']}, dbname: {config['dbname']}, user: {config['user']}"
    )

    return config


def check_postgres_env_vars():
    """Validate that all required PostgreSQL environment variables are configured.

    This function performs comprehensive validation of the environment configuration
    required for PostgreSQL connectivity, ensuring that all necessary parameters
    are available before attempting database operations.

    Returns:
        bool: True if all required variables are set, False if any are missing

    Required Environment Variables:
        - host: PostgreSQL server hostname
        - port: PostgreSQL server port
        - dbname: Target database name
        - user: Database username
        - password: Database password

    Validation Process:
        1. Checks each required variable for existence and non-empty value
        2. Reports missing variables for troubleshooting
        3. Provides debug logging for configuration verification
        4. Returns boolean result for conditional initialization logic

    Note:
        - Used during checkpointer initialization to fail fast on misconfiguration
        - Provides detailed feedback for missing configuration
        - Supports automated deployment validation
        - Essential for preventing runtime connection failures
    """
    # Log validation start for debugging and audit trail
    print__checkpointers_debug(
        "218 - ENV VARS CHECK START: Checking PostgreSQL environment variables"
    )

    # Define all required environment variables for PostgreSQL connectivity
    # These must be set before database operations can proceed
    required_vars = ["host", "port", "dbname", "user", "password"]

    # Iterate through required variables and collect any that are missing or empty
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):  # Check for both missing and empty string values
            missing_vars.append(var)

    # If any required variables are missing, log details and return failure status
    if missing_vars:
        print__checkpointers_debug(
            f"219 - ENV VARS MISSING: Missing required environment variables: {missing_vars}"
        )
        return False  # Return False to indicate validation failure

    # All required variables are present - log success and return True
    print__checkpointers_debug(
        "220 - ENV VARS COMPLETE: All required PostgreSQL environment variables are set"
    )
    return True  # Return True to indicate successful validation

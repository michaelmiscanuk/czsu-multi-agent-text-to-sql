"""Configuration constants and environment variable handling for PostgreSQL checkpointer.

This module centralizes all configuration constants and environment variable handling
for the PostgreSQL checkpointer system.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from api.utils.debug import print__checkpointers_debug

# This file will contain:
# - Retry configuration constants
# - Connection timeout constants
# - Pool configuration constants
# - String truncation constants
# - Checkpoint processing constants
# - get_db_config() function
# - check_postgres_env_vars() function
DEFAULT_MAX_RETRIES = 2  # Standard retry attempts for most operations
CHECKPOINTER_CREATION_MAX_RETRIES = 2  # Retry attempts for checkpointer creation
CONNECT_TIMEOUT = 20  # Initial connection timeout for cloud databases
TCP_USER_TIMEOUT = 30000  # TCP-level timeout (30 seconds in milliseconds)
KEEPALIVES_IDLE = 600  # 10 minutes before first keepalive
KEEPALIVES_INTERVAL = 30  # 30 seconds between keepalives
KEEPALIVES_COUNT = 3  # 3 failed keepalives before disconnect
DEFAULT_POOL_MIN_SIZE = 3  # Increased minimum pool size for higher concurrency
DEFAULT_POOL_MAX_SIZE = 10  # Increased maximum pool size to support more connections
DEFAULT_POOL_TIMEOUT = 20  # Pool connection timeout
DEFAULT_MAX_IDLE = 300  # 5 minutes idle timeout
DEFAULT_MAX_LIFETIME = 1800  # 30 minutes max connection lifetime
USER_MESSAGE_PREVIEW_LENGTH = 50  # Length for user message previews in logs
AI_MESSAGE_PREVIEW_LENGTH = 100  # Length for AI message previews in logs
THREAD_TITLE_MAX_LENGTH = 47  # Maximum length for thread titles
THREAD_TITLE_SUFFIX_LENGTH = 3  # Length of "..." suffix
MAX_RECENT_CHECKPOINTS = 10  # Limit checkpoints to recent ones only
MAX_DEBUG_MESSAGES_DETAILED = 6  # Show first N messages in detail
DEBUG_CHECKPOINT_LOG_INTERVAL = 5  # Log every Nth checkpoint
T = TypeVar("T")

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]


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
    print__checkpointers_debug(
        "212 - DB CONFIG START: Getting database configuration from environment variables"
    )
    config = {
        "user": os.environ.get("user"),
        "password": os.environ.get("password"),
        "host": os.environ.get("host"),
        "port": int(os.environ.get("port", 5432)),
        "dbname": os.environ.get("dbname"),
    }
    print__checkpointers_debug(
        f"213 - DB CONFIG RESULT: Configuration retrieved - host: {config['host']}, port: {config['port']}, dbname: {config['dbname']}, user: {config['user']}"
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
    print__checkpointers_debug(
        "218 - ENV VARS CHECK START: Checking PostgreSQL environment variables"
    )
    required_vars = ["host", "port", "dbname", "user", "password"]

    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        print__checkpointers_debug(
            f"219 - ENV VARS MISSING: Missing required environment variables: {missing_vars}"
        )
        return False
    else:
        print__checkpointers_debug(
            "220 - ENV VARS COMPLETE: All required PostgreSQL environment variables are set"
        )
        return True

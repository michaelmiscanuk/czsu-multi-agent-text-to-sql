"""Checkpointer Package - PostgreSQL Checkpoint Persistence System

This package provides comprehensive functionality for creating, managing, and monitoring
AsyncPostgresSaver instances that persist LangGraph agent conversation state and checkpoints
to PostgreSQL. It implements production-ready checkpoint persistence with connection pooling,
health monitoring, automatic recovery, and graceful lifecycle management.
"""

# ==============================================================================
# PACKAGE INITIALIZATION
# ==============================================================================
# This __init__.py file serves as the package entry point for the checkpointer
# subsystem. It provides package-level documentation and metadata but does not
# export any functions directly. Users should import from specific modules:
#
# - checkpointer.checkpointer.factory: Core factory and lifecycle functions
# - checkpointer.checkpointer.health: Health check functionality
#
# The package does not use __all__ to control exports, allowing explicit imports
# from submodules for better clarity and IDE support.
# ==============================================================================

MODULE_DESCRIPTION = r"""Checkpointer Package - PostgreSQL Checkpoint Persistence System

This package serves as the central module for all checkpointer-related functionality in
the multi-agent text-to-SQL application. It provides a unified interface for checkpoint
persistence, enabling stateful conversational AI with conversation history, multi-turn
interactions, and crash recovery capabilities.

Package Structure:
----------------
The package consists of three main modules:

1. factory.py:
   - Core factory functions for checkpointer creation
   - AsyncPostgresSaver instance lifecycle management
   - Connection pool management and configuration
   - Database table setup (LangGraph + custom tables)
   - Global checkpointer singleton management
   - Health monitoring and automatic recovery
   - Initialization and cleanup functions
   
   Key Functions:
   - create_async_postgres_saver(): Factory for AsyncPostgresSaver
   - close_async_postgres_saver(): Cleanup for checkpointer instances
   - get_global_checkpointer(): Unified access point with lazy init
   - initialize_checkpointer(): Application startup initialization
   - cleanup_checkpointer(): Application shutdown cleanup
   - check_pool_health_and_recreate(): Pool health validation

2. health.py:
   - Connection pool health monitoring
   - Re-exports health check functionality from factory
   - Provides logical import path for health operations
   - Compatibility layer for existing imports
   
   Key Functions:
   - check_pool_health_and_recreate(): Re-exported from factory

3. __init__.py (this file):
   - Package initialization and documentation
   - Defines package interface and organization
   - Documents overall package architecture
   - Provides usage examples and integration guidance

Key Features:
-------------
1. Checkpoint Persistence:
   - PostgreSQL-backed checkpoint storage
   - Conversation state preservation across requests
   - Multi-turn dialogue support
   - Crash recovery and resume capabilities
   - LangGraph integration via AsyncPostgresSaver

2. Connection Management:
   - AsyncConnectionPool for efficient connection reuse
   - Configurable pool sizing (min/max connections)
   - Connection health checking and validation
   - Automatic connection lifecycle management
   - SSL/TLS connection support
   - Timeout protection for connection operations

3. Health Monitoring:
   - Periodic connection pool health checks
   - Automatic pool recreation on failures
   - SSL connection error detection and recovery
   - Detailed error diagnostics and logging

4. Lifecycle Management:
   - Application startup initialization
   - Graceful shutdown with resource cleanup
   - Global singleton pattern for checkpointer
   - Thread-safe lazy initialization
   - Automatic fallback to MemorySaver on PostgreSQL failures

5. Error Handling:
   - Retry decorators for transient failures
   - SSL connection error retry (3 attempts)
   - Prepared statement error retry (configurable)
   - Graceful degradation to in-memory storage
   - Comprehensive error logging

6. Database Schema:
   - Automatic LangGraph table creation
   - Custom tracking tables (users_threads_runs)
   - Table existence checking to avoid duplicates
   - Autocommit mode for DDL operations

Architecture:
-----------
The package follows a factory pattern with singleton semantics:

┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│          (FastAPI endpoints, Agent graphs)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              get_global_checkpointer()                      │
│    (Unified access point with lazy initialization)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          _GLOBAL_CHECKPOINTER (Singleton)                   │
│              (AsyncPostgresSaver instance)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            AsyncConnectionPool                              │
│    (Manages PostgreSQL connections)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL Database                            │
│  (Stores checkpoints and conversation state)                │
└─────────────────────────────────────────────────────────────┘

Usage Example:
-------------
# 1. Application Startup (in FastAPI lifespan or main.py):
from checkpointer.checkpointer.factory import (
    initialize_checkpointer,
    cleanup_checkpointer
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    await initialize_checkpointer()
    yield
    # Cleanup on shutdown
    await cleanup_checkpointer()

# 2. Using Checkpointer in API Endpoints:
from checkpointer.checkpointer.factory import get_global_checkpointer

@app.post("/chat")
async def chat_endpoint(message: str, thread_id: str):
    # Get the global checkpointer instance
    checkpointer = await get_global_checkpointer()
    
    # Configure for this conversation thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Retrieve previous state
    state = await checkpointer.aget(config)
    
    # Process message with agent graph
    # ... agent logic ...
    
    # State is automatically saved by LangGraph
    return response

# 3. Health Monitoring (optional):
from checkpointer.checkpointer.health import check_pool_health_and_recreate

@app.get("/health/checkpointer")
async def checkpointer_health():
    is_healthy = await check_pool_health_and_recreate()
    return {
        "status": "healthy" if is_healthy else "recreated",
        "type": "AsyncPostgresSaver"
    }

Configuration:
------------
Required Environment Variables:
- POSTGRES_HOST: Database server hostname
- POSTGRES_PORT: Database server port (usually 5432)
- POSTGRES_USER: Database username
- POSTGRES_PASSWORD: Database password
- POSTGRES_DB: Database name
- POSTGRES_SSLMODE: SSL mode (prefer, require, etc.)

Configurable Constants (in checkpointer.config):
- DEFAULT_POOL_MIN_SIZE: Minimum connections in pool
- DEFAULT_POOL_MAX_SIZE: Maximum connections in pool
- DEFAULT_POOL_TIMEOUT: Connection acquisition timeout
- DEFAULT_MAX_IDLE: Maximum idle time for connections
- DEFAULT_MAX_LIFETIME: Maximum connection lifetime
- CHECKPOINTER_CREATION_MAX_RETRIES: Retry count for creation

Database Tables:
--------------
1. LangGraph Tables (auto-created):
   - checkpoints: Main checkpoint storage
   - checkpoint_writes: Incremental write storage

2. Custom Tables (auto-created):
   - users_threads_runs: User session tracking

Integration Points:
-----------------
- api.utils.debug: Debug logging functions
- checkpointer.config: Configuration constants
- checkpointer.error_handling: Retry decorators
- checkpointer.database: Database operations
- checkpointer.globals: Global state management
- langgraph.checkpoint.postgres.aio: AsyncPostgresSaver
- psycopg: PostgreSQL driver
- psycopg_pool: Connection pooling

Error Handling:
-------------
- PostgreSQL connection failures → Fallback to MemorySaver
- Pool health check failures → Automatic pool recreation
- SSL connection errors → Retry with fresh connection (3x)
- Prepared statement errors → Retry with cleanup
- Missing environment variables → Raise exception
- Cleanup errors → Log and continue shutdown

Testing:
-------
Each checkpointer creation is tested with:
- Test config: {"configurable": {"thread_id": "setup_test"}}
- aget() operation to verify functionality
- Result validation (should be None for new thread)

Notes:
-----
- Critical for stateful agent operation
- Automatic recovery from transient failures
- Global singleton ensures consistent state
- Fallback to MemorySaver ensures availability
- Health checks provide early failure detection
- Thorough cleanup prevents resource leaks

Dependencies:
-----------
- Python 3.8+
- PostgreSQL 12+
- psycopg 3.0+
- psycopg-pool
- langgraph
- asyncio
"""

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "CZSU Multi-Agent Text-to-SQL Team"
__description__ = "PostgreSQL checkpoint persistence for LangGraph agents"

# This package provides checkpointer creation, lifecycle management,
# and health monitoring functionality. Import specific functions from
# the submodules as needed.

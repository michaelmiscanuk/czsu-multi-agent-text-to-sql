# """PostgreSQL Checkpointer for LangGraph Multi-Agent Text-to-SQL System
#
# This module provides comprehensive PostgreSQL-based checkpointing functionality for the CZSU
# Multi-Agent Text-to-SQL system using LangGraph's AsyncPostgresSaver. It handles persistent
# conversation state management, user session tracking, and robust cloud database connectivity
# with advanced error recovery mechanisms.
#
# Key Features:
# -------------
# 1. AsyncPostgresSaver Integration:
#    - Official LangGraph AsyncPostgresSaver implementation
#    - Connection pool management with cloud database optimization
#    - Automatic table setup and schema initialization
#    - Context manager-based resource lifecycle management
#    - Prepared statement error handling and recovery
#    - Fallback mechanisms for different connection approaches
#
# 2. Cloud Database Compatibility:
#    - Optimized for cloud PostgreSQL services (Supabase, AWS RDS, etc.)
#    - Advanced connection string configuration with keepalive settings
#    - SSL connection management and timeout handling
#    - Connection pool sizing for concurrent workloads
#    - Automatic retry logic for transient cloud database issues
#    - Rate limiting and connection lifetime management
#
# 3. Conversation State Management:
#    - Persistent storage of LangGraph conversation checkpoints
#    - Extraction of user prompts and AI responses from checkpoints
#    - Chronological message ordering and conversation reconstruction
#    - Support for multi-turn conversations with proper sequencing
#    - Thread-based conversation isolation and security
#    - Checkpoint data validation and integrity checking
#
# 4. User Session Tracking:
#    - Custom users_threads_runs table for user-thread associations
#    - Thread ownership verification and access control
#    - Conversation metadata storage (prompts, timestamps, sentiments)
#    - Pagination support for large conversation histories
#    - Thread title generation from first user prompt
#    - Bulk conversation management and cleanup operations
#
# 5. Error Handling and Recovery:
#    - Comprehensive prepared statement error detection and recovery
#    - Automatic checkpointer recreation on connection failures
#    - Graceful degradation to in-memory storage on database failures
#    - Detailed error logging with debug mode support
#    - Connection pool health monitoring and automatic cleanup
#    - Transient error retry mechanisms with exponential backoff
#
# 6. Windows Compatibility:
#    - Windows SelectorEventLoop policy for PostgreSQL compatibility
#    - Cross-platform path handling and file system operations
#    - Proper async/await pattern implementation for Windows
#    - Event loop policy configuration for optimal performance
#
# 7. Performance Optimization:
#    - Connection pooling with configurable min/max connections
#    - Efficient checkpoint processing with limiting and pagination
#    - Prepared statement management to prevent memory leaks
#    - Garbage collection integration for resource cleanup
#    - Batch processing for multiple conversation operations
#    - Optimized SQL queries with proper indexing
#
# 8. Security Features:
#    - Thread ownership verification before data access
#    - SQL injection prevention through parameterized queries
#    - Secure connection string handling with environment variables
#    - Access control for conversation data and checkpoints
#    - Audit trail through comprehensive logging
#    - Resource isolation between different user sessions
#
# Core Components:
# ---------------
# 1. Connection Management:
#    - get_connection_string(): Cloud-optimized PostgreSQL connection strings
#    - get_connection_kwargs(): Connection parameters for cloud compatibility
#    - modern_psycopg_pool(): Async context manager for connection pooling
#    - get_direct_connection(): Direct database connections for utility operations
#
# 2. Checkpointer Lifecycle:
#    - create_async_postgres_saver(): Main checkpointer factory with retry logic
#    - setup_checkpointer_with_autocommit(): Table setup with transaction management
#    - close_async_postgres_saver(): Proper resource cleanup and connection closing
#    - initialize_checkpointer(): Global checkpointer initialization
#    - cleanup_checkpointer(): Shutdown cleanup and resource deallocation
#
# 3. Conversation Operations:
#    - create_thread_run_entry(): Create new user-thread associations
#    - get_user_chat_threads(): Retrieve user's conversation threads with pagination
#    - delete_user_thread_entries(): Secure thread deletion with ownership verification
#
# 4. Error Recovery:
#    - retry_on_prepared_statement_error(): Decorator for automatic retry logic
#    - clear_prepared_statements(): Cleanup utility for prepared statement conflicts
#    - is_prepared_statement_error(): Error classification for recovery decisions
#    - force_close_modern_pools(): Aggressive cleanup for troubleshooting
#
# 5. Utility Functions:
#    - setup_users_threads_runs_table(): Custom table creation and indexing
#    - update_thread_run_sentiment(): User feedback storage
#    - get_thread_run_sentiments(): Sentiment data retrieval
#
# Processing Flow:
# --------------
# 1. Initialization:
#    - Windows event loop policy configuration
#    - Environment variable validation
#    - Global state initialization
#    - Debug mode configuration
#
# 2. Checkpointer Creation:
#    - Connection string generation with unique application names
#    - Connection pool setup with cloud-optimized parameters
#    - AsyncPostgresSaver instantiation with error handling
#    - Table setup using autocommit connections to avoid transaction conflicts
#    - Context manager entry for proper resource management
#
# 3. Conversation Processing:
#    - Thread ownership verification for security
#    - Checkpoint retrieval with pagination and limiting
#    - Message extraction from checkpoint metadata and channel values
#    - Chronological ordering and conversation reconstruction
#    - Data validation and integrity checking
#    - Note: Message and metadata extraction is now handled by get_thread_messages_with_metadata in api/routes/chat.py
#
# 4. Database Operations:
#    - Parameterized SQL queries to prevent injection
#    - Transaction management for consistency
#    - Error handling with automatic retry logic
#    - Connection pool management for scalability
#    - Resource cleanup and connection lifecycle management
#
# 5. Error Recovery:
#    - Prepared statement error detection and cleanup
#    - Automatic checkpointer recreation on failures
#    - Graceful degradation to in-memory storage
#    - Detailed error logging for troubleshooting
#    - Connection pool health monitoring and recovery
#
# 6. Cleanup and Shutdown:
#    - Context manager exit for proper resource deallocation
#    - Connection pool closure and cleanup
#    - Global state reset and garbage collection
#    - Final error reporting and statistics
#
# Configuration:
# -------------
# Environment Variables Required:
# - host: PostgreSQL server hostname
# - port: PostgreSQL server port (default: 5432)
# - dbname: Database name
# - user: Database username
# - password: Database password
# - print__checkpointers_debug: Debug mode flag (0/1)
#
# Connection Parameters:
# - CONNECT_TIMEOUT: Initial connection timeout (20 seconds)
# - TCP_USER_TIMEOUT: TCP-level timeout (30 seconds in milliseconds)
# - KEEPALIVES_IDLE: Time before first keepalive (10 minutes)
# - KEEPALIVES_INTERVAL: Interval between keepalives (30 seconds)
# - KEEPALIVES_COUNT: Failed keepalives before disconnect (3)
#
# Pool Configuration:
# - DEFAULT_POOL_MIN_SIZE: Minimum pool connections (1)
# - DEFAULT_POOL_MAX_SIZE: Maximum pool connections (3)
# - DEFAULT_POOL_TIMEOUT: Pool connection timeout (20 seconds)
# - DEFAULT_MAX_IDLE: Maximum idle time (5 minutes)
# - DEFAULT_MAX_LIFETIME: Maximum connection lifetime (30 minutes)
#
# Processing Limits:
# - MAX_RECENT_CHECKPOINTS: Checkpoint processing limit (10)
# - MAX_DEBUG_MESSAGES_DETAILED: Detailed message logging limit (6)
# - DEBUG_CHECKPOINT_LOG_INTERVAL: Checkpoint logging frequency (every 5th)
#
# Usage Examples:
# --------------
# # Initialize global checkpointer
# await initialize_checkpointer()
#
# # Create new thread entry
# run_id = await create_thread_run_entry(
#     email="user@example.com",
#     thread_id="thread_123",
#     prompt="What is the population of Prague?"
# )
#
# # Get user's chat threads with pagination
# threads = await get_user_chat_threads(
#     email="user@example.com", limit=10, offset=0
# )
#
# # Note: Message and metadata extraction is now handled by
# # get_thread_messages_with_metadata() in api/routes/chat.py
#
# # Clean up on shutdown
# await cleanup_checkpointer()
#
# Database Schema:
# ---------------
# Core LangGraph Tables (managed by AsyncPostgresSaver):
# - checkpoints: Main checkpoint storage with thread isolation
# - checkpoint_blobs: Binary data storage for large checkpoint content
#
# Custom Application Tables:
# - users_threads_runs: User session tracking and thread ownership
#   - id: Serial primary key
#   - email: User email for ownership tracking
#   - thread_id: LangGraph thread identifier
#   - run_id: Unique run identifier for API operations
#   - prompt: User's initial prompt for thread title generation
#   - timestamp: Creation timestamp for ordering
#   - sentiment: User feedback (positive/negative/null)
#
# Indexes for Performance:
# - idx_users_threads_runs_email: Fast user lookup
# - idx_users_threads_runs_thread_id: Thread-based queries
# - idx_users_threads_runs_email_thread: Combined user-thread queries
#
# Error Handling:
# -------------
# 1. Connection Errors:
#    - Automatic retry with exponential backoff
#    - Graceful fallback to in-memory storage
#    - Connection pool recreation on persistent failures
#    - Detailed error logging for troubleshooting
#
# 2. Prepared Statement Conflicts:
#    - Automatic detection of prepared statement errors
#    - Cleanup utility to remove conflicting statements
#    - Checkpointer recreation with fresh connections
#    - Prevention through connection parameter tuning
#
# 3. Transaction Errors:
#    - Separate autocommit connections for DDL operations
#    - Transaction isolation for data consistency
#    - Rollback mechanisms for failed operations
#    - Deadlock detection and recovery
#
# 4. Data Integrity:
#    - Validation of checkpoint data structure
#    - Conversation message ordering verification
#    - Thread ownership security checks
#    - SQL injection prevention through parameterization
#
# Dependencies:
# ------------
# - asyncio: Async/await pattern implementation
# - psycopg: Modern PostgreSQL adapter for Python
# - psycopg_pool: Connection pool management
# - langgraph.checkpoint.postgres.aio: Official AsyncPostgresSaver
# - langgraph.checkpoint.memory: Fallback in-memory storage
# - threading, uuid, time: System utilities
# - pathlib, os, sys: Cross-platform file and system operations
#
# Performance Considerations:
# -------------------------
# - Connection pooling reduces overhead for frequent operations
# - Checkpoint limiting prevents memory issues with large conversations
# - Prepared statement management avoids memory leaks
# - Indexing on custom tables ensures fast query performance
# - Garbage collection integration for memory management
# - Efficient SQL patterns for bulk operations
#
# Security Considerations:
# ----------------------
# - Environment variable usage for sensitive configuration
# - Parameterized queries prevent SQL injection
# - Thread ownership verification ensures data isolation
# - SSL connections for encrypted communication
# - Access control through user email verification
# - Audit logging for security monitoring"""
#
# from __future__ import annotations
#
# import asyncio
# import os
# import sys
# from pathlib import Path
#
# # Import debug functions from utils
# from api.utils.debug import print__checkpointers_debug
# from checkpointer.checkpointer.factory import create_async_postgres_saver
# from checkpointer.config import check_postgres_env_vars
#
# # Windows event loop fix for PostgreSQL compatibility
# if sys.platform == "win32":
#     print(
#         "[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility..."
#     )
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     print("[POSTGRES-STARTUP] Event loop policy set successfully")
#
#
# # ==============================================================================
# # CONFIGURATION CONSTANTS
# # ==============================================================================
# # Retry configuration for error handling
#
# # Connection timeout constants (in seconds)
#
# # Pool configuration constants for cloud database optimization
#
# # String truncation constants for logging and display
#
# # Checkpoint processing constants for performance optimization
#
#
# # ==============================================================================
# # Global State Management
# # Single global checkpointer variable for proper cleanup
#
# # Cache the connection string to avoid timestamp conflicts
#
# # Lock for checkpointer initialization to prevent race conditions
#
# # Type variable for the retry decorator
#
#
# # Get base directory
# try:
#     pass
# except NameError:
#     BASE_DIR = Path(os.getcwd()).parents[0]
#
#
# # ==============================================================================
# # PREPARED STATEMENT ERROR HANDLING AND RECOVERY
# # ==============================================================================
#
#
# # ==============================================================================
# # PREPARED STATEMENT CLEANUP AND CONNECTION POOL MANAGEMENT
# # ==============================================================================
#
#
# # ASYNCPOSTGRESSAVER IMPLEMENTATION WITH CONNECTION POOL
#
#
# # ==============================================================================
# # DATABASE SETUP AND TABLE INITIALIZATION
# # ==============================================================================
#
#
# # USERS_THREADS_RUNS TABLE MANAGEMENT
# # Uses direct connection approach since AsyncPostgresSaver manages its own connections
#
#
# # HELPER FUNCTIONS FOR COMPATIBILITY - USING DIRECT CONNECTIONS
#
#
# # CHECKPOINTER MANAGEMENT
#
#
# # ==============================================================================
#
#
# # ==============================================================================
# # PSYCOPG CONNECTION POOL CONTEXT MANAGER
# # ==============================================================================
#
#
# if __name__ == "__main__":
#
#     async def test():
#         print__checkpointers_debug(
#             "Testing PostgreSQL checkpointer with official AsyncPostgresSaver..."
#         )
#
#         if not check_postgres_env_vars():
#             print__checkpointers_debug("Environment variables not set properly")
#             return
#
#         checkpointer = await create_async_postgres_saver()
#         print__checkpointers_debug(f"Checkpointer type: {type(checkpointer).__name__}")
#
#         # Test a simple operation
#         config = {"configurable": {"thread_id": "test_thread"}}
#         try:
#             # This should work with the official AsyncPostgresSaver
#             async for checkpoint in checkpointer.alist(config, limit=1):
#                 print__checkpointers_debug("alist() method works correctly")
#                 break
#             else:
#                 print__checkpointers_debug(
#                     "No checkpoints found (expected for fresh DB)"
#                 )
#         except Exception as e:
#             print__checkpointers_debug(f"Error testing alist(): {e}")
#
#         await close_async_postgres_saver()
#         print__checkpointers_debug("Official AsyncPostgresSaver test completed!")
#
#     asyncio.run(test())

"""User Management Package for PostgreSQL Checkpointer System

This package provides comprehensive user session tracking, conversation thread management,
and sentiment tracking functionality for the multi-agent text-to-SQL application's
PostgreSQL checkpointer system.
"""

MODULE_DESCRIPTION = r"""User Management Package for PostgreSQL Checkpointer System

This package serves as the user-facing layer of the checkpointer system, handling all
operations related to user conversation threads, thread runs, and user feedback tracking.
It provides a clean interface for managing user sessions and conversation history in the
users_threads_runs database table.

Package Structure:
----------------
The package consists of two main modules:

1. thread_operations.py:
   - Thread run entry creation and management
   - Thread retrieval with pagination support
   - Thread count operations for pagination
   - Thread deletion operations
   - Title generation from first prompts
   
   Key Functions:
   - create_thread_run_entry(): Create/update thread run entries
   - get_user_chat_threads(): Retrieve user threads with pagination
   - get_user_chat_threads_count(): Get total thread count
   - delete_user_thread_entries(): Delete all entries for a thread

2. sentiment_tracking.py:
   - User feedback (thumbs up/down) tracking
   - Sentiment update operations by run_id
   - Sentiment retrieval for entire threads
   - Analytics and quality improvement data
   
   Key Functions:
   - update_thread_run_sentiment(): Update sentiment for a run
   - get_thread_run_sentiments(): Retrieve all sentiments for a thread

3. __init__.py (this file):
   - Package initialization and documentation
   - Defines package interface and organization
   - Documents overall package architecture

Key Features:
-------------
1. Thread Lifecycle Management:
   - Create new conversation threads and runs
   - Retrieve thread history with metadata
   - Paginated thread listings for efficiency
   - Thread deletion with safety scoping
   - Automatic title generation from prompts

2. User Feedback Tracking:
   - Boolean sentiment values (True/False)
   - Per-run feedback collection
   - Thread-level sentiment aggregation
   - Analytics-ready data structure

3. Pagination Support:
   - Configurable page size (limit parameter)
   - Offset-based pagination
   - Total count for page calculation
   - Sorted by latest activity

4. Database Integration:
   - Direct PostgreSQL connections
   - Automatic retry on transient errors
   - Upsert semantics for idempotency
   - Efficient aggregation queries

5. Error Handling:
   - Graceful degradation on errors
   - Detailed debug logging
   - Exception handling with fallbacks
   - API stability guarantees

Database Schema:
--------------
Table: users_threads_runs
  - email (TEXT): User email address
  - thread_id (TEXT): Conversation thread identifier
  - run_id (TEXT, PRIMARY KEY): Unique run identifier
  - prompt (TEXT): User's initial prompt
  - timestamp (TIMESTAMP): Creation/update time
  - sentiment (BOOLEAN, nullable): User feedback

Usage Example:
-------------
# Thread Operations:
from checkpointer.user_management.thread_operations import (
    create_thread_run_entry,
    get_user_chat_threads,
    get_user_chat_threads_count,
    delete_user_thread_entries
)

# Create a new thread run:
run_id = await create_thread_run_entry(
    email="user@example.com",
    thread_id="thread-123",
    prompt="What were Q1 sales?"
)

# Get user's threads with pagination:
threads = await get_user_chat_threads(
    email="user@example.com",
    limit=20,
    offset=0
)

# Get total count for pagination:
total = await get_user_chat_threads_count(
    email="user@example.com"
)

# Delete a thread:
result = await delete_user_thread_entries(
    email="user@example.com",
    thread_id="thread-to-delete"
)

# Sentiment Tracking:
from checkpointer.user_management.sentiment_tracking import (
    update_thread_run_sentiment,
    get_thread_run_sentiments
)

# Update sentiment for a run:
success = await update_thread_run_sentiment(
    run_id="run-abc-123",
    sentiment=True  # thumbs up
)

# Get all sentiments for a thread:
sentiments = await get_thread_run_sentiments(
    email="user@example.com",
    thread_id="thread-123"
)

API Integration:
--------------
This package is typically used by FastAPI endpoints:

POST /api/threads
  - Creates new thread run entry
  - Returns run_id for tracking

GET /api/threads?limit=20&offset=0
  - Retrieves user's threads with pagination
  - Returns list of thread metadata

GET /api/threads/count
  - Returns total thread count for pagination

DELETE /api/threads/{thread_id}
  - Deletes all entries for a thread
  - Returns deletion summary

POST /api/feedback
  - Updates sentiment for a run
  - Returns success status

GET /api/threads/{thread_id}/sentiments
  - Retrieves all sentiments for thread
  - Returns sentiment dictionary

Dependencies:
-----------
- checkpointer.database.connection: Direct PostgreSQL connections
- checkpointer.error_handling.retry_decorators: Retry logic
- checkpointer.config: Configuration constants
- api.utils.debug: Debug logging
- psycopg: PostgreSQL async driver
- uuid: UUID generation
- traceback: Error diagnostics

Configuration:
------------
From checkpointer.config:
- DEFAULT_MAX_RETRIES: Maximum retry attempts
- THREAD_TITLE_MAX_LENGTH: Max chars in thread title
- THREAD_TITLE_SUFFIX_LENGTH: Length of "..." suffix

Notes:
-----
- All functions are async and use PostgreSQL async driver
- Retry decorators handle transient errors automatically
- Graceful error handling prevents API crashes
- Debug logging provides operation visibility
- Thread titles are user-friendly and auto-truncated
- Sentiments are optional (NULL if not provided)
- Deletion operations are scoped by email for security
- Pagination enables efficient handling of large datasets

Potential Improvements:
--------------------
- Add thread archival functionality
- Implement thread search and filtering
- Add thread tags/categories support
- Support for multi-sentiment ratings (1-5 stars)
- Add sentiment analytics and aggregation
- Implement soft delete with deleted_at timestamps
- Add thread sharing/collaboration features
- Support for thread export/import
"""

# ==============================================================================
# PACKAGE INITIALIZATION
# ==============================================================================
# This __init__.py file serves as the package entry point for the user_management
# subsystem. It provides package-level documentation but does not export any
# functions directly. Users should import from specific modules:
#
# - checkpointer.user_management.thread_operations: Thread CRUD operations
# - checkpointer.user_management.sentiment_tracking: Feedback tracking
#
# The package does not use __all__ to control exports, allowing explicit imports
# from submodules for better clarity and IDE support.
# ==============================================================================

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "CZSU Multi-Agent Text-to-SQL Team"
__description__ = "User session and thread management for PostgreSQL checkpointer"

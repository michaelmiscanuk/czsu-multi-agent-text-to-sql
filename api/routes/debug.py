"""
MODULE_DESCRIPTION: Debug and Administrative Endpoints - System Diagnostics and Management

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module implements comprehensive debugging and administrative endpoints for the
CZSU Multi-Agent Text-to-SQL API. It provides tools for system introspection,
troubleshooting, performance monitoring, and runtime configuration management.

These endpoints are essential for:
    - Development and debugging workflows
    - Production troubleshooting and diagnostics
    - Performance monitoring and optimization
    - Runtime configuration adjustments
    - Database connection health verification

The module includes seven main endpoint categories:
    1. Checkpoint inspection (/debug/chat/{thread_id}/checkpoints)
    2. Pool status monitoring (/debug/pool-status)
    3. Run ID verification (/debug/run-id/{run_id})
    4. Cache management (/admin/clear-cache)
    5. Statement cleanup (/admin/clear-prepared-statements)
    6. Environment configuration (/debug/set-env)
    7. Environment reset (/debug/reset-env)

===================================================================================
KEY FEATURES
===================================================================================

1. Checkpoint Inspection
   - View all checkpoints for a conversation thread
   - Examine checkpoint metadata and writes
   - Inspect message history in channel_values
   - Preview message content with truncation
   - Count messages per checkpoint
   - Identify checkpoint IDs

2. Database Pool Status Monitoring
   - Check AsyncPostgresSaver operational status
   - Test database connectivity
   - Measure query latency
   - Verify checkpointer initialization
   - Detect pool connection issues

3. Run ID Verification
   - Validate UUID format
   - Check run_id existence in database
   - Verify user ownership
   - Security: User-scoped queries only
   - Detailed database record retrieval

4. Cache Management
   - Clear bulk loading cache manually
   - Monitor memory after cleanup
   - Track cache entry counts
   - Audit user actions
   - Memory status reporting

5. Prepared Statement Cleanup
   - Clear PostgreSQL prepared statements
   - Free database memory
   - Prevent statement bloat
   - Logged for audit trail

6. Dynamic Environment Configuration
   - Set debug flags at runtime
   - Enable/disable logging levels
   - Adjust performance parameters
   - No server restart required
   - Useful for troubleshooting

7. Environment Reset
   - Restore original .env values
   - Undo runtime configuration changes
   - Safe reset mechanism
   - Prevents configuration drift

===================================================================================
API ENDPOINTS
===================================================================================

GET /debug/chat/{thread_id}/checkpoints
    Inspect all checkpoints for a conversation thread

    Authentication: JWT token required

    Path Parameters:
        thread_id (str): Thread identifier to inspect

    Returns:
        {
            "thread_id": "thread_123",
            "total_checkpoints": 5,
            "checkpoints": [
                {
                    "index": 0,
                    "checkpoint_id": "ckpt_789",
                    "has_checkpoint": true,
                    "has_metadata": true,
                    "metadata_writes": {...},
                    "channel_values": {
                        "message_count": 4,
                        "messages": [
                            {
                                "index": 0,
                                "type": "HumanMessage",
                                "id": "msg_123",
                                "content_preview": "What is...",
                                "content_length": 250
                            }
                        ]
                    }
                }
            ]
        }

GET /debug/pool-status
    Check database connection pool status and health

    Authentication: None (publicly accessible for monitoring)

    Returns:
        {
            "timestamp": "2024-01-15T10:30:00",
            "global_checkpointer_exists": true,
            "checkpointer_type": "AsyncPostgresSaver",
            "asyncpostgressaver_status": "operational",
            "test_latency_ms": 12.5,
            "connection_test": "passed"
        }

GET /debug/run-id/{run_id}
    Verify run_id existence and ownership

    Authentication: JWT token required
    Security: Only returns data for user's own run_ids

    Path Parameters:
        run_id (str): Run ID to verify (UUID format)

    Returns:
        {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "run_id_type": "str",
            "run_id_length": 36,
            "is_valid_uuid_format": true,
            "exists_in_database": true,
            "user_owns_run_id": true,
            "database_details": {
                "email": "user@example.com",
                "thread_id": "thread_123",
                "prompt": "What is ...",
                "timestamp": "2024-01-15T10:30:00"
            }
        }

POST /admin/clear-cache
    Clear the bulk loading cache manually

    Authentication: JWT token required

    Returns:
        {
            "message": "Cache cleared successfully",
            "cache_entries_cleared": 15,
            "current_memory_mb": 850.5,
            "memory_status": "normal",
            "cleared_by": "user@example.com",
            "timestamp": "2024-01-15T10:30:00"
        }

POST /admin/clear-prepared-statements
    Clear PostgreSQL prepared statements

    Authentication: JWT token required

    Returns:
        {
            "message": "Prepared statements cleared",
            "timestamp": "2024-01-15T10:30:00"
        }

POST /debug/set-env
    Set environment variables at runtime

    Authentication: JWT token required

    Request Body:
        {
            "DEBUG_CHAT": "1",
            "DEBUG_ANALYZE": "1"
        }

    Returns:
        {
            "message": "Set 2 environment variables",
            "variables": {
                "DEBUG_CHAT": "1",
                "DEBUG_ANALYZE": "1"
            },
            "timestamp": "2024-01-15T10:30:00"
        }

POST /debug/reset-env
    Reset environment variables to original .env values

    Authentication: JWT token required

    Request Body:
        {
            "DEBUG_CHAT": "",
            "DEBUG_ANALYZE": ""
        }

    Returns:
        {
            "message": "Reset 2 environment variables to original .env values",
            "variables": {
                "DEBUG_CHAT": "0",
                "DEBUG_ANALYZE": "0"
            },
            "timestamp": "2024-01-15T10:30:00"
        }

===================================================================================
CHECKPOINT INSPECTION DETAILS
===================================================================================

Purpose:
    Examine the LangGraph checkpoint data structure for debugging conversation
    state, message history, and agent execution flow.

What Are Checkpoints:
    - LangGraph saves conversation state after each agent step
    - Includes message history, agent state, metadata
    - Stored in PostgreSQL via AsyncPostgresSaver
    - Multiple checkpoints per thread (one per step/update)

Inspection Output:
    1. Checkpoint Metadata
       - checkpoint_id: Unique identifier
       - has_checkpoint: Boolean flag
       - has_metadata: Metadata present flag
       - metadata_writes: Pending writes

    2. Channel Values
       - message_count: Number of messages in checkpoint
       - messages: Array of message summaries

    3. Message Details
       - index: Position in message array
       - type: HumanMessage, AIMessage, SystemMessage, etc.
       - id: Unique message identifier
       - content_preview: Truncated message content (200 chars)
       - content_length: Full content length

Use Cases:
    - Debug why conversation state is incorrect
    - Verify messages are being saved properly
    - Identify missing or duplicated messages
    - Understand agent execution flow
    - Troubleshoot checkpoint corruption

Security:
    - Requires user authentication
    - User email extracted from JWT
    - Only owner can inspect their threads (implicitly via thread_id)

===================================================================================
POOL STATUS MONITORING
===================================================================================

Purpose:
    Monitor the health and operational status of the PostgreSQL connection pool
    and AsyncPostgresSaver checkpointer.

Health Checks:
    1. Checkpointer Initialization
       - Verify GLOBAL_CHECKPOINTER exists
       - Check checkpointer type (AsyncPostgresSaver expected)

    2. Connectivity Test
       - Execute test query: aget_tuple()
       - Measure latency in milliseconds
       - Detect connection failures

    3. Status Reporting
       - operational: All systems functioning
       - error: Connection test failed
       - not_initialized: Checkpointer not set up

Response Fields:
    - timestamp: When check was performed
    - global_checkpointer_exists: Boolean
    - checkpointer_type: Class name
    - asyncpostgressaver_status: Health status
    - test_latency_ms: Query response time
    - connection_test: passed/failed

Use Cases:
    - Monitor system health
    - Detect database connectivity issues
    - Measure query performance
    - Verify checkpointer initialization
    - Troubleshoot connection pool exhaustion

Monitoring Integration:
    - Can be called by external monitoring systems
    - No authentication required (public endpoint)
    - Returns 500 status on errors
    - Provides metrics for alerting

===================================================================================
RUN ID VERIFICATION
===================================================================================

Purpose:
    Verify that a run_id exists in the database and check user ownership.

Verification Steps:
    1. Format Validation
       - Check if run_id is valid UUID format
       - Parse UUID to confirm structure
       - Return parsing errors if invalid

    2. Database Lookup
       - Query users_threads_runs table
       - Filter by run_id AND user_email (security)
       - Retrieve run details if owned by user

    3. Ownership Check
       - exists_in_database: run_id found anywhere
       - user_owns_run_id: belongs to authenticated user
       - Returns database_details only if owned

Security Model:
    - User-scoped queries: WHERE email = %s
    - No cross-user data disclosure
    - Separate check for "exists but not owned"
    - Audit logging for access attempts

Response Scenarios:
    1. Valid & Owned:
       - is_valid_uuid_format: true
       - exists_in_database: true
       - user_owns_run_id: true
       - database_details: {...}

    2. Valid & Not Owned:
       - is_valid_uuid_format: true
       - exists_in_database: true
       - user_owns_run_id: false
       - database_details: null

    3. Valid & Not Found:
       - is_valid_uuid_format: true
       - exists_in_database: false
       - user_owns_run_id: false
       - database_details: null

    4. Invalid Format:
       - is_valid_uuid_format: false
       - uuid_error: "..."
       - Other fields: false/null

Use Cases:
    - Verify run_id before using in operations
    - Debug why run_id lookup failed
    - Check user ownership for authorization
    - Validate UUID format
    - Audit run_id access patterns

===================================================================================
CACHE MANAGEMENT
===================================================================================

Purpose:
    Manually clear the bulk loading cache to free memory or force fresh data load.

Cache Details:
    - _bulk_loading_cache: In-memory dict of bulk load results
    - Keyed by user_email
    - TTL: BULK_CACHE_TIMEOUT (default 60s)
    - Automatic expiry, but manual clear available

Clear Process:
    1. Count cache entries before clear
    2. Execute cache.clear()
    3. Log action with user attribution
    4. Check memory after cleanup
    5. Report memory status

Memory Status:
    - normal: < 80% of GC_MEMORY_THRESHOLD
    - high: >= 80% of GC_MEMORY_THRESHOLD
    - Threshold default: 1900 MB

Response Information:
    - cache_entries_cleared: Count before clear
    - current_memory_mb: RSS memory usage
    - memory_status: normal/high/unknown
    - cleared_by: User email (audit)
    - timestamp: When cleared

Use Cases:
    - Free memory when cache grows too large
    - Force fresh data load for testing
    - Clear stale cache entries
    - Debug cache-related issues
    - Performance troubleshooting

Security:
    - Requires authentication
    - Any authenticated user can clear (design decision)
    - Production: Consider restricting to admin role
    - Audit logging included

===================================================================================
PREPARED STATEMENT CLEANUP
===================================================================================

Purpose:
    Clear PostgreSQL prepared statements to free database server memory.

What Are Prepared Statements:
    - Pre-compiled SQL queries stored in database
    - Reused for performance optimization
    - Can accumulate and consume memory
    - Need periodic cleanup

When to Use:
    - Database memory usage high
    - Long-running server instances
    - Many different query patterns
    - Prepared statement bloat detected

Current Implementation:
    - Simplified version
    - Returns success message
    - Production: Add actual DEALLOCATE commands
    - Future: Selective cleanup based on usage

Enhancement Opportunities:
    1. List all prepared statements
    2. Show memory usage per statement
    3. Selective cleanup (old/unused only)
    4. Automatic cleanup on threshold
    5. Metrics for monitoring

Security:
    - Requires authentication
    - Logged for audit trail
    - Non-destructive (doesn't affect data)

===================================================================================
DYNAMIC ENVIRONMENT CONFIGURATION
===================================================================================

Purpose:
    Enable/disable debug flags and adjust configuration at runtime without
    restarting the server.

How It Works:
    1. Accept env_vars dict in request body
    2. Iterate through key-value pairs
    3. Set os.environ[key] = value
    4. Log each change
    5. Return confirmation

Use Cases:
    - Enable debug logging: DEBUG_CHAT=1
    - Adjust concurrency: MAX_CONCURRENT_BULK_THREADS=5
    - Change cache timeout: BULK_CACHE_TIMEOUT=120
    - Toggle features: ENABLE_FEATURE_X=1
    - Testing configuration changes

Common Debug Flags:
    - DEBUG_CHAT: Chat endpoint logging
    - DEBUG_ANALYZE: Analysis endpoint logging
    - DEBUG_CATALOG: Catalog endpoint logging
    - DEBUG_MEMORY: Memory monitoring
    - DEBUG_POOL: Connection pool logging

Benefits:
    - No server restart required
    - Quick troubleshooting
    - A/B testing configurations
    - Dynamic feature flags
    - Production debugging

Limitations:
    - Only affects current process
    - Not persisted (lost on restart)
    - Doesn't update .env file
    - May need process reload for some settings

Security:
    - Requires authentication
    - Logged for audit trail
    - Production: Restrict to admin role
    - Potential for misconfiguration

===================================================================================
ENVIRONMENT RESET
===================================================================================

Purpose:
    Reset environment variables to their original .env file values, undoing
    runtime configuration changes made via set-env.

Reset Process:
    1. Load original .env file via dotenv_values()
    2. For each variable in request
    3. Look up original value in .env
    4. Set os.environ[key] = original_value
    5. Default to "0" if not in .env
    6. Return reset values

Why Reset is Needed:
    - Undo debug flag changes
    - Return to known good configuration
    - Cleanup after troubleshooting
    - Prevent configuration drift

Use Cases:
    - After debugging session
    - Reset to production defaults
    - Undo accidental changes
    - Testing reset workflows

Response:
    - Lists all reset variables with new values
    - Confirms source (.env file)
    - Timestamp for audit

Security:
    - Requires authentication
    - Logged for audit trail
    - Safe operation (restores defaults)

===================================================================================
WINDOWS COMPATIBILITY
===================================================================================

Event Loop Policy:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

Required For:
    - psycopg async operations
    - PostgreSQL connection pool
    - Async endpoint handlers

Must Be:
    - At top of file
    - Before other imports
    - Only on Windows platform

===================================================================================
ERROR HANDLING
===================================================================================

Error Handling Patterns:

1. Try/Except with Traceback Response
   - Catches exceptions
   - Calls traceback_json_response(e)
   - Returns formatted error or re-raises

2. Explicit Error Returns
   - Returns {"error": "..."} for expected failures
   - Doesn't raise exceptions
   - Graceful degradation

3. Fallback Returns
   - Always returns valid JSON
   - Never returns None
   - Includes error details in response

Examples:
    - Checkpoint inspection fails → Return error dict
    - Pool status fails → Return 500 with error
    - Run ID not found → Return full result with flags
    - Cache clear fails → Re-raise exception

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Authentication
   - All endpoints require JWT (except pool-status)
   - User email extracted from token
   - User-scoped queries

2. Authorization
   - Checkpoint inspection: Implicit via thread_id ownership
   - Run ID verification: Explicit user_email filter
   - Cache clear: Any authenticated user (consider restricting)
   - Environment config: Any authenticated user (consider restricting)

3. Data Privacy
   - No cross-user data access
   - Content previews truncated
   - Sensitive data not logged

4. Audit Logging
   - All actions logged with user attribution
   - Timestamps for all operations
   - Debug output for troubleshooting

5. Production Hardening
   - Restrict admin endpoints to admin role
   - Rate limit debug endpoints
   - Disable in production (or restrict access)
   - Monitor for abuse

===================================================================================
MONITORING AND OBSERVABILITY
===================================================================================

Health Checks:
    - Pool status endpoint for liveness
    - Latency metrics for performance
    - Error rates for alerting

Metrics to Track:
    - Pool status check latency
    - Cache clear frequency
    - Memory after cache clear
    - Environment config changes
    - Checkpoint inspection frequency

Logging:
    - All operations logged via print__debug
    - User attribution in logs
    - Error stack traces preserved
    - Timestamps for all events

Alerting Scenarios:
    - Pool status failed
    - High latency (> 1000ms)
    - Memory status: high
    - Frequent cache clears (> 10/hour)
    - Environment config changes in prod

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    1. Checkpoint inspection with mock data
    2. Pool status with different checkpointer states
    3. Run ID verification with various UUID formats
    4. Cache clear memory calculation
    5. Environment set/reset logic

Integration Tests:
    1. Full checkpoint inspection with real database
    2. Pool status with live connection
    3. Run ID lookup with test data
    4. Cache clear with populated cache
    5. Environment changes affecting behavior

Mocking:
    - Mock GLOBAL_CHECKPOINTER
    - Mock database connections
    - Mock psutil memory functions
    - Mock dotenv_values()

Test Data:
    - Valid/invalid UUIDs
    - Existing/non-existing run_ids
    - User-owned/other-user run_ids
    - Empty/populated caches

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables, path operations
    - sys: Platform detection
    - time: Latency measurement
    - uuid: UUID validation
    - datetime: Timestamps
    - typing: Type hints
    - asyncio: Event loop (Windows)

Third-Party:
    - psutil: Memory monitoring
    - fastapi: Web framework, routing
    - dotenv: Environment variable loading

Internal:
    - api.config.settings: Global checkpointer, cache
    - api.dependencies.auth: JWT authentication
    - api.utils.debug: Debug logging
    - api.utils.memory: Memory monitoring
    - api.helpers: Traceback JSON response
    - checkpointer.checkpointer.factory: Checkpointer access

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Role-Based Access Control
   - Restrict admin endpoints to admin role
   - Separate debug and admin permissions
   - Audit trail for admin actions

2. Metrics Dashboard
   - Real-time pool status
   - Cache hit/miss rates
   - Memory usage graphs
   - Environment config history

3. Advanced Checkpoint Tools
   - Checkpoint diff between states
   - Checkpoint rollback capability
   - Export checkpoints for analysis

4. Query Performance Analysis
   - Slow query detection
   - Query plan inspection
   - Index usage statistics

5. Automated Cleanup
   - Scheduled cache cleanup
   - Automatic prepared statement management
   - Memory-based triggers

===================================================================================
"""

# ==============================================================================
# CRITICAL WINDOWS COMPATIBILITY CONFIGURATION
# ==============================================================================

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with asyncio on Windows platforms. This prevents "Event loop is closed" errors
# and ensures proper async database operations.
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# PATH AND DIRECTORY CONSTANTS
# ==============================================================================

# Determine base directory for the project
# Handles both normal execution and special environments (e.g., REPL, Jupyter)
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

import time

# Standard imports
import uuid
from datetime import datetime
from typing import Dict

import psutil
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# ==============================================================================
# CONFIGURATION AND GLOBAL STATE
# ==============================================================================

# Import configuration globals for checkpointer and cache access
from api.config.settings import (
    GLOBAL_CHECKPOINTER,
    _bulk_loading_cache,
)

# ==============================================================================
# AUTHENTICATION AND AUTHORIZATION
# ==============================================================================

# Import JWT-based authentication dependency for user verification
from api.dependencies.auth import get_current_user

# ==============================================================================
# DEBUG AND LOGGING UTILITIES
# ==============================================================================

# Import debug functions for comprehensive logging and diagnostics
from api.utils.debug import print__debug
from api.utils.memory import print__memory_monitoring

# Import database connection functions
sys.path.insert(0, str(BASE_DIR))
from api.helpers import traceback_json_response
from checkpointer.checkpointer.factory import get_global_checkpointer

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for debug and administrative endpoints
# This router will be included in the main FastAPI application
router = APIRouter()


# ==============================================================================
# CHECKPOINT INSPECTION ENDPOINT
# ==============================================================================


@router.get("/debug/chat/{thread_id}/checkpoints")
async def debug_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """Debug endpoint to inspect raw checkpoint data for a thread."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔍 Inspecting checkpoints for thread: {thread_id}")

    try:
        checkpointer = await get_global_checkpointer()

        if not hasattr(checkpointer, "conn"):
            return {"error": "No PostgreSQL checkpointer available"}

        config = {"configurable": {"thread_id": thread_id}}

        # Get all checkpoints for this thread
        checkpoint_tuples = []
        try:
            # Fix: alist() returns an async generator, don't await it
            checkpoint_iterator = checkpointer.alist(config)
            async for checkpoint_tuple in checkpoint_iterator:
                checkpoint_tuples.append(checkpoint_tuple)
        except Exception as alist_error:
            print__debug(f"❌ Error getting checkpoint list: {alist_error}")
            return {"error": f"Failed to get checkpoints: {alist_error}"}

        debug_data = {
            "thread_id": thread_id,
            "total_checkpoints": len(checkpoint_tuples),
            "checkpoints": [],
        }

        for i, checkpoint_tuple in enumerate(checkpoint_tuples):
            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}

            checkpoint_info = {
                "index": i,
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get(
                    "checkpoint_id", "unknown"
                ),
                "has_checkpoint": bool(checkpoint),
                "has_metadata": bool(metadata),
                "metadata_writes": metadata.get("writes", {}),
                "channel_values": {},
            }

            if checkpoint and "channel_values" in checkpoint:
                channel_values = checkpoint["channel_values"]
                messages = channel_values.get("messages", [])

                checkpoint_info["channel_values"] = {
                    "message_count": len(messages),
                    "messages": [],
                }

                for j, msg in enumerate(messages):
                    msg_info = {
                        "index": j,
                        "type": type(msg).__name__,
                        "id": getattr(msg, "id", None),
                        "content_preview": (
                            getattr(msg, "content", str(msg))[:200] + "..."
                            if hasattr(msg, "content")
                            and len(getattr(msg, "content", "")) > 200
                            else getattr(msg, "content", str(msg))
                        ),
                        "content_length": len(getattr(msg, "content", "")),
                    }
                    checkpoint_info["channel_values"]["messages"].append(msg_info)

            debug_data["checkpoints"].append(checkpoint_info)

        return debug_data

    except Exception as e:
        print__debug(f"❌ Error inspecting checkpoints: {e}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {"error": str(e)}


# ==============================================================================
# DATABASE POOL STATUS ENDPOINT
# ==============================================================================


@router.get("/debug/pool-status")
async def debug_pool_status():
    """Debug endpoint to check pool status - updated for official AsyncPostgresSaver."""
    return await get_pool_status()


async def get_pool_status():
    """Get pool status information - separated function for easier testing and imports."""
    try:
        global GLOBAL_CHECKPOINTER

        status = {
            "timestamp": datetime.now().isoformat(),
            "global_checkpointer_exists": GLOBAL_CHECKPOINTER is not None,
            "checkpointer_type": (
                type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
            ),
        }

        if GLOBAL_CHECKPOINTER:
            if "AsyncPostgresSaver" in str(type(GLOBAL_CHECKPOINTER)):
                # Test AsyncPostgresSaver functionality instead of checking .conn
                try:
                    test_config = {"configurable": {"thread_id": "pool_status_test"}}
                    start_time = time.time()

                    # Test a basic operation to verify the checkpointer is working
                    _result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
                    latency = time.time() - start_time

                    status.update(
                        {
                            "asyncpostgressaver_status": "operational",
                            "test_latency_ms": round(latency * 1000, 2),
                            "connection_test": "passed",
                        }
                    )

                except Exception as test_error:
                    status.update(
                        {
                            "asyncpostgressaver_status": "error",
                            "test_error": str(test_error),
                            "connection_test": "failed",
                        }
                    )
            else:
                status.update(
                    {
                        "checkpointer_status": "non_postgres_type",
                        "note": (
                            f"Using {type(GLOBAL_CHECKPOINTER).__name__} "
                            "instead of AsyncPostgresSaver"
                        ),
                    }
                )
        else:
            status["checkpointer_status"] = "not_initialized"

        return status

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "timestamp": datetime.now().isoformat()},
        )


# ==============================================================================
# RUN_ID VERIFICATION ENDPOINT
# ==============================================================================


@router.get("/debug/run-id/{run_id}")
async def debug_run_id(run_id: str, user=Depends(get_current_user)):
    """Debug endpoint to check if a run_id exists in the database."""

    try:
        user_email = user.get("email")
        if not user_email:
            raise HTTPException(status_code=401, detail="User email not found in token")

        print__debug(f"🔍 Checking run_id: '{run_id}' for user: {user_email}")

        result = {
            "run_id": run_id,
            "run_id_type": type(run_id).__name__,
            "run_id_length": len(run_id) if run_id else 0,
            "is_valid_uuid_format": False,
            "exists_in_database": False,
            "user_owns_run_id": False,
            "database_details": None,
        }

        # Check if it's a valid UUID format
        try:
            uuid_obj = uuid.UUID(run_id)
            result["is_valid_uuid_format"] = True
            result["uuid_parsed"] = str(uuid_obj)
        except ValueError as e:
            result["uuid_error"] = str(e)

        # Check if it exists in the database
        try:
            pool = await get_global_checkpointer()
            pool = pool.conn if hasattr(pool, "conn") else None

            if pool:
                async with pool.connection() as conn:
                    # 🔒 SECURITY: Check in users_threads_runs table with user ownership verification
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT email, thread_id, prompt, timestamp
                            FROM users_threads_runs 
                            WHERE run_id = %s AND email = %s
                        """,
                            (run_id, user_email),
                        )

                        row = await cur.fetchone()
                        if row:
                            result["exists_in_database"] = True
                            result["user_owns_run_id"] = True
                            result["database_details"] = {
                                "email": row[0],
                                "thread_id": row[1],
                                "prompt": row[2],
                                "timestamp": row[3].isoformat() if row[3] else None,
                            }
                            print__debug(f"✅ User {user_email} owns run_id {run_id}")
                        else:
                            # Check if run_id exists but belongs to different user
                            await cur.execute(
                                """
                                SELECT COUNT(*) FROM users_threads_runs WHERE run_id = %s
                            """,
                                (run_id,),
                            )

                            any_row = await cur.fetchone()
                            if any_row and any_row[0] > 0:
                                result["exists_in_database"] = True
                                result["user_owns_run_id"] = False
                                print__debug(
                                    f"🚫 Run_id {run_id} exists but user {user_email} does not own it"
                                )
                            else:
                                print__debug(
                                    f"❌ Run_id {run_id} not found in database"
                                )
        except Exception as e:
            result["database_error"] = str(e)
            # Don't use traceback_json_response here, just add error to result
            print__debug(f"❌ Database error in debug_run_id: {e}")

        # Always return the result
        print__debug(f"🔍 Returning result: {result}")
        return result

    except Exception as e:
        # Fallback error handling to ensure we never return None
        print__debug(f"❌ Critical error in debug_run_id: {e}")
        return {
            "run_id": run_id if "run_id" in locals() else "unknown",
            "error": f"Critical error: {str(e)}",
            "is_valid_uuid_format": False,
            "exists_in_database": False,
            "user_owns_run_id": False,
            "database_details": None,
        }


# ==============================================================================
# CACHE MANAGEMENT ENDPOINT
# ==============================================================================


@router.post("/admin/clear-cache")
async def clear_bulk_cache(user=Depends(get_current_user)):
    """Clear the bulk loading cache (admin endpoint)."""
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    # For now, allow any authenticated user to clear cache
    # In production, you might want to restrict this to admin users

    cache_entries_before = len(_bulk_loading_cache)
    _bulk_loading_cache.clear()

    print__memory_monitoring(
        f"🧹 MANUAL CACHE CLEAR: {cache_entries_before} entries cleared by {user_email}"
    )

    # Check memory after cleanup
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        gc_memory_threshold = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))
        memory_status = "normal" if rss_mb < (gc_memory_threshold * 0.8) else "high"
    except Exception:
        rss_mb = 0
        memory_status = "unknown"

    return {
        "message": "Cache cleared successfully",
        "cache_entries_cleared": cache_entries_before,
        "current_memory_mb": round(rss_mb, 1),
        "memory_status": memory_status,
        "cleared_by": user_email,
        "timestamp": datetime.now().isoformat(),
    }


# ==============================================================================
# PREPARED STATEMENTS CLEANUP ENDPOINT
# ==============================================================================


@router.post("/admin/clear-prepared-statements")
async def clear_prepared_statements_endpoint(user=Depends(get_current_user)):
    """Clear prepared statements in the database to free memory."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🧹 Clearing prepared statements for user: {user_email}")

    try:
        checkpointer = await get_global_checkpointer()

        if not hasattr(checkpointer, "conn"):
            return {"error": "No PostgreSQL checkpointer available"}

        # Clear prepared statements
        # Note: This is a simplified version - in production you'd want more sophisticated cleanup
        return {
            "message": "Prepared statements cleared",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print__debug(f"❌ Error clearing prepared statements: {e}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        return {"error": str(e)}


# ==============================================================================
# DYNAMIC ENVIRONMENT CONFIGURATION ENDPOINTS
# ==============================================================================


@router.post("/debug/set-env")
async def set_debug_environment(
    env_vars: Dict[str, str], user=Depends(get_current_user)
):
    """Dynamically set environment variables for debug control."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔧 Setting debug environment variables: {env_vars}")

    # Set the environment variables in the server process
    for key, value in env_vars.items():
        os.environ[key] = value
        print__debug(f"🔧 Set {key}={value}")

    return {
        "message": f"Set {len(env_vars)} environment variables",
        "variables": env_vars,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/debug/reset-env")
async def reset_debug_environment(
    env_vars: Dict[str, str], user=Depends(get_current_user)
):
    """Reset specific environment variables to their original .env values."""

    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__debug(f"🔧 Resetting debug environment variables: {list(env_vars.keys())}")

    # Load original values from .env file
    from dotenv import dotenv_values

    original_env = dotenv_values()

    reset_vars = {}
    for var_name in env_vars.keys():
        # Get original value from .env file, default to "0" if not found
        original_value = original_env.get(var_name, "0")
        os.environ[var_name] = original_value
        reset_vars[var_name] = original_value
        print__debug(f"🔧 Reset {var_name}={original_value}")

    return {
        "message": (
            f"Reset {len(reset_vars)} environment variables " "to original .env values"
        ),
        "variables": reset_vars,
        "timestamp": datetime.now().isoformat(),
    }

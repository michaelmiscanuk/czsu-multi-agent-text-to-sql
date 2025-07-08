# # ============================================================
# # CRITICAL INITIALIZATION
# # ============================================================
# # CRITICAL: Set Windows event loop policy FIRST, before any other imports
# # This must be the very first thing that happens to fix psycopg compatibility
# import os  # Import os early for environment variable access
# import sys
# import traceback


# def print__startup_debug(msg: str) -> None:
#     """Print startup debug messages when debug mode is enabled."""
#     debug_mode = os.environ.get("DEBUG", "0")
#     if debug_mode == "1":
#         print(f"[STARTUP-DEBUG] {msg}")
#         sys.stdout.flush()


# if sys.platform == "win32":
#     import asyncio

#     # AGGRESSIVE WINDOWS FIX: Force SelectorEventLoop before any other async operations
#     print__startup_debug(
#         f"🔧 API Server: Windows detected - forcing SelectorEventLoop for PostgreSQL compatibility"
#     )

#     # Set the policy first - this is CRITICAL and must happen before any async operations
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     print__startup_debug(
#         f"🔧 Windows event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}"
#     )

#     # Force close any existing event loop and create a fresh SelectorEventLoop
#     try:
#         current_loop = asyncio.get_event_loop()
#         if current_loop and not current_loop.is_closed():
#             current_loop_type = type(current_loop).__name__
#             print__startup_debug(f"🔧 API Server: Closing existing {current_loop_type}")
#             current_loop.close()
#     except RuntimeError:
#         # No event loop exists yet, which is fine
#         pass

#     # Create a new SelectorEventLoop explicitly and set it as the running loop
#     new_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
#     asyncio.set_event_loop(new_loop)
#     print__startup_debug(f"🔧 API Server: Created new {type(new_loop).__name__}")

#     # Verify the fix worked - this is critical for PostgreSQL compatibility
#     try:
#         current_loop = asyncio.get_event_loop()
#         current_loop_type = type(current_loop).__name__
#         print__startup_debug(
#             f"🔧 API Server: Current event loop type: {current_loop_type}"
#         )
#         if "Selector" in current_loop_type:
#             print__startup_debug(
#                 f"✅ API Server: PostgreSQL should work correctly on Windows now"
#             )
#         else:
#             print__startup_debug(
#                 f"⚠️ API Server: Event loop fix may not have worked properly"
#             )
#             # FORCE FIX: If we still don't have a SelectorEventLoop, create one
#             print__startup_debug(f"🔧 API Server: Force-creating SelectorEventLoop...")
#             if not current_loop.is_closed():
#                 current_loop.close()
#             selector_loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
#             asyncio.set_event_loop(selector_loop)
#             print__startup_debug(
#                 f"🔧 API Server: Force-created {type(selector_loop).__name__}"
#             )
#     except RuntimeError:
#         print__startup_debug(
#             f"🔧 API Server: No event loop set yet (will be created as needed)"
#         )


# # ============================================================
# # IMPORTS
# # ============================================================

# import asyncio
# import gc
# import json
# import os
# import signal
# import sqlite3
# import time
# import uuid
# from collections import defaultdict
# from contextlib import asynccontextmanager
# from datetime import datetime, timedelta, timezone
# from pathlib import Path
# from typing import Dict, List, Optional

# import jwt
# import psutil
# import requests
# from dotenv import load_dotenv
# from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
# from fastapi.exceptions import RequestValidationError
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware
# from fastapi.responses import JSONResponse, ORJSONResponse
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from jwt.algorithms import RSAAlgorithm
# from langchain_core.messages import BaseMessage
# from langsmith import Client
# from pydantic import BaseModel, Field, field_validator
# from starlette.exceptions import HTTPException as StarletteHTTPException

# # Load environment variables from .env file EARLY
# load_dotenv()

# # Read InMemorySaver fallback configuration from environment
# INMEMORY_FALLBACK_ENABLED = os.environ.get("InMemorySaver_fallback", "1") == "1"
# print__startup_debug(
#     f"🔧 API Server: InMemorySaver fallback {'ENABLED' if INMEMORY_FALLBACK_ENABLED else 'DISABLED'} (from environment)"
# )

# # Constants
# try:
#     BASE_DIR = Path(__file__).resolve().parents[2]
# except NameError:
#     BASE_DIR = Path(os.getcwd()).parents[0]

# from main import main as analysis_main

# # Additional imports for sentiment functionality
# from my_agent.utils.postgres_checkpointer import (
#     get_user_chat_threads_count,  # Add this import
# )
# from my_agent.utils.postgres_checkpointer import (
#     cleanup_checkpointer,
#     create_thread_run_entry,
#     delete_user_thread_entries,
#     get_conversation_messages_from_checkpoints,
#     get_direct_connection,
#     get_healthy_checkpointer,
#     get_queries_and_results_from_latest_checkpoint,
#     get_thread_run_sentiments,
#     get_user_chat_threads,
#     initialize_checkpointer,
#     update_thread_run_sentiment,
# )


# def print__memory_monitoring(msg: str) -> None:
#     """Print MEMORY-MONITORING messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("DEBUG", "0")
#     if debug_mode == "1":
#         print(f"[MEMORY-MONITORING] {msg}")
#         import sys

#         sys.stdout.flush()


# # ============================================================
# # CONFIGURATION AND CONSTANTS
# # ============================================================
# # Application startup time for uptime tracking
# start_time = time.time()

# # Read GC memory threshold from environment with default fallback
# GC_MEMORY_THRESHOLD = int(
#     os.environ.get("GC_MEMORY_THRESHOLD", "1900")
# )  # 1900MB for 2GB memory allocation
# print__startup_debug(
#     f"🔧 API Server: GC_MEMORY_THRESHOLD set to {GC_MEMORY_THRESHOLD}MB (from environment)"
# )
# # MEMORY LEAK PREVENTION: Simplified global tracking
# _app_startup_time = None
# _memory_baseline = None  # RSS memory at startup
# _request_count = 0  # Track total requests processed

# # Global shared checkpointer for conversation memory across API requests
# # This ensures that conversation state is preserved between frontend requests using PostgreSQL
# GLOBAL_CHECKPOINTER = None

# # Add a semaphore to limit concurrent analysis requests
# MAX_CONCURRENT_ANALYSES = int(
#     os.environ.get("MAX_CONCURRENT_ANALYSES", "3")
# )  # Read from .env with fallback to 3
# analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# # Log the concurrency setting for debugging
# print__startup_debug(
#     f"🔧 API Server: MAX_CONCURRENT_ANALYSES set to {MAX_CONCURRENT_ANALYSES} (from environment)"
# )
# print__memory_monitoring(
#     f"🔒 Concurrent analysis semaphore initialized with {MAX_CONCURRENT_ANALYSES} slots"
# )

# # RATE LIMITING: Global rate limiting storage
# rate_limit_storage = defaultdict(list)
# RATE_LIMIT_REQUESTS = 100  # requests per window
# RATE_LIMIT_WINDOW = 60  # 60 seconds window
# RATE_LIMIT_BURST = 20  # burst limit for rapid requests
# RATE_LIMIT_MAX_WAIT = 5  # maximum seconds to wait before giving up

# # Throttling semaphores per IP to limit concurrent requests
# throttle_semaphores = defaultdict(
#     lambda: asyncio.Semaphore(8)
# )  # Max 8 concurrent requests per IP

# # Global cache for bulk loading to prevent repeated calls
# _bulk_loading_cache = {}
# _bulk_loading_locks = defaultdict(asyncio.Lock)
# BULK_CACHE_TIMEOUT = 30  # Cache timeout in seconds

# GOOGLE_JWK_URL = "https://www.googleapis.com/oauth2/v3/certs"

# # Global counter for tracking JWT 'kid' missing events to reduce log spam
# _jwt_kid_missing_count = 0


# # ============================================================
# # UTILITY FUNCTIONS - MEMORY MANAGEMENT
# # ============================================================
# def cleanup_bulk_cache():
#     """Clean up expired cache entries."""
#     current_time = time.time()
#     expired_keys = []

#     for cache_key, (cached_data, cache_time) in _bulk_loading_cache.items():
#         if current_time - cache_time > BULK_CACHE_TIMEOUT:
#             expired_keys.append(cache_key)

#     for key in expired_keys:
#         del _bulk_loading_cache[key]

#     return len(expired_keys)


# def check_memory_and_gc():
#     """Enhanced memory check with cache cleanup and scaling strategy."""
#     try:
#         process = psutil.Process()
#         memory_info = process.memory_info()
#         rss_mb = memory_info.rss / 1024 / 1024

#         # Clean up cache first if memory is getting high
#         if rss_mb > (GC_MEMORY_THRESHOLD * 0.8):  # At 80% of threshold
#             print__memory_monitoring(
#                 f"📊 Memory at {rss_mb:.1f}MB (80% of {GC_MEMORY_THRESHOLD}MB threshold) - cleaning cache"
#             )
#             cleaned_entries = cleanup_bulk_cache()
#             if cleaned_entries > 0:
#                 # Check memory after cache cleanup
#                 new_memory = psutil.Process().memory_info().rss / 1024 / 1024
#                 freed = rss_mb - new_memory
#                 print__memory_monitoring(
#                     f"🧹 Cache cleanup freed {freed:.1f}MB, cleaned {cleaned_entries} entries"
#                 )
#                 rss_mb = new_memory

#         # Trigger GC only if above threshold
#         if rss_mb > GC_MEMORY_THRESHOLD:
#             print__memory_monitoring(
#                 f"🚨 MEMORY THRESHOLD EXCEEDED: {rss_mb:.1f}MB > {GC_MEMORY_THRESHOLD}MB - running GC"
#             )
#             import gc

#             collected = gc.collect()
#             print__memory_monitoring(f"🧹 GC collected {collected} objects")

#             # Log memory after GC
#             new_memory = psutil.Process().memory_info().rss / 1024 / 1024
#             freed = rss_mb - new_memory
#             print__memory_monitoring(
#                 f"🧹 Memory after GC: {new_memory:.1f}MB (freed: {freed:.1f}MB)"
#             )

#             # If memory is still high after GC, provide scaling guidance
#             if new_memory > (GC_MEMORY_THRESHOLD * 0.9):
#                 thread_count = len(_bulk_loading_cache)
#                 print__memory_monitoring(
#                     f"⚠ HIGH MEMORY WARNING: {new_memory:.1f}MB after GC"
#                 )
#                 print__memory_monitoring(f"📊 Current cache entries: {thread_count}")
#                 if thread_count > 20:
#                     print__memory_monitoring(
#                         f"💡 SCALING TIP: Consider implementing pagination for chat threads"
#                     )

#         return rss_mb

#     except Exception as e:
#         print__memory_monitoring(f"❌ Could not check memory: {e}")
#         return 0


# def log_memory_usage(context: str = ""):
#     """Simplified memory logging."""
#     try:
#         process = psutil.Process()
#         rss_mb = process.memory_info().rss / 1024 / 1024

#         print__memory_monitoring(
#             f"📊 Memory usage{f' [{context}]' if context else ''}: {rss_mb:.1f}MB RSS"
#         )

#         # Simple threshold check
#         if rss_mb > GC_MEMORY_THRESHOLD:
#             check_memory_and_gc()

#     except Exception as e:
#         print__memory_monitoring(f"❌ Could not check memory usage: {e}")


# def log_comprehensive_error(context: str, error: Exception, request: Request = None):
#     """Simplified error logging."""
#     error_details = {
#         "context": context,
#         "error_type": type(error).__name__,
#         "error_message": str(error),
#         "timestamp": datetime.now().isoformat(),
#     }

#     if request:
#         error_details.update(
#             {
#                 "method": request.method,
#                 "url": str(request.url),
#                 "client_ip": request.client.host if request.client else "unknown",
#             }
#         )

#     # Log to debug output
#     print__debug(f"🚨 ERROR: {json.dumps(error_details, indent=2)}")


# def check_rate_limit_with_throttling(client_ip: str) -> dict:
#     """Check rate limits and return throttling information instead of boolean."""
#     now = time.time()

#     # Clean old entries
#     rate_limit_storage[client_ip] = [
#         timestamp
#         for timestamp in rate_limit_storage[client_ip]
#         if now - timestamp < RATE_LIMIT_WINDOW
#     ]

#     # Check burst limit (last 10 seconds)
#     recent_requests = [
#         timestamp for timestamp in rate_limit_storage[client_ip] if now - timestamp < 10
#     ]

#     # Check window limit
#     window_requests = len(rate_limit_storage[client_ip])

#     # Calculate suggested wait time based on current load
#     suggested_wait = 0

#     if len(recent_requests) >= RATE_LIMIT_BURST:
#         # Burst limit exceeded - calculate wait time until oldest burst request expires
#         oldest_burst = min(recent_requests)
#         suggested_wait = max(0, 10 - (now - oldest_burst))
#     elif window_requests >= RATE_LIMIT_REQUESTS:
#         # Window limit exceeded - calculate wait time until oldest request expires
#         oldest_window = min(rate_limit_storage[client_ip])
#         suggested_wait = max(0, RATE_LIMIT_WINDOW - (now - oldest_window))

#     return {
#         "allowed": len(recent_requests) < RATE_LIMIT_BURST
#         and window_requests < RATE_LIMIT_REQUESTS,
#         "suggested_wait": min(suggested_wait, RATE_LIMIT_MAX_WAIT),
#         "burst_count": len(recent_requests),
#         "window_count": window_requests,
#         "burst_limit": RATE_LIMIT_BURST,
#         "window_limit": RATE_LIMIT_REQUESTS,
#     }


# async def wait_for_rate_limit(client_ip: str) -> bool:
#     """Wait for rate limit to allow request, with maximum wait time."""
#     max_attempts = 3

#     for attempt in range(max_attempts):
#         rate_info = check_rate_limit_with_throttling(client_ip)

#         if rate_info["allowed"]:
#             # Add current request to tracking
#             rate_limit_storage[client_ip].append(time.time())
#             return True

#         if rate_info["suggested_wait"] <= 0:
#             # Should be allowed but isn't - might be a race condition
#             await asyncio.sleep(0.1)
#             continue

#         if rate_info["suggested_wait"] > RATE_LIMIT_MAX_WAIT:
#             # Wait time too long, give up
#             print__debug(
#                 f"⚠️ Rate limit wait time ({rate_info['suggested_wait']:.1f}s) exceeds maximum ({RATE_LIMIT_MAX_WAIT}s) for {client_ip}"
#             )
#             return False

#         # Wait for the suggested time
#         print__debug(
#             f"⏳ Throttling request from {client_ip}: waiting {rate_info['suggested_wait']:.1f}s (burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, window: {rate_info['window_count']}/{rate_info['window_limit']}, attempt {attempt + 1})"
#         )
#         await asyncio.sleep(rate_info["suggested_wait"])

#     print__debug(
#         f"❌ Rate limit exceeded after {max_attempts} attempts for {client_ip}"
#     )
#     return False


# def check_rate_limit(client_ip: str) -> bool:
#     """Check if client IP is within rate limits."""
#     now = time.time()

#     # Clean old entries
#     rate_limit_storage[client_ip] = [
#         timestamp
#         for timestamp in rate_limit_storage[client_ip]
#         if now - timestamp < RATE_LIMIT_WINDOW
#     ]

#     # Check burst limit (last 10 seconds)
#     recent_requests = [
#         timestamp for timestamp in rate_limit_storage[client_ip] if now - timestamp < 10
#     ]

#     if len(recent_requests) >= RATE_LIMIT_BURST:
#         return False

#     # Check window limit
#     if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
#         return False

#     # Add current request
#     rate_limit_storage[client_ip].append(now)
#     return True


# def setup_graceful_shutdown():
#     """Setup graceful shutdown handlers."""

#     def signal_handler(signum, frame):
#         print__memory_monitoring(
#             f"📡 Received signal {signum} - preparing for graceful shutdown..."
#         )
#         log_memory_usage("shutdown_signal")

#     # Register signal handlers for common restart signals
#     signal.signal(signal.SIGTERM, signal_handler)  # Most common for container restarts
#     signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
#     if hasattr(signal, "SIGUSR1"):
#         signal.signal(signal.SIGUSR1, signal_handler)  # User-defined signal


# async def perform_deletion_operations(conn, user_email: str, thread_id: str):
#     """Perform the actual deletion operations on the given connection."""
#     print__api_postgresql(f"🔧 DEBUG: Starting deletion operations...")

#     print__api_postgresql(f"🔧 DEBUG: Setting autocommit...")
#     await conn.set_autocommit(True)
#     print__api_postgresql(f"🔧 DEBUG: Autocommit set successfully")

#     # 🔒 SECURITY CHECK: Verify user owns this thread before deleting
#     print__api_postgresql(
#         f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}"
#     )

#     print__api_postgresql(f"🔧 DEBUG: Creating cursor for ownership check...")
#     async with conn.cursor() as cur:
#         print__api_postgresql(f"🔧 DEBUG: Cursor created, executing ownership query...")
#         # Fix: Use correct psycopg approach with fetchone() instead of fetchval()
#         await cur.execute(
#             """
#             SELECT COUNT(*) FROM users_threads_runs
#             WHERE email = %s AND thread_id = %s
#         """,
#             (user_email, thread_id),
#         )

#         # Get the result row and extract the count value
#         result_row = await cur.fetchone()
#         # Fix: psycopg Row objects don't support [0] indexing, convert to tuple first
#         thread_entries_count = tuple(result_row)[0] if result_row else 0
#         print__api_postgresql(
#             f"🔧 DEBUG: Ownership check complete, count: {thread_entries_count}"
#         )

#     if thread_entries_count == 0:
#         print__api_postgresql(
#             f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - deletion denied"
#         )
#         return {
#             "message": "Thread not found or access denied",
#             "thread_id": thread_id,
#             "user_email": user_email,
#             "deleted_counts": {},
#         }

#     print__api_postgresql(
#         f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - deletion authorized"
#     )

#     print__api_postgresql(f"🔄 Deleting from checkpoint tables for thread {thread_id}")

#     # Delete from all checkpoint tables
#     tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
#     deleted_counts = {}

#     for table in tables:
#         try:
#             print__api_postgresql(f"🔧 DEBUG: Processing table {table}...")
#             # First check if the table exists
#             print__api_postgresql(
#                 f"🔧 DEBUG: Creating cursor for table existence check..."
#             )
#             async with conn.cursor() as cur:
#                 print__api_postgresql(
#                     f"🔧 DEBUG: Executing table existence query for {table}..."
#                 )
#                 # Fix: Use correct psycopg approach with fetchone() instead of fetchval()
#                 await cur.execute(
#                     """
#                     SELECT EXISTS (
#                         SELECT FROM information_schema.tables
#                         WHERE table_name = %s
#                     )
#                 """,
#                     (table,),
#                 )

#                 # Get the result row and extract the boolean value
#                 result_row = await cur.fetchone()
#                 # Fix: psycopg Row objects don't support [0] indexing, convert to tuple first
#                 table_exists = tuple(result_row)[0] if result_row else False
#                 print__api_postgresql(
#                     f"🔧 DEBUG: Table {table} exists check result: {table_exists}"
#                 )

#                 # Simple boolean check
#                 if not table_exists:
#                     print__api_postgresql(f"⚠ Table {table} does not exist, skipping")
#                     deleted_counts[table] = 0
#                     continue

#                 print__api_postgresql(
#                     f"🔧 DEBUG: Table {table} exists, proceeding with deletion..."
#                 )
#                 # Delete records for this thread_id
#                 print__api_postgresql(
#                     f"🔧 DEBUG: Creating cursor for deletion from {table}..."
#                 )
#                 async with conn.cursor() as del_cur:
#                     print__api_postgresql(
#                         f"🔧 DEBUG: Executing DELETE query for {table}..."
#                     )
#                     await del_cur.execute(
#                         f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
#                     )

#                     deleted_counts[table] = (
#                         del_cur.rowcount if hasattr(del_cur, "rowcount") else 0
#                     )
#                     print__api_postgresql(
#                         f"✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}"
#                     )

#         except Exception as table_error:
#             print__api_postgresql(f"⚠ Error deleting from table {table}: {table_error}")
#             print__api_postgresql(
#                 f"🔧 DEBUG: Table error type: {type(table_error).__name__}"
#             )
#             print__api_postgresql(
#                 f"🔧 DEBUG: Table error traceback: {traceback.format_exc()}"
#             )
#             deleted_counts[table] = f"Error: {str(table_error)}"

#     # Delete from users_threads_runs table directly within the same transaction
#     print__api_postgresql(
#         f"🔄 Deleting thread entries from users_threads_runs for user {user_email}, thread {thread_id}"
#     )
#     try:
#         print__api_postgresql(
#             "🔧 DEBUG: Creating cursor for users_threads_runs deletion..."
#         )
#         async with conn.cursor() as cur:
#             print__api_postgresql(
#                 f"🔧 DEBUG: Executing DELETE query for users_threads_runs..."
#             )
#             await cur.execute(
#                 """
#                 DELETE FROM users_threads_runs
#                 WHERE email = %s AND thread_id = %s
#             """,
#                 (user_email, thread_id),
#             )

#             users_threads_runs_deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
#             print__api_postgresql(
#                 f"✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}"
#             )

#             deleted_counts["users_threads_runs"] = users_threads_runs_deleted

#     except Exception as e:
#         print__api_postgresql(f"❌ Error deleting from users_threads_runs: {e}")
#         print__api_postgresql(
#             f"🔧 DEBUG: users_threads_runs error type: {type(e).__name__}"
#         )
#         print__api_postgresql(
#             f"🔧 DEBUG: users_threads_runs error traceback: {traceback.format_exc()}"
#         )
#         deleted_counts["users_threads_runs"] = f"Error: {str(e)}"

#     result_data = {
#         "message": f"Checkpoint records and thread entries deleted for thread_id: {thread_id}",
#         "deleted_counts": deleted_counts,
#         "thread_id": thread_id,
#         "user_email": user_email,
#     }

#     print__api_postgresql(
#         f"🎉 Successfully deleted thread {thread_id} for user {user_email}"
#     )
#     return result_data


# # ==============================================================================
# # DEBUG FUNCTIONS
# # ==============================================================================
# def print__api_postgresql(msg: str) -> None:
#     """Print API-PostgreSQL messages when debug mode is enabled."""
#     debug_mode = os.environ.get("print__api_postgresql", "0")
#     if debug_mode == "1":
#         print(f"[print__api_postgresql] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__feedback_flow(msg: str) -> None:
#     """Print FEEDBACK-FLOW messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("DEBUG", "0")
#     if debug_mode == "1":
#         print(f"[FEEDBACK-FLOW] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__token_debug(msg: str) -> None:
#     """Print print__token_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__token_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__token_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__sentiment_flow(msg: str) -> None:
#     """Print SENTIMENT-FLOW messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("DEBUG", "0")
#     if debug_mode == "1":
#         print(f"[SENTIMENT-FLOW] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__debug(msg: str) -> None:
#     """Print DEBUG messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("DEBUG", "0")
#     if debug_mode == "1":
#         print(f"[DEBUG] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__analyze_debug(msg: str) -> None:
#     """Print print__analyze_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__analyze_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__analyze_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_all_messages_debug(msg: str) -> None:
#     """Print print__chat_all_messages_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_all_messages_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_all_messages_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__feedback_debug(msg: str) -> None:
#     """Print print__feedback_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__feedback_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__feedback_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__sentiment_debug(msg: str) -> None:
#     """Print print__sentiment_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__sentiment_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__sentiment_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_threads_debug(msg: str) -> None:
#     """Print print__chat_threads_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_threads_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_threads_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_messages_debug(msg: str) -> None:
#     """Print print__chat_messages_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_messages_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_messages_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__delete_chat_debug(msg: str) -> None:
#     """Print print__delete_chat_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__delete_chat_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__delete_chat_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_sentiments_debug(msg: str) -> None:
#     """Print print__chat_sentiments_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_sentiments_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_sentiments_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__catalog_debug(msg: str) -> None:
#     """Print print__catalog_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__catalog_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__catalog_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__data_tables_debug(msg: str) -> None:
#     """Print print__data_tables_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__data_tables_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__data_tables_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__data_table_debug(msg: str) -> None:
#     """Print print__data_table_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__data_table_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__data_table_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_thread_id_checkpoints_debug(msg: str) -> None:
#     """Print print__chat_thread_id_checkpoints_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_thread_id_checkpoints_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_thread_id_checkpoints_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__debug_pool_status_debug(msg: str) -> None:
#     """Print print__debug_pool_status_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__debug_pool_status_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__debug_pool_status_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__chat_thread_id_run_ids_debug(msg: str) -> None:
#     """Print print__chat_thread_id_run_ids_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__chat_thread_id_run_ids_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__chat_thread_id_run_ids_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__debug_run_id_debug(msg: str) -> None:
#     """Print print__debug_run_id_debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     debug_mode = os.environ.get("print__debug_run_id_debug", "0")
#     if debug_mode == "1":
#         print(f"[print__debug_run_id_debug] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__admin_clear_cache_debug(msg: str) -> None:
#     """Print admin clear cache debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     admin_clear_cache_debug_mode = os.environ.get("ADMIN_CLEAR_CACHE_DEBUG", "0")
#     if admin_clear_cache_debug_mode == "1":
#         print(f"[ADMIN_CLEAR_CACHE_DEBUG] {msg}")
#         import sys

#         sys.stdout.flush()


# def print__analysis_tracing_debug(msg: str) -> None:
#     """Print analysis tracing debug messages when debug mode is enabled.

#     Args:
#         msg: The message to print
#     """
#     analysis_tracing_debug_mode = os.environ.get("print__analysis_tracing_debug", "0")
#     if analysis_tracing_debug_mode == "1":
#         print(f"[print__analysis_tracing_debug] 🔍 {msg}")
#         import sys

#         sys.stdout.flush()


# # ============================================================
# # APPLICATION SETUP - LIFESPAN AND MIDDLEWARE
# # ============================================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     global _app_startup_time, _memory_baseline
#     _app_startup_time = datetime.now()

#     print__startup_debug("🚀 FastAPI application starting up...")
#     print__memory_monitoring(
#         f"Application startup initiated at {_app_startup_time.isoformat()}"
#     )
#     log_memory_usage("app_startup")

#     # ROUTE REGISTRATION MONITORING: Track all routes that get registered
#     # This prevents the exact issue described in the "Needle in a haystack" article
#     print__memory_monitoring(
#         "🔍 Monitoring route registrations to prevent memory leaks..."
#     )

#     # Setup graceful shutdown handlers
#     setup_graceful_shutdown()

#     await initialize_checkpointer()

#     # Set memory baseline after initialization
#     if _memory_baseline is None:
#         try:
#             process = psutil.Process()
#             _memory_baseline = process.memory_info().rss / 1024 / 1024
#             print__memory_monitoring(
#                 f"Memory baseline established: {_memory_baseline:.1f}MB RSS"
#             )
#         except:
#             pass

#     log_memory_usage("app_ready")
#     print__startup_debug("✅ FastAPI application ready to serve requests")

#     yield

#     # Shutdown
#     print__startup_debug("🛑 FastAPI application shutting down...")
#     print__memory_monitoring(
#         f"Application ran for {datetime.now() - _app_startup_time}"
#     )

#     # Log final memory statistics
#     if _memory_baseline:
#         try:
#             process = psutil.Process()
#             final_memory = process.memory_info().rss / 1024 / 1024
#             total_growth = final_memory - _memory_baseline
#             print__memory_monitoring(
#                 f"Final memory stats: Started={_memory_baseline:.1f}MB, "
#                 f"Final={final_memory:.1f}MB, Growth={total_growth:.1f}MB"
#             )
#             if (
#                 total_growth > GC_MEMORY_THRESHOLD
#             ):  # More than threshold growth - app will restart soon
#                 print__memory_monitoring(
#                     "🚨 SIGNIFICANT MEMORY GROWTH DETECTED - investigate for leaks!"
#                 )
#         except:
#             pass

#     await cleanup_checkpointer()


# # CRITICAL: Route registration happens here ONCE during startup
# # This is the key fix from the "Needle in a haystack" article
# app = FastAPI(
#     lifespan=lifespan,
#     default_response_class=ORJSONResponse,  # Use ORJSON for faster, memory-efficient JSON responses
# )

# # Monitor all route registrations (including middleware and CORS)
# print__memory_monitoring("📋 Registering CORS middleware...")
# # Note: Route registration monitoring happens at runtime to avoid import-time global variable access

# # Allow CORS for local frontend dev
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# print__memory_monitoring("📋 Registering GZip middleware...")
# # Add GZip compression to reduce response sizes and memory usage
# app.add_middleware(GZipMiddleware, minimum_size=1000)


# # RATE LIMITING MIDDLEWARE
# @app.middleware("http")
# async def throttling_middleware(request: Request, call_next):
#     """Throttling middleware that makes requests wait instead of rejecting them."""

#     # Skip throttling for health checks and static endpoints
#     if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
#         return await call_next(request)

#     client_ip = request.client.host if request.client else "unknown"

#     # Use semaphore to limit concurrent requests per IP
#     semaphore = throttle_semaphores[client_ip]

#     async with semaphore:
#         # Try to wait for rate limit instead of immediately rejecting
#         if not await wait_for_rate_limit(client_ip):
#             # Only reject if we can't wait (wait time too long or max attempts exceeded)
#             rate_info = check_rate_limit_with_throttling(client_ip)
#             log_comprehensive_error(
#                 "rate_limit_exceeded_after_wait",
#                 Exception(
#                     f"Rate limit exceeded for IP: {client_ip} after waiting. Burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, Window: {rate_info['window_count']}/{rate_info['window_limit']}"
#                 ),
#                 request,
#             )
#             return JSONResponse(
#                 status_code=429,
#                 content={
#                     "detail": f"Rate limit exceeded. Please wait {rate_info['suggested_wait']:.1f}s before retrying.",
#                     "retry_after": max(rate_info["suggested_wait"], 1),
#                     "burst_usage": f"{rate_info['burst_count']}/{rate_info['burst_limit']}",
#                     "window_usage": f"{rate_info['window_count']}/{rate_info['window_limit']}",
#                 },
#                 headers={"Retry-After": str(max(int(rate_info["suggested_wait"]), 1))},
#             )

#         return await call_next(request)


# # Enhanced middleware to monitor memory patterns and detect leaks
# @app.middleware("http")
# async def simplified_memory_monitoring_middleware(request: Request, call_next):
#     """Simplified memory monitoring middleware."""
#     global _request_count

#     _request_count += 1

#     # Only check memory for heavy operations
#     request_path = request.url.path
#     if any(path in request_path for path in ["/analyze", "/chat/all-messages-for-all-threads"]):
#         log_memory_usage(f"before_{request_path.replace('/', '_')}")

#     response = await call_next(request)

#     # Check memory after heavy operations
#     if any(path in request_path for path in ["/analyze", "/chat/all-messages-for-all-threads"]):
#         log_memory_usage(f"after_{request_path.replace('/', '_')}")

#     return response


# # ============================================================
# # EXCEPTION HANDLERS
# # ============================================================
# # Global exception handlers for proper error handling
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     """Handle Pydantic validation errors with proper 422 status code."""
#     print__debug(f"Validation error: {exc.errors()}")
#     return JSONResponse(
#         status_code=422, content={"detail": "Validation error", "errors": exc.errors()}
#     )


# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request: Request, exc: StarletteHTTPException):
#     """Handle HTTP exceptions with comprehensive debugging for 401 errors."""

#     # Enhanced debugging for 401 errors since these are authentication-related
#     if exc.status_code == 401:
#         print__analyze_debug(f"🚨 HTTP 401 UNAUTHORIZED: {exc.detail}")
#         print__analysis_tracing_debug(f"🚨 HTTP 401 TRACE: Request URL: {request.url}")
#         print__analysis_tracing_debug(
#             f"🚨 HTTP 401 TRACE: Request method: {request.method}"
#         )
#         print__analysis_tracing_debug(
#             f"🚨 HTTP 401 TRACE: Request headers: {dict(request.headers)}"
#         )
#         print__analysis_tracing_debug(
#             f"🚨 HTTP 401 TRACE: Exception detail: {exc.detail}"
#         )
#         print__analysis_tracing_debug(
#             f"🚨 HTTP 401 TRACE: Full traceback:\n{traceback.format_exc()}"
#         )

#         # Log client IP for debugging
#         client_ip = request.client.host if request.client else "unknown"
#         print__analyze_debug(f"🚨 HTTP 401 CLIENT: IP address: {client_ip}")

#     # Debug prints for other HTTP exceptions
#     elif exc.status_code >= 400:
#         print__analyze_debug(f"🚨 HTTP {exc.status_code} ERROR: {exc.detail}")
#         print__analysis_tracing_debug(
#             f"🚨 HTTP {exc.status_code} TRACE: Request URL: {request.url}"
#         )
#         print__analysis_tracing_debug(
#             f"🚨 HTTP {exc.status_code} TRACE: Request method: {request.method}"
#         )
#         print__analysis_tracing_debug(
#             f"🚨 HTTP {exc.status_code} TRACE: Full traceback:\n{traceback.format_exc()}"
#         )

#     return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# @app.exception_handler(ValueError)
# async def value_error_handler(request: Request, exc: ValueError):
#     """Handle ValueError exceptions as 400 Bad Request."""
#     print__debug(f"ValueError: {str(exc)}")
#     return JSONResponse(status_code=400, content={"detail": str(exc)})


# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     """Handle unexpected exceptions."""
#     print__debug(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
#     return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# # ============================================================
# # PYDANTIC MODELS
# # ============================================================
# class AnalyzeRequest(BaseModel):
#     prompt: str = Field(
#         ..., min_length=1, max_length=10000, description="The prompt to analyze"
#     )
#     thread_id: str = Field(
#         ..., min_length=1, max_length=100, description="The thread ID"
#     )

#     @field_validator("prompt")
#     @classmethod
#     def validate_prompt(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Prompt cannot be empty or only whitespace")
#         return v.strip()

#     @field_validator("thread_id")
#     @classmethod
#     def validate_thread_id(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Thread ID cannot be empty or only whitespace")
#         return v.strip()


# class FeedbackRequest(BaseModel):
#     run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
#     feedback: Optional[int] = Field(
#         None,
#         ge=0,
#         le=1,
#         description="Feedback score: 1 for thumbs up, 0 for thumbs down",
#     )
#     comment: Optional[str] = Field(
#         None, max_length=1000, description="Optional comment"
#     )

#     @field_validator("run_id")
#     @classmethod
#     def validate_run_id(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Run ID cannot be empty")
#         # Basic UUID format validation
#         import uuid

#         try:
#             uuid.UUID(v.strip())
#         except ValueError:
#             raise ValueError("Run ID must be a valid UUID format")
#         return v.strip()

#     @field_validator("comment")
#     @classmethod
#     def validate_comment(cls, v):
#         if v is not None and len(v.strip()) == 0:
#             return None  # Convert empty string to None
#         return v


# class SentimentRequest(BaseModel):
#     run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
#     sentiment: Optional[bool] = Field(
#         None,
#         description="Sentiment: true for positive, false for negative, null to clear",
#     )

#     @field_validator("run_id")
#     @classmethod
#     def validate_run_id(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Run ID cannot be empty")
#         # Basic UUID format validation
#         import uuid

#         try:
#             uuid.UUID(v.strip())
#         except ValueError:
#             raise ValueError("Run ID must be a valid UUID format")
#         return v.strip()


# class ChatThreadResponse(BaseModel):
#     thread_id: str
#     latest_timestamp: datetime  # Changed from str to datetime
#     run_count: int
#     title: str  # Now includes the title from first prompt
#     full_prompt: str  # Full prompt text for tooltip


# class PaginatedChatThreadsResponse(BaseModel):
#     threads: List[ChatThreadResponse]
#     total_count: int
#     page: int
#     limit: int
#     has_more: bool


# class ChatMessage(BaseModel):
#     id: str
#     threadId: str
#     user: str
#     content: str
#     isUser: bool
#     createdAt: int
#     error: Optional[str] = None
#     meta: Optional[dict] = None
#     queriesAndResults: Optional[List[List[str]]] = None
#     isLoading: Optional[bool] = None
#     startedAt: Optional[int] = None
#     isError: Optional[bool] = None


# # ============================================================
# # AUTHENTICATION
# # ============================================================
# # FIXED: Enhanced JWT verification with NextAuth.js id_token support
# def verify_google_jwt(token: str):
#     global _jwt_kid_missing_count

#     try:
#         # EARLY VALIDATION: Check if token has basic JWT structure before processing
#         # JWT tokens must have exactly 3 parts separated by dots (header.payload.signature)
#         token_parts = token.split(".")
#         if len(token_parts) != 3:
#             # Don't log this as it's a common case for invalid tokens in tests
#             raise HTTPException(status_code=401, detail="Invalid JWT token format")

#         # Additional basic validation - each part should be non-empty and base64-like
#         for i, part in enumerate(token_parts):
#             if (
#                 not part or len(part) < 4
#             ):  # Base64 encoded parts should be at least 4 chars
#                 raise HTTPException(status_code=401, detail="Invalid JWT token format")

#         # Get unverified header and payload first - this should now work since we pre-validated the format
#         try:
#             unverified_header = jwt.get_unverified_header(token)
#             unverified_payload = jwt.decode(token, options={"verify_signature": False})
#         except jwt.DecodeError as e:
#             # This should be rare now due to pre-validation, but keep for edge cases
#             print__token_debug(f"JWT decode error after pre-validation: {e}")
#             raise HTTPException(status_code=401, detail="Invalid JWT token format")
#         except Exception as e:
#             print__token_debug(f"JWT header decode error: {e}")
#             raise HTTPException(status_code=401, detail="Invalid JWT token format")

#         # Debug: print the audience in the token and the expected audience
#         print__token_debug(f"Token aud: {unverified_payload.get('aud')}")
#         print__token_debug(f"Backend GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID')}")
#         print__token_debug(f"Token iss: {unverified_payload.get('iss')}")

#         # TEST MODE: Handle test tokens with test issuer (for development/testing only)

#         print__token_debug(
#             f"🔍 ENV DEBUG: USE_TEST_TOKENS = '{os.getenv('USE_TEST_TOKENS', 'NOT_SET')}'"
#         )
#         # Only enabled when USE_TEST_TOKENS environment variable is set to "1"
#         use_test_tokens = os.getenv("USE_TEST_TOKENS", "0") == "1"
#         if use_test_tokens and unverified_payload.get("iss") == "test_issuer":
#             print__token_debug(
#                 "🧪 TEST MODE: Detected test token with test issuer - skipping Google verification"
#             )
#             print__token_debug(
#                 f"🧪 TEST MODE: USE_TEST_TOKENS={os.getenv('USE_TEST_TOKENS', '0')} - test tokens enabled"
#             )

#             # Verify the audience still matches
#             expected_aud = os.getenv("GOOGLE_CLIENT_ID")
#             if unverified_payload.get("aud") != expected_aud:
#                 print__token_debug(
#                     f"Test token audience mismatch. Expected: {expected_aud}, Got: {unverified_payload.get('aud')}"
#                 )
#                 raise HTTPException(
#                     status_code=401, detail="Invalid test token audience"
#                 )

#             # Check expiration
#             if int(unverified_payload.get("exp", 0)) < time.time():
#                 print__token_debug("Test token has expired")
#                 raise HTTPException(status_code=401, detail="Test token has expired")

#             # Return the test payload directly (no Google verification needed)
#             print__token_debug("✅ TEST MODE: Test token validation successful")
#             return unverified_payload
#         elif unverified_payload.get("iss") == "test_issuer":
#             # Test token detected but test mode is disabled
#             print__token_debug(
#                 f"🚫 TEST MODE DISABLED: Test token detected but USE_TEST_TOKENS={os.getenv('USE_TEST_TOKENS', '0')} - rejecting token"
#             )
#             raise HTTPException(
#                 status_code=401,
#                 detail="Test tokens are not allowed in this environment",
#             )

#         # NEW: Check if this is a NextAuth.js id_token (missing 'kid' field)
#         if "kid" not in unverified_header:
#             # Reduce log noise - only log this every 10th occurrence
#             _jwt_kid_missing_count += 1
#             if _jwt_kid_missing_count % 10 == 1:  # Log 1st, 11th, 21st, etc.
#                 print__token_debug(
#                     f"JWT token missing 'kid' field (#{_jwt_kid_missing_count}) - attempting NextAuth.js id_token verification"
#                 )

#             # NEXTAUTH.JS SUPPORT: Verify id_token directly using Google's tokeninfo endpoint
#             try:
#                 print__token_debug(
#                     "Attempting NextAuth.js id_token verification via Google tokeninfo endpoint"
#                 )

#                 # Use Google's tokeninfo endpoint to verify the id_token
#                 tokeninfo_url = (
#                     f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
#                 )
#                 response = requests.get(tokeninfo_url, timeout=10)

#                 if response.status_code == 200:
#                     tokeninfo = response.json()
#                     print__token_debug(f"Google tokeninfo response: {tokeninfo}")

#                     # Verify the audience matches our client ID
#                     expected_aud = os.getenv("GOOGLE_CLIENT_ID")
#                     if tokeninfo.get("aud") != expected_aud:
#                         print__token_debug(
#                             f"Tokeninfo audience mismatch. Expected: {expected_aud}, Got: {tokeninfo.get('aud')}"
#                         )
#                         raise HTTPException(
#                             status_code=401, detail="Invalid token audience"
#                         )

#                     # Verify the token is not expired
#                     if int(tokeninfo.get("exp", 0)) < time.time():
#                         print__token_debug("Tokeninfo shows token has expired")
#                         raise HTTPException(status_code=401, detail="Token has expired")

#                     # Return the tokeninfo as the payload (it contains email, name, etc.)
#                     print__token_debug(
#                         "NextAuth.js id_token verification successful via Google tokeninfo"
#                     )
#                     return tokeninfo

#                 else:
#                     print__token_debug(
#                         f"Google tokeninfo endpoint returned error: {response.status_code} - {response.text}"
#                     )
#                     raise HTTPException(
#                         status_code=401, detail="Invalid NextAuth.js id_token"
#                     )

#             except requests.RequestException as e:
#                 print__token_debug(
#                     f"Failed to verify NextAuth.js id_token via Google tokeninfo: {e}"
#                 )
#                 raise HTTPException(
#                     status_code=401,
#                     detail="Token verification failed - unable to validate NextAuth.js token",
#                 )
#             except HTTPException:
#                 raise  # Re-raise HTTPException as-is
#             except Exception as e:
#                 print__token_debug(f"NextAuth.js id_token verification failed: {e}")
#                 raise HTTPException(
#                     status_code=401, detail="NextAuth.js token verification failed"
#                 )

#         # ORIGINAL FLOW: Standard Google JWT token with 'kid' field (for direct Google API calls)
#         try:
#             # Get Google public keys for JWKS verification
#             jwks = requests.get(GOOGLE_JWK_URL).json()
#         except requests.RequestException as e:
#             print__token_debug(f"Failed to fetch Google JWKS: {e}")
#             raise HTTPException(
#                 status_code=401,
#                 detail="Token verification failed - unable to fetch Google keys",
#             )

#         # Find matching key
#         for key in jwks["keys"]:
#             if key["kid"] == unverified_header["kid"]:
#                 public_key = RSAAlgorithm.from_jwk(key)
#                 try:
#                     payload = jwt.decode(
#                         token,
#                         public_key,
#                         algorithms=["RS256"],
#                         audience=os.getenv("GOOGLE_CLIENT_ID"),
#                     )
#                     print__token_debug(
#                         "Standard Google JWT token verification successful"
#                     )
#                     return payload
#                 except jwt.ExpiredSignatureError:
#                     print__token_debug("JWT token has expired")
#                     raise HTTPException(status_code=401, detail="Token has expired")
#                 except jwt.InvalidAudienceError:
#                     print__token_debug("JWT token has invalid audience")
#                     raise HTTPException(
#                         status_code=401, detail="Invalid token audience"
#                     )
#                 except jwt.InvalidSignatureError:
#                     print__token_debug("JWT token has invalid signature")
#                     raise HTTPException(
#                         status_code=401, detail="Invalid token signature"
#                     )
#                 except jwt.DecodeError as e:
#                     print__token_debug(f"JWT decode error: {e}")
#                     raise HTTPException(status_code=401, detail="Invalid token format")
#                 except jwt.InvalidTokenError as e:
#                     print__token_debug(f"JWT token is invalid: {e}")
#                     raise HTTPException(status_code=401, detail="Invalid token")
#                 except Exception as e:
#                     print__token_debug(f"JWT decode error: {e}")
#                     raise HTTPException(status_code=401, detail="Invalid token")

#         print__token_debug("JWT public key not found in Google JWKS")
#         raise HTTPException(
#             status_code=401, detail="Invalid token: public key not found"
#         )

#     except HTTPException:
#         raise  # Re-raise HTTPException as-is
#     except requests.RequestException as e:
#         print__token_debug(f"Failed to fetch Google JWKS: {e}")
#         raise HTTPException(
#             status_code=401, detail="Token verification failed - unable to validate"
#         )
#     except jwt.DecodeError as e:
#         # This should be rare now due to pre-validation
#         print__token_debug(f"JWT decode error in main handler: {e}")
#         raise HTTPException(status_code=401, detail="Invalid JWT token format")
#     except KeyError as e:
#         print__token_debug(f"JWT verification KeyError: {e}")
#         raise HTTPException(status_code=401, detail="Invalid JWT token structure")
#     except Exception as e:
#         print__token_debug(f"JWT verification failed: {e}")
#         raise HTTPException(status_code=401, detail="Token verification failed")


# # Enhanced dependency for JWT authentication with better error handling
# def get_current_user(authorization: str = Header(None)):
#     try:
#         # Add debug prints using the user's enabled environment variables
#         print__token_debug(
#             "🔑 AUTHENTICATION START: Beginning user authentication process"
#         )
#         print__token_debug(
#             "🔑 AUTH TRACE: get_current_user called with authorization header"
#         )

#         if not authorization:
#             print__token_debug("❌ AUTH ERROR: No authorization header provided")
#             print__token_debug(
#                 "❌ AUTH TRACE: Missing Authorization header - raising 401"
#             )
#             raise HTTPException(status_code=401, detail="Missing Authorization header")

#         print__token_debug(
#             f"🔍 AUTH CHECK: Authorization header present (length: {len(authorization)})"
#         )
#         print__token_debug(
#             f"🔍 AUTH TRACE: Authorization header format check - starts with 'Bearer ': {authorization.startswith('Bearer ')}"
#         )

#         if not authorization.startswith("Bearer "):
#             print__token_debug("❌ AUTH ERROR: Invalid authorization header format")
#             print__token_debug(
#                 "❌ AUTH TRACE: Invalid Authorization header format - raising 401"
#             )
#             raise HTTPException(
#                 status_code=401,
#                 detail="Invalid Authorization header format. Expected 'Bearer <token>'",
#             )

#         # Split and validate token extraction
#         auth_parts = authorization.split(" ", 1)
#         if len(auth_parts) != 2 or not auth_parts[1].strip():
#             print__token_debug("❌ AUTH ERROR: Malformed authorization header")
#             print__token_debug(
#                 f"❌ AUTH TRACE: Authorization header split failed - parts: {len(auth_parts)}"
#             )
#             raise HTTPException(
#                 status_code=401, detail="Invalid Authorization header format"
#             )

#         token = auth_parts[1].strip()
#         print__token_debug(
#             f"🔍 AUTH TOKEN: Token extracted successfully (length: {len(token)})"
#         )
#         print__token_debug(
#             f"🔍 AUTH TRACE: Token validation starting with verify_google_jwt"
#         )

#         # Call JWT verification with debug tracing
#         user_info = verify_google_jwt(token)
#         print__token_debug(
#             f"✅ AUTH SUCCESS: User authenticated successfully - {user_info.get('email', 'Unknown')}"
#         )
#         print__token_debug(
#             f"✅ AUTH TRACE: verify_google_jwt returned user info: {user_info}"
#         )

#         return user_info

#     except HTTPException as he:
#         # Re-raise HTTPException with enhanced debugging
#         print__token_debug(f"❌ AUTH HTTP EXCEPTION: {he.status_code} - {he.detail}")
#         print__token_debug(
#             f"❌ AUTH TRACE: HTTPException caught - status: {he.status_code}, detail: {he.detail}"
#         )
#         print__token_debug(
#             f"❌ AUTH TRACE: Full HTTPException traceback:\n{traceback.format_exc()}"
#         )
#         raise  # Re-raise HTTPException as-is
#     except Exception as e:
#         # Enhanced error handling with full traceback
#         print__token_debug(
#             f"❌ AUTH EXCEPTION: Unexpected authentication error - {type(e).__name__}: {str(e)}"
#         )
#         print__token_debug(f"❌ AUTH TRACE: Unexpected exception in authentication")
#         print__token_debug(f"❌ AUTH TRACE: Full traceback:\n{traceback.format_exc()}")

#         print__token_debug(f"Authentication error: {e}")
#         log_comprehensive_error("authentication", e)
#         raise HTTPException(status_code=401, detail="Authentication failed")


# # ============================================================
# # ROUTE REGISTRATION MONITORING
# # ============================================================
# # ROUTE REGISTRATION MONITORING: Track all endpoint registrations to prevent memory leaks
# # This directly addresses the core issue from the "Needle in a haystack" article
# print__memory_monitoring(
#     "📋 Monitoring route registrations for memory leak prevention..."
# )

# # Track all main routes that will be registered
# main_routes = [
#     ("/health", "GET"),
#     ("/health/database", "GET"),
#     ("/health/memory", "GET"),
#     ("/health/rate-limits", "GET"),
#     ("/health/prepared-statements", "GET"),
#     ("/admin/clear-prepared-statements", "POST"),
#     ("/analyze", "POST"),
#     ("/feedback", "POST"),
#     ("/sentiment", "POST"),
#     ("/chat/{thread_id}/sentiments", "GET"),
#     ("/chat-threads", "GET"),
#     ("/chat/{thread_id}", "DELETE"),
#     ("/catalog", "GET"),
#     ("/data-tables", "GET"),
#     ("/data-table", "GET"),
#     ("/chat/{thread_id}/messages", "GET"),
#     ("/chat/all-messages-for-all-threads", "GET"),
#     ("/debug/chat/{thread_id}/checkpoints", "GET"),
#     ("/debug/pool-status", "GET"),
#     ("/chat/{thread_id}/run-ids", "GET"),
#     ("/debug/run-id/{run_id}", "GET"),
# ]

# # Route monitoring is performed at runtime through middleware to ensure proper global variable access
# # for route_path, method in main_routes:
# #     monitor_route_registration(route_path, method)

# print__memory_monitoring(
#     f"📋 Route monitoring configured for {len(main_routes)} endpoints - tracking occurs at runtime"
# )


# # ============================================================
# # HEALTH CHECK ENDPOINTS
# # ============================================================
# @app.get("/health")
# async def health_check():
#     """Enhanced health check with memory monitoring and database verification."""
#     try:
#         # Memory check
#         import psutil

#         process = psutil.Process()
#         memory_info = process.memory_info()
#         memory_percent = process.memory_percent()

#         # Database check with proper AsyncPostgresSaver handling
#         database_healthy = True
#         database_error = None
#         checkpointer_type = "Unknown"

#         try:
#             if GLOBAL_CHECKPOINTER:
#                 checkpointer_type = type(GLOBAL_CHECKPOINTER).__name__

#                 if "AsyncPostgresSaver" in checkpointer_type:
#                     # Test AsyncPostgresSaver with a simple operation
#                     test_config = {"configurable": {"thread_id": "health_check_test"}}

#                     # Use aget_tuple() which is a basic read operation
#                     result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
#                     # If we get here without exception, the database is healthy
#                     database_healthy = True
#                 else:
#                     # For other checkpointer types (like MemorySaver)
#                     database_healthy = True

#         except Exception as e:
#             database_healthy = False
#             database_error = str(e)

#         # Response
#         status = "healthy" if database_healthy else "degraded"

#         health_data = {
#             "status": status,
#             "timestamp": datetime.now().isoformat(),
#             "uptime_seconds": time.time() - start_time,
#             "memory": {
#                 "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
#                 "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
#                 "percent": round(memory_percent, 2),
#             },
#             "database": {
#                 "healthy": database_healthy,
#                 "checkpointer_type": checkpointer_type,
#                 "error": database_error,
#             },
#             "version": "1.0.0",
#         }

#         if not database_healthy:
#             return JSONResponse(status_code=503, content=health_data)

#         return health_data

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat(),
#             },
#         )


# @app.get("/health/database")
# async def database_health_check():
#     """Detailed database health check."""
#     try:
#         health_status = {
#             "timestamp": datetime.now().isoformat(),
#             "checkpointer_available": GLOBAL_CHECKPOINTER is not None,
#             "checkpointer_type": (
#                 type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
#             ),
#         }

#         if GLOBAL_CHECKPOINTER and "AsyncPostgresSaver" in str(
#             type(GLOBAL_CHECKPOINTER)
#         ):
#             # Test AsyncPostgresSaver functionality
#             try:
#                 test_config = {"configurable": {"thread_id": "db_health_test"}}

#                 # Test basic read operation
#                 start_time = time.time()
#                 result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
#                 read_latency = time.time() - start_time

#                 health_status.update(
#                     {
#                         "database_connection": "healthy",
#                         "read_latency_ms": round(read_latency * 1000, 2),
#                         "read_test": "passed",
#                     }
#                 )

#             except Exception as e:
#                 health_status.update(
#                     {
#                         "database_connection": "error",
#                         "error": str(e),
#                         "read_test": "failed",
#                     }
#                 )
#                 return JSONResponse(status_code=503, content=health_status)
#         else:
#             health_status.update(
#                 {
#                     "database_connection": "using_memory_fallback",
#                     "note": "PostgreSQL checkpointer not available",
#                 }
#             )

#         return health_status

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "timestamp": datetime.now().isoformat(),
#                 "database_connection": "error",
#                 "error": str(e),
#             },
#         )


# @app.get("/health/memory")
# async def memory_health_check():
#     """Enhanced memory-specific health check with cache information."""
#     try:
#         process = psutil.Process()
#         rss_mb = process.memory_info().rss / 1024 / 1024

#         # Clean up expired cache entries
#         cleaned_entries = cleanup_bulk_cache()

#         status = "healthy"
#         if rss_mb > GC_MEMORY_THRESHOLD:
#             status = "high_memory"
#         elif rss_mb > (GC_MEMORY_THRESHOLD * 0.8):
#             status = "warning"

#         cache_info = {
#             "active_cache_entries": len(_bulk_loading_cache),
#             "cleaned_expired_entries": cleaned_entries,
#             "cache_timeout_seconds": BULK_CACHE_TIMEOUT,
#         }

#         # Calculate estimated memory per thread for scaling guidance
#         thread_count = len(_bulk_loading_cache)
#         memory_per_thread = rss_mb / max(thread_count, 1) if thread_count > 0 else 0
#         estimated_max_threads = (
#             int(GC_MEMORY_THRESHOLD / max(memory_per_thread, 38))
#             if memory_per_thread > 0
#             else 50
#         )

#         return {
#             "status": status,
#             "memory_rss_mb": round(rss_mb, 1),
#             "memory_threshold_mb": GC_MEMORY_THRESHOLD,
#             "memory_usage_percent": round((rss_mb / GC_MEMORY_THRESHOLD) * 100, 1),
#             "over_threshold": rss_mb > GC_MEMORY_THRESHOLD,
#             "total_requests_processed": _request_count,
#             "cache_info": cache_info,
#             "scaling_info": {
#                 "estimated_memory_per_thread_mb": round(memory_per_thread, 1),
#                 "estimated_max_threads_at_threshold": estimated_max_threads,
#                 "current_thread_count": thread_count,
#             },
#             "timestamp": datetime.now().isoformat(),
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat(),
#         }


# @app.get("/health/rate-limits")
# async def rate_limit_health_check():
#     """Rate limiting health check."""
#     try:
#         total_clients = len(rate_limit_storage)
#         active_clients = sum(1 for requests in rate_limit_storage.values() if requests)

#         return {
#             "status": "healthy",
#             "total_tracked_clients": total_clients,
#             "active_clients": active_clients,
#             "rate_limit_window": RATE_LIMIT_WINDOW,
#             "rate_limit_requests": RATE_LIMIT_REQUESTS,
#             "rate_limit_burst": RATE_LIMIT_BURST,
#             "timestamp": datetime.now().isoformat(),
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat(),
#         }


# @app.get("/health/prepared-statements")
# async def prepared_statements_health_check():
#     """Health check for prepared statements and database connection status."""
#     try:
#         from my_agent.utils.postgres_checkpointer import (
#             clear_prepared_statements,
#             get_global_checkpointer,
#         )

#         # Check if we can get a checkpointer
#         try:
#             checkpointer = await get_global_checkpointer()
#             checkpointer_status = "healthy" if checkpointer else "unavailable"
#         except Exception as e:
#             checkpointer_status = f"error: {str(e)}"

#         # Check prepared statements in the database
#         try:
#             import psycopg

#             from my_agent.utils.postgres_checkpointer import (
#                 get_connection_kwargs,
#                 get_db_config,
#             )

#             config = get_db_config()
#             # Create connection string without prepared statement parameters
#             connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require"

#             # Get connection kwargs for disabling prepared statements
#             connection_kwargs = get_connection_kwargs()

#             async with await psycopg.AsyncConnection.connect(
#                 connection_string, **connection_kwargs
#             ) as conn:
#                 async with conn.cursor() as cur:
#                     await cur.execute(
#                         """
#                         SELECT COUNT(*) as count,
#                                STRING_AGG(name, ', ') as statement_names
#                         FROM pg_prepared_statements
#                         WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
#                     """
#                     )
#                     result = await cur.fetchone()

#                     # Fix: Handle psycopg Row object properly - check if it exists and has data
#                     prepared_count = result[0] if result else 0
#                     statement_names = (
#                         result[1]
#                         if result and len(result) > 1 and result[1]
#                         else "none"
#                     )

#                     return {
#                         "status": "healthy",
#                         "checkpointer_status": checkpointer_status,
#                         "prepared_statements_count": prepared_count,
#                         "prepared_statement_names": statement_names,
#                         "connection_kwargs": connection_kwargs,
#                         "timestamp": datetime.now().isoformat(),
#                     }

#         except Exception as db_error:
#             return {
#                 "status": "degraded",
#                 "checkpointer_status": checkpointer_status,
#                 "database_error": str(db_error),
#                 "timestamp": datetime.now().isoformat(),
#             }

#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat(),
#         }


# # ============================================================
# # MAIN API ENDPOINTS - ANALYSIS
# # ============================================================
# @app.post("/analyze")
# async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
#     """Analyze request with simplified memory monitoring."""

#     print__analysis_tracing_debug("01 - ANALYZE ENDPOINT ENTRY: Request received")
#     print__analyze_debug(f"🔍 ANALYZE ENDPOINT - ENTRY POINT")
#     print__analyze_debug(f"🔍 Request received: thread_id={request.thread_id}")
#     print__analyze_debug(f"🔍 Request prompt length: {len(request.prompt)}")

#     try:
#         print__analysis_tracing_debug(
#             "02 - USER EXTRACTION: Getting user email from token"
#         )
#         user_email = user.get("email")
#         print__analyze_debug(f"🔍 User extraction: {user_email}")
#         if not user_email:
#             print__analysis_tracing_debug("03 - ERROR: No user email found in token")
#             print__analyze_debug(f"🚨 No user email found in token")
#             raise HTTPException(status_code=401, detail="User email not found in token")

#         print__analysis_tracing_debug(
#             f"04 - USER VALIDATION SUCCESS: User {user_email} validated"
#         )
#         print__feedback_flow(
#             f"📝 New analysis request - Thread: {request.thread_id}, User: {user_email}"
#         )
#         print__analyze_debug(
#             f"🔍 ANALYZE REQUEST RECEIVED: thread_id={request.thread_id}, user={user_email}"
#         )

#         print__analysis_tracing_debug("05 - MEMORY MONITORING: Starting memory logging")
#         # Simple memory check
#         print__analyze_debug(f"🔍 Starting memory logging")
#         log_memory_usage("analysis_start")
#         run_id = None

#         print__analysis_tracing_debug(
#             "06 - SEMAPHORE ACQUISITION: Attempting to acquire analysis semaphore"
#         )
#         print__analyze_debug(f"🔍 About to acquire analysis semaphore")
#         # Limit concurrent analyses to prevent resource exhaustion
#         async with analysis_semaphore:
#             print__analysis_tracing_debug(
#                 f"07 - SEMAPHORE ACQUIRED: Analysis semaphore acquired ({analysis_semaphore._value}/{MAX_CONCURRENT_ANALYSES} available)"
#             )
#             print__feedback_flow(
#                 f"🔒 Acquired analysis semaphore ({analysis_semaphore._value}/{MAX_CONCURRENT_ANALYSES} available)"
#             )
#             print__analyze_debug(f"🔍 Semaphore acquired successfully")

#             try:
#                 print__analysis_tracing_debug(
#                     "08 - CHECKPOINTER INITIALIZATION: Getting healthy checkpointer"
#                 )
#                 print__analyze_debug(f"🔍 About to get healthy checkpointer")
#                 print__feedback_flow(f"🔄 Getting healthy checkpointer")
#                 checkpointer = await get_healthy_checkpointer()
#                 print__analysis_tracing_debug(
#                     f"09 - CHECKPOINTER SUCCESS: Checkpointer obtained ({type(checkpointer).__name__})"
#                 )
#                 print__analyze_debug(
#                     f"🔍 Checkpointer obtained: {type(checkpointer).__name__}"
#                 )

#                 print__analysis_tracing_debug(
#                     "10 - THREAD RUN ENTRY: Creating thread run entry in database"
#                 )
#                 print__analyze_debug(f"🔍 About to create thread run entry")
#                 print__feedback_flow(f"🔄 Creating thread run entry")
#                 run_id = await create_thread_run_entry(
#                     user_email, request.thread_id, request.prompt
#                 )
#                 print__analysis_tracing_debug(
#                     f"11 - THREAD RUN SUCCESS: Thread run entry created with run_id {run_id}"
#                 )
#                 print__feedback_flow(f"✅ Generated new run_id: {run_id}")
#                 print__analyze_debug(
#                     f"🔍 Thread run entry created successfully: {run_id}"
#                 )

#                 print__analysis_tracing_debug(
#                     "12 - ANALYSIS MAIN START: Starting analysis_main function"
#                 )
#                 print__analyze_debug(f"🔍 About to start analysis_main")
#                 print__feedback_flow(f"🚀 Starting analysis")
#                 # 8 minute timeout for platform stability
#                 result = await asyncio.wait_for(
#                     analysis_main(
#                         request.prompt,
#                         thread_id=request.thread_id,
#                         checkpointer=checkpointer,
#                         run_id=run_id,
#                     ),
#                     timeout=480,  # 8 minutes timeout
#                 )

#                 print__analysis_tracing_debug(
#                     "13 - ANALYSIS MAIN SUCCESS: Analysis completed successfully"
#                 )
#                 print__analyze_debug(f"🔍 Analysis completed successfully")
#                 print__feedback_flow(f"✅ Analysis completed successfully")

#             except Exception as analysis_error:
#                 print__analysis_tracing_debug(
#                     f"14 - ANALYSIS ERROR: Exception in analysis block - {type(analysis_error).__name__}"
#                 )
#                 print__analyze_debug(
#                     f"🚨 Exception in analysis block: {type(analysis_error).__name__}: {str(analysis_error)}"
#                 )
#                 # If there's a database connection issue, try with InMemorySaver as fallback
#                 error_msg = str(analysis_error).lower()
#                 print__analyze_debug(f"🔍 Error message (lowercase): {error_msg}")

#                 # ENHANCED: Check for prepared statement errors specifically
#                 is_prepared_stmt_error = any(
#                     indicator in error_msg
#                     for indicator in [
#                         "prepared statement",
#                         "does not exist",
#                         "_pg3_",
#                         "_pg_",
#                         "invalidsqlstatementname",
#                     ]
#                 )

#                 if is_prepared_stmt_error:
#                     print__analysis_tracing_debug(
#                         "15 - PREPARED STATEMENT ERROR: Prepared statement error detected"
#                     )
#                     print__analyze_debug(
#                         f"🔧 PREPARED STATEMENT ERROR DETECTED: {analysis_error}"
#                     )
#                     print__feedback_flow(
#                         f"🔧 Prepared statement error detected - this should be handled by retry logic: {analysis_error}"
#                     )
#                     # Re-raise prepared statement errors - they should be handled by the retry decorator
#                     raise HTTPException(
#                         status_code=500,
#                         detail="Database prepared statement error. Please try again.",
#                     )
#                 elif any(
#                     keyword in error_msg
#                     for keyword in [
#                         "pool",
#                         "connection",
#                         "closed",
#                         "timeout",
#                         "ssl",
#                         "postgres",
#                     ]
#                 ):
#                     print__analysis_tracing_debug(
#                         "16 - DATABASE FALLBACK: Database issue detected, attempting fallback"
#                     )
#                     print__analyze_debug(
#                         f"🔍 Database issue detected, attempting fallback"
#                     )
#                     print__feedback_flow(
#                         f"⚠️ Database issue detected, trying with InMemorySaver fallback: {analysis_error}"
#                     )

#                     # Check if InMemorySaver fallback is enabled
#                     if not INMEMORY_FALLBACK_ENABLED:
#                         print__analysis_tracing_debug(
#                             "17 - FALLBACK DISABLED: InMemorySaver fallback is disabled by configuration"
#                         )
#                         print__analyze_debug(
#                             f"🚫 InMemorySaver fallback is DISABLED by configuration - re-raising database error"
#                         )
#                         print__feedback_flow(
#                             f"🚫 InMemorySaver fallback disabled - propagating database error: {analysis_error}"
#                         )
#                         raise HTTPException(
#                             status_code=500,
#                             detail="Database connection error. Please try again.",
#                         )

#                     try:
#                         print__analysis_tracing_debug(
#                             "17 - FALLBACK INITIALIZATION: Importing InMemorySaver"
#                         )
#                         print__analyze_debug(f"🔍 Importing InMemorySaver")
#                         from langgraph.checkpoint.memory import InMemorySaver

#                         fallback_checkpointer = InMemorySaver()
#                         print__analysis_tracing_debug(
#                             "18 - FALLBACK CHECKPOINTER: InMemorySaver created"
#                         )
#                         print__analyze_debug(f"🔍 InMemorySaver created")

#                         # Generate a fallback run_id since database creation might have failed
#                         if run_id is None:
#                             run_id = str(uuid.uuid4())
#                             print__analysis_tracing_debug(
#                                 f"19 - FALLBACK RUN ID: Generated fallback run_id {run_id}"
#                             )
#                             print__feedback_flow(
#                                 f"✅ Generated fallback run_id: {run_id}"
#                             )
#                             print__analyze_debug(
#                                 f"🔍 Generated fallback run_id: {run_id}"
#                             )

#                         print__analysis_tracing_debug(
#                             "20 - FALLBACK ANALYSIS: Starting fallback analysis"
#                         )
#                         print__analyze_debug(f"🔍 Starting fallback analysis")
#                         print__feedback_flow(
#                             f"🚀 Starting analysis with InMemorySaver fallback"
#                         )
#                         result = await asyncio.wait_for(
#                             analysis_main(
#                                 request.prompt,
#                                 thread_id=request.thread_id,
#                                 checkpointer=fallback_checkpointer,
#                                 run_id=run_id,
#                             ),
#                             timeout=480,  # 8 minutes timeout
#                         )

#                         print__analysis_tracing_debug(
#                             "21 - FALLBACK SUCCESS: Fallback analysis completed successfully"
#                         )
#                         print__analyze_debug(
#                             f"🔍 Fallback analysis completed successfully"
#                         )
#                         print__feedback_flow(
#                             f"✅ Analysis completed successfully with fallback"
#                         )

#                     except Exception as fallback_error:
#                         print__analysis_tracing_debug(
#                             f"22 - FALLBACK FAILED: Fallback also failed - {type(fallback_error).__name__}"
#                         )
#                         print__analyze_debug(
#                             f"🚨 Fallback also failed: {type(fallback_error).__name__}: {str(fallback_error)}"
#                         )
#                         print__feedback_flow(
#                             f"🚨 Fallback analysis also failed: {fallback_error}"
#                         )
#                         raise HTTPException(
#                             status_code=500,
#                             detail="Sorry, there was an error processing your request. Please try again.",
#                         )
#                 else:
#                     # Re-raise non-database errors
#                     print__analysis_tracing_debug(
#                         f"23 - NON-DATABASE ERROR: Non-database error - {type(analysis_error).__name__}"
#                     )
#                     print__analyze_debug(
#                         f"🚨 Non-database error, re-raising: {type(analysis_error).__name__}: {str(analysis_error)}"
#                     )
#                     print__feedback_flow(f"🚨 Non-database error: {analysis_error}")
#                     raise HTTPException(
#                         status_code=500,
#                         detail="Sorry, there was an error processing your request. Please try again.",
#                     )

#             print__analysis_tracing_debug(
#                 "24 - RESPONSE PREPARATION: Preparing response data"
#             )
#             print__analyze_debug(f"🔍 About to prepare response data")
#             # Simple response preparation
#             response_data = {
#                 "prompt": request.prompt,
#                 "result": (
#                     result["result"]
#                     if isinstance(result, dict) and "result" in result
#                     else str(result)
#                 ),
#                 "queries_and_results": (
#                     result.get("queries_and_results", [])
#                     if isinstance(result, dict)
#                     else []
#                 ),
#                 "thread_id": request.thread_id,
#                 "top_selection_codes": (
#                     result.get("top_selection_codes", [])
#                     if isinstance(result, dict)
#                     else []
#                 ),
#                 "iteration": (
#                     result.get("iteration", 0) if isinstance(result, dict) else 0
#                 ),
#                 "max_iterations": (
#                     result.get("max_iterations", 2) if isinstance(result, dict) else 2
#                 ),
#                 "sql": result.get("sql", None) if isinstance(result, dict) else None,
#                 "datasetUrl": (
#                     result.get("datasetUrl", None) if isinstance(result, dict) else None
#                 ),
#                 "run_id": run_id,
#                 "top_chunks": (
#                     result.get("top_chunks", []) if isinstance(result, dict) else []
#                 ),
#             }

#             print__analysis_tracing_debug(
#                 f"25 - RESPONSE SUCCESS: Response data prepared with {len(response_data.keys())} keys"
#             )
#             print__analyze_debug(f"🔍 Response data prepared successfully")
#             print__analyze_debug(f"🔍 Response data keys: {list(response_data.keys())}")
#             print__analyze_debug(f"🔍 ANALYZE ENDPOINT - SUCCESSFUL EXIT")
#             return response_data

#     except asyncio.TimeoutError:
#         error_msg = "Analysis timed out after 8 minutes"
#         print__analysis_tracing_debug(
#             "26 - TIMEOUT ERROR: Analysis timed out after 8 minutes"
#         )
#         print__analyze_debug(f"🚨 TIMEOUT ERROR: {error_msg}")
#         print__feedback_flow(f"🚨 {error_msg}")
#         raise HTTPException(status_code=408, detail=error_msg)

#     except HTTPException as http_exc:
#         print__analysis_tracing_debug(
#             f"27 - HTTP EXCEPTION: HTTP exception {http_exc.status_code}"
#         )
#         print__analyze_debug(
#             f"🚨 HTTP EXCEPTION: {http_exc.status_code} - {http_exc.detail}"
#         )
#         raise http_exc

#     except Exception as e:
#         error_msg = f"Analysis failed: {str(e)}"
#         print__analysis_tracing_debug(
#             f"28 - UNEXPECTED EXCEPTION: Unexpected exception - {type(e).__name__}"
#         )
#         print__analyze_debug(f"🚨 UNEXPECTED EXCEPTION: {type(e).__name__}: {str(e)}")
#         print__analyze_debug(f"🚨 Exception traceback: {traceback.format_exc()}")
#         print__feedback_flow(f"🚨 {error_msg}")
#         raise HTTPException(
#             status_code=500,
#             detail="Sorry, there was an error processing your request. Please try again.",
#         )


# # ============================================================
# # MAIN API ENDPOINTS - FEEDBACK AND SENTIMENT
# # ============================================================
# @app.post("/feedback")
# async def submit_feedback(request: FeedbackRequest, user=Depends(get_current_user)):
#     """Submit feedback for a specific run_id to LangSmith."""

#     print__feedback_debug(f"🔍 FEEDBACK ENDPOINT - ENTRY POINT")
#     print__feedback_debug(f"🔍 Request received: run_id={request.run_id}")
#     print__feedback_debug(f"🔍 Feedback value: {request.feedback}")
#     print__feedback_debug(
#         f"🔍 Comment length: {len(request.comment) if request.comment else 0}"
#     )

#     user_email = user.get("email")
#     print__feedback_debug(f"🔍 User email extracted: {user_email}")

#     if not user_email:
#         print__feedback_debug(f"🚨 No user email found in token")
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__feedback_flow(f"📥 Incoming feedback request:")
#     print__feedback_flow(f"👤 User: {user_email}")
#     print__feedback_flow(f"🔑 Run ID: '{request.run_id}'")
#     print__feedback_flow(
#         f"🔑 Run ID type: {type(request.run_id).__name__}, length: {len(request.run_id) if request.run_id else 0}"
#     )
#     print__feedback_flow(f"👍/👎 Feedback: {request.feedback}")
#     print__feedback_flow(f"💬 Comment: {request.comment}")

#     # Validate that at least one of feedback or comment is provided
#     print__feedback_debug(f"🔍 Validating request data")
#     if request.feedback is None and not request.comment:
#         print__feedback_debug(f"🚨 No feedback or comment provided")
#         raise HTTPException(
#             status_code=400,
#             detail="At least one of 'feedback' or 'comment' must be provided",
#         )

#     try:
#         try:
#             print__feedback_debug(f"🔍 Starting UUID validation")
#             print__feedback_flow(f"🔍 Validating UUID format for: '{request.run_id}'")
#             # Debug check if it resembles a UUID at all
#             if not request.run_id or len(request.run_id) < 32:
#                 print__feedback_debug(
#                     f"🚨 Run ID suspiciously short: '{request.run_id}' (length: {len(request.run_id)})"
#                 )
#                 print__feedback_flow(
#                     f"⚠️ Run ID is suspiciously short for a UUID: '{request.run_id}'"
#                 )

#             # Try to convert to UUID to validate format
#             try:
#                 run_uuid = str(uuid.UUID(request.run_id))
#                 print__feedback_debug(f"🔍 UUID validation successful: '{run_uuid}'")
#                 print__feedback_flow(f"✅ UUID validation successful: '{run_uuid}'")
#             except ValueError as uuid_error:
#                 print__feedback_debug(f"🚨 UUID validation failed: {str(uuid_error)}")
#                 print__feedback_flow(f"🚨 UUID ValueError details: {str(uuid_error)}")
#                 # More detailed diagnostic about the input
#                 for i, char in enumerate(request.run_id):
#                     if not (char.isalnum() or char == "-"):
#                         print__feedback_flow(
#                             f"🚨 Invalid character at position {i}: '{char}' (ord={ord(char)})"
#                         )
#                 raise
#         except ValueError:
#             print__feedback_debug(
#                 f"🚨 UUID format validation failed for: '{request.run_id}'"
#             )
#             print__feedback_flow(f"❌ UUID validation failed for: '{request.run_id}'")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}",
#             )

#         # 🔒 SECURITY CHECK: Verify user owns this run_id before submitting feedback
#         print__feedback_debug(f"🔍 Starting ownership verification")
#         print__feedback_flow(f"🔒 Verifying run_id ownership for user: {user_email}")

#         try:
#             # Get a healthy pool to check ownership
#             print__feedback_debug(f"🔍 Importing get_direct_connection")

#             print__feedback_debug(f"🔍 Getting healthy pool")
#             pool = await get_direct_connection()
#             print__feedback_debug(f"🔍 Pool obtained: {type(pool).__name__}")

#             print__feedback_debug(f"🔍 Getting connection from pool")
#             async with pool.connection() as conn:
#                 print__feedback_debug(f"🔍 Connection obtained: {type(conn).__name__}")
#                 async with conn.cursor() as cur:
#                     print__feedback_debug(f"🔍 Executing ownership query")
#                     await cur.execute(
#                         """
#                         SELECT COUNT(*) FROM users_threads_runs
#                         WHERE run_id = %s AND email = %s
#                     """,
#                         (run_uuid, user_email),
#                     )

#                     print__feedback_debug(
#                         f"🔍 Ownership query executed, fetching result"
#                     )
#                     ownership_row = await cur.fetchone()
#                     ownership_count = ownership_row[0] if ownership_row else 0
#                     print__feedback_debug(f"🔍 Ownership count: {ownership_count}")

#                     if ownership_count == 0:
#                         print__feedback_debug(
#                             f"🚨 User does not own run_id - access denied"
#                         )
#                         print__feedback_flow(
#                             f"🚫 SECURITY: User {user_email} does not own run_id {run_uuid} - feedback denied"
#                         )
#                         raise HTTPException(
#                             status_code=404, detail="Run ID not found or access denied"
#                         )

#                     print__feedback_debug(f"🔍 Ownership verification successful")
#                     print__feedback_flow(
#                         f"✅ SECURITY: User {user_email} owns run_id {run_uuid} - feedback authorized"
#                     )

#         except HTTPException:
#             raise
#         except Exception as ownership_error:
#             print__feedback_debug(
#                 f"🚨 Ownership verification error: {type(ownership_error).__name__}: {str(ownership_error)}"
#             )
#             print__feedback_debug(
#                 f"🚨 Ownership error traceback: {traceback.format_exc()}"
#             )
#             print__feedback_flow(f"⚠️ Could not verify ownership: {ownership_error}")
#             # Continue with feedback submission but log the warning
#             print__feedback_flow(
#                 f"⚠️ Proceeding with feedback submission despite ownership check failure"
#             )

#         print__feedback_debug(f"🔍 Initializing LangSmith client")
#         print__feedback_flow("🔄 Initializing LangSmith client")
#         client = Client()
#         print__feedback_debug(f"🔍 LangSmith client created: {type(client).__name__}")

#         # Prepare feedback data for LangSmith
#         print__feedback_debug(f"🔍 Preparing feedback data")
#         feedback_kwargs = {"run_id": run_uuid, "key": "SENTIMENT"}

#         # Only add score if feedback is provided
#         if request.feedback is not None:
#             feedback_kwargs["score"] = request.feedback
#             print__feedback_debug(f"🔍 Adding score to feedback: {request.feedback}")
#             print__feedback_flow(
#                 f"📤 Submitting feedback with score to LangSmith - run_id: '{run_uuid}', score: {request.feedback}"
#             )
#         else:
#             print__feedback_debug(f"🔍 No score provided - comment-only feedback")
#             print__feedback_flow(
#                 f"📤 Submitting comment-only feedback to LangSmith - run_id: '{run_uuid}'"
#             )

#         # Only add comment if provided
#         if request.comment:
#             feedback_kwargs["comment"] = request.comment
#             print__feedback_debug(
#                 f"🔍 Adding comment to feedback (length: {len(request.comment)})"
#             )

#         print__feedback_debug(f"🔍 Submitting feedback to LangSmith")
#         print__feedback_debug(f"🔍 Feedback kwargs: {feedback_kwargs}")
#         client.create_feedback(**feedback_kwargs)
#         print__feedback_debug(f"🔍 LangSmith feedback submission successful")

#         print__feedback_flow(f"✅ Feedback successfully submitted to LangSmith")

#         result = {
#             "message": "Feedback submitted successfully",
#             "run_id": run_uuid,
#             "feedback": request.feedback,
#             "comment": request.comment,
#         }
#         print__feedback_debug(f"🔍 FEEDBACK ENDPOINT - SUCCESSFUL EXIT")
#         return result

#     except HTTPException:
#         raise
#     except Exception as e:
#         print__feedback_debug(
#             f"🚨 Exception in feedback processing: {type(e).__name__}: {str(e)}"
#         )
#         print__feedback_debug(
#             f"🚨 Feedback processing traceback: {traceback.format_exc()}"
#         )
#         print__feedback_flow(f"🚨 LangSmith feedback submission error: {str(e)}")
#         print__feedback_flow(f"🔍 Error type: {type(e).__name__}")
#         raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")


# @app.post("/sentiment")
# async def update_sentiment(request: SentimentRequest, user=Depends(get_current_user)):
#     """Update sentiment for a specific run_id."""

#     print__sentiment_debug(f"🔍 SENTIMENT ENDPOINT - ENTRY POINT")
#     print__sentiment_debug(f"🔍 Request received: run_id={request.run_id}")
#     print__sentiment_debug(f"🔍 Sentiment value: {request.sentiment}")

#     user_email = user.get("email")
#     print__sentiment_debug(f"🔍 User email extracted: {user_email}")

#     if not user_email:
#         print__sentiment_debug(f"🚨 No user email found in token")
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__sentiment_flow("📥 Incoming sentiment update request:")
#     print__sentiment_flow(f"👤 User: {user_email}")
#     print__sentiment_flow(f"🔑 Run ID: '{request.run_id}'")
#     print__sentiment_flow(f"👍/👎 Sentiment: {request.sentiment}")

#     try:
#         # Validate UUID format
#         print__sentiment_debug("🔍 Starting UUID validation")
#         try:
#             run_uuid = str(uuid.UUID(request.run_id))
#             print__sentiment_debug(f"🔍 UUID validation successful: '{run_uuid}'")
#             print__sentiment_flow(f"✅ UUID validation successful: '{run_uuid}'")
#         except ValueError:
#             print__sentiment_debug(f"🚨 UUID validation failed for: '{request.run_id}'")
#             print__sentiment_flow(f"❌ UUID validation failed for: '{request.run_id}'")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid run_id format. Expected UUID, got: {request.run_id}",
#             )

#         # 🔒 SECURITY: Update sentiment with user email verification
#         print__sentiment_debug(
#             "🔍 Starting sentiment update with ownership verification"
#         )
#         print__sentiment_flow("🔒 Verifying ownership before sentiment update")
#         success = await update_thread_run_sentiment(run_uuid, request.sentiment)
#         print__sentiment_debug(f"🔍 Sentiment update result: {success}")

#         if success:
#             print__sentiment_debug("🔍 Sentiment update successful")
#             print__sentiment_flow("✅ Sentiment successfully updated")
#             result = {
#                 "message": "Sentiment updated successfully",
#                 "run_id": run_uuid,
#                 "sentiment": request.sentiment,
#             }
#             print__sentiment_debug(f"🔍 SENTIMENT ENDPOINT - SUCCESSFUL EXIT")
#             return result
#         else:
#             print__sentiment_debug(
#                 "🚨 Sentiment update failed - access denied or not found"
#             )
#             print__sentiment_flow(
#                 "❌ Failed to update sentiment - run_id may not exist or access denied"
#             )
#             raise HTTPException(
#                 status_code=404, detail=f"Run ID not found or access denied: {run_uuid}"
#             )

#     except HTTPException:
#         raise
#     except Exception as e:
#         print__sentiment_debug(
#             f"🚨 Exception in sentiment processing: {type(e).__name__}: {str(e)}"
#         )
#         print__sentiment_debug(
#             f"🚨 Sentiment processing traceback: {traceback.format_exc()}"
#         )
#         print__sentiment_flow(f"🚨 Sentiment update error: {str(e)}")
#         print__sentiment_flow(f"🔍 Error type: {type(e).__name__}")
#         raise HTTPException(status_code=500, detail=f"Failed to update sentiment: {e}")


# # ============================================================
# # MAIN API ENDPOINTS - CHAT MANAGEMENT
# # ============================================================
# @app.get("/chat/{thread_id}/sentiments")
# async def get_thread_sentiments(thread_id: str, user=Depends(get_current_user)):
#     """Get sentiment values for all messages in a thread."""

#     print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - ENTRY POINT")
#     print__chat_sentiments_debug(f"🔍 Request received: thread_id={thread_id}")

#     user_email = user.get("email")
#     print__chat_sentiments_debug(f"🔍 User email extracted: {user_email}")

#     if not user_email:
#         print__chat_sentiments_debug("🚨 No user email found in token")
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     try:
#         print__chat_sentiments_debug(
#             f"🔍 Getting sentiments for thread {thread_id}, user: {user_email}"
#         )
#         print__sentiment_flow(
#             f"📥 Getting sentiments for thread {thread_id}, user: {user_email}"
#         )
#         sentiments = await get_thread_run_sentiments(user_email, thread_id)
#         print__chat_sentiments_debug(f"🔍 Retrieved {len(sentiments)} sentiment values")

#         print__sentiment_flow(f"✅ Retrieved {len(sentiments)} sentiment values")
#         print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - SUCCESSFUL EXIT")
#         return sentiments

#     except Exception as e:
#         print__chat_sentiments_debug(
#             f"🚨 Exception in chat sentiments processing: {type(e).__name__}: {str(e)}"
#         )
#         print__chat_sentiments_debug(
#             f"🚨 Chat sentiments processing traceback: {traceback.format_exc()}"
#         )
#         print__sentiment_flow(
#             f"❌ Failed to get sentiments for thread {thread_id}: {e}"
#         )
#         raise HTTPException(status_code=500, detail=f"Failed to get sentiments: {e}")


# @app.get("/chat-threads")
# async def get_chat_threads(
#     page: int = Query(1, ge=1, description="Page number (1-indexed)"),
#     limit: int = Query(10, ge=1, le=50, description="Number of threads per page"),
#     user=Depends(get_current_user),
# ) -> PaginatedChatThreadsResponse:
#     """Get paginated chat threads for the authenticated user."""

#     print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ENTRY POINT")
#     print__chat_threads_debug(f"🔍 Request parameters: page={page}, limit={limit}")

#     try:
#         user_email = user["email"]
#         print__chat_threads_debug(f"🔍 User email extracted: {user_email}")
#         print__chat_threads_debug(
#             f"Loading chat threads for user: {user_email} (page: {page}, limit: {limit})"
#         )

#         print__chat_threads_debug("🔍 Starting simplified approach")
#         print__chat_threads_debug("Getting chat threads with simplified approach")

#         # Get total count first
#         print__chat_threads_debug("🔍 Getting total threads count")
#         print__chat_threads_debug(f"Getting chat threads count for user: {user_email}")
#         total_count = await get_user_chat_threads_count(user_email)
#         print__chat_threads_debug(f"🔍 Total count retrieved: {total_count}")
#         print__chat_threads_debug(
#             f"Total threads count for user {user_email}: {total_count}"
#         )

#         # Calculate offset for pagination
#         offset = (page - 1) * limit
#         print__chat_threads_debug(f"🔍 Calculated offset: {offset}")

#         # Get threads for this page
#         print__chat_threads_debug(
#             f"🔍 Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
#         )
#         print__chat_threads_debug(
#             f"Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
#         )
#         threads = await get_user_chat_threads(user_email, limit=limit, offset=offset)
#         print__chat_threads_debug(f"🔍 Retrieved threads: {threads}")
#         if threads is None:
#             print__chat_threads_debug(
#                 "get_user_chat_threads returned None! Setting to empty list."
#             )
#             threads = []
#         print__chat_threads_debug(f"🔍 Retrieved {len(threads)} threads from database")
#         print__chat_threads_debug(
#             f"Retrieved {len(threads)} threads for user {user_email}"
#         )

#         # Try/except around the for-loop to catch and print any errors
#         try:
#             chat_thread_responses = []
#             for thread in threads:
#                 print("[GENERIC-DEBUG] Processing thread dict:", thread)
#                 chat_thread_response = ChatThreadResponse(
#                     thread_id=thread["thread_id"],
#                     latest_timestamp=thread["latest_timestamp"],
#                     run_count=thread["run_count"],
#                     title=thread["title"],
#                     full_prompt=thread["full_prompt"],
#                 )
#                 chat_thread_responses.append(chat_thread_response)
#         except Exception as e:
#             print("[GENERIC-ERROR] Exception in /chat-threads for-loop:", e)
#             print(traceback.format_exc())
#             # Return empty result on error
#             return PaginatedChatThreadsResponse(
#                 threads=[], total_count=0, page=page, limit=limit, has_more=False
#             )

#         # Convert to response format
#         print__chat_threads_debug("🔍 Converting threads to response format")
#         chat_thread_responses = []
#         for thread in threads:
#             chat_thread_response = ChatThreadResponse(
#                 thread_id=thread["thread_id"],
#                 latest_timestamp=thread["latest_timestamp"],
#                 run_count=thread["run_count"],
#                 title=thread["title"],
#                 full_prompt=thread["full_prompt"],
#             )
#             chat_thread_responses.append(chat_thread_response)

#         # Calculate pagination info
#         has_more = (offset + len(chat_thread_responses)) < total_count
#         print__chat_threads_debug(f"🔍 Pagination calculated: has_more={has_more}")

#         print__chat_threads_debug(
#             f"Retrieved {len(threads)} threads for user {user_email} (total: {total_count})"
#         )
#         print__chat_threads_debug(
#             f"Returning {len(chat_thread_responses)} threads to frontend (page {page}/{(total_count + limit - 1) // limit})"
#         )

#         result = PaginatedChatThreadsResponse(
#             threads=chat_thread_responses,
#             total_count=total_count,
#             page=page,
#             limit=limit,
#             has_more=has_more,
#         )
#         print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - SUCCESSFUL EXIT")
#         return result

#     except Exception as e:
#         print__chat_threads_debug(
#             f"🚨 Exception in chat threads processing: {type(e).__name__}: {str(e)}"
#         )
#         print__chat_threads_debug(
#             f"🚨 Chat threads processing traceback: {traceback.format_exc()}"
#         )
#         print__chat_threads_debug(f"❌ Error getting chat threads: {e}")
#         print__chat_threads_debug(f"Full traceback: {traceback.format_exc()}")

#         # Return error response
#         result = PaginatedChatThreadsResponse(
#             threads=[], total_count=0, page=page, limit=limit, has_more=False
#         )
#         print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ERROR EXIT")
#         return result


# @app.delete("/chat/{thread_id}")
# async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
#     """Delete all PostgreSQL checkpoint records and thread entries for a specific thread_id."""

#     print__delete_chat_debug(f"🔍 DELETE_CHAT ENDPOINT - ENTRY POINT")
#     print__delete_chat_debug(f"🔍 Request received: thread_id={thread_id}")

#     user_email = user.get("email")
#     print__delete_chat_debug(f"🔍 User email extracted: {user_email}")

#     if not user_email:
#         print__delete_chat_debug(f"🚨 No user email found in token")
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__delete_chat_debug(
#         f"🗑️ Deleting chat thread {thread_id} for user {user_email}"
#     )

#     try:
#         # Get a healthy checkpointer
#         print__delete_chat_debug(f"🔧 DEBUG: Getting healthy checkpointer...")
#         checkpointer = await get_healthy_checkpointer()
#         print__delete_chat_debug(
#             f"🔧 DEBUG: Checkpointer type: {type(checkpointer).__name__}"
#         )

#         # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
#         print__delete_chat_debug(
#             f"🔧 DEBUG: Checking if checkpointer has 'conn' attribute..."
#         )
#         if not hasattr(checkpointer, "conn"):
#             print__delete_chat_debug(
#                 f"⚠️ No PostgreSQL checkpointer available - nothing to delete"
#             )
#             return {
#                 "message": "No PostgreSQL checkpointer available - nothing to delete"
#             }

#         print__delete_chat_debug(f"🔧 DEBUG: Checkpointer has 'conn' attribute")
#         print__delete_chat_debug(
#             f"🔧 DEBUG: checkpointer.conn type: {type(checkpointer.conn).__name__}"
#         )

#         # Access the connection through the conn attribute
#         conn_obj = checkpointer.conn
#         print__delete_chat_debug(
#             f"🔧 DEBUG: Connection object set, type: {type(conn_obj).__name__}"
#         )

#         # FIXED: Handle both connection pool and single connection cases
#         if hasattr(conn_obj, "connection") and callable(
#             getattr(conn_obj, "connection", None)
#         ):
#             # It's a connection pool - use pool.connection()
#             print__delete_chat_debug(f"🔧 DEBUG: Using connection pool pattern...")
#             async with conn_obj.connection() as conn:
#                 print__delete_chat_debug(
#                     f"🔧 DEBUG: Successfully got connection from pool, type: {type(conn).__name__}"
#                 )
#                 result_data = await perform_deletion_operations(
#                     conn, user_email, thread_id
#                 )
#                 return result_data
#         else:
#             # It's a single connection - use it directly
#             print__delete_chat_debug(f"🔧 DEBUG: Using single connection pattern...")
#             conn = conn_obj
#             print__delete_chat_debug(
#                 f"🔧 DEBUG: Using direct connection, type: {type(conn).__name__}"
#             )
#             result_data = await perform_deletion_operations(conn, user_email, thread_id)
#             return result_data

#     except Exception as e:
#         error_msg = str(e)
#         print__delete_chat_debug(
#             f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}"
#         )
#         print__delete_chat_debug(f"🔧 DEBUG: Main exception type: {type(e).__name__}")
#         print__delete_chat_debug(
#             f"🔧 DEBUG: Main exception traceback: {traceback.format_exc()}"
#         )

#         # If it's a connection error, don't treat it as a failure since it means
#         # there are likely no records to delete anyway
#         if any(
#             keyword in error_msg.lower()
#             for keyword in [
#                 "ssl error",
#                 "connection",
#                 "timeout",
#                 "operational error",
#                 "server closed",
#                 "bad connection",
#                 "consuming input failed",
#             ]
#         ):
#             print__delete_chat_debug(
#                 f"⚠️ PostgreSQL connection unavailable - no records to delete"
#             )
#             return {
#                 "message": "PostgreSQL connection unavailable - no records to delete",
#                 "thread_id": thread_id,
#                 "user_email": user_email,
#                 "warning": "Database connection issues",
#             }
#         else:
#             raise HTTPException(
#                 status_code=500, detail=f"Failed to delete checkpoint records: {e}"
#             )


# # ============================================================
# # MAIN API ENDPOINTS - DATA CATALOG
# # ============================================================
# @app.get("/catalog")
# def get_catalog(
#     page: int = Query(1, ge=1),
#     q: Optional[str] = None,
#     page_size: int = Query(10, ge=1, le=10000),
#     user=Depends(get_current_user),
# ):
#     db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
#     offset = (page - 1) * page_size
#     where_clause = ""
#     params = []
#     if q:
#         where_clause = "WHERE selection_code LIKE ? OR extended_description LIKE ?"
#         like_q = f"%{q}%"
#         params.extend([like_q, like_q])
#     query = f"""
#         SELECT selection_code, extended_description
#         FROM selection_descriptions
#         {where_clause}
#         ORDER BY selection_code
#         LIMIT ? OFFSET ?
#     """
#     params.extend([page_size, offset])
#     count_query = f"SELECT COUNT(*) FROM selection_descriptions {where_clause}"
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#         cursor.execute(count_query, params[:-2] if q else [])
#         total = cursor.fetchone()[0]
#         cursor.execute(query, params)
#         rows = cursor.fetchall()
#     results = [
#         {"selection_code": row[0], "extended_description": row[1]} for row in rows
#     ]
#     return {"results": results, "total": total, "page": page, "page_size": page_size}


# @app.get("/data-tables")
# def get_data_tables(q: Optional[str] = None, user=Depends(get_current_user)):
#     db_path = "data/czsu_data.db"
#     desc_db_path = "metadata/llm_selection_descriptions/selection_descriptions.db"
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#         cursor.execute(
#             "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
#         )
#         tables = [row[0] for row in cursor.fetchall()]
#     if q:
#         q_lower = q.lower()
#         tables = [t for t in tables if q_lower in t.lower()]
#     # Fetch short_descriptions from the other DB
#     desc_map = {}
#     try:
#         with sqlite3.connect(desc_db_path) as desc_conn:
#             desc_cursor = desc_conn.cursor()
#             desc_cursor.execute(
#                 "SELECT selection_code, short_description FROM selection_descriptions"
#             )
#             for code, short_desc in desc_cursor.fetchall():
#                 desc_map[code] = short_desc
#     except Exception as e:
#         print__debug(f"Error fetching short_descriptions: {e}")
#     # Build result list
#     result = [
#         {"selection_code": t, "short_description": desc_map.get(t, "")} for t in tables
#     ]
#     return {"tables": result}


# @app.get("/data-table")
# def get_data_table(table: Optional[str] = None, user=Depends(get_current_user)):
#     db_path = "data/czsu_data.db"
#     if not table:
#         print__debug("No table specified")
#         return {"columns": [], "rows": []}
#     print__debug(f"Requested table: {table}")
#     with sqlite3.connect(db_path) as conn:
#         cursor = conn.cursor()
#         try:
#             cursor.execute(f"SELECT * FROM '{table}' LIMIT 10000")
#             columns = [desc[0] for desc in cursor.description]
#             rows = cursor.fetchall()
#             print__debug(f"Columns: {columns}, Rows count: {len(rows)}")
#         except Exception as e:
#             print__debug(f"Error fetching table '{table}': {e}")
#             return {"columns": [], "rows": []}
#     return {"columns": columns, "rows": rows}


# # ============================================================
# # MAIN API ENDPOINTS - CHAT MESSAGES
# # ============================================================
# @app.get("/chat/{thread_id}/messages")
# async def get_chat_messages(
#     thread_id: str, user=Depends(get_current_user)
# ) -> List[ChatMessage]:
#     """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages."""

#     user_email = user.get("email")
#     if not user_email:
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__api_postgresql(
#         f"📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}"
#     )

#     try:
#         # 🔒 SECURITY CHECK: Verify user owns this thread before retrieving messages
#         print__api_postgresql(
#             f"🔒 Verifying thread ownership for user: {user_email}, thread: {thread_id}"
#         )

#         # Check if this user has any entries in users_threads_runs for this thread
#         checkpointer = await get_healthy_checkpointer()

#         if not hasattr(checkpointer, "conn"):
#             print__api_postgresql(
#                 f"⚠️ No PostgreSQL checkpointer available - returning empty messages"
#             )
#             return []

#         # Verify thread ownership using users_threads_runs table
#         async with checkpointer.conn.connection() as conn:
#             async with conn.cursor() as cur:
#                 await cur.execute(
#                     """
#             SELECT COUNT(*) FROM users_threads_runs
#             WHERE email = %s AND thread_id = %s
#         """,
#                     (user_email, thread_id),
#                 )

#                 ownership_row = await cur.fetchone()
#                 thread_entries_count = ownership_row[0] if ownership_row else 0

#             if thread_entries_count == 0:
#                 print__api_postgresql(
#                     f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - access denied"
#                 )
#                 # Return empty instead of error to avoid information disclosure
#                 return []

#             print__api_postgresql(
#                 f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted"
#             )

#         # Get conversation messages from checkpoint history
#         stored_messages = await get_conversation_messages_from_checkpoints(
#             checkpointer, thread_id, user_email
#         )

#         if not stored_messages:
#             print__api_postgresql(
#                 f"⚠ No messages found in checkpoints for thread {thread_id}"
#             )
#             return []

#         print__api_postgresql(
#             f"📄 Found {len(stored_messages)} messages from checkpoints"
#         )

#         # Get additional metadata from latest checkpoint (like queries_and_results and top_selection_codes)
#         queries_and_results = await get_queries_and_results_from_latest_checkpoint(
#             checkpointer, thread_id
#         )

#         # Get dataset information and SQL query from latest checkpoint
#         datasets_used = []
#         sql_query = None
#         top_chunks = []

#         try:
#             config = {"configurable": {"thread_id": thread_id}}
#             state_snapshot = await checkpointer.aget_tuple(config)

#             if state_snapshot and state_snapshot.checkpoint:
#                 channel_values = state_snapshot.checkpoint.get("channel_values", {})
#                 top_selection_codes = channel_values.get("top_selection_codes", [])

#                 # Use the datasets directly
#                 datasets_used = top_selection_codes

#                 # Get PDF chunks from checkpoint state
#                 checkpoint_top_chunks = channel_values.get("top_chunks", [])
#                 print__api_postgresql(
#                     f"📄 Found {len(checkpoint_top_chunks)} PDF chunks in checkpoint for thread {thread_id}"
#                 )

#                 # Convert Document objects to serializable format
#                 if checkpoint_top_chunks:
#                     for chunk in checkpoint_top_chunks:
#                         chunk_data = {
#                             "content": (
#                                 chunk.page_content
#                                 if hasattr(chunk, "page_content")
#                                 else str(chunk)
#                             ),
#                             "metadata": (
#                                 chunk.metadata if hasattr(chunk, "metadata") else {}
#                             ),
#                         }
#                         top_chunks.append(chunk_data)
#                     print__api_postgresql(
#                         f"📄 Serialized {len(top_chunks)} PDF chunks for frontend"
#                     )

#                 # Extract SQL query from queries_and_results for SQL button
#                 if queries_and_results:
#                     # Get the last (most recent) SQL query
#                     sql_query = (
#                         queries_and_results[-1][0] if queries_and_results[-1] else None
#                     )

#         except Exception as e:
#             print__api_postgresql(
#                 f"⚠️ Could not get datasets/SQL/chunks from checkpoint: {e}"
#             )
#             print__api_postgresql(
#                 f"🔧 Using fallback empty values: datasets=[], sql=None, chunks=[]"
#             )

#         # Convert stored messages to frontend format
#         chat_messages = []

#         for i, stored_msg in enumerate(stored_messages):
#             # Debug: Log the raw stored message
#             print__api_postgresql(
#                 f"🔍 Processing stored message {i+1}: is_user={stored_msg.get('is_user')}, content='{stored_msg.get('content', '')[:30]}...'"
#             )

#             # Create meta information for messages
#             meta_info = {}

#             # For AI messages, include queries/results, datasets used, and SQL query
#             if not stored_msg["is_user"]:
#                 if queries_and_results:
#                     meta_info["queriesAndResults"] = queries_and_results
#                 if datasets_used:
#                     meta_info["datasetsUsed"] = datasets_used
#                 if sql_query:
#                     meta_info["sqlQuery"] = sql_query
#                 if top_chunks:
#                     meta_info["topChunks"] = top_chunks
#                 meta_info["source"] = "checkpoint_history"
#                 print__api_postgresql(
#                     f"🔍 Added metadata to AI message: datasets={len(datasets_used)}, sql={'Yes' if sql_query else 'No'}, chunks={len(top_chunks)}"
#                 )

#             # Convert queries_and_results for AI messages
#             queries_results_for_frontend = None
#             if not stored_msg["is_user"] and queries_and_results:
#                 queries_results_for_frontend = queries_and_results

#             # Create ChatMessage with explicit debugging
#             is_user_flag = stored_msg["is_user"]
#             print__api_postgresql(f"🔍 Creating ChatMessage: isUser={is_user_flag}")

#             chat_message = ChatMessage(
#                 id=stored_msg["id"],
#                 threadId=thread_id,
#                 user=user_email if is_user_flag else "AI",
#                 content=stored_msg["content"],
#                 isUser=is_user_flag,  # Explicitly use the flag
#                 createdAt=int(stored_msg["timestamp"].timestamp() * 1000),
#                 error=None,
#                 meta=(
#                     meta_info if meta_info else None
#                 ),  # Only add meta if it has content
#                 queriesAndResults=queries_results_for_frontend,
#                 isLoading=False,
#                 startedAt=None,
#                 isError=False,
#             )

#             # Debug: Verify the ChatMessage was created correctly
#             print__api_postgresql(
#                 f"🔍 ChatMessage created: isUser={chat_message.isUser}, user='{chat_message.user}'"
#             )

#             chat_messages.append(chat_message)

#         print__api_postgresql(
#             f"✅ Converted {len(chat_messages)} messages to frontend format"
#         )

#         # Log the messages for debugging
#         for i, msg in enumerate(chat_messages):
#             user_type = "👤 User" if msg.isUser else "🤖 AI"
#             content_preview = (
#                 msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
#             )
#             datasets_info = (
#                 f" (datasets: {msg.meta.get('datasetsUsed', [])})"
#                 if msg.meta and msg.meta.get("datasetsUsed")
#                 else ""
#             )
#             sql_info = (
#                 f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)"
#                 if msg.meta and msg.meta.get("sqlQuery")
#                 else ""
#             )
#             print__api_postgresql(
#                 f"{i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}"
#             )

#         return chat_messages

#     except Exception as e:
#         error_msg = str(e)
#         print__api_postgresql(
#             f"❌ Failed to load checkpoint messages for thread {thread_id}: {e}"
#         )

#         # Handle specific database connection errors gracefully
#         if any(
#             keyword in error_msg.lower()
#             for keyword in [
#                 "ssl error",
#                 "connection",
#                 "timeout",
#                 "operational error",
#                 "server closed",
#                 "bad connection",
#                 "consuming input failed",
#             ]
#         ):
#             print__api_postgresql(
#                 f"⚠ Database connection error - returning empty messages"
#             )
#             return []
#         else:
#             raise HTTPException(
#                 status_code=500, detail=f"Failed to load checkpoint messages: {e}"
#             )


# @app.get("/chat/{thread_id}/run-ids")
# async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
#     """Get run_ids for messages in a thread to enable feedback submission."""

#     user_email = user.get("email")
#     if not user_email:
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__feedback_flow(f"🔍 Fetching run_ids for thread {thread_id}")

#     try:
#         pool = await get_healthy_checkpointer()
#         pool = pool.conn if hasattr(pool, "conn") else None

#         if not pool:
#             print__feedback_flow("⚠ No pool available for run_id lookup")
#             return {"run_ids": []}

#         async with pool.connection() as conn:
#             print__feedback_flow(f"📊 Executing SQL query for thread {thread_id}")
#             async with conn.cursor() as cur:
#                 await cur.execute(
#                     """
#                     SELECT run_id, prompt, timestamp
#                     FROM users_threads_runs
#                     WHERE email = %s AND thread_id = %s
#                     ORDER BY timestamp ASC
#                 """,
#                     (user_email, thread_id),
#                 )

#                 run_id_data = []
#                 rows = await cur.fetchall()
#                 for row in rows:
#                     print__feedback_flow(
#                         f"📝 Processing database row - run_id: {row[0]}, prompt: {row[1][:50]}..."
#                     )
#                     try:
#                         run_uuid = str(uuid.UUID(row[0])) if row[0] else None
#                         if run_uuid:
#                             run_id_data.append(
#                                 {
#                                     "run_id": run_uuid,
#                                     "prompt": row[1],
#                                     "timestamp": row[2].isoformat(),
#                                 }
#                             )
#                             print__feedback_flow(f"✅ Valid UUID found: {run_uuid}")
#                         else:
#                             print__feedback_flow(
#                                 f"⚠ Null run_id found for prompt: {row[1][:50]}..."
#                             )
#                     except ValueError:
#                         print__feedback_flow(f"❌ Invalid UUID in database: {row[0]}")
#                         continue

#                 print__feedback_flow(
#                     f"📊 Total valid run_ids found: {len(run_id_data)}"
#                 )
#                 return {"run_ids": run_id_data}

#     except Exception as e:
#         print__feedback_flow(f"🚨 Error fetching run_ids: {str(e)}")
#         return {"run_ids": []}


# # ============================================================
# # MAIN API ENDPOINTS - BULK OPERATIONS
# # ============================================================


# @app.get("/chat/all-messages-for-all-threads")
# async def get_all_chat_messages(user=Depends(get_current_user)) -> Dict:
#     """Get all chat messages for the authenticated user using bulk loading with improved caching."""

#     print__chat_all_messages_debug(f"🔍 CHAT_ALL_MESSAGES ENDPOINT - ENTRY POINT")

#     user_email = user["email"]
#     print__chat_all_messages_debug(f"🔍 User email extracted: {user_email}")
#     print__chat_all_messages_debug(
#         f"📥 BULK REQUEST: Loading ALL chat messages for user: {user_email}"
#     )

#     # Check if we have a recent cached result
#     cache_key = f"bulk_messages_{user_email}"
#     current_time = time.time()
#     print__chat_all_messages_debug(f"🔍 Cache key: {cache_key}")
#     print__chat_all_messages_debug(f"🔍 Current time: {current_time}")

#     if cache_key in _bulk_loading_cache:
#         print__chat_all_messages_debug(f"🔍 Cache entry found for user")
#         cached_data, cache_time = _bulk_loading_cache[cache_key]
#         cache_age = current_time - cache_time
#         print__chat_all_messages_debug(
#             f"🔍 Cache age: {cache_age:.1f}s (timeout: {BULK_CACHE_TIMEOUT}s)"
#         )

#         if cache_age < BULK_CACHE_TIMEOUT:
#             print__chat_all_messages_debug(
#                 f"✅ CACHE HIT: Returning cached bulk data for {user_email} (age: {cache_age:.1f}s)"
#             )

#             # Return cached data with appropriate headers
#             from fastapi.responses import JSONResponse

#             response = JSONResponse(content=cached_data)
#             response.headers["Cache-Control"] = (
#                 f"public, max-age={int(BULK_CACHE_TIMEOUT - cache_age)}"
#             )
#             response.headers["ETag"] = f"bulk-{user_email}-{int(cache_time)}"
#             print__chat_all_messages_debug(
#                 f"🔍 CHAT_ALL_MESSAGES ENDPOINT - CACHE HIT EXIT"
#             )
#             return response
#         else:
#             print__chat_all_messages_debug(
#                 f"⏰ CACHE EXPIRED: Cached data too old ({cache_age:.1f}s), will refresh"
#             )
#             del _bulk_loading_cache[cache_key]
#             print__chat_all_messages_debug(f"🔍 Expired cache entry deleted")
#     else:
#         print__chat_all_messages_debug(f"🔍 No cache entry found for user")

#     # Use a lock to prevent multiple simultaneous requests from the same user
#     print__chat_all_messages_debug(
#         f"🔍 Attempting to acquire lock for user: {user_email}"
#     )
#     async with _bulk_loading_locks[user_email]:
#         print__chat_all_messages_debug(f"🔒 Lock acquired for user: {user_email}")

#         # Double-check cache after acquiring lock (another request might have completed)
#         if cache_key in _bulk_loading_cache:
#             print__chat_all_messages_debug(
#                 f"🔍 Double-checking cache after lock acquisition"
#             )
#             cached_data, cache_time = _bulk_loading_cache[cache_key]
#             cache_age = current_time - cache_time
#             if cache_age < BULK_CACHE_TIMEOUT:
#                 print__chat_all_messages_debug(
#                     f"✅ CACHE HIT (after lock): Returning cached bulk data for {user_email}"
#                 )
#                 print__chat_all_messages_debug(
#                     f"🔍 CHAT_ALL_MESSAGES ENDPOINT - CACHE HIT AFTER LOCK EXIT"
#                 )
#                 return cached_data
#             else:
#                 print__chat_all_messages_debug(
#                     f"🔍 Cache still expired after lock, proceeding with fresh request"
#                 )

#         print__chat_all_messages_debug(
#             f"🔄 CACHE MISS: Processing fresh bulk request for {user_email}"
#         )

#         # Simple memory check before starting
#         print__chat_all_messages_debug(f"🔍 Starting memory check")
#         log_memory_usage("bulk_start")
#         print__chat_all_messages_debug(f"🔍 Memory check completed")

#         try:
#             print__chat_all_messages_debug(f"🔍 Getting healthy checkpointer")
#             checkpointer = await get_healthy_checkpointer()
#             print__chat_all_messages_debug(
#                 f"🔍 Checkpointer obtained: {type(checkpointer).__name__}"
#             )

#             # STEP 1: Get all user threads, run-ids, and sentiments in ONE query
#             print__chat_all_messages_debug(
#                 f"🔍 BULK QUERY: Getting all user threads, run-ids, and sentiments"
#             )
#             user_thread_ids = []
#             all_run_ids = {}
#             all_sentiments = {}

#             # FIXED: Use our working get_direct_connection() function instead of checkpointer.conn
#             print__chat_all_messages_debug(f"🔍 Importing get_direct_connection")
#             from my_agent.utils.postgres_checkpointer import get_direct_connection

#             print__chat_all_messages_debug(f"🔍 Getting direct connection")

#             # FIXED: Use get_direct_connection() as async context manager
#             print__chat_all_messages_debug(
#                 f"🔍 Using direct connection context manager"
#             )
#             async with get_direct_connection() as conn:
#                 print__chat_all_messages_debug(
#                     f"🔍 Connection obtained: {type(conn).__name__}"
#                 )
#                 async with conn.cursor() as cur:
#                     print__chat_all_messages_debug(
#                         f"🔍 Cursor created, executing bulk query"
#                     )
#                     # Single query for all threads, run-ids, and sentiments
#                     # FIXED: Use psycopg format (%s) instead of asyncpg format ($1)
#                     await cur.execute(
#                         """
#                         SELECT
#                             thread_id,
#                             run_id,
#                             prompt,
#                             timestamp,
#                             sentiment
#                         FROM users_threads_runs
#                         WHERE email = %s
#                         ORDER BY thread_id, timestamp ASC
#                     """,
#                         (user_email,),
#                     )

#                     print__chat_all_messages_debug(
#                         f"🔍 Bulk query executed, fetching results"
#                     )
#                     rows = await cur.fetchall()
#                     print__chat_all_messages_debug(
#                         f"🔍 Retrieved {len(rows)} rows from database"
#                     )

#                 for i, row in enumerate(rows):
#                     print__chat_all_messages_debug(
#                         f"🔍 Processing row {i+1}/{len(rows)}"
#                     )
#                     # FIXED: Use index-based access instead of dict-based for psycopg
#                     thread_id = row[0]  # thread_id
#                     run_id = row[1]  # run_id
#                     prompt = row[2]  # prompt
#                     timestamp = row[3]  # timestamp
#                     sentiment = row[4]  # sentiment

#                     print__chat_all_messages_debug(
#                         f"🔍 Row data: thread_id={thread_id}, run_id={run_id}, prompt_length={len(prompt) if prompt else 0}"
#                     )

#                     # Track unique thread IDs
#                     if thread_id not in user_thread_ids:
#                         user_thread_ids.append(thread_id)
#                         print__chat_all_messages_debug(
#                             f"🔍 New thread discovered: {thread_id}"
#                         )

#                     # Build run-ids dictionary
#                     if thread_id not in all_run_ids:
#                         all_run_ids[thread_id] = []
#                         print__chat_all_messages_debug(
#                             f"🔍 Initializing run_ids list for thread: {thread_id}"
#                         )
#                     all_run_ids[thread_id].append(
#                         {
#                             "run_id": run_id,
#                             "prompt": prompt,
#                             "timestamp": timestamp.isoformat(),
#                         }
#                     )

#                     # Build sentiments dictionary
#                     if sentiment is not None:
#                         if thread_id not in all_sentiments:
#                             all_sentiments[thread_id] = {}
#                             print__chat_all_messages_debug(
#                                 f"🔍 Initializing sentiments dict for thread: {thread_id}"
#                             )
#                         all_sentiments[thread_id][run_id] = sentiment
#                         print__chat_all_messages_debug(
#                             f"🔍 Added sentiment for run_id {run_id}: {sentiment}"
#                         )

#             print__chat_all_messages_debug(
#                 f"📊 BULK: Found {len(user_thread_ids)} threads"
#             )
#             print__chat_all_messages_debug(
#                 f"📊 BULK: Found {len(all_run_ids)} thread run_ids"
#             )
#             print__chat_all_messages_debug(
#                 f"📊 BULK: Found {len(all_sentiments)} thread sentiments"
#             )

#             if not user_thread_ids:
#                 print__chat_all_messages_debug(
#                     f"⚠ No threads found for user - returning empty dictionary"
#                 )
#                 empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
#                 _bulk_loading_cache[cache_key] = (empty_result, current_time)
#                 print__chat_all_messages_debug(
#                     f"🔍 CHAT_ALL_MESSAGES ENDPOINT - EMPTY RESULT EXIT"
#                 )
#                 return empty_result

#             # STEP 2: Process threads with limited concurrency (max 3 concurrent)
#             print__chat_all_messages_debug(
#                 f"🔄 Processing {len(user_thread_ids)} threads with limited concurrency"
#             )

#             async def process_single_thread(thread_id: str):
#                 """Process a single thread using the proven working functions."""
#                 try:
#                     print__chat_all_messages_debug(f"🔄 Processing thread {thread_id}")

#                     # Use the working function
#                     print__chat_all_messages_debug(
#                         f"🔍 Getting conversation messages from checkpoints for thread: {thread_id}"
#                     )
#                     stored_messages = await get_conversation_messages_from_checkpoints(
#                         checkpointer, thread_id, user_email
#                     )

#                     if not stored_messages:
#                         print__chat_all_messages_debug(
#                             f"⚠ No messages found in checkpoints for thread {thread_id}"
#                         )
#                         return thread_id, []

#                     print__chat_all_messages_debug(
#                         f"📄 Found {len(stored_messages)} messages for thread {thread_id}"
#                     )

#                     # Get additional metadata from latest checkpoint
#                     print__chat_all_messages_debug(
#                         f"🔍 Getting queries and results from latest checkpoint for thread: {thread_id}"
#                     )
#                     queries_and_results = (
#                         await get_queries_and_results_from_latest_checkpoint(
#                             checkpointer, thread_id
#                         )
#                     )
#                     print__chat_all_messages_debug(
#                         f"🔍 Retrieved {len(queries_and_results) if queries_and_results else 0} queries and results"
#                     )

#                     # Get dataset information and SQL query from latest checkpoint
#                     datasets_used = []
#                     sql_query = None
#                     top_chunks = []

#                     try:
#                         print__chat_all_messages_debug(
#                             f"🔍 Getting state snapshot for thread: {thread_id}"
#                         )
#                         config = {"configurable": {"thread_id": thread_id}}
#                         state_snapshot = await checkpointer.aget_tuple(config)

#                         if state_snapshot and state_snapshot.checkpoint:
#                             print__chat_all_messages_debug(
#                                 f"🔍 State snapshot found for thread: {thread_id}"
#                             )
#                             channel_values = state_snapshot.checkpoint.get(
#                                 "channel_values", {}
#                             )
#                             top_selection_codes = channel_values.get(
#                                 "top_selection_codes", []
#                             )
#                             datasets_used = top_selection_codes
#                             print__chat_all_messages_debug(
#                                 f"🔍 Found {len(datasets_used)} datasets used"
#                             )

#                             # Get PDF chunks
#                             checkpoint_top_chunks = channel_values.get("top_chunks", [])
#                             print__chat_all_messages_debug(
#                                 f"🔍 Found {len(checkpoint_top_chunks)} PDF chunks in checkpoint"
#                             )
#                             if checkpoint_top_chunks:
#                                 for j, chunk in enumerate(checkpoint_top_chunks):
#                                     print__chat_all_messages_debug(
#                                         f"🔍 Processing PDF chunk {j+1}/{len(checkpoint_top_chunks)}"
#                                     )
#                                     chunk_data = {
#                                         "content": (
#                                             chunk.page_content
#                                             if hasattr(chunk, "page_content")
#                                             else str(chunk)
#                                         ),
#                                         "metadata": (
#                                             chunk.metadata
#                                             if hasattr(chunk, "metadata")
#                                             else {}
#                                         ),
#                                     }
#                                     top_chunks.append(chunk_data)
#                                 print__chat_all_messages_debug(
#                                     f"🔍 Processed {len(top_chunks)} PDF chunks"
#                                 )

#                             # Extract SQL query
#                             if queries_and_results:
#                                 sql_query = (
#                                     queries_and_results[-1][0]
#                                     if queries_and_results[-1]
#                                     else None
#                                 )
#                                 print__chat_all_messages_debug(
#                                     f"🔍 SQL query extracted: {'Yes' if sql_query else 'No'}"
#                                 )
#                         else:
#                             print__chat_all_messages_debug(
#                                 f"🔍 No state snapshot found for thread: {thread_id}"
#                             )

#                     except Exception as e:
#                         print__chat_all_messages_debug(
#                             f"⚠️ Could not get datasets/SQL/chunks from checkpoint for thread {thread_id}: {e}"
#                         )
#                         print__chat_all_messages_debug(
#                             f"🔍 Checkpoint metadata error type: {type(e).__name__}"
#                         )
#                         print__chat_all_messages_debug(
#                             f"🔍 Checkpoint metadata error traceback: {traceback.format_exc()}"
#                         )

#                     # Convert stored messages to frontend format
#                     chat_messages = []
#                     print__chat_all_messages_debug(
#                         f"🔍 Converting {len(stored_messages)} stored messages to frontend format"
#                     )

#                     for i, stored_msg in enumerate(stored_messages):
#                         print__chat_all_messages_debug(
#                             f"🔍 Processing stored message {i+1}/{len(stored_messages)}"
#                         )
#                         # Create meta information for AI messages
#                         meta_info = {}
#                         if not stored_msg["is_user"]:
#                             print__chat_all_messages_debug(
#                                 f"🔍 Processing AI message - adding metadata"
#                             )
#                             if queries_and_results:
#                                 meta_info["queriesAndResults"] = queries_and_results
#                                 print__chat_all_messages_debug(
#                                     f"🔍 Added queries and results to meta"
#                                 )
#                             if datasets_used:
#                                 meta_info["datasetsUsed"] = datasets_used
#                                 print__chat_all_messages_debug(
#                                     f"🔍 Added {len(datasets_used)} datasets to meta"
#                                 )
#                             if sql_query:
#                                 meta_info["sqlQuery"] = sql_query
#                                 print__chat_all_messages_debug(
#                                     f"🔍 Added SQL query to meta"
#                                 )
#                             if top_chunks:
#                                 meta_info["topChunks"] = top_chunks
#                                 print__chat_all_messages_debug(
#                                     f"🔍 Added {len(top_chunks)} chunks to meta"
#                                 )
#                             meta_info["source"] = "cached_bulk_processing"
#                         else:
#                             print__chat_all_messages_debug(
#                                 f"🔍 Processing user message - no metadata needed"
#                             )

#                         queries_results_for_frontend = None
#                         if not stored_msg["is_user"] and queries_and_results:
#                             queries_results_for_frontend = queries_and_results
#                             print__chat_all_messages_debug(
#                                 f"🔍 Set queries_results_for_frontend for AI message"
#                             )

#                         is_user_flag = stored_msg["is_user"]
#                         print__chat_all_messages_debug(
#                             f"🔍 Creating ChatMessage: isUser={is_user_flag}"
#                         )

#                         chat_message = ChatMessage(
#                             id=stored_msg["id"],
#                             threadId=thread_id,
#                             user=user_email if is_user_flag else "AI",
#                             content=stored_msg["content"],
#                             isUser=is_user_flag,
#                             createdAt=int(stored_msg["timestamp"].timestamp() * 1000),
#                             error=None,
#                             meta=meta_info if meta_info else None,
#                             queriesAndResults=queries_results_for_frontend,
#                             isLoading=False,
#                             startedAt=None,
#                             isError=False,
#                         )

#                         chat_messages.append(chat_message)
#                         print__chat_all_messages_debug(
#                             f"🔍 ChatMessage created and added to list"
#                         )

#                     print__chat_all_messages_debug(
#                         f"✅ Processed {len(chat_messages)} messages for thread {thread_id}"
#                     )
#                     return thread_id, chat_messages

#                 except Exception as e:
#                     print__chat_all_messages_debug(
#                         f"❌ Error processing thread {thread_id}: {e}"
#                     )
#                     print__chat_all_messages_debug(
#                         f"🔍 Thread processing error type: {type(e).__name__}"
#                     )
#                     print__chat_all_messages_debug(
#                         f"🔍 Thread processing error traceback: {traceback.format_exc()}"
#                     )
#                     return thread_id, []

#             MAX_CONCURRENT_BULK_THREADS = 3
#             semaphore = asyncio.Semaphore(MAX_CONCURRENT_BULK_THREADS)
#             print__chat_all_messages_debug(
#                 f"🔍 Semaphore created with {MAX_CONCURRENT_BULK_THREADS} slots"
#             )

#             async def process_single_thread_with_limit(thread_id: str):
#                 """Process a single thread with concurrency limiting."""
#                 print__chat_all_messages_debug(
#                     f"🔍 Waiting for semaphore slot for thread: {thread_id}"
#                 )
#                 async with semaphore:
#                     print__chat_all_messages_debug(
#                         f"🔍 Semaphore acquired for thread: {thread_id}"
#                     )
#                     result = await process_single_thread(thread_id)
#                     print__chat_all_messages_debug(
#                         f"🔍 Semaphore released for thread: {thread_id}"
#                     )
#                     return result

#             print__chat_all_messages_debug(
#                 f"🔒 Processing with max {MAX_CONCURRENT_BULK_THREADS} concurrent operations"
#             )

#             # Use asyncio.gather with limited concurrency
#             print__chat_all_messages_debug(
#                 f"🔍 Starting asyncio.gather for {len(user_thread_ids)} threads"
#             )
#             thread_results = await asyncio.gather(
#                 *[
#                     process_single_thread_with_limit(thread_id)
#                     for thread_id in user_thread_ids
#                 ],
#                 return_exceptions=True,
#             )
#             print__chat_all_messages_debug(
#                 f"🔍 asyncio.gather completed, processing results"
#             )

#             # Collect results
#             all_messages = {}
#             total_messages = 0

#             for i, result in enumerate(thread_results):
#                 print__chat_all_messages_debug(
#                     f"🔍 Processing thread result {i+1}/{len(thread_results)}"
#                 )
#                 if isinstance(result, Exception):
#                     print__chat_all_messages_debug(
#                         f"⚠ Exception in thread processing: {result}"
#                     )
#                     print__chat_all_messages_debug(
#                         f"🔍 Exception type: {type(result).__name__}"
#                     )
#                     print__chat_all_messages_debug(
#                         f"🔍 Exception traceback: {traceback.format_exc()}"
#                     )
#                     continue

#                 thread_id, chat_messages = result
#                 all_messages[thread_id] = chat_messages
#                 total_messages += len(chat_messages)
#                 print__chat_all_messages_debug(
#                     f"🔍 Added {len(chat_messages)} messages for thread {thread_id}"
#                 )

#             print__chat_all_messages_debug(
#                 f"✅ BULK LOADING COMPLETE: {len(all_messages)} threads, {total_messages} total messages"
#             )

#             # Simple memory check after completion
#             print__chat_all_messages_debug(f"🔍 Starting post-completion memory check")
#             log_memory_usage("bulk_complete")
#             print__chat_all_messages_debug(f"🔍 Post-completion memory check completed")

#             # Convert all ChatMessage objects to dicts for JSON serialization
#             for thread_id in all_messages:
#                 all_messages[thread_id] = [
#                     msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
#                     for msg in all_messages[thread_id]
#                 ]

#             result = {
#                 "messages": all_messages,
#                 "runIds": all_run_ids,
#                 "sentiments": all_sentiments,
#             }
#             print__chat_all_messages_debug(
#                 f"🔍 Result dictionary created with {len(result)} keys"
#             )

#             # Cache the result
#             _bulk_loading_cache[cache_key] = (result, current_time)
#             print__chat_all_messages_debug(
#                 f"💾 CACHED: Bulk result for {user_email} (expires in {BULK_CACHE_TIMEOUT}s)"
#             )

#             # Return with cache headers
#             from fastapi.responses import JSONResponse

#             response = JSONResponse(content=result)
#             response.headers["Cache-Control"] = f"public, max-age={BULK_CACHE_TIMEOUT}"
#             response.headers["ETag"] = f"bulk-{user_email}-{int(current_time)}"
#             print__chat_all_messages_debug(
#                 f"🔍 JSONResponse created with cache headers"
#             )
#             print__chat_all_messages_debug(
#                 f"🔍 CHAT_ALL_MESSAGES ENDPOINT - SUCCESSFUL EXIT"
#             )
#             return response

#         except Exception as e:
#             print__chat_all_messages_debug(
#                 f"❌ BULK ERROR: Failed to process bulk request for {user_email}: {e}"
#             )
#             print__chat_all_messages_debug(
#                 f"🔍 Main exception type: {type(e).__name__}"
#             )
#             print__chat_all_messages_debug(
#                 f"Full error traceback: {traceback.format_exc()}"
#             )

#             # Return empty result but cache it briefly to prevent error loops
#             empty_result = {"messages": {}, "runIds": {}, "sentiments": {}}
#             _bulk_loading_cache[cache_key] = (empty_result, current_time)
#             print__chat_all_messages_debug(f"🔍 Cached empty result due to error")

#             response = JSONResponse(content=empty_result, status_code=500)
#             response.headers["Cache-Control"] = (
#                 "no-cache, no-store"  # Don't cache errors
#             )
#             print__chat_all_messages_debug(
#                 f"🔍 CHAT_ALL_MESSAGES ENDPOINT - ERROR EXIT"
#             )
#             return response


# # ============================================================
# # DEBUG AND ADMIN ENDPOINTS
# # ============================================================
# @app.get("/debug/chat/{thread_id}/checkpoints")
# async def debug_checkpoints(thread_id: str, user=Depends(get_current_user)):
#     """Debug endpoint to inspect raw checkpoint data for a thread."""

#     user_email = user.get("email")
#     if not user_email:
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__debug(f"🔍 Inspecting checkpoints for thread: {thread_id}")

#     try:
#         checkpointer = await get_healthy_checkpointer()

#         if not hasattr(checkpointer, "conn"):
#             return {"error": "No PostgreSQL checkpointer available"}

#         config = {"configurable": {"thread_id": thread_id}}

#         # Get all checkpoints for this thread
#         checkpoint_tuples = []
#         try:
#             # Fix: alist() returns an async generator, don't await it
#             checkpoint_iterator = checkpointer.alist(config)
#             async for checkpoint_tuple in checkpoint_iterator:
#                 checkpoint_tuples.append(checkpoint_tuple)
#         except Exception as alist_error:
#             print__debug(f"❌ Error getting checkpoint list: {alist_error}")
#             return {"error": f"Failed to get checkpoints: {alist_error}"}

#         debug_data = {
#             "thread_id": thread_id,
#             "total_checkpoints": len(checkpoint_tuples),
#             "checkpoints": [],
#         }

#         for i, checkpoint_tuple in enumerate(checkpoint_tuples):
#             checkpoint = checkpoint_tuple.checkpoint
#             metadata = checkpoint_tuple.metadata or {}

#             checkpoint_info = {
#                 "index": i,
#                 "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get(
#                     "checkpoint_id", "unknown"
#                 ),
#                 "has_checkpoint": bool(checkpoint),
#                 "has_metadata": bool(metadata),
#                 "metadata_writes": metadata.get("writes", {}),
#                 "channel_values": {},
#             }

#             if checkpoint and "channel_values" in checkpoint:
#                 channel_values = checkpoint["channel_values"]
#                 messages = channel_values.get("messages", [])

#                 checkpoint_info["channel_values"] = {
#                     "message_count": len(messages),
#                     "messages": [],
#                 }

#                 for j, msg in enumerate(messages):
#                     msg_info = {
#                         "index": j,
#                         "type": type(msg).__name__,
#                         "id": getattr(msg, "id", None),
#                         "content_preview": (
#                             getattr(msg, "content", str(msg))[:200] + "..."
#                             if hasattr(msg, "content")
#                             and len(getattr(msg, "content", "")) > 200
#                             else getattr(msg, "content", str(msg))
#                         ),
#                         "content_length": len(getattr(msg, "content", "")),
#                     }
#                     checkpoint_info["channel_values"]["messages"].append(msg_info)

#             debug_data["checkpoints"].append(checkpoint_info)

#         return debug_data

#     except Exception as e:
#         print__debug(f"❌ Error inspecting checkpoints: {e}")
#         return {"error": str(e)}


# @app.get("/debug/pool-status")
# async def debug_pool_status():
#     """Debug endpoint to check pool status - updated for official AsyncPostgresSaver."""
#     try:
#         global GLOBAL_CHECKPOINTER

#         status = {
#             "timestamp": datetime.now().isoformat(),
#             "global_checkpointer_exists": GLOBAL_CHECKPOINTER is not None,
#             "checkpointer_type": (
#                 type(GLOBAL_CHECKPOINTER).__name__ if GLOBAL_CHECKPOINTER else None
#             ),
#         }

#         if GLOBAL_CHECKPOINTER:
#             if "AsyncPostgresSaver" in str(type(GLOBAL_CHECKPOINTER)):
#                 # Test AsyncPostgresSaver functionality instead of checking .conn
#                 try:
#                     test_config = {"configurable": {"thread_id": "pool_status_test"}}
#                     start_time = time.time()

#                     # Test a basic operation to verify the checkpointer is working
#                     result = await GLOBAL_CHECKPOINTER.aget_tuple(test_config)
#                     latency = time.time() - start_time

#                     status.update(
#                         {
#                             "asyncpostgressaver_status": "operational",
#                             "test_latency_ms": round(latency * 1000, 2),
#                             "connection_test": "passed",
#                         }
#                     )

#                 except Exception as test_error:
#                     status.update(
#                         {
#                             "asyncpostgressaver_status": "error",
#                             "test_error": str(test_error),
#                             "connection_test": "failed",
#                         }
#                     )
#             else:
#                 status.update(
#                     {
#                         "checkpointer_status": "non_postgres_type",
#                         "note": f"Using {type(GLOBAL_CHECKPOINTER).__name__} instead of AsyncPostgresSaver",
#                     }
#                 )
#         else:
#             status["checkpointer_status"] = "not_initialized"

#         return status

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e), "timestamp": datetime.now().isoformat()},
#         )


# @app.get("/debug/run-id/{run_id}")
# async def debug_run_id(run_id: str, user=Depends(get_current_user)):
#     """Debug endpoint to check if a run_id exists in the database."""

#     user_email = user.get("email")
#     if not user_email:
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     print__debug(f"🔍 Checking run_id: '{run_id}' for user: {user_email}")

#     result = {
#         "run_id": run_id,
#         "run_id_type": type(run_id).__name__,
#         "run_id_length": len(run_id) if run_id else 0,
#         "is_valid_uuid_format": False,
#         "exists_in_database": False,
#         "user_owns_run_id": False,
#         "database_details": None,
#     }

#     # Check if it's a valid UUID format
#     try:
#         uuid_obj = uuid.UUID(run_id)
#         result["is_valid_uuid_format"] = True
#         result["uuid_parsed"] = str(uuid_obj)
#     except ValueError as e:
#         result["uuid_error"] = str(e)

#     # Check if it exists in the database
#     try:
#         pool = await get_healthy_checkpointer()
#         pool = pool.conn if hasattr(pool, "conn") else None

#         if pool:
#             async with pool.connection() as conn:
#                 # 🔒 SECURITY: Check in users_threads_runs table with user ownership verification
#                 async with conn.cursor() as cur:
#                     await cur.execute(
#                         """
#                         SELECT email, thread_id, prompt, timestamp
#                         FROM users_threads_runs
#                         WHERE run_id = %s AND email = %s
#                     """,
#                         (run_id, user_email),
#                     )

#                     row = await cur.fetchone()
#                     if row:
#                         result["exists_in_database"] = True
#                         result["user_owns_run_id"] = True
#                         result["database_details"] = {
#                             "email": row[0],
#                             "thread_id": row[1],
#                             "prompt": row[2],
#                             "timestamp": row[3].isoformat() if row[3] else None,
#                         }
#                         print__debug(f"✅ User {user_email} owns run_id {run_id}")
#                     else:
#                         # Check if run_id exists but belongs to different user
#                         await cur.execute(
#                             """
#                             SELECT COUNT(*) FROM users_threads_runs WHERE run_id = %s
#                         """,
#                             (run_id,),
#                         )

#                         any_row = await cur.fetchone()
#                         if any_row and any_row[0] > 0:
#                             result["exists_in_database"] = True
#                             result["user_owns_run_id"] = False
#                             print__debug(
#                                 f"🚫 Run_id {run_id} exists but user {user_email} does not own it"
#                             )
#                         else:
#                             print__debug(f"❌ Run_id {run_id} not found in database")
#     except Exception as e:
#         result["database_error"] = str(e)

#     return result


# @app.post("/admin/clear-cache")
# async def clear_bulk_cache(user=Depends(get_current_user)):
#     """Clear the bulk loading cache (admin endpoint)."""
#     user_email = user.get("email")
#     if not user_email:
#         raise HTTPException(status_code=401, detail="User email not found in token")

#     # For now, allow any authenticated user to clear cache
#     # In production, you might want to restrict this to admin users

#     cache_entries_before = len(_bulk_loading_cache)
#     _bulk_loading_cache.clear()

#     print__memory_monitoring(
#         f"🧹 MANUAL CACHE CLEAR: {cache_entries_before} entries cleared by {user_email}"
#     )

#     # Check memory after cleanup
#     try:
#         process = psutil.Process()
#         rss_mb = process.memory_info().rss / 1024 / 1024
#         memory_status = "normal" if rss_mb < (GC_MEMORY_THRESHOLD * 0.8) else "high"
#     except:
#         rss_mb = 0
#         memory_status = "unknown"

#     return {
#         "message": "Cache cleared successfully",
#         "cache_entries_cleared": cache_entries_before,
#         "current_memory_mb": round(rss_mb, 1),
#         "memory_status": memory_status,
#         "cleared_by": user_email,
#         "timestamp": datetime.now().isoformat(),
#     }


# @app.post("/admin/clear-prepared-statements")
# async def clear_prepared_statements_endpoint(user=Depends(get_current_user)):
#     """Manually clear prepared statements (admin endpoint)."""
#     try:
#         from my_agent.utils.postgres_checkpointer import clear_prepared_statements

#         # Clear prepared statements
#         await clear_prepared_statements()

#         return {
#             "status": "success",
#             "message": "Prepared statements cleared successfully",
#             "timestamp": datetime.now().isoformat(),
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat(),
#         }


# # ============================================================
# # Placeholder Endpoint
# # ============================================================
# @app.get("/placeholder/{width}/{height}")
# async def get_placeholder_image(width: int, height: int):
#     """Generate a placeholder image with specified dimensions."""
#     try:
#         # Validate dimensions
#         width = max(1, min(width, 2000))  # Limit between 1 and 2000 pixels
#         height = max(1, min(height, 2000))

#         # Create a simple SVG placeholder
#         svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
#             <rect width="100%" height="100%" fill="#e5e7eb"/>
#             <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="20">{width}x{height}</text>
#         </svg>"""

#         from fastapi.responses import Response

#         return Response(
#             content=svg_content,
#             media_type="image/svg+xml",
#             headers={
#                 "Cache-Control": "public, max-age=3600",
#                 "Access-Control-Allow-Origin": "*",
#             },
#         )

#     except Exception as e:
#         # Fallback for any errors
#         simple_svg = f"""<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
#             <rect width="100%" height="100%" fill="#f3f4f6"/>
#             <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#6b7280" font-size="12">Error</text>
#         </svg>"""

#         from fastapi.responses import Response

#         return Response(
#             content=simple_svg,
#             media_type="image/svg+xml",
#             headers={
#                 "Cache-Control": "public, max-age=3600",
#                 "Access-Control-Allow-Origin": "*",
#             },
#         )

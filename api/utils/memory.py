# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

import asyncio

# Standard imports
import gc
import json
import signal
import time
import traceback
from collections import defaultdict
from datetime import datetime

import psutil
from fastapi import Request

# Import global variables from api.config.settings
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    GC_MEMORY_THRESHOLD,
    _bulk_loading_cache,
)

# Import debug functions from utils
from api.utils.debug import print__memory_monitoring


# ============================================================
# UTILITY FUNCTIONS - MEMORY MANAGEMENT
# ============================================================
def cleanup_bulk_cache():
    """Clean up expired cache entries."""
    current_time = time.time()
    expired_keys = []

    for cache_key, (cached_data, cache_time) in _bulk_loading_cache.items():
        if current_time - cache_time > BULK_CACHE_TIMEOUT:
            expired_keys.append(cache_key)

    for key in expired_keys:
        del _bulk_loading_cache[key]

    return len(expired_keys)


def check_memory_and_gc():
    """Enhanced memory check with cache cleanup and scaling strategy."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024

        # Clean up cache first if memory is getting high
        if rss_mb > (GC_MEMORY_THRESHOLD * 0.8):  # At 80% of threshold
            print__memory_monitoring(
                f"📊 Memory at {rss_mb:.1f}MB (80% of {GC_MEMORY_THRESHOLD}MB threshold) - cleaning cache"
            )
            cleaned_entries = cleanup_bulk_cache()
            if cleaned_entries > 0:
                # Check memory after cache cleanup
                new_memory = psutil.Process().memory_info().rss / 1024 / 1024
                freed = rss_mb - new_memory
                print__memory_monitoring(
                    f"🧹 Cache cleanup freed {freed:.1f}MB, cleaned {cleaned_entries} entries"
                )
                rss_mb = new_memory

        # Trigger GC only if above threshold
        if rss_mb > GC_MEMORY_THRESHOLD:
            print__memory_monitoring(
                f"🚨 MEMORY THRESHOLD EXCEEDED: {rss_mb:.1f}MB > {GC_MEMORY_THRESHOLD}MB - running GC"
            )
            import gc

            collected = gc.collect()
            print__memory_monitoring(f"🧹 GC collected {collected} objects")

            # Log memory after GC
            new_memory = psutil.Process().memory_info().rss / 1024 / 1024
            freed = rss_mb - new_memory
            print__memory_monitoring(
                f"🧹 Memory after GC: {new_memory:.1f}MB (freed: {freed:.1f}MB)"
            )

            # If memory is still high after GC, provide scaling guidance
            if new_memory > (GC_MEMORY_THRESHOLD * 0.9):
                thread_count = len(_bulk_loading_cache)
                print__memory_monitoring(
                    f"⚠ HIGH MEMORY WARNING: {new_memory:.1f}MB after GC"
                )
                print__memory_monitoring(f"📊 Current cache entries: {thread_count}")
                if thread_count > 20:
                    print__memory_monitoring(
                        f"💡 SCALING TIP: Consider implementing pagination for chat threads"
                    )

        return rss_mb

    except Exception as e:
        print__memory_monitoring(f"❌ Could not check memory: {e}")
        return 0


def log_memory_usage(context: str = ""):
    """Simplified memory logging."""
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024

        print__memory_monitoring(
            f"📊 Memory usage{f' [{context}]' if context else ''}: {rss_mb:.1f}MB RSS"
        )

        # Simple threshold check
        if rss_mb > GC_MEMORY_THRESHOLD:
            check_memory_and_gc()

    except Exception as e:
        print__memory_monitoring(f"❌ Could not check memory usage: {e}")


def log_comprehensive_error(context: str, error: Exception, request: Request = None):
    """Simplified error logging."""
    # Import debug function
    from api.utils.debug import print__debug

    error_details = {
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
    }

    if request:
        error_details.update(
            {
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
            }
        )

    # Log to debug output
    print__debug(f"🚨 ERROR: {json.dumps(error_details, indent=2)}")


def setup_graceful_shutdown():
    """Setup graceful shutdown handlers."""

    def signal_handler(signum, frame):
        print__memory_monitoring(
            f"📡 Received signal {signum} - preparing for graceful shutdown..."
        )
        log_memory_usage("shutdown_signal")

    # Register signal handlers for common restart signals
    signal.signal(signal.SIGTERM, signal_handler)  # Most common for container restarts
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal_handler)  # User-defined signal


async def perform_deletion_operations(conn, user_email: str, thread_id: str):
    """Perform the actual deletion operations on the given connection."""
    # Import debug function
    from api.utils.debug import print__api_postgresql

    print__api_postgresql(f"🔧 DEBUG: Starting deletion operations...")

    print__api_postgresql(f"🔧 DEBUG: Setting autocommit...")
    await conn.set_autocommit(True)
    print__api_postgresql(f"🔧 DEBUG: Autocommit set successfully")

    # 🔒 SECURITY CHECK: Verify user owns this thread before deleting
    print__api_postgresql(
        f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}"
    )

    print__api_postgresql(f"🔧 DEBUG: Creating cursor for ownership check...")
    async with conn.cursor() as cur:
        print__api_postgresql(f"🔧 DEBUG: Cursor created, executing ownership query...")
        # Fix: Use correct psycopg approach with fetchone() instead of fetchval()
        await cur.execute(
            """
            SELECT COUNT(*) FROM users_threads_runs 
            WHERE email = %s AND thread_id = %s
        """,
            (user_email, thread_id),
        )

        # Get the result row and extract the count value
        result_row = await cur.fetchone()
        # Fix: psycopg Row objects don't support [0] indexing, convert to tuple first
        thread_entries_count = tuple(result_row)[0] if result_row else 0
        print__api_postgresql(
            f"🔧 DEBUG: Ownership check complete, count: {thread_entries_count}"
        )

    if thread_entries_count == 0:
        print__api_postgresql(
            f"🚫 SECURITY: User {user_email} does not own thread {thread_id} - deletion denied"
        )
        return {
            "message": "Thread not found or access denied",
            "thread_id": thread_id,
            "user_email": user_email,
            "deleted_counts": {},
        }

    print__api_postgresql(
        f"✅ SECURITY: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - deletion authorized"
    )

    print__api_postgresql(f"🔄 Deleting from checkpoint tables for thread {thread_id}")

    # Delete from all checkpoint tables
    tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
    deleted_counts = {}

    for table in tables:
        try:
            print__api_postgresql(f"🔧 DEBUG: Processing table {table}...")
            # First check if the table exists
            print__api_postgresql(
                f"🔧 DEBUG: Creating cursor for table existence check..."
            )
            async with conn.cursor() as cur:
                print__api_postgresql(
                    f"🔧 DEBUG: Executing table existence query for {table}..."
                )
                # Fix: Use correct psycopg approach with fetchone() instead of fetchval()
                await cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """,
                    (table,),
                )

                # Get the result row and extract the boolean value
                result_row = await cur.fetchone()
                # Fix: psycopg Row objects don't support [0] indexing, convert to tuple first
                table_exists = tuple(result_row)[0] if result_row else False
                print__api_postgresql(
                    f"🔧 DEBUG: Table {table} exists check result: {table_exists}"
                )

                # Simple boolean check
                if not table_exists:
                    print__api_postgresql(f"⚠ Table {table} does not exist, skipping")
                    deleted_counts[table] = 0
                    continue

                print__api_postgresql(
                    f"🔧 DEBUG: Table {table} exists, proceeding with deletion..."
                )
                # Delete records for this thread_id
                print__api_postgresql(
                    f"🔧 DEBUG: Creating cursor for deletion from {table}..."
                )
                async with conn.cursor() as del_cur:
                    print__api_postgresql(
                        f"🔧 DEBUG: Executing DELETE query for {table}..."
                    )
                    await del_cur.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
                    )

                    deleted_counts[table] = (
                        del_cur.rowcount if hasattr(del_cur, "rowcount") else 0
                    )
                    print__api_postgresql(
                        f"✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}"
                    )

        except Exception as table_error:
            print__api_postgresql(f"⚠ Error deleting from table {table}: {table_error}")
            print__api_postgresql(
                f"🔧 DEBUG: Table error type: {type(table_error).__name__}"
            )
            print__api_postgresql(
                f"🔧 DEBUG: Table error traceback: {traceback.format_exc()}"
            )
            deleted_counts[table] = f"Error: {str(table_error)}"

    # Delete from users_threads_runs table directly within the same transaction
    print__api_postgresql(
        f"🔄 Deleting thread entries from users_threads_runs for user {user_email}, thread {thread_id}"
    )
    try:
        print__api_postgresql(
            "🔧 DEBUG: Creating cursor for users_threads_runs deletion..."
        )
        async with conn.cursor() as cur:
            print__api_postgresql(
                f"🔧 DEBUG: Executing DELETE query for users_threads_runs..."
            )
            await cur.execute(
                """
                DELETE FROM users_threads_runs 
                WHERE email = %s AND thread_id = %s
            """,
                (user_email, thread_id),
            )

            users_threads_runs_deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
            print__api_postgresql(
                f"✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}"
            )

            deleted_counts["users_threads_runs"] = users_threads_runs_deleted

    except Exception as e:
        print__api_postgresql(f"❌ Error deleting from users_threads_runs: {e}")
        print__api_postgresql(
            f"🔧 DEBUG: users_threads_runs error type: {type(e).__name__}"
        )
        print__api_postgresql(
            f"🔧 DEBUG: users_threads_runs error traceback: {traceback.format_exc()}"
        )
        deleted_counts["users_threads_runs"] = f"Error: {str(e)}"

    result_data = {
        "message": f"Checkpoint records and thread entries deleted for thread_id: {thread_id}",
        "deleted_counts": deleted_counts,
        "thread_id": thread_id,
        "user_email": user_email,
    }

    print__api_postgresql(
        f"🎉 Successfully deleted thread {thread_id} for user {user_email}"
    )
    return result_data

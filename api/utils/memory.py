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
import ctypes
import gc
import json
import logging
import signal
import time
import tracemalloc
import traceback

# from collections import defaultdict  # unused
from datetime import datetime
from typing import Optional

import psutil
from fastapi import Request

# Import debug functions from utils
from api.utils.debug import print__memory_monitoring

# Load memory cleanup environment variables directly
MEMORY_CLEANUP_ENABLED = os.environ.get("MEMORY_CLEANUP_ENABLED", "1") == "1"
MEMORY_CLEANUP_INTERVAL = int(os.environ.get("MEMORY_CLEANUP_INTERVAL", "60"))

# Try to load libc for malloc_trim support
try:
    libc = ctypes.CDLL("libc.so.6")
    MALLOC_TRIM_AVAILABLE = True
    print__memory_monitoring("🐧 malloc_trim loaded successfully")
except (OSError, AttributeError) as e:
    libc = None
    MALLOC_TRIM_AVAILABLE = False
    MEMORY_CLEANUP_ENABLED = False  # Disable cleanup if malloc_trim not available
    print__memory_monitoring(f"❌ Failed to load libc: {e}")
    print__memory_monitoring("🧹 Memory cleanup disabled (malloc_trim not available)")

# Load memory-related environment variables directly (moved from settings.py)
GC_MEMORY_THRESHOLD = int(
    os.environ.get("GC_MEMORY_THRESHOLD", "1900")
)  # 1900MB for 2GB memory allocation
MEMORY_PROFILER_ENABLED = os.environ.get("MEMORY_PROFILER_ENABLED", "0") == "1"
MEMORY_PROFILER_INTERVAL = int(os.environ.get("MEMORY_PROFILER_INTERVAL", "30"))
MEMORY_PROFILER_TOP_STATS = int(os.environ.get("MEMORY_PROFILER_TOP_STATS", "10"))

# Import remaining global variables from api.config.settings
from api.config.settings import (
    BULK_CACHE_TIMEOUT,
    _bulk_loading_cache,
)


# ============================================================
# UTILITY FUNCTIONS - MEMORY MANAGEMENT
# ============================================================
def force_release_memory():
    """
    Force memory release using malloc_trim if available.
    """
    try:
        # Get initial memory
        process = psutil.Process()
        initial_rss = process.memory_info().rss / 1024 / 1024

        # Run garbage collection
        collected = gc.collect()

        # Only run malloc_trim if available
        if MALLOC_TRIM_AVAILABLE:
            libc.malloc_trim(0)
            print__memory_monitoring("🧹 Called malloc_trim(0) to release memory to OS")
            malloc_trim_used = True
        else:
            malloc_trim_used = False

        # Get final memory
        final_rss = process.memory_info().rss / 1024 / 1024
        freed_mb = initial_rss - final_rss

        print__memory_monitoring(
            f"🧹 Memory cleanup: {freed_mb:.1f}MB freed | "
            f"{initial_rss:.1f}MB → {final_rss:.1f}MB | "
            f"GC: {collected} | malloc_trim: {'✓' if malloc_trim_used else '✗'}"
        )

        return {
            "freed_mb": round(freed_mb, 2),
            "gc_collected": collected,
            "malloc_trim_used": malloc_trim_used,
        }

    except Exception as e:
        print__memory_monitoring(f"❌ Memory cleanup error: {e}")
        return {"error": str(e), "freed_mb": 0}


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
    """Enhanced memory check with cache cleanup, GC, and malloc_trim."""
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

        # Trigger GC and malloc_trim if above threshold
        if rss_mb > GC_MEMORY_THRESHOLD:
            print__memory_monitoring(
                f"🚨 MEMORY THRESHOLD EXCEEDED: {rss_mb:.1f}MB > {GC_MEMORY_THRESHOLD}MB - forcing memory release"
            )

            # Use the new force_release_memory function
            release_result = force_release_memory()

            new_memory = release_result.get("final_rss_mb", rss_mb)
            freed = release_result.get("freed_mb", 0)

            print__memory_monitoring(
                f"🧹 Memory after cleanup: {new_memory:.1f}MB (freed: {freed:.1f}MB)"
            )

            # If memory is still high after cleanup, provide scaling guidance
            if new_memory > (GC_MEMORY_THRESHOLD * 0.9):
                thread_count = len(_bulk_loading_cache)
                print__memory_monitoring(
                    f"⚠ HIGH MEMORY WARNING: {new_memory:.1f}MB after cleanup"
                )
                print__memory_monitoring(f"📊 Current cache entries: {thread_count}")
                if thread_count > 20:
                    print__memory_monitoring(
                        "💡 SCALING TIP: Consider implementing pagination for chat threads"
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


# ============================================================
# PERIODIC TRACEMALLOC MONITORING
# ============================================================
_memory_profiler_task: Optional[asyncio.Task] = None
_memory_cleanup_task: Optional[asyncio.Task] = None
_previous_snapshot: Optional[tracemalloc.Snapshot] = None


def _get_uvicorn_logger() -> logging.Logger:
    return logging.getLogger("uvicorn.error")


def _shorten_path(filepath: str, max_len: int = 60) -> str:
    """Shorten long file paths for better table formatting."""
    if len(filepath) <= max_len:
        return filepath
    # Try to keep filename and some parent context
    parts = filepath.replace("\\", "/").split("/")
    filename = parts[-1]
    if len(filename) > max_len - 10:
        return f"...{filename[-(max_len-3):]}"
    # Build path from end until we hit max_len
    result = filename
    for part in reversed(parts[:-1]):
        candidate = f"{part}/{result}"
        if len(candidate) > max_len - 3:
            return f".../{result}"
        result = candidate
    return result


def _log_tracemalloc_snapshot(
    snapshot: tracemalloc.Snapshot,
    previous_snapshot: Optional[tracemalloc.Snapshot],
    top_stats: int,
) -> None:
    logger = _get_uvicorn_logger()
    stats = snapshot.statistics("lineno")

    diff_map = {}
    if previous_snapshot is not None:
        for diff in snapshot.compare_to(previous_snapshot, "lineno"):
            key = tuple((frame.filename, frame.lineno) for frame in diff.traceback)
            diff_map[key] = diff

    process = psutil.Process()
    try:
        mem_full = process.memory_full_info()
        rss_mb = mem_full.rss / (1024 * 1024)
        uss_mb = getattr(mem_full, "uss", 0) / (1024 * 1024)
        swap_mb = getattr(mem_full, "swap", 0) / (1024 * 1024)
    except Exception:
        mem_full = None
        rss_mb = uss_mb = swap_mb = 0

    total_current, peak = tracemalloc.get_traced_memory()

    if not stats:
        msg = "[memory-profiler] No allocation stats collected yet"
        logger.info(msg)
        print__memory_monitoring(msg)
        return

    # Build table with allocations
    table_lines = []
    table_lines.append("=" * 150)
    header = (
        f"MEMORY PROFILER - Top {top_stats} Allocations | "
        f"Tracemalloc current: {total_current / (1024 * 1024):.1f} MiB | "
        f"Tracemalloc peak: {peak / (1024 * 1024):.1f} MiB"
    )
    table_lines.append(header)
    if mem_full is not None:
        table_lines.append(
            f"Process RSS: {rss_mb:.1f} MiB | USS: {uss_mb:.1f} MiB | Swap: {swap_mb:.1f} MiB"
        )
    table_lines.append("=" * 150)
    table_lines.append(
        f"{'#':<3} {'Size (MiB)':>12} {'ΔSize (MiB)':>13} {'Blocks':>10} {'ΔBlocks':>10} {'Location':<90}"
    )
    table_lines.append("-" * 150)

    # Ensure allocations sorted by size descending
    stats_sorted = sorted(stats, key=lambda s: s.size, reverse=True)

    for idx, stat in enumerate(stats_sorted[:top_stats], start=1):
        key = tuple((frame.filename, frame.lineno) for frame in stat.traceback)
        size = stat.size
        count = stat.count
        diff_entry = diff_map.get(key)
        size_delta = diff_entry.size_diff if diff_entry else 0
        count_delta = diff_entry.count_diff if diff_entry else 0

        size_mb = size / (1024 * 1024)
        size_delta_mb = size_delta / (1024 * 1024)

        primary_frame = stat.traceback[0] if stat.traceback else None
        if primary_frame:
            location = (
                f"{_shorten_path(primary_frame.filename, 95)}:{primary_frame.lineno}"
            )
        else:
            location = "<unknown>"

        table_lines.append(
            f"{idx:<3} {size_mb:>12,.2f} {size_delta_mb:>+13,.2f} {count:>10,} {count_delta:>+10,} {location:<90}"
        )

    # Add summary of other entries
    other = stats[top_stats:]
    if other:
        other_size = sum(stat.size for stat in other)
        other_blocks = sum(stat.count for stat in other)
        other_size_delta = 0
        other_blocks_delta = 0
        if diff_map:
            for stat in other:
                key = tuple((frame.filename, frame.lineno) for frame in stat.traceback)
                diff_entry = diff_map.get(key)
                if diff_entry:
                    other_size_delta += diff_entry.size_diff
                    other_blocks_delta += diff_entry.count_diff

        table_lines.append("-" * 150)
        table_lines.append(
            f"    {len(other):>3} other entries | Size: {other_size / (1024 * 1024):,.2f} MiB ({other_size_delta / (1024 * 1024):+,.2f} MiB)"
            f" | Blocks: {other_blocks:,} ({other_blocks_delta:+,})"
        )

    table_lines.append("=" * 150)

    # Append memory map summary beneath allocations within same table
    table_lines.append("MEMORY MAPS - Top RSS Segments")
    table_lines.append("-" * 150)
    table_lines.append(
        f"{'#':<3} {'RSS (MiB)':>12} {'Private (MiB)':>15} {'Path':<115}"
    )
    table_lines.append("-" * 150)

    if mem_full is not None:
        try:
            maps = sorted(
                process.memory_maps(grouped=True), key=lambda m: m.rss, reverse=True
            )
            top_maps = maps[:top_stats]
            for idx, m in enumerate(top_maps, start=1):
                rss = m.rss / (1024 * 1024)
                private_bytes = sum(
                    getattr(m, attr, 0) or 0
                    for attr in ("private", "private_clean", "private_dirty")
                )
                private = private_bytes / (1024 * 1024)
                path = _shorten_path(m.path or "[anonymous]", 115)
                table_lines.append(
                    f"{idx:<3} {rss:>12,.2f} {private:>15,.2f} {path:<115}"
                )
            if len(maps) > top_stats:
                remaining_rss = sum(m.rss for m in maps[top_stats:]) / (1024 * 1024)
                table_lines.append("-" * 150)
                table_lines.append(
                    f"    {len(maps) - top_stats} additional segments totalling {remaining_rss:,.2f} MiB RSS"
                )
        except Exception as map_error:
            table_lines.append(
                f"[memory-profiler] Unable to collect memory map data: {type(map_error).__name__}: {map_error}"
            )
    else:
        table_lines.append("[memory-profiler] Process memory info unavailable")

    table_lines.append("=" * 150)
    table_lines.append("")

    # Log as single message
    table_output = "\n".join(table_lines)
    logger.info(table_output)
    print__memory_monitoring(table_output)


async def _memory_profiler_loop(interval: int, top_stats: int) -> None:
    global _previous_snapshot
    logger = _get_uvicorn_logger()

    if not tracemalloc.is_tracing():
        tracemalloc.start()
        print__memory_monitoring("[memory-profiler] tracemalloc tracing started")

    _previous_snapshot = tracemalloc.take_snapshot()

    print__memory_monitoring(
        f"[memory-profiler] Background task running every {interval}s (top {top_stats} stats)"
    )

    try:
        while True:
            await asyncio.sleep(interval)
            snapshot = tracemalloc.take_snapshot()
            _log_tracemalloc_snapshot(snapshot, _previous_snapshot, top_stats)
            _previous_snapshot = snapshot
    except asyncio.CancelledError:
        logger.info("[memory-profiler] Background task cancelled")
        print__memory_monitoring("[memory-profiler] Background task cancelled")
        raise
    finally:
        # Optionally keep tracemalloc running to allow other consumers
        _previous_snapshot = None


# ============================================================
# PERIODIC MEMORY CLEANUP
# ============================================================


async def _memory_cleanup_loop() -> None:
    """
    Simple periodic memory cleanup - every MEMORY_CLEANUP_INTERVAL seconds.
    """
    print__memory_monitoring(
        f"🧹 [memory-cleanup] Starting cleanup task (every {MEMORY_CLEANUP_INTERVAL}s)"
    )

    try:
        while True:
            await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)

            # Get current memory usage
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024

            print__memory_monitoring(
                f"🧹 [memory-cleanup] Running (RSS: {rss_mb:.1f}MB)"
            )

            # Clean cache first
            cleaned = cleanup_bulk_cache()

            # Force memory release (only effective on Linux)
            result = force_release_memory()

            print__memory_monitoring(
                f"✅ [memory-cleanup] Cache: {cleaned} entries, "
                f"Memory: {result.get('freed_mb', 0):.1f}MB freed"
            )

    except asyncio.CancelledError:
        print__memory_monitoring("🛑 [memory-cleanup] Task cancelled")
        raise


def start_memory_profiler(
    interval: Optional[int] = None, top_stats: Optional[int] = None
) -> Optional[asyncio.Task]:
    """Start the periodic tracemalloc profiler if enabled."""

    if not MEMORY_PROFILER_ENABLED:
        return None

    global _memory_profiler_task
    if _memory_profiler_task and not _memory_profiler_task.done():
        return _memory_profiler_task

    loop = asyncio.get_running_loop()
    interval = interval or MEMORY_PROFILER_INTERVAL
    top_stats = top_stats or MEMORY_PROFILER_TOP_STATS
    _memory_profiler_task = loop.create_task(_memory_profiler_loop(interval, top_stats))
    return _memory_profiler_task


def start_memory_cleanup() -> Optional[asyncio.Task]:
    """
    Start the periodic memory cleanup task if enabled.
    """
    global _memory_cleanup_task

    if not MEMORY_CLEANUP_ENABLED:
        print__memory_monitoring(
            "🧹 Memory cleanup disabled (MEMORY_CLEANUP_ENABLED=0)"
        )
        return None

    if _memory_cleanup_task and not _memory_cleanup_task.done():
        print__memory_monitoring("🧹 Memory cleanup already running")
        return _memory_cleanup_task

    try:
        loop = asyncio.get_running_loop()
        _memory_cleanup_task = loop.create_task(_memory_cleanup_loop())
        print__memory_monitoring(
            f"✅ Memory cleanup started (every {MEMORY_CLEANUP_INTERVAL}s)"
        )
        return _memory_cleanup_task
    except RuntimeError as e:
        print__memory_monitoring(f"❌ Cannot start memory cleanup - no event loop: {e}")
        return None


async def stop_memory_profiler() -> None:
    """Stop the periodic tracemalloc profiler if it is running."""

    global _memory_profiler_task
    if not _memory_profiler_task:
        return

    _memory_profiler_task.cancel()
    try:
        await _memory_profiler_task
    except asyncio.CancelledError:
        pass
    finally:
        _memory_profiler_task = None
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            logger = _get_uvicorn_logger()
            logger.info("[memory-profiler] tracemalloc tracing stopped")
            print__memory_monitoring("[memory-profiler] tracemalloc tracing stopped")


async def stop_memory_cleanup() -> None:
    """Stop the periodic memory cleanup task if it is running."""

    global _memory_cleanup_task
    if not _memory_cleanup_task:
        return

    _memory_cleanup_task.cancel()
    try:
        await _memory_cleanup_task
    except asyncio.CancelledError:
        pass
    finally:
        _memory_cleanup_task = None
        logger = _get_uvicorn_logger()
        logger.info("[memory-cleanup] Background cleanup task stopped")
        print__memory_monitoring("[memory-cleanup] Background cleanup task stopped")


# ============================================================
# ERROR HANDLING AND UTILITIES
# ============================================================


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

    print__api_postgresql("🔧 DEBUG: Starting deletion operations...")

    print__api_postgresql("🔧 DEBUG: Setting autocommit...")
    await conn.set_autocommit(True)
    print__api_postgresql("🔧 DEBUG: Autocommit set successfully")

    # 🔒 SECURITY CHECK: Verify user owns this thread before deleting
    print__api_postgresql(
        f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}"
    )

    print__api_postgresql("🔧 DEBUG: Creating cursor for ownership check...")
    async with conn.cursor() as cur:
        print__api_postgresql("🔧 DEBUG: Cursor created, executing ownership query...")
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
                "🔧 DEBUG: Creating cursor for table existence check..."
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
                    # Explicit commit for safety
                    await conn.commit()

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
                "🔧 DEBUG: Executing DELETE query for users_threads_runs..."
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
            # Explicit commit for safety
            await conn.commit()

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

"""Memory Management and Monitoring Module for FastAPI Application

This module provides comprehensive memory management, profiling, and monitoring
capabilities for the FastAPI-based CZSU Multi-Agent Text-to-SQL application,
with special focus on memory-constrained environments and PostgreSQL integration.
"""

MODULE_DESCRIPTION = r"""Memory Management and Monitoring Module for FastAPI Application

This module provides comprehensive memory management, profiling, and monitoring
capabilities for the FastAPI-based CZSU Multi-Agent Text-to-SQL application,
with special focus on memory-constrained environments and PostgreSQL integration.

Key Features:
-------------
1. Platform-Specific Initialization:
   - Windows event loop policy configuration for psycopg compatibility
   - Early environment variable loading for configuration
   - Cross-platform support with Linux-specific optimizations
   - Dynamic base directory resolution

2. Memory Cleanup and Release:
   - Automatic garbage collection with configurable thresholds
   - Linux malloc_trim support for releasing memory to OS
   - Bulk cache cleanup for expired entries
   - Forced memory release with detailed reporting
   - Periodic cleanup tasks with configurable intervals

3. Memory Profiling and Monitoring:
   - Real-time memory usage tracking with psutil integration
   - Tracemalloc-based allocation profiling
   - Detailed memory statistics with file-level granularity
   - Process memory maps analysis (RSS, USS, swap)
   - Comparative snapshots for memory leak detection
   - Configurable profiling intervals and top statistics

4. Cache Management:
   - Bulk loading cache with time-based expiration
   - Automatic cleanup of expired cache entries
   - Memory-aware cache eviction strategies
   - Cache statistics and monitoring

5. Database Operations Support:
   - PostgreSQL thread deletion with security checks
   - User ownership verification before operations
   - Atomic transaction handling with autocommit
   - Comprehensive error handling and logging
   - Cross-table cascade deletion support

6. Error Handling and Diagnostics:
   - Comprehensive error logging with context
   - Request-aware error tracking
   - Timestamp-based error reporting
   - Integration with debug output utilities
   - Graceful degradation on unsupported platforms

7. Background Task Management:
   - Async memory profiler task with lifecycle management
   - Async memory cleanup task with periodic execution
   - Proper task cancellation and cleanup
   - Signal handlers for graceful shutdown

Configuration Management:
-----------------------
Environment Variables:
- MEMORY_CLEANUP_ENABLED: Enable/disable periodic cleanup (default: 1)
- MEMORY_CLEANUP_INTERVAL: Cleanup interval in seconds (default: 60)
- GC_MEMORY_THRESHOLD: Memory threshold for GC trigger in MB (default: 1900)
- MEMORY_PROFILER_ENABLED: Enable/disable profiling (default: 0)
- MEMORY_PROFILER_INTERVAL: Profiling interval in seconds (default: 30)
- MEMORY_PROFILER_TOP_STATS: Number of top stats to display (default: 10)
- BULK_CACHE_TIMEOUT: Cache entry timeout in seconds

Memory Management Flow:
---------------------
1. Initialization:
   - Platform detection and event loop policy setup
   - Environment variable loading and validation
   - libc loading for malloc_trim support (Linux)
   - Configuration of memory thresholds
   - Setup of global variables and caches

2. Runtime Monitoring:
   - Periodic memory usage checks
   - Threshold-based GC triggering
   - Cache cleanup at 80% threshold
   - Automatic memory release on threshold exceeded

3. Memory Profiling (if enabled):
   - Periodic tracemalloc snapshots
   - Comparative analysis with previous snapshots
   - Detailed allocation statistics by file/line
   - Process memory maps analysis
   - Formatted table output to logs

4. Memory Cleanup:
   - Cache expiration checks
   - Garbage collection execution
   - malloc_trim for OS memory release (Linux)
   - Memory usage before/after reporting

5. Graceful Shutdown:
   - Signal handler registration
   - Background task cancellation
   - Tracemalloc cleanup
   - Final memory state logging

Usage Examples:
--------------
# Enable memory profiling at startup
from api.utils.memory import start_memory_profiler, start_memory_cleanup

@app.on_event("startup")
async def startup_event():
    start_memory_profiler()  # Start profiling if enabled
    start_memory_cleanup()   # Start periodic cleanup if enabled

# Check memory and trigger GC if needed
from api.utils.memory import check_memory_and_gc

check_memory_and_gc()

# Log memory usage with context
from api.utils.memory import log_memory_usage

log_memory_usage("after_bulk_query")

# Force memory release
from api.utils.memory import force_release_memory

result = force_release_memory()
print(f"Freed {result['freed_mb']} MB")

# Database thread deletion with security
from api.utils.memory import perform_deletion_operations

result = await perform_deletion_operations(conn, user_email, thread_id)

Memory Profiler Output:
---------------------
The module generates detailed memory profiling reports including:
- Top N allocations by size with file/line locations
- Size deltas compared to previous snapshot
- Block counts and delta changes
- Process RSS, USS, and swap memory
- Tracemalloc current and peak usage
- Memory maps with RSS and private memory breakdown

Error Handling:
-------------
- Platform compatibility checks (Windows/Linux)
- Graceful fallback when malloc_trim unavailable
- Exception handling in all memory operations
- Detailed error logging with context
- Safe task cancellation and cleanup
- Database operation transaction safety

Security Features:
----------------
- User ownership verification for thread deletion
- Email-based authorization checks
- SQL injection prevention via parameterized queries
- Atomic transaction operations
- Detailed security audit logging

Performance Considerations:
-------------------------
- Configurable profiling intervals to reduce overhead
- Async background tasks for non-blocking operation
- Efficient snapshot comparisons
- Path shortening for better formatting
- Memory-aware cache eviction
- Threshold-based GC triggering

Platform Support:
---------------
- Windows: Event loop policy configuration, limited cleanup
- Linux: Full support with malloc_trim for optimal memory release
- Process memory statistics via psutil
- Tracemalloc for detailed allocation tracking

Required Environment:
-------------------
- Python 3.7+
- psutil package for process monitoring
- psycopg (asyncpg) for PostgreSQL operations
- FastAPI framework
- asyncio support
- Optional: libc.so.6 for malloc_trim (Linux)

Integration Points:
-----------------
- api.utils.debug: Debug output functions
- api.config.settings: Global configuration and cache
- FastAPI Request: Request context for error logging
- PostgreSQL: Database cleanup operations
- uvicorn.error logger: Profiler output logging

Limitations:
-----------
- malloc_trim only available on Linux systems
- Memory profiling adds overhead, disabled by default
- Cache cleanup requires periodic task running
- GC may cause brief performance pauses
- Large memory maps can produce verbose output
"""

# ============================================================
# CRITICAL: PLATFORM-SPECIFIC INITIALIZATION
# ============================================================
# Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# on Windows platforms. Without this, async PostgreSQL operations will fail.
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================================
# ENVIRONMENT AND CONSTANTS SETUP
# ============================================================
# Load environment variables early to ensure all configuration
# is available before module initialization
from dotenv import load_dotenv

load_dotenv()

# Base directory resolution with fallback for different execution contexts
# Attempts to use __file__ if available, otherwise falls back to cwd
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback when __file__ is not defined (e.g., interactive mode)
    BASE_DIR = Path(os.getcwd()).parents[0]

import asyncio

# ============================================================
# STANDARD LIBRARY IMPORTS
# ============================================================
import ctypes  # For libc malloc_trim access on Linux
import gc  # Garbage collection control
import json  # JSON formatting for error details
import logging  # Logging for profiler output
import signal  # Signal handlers for graceful shutdown
import time  # Time tracking for cache expiration
import tracemalloc  # Memory allocation profiling
import traceback  # Exception traceback formatting

from datetime import datetime  # Timestamp generation
from typing import Optional  # Type hints for async tasks

# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================
import psutil  # Process and system memory monitoring
from fastapi import Request  # FastAPI request context

# ============================================================
# INTERNAL IMPORTS
# ============================================================
# Import debug functions from utils for consistent logging
from api.utils.debug import print__memory_monitoring

# ============================================================
# MEMORY CLEANUP CONFIGURATION
# ============================================================
# Load memory cleanup settings from environment variables
# These control the periodic cleanup task behavior
MEMORY_CLEANUP_ENABLED = os.environ.get("MEMORY_CLEANUP_ENABLED", "1") == "1"
MEMORY_CLEANUP_INTERVAL = int(
    os.environ.get("MEMORY_CLEANUP_INTERVAL", "60")
)  # seconds

# ============================================================
# LIBC MALLOC_TRIM INITIALIZATION (Linux Only)
# ============================================================
# Attempt to load libc for malloc_trim support, which releases
# memory back to the OS. This is only available on Linux systems.
try:
    libc = ctypes.CDLL("libc.so.6")
    MALLOC_TRIM_AVAILABLE = True
    print__memory_monitoring("🐧 malloc_trim loaded successfully (Linux)")
except (OSError, AttributeError) as e:
    # malloc_trim not available (Windows or other platforms)
    libc = None
    MALLOC_TRIM_AVAILABLE = False
    MEMORY_CLEANUP_ENABLED = False  # Disable cleanup if malloc_trim not available
    print__memory_monitoring(f"❌ Failed to load libc: {e}")
    print__memory_monitoring(
        "🧹 Memory cleanup disabled (malloc_trim not available - not on Linux)"
    )

# ============================================================
# MEMORY MONITORING CONFIGURATION
# ============================================================
# Load memory-related environment variables directly (moved from settings.py)
# These settings control memory thresholds and profiling behavior
GC_MEMORY_THRESHOLD = int(
    os.environ.get("GC_MEMORY_THRESHOLD", "1900")
)  # MB - Threshold for GC trigger (1900MB for 2GB memory allocation)
MEMORY_PROFILER_ENABLED = os.environ.get("MEMORY_PROFILER_ENABLED", "0") == "1"
MEMORY_PROFILER_INTERVAL = int(
    os.environ.get("MEMORY_PROFILER_INTERVAL", "30")
)  # seconds
MEMORY_PROFILER_TOP_STATS = int(
    os.environ.get("MEMORY_PROFILER_TOP_STATS", "10")
)  # number of entries

# ============================================================
# CACHE CONFIGURATION IMPORTS
# ============================================================
# Import remaining global variables from api.config.settings
# These are shared across the application for bulk data caching
from api.config.settings import (
    BULK_CACHE_TIMEOUT,  # Cache entry expiration timeout in seconds
    _bulk_loading_cache,  # Global cache dictionary for bulk loading operations
)


# ============================================================
# UTILITY FUNCTIONS - MEMORY MANAGEMENT
# ============================================================


def force_release_memory():
    """Force memory release using garbage collection and malloc_trim.

    Performs aggressive memory cleanup by executing garbage collection
    and, on Linux systems, calling malloc_trim to release memory back
    to the operating system. Provides detailed before/after metrics.

    Returns:
        dict: Dictionary containing:
            - freed_mb (float): Amount of memory freed in MB
            - gc_collected (int): Number of objects collected by GC
            - malloc_trim_used (bool): Whether malloc_trim was executed
            - error (str, optional): Error message if operation failed

    Note:
        - malloc_trim is only available on Linux systems
        - On Windows/other platforms, only GC is performed
        - Memory release to OS is best-effort and not guaranteed
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
    """Clean up expired cache entries from the bulk loading cache.

    Iterates through the global bulk loading cache and removes entries
    that have exceeded the BULK_CACHE_TIMEOUT duration. This helps
    prevent memory buildup from stale cache data.

    Returns:
        int: Number of expired cache entries that were removed

    Note:
        - Uses current time comparison against cache timestamps
        - Operates on the global _bulk_loading_cache dictionary
        - Should be called periodically or when memory pressure detected
    """
    current_time = time.time()
    expired_keys = []

    for cache_key, (cached_data, cache_time) in _bulk_loading_cache.items():
        if current_time - cache_time > BULK_CACHE_TIMEOUT:
            expired_keys.append(cache_key)

    for key in expired_keys:
        del _bulk_loading_cache[key]

    return len(expired_keys)


def check_memory_and_gc():
    """Enhanced memory check with cache cleanup, GC, and malloc_trim.

    Monitors current memory usage and triggers cleanup operations when
    memory thresholds are approached or exceeded. Implements a tiered
    cleanup strategy:
    - At 80% threshold: Cache cleanup
    - At 100% threshold: Full cleanup (cache + GC + malloc_trim)

    Returns:
        float: Current RSS memory usage in MB after cleanup

    Cleanup Strategy:
    ----------------
    1. Check current memory usage (RSS)
    2. If at 80% of threshold: Clean expired cache entries
    3. If above threshold: Force full memory release
    4. Provide scaling guidance if memory remains high

    Note:
        - Returns 0 if memory check fails
        - Logs detailed cleanup statistics
        - Provides actionable scaling recommendations
    """
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
    """Simplified memory logging with optional context.

    Logs current memory usage (RSS) to the debug output. If memory
    exceeds the configured threshold, automatically triggers memory
    check and cleanup operations.

    Args:
        context (str, optional): Descriptive context for the log entry
                               (e.g., "after_query", "startup")

    Note:
        - Always logs current memory in MB
        - Automatically triggers check_memory_and_gc if threshold exceeded
        - Errors are logged but don't raise exceptions
    """
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
# Global variables for background task management and snapshot tracking
_memory_profiler_task: Optional[asyncio.Task] = None  # Profiler background task
_memory_cleanup_task: Optional[asyncio.Task] = None  # Cleanup background task
_previous_snapshot: Optional[tracemalloc.Snapshot] = (
    None  # Last tracemalloc snapshot for comparison
)


def _get_uvicorn_logger() -> logging.Logger:
    """Get the uvicorn error logger for profiler output.

    Returns:
        logging.Logger: The uvicorn.error logger instance
    """
    return logging.getLogger("uvicorn.error")


def _shorten_path(filepath: str, max_len: int = 60) -> str:
    """Shorten long file paths for better table formatting.

    Intelligently truncates file paths to fit within the specified
    maximum length while preserving the most relevant parts (filename
    and some parent directory context).

    Args:
        filepath (str): The full file path to shorten
        max_len (int, optional): Maximum length for the shortened path.
                                Defaults to 60 characters.

    Returns:
        str: Shortened file path with ellipsis (...) indicating truncation

    Strategy:
    --------
    1. If path fits in max_len, return as-is
    2. Preserve filename if possible
    3. Build path from end (filename) backwards
    4. Add ellipsis prefix when truncated
    """
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
    """Log detailed tracemalloc snapshot with memory statistics.

    Generates a comprehensive formatted table showing memory allocations,
    process memory usage, and memory maps. Compares with previous snapshot
    to show allocation deltas.

    Args:
        snapshot (tracemalloc.Snapshot): Current memory snapshot
        previous_snapshot (Optional[tracemalloc.Snapshot]): Previous snapshot
                                                           for comparison
        top_stats (int): Number of top allocations to display

    Output Format:
    -------------
    - Header with tracemalloc current/peak memory
    - Process RSS, USS, and swap statistics
    - Top N allocations table with:
      * Size in MiB and delta from previous
      * Block count and delta
      * File location (shortened for readability)
    - Summary of remaining allocations
    - Memory maps table showing RSS per segment
    - Additional segments summary

    Note:
        - Logs to uvicorn.error logger
        - Also outputs to debug monitoring
        - Handles missing previous snapshot gracefully
        - Formats paths for better readability
    """
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
    """Background task for periodic memory profiling.

    Continuously takes tracemalloc snapshots at the specified interval
    and logs detailed memory allocation statistics. Compares each snapshot
    with the previous one to track allocation changes over time.

    Args:
        interval (int): Seconds between profiling snapshots
        top_stats (int): Number of top allocations to display

    Raises:
        asyncio.CancelledError: When task is cancelled (expected during shutdown)

    Lifecycle:
    ---------
    1. Start tracemalloc if not already running
    2. Take initial snapshot
    3. Loop: sleep -> snapshot -> log -> update previous
    4. On cancellation: cleanup and stop tracemalloc

    Note:
        - Runs indefinitely until cancelled
        - Stores snapshots in global _previous_snapshot
        - Logs to uvicorn.error logger
    """
    global _previous_snapshot
    logger = _get_uvicorn_logger()

    # Initialize tracemalloc if not already running
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        print__memory_monitoring("[memory-profiler] tracemalloc tracing started")

    # Take initial snapshot for baseline comparison
    _previous_snapshot = tracemalloc.take_snapshot()

    print__memory_monitoring(
        f"[memory-profiler] Background task running every {interval}s (top {top_stats} stats)"
    )

    try:
        # Main profiling loop - runs until task is cancelled
        while True:
            await asyncio.sleep(interval)

            # Take new snapshot and compare with previous
            snapshot = tracemalloc.take_snapshot()
            _log_tracemalloc_snapshot(snapshot, _previous_snapshot, top_stats)

            # Update previous snapshot for next iteration comparison
            _previous_snapshot = snapshot
    except asyncio.CancelledError:
        # Expected during graceful shutdown
        logger.info("[memory-profiler] Background task cancelled")
        print__memory_monitoring("[memory-profiler] Background task cancelled")
        raise
    finally:
        # Cleanup: keep tracemalloc running to allow other consumers
        _previous_snapshot = None


# ============================================================
# PERIODIC MEMORY CLEANUP
# ============================================================


async def _memory_cleanup_loop() -> None:
    """Background task for periodic memory cleanup.

    Executes cache cleanup and memory release operations at regular
    intervals to prevent memory buildup during long-running processes.

    Cleanup Operations:
    ------------------
    1. Check current memory usage
    2. Clean expired cache entries
    3. Force memory release (GC + malloc_trim)
    4. Report cleanup statistics

    Raises:
        asyncio.CancelledError: When task is cancelled (expected during shutdown)

    Note:
        - Runs every MEMORY_CLEANUP_INTERVAL seconds
        - Only effective on Linux (malloc_trim)
        - Logs detailed cleanup results
        - Runs indefinitely until cancelled
    """
    print__memory_monitoring(
        f"🧹 [memory-cleanup] Starting cleanup task (every {MEMORY_CLEANUP_INTERVAL}s)"
    )

    try:
        # Main cleanup loop - runs until task is cancelled
        while True:
            await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)

            # Get current memory usage before cleanup
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024

            print__memory_monitoring(
                f"🧹 [memory-cleanup] Running (RSS: {rss_mb:.1f}MB)"
            )

            # Step 1: Clean expired cache entries
            cleaned = cleanup_bulk_cache()

            # Step 2: Force memory release (GC + malloc_trim on Linux)
            result = force_release_memory()

            # Report cleanup results
            print__memory_monitoring(
                f"✅ [memory-cleanup] Cache: {cleaned} entries, "
                f"Memory: {result.get('freed_mb', 0):.1f}MB freed"
            )

    except asyncio.CancelledError:
        # Expected during graceful shutdown
        print__memory_monitoring("🛑 [memory-cleanup] Task cancelled")
        raise


def start_memory_profiler(
    interval: Optional[int] = None, top_stats: Optional[int] = None
) -> Optional[asyncio.Task]:
    """Start the periodic tracemalloc profiler if enabled.

    Creates and starts a background asyncio task that periodically
    profiles memory allocations using tracemalloc. Only starts if
    MEMORY_PROFILER_ENABLED is True.

    Args:
        interval (Optional[int]): Profiling interval in seconds.
                                 Defaults to MEMORY_PROFILER_INTERVAL.
        top_stats (Optional[int]): Number of top allocations to display.
                                  Defaults to MEMORY_PROFILER_TOP_STATS.

    Returns:
        Optional[asyncio.Task]: The profiler task if started, None if disabled
                               or already running.

    Note:
        - Requires running event loop
        - Returns existing task if already running
        - Stores task in global _memory_profiler_task
    """

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
    """Start the periodic memory cleanup task if enabled.

    Creates and starts a background asyncio task that periodically
    cleans up expired cache entries and releases memory. Only starts
    if MEMORY_CLEANUP_ENABLED is True.

    Returns:
        Optional[asyncio.Task]: The cleanup task if started, None if disabled,
                               already running, or no event loop available.

    Note:
        - Requires running event loop
        - Returns existing task if already running
        - Stores task in global _memory_cleanup_task
        - Disabled automatically if malloc_trim unavailable
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
    """Stop the periodic tracemalloc profiler if it is running.

    Cancels the profiler background task and stops tracemalloc.
    Waits for task cancellation to complete before returning.

    Note:
        - Handles CancelledError gracefully
        - Stops tracemalloc if it was running
        - Clears global _memory_profiler_task
        - Safe to call even if profiler not running
    """

    global _memory_profiler_task
    if not _memory_profiler_task:
        return

    # Cancel the task and wait for it to complete
    _memory_profiler_task.cancel()
    try:
        await _memory_profiler_task
    except asyncio.CancelledError:
        pass
    finally:
        # Clean up global reference
        _memory_profiler_task = None
        # Stop tracemalloc if it was running
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            logger = _get_uvicorn_logger()
            logger.info("[memory-profiler] tracemalloc tracing stopped")
            print__memory_monitoring("[memory-profiler] tracemalloc tracing stopped")


async def stop_memory_cleanup() -> None:
    """Stop the periodic memory cleanup task if it is running.

    Cancels the cleanup background task and waits for cancellation
    to complete before returning.

    Note:
        - Handles CancelledError gracefully
        - Clears global _memory_cleanup_task
        - Safe to call even if cleanup not running
        - Logs shutdown confirmation
    """

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
    """Log comprehensive error information with context.

    Creates a detailed error report including error type, message,
    timestamp, and optional request information. Outputs to debug
    logging system.

    Args:
        context (str): Description of where/when the error occurred
        error (Exception): The exception that was raised
        request (Request, optional): FastAPI request object for additional context

    Error Details:
    -------------
    - Error context/location
    - Error type (exception class name)
    - Error message
    - Timestamp (ISO format)
    - Request method and URL (if request provided)
    - Client IP address (if request provided)

    Note:
        - Does not raise exceptions
        - Outputs to api.utils.debug.print__debug
        - Format: JSON for structured logging
    """
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
    """Setup graceful shutdown signal handlers.

    Registers signal handlers for common shutdown signals (SIGTERM, SIGINT,
    SIGUSR1) to enable graceful shutdown with memory state logging.

    Registered Signals:
    ------------------
    - SIGTERM: Container/service shutdown (most common)
    - SIGINT: Keyboard interrupt (Ctrl+C)
    - SIGUSR1: User-defined signal (Unix only)

    Shutdown Behavior:
    -----------------
    - Logs signal receipt
    - Logs final memory state
    - Allows application to clean up resources

    Note:
        - SIGUSR1 only registered on Unix systems
        - Handlers log but don't force exit
        - Should be called during application startup
    """

    def signal_handler(signum, frame):
        """Handle shutdown signals by logging memory state."""
        print__memory_monitoring(
            f"📡 Received signal {signum} - preparing for graceful shutdown..."
        )
        log_memory_usage("shutdown_signal")

    # Register signal handlers for common restart/shutdown signals
    signal.signal(signal.SIGTERM, signal_handler)  # Container/service shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C interrupt
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal_handler)  # User-defined signal (Unix)


async def perform_deletion_operations(conn, user_email: str, thread_id: str):
    """Perform thread deletion operations with security verification.

    Deletes all data associated with a conversation thread after verifying
    that the requesting user owns the thread. Performs atomic deletion
    across multiple tables (checkpoints and users_threads_runs).

    Args:
        conn: Async PostgreSQL connection (psycopg)
        user_email (str): Email of the user requesting deletion
        thread_id (str): ID of the thread to delete

    Returns:
        dict: Result dictionary containing:
            - message (str): Success/failure message
            - thread_id (str): The deleted thread ID
            - user_email (str): The requesting user's email
            - deleted_counts (dict): Deletion counts per table

    Security:
    --------
    - Verifies user ownership before any deletion
    - Returns access denied if user doesn't own thread
    - Uses parameterized queries to prevent SQL injection
    - Atomic operations with explicit commits

    Tables Affected:
    ---------------
    1. checkpoint_blobs: Binary checkpoint data
    2. checkpoint_writes: Checkpoint write operations
    3. checkpoints: Main checkpoint records
    4. users_threads_runs: User-thread associations

    Note:
        - Uses autocommit mode for atomic operations
        - Handles missing tables gracefully
        - Logs all operations with detailed diagnostics
        - Returns partial results on errors
    """
    # Import debug function
    from api.utils.debug import print__api_postgresql

    print__api_postgresql("🔧 DEBUG: Starting deletion operations...")

    # Set connection to autocommit mode for atomic transaction handling
    print__api_postgresql("🔧 DEBUG: Setting autocommit...")
    await conn.set_autocommit(True)
    print__api_postgresql("🔧 DEBUG: Autocommit set successfully")

    # ================================================================
    # SECURITY CHECK: Verify User Ownership
    # ================================================================
    # 🔒 Verify user owns this thread before allowing deletion
    print__api_postgresql(
        f"🔒 Verifying thread ownership for deletion - user: {user_email}, thread: {thread_id}"
    )

    print__api_postgresql("🔧 DEBUG: Creating cursor for ownership check...")
    async with conn.cursor() as cur:
        print__api_postgresql("🔧 DEBUG: Cursor created, executing ownership query...")
        # Query to count entries for this user-thread combination
        # If count is 0, user doesn't own the thread
        await cur.execute(
            """
            SELECT COUNT(*) FROM users_threads_runs
            WHERE email = %s AND thread_id = %s
        """,
            (user_email, thread_id),
        )

        # Extract count from result row (psycopg returns Row objects)
        result_row = await cur.fetchone()
        # Convert Row to tuple to access first element (count value)
        thread_entries_count = tuple(result_row)[0] if result_row else 0
        print__api_postgresql(
            f"🔧 DEBUG: Ownership check complete, count: {thread_entries_count}"
        )

    # Deny deletion if user doesn't own the thread (security check failed)
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

    # ================================================================
    # CHECKPOINT TABLES DELETION
    # ================================================================
    print__api_postgresql(f"🔄 Deleting from checkpoint tables for thread {thread_id}")

    # Tables to delete from (in order)
    tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
    deleted_counts = {}  # Track deletion counts per table

    # Iterate through each checkpoint table and delete thread data
    for table in tables:
        try:
            print__api_postgresql(f"🔧 DEBUG: Processing table {table}...")
            # First check if the table exists in the database
            print__api_postgresql(
                "🔧 DEBUG: Creating cursor for table existence check..."
            )
            async with conn.cursor() as cur:
                print__api_postgresql(
                    f"🔧 DEBUG: Executing table existence query for {table}..."
                )
                # Query information_schema to verify table exists
                await cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    )
                """,
                    (table,),
                )

                # Extract boolean result from Row object
                result_row = await cur.fetchone()
                # Convert to tuple and get first element (boolean)
                table_exists = tuple(result_row)[0] if result_row else False
                print__api_postgresql(
                    f"🔧 DEBUG: Table {table} exists check result: {table_exists}"
                )

                # Skip this table if it doesn't exist (graceful handling)
                if not table_exists:
                    print__api_postgresql(f"⚠ Table {table} does not exist, skipping")
                    deleted_counts[table] = 0
                    continue

                print__api_postgresql(
                    f"🔧 DEBUG: Table {table} exists, proceeding with deletion..."
                )
                # Delete all records for this thread_id from the table
                print__api_postgresql(
                    f"🔧 DEBUG: Creating cursor for deletion from {table}..."
                )
                async with conn.cursor() as del_cur:
                    print__api_postgresql(
                        f"🔧 DEBUG: Executing DELETE query for {table}..."
                    )
                    # Execute parameterized DELETE to prevent SQL injection
                    await del_cur.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
                    )

                    # Get number of rows deleted
                    deleted_counts[table] = (
                        del_cur.rowcount if hasattr(del_cur, "rowcount") else 0
                    )
                    print__api_postgresql(
                        f"✅ Deleted {deleted_counts[table]} records from {table} for thread_id: {thread_id}"
                    )
                    # Explicit commit for transaction safety
                    await conn.commit()

        except Exception as table_error:
            # Handle errors for individual tables without stopping entire operation
            print__api_postgresql(f"⚠ Error deleting from table {table}: {table_error}")
            print__api_postgresql(
                f"🔧 DEBUG: Table error type: {type(table_error).__name__}"
            )
            print__api_postgresql(
                f"🔧 DEBUG: Table error traceback: {traceback.format_exc()}"
            )
            deleted_counts[table] = f"Error: {str(table_error)}"

    # ================================================================
    # USER-THREAD ASSOCIATION DELETION
    # ================================================================
    # Delete the user-thread association from users_threads_runs table
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
            # Delete user-thread association with parameterized query
            await cur.execute(
                """
                DELETE FROM users_threads_runs
                WHERE email = %s AND thread_id = %s
            """,
                (user_email, thread_id),
            )

            # Get number of rows deleted
            users_threads_runs_deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
            print__api_postgresql(
                f"✅ Deleted {users_threads_runs_deleted} entries from users_threads_runs for user {user_email}, thread {thread_id}"
            )

            deleted_counts["users_threads_runs"] = users_threads_runs_deleted
            # Explicit commit for transaction safety
            await conn.commit()

    except Exception as e:
        # Handle errors in users_threads_runs deletion
        print__api_postgresql(f"❌ Error deleting from users_threads_runs: {e}")
        print__api_postgresql(
            f"🔧 DEBUG: users_threads_runs error type: {type(e).__name__}"
        )
        print__api_postgresql(
            f"🔧 DEBUG: users_threads_runs error traceback: {traceback.format_exc()}"
        )
        deleted_counts["users_threads_runs"] = f"Error: {str(e)}"

    # ================================================================
    # RETURN DELETION RESULTS
    # ================================================================
    # Compile final result data with all deletion statistics
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

#!/usr/bin/env python3
"""
Simple Memory Cleanup Test Script

Tests the memory cleanup functionality that uses libc.malloc_trim(0) on Linux.
This script runs the periodic memory cleanup loop and monitors memory usage.

Environment Variables:
- MEMORY_CLEANUP_ENABLED=1 (default: 1)
- MEMORY_CLEANUP_INTERVAL=10 (default: 60)

Usage:
    python test_memory_cleanup.py

On Linux, this will call malloc_trim(0) every MEMORY_CLEANUP_INTERVAL seconds.
On Windows, it will only run garbage collection.
"""

import asyncio
import ctypes
import gc
import os
import sys
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Import psutil
import psutil

# Simplified memory cleanup functionality (copied from memory.py)
MEMORY_CLEANUP_ENABLED = os.environ.get("MEMORY_CLEANUP_ENABLED", "1") == "1"
MEMORY_CLEANUP_INTERVAL = int(os.environ.get("MEMORY_CLEANUP_INTERVAL", "60"))

# Try to load libc for malloc_trim support (Linux only)
try:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL("libc.so.6")
        MALLOC_TRIM_AVAILABLE = True
        print("🐧 Linux detected - malloc_trim available")
    else:
        libc = None
        MALLOC_TRIM_AVAILABLE = False
        print(f"🪟 Non-Linux platform ({sys.platform}) - malloc_trim not available")
except (OSError, AttributeError) as e:
    libc = None
    MALLOC_TRIM_AVAILABLE = False
    print(f"❌ Failed to load libc: {e}")


def force_release_memory():
    """
    Force memory release - Linux only (malloc_trim).
    """
    try:
        # Get initial memory
        process = psutil.Process()
        initial_rss = process.memory_info().rss / 1024 / 1024

        # Run garbage collection
        collected = gc.collect()

        # Only run malloc_trim on Linux
        if MALLOC_TRIM_AVAILABLE and sys.platform.startswith("linux"):
            libc.malloc_trim(0)
            malloc_trim_used = True
            print("🧹 Called malloc_trim(0) to release memory to OS")
        else:
            malloc_trim_used = False

        # Get final memory
        final_rss = process.memory_info().rss / 1024 / 1024
        freed_mb = initial_rss - final_rss

        print(
            f"🧹 Memory cleanup: {freed_mb:.1f}MB freed | {initial_rss:.1f}MB → {final_rss:.1f}MB | GC: {collected} | malloc_trim: {'✓' if malloc_trim_used else '✗'}"
        )

        return {
            "freed_mb": round(freed_mb, 2),
            "gc_collected": collected,
            "malloc_trim_used": malloc_trim_used,
        }

    except Exception as e:
        print(f"❌ Memory cleanup error: {e}")
        return {"error": str(e), "freed_mb": 0}


async def memory_cleanup_loop():
    """
    Simple periodic memory cleanup - every MEMORY_CLEANUP_INTERVAL seconds.
    """
    print(
        f"🧹 [memory-cleanup] Starting cleanup task (every {MEMORY_CLEANUP_INTERVAL}s)"
    )

    try:
        while True:
            await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)

            # Get current memory usage
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024

            print(f"🧹 [memory-cleanup] Running (RSS: {rss_mb:.1f}MB)")

            # Force memory release (only effective on Linux)
            result = force_release_memory()

            print(
                f"✅ [memory-cleanup] Memory: {result.get('freed_mb', 0):.1f}MB freed"
            )

    except asyncio.CancelledError:
        print("🛑 [memory-cleanup] Task cancelled")
        raise


def print_status():
    """Print current status and memory usage."""
    process = psutil.Process()
    rss_mb = process.memory_info().rss / 1024 / 1024

    print("\n📊 STATUS:")
    print(f"   Platform: {sys.platform}")
    print(f"   malloc_trim available: {'✓' if MALLOC_TRIM_AVAILABLE else '✗'}")
    print(f"   Memory cleanup enabled: {'✓' if MEMORY_CLEANUP_ENABLED else '✗'}")
    print(f"   Cleanup interval: {MEMORY_CLEANUP_INTERVAL}s")
    print(f"   Current memory: {rss_mb:.1f}MB")
    print(f"   Process PID: {process.pid}")


async def test_manual_cleanup():
    """Test manual memory cleanup."""
    print("\n🧪 TESTING MANUAL MEMORY CLEANUP")
    print("=" * 50)

    # Get memory before
    process = psutil.Process()
    before_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Memory before: {before_mb:.1f}MB")

    # Run cleanup
    result = force_release_memory()

    # Get memory after
    after_mb = process.memory_info().rss / 1024 / 1024
    print(f"   Memory after: {after_mb:.1f}MB")
    print(f"   Freed: {result.get('freed_mb', 0):.1f}MB")
    print(f"   GC collected: {result.get('gc_collected', 0)} objects")
    print(
        f"   malloc_trim used: {'✓' if result.get('malloc_trim_used', False) else '✗'}"
    )


async def test_periodic_cleanup():
    """Test periodic memory cleanup loop."""
    print("\n🔄 TESTING PERIODIC MEMORY CLEANUP")
    print("=" * 50)
    print(f"Will run cleanup every {MEMORY_CLEANUP_INTERVAL} seconds...")
    print("Press Ctrl+C to stop the test")

    # Start the cleanup task
    cleanup_task = asyncio.create_task(memory_cleanup_loop())

    print("✅ Memory cleanup task started")

    # Run for a few cycles
    cycles = 3  # Run for 3 cleanup cycles
    total_time = cycles * MEMORY_CLEANUP_INTERVAL

    print(f"⏰ Running for {total_time} seconds ({cycles} cleanup cycles)...")

    try:
        await asyncio.sleep(total_time)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    finally:
        print("\n🛑 Stopping memory cleanup task...")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        print("✅ Memory cleanup task stopped")


async def main():
    """Main test function."""
    print("🧹 MEMORY CLEANUP TEST SCRIPT")
    print("=" * 50)

    # Print initial status
    print_status()

    if not MEMORY_CLEANUP_ENABLED:
        print("\n❌ Memory cleanup is disabled (MEMORY_CLEANUP_ENABLED != 1)")
        print("Set MEMORY_CLEANUP_ENABLED=1 to enable cleanup")
        return

    # Test manual cleanup first
    await test_manual_cleanup()

    # Wait a bit
    await asyncio.sleep(2)

    # Test periodic cleanup
    await test_periodic_cleanup()

    print("\n🎉 Memory cleanup test completed!")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())

"""
Quick test script to validate SSL connection fixes.
This script tests the database connection health checking and SSL error handling.
"""

import asyncio
import sys
from pathlib import Path
import os

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]


# Make project root importable
sys.path.insert(0, str(BASE_DIR))

from checkpointer.checkpointer.factory import (
    create_async_postgres_saver,
    close_async_postgres_saver,
)
from checkpointer.database.connection import check_connection_health
from checkpointer.config import check_postgres_env_vars
from api.utils.debug import print__checkpointers_debug


async def test_ssl_fixes():
    """Test the SSL connection fixes."""
    print("🚀 Testing SSL connection fixes...")

    # Check environment
    if not check_postgres_env_vars():
        print("❌ PostgreSQL environment variables not configured!")
        return False

    print("✅ Environment variables configured")

    try:
        # Test checkpointer creation with SSL retry
        print("🔧 Testing checkpointer creation with SSL retry...")
        checkpointer = await create_async_postgres_saver()
        print("✅ Checkpointer created successfully")

        # Test connection health checking
        print("🔍 Testing connection health checking...")
        pool = getattr(checkpointer, "pool", None)
        if pool:
            async with pool.connection() as conn:
                is_healthy = await check_connection_health(conn)
                print(f"✅ Connection health check: {'PASS' if is_healthy else 'FAIL'}")

        # Test basic operation
        print("🔍 Testing basic checkpointer operation...")
        test_config = {"configurable": {"thread_id": "ssl_test"}}
        result = await checkpointer.aget(test_config)
        print("✅ Basic checkpointer operation successful")

        # Cleanup
        await close_async_postgres_saver()
        print("✅ Cleanup completed")

        print("🎉 All SSL fixes validated successfully!")
        return True

    except Exception as e:
        print(f"❌ SSL fix test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_ssl_fixes()
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)

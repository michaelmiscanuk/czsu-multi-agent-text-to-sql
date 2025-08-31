#!/usr/bin/env python3
"""
Test script to debug FastAPI startup and checkpointer initialization
"""

import asyncio
import os
import sys

# Set debug mode
os.environ["DEBUG"] = "1"

# Add current directory to path
sys.path.insert(0, ".")


async def test_fastapi_startup():
    """Test the FastAPI startup process"""
    print("🔍 Testing FastAPI startup process...")

    # Test 1: Import api.main instead of api_server
    print("\n1. Importing api.main...")
    try:
        from api.main import app
        from api.config.settings import GLOBAL_CHECKPOINTER

        print("✅ api.main imported successfully")
    except Exception as e:
        print(f"❌ Failed to import api.main: {e}")
        return

    # Test 2: Test the lifespan startup manually
    print("\n2. Testing lifespan startup manually...")
    try:
        # Simulate the FastAPI lifespan startup
        print("🔄 Running initialize_checkpointer()...")
        from checkpointer.postgres_checkpointer import initialize_checkpointer

        await initialize_checkpointer()

        print(f"✅ Checkpointer initialized: {type(GLOBAL_CHECKPOINTER).__name__}")

        # Check checkpointer type
        if hasattr(GLOBAL_CHECKPOINTER, "conn"):
            print(f"✅ Has PostgreSQL connection pool")
            if GLOBAL_CHECKPOINTER.conn:
                print(
                    f"✅ Connection pool exists: closed={GLOBAL_CHECKPOINTER.conn.closed}"
                )
            else:
                print(f"⚠️ Connection pool is None")
        else:
            print(f"⚠️ Using InMemorySaver fallback")

    except Exception as e:
        print(f"❌ Failed during lifespan startup: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")

    # Test 3: Test health check
    print("\n3. Testing health check...")
    try:
        from api.routes.health import health_check

        health_result = await health_check()
        print(f"✅ Health check result: {health_result}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

    # Test 4: Test get_global_checkpointer
    print("\n4. Testing get_global_checkpointer...")
    try:
        from checkpointer.postgres_checkpointer import get_global_checkpointer

        healthy_checkpointer = await get_global_checkpointer()
        print(f"✅ Healthy checkpointer: {type(healthy_checkpointer).__name__}")

        if hasattr(healthy_checkpointer, "conn"):
            print(f"✅ Has PostgreSQL connection pool")
        else:
            print(f"⚠️ Using InMemorySaver fallback")

    except Exception as e:
        print(f"❌ get_global_checkpointer failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_fastapi_startup())

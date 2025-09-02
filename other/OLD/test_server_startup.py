#!/usr/bin/env python3
"""
Test script to debug server startup and checkpointer initialization
"""

import asyncio
import os
import sys

# Set debug mode
os.environ["DEBUG"] = "1"

# Add current directory to path
sys.path.insert(0, ".")


async def test_server_startup():
    """Test the server startup components individually"""
    print("🔍 Testing server startup components...")

    # Test 1: Import api.main instead of api_server
    print("\n1. Testing api.main import...")
    try:
        from api.main import app
        from api.config.settings import GLOBAL_CHECKPOINTER

        print("✅ api.main imported successfully")
    except Exception as e:
        print(f"❌ Failed to import api.main: {e}")
        return

    # Test 2: Test checkpointer initialization directly
    print("\n2. Testing checkpointer initialization...")
    try:
        from checkpointer.checkpointer.factory import initialize_checkpointer

        await initialize_checkpointer()
        print(f"✅ Checkpointer initialized: {type(GLOBAL_CHECKPOINTER).__name__}")

        # Check checkpointer type
        if hasattr(GLOBAL_CHECKPOINTER, "conn"):
            print(f"✅ Has PostgreSQL connection pool")
        else:
            print(f"⚠️ Using InMemorySaver fallback")

    except Exception as e:
        print(f"❌ Failed to initialize checkpointer: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")

    # Test 3: Test FastAPI app creation
    print("\n3. Testing FastAPI app...")
    try:
        print(f"✅ FastAPI app created: {type(app).__name__}")
    except Exception as e:
        print(f"❌ Failed to create FastAPI app: {e}")

    # Test 4: Test health endpoint directly
    print("\n4. Testing health endpoint function...")
    try:
        from api.routes.health import health_check

        health_result = await health_check()
        print(f"✅ Health check result: {health_result}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_server_startup())

#!/usr/bin/env python3
"""
Test script to debug server startup and checkpointer initialization
"""

import asyncio
import os
import sys

# Set debug mode
os.environ['DEBUG'] = '1'

# Add current directory to path
sys.path.insert(0, '.')

async def test_server_startup():
    """Test the server startup components individually"""
    print("🔍 Testing server startup components...")
    
    # Test 1: Import api_server
    print("\n1. Testing api_server import...")
    try:
        import api_server
        print("✅ api_server imported successfully")
    except Exception as e:
        print(f"❌ Failed to import api_server: {e}")
        return
    
    # Test 2: Test checkpointer initialization directly
    print("\n2. Testing checkpointer initialization...")
    try:
        await api_server.initialize_checkpointer()
        print(f"✅ Checkpointer initialized: {type(api_server.GLOBAL_CHECKPOINTER).__name__}")
        
        # Check checkpointer type
        if hasattr(api_server.GLOBAL_CHECKPOINTER, 'conn'):
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
        app = api_server.app
        print(f"✅ FastAPI app created: {type(app).__name__}")
    except Exception as e:
        print(f"❌ Failed to create FastAPI app: {e}")
    
    # Test 4: Test health endpoint directly
    print("\n4. Testing health endpoint function...")
    try:
        health_result = await api_server.health_check()
        print(f"✅ Health check result: {health_result}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_server_startup()) 
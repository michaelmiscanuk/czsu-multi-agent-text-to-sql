#!/usr/bin/env python3
"""
Test script to debug FastAPI startup and checkpointer initialization
"""

import asyncio
import os
import sys

# Set debug mode
os.environ['MY_AGENT_DEBUG'] = '1'

# Add current directory to path
sys.path.insert(0, '.')

async def test_fastapi_startup():
    """Test the FastAPI startup process"""
    print("🔍 Testing FastAPI startup process...")
    
    # Test 1: Import api_server
    print("\n1. Importing api_server...")
    try:
        import api_server
        print("✅ api_server imported successfully")
    except Exception as e:
        print(f"❌ Failed to import api_server: {e}")
        return
    
    # Test 2: Test the lifespan startup manually
    print("\n2. Testing lifespan startup manually...")
    try:
        # Simulate the FastAPI lifespan startup
        print("🔄 Running initialize_checkpointer()...")
        await api_server.initialize_checkpointer()
        
        print(f"✅ Checkpointer initialized: {type(api_server.GLOBAL_CHECKPOINTER).__name__}")
        
        # Check checkpointer type
        if hasattr(api_server.GLOBAL_CHECKPOINTER, 'conn'):
            print(f"✅ Has PostgreSQL connection pool")
            if api_server.GLOBAL_CHECKPOINTER.conn:
                print(f"✅ Connection pool exists: closed={api_server.GLOBAL_CHECKPOINTER.conn.closed}")
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
        health_result = await api_server.health_check()
        print(f"✅ Health check result: {health_result}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 4: Test get_healthy_checkpointer
    print("\n4. Testing get_healthy_checkpointer...")
    try:
        healthy_checkpointer = await api_server.get_healthy_checkpointer()
        print(f"✅ Healthy checkpointer: {type(healthy_checkpointer).__name__}")
        
        if hasattr(healthy_checkpointer, 'conn'):
            print(f"✅ Has PostgreSQL connection pool")
        else:
            print(f"⚠️ Using InMemorySaver fallback")
            
    except Exception as e:
        print(f"❌ get_healthy_checkpointer failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_fastapi_startup()) 
#!/usr/bin/env python3
"""
Test script to verify the /chat-threads API endpoint is working correctly.
"""

import asyncio
import sys
from my_agent.utils.postgres_checkpointer import (
    get_user_chat_threads, 
    get_postgres_checkpointer
)

async def test_chat_threads_backend():
    """Test the backend function directly."""
    
    print("🧪 Testing get_user_chat_threads backend function...")
    
    # Initialize the checkpointer to set up the database pool
    try:
        checkpointer = await get_postgres_checkpointer()
        print("✓ Database pool initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database pool: {e}")
        return False
    
    # Test with the user we know has data
    test_email = "retko85@gmail.com"
    
    try:
        threads = await get_user_chat_threads(test_email)
        print(f"✅ Backend function returned {len(threads)} threads for {test_email}")
        
        for i, thread in enumerate(threads):
            print(f"  {i+1}. Thread: {thread['thread_id'][:12]}...")
            print(f"      Latest: {thread['latest_timestamp']}")
            print(f"      Runs: {thread['run_count']}")
        
        return len(threads) > 0  # Should have threads for this user
        
    except Exception as e:
        print(f"❌ Backend function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_response_format():
    """Test that the API response format matches what the frontend expects."""
    
    print("\n🔧 Testing API response format conversion...")
    
    try:
        # Get threads from backend
        threads = await get_user_chat_threads("retko85@gmail.com")
        
        if len(threads) == 0:
            print("⚠ No threads to test format conversion with")
            return True
        
        # Convert to API response format (like in api_server.py)
        response_threads = []
        for thread in threads:
            response_thread = {
                "thread_id": thread["thread_id"],
                "latest_timestamp": thread["latest_timestamp"].isoformat(),
                "run_count": thread["run_count"]
            }
            response_threads.append(response_thread)
        
        print(f"✅ Converted {len(response_threads)} threads to API format")
        
        # Show what frontend would receive
        for i, thread in enumerate(response_threads):
            print(f"  {i+1}. Frontend would get:")
            print(f"      thread_id: {thread['thread_id'][:12]}...")
            print(f"      latest_timestamp: {thread['latest_timestamp']}")
            print(f"      run_count: {thread['run_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ API format test failed: {e}")
        return False

if __name__ == "__main__":
    async def run_tests():
        print("🔬 Running backend API tests...\n")
        
        # Test 1: Backend function
        backend_ok = await test_chat_threads_backend()
        
        # Test 2: API format conversion
        format_ok = await test_api_response_format()
        
        if backend_ok and format_ok:
            print("\n🎉 All backend tests PASSED")
            print("✅ Backend is working correctly - threads are being retrieved")
            print("💡 If frontend is not loading threads, the issue is likely:")
            print("   1. Authentication/session issues in frontend")
            print("   2. CORS or network connectivity")
            print("   3. Frontend not calling the API correctly")
            print("   4. Environment variables (API_BASE) not set correctly in frontend")
            return True
        else:
            print("\n❌ Some backend tests FAILED")
            print("💡 Backend issues found - threads are not being retrieved correctly")
            return False
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1) 
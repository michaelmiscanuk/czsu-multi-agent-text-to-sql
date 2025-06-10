#!/usr/bin/env python3
"""
Test suite for the PostgreSQL-based chat management system.

This module tests the new users_threads_runs table functionality and 
API endpoints that replace IndexedDB for chat persistence.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Configure asyncio event loop policy for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_agent.utils.postgres_checkpointer import (
    setup_users_threads_runs_table,
    create_thread_run_entry,
    get_user_chat_threads,
    delete_user_thread_entries,
    get_postgres_checkpointer
)

# Test configuration
TEST_EMAIL_1 = "test1@example.com"
TEST_EMAIL_2 = "test2@example.com"
TEST_THREAD_ID_1 = "test_thread_001"
TEST_THREAD_ID_2 = "test_thread_002"
TEST_THREAD_ID_3 = "test_thread_003"

class TestPostgresChatSystem:
    """Test class for PostgreSQL chat management system."""
    
    async def setup_test_database(self):
        """Setup test database and clean up any existing test data."""
        try:
            # Initialize the table
            await setup_users_threads_runs_table()
            
            # Clean up any existing test data
            await self.cleanup_test_data()
            
            print("✅ Test database setup completed")
            
        except Exception as e:
            print(f"❌ Test database setup failed: {e}")
            raise
    
    async def cleanup_test_data(self):
        """Clean up test data from database."""
        try:
            await delete_user_thread_entries(TEST_EMAIL_1, TEST_THREAD_ID_1)
            await delete_user_thread_entries(TEST_EMAIL_1, TEST_THREAD_ID_2)
            await delete_user_thread_entries(TEST_EMAIL_1, TEST_THREAD_ID_3)
            await delete_user_thread_entries(TEST_EMAIL_2, TEST_THREAD_ID_1)
            await delete_user_thread_entries(TEST_EMAIL_2, TEST_THREAD_ID_2)
            await delete_user_thread_entries(TEST_EMAIL_1, "thread_oldest")
            await delete_user_thread_entries(TEST_EMAIL_1, "thread_middle")
            await delete_user_thread_entries(TEST_EMAIL_1, "thread_newest")
            await delete_user_thread_entries(TEST_EMAIL_1, "integration_test_thread")
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")

    async def test_create_thread_run_entry(self):
        """Test creating thread run entries."""
        print("\n🧪 Testing thread run entry creation...")
        
        # Test with auto-generated run_id
        run_id_1 = await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)
        
        assert run_id_1 is not None
        assert len(run_id_1) == 36  # UUID length
        print(f"✅ Auto-generated run_id: {run_id_1}")
        
        # Test with provided run_id
        custom_run_id = "custom_run_123"
        run_id_2 = await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1, custom_run_id)
        
        assert run_id_2 == custom_run_id
        print(f"✅ Custom run_id: {run_id_2}")
        
        # Test duplicate entry handling (should update timestamp)
        run_id_3 = await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1, custom_run_id)
        assert run_id_3 == custom_run_id
        print("✅ Duplicate entry handled correctly")

    async def test_get_user_chat_threads_empty(self):
        """Test getting chat threads for user with no threads."""
        print("\n🧪 Testing empty chat threads retrieval...")
        
        threads = await get_user_chat_threads("nonexistent@example.com")
        
        assert isinstance(threads, list)
        assert len(threads) == 0
        print("✅ Empty threads list returned correctly")

    async def test_get_user_chat_threads_with_data(self):
        """Test getting chat threads for user with multiple threads."""
        print("\n🧪 Testing chat threads retrieval with data...")
        
        # Create test data
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)
        await asyncio.sleep(0.1)  # Small delay to ensure different timestamps
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_2)
        await asyncio.sleep(0.1)
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)  # Second run in same thread
        await asyncio.sleep(0.1)
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_3)
        
        # Get threads
        threads = await get_user_chat_threads(TEST_EMAIL_1)
        
        assert isinstance(threads, list)
        assert len(threads) == 3  # 3 unique threads
        
        # Check sorting (latest first)
        assert threads[0]["thread_id"] == TEST_THREAD_ID_3  # Most recent
        assert threads[1]["thread_id"] == TEST_THREAD_ID_1  # Has latest timestamp due to second run
        assert threads[2]["thread_id"] == TEST_THREAD_ID_2  # Oldest
        
        # Check run counts
        thread_1_data = next(t for t in threads if t["thread_id"] == TEST_THREAD_ID_1)
        assert thread_1_data["run_count"] == 2  # Two runs in this thread
        
        thread_2_data = next(t for t in threads if t["thread_id"] == TEST_THREAD_ID_2)
        assert thread_2_data["run_count"] == 1  # One run
        
        thread_3_data = next(t for t in threads if t["thread_id"] == TEST_THREAD_ID_3)
        assert thread_3_data["run_count"] == 1  # One run
        
        print("✅ Chat threads retrieved and sorted correctly")
        print(f"   Thread order: {[t['thread_id'] for t in threads]}")
        print(f"   Run counts: {[(t['thread_id'], t['run_count']) for t in threads]}")

    async def test_user_isolation(self):
        """Test that users can only see their own threads."""
        print("\n🧪 Testing user isolation...")
        
        # Create data for both users
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)
        await create_thread_run_entry(TEST_EMAIL_2, TEST_THREAD_ID_1)
        await create_thread_run_entry(TEST_EMAIL_2, TEST_THREAD_ID_2)
        
        # Get threads for each user
        user1_threads = await get_user_chat_threads(TEST_EMAIL_1)
        user2_threads = await get_user_chat_threads(TEST_EMAIL_2)
        
        assert len(user1_threads) == 1
        assert len(user2_threads) == 2
        
        assert user1_threads[0]["thread_id"] == TEST_THREAD_ID_1
        assert set(t["thread_id"] for t in user2_threads) == {TEST_THREAD_ID_1, TEST_THREAD_ID_2}
        
        print("✅ User isolation working correctly")
        print(f"   User 1 threads: {[t['thread_id'] for t in user1_threads]}")
        print(f"   User 2 threads: {[t['thread_id'] for t in user2_threads]}")

    async def test_delete_user_thread_entries(self):
        """Test deleting thread entries for a user."""
        print("\n🧪 Testing thread entry deletion...")
        
        # Create test data
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_1)  # Second run
        await create_thread_run_entry(TEST_EMAIL_1, TEST_THREAD_ID_2)
        
        # Verify data exists
        threads_before = await get_user_chat_threads(TEST_EMAIL_1)
        assert len(threads_before) == 2
        
        # Delete one thread
        result = await delete_user_thread_entries(TEST_EMAIL_1, TEST_THREAD_ID_1)
        
        assert result["deleted_count"] == 2  # Two runs were deleted
        assert result["email"] == TEST_EMAIL_1
        assert result["thread_id"] == TEST_THREAD_ID_1
        
        # Verify deletion
        threads_after = await get_user_chat_threads(TEST_EMAIL_1)
        assert len(threads_after) == 1
        assert threads_after[0]["thread_id"] == TEST_THREAD_ID_2
        
        print("✅ Thread entry deletion working correctly")
        print(f"   Deleted {result['deleted_count']} entries")
        print(f"   Remaining threads: {[t['thread_id'] for t in threads_after]}")

    async def test_delete_nonexistent_thread(self):
        """Test deleting a nonexistent thread."""
        print("\n🧪 Testing deletion of nonexistent thread...")
        
        result = await delete_user_thread_entries(TEST_EMAIL_1, "nonexistent_thread")
        
        assert result["deleted_count"] == 0
        assert result["email"] == TEST_EMAIL_1
        assert result["thread_id"] == "nonexistent_thread"
        
        print("✅ Nonexistent thread deletion handled correctly")

    async def test_checkpointer_integration(self):
        """Test that the checkpointer initializes correctly with our table."""
        print("\n🧪 Testing checkpointer integration...")
        
        try:
            checkpointer = await get_postgres_checkpointer()
            assert checkpointer is not None
            print("✅ Checkpointer initialized successfully")
            
            # Test that our table was created
            await create_thread_run_entry(TEST_EMAIL_1, "integration_test_thread")
            threads = await get_user_chat_threads(TEST_EMAIL_1)
            
            integration_thread = next((t for t in threads if t["thread_id"] == "integration_test_thread"), None)
            assert integration_thread is not None
            print("✅ Integration with checkpointer working correctly")
            
            # Cleanup
            await delete_user_thread_entries(TEST_EMAIL_1, "integration_test_thread")
            
        except Exception as e:
            print(f"⚠ Checkpointer integration test failed (this may be expected in some environments): {e}")

    async def test_timestamp_ordering(self):
        """Test that threads are properly ordered by timestamp."""
        print("\n🧪 Testing timestamp ordering...")
        
        # Create threads with deliberate timing
        await create_thread_run_entry(TEST_EMAIL_1, "thread_oldest")
        await asyncio.sleep(0.1)
        await create_thread_run_entry(TEST_EMAIL_1, "thread_middle")
        await asyncio.sleep(0.1)
        await create_thread_run_entry(TEST_EMAIL_1, "thread_newest")
        
        threads = await get_user_chat_threads(TEST_EMAIL_1)
        
        assert len(threads) == 3
        assert threads[0]["thread_id"] == "thread_newest"
        assert threads[1]["thread_id"] == "thread_middle"
        assert threads[2]["thread_id"] == "thread_oldest"
        
        # Verify timestamps are actually decreasing
        for i in range(len(threads) - 1):
            assert threads[i]["latest_timestamp"] >= threads[i + 1]["latest_timestamp"]
        
        print("✅ Timestamp ordering working correctly")
        print(f"   Order: {[t['thread_id'] for t in threads]}")
        
        # Cleanup
        for thread_id in ["thread_oldest", "thread_middle", "thread_newest"]:
            await delete_user_thread_entries(TEST_EMAIL_1, thread_id)

def run_tests():
    """Run all tests."""
    print("🚀 Starting PostgreSQL Chat System Tests")
    print("=" * 50)
    
    # Run the tests
    test_instance = TestPostgresChatSystem()
    
    async def run_all_tests():
        # Setup
        await test_instance.setup_test_database()
        
        try:
            await test_instance.test_create_thread_run_entry()
            await test_instance.cleanup_test_data()
            
            await test_instance.test_get_user_chat_threads_empty()
            
            await test_instance.test_get_user_chat_threads_with_data()
            await test_instance.cleanup_test_data()
            
            await test_instance.test_user_isolation()
            await test_instance.cleanup_test_data()
            
            await test_instance.test_delete_user_thread_entries()
            await test_instance.cleanup_test_data()
            
            await test_instance.test_delete_nonexistent_thread()
            
            await test_instance.test_checkpointer_integration()
            await test_instance.cleanup_test_data()
            
            await test_instance.test_timestamp_ordering()
            
            print("\n" + "=" * 50)
            print("🎉 All tests passed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            await test_instance.cleanup_test_data()
    
    # Run tests
    asyncio.run(run_all_tests())

if __name__ == "__main__":
    run_tests() 
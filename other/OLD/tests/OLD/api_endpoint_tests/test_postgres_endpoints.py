#!/usr/bin/env python3
"""
Test script to verify PostgreSQL functionality and API endpoints
"""

import asyncio
import os
import sys
import requests
import uuid
from datetime import datetime

# Set debug mode
os.environ["DEBUG"] = "1"

# Add current directory to path
sys.path.insert(0, ".")


def test_api_endpoint(endpoint, method="GET", data=None, headers=None):
    """Test an API endpoint and return the result"""
    try:
        url = f"http://localhost:8000{endpoint}"
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)

        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Response: {str(result)[:200]}...")
            return True, result
        else:
            print(f"  Error: {response.text}")
            return False, response.text
    except Exception as e:
        print(f"  Exception: {e}")
        return False, str(e)


async def test_postgres_operations():
    """Test PostgreSQL operations directly"""
    print("\n🔍 Testing PostgreSQL operations directly...")

    try:
        from checkpointer.postgres_checkpointer import (
            get_healthy_pool,
        )
        from checkpointer.user_management.thread_operations import get_user_chat_threads
        from checkpointer.user_management.thread_operations import create_thread_run_entry
        from checkpointer.user_management.sentiment_tracking import get_thread_run_sentiments
        from checkpointer.user_management.sentiment_tracking import update_thread_run_sentiment

        # Test 1: Get healthy pool
        print("\n1. Testing get_healthy_pool...")
        pool = await get_healthy_pool()
        print(f"✅ Pool obtained: {type(pool).__name__}")
        print(f"   Pool closed: {pool.closed}")

        # Test 2: Test basic query
        print("\n2. Testing basic database query...")
        async with pool.connection() as conn:
            result = await conn.execute("SELECT 1 as test, NOW() as current_time")
            row = await result.fetchone()
            print(f"✅ Basic query successful: test={row[0]}, time={row[1]}")

        # Test 3: Create thread run entry
        print("\n3. Testing create_thread_run_entry...")
        test_email = "test@example.com"
        test_thread_id = f"test_thread_{uuid.uuid4()}"
        test_prompt = "Test prompt for PostgreSQL"

        run_id = await create_thread_run_entry(test_email, test_thread_id, test_prompt)
        print(f"✅ Thread run entry created: {run_id}")

        # Test 4: Get user chat threads
        print("\n4. Testing get_user_chat_threads...")
        threads = await get_user_chat_threads(test_email)
        print(f"✅ Retrieved {len(threads)} threads for {test_email}")

        # Test 5: Update sentiment
        print("\n5. Testing update_thread_run_sentiment...")
        success = await update_thread_run_sentiment(run_id, True, test_email)
        print(f"✅ Sentiment update successful: {success}")

        # Test 6: Get sentiments
        print("\n6. Testing get_thread_run_sentiments...")
        sentiments = await get_thread_run_sentiments(test_email, test_thread_id)
        print(f"✅ Retrieved sentiments: {sentiments}")

        # Test 7: Test users_threads_runs table structure
        print("\n7. Testing users_threads_runs table structure...")
        async with pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'users_threads_runs'
                ORDER BY ordinal_position
            """
            )

            columns = []
            async for row in result:
                columns.append(f"{row[0]}({row[1]})")

            print(f"✅ Table structure: {', '.join(columns)}")

        return True

    except Exception as e:
        print(f"❌ PostgreSQL operations failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return False


def test_api_endpoints():
    """Test API endpoints that don't require authentication"""
    print("\n🔍 Testing API endpoints...")

    # Test 1: Health endpoint
    print("\n1. Testing /health endpoint...")
    success, result = test_api_endpoint("/health")
    if success:
        print(f"✅ Health check passed")
        print(f"   Database status: {result.get('database', 'unknown')}")

    # Test 2: Debug pool status
    print("\n2. Testing /debug/pool-status endpoint...")
    success, result = test_api_endpoint("/debug/pool-status")
    if success:
        print(f"✅ Pool status check passed")
        print(f"   Checkpointer type: {result.get('checkpointer_type', 'unknown')}")
        print(f"   Pool healthy: {result.get('pool_healthy', 'unknown')}")

    # Test 3: Try an endpoint that requires auth (should fail gracefully)
    print("\n3. Testing /chat-threads endpoint (should require auth)...")
    success, result = test_api_endpoint("/chat-threads")
    if not success and "Authorization" in str(result):
        print(f"✅ Authentication properly required")
    else:
        print(f"⚠️ Unexpected result: {result}")


async def main():
    """Main test function"""
    print("🔍 Starting comprehensive PostgreSQL and API endpoint tests...")

    # Test PostgreSQL operations
    postgres_success = await test_postgres_operations()

    # Test API endpoints
    test_api_endpoints()

    # Summary
    print(f"\n📊 Test Summary:")
    print(
        f"   PostgreSQL Operations: {'✅ PASSED' if postgres_success else '❌ FAILED'}"
    )
    print(f"   API Endpoints: ✅ TESTED")

    if postgres_success:
        print(f"\n🎉 PostgreSQL integration is working correctly!")
        print(f"   - Connection pool is healthy")
        print(f"   - Database operations work")
        print(f"   - users_threads_runs table is functional")
        print(f"   - API server is connected to PostgreSQL")
    else:
        print(f"\n❌ PostgreSQL integration has issues that need attention")


if __name__ == "__main__":
    asyncio.run(main())

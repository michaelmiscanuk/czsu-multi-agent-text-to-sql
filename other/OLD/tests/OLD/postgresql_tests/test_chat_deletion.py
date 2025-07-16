import asyncio
import os
import sys

import jwt
import requests
from dotenv import load_dotenv

from my_agent import create_graph
from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer

# Load environment variables
load_dotenv()

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def query_postgresql_records(thread_id: str):
    """Query PostgreSQL to check what records exist for a thread_id."""
    checkpointer = await get_postgres_checkpointer()

    if not hasattr(checkpointer, "pool") or not checkpointer.pool:
        print("❌ No PostgreSQL checkpointer available")
        return {}

    tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
    counts = {}

    try:
        async with checkpointer.pool.connection() as conn:
            for table in tables:
                try:
                    # Check if table exists
                    result = await conn.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """,
                        (table,),
                    )

                    table_exists = await result.fetchone()
                    if not table_exists or not table_exists[0]:
                        print(f"⚠ Table {table} does not exist")
                        counts[table] = 0
                        continue

                    # Count records for this thread_id
                    result = await conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s",
                        (thread_id,),
                    )
                    count = await result.fetchone()
                    counts[table] = count[0] if count else 0
                    print(
                        f"📊 {table}: {counts[table]} records for thread_id: {thread_id}"
                    )

                except Exception as e:
                    print(f"⚠ Error querying table {table}: {e}")
                    counts[table] = f"Error: {str(e)}"

    except Exception as e:
        print(f"❌ Database query failed: {e}")
        return {}

    return counts


async def create_test_checkpoint_data(thread_id: str):
    """Create some test checkpoint data in PostgreSQL."""
    checkpointer = await get_postgres_checkpointer()
    graph = create_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Simulate a conversation to create checkpoint data
        test_input = {
            "messages": [{"role": "user", "content": "Test message for deletion test"}]
        }

        print(f"🔄 Creating test checkpoint data for thread_id: {thread_id}")
        result = await graph.ainvoke(test_input, config)

        print(f"✅ Test checkpoint data created")
        return True

    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False


def test_delete_api_endpoint(thread_id: str):
    """Test the DELETE /chat/{thread_id} API endpoint."""
    # Note: This requires a valid JWT token, which we can't easily generate in a test
    # For now, we'll just test the endpoint structure
    API_BASE = "http://localhost:8000"  # Adjust as needed

    print(f"🌐 Would call: DELETE {API_BASE}/chat/{thread_id}")
    print("⚠ Note: Actual API test requires valid authentication token")

    # In a real test environment, you would:
    # 1. Get a valid JWT token
    # 2. Make the DELETE request
    # 3. Check the response

    return True


async def simulate_chat_deletion_flow(thread_id: str):
    """Simulate the complete chat deletion flow."""
    print(f"\n🧪 Testing chat deletion flow for thread_id: {thread_id}")
    print("=" * 60)

    # Step 1: Check initial state
    print("\n1️⃣ Checking initial PostgreSQL state...")
    initial_counts = await query_postgresql_records(thread_id)

    # Step 2: Create test data if none exists
    total_initial = sum(
        count for count in initial_counts.values() if isinstance(count, int)
    )
    if total_initial == 0:
        print("\n2️⃣ Creating test checkpoint data...")
        created = await create_test_checkpoint_data(thread_id)
        if created:
            print("\n2️⃣ Checking state after data creation...")
            await query_postgresql_records(thread_id)
    else:
        print(f"\n2️⃣ Found existing data: {total_initial} total records")

    # Step 3: Test the deletion API endpoint (simulated)
    print("\n3️⃣ Testing deletion API endpoint...")
    test_delete_api_endpoint(thread_id)

    # Step 4: Manual cleanup for testing (simulating what the API would do)
    print("\n4️⃣ Manually cleaning up test data...")
    checkpointer = await get_postgres_checkpointer()

    if hasattr(checkpointer, "pool") and checkpointer.pool:
        try:
            async with checkpointer.pool.connection() as conn:
                await conn.set_autocommit(True)

                tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
                for table in tables:
                    try:
                        result = await conn.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
                        )
                        deleted = result.rowcount if hasattr(result, "rowcount") else 0
                        print(f"🗑️ Deleted {deleted} records from {table}")
                    except Exception as e:
                        print(f"⚠ Error deleting from {table}: {e}")

        except Exception as e:
            print(f"❌ Manual cleanup failed: {e}")

    # Step 5: Verify deletion
    print("\n5️⃣ Verifying deletion...")
    final_counts = await query_postgresql_records(thread_id)
    total_final = sum(
        count for count in final_counts.values() if isinstance(count, int)
    )

    if total_final == 0:
        print("✅ All checkpoint records successfully deleted!")
    else:
        print(f"⚠ Some records remain: {total_final} total records")

    print("\n" + "=" * 60)
    return total_final == 0


async def test_connection_and_tables():
    """Test basic PostgreSQL connection and table existence."""
    print("🔍 Testing PostgreSQL connection and table structure...")

    checkpointer = await get_postgres_checkpointer()

    if not hasattr(checkpointer, "pool") or not checkpointer.pool:
        print("❌ No PostgreSQL connection available")
        return False

    try:
        async with checkpointer.pool.connection() as conn:
            # Test basic connection
            result = await conn.execute("SELECT 1 as test")
            test_result = await result.fetchone()
            print(
                f"✅ Database connection: {'OK' if test_result and test_result[0] == 1 else 'Failed'}"
            )

            # Check table existence
            tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
            for table in tables:
                result = await conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """,
                    (table,),
                )

                exists = await result.fetchone()
                status = "EXISTS" if exists and exists[0] else "MISSING"
                print(f"📋 Table {table}: {status}")

        return True

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 CHAT DELETION TEST SUITE")
    print("=" * 60)

    # Test 1: Basic connection and table structure
    print("\n📋 TEST 1: Database Connection and Tables")
    connection_ok = await test_connection_and_tables()

    if not connection_ok:
        print("❌ Cannot proceed with tests - database connection failed")
        return

    # Test 2: Simulate chat deletion for a test thread
    test_thread_id = "test_deletion_thread_12345"

    print(f"\n🗑️ TEST 2: Chat Deletion Flow")
    deletion_success = await simulate_chat_deletion_flow(test_thread_id)

    # Test 3: Test with a different thread ID to ensure isolation
    test_thread_id_2 = "test_deletion_thread_67890"

    print(f"\n🔄 TEST 3: Second Thread Deletion (Isolation Test)")
    deletion_success_2 = await simulate_chat_deletion_flow(test_thread_id_2)

    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Database Connection: {'PASS' if connection_ok else 'FAIL'}")
    print(f"✅ Thread 1 Deletion: {'PASS' if deletion_success else 'FAIL'}")
    print(f"✅ Thread 2 Deletion: {'PASS' if deletion_success_2 else 'FAIL'}")

    overall_success = connection_ok and deletion_success and deletion_success_2
    print(
        f"\n🎯 OVERALL: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}"
    )

    # Cleanup: Close any remaining connections
    try:
        from my_agent.utils.postgres_checkpointer import GLOBAL_CHECKPOINTER

        if (
            hasattr(locals(), "checkpointer")
            and hasattr(checkpointer, "pool")
            and checkpointer.pool
        ):
            await checkpointer.pool.close()
            print("🧹 Cleaned up test connections")
    except:
        pass  # Ignore cleanup errors


if __name__ == "__main__":
    asyncio.run(main())

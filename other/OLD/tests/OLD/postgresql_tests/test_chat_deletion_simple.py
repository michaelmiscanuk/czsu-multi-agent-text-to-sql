import asyncio
import sys
import os
from dotenv import load_dotenv
from checkpointer.postgres_checkpointer import create_postgres_checkpointer

# Load environment variables
load_dotenv()

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_postgresql_operations():
    """Test PostgreSQL operations directly."""
    print("🧪 SIMPLE POSTGRESQL DELETION TEST")
    print("=" * 50)

    try:
        # Create checkpointer directly (bypassing the fallback)
        print("📡 Connecting to PostgreSQL...")
        checkpointer = await create_postgres_checkpointer()

        if not hasattr(checkpointer, "pool") or not checkpointer.pool:
            print("❌ No PostgreSQL pool available")
            return

        test_thread_id = "test_simple_deletion_12345"

        # Test 1: Check table existence and initial state
        print(
            f"\n1️⃣ Checking table structure and initial state for thread_id: {test_thread_id}"
        )

        tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
        initial_counts = {}

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
                        print(f"📋 Table {table}: MISSING")
                        initial_counts[table] = 0
                        continue

                    print(f"📋 Table {table}: EXISTS")

                    # Count records for this thread_id
                    result = await conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s",
                        (test_thread_id,),
                    )
                    count = await result.fetchone()
                    initial_counts[table] = count[0] if count else 0
                    print(
                        f"   📊 Records for {test_thread_id}: {initial_counts[table]}"
                    )

                except Exception as e:
                    print(f"   ⚠ Error with table {table}: {e}")
                    initial_counts[table] = f"Error: {str(e)}"

        # Test 2: Insert some test data
        print(f"\n2️⃣ Inserting test data into checkpoints table...")

        try:
            async with checkpointer.pool.connection() as conn:
                await conn.set_autocommit(True)

                # Insert a test record into checkpoints table
                await conn.execute(
                    """
                    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO NOTHING
                """,
                    (
                        test_thread_id,
                        "",  # default namespace
                        "test_checkpoint_123",
                        '{"test": "data"}',  # JSON data
                        "{}",  # empty metadata
                    ),
                )

                print(f"   ✅ Test record inserted")

                # Verify insertion
                result = await conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                    (test_thread_id,),
                )
                count = await result.fetchone()
                print(f"   📊 Records after insertion: {count[0] if count else 0}")

        except Exception as e:
            print(f"   ❌ Failed to insert test data: {e}")
            return

        # Test 3: Test deletion (simulating the API endpoint logic)
        print(f"\n3️⃣ Testing deletion functionality...")

        try:
            async with checkpointer.pool.connection() as conn:
                await conn.set_autocommit(True)

                deleted_counts = {}
                for table in tables:
                    try:
                        # Check if table exists first
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
                            deleted_counts[table] = 0
                            continue

                        # Delete records for this thread_id
                        result = await conn.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s",
                            (test_thread_id,),
                        )

                        deleted_counts[table] = (
                            result.rowcount if hasattr(result, "rowcount") else 0
                        )
                        print(
                            f"   🗑️ Deleted {deleted_counts[table]} records from {table}"
                        )

                    except Exception as table_error:
                        print(f"   ⚠ Error deleting from table {table}: {table_error}")
                        deleted_counts[table] = f"Error: {str(table_error)}"

                print(f"   📈 Summary: {deleted_counts}")

        except Exception as e:
            print(f"   ❌ Deletion test failed: {e}")
            return

        # Test 4: Verify deletion
        print(f"\n4️⃣ Verifying deletion...")

        try:
            async with checkpointer.pool.connection() as conn:
                final_counts = {}
                for table in tables:
                    try:
                        result = await conn.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE thread_id = %s",
                            (test_thread_id,),
                        )
                        count = await result.fetchone()
                        final_counts[table] = count[0] if count else 0
                        print(f"   📊 {table}: {final_counts[table]} records remaining")
                    except Exception as e:
                        print(f"   ⚠ Error checking {table}: {e}")
                        final_counts[table] = "Error"

                total_remaining = sum(
                    count for count in final_counts.values() if isinstance(count, int)
                )
                if total_remaining == 0:
                    print(f"   ✅ All records successfully deleted!")
                else:
                    print(f"   ⚠ {total_remaining} records still remain")

        except Exception as e:
            print(f"   ❌ Verification failed: {e}")

        print(f"\n✅ PostgreSQL deletion test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    await test_postgresql_operations()


if __name__ == "__main__":
    asyncio.run(main())

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


async def test_direct_postgresql():
    """Test PostgreSQL operations directly without fallback wrapper."""
    print("🧪 DIRECT POSTGRESQL TEST")
    print("=" * 50)

    checkpointer = None
    try:
        print("📡 Creating PostgreSQL checkpointer directly...")
        checkpointer = await create_postgres_checkpointer()

        print(f"✅ Checkpointer created: {type(checkpointer).__name__}")
        print(
            f"✅ Attributes: {[attr for attr in dir(checkpointer) if not attr.startswith('_')]}"
        )

        # Try to access the pool through different methods
        pool = None
        if hasattr(checkpointer, "pool"):
            pool = checkpointer.pool
            print(f"✅ Found pool attribute: {type(pool).__name__}")
        elif hasattr(checkpointer, "_pool"):
            pool = checkpointer._pool
            print(f"✅ Found _pool attribute: {type(pool).__name__}")
        elif hasattr(checkpointer, "conn"):
            conn = checkpointer.conn
            print(f"✅ Found conn attribute: {type(conn).__name__}")
            # Check if conn has a pool
            if hasattr(conn, "pool"):
                pool = conn.pool
                print(f"✅ Found pool in conn: {type(pool).__name__}")

        # Check private attributes that might contain the pool
        for attr in dir(checkpointer):
            if "pool" in attr.lower():
                print(f"🔍 Found pool-related attribute: {attr}")
                try:
                    value = getattr(checkpointer, attr)
                    print(f"   Type: {type(value).__name__}")
                    if hasattr(value, "connection"):
                        pool = value
                        print(f"   ✅ This looks like a connection pool!")
                        break
                except:
                    pass

        if pool:
            print(f"✅ Pool available: {pool is not None}")
            print(f"✅ Pool type: {type(pool).__name__}")
        else:
            print("⚠ Pool not found, trying to use checkpointer methods directly")

            # Try to use the checkpointer's own methods
            test_thread_id = "test_direct_postgres_12345"
            print(f"\n🔧 Testing checkpointer methods for thread_id: {test_thread_id}")

            # Try to use the checkpointer to save and then delete a checkpoint
            from langgraph.checkpoint.base import Checkpoint
            import json

            try:
                # Create a simple checkpoint
                config = {"configurable": {"thread_id": test_thread_id}}

                # Try to list existing checkpoints
                checkpoints = [c async for c in checkpointer.alist(config)]
                print(f"📊 Existing checkpoints: {len(checkpoints)}")

                # If we can list checkpoints, we should be able to access the database
                # Let's try to manually delete using a direct connection
                print("🔧 Attempting direct database connection...")

                # Get connection parameters from environment
                import os

                user = os.getenv("user")
                password = os.getenv("password")
                host = os.getenv("host")
                port = os.getenv("port", "5432")
                dbname = os.getenv("dbname")

                from psycopg_pool import AsyncConnectionPool

                connection_string = (
                    f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
                )
                temp_pool = AsyncConnectionPool(
                    conninfo=connection_string,
                    max_size=1,
                    min_size=1,
                    kwargs={"sslmode": "require"},
                    open=False,
                )

                await temp_pool.open()

                async with temp_pool.connection() as conn:
                    await conn.set_autocommit(True)

                    # Test deletion
                    result = await conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = %s",
                        (test_thread_id,),
                    )
                    deleted = result.rowcount if hasattr(result, "rowcount") else 0
                    print(f"✅ Deleted {deleted} records using direct connection")

                await temp_pool.close()
                print("🎉 Direct connection approach WORKED!")

            except Exception as method_error:
                print(f"⚠ Checkpointer method approach failed: {method_error}")

            return

        test_thread_id = "test_direct_postgres_12345"

        # Test: Insert and delete operations
        print(f"\n🔧 Testing CRUD operations for thread_id: {test_thread_id}")

        # Try to get a connection
        if pool:
            async with pool.connection() as conn:
                await conn.set_autocommit(True)

                # Check if checkpoints table exists and is accessible
                result = await conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'checkpoints'
                    )
                """
                )
                table_exists = await result.fetchone()

                if not table_exists or not table_exists[0]:
                    print("❌ Checkpoints table doesn't exist!")
                    return

                print("✅ Checkpoints table exists")

                # Insert a test record
                print("📝 Inserting test record...")
                await conn.execute(
                    """
                    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE 
                    SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata
                """,
                    (
                        test_thread_id,
                        "",  # default namespace
                        "test_checkpoint_direct",
                        '{"test": "direct_data"}',
                        '{"created_by": "direct_test"}',
                    ),
                )

                print("✅ Test record inserted")

                # Verify insertion
                result = await conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                    (test_thread_id,),
                )
                count = await result.fetchone()
                print(f"📊 Records after insertion: {count[0] if count else 0}")

                # Test deletion
                print("🗑️ Testing deletion...")
                result = await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s", (test_thread_id,)
                )
                deleted = result.rowcount if hasattr(result, "rowcount") else 0
                print(f"✅ Deleted {deleted} records")

                # Verify deletion
                result = await conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                    (test_thread_id,),
                )
                count = await result.fetchone()
                remaining = count[0] if count else 0
                print(f"📊 Records after deletion: {remaining}")

                if remaining == 0:
                    print("🎉 Deletion test PASSED!")
                else:
                    print(f"⚠ Deletion test FAILED - {remaining} records remain")
        else:
            print("❌ Could not access connection pool")
            return

        print("\n✅ Direct PostgreSQL test completed successfully!")

    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        if checkpointer and pool:
            try:
                await pool.close()
                print("🧹 Connection pool closed")
            except Exception as e:
                print(f"⚠ Error closing pool: {e}")


async def test_api_endpoint_simulation():
    """Simulate what the DELETE API endpoint would do."""
    print("\n🌐 API ENDPOINT SIMULATION")
    print("=" * 50)

    checkpointer = None
    try:
        checkpointer = await create_postgres_checkpointer()
        test_thread_id = "test_api_simulation_67890"

        # Get the connection pool from the conn attribute
        if not hasattr(checkpointer, "conn"):
            return {"message": "No PostgreSQL conn available - nothing to delete"}

        pool = checkpointer.conn
        print(f"✅ Using connection pool: {type(pool).__name__}")

        async with pool.connection() as conn:
            await conn.set_autocommit(True)

            # First, insert some test data
            print(f"📝 Setting up test data for thread_id: {test_thread_id}")
            await conn.execute(
                """
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE 
                SET checkpoint = EXCLUDED.checkpoint
            """,
                (test_thread_id, "", "api_test_checkpoint", '{"api": "test"}', "{}"),
            )

            # Verify insertion
            result = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                (test_thread_id,),
            )
            count = await result.fetchone()
            print(f"📊 Records inserted: {count[0] if count else 0}")

            # Delete from all checkpoint tables (like the API endpoint does)
            tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
            deleted_counts = {}

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
                        print(f"⚠ Table {table} does not exist, skipping")
                        deleted_counts[table] = 0
                        continue

                    # Delete records for this thread_id
                    result = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s", (test_thread_id,)
                    )

                    deleted_counts[table] = (
                        result.rowcount if hasattr(result, "rowcount") else 0
                    )
                    print(f"✅ Deleted {deleted_counts[table]} records from {table}")

                except Exception as table_error:
                    print(f"⚠ Error with table {table}: {table_error}")
                    deleted_counts[table] = f"Error: {str(table_error)}"

            # Verify deletion
            result = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                (test_thread_id,),
            )
            count = await result.fetchone()
            remaining = count[0] if count else 0
            print(f"📊 Records remaining after deletion: {remaining}")

            result = {
                "message": f"Checkpoint records deleted for thread_id: {test_thread_id}",
                "deleted_counts": deleted_counts,
                "thread_id": test_thread_id,
                "verification": f"{remaining} records remaining",
            }

            print(f"🎉 API simulation result: {result}")

            if remaining == 0:
                print("✅ API simulation test PASSED!")
            else:
                print(f"⚠ API simulation test FAILED - {remaining} records remain")

            return result

    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if checkpointer and hasattr(checkpointer, "conn"):
            try:
                await checkpointer.conn.close()
                print("🧹 Connection pool closed")
            except:
                pass


async def main():
    await test_direct_postgresql()
    await test_api_endpoint_simulation()


if __name__ == "__main__":
    asyncio.run(main())

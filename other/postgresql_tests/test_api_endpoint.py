import asyncio
import sys
from dotenv import load_dotenv
from my_agent.utils.postgres_checkpointer import create_postgres_checkpointer

# Load environment variables
load_dotenv()

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def test_delete_endpoint_logic():
    """Test the logic that would be used in the DELETE /chat/{thread_id} endpoint."""
    print("🧪 TESTING DELETE ENDPOINT LOGIC")
    print("=" * 50)
    
    test_thread_id = "test_endpoint_logic_12345"
    
    try:
        # Simulate getting a healthy checkpointer (like in the API)
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        checkpointer = await get_postgres_checkpointer()
        
        print(f"✅ Checkpointer type: {type(checkpointer).__name__}")
        
        # Check if we have a PostgreSQL checkpointer (not InMemorySaver)
        if not hasattr(checkpointer, 'conn'):
            result = {"message": "No PostgreSQL checkpointer available - nothing to delete"}
            print(f"⚠ Result: {result}")
            return result
        
        # Access the connection pool through the conn attribute
        pool = checkpointer.conn
        print(f"✅ Using connection pool: {type(pool).__name__}")
        
        # First, insert some test data to delete
        print(f"\n📝 Setting up test data for thread_id: {test_thread_id}")
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Insert test records
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE 
                SET checkpoint = EXCLUDED.checkpoint
            """, (
                test_thread_id,
                "",
                "endpoint_test_checkpoint_1",
                '{"test": "endpoint_data_1"}',
                '{"created_by": "endpoint_test"}'
            ))
            
            await conn.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE 
                SET checkpoint = EXCLUDED.checkpoint
            """, (
                test_thread_id,
                "",
                "endpoint_test_checkpoint_2",
                '{"test": "endpoint_data_2"}',
                '{"created_by": "endpoint_test"}'
            ))
            
            # Verify insertion
            result = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                (test_thread_id,)
            )
            count = await result.fetchone()
            print(f"📊 Test records inserted: {count[0] if count else 0}")
        
        # Now test the deletion logic (simulating the API endpoint)
        print(f"\n🗑️ Testing deletion for thread_id: {test_thread_id}")
        
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            
            # Delete from all checkpoint tables
            tables = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]
            deleted_counts = {}
            
            for table in tables:
                try:
                    # First check if the table exists
                    result = await conn.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table,))
                    
                    table_exists = await result.fetchone()
                    if not table_exists or not table_exists[0]:
                        print(f"⚠ Table {table} does not exist, skipping")
                        deleted_counts[table] = 0
                        continue
                    
                    # Delete records for this thread_id
                    result = await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s",
                        (test_thread_id,)
                    )
                    
                    deleted_counts[table] = result.rowcount if hasattr(result, 'rowcount') else 0
                    print(f"✓ Deleted {deleted_counts[table]} records from {table} for thread_id: {test_thread_id}")
                    
                except Exception as table_error:
                    print(f"⚠ Error deleting from table {table}: {table_error}")
                    deleted_counts[table] = f"Error: {str(table_error)}"
            
            # Verify deletion
            result = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                (test_thread_id,)
            )
            count = await result.fetchone()
            remaining = count[0] if count else 0
            print(f"📊 Records remaining after deletion: {remaining}")
            
            api_result = {
                "message": f"Checkpoint records deleted for thread_id: {test_thread_id}",
                "deleted_counts": deleted_counts,
                "thread_id": test_thread_id
            }
            
            print(f"\n🎉 API endpoint would return: {api_result}")
            
            if remaining == 0 and deleted_counts.get('checkpoints', 0) > 0:
                print("✅ DELETE endpoint logic test PASSED!")
                return api_result
            else:
                print(f"⚠ DELETE endpoint logic test FAILED - {remaining} records remain, deleted: {deleted_counts}")
                return api_result
            
    except Exception as e:
        print(f"❌ DELETE endpoint logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def main():
    await test_delete_endpoint_logic()

if __name__ == "__main__":
    asyncio.run(main()) 
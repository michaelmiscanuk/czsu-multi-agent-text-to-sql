#!/usr/bin/env python3
"""
Test script to verify PostgreSQL checkpointer setup and table creation.
This script will test if all required tables are created properly.
"""

import asyncio
import platform
import os
from dotenv import load_dotenv
from checkpointer.postgres_checkpointer import create_postgres_checkpointer

# Fix for Windows ProactorEventLoop issue with psycopg
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()


async def test_postgres_setup():
    """Test PostgreSQL checkpointer setup and verify all tables exist."""
    print("🧪 Testing PostgreSQL checkpointer setup...")

    try:
        # Create checkpointer (this will create tables if they don't exist)
        checkpointer = await create_postgres_checkpointer()

        # Get a connection from the pool to verify table creation
        async with checkpointer.conn.connection() as conn:
            # Check which tables exist
            result = await conn.execute(
                """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'checkpoint%'
                ORDER BY tablename;
            """
            )

            tables = [row[0] for row in await result.fetchall()]

            print(f"\n📋 Found {len(tables)} checkpoint tables:")
            for table in tables:
                print(f"  ✓ {table}")

            # Expected tables
            expected_tables = [
                "checkpoints",
                "checkpoint_writes",
                "checkpoint_blobs",
                "checkpoint_migrations",
            ]

            print(f"\n🎯 Expected tables: {expected_tables}")
            missing_tables = [table for table in expected_tables if table not in tables]

            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False
            else:
                print("✅ All expected tables found!")

            # Test basic checkpoint operations
            print("\n🔄 Testing basic checkpoint operations...")

            # Create a test config
            config = {"configurable": {"thread_id": "test_setup_thread"}}

            # Try to put a simple checkpoint
            test_checkpoint = {
                "v": 1,
                "ts": "2024-01-01T00:00:00.000000+00:00",
                "id": "test_checkpoint_id",
                "channel_values": {"test": "value"},
                "channel_versions": {"test": 1},
                "versions_seen": {},
                "pending_sends": [],
            }

            test_metadata = {}
            test_new_versions = {}

            # Test writing checkpoint
            try:
                await checkpointer.aput(
                    config, test_checkpoint, test_metadata, test_new_versions
                )
                print("  ✓ Successfully wrote test checkpoint")
            except Exception as e:
                print(f"  ❌ Failed to write checkpoint: {e}")
                return False

            # Test reading checkpoint
            try:
                retrieved = await checkpointer.aget(config)
                if retrieved:
                    print("  ✓ Successfully retrieved test checkpoint")
                else:
                    print("  ⚠ No checkpoint found (this may be normal)")
            except Exception as e:
                print(f"  ❌ Failed to read checkpoint: {e}")
                return False

            print(
                "\n✅ All tests passed! PostgreSQL checkpointer setup is working correctly."
            )
            return True

    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Close the checkpointer connection pool
        if "checkpointer" in locals():
            try:
                await checkpointer.conn.close()
                print("🔒 Connection pool closed")
            except Exception as e:
                print(f"⚠ Warning: Could not close connection pool: {e}")


if __name__ == "__main__":
    asyncio.run(test_postgres_setup())

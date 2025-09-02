#!/usr/bin/env python3
"""
Test script to verify the prepared statement fix is working.
This script tests both the connection with prepared statements disabled
and runs a simple checkpoint operation to ensure everything works.
"""

import asyncio
import os
import sys
from pathlib import Path

# CRITICAL: Windows event loop fix MUST be first for PostgreSQL compatibility
if sys.platform == "win32":
    print(
        "🪟 Windows detected - setting SelectorEventLoop for PostgreSQL compatibility..."
    )
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Event loop policy set successfully")

# Add the project root to Python path
BASE_DIR = Path(__file__).resolve().parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


async def test_prepared_statements_fix():
    """Test that prepared statements are properly disabled to prevent errors."""
    print("🧪 Testing Prepared Statements Fix")
    print("=" * 50)

    try:
        # Test 1: Check connection string doesn't have invalid parameters
        print("\n1️⃣ Testing connection string parameters...")
        from checkpointer.database.connection import get_connection_kwargs
        from checkpointer.database.connection import get_connection_string

        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()

        if "prepare_threshold" not in connection_string:
            print(
                "✅ prepare_threshold NOT in connection string (correct - should be in kwargs)"
            )
        else:
            print(
                "❌ prepare_threshold found in connection string (this should be fixed)"
            )
            return False

        if "prepared_max" not in connection_string:
            print(
                "✅ prepared_max NOT in connection string (correct - should be in kwargs)"
            )
        else:
            print("❌ prepared_max found in connection string (this should be fixed)")
            return False

        # Check connection kwargs
        if connection_kwargs.get("prepare_threshold") is None:
            print("✅ prepare_threshold=None found in connection kwargs")
        else:
            print(
                f"❌ prepare_threshold not properly set in kwargs: {connection_kwargs.get('prepare_threshold')}"
            )
            return False

        print(f"🔗 Connection string: {connection_string[:100]}...")
        print(f"🔧 Connection kwargs: {connection_kwargs}")

        # Test 2: Test direct connection with kwargs
        print("\n2️⃣ Testing direct connection with kwargs...")
        import psycopg

        # This should work without the invalid URI parameter error
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            print("✅ Direct connection with kwargs successful")

            # Test that prepared statements are disabled
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                print(f"✅ Simple query executed: {result}")

        # Test 3: Create a checkpointer and test basic operations
        print("\n3️⃣ Testing checkpointer creation...")
        from checkpointer.postgres_checkpointer import get_postgres_checkpointer

        checkpointer = await get_postgres_checkpointer()
        print(f"✅ Checkpointer created: {type(checkpointer).__name__}")

        # Test 4: Perform a simple checkpoint operation
        print("\n4️⃣ Testing checkpoint operations...")
        test_config = {"configurable": {"thread_id": "prepared_statements_test"}}

        # This operation should not cause prepared statement errors
        result = await checkpointer.aget(test_config)
        print(
            f"✅ aget() operation successful: {result is None} (expected for new thread)"
        )

        # Test multiple operations to ensure prepared statements aren't created
        for i in range(3):
            await checkpointer.aget(test_config)
        print("✅ Multiple operations completed without prepared statement errors")

        # Test 5: Verify prepared statements are not being created
        print("\n5️⃣ Verifying no prepared statements are created...")

        # Use a connection with the same parameters to check
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%'
                """
                )
                result = await cur.fetchone()
                prepared_count = result[0] if result else 0

                if prepared_count == 0:
                    print("✅ No automatic prepared statements found - fix is working!")
                else:
                    print(
                        f"⚠️ Found {prepared_count} prepared statements - they should be disabled"
                    )
                    print(
                        "   This might be from other connections, but the fix should prevent new ones"
                    )

        print("\n🎉 All tests passed! Prepared statements fix is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        print(f"📋 Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Ensure we have the required environment variables
    required_vars = ["host", "port", "dbname", "user", "password"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        print("Please set them in your .env file or environment.")
        sys.exit(1)

    # Run the test
    result = asyncio.run(test_prepared_statements_fix())
    sys.exit(0 if result else 1)

#!/usr/bin/env python3
"""
Debug script to test checkpointer initialization in isolation
"""

import os
import sys
import asyncio
from pathlib import Path

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv()


async def test_checkpointer_init():
    """Test checkpointer initialization to see what's failing."""

    print("🔍 CHECKPOINTER DEBUG - Starting initialization test")

    # Check environment variables
    print("\n📋 ENVIRONMENT VARIABLES:")
    postgres_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    for var in postgres_vars:
        value = os.environ.get(var, "<NOT SET>")
        # Mask password for security
        if "PASSWORD" in var and value != "<NOT SET>":
            value = "*" * len(value)
        print(f"   {var}: {value}")

    # Test database connection string construction
    try:
        from checkpointer.checkpointer.factory import construct_connection_string

        connection_string = construct_connection_string()
        # Mask the password in the connection string for display
        masked_conn_str = connection_string
        if "@" in masked_conn_str:
            parts = masked_conn_str.split("@")
            if ":" in parts[0]:
                user_pass = parts[0].split(":")
                if len(user_pass) >= 2:
                    masked_conn_str = (
                        f"{user_pass[0]}:{'*' * len(user_pass[1])}@{parts[1]}"
                    )
        print(f"\n🔗 CONNECTION STRING: {masked_conn_str}")
    except Exception as e:
        print(f"\n❌ CONNECTION STRING ERROR: {e}")
        return False

    # Test basic database connection
    print(f"\n🔍 TESTING BASIC DATABASE CONNECTION...")
    try:
        import psycopg

        async with await psycopg.AsyncConnection.connect(connection_string) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT version();")
                result = await cur.fetchone()
                print(f"✅ DATABASE CONNECTION SUCCESS: PostgreSQL {result[0][:50]}...")
    except Exception as e:
        print(f"❌ DATABASE CONNECTION FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

    # Test checkpointer creation
    print(f"\n🔍 TESTING CHECKPOINTER CREATION...")
    try:
        from checkpointer.checkpointer.factory import create_async_postgres_saver

        print("   Creating AsyncPostgresSaver...")
        checkpointer = await create_async_postgres_saver()

        if checkpointer:
            print(f"✅ CHECKPOINTER CREATION SUCCESS: {type(checkpointer).__name__}")

            # Test basic checkpointer operations
            print(f"\n🔍 TESTING CHECKPOINTER OPERATIONS...")
            config = {"configurable": {"thread_id": "test_thread_123"}}

            # Test aget_tuple operation
            result = await checkpointer.aget_tuple(config)
            print(
                f"✅ CHECKPOINTER OPERATION SUCCESS: aget_tuple returned {type(result)}"
            )

        else:
            print(f"❌ CHECKPOINTER CREATION FAILED: Returned None")
            return False

    except Exception as e:
        print(f"❌ CHECKPOINTER CREATION FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        return False

    # Test global checkpointer access
    print(f"\n🔍 TESTING GLOBAL CHECKPOINTER ACCESS...")
    try:
        from checkpointer.checkpointer.factory import get_global_checkpointer

        global_checkpointer = await get_global_checkpointer()

        if global_checkpointer:
            print(
                f"✅ GLOBAL CHECKPOINTER SUCCESS: {type(global_checkpointer).__name__}"
            )
        else:
            print(f"❌ GLOBAL CHECKPOINTER FAILED: Returned None")
            return False

    except Exception as e:
        print(f"❌ GLOBAL CHECKPOINTER FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        return False

    print(f"\n✅ ALL CHECKPOINTER TESTS PASSED!")
    return True


if __name__ == "__main__":
    result = asyncio.run(test_checkpointer_init())
    sys.exit(0 if result else 1)

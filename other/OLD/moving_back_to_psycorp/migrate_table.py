#!/usr/bin/env python3
"""
Manual migration script to fix the VARCHAR(50) issue in users_threads_runs table.
Run this script to apply the database schema migration.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Windows event loop policy if on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpointer.database.table_setup import setup_users_threads_runs_table


async def main():
    """Run the migration to fix VARCHAR(50) constraint issue."""
    print("🔧 Running database migration to fix VARCHAR(50) constraint...")

    try:
        # This will check for the old schema and migrate if needed
        await setup_users_threads_runs_table()
        print("✅ Migration completed successfully!")
        print("✅ Your application should now work correctly.")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 You can now try your question again in the application!")
    else:
        print(
            "\n💡 If migration failed, check your database connection settings in .env"
        )
        sys.exit(1)

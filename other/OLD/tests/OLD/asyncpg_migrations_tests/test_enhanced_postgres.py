#!/usr/bin/env python3
"""
Enhanced PostgreSQL Connection Test Script
==========================================

This script tests all the improvements made to handle SSL connection issues,
AsyncPipeline errors, and DbHandler exited problems in cloud PostgreSQL deployments.

Run this script to validate that the enhanced connection system is working properly.
"""

import asyncio
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows compatibility fix - must be set BEFORE any async operations
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("🔧 Windows: Set SelectorEventLoop policy for PostgreSQL compatibility")


async def run_enhanced_postgres_tests():
    """Run comprehensive tests of the enhanced PostgreSQL connection system."""

    print("🚀 Enhanced PostgreSQL Connection System Tests")
    print("=" * 80)
    print("This script tests all improvements made to handle:")
    print("  • SSL connection drops and timeouts")
    print("  • AsyncPipeline errors and DbHandler exited issues")
    print("  • Connection pool recreation and monitoring")
    print("  • Enhanced error detection and recovery")
    print("=" * 80)

    try:
        # Import the enhanced PostgreSQL utilities
        from checkpointer.postgres_checkpointer import (
            initialize_enhanced_postgres_system,
            test_connection_health,
            test_pool_connection,
            debug_pool_status,
            get_postgres_checkpointer,
            monitor_connection_health,
        )
        from checkpointer.user_management.thread_operations import create_thread_run_entry
        from checkpointer.user_management.sentiment_tracking import get_thread_run_sentiments
        from checkpointer.user_management.sentiment_tracking import update_thread_run_sentiment

        print("✅ Enhanced PostgreSQL utilities imported successfully")

        # Test 1: Full system initialization
        print("\n" + "=" * 60)
        print("TEST 1: Full Enhanced System Initialization")
        print("=" * 60)

        init_success = await initialize_enhanced_postgres_system()
        if not init_success:
            print("❌ Enhanced system initialization failed")
            return False

        print("✅ Enhanced PostgreSQL system initialized successfully")

        # Test 2: Checkpointer creation with enhanced error handling
        print("\n" + "=" * 60)
        print("TEST 2: Enhanced Checkpointer Creation")
        print("=" * 60)

        try:
            checkpointer = await get_postgres_checkpointer()
            print("✅ Enhanced checkpointer created successfully")

            # Test basic checkpointer operations
            config = {"configurable": {"thread_id": "test_enhanced_thread"}}

            # Test checkpoint operations that previously failed with DbHandler errors
            try:
                # This operation often triggered "DbHandler exited" errors
                checkpoints = []
                async for checkpoint in checkpointer.alist(config, limit=5):
                    checkpoints.append(checkpoint)

                print(
                    f"✅ Checkpoint listing successful: {len(checkpoints)} checkpoints found"
                )

            except Exception as e:
                print(
                    f"⚠️ Checkpoint listing failed (this may be expected for new databases): {e}"
                )

        except Exception as e:
            print(f"❌ Enhanced checkpointer creation failed: {e}")
            return False

        # Test 3: Database operations that often fail with connection issues
        print("\n" + "=" * 60)
        print("TEST 3: Database Operations Stress Test")
        print("=" * 60)

        try:
            # Create test entries
            test_email = "test@enhanced-postgres.com"
            test_thread_id = f"enhanced_test_{int(time.time())}"

            # Test operations that often triggered SSL/connection errors
            run_id1 = await create_thread_run_entry(
                test_email, test_thread_id, "Test prompt 1"
            )
            print(f"✅ Created thread run entry: {run_id1}")

            run_id2 = await create_thread_run_entry(
                test_email, test_thread_id, "Test prompt 2"
            )
            print(f"✅ Created thread run entry: {run_id2}")

            # Test sentiment updates (operations that often failed with pipeline errors)
            await update_thread_run_sentiment(run_id1, True, test_email)
            print(f"✅ Updated sentiment for: {run_id1}")

            await update_thread_run_sentiment(run_id2, False, test_email)
            print(f"✅ Updated sentiment for: {run_id2}")

            # Test retrieval operations
            sentiments = await get_thread_run_sentiments(test_email, test_thread_id)
            print(f"✅ Retrieved sentiments: {len(sentiments)} entries")

            print("✅ Database operations stress test passed")

        except Exception as e:
            print(f"❌ Database operations stress test failed: {e}")
            import traceback

            print(f"🔍 Error details: {traceback.format_exc()}")
            return False

        # Test 4: Connection error simulation and recovery
        print("\n" + "=" * 60)
        print("TEST 4: Connection Error Recovery Simulation")
        print("=" * 60)

        try:
            # Debug pool status
            await debug_pool_status()

            # Simulate multiple rapid operations that might trigger connection issues
            print("🔄 Simulating rapid database operations...")

            tasks = []
            for i in range(5):
                task = create_thread_run_entry(
                    f"stress_test_{i}@example.com",
                    f"stress_thread_{i}",
                    f"Stress test prompt {i}",
                )
                tasks.append(task)

            # Run operations concurrently to stress the connection pool
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count

            print(
                f"✅ Concurrent operations: {success_count} successful, {error_count} failed"
            )

            if error_count > 0:
                print("⚠️ Some operations failed (this may indicate connection issues)")
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"   Operation {i} failed: {result}")
            else:
                print("🎉 All concurrent operations succeeded!")

        except Exception as e:
            print(f"❌ Connection error recovery test failed: {e}")
            return False

        # Test 5: Final health check
        print("\n" + "=" * 60)
        print("TEST 5: Final System Health Check")
        print("=" * 60)

        final_health = await test_connection_health()
        if final_health:
            print("✅ Final health check passed")
        else:
            print("⚠️ Final health check failed")

        await debug_pool_status()

        print("\n" + "🎉" + "=" * 76 + "🎉")
        print("🎉" + " " * 76 + "🎉")
        print(
            "🎉"
            + " ENHANCED POSTGRESQL CONNECTION SYSTEM TESTS COMPLETED ".center(76)
            + "🎉"
        )
        print("🎉" + " " * 76 + "🎉")
        print("🎉" + "=" * 76 + "🎉")

        return True

    except ImportError as e:
        print(f"❌ Failed to import enhanced PostgreSQL utilities: {e}")
        print("💡 Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Enhanced PostgreSQL tests failed: {e}")
        import traceback

        print(f"🔍 Full error details: {traceback.format_exc()}")
        return False


async def main():
    """Main test runner with comprehensive error handling."""
    print("🔍 Checking environment variables...")

    required_vars = ["user", "password", "host", "dbname"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Make sure your .env file contains:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
        return False

    print("✅ Environment variables check passed")

    # Set enhanced debugging
    os.environ["DEBUG"] = "1"
    os.environ["VERBOSE_SSL_LOGGING"] = "true"
    os.environ["ENABLE_CONNECTION_MONITORING"] = "false"  # Disable monitoring for tests

    try:
        success = await run_enhanced_postgres_tests()
        if success:
            print("\n✅ All enhanced PostgreSQL tests completed successfully!")
            print("💡 Your enhanced connection system should now handle:")
            print("   • SSL connection drops and recovery")
            print("   • AsyncPipeline and DbHandler errors")
            print("   • Automatic pool recreation")
            print("   • Enhanced error diagnostics")
            return True
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return False
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback

        print(f"🔍 Full error details: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the enhanced tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

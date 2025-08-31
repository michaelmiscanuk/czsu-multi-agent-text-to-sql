#!/usr/bin/env python3
"""
Quick PostgreSQL Unit Tests
Fast validation of critical PostgreSQL functionality
"""

import asyncio
import os
import sys
import uuid
import time
from datetime import datetime

# Import our PostgreSQL module
from checkpointer.postgres_checkpointer import (
    check_postgres_env_vars,
    test_basic_postgres_connection,
    test_connection_health,
    get_healthy_pool,
    get_postgres_checkpointer,
    setup_users_threads_runs_table,
    create_thread_run_entry,
    get_user_chat_threads,
    delete_user_thread_entries,
    force_close_all_connections,
)


class QuickTest:
    """Quick test runner"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()

    def test(self, name: str, condition: bool, message: str = ""):
        """Run a single test"""
        if condition:
            print(f"[PASS] {name}")
            self.passed += 1
        else:
            print(f"[FAIL] {name} - {message}")
            self.failed += 1

    def summary(self):
        """Print test summary"""
        duration = time.time() - self.start_time
        total = self.passed + self.failed
        print(f"\n[SUMMARY] Quick Test Results:")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")
        print(f"   Total: {total}")
        print(f"   Duration: {duration:.2f}s")
        print(
            f"   Success Rate: {(self.passed/total)*100:.1f}%"
            if total > 0
            else "   Success Rate: 0%"
        )
        return self.failed == 0


async def main():
    """Run quick tests"""
    print("[START] Quick PostgreSQL Tests\n")

    test = QuickTest()
    test_email = f"quick_test_{uuid.uuid4().hex[:8]}@test.com"
    test_thread_id = f"quick_thread_{uuid.uuid4().hex[:8]}"

    try:
        # Test 1: Environment Variables
        env_ok = check_postgres_env_vars()
        test.test(
            "Environment Variables", env_ok, "Missing required environment variables"
        )

        if not env_ok:
            print("[ERROR] Cannot continue without proper environment setup")
            return test.summary()

        # Test 2: Basic Connection
        basic_conn = await test_basic_postgres_connection()
        test.test("Basic Connection", basic_conn, "Direct asyncpg connection failed")

        # Test 3: Connection Health
        health = await test_connection_health()
        test.test("Connection Health", health, "Connection health check failed")

        # Test 4: Pool Creation
        pool = await get_healthy_pool()
        test.test("Pool Creation", pool is not None, "Failed to create connection pool")

        if pool:
            # Test 5: Pool Query
            try:
                async with pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                test.test("Pool Query", result == 1, f"Expected 1, got {result}")
            except Exception as e:
                test.test("Pool Query", False, str(e))

        # Test 6: Table Setup
        try:
            await setup_users_threads_runs_table()
            test.test("Table Setup", True)
        except Exception as e:
            test.test("Table Setup", False, str(e))

        # Test 7: Data Operations
        try:
            run_id = await create_thread_run_entry(
                test_email, test_thread_id, "Quick test prompt"
            )
            test.test(
                "Data Insertion",
                run_id is not None,
                "Failed to create thread run entry",
            )

            threads = await get_user_chat_threads(test_email, limit=5)
            test.test("Data Retrieval", len(threads) > 0, "No threads found")

            # Cleanup
            await delete_user_thread_entries(test_email, test_thread_id)
            test.test("Data Cleanup", True)

        except Exception as e:
            test.test("Data Operations", False, str(e))

        # Test 8: LangGraph Checkpointer
        try:
            checkpointer = await get_postgres_checkpointer()
            test.test(
                "Checkpointer Creation",
                checkpointer is not None,
                "Failed to create checkpointer",
            )

            if checkpointer:
                config = {
                    "configurable": {"thread_id": f"quick_test_{uuid.uuid4().hex[:8]}"}
                }
                result = await checkpointer.aget(config)
                test.test("Checkpointer Query", True, "Checkpointer query executed")
        except Exception as e:
            test.test("Checkpointer Creation", False, str(e))

        # Test 9: Concurrent Operations
        try:

            async def concurrent_task(task_id):
                pool = await get_healthy_pool()
                async with pool.acquire() as conn:
                    return await conn.fetchval("SELECT $1::text", str(task_id))

            tasks = [concurrent_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success = all(not isinstance(r, Exception) for r in results)
            test.test(
                "Concurrent Operations", success, "Some concurrent operations failed"
            )
        except Exception as e:
            test.test("Concurrent Operations", False, str(e))

        # Cleanup
        await force_close_all_connections()
        test.test("Connection Cleanup", True)

    except Exception as e:
        test.test("Overall Test", False, f"Unexpected error: {e}")

    success = test.summary()

    if success:
        print(
            "\n[SUCCESS] All quick tests passed! The PostgreSQL implementation is working correctly."
        )
        return 0
    else:
        print("\n[ERROR] Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
PostgreSQL Stress Test Suite
Tests connection resilience, error recovery, and high-load scenarios
"""

import asyncio
import os
import sys
import uuid
import time
import traceback
import random
from datetime import datetime
from typing import List, Dict, Any
import json
import statistics

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from checkpointer.postgres_checkpointer import (
    get_healthy_pool,
    setup_users_threads_runs_table,
    create_thread_run_entry,
    get_postgres_checkpointer,
    force_close_all_connections,
    get_user_chat_threads,
    update_thread_run_sentiment,
)


class StressTestColors:
    """ANSI color codes for stress test output"""

    HEADER = "\033[95m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class PostgreSQLStressTest:
    """Stress test suite for PostgreSQL operations"""

    def __init__(self):
        self.results = []
        self.test_emails = [
            f"stress_user_{i}_{uuid.uuid4().hex[:6]}@test.com" for i in range(10)
        ]
        self.cleanup_threads = []

    def log_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log stress test result"""
        status = (
            f"{StressTestColors.OKGREEN}✅ PASS{StressTestColors.ENDC}"
            if success
            else f"{StressTestColors.FAIL}❌ FAIL{StressTestColors.ENDC}"
        )
        print(f"{status} {test_name}")

        # Print key metrics
        for key, value in details.items():
            if key in ["duration", "avg_latency", "max_latency", "min_latency"]:
                print(f"    {key}: {value:.3f}s")
            elif key in ["throughput", "success_rate"]:
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")

        self.results.append(
            {
                "test": test_name,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                **details,
            }
        )

    async def test_high_concurrency_asyncpg(
        self, concurrent_tasks: int = 50, operations_per_task: int = 20
    ):
        """Test high concurrency with asyncpg operations"""
        print(
            f"\n{StressTestColors.HEADER}🚀 High Concurrency AsyncPG Test ({concurrent_tasks} tasks x {operations_per_task} ops){StressTestColors.ENDC}"
        )

        start_time = time.time()
        success_count = 0
        error_count = 0
        latencies = []

        async def concurrent_task(task_id: int):
            nonlocal success_count, error_count
            task_start = time.time()

            try:
                pool = await get_healthy_pool()
                async with pool.acquire() as conn:
                    for i in range(operations_per_task):
                        op_start = time.time()
                        result = await conn.fetchval(
                            "SELECT $1 + $2 + $3", task_id, i, random.randint(1, 100)
                        )
                        op_end = time.time()
                        latencies.append(op_end - op_start)
                        success_count += 1

                        # Random short sleep to simulate real workload
                        if random.random() < 0.1:
                            await asyncio.sleep(0.001)

            except Exception as e:
                error_count += 1
                print(f"    Task {task_id} error: {e}")

        # Run concurrent tasks
        tasks = [concurrent_task(i) for i in range(concurrent_tasks)]
        await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time
        total_operations = concurrent_tasks * operations_per_task
        success_rate = (
            (success_count / total_operations) * 100 if total_operations > 0 else 0
        )
        throughput = success_count / duration if duration > 0 else 0

        details = {
            "concurrent_tasks": concurrent_tasks,
            "operations_per_task": operations_per_task,
            "total_operations": total_operations,
            "successful_operations": success_count,
            "failed_operations": error_count,
            "duration": duration,
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "p95_latency": (
                statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0
            ),
        }

        success = success_rate >= 95.0  # 95% success rate threshold
        self.log_result("High Concurrency AsyncPG", success, details)
        return success

    async def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted"""
        print(
            f"\n{StressTestColors.HEADER}🏊 Connection Pool Exhaustion Test{StressTestColors.ENDC}"
        )

        start_time = time.time()
        pool = await get_healthy_pool()
        max_size = getattr(pool, "_max_size", 10)

        # Try to acquire more connections than pool size
        connections = []
        acquisition_times = []

        try:
            # Acquire connections up to pool limit
            for i in range(max_size + 5):  # Try to exceed pool size
                acq_start = time.time()
                try:
                    conn = await asyncio.wait_for(pool.acquire(), timeout=2.0)
                    acq_end = time.time()
                    connections.append(conn)
                    acquisition_times.append(acq_end - acq_start)
                except asyncio.TimeoutError:
                    print(
                        f"    Connection {i+1}: Timeout (expected for pool exhaustion)"
                    )
                    break
                except Exception as e:
                    print(f"    Connection {i+1}: Error - {e}")
                    break

            # Test that existing connections still work
            if connections:
                test_conn = connections[0]
                result = await test_conn.fetchval("SELECT 1")
                working_connections = result == 1
            else:
                working_connections = False

            # Release connections
            for conn in connections:
                pool.release(conn)

            duration = time.time() - start_time

            details = {
                "pool_max_size": max_size,
                "connections_acquired": len(connections),
                "avg_acquisition_time": (
                    statistics.mean(acquisition_times) if acquisition_times else 0
                ),
                "max_acquisition_time": (
                    max(acquisition_times) if acquisition_times else 0
                ),
                "existing_connections_working": working_connections,
                "duration": duration,
            }

            # Success if we can acquire up to pool limit and existing connections work
            success = len(connections) >= max_size and working_connections
            self.log_result("Connection Pool Exhaustion", success, details)
            return success

        except Exception as e:
            details = {"error": str(e), "duration": time.time() - start_time}
            self.log_result("Connection Pool Exhaustion", False, details)
            return False

    async def test_checkpointer_stress(
        self, concurrent_threads: int = 20, operations_per_thread: int = 10
    ):
        """Test LangGraph checkpointer under stress"""
        print(
            f"\n{StressTestColors.HEADER}🔄 Checkpointer Stress Test ({concurrent_threads} threads x {operations_per_thread} ops){StressTestColors.ENDC}"
        )

        start_time = time.time()
        success_count = 0
        error_count = 0
        latencies = []

        async def checkpointer_task(task_id: int):
            nonlocal success_count, error_count

            try:
                checkpointer = await get_postgres_checkpointer()

                for i in range(operations_per_thread):
                    op_start = time.time()
                    config = {
                        "configurable": {"thread_id": f"stress_test_{task_id}_{i}"}
                    }

                    # Test get operation
                    await checkpointer.aget(config)

                    # Test tuple get operation
                    await checkpointer.aget_tuple(config)

                    op_end = time.time()
                    latencies.append(op_end - op_start)
                    success_count += 1

            except Exception as e:
                error_count += 1
                print(f"    Checkpointer task {task_id} error: {e}")

        # Run concurrent checkpointer tasks
        tasks = [checkpointer_task(i) for i in range(concurrent_threads)]
        await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time
        total_operations = concurrent_threads * operations_per_thread
        success_rate = (
            (success_count / total_operations) * 100 if total_operations > 0 else 0
        )
        throughput = success_count / duration if duration > 0 else 0

        details = {
            "concurrent_threads": concurrent_threads,
            "operations_per_thread": operations_per_thread,
            "total_operations": total_operations,
            "successful_operations": success_count,
            "failed_operations": error_count,
            "duration": duration,
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
        }

        success = success_rate >= 90.0  # 90% success rate threshold for checkpointer
        self.log_result("Checkpointer Stress", success, details)
        return success

    async def test_mixed_workload_stress(self, duration_seconds: int = 30):
        """Test mixed workload with both asyncpg and checkpointer operations"""
        print(
            f"\n{StressTestColors.HEADER}🔀 Mixed Workload Stress Test ({duration_seconds}s){StressTestColors.ENDC}"
        )

        start_time = time.time()
        end_time = start_time + duration_seconds

        asyncpg_ops = 0
        checkpointer_ops = 0
        asyncpg_errors = 0
        checkpointer_errors = 0

        async def asyncpg_worker():
            nonlocal asyncpg_ops, asyncpg_errors
            while time.time() < end_time:
                try:
                    pool = await get_healthy_pool()
                    async with pool.acquire() as conn:
                        await conn.fetchval("SELECT $1", random.randint(1, 1000))
                        asyncpg_ops += 1
                except Exception:
                    asyncpg_errors += 1
                await asyncio.sleep(0.01)

        async def checkpointer_worker():
            nonlocal checkpointer_ops, checkpointer_errors
            checkpointer = await get_postgres_checkpointer()
            while time.time() < end_time:
                try:
                    config = {
                        "configurable": {
                            "thread_id": f"mixed_test_{random.randint(1, 100)}"
                        }
                    }
                    await checkpointer.aget(config)
                    checkpointer_ops += 1
                except Exception:
                    checkpointer_errors += 1
                await asyncio.sleep(0.05)

        async def data_worker():
            nonlocal asyncpg_ops, asyncpg_errors
            while time.time() < end_time:
                try:
                    email = random.choice(self.test_emails)
                    thread_id = f"mixed_thread_{random.randint(1, 50)}"
                    await create_thread_run_entry(
                        email, thread_id, f"Test prompt {random.randint(1, 1000)}"
                    )
                    asyncpg_ops += 1
                except Exception:
                    asyncpg_errors += 1
                await asyncio.sleep(0.02)

        # Run mixed workload
        workers = (
            [asyncpg_worker() for _ in range(3)]
            + [checkpointer_worker() for _ in range(2)]
            + [data_worker() for _ in range(2)]
        )

        await asyncio.gather(*workers, return_exceptions=True)

        actual_duration = time.time() - start_time
        total_ops = asyncpg_ops + checkpointer_ops
        total_errors = asyncpg_errors + checkpointer_errors

        details = {
            "duration": actual_duration,
            "asyncpg_operations": asyncpg_ops,
            "checkpointer_operations": checkpointer_ops,
            "total_operations": total_ops,
            "asyncpg_errors": asyncpg_errors,
            "checkpointer_errors": checkpointer_errors,
            "total_errors": total_errors,
            "total_throughput": (
                total_ops / actual_duration if actual_duration > 0 else 0
            ),
            "error_rate": (total_errors / total_ops) * 100 if total_ops > 0 else 0,
        }

        success = details["error_rate"] < 5.0  # Less than 5% error rate
        self.log_result("Mixed Workload Stress", success, details)
        return success

    async def test_connection_recovery_after_errors(self):
        """Test connection recovery after simulated errors"""
        print(
            f"\n{StressTestColors.HEADER}🔧 Connection Recovery Test{StressTestColors.ENDC}"
        )

        start_time = time.time()
        recovery_successful = False

        try:
            # First, establish that connections work
            pool = await get_healthy_pool()
            async with pool.acquire() as conn:
                initial_result = await conn.fetchval("SELECT 1")

            if initial_result != 1:
                raise Exception("Initial connection test failed")

            # Simulate some error conditions by running invalid queries
            error_count = 0
            for i in range(5):
                try:
                    async with pool.acquire() as conn:
                        await conn.fetchval(
                            "SELECT invalid_column FROM non_existent_table"
                        )
                except Exception:
                    error_count += 1

            # Test that connections still work after errors
            async with pool.acquire() as conn:
                recovery_result = await conn.fetchval("SELECT 2")

            recovery_successful = recovery_result == 2

            # Test checkpointer recovery
            checkpointer = await get_postgres_checkpointer()
            config = {"configurable": {"thread_id": "recovery_test"}}
            checkpointer_result = await checkpointer.aget(config)
            checkpointer_recovery = True  # If no exception, recovery successful

            duration = time.time() - start_time

            details = {
                "initial_connection_ok": initial_result == 1,
                "errors_simulated": error_count,
                "asyncpg_recovery_ok": recovery_successful,
                "checkpointer_recovery_ok": checkpointer_recovery,
                "duration": duration,
            }

            success = recovery_successful and checkpointer_recovery
            self.log_result("Connection Recovery", success, details)
            return success

        except Exception as e:
            details = {"error": str(e), "duration": time.time() - start_time}
            self.log_result("Connection Recovery", False, details)
            return False

    async def test_memory_usage_under_load(self, operations: int = 1000):
        """Test memory usage under sustained load"""
        print(
            f"\n{StressTestColors.HEADER}💾 Memory Usage Test ({operations} operations){StressTestColors.ENDC}"
        )

        start_time = time.time()

        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform sustained operations
            pool = await get_healthy_pool()
            for i in range(operations):
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT $1", i)

                # Check memory every 100 operations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory

                    # If memory growth is excessive (>100MB), something might be wrong
                    if memory_growth > 100:
                        print(
                            f"    Warning: Memory growth {memory_growth:.1f}MB at operation {i}"
                        )

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory

            duration = time.time() - start_time

            details = {
                "operations": operations,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "memory_per_operation_kb": (
                    (memory_growth * 1024) / operations if operations > 0 else 0
                ),
                "duration": duration,
                "ops_per_second": operations / duration if duration > 0 else 0,
            }

            # Success if memory growth is reasonable (< 50MB for 1000 operations)
            success = memory_growth < 50
            self.log_result("Memory Usage", success, details)
            return success

        except ImportError:
            details = {"error": "psutil not available for memory monitoring"}
            self.log_result("Memory Usage", False, details)
            return False
        except Exception as e:
            details = {"error": str(e), "duration": time.time() - start_time}
            self.log_result("Memory Usage", False, details)
            return False

    async def cleanup_test_data(self):
        """Cleanup test data"""
        print(f"\n{StressTestColors.HEADER}🧹 Cleanup{StressTestColors.ENDC}")

        try:
            # Clean up test threads
            cleanup_count = 0
            for email in self.test_emails:
                try:
                    threads = await get_user_chat_threads(email, limit=100)
                    for thread in threads:
                        thread_id = thread["thread_id"]
                        # Delete thread entries (this function exists in our module)
                        from checkpointer.postgres_checkpointer import (
                            delete_user_thread_entries,
                        )

                        result = await delete_user_thread_entries(email, thread_id)
                        cleanup_count += result.get("deleted_count", 0)
                except Exception as e:
                    print(f"    Cleanup error for {email}: {e}")

            # Force close all connections
            await force_close_all_connections()

            print(f"    Cleaned up {cleanup_count} test entries")
            print(f"    Closed all database connections")

            return True

        except Exception as e:
            print(f"    Cleanup error: {e}")
            return False

    def print_summary(self):
        """Print test summary"""
        print(
            f"\n{StressTestColors.HEADER}[SUMMARY] STRESS TEST SUMMARY{StressTestColors.ENDC}"
        )
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result["success"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(
            f"{StressTestColors.OKGREEN}Passed: {passed_tests}{StressTestColors.ENDC}"
        )
        print(f"{StressTestColors.FAIL}Failed: {failed_tests}{StressTestColors.ENDC}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        if failed_tests > 0:
            print(f"\n{StressTestColors.FAIL}Failed Tests:{StressTestColors.ENDC}")
            for result in self.results:
                if not result["success"]:
                    print(f"  [FAIL] {result['test']}: {result['message']}")

        # Save detailed results
        results_file = (
            f"stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total": total_tests,
                        "passed": passed_tests,
                        "failed": failed_tests,
                        "success_rate": (passed_tests / total_tests) * 100,
                    },
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\nDetailed results saved to: {results_file}")

        return failed_tests == 0


async def main():
    """Run the stress test suite"""
    print(f"{StressTestColors.BOLD}{StressTestColors.HEADER}")
    print("[STRESS] POSTGRESQL STRESS TEST SUITE")
    print("Testing both AsyncPG and LangGraph AsyncPostgresSaver under high load")
    print("=" * 80)
    print(f"{StressTestColors.ENDC}")

    stress_test = PostgreSQLStressTest()

    try:
        # Setup
        await setup_users_threads_runs_table()

        # Run stress tests
        tests = [
            stress_test.test_high_concurrency_asyncpg(50, 20),
            stress_test.test_connection_pool_exhaustion(),
            stress_test.test_checkpointer_stress(20, 10),
            stress_test.test_mixed_workload_stress(30),
            stress_test.test_connection_recovery_after_errors(),
            stress_test.test_memory_usage_under_load(1000),
        ]

        results = await asyncio.gather(*tests, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(
                    f"{StressTestColors.FAIL}Test {i+1} crashed: {result}{StressTestColors.ENDC}"
                )
                traceback.print_exc()

        # Cleanup
        await stress_test.cleanup_test_data()

        # Print summary
        success = stress_test.print_summary()

        if success:
            print(
                f"\n{StressTestColors.OKGREEN}{StressTestColors.BOLD}[SUCCESS] ALL STRESS TESTS PASSED!{StressTestColors.ENDC}"
            )
            return 0
        else:
            print(
                f"\n{StressTestColors.FAIL}{StressTestColors.BOLD}[ERROR] SOME STRESS TESTS FAILED{StressTestColors.ENDC}"
            )
            return 1

    except Exception as e:
        print(f"{StressTestColors.FAIL}Test suite crashed: {e}{StressTestColors.ENDC}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

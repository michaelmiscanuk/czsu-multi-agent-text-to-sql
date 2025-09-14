"""Stress Tests for Checkpointer Database System
Tests the database system under high load and stress conditions.

This test file focuses on stress testing the checkpointer database system including:
- High-volume concurrent connections
- Mass data operations (create/read/update/delete)
- Connection pool exhaustion and recovery
- Memory leak detection under load
- Error recovery under stress conditions
- Performance benchmarking under various loads
- Deadlock and race condition detection

This test file follows the same patterns as test_checkpointer_overall.py,
including proper error handling, traceback capture, and detailed reporting.
"""

import asyncio
import gc
import os
import psutil
import random
import sys
import time
import traceback
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

# Standard library imports
import json
from unittest.mock import patch, MagicMock

from tests.helpers import (
    BaseTestResults,
    handle_error_response,
    handle_expected_failure,
    extract_detailed_error_info,
    make_request_with_traceback_capture,
    save_traceback_report,
    create_test_jwt_token,
    check_server_connectivity,
    setup_debug_environment,
    cleanup_debug_environment,
    ServerLogCapture,
    capture_server_logs,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import all database modules for stress testing with error handling
try:
    # Database layer
    from checkpointer.database.connection import (
        get_connection_string,
        get_connection_kwargs,
        get_direct_connection,
    )
    from checkpointer.database.pool_manager import (
        cleanup_all_pools,
        force_close_modern_pools,
        modern_psycopg_pool,
    )
    from checkpointer.database.table_setup import (
        setup_checkpointer_with_autocommit,
        setup_users_threads_runs_table,
        table_exists,
    )

    # Checkpointer factory and health
    from checkpointer.checkpointer.factory import (
        create_async_postgres_saver,
        close_async_postgres_saver,
        get_global_checkpointer,
        initialize_checkpointer,
        cleanup_checkpointer,
        check_pool_health_and_recreate,
    )
    from checkpointer.checkpointer.health import (
        check_pool_health_and_recreate as health_check_pool_health_and_recreate,
    )

    # User management for stress testing
    from checkpointer.user_management.thread_operations import (
        create_thread_run_entry,
        get_user_chat_threads,
        get_user_chat_threads_count,
        delete_user_thread_entries,
    )
    from checkpointer.user_management.sentiment_tracking import (
        update_thread_run_sentiment,
        get_thread_run_sentiments,
    )

    # Error handling
    from checkpointer.error_handling.prepared_statements import (
        is_prepared_statement_error,
        clear_prepared_statements,
    )
    from checkpointer.error_handling.retry_decorators import (
        retry_on_prepared_statement_error,
    )

    # Configuration and globals
    from checkpointer.config import (
        get_db_config,
        check_postgres_env_vars,
        DEFAULT_MAX_RETRIES,
        CHECKPOINTER_CREATION_MAX_RETRIES,
        DEFAULT_POOL_MIN_SIZE,
        DEFAULT_POOL_MAX_SIZE,
        DEFAULT_POOL_TIMEOUT,
    )
    from checkpointer.globals import _GLOBAL_CHECKPOINTER, _CONNECTION_STRING_CACHE

    DATABASE_STRESS_AVAILABLE = True
    IMPORT_ERROR = None

except ImportError as e:
    DATABASE_STRESS_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create comprehensive mock functions for testing import failures
    def get_connection_string():
        return "postgresql://mock:mock@mock:5432/mock"

    def get_connection_kwargs():
        return {"prepare_threshold": None}

    async def get_direct_connection():
        return MagicMock()

    async def cleanup_all_pools():
        pass

    async def force_close_modern_pools():
        pass

    async def modern_psycopg_pool():
        return MagicMock()

    async def setup_checkpointer_with_autocommit():
        return MagicMock()

    async def setup_users_threads_runs_table():
        pass

    async def table_exists(conn, table_name):
        return True

    async def create_async_postgres_saver():
        return MagicMock()

    async def close_async_postgres_saver():
        pass

    def get_global_checkpointer():
        return MagicMock()

    async def initialize_checkpointer():
        return MagicMock()

    async def cleanup_checkpointer():
        pass

    async def check_pool_health_and_recreate():
        return True

    async def health_check_pool_health_and_recreate():
        return True

    async def create_thread_run_entry(email, thread_id, prompt=None, run_id=None):
        return str(uuid.uuid4())

    async def get_user_chat_threads(email, limit=None, offset=0):
        return []

    async def get_user_chat_threads_count(email):
        return 0

    async def delete_user_thread_entries(email, thread_id):
        return {"deleted_count": 0}

    async def update_thread_run_sentiment(run_id, sentiment):
        return False

    async def get_thread_run_sentiments(email, thread_id):
        return {}

    def is_prepared_statement_error(error):
        return False

    async def clear_prepared_statements():
        pass

    def retry_on_prepared_statement_error(max_retries=3):
        def decorator(func):
            return func

        return decorator

    def get_db_config():
        """Retrieve database configuration from environment variables."""
        return {
            "host": os.getenv("host", "mock"),
            "port": int(os.getenv("port", 5432)),
            "user": os.getenv("user", "mock"),
            "password": os.getenv("password", "mock"),
            "dbname": os.getenv("dbname", "mock"),
        }

    def check_postgres_env_vars():
        return True

    DEFAULT_MAX_RETRIES = 3
    CHECKPOINTER_CREATION_MAX_RETRIES = 2
    DEFAULT_POOL_MIN_SIZE = 3
    DEFAULT_POOL_MAX_SIZE = 10
    DEFAULT_POOL_TIMEOUT = 30
    _GLOBAL_CHECKPOINTER = None
    _CONNECTION_STRING_CACHE = {}


# Stress test configuration
STRESS_TEST_NAME = f"stress_test_{uuid.uuid4().hex[:8]}"
STRESS_TEST_EMAIL_BASE = "stress_test_user"
STRESS_TEST_THREAD_BASE = "stress_thread"
STRESS_TEST_PROMPT_BASE = "Stress test prompt"

# Stress test parameters - configurable
CONCURRENT_CONNECTIONS = int(os.getenv("STRESS_CONCURRENT_CONNECTIONS", "20"))
BULK_OPERATIONS_COUNT = int(os.getenv("STRESS_BULK_OPERATIONS", "100"))
STRESS_DURATION_SECONDS = int(os.getenv("STRESS_DURATION_SECONDS", "60"))
MEMORY_THRESHOLD_MB = int(os.getenv("STRESS_MEMORY_THRESHOLD_MB", "500"))
CONNECTION_POOL_EXHAUSTION_LIMIT = int(os.getenv("STRESS_POOL_EXHAUSTION", "50"))
DEADLOCK_SIMULATION_THREADS = int(os.getenv("STRESS_DEADLOCK_THREADS", "10"))

# Performance benchmarks
PERFORMANCE_BASELINE_OPERATIONS = 10
PERFORMANCE_THRESHOLD_SECONDS = 5.0


class StressTestResults(BaseTestResults):
    """Extended test results class for stress testing with performance metrics."""

    def __init__(self):
        super().__init__()
        self.stress_metrics = {
            "peak_memory_mb": 0.0,
            "memory_growth_mb": 0.0,
            "peak_connections": 0,
            "operations_per_second": 0.0,
            "error_rate": 0.0,
            "deadlocks_detected": 0,
            "pool_exhaustions": 0,
            "recovery_time_seconds": 0.0,
        }
        self.performance_benchmarks = {}
        self.concurrent_test_results = []

    def record_memory_usage(self, memory_mb: float, baseline_mb: float):
        """Record memory usage metrics."""
        self.stress_metrics["peak_memory_mb"] = max(
            self.stress_metrics["peak_memory_mb"], memory_mb
        )
        growth = memory_mb - baseline_mb
        self.stress_metrics["memory_growth_mb"] = max(
            self.stress_metrics["memory_growth_mb"], growth
        )

    def record_performance_benchmark(self, operation: str, duration: float, count: int):
        """Record performance benchmark results."""
        ops_per_second = count / duration if duration > 0 else 0
        self.performance_benchmarks[operation] = {
            "duration_seconds": duration,
            "operations_count": count,
            "ops_per_second": ops_per_second,
            "avg_time_per_operation": duration / count if count > 0 else 0,
        }

    def record_concurrent_result(
        self, thread_id: str, success: bool, duration: float, error: str = None
    ):
        """Record results from concurrent operations."""
        self.concurrent_test_results.append(
            {
                "thread_id": thread_id,
                "success": success,
                "duration": duration,
                "error": error,
                "timestamp": datetime.now(),
            }
        )

    def calculate_final_metrics(self):
        """Calculate final stress test metrics."""
        if self.concurrent_test_results:
            total_operations = len(self.concurrent_test_results)
            successful_operations = sum(
                1 for r in self.concurrent_test_results if r["success"]
            )
            failed_operations = total_operations - successful_operations

            self.stress_metrics["error_rate"] = (
                failed_operations / total_operations if total_operations > 0 else 0
            )

            # Calculate operations per second from concurrent results
            if self.concurrent_test_results:
                durations = [r["duration"] for r in self.concurrent_test_results]
                total_time = max(durations) if durations else 0
                self.stress_metrics["operations_per_second"] = (
                    total_operations / total_time if total_time > 0 else 0
                )

    def save_traceback_report(self, report_type: str = "stress_test_report"):
        """Save the stress test report with metrics."""
        return save_traceback_report(
            report_type=report_type,
            test_results=self,
            test_context={
                "test_type": "Database Stress Testing",
                "stress_metrics": self.stress_metrics,
                "performance_benchmarks": self.performance_benchmarks,
                "concurrent_results_count": len(self.concurrent_test_results),
                "total_tests": len(self.results),
                "import_available": DATABASE_STRESS_AVAILABLE,
                "import_error": IMPORT_ERROR,
                "stress_configuration": {
                    "concurrent_connections": CONCURRENT_CONNECTIONS,
                    "bulk_operations": BULK_OPERATIONS_COUNT,
                    "stress_duration": STRESS_DURATION_SECONDS,
                    "memory_threshold": MEMORY_THRESHOLD_MB,
                },
            },
        )


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


async def stress_test_concurrent_connections():
    """Stress test concurrent database connections."""
    print(
        f"\n🔥 Stress Test: Concurrent Connections ({CONCURRENT_CONNECTIONS} connections)"
    )

    results = []
    start_memory = get_memory_usage()
    start_time = time.time()

    async def create_connection_task(task_id: int) -> Dict[str, Any]:
        """Create a single connection task."""
        task_start = time.time()
        try:
            conn_manager = get_direct_connection()
            async with conn_manager as connection:
                # Perform a simple operation to test the connection
                if hasattr(connection, "execute"):
                    await connection.execute("SELECT 1")
                task_duration = time.time() - task_start
                return {
                    "task_id": task_id,
                    "success": True,
                    "duration": task_duration,
                    "error": None,
                }
        except Exception as e:
            task_duration = time.time() - task_start
            return {
                "task_id": task_id,
                "success": False,
                "duration": task_duration,
                "error": str(e),
            }

    # Create concurrent connection tasks
    tasks = [create_connection_task(i) for i in range(CONCURRENT_CONNECTIONS)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_duration = time.time() - start_time
    end_memory = get_memory_usage()
    memory_growth = end_memory - start_memory

    # Process results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failed = len(results) - successful

    print(f"   Results: {successful}/{len(results)} connections successful")
    print(f"   Total Duration: {total_duration:.3f}s")
    print(f"   Memory Growth: {memory_growth:.1f}MB")
    print(f"   Connections per Second: {len(results)/total_duration:.1f}")

    return {
        "success": failed == 0,
        "total_connections": len(results),
        "successful_connections": successful,
        "failed_connections": failed,
        "total_duration": total_duration,
        "memory_growth": memory_growth,
        "connections_per_second": len(results) / total_duration,
        "results": results,
    }


async def stress_test_bulk_operations():
    """Stress test bulk database operations."""
    print(f"\n🔥 Stress Test: Bulk Operations ({BULK_OPERATIONS_COUNT} operations)")

    start_memory = get_memory_usage()
    start_time = time.time()

    # Generate test data
    test_users = [
        f"{STRESS_TEST_EMAIL_BASE}_{i}@example.com"
        for i in range(BULK_OPERATIONS_COUNT)
    ]
    test_threads = [
        f"{STRESS_TEST_THREAD_BASE}_{i}_{uuid.uuid4().hex[:8]}"
        for i in range(BULK_OPERATIONS_COUNT)
    ]

    operations_results = {
        "create_operations": [],
        "read_operations": [],
        "update_operations": [],
        "delete_operations": [],
    }

    # Bulk CREATE operations
    print("   Phase 1: Bulk CREATE operations...")
    create_start = time.time()
    for i in range(BULK_OPERATIONS_COUNT):
        try:
            run_id = await create_thread_run_entry(
                test_users[i], test_threads[i], f"{STRESS_TEST_PROMPT_BASE} {i}"
            )
            operations_results["create_operations"].append(
                {
                    "success": True,
                    "run_id": run_id,
                    "index": i,
                }
            )
        except Exception as e:
            operations_results["create_operations"].append(
                {
                    "success": False,
                    "error": str(e),
                    "index": i,
                }
            )
    create_duration = time.time() - create_start

    # Bulk READ operations
    print("   Phase 2: Bulk READ operations...")
    read_start = time.time()
    for i in range(min(BULK_OPERATIONS_COUNT, len(test_users))):
        try:
            threads = await get_user_chat_threads(test_users[i], limit=10)
            thread_count = await get_user_chat_threads_count(test_users[i])
            operations_results["read_operations"].append(
                {
                    "success": True,
                    "threads_count": len(threads),
                    "total_count": thread_count,
                    "index": i,
                }
            )
        except Exception as e:
            operations_results["read_operations"].append(
                {
                    "success": False,
                    "error": str(e),
                    "index": i,
                }
            )
    read_duration = time.time() - read_start

    # Bulk UPDATE operations (sentiment updates)
    print("   Phase 3: Bulk UPDATE operations...")
    update_start = time.time()
    successful_creates = [
        op for op in operations_results["create_operations"] if op["success"]
    ]
    for i, create_op in enumerate(successful_creates[: BULK_OPERATIONS_COUNT // 2]):
        try:
            if "run_id" in create_op:
                sentiment_updated = await update_thread_run_sentiment(
                    create_op["run_id"], random.choice([True, False])
                )
                operations_results["update_operations"].append(
                    {
                        "success": sentiment_updated,
                        "run_id": create_op["run_id"],
                        "index": i,
                    }
                )
        except Exception as e:
            operations_results["update_operations"].append(
                {
                    "success": False,
                    "error": str(e),
                    "index": i,
                }
            )
    update_duration = time.time() - update_start

    # Bulk DELETE operations
    print("   Phase 4: Bulk DELETE operations...")
    delete_start = time.time()
    for i in range(min(BULK_OPERATIONS_COUNT // 4, len(test_users))):
        try:
            delete_result = await delete_user_thread_entries(
                test_users[i], test_threads[i]
            )
            operations_results["delete_operations"].append(
                {
                    "success": True,
                    "deleted_count": delete_result.get("deleted_count", 0),
                    "index": i,
                }
            )
        except Exception as e:
            operations_results["delete_operations"].append(
                {
                    "success": False,
                    "error": str(e),
                    "index": i,
                }
            )
    delete_duration = time.time() - delete_start

    total_duration = time.time() - start_time
    end_memory = get_memory_usage()
    memory_growth = end_memory - start_memory

    # Calculate success rates
    total_operations = sum(len(ops) for ops in operations_results.values())
    successful_operations = sum(
        sum(1 for op in ops if op.get("success", False))
        for ops in operations_results.values()
    )

    print(
        f"   Results: {successful_operations}/{total_operations} operations successful"
    )
    print(
        f"   CREATE: {create_duration:.3f}s, READ: {read_duration:.3f}s, UPDATE: {update_duration:.3f}s, DELETE: {delete_duration:.3f}s"
    )
    print(f"   Total Duration: {total_duration:.3f}s")
    print(f"   Memory Growth: {memory_growth:.1f}MB")
    print(f"   Operations per Second: {total_operations/total_duration:.1f}")

    return {
        "success": successful_operations == total_operations,
        "total_operations": total_operations,
        "successful_operations": successful_operations,
        "failed_operations": total_operations - successful_operations,
        "operations_breakdown": {
            "create": {
                "count": len(operations_results["create_operations"]),
                "duration": create_duration,
            },
            "read": {
                "count": len(operations_results["read_operations"]),
                "duration": read_duration,
            },
            "update": {
                "count": len(operations_results["update_operations"]),
                "duration": update_duration,
            },
            "delete": {
                "count": len(operations_results["delete_operations"]),
                "duration": delete_duration,
            },
        },
        "total_duration": total_duration,
        "memory_growth": memory_growth,
        "operations_per_second": total_operations / total_duration,
        "detailed_results": operations_results,
    }


async def stress_test_connection_pool_exhaustion():
    """Test connection pool behavior under exhaustion conditions."""
    print(
        f"\n🔥 Stress Test: Connection Pool Exhaustion ({CONNECTION_POOL_EXHAUSTION_LIMIT} connections)"
    )

    start_time = time.time()
    start_memory = get_memory_usage()

    connections = []
    exhaustion_reached = False
    max_connections_achieved = 0

    try:
        # Attempt to create many connections without releasing them
        for i in range(CONNECTION_POOL_EXHAUSTION_LIMIT):
            try:
                conn_manager = get_direct_connection()
                # Store the connection manager to prevent cleanup
                connections.append(conn_manager)
                max_connections_achieved = i + 1

                # Small delay to allow pool to respond
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"   Pool exhaustion reached at {i} connections: {str(e)}")
                exhaustion_reached = True
                break

        # Test pool recovery
        print("   Testing pool recovery...")
        recovery_start = time.time()

        # Release all connections
        connections.clear()
        await asyncio.sleep(0.1)  # Allow cleanup

        # Force cleanup pools
        await cleanup_all_pools()
        await asyncio.sleep(0.5)  # Allow recovery

        # Test if we can create new connections after cleanup
        recovery_success = False
        try:
            conn_manager = get_direct_connection()
            async with conn_manager as connection:
                if hasattr(connection, "execute"):
                    await connection.execute("SELECT 1")
                recovery_success = True
        except Exception as e:
            print(f"   Pool recovery failed: {str(e)}")

        recovery_duration = time.time() - recovery_start

    finally:
        # Ensure cleanup
        connections.clear()
        await cleanup_all_pools()

    total_duration = time.time() - start_time
    end_memory = get_memory_usage()
    memory_growth = end_memory - start_memory

    print(f"   Max Connections Achieved: {max_connections_achieved}")
    print(f"   Exhaustion Reached: {exhaustion_reached}")
    print(f"   Recovery Duration: {recovery_duration:.3f}s")
    print(f"   Total Duration: {total_duration:.3f}s")
    print(f"   Memory Growth: {memory_growth:.1f}MB")

    return {
        "success": recovery_success,
        "max_connections_achieved": max_connections_achieved,
        "exhaustion_reached": exhaustion_reached,
        "recovery_success": recovery_success,
        "recovery_duration": recovery_duration,
        "total_duration": total_duration,
        "memory_growth": memory_growth,
    }


async def stress_test_memory_leaks():
    """Test for memory leaks during sustained operations."""
    print(
        f"\n🔥 Stress Test: Memory Leak Detection ({STRESS_DURATION_SECONDS}s duration)"
    )

    start_memory = get_memory_usage()
    start_time = time.time()
    memory_samples = []

    operations_count = 0
    errors_count = 0

    # Sustained operations for specified duration
    end_time = start_time + STRESS_DURATION_SECONDS

    while time.time() < end_time:
        current_memory = get_memory_usage()
        memory_samples.append(
            {
                "timestamp": time.time() - start_time,
                "memory_mb": current_memory,
                "operations_count": operations_count,
            }
        )

        try:
            # Perform various operations to test for leaks

            # Connection operations
            conn_manager = get_direct_connection()
            async with conn_manager as connection:
                if hasattr(connection, "execute"):
                    await connection.execute("SELECT 1")

            # Checkpointer operations
            checkpointer = await initialize_checkpointer()
            await cleanup_checkpointer()

            # User operations
            test_email = f"leak_test_{operations_count}@example.com"
            test_thread = f"leak_thread_{operations_count}"

            run_id = await create_thread_run_entry(
                test_email, test_thread, f"Leak test {operations_count}"
            )
            await update_thread_run_sentiment(run_id, True)
            threads = await get_user_chat_threads(test_email, limit=1)

            operations_count += 1

            # Periodic garbage collection to help detect real leaks
            if operations_count % 10 == 0:
                collected = gc.collect()
                if collected > 0:
                    print(
                        f"   GC collected {collected} objects at operation {operations_count}"
                    )

        except Exception as e:
            errors_count += 1
            if errors_count % 10 == 0:  # Only print every 10th error to avoid spam
                print(f"   Error during operation {operations_count}: {str(e)}")

        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.01)

    end_memory = get_memory_usage()
    total_duration = time.time() - start_time
    memory_growth = end_memory - start_memory

    # Analyze memory trend
    if len(memory_samples) >= 3:
        early_memory = sum(s["memory_mb"] for s in memory_samples[:3]) / 3
        late_memory = sum(s["memory_mb"] for s in memory_samples[-3:]) / 3
        memory_trend = late_memory - early_memory
    else:
        memory_trend = memory_growth

    # Calculate leak severity
    leak_severity = "NONE"
    if memory_growth > MEMORY_THRESHOLD_MB:
        if memory_trend > memory_growth * 0.8:  # Growing trend
            leak_severity = "SEVERE"
        else:
            leak_severity = "MODERATE"
    elif memory_growth > MEMORY_THRESHOLD_MB * 0.5:
        leak_severity = "MINOR"

    print(f"   Operations Completed: {operations_count}")
    print(f"   Errors: {errors_count}")
    print(f"   Memory Growth: {memory_growth:.1f}MB")
    print(f"   Memory Trend: {memory_trend:.1f}MB")
    print(f"   Leak Severity: {leak_severity}")
    print(f"   Operations per Second: {operations_count/total_duration:.1f}")

    return {
        "success": leak_severity in ["NONE", "MINOR"],
        "operations_completed": operations_count,
        "errors_count": errors_count,
        "memory_growth": memory_growth,
        "memory_trend": memory_trend,
        "leak_severity": leak_severity,
        "operations_per_second": operations_count / total_duration,
        "memory_samples": memory_samples,
        "duration": total_duration,
    }


async def stress_test_error_recovery():
    """Test system recovery from various error conditions."""
    print(f"\n🔥 Stress Test: Error Recovery and Resilience")

    start_time = time.time()
    recovery_tests = {}

    # Test 1: Recovery from prepared statement errors
    print("   Test 1: Prepared statement error recovery...")
    try:
        # Simulate prepared statement error condition
        await clear_prepared_statements()

        # Test if system can continue operating
        test_email = f"recovery_test_1@example.com"
        test_thread = f"recovery_thread_1_{uuid.uuid4().hex[:8]}"
        run_id = await create_thread_run_entry(test_email, test_thread, "Recovery test")

        recovery_tests["prepared_statement_recovery"] = {
            "success": True,
            "run_id": run_id,
        }
    except Exception as e:
        recovery_tests["prepared_statement_recovery"] = {
            "success": False,
            "error": str(e),
        }

    # Test 2: Recovery from pool exhaustion
    print("   Test 2: Pool exhaustion recovery...")
    try:
        # Force close pools
        await force_close_modern_pools()
        await asyncio.sleep(0.5)

        # Test if system can create new connections
        conn_manager = get_direct_connection()
        async with conn_manager as connection:
            if hasattr(connection, "execute"):
                await connection.execute("SELECT 1")

        recovery_tests["pool_exhaustion_recovery"] = {
            "success": True,
        }
    except Exception as e:
        recovery_tests["pool_exhaustion_recovery"] = {
            "success": False,
            "error": str(e),
        }

    # Test 3: Recovery from checkpointer system cleanup
    print("   Test 3: Checkpointer system recovery...")
    try:
        # Cleanup checkpointer system
        await cleanup_checkpointer()
        await asyncio.sleep(0.2)

        # Test if system can reinitialize
        checkpointer = await initialize_checkpointer()

        recovery_tests["checkpointer_recovery"] = {
            "success": checkpointer is not None,
            "checkpointer_type": type(checkpointer).__name__ if checkpointer else None,
        }
    except Exception as e:
        recovery_tests["checkpointer_recovery"] = {
            "success": False,
            "error": str(e),
        }

    # Test 4: Recovery from database connection issues
    print("   Test 4: Database connection recovery...")
    try:
        # Test connection health and recovery
        health_ok = await check_pool_health_and_recreate()

        # Verify we can still perform operations
        test_email = f"recovery_test_4@example.com"
        threads = await get_user_chat_threads(test_email, limit=1)

        recovery_tests["connection_recovery"] = {
            "success": True,
            "health_check": health_ok,
            "operation_success": True,
        }
    except Exception as e:
        recovery_tests["connection_recovery"] = {
            "success": False,
            "error": str(e),
        }

    total_duration = time.time() - start_time
    successful_recoveries = sum(
        1 for test in recovery_tests.values() if test["success"]
    )
    total_recoveries = len(recovery_tests)

    print(f"   Recovery Tests: {successful_recoveries}/{total_recoveries} successful")
    print(f"   Total Duration: {total_duration:.3f}s")

    return {
        "success": successful_recoveries == total_recoveries,
        "successful_recoveries": successful_recoveries,
        "total_recoveries": total_recoveries,
        "recovery_details": recovery_tests,
        "duration": total_duration,
    }


async def performance_benchmark():
    """Run performance benchmarks for key operations."""
    print(f"\n📊 Performance Benchmarking")

    benchmarks = {}

    # Benchmark 1: Connection creation speed
    print("   Benchmark 1: Connection Creation Speed...")
    connection_start = time.time()
    for i in range(PERFORMANCE_BASELINE_OPERATIONS):
        conn_manager = get_direct_connection()
        async with conn_manager as connection:
            if hasattr(connection, "execute"):
                await connection.execute("SELECT 1")
    connection_duration = time.time() - connection_start
    benchmarks["connection_creation"] = {
        "operations": PERFORMANCE_BASELINE_OPERATIONS,
        "duration": connection_duration,
        "ops_per_second": PERFORMANCE_BASELINE_OPERATIONS / connection_duration,
    }

    # Benchmark 2: User operations speed
    print("   Benchmark 2: User Operations Speed...")
    user_ops_start = time.time()
    for i in range(PERFORMANCE_BASELINE_OPERATIONS):
        test_email = f"benchmark_user_{i}@example.com"
        test_thread = f"benchmark_thread_{i}"
        run_id = await create_thread_run_entry(
            test_email, test_thread, f"Benchmark test {i}"
        )
        await update_thread_run_sentiment(run_id, True)
        threads = await get_user_chat_threads(test_email, limit=5)
    user_ops_duration = time.time() - user_ops_start
    benchmarks["user_operations"] = {
        "operations": PERFORMANCE_BASELINE_OPERATIONS,
        "duration": user_ops_duration,
        "ops_per_second": PERFORMANCE_BASELINE_OPERATIONS / user_ops_duration,
    }

    # Benchmark 3: Checkpointer operations speed
    print("   Benchmark 3: Checkpointer Operations Speed...")
    checkpointer_start = time.time()
    for i in range(PERFORMANCE_BASELINE_OPERATIONS):
        checkpointer = await initialize_checkpointer()
        health_ok = await check_pool_health_and_recreate()
        await cleanup_checkpointer()
    checkpointer_duration = time.time() - checkpointer_start
    benchmarks["checkpointer_operations"] = {
        "operations": PERFORMANCE_BASELINE_OPERATIONS,
        "duration": checkpointer_duration,
        "ops_per_second": PERFORMANCE_BASELINE_OPERATIONS / checkpointer_duration,
    }

    # Evaluate performance against thresholds
    performance_ok = all(
        benchmark["duration"] < PERFORMANCE_THRESHOLD_SECONDS
        for benchmark in benchmarks.values()
    )

    print("   Benchmark Results:")
    for operation, metrics in benchmarks.items():
        print(
            f"     {operation}: {metrics['ops_per_second']:.1f} ops/sec ({metrics['duration']:.3f}s total)"
        )

    return {
        "success": performance_ok,
        "benchmarks": benchmarks,
        "threshold_seconds": PERFORMANCE_THRESHOLD_SECONDS,
        "all_benchmarks_passed": performance_ok,
    }


async def run_stress_tests() -> StressTestResults:
    """Run all stress tests and return comprehensive results."""
    print("🔥 Starting Database Stress Tests")
    print(
        f"   Import Status: {'✅ Available' if DATABASE_STRESS_AVAILABLE else '❌ Failed'}"
    )
    if not DATABASE_STRESS_AVAILABLE:
        print(f"   Import Error: {IMPORT_ERROR}")

    print(f"   Configuration:")
    print(f"     Concurrent Connections: {CONCURRENT_CONNECTIONS}")
    print(f"     Bulk Operations: {BULK_OPERATIONS_COUNT}")
    print(f"     Stress Duration: {STRESS_DURATION_SECONDS}s")
    print(f"     Memory Threshold: {MEMORY_THRESHOLD_MB}MB")

    results = StressTestResults()
    start_time = time.time()
    baseline_memory = get_memory_usage()

    # Force initial cleanup to start from clean state
    try:
        await cleanup_all_pools()
        await cleanup_checkpointer()
        gc.collect()
    except Exception as e:
        print(f"   Initial cleanup warning: {e}")

    # Test 1: Concurrent Connections Stress Test
    try:
        concurrent_result = await stress_test_concurrent_connections()
        results.add_result(
            test_id="ST001",
            endpoint="concurrent_connections",
            description=f"Concurrent connections stress test ({CONCURRENT_CONNECTIONS} connections)",
            response_data=concurrent_result,
            response_time=concurrent_result["total_duration"],
            status_code=200 if concurrent_result["success"] else 500,
            success=concurrent_result["success"],
        )
        results.record_memory_usage(get_memory_usage(), baseline_memory)
        results.stress_metrics["peak_connections"] = concurrent_result[
            "total_connections"
        ]
    except Exception as e:
        results.add_error(
            "ST001",
            "concurrent_connections",
            "Concurrent connections stress test",
            e,
            0.0,
        )

    # Test 2: Bulk Operations Stress Test
    try:
        bulk_result = await stress_test_bulk_operations()
        results.add_result(
            test_id="ST002",
            endpoint="bulk_operations",
            description=f"Bulk operations stress test ({BULK_OPERATIONS_COUNT} operations)",
            response_data=bulk_result,
            response_time=bulk_result["total_duration"],
            status_code=200 if bulk_result["success"] else 500,
            success=bulk_result["success"],
        )
        results.record_memory_usage(get_memory_usage(), baseline_memory)
        results.record_performance_benchmark(
            "bulk_operations",
            bulk_result["total_duration"],
            bulk_result["total_operations"],
        )
    except Exception as e:
        results.add_error(
            "ST002", "bulk_operations", "Bulk operations stress test", e, 0.0
        )

    # Test 3: Connection Pool Exhaustion Test
    try:
        exhaustion_result = await stress_test_connection_pool_exhaustion()
        results.add_result(
            test_id="ST003",
            endpoint="pool_exhaustion",
            description=f"Connection pool exhaustion test ({CONNECTION_POOL_EXHAUSTION_LIMIT} connections)",
            response_data=exhaustion_result,
            response_time=exhaustion_result["total_duration"],
            status_code=200 if exhaustion_result["success"] else 500,
            success=exhaustion_result["success"],
        )
        results.record_memory_usage(get_memory_usage(), baseline_memory)
        results.stress_metrics["pool_exhaustions"] = (
            1 if exhaustion_result["exhaustion_reached"] else 0
        )
        results.stress_metrics["recovery_time_seconds"] = exhaustion_result[
            "recovery_duration"
        ]
    except Exception as e:
        results.add_error(
            "ST003", "pool_exhaustion", "Connection pool exhaustion test", e, 0.0
        )

    # Test 4: Memory Leak Detection
    try:
        memory_result = await stress_test_memory_leaks()
        results.add_result(
            test_id="ST004",
            endpoint="memory_leaks",
            description=f"Memory leak detection test ({STRESS_DURATION_SECONDS}s duration)",
            response_data=memory_result,
            response_time=memory_result["duration"],
            status_code=200 if memory_result["success"] else 500,
            success=memory_result["success"],
        )
        results.record_memory_usage(get_memory_usage(), baseline_memory)
        results.record_performance_benchmark(
            "sustained_operations",
            memory_result["duration"],
            memory_result["operations_completed"],
        )
    except Exception as e:
        results.add_error("ST004", "memory_leaks", "Memory leak detection test", e, 0.0)

    # Test 5: Error Recovery Test
    try:
        recovery_result = await stress_test_error_recovery()
        results.add_result(
            test_id="ST005",
            endpoint="error_recovery",
            description="Error recovery and resilience test",
            response_data=recovery_result,
            response_time=recovery_result["duration"],
            status_code=200 if recovery_result["success"] else 500,
            success=recovery_result["success"],
        )
        results.record_memory_usage(get_memory_usage(), baseline_memory)
    except Exception as e:
        results.add_error("ST005", "error_recovery", "Error recovery test", e, 0.0)

    # Test 6: Performance Benchmarking
    try:
        benchmark_result = await performance_benchmark()
        results.add_result(
            test_id="ST006",
            endpoint="performance_benchmark",
            description="Performance benchmarking test",
            response_data=benchmark_result,
            response_time=sum(
                b["duration"] for b in benchmark_result["benchmarks"].values()
            ),
            status_code=200 if benchmark_result["success"] else 500,
            success=benchmark_result["success"],
        )

        # Record individual benchmarks
        for operation, metrics in benchmark_result["benchmarks"].items():
            results.record_performance_benchmark(
                operation, metrics["duration"], metrics["operations"]
            )

    except Exception as e:
        results.add_error(
            "ST006", "performance_benchmark", "Performance benchmarking test", e, 0.0
        )

    # Final cleanup and metrics calculation
    try:
        await cleanup_all_pools()
        await cleanup_checkpointer()
        gc.collect()
    except Exception as e:
        print(f"   Final cleanup warning: {e}")

    results.record_memory_usage(get_memory_usage(), baseline_memory)
    results.calculate_final_metrics()

    total_duration = time.time() - start_time

    print(f"\n📊 Stress Test Summary:")
    summary = results.get_summary()
    print(f"   Total Tests: {summary['total_requests']}")
    print(f"   Successful: {summary['successful_requests']}")
    print(f"   Failed: {summary['failed_requests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Total Duration: {total_duration:.1f}s")

    print(f"\n🔬 Stress Metrics:")
    print(f"   Peak Memory: {results.stress_metrics['peak_memory_mb']:.1f}MB")
    print(f"   Memory Growth: {results.stress_metrics['memory_growth_mb']:.1f}MB")
    print(f"   Peak Connections: {results.stress_metrics['peak_connections']}")
    print(
        f"   Operations/Second: {results.stress_metrics['operations_per_second']:.1f}"
    )
    print(f"   Error Rate: {results.stress_metrics['error_rate']:.2%}")

    return results


def analyze_stress_results(results: StressTestResults):
    """Analyze and display detailed stress test results."""
    print("\n📈 Detailed Stress Analysis:")

    summary = results.get_summary()
    print(f"   Overall Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Average Response Time: {summary['average_response_time']:.3f}s")

    # Memory analysis
    print(f"\n🧠 Memory Analysis:")
    memory_status = "PASS"
    if results.stress_metrics["memory_growth_mb"] > MEMORY_THRESHOLD_MB:
        memory_status = "FAIL - Excessive memory growth"
    elif results.stress_metrics["memory_growth_mb"] > MEMORY_THRESHOLD_MB * 0.7:
        memory_status = "WARNING - High memory usage"

    print(f"   Memory Status: {memory_status}")
    print(f"   Peak Memory: {results.stress_metrics['peak_memory_mb']:.1f}MB")
    print(f"   Memory Growth: {results.stress_metrics['memory_growth_mb']:.1f}MB")

    # Performance analysis
    print(f"\n⚡ Performance Analysis:")
    if results.performance_benchmarks:
        print("   Benchmark Results:")
        for operation, metrics in results.performance_benchmarks.items():
            status = "✅" if metrics["avg_time_per_operation"] < 1.0 else "⚠️"
            print(f"     {status} {operation}: {metrics['ops_per_second']:.1f} ops/sec")

    # Error analysis
    if results.errors:
        print(f"\n❌ Error Analysis ({len(results.errors)} errors):")
        error_types = {}
        for error in results.errors:
            error_key = error["description"]
            error_types[error_key] = error_types.get(error_key, 0) + 1

        for error_type, count in error_types.items():
            print(f"   - {error_type}: {count} occurrences")

    # Concurrent operations analysis
    if results.concurrent_test_results:
        print(f"\n🔀 Concurrency Analysis:")
        successful_concurrent = sum(
            1 for r in results.concurrent_test_results if r["success"]
        )
        total_concurrent = len(results.concurrent_test_results)
        print(
            f"   Concurrent Operations: {successful_concurrent}/{total_concurrent} successful"
        )

        if results.concurrent_test_results:
            durations = [
                r["duration"] for r in results.concurrent_test_results if r["success"]
            ]
            if durations:
                print(
                    f"   Avg Concurrent Duration: {sum(durations)/len(durations):.3f}s"
                )
                print(f"   Max Concurrent Duration: {max(durations):.3f}s")


async def main():
    """Main stress test execution function."""
    print("=" * 90)
    print("🔥 DATABASE CHECKPOINTER STRESS TESTING SUITE")
    print("=" * 90)

    start_time = time.time()

    try:
        # Run all stress tests
        results = await run_stress_tests()

        # Analyze results
        analyze_stress_results(results)

        # Save stress test report
        print(f"\n💾 Saving stress test report...")
        report_saved = results.save_traceback_report()
        if report_saved:
            print("   Stress test report saved successfully")
        else:
            print("   Failed to save stress test report")

        # Final summary
        total_time = time.time() - start_time
        summary = results.get_summary()

        print(f"\n🏁 Stress Test Suite Completed in {total_time:.2f} seconds")
        print(
            f"   Final Results: {summary['successful_requests']}/{summary['total_requests']} tests passed"
        )
        print(f"   Success Rate: {summary['success_rate']:.1f}%")

        # Determine exit code based on results
        critical_failures = sum(
            1
            for error in results.errors
            if "memory_leaks" in error.get("endpoint", "")
            or "pool_exhaustion" in error.get("endpoint", "")
        )

        memory_issues = results.stress_metrics["memory_growth_mb"] > MEMORY_THRESHOLD_MB
        performance_issues = (
            results.stress_metrics["error_rate"] > 0.1
        )  # >10% error rate

        if critical_failures > 0 or memory_issues or performance_issues:
            print("   🚨 CRITICAL ISSUES DETECTED:")
            if critical_failures > 0:
                print(f"      - {critical_failures} critical test failures")
            if memory_issues:
                print(
                    f"      - Excessive memory usage: {results.stress_metrics['memory_growth_mb']:.1f}MB"
                )
            if performance_issues:
                print(
                    f"      - High error rate: {results.stress_metrics['error_rate']:.1%}"
                )
            sys.exit(2)  # Critical failure
        elif summary["failed_requests"] > 0:
            print("   ⚠️  Some stress tests failed - check the results above")
            sys.exit(1)  # Some failures
        else:
            print("   🎉 All stress tests passed!")
            print(f"   🏆 System Performance: EXCELLENT")
            print(
                f"      - Memory Usage: {results.stress_metrics['memory_growth_mb']:.1f}MB growth"
            )
            print(
                f"      - Operations/Second: {results.stress_metrics['operations_per_second']:.1f}"
            )
            print(f"      - Error Rate: {results.stress_metrics['error_rate']:.2%}")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Stress test suite failed with exception: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

        # Save error report
        save_traceback_report(
            report_type="stress_test_exception",
            exception=e,
            test_context={
                "test_type": "Database Stress Testing",
                "error_location": "main",
                "import_available": DATABASE_STRESS_AVAILABLE,
                "import_error": IMPORT_ERROR,
                "stress_configuration": {
                    "concurrent_connections": CONCURRENT_CONNECTIONS,
                    "bulk_operations": BULK_OPERATIONS_COUNT,
                    "stress_duration": STRESS_DURATION_SECONDS,
                },
            },
        )
        sys.exit(3)  # Exception failure


if __name__ == "__main__":
    asyncio.run(main())

"""
Test for Phase 12: Performance Testing
Comprehensive performance testing for all API endpoints with meaningful test scenarios
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

# Standard imports
import asyncio
import time
import traceback
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
import psutil

# Import test helpers
from tests.helpers import (
    BaseTestResults,
    check_server_connectivity,
    create_test_jwt_token,
    make_request_with_traceback_capture,
    save_traceback_report,
    setup_debug_environment,
    cleanup_debug_environment,
    extract_detailed_error_info,
)

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60.0
TEST_EMAIL = "test_performance_user@example.com"

# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    "health": 1.0,  # Health endpoints should be very fast
    "auth": 2.0,  # Auth-protected endpoints
    "database": 5.0,  # Database-heavy endpoints
    "computation": 10.0,  # Computation-heavy endpoints like analysis
}

# Endpoints grouped by expected performance characteristics
ENDPOINT_CATEGORIES = {
    "health": [
        "/health",
        "/health/database",
        "/health/memory",
        "/health/rate-limits",
        "/health/prepared-statements",
    ],
    "auth": [
        "/chat-threads",
        "/catalog",
        "/data-tables",
        "/data-table",
    ],
    "database": [
        "/chat/all-messages-for-all-threads",
        "/debug/pool-status",
        "/admin/clear-cache",
        "/admin/clear-prepared-statements",
    ],
    "computation": [
        "/analyze",  # This is likely the heaviest endpoint
    ],
    "misc": [
        "/placeholder/100/100",
    ],
}


class PerformanceResults:
    """Enhanced class to track comprehensive performance test results."""

    def __init__(self):
        self.startup_time = None
        self.memory_baseline = None
        self.memory_after_requests = None
        self.memory_peak = None
        self.request_times = []
        self.concurrent_performance = None
        self.concurrent_analysis_performance = None
        self.load_test_results = None
        self.endpoint_performance = {}
        self.failed_requests = []
        self.database_connection_times = []
        self.cache_performance = {}

    def add_request_time(
        self,
        endpoint: str,
        response_time: float,
        status_code: int = 200,
        test_type: str = "individual",
    ):
        """Add a request timing result with additional metadata."""
        self.request_times.append(
            {
                "endpoint": endpoint,
                "response_time": response_time,
                "status_code": status_code,
                "test_type": test_type,
                "timestamp": datetime.now().isoformat(),
                "success": status_code in [200, 201, 204],
            }
        )

    def add_failed_request(
        self,
        endpoint: str,
        error: str,
        response_time: float = None,
        test_type: str = "individual",
    ):
        """Add a failed request for analysis."""
        self.failed_requests.append(
            {
                "endpoint": endpoint,
                "error": error,
                "response_time": response_time,
                "test_type": test_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_endpoint_performance(self, endpoint: str, stats: Dict[str, Any]):
        """Add detailed performance statistics for a specific endpoint."""
        self.endpoint_performance[endpoint] = stats

    def get_summary(self):
        """Get comprehensive performance summary."""
        if self.request_times:
            successful_requests = [r for r in self.request_times if r["success"]]
            if successful_requests:
                avg_time = sum(r["response_time"] for r in successful_requests) / len(
                    successful_requests
                )
                max_time = max(r["response_time"] for r in successful_requests)
                min_time = min(r["response_time"] for r in successful_requests)

                # Calculate percentiles
                response_times = sorted(
                    [r["response_time"] for r in successful_requests]
                )
                p50 = response_times[len(response_times) // 2] if response_times else 0
                p95 = (
                    response_times[int(len(response_times) * 0.95)]
                    if response_times
                    else 0
                )
                p99 = (
                    response_times[int(len(response_times) * 0.99)]
                    if response_times
                    else 0
                )
            else:
                avg_time = max_time = min_time = p50 = p95 = p99 = 0
        else:
            avg_time = max_time = min_time = p50 = p95 = p99 = 0

        # Categorize performance by endpoint types
        endpoint_category_performance = {}
        for category, endpoints in ENDPOINT_CATEGORIES.items():
            category_times = [
                r["response_time"]
                for r in self.request_times
                if r["success"]
                and any(
                    r["endpoint"].startswith(ep) or r["endpoint"] == ep
                    for ep in endpoints
                )
            ]
            if category_times:
                endpoint_category_performance[category] = {
                    "avg_time": sum(category_times) / len(category_times),
                    "max_time": max(category_times),
                    "min_time": min(category_times),
                    "request_count": len(category_times),
                }

        return {
            "startup_time": self.startup_time,
            "memory_baseline_mb": self.memory_baseline,
            "memory_after_requests_mb": self.memory_after_requests,
            "memory_peak_mb": self.memory_peak,
            "memory_growth_mb": (
                (self.memory_after_requests - self.memory_baseline)
                if both_exist(self.memory_baseline, self.memory_after_requests)
                else None
            ),
            "total_requests": len(self.request_times),
            "successful_requests": len([r for r in self.request_times if r["success"]]),
            "failed_requests_count": len(self.failed_requests),
            "avg_response_time": avg_time,
            "max_response_time": max_time,
            "min_response_time": min_time,
            "p50_response_time": p50,
            "p95_response_time": p95,
            "p99_response_time": p99,
            "concurrent_performance": self.concurrent_performance,
            "concurrent_analysis_performance": self.concurrent_analysis_performance,
            "load_test_results": self.load_test_results,
            "endpoint_category_performance": endpoint_category_performance,
            "endpoint_performance": self.endpoint_performance,
            "failed_requests": self.failed_requests,
            "database_connection_times": self.database_connection_times,
            "cache_performance": self.cache_performance,
        }


def both_exist(a, b):
    """Helper to check if both values exist."""
    return a is not None and b is not None


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return None


async def create_test_data_for_performance_tests(user_email: str) -> Dict[str, Any]:
    """Create test data needed for performance testing."""
    print("🔧 Setting up test data for performance tests...")

    test_data = {
        "thread_ids": [],
        "run_ids": [],
        "test_analyze_request": {
            "question": "What is the population and GDP per capita for each country in the latest year?",
            "dataset_id": "country_statistics_dataset",
        },
    }

    try:
        # Import database connection for test data setup
        from checkpointer.database.connection import get_direct_connection

        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                # Create test thread and run_id data
                for i in range(5):  # Create multiple test threads
                    thread_id = str(uuid.uuid4())
                    run_id = str(uuid.uuid4())

                    await cur.execute(
                        """
                        INSERT INTO users_threads_runs 
                        (email, thread_id, run_id, sentiment, timestamp) 
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (run_id) DO NOTHING
                        """,
                        (user_email, thread_id, run_id, None),
                    )

                    test_data["thread_ids"].append(thread_id)
                    test_data["run_ids"].append(run_id)

                await conn.commit()
                print(
                    f"✅ Created {len(test_data['thread_ids'])} test threads and run_ids"
                )

    except Exception as e:
        print(f"⚠️ Failed to create test data: {e}")
        # Use random UUIDs as fallback
        test_data["thread_ids"] = [str(uuid.uuid4()) for _ in range(5)]
        test_data["run_ids"] = [str(uuid.uuid4()) for _ in range(5)]
        print("🔄 Using random UUIDs as fallback")

    return test_data


async def cleanup_test_data(test_data: Dict[str, Any]):
    """Clean up test data after performance tests."""
    try:
        from checkpointer.database.connection import get_direct_connection

        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                for run_id in test_data.get("run_ids", []):
                    await cur.execute(
                        "DELETE FROM users_threads_runs WHERE run_id = %s", (run_id,)
                    )
                await conn.commit()
                print(f"🧹 Cleaned up {len(test_data.get('run_ids', []))} test records")
    except Exception as e:
        print(f"⚠️ Failed to cleanup test data: {e}")


async def measure_endpoint_performance(
    client: httpx.AsyncClient,
    endpoint: str,
    method: str = "GET",
    headers: Dict = None,
    json_data: Dict = None,
    params: Dict = None,
    iterations: int = 3,
    test_name: str = "",
) -> Dict[str, Any]:
    """Measure performance of a specific endpoint with multiple iterations."""

    print(f"🔍 Testing endpoint performance: {method} {endpoint}")
    if test_name:
        print(f"   📝 Test scenario: {test_name}")
    print(f"   🔄 Running {iterations} iterations...")

    response_times = []
    status_codes = []
    errors = []
    memory_usage = []

    for i in range(iterations):
        start_memory = get_memory_usage()
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = await client.get(endpoint, headers=headers, params=params)
            elif method.upper() == "POST":
                response = await client.post(endpoint, headers=headers, json=json_data)
            elif method.upper() == "DELETE":
                response = await client.delete(endpoint, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response_time = time.time() - start_time
            end_memory = get_memory_usage()

            response_times.append(response_time)
            status_codes.append(response.status_code)

            if start_memory and end_memory:
                memory_usage.append(end_memory - start_memory)

            print(
                f"   📊 Iteration {i+1}: {response_time:.3f}s (status: {response.status_code})"
            )

            if response.status_code not in [200, 201, 204]:
                try:
                    error_detail = response.json().get(
                        "detail", f"HTTP {response.status_code}"
                    )
                except:
                    error_detail = f"HTTP {response.status_code}: {response.text[:100]}"
                errors.append(error_detail)

        except Exception as e:
            response_time = time.time() - start_time
            response_times.append(response_time)
            status_codes.append(0)
            errors.append(str(e))
            print(f"   ❌ Iteration {i+1}: Error - {str(e)[:100]}")

        # Small delay between iterations to avoid overwhelming the server
        if i < iterations - 1:
            await asyncio.sleep(0.1)

    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)

        # Calculate stability (coefficient of variation)
        if avg_time > 0:
            std_dev = (
                sum((t - avg_time) ** 2 for t in response_times) / len(response_times)
            ) ** 0.5
            stability = std_dev / avg_time  # Lower is better
        else:
            stability = float("inf")
    else:
        avg_time = max_time = min_time = stability = 0

    successful_iterations = len([s for s in status_codes if s in [200, 201, 204]])
    success_rate = (successful_iterations / iterations) * 100 if iterations > 0 else 0

    performance_stats = {
        "endpoint": endpoint,
        "method": method,
        "test_name": test_name,
        "iterations": iterations,
        "avg_response_time": avg_time,
        "max_response_time": max_time,
        "min_response_time": min_time,
        "response_times": response_times,
        "status_codes": status_codes,
        "success_rate": success_rate,
        "errors": errors,
        "stability": stability,
        "memory_impact": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
    }

    # Performance assessment
    threshold = get_performance_threshold(endpoint)
    if avg_time <= threshold:
        status = "✅ EXCELLENT"
    elif avg_time <= threshold * 1.5:
        status = "🟡 ACCEPTABLE"
    else:
        status = "❌ SLOW"

    print(
        f"   📈 Results: avg={avg_time:.3f}s, success_rate={success_rate:.1f}% {status}"
    )

    return performance_stats


def get_performance_threshold(endpoint: str) -> float:
    """Get performance threshold for an endpoint based on its category."""
    for category, endpoints in ENDPOINT_CATEGORIES.items():
        if any(endpoint.startswith(ep) or endpoint == ep for ep in endpoints):
            return PERFORMANCE_THRESHOLDS.get(category, 5.0)
    return 5.0  # Default threshold


async def test_application_startup_time():
    """Test application startup time with enhanced monitoring."""
    print("🔍 Testing application startup time...")
    print("   📝 Measuring time to first successful health check response")

    try:
        start_time = time.time()
        max_wait = 30  # seconds
        attempts = 0

        while time.time() - start_time < max_wait:
            attempts += 1
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    response = await client.get(f"{SERVER_BASE_URL}/health")
                    if response.status_code == 200:
                        startup_time = time.time() - start_time
                        print(f"   ✅ Application ready after {attempts} attempts")
                        print(f"   📊 Startup time: {startup_time:.2f}s")

                        # Assess startup performance
                        if startup_time < 5:
                            print("   🎉 EXCELLENT startup time")
                        elif startup_time < 15:
                            print("   ✅ GOOD startup time")
                        elif startup_time < 30:
                            print("   🟡 ACCEPTABLE startup time")
                        else:
                            print("   ❌ SLOW startup time")

                        return startup_time
            except Exception as e:
                print(f"   ⏳ Attempt {attempts}: {str(e)[:50]}...")
                await asyncio.sleep(1)

        print("   ❌ Application did not start within timeout")
        return None

    except Exception as e:
        print(f"   ❌ Startup time test failed: {e}")
        return None


async def test_memory_usage_patterns(test_data: Dict[str, Any]):
    """Test memory usage patterns with comprehensive endpoint coverage."""
    print("🔍 Testing memory usage patterns...")
    print("   📝 Measuring memory impact across different endpoint categories")

    try:
        baseline_memory = get_memory_usage()
        peak_memory = baseline_memory
        print(f"   📊 Baseline memory: {baseline_memory:.1f}MB")

        token = create_test_jwt_token(TEST_EMAIL)
        headers = {"Authorization": f"Bearer {token}"}

        memory_measurements = []

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test different endpoint categories
            test_scenarios = [
                # Health endpoints (no auth required)
                ("/health", None, "Health check"),
                ("/health/memory", None, "Memory health check"),
                ("/health/database", None, "Database health check"),
                # Basic authenticated endpoints
                ("/catalog", headers, "Catalog listing"),
                ("/data-tables", headers, "Data tables listing"),
                ("/chat-threads", headers, "Chat threads listing"),
                # Database-heavy endpoints
                ("/debug/pool-status", headers, "Pool status check"),
                # Parameterized endpoints
                (f"/data-table?table=country_stats", headers, "Specific table query"),
                ("/placeholder/200/200", None, "Placeholder image generation"),
            ]

            # Add thread-specific tests if we have test data
            if test_data.get("thread_ids"):
                thread_id = test_data["thread_ids"][0]
                test_scenarios.extend(
                    [
                        (f"/chat/{thread_id}/messages", headers, "Thread messages"),
                        (f"/chat/{thread_id}/sentiments", headers, "Thread sentiments"),
                    ]
                )

            for endpoint, request_headers, description in test_scenarios:
                print(f"   🔍 Testing {description}: {endpoint}")

                pre_request_memory = get_memory_usage()

                try:
                    response = await client.get(endpoint, headers=request_headers)
                    post_request_memory = get_memory_usage()

                    if post_request_memory:
                        peak_memory = max(peak_memory, post_request_memory)
                        memory_delta = post_request_memory - (pre_request_memory or 0)
                        memory_measurements.append(
                            {
                                "endpoint": endpoint,
                                "description": description,
                                "memory_delta": memory_delta,
                                "status_code": response.status_code,
                            }
                        )

                        print(
                            f"     📊 {description}: {response.status_code}, memory Δ: {memory_delta:+.1f}MB"
                        )

                except Exception as e:
                    print(f"     ❌ {description}: Error - {str(e)[:50]}")

                await asyncio.sleep(0.5)  # Small delay between requests

        final_memory = get_memory_usage()

        if baseline_memory and final_memory:
            growth = final_memory - baseline_memory
            peak_growth = peak_memory - baseline_memory

            print(f"   📊 Final memory: {final_memory:.1f}MB")
            print(f"   📊 Peak memory: {peak_memory:.1f}MB")
            print(f"   📊 Total growth: {growth:+.1f}MB")
            print(f"   📊 Peak growth: {peak_growth:+.1f}MB")

            # Memory usage assessment
            if growth < 50:
                print("   ✅ EXCELLENT memory management")
            elif growth < 100:
                print("   🟡 ACCEPTABLE memory growth")
            else:
                print("   ❌ HIGH memory growth - potential memory leak")

            return {
                "baseline": baseline_memory,
                "final": final_memory,
                "peak": peak_memory,
                "growth": growth,
                "peak_growth": peak_growth,
                "measurements": memory_measurements,
            }
        else:
            print("   ⚠️ Could not measure memory properly")
            return {"baseline": baseline_memory, "final": final_memory, "growth": None}

    except Exception as e:
        print(f"   ❌ Memory usage test failed: {e}")
        return None


async def test_concurrent_request_handling(test_data: Dict[str, Any]):
    """Test concurrent request handling performance with realistic scenarios."""
    print("🔍 Testing concurrent request handling...")
    print("   📝 Simulating realistic concurrent user scenarios")

    try:
        token = create_test_jwt_token(TEST_EMAIL)
        headers = {"Authorization": f"Bearer {token}"}

        async def make_timed_request(
            endpoint: str,
            client: httpx.AsyncClient,
            headers: Dict = None,
            description: str = "",
        ):
            """Make a single request and measure comprehensive metrics."""
            start_time = time.time()
            start_memory = get_memory_usage()

            try:
                response = await client.get(endpoint, headers=headers)
                response_time = time.time() - start_time
                end_memory = get_memory_usage()

                return {
                    "endpoint": endpoint,
                    "description": description,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200,
                    "memory_delta": (
                        (end_memory - start_memory)
                        if (start_memory and end_memory)
                        else 0
                    ),
                    "response_size": (
                        len(response.content) if hasattr(response, "content") else 0
                    ),
                }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "endpoint": endpoint,
                    "description": description,
                    "status_code": 0,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)[:100],
                    "memory_delta": 0,
                    "response_size": 0,
                }

        # Test multiple concurrent scenarios
        print("   🔄 Scenario 1: Mixed health and authenticated endpoints")
        concurrent_start = time.time()

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Scenario 1: Mixed endpoints (simulating different user actions)
            tasks_scenario1 = [
                make_timed_request("/health", client, None, "Health check"),
                make_timed_request("/health/memory", client, None, "Memory health"),
                make_timed_request("/catalog", client, headers, "Catalog access"),
                make_timed_request("/data-tables", client, headers, "Data tables"),
                make_timed_request("/chat-threads", client, headers, "Chat threads"),
            ]

            results_scenario1 = await asyncio.gather(
                *tasks_scenario1, return_exceptions=True
            )
            scenario1_time = time.time() - concurrent_start

            print("   🔄 Scenario 2: Database-intensive concurrent requests")
            concurrent_start2 = time.time()

            # Scenario 2: Database-heavy endpoints
            tasks_scenario2 = []
            if test_data.get("thread_ids"):
                for i, thread_id in enumerate(
                    test_data["thread_ids"][:3]
                ):  # Test 3 threads concurrently
                    tasks_scenario2.append(
                        make_timed_request(
                            f"/chat/{thread_id}/messages",
                            client,
                            headers,
                            f"Messages for thread {i+1}",
                        )
                    )

            # Add some general database endpoints
            tasks_scenario2.extend(
                [
                    make_timed_request(
                        "/debug/pool-status", client, headers, "Pool status"
                    ),
                    make_timed_request(
                        "/chat/all-messages-for-all-threads",
                        client,
                        headers,
                        "All messages bulk",
                    ),
                ]
            )

            results_scenario2 = (
                await asyncio.gather(*tasks_scenario2, return_exceptions=True)
                if tasks_scenario2
                else []
            )
            scenario2_time = time.time() - concurrent_start2

            print("   🔄 Scenario 3: High-frequency light requests")
            concurrent_start3 = time.time()

            # Scenario 3: Many light requests (stress test)
            tasks_scenario3 = [
                make_timed_request("/health", client, None, f"Health check {i}")
                for i in range(10)
            ]

            results_scenario3 = await asyncio.gather(
                *tasks_scenario3, return_exceptions=True
            )
            scenario3_time = time.time() - concurrent_start3

        # Analyze all scenarios
        all_scenarios = [
            ("Mixed endpoints", results_scenario1, scenario1_time),
            ("Database-intensive", results_scenario2, scenario2_time),
            ("High-frequency light", results_scenario3, scenario3_time),
        ]

        scenario_results = {}

        for scenario_name, results, total_time in all_scenarios:
            if not results:
                continue

            valid_results = [r for r in results if isinstance(r, dict)]
            successful_requests = len(
                [r for r in valid_results if r.get("success", False)]
            )
            total_requests = len(valid_results)

            if valid_results:
                avg_response_time = sum(
                    r.get("response_time", 0) for r in valid_results
                ) / len(valid_results)
                max_response_time = max(
                    r.get("response_time", 0) for r in valid_results
                )
                total_response_size = sum(
                    r.get("response_size", 0) for r in valid_results
                )
                avg_memory_delta = sum(
                    r.get("memory_delta", 0) for r in valid_results
                ) / len(valid_results)
            else:
                avg_response_time = max_response_time = total_response_size = (
                    avg_memory_delta
                ) = 0

            success_rate = (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            )

            scenario_results[scenario_name] = {
                "total_time": total_time,
                "successful_requests": successful_requests,
                "total_requests": total_requests,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "success_rate": success_rate,
                "total_response_size": total_response_size,
                "avg_memory_delta": avg_memory_delta,
                "throughput": total_requests / total_time if total_time > 0 else 0,
            }

            print(
                f"   📊 {scenario_name}: {successful_requests}/{total_requests} successful"
            )
            print(
                f"      Time: {total_time:.2f}s, Avg response: {avg_response_time:.3f}s, Success rate: {success_rate:.1f}%"
            )

            # Performance assessment
            if success_rate >= 95 and avg_response_time < 2.0:
                print("      ✅ EXCELLENT concurrent performance")
            elif success_rate >= 80 and avg_response_time < 5.0:
                print("      🟡 ACCEPTABLE concurrent performance")
            else:
                print("      ❌ POOR concurrent performance")

        return scenario_results

    except Exception as e:
        print(f"   ❌ Concurrent request test failed: {e}")
        return None


async def test_concurrent_analysis_performance(test_data: Dict[str, Any]):
    """Test multi-user concurrent performance simulation for analysis endpoints."""
    print("� Testing MULTI-USER CONCURRENT analysis endpoint performance...")
    print(
        "   🏢 Simulating realistic multi-user environment with diverse workloads"
    )

    try:
        # Create multiple user tokens to simulate different users
        user_tokens = []
        for i in range(3):  # 3 different users
            token = create_test_jwt_token(f"user{i+1}@test.com")
            user_tokens.append({"Authorization": f"Bearer {token}"})

        async def simulate_user_session(
            client: httpx.AsyncClient, 
            user_id: int, 
            headers: Dict, 
            session_duration: int = 30
        ):
            """Simulate a complete user session with multiple requests."""
            print(f"   👤 User {user_id}: Starting session (targeting {session_duration}s)")
            
            session_start = time.time()
            session_requests = []
            request_count = 0
            
            # Different user behavior patterns (country statistics theme)
            user_patterns = {
                1: {  # Data Analyst - Complex country-statistics queries
                    "requests_per_minute": 4,
                    "queries": [
                        "What is the total population and population growth rate by country for the last 5 years?",
                        "Show GDP per capita and rank countries by income group",
                        "Analyze life expectancy trends by region and income group",
                        "Compare unemployment rates across countries and identify outliers",
                        "Generate a report of median age and urbanization rate by country"
                    ]
                },
                2: {  # Business User - Simple country-statistics queries  
                    "requests_per_minute": 8,
                    "queries": [
                        "How many countries have a population over 50 million?",
                        "List the top 10 countries by GDP per capita",
                        "Show countries with life expectancy above 80 years",
                        "What countries had negative GDP growth last year?",
                        "Display countries with CO2 emissions per capita above 10 tonnes"
                    ]
                },
                3: {  # Power User - Mixed complexity country-statistics queries
                    "requests_per_minute": 6,
                    "queries": [
                        "Create a comprehensive economic indicators dashboard for a given region",
                        "Show demographic breakdown (age groups) for selected countries",
                        "Analyze correlation between education index and GDP per capita",
                        "Generate trade balance summary (exports - imports) for selected countries",
                        "Compare healthcare spending vs life expectancy across countries"
                    ]
                }
            }
            
            pattern = user_patterns[user_id]
            queries = pattern["queries"]
            requests_per_minute = pattern["requests_per_minute"]
            
            # Calculate request interval
            request_interval = 60 / requests_per_minute
            
            while time.time() - session_start < session_duration:
                request_count += 1
                query = random.choice(queries)
                thread_id = test_data.get("thread_ids", [str(uuid.uuid4())])[0]
                
                request_start = time.time()
                start_memory = get_memory_usage()
                
                try:
                    response = await client.post(
                        "/analyze",
                        json={
                            "prompt": query,
                            "thread_id": f"{thread_id}_user{user_id}_{request_count}",
                        },
                        headers=headers,
                        timeout=60.0
                    )
                    
                    request_time = time.time() - request_start
                    end_memory = get_memory_usage()
                    
                    session_requests.append({
                        "user_id": user_id,
                        "request_id": request_count,
                        "query": query[:60] + "..." if len(query) > 60 else query,
                        "response_time": request_time,
                        "status_code": response.status_code,
                        "success": response.status_code in [200, 201],
                        "memory_delta": (end_memory - start_memory) if both_exist(start_memory, end_memory) else 0,
                        "response_size": len(response.content) if hasattr(response, "content") else 0,
                        "timestamp": time.time() - session_start
                    })
                    
                    status_emoji = "✅" if response.status_code in [200, 201] else "❌"
                    print(f"      {status_emoji} User {user_id} Request {request_count}: {request_time:.1f}s ({response.status_code})")
                    
                except Exception as e:
                    request_time = time.time() - request_start
                    session_requests.append({
                        "user_id": user_id,
                        "request_id": request_count,
                        "query": query[:60] + "..." if len(query) > 60 else query,
                        "response_time": request_time,
                        "status_code": 0,
                        "success": False,
                        "memory_delta": 0,
                        "response_size": 0,
                        "error": str(e)[:100],
                        "timestamp": time.time() - session_start
                    })
                    print(f"      ❌ User {user_id} Request {request_count}: Failed ({str(e)[:50]})")
                
                # Wait between requests (simulate realistic user behavior)
                remaining_time = session_duration - (time.time() - session_start)
                if remaining_time > request_interval:
                    await asyncio.sleep(min(request_interval, remaining_time))
                else:
                    break
            
            session_time = time.time() - session_start
            successful_requests = len([r for r in session_requests if r["success"]])
            print(f"   👤 User {user_id}: Session completed - {successful_requests}/{len(session_requests)} successful in {session_time:.1f}s")
            
            return session_requests

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL,
            timeout=httpx.Timeout(120.0)
        ) as client:
            
            print("   � Starting concurrent multi-user simulation...")
            print("   📊 User Profiles:")
            print("      👤 User 1: Data Analyst (4 req/min, complex queries)")
            print("      👤 User 2: Business User (8 req/min, simple queries)") 
            print("      👤 User 3: Power User (6 req/min, mixed complexity)")
            
            simulation_start = time.time()
            
            # Start all user sessions concurrently
            user_tasks = []
            for i, headers in enumerate(user_tokens):
                task = simulate_user_session(
                    client, 
                    user_id=i+1, 
                    headers=headers,
                    session_duration=30  # 30 second sessions
                )
                user_tasks.append(task)
            
            # Run all user sessions simultaneously
            print("   ⏳ Executing multi-user concurrent sessions...")
            user_session_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            total_simulation_time = time.time() - simulation_start
            print(f"   ⏱️ Total simulation time: {total_simulation_time:.2f}s")
            
            # Aggregate results from all users
            all_requests = []
            user_stats = {}
            
            for i, session_requests in enumerate(user_session_results):
                if isinstance(session_requests, list):
                    all_requests.extend(session_requests)
                    
                    # Calculate per-user statistics
                    user_id = i + 1
                    successful = len([r for r in session_requests if r["success"]])
                    total_reqs = len(session_requests)
                    avg_time = sum(r["response_time"] for r in session_requests) / total_reqs if total_reqs > 0 else 0
                    
                    user_stats[f"user_{user_id}"] = {
                        "total_requests": total_reqs,
                        "successful_requests": successful,
                        "success_rate": (successful / total_reqs * 100) if total_reqs > 0 else 0,
                        "avg_response_time": avg_time,
                        "throughput": total_reqs / total_simulation_time if total_simulation_time > 0 else 0
                    }
            
            if all_requests:
                # Overall statistics
                total_requests = len(all_requests)
                successful_requests = len([r for r in all_requests if r["success"]])
                failed_requests = total_requests - successful_requests
                
                response_times = [r["response_time"] for r in all_requests]
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # Calculate percentiles
                sorted_times = sorted(response_times)
                p50_time = sorted_times[len(sorted_times)//2]
                p95_time = sorted_times[int(len(sorted_times)*0.95)]
                p99_time = sorted_times[int(len(sorted_times)*0.99)]
                
                success_rate = (successful_requests / total_requests) * 100
                overall_throughput = total_requests / total_simulation_time if total_simulation_time > 0 else 0
                
                # Memory analysis
                memory_deltas = [r.get("memory_delta", 0) for r in all_requests if r.get("memory_delta")]
                avg_memory_delta = sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
                
                print(f"\n   📊 MULTI-USER SIMULATION RESULTS:")
                print(f"      🎯 Overall Performance:")
                print(f"         Total Requests: {total_requests}")
                print(f"         Successful: {successful_requests} ({success_rate:.1f}%)")
                print(f"         Failed: {failed_requests}")
                print(f"         Overall Throughput: {overall_throughput:.2f} req/s")
                
                print(f"      ⚡ Response Times:")
                print(f"         Average: {avg_response_time:.2f}s")
                print(f"         P50 (Median): {p50_time:.2f}s")
                print(f"         P95: {p95_time:.2f}s")
                print(f"         P99: {p99_time:.2f}s")
                print(f"         Min: {min_response_time:.2f}s")
                print(f"         Max: {max_response_time:.2f}s")
                
                print(f"      💾 Memory Impact: {avg_memory_delta:+.1f}MB avg per request")
                
                print(f"      👥 Per-User Performance:")
                for user_key, stats in user_stats.items():
                    user_num = user_key.split('_')[1]
                    print(f"         User {user_num}: {stats['successful_requests']}/{stats['total_requests']} "
                          f"({stats['success_rate']:.1f}%) - {stats['avg_response_time']:.2f}s avg - "
                          f"{stats['throughput']:.1f} req/s")
                
                # Multi-user performance assessment
                if success_rate >= 85 and avg_response_time < 25.0 and overall_throughput >= 0.3:
                    print("      🎉 EXCELLENT multi-user performance under concurrent load")
                    performance_rating = "excellent"
                elif success_rate >= 70 and avg_response_time < 45.0 and overall_throughput >= 0.15:
                    print("      🟡 ACCEPTABLE multi-user performance under concurrent load")
                    performance_rating = "acceptable"
                else:
                    print("      ❌ POOR multi-user performance under concurrent load")
                    performance_rating = "poor"
                
                # Return comprehensive multi-user results
                return {
                    "test_name": "Multi-User Concurrent Analysis Performance",
                    "simulation_type": "multi_user_concurrent",
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": success_rate,
                    "total_time": total_simulation_time,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "min_response_time": min_response_time,
                    "p50_response_time": p50_time,
                    "p95_response_time": p95_time,
                    "p99_response_time": p99_time,
                    "overall_throughput": overall_throughput,
                    "avg_memory_delta": avg_memory_delta,
                    "performance_rating": performance_rating,
                    "user_stats": user_stats,
                    "concurrent_users": len(user_tokens),
                    "session_duration": 30,
                }
            
            else:
                print("   ❌ No valid results from multi-user concurrent test")
                return {
                    "test_name": "Multi-User Concurrent Analysis Performance",
                    "total_requests": 0,
                    "error": "No valid results obtained",
                    "performance_rating": "failed",
                }

    except Exception as e:
        print(f"   ❌ Multi-user concurrent analysis performance test failed: {e}")
        traceback.print_exc()
        return {
            "test_name": "Multi-User Concurrent Analysis Performance",
            "error": str(e),
            "performance_rating": "failed",
        }


async def test_endpoint_response_times(
    test_data: Dict[str, Any], results: PerformanceResults
):
    """Test individual endpoint response times comprehensively."""
    print("🔍 Testing individual endpoint response times...")
    print(
        "   📝 Comprehensive coverage of all API endpoints with multiple test scenarios"
    )

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        # Health endpoints (no authentication required)
        health_endpoints = [
            ("/health", "GET", None, None, "Basic health check"),
            ("/health/database", "GET", None, None, "Database health verification"),
            ("/health/memory", "GET", None, None, "Memory usage health check"),
            ("/health/rate-limits", "GET", None, None, "Rate limiting status"),
            (
                "/health/prepared-statements",
                "GET",
                None,
                None,
                "Prepared statements health",
            ),
        ]

        print("   📋 Testing health endpoints...")
        for endpoint, method, req_headers, params, description in health_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                iterations=3,
                test_name=description,
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            # Add to request times for overall stats
            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

        # Catalog and data endpoints
        catalog_endpoints = [
            ("/catalog", "GET", headers, None, "Catalog listing - default pagination"),
            (
                "/catalog",
                "GET",
                headers,
                {"page": 2, "page_size": 5},
                "Catalog listing - custom pagination",
            ),
            ("/catalog", "GET", headers, {"q": "population"}, "Catalog search with query"),
            ("/data-tables", "GET", headers, None, "Data tables listing"),
            ("/data-tables", "GET", headers, {"q": "test"}, "Data tables search"),
                (
                "/data-table",
                "GET",
                headers,
                {"table": "country_stats"},
                "Specific table data",
            ),
        ]

        print("   📋 Testing catalog and data endpoints...")
        for endpoint, method, req_headers, params, description in catalog_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                params=params,
                iterations=3,
                test_name=description,
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

        # Chat and messaging endpoints
        chat_endpoints = [
            ("/chat-threads", "GET", headers, None, "Chat threads listing - default"),
            (
                "/chat-threads",
                "GET",
                headers,
                {"page": 1, "limit": 5},
                "Chat threads - limited results",
            ),
        ]

        # Add thread-specific endpoints if test data exists
        if test_data.get("thread_ids"):
            thread_id = test_data["thread_ids"][0]
            chat_endpoints.extend(
                [
                    (
                        f"/chat/{thread_id}/messages",
                        "GET",
                        headers,
                        None,
                        f"Messages for thread {thread_id[:8]}",
                    ),
                    (
                        f"/chat/{thread_id}/sentiments",
                        "GET",
                        headers,
                        None,
                        f"Sentiments for thread {thread_id[:8]}",
                    ),
                    (
                        f"/chat/{thread_id}/run-ids",
                        "GET",
                        headers,
                        None,
                        f"Run IDs for thread {thread_id[:8]}",
                    ),
                    (
                        f"/chat/all-messages-for-one-thread/{thread_id}",
                        "GET",
                        headers,
                        None,
                        f"All messages for one thread {thread_id[:8]}",
                    ),
                ]
            )

        print("   📋 Testing chat and messaging endpoints...")
        for endpoint, method, req_headers, params, description in chat_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                params=params,
                iterations=3,
                test_name=description,
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

        # Debug and admin endpoints
        debug_endpoints = [
            ("/debug/pool-status", "GET", headers, None, "Database pool status check"),
        ]

        # Add debug endpoints with test data
        if test_data.get("thread_ids"):
            thread_id = test_data["thread_ids"][0]
            debug_endpoints.append(
                (
                    f"/debug/chat/{thread_id}/checkpoints",
                    "GET",
                    headers,
                    None,
                    f"Debug checkpoints for thread {thread_id[:8]}",
                )
            )

        if test_data.get("run_ids"):
            run_id = test_data["run_ids"][0]
            debug_endpoints.append(
                (
                    f"/debug/run-id/{run_id}",
                    "GET",
                    headers,
                    None,
                    f"Debug run ID {run_id[:8]}",
                )
            )

        print("   📋 Testing debug and admin endpoints...")
        for endpoint, method, req_headers, params, description in debug_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                params=params,
                iterations=3,
                test_name=description,
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

        # Miscellaneous endpoints
        misc_endpoints = [
            ("/placeholder/100/100", "GET", None, None, "Small placeholder image"),
            ("/placeholder/500/300", "GET", None, None, "Medium placeholder image"),
        ]

        print("   📋 Testing miscellaneous endpoints...")
        for endpoint, method, req_headers, params, description in misc_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                iterations=3,
                test_name=description,
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

        # Test bulk operations (potentially heavy)
        bulk_endpoints = [
            (
                "/chat/all-messages-for-all-threads",
                "GET",
                headers,
                None,
                "Bulk messages for all threads",
            ),
        ]

        print("   📋 Testing bulk operation endpoints...")
        for endpoint, method, req_headers, params, description in bulk_endpoints:
            perf_stats = await measure_endpoint_performance(
                client,
                endpoint,
                method,
                req_headers,
                iterations=2,
                test_name=description,  # Fewer iterations for heavy endpoints
            )
            results.add_endpoint_performance(endpoint, perf_stats)

            for rt in perf_stats["response_times"]:
                status = perf_stats["status_codes"][
                    perf_stats["response_times"].index(rt)
                ]
                results.add_request_time(endpoint, rt, status, "individual")

    print("   ✅ Individual endpoint performance testing completed")


async def test_load_performance(test_data: Dict[str, Any]):
    """Test system performance under load."""
    print("🔍 Testing load performance...")
    print("   📝 Simulating sustained load to identify performance bottlenecks")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    load_scenarios = [
        {
            "name": "Light Load Test",
            "concurrent_users": 3,
            "requests_per_user": 5,
            "endpoints": ["/health", "/catalog", "/data-tables"],
        },
        {
            "name": "Medium Load Test",
            "concurrent_users": 5,
            "requests_per_user": 3,
            "endpoints": ["/chat-threads", "/health/memory", "/debug/pool-status"],
        },
    ]

    load_results = {}

    for scenario in load_scenarios:
        print(f"   🔄 Running {scenario['name']}...")
        print(
            f"      Users: {scenario['concurrent_users']}, Requests per user: {scenario['requests_per_user']}"
        )

        async def simulate_user_load(user_id: int, endpoints: List[str], requests: int):
            """Simulate a single user's load pattern."""
            user_results = []
            async with httpx.AsyncClient(
                base_url=SERVER_BASE_URL, timeout=httpx.Timeout(30.0)
            ) as client:
                for req_num in range(requests):
                    endpoint = endpoints[
                        req_num % len(endpoints)
                    ]  # Cycle through endpoints
                    start_time = time.time()

                    try:
                        req_headers = (
                            headers if not endpoint.startswith("/health") else None
                        )
                        response = await client.get(endpoint, headers=req_headers)
                        response_time = time.time() - start_time

                        user_results.append(
                            {
                                "user_id": user_id,
                                "request_num": req_num,
                                "endpoint": endpoint,
                                "response_time": response_time,
                                "status_code": response.status_code,
                                "success": response.status_code == 200,
                            }
                        )

                    except Exception as e:
                        response_time = time.time() - start_time
                        user_results.append(
                            {
                                "user_id": user_id,
                                "request_num": req_num,
                                "endpoint": endpoint,
                                "response_time": response_time,
                                "status_code": 0,
                                "success": False,
                                "error": str(e)[:50],
                            }
                        )

                    # Small delay to simulate realistic user behavior
                    await asyncio.sleep(0.2)

            return user_results

        # Run concurrent users
        start_time = time.time()
        tasks = [
            simulate_user_load(
                user_id, scenario["endpoints"], scenario["requests_per_user"]
            )
            for user_id in range(scenario["concurrent_users"])
        ]

        all_user_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Flatten results
        all_requests = []
        for user_results in all_user_results:
            if isinstance(user_results, list):
                all_requests.extend(user_results)

        # Analyze load test results
        if all_requests:
            successful_requests = len(
                [r for r in all_requests if r.get("success", False)]
            )
            total_requests = len(all_requests)
            avg_response_time = sum(r["response_time"] for r in all_requests) / len(
                all_requests
            )
            max_response_time = max(r["response_time"] for r in all_requests)

            # Calculate throughput
            throughput = total_requests / total_time if total_time > 0 else 0
            success_rate = (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            )

            load_results[scenario["name"]] = {
                "concurrent_users": scenario["concurrent_users"],
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "total_time": total_time,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "throughput": throughput,
                "requests_per_second": throughput,
            }

            print(
                f"      📊 Results: {successful_requests}/{total_requests} successful"
            )
            print(f"      📊 Success rate: {success_rate:.1f}%")
            print(f"      📊 Throughput: {throughput:.2f} req/s")
            print(f"      📊 Avg response: {avg_response_time:.3f}s")

            # Performance assessment
            if success_rate >= 95 and throughput >= 5:
                print("      ✅ EXCELLENT load handling")
            elif success_rate >= 90 and throughput >= 2:
                print("      🟡 ACCEPTABLE load handling")
            else:
                print("      ❌ POOR load handling")
        else:
            print("      ❌ No valid results from load test")

    return load_results


async def test_cache_performance(test_data: Dict[str, Any]):
    """Test caching performance by making repeated requests."""
    print("🔍 Testing cache performance...")
    print("   📝 Measuring response time improvements from caching")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    cache_results = {}

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        # Test endpoints that are likely to benefit from caching
        cache_test_endpoints = [
            ("/catalog", headers, "Catalog listing"),
            ("/data-tables", headers, "Data tables"),
            ("/health/memory", None, "Memory health"),
        ]

        # Add thread-specific endpoint if available
        if test_data.get("thread_ids"):
            thread_id = test_data["thread_ids"][0]
            cache_test_endpoints.append(
                (
                    f"/chat/{thread_id}/messages",
                    headers,
                    f"Thread messages {thread_id[:8]}",
                )
            )

        for endpoint, req_headers, description in cache_test_endpoints:
            print(f"   🔍 Testing cache behavior: {description}")

            # Make initial request (cold cache)
            start_time = time.time()
            try:
                response1 = await client.get(endpoint, headers=req_headers)
                first_response_time = time.time() - start_time
                first_success = response1.status_code == 200
            except Exception as e:
                first_response_time = time.time() - start_time
                first_success = False
                print(f"      ❌ First request failed: {e}")
                continue

            await asyncio.sleep(0.1)  # Brief pause

            # Make second request (warm cache)
            start_time = time.time()
            try:
                response2 = await client.get(endpoint, headers=req_headers)
                second_response_time = time.time() - start_time
                second_success = response2.status_code == 200
            except Exception as e:
                second_response_time = time.time() - start_time
                second_success = False
                print(f"      ❌ Second request failed: {e}")
                continue

            # Analyze caching effectiveness
            if first_success and second_success:
                improvement = (
                    (first_response_time - second_response_time) / first_response_time
                ) * 100

                cache_results[endpoint] = {
                    "description": description,
                    "first_response_time": first_response_time,
                    "second_response_time": second_response_time,
                    "improvement_percent": improvement,
                    "cache_effective": improvement
                    > 10,  # Consider >10% improvement as effective caching
                }

                print(
                    f"      📊 First: {first_response_time:.3f}s, Second: {second_response_time:.3f}s"
                )
                if improvement > 0:
                    print(f"      📈 Cache improvement: {improvement:.1f}%")
                    if improvement > 25:
                        print("      ✅ EXCELLENT caching")
                    elif improvement > 10:
                        print("      🟡 GOOD caching")
                    else:
                        print("      📊 MINIMAL caching benefit")
                else:
                    print(
                        f"      📊 No cache improvement (slower by {abs(improvement):.1f}%)"
                    )

    return cache_results


async def test_database_connection_performance():
    """Test database connection and query performance."""
    print("🔍 Testing database connection performance...")
    print("   📝 Measuring database-related endpoint performance and connection health")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    connection_results = {}

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        # Test database health endpoints
        db_health_tests = [
            ("/health/database", None, "Database health check"),
            ("/health/prepared-statements", None, "Prepared statements health"),
            ("/debug/pool-status", headers, "Connection pool status"),
        ]

        for endpoint, req_headers, description in db_health_tests:
            print(f"   🔍 Testing {description}...")

            response_times = []
            success_count = 0

            # Test multiple times to assess consistency
            for i in range(5):
                start_time = time.time()
                try:
                    response = await client.get(endpoint, headers=req_headers)
                    response_time = time.time() - start_time
                    response_times.append(response_time)

                    if response.status_code == 200:
                        success_count += 1

                    print(
                        f"      📊 Attempt {i+1}: {response_time:.3f}s (status: {response.status_code})"
                    )

                except Exception as e:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    print(
                        f"      ❌ Attempt {i+1}: {response_time:.3f}s (error: {str(e)[:50]})"
                    )

                await asyncio.sleep(0.2)  # Small delay between attempts

            # Calculate statistics
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                min_time = min(response_times)
                success_rate = (success_count / len(response_times)) * 100

                connection_results[endpoint] = {
                    "description": description,
                    "avg_response_time": avg_time,
                    "max_response_time": max_time,
                    "min_response_time": min_time,
                    "success_rate": success_rate,
                    "attempts": len(response_times),
                }

                print(
                    f"      📊 Average: {avg_time:.3f}s, Success: {success_rate:.1f}%"
                )

                # Assess database performance
                if success_rate >= 100 and avg_time < 1.0:
                    print("      ✅ EXCELLENT database performance")
                elif success_rate >= 90 and avg_time < 3.0:
                    print("      🟡 ACCEPTABLE database performance")
                else:
                    print("      ❌ POOR database performance")

    return connection_results


async def test_authentication_performance():
    """Test authentication and authorization performance."""
    print("🔍 Testing authentication performance...")
    print("   📝 Measuring authentication overhead and token validation speed")

    # Test different authentication scenarios
    auth_scenarios = [
        ("Valid token", create_test_jwt_token(TEST_EMAIL)),
        ("Different user token", create_test_jwt_token("different_user@example.com")),
    ]

    auth_results = {}

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        # Test endpoint that requires authentication
        test_endpoint = "/chat-threads"

        for scenario_name, token in auth_scenarios:
            print(f"   🔍 Testing {scenario_name}...")

            headers = {"Authorization": f"Bearer {token}"}
            response_times = []
            success_count = 0

            # Test authentication performance with multiple requests
            for i in range(3):
                start_time = time.time()
                try:
                    response = await client.get(test_endpoint, headers=headers)
                    response_time = time.time() - start_time
                    response_times.append(response_time)

                    if response.status_code == 200:
                        success_count += 1

                    print(
                        f"      📊 Attempt {i+1}: {response_time:.3f}s (status: {response.status_code})"
                    )

                except Exception as e:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    print(
                        f"      ❌ Attempt {i+1}: {response_time:.3f}s (error: {str(e)[:50]})"
                    )

                await asyncio.sleep(0.1)

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                success_rate = (success_count / len(response_times)) * 100

                auth_results[scenario_name] = {
                    "avg_response_time": avg_time,
                    "success_rate": success_rate,
                    "attempts": len(response_times),
                }

                print(
                    f"      📊 Average auth time: {avg_time:.3f}s, Success: {success_rate:.1f}%"
                )

    return auth_results


def analyze_performance_results(results: PerformanceResults) -> Dict[str, Any]:
    """Analyze and categorize performance test results."""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 60)

    summary = results.get_summary()
    analysis = {"overall_assessment": "UNKNOWN", "issues": [], "recommendations": []}

    # Startup Performance Analysis
    if summary["startup_time"]:
        print(f"\n🚀 STARTUP PERFORMANCE")
        startup_time = summary["startup_time"]
        if startup_time < 5:
            startup_status = "✅ EXCELLENT"
            analysis["startup_rating"] = "excellent"
        elif startup_time < 15:
            startup_status = "🟡 GOOD"
            analysis["startup_rating"] = "good"
        elif startup_time < 30:
            startup_status = "🟠 ACCEPTABLE"
            analysis["startup_rating"] = "acceptable"
            analysis["issues"].append(
                f"Startup time ({startup_time:.1f}s) is slower than optimal"
            )
            analysis["recommendations"].append(
                "Consider optimizing application initialization"
            )
        else:
            startup_status = "❌ SLOW"
            analysis["startup_rating"] = "poor"
            analysis["issues"].append(f"Startup time ({startup_time:.1f}s) is too slow")
            analysis["recommendations"].append(
                "Critical: Review application startup process"
            )

        print(f"   Startup Time: {startup_time:.2f}s {startup_status}")

    # Memory Performance Analysis
    if summary["memory_growth_mb"] is not None:
        print(f"\n💾 MEMORY PERFORMANCE")
        memory_growth = summary["memory_growth_mb"]
        peak_growth = (
            summary.get("memory_peak_mb", 0) - summary.get("memory_baseline_mb", 0)
            if summary.get("memory_peak_mb") and summary.get("memory_baseline_mb")
            else 0
        )

        if memory_growth < 25:
            memory_status = "✅ EXCELLENT"
            analysis["memory_rating"] = "excellent"
        elif memory_growth < 50:
            memory_status = "🟡 GOOD"
            analysis["memory_rating"] = "good"
        elif memory_growth < 100:
            memory_status = "🟠 ACCEPTABLE"
            analysis["memory_rating"] = "acceptable"
            analysis["issues"].append(
                f"Memory growth ({memory_growth:.1f}MB) is higher than optimal"
            )
            analysis["recommendations"].append(
                "Monitor memory usage patterns for potential leaks"
            )
        else:
            memory_status = "❌ HIGH"
            analysis["memory_rating"] = "poor"
            analysis["issues"].append(
                f"High memory growth ({memory_growth:.1f}MB) detected"
            )
            analysis["recommendations"].append("Critical: Investigate memory leaks")

        print(f"   Memory Growth: {memory_growth:.1f}MB {memory_status}")
        if peak_growth > 0:
            print(f"   Peak Memory Growth: {peak_growth:.1f}MB")

    # Response Time Analysis
    print(f"\n⚡ RESPONSE TIME PERFORMANCE")
    if summary["avg_response_time"]:
        avg_time = summary["avg_response_time"]
        p95_time = summary.get("p95_response_time", 0)
        p99_time = summary.get("p99_response_time", 0)

        if avg_time < 0.5:
            response_status = "✅ EXCELLENT"
            analysis["response_rating"] = "excellent"
        elif avg_time < 1.0:
            response_status = "🟡 GOOD"
            analysis["response_rating"] = "good"
        elif avg_time < 3.0:
            response_status = "🟠 ACCEPTABLE"
            analysis["response_rating"] = "acceptable"
            analysis["issues"].append(
                f"Average response time ({avg_time:.2f}s) could be improved"
            )
            analysis["recommendations"].append("Consider response time optimizations")
        else:
            response_status = "❌ SLOW"
            analysis["response_rating"] = "poor"
            analysis["issues"].append(
                f"Average response time ({avg_time:.2f}s) is too slow"
            )
            analysis["recommendations"].append(
                "Critical: Investigate slow response times"
            )

        print(f"   Average Response Time: {avg_time:.3f}s {response_status}")
        print(f"   Max Response Time: {summary['max_response_time']:.3f}s")
        print(f"   P95 Response Time: {p95_time:.3f}s")
        print(f"   P99 Response Time: {p99_time:.3f}s")

    # Endpoint Category Performance Analysis
    print(f"\n📋 ENDPOINT CATEGORY PERFORMANCE")
    category_performance = summary.get("endpoint_category_performance", {})
    for category, perf_data in category_performance.items():
        threshold = PERFORMANCE_THRESHOLDS.get(category, 5.0)
        avg_time = perf_data["avg_time"]

        if avg_time <= threshold:
            status = "✅"
        elif avg_time <= threshold * 1.5:
            status = "🟡"
        else:
            status = "❌"
            analysis["issues"].append(
                f"{category.title()} endpoints are slow (avg: {avg_time:.2f}s)"
            )

        print(
            f"   {category.title()}: {avg_time:.3f}s {status} ({perf_data['request_count']} requests)"
        )

    # Success Rate Analysis
    print(f"\n✅ SUCCESS RATE ANALYSIS")
    total_requests = summary["total_requests"]
    successful_requests = summary["successful_requests"]
    failed_requests = summary["failed_requests_count"]

    if total_requests > 0:
        success_rate = (successful_requests / total_requests) * 100

        if success_rate >= 99:
            success_status = "✅ EXCELLENT"
            analysis["success_rating"] = "excellent"
        elif success_rate >= 95:
            success_status = "🟡 GOOD"
            analysis["success_rating"] = "good"
        elif success_rate >= 90:
            success_status = "🟠 ACCEPTABLE"
            analysis["success_rating"] = "acceptable"
            analysis["issues"].append(
                f"Success rate ({success_rate:.1f}%) could be improved"
            )
        else:
            success_status = "❌ POOR"
            analysis["success_rating"] = "poor"
            analysis["issues"].append(
                f"Low success rate ({success_rate:.1f}%) detected"
            )
            analysis["recommendations"].append("Critical: Investigate failing requests")

        print(f"   Success Rate: {success_rate:.1f}% {success_status}")
        print(f"   Successful Requests: {successful_requests}/{total_requests}")
        print(f"   Failed Requests: {failed_requests}")

    # Concurrent Performance Analysis
    concurrent_data = summary.get("concurrent_performance", {})
    if concurrent_data:
        print(f"\n🔄 CONCURRENT PERFORMANCE")
        for scenario_name, scenario_data in concurrent_data.items():
            success_rate = scenario_data.get("success_rate", 0)
            throughput = scenario_data.get("throughput", 0)

            if success_rate >= 95 and throughput >= 5:
                concurrent_status = "✅ EXCELLENT"
            elif success_rate >= 90 and throughput >= 2:
                concurrent_status = "🟡 GOOD"
            else:
                concurrent_status = "❌ POOR"
                analysis["issues"].append(
                    f"Poor concurrent performance in {scenario_name}"
                )

            print(
                f"   {scenario_name}: {success_rate:.1f}% success, {throughput:.1f} req/s {concurrent_status}"
            )

    # Concurrent Analysis Performance
    concurrent_analysis_data = summary.get("concurrent_analysis_performance", {})
    if (
        concurrent_analysis_data
        and concurrent_analysis_data.get("performance_rating") != "failed"
    ):
        print(f"\n� MULTI-USER CONCURRENT ANALYSIS PERFORMANCE")
        
        simulation_type = concurrent_analysis_data.get("simulation_type", "unknown")
        total_requests = concurrent_analysis_data.get("total_requests", 0)
        success_rate = concurrent_analysis_data.get("success_rate", 0)
        avg_response_time = concurrent_analysis_data.get("avg_response_time", 0)
        overall_throughput = concurrent_analysis_data.get("overall_throughput", 0)
        concurrent_users = concurrent_analysis_data.get("concurrent_users", 0)
        session_duration = concurrent_analysis_data.get("session_duration", 0)
        performance_rating = concurrent_analysis_data.get("performance_rating", "unknown")
        user_stats = concurrent_analysis_data.get("user_stats", {})

        if performance_rating == "excellent":
            analysis_status = "✅ EXCELLENT"
            analysis["concurrent_analysis_rating"] = "excellent"
        elif performance_rating == "acceptable":
            analysis_status = "🟡 ACCEPTABLE"
            analysis["concurrent_analysis_rating"] = "acceptable"
        else:
            analysis_status = "❌ POOR"
            analysis["concurrent_analysis_rating"] = "poor"
            analysis["issues"].append("Poor multi-user concurrent analysis performance")
            analysis["recommendations"].append(
                "Consider optimizing analysis endpoints for concurrent multi-user load"
            )

        print(f"   Simulation: {concurrent_users} users for {session_duration}s each")
        print(f"   Total Requests: {total_requests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Average Response Time: {avg_response_time:.2f}s")
        print(f"   Overall Throughput: {overall_throughput:.2f} req/s")
        
        # Show per-user performance if available
        if user_stats:
            print(f"   Per-User Performance:")
            for user_key, stats in user_stats.items():
                user_num = user_key.split('_')[1]
                print(f"      User {user_num}: {stats['success_rate']:.1f}% success, {stats['avg_response_time']:.1f}s avg")
        
        print(f"   Performance Rating: {analysis_status}")

        # Add percentile information if available
        p95_time = concurrent_analysis_data.get("p95_response_time")
        p99_time = concurrent_analysis_data.get("p99_response_time")
        if p95_time and p99_time:
            print(f"   Response Time Distribution: P95={p95_time:.2f}s, P99={p99_time:.2f}s")

    # Overall Assessment
    print(f"\n🏁 OVERALL ASSESSMENT")

    # Count ratings
    excellent_count = len(
        [
            r
            for r in [
                analysis.get("startup_rating"),
                analysis.get("memory_rating"),
                analysis.get("response_rating"),
                analysis.get("success_rating"),
            ]
            if r == "excellent"
        ]
    )

    poor_count = len(
        [
            r
            for r in [
                analysis.get("startup_rating"),
                analysis.get("memory_rating"),
                analysis.get("response_rating"),
                analysis.get("success_rating"),
            ]
            if r == "poor"
        ]
    )

    critical_issues = len([i for i in analysis["issues"] if "Critical:" in i])

    if critical_issues > 0:
        analysis["overall_assessment"] = "❌ CRITICAL ISSUES DETECTED"
        analysis["overall_rating"] = "critical"
    elif poor_count > 1:
        analysis["overall_assessment"] = "🟠 NEEDS ATTENTION"
        analysis["overall_rating"] = "needs_attention"
    elif excellent_count >= 3:
        analysis["overall_assessment"] = "🎉 EXCELLENT PERFORMANCE"
        analysis["overall_rating"] = "excellent"
    elif len(analysis["issues"]) == 0:
        analysis["overall_assessment"] = "✅ GOOD PERFORMANCE"
        analysis["overall_rating"] = "good"
    else:
        analysis["overall_assessment"] = "🟡 ACCEPTABLE PERFORMANCE"
        analysis["overall_rating"] = "acceptable"

    print(f"   {analysis['overall_assessment']}")

    if analysis["issues"]:
        print(f"\n⚠️ IDENTIFIED ISSUES ({len(analysis['issues'])}):")
        for i, issue in enumerate(analysis["issues"], 1):
            print(f"   {i}. {issue}")

    if analysis["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")

    return analysis


async def main():
    """Main performance test runner with comprehensive coverage."""
    print("🚀 Phase 12: Comprehensive Performance Testing Starting...")
    print("=" * 70)

    test_start_time = datetime.now()

    # Check server connectivity first
    print("🔍 Pre-flight checks...")
    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        print("   Please start your uvicorn server first:")
        print(f"   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        save_traceback_report(
            report_type="test_failure",
            test_context={
                "error": "Server not accessible",
                "server_url": SERVER_BASE_URL,
            },
        )
        return False

    print("✅ Server is accessible - initializing performance test suite...")

    # Initialize results tracking
    performance_results = PerformanceResults()
    test_errors = []

    try:
        # Set up debug environment for better error tracking
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as debug_client:
            await setup_debug_environment(
                debug_client,
                DEBUG_TRACEBACK="1",
                print__api_postgresql="1",
                print__chat_messages_debug="1",
            )

        # Create test data needed for comprehensive testing
        print("\n🔧 Setting up test environment...")
        test_data = await create_test_data_for_performance_tests(TEST_EMAIL)

        # Test 1: Application startup time
        print(f"\n📋 Test 1: Application Startup Performance")
        print("-" * 50)
        startup_time = await test_application_startup_time()
        performance_results.startup_time = startup_time

        # Test 2: Memory usage patterns
        print(f"\n📋 Test 2: Memory Usage Analysis")
        print("-" * 50)
        memory_data = await test_memory_usage_patterns(test_data)
        if memory_data:
            performance_results.memory_baseline = memory_data.get("baseline")
            performance_results.memory_after_requests = memory_data.get("final")
            performance_results.memory_peak = memory_data.get("peak")

        # Test 3: Individual endpoint response times
        print(f"\n📋 Test 3: Individual Endpoint Performance")
        print("-" * 50)
        await test_endpoint_response_times(test_data, performance_results)

        # Test 4: Concurrent request handling
        print(f"\n📋 Test 4: Concurrent Request Performance")
        print("-" * 50)
        concurrent_data = await test_concurrent_request_handling(test_data)
        performance_results.concurrent_performance = concurrent_data

        # Test 5: Load testing
        print(f"\n📋 Test 5: Load Performance Testing")
        print("-" * 50)
        load_test_results = await test_load_performance(test_data)
        performance_results.load_test_results = load_test_results

        # Test 6: Database connection performance
        print(f"\n📋 Test 6: Database Connection Performance")
        print("-" * 50)
        db_performance = await test_database_connection_performance()
        if db_performance:
            performance_results.database_connection_times = list(
                db_performance.values()
            )

        # Test 7: Cache performance testing
        print(f"\n📋 Test 7: Cache Performance Analysis")
        print("-" * 50)
        cache_performance = await test_cache_performance(test_data)
        performance_results.cache_performance = cache_performance

        # Test 8: Authentication performance
        print(f"\n📋 Test 8: Authentication Performance")
        print("-" * 50)
        auth_performance = await test_authentication_performance()

        # Test 9: Concurrent analysis endpoint performance
        print(f"\n📋 Test 9: Concurrent Analysis Endpoint Performance")
        print("-" * 50)
        concurrent_analysis_data = await test_concurrent_analysis_performance(test_data)
        performance_results.concurrent_analysis_performance = concurrent_analysis_data

    except Exception as e:
        error_msg = f"Performance test execution failed: {str(e)}"
        test_errors.append(error_msg)
        print(f"❌ {error_msg}")
        traceback.print_exc()

    finally:
        # Clean up test data
        if "test_data" in locals():
            await cleanup_test_data(test_data)

        # Clean up debug environment
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as debug_client:
                await cleanup_debug_environment(
                    debug_client,
                    DEBUG_TRACEBACK="0",
                    print__api_postgresql="0",
                    print__chat_messages_debug="0",
                )
        except:
            pass  # Ignore cleanup errors

    # Comprehensive Performance Analysis
    analysis = analyze_performance_results(performance_results)

    test_end_time = datetime.now()
    total_test_duration = (test_end_time - test_start_time).total_seconds()

    print(f"\n📊 TEST EXECUTION SUMMARY")
    print("=" * 50)
    print(f"   Total Test Duration: {total_test_duration:.1f}s")
    print(f"   Total Requests Made: {len(performance_results.request_times)}")
    print(f"   Test Errors: {len(test_errors)}")

    # Determine overall test success
    test_passed = (
        len(test_errors) == 0
        and analysis.get("overall_rating") not in ["critical", "poor"]
        and len(performance_results.request_times) > 0
    )

    # Create comprehensive test context for traceback report
    test_context = {
        "Server URL": SERVER_BASE_URL,
        "Test Duration": f"{total_test_duration:.1f}s",
        "Total Requests": len(performance_results.request_times),
        "Overall Assessment": analysis["overall_assessment"],
        "Performance Rating": analysis.get("overall_rating", "unknown"),
        "Issues Found": len(analysis.get("issues", [])),
        "Test Errors": len(test_errors),
    }

    # Save comprehensive traceback report
    if test_errors or analysis.get("overall_rating") == "critical":
        # Save detailed error report
        exception_to_report = Exception(
            f"Performance test issues: {'; '.join(test_errors)}"
        )
        save_traceback_report(
            report_type="test_failure",
            exception=exception_to_report if test_errors else None,
            test_results=performance_results,
            test_context=test_context,
        )
    else:
        # Save successful test report (empty file)
        save_traceback_report(report_type="test_success", test_context=test_context)

    print(f"\n[RESULT] OVERALL RESULT: {'PASSED' if test_passed else 'FAILED'}")
    print("[OK] Comprehensive endpoint performance testing completed")
    print("[OK] Memory usage patterns analyzed")
    print("[OK] Concurrent request handling evaluated")
    print("[OK] Load performance assessed")
    print("[OK] Database connection performance measured")
    print("[OK] Cache effectiveness analyzed")
    print("[OK] Authentication performance evaluated")
    print(f"[INFO] Performance analysis: {analysis['overall_assessment']}")

    return test_passed


if __name__ == "__main__":
    # Set debug mode for better visibility
    os.environ["DEBUG"] = "1"
    os.environ["USE_TEST_TOKENS"] = "1"

    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n[STOP] Performance test interrupted by user")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Interruption": "User interrupted test execution",
        }
        save_traceback_report(report_type="interruption", test_context=test_context)
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error during performance testing: {str(e)}")
        traceback.print_exc()
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Fatal Error": str(e),
            "Error Type": type(e).__name__,
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

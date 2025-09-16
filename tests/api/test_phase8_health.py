"""Test for Phase 8.2: Health Routes
Tests the health endpoints with real HTTP requests and proper authentication.
"""

import httpx
from typing import Dict
from datetime import datetime
import time
import asyncio
import sys
import os

from pathlib import Path

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

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
)


# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_user@example.com"
REQUIRED_ENDPOINTS = {
    "/health",
    "/health/database",
    "/health/memory",
    "/health/rate-limits",
    "/health/prepared-statements",
}
TEST_QUERIES = [
    {
        "endpoint": "/health",
        "params": {},
        "description": "Basic health check",
        "should_succeed": True,
        "test_focus": "Basic health endpoint validation",
    },
    {
        "endpoint": "/health/database",
        "params": {},
        "description": "Database health check",
        "should_succeed": True,
        "test_focus": "Database connectivity verification",
    },
    {
        "endpoint": "/health/memory",
        "params": {},
        "description": "Memory health check",
        "should_succeed": True,
        "test_focus": "Memory usage monitoring",
    },
    {
        "endpoint": "/health/rate-limits",
        "params": {},
        "description": "Rate limits health check",
        "should_succeed": True,
        "test_focus": "Rate limiting status verification",
    },
    {
        "endpoint": "/health/prepared-statements",
        "params": {},
        "description": "Prepared statements health check",
        "should_succeed": True,
        "test_focus": "Prepared statements cleanup verification",
    },
    # Test invalid endpoints to ensure proper error handling
    {
        "endpoint": "/health/invalid",
        "params": {},
        "description": "Invalid health endpoint",
        "should_succeed": False,
        "expected_status": 404,
        "test_focus": "Invalid endpoint error handling",
    },
    # Test with invalid parameters to ensure robustness
    {
        "endpoint": "/health",
        "params": {"invalid_param": "test"},
        "description": "Health check with invalid parameters",
        "should_succeed": True,  # Should ignore invalid params
        "test_focus": "Parameter validation robustness",
    },
]


def _validate_response_structure(endpoint: str, data: dict, test_focus: str = None):
    """Validate response structure based on endpoint."""
    print(
        f"🔍 Testing: {test_focus or f'Response structure validation for {endpoint}'}"
    )

    if endpoint == "/health":
        # Main health check response validation
        assert "status" in data, "Missing 'status' field"
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert "uptime_seconds" in data, "Missing 'uptime_seconds' field"
        assert "memory" in data, "Missing 'memory' field"
        assert "database" in data, "Missing 'database' field"
        assert "version" in data, "Missing 'version' field"

        # Validate status values
        assert data["status"] in [
            "healthy",
            "degraded",
            "error",
        ], f"Invalid status: {data['status']}"
        assert isinstance(
            data["uptime_seconds"], (int, float)
        ), "'uptime_seconds' must be numeric"
        assert data["uptime_seconds"] >= 0, "'uptime_seconds' must be non-negative"

        # Validate memory structure
        memory = data["memory"]
        assert "rss_mb" in memory, "Missing 'rss_mb' in memory"
        assert "vms_mb" in memory, "Missing 'vms_mb' in memory"
        assert "percent" in memory, "Missing 'percent' in memory"
        assert isinstance(memory["rss_mb"], (int, float)), "'rss_mb' must be numeric"
        assert isinstance(memory["vms_mb"], (int, float)), "'vms_mb' must be numeric"
        assert isinstance(memory["percent"], (int, float)), "'percent' must be numeric"
        assert memory["rss_mb"] > 0, "'rss_mb' must be positive"
        assert memory["percent"] >= 0, "'percent' must be non-negative"

        # Validate database structure
        database = data["database"]
        assert "healthy" in database, "Missing 'healthy' in database"
        assert (
            "checkpointer_type" in database
        ), "Missing 'checkpointer_type' in database"
        assert isinstance(database["healthy"], bool), "'healthy' must be boolean"
        assert isinstance(
            database["checkpointer_type"], str
        ), "'checkpointer_type' must be string"

        print(
            f"✅ {endpoint} validation passed - Status: {data['status']}, Memory: {memory['rss_mb']}MB"
        )

    elif endpoint == "/health/database":
        # Database health check validation
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert (
            "checkpointer_available" in data
        ), "Missing 'checkpointer_available' field"
        assert "checkpointer_type" in data, "Missing 'checkpointer_type' field"
        assert isinstance(
            data["checkpointer_available"], bool
        ), "'checkpointer_available' must be boolean"

        # If database connection info is present, validate it
        if "database_connection" in data:
            assert data["database_connection"] in [
                "healthy",
                "error",
                "using_memory_fallback",
            ], f"Invalid database_connection: {data['database_connection']}"

            if data["database_connection"] == "healthy" and "read_latency_ms" in data:
                assert isinstance(
                    data["read_latency_ms"], (int, float)
                ), "'read_latency_ms' must be numeric"
                assert (
                    data["read_latency_ms"] >= 0
                ), "'read_latency_ms' must be non-negative"

        print(
            f"✅ {endpoint} validation passed - Checkpointer: {data.get('checkpointer_type', 'Unknown')}"
        )

    elif endpoint == "/health/memory":
        # Memory health check validation
        assert "status" in data, "Missing 'status' field"
        assert "memory_rss_mb" in data, "Missing 'memory_rss_mb' field"
        assert "memory_threshold_mb" in data, "Missing 'memory_threshold_mb' field"
        assert "memory_usage_percent" in data, "Missing 'memory_usage_percent' field"
        assert "over_threshold" in data, "Missing 'over_threshold' field"
        assert (
            "total_requests_processed" in data
        ), "Missing 'total_requests_processed' field"
        assert "cache_info" in data, "Missing 'cache_info' field"
        assert "scaling_info" in data, "Missing 'scaling_info' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        # Validate status values
        assert data["status"] in [
            "healthy",
            "warning",
            "high_memory",
            "error",
        ], f"Invalid memory status: {data['status']}"
        assert isinstance(
            data["memory_rss_mb"], (int, float)
        ), "'memory_rss_mb' must be numeric"
        assert isinstance(
            data["memory_threshold_mb"], (int, float)
        ), "'memory_threshold_mb' must be numeric"
        assert isinstance(
            data["memory_usage_percent"], (int, float)
        ), "'memory_usage_percent' must be numeric"
        assert isinstance(
            data["over_threshold"], bool
        ), "'over_threshold' must be boolean"
        assert isinstance(
            data["total_requests_processed"], int
        ), "'total_requests_processed' must be integer"

        # Validate cache_info structure
        cache_info = data["cache_info"]
        assert (
            "active_cache_entries" in cache_info
        ), "Missing 'active_cache_entries' in cache_info"
        assert (
            "cleaned_expired_entries" in cache_info
        ), "Missing 'cleaned_expired_entries' in cache_info"
        assert (
            "cache_timeout_seconds" in cache_info
        ), "Missing 'cache_timeout_seconds' in cache_info"

        # Validate scaling_info structure
        scaling_info = data["scaling_info"]
        assert (
            "estimated_memory_per_thread_mb" in scaling_info
        ), "Missing 'estimated_memory_per_thread_mb'"
        assert (
            "estimated_max_threads_at_threshold" in scaling_info
        ), "Missing 'estimated_max_threads_at_threshold'"
        assert "current_thread_count" in scaling_info, "Missing 'current_thread_count'"

        print(
            f"✅ {endpoint} validation passed - Status: {data['status']}, Memory: {data['memory_rss_mb']}MB"
        )

    elif endpoint == "/health/rate-limits":
        # Rate limits health check validation
        assert "status" in data, "Missing 'status' field"
        assert "total_tracked_clients" in data, "Missing 'total_tracked_clients' field"
        assert "active_clients" in data, "Missing 'active_clients' field"
        assert "rate_limit_window" in data, "Missing 'rate_limit_window' field"
        assert "rate_limit_requests" in data, "Missing 'rate_limit_requests' field"
        assert "rate_limit_burst" in data, "Missing 'rate_limit_burst' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        # Validate numeric values
        assert isinstance(
            data["total_tracked_clients"], int
        ), "'total_tracked_clients' must be integer"
        assert isinstance(
            data["active_clients"], int
        ), "'active_clients' must be integer"
        assert isinstance(
            data["rate_limit_window"], (int, float)
        ), "'rate_limit_window' must be numeric"
        assert isinstance(
            data["rate_limit_requests"], int
        ), "'rate_limit_requests' must be integer"
        assert isinstance(
            data["rate_limit_burst"], int
        ), "'rate_limit_burst' must be integer"
        assert (
            data["total_tracked_clients"] >= 0
        ), "'total_tracked_clients' must be non-negative"
        assert data["active_clients"] >= 0, "'active_clients' must be non-negative"
        assert (
            data["active_clients"] <= data["total_tracked_clients"]
        ), "'active_clients' cannot exceed 'total_tracked_clients'"

        print(
            f"✅ {endpoint} validation passed - Status: {data['status']}, Clients: {data['active_clients']}/{data['total_tracked_clients']}"
        )

    elif endpoint == "/health/prepared-statements":
        # Prepared statements health check validation
        assert "status" in data, "Missing 'status' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        # Validate status values
        assert data["status"] in [
            "healthy",
            "degraded",
            "unhealthy",
        ], f"Invalid prepared statements status: {data['status']}"

        # If checkpointer_status is present, validate it
        if "checkpointer_status" in data:
            assert isinstance(
                data["checkpointer_status"], str
            ), "'checkpointer_status' must be string"

        # If prepared statements count is present, validate it
        if "prepared_statements_count" in data:
            assert isinstance(
                data["prepared_statements_count"], int
            ), "'prepared_statements_count' must be integer"
            assert (
                data["prepared_statements_count"] >= 0
            ), "'prepared_statements_count' must be non-negative"

        # If connection kwargs are present, validate structure
        if "connection_kwargs" in data:
            assert isinstance(
                data["connection_kwargs"], dict
            ), "'connection_kwargs' must be dict"

        print(f"✅ {endpoint} validation passed - Status: {data['status']}")


def _get_test_explanation(
    test_focus: str,
    should_succeed: bool,
    expected_status: int,
    endpoint: str,
    params: dict = None,
) -> str:
    """Generate a detailed explanation of what the test is validating."""

    if should_succeed:
        # Success cases - explain what functionality we're testing
        if endpoint == "/health":
            return "Main health check: validates server status, uptime, memory usage, and database connectivity"
        elif endpoint == "/health/database":
            return "Database health: verifies checkpointer availability, connection status, and read latency"
        elif endpoint == "/health/memory":
            return "Memory monitoring: checks memory usage, thresholds, cache status, and scaling information"
        elif endpoint == "/health/rate-limits":
            return "Rate limiting status: validates client tracking, limits configuration, and active connections"
        elif endpoint == "/health/prepared-statements":
            return "Prepared statements: verifies PostgreSQL prepared statement cleanup and connection health"
        elif params and "invalid_param" in params:
            return "Parameter robustness: ensures endpoint gracefully handles unexpected query parameters"
    else:
        # Failure cases - explain what validation we're testing
        if expected_status == 404:
            return f"Error handling: invalid endpoint '{endpoint}' should return 404 Not Found"
        elif expected_status == 422:
            return "Validation error: malformed request should be rejected with proper error response"

    return f"Testing {test_focus} - verifying proper health endpoint behavior"


async def make_health_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    params: Dict,
    description: str,
    should_succeed: bool,
    results: BaseTestResults,
    expected_status: int = None,
    test_focus: str = None,
):
    """Make a request to a health endpoint with server traceback capture."""
    # Print detailed test information
    print(f"\n� TEST {test_id}: {test_focus or description}")
    print(f"   📍 Endpoint: GET {endpoint}")
    print(f"   📋 Parameters: {params or 'None'}")
    print(
        f"   ✅ Expected Result: {'Success (200/503)' if should_succeed else f'Failure ({expected_status or 422})'}"
    )
    print(
        f"   🎯 What we're testing: {_get_test_explanation(test_focus, should_succeed, expected_status, endpoint, params)}"
    )

    # Health endpoints typically don't require authentication, but we'll include it for consistency
    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    start_time = time.time()
    try:
        result = await make_request_with_traceback_capture(
            client,
            "GET",
            f"{SERVER_BASE_URL}{endpoint}",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(result)

        if result["response"] is None:
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"   ❌ [ERROR] Client Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)
            return

        response = result["response"]
        print(f"   📊 Response: {response.status_code} in {response_time:.2f}s")

        # Print response details for detailed output
        try:
            response_data = response.json()
            if isinstance(response_data, dict) and "status" in response_data:
                print(f"   🏥 Health Status: {response_data['status']}")
        except:
            pass

        if should_succeed:
            if response.status_code == 200:
                try:
                    data = response.json()
                    _validate_response_structure(endpoint, data, test_focus)
                    print(
                        f"   ✅ [OK] Test passed - Health status: {data.get('status', 'unknown')}"
                    )
                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        data,
                        response_time,
                        response.status_code,
                    )
                except (AssertionError, Exception) as e:
                    print(f"   ❌ [ERROR] Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id, endpoint, description, error_obj, response_time
                    )
            elif response.status_code == 503:
                # Health endpoints may return 503 for degraded services - this is valid
                try:
                    data = response.json()
                    if "status" in data and data["status"] in ["degraded", "unhealthy"]:
                        print(
                            f"   ⚠️ [INFO] Service degraded (HTTP 503): {data.get('status')}"
                        )
                        results.add_result(
                            test_id,
                            endpoint,
                            description,
                            data,
                            response_time,
                            response.status_code,
                        )
                    else:
                        raise ValueError("Invalid degraded service response structure")
                except Exception as e:
                    print(f"   ❌ [ERROR] Invalid 503 response: {e}")
                    error_obj = Exception(f"Invalid degraded service response: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id, endpoint, description, error_obj, response_time
                    )
            else:
                handle_error_response(
                    test_id,
                    endpoint,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )
        else:
            # Handle expected failure cases
            if expected_status and response.status_code == expected_status:
                print(
                    f"✅ Test {test_id} - Correctly failed with HTTP {expected_status}"
                )
                data = {"expected_failure": True, "status_code": expected_status}
                results.add_result(
                    test_id,
                    endpoint,
                    description,
                    data,
                    response_time,
                    response.status_code,
                )
            else:
                handle_expected_failure(
                    test_id,
                    endpoint,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Unknown error of type {type(e).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}")
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(
            test_id, endpoint, description, error_obj, response_time, response_data=None
        )


async def run_health_tests() -> BaseTestResults:
    """Run all health endpoint tests."""
    print("🚀 Starting health tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Test connectivity first
        connectivity_test = await check_server_connectivity(SERVER_BASE_URL)
        if not connectivity_test:
            print("❌ Server connectivity check failed - aborting health tests")
            return results

        # Run all test cases
        for i, test_case in enumerate(TEST_QUERIES, 1):
            test_id = f"test_{i}"
            await make_health_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["params"],
                test_case["description"],
                test_case["should_succeed"],
                results,
                expected_status=test_case.get("expected_status"),
                test_focus=test_case.get("test_focus"),
            )
            await asyncio.sleep(0.1)  # Small delay between requests

        # Additional stress test - multiple rapid health checks
        print("🔄 Running rapid health check stress test...")
        for i in range(5):
            test_id = f"stress_test_{i+1}"
            await make_health_request(
                client,
                test_id,
                "/health",
                {},
                f"Rapid health check #{i+1}",
                True,
                results,
                test_focus="Performance testing under rapid consecutive requests",
            )
            await asyncio.sleep(0.05)  # Very short delay for stress testing

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"Max Response Time: {summary['max_response_time']:.2f}s")
        print(f"Min Response Time: {summary['min_response_time']:.2f}s")

    if not summary["all_endpoints_tested"]:
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

    # Show specific health insights
    if results.results:
        health_statuses = {}
        memory_usage = []
        database_statuses = {}

        for result in results.results:
            if result["endpoint"] == "/health" and result["success"]:
                data = result["response_data"]
                health_statuses[result["test_id"]] = data.get("status", "unknown")
                if "memory" in data:
                    memory_usage.append(data["memory"].get("rss_mb", 0))
                if "database" in data:
                    database_statuses[result["test_id"]] = data["database"].get(
                        "healthy", False
                    )

        if health_statuses:
            print(f"\n📋 Health Status Summary:")
            status_counts = {}
            for status in health_statuses.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            for status, count in status_counts.items():
                print(f"  {status}: {count} checks")

        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
            min_memory = min(memory_usage)
            print(
                f"\n💾 Memory Usage: Avg={avg_memory:.1f}MB, Max={max_memory:.1f}MB, Min={min_memory:.1f}MB"
            )

        if database_statuses:
            healthy_db_count = sum(
                1 for healthy in database_statuses.values() if healthy
            )
            print(
                f"\n🗄️ Database Health: {healthy_db_count}/{len(database_statuses)} checks healthy"
            )

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors:
            print(f"  {error['test_id']}: {error['error']}")

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("🚀 Health Endpoints Test Starting...")
    print(f"📋 Testing endpoints: {', '.join(REQUIRED_ENDPOINTS)}")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                DEBUG_TRACEBACK="1",
                HEALTH_DEBUG="1",  # Enable health-specific debugging
            )
            results = await run_health_tests()
            await cleanup_debug_environment(
                client,
                DEBUG_TRACEBACK="0",
                HEALTH_DEBUG="0",
            )

        summary = analyze_test_results(results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )
        has_critical_errors = any(
            "no such variable" in error.get("error", "").lower()
            or "nameError" in error.get("error", "")
            or "undefined" in error.get("error", "").lower()
            for error in summary["errors"]
        )

        # Health endpoints should be more tolerant of degraded states
        test_passed = (
            not has_empty_errors
            and not has_critical_errors
            and summary["total_requests"] > 0
            and summary["all_endpoints_tested"]
            and summary["successful_requests"] > 0
            and summary["success_rate"] >= 80.0  # Allow some degraded responses
        )

        if has_empty_errors:
            print("❌ Test failed: Server returned empty error messages")
        elif has_critical_errors:
            print("❌ Test failed: Critical errors detected")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded")
        elif not summary["all_endpoints_tested"]:
            print("❌ Test failed: Not all required endpoints were tested")
        elif summary["success_rate"] < 80.0:
            print(
                f"❌ Test failed: Success rate too low ({summary['success_rate']:.1f}% < 80%)"
            )

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")

        # Print what was tested
        print(f"\n📝 TESTS PERFORMED:")
        print(f"   • Basic health check endpoint validation")
        print(f"   • Database connectivity and health verification")
        print(f"   • Memory usage monitoring and threshold checking")
        print(f"   • Rate limiting status and client tracking")
        print(f"   • Prepared statements health and cleanup verification")
        print(f"   • Invalid endpoint error handling")
        print(f"   • Parameter validation and robustness testing")
        print(f"   • Rapid request stress testing")
        print(f"   • Response structure and data type validation")
        print(f"   • Server-side error capture and traceback analysis")

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Required Endpoints": ", ".join(REQUIRED_ENDPOINTS),
            "Error Location": "main() function",
            "Error During": "Health endpoint test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Required Endpoints": ", ".join(REQUIRED_ENDPOINTS),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

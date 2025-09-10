"""Test for Phase 8: Debug Routes
Tests the debug endpoints with real HTTP requests and proper authentication.
"""

import os
import sys
import time
import traceback
import uuid
from typing import Dict, Any
from datetime import datetime

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx

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


class DebugTestResults(BaseTestResults):
    """Extended test results class that handles endpoint pattern matching."""

    def __init__(self, required_endpoint_patterns: set = None):
        super().__init__(required_endpoints=set())  # Initialize parent with empty set
        self.required_endpoint_patterns = required_endpoint_patterns or set()
        self.setup_failures = []  # Track setup/cleanup failures separately

    def add_setup_failure(self, operation: str, error: str):
        """Add a setup/cleanup failure (not counted as test failure)."""
        self.setup_failures.append(
            {
                "operation": operation,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_expected_failure_result(
        self,
        test_id: str,
        endpoint: str,
        description: str,
        response_data: Dict,
        response_time: float,
        status_code: int,
    ):
        """Add a test result for an expected failure (counts as success)."""
        result = {
            "test_id": test_id,
            "endpoint": endpoint,
            "description": description,
            "response_data": response_data,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "success": True,  # Expected failures count as successes
            "expected_failure": True,
        }
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results with proper endpoint pattern matching."""
        # Get base summary
        summary = super().get_summary()

        # Recalculate based on actual tests only (exclude setup failures)
        actual_test_requests = len(self.results) + len(self.errors)
        actual_successful_requests = len([r for r in self.results if r["success"]])
        actual_failed_requests = len(self.errors) + len(
            [r for r in self.results if not r["success"]]
        )

        # Check endpoint coverage using pattern matching
        tested_endpoints = set(r["endpoint"] for r in self.results if r["success"])
        covered_patterns = set()
        missing_patterns = set()

        for pattern in self.required_endpoint_patterns:
            pattern_covered = any(
                endpoint_matches_pattern(endpoint, pattern)
                for endpoint in tested_endpoints
            )
            if pattern_covered:
                covered_patterns.add(pattern)
            else:
                missing_patterns.add(pattern)

        # Update summary with corrected values
        summary.update(
            {
                "total_requests": actual_test_requests,
                "successful_requests": actual_successful_requests,
                "failed_requests": actual_failed_requests,
                "success_rate": (
                    (actual_successful_requests / actual_test_requests * 100)
                    if actual_test_requests > 0
                    else 0
                ),
                "all_endpoints_tested": len(missing_patterns) == 0,
                "covered_patterns": covered_patterns,
                "missing_patterns": missing_patterns,
                "setup_failures": self.setup_failures,
            }
        )

        return summary


# Set Windows event loop policy FIRST
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_user@example.com"


# Function to check if an endpoint matches a pattern
def endpoint_matches_pattern(endpoint: str, pattern: str) -> bool:
    """Check if an endpoint matches a pattern with placeholders like {thread_id}."""
    import re

    # Convert pattern to regex by replacing {param} with a wildcard
    regex_pattern = re.sub(r"\{[^}]+\}", r"[^/]+", pattern)
    regex_pattern = f"^{regex_pattern}$"
    return bool(re.match(regex_pattern, endpoint))


# Required endpoint patterns (will be matched against actual endpoints)
REQUIRED_ENDPOINT_PATTERNS = {
    "/debug/pool-status",
    "/debug/chat/{thread_id}/checkpoints",
    "/debug/run-id/{run_id}",
    "/admin/clear-cache",
    "/admin/clear-prepared-statements",
    "/debug/set-env",
    "/debug/reset-env",
}

# Test cases for debug endpoints
TEST_QUERIES = [
    # Pool status endpoint (no auth required)
    {
        "endpoint": "/debug/pool-status",
        "method": "GET",
        "params": {},
        "description": "Check database pool status",
        "should_succeed": True,
        "requires_auth": False,
    },
    # Checkpoint inspection (requires auth and thread_id)
    {
        "endpoint": "/debug/chat/test-thread-123/checkpoints",
        "method": "GET",
        "params": {},
        "description": "Inspect checkpoints for test thread",
        "should_succeed": True,
        "requires_auth": True,
    },
    {
        "endpoint": "/debug/chat/non-existent-thread/checkpoints",
        "method": "GET",
        "params": {},
        "description": "Inspect checkpoints for non-existent thread",
        "should_succeed": True,  # Should return empty results, not fail
        "requires_auth": True,
    },
    # Run ID debugging (requires auth and run_id)
    {
        "endpoint": f"/debug/run-id/{uuid.uuid4()}",
        "method": "GET",
        "params": {},
        "description": "Check valid UUID format run_id",
        "should_succeed": True,
        "requires_auth": True,
    },
    {
        "endpoint": "/debug/run-id/invalid-uuid-format",
        "method": "GET",
        "params": {},
        "description": "Check invalid UUID format run_id",
        "should_succeed": True,  # Should handle gracefully
        "requires_auth": True,
    },
    # Admin cache operations (requires auth)
    {
        "endpoint": "/admin/clear-cache",
        "method": "POST",
        "params": {},
        "description": "Clear bulk loading cache",
        "should_succeed": True,
        "requires_auth": True,
    },
    {
        "endpoint": "/admin/clear-prepared-statements",
        "method": "POST",
        "params": {},
        "description": "Clear database prepared statements",
        "should_succeed": True,
        "requires_auth": True,
    },
    # Environment variable management (requires auth)
    {
        "endpoint": "/debug/set-env",
        "method": "POST",
        "json": {"TEST_DEBUG_VAR": "test_value", "ANOTHER_VAR": "another_value"},
        "description": "Set debug environment variables",
        "should_succeed": True,
        "requires_auth": True,
    },
    {
        "endpoint": "/debug/reset-env",
        "method": "POST",
        "json": {"TEST_DEBUG_VAR": "", "ANOTHER_VAR": ""},
        "description": "Reset debug environment variables",
        "should_succeed": True,
        "requires_auth": True,
    },
    # Authentication failure tests
    {
        "endpoint": "/debug/chat/test-thread/checkpoints",
        "method": "GET",
        "params": {},
        "description": "Checkpoint access without auth",
        "should_succeed": False,
        "requires_auth": False,  # Explicitly no auth to test failure
        "expected_status": 401,
    },
    {
        "endpoint": "/admin/clear-cache",
        "method": "POST",
        "params": {},
        "description": "Cache clear without auth",
        "should_succeed": False,
        "requires_auth": False,  # Explicitly no auth to test failure
        "expected_status": 401,
    },
]


def _validate_response_structure(endpoint: str, data: dict, method: str):
    """Validate response structure based on endpoint."""

    if endpoint == "/debug/pool-status":
        print(f"🔍 Testing pool status response structure...")
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert (
            "global_checkpointer_exists" in data
        ), "Missing 'global_checkpointer_exists' field"
        assert isinstance(
            data["global_checkpointer_exists"], bool
        ), "'global_checkpointer_exists' must be boolean"

        if data.get("checkpointer_type"):
            assert isinstance(
                data["checkpointer_type"], str
            ), "'checkpointer_type' must be string"

        print(
            f"✅ Pool status validation passed - Checkpointer exists: {data['global_checkpointer_exists']}"
        )

    elif "checkpoints" in endpoint:
        print(f"🔍 Testing checkpoint response structure...")

        # Handle error response case when checkpointer is not available
        if "error" in data:
            print(f"⚠️ Checkpoint endpoint returned error: {data['error']}")
            assert isinstance(data["error"], str), "'error' must be string"
            print(f"✅ Checkpoint error validation passed - Error handled gracefully")
            return

        # Normal response validation
        assert "thread_id" in data, "Missing 'thread_id' field"
        assert "total_checkpoints" in data, "Missing 'total_checkpoints' field"
        assert "checkpoints" in data, "Missing 'checkpoints' field"
        assert isinstance(
            data["total_checkpoints"], int
        ), "'total_checkpoints' must be integer"
        assert isinstance(data["checkpoints"], list), "'checkpoints' must be list"

        for checkpoint in data["checkpoints"]:
            assert "index" in checkpoint, "Missing 'index' in checkpoint"
            assert (
                "has_checkpoint" in checkpoint
            ), "Missing 'has_checkpoint' in checkpoint"
            assert "has_metadata" in checkpoint, "Missing 'has_metadata' in checkpoint"

        print(
            f"✅ Checkpoint validation passed - Found {data['total_checkpoints']} checkpoints"
        )

    elif "run-id" in endpoint:
        print(f"🔍 Testing run-id response structure...")

        if data is None:
            print(f"❌ Response data is None for run-id endpoint")
            raise Exception(
                "Response data is None - this indicates a server-side issue"
            )

        assert "run_id" in data, "Missing 'run_id' field"
        assert "run_id_type" in data, "Missing 'run_id_type' field"
        assert "run_id_length" in data, "Missing 'run_id_length' field"
        assert "is_valid_uuid_format" in data, "Missing 'is_valid_uuid_format' field"
        assert "exists_in_database" in data, "Missing 'exists_in_database' field"
        assert "user_owns_run_id" in data, "Missing 'user_owns_run_id' field"

        assert isinstance(
            data["is_valid_uuid_format"], bool
        ), "'is_valid_uuid_format' must be boolean"
        assert isinstance(
            data["exists_in_database"], bool
        ), "'exists_in_database' must be boolean"
        assert isinstance(
            data["user_owns_run_id"], bool
        ), "'user_owns_run_id' must be boolean"

        # Handle database_details which can be None
        if "database_details" in data and data["database_details"] is not None:
            db_details = data["database_details"]
            assert isinstance(
                db_details, dict
            ), "'database_details' must be dict or None"
            if "timestamp" in db_details and db_details["timestamp"] is not None:
                # Timestamp validation - should be ISO format string if not None
                assert isinstance(
                    db_details["timestamp"], str
                ), "'timestamp' must be string if not None"

        print(
            f"✅ Run-id validation passed - UUID format: {data['is_valid_uuid_format']}, DB exists: {data['exists_in_database']}"
        )

    elif "clear-cache" in endpoint and method == "POST":
        print(f"🔍 Testing cache clear response structure...")
        assert "message" in data, "Missing 'message' field"
        assert "cache_entries_cleared" in data, "Missing 'cache_entries_cleared' field"
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert "cleared_by" in data, "Missing 'cleared_by' field"

        assert isinstance(
            data["cache_entries_cleared"], int
        ), "'cache_entries_cleared' must be integer"
        assert (
            data["cache_entries_cleared"] >= 0
        ), "'cache_entries_cleared' must be non-negative"

        print(
            f"✅ Cache clear validation passed - Cleared {data['cache_entries_cleared']} entries"
        )

    elif "clear-prepared-statements" in endpoint and method == "POST":
        print(f"🔍 Testing prepared statements clear response structure...")
        assert "message" in data, "Missing 'message' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        print(f"✅ Prepared statements clear validation passed")

    # Check reset-env before set-env to avoid substring matching ("set-env" in "reset-env")
    elif "reset-env" in endpoint and method == "POST":
        print(f"🔍 Testing reset environment variables response structure...")
        assert "message" in data, "Missing 'message' field"
        assert "variables" in data, "Missing 'variables' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        assert isinstance(data["variables"], dict), "'variables' must be dict"

        print(
            f"✅ Reset environment validation passed - Reset {len(data['variables'])} variables"
        )

    elif "set-env" in endpoint and method == "POST":
        print(f"🔍 Testing set environment variables response structure...")
        assert "message" in data, "Missing 'message' field"
        assert "variables" in data, "Missing 'variables' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

        assert isinstance(data["variables"], dict), "'variables' must be dict"

        print(
            f"✅ Set environment validation passed - Set {len(data['variables'])} variables"
        )


async def make_debug_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    method: str,
    params: Dict,
    json_data: Dict,
    description: str,
    should_succeed: bool,
    requires_auth: bool,
    results: BaseTestResults,
    expected_status: int = None,
):
    """Make a request to a debug endpoint with server traceback capture."""
    headers = {}
    if requires_auth:
        token = create_test_jwt_token(TEST_EMAIL)
        headers["Authorization"] = f"Bearer {token}"

    start_time = time.time()
    try:
        request_kwargs = {
            "headers": headers,
            "timeout": REQUEST_TIMEOUT,
        }

        if params:
            request_kwargs["params"] = params
        if json_data:
            request_kwargs["json"] = json_data

        result = await make_request_with_traceback_capture(
            client, method.upper(), f"{SERVER_BASE_URL}{endpoint}", **request_kwargs
        )

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(result)

        if result["response"] is None:
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"❌ Test {test_id} - Client Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id,
                endpoint,
                description,
                error_obj,
                response_time,
                response_data=None,
            )
            return

        response = result["response"]
        print(
            f"Test {test_id}: {method} {endpoint} -> {response.status_code} ({response_time:.2f}s)"
        )

        if should_succeed:
            if response.status_code == 200:
                try:
                    response_data = response.json()

                    # Handle case where response_data is None (shouldn't happen with 200 status)
                    if response_data is None:
                        print(
                            f"⚠️ Test {test_id} - Received None response data with 200 status"
                        )
                        print(f"   Raw response: {response.text}")
                        error_obj = Exception(
                            "Received None response data with 200 status code"
                        )
                        error_obj.server_tracebacks = error_info["server_tracebacks"]
                        results.add_error(
                            test_id,
                            endpoint,
                            description,
                            error_obj,
                            response_time,
                            response_data=None,
                        )
                        return

                    _validate_response_structure(endpoint, response_data, method)

                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        response_data,
                        response_time,
                        response.status_code,
                    )
                    print(f"✅ Test {test_id} - Success: {description}")

                except Exception as validation_error:
                    print(f"❌ Test {test_id} - Validation failed: {validation_error}")
                    error_obj = Exception(
                        f"Response validation failed: {validation_error}"
                    )
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id,
                        endpoint,
                        description,
                        error_obj,
                        response_time,
                        response_data=response_data,
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
            # Expected failure case
            # Check if results object has the special method for expected failures
            if (
                hasattr(results, "add_expected_failure_result")
                and expected_status
                and response.status_code == expected_status
            ):
                print(
                    f"✅ Test {test_id} - Correctly failed with HTTP {expected_status}"
                )
                data = {"expected_failure": True, "status_code": expected_status}
                results.add_expected_failure_result(
                    test_id,
                    endpoint,
                    description,
                    data,
                    response_time,
                    response.status_code,
                )
            else:
                # Use original handle_expected_failure for other cases
                handle_expected_failure(
                    test_id,
                    endpoint,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                    expected_status=expected_status,
                )

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Empty error from {type(e).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}")
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(
            test_id, endpoint, description, error_obj, response_time, response_data=None
        )


async def run_debug_tests() -> DebugTestResults:
    """Run all debug endpoint tests."""
    print("🚀 Starting debug tests...")

    results = DebugTestResults(required_endpoint_patterns=REQUIRED_ENDPOINT_PATTERNS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Set up debug environment for testing
        print("🔧 Setting up debug environment...")
        setup_success = await setup_debug_environment(
            client, DEBUG_MEMORY_MONITORING="1", DEBUG_PRINT="1", CLEANED_TRACEBACK="1"
        )
        if setup_success:
            print("✅ Debug environment setup successful")
        else:
            print("⚠️ Debug environment setup failed, continuing anyway...")
            results.add_setup_failure(
                "debug_environment_setup", "Failed to set debug environment variables"
            )

        # Run all test cases
        for i, test_case in enumerate(TEST_QUERIES, 1):
            test_id = f"DEBUG_{i:02d}"
            endpoint = test_case["endpoint"]
            method = test_case["method"]
            params = test_case.get("params", {})
            json_data = test_case.get("json", {})
            description = test_case["description"]
            should_succeed = test_case["should_succeed"]
            requires_auth = test_case["requires_auth"]
            expected_status = test_case.get("expected_status")

            print(f"\n🧪 Running Test {test_id}: {description}")
            print(f"   Method: {method}, Endpoint: {endpoint}")
            print(
                f"   Auth Required: {requires_auth}, Should Succeed: {should_succeed}"
            )

            await make_debug_request(
                client,
                test_id,
                endpoint,
                method,
                params,
                json_data,
                description,
                should_succeed,
                requires_auth,
                results,
                expected_status,
            )

        # Clean up debug environment
        print("\n🧹 Cleaning up debug environment...")
        cleanup_success = await cleanup_debug_environment(
            client, DEBUG_MEMORY_MONITORING="0", DEBUG_PRINT="0"
        )
        if cleanup_success:
            print("✅ Debug environment cleanup successful")
        else:
            print("⚠️ Debug environment cleanup failed")
            results.add_setup_failure(
                "debug_environment_cleanup",
                "Failed to reset debug environment variables",
            )

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: DebugTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.2f}s")

    # Show setup/cleanup failures separately
    if summary["setup_failures"]:
        print(f"\n⚠️ Setup/Cleanup Issues ({len(summary['setup_failures'])}):")
        for failure in summary["setup_failures"]:
            print(f"   - {failure['operation']}: {failure['error']}")

    # Show endpoint coverage
    if not summary["all_endpoints_tested"]:
        print(
            f"\n❌ Missing endpoint patterns: {', '.join(summary['missing_patterns'])}"
        )
    else:
        print(
            f"\n✅ All required endpoint patterns covered ({len(summary['covered_patterns'])})"
        )

    # Show actual test errors with details
    if results.errors:
        print(f"\n❌ {len(results.errors)} Test Errors:")
        for i, error in enumerate(results.errors, 1):
            print(f"\n   {i}. Test {error['test_id']}: {error['description']}")
            print(f"      Endpoint: {error['endpoint']}")
            print(f"      Error: {error['error']}")
            print(f"      Type: {error['error_type']}")

            # Show server traceback if available
            if hasattr(error.get("error_obj"), "server_tracebacks"):
                tracebacks = error["error_obj"].server_tracebacks
                if tracebacks:
                    print(f"      Server Tracebacks: {len(tracebacks)} found")
                    for j, tb in enumerate(
                        tracebacks[:2], 1
                    ):  # Show first 2 tracebacks
                        print(
                            f"        TB{j}: {tb.get('exception_type', 'Unknown')} - {tb.get('exception_message', 'Unknown')}"
                        )

            # Show response data if available and contains error info
            if error.get("response_data") and isinstance(error["response_data"], dict):
                if "detail" in error["response_data"]:
                    print(f"      Server Detail: {error['response_data']['detail']}")
                if "traceback" in error["response_data"]:
                    print(f"      Server Traceback: Available in response")

    # Show failed results (different from errors, exclude expected failures)
    failed_results = [
        r
        for r in results.results
        if not r.get("success", True) and not r.get("expected_failure", False)
    ]
    if failed_results:
        print(f"\n❌ {len(failed_results)} Failed Results:")
        for i, result in enumerate(failed_results, 1):
            print(f"   {i}. Test {result['test_id']}: {result['description']}")
            print(f"      Endpoint: {result['endpoint']}")
            print(f"      Status: {result['status_code']}")
            if result.get("response_data"):
                print(f"      Response: {result['response_data']}")

    # Show expected failures separately as informational
    expected_failures = [r for r in results.results if r.get("expected_failure", False)]
    if expected_failures:
        print(f"\n✅ {len(expected_failures)} Expected Failures (Successful):")
        for i, result in enumerate(expected_failures, 1):
            print(f"   {i}. Test {result['test_id']}: {result['description']}")
            print(f"      Endpoint: {result['endpoint']}")
            print(f"      Expected Status: {result['status_code']}")

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("🚀 Debug Endpoints Test Starting...")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            results = await run_debug_tests()

        summary = analyze_test_results(results)

        # Determine overall test success
        has_actual_test_errors = len(results.errors) > 0
        # Exclude expected failures from failed results count
        has_failed_results = (
            len(
                [
                    r
                    for r in results.results
                    if not r.get("success", True)
                    and not r.get("expected_failure", False)
                ]
            )
            > 0
        )
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in results.errors
        )
        has_database_errors = any(
            "no such variable" in error.get("error", "").lower()
            or "nameError" in error.get("error", "")
            or "undefined" in error.get("error", "").lower()
            for error in results.errors
        )

        test_passed = (
            not has_empty_errors
            and not has_database_errors
            and not has_actual_test_errors
            and not has_failed_results
            and summary["total_requests"] > 0
            and summary["successful_requests"] > 0
            and summary["all_endpoints_tested"]
        )

        if has_empty_errors:
            print("❌ Test failed due to empty/unknown errors")
        elif has_database_errors:
            print("❌ Test failed due to database configuration errors")
        elif has_actual_test_errors:
            print(f"❌ Test failed due to {len(results.errors)} test errors")
        elif has_failed_results:
            failed_count = len(
                [
                    r
                    for r in results.results
                    if not r.get("success", True)
                    and not r.get("expected_failure", False)
                ]
            )
            print(f"❌ Test failed due to {failed_count} unexpected failed results")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed - no successful requests")
        elif not summary["all_endpoints_tested"]:
            print(
                f"❌ Test failed - missing endpoint coverage: {', '.join(summary['missing_patterns'])}"
            )

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        import asyncio

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
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

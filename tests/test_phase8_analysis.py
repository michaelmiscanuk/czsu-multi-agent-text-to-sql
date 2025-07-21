"""Test for Phase 8.3: Analysis Routes
Tests the analysis endpoints with real HTTP requests and proper authentication.
"""

import httpx
from typing import Dict
from datetime import datetime
import time
import asyncio
import sys
import uuid

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

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 180  # 3 minutes for analysis requests (they can take longer)
TEST_EMAIL = "test_user@example.com"
REQUIRED_ENDPOINTS = {"/analyze"}
TEST_QUERIES = [
    # Basic analysis tests
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "Kolik Lidi zije v Praze?",
            "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
        },
        "description": "Basic population query",
        "should_succeed": True,
    },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "Show me the structure of available tables",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Table structure analysis",
    #     "should_succeed": True,
    # },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "List the first 5 rows from any available table",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Data query analysis",
    #     "should_succeed": True,
    # },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "Explain the available datasets",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Dataset explanation query",
    #     "should_succeed": True,
    # },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "Help me understand the data structure",
    #         "thread_id": "existing_thread_123",
    #     },
    #     "description": "Analysis with existing thread",
    #     "should_succeed": True,
    # },
    # SQL generation tests
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "Generate SQL to count rows in tables",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "SQL generation query",
    #     "should_succeed": True,
    # },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "Create a query to find unique values",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Complex SQL query request",
    #     "should_succeed": True,
    # },
    # Invalid request tests (should fail validation)
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "",
            "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
        },
        "description": "Empty prompt",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {"prompt": "Valid prompt", "thread_id": ""},
        "description": "Empty thread_id",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}"
            # Missing prompt field
        },
        "description": "Missing prompt field",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "Valid prompt"
            # Missing thread_id field
        },
        "description": "Missing thread_id field",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": None,
            "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
        },
        "description": "Null prompt value",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {"prompt": "Valid prompt", "thread_id": None},
        "description": "Null thread_id value",
        "should_succeed": False,
    },
    # Edge cases
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "A" * 9999,  # Very long prompt (just under potential limit)
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Very long prompt",
    #     "should_succeed": True,
    # },
    # {
    #     "endpoint": "/analyze",
    #     "method": "POST",
    #     "data": {
    #         "prompt": "What is the meaning of life, the universe, and everything? Please provide detailed analysis of all available data sources and their relationships while considering the philosophical implications of data analysis in the context of existential questions about the nature of reality and consciousness as it relates to database management systems and their role in answering fundamental questions about existence.",
    #         "thread_id": lambda: f"test_thread_{uuid.uuid4().hex[:8]}",
    #     },
    #     "description": "Complex philosophical query",
    #     "should_succeed": True,
    # },
]


def _validate_response_structure(endpoint: str, data: dict):
    """Validate response structure based on endpoint."""
    if endpoint == "/analyze":
        # Core required fields
        assert "prompt" in data, "Missing 'prompt' field"
        assert "result" in data, "Missing 'result' field"
        assert "thread_id" in data, "Missing 'thread_id' field"
        assert "run_id" in data, "Missing 'run_id' field"

        # Type validation
        assert isinstance(data["prompt"], str), "'prompt' must be a string"
        assert isinstance(data["result"], str), "'result' must be a string"
        assert isinstance(data["thread_id"], str), "'thread_id' must be a string"
        assert isinstance(data["run_id"], str), "'run_id' must be a string"

        # Content validation
        assert data["prompt"].strip(), "'prompt' must not be empty"
        assert data["result"].strip(), "'result' must not be empty"
        assert data["thread_id"].strip(), "'thread_id' must not be empty"
        assert data["run_id"].strip(), "'run_id' must not be empty"

        # UUID validation for run_id
        try:
            uuid.UUID(data["run_id"])
        except ValueError:
            raise AssertionError("'run_id' must be a valid UUID")

        # Optional fields validation (if present)
        if "queries_and_results" in data:
            assert isinstance(
                data["queries_and_results"], list
            ), "'queries_and_results' must be a list"

        if "top_selection_codes" in data:
            assert isinstance(
                data["top_selection_codes"], list
            ), "'top_selection_codes' must be a list"

        if "datasets_used" in data:
            assert isinstance(
                data["datasets_used"], list
            ), "'datasets_used' must be a list"

        if "sql" in data and data["sql"] is not None:
            assert isinstance(data["sql"], str), "'sql' must be a string"

        if "datasetUrl" in data and data["datasetUrl"] is not None:
            assert isinstance(data["datasetUrl"], str), "'datasetUrl' must be a string"

        if "top_chunks" in data:
            assert isinstance(data["top_chunks"], list), "'top_chunks' must be a list"

        if "iteration" in data:
            assert isinstance(data["iteration"], int), "'iteration' must be an integer"
            assert data["iteration"] >= 0, "'iteration' must be non-negative"

        if "max_iterations" in data:
            assert isinstance(
                data["max_iterations"], int
            ), "'max_iterations' must be an integer"
            assert data["max_iterations"] > 0, "'max_iterations' must be positive"

        print(f"✅ {endpoint} validation passed")


async def make_analysis_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    method: str,
    data: Dict,
    description: str,
    should_succeed: bool,
    results: BaseTestResults,
):
    """Make a request to an analysis endpoint with server traceback capture."""
    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    # Resolve any lambda functions in data
    resolved_data = {}
    for key, value in data.items():
        if callable(value):
            resolved_data[key] = value()
        else:
            resolved_data[key] = value

    start_time = time.time()
    try:
        if method.upper() == "POST":
            result = await make_request_with_traceback_capture(
                client,
                "POST",
                f"{SERVER_BASE_URL}{endpoint}",
                json=resolved_data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        else:
            result = await make_request_with_traceback_capture(
                client,
                method.upper(),
                f"{SERVER_BASE_URL}{endpoint}",
                params=resolved_data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(result)

        if result["response"] is None:
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"❌ Test {test_id} - Client Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)
            return

        response = result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.2f}s)")

        if should_succeed:
            if response.status_code == 200:
                try:
                    data_response = response.json()
                    _validate_response_structure(endpoint, data_response)

                    # Additional validation for specific test cases
                    if "prompt" in resolved_data and resolved_data["prompt"]:
                        assert (
                            data_response["prompt"] == resolved_data["prompt"]
                        ), "Prompt mismatch in response"
                    if "thread_id" in resolved_data and resolved_data["thread_id"]:
                        assert (
                            data_response["thread_id"] == resolved_data["thread_id"]
                        ), "Thread ID mismatch in response"

                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        data_response,
                        response_time,
                        response.status_code,
                    )
                except (AssertionError, Exception) as e:
                    print(f"❌ Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
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


async def run_analysis_tests() -> BaseTestResults:
    """Run all analysis endpoint tests."""
    print("🚀 Starting analysis tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Run all test cases
        for i, test_case in enumerate(TEST_QUERIES, 1):
            test_id = f"test_{i}"
            await make_analysis_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["method"],
                test_case["data"],
                test_case["description"],
                test_case["should_succeed"],
                results,
            )
            await asyncio.sleep(1.0)  # Small delay between analysis requests

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

    if not summary["all_endpoints_tested"]:
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors:
            print(f"  {error['test_id']}: {error['error']}")

    # Save traceback information if there are failures
    if results.errors or summary["failed_requests"] > 0:
        save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("🚀 Analysis Endpoints Test Starting...")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__analyze_debug="1",
                print__analysis_tracing_debug="1",
                print__feedback_flow="1",
                DEBUG_TRACEBACK="1",
            )
            results = await run_analysis_tests()
            await cleanup_debug_environment(
                client,
                print__analyze_debug="0",
                print__analysis_tracing_debug="0",
                print__feedback_flow="0",
                DEBUG_TRACEBACK="0",
            )

        summary = analyze_test_results(results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )
        has_database_errors = any(
            "no such variable" in error.get("error", "").lower()
            or "nameError" in error.get("error", "")
            or "undefined" in error.get("error", "").lower()
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors
            and not has_database_errors
            and summary["total_requests"] > 0
            and summary["all_endpoints_tested"]
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
        )

        if has_empty_errors:
            print("❌ Test failed: Server returned empty error messages")
        elif has_database_errors:
            print("❌ Test failed: Database errors detected")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded")
        elif not summary["all_endpoints_tested"]:
            print("❌ Test failed: Not all required endpoints were tested")
        elif summary["failed_requests"] > 0:
            print(f"❌ Test failed: {summary['failed_requests']} requests failed")

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

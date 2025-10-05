"""Test for Phase 8.2: Catalog Routes
Tests the catalog endpoints with real HTTP requests and proper authentication.
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
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_user@example.com"
REQUIRED_ENDPOINTS = {"/catalog", "/data-tables", "/data-table"}
TEST_QUERIES = [
    {
        "endpoint": "/catalog",
        "params": {},
        "description": "Basic catalog query",
        "should_succeed": True,
    },
    {
        "endpoint": "/catalog",
        "params": {"page": 1, "page_size": 5},
        "description": "Paginated catalog query",
        "should_succeed": True,
    },
    {
        "endpoint": "/catalog",
        "params": {"q": "test"},
        "description": "Catalog search query",
        "should_succeed": True,
    },
    {
        "endpoint": "/catalog",
        "params": {"page": 0},
        "description": "Invalid page number",
        "should_succeed": False,
    },
    {
        "endpoint": "/catalog",
        "params": {"page_size": 0},
        "description": "Invalid page size",
        "should_succeed": False,
    },
    {
        "endpoint": "/catalog",
        "params": {"page_size": 20000},
        "description": "Page size too large",
        "should_succeed": False,
    },
    {
        "endpoint": "/data-tables",
        "params": {},
        "description": "List all tables",
        "should_succeed": True,
    },
    {
        "endpoint": "/data-tables",
        "params": {"q": "test"},
        "description": "Search tables",
        "should_succeed": True,
    },
    {
        "endpoint": "/data-table",
        "params": {},
        "description": "Empty table query",
        "should_succeed": True,
    },
    {
        "endpoint": "/data-table",
        "params": {"table": "non_existent_table"},
        "description": "Invalid table query",
        "should_succeed": True,
    },
]


def _validate_response_structure(endpoint: str, data: dict):
    """Validate response structure based on endpoint."""
    if endpoint == "/catalog":
        assert "results" in data, "Missing 'results' field"
        assert "total" in data, "Missing 'total' field"
        assert "page" in data, "Missing 'page' field"
        assert "page_size" in data, "Missing 'page_size' field"
        assert isinstance(data["results"], list), "'results' must be a list"
        assert isinstance(data["total"], int), "'total' must be an integer"
        assert isinstance(data["page"], int), "'page' must be an integer"
        assert isinstance(data["page_size"], int), "'page_size' must be an integer"
        assert data["total"] >= 0, "'total' must be non-negative"
        assert data["page"] >= 1, "'page' must be >= 1"
        assert data["page_size"] >= 1, "'page_size' must be >= 1"

        for item in data["results"]:
            assert "selection_code" in item, "Missing 'selection_code' in result item"
            assert (
                "extended_description" in item
            ), "Missing 'extended_description' in result item"
            assert isinstance(
                item["selection_code"], str
            ), "'selection_code' must be a string"
            assert isinstance(
                item["extended_description"], str
            ), "'extended_description' must be a string"

        print(f"✅ {endpoint} validation passed")

    elif endpoint == "/data-tables":
        assert "tables" in data, "Missing 'tables' field"
        assert isinstance(data["tables"], list), "'tables' must be a list"

        for table in data["tables"]:
            assert "selection_code" in table, "Missing 'selection_code' in table item"
            assert (
                "short_description" in table
            ), "Missing 'short_description' in table item"
            assert isinstance(
                table["selection_code"], str
            ), "'selection_code' must be a string"
            assert isinstance(
                table["short_description"], str
            ), "'short_description' must be a string"

        print(f"✅ {endpoint} validation passed")

    elif endpoint == "/data-table":
        assert "columns" in data, "Missing 'columns' field"
        assert "rows" in data, "Missing 'rows' field"
        assert isinstance(data["columns"], list), "'columns' must be a list"
        assert isinstance(data["rows"], list), "'rows' must be a list"

        for col in data["columns"]:
            assert isinstance(col, str), f"Column name '{col}' must be a string"

        for i, row in enumerate(data["rows"]):
            assert isinstance(row, list), f"Row {i} must be a list"
            if data["columns"]:
                assert len(row) == len(
                    data["columns"]
                ), f"Row {i} has {len(row)} values but {len(data['columns'])} columns"

        print(f"✅ {endpoint} validation passed")


async def make_catalog_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    params: Dict,
    description: str,
    should_succeed: bool,
    results: BaseTestResults,
):
    """Make a request to a catalog endpoint with server traceback capture."""
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
                    data = response.json()
                    _validate_response_structure(endpoint, data)
                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        data,
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


async def run_catalog_tests() -> BaseTestResults:
    """Run all catalog endpoint tests."""
    print("🚀 Starting catalog tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Get real table names to test data-table endpoint more thoroughly
        real_table_name = None
        try:
            token = create_test_jwt_token(TEST_EMAIL)
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(
                f"{SERVER_BASE_URL}/data-tables", headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                if data["tables"]:
                    real_table_name = data["tables"][0]["selection_code"]
        except Exception:
            pass

        test_queries = TEST_QUERIES.copy()
        if real_table_name:
            test_queries.append(
                {
                    "endpoint": "/data-table",
                    "params": {"table": real_table_name},
                    "description": f"Real table query: {real_table_name}",
                    "should_succeed": True,
                }
            )

        # Run all test cases
        for i, test_case in enumerate(test_queries, 1):
            test_id = f"test_{i}"
            await make_catalog_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["params"],
                test_case["description"],
                test_case["should_succeed"],
                results,
            )
            await asyncio.sleep(0.1)  # Small delay between requests

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

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("🚀 Catalog Endpoints Test Starting...")
    print(f"🌐 Testing server at: {SERVER_BASE_URL}")

    # Provide helpful information about the server URL
    if "vercel.app/api" in SERVER_BASE_URL:
        print("📝 Using Vercel frontend proxy (routes to Render backend)")
    elif "vercel.app" in SERVER_BASE_URL:
        print("⚠️  WARNING: Direct Vercel access - consider using /api proxy")
    elif "railway.com" in SERVER_BASE_URL:
        print("📝 Direct backend server access")
    elif "localhost" in SERVER_BASE_URL:
        print("📝 Local development server")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        print(f"💡 Tips for fixing connectivity issues:")
        print(
            f"   • For Vercel proxy: Use 'https://czsu-multi-agent-text-to-sql.vercel.app/api'"
        )
        print(
            f"   • For direct backend: Use 'https://czsu-multi-agent-backend-production.up.railway.app'"
        )
        print(f"   • For local dev: Use 'http://localhost:8000'")
        print(f"   • Set environment variable: TEST_SERVER_URL=<your-url>")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__catalog_debug="1",
                print__data_tables_debug="0",
                DEBUG_TRACEBACK="1",
            )
            results = await run_catalog_tests()
            await cleanup_debug_environment(
                client,
                print__catalog_debug="0",
                print__data_tables_debug="0",
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

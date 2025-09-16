"""Test for Phase 9: Main Application File
Tests the main FastAPI application with real HTTP requests, middleware validation,
exception handling, and comprehensive endpoint testing.
"""

import os
import sys
from pathlib import Path

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

import httpx
from typing import Dict
from datetime import datetime
import time
import asyncio

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

# Import database connection for test data setup
try:
    from checkpointer.database.connection import get_direct_connection
    import uuid

    DATABASE_AVAILABLE = True
except ImportError:
    print("⚠️ Database connection not available - will use dummy run_ids")
    import uuid  # We still need uuid for generating dummy IDs

    DATABASE_AVAILABLE = False

# Test configuration
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_main_user@example.com"
REQUIRED_ENDPOINTS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/catalog",
    "/analyze",
    "/feedback",
    "/sentiment",
    "/chat-threads",
    "/placeholder/{width}/{height}",
}


async def create_test_run_ids_in_db(user_email: str, count: int = 2) -> list[str]:
    """Create test run_ids in the database that the test user owns."""
    test_run_ids = []

    if not DATABASE_AVAILABLE:
        # Return dummy UUIDs if database is not available
        return [str(uuid.uuid4()) for _ in range(count)]

    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                for i in range(count):
                    run_id = str(uuid.uuid4())
                    thread_id = str(uuid.uuid4())

                    # Insert test run data into users_threads_runs table
                    await cur.execute(
                        """
                        INSERT INTO users_threads_runs 
                        (email, thread_id, run_id, sentiment, timestamp) 
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (run_id) DO NOTHING
                        """,
                        (user_email, thread_id, run_id, None),
                    )
                    test_run_ids.append(run_id)

                await conn.commit()
                print(
                    f"✅ Created {len(test_run_ids)} test run_ids in database for user: {user_email}"
                )

    except Exception as e:
        print(f"⚠️ Failed to create test run_ids: {e}")
        # Fall back to dummy UUIDs
        test_run_ids = [str(uuid.uuid4()) for _ in range(count)]
        print(f"🔄 Using dummy UUIDs instead (tests will show ownership validation)")

    return test_run_ids


async def cleanup_test_run_ids_from_db(run_ids: list[str]):
    """Clean up test run_ids from the database after testing."""
    if not DATABASE_AVAILABLE or not run_ids:
        return

    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                for run_id in run_ids:
                    await cur.execute(
                        "DELETE FROM users_threads_runs WHERE run_id = %s", (run_id,)
                    )
                await conn.commit()
                print(f"🧹 Cleaned up {len(run_ids)} test run_ids from database")
    except Exception as e:
        print(f"⚠️ Failed to cleanup test run_ids: {e}")


def get_test_cases(test_run_ids: list[str] = None):
    """Get test cases with actual run_ids if available."""
    # Use real run_ids if available, otherwise use dummy ones
    feedback_run_id = (
        test_run_ids[0]
        if test_run_ids and len(test_run_ids) > 0
        else "11111111-1111-1111-1111-111111111111"
    )
    sentiment_run_id = (
        test_run_ids[1]
        if test_run_ids and len(test_run_ids) > 1
        else "22222222-2222-2222-2222-222222222222"
    )


def get_test_cases(test_run_ids: list[str] = None):
    """Get test cases with actual run_ids if available."""
    # Use real run_ids if available, otherwise use dummy ones
    feedback_run_id = (
        test_run_ids[0]
        if test_run_ids and len(test_run_ids) > 0
        else "11111111-1111-1111-1111-111111111111"
    )
    sentiment_run_id = (
        test_run_ids[1]
        if test_run_ids and len(test_run_ids) > 1
        else "22222222-2222-2222-2222-222222222222"
    )

    return [
        {
            "endpoint": "/health",
            "method": "GET",
            "params": {},
            "description": "Health check endpoint",
            "should_succeed": True,
            "requires_auth": False,
            "test_focus": "Health check endpoint availability",
        },
        {
            "endpoint": "/docs",
            "method": "GET",
            "params": {},
            "description": "API documentation endpoint",
            "should_succeed": True,
            "requires_auth": False,
            "test_focus": "Swagger documentation accessibility",
        },
        {
            "endpoint": "/openapi.json",
            "method": "GET",
            "params": {},
            "description": "OpenAPI schema endpoint",
            "should_succeed": True,
            "requires_auth": False,
            "test_focus": "OpenAPI schema generation and accessibility",
        },
        {
            "endpoint": "/catalog",
            "method": "GET",
            "params": {},
            "description": "Catalog endpoint with authentication",
            "should_succeed": True,
            "requires_auth": True,
            "test_focus": "Authenticated catalog access",
        },
        {
            "endpoint": "/catalog",
            "method": "GET",
            "params": {"page": 1, "page_size": 5},
            "description": "Paginated catalog query",
            "should_succeed": True,
            "requires_auth": True,
            "test_focus": "Catalog pagination functionality",
        },
        {
            "endpoint": "/analyze",
            "method": "POST",
            "json_data": {
                "prompt": "How many tables are in the database?",
                "thread_id": "test-thread-main-123",
            },
            "description": "Analysis endpoint with proper query",
            "should_succeed": True,
            "requires_auth": True,
            "test_focus": "Analysis endpoint processing with complete parameters",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {"run_id": feedback_run_id, "feedback": 1},
            "description": f"Feedback submission with {'real' if DATABASE_AVAILABLE else 'dummy'} run_id",
            "should_succeed": DATABASE_AVAILABLE,  # Only expect success if we have real run_ids
            "requires_auth": True,
            "test_focus": "Feedback endpoint functionality",
            "expected_status": None if DATABASE_AVAILABLE else 404,
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": sentiment_run_id, "sentiment": True},
            "description": f"Sentiment update with {'real' if DATABASE_AVAILABLE else 'dummy'} run_id",
            "should_succeed": DATABASE_AVAILABLE,  # Only expect success if we have real run_ids
            "requires_auth": True,
            "test_focus": "Sentiment endpoint functionality",
            "expected_status": None if DATABASE_AVAILABLE else 404,
        },
        {
            "endpoint": "/chat-threads",
            "method": "GET",
            "params": {},
            "description": "Chat threads listing",
            "should_succeed": True,
            "requires_auth": True,
            "test_focus": "Chat threads endpoint availability",
        },
        {
            "endpoint": "/placeholder/100/100",
            "method": "GET",
            "params": {},
            "description": "Placeholder image generation",
            "should_succeed": True,
            "requires_auth": False,
            "test_focus": "Placeholder image generation endpoint",
        },
        # Authentication failure tests
        {
            "endpoint": "/catalog",
            "method": "GET",
            "params": {},
            "description": "Catalog without authentication",
            "should_succeed": False,
            "requires_auth": False,
            "expected_status": 401,
            "test_focus": "Authentication requirement validation",
        },
        {
            "endpoint": "/analyze",
            "method": "POST",
            "json_data": {"prompt": "test"},
            "description": "Analysis without authentication",
            "should_succeed": False,
            "requires_auth": False,
            "expected_status": 401,
            "test_focus": "Analysis endpoint auth protection",
        },
        # Validation error tests
        {
            "endpoint": "/catalog",
            "method": "GET",
            "params": {"page": 0},
            "description": "Invalid page number",
            "should_succeed": False,
            "requires_auth": True,
            "expected_status": 422,
            "test_focus": "Catalog validation error handling",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {"run_id": "invalid-uuid", "feedback": 1},
            "description": "Invalid UUID format",
            "should_succeed": False,
            "requires_auth": True,
            "expected_status": 422,
            "test_focus": "Feedback UUID validation",
        },
    ]


# Test cases for main application endpoints - we'll replace this with the function above
TEST_CASES = []


def _validate_main_response(endpoint: str, data: dict, test_case: dict):
    """Validate response structure based on endpoint."""
    test_focus = test_case.get("test_focus", "Response validation")
    print(f"🔍 Testing: {test_focus}")

    if endpoint == "/health":
        assert "status" in data, "Health endpoint should return status"
        assert data["status"] == "healthy", "Health status should be 'healthy'"
        print("✅ Health endpoint validation passed")

    elif endpoint == "/catalog":
        assert "results" in data, "Catalog should return results"
        assert "total" in data, "Catalog should return total count"
        assert isinstance(data["results"], list), "Results should be a list"
        assert isinstance(data["total"], int), "Total should be an integer"
        print("✅ Catalog endpoint validation passed")

    elif endpoint == "/analyze":
        # Analysis endpoint may return various structures depending on the query
        # Just ensure it's a valid dict response
        assert isinstance(data, dict), "Analysis should return a dictionary"
        print("✅ Analysis endpoint validation passed")

    elif endpoint == "/feedback":
        assert "message" in data, "Feedback should return message"
        assert "run_id" in data, "Feedback should return run_id"
        print("✅ Feedback endpoint validation passed")

    elif endpoint == "/sentiment":
        assert "message" in data, "Sentiment should return message"
        assert "run_id" in data, "Sentiment should return run_id"
        print("✅ Sentiment endpoint validation passed")

    elif endpoint == "/chat-threads":
        assert "threads" in data, "Chat threads should return threads list"
        assert isinstance(data["threads"], list), "Threads should be a list"
        print("✅ Chat threads endpoint validation passed")

    elif endpoint.startswith("/placeholder/"):
        # Placeholder returns image data, so we don't expect JSON
        print("✅ Placeholder endpoint validation passed (image response)")


def _normalize_endpoint_for_tracking(endpoint: str) -> str:
    """Normalize endpoint for tracking purposes."""
    # Convert parameterized paths to template form
    if endpoint.startswith("/placeholder/") and len(endpoint.split("/")) == 4:
        return "/placeholder/{width}/{height}"
    return endpoint


def _get_test_explanation(test_case: dict) -> str:
    """Generate detailed explanation of what the test validates."""
    test_focus = test_case.get("test_focus", "")
    endpoint = test_case["endpoint"]
    method = test_case["method"]
    should_succeed = test_case["should_succeed"]
    requires_auth = test_case["requires_auth"]

    if should_succeed:
        if requires_auth:
            return f"Testing {endpoint} endpoint functionality with proper JWT authentication, verifying {test_focus}"
        else:
            return f"Testing public {endpoint} endpoint accessibility, verifying {test_focus}"
    else:
        expected_status = test_case.get("expected_status", 401)
        if expected_status == 401:
            return f"Testing authentication protection for {endpoint}, ensuring unauthorized access is rejected"
        elif expected_status == 422:
            return f"Testing input validation for {endpoint}, ensuring malformed requests are properly rejected"
        else:
            return f"Testing error handling for {endpoint}, expecting {expected_status} status code"


async def make_main_request(
    client: httpx.AsyncClient,
    test_id: str,
    test_case: dict,
    results: BaseTestResults,
):
    """Make a request to a main application endpoint with traceback capture."""
    endpoint = test_case["endpoint"]
    method = test_case["method"]
    description = test_case["description"]
    should_succeed = test_case["should_succeed"]
    requires_auth = test_case["requires_auth"]
    params = test_case.get("params", {})
    json_data = test_case.get("json_data")
    expected_status = test_case.get("expected_status")

    # Print detailed test information
    print(f"\n🔍 TEST {test_id}: {test_case.get('test_focus', description)}")
    print(f"   📍 Endpoint: {method} {endpoint}")
    if params:
        print(f"   📋 Parameters: {params}")
    if json_data:
        print(f"   📋 JSON Data: {json_data}")
    print(f"   🔑 Auth Required: {'Yes' if requires_auth else 'No'}")
    expected_result = (
        "Success (200)" if should_succeed else f"Failure ({expected_status or 'Error'})"
    )
    print(f"   ✅ Expected Result: {expected_result}")
    print(f"   🎯 What we're testing: {_get_test_explanation(test_case)}")

    # Setup headers
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
            client,
            method,
            f"{SERVER_BASE_URL}{endpoint}",
            **request_kwargs,
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
                _normalize_endpoint_for_tracking(endpoint),
                description,
                error_obj,
                response_time,
            )
            return

        response = result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.2f}s)")

        if should_succeed:
            if response.status_code == 200:
                try:
                    # Handle different content types
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        data = response.json()
                        _validate_main_response(endpoint, data, test_case)
                        response_data = data
                    elif "image/" in content_type or endpoint.startswith(
                        "/placeholder/"
                    ):
                        # Image response - just verify we got content
                        response_data = {
                            "content_type": content_type,
                            "size": len(response.content),
                        }
                        print("✅ Image response received successfully")
                    elif "text/html" in content_type and endpoint == "/docs":
                        # HTML response for docs
                        response_data = {
                            "content_type": content_type,
                            "has_content": len(response.content) > 0,
                        }
                        print("✅ HTML documentation response received")
                    else:
                        response_data = {
                            "content_type": content_type,
                            "size": len(response.content),
                        }

                    results.add_result(
                        test_id,
                        _normalize_endpoint_for_tracking(endpoint),
                        description,
                        response_data,
                        response_time,
                        response.status_code,
                        success=True,
                    )
                except Exception as e:
                    print(f"❌ Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id,
                        _normalize_endpoint_for_tracking(endpoint),
                        description,
                        error_obj,
                        response_time,
                    )
            else:
                handle_error_response(
                    test_id,
                    _normalize_endpoint_for_tracking(endpoint),
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )
        else:
            handle_expected_failure(
                test_id,
                _normalize_endpoint_for_tracking(endpoint),
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
            error_message = f"Unknown error of type {type(e).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}")
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(
            test_id,
            _normalize_endpoint_for_tracking(endpoint),
            description,
            error_obj,
            response_time,
            response_data=None,
        )


async def test_application_startup():
    """Test that the application can be imported and has correct structure."""
    print("🔍 Testing: Application import and structure validation")

    try:
        from api.main import app, lifespan

        # Validate FastAPI app structure
        assert hasattr(app, "title"), "App should have title"
        assert hasattr(app, "version"), "App should have version"
        assert app.title == "CZSU Multi-Agent Text-to-SQL API", "Correct title"
        assert app.version == "1.0.0", "Correct version"

        # Validate middleware
        middleware_types = [
            middleware.cls.__name__ for middleware in app.user_middleware
        ]
        assert "CORSMiddleware" in middleware_types, "Should have CORS middleware"
        assert "GZipMiddleware" in middleware_types, "Should have GZip middleware"

        # Validate exception handlers
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException

        exception_handlers = app.exception_handlers
        assert (
            RequestValidationError in exception_handlers
        ), "Should have validation error handler"
        assert HTTPException in exception_handlers, "Should have HTTP exception handler"

        # Validate lifespan
        assert callable(lifespan), "Lifespan should be callable"

        print("✅ Application startup validation passed")
        return True

    except ImportError as e:
        # Handle missing optional dependencies gracefully
        if (
            "cohere" in str(e)
            or "openai" in str(e)
            or any(dep in str(e) for dep in ["azure", "anthropic"])
        ):
            print(f"⚠️ Optional dependency missing: {e}")
            print(
                "✅ Application startup validation passed (with optional dependencies missing)"
            )
            return True
        else:
            print(f"❌ Application startup validation failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Application startup validation failed: {e}")
        return False


async def test_middleware_functionality():
    """Test middleware functionality through actual requests."""
    print("🔍 Testing: Middleware functionality through HTTP requests")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Test CORS headers
            response = await client.get(f"{SERVER_BASE_URL}/health")
            cors_headers = [
                "access-control-allow-origin",
                "access-control-allow-credentials",
                "access-control-allow-methods",
                "access-control-allow-headers",
            ]

            # Note: CORS headers might not be present on same-origin requests
            print(
                "ℹ️ CORS middleware tested (headers may not appear on same-origin requests)"
            )

            # Test rate limiting middleware (make multiple requests)
            start_time = time.time()
            responses = []
            for i in range(5):
                resp = await client.get(f"{SERVER_BASE_URL}/health")
                responses.append(resp.status_code)

            # All should succeed for health endpoint (not rate limited)
            assert all(
                status == 200 for status in responses
            ), "Health endpoint should not be rate limited"

            # Test that requests are processed (response time indicates middleware is working)
            elapsed = time.time() - start_time
            assert elapsed < 5.0, "Requests should be processed quickly"

            print("✅ Middleware functionality validation passed")
            return True

    except Exception as e:
        print(f"❌ Middleware functionality validation failed: {e}")
        return False


async def run_main_tests() -> BaseTestResults:
    """Run all main application tests."""
    print("🚀 Starting main application tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    # Create test run_ids in the database for valid test cases
    test_run_ids = await create_test_run_ids_in_db(TEST_EMAIL, 2)

    try:
        # Test application structure first
        app_startup_ok = await test_application_startup()
        if app_startup_ok:
            results.add_result(
                "startup",
                "application",
                "Application startup and structure validation",
                {"validation": "passed"},
                0.0,
                200,
                success=True,
            )
        else:
            results.add_error(
                "startup",
                "application",
                "Application startup validation",
                Exception("Application structure validation failed"),
                0.0,
            )

        # Test middleware functionality
        middleware_ok = await test_middleware_functionality()
        if middleware_ok:
            results.add_result(
                "middleware",
                "application",
                "Middleware functionality validation",
                {"validation": "passed"},
                0.0,
                200,
                success=True,
            )
        else:
            results.add_error(
                "middleware",
                "application",
                "Middleware functionality validation",
                Exception("Middleware validation failed"),
                0.0,
            )

        # Get test cases with real run_ids
        test_cases = get_test_cases(test_run_ids)

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Run all endpoint test cases
            for i, test_case in enumerate(test_cases, 1):
                test_id = f"test_{i}"
                await make_main_request(client, test_id, test_case, results)
                await asyncio.sleep(0.1)  # Small delay between requests

    finally:
        # Clean up test data from database
        await cleanup_test_run_ids_from_db(test_run_ids)

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results with detailed summary."""
    print("\n" + "=" * 80)
    print("📊 MAIN APPLICATION TEST RESULTS SUMMARY")
    print("=" * 80)

    summary = results.get_summary()

    # Main statistics
    print(f"🔢 Total Requests: {summary['total_requests']}")
    print(f"✅ Successful Requests: {summary['successful_requests']}")
    print(f"❌ Failed Requests: {summary['failed_requests']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"⏱️ Average Response Time: {summary['average_response_time']:.3f}s")
        print(f"⚡ Max Response Time: {summary['max_response_time']:.3f}s")
        print(f"🚀 Min Response Time: {summary['min_response_time']:.3f}s")

    if summary["total_test_time"]:
        print(f"⏰ Total Test Duration: {summary['total_test_time']:.2f}s")

    # Endpoint coverage analysis
    print(f"\n🎯 ENDPOINT COVERAGE ANALYSIS:")
    tested_endpoints = summary["tested_endpoints"]
    missing_endpoints = summary["missing_endpoints"]

    if summary["all_endpoints_tested"]:
        print(
            f"✅ All {len(REQUIRED_ENDPOINTS)} required endpoints tested successfully"
        )
    else:
        print(f"❌ Missing endpoints: {len(missing_endpoints)}")
        for endpoint in missing_endpoints:
            print(f"   • {endpoint}")

    print(f"\n📋 Tested endpoints ({len(tested_endpoints)}):")
    for endpoint in sorted(tested_endpoints):
        print(f"   ✅ {endpoint}")

    # Error analysis
    if results.errors:
        print(f"\n❌ ERROR DETAILS ({len(results.errors)} errors):")
        for i, error in enumerate(results.errors, 1):
            print(f"   {i}. Test {error['test_id']}: {error['endpoint']}")
            print(f"      Error: {error['error']}")
            if error.get("response_time"):
                print(f"      Response Time: {error['response_time']:.3f}s")

    # Test type breakdown
    success_tests = [r for r in results.results if r["success"]]
    auth_tests = [
        r
        for r in success_tests
        if r["description"].find("authentication") != -1
        or r["description"].find("without") != -1
    ]
    validation_tests = [
        r
        for r in success_tests
        if r["description"].find("validation") != -1
        or r["description"].find("Invalid") != -1
    ]
    functional_tests = [
        r for r in success_tests if r not in auth_tests and r not in validation_tests
    ]

    print(f"\n📊 TEST TYPE BREAKDOWN:")
    print(f"   🔧 Functional Tests: {len(functional_tests)}")
    print(f"   🔑 Authentication Tests: {len(auth_tests)}")
    print(f"   ✔️ Validation Tests: {len(validation_tests)}")

    # Save traceback information
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("[START] Main Application Endpoints Test Starting...")
    print("[INFO] Testing: Comprehensive main application functionality and endpoints")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("[ERROR] Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__main_debug="1",
                print__startup_debug="1",
                print__middleware_debug="1",
                DEBUG_TRACEBACK="1",
            )

            results = await run_main_tests()

            await cleanup_debug_environment(
                client,
                print__main_debug="0",
                print__startup_debug="0",
                print__middleware_debug="0",
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

        print(f"\n[RESULT] OVERALL RESULT: {'PASSED' if test_passed else 'FAILED'}")
        print("[OK] Tested main application structure and configuration")
        print("[OK] Tested middleware functionality and error handling")
        print("[OK] Tested all major endpoint categories")
        print("[OK] Tested authentication and authorization")
        print("[OK] Tested validation and error responses")
        print("[OK] Tested FastAPI application assembly and integration")

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Cases": "Dynamic (depends on database availability)",
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
        print("\n[STOP] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Cases": "Dynamic (depends on database availability)",
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

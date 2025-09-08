"""
Test for Phase 6: Exception Handlers
Tests exception handler functions following the established project patterns.
"""

import os
import sys
import time
import traceback
import json
import asyncio
from typing import Dict, Any
from datetime import datetime

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(BASE_DIR))
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    sys.path.insert(0, str(BASE_DIR))

# Import test helpers following Phase 8 patterns
from tests.helpers import (
    BaseTestResults,
    save_traceback_report,
)

# Test imports from extracted modules
try:
    from api.exceptions.handlers import (
        general_exception_handler,
        http_exception_handler,
        validation_exception_handler,
        value_error_handler,
    )
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    print("✅ Successfully imported exception handlers")
except Exception as e:
    print(f"❌ Failed to import exception handlers: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)

# Test configuration following Phase 8 patterns
REQUIRED_HANDLERS = {
    "validation_exception_handler",
    "http_exception_handler",
    "value_error_handler",
    "general_exception_handler",
}

EXCEPTION_VALIDATION_TESTS = [
    {
        "test_id": "EXC_001",
        "handler": "validation_exception_handler",
        "description": "RequestValidationError with field errors",
        "exception_type": "RequestValidationError",
        "exception_data": [
            {"loc": ["field1"], "msg": "field required", "type": "value_error.missing"},
            {"loc": ["field2"], "msg": "invalid value", "type": "value_error"},
        ],
        "expected_status": 422,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_002",
        "handler": "http_exception_handler",
        "description": "HTTP 401 Unauthorized error",
        "exception_type": "StarletteHTTPException",
        "exception_data": {"status_code": 401, "detail": "Authentication failed"},
        "expected_status": 401,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_003",
        "handler": "http_exception_handler",
        "description": "HTTP 404 Not Found error",
        "exception_type": "StarletteHTTPException",
        "exception_data": {"status_code": 404, "detail": "Resource not found"},
        "expected_status": 404,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_004",
        "handler": "http_exception_handler",
        "description": "HTTP 500 Internal Server Error",
        "exception_type": "StarletteHTTPException",
        "exception_data": {"status_code": 500, "detail": "Internal server error"},
        "expected_status": 500,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_005",
        "handler": "value_error_handler",
        "description": "ValueError with custom message",
        "exception_type": "ValueError",
        "exception_data": "Invalid input value provided",
        "expected_status": 400,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_006",
        "handler": "value_error_handler",
        "description": "ValueError with empty message",
        "exception_type": "ValueError",
        "exception_data": "",
        "expected_status": 400,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_007",
        "handler": "general_exception_handler",
        "description": "RuntimeError exception",
        "exception_type": "RuntimeError",
        "exception_data": "Unexpected runtime error occurred",
        "expected_status": 500,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_008",
        "handler": "general_exception_handler",
        "description": "Generic Exception",
        "exception_type": "Exception",
        "exception_data": "Generic error message",
        "expected_status": 500,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_009",
        "handler": "http_exception_handler",
        "description": "HTTP 200 OK (edge case)",
        "exception_type": "StarletteHTTPException",
        "exception_data": {"status_code": 200, "detail": "Success"},
        "expected_status": 200,
        "should_succeed": True,
    },
    {
        "test_id": "EXC_010",
        "handler": "validation_exception_handler",
        "description": "RequestValidationError with empty errors",
        "exception_type": "RequestValidationError",
        "exception_data": [],
        "expected_status": 422,
        "should_succeed": True,
    },
]


class MockRequest:
    """Mock request object for testing exception handlers."""

    def __init__(
        self,
        url="http://test.example.com/test",
        method="GET",
        client_host="127.0.0.1",
        headers=None,
    ):
        self.url = MockURL(url)
        self.method = method
        self.client = MockClient(client_host)
        self.headers = headers or {}


class MockURL:
    """Mock URL object."""

    def __init__(self, url):
        self._url = url

    def __str__(self):
        return self._url


class MockClient:
    """Mock client object."""

    def __init__(self, host):
        self.host = host


class MockValidationError(RequestValidationError):
    """Mock validation error for testing."""

    def __init__(self, errors):
        self._errors = errors

    def errors(self):
        return self._errors


def create_test_exception(exception_type: str, exception_data: Any):
    """Create a test exception based on type and data."""
    if exception_type == "RequestValidationError":
        return MockValidationError(exception_data)
    elif exception_type == "StarletteHTTPException":
        return StarletteHTTPException(
            status_code=exception_data["status_code"], detail=exception_data["detail"]
        )
    elif exception_type == "ValueError":
        return ValueError(exception_data)
    elif exception_type == "RuntimeError":
        return RuntimeError(exception_data)
    elif exception_type == "Exception":
        return Exception(exception_data)
    else:
        raise ValueError(f"Unknown exception type: {exception_type}")


async def validate_exception_handler(
    test_id: str,
    handler_name: str,
    description: str,
    test_case: Dict[str, Any],
    should_succeed: bool,
    results: BaseTestResults,
):
    """Validate an exception handler with error tracking."""
    print(f"\n🔍 Test {test_id}: {description}")
    print(f"   📋 Testing: {handler_name}()")
    print(f"   🎯 Expected: {'SUCCESS' if should_succeed else 'FAILURE'}")

    start_time = time.time()

    try:
        # Create mock request
        mock_request = MockRequest(url=f"http://test.example.com/{test_id.lower()}")

        # Create test exception
        exception_type = test_case.get("exception_type")
        exception_data = test_case.get("exception_data")
        test_exception = create_test_exception(exception_type, exception_data)

        # Show what we're testing with
        if exception_type == "RequestValidationError":
            print(
                f"   📥 Input: RequestValidationError with {len(exception_data)} errors"
            )
        elif exception_type == "StarletteHTTPException":
            print(
                f"   📥 Input: HTTP {exception_data['status_code']} - '{exception_data['detail']}'"
            )
        elif exception_type in ["ValueError", "RuntimeError", "Exception"]:
            print(f"   📥 Input: {exception_type}('{exception_data}')")

        # Get the handler function
        handler_func = None
        if handler_name == "validation_exception_handler":
            handler_func = validation_exception_handler
        elif handler_name == "http_exception_handler":
            handler_func = http_exception_handler
        elif handler_name == "value_error_handler":
            handler_func = value_error_handler
        elif handler_name == "general_exception_handler":
            handler_func = general_exception_handler
        else:
            raise ValueError(f"Unknown handler: {handler_name}")

        # Call the handler
        print(f"   🔧 Calling: {handler_name}(request, exception)")
        response = await handler_func(mock_request, test_exception)

        response_time = time.time() - start_time
        expected_status = test_case.get("expected_status", 500)

        # Validate response
        assert isinstance(
            response, JSONResponse
        ), f"Response should be JSONResponse, got {type(response)}"
        assert (
            response.status_code == expected_status
        ), f"Expected status {expected_status}, got {response.status_code}"

        # Parse response content
        try:
            content = json.loads(response.body)
        except:
            content = {"body": str(response.body)}

        if should_succeed:
            print(f"   ✅ RESULT: SUCCESS ({response_time:.3f}s)")
            print(
                f"   📤 Output: HTTP {response.status_code} - {content.get('detail', 'No detail')}"
            )

            # Additional validations based on handler type
            if handler_name == "validation_exception_handler":
                assert "detail" in content, "Response should contain 'detail'"
                assert "errors" in content, "Response should contain 'errors'"
                assert (
                    content["detail"] == "Validation error"
                ), "Detail should be 'Validation error'"
            elif handler_name in [
                "http_exception_handler",
                "value_error_handler",
                "general_exception_handler",
            ]:
                assert "detail" in content, "Response should contain 'detail'"

            results.add_result(
                test_id,
                handler_name,
                description,
                {"status_code": response.status_code, "content": content},
                response_time,
                200,  # Mark as success
            )
        else:
            print(f"   ❌ UNEXPECTED: Expected failure but got success")
            print(f"   📤 Output: HTTP {response.status_code} - {content}")
            results.add_error(
                test_id,
                handler_name,
                description,
                Exception(f"Expected failure but handler succeeded with: {content}"),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        if should_succeed:
            print(f"   ❌ UNEXPECTED: Handler failed with error: {str(e)}")
            print(f"   📤 Output: {type(e).__name__} - {str(e)}")
            results.add_error(test_id, handler_name, description, e, response_time)
        else:
            print(f"   ✅ RESULT: Expected failure with error ({response_time:.3f}s)")
            print(f"   📤 Output: {type(e).__name__} - {str(e)}")
            results.add_result(
                test_id,
                handler_name,
                description,
                {"error_type": type(e).__name__, "error_message": str(e)},
                response_time,
                200,  # Mark as success for expected failure
            )


async def run_exception_validation_tests() -> BaseTestResults:
    """Run all exception handler validation tests following Phase 8 patterns."""
    print("🚀 Starting exception handler validation tests...")

    # Show what we're going to test
    print("\n📋 Test Plan Overview:")
    print("=" * 70)

    validation_tests = [
        t
        for t in EXCEPTION_VALIDATION_TESTS
        if t["handler"] == "validation_exception_handler"
    ]
    http_tests = [
        t
        for t in EXCEPTION_VALIDATION_TESTS
        if t["handler"] == "http_exception_handler"
    ]
    value_tests = [
        t for t in EXCEPTION_VALIDATION_TESTS if t["handler"] == "value_error_handler"
    ]
    general_tests = [
        t
        for t in EXCEPTION_VALIDATION_TESTS
        if t["handler"] == "general_exception_handler"
    ]

    print(f"📋 validation_exception_handler() - {len(validation_tests)} tests:")
    for test in validation_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        print(f"   • {test['test_id']}: {test['description']} → {status}")

    print(f"\n🌐 http_exception_handler() - {len(http_tests)} tests:")
    for test in http_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        print(f"   • {test['test_id']}: {test['description']} → {status}")

    print(f"\n❗ value_error_handler() - {len(value_tests)} tests:")
    for test in value_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        print(f"   • {test['test_id']}: {test['description']} → {status}")

    print(f"\n⚠️ general_exception_handler() - {len(general_tests)} tests:")
    for test in general_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        print(f"   • {test['test_id']}: {test['description']} → {status}")

    print("=" * 70)

    results = BaseTestResults(required_endpoints=REQUIRED_HANDLERS)
    results.start_time = datetime.now()

    # Run all test cases
    for test_case in EXCEPTION_VALIDATION_TESTS:
        await validate_exception_handler(
            test_case["test_id"],
            test_case["handler"],
            test_case["description"],
            test_case,
            test_case["should_succeed"],
            results,
        )

    results.end_time = datetime.now()
    return results


def analyze_exception_test_results(results: BaseTestResults):
    """Analyze and print exception handler test results following Phase 8 patterns."""
    print("\n📊 Exception Handler Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.3f}s")

    # Check handler coverage
    tested_handlers = set()
    for test_case in EXCEPTION_VALIDATION_TESTS:
        tested_handlers.add(test_case["handler"])

    missing_handlers = REQUIRED_HANDLERS - tested_handlers
    if missing_handlers:
        print(f"❌ Missing handler tests: {', '.join(missing_handlers)}")
    else:
        print(f"✅ All required handlers tested: {', '.join(REQUIRED_HANDLERS)}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors[:5]:  # Show first 5 errors
            print(
                f"  - Test {error.get('test_id', 'Unknown')}: {error.get('error', 'Unknown error')}"
            )
        if len(results.errors) > 5:
            print(f"  ... and {len(results.errors) - 5} more errors")

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function following Phase 8 patterns."""
    print("🚀 Phase 6 Exception Handler Tests Starting...")
    print(f"📂 BASE_DIR: {BASE_DIR}")
    print("=" * 80)

    try:
        # Run exception handler validation tests
        results = await run_exception_validation_tests()

        # Analyze results
        summary = analyze_exception_test_results(results)

        # Determine overall test success
        test_passed = (
            summary["total_requests"] > 0
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
            and len(
                REQUIRED_HANDLERS
                - {test["handler"] for test in EXCEPTION_VALIDATION_TESTS}
            )
            == 0
        )

        if summary["total_requests"] == 0:
            print("❌ No tests were executed")
            test_passed = False
        elif summary["successful_requests"] == 0:
            print("❌ All tests failed")
            test_passed = False

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "BASE_DIR": str(BASE_DIR),
            "Total Test Cases": len(EXCEPTION_VALIDATION_TESTS),
            "Error Location": "main() function",
            "Error During": "Exception handler testing",
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
        save_traceback_report(report_type="exception", exception=e)
        sys.exit(1)

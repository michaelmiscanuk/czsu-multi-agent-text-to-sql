"""
Test for Phase 6: Extract Exception Handlers
Based on test_concurrency.py pattern - imports functionality from main scripts
"""

import os

# CRITICAL: Set Windows event loop policy FIRST, before other imports
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add project root to path
sys.path.insert(0, str(BASE_DIR))

# Test imports from extracted modules
try:
    from api.exceptions.handlers import (
        general_exception_handler,
        http_exception_handler,
        validation_exception_handler,
        value_error_handler,
    )

    print("✅ Successfully imported exception handlers")
except Exception as e:
    print(f"❌ Failed to import exception handlers: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)


def print_test_status(message: str):
    """Print test status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


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


def test_phase6_exception_imports():
    """Test that exception handler modules can be imported successfully."""
    print_test_status("🔍 Testing Phase 6 exception handler imports...")

    try:
        # Test individual function imports
        from api.exceptions.handlers import validation_exception_handler

        assert callable(
            validation_exception_handler
        ), "validation_exception_handler should be callable"
        print_test_status("✅ validation_exception_handler imported successfully")

        from api.exceptions.handlers import http_exception_handler

        assert callable(
            http_exception_handler
        ), "http_exception_handler should be callable"
        print_test_status("✅ http_exception_handler imported successfully")

        from api.exceptions.handlers import value_error_handler

        assert callable(value_error_handler), "value_error_handler should be callable"
        print_test_status("✅ value_error_handler imported successfully")

        from api.exceptions.handlers import general_exception_handler

        assert callable(
            general_exception_handler
        ), "general_exception_handler should be callable"
        print_test_status("✅ general_exception_handler imported successfully")

        print_test_status("✅ Phase 6 exception handler imports test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Phase 6 exception handler imports test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_validation_exception_handler():
    """Test the validation exception handler."""
    print_test_status("🔍 Testing validation exception handler...")

    try:
        # Create a mock validation error
        class MockValidationError(RequestValidationError):
            def __init__(self, errors):
                self._errors = errors

            def errors(self):
                return self._errors

        # Test with mock request and validation error
        mock_request = MockRequest()
        mock_error = MockValidationError(
            [
                {
                    "loc": ["field1"],
                    "msg": "field required",
                    "type": "value_error.missing",
                },
                {"loc": ["field2"], "msg": "invalid value", "type": "value_error"},
            ]
        )

        response = await validation_exception_handler(mock_request, mock_error)

        # Verify response
        assert isinstance(response, JSONResponse), "Response should be JSONResponse"
        assert (
            response.status_code == 422
        ), f"Expected status 422, got {response.status_code}"

        # Check response content
        content = json.loads(response.body)
        assert "detail" in content, "Response should contain 'detail'"
        assert "errors" in content, "Response should contain 'errors'"
        assert (
            content["detail"] == "Validation error"
        ), "Detail should be 'Validation error'"

        print_test_status("✅ Validation exception handler test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Validation exception handler test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_http_exception_handler():
    """Test the HTTP exception handler."""
    print_test_status("🔍 Testing HTTP exception handler...")

    try:
        # Test 401 Unauthorized
        mock_request = MockRequest()
        auth_error = StarletteHTTPException(
            status_code=401, detail="Authentication failed"
        )

        response = await http_exception_handler(mock_request, auth_error)

        # Verify response
        assert isinstance(response, JSONResponse), "Response should be JSONResponse"
        assert (
            response.status_code == 401
        ), f"Expected status 401, got {response.status_code}"

        content = json.loads(response.body)
        assert (
            content["detail"] == "Authentication failed"
        ), "Detail should match original error"

        # Test other 4xx error
        not_found_error = StarletteHTTPException(status_code=404, detail="Not found")
        response = await http_exception_handler(mock_request, not_found_error)

        assert (
            response.status_code == 404
        ), f"Expected status 404, got {response.status_code}"
        content = json.loads(response.body)
        assert content["detail"] == "Not found", "Detail should match original error"

        # Test non-error status code (should not trigger debug prints)
        success_error = StarletteHTTPException(status_code=200, detail="OK")
        response = await http_exception_handler(mock_request, success_error)

        assert (
            response.status_code == 200
        ), f"Expected status 200, got {response.status_code}"

        print_test_status("✅ HTTP exception handler test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ HTTP exception handler test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_value_error_handler():
    """Test the ValueError exception handler."""
    print_test_status("🔍 Testing ValueError exception handler...")

    try:
        mock_request = MockRequest()
        value_error = ValueError("Invalid input value")

        response = await value_error_handler(mock_request, value_error)

        # Verify response
        assert isinstance(response, JSONResponse), "Response should be JSONResponse"
        assert (
            response.status_code == 400
        ), f"Expected status 400, got {response.status_code}"

        content = json.loads(response.body)
        assert (
            content["detail"] == "Invalid input value"
        ), "Detail should match original error message"

        print_test_status("✅ ValueError exception handler test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ ValueError exception handler test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_general_exception_handler():
    """Test the general exception handler."""
    print_test_status("🔍 Testing general exception handler...")

    try:
        mock_request = MockRequest()
        general_error = RuntimeError("Unexpected runtime error")

        response = await general_exception_handler(mock_request, general_error)

        # Verify response
        assert isinstance(response, JSONResponse), "Response should be JSONResponse"
        assert (
            response.status_code == 500
        ), f"Expected status 500, got {response.status_code}"

        content = json.loads(response.body)
        assert (
            content["detail"] == "Internal server error"
        ), "Detail should be generic error message"

        print_test_status("✅ General exception handler test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ General exception handler test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_exception_handler_integration():
    """Test that exception handlers work with proper imports and dependencies."""
    print_test_status("🔍 Testing exception handler integration...")

    try:
        # Test that debug functions are properly imported and accessible
        from api.utils.debug import (
            print__analysis_tracing_debug,
            print__analyze_debug,
            print__debug,
        )

        # Verify the handlers can access debug functions
        mock_request = MockRequest(url="http://test.example.com/integration-test")

        # Test with 401 to trigger debug functions
        auth_error = StarletteHTTPException(
            status_code=401, detail="Integration test error"
        )
        response = await http_exception_handler(mock_request, auth_error)

        assert response.status_code == 401, "Should handle 401 errors correctly"

        # Test with ValueError to trigger debug function
        value_error = ValueError("Integration test value error")
        response = await value_error_handler(mock_request, value_error)

        assert response.status_code == 400, "Should handle ValueError correctly"

        print_test_status("✅ Exception handler integration test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Exception handler integration test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def main():
    """Run all Phase 6 exception handler tests."""
    print_test_status("🚀 Starting Phase 6 Exception Handler Tests")
    print_test_status(f"📂 BASE_DIR: {BASE_DIR}")
    print_test_status("=" * 80)

    all_tests_passed = True

    # Run all tests
    tests = [
        ("Exception Handler Imports", test_phase6_exception_imports),
        ("Validation Exception Handler", test_validation_exception_handler),
        ("HTTP Exception Handler", test_http_exception_handler),
        ("ValueError Exception Handler", test_value_error_handler),
        ("General Exception Handler", test_general_exception_handler),
        ("Exception Handler Integration", test_exception_handler_integration),
    ]

    for test_name, test_func in tests:
        print_test_status(f"\n📋 Running test: {test_name}")
        print_test_status("-" * 60)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if not result:
                all_tests_passed = False
        except Exception as e:
            print_test_status(f"❌ Test {test_name} crashed: {e}")
            print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
            all_tests_passed = False

    # Final summary
    print_test_status("=" * 80)
    if all_tests_passed:
        print_test_status("🎉 ALL PHASE 6 EXCEPTION HANDLER TESTS PASSED!")
        print_test_status("✅ Exception handler extraction successful")
        print_test_status("✅ All exception handlers working correctly")
        print_test_status("✅ Debug function integration working correctly")
        print_test_status("✅ Response formatting working correctly")
    else:
        print_test_status("❌ SOME PHASE 6 EXCEPTION HANDLER TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

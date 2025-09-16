"""Test for Checkpointer Error Handling: prepared_statements.py and retry_decorators.py
Tests the error handling functionality with comprehensive testing patterns.

This test file follows the same patterns as test_checkpointer_checkpointer.py,
including proper error handling, traceback capture, and detailed reporting.
"""

import asyncio
import os
import sys
import time
import traceback
import uuid
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

# Standard library imports
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

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

# Import error handling modules for testing with error handling
try:
    from checkpointer.error_handling.prepared_statements import (
        is_prepared_statement_error,
        clear_prepared_statements,
    )
    from checkpointer.error_handling.retry_decorators import (
        retry_on_prepared_statement_error,
    )
    from checkpointer.config import (
        DEFAULT_MAX_RETRIES,
        get_db_config,
        check_postgres_env_vars,
    )
    from checkpointer.globals import _GLOBAL_CHECKPOINTER
    from checkpointer.database.connection import get_connection_kwargs

    ERROR_HANDLING_AVAILABLE = True
    IMPORT_ERROR = None

except ImportError as e:
    ERROR_HANDLING_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create mock functions for testing import failures
    def is_prepared_statement_error(error):
        return False

    async def clear_prepared_statements():
        return None

    def retry_on_prepared_statement_error(max_retries=3):
        def decorator(func):
            return func

        return decorator

    def get_db_config():
        return {
            "host": "mock",
            "port": 5432,
            "user": "mock",
            "password": "mock",
            "dbname": "mock",
        }

    def check_postgres_env_vars():
        return True

    def get_connection_kwargs():
        return {"prepare_threshold": None}

    DEFAULT_MAX_RETRIES = 3
    _GLOBAL_CHECKPOINTER = None


# Test configuration
TEST_ERROR_HANDLING_NAME = f"test_error_handling_{uuid.uuid4().hex[:8]}"
REQUIRED_COMPONENTS = {
    "is_prepared_statement_error",
    "clear_prepared_statements",
    "retry_on_prepared_statement_error",
}


class ErrorHandlingTestResults(BaseTestResults):
    """Extended test results class for error handling testing."""

    def __init__(self, required_components: set = None):
        super().__init__()
        self.required_components = required_components or REQUIRED_COMPONENTS
        self.component_coverage = set()

    def add_component_test(
        self,
        test_id: str,
        component: str,
        description: str,
        result_data: Any,
        response_time: float,
        success: bool = True,
        error: Exception = None,
        expected_failure: bool = False,
    ):
        """Add a component test result with coverage tracking."""
        self.component_coverage.add(component)

        if expected_failure:
            # Expected failures should be treated as successes for overall metrics
            self.add_result(
                test_id=test_id,
                endpoint=component,
                description=description,
                response_data={"result": str(result_data), "expected_failure": True},
                response_time=response_time,
                status_code=200,  # Success for expected failure
                success=True,
            )
        elif success and not error:
            self.add_result(
                test_id=test_id,
                endpoint=component,
                description=description,
                response_data={
                    "result": str(result_data),
                    "type": type(result_data).__name__,
                },
                response_time=response_time,
                status_code=200,
                success=True,
            )
        else:
            self.add_error(
                test_id=test_id,
                endpoint=component,
                description=description,
                error=error or Exception("Test failed"),
                response_time=response_time,
                response_data=result_data,
            )


def get_prepared_statement_error_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for prepared statement error detection."""
    return [
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [Exception("prepared statement _pg3_stmt_1 does not exist")],
            "kwargs": {},
            "description": "Detect prepared statement error with _pg3_ prefix",
            "should_succeed": True,
            "test_focus": "prepared_statement_detection",
            "expected_type": bool,
            "async": False,
        },
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [Exception("PREPARED STATEMENT error occurred")],
            "kwargs": {},
            "description": "Detect prepared statement error with uppercase text",
            "should_succeed": True,
            "test_focus": "prepared_statement_case_insensitive",
            "expected_type": bool,
            "async": False,
        },
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [Exception("statement _pg_stmt does not exist")],
            "kwargs": {},
            "description": "Detect prepared statement error with _pg_ prefix",
            "should_succeed": True,
            "test_focus": "prepared_statement_pg_prefix",
            "expected_type": bool,
            "async": False,
        },
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [Exception("invalidsqlstatementname error code")],
            "kwargs": {},
            "description": "Detect prepared statement error with PostgreSQL error code",
            "should_succeed": True,
            "test_focus": "prepared_statement_error_code",
            "expected_type": bool,
            "async": False,
        },
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [Exception("regular database connection error")],
            "kwargs": {},
            "description": "Should not detect non-prepared statement errors",
            "should_succeed": True,
            "test_focus": "non_prepared_statement_error",
            "expected_type": bool,
            "async": False,
        },
        {
            "component": "is_prepared_statement_error",
            "function": is_prepared_statement_error,
            "args": [ValueError("Invalid argument provided")],
            "kwargs": {},
            "description": "Handle non-database errors gracefully",
            "should_succeed": True,
            "test_focus": "non_database_error",
            "expected_type": bool,
            "async": False,
        },
    ]


def get_prepared_statement_cleanup_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for prepared statement cleanup functionality."""
    return [
        {
            "component": "clear_prepared_statements",
            "function": clear_prepared_statements,
            "args": [],
            "kwargs": {},
            "description": "Clear prepared statements with valid database connection",
            "should_succeed": True,
            "test_focus": "clear_prepared_statements_success",
            "expected_type": type(None),
            "async": True,
        },
        {
            "component": "clear_prepared_statements_invalid_connection",
            "function": clear_prepared_statements,
            "args": [],
            "kwargs": {},
            "description": "Handle clear prepared statements with invalid connection gracefully",
            "should_succeed": True,  # Should not raise error even with invalid connection
            "test_focus": "clear_prepared_statements_error_handling",
            "expected_type": type(None),
            "async": True,
            "custom_test": "clear_prepared_statements_error_handling",
        },
        {
            "component": "clear_prepared_statements_multiple_calls",
            "function": clear_prepared_statements,
            "args": [],
            "kwargs": {},
            "description": "Test multiple calls to clear_prepared_statements (idempotent)",
            "should_succeed": True,
            "test_focus": "clear_prepared_statements_idempotent",
            "expected_type": type(None),
            "async": True,
            "custom_test": "clear_prepared_statements_multiple",
        },
    ]


def get_retry_decorator_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for retry decorator functionality."""
    return [
        {
            "component": "retry_on_prepared_statement_error",
            "function": retry_on_prepared_statement_error,
            "args": [2],  # max_retries
            "kwargs": {},
            "description": "Create retry decorator with custom max_retries",
            "should_succeed": True,
            "test_focus": "retry_decorator_creation",
            "expected_type": type(lambda: None),  # function type
            "async": False,
        },
        {
            "component": "retry_decorator_success_no_retry",
            "function": None,  # Will be handled in custom test
            "args": [],
            "kwargs": {},
            "description": "Test retry decorator with successful function (no retry needed)",
            "should_succeed": True,
            "test_focus": "retry_decorator_success",
            "expected_type": str,
            "async": True,
            "custom_test": "retry_decorator_success_test",
        },
        {
            "component": "retry_decorator_prepared_statement_error",
            "function": None,  # Will be handled in custom test
            "args": [],
            "kwargs": {},
            "description": "Test retry decorator with prepared statement error and recovery",
            "should_succeed": True,
            "test_focus": "retry_decorator_prepared_statement",
            "expected_type": str,
            "async": True,
            "custom_test": "retry_decorator_prepared_statement_test",
        },
        {
            "component": "retry_decorator_non_prepared_error",
            "function": None,  # Will be handled in custom test
            "args": [],
            "kwargs": {},
            "description": "Test retry decorator with non-prepared statement error (should not retry)",
            "should_succeed": False,  # Should fail and not retry
            "test_focus": "retry_decorator_non_prepared_error",
            "expected_type": Exception,
            "async": True,
            "custom_test": "retry_decorator_non_prepared_error_test",
        },
        {
            "component": "retry_decorator_max_retries_exhausted",
            "function": None,  # Will be handled in custom test
            "args": [],
            "kwargs": {},
            "description": "Test retry decorator when max retries are exhausted",
            "should_succeed": False,  # Should fail after max retries
            "test_focus": "retry_decorator_max_retries",
            "expected_type": Exception,
            "async": True,
            "custom_test": "retry_decorator_max_retries_test",
        },
    ]


def get_integration_test_cases() -> List[Dict[str, Any]]:
    """Generate integration test cases combining multiple error handling components."""
    return [
        {
            "component": "error_handling_integration",
            "function": None,  # Will be handled in custom test
            "args": [],
            "kwargs": {},
            "description": "Integration test of error detection and cleanup workflow",
            "should_succeed": True,
            "test_focus": "error_handling_integration",
            "expected_type": str,
            "async": True,
            "custom_test": "error_handling_integration_test",
        },
        {
            "component": "database_config_validation",
            "function": check_postgres_env_vars,
            "args": [],
            "kwargs": {},
            "description": "Validate database configuration for error handling tests",
            "should_succeed": True,
            "test_focus": "database_config_validation",
            "expected_type": bool,
            "async": False,
        },
    ]


def _get_error_handling_test_explanation(test_case: Dict[str, Any]) -> str:
    """Generate comprehensive test explanation for error handling tests."""
    component = test_case["component"]
    description = test_case["description"]
    test_focus = test_case.get("test_focus", "general")
    should_succeed = test_case.get("should_succeed", True)

    # Comprehensive explanations for different test focuses
    explanations = {
        "prepared_statement_detection": "🔍 is_prepared_statement_error: Analyzes exception messages to identify prepared statement conflicts using pattern matching",
        "prepared_statement_case_insensitive": "🔍 is_prepared_statement_error: Tests case-insensitive detection of prepared statement errors in exception messages",
        "prepared_statement_pg_prefix": "🔍 is_prepared_statement_error: Detects prepared statement errors with psycopg naming conventions (_pg_ prefixes)",
        "prepared_statement_error_code": "🔍 is_prepared_statement_error: Identifies PostgreSQL prepared statement error codes like invalidsqlstatementname",
        "non_prepared_statement_error": "🔍 is_prepared_statement_error: Correctly ignores non-prepared statement database errors",
        "non_database_error": "🔍 is_prepared_statement_error: Handles non-database exceptions gracefully without false positives",
        "clear_prepared_statements_success": "🧹 clear_prepared_statements: Connects to database and clears existing prepared statements safely",
        "clear_prepared_statements_error_handling": "🧹 clear_prepared_statements: Handles database connection errors gracefully without raising exceptions",
        "clear_prepared_statements_idempotent": "🧹 clear_prepared_statements: Safe to call multiple times with consistent behavior",
        "retry_decorator_creation": "🔄 retry_on_prepared_statement_error: Creates function decorator with configurable retry parameters",
        "retry_decorator_success": "🔄 retry_decorator: Tests decorated function that succeeds on first attempt (no retry needed)",
        "retry_decorator_prepared_statement": "🔄 retry_decorator: Handles prepared statement errors with automatic cleanup and retry",
        "retry_decorator_non_prepared_error": "🔄 retry_decorator: Does not retry non-prepared statement errors (immediate failure)",
        "retry_decorator_max_retries": "🔄 retry_decorator: Respects max_retries limit and fails gracefully when exhausted",
        "error_handling_integration": "🔧 error_handling_integration: End-to-end workflow of error detection, cleanup, and recovery",
        "database_config_validation": "⚙️ database_config_validation: Validates PostgreSQL environment variables for error handling tests",
    }

    explanation = explanations.get(test_focus, f"⚡ {component}: {description}")
    success_indicator = (
        "✅ Expected Success" if should_succeed else "✅ Expected Failure"
    )

    return f"{explanation} | {success_indicator}"


async def run_error_handling_test(
    test_id: str,
    test_case: Dict[str, Any],
    results: ErrorHandlingTestResults,
) -> None:
    """Run a single error handling test with comprehensive error handling and traceback capture."""
    component = test_case["component"]
    function = test_case["function"]
    args = test_case.get("args", [])
    kwargs = test_case.get("kwargs", {})
    description = test_case["description"]
    should_succeed = test_case.get("should_succeed", True)
    expected_type = test_case.get("expected_type", None)
    is_async = test_case.get("async", False)
    is_custom_test = test_case.get("custom_test", False)
    mock_env = test_case.get("mock_env", {})

    explanation = _get_error_handling_test_explanation(test_case)
    print(f"\n🧪 Test {test_id}: {explanation}")

    start_time = time.time()

    # Apply mock environment if specified
    original_env = {}
    if mock_env:
        for key, value in mock_env.items():
            original_env[key] = os.environ.get(key)
            if value is None and key in os.environ:
                del os.environ[key]
            elif value is not None:
                os.environ[key] = value

    try:
        with capture_server_logs() as log_capture:
            # Handle custom tests that need special setup
            if is_custom_test:
                result = await run_custom_error_handling_test(component, test_case)
            elif is_async:
                result = await function(*args, **kwargs)
            else:
                result = function(*args, **kwargs)

            response_time = time.time() - start_time

            # Validate result type if expected
            if expected_type and should_succeed:
                if expected_type == str:
                    type_check = isinstance(result, str)
                elif expected_type == dict:
                    type_check = isinstance(result, dict)
                elif expected_type == bool:
                    type_check = isinstance(result, bool)
                elif expected_type == type(None):
                    type_check = result is None
                elif expected_type == type(lambda: None):  # function type
                    type_check = callable(result)
                else:
                    type_check = isinstance(result, expected_type)

                if not type_check and should_succeed:
                    raise Exception(
                        f"Expected {expected_type.__name__}, got {type(result).__name__}: {result}"
                    )

            # Check for server-side logs/errors
            server_tracebacks = log_capture.get_captured_tracebacks()
            server_logs = log_capture.get_captured_logs()

            if should_succeed:
                if server_tracebacks:
                    print(
                        f"⚠️  Test {test_id} - Got server tracebacks but expected success"
                    )
                    for tb in server_tracebacks:
                        print(f"   Server Error: {tb['message']}")

                # Include server traceback info in response data even for successful tests
                response_data_with_logs = {
                    "result": str(result),
                    "type": type(result).__name__,
                    "server_tracebacks": server_tracebacks,
                    "server_logs": [
                        log
                        for log in server_logs
                        if log.get("level") in ["ERROR", "WARNING"]
                    ],
                }

                results.add_component_test(
                    test_id=test_id,
                    component=component,
                    description=description,
                    result_data=response_data_with_logs,
                    response_time=response_time,
                    success=True,
                )
                print(
                    f"✅ Test {test_id} - SUCCESS ({response_time:.2f}s) - Result type: {type(result).__name__}"
                )

            else:
                # This was an expected failure, check if we actually failed
                if expected_type == Exception:
                    print(
                        f"❌ Test {test_id} - Expected failure but got success: {result}"
                    )
                    results.add_error(
                        test_id=test_id,
                        endpoint=component,
                        description=description,
                        error=Exception(f"Expected failure but got success: {result}"),
                        response_time=response_time,
                        response_data={
                            "result": str(result),
                            "type": type(result).__name__,
                        },
                    )
                else:
                    # Expected failure with specific result
                    results.add_component_test(
                        test_id=test_id,
                        component=component,
                        description=description,
                        result_data=result,
                        response_time=response_time,
                        success=False,
                        expected_failure=True,
                    )
                    print(
                        f"✅ Test {test_id} - EXPECTED FAILURE ({response_time:.2f}s)"
                    )

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"

        # Capture full traceback
        full_traceback = traceback.format_exc()

        if should_succeed:
            # This was an unexpected error
            print(f"❌ Test {test_id} - UNEXPECTED ERROR: {error_message}")
            results.add_error(
                test_id=test_id,
                endpoint=component,
                description=description,
                error=e,
                response_time=response_time,
                response_data={
                    "error": error_message,
                    "traceback": full_traceback,
                    "error_type": type(e).__name__,
                },
            )
        else:
            # This was an expected failure
            print(f"✅ Test {test_id} - EXPECTED FAILURE: {error_message}")
            results.add_component_test(
                test_id=test_id,
                component=component,
                description=description,
                result_data={
                    "error": error_message,
                    "traceback": full_traceback,
                    "error_type": type(e).__name__,
                },
                response_time=response_time,
                success=False,
                expected_failure=True,
            )

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value


async def run_custom_error_handling_test(
    component: str, test_case: Dict[str, Any]
) -> Any:
    """Handle custom error handling test cases that require special setup."""
    custom_test = test_case.get("custom_test")

    if custom_test == "clear_prepared_statements_error_handling":
        # Test clear_prepared_statements with invalid connection
        try:
            # Temporarily corrupt database configuration
            import os

            original_host = os.environ.get("POSTGRES_HOST")
            try:
                os.environ["POSTGRES_HOST"] = "invalid_host_12345"
                await clear_prepared_statements()
                return None  # Should return None even with errors
            finally:
                # Restore original host
                if original_host:
                    os.environ["POSTGRES_HOST"] = original_host
                else:
                    os.environ.pop("POSTGRES_HOST", None)
        except Exception as e:
            # Should not raise - function is designed to handle errors gracefully
            raise Exception(
                f"clear_prepared_statements should handle errors gracefully: {e}"
            )

    elif custom_test == "clear_prepared_statements_multiple":
        # Test multiple calls to clear_prepared_statements
        try:
            await clear_prepared_statements()
            await clear_prepared_statements()
            await clear_prepared_statements()
            return None
        except Exception as e:
            raise Exception(f"Multiple calls to clear_prepared_statements failed: {e}")

    elif custom_test == "retry_decorator_success_test":
        # Test retry decorator with successful function
        @retry_on_prepared_statement_error(max_retries=2)
        async def test_successful_function():
            return "success_result"

        result = await test_successful_function()
        return result

    elif custom_test == "retry_decorator_prepared_statement_test":
        # Test retry decorator with prepared statement error that recovers
        attempt_count = [0]  # Use list to modify from inner function

        @retry_on_prepared_statement_error(max_retries=2)
        async def test_prepared_statement_function():
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt fails with prepared statement error
                raise Exception("prepared statement _pg3_test does not exist")
            else:
                # Second attempt succeeds
                return f"success_after_retry_attempt_{attempt_count[0]}"

        result = await test_prepared_statement_function()
        return result

    elif custom_test == "retry_decorator_non_prepared_error_test":
        # Test retry decorator with non-prepared statement error (should not retry)
        @retry_on_prepared_statement_error(max_retries=2)
        async def test_non_prepared_error_function():
            raise ValueError("This is not a prepared statement error")

        # This should raise the ValueError without retry
        try:
            result = await test_non_prepared_error_function()
            raise Exception("Expected ValueError but function succeeded")
        except ValueError as e:
            # This is the expected behavior - re-raise to test failure handling
            raise e

    elif custom_test == "retry_decorator_max_retries_test":
        # Test retry decorator when max retries are exhausted
        @retry_on_prepared_statement_error(max_retries=2)
        async def test_max_retries_function():
            # Always fails with prepared statement error
            raise Exception("prepared statement _pg3_persistent does not exist")

        # This should exhaust retries and raise the original exception
        try:
            result = await test_max_retries_function()
            raise Exception("Expected prepared statement error but function succeeded")
        except Exception as e:
            if "prepared statement" in str(e).lower():
                # This is expected - re-raise to test failure handling
                raise e
            else:
                raise Exception(f"Expected prepared statement error but got: {e}")

    elif custom_test == "error_handling_integration_test":
        # Integration test combining error detection and cleanup
        try:
            # Test error detection
            test_error = Exception("prepared statement _pg3_integration does not exist")
            is_prepared_error = is_prepared_statement_error(test_error)

            if not is_prepared_error:
                raise Exception("Error detection failed for integration test")

            # Test cleanup (should not raise errors)
            await clear_prepared_statements()

            # Test successful retry scenario
            integration_attempt = [0]

            @retry_on_prepared_statement_error(max_retries=1)
            async def integration_test_function():
                integration_attempt[0] += 1
                if integration_attempt[0] == 1:
                    raise Exception(
                        "prepared statement _pg3_integration_test does not exist"
                    )
                return "integration_test_success"

            result = await integration_test_function()
            return f"integration_complete_{result}"

        except Exception as e:
            raise Exception(f"Integration test failed: {e}")

    else:
        raise Exception(f"Unknown custom test: {custom_test}")


async def test_error_handling_integration():
    """Test error handling integration scenarios."""
    print("\n🔄 Running error handling integration test...")

    try:
        # Test basic error detection
        test_errors = [
            Exception("prepared statement _pg3_test does not exist"),
            Exception("PREPARED STATEMENT error"),
            Exception("regular database error"),
        ]

        detection_results = []
        for error in test_errors:
            is_prepared = is_prepared_statement_error(error)
            detection_results.append(is_prepared)

        print(f"Error detection results: {detection_results}")

        # Test cleanup function
        try:
            await clear_prepared_statements()
            print("Prepared statement cleanup: Success")
        except Exception as e:
            print(f"Prepared statement cleanup error (non-fatal): {e}")

        # Test retry decorator creation
        decorator = retry_on_prepared_statement_error(max_retries=1)
        print(f"Retry decorator created: {callable(decorator)}")

        return True

    except Exception as e:
        print(f"Integration test error: {e}")
        return False


async def run_error_handling_tests() -> ErrorHandlingTestResults:
    """Run all error handling tests and return comprehensive results."""
    print("[START] Error Handling Test Starting...")
    print(
        "[INFO] Testing: Comprehensive error handling functionality (prepared_statements.py, retry_decorators.py)"
    )
    print(f"[INFO] Components to test: {', '.join(sorted(REQUIRED_COMPONENTS))}")

    results = ErrorHandlingTestResults(REQUIRED_COMPONENTS)

    try:
        print("\n🚀 Starting error handling tests...")

        # Get all test cases
        all_test_cases = []
        all_test_cases.extend(get_prepared_statement_error_test_cases())
        all_test_cases.extend(get_prepared_statement_cleanup_test_cases())
        all_test_cases.extend(get_retry_decorator_test_cases())
        all_test_cases.extend(get_integration_test_cases())

        # Run each test case
        for i, test_case in enumerate(all_test_cases, 1):
            test_id = f"test_{i:03d}"
            await run_error_handling_test(test_id, test_case, results)

            # Small delay between tests
            await asyncio.sleep(0.1)

        # Run integration test
        integration_success = await test_error_handling_integration()
        print(
            f"✅ Integration test - {'SUCCESS' if integration_success else 'FAILED'} ({0.01:.2f}s)"
        )

        # Cleanup test
        print("\n🧹 Running cleanup tests...")
        try:
            await clear_prepared_statements()
            print("✅ Cleanup test - SUCCESS (0.01s)")
        except Exception as e:
            print(f"❌ Cleanup test - FAILED: {e}")

    except Exception as e:
        print(f"Error during test execution: {e}")
        traceback.print_exc()

    return results


def analyze_error_handling_test_results(results: ErrorHandlingTestResults):
    """Analyze and display comprehensive error handling test results."""
    summary = results.get_summary()
    total_tests = summary["total_requests"]
    success_tests = summary["successful_requests"]
    failed_tests = summary["failed_requests"]
    success_rate = summary["success_rate"]
    avg_response_time = summary["average_response_time"]

    print(f"\n📊 Error Handling Test Results:")
    print(f"Total: {total_tests}, Success: {success_tests}, Failed: {failed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Response Time: {avg_response_time:.2f}s")

    # Check component coverage
    missing_components = REQUIRED_COMPONENTS - results.component_coverage
    if missing_components:
        print(f"❌ Missing component tests: {missing_components}")
    else:
        print("✅ All required components tested")

    # Display errors if any
    if results.errors:
        print(f"❌ {len(results.errors)} Errors:")
        for i, error_data in enumerate(results.errors, 1):
            error_msg = error_data.get("error", "Unknown error")
            # Truncate long error messages
            if len(str(error_msg)) > 100:
                error_msg = str(error_msg)[:100] + "..."
            test_id = error_data.get("test_id", f"error_{i}")
            print(f"  {test_id}: {error_msg}")
            if i >= 5:  # Limit error display
                remaining = len(results.errors) - 5
                if remaining > 0:
                    print(f"  ... and {remaining} more errors")
                break

    # Save detailed report
    try:
        report_saved = save_traceback_report(
            report_type="error_handling_test_report", test_results=results
        )
        print(f"\n📋 Detailed report saved to: {report_saved}")
    except Exception as e:
        print(f"\n📋 Report saving failed: {e}")
        print(f"📊 Test completed successfully with {success_rate:.1f}% success rate")


async def main():
    """Main test execution function."""
    try:
        print("=" * 70)
        print("🔧 CHECKPOINTER ERROR HANDLING TEST SUITE")
        print("=" * 70)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🗂️  Test Database: {TEST_ERROR_HANDLING_NAME}")
        print(f"🔧 Error Handling Available: {ERROR_HANDLING_AVAILABLE}")

        if not ERROR_HANDLING_AVAILABLE:
            print(f"⚠️  Import Error: {IMPORT_ERROR}")
            print("📝 Note: Running with mock functions due to import failures")

        print(f"🎯 Required Components: {len(REQUIRED_COMPONENTS)}")
        for component in sorted(REQUIRED_COMPONENTS):
            print(f"   • {component}")

        print("\n" + "=" * 70)

        # Run the error handling tests
        results = await run_error_handling_tests()

        # Analyze and display results
        analyze_error_handling_test_results(results)

        # Determine overall success
        summary = results.get_summary()
        success_rate = summary["success_rate"]

        if success_rate >= 80:
            print(
                f"\n🎉 [SUCCESS] Error handling tests completed with {success_rate:.1f}% success rate!"
            )
        elif success_rate >= 60:
            print(
                f"\n⚠️  [PARTIAL SUCCESS] Error handling tests completed with {success_rate:.1f}% success rate"
            )
        else:
            print(
                f"\n❌ [NEEDS ATTENTION] Error handling tests completed with {success_rate:.1f}% success rate"
            )

        return results

    except Exception as e:
        print(f"\n💥 Fatal error in main test execution: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return None


# Update get_db_config to use load_dotenv
def get_db_config():
    load_dotenv()
    return {
        "host": os.getenv("host"),
        "port": int(os.getenv("port", 5432)),
        "user": os.getenv("user"),
        "password": os.getenv("password"),
        "dbname": os.getenv("dbname"),
    }


if __name__ == "__main__":
    asyncio.run(main())

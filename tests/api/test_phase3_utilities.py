"""
Test for Phase 3: Extract Utilities (Debug, Memory, Rate Limiting)
Tests the utility functions with comprehensive validation and environment setup.
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import tempfile
import threading
from dotenv import load_dotenv

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


try:
    # Import utilities for testing
    from api.utils.debug import (
        print__admin_clear_cache_debug,
        print__analysis_tracing_debug,
        print__analyze_debug,
        print__api_postgresql,
        print__chat_all_messages_debug,
        print__debug,
        print__feedback_flow,
        print__memory_monitoring,
        print__token_debug,
        print__sentiment_flow,
    )
    from api.utils.memory import (
        check_memory_and_gc,
        cleanup_bulk_cache,
        log_comprehensive_error,
        log_memory_usage,
        setup_graceful_shutdown,
    )
    from api.utils.rate_limiting import (
        check_rate_limit,
        check_rate_limit_with_throttling,
        wait_for_rate_limit,
    )
except ImportError as e:
    print(f"❌ Failed to import utilities: {e}")
    print("This may be due to missing dependencies. Please run: pip install psutil")
    sys.exit(1)

load_dotenv()

# Test configuration
TEST_UTILITIES = {
    "debug": [
        "print__debug",
        "print__api_postgresql",
        "print__feedback_flow",
        "print__memory_monitoring",
        "print__token_debug",
        "print__sentiment_flow",
        "print__analyze_debug",
        "print__chat_all_messages_debug",
    ],
    "memory": [
        "check_memory_and_gc",
        "cleanup_bulk_cache",
        "log_memory_usage",
        "setup_graceful_shutdown",
        "log_comprehensive_error",
    ],
    "rate_limiting": [
        "check_rate_limit",
        "check_rate_limit_with_throttling",
        "wait_for_rate_limit",
    ],
    "edge_cases": ["debug_edge_cases", "memory_edge_cases", "rate_limiting_edge_cases"],
    "performance": [
        "debug_performance",
        "memory_performance",
        "rate_limiting_performance",
    ],
}

# Test cases for different utility categories
DEBUG_TEST_CASES = [
    {
        "function": "print__debug",
        "env_var": "print__debug",
        "test_message": "Test debug message",
        "description": "Basic debug logging",
        "should_output": True,
    },
    {
        "function": "print__api_postgresql",
        "env_var": "print__api_postgresql",
        "test_message": "Test PostgreSQL debug message",
        "description": "PostgreSQL debug logging",
        "should_output": True,
    },
    {
        "function": "print__feedback_flow",
        "env_var": "print__feedback_flow",
        "test_message": "Test feedback flow message",
        "description": "Feedback flow debug logging",
        "should_output": True,
    },
    {
        "function": "print__memory_monitoring",
        "env_var": "print__memory_monitoring",
        "test_message": "Test memory monitoring message",
        "description": "Memory monitoring debug logging",
        "should_output": True,
    },
    {
        "function": "print__token_debug",
        "env_var": "print__token_debug",
        "test_message": "Test token debug message",
        "description": "Token debug logging",
        "should_output": True,
    },
    {
        "function": "print__sentiment_flow",
        "env_var": "print__sentiment_flow",
        "test_message": "Test sentiment flow message",
        "description": "Sentiment flow debug logging",
        "should_output": True,
    },
    {
        "function": "print__analyze_debug",
        "env_var": "print__analyze_debug",
        "test_message": "Test analyze debug message",
        "description": "Analyze debug logging",
        "should_output": True,
    },
    {
        "function": "print__chat_all_messages_debug",
        "env_var": "print__chat_all_messages_debug",
        "test_message": "Test chat messages debug",
        "description": "Chat messages debug logging",
        "should_output": True,
    },
]

MEMORY_TEST_CASES = [
    {
        "function": "check_memory_and_gc",
        "description": "Memory check and garbage collection",
        "expected_type": "float",
        "should_succeed": True,
        "validate_range": True,
        "min_value": 0,
        "max_value": 2000,  # 2GB should be reasonable upper limit
    },
    {
        "function": "cleanup_bulk_cache",
        "description": "Bulk cache cleanup",
        "expected_type": "int",
        "should_succeed": True,
        "validate_range": True,
        "min_value": 0,
    },
    {
        "function": "log_memory_usage",
        "description": "Memory usage logging with context",
        "args": ["test_context"],
        "expected_type": "None",
        "should_succeed": True,
    },
    {
        "function": "log_memory_usage",
        "description": "Memory usage logging without context",
        "args": [],
        "expected_type": "None",
        "should_succeed": True,
    },
    {
        "function": "setup_graceful_shutdown",
        "description": "Graceful shutdown setup",
        "expected_type": "None",
        "should_succeed": True,
    },
    {
        "function": "log_comprehensive_error",
        "description": "Comprehensive error logging with request",
        "args": ["test_context", "ValueError('Test error')", "mock_request"],
        "expected_type": "None",
        "should_succeed": True,
        "requires_mock": True,
    },
    {
        "function": "log_comprehensive_error",
        "description": "Comprehensive error logging without request",
        "args": ["test_context", "ValueError('Test error')", "None"],
        "expected_type": "None",
        "should_succeed": True,
        "requires_mock": True,
    },
]

RATE_LIMITING_TEST_CASES = [
    {
        "function": "check_rate_limit_with_throttling",
        "description": "Rate limit check with throttling info",
        "args": ["127.0.0.1"],
        "expected_keys": [
            "allowed",
            "suggested_wait",
            "burst_count",
            "window_count",
            "burst_limit",
            "window_limit",
        ],
        "should_succeed": True,
    },
    {
        "function": "check_rate_limit",
        "description": "Simple rate limit check - first request",
        "args": ["127.0.0.2"],  # Use different IP to avoid conflicts
        "expected_type": "bool",
        "should_succeed": True,
        "expected_value": True,  # First request should succeed
    },
    {
        "function": "check_rate_limit",
        "description": "Simple rate limit check - rapid requests",
        "args": ["127.0.0.3"],
        "expected_type": "bool",
        "should_succeed": True,
        "test_burst": True,  # Test burst behavior
    },
    {
        "function": "wait_for_rate_limit",
        "description": "Async rate limit wait - allowed request",
        "args": ["127.0.0.4"],
        "expected_type": "bool",
        "should_succeed": True,
        "is_async": True,
        "expected_value": True,
    },
    {
        "function": "check_rate_limit_with_throttling",
        "description": "Rate limit throttling with different IP",
        "args": ["192.168.1.1"],
        "expected_keys": ["allowed", "suggested_wait", "burst_count", "window_count"],
        "should_succeed": True,
        "validate_values": True,
    },
    {
        "function": "wait_for_rate_limit",
        "description": "Async rate limit wait with timeout scenarios",
        "args": ["127.0.0.5"],
        "expected_type": "bool",
        "should_succeed": True,
        "is_async": True,
        "test_timeout": True,
    },
]

# Additional edge case tests
EDGE_CASE_TEST_CASES = [
    {
        "category": "debug",
        "function": "print__debug",
        "description": "Debug with empty message",
        "env_var": "print__debug",
        "test_message": "",
        "should_succeed": True,
    },
    {
        "category": "debug",
        "function": "print__api_postgresql",
        "description": "Debug with very long message",
        "env_var": "print__api_postgresql",
        "test_message": "x" * 1000,  # Very long message
        "should_succeed": True,
    },
    {
        "category": "debug",
        "function": "print__debug",
        "description": "Debug with special characters",
        "env_var": "print__debug",
        "test_message": "Test with unicode: 🚀 and special chars: <>&\"'",
        "should_succeed": True,
    },
    {
        "category": "rate_limiting",
        "function": "check_rate_limit",
        "description": "Rate limit with invalid IP format",
        "args": ["invalid_ip"],
        "should_succeed": True,  # Should handle gracefully
    },
    {
        "category": "memory",
        "function": "check_memory_and_gc",
        "description": "Memory check under stress",
        "should_succeed": True,
        "stress_test": True,
    },
]


async def make_utility_test_request(
    test_id: str,
    test_category: str,
    description: str,
    test_function: callable,
    should_succeed: bool,
    results: BaseTestResults,
    *args,
    **kwargs,
):
    """Make a utility test request with traceback capture, following catalog test pattern."""
    start_time = time.time()

    try:
        # Create a mock result structure similar to HTTP requests
        mock_result = {
            "response": None,
            "success": True,
            "server_tracebacks": [],
            "client_exception": None,
        }

        try:
            # Execute the test function
            test_result = (
                await test_function(*args, **kwargs)
                if asyncio.iscoroutinefunction(test_function)
                else test_function(*args, **kwargs)
            )

            # Create a mock response object
            mock_response = type(
                "MockResponse",
                (),
                {
                    "status_code": 200,
                    "json": lambda self=None: {"result": test_result, "success": True},
                },
            )()

            mock_result["response"] = mock_response

        except Exception as test_error:
            mock_result["success"] = False
            mock_result["client_exception"] = test_error

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(mock_result)

        if mock_result["response"] is None:
            error_message = error_info["client_error"] or "Unknown utility test error"
            print(f"❌ Test {test_id} - Utility Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id, test_category, description, error_obj, response_time
            )
            return

        response = mock_result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.3f}s)")

        if should_succeed:
            if response.status_code == 200:
                try:
                    data = response.json()
                    results.add_result(
                        test_id,
                        test_category,
                        description,
                        data,
                        response_time,
                        response.status_code,
                    )
                except Exception as e:
                    print(f"❌ Validation failed: {e}")
                    error_obj = Exception(f"Test validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id, test_category, description, error_obj, response_time
                    )
            else:
                handle_error_response(
                    test_id,
                    test_category,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )
        else:
            handle_expected_failure(
                test_id,
                test_category,
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
            test_id,
            test_category,
            description,
            error_obj,
            response_time,
            response_data=None,
        )


def _validate_debug_output(
    function_name: str, env_var: str, message: str, should_output: bool, **kwargs
):
    """Validate debug function output behavior."""
    # Save original environment
    original_value = os.environ.get(env_var)

    try:
        # Test with debug enabled
        os.environ[env_var] = "1"

        # Force reload of debug module to pick up new environment
        import importlib
        import api.utils.debug

        importlib.reload(api.utils.debug)
        from api.utils.debug import (
            print__debug,
            print__api_postgresql,
            print__feedback_flow,
            print__memory_monitoring,
            print__token_debug,
            print__sentiment_flow,
            print__analyze_debug,
            print__chat_all_messages_debug,
        )

        # Capture stdout for validation
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Get the function and call it
            if function_name == "print__debug":
                print__debug(message)
            elif function_name == "print__api_postgresql":
                print__api_postgresql(message)
            elif function_name == "print__feedback_flow":
                print__feedback_flow(message)
            elif function_name == "print__memory_monitoring":
                print__memory_monitoring(message)
            elif function_name == "print__token_debug":
                print__token_debug(message)
            elif function_name == "print__sentiment_flow":
                print__sentiment_flow(message)
            elif function_name == "print__analyze_debug":
                print__analyze_debug(message)
            elif function_name == "print__chat_all_messages_debug":
                print__chat_all_messages_debug(message)

        output_with_debug = captured_output.getvalue()

        # Test with debug disabled
        os.environ[env_var] = "0"

        # Force reload again to pick up disabled environment
        importlib.reload(api.utils.debug)
        from api.utils.debug import (
            print__debug,
            print__api_postgresql,
            print__feedback_flow,
            print__memory_monitoring,
            print__token_debug,
            print__sentiment_flow,
            print__analyze_debug,
            print__chat_all_messages_debug,
        )

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            if function_name == "print__debug":
                print__debug(message)
            elif function_name == "print__api_postgresql":
                print__api_postgresql(message)
            elif function_name == "print__feedback_flow":
                print__feedback_flow(message)
            elif function_name == "print__memory_monitoring":
                print__memory_monitoring(message)
            elif function_name == "print__token_debug":
                print__token_debug(message)
            elif function_name == "print__sentiment_flow":
                print__sentiment_flow(message)
            elif function_name == "print__analyze_debug":
                print__analyze_debug(message)
            elif function_name == "print__chat_all_messages_debug":
                print__chat_all_messages_debug(message)

        output_without_debug = captured_output.getvalue()

        # Validate behavior
        if should_output:
            # For empty messages, just check that something was output
            if message == "":
                assert (
                    len(output_with_debug) > 0
                ), "Expected some output with debug enabled"
            else:
                assert (
                    message in output_with_debug
                ), f"Expected message '{message}' in debug output"
            assert output_without_debug == "", f"Expected no output when debug disabled"

        return True

    finally:
        # Restore original environment
        if original_value is not None:
            os.environ[env_var] = original_value
        elif env_var in os.environ:
            del os.environ[env_var]


def _validate_memory_function(
    function_name: str,
    args: List = None,
    expected_type: str = None,
    test_case: Dict = None,
    **kwargs,
):
    """Validate memory management function behavior."""
    if function_name == "check_memory_and_gc":
        # Optionally create memory pressure for stress test
        if test_case and test_case.get("stress_test"):
            # Create some memory pressure
            temp_data = []
            try:
                for i in range(100):
                    temp_data.append([0] * 1000)
                result = check_memory_and_gc()
            finally:
                del temp_data
        else:
            result = check_memory_and_gc()

        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result >= 0, f"Expected non-negative memory value, got {result}"

        # Validate range if specified
        if test_case and test_case.get("validate_range"):
            min_val = test_case.get("min_value", 0)
            max_val = test_case.get("max_value", float("inf"))
            assert (
                min_val <= result <= max_val
            ), f"Memory value {result} not in range [{min_val}, {max_val}]"

    elif function_name == "cleanup_bulk_cache":
        result = cleanup_bulk_cache()
        assert isinstance(result, int), f"Expected int, got {type(result)}"
        assert result >= 0, f"Expected non-negative cleanup count, got {result}"

        # Validate range if specified
        if test_case and test_case.get("validate_range"):
            min_val = test_case.get("min_value", 0)
            assert result >= min_val, f"Cleanup count {result} below minimum {min_val}"

    elif function_name == "log_memory_usage":
        # Should not raise exception
        if args:
            log_memory_usage(args[0])
        else:
            log_memory_usage()
        result = None

    elif function_name == "setup_graceful_shutdown":
        # Should not raise exception
        setup_graceful_shutdown()
        result = None

    elif function_name == "log_comprehensive_error":
        # Handle mock requirements
        if test_case and test_case.get("requires_mock"):
            if len(args) >= 3 and args[2] == "mock_request":
                # Create mock request
                class MockRequest:
                    def __init__(self):
                        self.method = "GET"
                        self.url = "http://test.com/test"
                        self.client = type("MockClient", (), {"host": "127.0.0.1"})()

                mock_request = MockRequest()
                test_error = ValueError("Test error")
                log_comprehensive_error(args[0], test_error, mock_request)
            elif len(args) >= 3 and args[2] == "None":
                test_error = ValueError("Test error")
                log_comprehensive_error(args[0], test_error, None)
        result = None

    return True


def _validate_rate_limiting_function(
    function_name: str,
    args: List = None,
    expected_keys: List = None,
    expected_type: str = None,
    test_case: Dict = None,
    **kwargs,
):
    """Validate rate limiting function behavior."""
    if function_name == "check_rate_limit_with_throttling":
        result = check_rate_limit_with_throttling(args[0])
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

        for key in expected_keys:
            assert key in result, f"Expected key '{key}' in result"

        # Validate specific value types
        assert isinstance(result["allowed"], bool), "Expected 'allowed' to be bool"
        assert isinstance(
            result["suggested_wait"], (int, float)
        ), "Expected 'suggested_wait' to be numeric"
        assert isinstance(
            result["burst_count"], int
        ), "Expected 'burst_count' to be int"
        assert isinstance(
            result["window_count"], int
        ), "Expected 'window_count' to be int"

        # Validate value ranges if specified
        if test_case and test_case.get("validate_values"):
            assert (
                result["suggested_wait"] >= 0
            ), "suggested_wait should be non-negative"
            assert result["burst_count"] >= 0, "burst_count should be non-negative"
            assert result["window_count"] >= 0, "window_count should be non-negative"

            if "burst_limit" in result:
                assert (
                    result["burst_count"] <= result["burst_limit"]
                ), "burst_count should not exceed burst_limit"
            if "window_limit" in result:
                assert (
                    result["window_count"] <= result["window_limit"]
                ), "window_count should not exceed window_limit"

    elif function_name == "check_rate_limit":
        if test_case and test_case.get("test_burst"):
            # Test burst behavior by making multiple rapid requests
            results = []
            for i in range(10):  # Make multiple requests
                result = check_rate_limit(args[0])
                results.append(result)
                assert isinstance(result, bool), f"Expected bool, got {type(result)}"

            # At least the first request should succeed
            assert any(results), "At least one request should succeed in burst test"
        else:
            result = check_rate_limit(args[0])
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"

            # Check expected value if specified
            if test_case and "expected_value" in test_case:
                assert (
                    result == test_case["expected_value"]
                ), f"Expected {test_case['expected_value']}, got {result}"

    return True


async def _validate_async_rate_limiting_function(
    function_name: str,
    args: List = None,
    expected_type: str = None,
    test_case: Dict = None,
    **kwargs,
):
    """Validate async rate limiting function behavior."""
    if function_name == "wait_for_rate_limit":
        if test_case and test_case.get("test_timeout"):
            # Test with potential timeout scenarios
            start_time = time.time()
            result = await wait_for_rate_limit(args[0])
            elapsed = time.time() - start_time

            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            # Should complete in reasonable time (less than 30 seconds)
            assert elapsed < 30, f"Function took too long: {elapsed:.2f}s"
        else:
            result = await wait_for_rate_limit(args[0])
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"

            # Check expected value if specified
            if test_case and "expected_value" in test_case:
                assert (
                    result == test_case["expected_value"]
                ), f"Expected {test_case['expected_value']}, got {result}"

    return True


async def run_debug_tests(results: BaseTestResults):
    """Run debug utility tests."""
    print("\n🔧 Testing Debug Utilities...")

    for i, test_case in enumerate(DEBUG_TEST_CASES, 1):
        test_id = f"debug_test_{i}"
        description = test_case["description"]

        await make_utility_test_request(
            test_id=test_id,
            test_category="debug_utils",
            description=description,
            test_function=_validate_debug_output,
            should_succeed=True,
            results=results,
            function_name=test_case["function"],
            env_var=test_case["env_var"],
            message=test_case["test_message"],
            should_output=test_case["should_output"],
        )


async def run_memory_tests(results: BaseTestResults):
    """Run memory management utility tests."""
    print("\n🧠 Testing Memory Management Utilities...")

    for i, test_case in enumerate(MEMORY_TEST_CASES, 1):
        test_id = f"memory_test_{i}"
        description = test_case["description"]

        await make_utility_test_request(
            test_id=test_id,
            test_category="memory_utils",
            description=description,
            test_function=_validate_memory_function,
            should_succeed=True,
            results=results,
            function_name=test_case["function"],
            args=test_case.get("args"),
            expected_type=test_case.get("expected_type"),
            test_case=test_case,
        )


async def run_rate_limiting_tests(results: BaseTestResults):
    """Run rate limiting utility tests."""
    print("\n⏱️ Testing Rate Limiting Utilities...")

    for i, test_case in enumerate(RATE_LIMITING_TEST_CASES, 1):
        test_id = f"rate_test_{i}"
        description = test_case["description"]

        if test_case.get("is_async"):
            test_function = _validate_async_rate_limiting_function
        else:
            test_function = _validate_rate_limiting_function

        await make_utility_test_request(
            test_id=test_id,
            test_category="rate_limiting_utils",
            description=description,
            test_function=test_function,
            should_succeed=True,
            results=results,
            function_name=test_case["function"],
            args=test_case.get("args"),
            expected_keys=test_case.get("expected_keys"),
            expected_type=test_case.get("expected_type"),
            test_case=test_case,
        )


async def run_integration_tests(results: BaseTestResults):
    """Run cross-module integration tests."""
    print("\n🔗 Testing Cross-Module Integration...")

    # Test 1: Debug functions work across modules
    test_id = "integration_test_1"
    start_time = time.time()

    try:
        # Test that debug functions can be called from memory module
        class MockRequest:
            def __init__(self):
                self.method = "GET"
                self.url = "http://test.com/test"
                self.client = type("MockClient", (), {"host": "127.0.0.1"})()

        mock_request = MockRequest()
        test_error = ValueError("Test error for integration")

        # This should work without error
        log_comprehensive_error("integration_test", test_error, mock_request)

        response_time = time.time() - start_time
        print(f"✅ Test {test_id}: Cross-module error logging ({response_time:.3f}s)")

        results.add_result(
            test_id,
            "integration",
            "Cross-module error logging",
            {"success": True},
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Cross-module error logging failed - {str(e)}")
        results.add_error(
            test_id, "integration", "Cross-module error logging", e, response_time
        )

    # Test 2: Rate limiting with burst behavior
    test_id = "integration_test_2"
    start_time = time.time()

    try:
        test_ip = "192.168.1.100"  # Use different IP for this test

        # Clear any existing rate limit data for this IP
        from api.config.settings import rate_limit_storage

        if test_ip in rate_limit_storage:
            del rate_limit_storage[test_ip]

        # Test multiple rapid requests to trigger rate limiting
        results_list = []
        for i in range(5):
            result = check_rate_limit(test_ip)
            results_list.append(result)

        # First few should succeed, later ones might fail due to rate limiting
        assert any(results_list), "At least one request should succeed"

        response_time = time.time() - start_time
        print(f"✅ Test {test_id}: Rate limiting burst behavior ({response_time:.3f}s)")

        results.add_result(
            test_id,
            "integration",
            "Rate limiting burst behavior",
            {"results": results_list, "success": True},
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Rate limiting burst behavior failed - {str(e)}")
        results.add_error(
            test_id, "integration", "Rate limiting burst behavior", e, response_time
        )

    # Test 3: Memory monitoring with actual memory pressure
    test_id = "integration_test_3"
    start_time = time.time()

    try:
        # Get baseline memory
        baseline_memory = check_memory_and_gc()

        # Create some memory pressure by allocating data
        large_data = []
        for i in range(1000):
            large_data.append([0] * 1000)  # Allocate some memory

        # Check memory again
        after_allocation = check_memory_and_gc()

        # Clean up
        del large_data

        # Memory should have increased
        assert (
            after_allocation >= baseline_memory
        ), "Memory should increase after allocation"

        response_time = time.time() - start_time
        print(
            f"✅ Test {test_id}: Memory monitoring under pressure ({response_time:.3f}s)"
        )

        results.add_result(
            test_id,
            "integration",
            "Memory monitoring under pressure",
            {
                "baseline": baseline_memory,
                "after_allocation": after_allocation,
                "success": True,
            },
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Memory monitoring under pressure failed - {str(e)}")
        results.add_error(
            test_id, "integration", "Memory monitoring under pressure", e, response_time
        )


async def run_edge_case_tests(results: BaseTestResults):
    """Run edge case and stress tests."""
    print("\n🧪 Testing Edge Cases and Stress Scenarios...")

    for i, test_case in enumerate(EDGE_CASE_TEST_CASES, 1):
        test_id = f"edge_test_{i}"
        start_time = time.time()

        try:
            if test_case["category"] == "debug":
                success = _validate_debug_output(
                    test_case["function"],
                    test_case["env_var"],
                    test_case["test_message"],
                    test_case.get("should_output", True),
                )
            elif test_case["category"] == "memory":
                success = _validate_memory_function(
                    test_case["function"],
                    test_case.get("args"),
                    test_case.get("expected_type"),
                    test_case,
                )
            elif test_case["category"] == "rate_limiting":
                success = _validate_rate_limiting_function(
                    test_case["function"],
                    test_case.get("args"),
                    test_case.get("expected_keys"),
                    test_case.get("expected_type"),
                    test_case,
                )
            else:
                raise ValueError(f"Unknown test category: {test_case['category']}")

            response_time = time.time() - start_time
            print(
                f"✅ Test {test_id}: {test_case['description']} ({response_time:.3f}s)"
            )

            results.add_result(
                test_id,
                "edge_cases",
                test_case["description"],
                {"function": test_case["function"], "success": success},
                response_time,
                200,
            )

        except Exception as e:
            response_time = time.time() - start_time
            print(f"❌ Test {test_id}: {test_case['description']} failed - {str(e)}")
            results.add_error(
                test_id, "edge_cases", test_case["description"], e, response_time
            )


async def run_performance_tests(results: BaseTestResults):
    """Run performance and load tests."""
    print("\n🚄 Testing Performance and Load Scenarios...")

    # Performance test 1: Debug function performance under load
    test_id = "perf_test_1"
    start_time = time.time()

    try:
        os.environ["DEBUG"] = "1"

        # Test multiple rapid debug calls
        for i in range(100):
            print__debug(f"Performance test message {i}")

        response_time = time.time() - start_time
        print(
            f"✅ Test {test_id}: Debug function performance (100 calls) ({response_time:.3f}s)"
        )

        results.add_result(
            test_id,
            "performance",
            "Debug function performance under load",
            {"calls": 100, "total_time": response_time},
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Debug function performance failed - {str(e)}")
        results.add_error(
            test_id, "performance", "Debug function performance", e, response_time
        )

    # Performance test 2: Memory management under repeated calls
    test_id = "perf_test_2"
    start_time = time.time()

    try:
        memory_readings = []

        # Test repeated memory checks
        for i in range(10):
            memory_mb = check_memory_and_gc()
            memory_readings.append(memory_mb)
            cleanup_bulk_cache()

        response_time = time.time() - start_time
        avg_memory = sum(memory_readings) / len(memory_readings)
        print(
            f"✅ Test {test_id}: Memory management performance (10 cycles, avg: {avg_memory:.1f}MB) ({response_time:.3f}s)"
        )

        results.add_result(
            test_id,
            "performance",
            "Memory management performance under repeated calls",
            {"cycles": 10, "avg_memory": avg_memory, "readings": memory_readings},
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Memory management performance failed - {str(e)}")
        results.add_error(
            test_id, "performance", "Memory management performance", e, response_time
        )

    # Performance test 3: Rate limiting under concurrent scenarios
    test_id = "perf_test_3"
    start_time = time.time()

    try:
        # Test rate limiting with multiple IPs
        ips = [f"192.168.1.{i}" for i in range(1, 21)]  # 20 different IPs
        results_matrix = []

        for ip in ips:
            ip_results = []
            for j in range(5):  # 5 requests per IP
                result = check_rate_limit(ip)
                ip_results.append(result)
            results_matrix.append(ip_results)

        response_time = time.time() - start_time
        total_requests = len(ips) * 5
        successful_requests = sum(sum(ip_results) for ip_results in results_matrix)

        print(
            f"✅ Test {test_id}: Rate limiting concurrency ({successful_requests}/{total_requests} allowed) ({response_time:.3f}s)"
        )

        results.add_result(
            test_id,
            "performance",
            "Rate limiting under concurrent scenarios",
            {
                "total_requests": total_requests,
                "successful": successful_requests,
                "ips_tested": len(ips),
            },
            response_time,
            200,
        )

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Test {test_id}: Rate limiting concurrency failed - {str(e)}")
        results.add_error(
            test_id, "performance", "Rate limiting concurrency", e, response_time
        )


async def run_utilities_tests() -> BaseTestResults:
    """Run all utility function tests."""
    print("🚀 Starting Phase 3 Utilities Tests...")

    required_utilities = set(
        [
            "debug_utils",
            "memory_utils",
            "rate_limiting_utils",
            "integration",
            "edge_cases",
            "performance",
        ]
    )
    results = BaseTestResults(required_endpoints=required_utilities)
    results.start_time = datetime.now()

    # Run all test categories
    await run_debug_tests(results)
    await run_memory_tests(results)
    await run_rate_limiting_tests(results)
    await run_integration_tests(results)
    await run_edge_case_tests(results)
    await run_performance_tests(results)

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results Analysis:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.3f}s")
        print(f"Max Response Time: {summary['max_response_time']:.3f}s")
        print(f"Min Response Time: {summary['min_response_time']:.3f}s")

    if not summary["all_endpoints_tested"]:
        missing = set(results.required_endpoints) - set(summary["tested_endpoints"])
        print(f"⚠️ Missing utility categories: {missing}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Error(s):")
        for error in results.errors[:5]:  # Show first 5 errors
            print(f"  - {error['test_id']}: {error['description']} - {error['error']}")
        if len(results.errors) > 5:
            print(f"  ... and {len(results.errors) - 5} more errors")

    # Save traceback information
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("🚀 Phase 3 Utilities Test Starting...")

    try:
        # Setup debug environment for comprehensive testing
        print("🔧 Setting up debug environment...")

        # Set up various debug flags for testing
        debug_setup = {
            "DEBUG": "1",
            "print__api_postgresql": "1",
            "print__memory_monitoring": "1",
            "print__token_debug": "1",
            "print__sentiment_flow": "1",
            "print__analyze_debug": "1",
            "print__chat_all_messages_debug": "1",
            "DEBUG_TRACEBACK": "1",
        }

        # Apply debug environment
        original_env = {}
        for key, value in debug_setup.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            results = await run_utilities_tests()
            summary = analyze_test_results(results)

            # Determine overall test success
            has_empty_errors = any(
                error.get("error", "").strip() == ""
                or "Unknown error" in error.get("error", "")
                for error in summary["errors"]
            )

            has_critical_errors = any(
                "ImportError" in error.get("error", "")
                or "ModuleNotFoundError" in error.get("error", "")
                or "AttributeError" in error.get("error", "")
                for error in summary["errors"]
            )

            test_passed = (
                not has_empty_errors
                and not has_critical_errors
                and summary["total_requests"] > 0
                and summary["all_endpoints_tested"]
                and summary["failed_requests"] == 0
                and summary["successful_requests"] > 0
            )

            if has_empty_errors:
                print("❌ Test failed: Empty error messages detected")
            elif has_critical_errors:
                print("❌ Test failed: Critical import/module errors detected")
            elif summary["successful_requests"] == 0:
                print("❌ Test failed: No tests succeeded")
            elif not summary["all_endpoints_tested"]:
                print("❌ Test failed: Not all utility categories were tested")
            elif summary["failed_requests"] > 0:
                print(f"❌ Test failed: {summary['failed_requests']} tests failed")

            print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
            return test_passed

        finally:
            # Restore original environment
            print("🧹 Cleaning up debug environment...")
            for key, original_value in original_env.items():
                if original_value is not None:
                    os.environ[key] = original_value
                elif key in os.environ:
                    del os.environ[key]

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Test Type": "Phase 3 Utilities",
            "Test Categories": list(TEST_UTILITIES.keys()),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


def test_phase3_utilities():
    """Backward compatibility function for existing calls."""
    return asyncio.run(main())


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        test_context = {
            "Test Type": "Phase 3 Utilities",
            "Test Categories": list(TEST_UTILITIES.keys()),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

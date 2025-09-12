"""Test for Checkpointer Module: factory.py and health.py
Tests the checkpointer functionality with comprehensive testing patterns.

This test file follows the same patterns as test_phase8_catalog.py and test_phase8_feedback.py,
including proper error handling, traceback capture, and detailed reporting.
"""

import os
import sys
from pathlib import Path

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback
from unittest.mock import patch, MagicMock

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

# Import checkpointer modules for testing with error handling
try:
    from checkpointer.checkpointer.factory import (
        create_async_postgres_saver,
        close_async_postgres_saver,
        get_global_checkpointer,
        initialize_checkpointer,
        cleanup_checkpointer,
        check_pool_health_and_recreate,
    )
    from checkpointer.checkpointer.health import (
        check_pool_health_and_recreate as health_check_pool_health_and_recreate,
    )
    from checkpointer.config import (
        check_postgres_env_vars,
        get_db_config,
        DEFAULT_MAX_RETRIES,
        CHECKPOINTER_CREATION_MAX_RETRIES,
        DEFAULT_POOL_MIN_SIZE,
        DEFAULT_POOL_MAX_SIZE,
    )
    from checkpointer.globals import _GLOBAL_CHECKPOINTER
    from checkpointer.database.connection import (
        get_connection_string,
        get_direct_connection,
    )

    CHECKPOINTER_AVAILABLE = True
    IMPORT_ERROR = None

except ImportError as e:
    CHECKPOINTER_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create mock functions for testing import failures
    def create_async_postgres_saver():
        raise ImportError(IMPORT_ERROR)

    def close_async_postgres_saver():
        raise ImportError(IMPORT_ERROR)

    def get_global_checkpointer():
        raise ImportError(IMPORT_ERROR)

    def initialize_checkpointer():
        raise ImportError(IMPORT_ERROR)

    def cleanup_checkpointer():
        raise ImportError(IMPORT_ERROR)

    def check_pool_health_and_recreate():
        raise ImportError(IMPORT_ERROR)

    def health_check_pool_health_and_recreate():
        raise ImportError(IMPORT_ERROR)

    def check_postgres_env_vars():
        return False

    def get_db_config():
        return {}

    def get_connection_string():
        raise ImportError(IMPORT_ERROR)

    def get_direct_connection():
        raise ImportError(IMPORT_ERROR)

    DEFAULT_MAX_RETRIES = 2
    CHECKPOINTER_CREATION_MAX_RETRIES = 2
    DEFAULT_POOL_MIN_SIZE = 3
    DEFAULT_POOL_MAX_SIZE = 10
    _GLOBAL_CHECKPOINTER = None

# Test configuration
TEST_CONFIG_THREAD_ID = "test_checkpointer_thread"
TEST_CONFIG = {"configurable": {"thread_id": TEST_CONFIG_THREAD_ID}}
REQUIRED_COMPONENTS = {
    "create_async_postgres_saver",
    "close_async_postgres_saver",
    "get_global_checkpointer",
    "initialize_checkpointer",
    "cleanup_checkpointer",
    "check_pool_health_and_recreate",
    "health_check_pool_health_and_recreate",
}


class CheckpointerTestResults(BaseTestResults):
    """Extended test results class for checkpointer testing."""

    def __init__(self, required_components: set = None):
        super().__init__(required_endpoints=required_components)
        self.required_components = required_components or set()
        self.component_tests = {}

    def add_component_test(
        self,
        test_id: str,
        component: str,
        description: str,
        result_data: Any,
        response_time: float,
        success: bool = True,
        error: Exception = None,
    ):
        """Add a component test result."""
        # Handle complex result_data that might already contain structured info
        if isinstance(result_data, dict) and "result" in result_data:
            response_data = result_data  # Already structured
        else:
            response_data = {
                "result": str(result_data),
                "type": type(result_data).__name__,
            }

        self.add_result(
            test_id=test_id,
            endpoint=component,
            description=description,
            response_data=response_data,
            response_time=response_time,
            status_code=200 if success else 500,
            success=success,
        )

        if error:
            full_traceback = (
                traceback.format_exc()
                if hasattr(traceback, "format_exc")
                else str(error)
            )
            self.add_error(
                test_id=test_id,
                endpoint=component,
                description=description,
                error=error,
                response_time=response_time,
                response_data={
                    **response_data,
                    "traceback": full_traceback,
                    "error_type": type(error).__name__,
                },
            )


def get_checkpointer_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for checkpointer functionality."""
    test_cases = [
        # Environment and Configuration Tests
        {
            "component": "check_postgres_env_vars",
            "function": check_postgres_env_vars,
            "args": [],
            "kwargs": {},
            "description": "Validate PostgreSQL environment variables",
            "should_succeed": True,
            "test_focus": "config_validation",
            "expected_type": bool,
        },
        {
            "component": "get_db_config",
            "function": get_db_config,
            "args": [],
            "kwargs": {},
            "description": "Get database configuration from environment",
            "should_succeed": True,
            "test_focus": "config_retrieval",
            "expected_type": dict,
        },
        {
            "component": "get_connection_string",
            "function": get_connection_string,
            "args": [],
            "kwargs": {},
            "description": "Generate PostgreSQL connection string",
            "should_succeed": True,
            "test_focus": "connection_string_generation",
            "expected_type": str,
        },
        # Factory Function Tests
        {
            "component": "create_async_postgres_saver",
            "function": create_async_postgres_saver,
            "args": [],
            "kwargs": {},
            "description": "Create AsyncPostgresSaver instance",
            "should_succeed": True,
            "test_focus": "checkpointer_creation",
            "expected_type": object,
            "async": True,
        },
        {
            "component": "get_global_checkpointer",
            "function": get_global_checkpointer,
            "args": [],
            "kwargs": {},
            "description": "Get global checkpointer instance",
            "should_succeed": True,
            "test_focus": "checkpointer_access",
            "expected_type": object,
            "async": True,
        },
        {
            "component": "initialize_checkpointer",
            "function": initialize_checkpointer,
            "args": [],
            "kwargs": {},
            "description": "Initialize global checkpointer",
            "should_succeed": True,
            "test_focus": "checkpointer_initialization",
            "expected_type": type(None),
            "async": True,
        },
        {
            "component": "check_pool_health_and_recreate",
            "function": check_pool_health_and_recreate,
            "args": [],
            "kwargs": {},
            "description": "Check pool health and recreate if needed",
            "should_succeed": True,
            "test_focus": "pool_health_check",
            "expected_type": bool,
            "async": True,
        },
        {
            "component": "health_check_pool_health_and_recreate",
            "function": health_check_pool_health_and_recreate,
            "args": [],
            "kwargs": {},
            "description": "Health module pool health check",
            "should_succeed": True,
            "test_focus": "health_module_check",
            "expected_type": object,  # returns coroutine
            "async": True,
        },
        {
            "component": "cleanup_checkpointer",
            "function": cleanup_checkpointer,
            "args": [],
            "kwargs": {},
            "description": "Clean up global checkpointer",
            "should_succeed": True,
            "test_focus": "checkpointer_cleanup",
            "expected_type": type(None),
            "async": True,
        },
        {
            "component": "close_async_postgres_saver",
            "function": close_async_postgres_saver,
            "args": [],
            "kwargs": {},
            "description": "Close AsyncPostgresSaver instance",
            "should_succeed": True,
            "test_focus": "checkpointer_closing",
            "expected_type": type(None),
            "async": True,
        },
    ]

    return test_cases


def get_checkpointer_error_test_cases() -> List[Dict[str, Any]]:
    """Generate error test cases for checkpointer functionality."""
    error_test_cases = [
        # Environment variable missing tests
        {
            "component": "check_postgres_env_vars",
            "function": check_postgres_env_vars,
            "args": [],
            "kwargs": {},
            "description": "Check env vars with missing variables",
            "should_succeed": False,
            "test_focus": "missing_env_vars",
            "expected_type": bool,
            "mock_env": {"user": None},  # Remove user env var
        },
        # Connection failures
        {
            "component": "create_async_postgres_saver",
            "function": create_async_postgres_saver,
            "args": [],
            "kwargs": {},
            "description": "Create saver with invalid connection",
            "should_succeed": False,
            "test_focus": "connection_failure",
            "expected_type": Exception,
            "async": True,
            "mock_env": {"host": "invalid_host_name_12345"},
        },
    ]

    return error_test_cases


def get_checkpointer_functionality_test_cases() -> List[Dict[str, Any]]:
    """Generate functional test cases that test actual checkpointer operations."""
    functionality_test_cases = [
        {
            "component": "checkpointer_get_operation",
            "function": None,  # Custom test function
            "args": [],
            "kwargs": {},
            "description": "Test checkpointer get operation",
            "should_succeed": True,
            "test_focus": "checkpointer_get",
            "expected_type": type(None),  # get() returns None for new threads
            "async": True,
            "custom_test": True,
        },
        {
            "component": "checkpointer_put_operation",
            "function": None,  # Custom test function
            "args": [],
            "kwargs": {},
            "description": "Test checkpointer put operation",
            "should_succeed": True,
            "test_focus": "checkpointer_put",
            "expected_type": object,  # put() returns RunnableConfig
            "async": True,
            "custom_test": True,
        },
        {
            "component": "checkpointer_list_operation",
            "function": None,  # Custom test function
            "args": [],
            "kwargs": {},
            "description": "Test checkpointer list operation",
            "should_succeed": True,
            "test_focus": "checkpointer_list",
            "expected_type": list,  # list() returns list of checkpoints
            "async": True,
            "custom_test": True,
        },
    ]

    return functionality_test_cases


def _get_test_explanation(test_case: Dict[str, Any]) -> str:
    """Generate detailed test explanation based on test case."""
    component = test_case["component"]
    test_focus = test_case.get("test_focus", "general")
    description = test_case["description"]
    should_succeed = test_case["should_succeed"]

    explanations = {
        "config_validation": f"🔧 {component}: Validates that all required PostgreSQL environment variables are properly set and accessible for database connections",
        "config_retrieval": f"📋 {component}: Retrieves database configuration from environment variables and formats them for connection use",
        "connection_string_generation": f"🔗 {component}: Generates optimized PostgreSQL connection string with cloud-specific parameters and unique application naming",
        "checkpointer_creation": f"🏭 {component}: Creates new AsyncPostgresSaver instance with proper connection pool and table setup",
        "checkpointer_access": f"🎯 {component}: Provides unified access to global checkpointer with lazy initialization and health checks",
        "checkpointer_initialization": f"🚀 {component}: Initializes global checkpointer with proper async context management and retry logic",
        "pool_health_check": f"🏥 {component}: Checks connection pool health and recreates if unhealthy, ensuring reliable database connectivity",
        "health_module_check": f"🩺 {component}: Health module wrapper for pool health checks, maintains API compatibility",
        "checkpointer_cleanup": f"🧹 {component}: Properly cleans up global checkpointer and closes all connections on shutdown",
        "checkpointer_closing": f"🔒 {component}: Closes AsyncPostgresSaver instance and associated connection pools safely",
        "checkpointer_get": f"📖 {component}: Tests checkpointer get operation to retrieve checkpoint data for a thread",
        "checkpointer_put": f"💾 {component}: Tests checkpointer put operation to store checkpoint data for a thread",
        "checkpointer_list": f"📝 {component}: Tests checkpointer list operation to enumerate checkpoints for a thread",
        "missing_env_vars": f"❌ {component}: Tests behavior when required environment variables are missing or invalid",
        "runtime_exception": f"🔥 {component}: Tests runtime exception handling and ensures traceback information is properly captured",
        "connection_failure": f"🔥 {component}: Tests error handling when database connection fails due to invalid parameters",
    }

    explanation = explanations.get(test_focus, f"⚡ {component}: {description}")
    success_indicator = (
        "✅ Expected Success" if should_succeed else "❌ Expected Failure"
    )

    return f"{explanation} | {success_indicator}"


async def run_checkpointer_test(
    test_id: str,
    test_case: Dict[str, Any],
    results: CheckpointerTestResults,
) -> None:
    """Run a single checkpointer test case."""
    component = test_case["component"]
    function = test_case["function"]
    args = test_case.get("args", [])
    kwargs = test_case.get("kwargs", {})
    description = test_case["description"]
    should_succeed = test_case["should_succeed"]
    expected_type = test_case.get("expected_type", object)
    is_async = test_case.get("async", False)
    is_custom_test = test_case.get("custom_test", False)
    mock_env = test_case.get("mock_env", {})

    explanation = _get_test_explanation(test_case)
    print(f"\n🧪 Test {test_id}: {explanation}")

    start_time = time.time()

    try:
        with capture_server_logs() as log_capture:
            result = None
            error = None

            # Apply environment mocking if specified
            if mock_env:
                original_env = {}
                for key, value in mock_env.items():
                    original_env[key] = os.environ.get(key)
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value

                try:
                    if is_custom_test:
                        result = await run_custom_test(component, test_case)
                    elif is_async:
                        result = await function(*args, **kwargs)
                    else:
                        result = function(*args, **kwargs)
                finally:
                    # Restore original environment
                    for key, original_value in original_env.items():
                        if original_value is None:
                            if key in os.environ:
                                del os.environ[key]
                        else:
                            os.environ[key] = original_value
            else:
                if is_custom_test:
                    result = await run_custom_test(component, test_case)
                elif is_async:
                    result = await function(*args, **kwargs)
                else:
                    result = function(*args, **kwargs)

            response_time = time.time() - start_time

            # Validate result type
            if expected_type != Exception:
                if expected_type == type(None):
                    type_check = result is None
                elif expected_type == object:
                    type_check = result is not None
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
            print(
                f"✅ Test {test_id} - EXPECTED ERROR ({response_time:.2f}s): {error_message}"
            )
            results.add_component_test(
                test_id=test_id,
                component=component,
                description=description,
                result_data=error_message,
                response_time=response_time,
                success=False,
            )


async def run_custom_test(component: str, test_case: Dict[str, Any]) -> Any:
    """Run custom test functions for complex checkpointer operations."""
    if component == "checkpointer_get_operation":
        # Test get operation
        checkpointer = await get_global_checkpointer()
        if not checkpointer:
            raise Exception("No global checkpointer available")

        result = await checkpointer.aget(TEST_CONFIG)
        return result

    elif component == "checkpointer_put_operation":
        # Test put operation
        checkpointer = await get_global_checkpointer()
        if not checkpointer:
            raise Exception("No global checkpointer available")

        # Create a simple checkpoint to save
        checkpoint_data = {
            "v": 1,
            "ts": datetime.now().isoformat(),
            "id": str(uuid.uuid4()),
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": 1},
            "versions_seen": {"test_channel": {}},
        }

        result = await checkpointer.aput(TEST_CONFIG, checkpoint_data, {})
        return result

    elif component == "checkpointer_list_operation":
        # Test list operation
        checkpointer = await get_global_checkpointer()
        if not checkpointer:
            raise Exception("No global checkpointer available")

        result = []
        async for checkpoint in checkpointer.alist(TEST_CONFIG):
            result.append(checkpoint)

        return result

    else:
        raise Exception(f"Unknown custom test component: {component}")


async def test_checkpointer_integration():
    """Test complete checkpointer integration workflow."""
    print("\n🔄 Running checkpointer integration test...")

    integration_results = CheckpointerTestResults()
    test_id = "integration_001"

    start_time = time.time()

    try:
        with capture_server_logs() as log_capture:
            # Step 1: Initialize checkpointer
            await initialize_checkpointer()

            # Step 2: Get checkpointer instance
            checkpointer = await get_global_checkpointer()
            if not checkpointer:
                raise Exception("Failed to get checkpointer instance")

            # Step 3: Test basic operations
            test_config = {
                "configurable": {"thread_id": f"integration_test_{uuid.uuid4()}"}
            }

            # Get (should return None for new thread)
            get_result = await checkpointer.aget(test_config)
            if get_result is not None:
                print(f"⚠️  Expected None but got: {get_result}")

            # Put a checkpoint
            checkpoint_data = {
                "v": 1,
                "ts": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
                "channel_values": {"test": "integration_value"},
                "channel_versions": {"test": 1},
                "versions_seen": {"test": {}},
            }

            put_result = await checkpointer.aput(test_config, checkpoint_data, {})
            if not put_result:
                raise Exception("Put operation returned None")

            # Get again (should return the checkpoint now)
            get_result_2 = await checkpointer.aget(test_config)
            if get_result_2 is None:
                raise Exception("Get operation returned None after put")

            # List checkpoints
            list_result = []
            async for checkpoint in checkpointer.alist(test_config):
                list_result.append(checkpoint)

            if not list_result:
                print("⚠️  List operation returned empty result")

            # Step 4: Health check
            health_result = await check_pool_health_and_recreate()

            response_time = time.time() - start_time

            integration_results.add_component_test(
                test_id=test_id,
                component="checkpointer_integration",
                description="Complete checkpointer workflow integration test",
                result_data={
                    "checkpointer_available": checkpointer is not None,
                    "get_before_put": get_result,
                    "put_success": put_result is not None,
                    "get_after_put": get_result_2 is not None,
                    "list_count": len(list_result),
                    "health_check": health_result,
                },
                response_time=response_time,
                success=True,
            )

            print(f"✅ Integration test - SUCCESS ({response_time:.2f}s)")

    except Exception as e:
        response_time = time.time() - start_time
        full_traceback = traceback.format_exc()
        print(f"❌ Integration test - ERROR: {e}")
        integration_results.add_error(
            test_id=test_id,
            endpoint="checkpointer_integration",
            description="Complete checkpointer workflow integration test",
            error=e,
            response_time=response_time,
            response_data={
                "error": str(e),
                "traceback": full_traceback,
                "error_type": type(e).__name__,
            },
        )

    return integration_results


async def run_checkpointer_tests() -> CheckpointerTestResults:
    """Run all checkpointer tests."""
    print("🚀 Starting checkpointer tests...")

    results = CheckpointerTestResults(required_components=REQUIRED_COMPONENTS)
    results.start_time = datetime.now()

    try:
        # Get all test cases
        basic_tests = get_checkpointer_test_cases()
        error_tests = get_checkpointer_error_test_cases()
        functionality_tests = get_checkpointer_functionality_test_cases()

        all_tests = basic_tests + error_tests + functionality_tests

        # Run all test cases
        for i, test_case in enumerate(all_tests, 1):
            test_id = f"test_{i:03d}"
            await run_checkpointer_test(test_id, test_case, results)
            await asyncio.sleep(0.1)  # Small delay between tests

        # Run integration test
        integration_results = await test_checkpointer_integration()
        results.results.extend(integration_results.results)
        results.errors.extend(integration_results.errors)

        # Test cleanup
        print("\n🧹 Running cleanup tests...")
        cleanup_start = time.time()
        try:
            await cleanup_checkpointer()
            await close_async_postgres_saver()
            cleanup_time = time.time() - cleanup_start

            results.add_component_test(
                test_id="cleanup_001",
                component="final_cleanup",
                description="Final cleanup of all checkpointer resources",
                result_data="Cleanup completed successfully",
                response_time=cleanup_time,
                success=True,
            )
            print(f"✅ Cleanup test - SUCCESS ({cleanup_time:.2f}s)")

        except Exception as e:
            cleanup_time = time.time() - cleanup_start
            full_traceback = traceback.format_exc()
            results.add_error(
                test_id="cleanup_001",
                endpoint="final_cleanup",
                description="Final cleanup of all checkpointer resources",
                error=e,
                response_time=cleanup_time,
                response_data={
                    "error": str(e),
                    "traceback": full_traceback,
                    "error_type": type(e).__name__,
                },
            )
            print(f"❌ Cleanup test - ERROR: {e}")

    except Exception as e:
        print(f"❌ Test execution error: {e}")
        full_traceback = traceback.format_exc()
        results.add_error(
            test_id="execution_error",
            endpoint="test_execution",
            description="Test execution framework error",
            error=e,
            response_time=0.0,
            response_data={
                "error": str(e),
                "traceback": full_traceback,
                "error_type": type(e).__name__,
            },
        )

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: CheckpointerTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.2f}s")

    # Component coverage
    tested_components = set()
    for result in results.results:
        tested_components.add(result["endpoint"])

    missing_components = REQUIRED_COMPONENTS - tested_components
    if missing_components:
        print(f"❌ Missing components: {', '.join(missing_components)}")
    else:
        print("✅ All required components tested")

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
    print("[START] Checkpointer Module Test Starting...")
    print(
        "[INFO] Testing: Comprehensive checkpointer functionality (factory.py and health.py)"
    )
    print(f"[INFO] Components to test: {', '.join(REQUIRED_COMPONENTS)}")

    # Check if imports are available
    if not CHECKPOINTER_AVAILABLE:
        print(f"[ERROR] Checkpointer modules not available: {IMPORT_ERROR}")
        print(
            "[INFO] This could be due to missing dependencies or environment configuration"
        )
        print("\n[INFO] Missing dependencies likely include:")
        print("  - psycopg (PostgreSQL adapter)")
        print("  - langgraph (for AsyncPostgresSaver)")
        print("  - psycopg_pool (for connection pooling)")
        print("\n[INFO] If dependencies were available, this test would:")
        print("  - Test environment variable validation")
        print("  - Test database configuration retrieval")
        print("  - Test connection string generation")
        print("  - Test checkpointer creation and initialization")
        print("  - Test pool health checks and recreation")
        print("  - Test basic CRUD operations (get, put, list)")
        print("  - Test error handling and retry logic")
        print("  - Test cleanup and resource management")
        print("  - Generate detailed traceback reports")

        print(
            "\n[INFO] Running limited tests to demonstrate error handling and traceback capture..."
        )

        # Still run basic import tests to show what's missing
        results = CheckpointerTestResults(required_components=REQUIRED_COMPONENTS)
        results.start_time = datetime.now()

        # Create exception with traceback info
        import_exception = Exception(IMPORT_ERROR)
        results.add_error(
            test_id="import_error",
            endpoint="module_import",
            description="Failed to import checkpointer modules",
            error=import_exception,
            response_time=0.0,
            response_data={
                "import_error": IMPORT_ERROR,
                "traceback": f"ImportError: {IMPORT_ERROR}\n  Module dependencies missing: psycopg, langgraph, psycopg_pool",
                "error_type": "ImportError",
            },
        )

        # Traceback capture is now properly integrated into all real tests

        results.end_time = datetime.now()
        analyze_test_results(results)
        return False

    # Check environment setup
    if not check_postgres_env_vars():
        print("[WARNING] Required PostgreSQL environment variables are not set!")
        print("[INFO] Required variables: host, port, dbname, user, password")
        print("[INFO] Tests will run but may fail due to missing configuration")

    try:
        # Setup debug environment (not HTTP-based, so we'll just set env vars)
        debug_vars = {
            "print__checkpointers_debug": "1",
            "print__checkpointer_flow": "1",
            "DEBUG_TRACEBACK": "1",
        }

        original_env = {}
        for key, value in debug_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Run tests
            results = await run_checkpointer_tests()

            # Analyze results
            summary = analyze_test_results(results)

            # Determine success
            success = summary["success_rate"] >= 70.0  # 70% success rate threshold

            if success:
                print(
                    f"\n🎉 [SUCCESS] Checkpointer tests completed with {summary['success_rate']:.1f}% success rate!"
                )
            else:
                print(
                    f"\n💥 [FAILURE] Checkpointer tests failed with {summary['success_rate']:.1f}% success rate!"
                )

            return success

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = original_value

    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        save_traceback_report(
            report_type="exception",
            exception=e,
            test_context={
                "test_type": "checkpointer_module",
                "test_file": "test_checkpointer_checkpointer.py",
                "error_location": "main_execution",
                "checkpointer_available": CHECKPOINTER_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        save_traceback_report(
            report_type="exception",
            exception=e,
            test_context={
                "test_type": "checkpointer_module",
                "test_file": "test_checkpointer_checkpointer.py",
                "error_location": "script_execution",
                "checkpointer_available": CHECKPOINTER_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )
        sys.exit(1)

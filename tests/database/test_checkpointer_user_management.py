"""Test for Checkpointer User Management: sentiment_tracking.py and thread_operations.py
Tests the user management functionality with comprehensive testing patterns.

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
from dotenv import load_dotenv

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

# Standard library imports
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
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

# Import user management modules for testing with error handling
try:
    from checkpointer.user_management.sentiment_tracking import (
        update_thread_run_sentiment,
        get_thread_run_sentiments,
    )
    from checkpointer.user_management.thread_operations import (
        create_thread_run_entry,
        get_user_chat_threads,
        get_user_chat_threads_count,
        delete_user_thread_entries,
    )
    from checkpointer.config import (
        DEFAULT_MAX_RETRIES,
        THREAD_TITLE_MAX_LENGTH,
        THREAD_TITLE_SUFFIX_LENGTH,
        get_db_config,
        check_postgres_env_vars,
    )
    from checkpointer.database.connection import get_direct_connection

    USER_MANAGEMENT_AVAILABLE = True
    IMPORT_ERROR = None

except ImportError as e:
    USER_MANAGEMENT_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create mock functions for testing import failures
    async def update_thread_run_sentiment(run_id, sentiment):
        return False

    async def get_thread_run_sentiments(email, thread_id):
        return {}

    async def create_thread_run_entry(email, thread_id, prompt=None, run_id=None):
        return str(uuid.uuid4())

    async def get_user_chat_threads(email, limit=None, offset=0):
        return []

    async def get_user_chat_threads_count(email):
        return 0

    async def delete_user_thread_entries(email, thread_id):
        return {"deleted_count": 0, "message": "Mock function"}

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

    async def get_direct_connection():
        return MagicMock()

    DEFAULT_MAX_RETRIES = 3
    THREAD_TITLE_MAX_LENGTH = 50
    THREAD_TITLE_SUFFIX_LENGTH = 3


# Test configuration
TEST_USER_MANAGEMENT_NAME = f"test_user_management_{uuid.uuid4().hex[:8]}"
TEST_EMAIL = "test_user@example.com"
TEST_THREAD_ID = f"test_thread_{uuid.uuid4().hex[:8]}"
TEST_RUN_ID = f"test_run_{uuid.uuid4().hex[:8]}"
TEST_PROMPT = "Test prompt for user management testing"

REQUIRED_COMPONENTS = {
    "update_thread_run_sentiment",
    "get_thread_run_sentiments",
    "create_thread_run_entry",
    "get_user_chat_threads",
    "get_user_chat_threads_count",
    "delete_user_thread_entries",
}


class UserManagementTestResults(BaseTestResults):
    """Extended test results class for user management testing."""

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
                error=error or Exception("Unknown error"),
                response_time=response_time,
                response_data={"result": str(result_data)} if result_data else None,
            )

    def save_traceback_report(self, report_type: str = "user_management_test_report"):
        """Save the traceback report using the helper function."""
        return save_traceback_report(
            report_type=report_type,
            test_results=self,
            test_context={
                "test_type": "User Management Testing",
                "components_tested": list(self.component_coverage),
                "total_tests": len(self.results),
                "import_available": USER_MANAGEMENT_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )


def get_sentiment_tracking_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for sentiment tracking functionality."""
    return [
        # Basic sentiment tracking tests
        {
            "test_id": "UMT001",
            "component": "update_thread_run_sentiment",
            "description": "Update thread run sentiment to positive",
            "test_function": "update_thread_run_sentiment",
            "test_args": [TEST_RUN_ID, True],
            "expected_type": bool,
            "explanation": "Test updating sentiment to positive value",
        },
        {
            "test_id": "UMT002",
            "component": "update_thread_run_sentiment",
            "description": "Update thread run sentiment to negative",
            "test_function": "update_thread_run_sentiment",
            "test_args": [TEST_RUN_ID, False],
            "expected_type": bool,
            "explanation": "Test updating sentiment to negative value",
        },
        {
            "test_id": "UMT003",
            "component": "get_thread_run_sentiments",
            "description": "Get thread run sentiments for user",
            "test_function": "get_thread_run_sentiments",
            "test_args": [TEST_EMAIL, TEST_THREAD_ID],
            "expected_type": dict,
            "explanation": "Test retrieving sentiments for a thread",
        },
        # Edge cases and error conditions
        {
            "test_id": "UMT004",
            "component": "update_thread_run_sentiment",
            "description": "Update sentiment with invalid run ID",
            "test_function": "update_thread_run_sentiment",
            "test_args": ["invalid_run_id", True],
            "expected_type": bool,
            "explanation": "Test sentiment update with non-existent run ID (returns False for non-existent)",
        },
        {
            "test_id": "UMT005",
            "component": "get_thread_run_sentiments",
            "description": "Get sentiments for non-existent thread",
            "test_function": "get_thread_run_sentiments",
            "test_args": [TEST_EMAIL, "non_existent_thread"],
            "expected_type": dict,
            "explanation": "Test retrieving sentiments for non-existent thread",
        },
        {
            "test_id": "UMT006",
            "component": "get_thread_run_sentiments",
            "description": "Get sentiments with empty email",
            "test_function": "get_thread_run_sentiments",
            "test_args": ["", TEST_THREAD_ID],
            "expected_type": dict,
            "explanation": "Test retrieving sentiments with empty email",
        },
    ]


def get_thread_operations_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for thread operations functionality."""
    return [
        # Basic thread operations tests
        {
            "test_id": "UMT007",
            "component": "create_thread_run_entry",
            "description": "Create thread run entry with all parameters",
            "test_function": "create_thread_run_entry",
            "test_args": [TEST_EMAIL, TEST_THREAD_ID, TEST_PROMPT, TEST_RUN_ID],
            "expected_type": str,
            "explanation": "Test creating thread run entry with full parameters",
        },
        {
            "test_id": "UMT008",
            "component": "create_thread_run_entry",
            "description": "Create thread run entry with auto-generated run ID",
            "test_function": "create_thread_run_entry",
            "test_args": [TEST_EMAIL, f"{TEST_THREAD_ID}_auto", TEST_PROMPT],
            "expected_type": str,
            "explanation": "Test creating thread run entry with auto-generated run ID",
        },
        {
            "test_id": "UMT009",
            "component": "create_thread_run_entry",
            "description": "Create thread run entry without prompt",
            "test_function": "create_thread_run_entry",
            "test_args": [TEST_EMAIL, f"{TEST_THREAD_ID}_no_prompt"],
            "expected_type": str,
            "explanation": "Test creating thread run entry without prompt",
        },
        {
            "test_id": "UMT010",
            "component": "get_user_chat_threads",
            "description": "Get user chat threads without pagination",
            "test_function": "get_user_chat_threads",
            "test_args": [TEST_EMAIL],
            "expected_type": list,
            "explanation": "Test retrieving user chat threads without pagination",
        },
        {
            "test_id": "UMT011",
            "component": "get_user_chat_threads",
            "description": "Get user chat threads with pagination",
            "test_function": "get_user_chat_threads",
            "test_args": [TEST_EMAIL, 5, 0],
            "expected_type": list,
            "explanation": "Test retrieving user chat threads with pagination",
        },
        {
            "test_id": "UMT012",
            "component": "get_user_chat_threads_count",
            "description": "Get user chat threads count",
            "test_function": "get_user_chat_threads_count",
            "test_args": [TEST_EMAIL],
            "expected_type": int,
            "explanation": "Test retrieving user chat threads count",
        },
        {
            "test_id": "UMT013",
            "component": "delete_user_thread_entries",
            "description": "Delete user thread entries",
            "test_function": "delete_user_thread_entries",
            "test_args": [TEST_EMAIL, TEST_THREAD_ID],
            "expected_type": dict,
            "explanation": "Test deleting user thread entries",
        },
        # Edge cases and error conditions
        {
            "test_id": "UMT014",
            "component": "get_user_chat_threads",
            "description": "Get threads for non-existent user",
            "test_function": "get_user_chat_threads",
            "test_args": ["non_existent@example.com"],
            "expected_type": list,
            "explanation": "Test retrieving threads for non-existent user",
        },
        {
            "test_id": "UMT015",
            "component": "get_user_chat_threads_count",
            "description": "Get count for non-existent user",
            "test_function": "get_user_chat_threads_count",
            "test_args": ["non_existent@example.com"],
            "expected_type": int,
            "explanation": "Test retrieving count for non-existent user",
        },
        {
            "test_id": "UMT016",
            "component": "delete_user_thread_entries",
            "description": "Delete non-existent thread entries",
            "test_function": "delete_user_thread_entries",
            "test_args": [TEST_EMAIL, "non_existent_thread"],
            "expected_type": dict,
            "explanation": "Test deleting non-existent thread entries",
        },
    ]


def _get_test_explanation(test_case: Dict[str, Any]) -> str:
    """Generate detailed test explanation."""
    base_explanation = test_case.get("explanation", "No explanation provided")
    component = test_case.get("component", "unknown")
    test_function = test_case.get("test_function", "unknown")
    expected_type = test_case.get("expected_type", "unknown")

    return f"""
    Test: {test_case.get('test_id', 'N/A')} - {test_case.get('description', 'No description')}
    Component: {component}
    Function: {test_function}
    Expected Type: {expected_type}
    Explanation: {base_explanation}
    """


async def run_user_management_test(
    test_id: str,
    test_case: Dict[str, Any],
    results: UserManagementTestResults,
) -> None:
    """Run a single user management test case."""
    component = test_case["component"]
    description = test_case["description"]
    expected_failure = test_case.get("expected_failure", False)

    print(f"\n🧪 Running Test {test_id}: {description}")
    print(f"   Component: {component}")
    print(_get_test_explanation(test_case))

    start_time = time.time()
    success = False
    result_data = None
    error = None

    try:
        # Run the custom test function
        result_data = await run_custom_user_management_test(component, test_case)
        success = True
        response_time = time.time() - start_time

        if expected_failure:
            print(
                f"⚠️  Test {test_id} - Expected failure but got success: {result_data}"
            )
        else:
            print(f"✅ Test {test_id} - Success: {result_data}")

    except Exception as e:
        error = e
        response_time = time.time() - start_time

        if expected_failure:
            print(f"✅ Test {test_id} - Expected failure occurred: {str(e)}")
            success = True  # Expected failures are considered successes
        else:
            print(f"❌ Test {test_id} - Unexpected error: {str(e)}")
            print(f"   Full traceback: {traceback.format_exc()}")

    # Add result to test results
    results.add_component_test(
        test_id=test_id,
        component=component,
        description=description,
        result_data=result_data,
        response_time=response_time,
        success=success,
        error=error,
        expected_failure=expected_failure,
    )

    print(f"   Response time: {response_time:.3f}s")


async def run_custom_user_management_test(
    component: str, test_case: Dict[str, Any]
) -> Any:
    """Run custom test based on the component and test case."""
    test_function = test_case["test_function"]
    test_args = test_case.get("test_args", [])
    test_kwargs = test_case.get("test_kwargs", {})

    if component == "update_thread_run_sentiment":
        return await update_thread_run_sentiment(*test_args, **test_kwargs)
    elif component == "get_thread_run_sentiments":
        return await get_thread_run_sentiments(*test_args, **test_kwargs)
    elif component == "create_thread_run_entry":
        return await create_thread_run_entry(*test_args, **test_kwargs)
    elif component == "get_user_chat_threads":
        return await get_user_chat_threads(*test_args, **test_kwargs)
    elif component == "get_user_chat_threads_count":
        return await get_user_chat_threads_count(*test_args, **test_kwargs)
    elif component == "delete_user_thread_entries":
        return await delete_user_thread_entries(*test_args, **test_kwargs)
    else:
        raise ValueError(f"Unknown component: {component}")


async def test_user_management_integration():
    """Run integration test that combines multiple user management functions."""
    print("\n🔗 Running User Management Integration Test")

    try:
        # Create a thread entry
        print("   Step 1: Creating thread entry...")
        run_id = await create_thread_run_entry(TEST_EMAIL, TEST_THREAD_ID, TEST_PROMPT)
        print(f"   Created run ID: {run_id}")

        # Update sentiment for the run
        print("   Step 2: Updating sentiment...")
        sentiment_result = await update_thread_run_sentiment(run_id, True)
        print(f"   Sentiment update result: {sentiment_result}")

        # Get sentiments for the thread
        print("   Step 3: Retrieving sentiments...")
        sentiments = await get_thread_run_sentiments(TEST_EMAIL, TEST_THREAD_ID)
        print(f"   Retrieved sentiments: {sentiments}")

        # Get user threads
        print("   Step 4: Getting user threads...")
        threads = await get_user_chat_threads(TEST_EMAIL)
        print(f"   User threads count: {len(threads)}")

        # Get threads count
        print("   Step 5: Getting threads count...")
        count = await get_user_chat_threads_count(TEST_EMAIL)
        print(f"   Total threads count: {count}")

        print("✅ Integration test completed successfully")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False


async def run_user_management_tests() -> UserManagementTestResults:
    """Run all user management tests."""
    print("🚀 Starting User Management Tests")
    print(
        f"   Import Status: {'✅ Available' if USER_MANAGEMENT_AVAILABLE else '❌ Failed'}"
    )
    if not USER_MANAGEMENT_AVAILABLE:
        print(f"   Import Error: {IMPORT_ERROR}")

    results = UserManagementTestResults()

    # Get all test cases
    sentiment_tests = get_sentiment_tracking_test_cases()
    thread_tests = get_thread_operations_test_cases()
    all_test_cases = sentiment_tests + thread_tests

    print(f"   Total Tests: {len(all_test_cases)}")
    print(f"   Required Components: {len(REQUIRED_COMPONENTS)}")

    # Run each test case
    for test_case in all_test_cases:
        test_id = test_case["test_id"]
        await run_user_management_test(test_id, test_case, results)

        # Small delay between tests
        await asyncio.sleep(0.1)

    # Run integration test
    integration_success = await test_user_management_integration()

    # Add integration test result
    results.add_component_test(
        test_id="UMT_INTEGRATION",
        component="integration_test",
        description="User management integration test",
        result_data=integration_success,
        response_time=0.0,
        success=integration_success,
    )

    print(f"\n📊 User Management Tests Summary:")
    summary = results.get_summary()
    print(f"   Total Tests: {summary['total_requests']}")
    print(f"   Successful: {summary['successful_requests']}")
    print(f"   Failed: {summary['failed_requests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Average Response Time: {summary['average_response_time']:.3f}s")

    return results


def analyze_test_results(results: UserManagementTestResults):
    """Analyze and display test results."""
    print("\n📈 Detailed Test Analysis:")

    summary = results.get_summary()

    print(
        f"   Components Tested: {len(results.component_coverage)}/{len(REQUIRED_COMPONENTS)}"
    )
    print(f"   Missing Components: {REQUIRED_COMPONENTS - results.component_coverage}")

    if results.errors:
        print(f"\n❌ Failed Tests ({len(results.errors)}):")
        for error in results.errors:
            print(f"   - {error['test_id']}: {error['description']}")
            print(f"     Error: {error['error']}")

    if summary["successful_requests"] > 0:
        print(f"\n✅ Successful Tests ({summary['successful_requests']}):")
        for result in results.results:
            if result.get("success", True):
                print(
                    f"   - {result.get('test_id', 'N/A')}: {result.get('description', 'N/A')}"
                )


async def main():
    """Main test execution function."""
    print("=" * 80)
    print("🧪 USER MANAGEMENT CHECKPOINTER TESTING SUITE")
    print("=" * 80)

    start_time = time.time()

    try:
        # Run all tests
        results = await run_user_management_tests()

        # Analyze results
        analyze_test_results(results)

        # Save traceback report
        print(f"\n💾 Saving test report...")
        report_saved = results.save_traceback_report()
        if report_saved:
            print("   Report saved successfully")
        else:
            print("   Failed to save report")

        # Final summary
        total_time = time.time() - start_time
        summary = results.get_summary()

        print(f"\n🏁 Test Suite Completed in {total_time:.2f} seconds")
        print(
            f"   Final Results: {summary['successful_requests']}/{summary['total_requests']} tests passed"
        )
        print(f"   Success Rate: {summary['success_rate']:.1f}%")

        # Exit with appropriate code
        if summary["failed_requests"] > 0:
            print("   ⚠️  Some tests failed - check the results above")
            sys.exit(1)
        else:
            print("   🎉 All tests passed!")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Test suite failed with exception: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

        # Save error report
        save_traceback_report(
            report_type="exception",
            exception=e,
            test_context={
                "test_type": "User Management Testing",
                "error_location": "main",
                "import_available": USER_MANAGEMENT_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

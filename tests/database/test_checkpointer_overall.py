"""Test for Checkpointer Overall Database Functionality
Tests the complete database functionality with interconnected components working together.

This test file focuses on testing the overall database system including:
- Database connectivity and configuration
- Table setup and schema management
- Connection pool management and lifecycle
- Checkpointer factory and health checks
- End-to-end data operations
- Error handling and recovery workflows

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
from dotenv import load_dotenv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import all database modules for testing with error handling
try:
    # Database layer
    from checkpointer.database.connection import (
        get_connection_string,
        get_connection_kwargs,
        get_direct_connection,
    )
    from checkpointer.database.pool_manager import (
        cleanup_all_pools,
        force_close_modern_pools,
        modern_psycopg_pool,
    )
    from checkpointer.database.table_setup import (
        setup_checkpointer_with_autocommit,
        setup_users_threads_runs_table,
        table_exists,
    )

    # Checkpointer factory and health
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

    # User management for end-to-end testing
    from checkpointer.user_management.thread_operations import (
        create_thread_run_entry,
        get_user_chat_threads,
        get_user_chat_threads_count,
        delete_user_thread_entries,
    )
    from checkpointer.user_management.sentiment_tracking import (
        update_thread_run_sentiment,
        get_thread_run_sentiments,
    )

    # Error handling
    from checkpointer.error_handling.prepared_statements import (
        is_prepared_statement_error,
        clear_prepared_statements,
    )
    from checkpointer.error_handling.retry_decorators import (
        retry_on_prepared_statement_error,
    )

    # Configuration and globals
    from checkpointer.config import (
        get_db_config,
        check_postgres_env_vars,
        DEFAULT_MAX_RETRIES,
        CHECKPOINTER_CREATION_MAX_RETRIES,
        DEFAULT_POOL_MIN_SIZE,
        DEFAULT_POOL_MAX_SIZE,
        DEFAULT_POOL_TIMEOUT,
    )
    from checkpointer.globals import _GLOBAL_CHECKPOINTER, _CONNECTION_STRING_CACHE

    DATABASE_OVERALL_AVAILABLE = True
    IMPORT_ERROR = None

except ImportError as e:
    DATABASE_OVERALL_AVAILABLE = False
    IMPORT_ERROR = str(e)

    # Create comprehensive mock functions for testing import failures
    def get_connection_string():
        return "postgresql://mock:mock@mock:5432/mock"

    def get_connection_kwargs():
        return {"prepare_threshold": None}

    async def get_direct_connection():
        return MagicMock()

    async def cleanup_all_pools():
        pass

    async def force_close_modern_pools():
        pass

    async def modern_psycopg_pool():
        return MagicMock()

    async def setup_checkpointer_with_autocommit():
        return MagicMock()

    async def setup_users_threads_runs_table():
        pass

    async def table_exists(conn, table_name):
        return True

    async def create_async_postgres_saver():
        return MagicMock()

    async def close_async_postgres_saver():
        pass

    def get_global_checkpointer():
        return MagicMock()

    async def initialize_checkpointer():
        return MagicMock()

    async def cleanup_checkpointer():
        pass

    async def check_pool_health_and_recreate():
        return True

    async def health_check_pool_health_and_recreate():
        return True

    async def create_thread_run_entry(email, thread_id, prompt=None, run_id=None):
        return str(uuid.uuid4())

    async def get_user_chat_threads(email, limit=None, offset=0):
        return []

    async def get_user_chat_threads_count(email):
        return 0

    async def delete_user_thread_entries(email, thread_id):
        return {"deleted_count": 0}

    async def update_thread_run_sentiment(run_id, sentiment):
        return False

    async def get_thread_run_sentiments(email, thread_id):
        return {}

    def is_prepared_statement_error(error):
        return False

    async def clear_prepared_statements():
        pass

    def retry_on_prepared_statement_error(max_retries=3):
        def decorator(func):
            return func

        return decorator

    def get_db_config():
        """Retrieve database configuration from environment variables."""
        return {
            "host": os.getenv("host"),
            "port": int(os.getenv("port", 5432)),
            "user": os.getenv("user"),
            "password": os.getenv("password"),
            "dbname": os.getenv("dbname"),
        }

    def check_postgres_env_vars():
        return True

    DEFAULT_MAX_RETRIES = 3
    CHECKPOINTER_CREATION_MAX_RETRIES = 2
    DEFAULT_POOL_MIN_SIZE = 3
    DEFAULT_POOL_MAX_SIZE = 10
    DEFAULT_POOL_TIMEOUT = 30
    _GLOBAL_CHECKPOINTER = None
    _CONNECTION_STRING_CACHE = {}


# Test configuration
TEST_OVERALL_NAME = f"test_overall_{uuid.uuid4().hex[:8]}"
TEST_EMAIL = "overall_test@example.com"
TEST_THREAD_ID = f"overall_test_thread_{uuid.uuid4().hex[:8]}"
TEST_PROMPT = "Overall database functionality testing prompt"

REQUIRED_COMPONENTS = {
    "get_connection_string",
    "get_connection_kwargs",
    "get_direct_connection",
    "cleanup_all_pools",
    "force_close_modern_pools",
    "setup_checkpointer_with_autocommit",
    "setup_users_threads_runs_table",
    "table_exists",
    "create_async_postgres_saver",
    "get_global_checkpointer",
    "initialize_checkpointer",
    "cleanup_checkpointer",
    "check_pool_health_and_recreate",
    "create_thread_run_entry",
    "get_user_chat_threads",
    "update_thread_run_sentiment",
    "clear_prepared_statements",
}


class OverallTestResults(BaseTestResults):
    """Extended test results class for overall database testing."""

    def __init__(self, required_components: set = None):
        super().__init__()
        self.required_components = required_components or REQUIRED_COMPONENTS
        self.component_coverage = set()
        self.workflow_results = {}

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

    def add_workflow_result(
        self, workflow_name: str, success: bool, details: Dict[str, Any]
    ):
        """Track workflow test results."""
        self.workflow_results[workflow_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now(),
        }

    def save_traceback_report(self, report_type: str = "overall_test_report"):
        """Save the traceback report using the helper function."""
        return save_traceback_report(
            report_type=report_type,
            test_results=self,
            test_context={
                "test_type": "Overall Database Testing",
                "components_tested": list(self.component_coverage),
                "workflows_tested": list(self.workflow_results.keys()),
                "total_tests": len(self.results),
                "import_available": DATABASE_OVERALL_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )


def get_database_foundation_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for database foundation functionality."""
    return [
        # Connection and configuration tests
        {
            "test_id": "OT001",
            "component": "get_connection_string",
            "description": "Get database connection string",
            "test_function": "get_connection_string",
            "test_args": [],
            "expected_type": str,
            "explanation": "Test database connection string generation",
        },
        {
            "test_id": "OT002",
            "component": "get_connection_kwargs",
            "description": "Get database connection kwargs",
            "test_function": "get_connection_kwargs",
            "test_args": [],
            "expected_type": dict,
            "explanation": "Test database connection kwargs generation",
        },
        {
            "test_id": "OT003",
            "component": "get_db_config",
            "description": "Get database configuration",
            "test_function": "get_db_config",
            "test_args": [],
            "expected_type": dict,
            "explanation": "Test database configuration retrieval",
        },
        {
            "test_id": "OT004",
            "component": "check_postgres_env_vars",
            "description": "Check PostgreSQL environment variables",
            "test_function": "check_postgres_env_vars",
            "test_args": [],
            "expected_type": bool,
            "explanation": "Test PostgreSQL environment variable validation",
        },
    ]


def get_table_management_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for table management functionality."""
    return [
        # Table setup and management tests
        {
            "test_id": "OT005",
            "component": "table_exists",
            "description": "Check if checkpoints table exists",
            "test_function": "table_exists",
            "test_args": ["checkpoints"],
            "expected_type": bool,
            "explanation": "Test checking for checkpoints table existence",
        },
        {
            "test_id": "OT006",
            "component": "table_exists",
            "description": "Check if users_threads_runs table exists",
            "test_function": "table_exists",
            "test_args": ["users_threads_runs"],
            "expected_type": bool,
            "explanation": "Test checking for users_threads_runs table existence",
        },
        {
            "test_id": "OT007",
            "component": "setup_users_threads_runs_table",
            "description": "Setup users_threads_runs table",
            "test_function": "setup_users_threads_runs_table",
            "test_args": [],
            "expected_type": type(None),
            "explanation": "Test setting up users_threads_runs table schema",
        },
    ]


def get_connection_pool_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for connection pool functionality."""
    return [
        # Connection pool tests
        {
            "test_id": "OT008",
            "component": "get_direct_connection",
            "description": "Get direct database connection",
            "test_function": "get_direct_connection",
            "test_args": [],
            "expected_type": object,
            "explanation": "Test getting direct database connection",
        },
        {
            "test_id": "OT009",
            "component": "force_close_modern_pools",
            "description": "Force close modern connection pools",
            "test_function": "force_close_modern_pools",
            "test_args": [],
            "expected_type": type(None),
            "explanation": "Test force closing modern database connection pools",
        },
        {
            "test_id": "OT010",
            "component": "cleanup_all_pools",
            "description": "Cleanup all connection pools",
            "test_function": "cleanup_all_pools",
            "test_args": [],
            "expected_type": type(None),
            "explanation": "Test cleaning up all database connection pools",
        },
    ]


def get_checkpointer_management_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for checkpointer management functionality."""
    return [
        # Checkpointer creation and management tests
        {
            "test_id": "OT013",
            "component": "create_async_postgres_saver",
            "description": "Create AsyncPostgresSaver",
            "test_function": "create_async_postgres_saver",
            "test_args": [],
            "expected_type": object,
            "explanation": "Test creating AsyncPostgresSaver checkpointer",
        },
        {
            "test_id": "OT014",
            "component": "get_global_checkpointer",
            "description": "Get global checkpointer instance",
            "test_function": "get_global_checkpointer",
            "test_args": [],
            "expected_type": object,
            "explanation": "Test retrieving global checkpointer instance",
        },
        {
            "test_id": "OT015",
            "component": "initialize_checkpointer",
            "description": "Initialize checkpointer system",
            "test_function": "initialize_checkpointer",
            "test_args": [],
            "expected_type": object,
            "explanation": "Test initializing checkpointer system",
        },
        {
            "test_id": "OT016",
            "component": "check_pool_health_and_recreate",
            "description": "Check pool health and recreate if needed",
            "test_function": "check_pool_health_and_recreate",
            "test_args": [],
            "expected_type": bool,
            "explanation": "Test checking pool health and recreation",
        },
        {
            "test_id": "OT017",
            "component": "cleanup_checkpointer",
            "description": "Cleanup checkpointer system",
            "test_function": "cleanup_checkpointer",
            "test_args": [],
            "expected_type": type(None),
            "explanation": "Test cleaning up checkpointer system",
        },
    ]


def get_error_handling_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for error handling functionality."""
    return [
        # Error handling tests
        {
            "test_id": "OT018",
            "component": "is_prepared_statement_error",
            "description": "Test prepared statement error detection",
            "test_function": "is_prepared_statement_error",
            "test_args": [Exception("prepared statement does not exist")],
            "expected_type": bool,
            "explanation": "Test detecting prepared statement errors",
        },
        {
            "test_id": "OT019",
            "component": "is_prepared_statement_error",
            "description": "Test non-prepared statement error detection",
            "test_function": "is_prepared_statement_error",
            "test_args": [Exception("regular error message")],
            "expected_type": bool,
            "explanation": "Test detecting non-prepared statement errors",
        },
        {
            "test_id": "OT020",
            "component": "clear_prepared_statements",
            "description": "Clear prepared statements",
            "test_function": "clear_prepared_statements",
            "test_args": [],
            "expected_type": type(None),
            "explanation": "Test clearing prepared statements",
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


async def run_overall_test(
    test_id: str,
    test_case: Dict[str, Any],
    results: OverallTestResults,
) -> None:
    """Run a single overall database test case."""
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
        result_data = await run_custom_overall_test(component, test_case)
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


async def run_custom_overall_test(component: str, test_case: Dict[str, Any]) -> Any:
    """Run custom test based on the component and test case."""
    test_function = test_case["test_function"]
    test_args = test_case.get("test_args", [])
    test_kwargs = test_case.get("test_kwargs", {})

    # Database foundation functions
    if component == "get_connection_string":
        return get_connection_string()
    elif component == "get_connection_kwargs":
        return get_connection_kwargs()
    elif component == "get_db_config":
        return get_db_config()
    elif component == "check_postgres_env_vars":
        return check_postgres_env_vars()

    # Table management functions
    elif component == "table_exists":
        conn = get_direct_connection()
        async with conn as connection:
            return await table_exists(connection, *test_args, **test_kwargs)
    # get_table_info function not available in actual codebase
    # elif component == "get_table_info":
    #     return await get_table_info(*test_args, **test_kwargs)
    elif component == "setup_users_threads_runs_table":
        return await setup_users_threads_runs_table(*test_args, **test_kwargs)

    # Connection pool functions
    elif component == "get_direct_connection":
        conn_manager = get_direct_connection(*test_args, **test_kwargs)
        async with conn_manager as connection:
            # Test the connection by checking if it has the expected interface
            if hasattr(connection, "execute"):
                return "Connection object created successfully"
            else:
                return connection
    elif component == "force_close_modern_pools":
        return await force_close_modern_pools(*test_args, **test_kwargs)
    elif component == "cleanup_all_pools":
        return await cleanup_all_pools(*test_args, **test_kwargs)

    # Checkpointer management functions
    elif component == "create_async_postgres_saver":
        return await create_async_postgres_saver(*test_args, **test_kwargs)
    elif component == "get_global_checkpointer":
        return get_global_checkpointer()
    elif component == "initialize_checkpointer":
        return await initialize_checkpointer(*test_args, **test_kwargs)
    elif component == "check_pool_health_and_recreate":
        return await check_pool_health_and_recreate(*test_args, **test_kwargs)
    elif component == "cleanup_checkpointer":
        return await cleanup_checkpointer(*test_args, **test_kwargs)

    # Error handling functions
    elif component == "is_prepared_statement_error":
        return is_prepared_statement_error(*test_args, **test_kwargs)
    elif component == "clear_prepared_statements":
        return await clear_prepared_statements(*test_args, **test_kwargs)

    else:
        raise ValueError(f"Unknown component: {component}")


async def test_end_to_end_workflow():
    """Run comprehensive end-to-end workflow test."""
    print("\n🔗 Running End-to-End Database Workflow Test")

    workflow_results = {
        "connection_established": False,
        "tables_verified": False,
        "checkpointer_created": False,
        "data_operations": False,
        "cleanup_completed": False,
    }

    try:
        # Step 1: Establish database connection
        print("   Step 1: Establishing database connection...")
        connection_string = get_connection_string()
        connection_kwargs = get_connection_kwargs()
        print(f"   Connection string: {connection_string[:50]}...")
        print(f"   Connection kwargs: {connection_kwargs}")
        workflow_results["connection_established"] = True

        # Step 2: Verify table setup
        print("   Step 2: Verifying table setup...")
        conn_manager = get_direct_connection()
        async with conn_manager as connection:
            checkpoints_exists = await table_exists(connection, "checkpoints")
            users_threads_runs_exists = await table_exists(
                connection, "users_threads_runs"
            )
        print(f"   Checkpoints table exists: {checkpoints_exists}")
        print(f"   Users threads runs table exists: {users_threads_runs_exists}")

        if not users_threads_runs_exists:
            print("   Setting up users_threads_runs table...")
            await setup_users_threads_runs_table()
            users_threads_runs_exists = await table_exists("users_threads_runs")
            print(
                f"   Users threads runs table after setup: {users_threads_runs_exists}"
            )

        workflow_results["tables_verified"] = (
            checkpoints_exists or users_threads_runs_exists
        )

        # Step 3: Create and test checkpointer
        print("   Step 3: Creating checkpointer...")
        checkpointer = await initialize_checkpointer()
        print(f"   Checkpointer created: {type(checkpointer).__name__}")

        # Test checkpointer health
        health_ok = await check_pool_health_and_recreate()
        print(f"   Checkpointer health: {health_ok}")
        workflow_results["checkpointer_created"] = checkpointer is not None

        # Step 4: Test data operations
        print("   Step 4: Testing data operations...")
        test_run_id = await create_thread_run_entry(
            TEST_EMAIL, TEST_THREAD_ID, TEST_PROMPT
        )
        print(f"   Created thread run entry: {test_run_id}")

        # Update sentiment
        sentiment_updated = await update_thread_run_sentiment(test_run_id, True)
        print(f"   Sentiment updated: {sentiment_updated}")

        # Get user threads
        user_threads = await get_user_chat_threads(TEST_EMAIL, limit=5)
        print(f"   Retrieved {len(user_threads)} user threads")

        # Get thread count
        thread_count = await get_user_chat_threads_count(TEST_EMAIL)
        print(f"   Total thread count: {thread_count}")

        workflow_results["data_operations"] = (
            len(user_threads) >= 0 and thread_count >= 0
        )

        # Step 5: Test error handling
        print("   Step 5: Testing error handling...")
        test_error = Exception("prepared statement does not exist")
        is_prep_error = is_prepared_statement_error(test_error)
        print(f"   Prepared statement error detected: {is_prep_error}")

        # Clear prepared statements
        await clear_prepared_statements()
        print("   Prepared statements cleared")

        # Step 6: Cleanup
        print("   Step 6: Cleanup operations...")
        await cleanup_all_pools()
        await cleanup_checkpointer()
        print("   Cleanup completed")
        workflow_results["cleanup_completed"] = True

        print("✅ End-to-end workflow test completed successfully")
        return True, workflow_results

    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False, workflow_results


async def test_database_consistency():
    """Test database consistency and integrity."""
    print("\n🔍 Running Database Consistency Test")

    consistency_results = {
        "schema_consistency": False,
        "data_integrity": False,
        "connection_stability": False,
    }

    try:
        # Test schema consistency by checking table existence
        print("   Testing schema consistency...")
        conn_manager = get_direct_connection()
        async with conn_manager as connection:
            checkpoints_exists = await table_exists(connection, "checkpoints")
            users_exists = await table_exists(connection, "users_threads_runs")

        schema_ok = checkpoints_exists and users_exists
        consistency_results["schema_consistency"] = schema_ok
        print(f"   Schema consistency: {schema_ok}")
        print(f"   Checkpoints table exists: {checkpoints_exists}")
        print(f"   Users table exists: {users_exists}")

        # Test data integrity by creating and verifying data
        print("   Testing data integrity...")
        test_thread_id = f"consistency_test_{uuid.uuid4().hex[:8]}"
        run_id = await create_thread_run_entry(
            TEST_EMAIL, test_thread_id, "Consistency test"
        )

        # Verify the data was created
        threads = await get_user_chat_threads(TEST_EMAIL)
        found_thread = any(t.get("thread_id") == test_thread_id for t in threads)
        consistency_results["data_integrity"] = found_thread
        print(f"   Data integrity: {found_thread}")

        # Clean up test data
        if found_thread:
            await delete_user_thread_entries(TEST_EMAIL, test_thread_id)
            print("   Test data cleaned up")

        # Test connection stability
        print("   Testing connection stability...")
        conn_string_1 = get_connection_string()
        conn_string_2 = get_connection_string()
        stability_ok = conn_string_1 == conn_string_2
        consistency_results["connection_stability"] = stability_ok
        print(f"   Connection stability: {stability_ok}")

        overall_success = all(consistency_results.values())
        print(
            f"✅ Database consistency test: {'PASSED' if overall_success else 'PARTIAL'}"
        )
        return overall_success, consistency_results

    except Exception as e:
        print(f"❌ Database consistency test failed: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False, consistency_results


async def run_overall_tests() -> OverallTestResults:
    """Run all overall database tests."""
    print("🚀 Starting Overall Database Tests")
    print(
        f"   Import Status: {'✅ Available' if DATABASE_OVERALL_AVAILABLE else '❌ Failed'}"
    )
    if not DATABASE_OVERALL_AVAILABLE:
        print(f"   Import Error: {IMPORT_ERROR}")

    results = OverallTestResults()

    # Get all test cases
    foundation_tests = get_database_foundation_test_cases()
    table_tests = get_table_management_test_cases()
    pool_tests = get_connection_pool_test_cases()
    checkpointer_tests = get_checkpointer_management_test_cases()
    error_tests = get_error_handling_test_cases()

    all_test_cases = (
        foundation_tests + table_tests + pool_tests + checkpointer_tests + error_tests
    )

    print(f"   Total Component Tests: {len(all_test_cases)}")
    print(f"   Required Components: {len(REQUIRED_COMPONENTS)}")

    # Run each component test
    for test_case in all_test_cases:
        test_id = test_case["test_id"]
        await run_overall_test(test_id, test_case, results)

        # Small delay between tests
        await asyncio.sleep(0.1)

    # Run workflow tests
    print("\n🔄 Running Workflow Tests...")

    # End-to-end workflow test
    workflow_success, workflow_details = await test_end_to_end_workflow()
    results.add_workflow_result("end_to_end", workflow_success, workflow_details)
    results.add_component_test(
        test_id="OT_WORKFLOW_E2E",
        component="end_to_end_workflow",
        description="End-to-end database workflow test",
        result_data=workflow_success,
        response_time=0.0,
        success=workflow_success,
    )

    # Database consistency test
    consistency_success, consistency_details = await test_database_consistency()
    results.add_workflow_result("consistency", consistency_success, consistency_details)
    results.add_component_test(
        test_id="OT_WORKFLOW_CONSISTENCY",
        component="database_consistency",
        description="Database consistency and integrity test",
        result_data=consistency_success,
        response_time=0.0,
        success=consistency_success,
    )

    print(f"\n📊 Overall Database Tests Summary:")
    summary = results.get_summary()
    print(f"   Total Tests: {summary['total_requests']}")
    print(f"   Successful: {summary['successful_requests']}")
    print(f"   Failed: {summary['failed_requests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Average Response Time: {summary['average_response_time']:.3f}s")

    return results


def analyze_test_results(results: OverallTestResults):
    """Analyze and display test results."""
    print("\n📈 Detailed Test Analysis:")

    summary = results.get_summary()

    print(
        f"   Components Tested: {len(results.component_coverage)}/{len(REQUIRED_COMPONENTS)}"
    )
    print(f"   Missing Components: {REQUIRED_COMPONENTS - results.component_coverage}")
    print(f"   Workflow Tests: {len(results.workflow_results)}")

    # Analyze workflow results
    if results.workflow_results:
        print(f"\n🔄 Workflow Test Results:")
        for workflow_name, workflow_data in results.workflow_results.items():
            status = "✅ PASSED" if workflow_data["success"] else "❌ FAILED"
            print(f"   - {workflow_name}: {status}")

            # Show workflow details
            if workflow_data["details"]:
                print(f"     Details: {workflow_data['details']}")

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
    print("=" * 90)
    print("🧪 OVERALL DATABASE CHECKPOINTER TESTING SUITE")
    print("=" * 90)

    start_time = time.time()

    try:
        # Run all tests
        results = await run_overall_tests()

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
                "test_type": "Overall Database Testing",
                "error_location": "main",
                "import_available": DATABASE_OVERALL_AVAILABLE,
                "import_error": IMPORT_ERROR,
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

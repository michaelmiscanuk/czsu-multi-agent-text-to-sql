"""
Test for Phase 8.2: Catalog Routes
Tests the catalog endpoints with real HTTP requests and proper authentication.

This test uses real functionality from the main scripts without hardcoding values.
"""

import asyncio
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    print(
        "[CATALOG-STARTUP] Windows detected - setting SelectorEventLoop for compatibility..."
    )
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[CATALOG-STARTUP] Event loop policy set successfully")

# Add project root to path
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

sys.path.insert(0, str(BASE_DIR))

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Import test helpers
from tests.helpers import (
    extract_detailed_error_info,
    make_request_with_traceback_capture,
    save_exception_traceback,
    save_server_traceback_report,
    save_test_failures_traceback,
)

# Test configuration
TEST_EMAIL = "test_user@example.com"
TEST_QUERIES = [
    # /catalog endpoint tests
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
    # /data-tables endpoint tests
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
    # /data-table endpoint tests - these should reveal the bug
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
    },  # Should return empty but not crash
]

# Server configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30  # seconds


def create_test_jwt_token(email: str = TEST_EMAIL):
    """Create a simple test JWT token for authentication."""
    try:
        import jwt

        google_client_id = (
            "722331814120-9kdm64s2mp9cq8kig0mvrluf1eqkso74.apps.googleusercontent.com"
        )

        payload = {
            "email": email,
            "aud": google_client_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "test_issuer",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
        }
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        print(
            f"🔧 TEST TOKEN: Created JWT token with correct audience: {google_client_id}"
        )
        return token
    except ImportError:
        print("⚠️ JWT library not available, using simple Bearer token")
        return "test_token_placeholder"


class CatalogTestResults:
    """Class to track and analyze catalog test results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        self.errors: List[Dict[str, Any]] = []

    def add_result(
        self,
        test_id: str,
        endpoint: str,
        description: str,
        response_data: Dict,
        response_time: float,
        status_code: int,
    ):
        """Add a test result."""
        result = {
            "test_id": test_id,
            "endpoint": endpoint,
            "description": description,
            "response_data": response_data,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "success": status_code
            in [200, 422],  # Both success and validation errors are valid outcomes
        }
        self.results.append(result)
        print(
            f"✅ Result added: Test {test_id}, {endpoint} ({description}), "
            f"Status {status_code}, Time {response_time:.2f}s"
        )

    def add_error(
        self,
        test_id: str,
        endpoint: str,
        description: str,
        error: Exception,
        response_time: float = None,
        response_data: dict = None,
    ):
        """Add an error result."""
        error_info = {
            "test_id": test_id,
            "endpoint": endpoint,
            "description": description,
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "error_obj": error,  # Store the actual error object to access server_tracebacks
            "response_data": response_data,  # Store server response data (may include traceback)
        }
        self.errors.append(error_info)
        print(
            f"❌ Error added: Test {test_id}, {endpoint} ({description}), Error: {str(error)}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        total_requests = len(self.results) + len(self.errors)
        successful_requests = len([r for r in self.results if r["success"]])
        failed_requests = len(self.errors) + len(
            [r for r in self.results if not r["success"]]
        )

        if self.results:
            avg_response_time = sum(r["response_time"] for r in self.results) / len(
                self.results
            )
            max_response_time = max(r["response_time"] for r in self.results)
            min_response_time = min(r["response_time"] for r in self.results)
        else:
            avg_response_time = max_response_time = min_response_time = 0

        total_test_time = None
        if self.start_time and self.end_time:
            total_test_time = (self.end_time - self.start_time).total_seconds()

        # Count unique endpoints tested successfully
        tested_endpoints = set(r["endpoint"] for r in self.results if r["success"])
        required_endpoints = {"/catalog", "/data-tables", "/data-table"}

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            ),
            "average_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "total_test_time": total_test_time,
            "errors": self.errors,
            "all_endpoints_tested": tested_endpoints.issuperset(required_endpoints),
            "tested_endpoints": tested_endpoints,
            "missing_endpoints": required_endpoints - tested_endpoints,
        }


async def check_server_connectivity():
    """Check if the server is running and accessible."""
    print("\n" + "=" * 80)
    print("🔍 CHECKING SERVER CONNECTIVITY")
    print("=" * 80)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{SERVER_BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is running and accessible")
                return True
            else:
                print(f"❌ Server responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure uvicorn is running at {SERVER_BASE_URL}")
        return False


def setup_test_environment():
    """Set up the test environment and check prerequisites."""
    print("\n" + "=" * 80)
    print("🔧 SETTING UP TEST ENVIRONMENT")
    print("=" * 80)

    # Check if USE_TEST_TOKENS is set for the server
    use_test_tokens = os.getenv("USE_TEST_TOKENS", "0")
    if use_test_tokens != "1":
        print("⚠️  WARNING: USE_TEST_TOKENS environment variable is not set to '1'")
        print("   The server needs USE_TEST_TOKENS=1 to accept test tokens")
        print("   Set this environment variable in your server environment:")
        print("   SET USE_TEST_TOKENS=1 (Windows)")
        print("   export USE_TEST_TOKENS=1 (Linux/Mac)")
        print("   Continuing test anyway - this may cause 401 authentication errors")
    else:
        print("✅ USE_TEST_TOKENS=1 - test tokens will be accepted by server")

    # Check if required database files exist
    db_paths = [
        "metadata/llm_selection_descriptions/selection_descriptions.db",
        "data/czsu_data.db",
    ]
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"⚠️  WARNING: Database file not found: {db_path}")
            print("   Some tests may fail if the database is not available")
        else:
            print(f"✅ Database found: {db_path}")

    print("✅ Test environment setup complete")
    return True


async def setup_debug_environment(client: httpx.AsyncClient):
    """Setup debug environment for this specific test."""
    print("\n" + "=" * 80)
    print("🔧 SETTING UP DEBUG ENVIRONMENT")
    print("=" * 80)

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Set debug variables specific to this test
    debug_vars = {
        "print__catalog_debug": "1",
        "print__data_tables_debug": "1",
        "DEBUG_TRACEBACK": "1",  # Enable traceback in error responses
    }

    try:
        response = await client.post("/debug/set-env", headers=headers, json=debug_vars)
        if response.status_code == 200:
            print("✅ Debug environment configured successfully")
            return True
        else:
            print(f"⚠️ Debug setup failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Debug setup error: {e}")
        return False


async def cleanup_debug_environment(client: httpx.AsyncClient):
    """Reset debug environment after test."""
    print("\n" + "=" * 80)
    print("🧹 CLEANING UP DEBUG ENVIRONMENT")
    print("=" * 80)

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    debug_vars = {"print__catalog_debug": "1", "print__data_tables_debug": "1"}

    try:
        response = await client.post(
            "/debug/reset-env", headers=headers, json=debug_vars
        )
        if response.status_code == 200:
            print("✅ Debug environment reset to original .env values")
        else:
            print(f"⚠️ Debug reset failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Debug reset error: {e}")


async def make_catalog_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    params: Dict,
    description: str,
    should_succeed: bool,
    results: CatalogTestResults,
):
    """Make a request to a catalog endpoint with server traceback capture."""
    print(f"\n🔍 Testing {endpoint} (Test ID: {test_id})")
    print(f"   Description: {description}")
    print(f"   Parameters: {params}")
    print(f"   Expected to succeed: {should_succeed}")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    start_time = time.time()
    try:
        # Use the new helper function to capture server tracebacks
        result = await make_request_with_traceback_capture(
            client,
            "GET",
            f"{SERVER_BASE_URL}{endpoint}",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response_time = time.time() - start_time

        # Extract detailed error information
        error_info = extract_detailed_error_info(result)

        # Check if we got a response
        if result["response"] is None:
            # Client-side error (connection failed, etc.)
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"🔌 Test {test_id} - Client Error: {error_message}")

            # Create error object with server traceback info
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]

            results.add_error(test_id, endpoint, description, error_obj, response_time)
            return

        response = result["response"]
        print(
            f"📝 Test {test_id} - Status: {response.status_code}, Time: {response_time:.2f}s"
        )

        # Check if response matches expectation
        if should_succeed:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Strict validation of response structure
                    if endpoint == "/catalog":
                        # Validate catalog response structure
                        assert "results" in data, "Missing 'results' field"
                        assert "total" in data, "Missing 'total' field"
                        assert "page" in data, "Missing 'page' field"
                        assert "page_size" in data, "Missing 'page_size' field"
                        assert isinstance(
                            data["results"], list
                        ), "'results' must be a list"
                        assert isinstance(
                            data["total"], int
                        ), "'total' must be an integer"
                        assert isinstance(
                            data["page"], int
                        ), "'page' must be an integer"
                        assert isinstance(
                            data["page_size"], int
                        ), "'page_size' must be an integer"
                        assert data["total"] >= 0, "'total' must be non-negative"
                        assert data["page"] >= 1, "'page' must be >= 1"
                        assert data["page_size"] >= 1, "'page_size' must be >= 1"

                        # Validate each result item
                        for item in data["results"]:
                            assert (
                                "selection_code" in item
                            ), "Missing 'selection_code' in result item"
                            assert (
                                "extended_description" in item
                            ), "Missing 'extended_description' in result item"
                            assert isinstance(
                                item["selection_code"], str
                            ), "'selection_code' must be a string"
                            assert isinstance(
                                item["extended_description"], str
                            ), "'extended_description' must be a string"

                        print(
                            f"✅ Catalog validation passed: {len(data['results'])} items, total {data['total']}"
                        )

                    elif endpoint == "/data-tables":
                        # Validate data-tables response structure
                        assert "tables" in data, "Missing 'tables' field"
                        assert isinstance(
                            data["tables"], list
                        ), "'tables' must be a list"

                        # Validate each table item
                        for table in data["tables"]:
                            assert (
                                "selection_code" in table
                            ), "Missing 'selection_code' in table item"
                            assert (
                                "short_description" in table
                            ), "Missing 'short_description' in table item"
                            assert isinstance(
                                table["selection_code"], str
                            ), "'selection_code' must be a string"
                            assert isinstance(
                                table["short_description"], str
                            ), "'short_description' must be a string"

                        print(
                            f"✅ Data-tables validation passed: {len(data['tables'])} tables"
                        )

                    elif endpoint == "/data-table":
                        # Validate data-table response structure
                        assert "columns" in data, "Missing 'columns' field"
                        assert "rows" in data, "Missing 'rows' field"
                        assert isinstance(
                            data["columns"], list
                        ), "'columns' must be a list"
                        assert isinstance(data["rows"], list), "'rows' must be a list"

                        # Validate column names are strings
                        for col in data["columns"]:
                            assert isinstance(
                                col, str
                            ), f"Column name '{col}' must be a string"

                        # Validate row structure
                        for i, row in enumerate(data["rows"]):
                            assert isinstance(row, list), f"Row {i} must be a list"
                            if data["columns"]:  # Only check if we have columns
                                assert len(row) == len(
                                    data["columns"]
                                ), f"Row {i} has {len(row)} values but {len(data['columns'])} columns"

                        print(
                            f"✅ Data-table validation passed: {len(data['columns'])} columns, {len(data['rows'])} rows"
                        )

                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        data,
                        response_time,
                        response.status_code,
                    )

                except AssertionError as e:
                    print(f"❌ Validation failed: {e}")

                    # Create error object with server traceback info
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]

                    results.add_error(
                        test_id, endpoint, description, error_obj, response_time
                    )
                except Exception as e:
                    print(f"❌ JSON parsing or validation error: {e}")

                    # Create error object with server traceback info
                    error_obj = Exception(f"Response processing failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]

                    results.add_error(
                        test_id, endpoint, description, error_obj, response_time
                    )
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "detail", f"HTTP {response.status_code}: {response.text}"
                    )
                except Exception:
                    error_data = None
                    error_message = f"HTTP {response.status_code}: {response.text}"

                print(
                    f"❌ Test {test_id} - Expected success but got HTTP {response.status_code}: {error_message}"
                )

                # Create error object with server traceback info
                error_obj = Exception(f"Expected success but got: {error_message}")
                error_obj.server_tracebacks = error_info["server_tracebacks"]

                # Log server tracebacks if any were captured
                if error_info["server_tracebacks"]:
                    print(
                        f"🔍 Server tracebacks captured: {len(error_info['server_tracebacks'])}"
                    )
                    for i, tb in enumerate(error_info["server_tracebacks"], 1):
                        print(
                            f"   Server Traceback #{i}: {tb['exception_type']}: {tb['exception_message']}"
                        )

                results.add_error(
                    test_id,
                    endpoint,
                    description,
                    error_obj,
                    response_time,
                    response_data=error_data,
                )
        else:
            if response.status_code == 422:  # Validation error
                print(f"✅ Test {test_id} - Correctly failed with validation error")
                data = {"validation_error": True}
                results.add_result(
                    test_id,
                    endpoint,
                    description,
                    data,
                    response_time,
                    response.status_code,
                )
            elif response.status_code == 200:
                print(f"❌ Test {test_id} - Expected validation error but got success")

                # Create error object with server traceback info
                error_obj = Exception("Expected validation error but request succeeded")
                error_obj.server_tracebacks = error_info["server_tracebacks"]

                results.add_error(
                    test_id, endpoint, description, error_obj, response_time
                )
            else:
                print(
                    f"❌ Test {test_id} - Expected validation error but got HTTP {response.status_code}"
                )

                # Create error object with server traceback info
                error_obj = Exception(
                    f"Expected validation error but got HTTP {response.status_code}"
                )
                error_obj.server_tracebacks = error_info["server_tracebacks"]

                # Log server tracebacks if any were captured
                if error_info["server_tracebacks"]:
                    print(
                        f"🔍 Server tracebacks captured: {len(error_info['server_tracebacks'])}"
                    )
                    for i, tb in enumerate(error_info["server_tracebacks"], 1):
                        print(
                            f"   Server Traceback #{i}: {tb['exception_type']}: {tb['exception_message']}"
                        )

                results.add_error(
                    test_id, endpoint, description, error_obj, response_time
                )

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Unknown error of type {type(e).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}, Time: {response_time:.2f}s")

        # Create error object (this shouldn't have server tracebacks since it's a client-side exception)
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []

        results.add_error(
            test_id, endpoint, description, error_obj, response_time, response_data=None
        )


async def run_catalog_tests() -> CatalogTestResults:
    """Run all catalog endpoint tests."""
    print("\n" + "=" * 80)
    print("🚀 STARTING CATALOG ENDPOINTS TESTS")
    print(f"📂 BASE_DIR: {BASE_DIR}")
    print("=" * 80)

    results = CatalogTestResults()
    results.start_time = datetime.now()

    print(f"📋 Test queries: {len(TEST_QUERIES)} queries defined")
    print(f"🌐 Server URL: {SERVER_BASE_URL}")
    print(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # First, get real table names to test data-table endpoint more thoroughly
        real_table_name = None
        try:
            token = create_test_jwt_token()
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(
                f"{SERVER_BASE_URL}/data-tables", headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                if data["tables"]:
                    real_table_name = data["tables"][0]["selection_code"]
                    print(f"🔍 Found real table for testing: {real_table_name}")
        except Exception as e:
            print(f"⚠️ Could not get real table names: {e}")

        # Add real table test if we found one
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
            print("\n" + "=" * 80)
            print(f"🔍 RUNNING TEST {test_id}")
            print("=" * 80)

            await make_catalog_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["params"],
                test_case["description"],
                test_case["should_succeed"],
                results,
            )
            # Add small delay between requests
            await asyncio.sleep(0.5)

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: CatalogTestResults):
    """Analyze and print test results."""
    print("\n" + "=" * 80)
    print("=" * 80)
    print("=" * 80)
    print("📊 CATALOG ENDPOINTS TEST RESULTS")
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)

    summary = results.get_summary()

    print(f"\n🔢 Total Requests: {summary['total_requests']}")
    print(f"✅ Successful: {summary['successful_requests']}")
    print(f"❌ Failed: {summary['failed_requests']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

    if summary["total_test_time"]:
        print(f"\n⏱️  Total Test Time: {summary['total_test_time']:.2f}s")

    if summary["successful_requests"] > 0:
        print(f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"🏆 Best Response Time: {summary['min_response_time']:.2f}s")
        print(f"🐌 Worst Response Time: {summary['max_response_time']:.2f}s")

    print(
        f"\n🎯 All Required Endpoints Tested: {'✅ YES' if summary['all_endpoints_tested'] else '❌ NO'}"
    )
    if not summary["all_endpoints_tested"]:
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

    # Show individual results
    print("\n📋 Individual Request Results:")
    for i, result in enumerate(results.results, 1):
        status_emoji = "✅" if result["success"] else "❌"
        print(
            f"  {i}. {status_emoji} Test: {result['test_id']} | "
            f"Endpoint: {result['endpoint']} | "
            f"Description: {result['description']} | "
            f"Status: {result['status_code']} | "
            f"Time: {result['response_time']:.2f}s"
        )

    # Show errors if any
    if results.errors:
        print("\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print(
                f"  {i}. Test: {error['test_id']} | "
                f"Endpoint: {error['endpoint']} | "
                f"Description: {error['description']} | "
                f"Error: {error['error']}"
            )

    # Endpoint analysis
    print("\n🔍 ENDPOINT ANALYSIS:")
    if summary["all_endpoints_tested"]:
        print("✅ All required endpoints tested successfully")
        if summary["max_response_time"] - summary["min_response_time"] < 1.0:
            print("✅ Response times are consistent - good performance")
        else:
            print("⚠️  Response times vary significantly - possible performance issues")
    else:
        print("❌ Not all required endpoints were tested successfully")
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

    # Collect all server tracebacks from errors
    all_server_tracebacks = []
    for error in results.errors:
        error_obj = error.get("error_obj")  # This might be the actual Exception object
        if hasattr(error_obj, "server_tracebacks") and error_obj.server_tracebacks:
            all_server_tracebacks.extend(error_obj.server_tracebacks)

    # Save traceback information if there are failures
    if results.errors or summary["failed_requests"] > 0:
        print("\n📝 Saving failure traceback information...")

        # Prepare additional info for the traceback report
        additional_info = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Test Start Time": (
                results.start_time.strftime("%Y-%m-%d %H:%M:%S")
                if results.start_time
                else "Unknown"
            ),
            "Test End Time": (
                results.end_time.strftime("%Y-%m-%d %H:%M:%S")
                if results.end_time
                else "Unknown"
            ),
            "Server Tracebacks Captured": len(all_server_tracebacks),
        }

        # Save the regular traceback report
        save_test_failures_traceback(
            test_file_name="test_phase8_catalog.py",
            test_results=results,
            additional_info=additional_info,
        )

        # Save the server traceback report if we captured any
        if all_server_tracebacks:
            print(f"📝 Saving {len(all_server_tracebacks)} server traceback(s)...")
            save_server_traceback_report(
                test_file_name="test_phase8_catalog.py",
                test_results=results,
                server_tracebacks=all_server_tracebacks,
                additional_info=additional_info,
            )
        else:
            print(
                "ℹ️  No server tracebacks were captured (this might indicate the server didn't log errors)"
            )

    return summary


def cleanup_test_environment():
    """Clean up the test environment."""
    print("\n" + "=" * 80)
    print("🧹 CLEANING UP TEST ENVIRONMENT")
    print("=" * 80)
    print("✅ Test environment cleanup complete")


async def main():
    """Main test execution function."""
    print("🚀 Catalog Endpoints Test Starting...")
    print("=" * 60)

    # Check if server is running
    if not await check_server_connectivity():
        print("❌ Server connectivity check failed!")
        print("   Please start your uvicorn server first:")
        print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False

    # Setup test environment
    if not setup_test_environment():
        print("❌ Test environment setup failed!")
        return False

    try:
        # Create HTTP client for the entire test session
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Setup debug environment for this test
            if not await setup_debug_environment(client):
                print("⚠️ Debug environment setup failed, continuing without debug")

            # Run the tests
            results = await run_catalog_tests()

            # Cleanup debug environment
            await cleanup_debug_environment(client)

        # Analyze results
        summary = analyze_test_results(results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )

        # Check for specific database errors that indicate bugs
        has_database_errors = any(
            "no such variable" in error.get("error", "").lower()
            or "nameError" in error.get("error", "")
            or "undefined" in error.get("error", "").lower()
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors  # No empty/unknown errors
            and not has_database_errors  # No database variable errors
            and summary["total_requests"] > 0  # Some requests were made
            and summary["all_endpoints_tested"]  # All required endpoints tested
            and summary["failed_requests"] == 0  # No failed requests
            and summary["successful_requests"] > 0  # At least some succeeded
        )

        if has_empty_errors:
            print(
                "❌ Test failed: Server returned empty error messages (potential crash/hang)"
            )
        elif has_database_errors:
            print("❌ Test failed: Database errors detected (potential code bugs)")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded (server may be down)")
        elif not summary["all_endpoints_tested"]:
            print("❌ Test failed: Not all required endpoints were tested successfully")
        elif summary["failed_requests"] > 0:
            print(f"❌ Test failed: {summary['failed_requests']} requests failed")
        else:
            print(
                f"✅ Test criteria met: {summary['successful_requests']}/{summary['total_requests']} "
                f"requests successful with proper error handling"
            )

        print(
            f"\n🏁 OVERALL TEST RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}"
        )

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        traceback.print_exc()

        # Save exception traceback
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }

        save_exception_traceback(
            test_file_name="test_phase8_catalog.py",
            exception=e,
            test_context=test_context,
        )

        return False

    finally:
        cleanup_test_environment()


@pytest.mark.asyncio
async def test_catalog_endpoints():
    """Pytest-compatible test function."""

    async def main_with_cleanup():
        try:
            result = await main()
            return result
        except KeyboardInterrupt:
            print("\n⛔ Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n💥 Unexpected error: {str(e)}")
            traceback.print_exc()

            # Save exception traceback for pytest
            test_context = {
                "Server URL": SERVER_BASE_URL,
                "Request Timeout": f"{REQUEST_TIMEOUT}s",
                "Total Test Queries": len(TEST_QUERIES),
                "Error Location": "pytest test_catalog_endpoints()",
                "Error During": "Pytest execution",
            }

            save_exception_traceback(
                test_file_name="test_phase8_catalog.py",
                exception=e,
                test_context=test_context,
            )

            return False
        finally:
            cleanup_test_environment()

    result = await main_with_cleanup()
    assert result, "Catalog endpoints test failed"


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()

        # Save exception traceback for direct execution
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }

        save_exception_traceback(
            test_file_name="test_phase8_catalog.py",
            exception=e,
            test_context=test_context,
        )

        sys.exit(1)

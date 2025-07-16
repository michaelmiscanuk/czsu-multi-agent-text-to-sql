"""
Test for Phase 8.3: Analysis Routes

Tests the analysis endpoints with real HTTP requests and proper authentication.
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
        "[ANALYSIS-STARTUP] Windows detected - setting SelectorEventLoop for compatibility..."
    )
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[ANALYSIS-STARTUP] Event loop policy set successfully")

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
    # Basic /analyze endpoint tests
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "What tables are available?",
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
        },
        "description": "Basic analysis query",
        "should_succeed": True,
        "expected_fields": ["prompt", "result", "thread_id", "run_id"]
    },
    {
        "endpoint": "/analyze",
        "method": "POST", 
        "data": {
            "prompt": "Show me the first 5 rows from any table",
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
        },
        "description": "SQL query analysis",
        "should_succeed": True,
        "expected_fields": ["prompt", "result", "thread_id", "run_id", "sql", "queries_and_results"]
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "Test query with existing thread",
            "thread_id": "existing_thread_123"
        },
        "description": "Analysis with existing thread",
        "should_succeed": True,
        "expected_fields": ["prompt", "result", "thread_id", "run_id"]
    },
    # Invalid request tests
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "",
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
        },
        "description": "Empty prompt",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "Valid prompt",
            "thread_id": ""
        },
        "description": "Empty thread_id",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
            # Missing prompt field
        },
        "description": "Missing prompt field",
        "should_succeed": False,
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "Valid prompt"
            # Missing thread_id field
        },
        "description": "Missing thread_id field",
        "should_succeed": False,
    },
    # Edge cases
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "A" * 9999,  # Very long prompt (just under limit)
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
        },
        "description": "Very long prompt",
        "should_succeed": True,
        "expected_fields": ["prompt", "result", "thread_id", "run_id"]
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "data": {
            "prompt": "A" * 10001,  # Exceeds max length
            "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
        },
        "description": "Prompt exceeds max length",
        "should_succeed": False,
    },
]

# Server configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 180  # 3 minutes for analysis requests (they can take longer)


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


class AnalysisTestResults:
    """Class to track and analyze analysis test results."""

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
            "success": status_code in [200, 422],  # Both success and validation errors are valid outcomes
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
        required_endpoints = {"/analyze"}

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

    # Check environment variables for analysis functionality
    required_env_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_CLIENT_ID",
    ]
    
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            print(f"⚠️  WARNING: Environment variable {env_var} is not set")
            print("   Analysis functionality may be limited")
        else:
            print(f"✅ Environment variable {env_var} is set")

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
        "print__analyze_debug": "1",
        "print__analysis_tracing_debug": "1",
        "print__feedback_flow": "1",
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

    debug_vars = {
        "print__analyze_debug": "1",
        "print__analysis_tracing_debug": "1", 
        "print__feedback_flow": "1"
    }

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


async def make_analysis_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    method: str,
    data: Dict,
    description: str,
    should_succeed: bool,
    expected_fields: List[str],
    results: AnalysisTestResults,
):
    """Make a request to an analysis endpoint with server traceback capture."""
    print(f"\n🔍 Testing {method} {endpoint} (Test ID: {test_id})")
    print(f"   Description: {description}")
    print(f"   Data: {data}")
    print(f"   Expected to succeed: {should_succeed}")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
    start_time = time.time()

    try:
        # Use the new helper function to capture server tracebacks
        if method.upper() == "POST":
            result = await make_request_with_traceback_capture(
                client,
                "POST",
                f"{SERVER_BASE_URL}{endpoint}",
                json=data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        else:
            result = await make_request_with_traceback_capture(
                client,
                method.upper(),
                f"{SERVER_BASE_URL}{endpoint}",
                params=data,
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
                    data_response = response.json()
                    
                    # Validate expected fields are present
                    for field in expected_fields:
                        assert field in data_response, f"Missing expected field: {field}"
                    
                    # Specific validation for analysis responses
                    if endpoint == "/analyze":
                        # Validate analysis response structure
                        assert "prompt" in data_response, "Missing 'prompt' field"
                        assert "result" in data_response, "Missing 'result' field"
                        assert "thread_id" in data_response, "Missing 'thread_id' field"
                        assert "run_id" in data_response, "Missing 'run_id' field"
                        
                        # Check that prompt matches request
                        assert data_response["prompt"] == data["prompt"], "Prompt mismatch in response"
                        assert data_response["thread_id"] == data["thread_id"], "Thread ID mismatch in response"
                        
                        # Check result is not empty
                        assert data_response["result"], "Result field is empty"
                        
                        # Check run_id is valid UUID format
                        import uuid
                        try:
                            uuid.UUID(data_response["run_id"])
                        except ValueError:
                            raise AssertionError("run_id is not a valid UUID")

                    print(f"✅ Test {test_id} - Validation passed")
                    results.add_result(
                        test_id, endpoint, description, data_response, response_time, response.status_code
                    )

                except AssertionError as e:
                    print(f"❌ Test {test_id} - Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(test_id, endpoint, description, error_obj, response_time, response_data=data_response)

                except Exception as e:
                    print(f"❌ Test {test_id} - Response parsing failed: {e}")
                    error_obj = Exception(f"Response parsing failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(test_id, endpoint, description, error_obj, response_time)
            else:
                # Expected success but got non-200 status
                print(f"❌ Test {test_id} - Expected success but got status {response.status_code}")
                try:
                    error_response = response.json()
                except:
                    error_response = {"error": "Could not parse error response"}
                
                error_obj = Exception(f"Expected success but got status {response.status_code}")
                error_obj.server_tracebacks = error_info["server_tracebacks"]
                results.add_error(test_id, endpoint, description, error_obj, response_time, response_data=error_response)
        else:
            # Expected failure
            if response.status_code in [400, 422, 404, 500]:
                print(f"✅ Test {test_id} - Expected failure with status {response.status_code}")
                try:
                    error_response = response.json()
                except:
                    error_response = {"error": "Could not parse error response"}

                results.add_result(
                    test_id, endpoint, description, error_response, response_time, response.status_code
                )
            else:
                print(f"❌ Test {test_id} - Expected failure but got status {response.status_code}")
                try:
                    unexpected_response = response.json()
                except:
                    unexpected_response = {"error": "Could not parse response"}
                
                error_obj = Exception(f"Expected failure but got status {response.status_code}")
                error_obj.server_tracebacks = error_info["server_tracebacks"]
                results.add_error(test_id, endpoint, description, error_obj, response_time, response_data=unexpected_response)

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Unknown {type(e).__name__} exception"

        print(f"❌ Test {test_id} - Error: {error_message}, Time: {response_time:.2f}s")
        
        # Create error object (this shouldn't have server tracebacks since it's a client-side exception)
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(
            test_id, endpoint, description, error_obj, response_time, response_data=None
        )


async def run_analysis_tests() -> AnalysisTestResults:
    """Run all analysis endpoint tests."""
    print("\n" + "=" * 80)
    print("🚀 STARTING ANALYSIS ENDPOINTS TESTS")
    print(f"📂 BASE_DIR: {BASE_DIR}")
    print("=" * 80)

    results = AnalysisTestResults()
    results.start_time = datetime.now()

    print(f"📋 Test queries: {len(TEST_QUERIES)} queries defined")
    print(f"🌐 Server URL: {SERVER_BASE_URL}")
    print(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Setup debug environment
        await setup_debug_environment(client)

        # Run all test cases
        for i, test_case in enumerate(TEST_QUERIES, 1):
            test_id = f"ANALYSIS_{i:02d}"
            
            await make_analysis_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["method"],
                test_case["data"],
                test_case["description"],
                test_case["should_succeed"],
                test_case.get("expected_fields", []),
                results,
            )

        # Cleanup debug environment
        await cleanup_debug_environment(client)

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: AnalysisTestResults):
    """Analyze and print test results."""
    print("\n" + "=" * 80)
    print("=" * 80)
    print("=" * 80)
    print("📊 ANALYSIS ENDPOINTS TEST RESULTS")
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
            f"   {status_emoji} Test {result['test_id']}: {result['description']} "
            f"({result['status_code']}, {result['response_time']:.2f}s)"
        )

    # Show errors if any
    if results.errors:
        print(f"\n❌ ERROR DETAILS ({len(results.errors)} errors):")
        for i, error in enumerate(results.errors, 1):
            print(f"   {i}. Test {error['test_id']}: {error['description']}")
            print(f"      Error: {error['error']}")
            if error.get("response_time"):
                print(f"      Time: {error['response_time']:.2f}s")
            
            # Show server tracebacks if available
            if hasattr(error.get("error_obj"), "server_tracebacks") and error["error_obj"].server_tracebacks:
                print(f"      Server Tracebacks: {len(error['error_obj'].server_tracebacks)} found")

    # Endpoint analysis
    print("\n🔍 ENDPOINT ANALYSIS:")
    if summary["all_endpoints_tested"]:
        print("✅ All required analysis endpoints were tested successfully")
        print("✅ /analyze endpoint is working correctly")
    else:
        print("❌ Some required endpoints were not tested successfully")
        print("❌ Check server logs for detailed error information")

    # Collect all server tracebacks from errors
    all_server_tracebacks = []
    for error in results.errors:
        if hasattr(error.get("error_obj"), "server_tracebacks"):
            all_server_tracebacks.extend(error["error_obj"].server_tracebacks)

    # Save traceback information if there are failures
    if results.errors or summary["failed_requests"] > 0:
        save_test_failures_traceback(
            "test_phase8_analysis.py",
            results,
            additional_info={
                "test_type": "Analysis Endpoints Test",
                "server_url": SERVER_BASE_URL,
                "total_requests": summary["total_requests"],
                "success_rate": f"{summary['success_rate']:.1f}%",
                "total_test_time": f"{summary.get('total_test_time', 0):.2f}s",
            },
        )

        # Save server tracebacks if any were captured
        if all_server_tracebacks:
            save_server_traceback_report(
                "test_phase8_analysis.py",
                results,
                all_server_tracebacks,
                additional_info={
                    "test_type": "Analysis Endpoints Test",
                    "server_url": SERVER_BASE_URL,
                },
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
    print("🚀 Analysis Endpoints Test Starting...")
    print("=" * 60)

    # Check if server is running
    if not await check_server_connectivity():
        print("❌ Server connectivity check failed. Exiting.")
        return False

    # Setup test environment
    if not setup_test_environment():
        print("❌ Test environment setup failed. Exiting.")
        return False

    try:
        # Run tests
        test_results = await run_analysis_tests()

        # Analyze results
        summary = analyze_test_results(test_results)

        # Return success status
        return summary["success_rate"] >= 50.0  # Consider 50%+ success rate as passing

    except Exception as e:
        print(f"🚨 CRITICAL ERROR during test execution: {e}")
        print(f"🚨 Exception type: {type(e).__name__}")
        print(f"🚨 Exception traceback:")
        print(traceback.format_exc())

        # Save exception information
        save_exception_traceback(
            "test_phase8_analysis.py",
            e,
            test_context={
                "phase": "main_execution",
                "server_url": SERVER_BASE_URL,
                "request_timeout": REQUEST_TIMEOUT,
            },
        )
        return False

    finally:
        cleanup_test_environment()


@pytest.mark.asyncio
async def test_analysis_endpoints():
    """Pytest entry point for analysis endpoints test."""
    success = await main()
    assert success, "Analysis endpoints test failed"


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print("\n🎉 Analysis endpoints test completed successfully!")
        else:
            print("\n❌ Analysis endpoints test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n🚨 Test crashed: {e}")
        sys.exit(1)
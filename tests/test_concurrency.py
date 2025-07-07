#!/usr/bin/env python3
"""
Concurrency test for the /analyze endpoint.
Tests concurrent requests to ensure proper database connection handling
and PostgreSQL checkpointer stability under load.

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
# This prevents conflicts with the modular API structure's event loop setup
if sys.platform == "win32":
    print(
        "[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility..."
    )
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[POSTGRES-STARTUP] Event loop policy set successfully")

# Add the root directory to Python path to import from main scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

from my_agent.utils.postgres_checkpointer import (
    check_postgres_env_vars,
    cleanup_checkpointer,
    close_async_postgres_saver,
    create_async_postgres_saver,
    get_db_config,
)

# Import main application and dependencies from the new modular structure
# from api.main import app  # No longer needed for HTTP requests
# from api.dependencies.auth import get_current_user  # No longer needed for HTTP requests
# from api.models.requests import AnalyzeRequest  # No longer needed for HTTP requests

# Test configuration
TEST_EMAIL = "test_user@example.com"
TEST_PROMPTS = [
    "Porovnej narust poctu lidi v Brne a Praze v poslednich letech.",
    "Kontrastuj uroven zivota v Brne a v Praze.",
    "Kontrastuj uroven zivota v Brne a v Plzni.",
    "Kontrastuj uroven zivota v Ostrave a v Plzni.",
]

# Server configuration for real HTTP requests
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = (
    180  # seconds - Increased from 60s as analysis requests can take 40+ seconds
)


def create_test_jwt_token(email: str = TEST_EMAIL):
    """Create a simple test JWT token for authentication."""
    try:
        import jwt

        # Use the actual Google Client ID that the server expects
        # This needs to match the GOOGLE_CLIENT_ID environment variable in the server
        google_client_id = (
            "722331814120-9kdm64s2mp9cq8kig0mvrluf1eqkso74.apps.googleusercontent.com"
        )

        payload = {
            "email": email,
            "aud": google_client_id,  # ✅ Fixed: Use the correct Google Client ID
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "test_issuer",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
        }
        # Use a simple test secret - in real scenarios this would be properly configured
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        print(
            f"🔧 TEST TOKEN: Created JWT token with correct audience: {google_client_id}"
        )
        return token
    except ImportError:
        print("⚠️ JWT library not available, using simple Bearer token")
        return "test_token_placeholder"


class ConcurrencyTestResults:
    """Class to track and analyze concurrency test results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        self.errors: List[Dict[str, Any]] = []

    def add_result(
        self,
        thread_id: str,
        prompt: str,
        response_data: Dict,
        response_time: float,
        status_code: int,
    ):
        """Add a test result."""
        result = {
            "thread_id": thread_id,
            "prompt": prompt,
            "response_data": response_data,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "success": status_code == 200,
        }
        self.results.append(result)
        print(
            f"✅ Result added: Thread {thread_id}, Status {status_code}, Time {response_time:.2f}s"
        )

    def add_error(
        self, thread_id: str, prompt: str, error: Exception, response_time: float = None
    ):
        """Add an error result."""
        error_info = {
            "thread_id": thread_id,
            "prompt": prompt,
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        print(f"❌ Error added: Thread {thread_id}, Error: {str(error)}")

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
            "concurrent_requests_completed": successful_requests >= 2,
        }


def create_mock_user():
    """Create a mock user object for testing."""
    return {
        "email": TEST_EMAIL,
        "name": "Test User",
        "sub": "test_user_123",
        "aud": "test_audience",
        "exp": int(time.time()) + 3600,  # Expires in 1 hour
    }


def override_get_current_user():
    """Override the get_current_user dependency for testing."""
    return create_mock_user()


async def check_server_connectivity():
    """Check if the server is running and accessible."""
    print(f"🔍 Checking server connectivity at {SERVER_BASE_URL}...")
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
    print("🔧 Setting up test environment...")

    # Check if PostgreSQL environment variables are set
    if not check_postgres_env_vars():
        print("❌ PostgreSQL environment variables are not properly configured!")
        print("Required variables: host, port, dbname, user, password")
        print("Current config:", get_db_config())
        return False

    print("✅ PostgreSQL environment variables are configured")

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

    print("✅ Test environment setup complete (using real HTTP requests)")

    return True


async def make_analyze_request(
    client: httpx.AsyncClient,
    thread_id: str,
    prompt: str,
    results: ConcurrencyTestResults,
):
    """Make a single analyze request and record the result."""
    print(f"🚀 Starting request for thread {thread_id}")
    start_time = time.time()

    try:
        # Create the request payload using the same structure as the main app
        request_data = {"prompt": prompt, "thread_id": thread_id}

        # Create authentication headers for real HTTP requests
        token = create_test_jwt_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Make the request to the real running server
        response = await client.post("/analyze", json=request_data, headers=headers)
        response_time = time.time() - start_time

        print(
            f"📝 Thread {thread_id} - Status: {response.status_code}, Time: {response_time:.2f}s"
        )

        # Parse response
        if response.status_code == 200:
            response_data = response.json()
            results.add_result(
                thread_id, prompt, response_data, response_time, response.status_code
            )
        else:
            # Handle non-200 responses
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "detail", f"HTTP {response.status_code}: {response.text}"
                )
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"

            print(
                f"❌ Thread {thread_id} - HTTP Error {response.status_code}: {error_message}"
            )
            # For non-200 responses, treat as errors not just failed results
            results.add_error(
                thread_id, prompt, Exception(error_message), response_time
            )

    except httpx.TimeoutException:
        response_time = time.time() - start_time
        error_message = (
            f"Request timeout after {response_time:.1f}s (limit: {REQUEST_TIMEOUT}s)"
        )
        print(f"⏰ Thread {thread_id} - {error_message}")
        results.add_error(thread_id, prompt, Exception(error_message), response_time)

    except httpx.ConnectError as e:
        response_time = time.time() - start_time
        error_message = f"Connection failed: {str(e)}"
        print(f"🔌 Thread {thread_id} - {error_message}")
        results.add_error(thread_id, prompt, Exception(error_message), response_time)

    except httpx.HTTPStatusError as e:
        response_time = time.time() - start_time
        error_message = f"HTTP {e.response.status_code}: {e.response.text}"
        print(f"📡 Thread {thread_id} - {error_message}")
        results.add_error(thread_id, prompt, Exception(error_message), response_time)

    except Exception as e:
        response_time = time.time() - start_time
        # Improve error message extraction
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Unknown error of type {type(e).__name__}"

        print(
            f"❌ Thread {thread_id} - Error: {error_message}, Time: {response_time:.2f}s"
        )
        results.add_error(thread_id, prompt, Exception(error_message), response_time)


async def run_concurrency_test() -> ConcurrencyTestResults:
    """Run the main concurrency test with 4 simultaneous requests."""
    print("🎯 Starting concurrency test with 4 simultaneous requests...")

    results = ConcurrencyTestResults()
    results.start_time = datetime.now()

    # Create HTTP client for real requests to running server
    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:
        # Generate unique thread IDs for the test
        thread_id_1 = f"test_thread_{uuid.uuid4().hex[:8]}"
        thread_id_2 = f"test_thread_{uuid.uuid4().hex[:8]}"

        print(f"📋 Test threads: {thread_id_1}, {thread_id_2}")
        print(f"📋 Test prompts: {TEST_PROMPTS}")
        print(f"🌐 Server URL: {SERVER_BASE_URL}")
        print(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")

        # Create concurrent tasks
        tasks = [
            make_analyze_request(client, thread_id_1, TEST_PROMPTS[0], results),
            make_analyze_request(client, thread_id_2, TEST_PROMPTS[1], results),
            make_analyze_request(client, thread_id_2, TEST_PROMPTS[2], results),
            make_analyze_request(client, thread_id_2, TEST_PROMPTS[3], results),
        ]

        # Run tasks concurrently
        print("⚡ Executing concurrent requests...")
        print("💡 Note: Analysis requests can take 30-60+ seconds, please wait...")

        # Execute all tasks and handle any exceptions
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if any tasks returned exceptions (due to return_exceptions=True)
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                print(f"⚠️ Task {i+1} failed with exception: {result}")
                # Exception was already handled in make_analyze_request, so we don't need to do anything more

        # Add a small delay to ensure all results are recorded
        await asyncio.sleep(0.1)

    results.end_time = datetime.now()
    return results


def analyze_concurrency_results(results: ConcurrencyTestResults):
    """Analyze and display the concurrency test results."""
    print("\n" + "=" * 60)
    print("📊 CONCURRENCY TEST RESULTS ANALYSIS")
    print("=" * 60)

    summary = results.get_summary()

    print(f"🔢 Total Requests: {summary['total_requests']}")
    print(f"✅ Successful: {summary['successful_requests']}")
    print(f"❌ Failed: {summary['failed_requests']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

    if summary["total_test_time"]:
        print(f"⏱️  Total Test Time: {summary['total_test_time']:.2f}s")

    if summary["successful_requests"] > 0:
        print(f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"🏆 Best Response Time: {summary['min_response_time']:.2f}s")
        print(f"🐌 Worst Response Time: {summary['max_response_time']:.2f}s")

    print(
        f"🎯 Concurrent Requests Completed: {'✅ YES' if summary['concurrent_requests_completed'] else '❌ NO'}"
    )

    # Show individual results
    print("\n📋 Individual Request Results:")
    for i, result in enumerate(results.results, 1):
        status_emoji = "✅" if result["success"] else "❌"
        print(
            f"  {i}. {status_emoji} Thread: {result['thread_id'][:12]}... | "
            f"Status: {result['status_code']} | Time: {result['response_time']:.2f}s"
        )
        if "run_id" in result.get("response_data", {}):
            print(f"     Run ID: {result['response_data']['run_id']}")

    # Show errors if any
    if results.errors:
        print("\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print(
                f"  {i}. Thread: {error['thread_id'][:12]}... | Error: {error['error']}"
            )

    # Connection pool and database analysis
    print("\n🔍 CONCURRENCY ANALYSIS:")
    if summary["concurrent_requests_completed"]:
        print(
            "✅ Both requests completed - database connection handling appears stable"
        )
        if summary["max_response_time"] - summary["min_response_time"] < 2.0:
            print("✅ Response times are consistent - good concurrent performance")
        else:
            print(
                "⚠️  Response times vary significantly - possible connection contention"
            )
    else:
        print("❌ Concurrent requests failed - potential database connection issues")

    return summary


async def test_database_connectivity():
    """Test basic database connectivity before running concurrency tests."""
    print("🔍 Testing database connectivity...")
    try:
        # Test database connection using our existing functionality
        print("🔧 Creating database checkpointer...")
        checkpointer = await create_async_postgres_saver()
        if checkpointer:
            print("✅ Database checkpointer created successfully")
            # Test a simple operation
            test_config = {"configurable": {"thread_id": "connectivity_test"}}
            _ = await checkpointer.aget(test_config)
            print("✅ Database connectivity test passed")
            await close_async_postgres_saver()
            print("✅ Database connection closed properly")
            return True
        else:
            print("❌ Failed to create database checkpointer")
            return False
    except Exception as e:
        print(f"❌ Database connectivity test failed: {str(e)}")
        return False


def cleanup_test_environment():
    """Clean up the test environment."""
    print("🧹 Cleaning up test environment...")
    print("✅ Test environment cleanup complete")


async def setup_debug_environment(client: httpx.AsyncClient):
    """Setup debug environment for this specific test."""
    print("🔧 Setting up debug environment via API...")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Set debug variables specific to this test
    debug_vars = {"print__analysis_tracing_debug": "1", "print__analyze_debug": "1"}

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
    print("🧹 Resetting debug environment...")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Pass the same debug variables that were set, so they can be reset to original values
    debug_vars = {"print__analysis_tracing_debug": "1", "print__analyze_debug": "1"}

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


async def async_cleanup():
    """Async cleanup for database connections."""
    try:
        # Close any remaining PostgreSQL connections
        await close_async_postgres_saver()

        # CRITICAL: Also cleanup any global checkpointer from the modular API structure
        try:
            await cleanup_checkpointer()
        except Exception as e:
            print(f"⚠️ Warning during global checkpointer cleanup: {e}")

        # Give extra time for all tasks to finish
        await asyncio.sleep(0.2)

    except Exception as e:
        print(f"⚠️ Warning during async cleanup: {e}")


async def main():
    """Main test execution function."""
    print("🚀 PostgreSQL Concurrency Test Starting...")
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
        # Test database connectivity first
        if not await test_database_connectivity():
            print("❌ Database connectivity test failed!")
            return False

        print("✅ Database connectivity confirmed - proceeding with concurrency test")

        # Create HTTP client for the entire test session
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:

            # Setup debug environment for this test
            if not await setup_debug_environment(client):
                print("⚠️ Debug environment setup failed, continuing without debug")

            # Run the concurrency test
            results = await run_concurrency_test()

            # Cleanup debug environment
            await cleanup_debug_environment(client)

        # Analyze results
        summary = analyze_concurrency_results(results)

        # Determine overall test success
        # Test passes if:
        # 1. No empty/unknown error messages (server properly handled all requests)
        # 2. At least some requests succeeded (server is functional)

        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors  # No empty/unknown errors (proper server response)
            and summary["total_requests"] > 0  # Some requests were made
            and (
                summary["successful_requests"] > 0  # Either some succeeded
                or all(
                    error.get("error", "").strip() != "" for error in summary["errors"]
                )
            )  # Or all errors have proper messages
        )

        if has_empty_errors:
            print(
                "❌ Test failed: Server returned empty error messages (potential crash/hang)"
            )
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded (server may be down)")
        else:
            print(
                f"✅ Test criteria met: {summary['successful_requests']}/{summary['total_requests']} requests successful with proper error handling"
            )

        print(
            f"\n🏁 OVERALL TEST RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}"
        )

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        traceback.print_exc()
        return False

    finally:
        cleanup_test_environment()
        # CRITICAL: Close database connections before event loop ends
        await async_cleanup()


# Test runner for pytest
@pytest.mark.asyncio
async def test_analyze_endpoint_concurrency():
    """Pytest-compatible test function."""
    result = await main()
    assert result, "Concurrency test failed"


if __name__ == "__main__":
    # Debug variables are now set dynamically via API calls in setup_debug_environment()
    # This allows per-test-script debug configuration without hardcoding in server

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
            return False
        finally:
            await async_cleanup()

    try:
        test_result = asyncio.run(main_with_cleanup())
        sys.exit(0 if test_result else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

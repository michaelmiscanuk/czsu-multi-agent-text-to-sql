"""
Test for Phase 8.5: Chat Routes with Real HTTP Testing
Based on test_concurrency.py pattern - makes real HTTP requests to running server
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

from checkpointer.config import get_db_config, check_postgres_env_vars
from checkpointer.checkpointer.factory import (
    create_async_postgres_saver,
    close_async_postgres_saver,
    cleanup_checkpointer,
)

# Test configuration
TEST_EMAIL = "test_user@example.com"
TEST_THREAD_ID = f"test_thread_{uuid.uuid4().hex[:8]}"
TEST_THREAD_ID_2 = f"test_thread_{uuid.uuid4().hex[:8]}"

# Server configuration for real HTTP requests
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 30  # seconds for chat endpoints (less intensive than analyze)


def print_test_status(message: str):
    """Print test status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def create_test_jwt_token(email: str = TEST_EMAIL):
    """Create a simple test JWT token for authentication."""
    try:
        import jwt

        # Use the actual Google Client ID that the server expects
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
        # Use a simple test secret - in real scenarios this would be properly configured
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        print_test_status(
            f"🔧 TEST TOKEN: Created JWT token with correct audience: {google_client_id}"
        )
        return token
    except ImportError:
        print_test_status("⚠️ JWT library not available, using simple Bearer token")
        return "test_token_placeholder"


class ChatTestResults:
    """Class to track and analyze chat test results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None

    def add_result(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_data: Dict,
        response_time: float,
    ):
        """Add a successful test result."""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_data": response_data,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "success": 200 <= status_code < 300,
        }
        self.results.append(result)
        print_test_status(
            f"✅ Result: {method} {endpoint} - Status {status_code}, Time {response_time:.2f}s"
        )

    def add_error(
        self, endpoint: str, method: str, error: Exception, response_time: float = None
    ):
        """Add an error result."""
        error_info = {
            "endpoint": endpoint,
            "method": method,
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        print_test_status(f"❌ Error: {method} {endpoint} - {str(error)}")

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
        }


async def check_server_connectivity():
    """Check if the server is running and accessible."""
    print_test_status(f"🔍 Checking server connectivity at {SERVER_BASE_URL}...")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{SERVER_BASE_URL}/health")
            if response.status_code == 200:
                print_test_status("✅ Server is running and accessible")
                return True
            else:
                print_test_status(
                    f"❌ Server responded with status {response.status_code}"
                )
                return False
    except Exception as e:
        print_test_status(f"❌ Cannot connect to server: {e}")
        print_test_status(f"   Make sure uvicorn is running at {SERVER_BASE_URL}")
        return False


def setup_test_environment():
    """Set up the test environment and check prerequisites."""
    print_test_status("🔧 Setting up test environment...")

    # Check if PostgreSQL environment variables are set
    if not check_postgres_env_vars():
        print_test_status(
            "❌ PostgreSQL environment variables are not properly configured!"
        )
        print_test_status("Required variables: host, port, dbname, user, password")
        print_test_status(f"Current config: {get_db_config()}")
        return False

    print_test_status("✅ PostgreSQL environment variables are configured")

    # Check if USE_TEST_TOKENS is set for the server
    use_test_tokens = os.getenv("USE_TEST_TOKENS", "0")
    if use_test_tokens != "1":
        print_test_status(
            "⚠️  WARNING: USE_TEST_TOKENS environment variable is not set to '1'"
        )
        print_test_status("   The server needs USE_TEST_TOKENS=1 to accept test tokens")
        print_test_status(
            "   Set this environment variable in your server environment:"
        )
        print_test_status("   SET USE_TEST_TOKENS=1 (Windows)")
        print_test_status("   export USE_TEST_TOKENS=1 (Linux/Mac)")
        print_test_status(
            "   Continuing test anyway - this may cause 401 authentication errors"
        )
    else:
        print_test_status(
            "✅ USE_TEST_TOKENS=1 - test tokens will be accepted by server"
        )

    print_test_status("✅ Test environment setup complete")
    return True


async def test_get_thread_sentiments(
    client: httpx.AsyncClient, results: ChatTestResults
):
    """Test GET /chat/{thread_id}/sentiments endpoint."""
    print_test_status(f"🔍 Testing GET /chat/{TEST_THREAD_ID}/sentiments")
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            f"/chat/{TEST_THREAD_ID}/sentiments", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()
            results.add_result(
                f"/chat/{TEST_THREAD_ID}/sentiments",
                "GET",
                response.status_code,
                response_data,
                response_time,
            )
            print_test_status(
                f"✅ Sentiments response: {len(response_data)} sentiment entries"
            )
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                f"/chat/{TEST_THREAD_ID}/sentiments",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error(f"/chat/{TEST_THREAD_ID}/sentiments", "GET", e, response_time)


async def test_get_chat_threads(client: httpx.AsyncClient, results: ChatTestResults):
    """Test GET /chat-threads endpoint with pagination."""
    print_test_status("🔍 Testing GET /chat-threads")
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        # Test with pagination parameters
        params = {"page": 1, "limit": 10}
        response = await client.get("/chat-threads", headers=headers, params=params)
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()
            results.add_result(
                "/chat-threads",
                "GET",
                response.status_code,
                response_data,
                response_time,
            )
            print_test_status(
                f"✅ Chat threads response: {len(response_data.get('threads', []))} threads"
            )
            print_test_status(f"   Total count: {response_data.get('total_count', 0)}")
            print_test_status(f"   Has more: {response_data.get('has_more', False)}")
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                "/chat-threads", "GET", Exception(error_message), response_time
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error("/chat-threads", "GET", e, response_time)


async def test_get_all_messages_for_one_thread(
    client: httpx.AsyncClient, results: ChatTestResults
):
    """Test GET /chat/all-messages-for-one-thread/{thread_id} endpoint."""
    print_test_status(
        f"🔍 Testing GET /chat/all-messages-for-one-thread/{TEST_THREAD_ID}"
    )
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            f"/chat/all-messages-for-one-thread/{TEST_THREAD_ID}", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()
            results.add_result(
                f"/chat/all-messages-for-one-thread/{TEST_THREAD_ID}",
                "GET",
                response.status_code,
                response_data,
                response_time,
            )

            # Analyze the single thread response structure
            messages = response_data.get("messages", [])
            run_ids = response_data.get("runIds", [])
            sentiments = response_data.get("sentiments", {})

            print_test_status("✅ Single thread messages response:")
            print_test_status(f"   Messages: {len(messages)}")
            print_test_status(f"   Run IDs: {len(run_ids)}")
            print_test_status(f"   Sentiments: {len(sentiments)}")

            # Check for cache headers
            cache_control = response.headers.get("Cache-Control", "")
            etag = response.headers.get("ETag", "")
            if cache_control or etag:
                print_test_status(f"   Cache-Control: {cache_control}")
                print_test_status(f"   ETag: {etag}")

            # Validate response structure
            if isinstance(messages, list):
                print_test_status("✅ Messages is a list (single thread format)")
            else:
                print_test_status("⚠️ Messages is not a list - unexpected format")

            if isinstance(run_ids, list):
                print_test_status("✅ Run IDs is a list (single thread format)")
            else:
                print_test_status("⚠️ Run IDs is not a list - unexpected format")

        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                f"/chat/all-messages-for-one-thread/{TEST_THREAD_ID}",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error(
            f"/chat/all-messages-for-one-thread/{TEST_THREAD_ID}",
            "GET",
            e,
            response_time,
        )


async def test_delete_chat_checkpoints(
    client: httpx.AsyncClient, results: ChatTestResults
):
    """Test DELETE /chat/{thread_id} endpoint."""
    print_test_status(f"🔍 Testing DELETE /chat/{TEST_THREAD_ID_2}")
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.delete(f"/chat/{TEST_THREAD_ID_2}", headers=headers)
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()
            results.add_result(
                f"/chat/{TEST_THREAD_ID_2}",
                "DELETE",
                response.status_code,
                response_data,
                response_time,
            )
            print_test_status(
                f"✅ Delete response: {response_data.get('message', 'Success')}"
            )
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                f"/chat/{TEST_THREAD_ID_2}",
                "DELETE",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error(f"/chat/{TEST_THREAD_ID_2}", "DELETE", e, response_time)


async def run_chat_endpoints_test() -> ChatTestResults:
    """Run comprehensive tests for all chat endpoints."""
    print_test_status("🎯 Starting comprehensive chat endpoints test...")

    results = ChatTestResults()
    results.start_time = datetime.now()

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        print_test_status(f"🌐 Server URL: {SERVER_BASE_URL}")
        print_test_status(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")
        print_test_status(f"🔖 Test thread IDs: {TEST_THREAD_ID}, {TEST_THREAD_ID_2}")

        # Test all chat endpoints
        print("============================================================")
        print("🔍 TESTING GET THREAD SENTIMENTS")
        print("============================================================")
        await test_get_thread_sentiments(client, results)

        print("============================================================")
        print("🔍 TESTING GET CHAT THREADS")
        print("============================================================")
        await test_get_chat_threads(client, results)

        print("============================================================")
        print("🔍 TESTING GET ALL MESSAGES FOR ONE THREAD")
        print("============================================================")
        await test_get_all_messages_for_one_thread(client, results)

        print("============================================================")
        print("🔍 TESTING DELETE CHAT CHECKPOINTS")
        print("============================================================")
        await test_delete_chat_checkpoints(client, results)

        # Add a small delay to ensure all results are recorded
        await asyncio.sleep(0.1)

    results.end_time = datetime.now()
    return results


def analyze_chat_test_results(results: ChatTestResults):
    """Analyze and display the chat test results."""
    print_test_status("============================================================")
    print_test_status("============================================================")
    print_test_status("============================================================")
    print_test_status("📊 CHAT ENDPOINTS TEST RESULTS")
    print_test_status("============================================================")
    print_test_status("============================================================")
    print_test_status("============================================================")

    summary = results.get_summary()

    print_test_status(f"🔢 Total Requests: {summary['total_requests']}")
    print_test_status(f"✅ Successful: {summary['successful_requests']}")
    print_test_status(f"❌ Failed: {summary['failed_requests']}")
    print_test_status(f"📈 Success Rate: {summary['success_rate']:.1f}%")

    if summary["total_test_time"]:
        print_test_status(f"⏱️  Total Test Time: {summary['total_test_time']:.2f}s")

    if summary["successful_requests"] > 0:
        print_test_status(
            f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s"
        )
        print_test_status(f"🏆 Best Response Time: {summary['min_response_time']:.2f}s")
        print_test_status(
            f"🐌 Worst Response Time: {summary['max_response_time']:.2f}s"
        )

    # Show individual results
    print_test_status("\n📋 Individual Endpoint Results:")
    for i, result in enumerate(results.results, 1):
        status_emoji = "✅" if result["success"] else "❌"
        print_test_status(
            f"  {i}. {status_emoji} {result['method']} {result['endpoint']} | "
            f"Status: {result['status_code']} | Time: {result['response_time']:.2f}s"
        )

    # Show errors if any
    if results.errors:
        print_test_status("\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print_test_status(
                f"  {i}. {error['method']} {error['endpoint']} | Error: {error['error']}"
            )

    # Analysis
    print_test_status("\n🔍 CHAT ENDPOINTS ANALYSIS:")

    # Check which endpoints are working
    endpoint_status = {}
    for result in results.results:
        endpoint = result["endpoint"]
        if endpoint not in endpoint_status:
            endpoint_status[endpoint] = {"success": 0, "total": 0}
        endpoint_status[endpoint]["total"] += 1
        if result["success"]:
            endpoint_status[endpoint]["success"] += 1

    for error in results.errors:
        endpoint = error["endpoint"]
        if endpoint not in endpoint_status:
            endpoint_status[endpoint] = {"success": 0, "total": 0}
        endpoint_status[endpoint]["total"] += 1

    for endpoint, stats in endpoint_status.items():
        success_rate = (
            (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )
        status_emoji = (
            "✅" if success_rate == 100 else "⚠️" if success_rate > 0 else "❌"
        )
        print_test_status(
            f"{status_emoji} {endpoint}: {success_rate:.0f}% success ({stats['success']}/{stats['total']})"
        )

    return summary


async def test_database_connectivity():
    """Test basic database connectivity before running chat tests."""
    print_test_status("🔍 Testing database connectivity...")
    try:
        # Test database connection using our existing functionality
        print_test_status("🔧 Creating database checkpointer...")
        checkpointer = await create_async_postgres_saver()
        if checkpointer:
            print_test_status("✅ Database checkpointer created successfully")
            # Test a simple operation
            test_config = {"configurable": {"thread_id": "connectivity_test"}}
            _ = await checkpointer.aget(test_config)
            print_test_status("✅ Database connectivity test passed")
            await close_async_postgres_saver()
            print_test_status("✅ Database connection closed properly")
            return True
        else:
            print_test_status("❌ Failed to create database checkpointer")
            return False
    except Exception as e:
        print_test_status(f"❌ Database connectivity test failed: {str(e)}")
        return False


def cleanup_test_environment():
    """Clean up the test environment."""
    print_test_status("🧹 Cleaning up test environment...")
    print_test_status("✅ Test environment cleanup complete")


async def setup_debug_environment(client: httpx.AsyncClient):
    """Setup debug environment for this specific test."""
    print_test_status("🔧 Setting up debug environment via API...")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Set debug variables specific to this test
    debug_vars = {
        "print__chat_all_messages_debug": "1",
        "print__chat_sentiments_debug": "1",
        "print__chat_threads_debug": "1",
        "print__delete_chat_debug": "1",
        "print__sentiment_flow": "1",
    }

    try:
        response = await client.post("/debug/set-env", headers=headers, json=debug_vars)
        if response.status_code == 200:
            print_test_status("✅ Debug environment configured successfully")
            return True
        else:
            print_test_status(f"⚠️ Debug setup failed: {response.status_code}")
            return False
    except Exception as e:
        print_test_status(f"⚠️ Debug setup error: {e}")
        return False


async def cleanup_debug_environment(client: httpx.AsyncClient):
    """Reset debug environment after test."""
    print_test_status("🧹 Resetting debug environment...")

    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Pass the same debug variables that were set, so they can be reset to original values
    debug_vars = {
        "print__chat_all_messages_debug": "1",
        "print__chat_sentiments_debug": "1",
        "print__chat_threads_debug": "1",
        "print__delete_chat_debug": "1",
        "print__sentiment_flow": "1",
    }

    try:
        response = await client.post(
            "/debug/reset-env", headers=headers, json=debug_vars
        )
        if response.status_code == 200:
            print_test_status("✅ Debug environment reset to original .env values")
        else:
            print_test_status(f"⚠️ Debug reset failed: {response.status_code}")
    except Exception as e:
        print_test_status(f"⚠️ Debug reset error: {e}")


async def async_cleanup():
    """Async cleanup for database connections."""
    try:
        # Close any remaining PostgreSQL connections
        await close_async_postgres_saver()

        # CRITICAL: Also cleanup any global checkpointer from the modular API structure
        try:
            await cleanup_checkpointer()
        except Exception as e:
            print_test_status(f"⚠️ Warning during global checkpointer cleanup: {e}")

        # Give extra time for all tasks to finish
        await asyncio.sleep(0.2)

    except Exception as e:
        print_test_status(f"⚠️ Warning during async cleanup: {e}")


async def main():
    """Main test execution function."""
    print_test_status("🚀 Chat Endpoints Test Starting...")
    print_test_status("=" * 60)

    # Check if server is running
    if not await check_server_connectivity():
        print_test_status("❌ Server connectivity check failed!")
        print_test_status("   Please start your uvicorn server first:")
        print_test_status(f"   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False

    # Setup test environment
    if not setup_test_environment():
        print_test_status("❌ Test environment setup failed!")
        return False

    try:
        # Test database connectivity first
        if not await test_database_connectivity():
            print_test_status("❌ Database connectivity test failed!")
            return False

        print_test_status(
            "✅ Database connectivity confirmed - proceeding with chat endpoints test"
        )

        # Create HTTP client for the entire test session
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:

            # Setup debug environment for this test
            if not await setup_debug_environment(client):
                print_test_status(
                    "⚠️ Debug environment setup failed, continuing without debug"
                )

            # Run the chat endpoints test
            results = await run_chat_endpoints_test()

            # Cleanup debug environment
            await cleanup_debug_environment(client)

        # Analyze results
        summary = analyze_chat_test_results(results)

        # Determine overall test success
        # Test passes if:
        # 1. At least 75% of requests succeeded (allows for some endpoints to be empty but functional)
        # 2. No empty/unknown error messages (server properly handled all requests)

        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors  # No empty/unknown errors
            and summary["total_requests"] > 0  # Some requests were made
            and summary["success_rate"] >= 75.0  # At least 75% success rate
        )

        if has_empty_errors:
            print_test_status(
                "❌ Test failed: Server returned empty error messages (potential crash/hang)"
            )
        elif summary["success_rate"] < 75.0:
            print_test_status(
                f"❌ Test failed: Success rate too low ({summary['success_rate']:.1f}% < 75%)"
            )
        else:
            print_test_status(
                f"✅ Test criteria met: {summary['success_rate']:.1f}% success rate with proper error handling"
            )

        print_test_status(
            f"\n🏁 OVERALL TEST RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}"
        )

        return test_passed

    except Exception as e:
        print_test_status(f"❌ Test execution failed: {str(e)}")
        traceback.print_exc()
        return False

    finally:
        cleanup_test_environment()
        # CRITICAL: Close database connections before event loop ends
        await async_cleanup()


# Test runner for pytest
@pytest.mark.asyncio
async def test_chat_endpoints():
    """Pytest-compatible test function."""
    result = await main()
    assert result, "Chat endpoints test failed"


if __name__ == "__main__":
    # Debug variables are now set dynamically via API calls in setup_debug_environment()
    # This allows per-test-script debug configuration without hardcoding in server

    async def main_with_cleanup():
        try:
            result = await main()
            return result
        except KeyboardInterrupt:
            print_test_status("\n⛔ Test interrupted by user")
            return False
        except Exception as e:
            print_test_status(f"\n💥 Unexpected error: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            await async_cleanup()

    try:
        test_result = asyncio.run(main_with_cleanup())
        sys.exit(0 if test_result else 1)
    except Exception as e:
        print_test_status(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

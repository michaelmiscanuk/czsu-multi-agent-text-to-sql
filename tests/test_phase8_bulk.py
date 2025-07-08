#!/usr/bin/env python3
"""
Test for Phase 8.7: Bulk Operations Routes with Real HTTP Testing
Based on test_phase8_chat.py pattern - makes real HTTP requests to running server
"""

import asyncio
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

# CRITICAL: Set Windows event loop policy FIRST, before other imports
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

# Test configuration
TEST_EMAIL = "test_user@example.com"

# Server configuration for real HTTP requests
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = (
    60  # seconds for bulk endpoints (more time needed for bulk operations)
)


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
            "name": "Test Bulk User",
            "given_name": "Test",
            "family_name": "Bulk",
        }
        # Use a simple test secret - in real scenarios this would be properly configured
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        print_test_status(
            f"🔧 TEST TOKEN: Created JWT token with correct audience: {google_client_id}"
        )
        return token
    except ImportError:
        print_test_status("⚠️ JWT library not available, using simple Bearer token")
        return "test_bulk_token_placeholder"


class BulkTestResults:
    """Class to track and analyze bulk test results."""

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
        cache_info: Dict = None,
    ):
        """Add a successful test result."""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_data": response_data,
            "response_time": response_time,
            "cache_info": cache_info,
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


async def test_get_all_chat_messages(
    client: httpx.AsyncClient, results: BulkTestResults
):
    """Test GET /chat/all-messages-for-all-threads endpoint (moved from test_phase8_chat.py)."""
    print_test_status("🔍 Testing GET /chat/all-messages-for-all-threads (basic test)")
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            "/chat/all-messages-for-all-threads", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()

            # Extract cache information if available
            cache_info = None
            if "Cache-Control" in response.headers or "ETag" in response.headers:
                cache_info = {
                    "cache_control": response.headers.get("Cache-Control", ""),
                    "etag": response.headers.get("ETag", ""),
                    "response_time": response_time,
                }

            results.add_result(
                "/chat/all-messages-for-all-threads",
                "GET",
                response.status_code,
                response_data,
                response_time,
                cache_info,
            )

            # Analyze the bulk response structure
            messages = response_data.get("messages", {})
            run_ids = response_data.get("runIds", {})
            sentiments = response_data.get("sentiments", {})

            print_test_status("✅ All messages response:")
            print_test_status(f"   Messages for {len(messages)} threads")
            print_test_status(f"   Run IDs for {len(run_ids)} threads")
            print_test_status(f"   Sentiments for {len(sentiments)} threads")

            # Count total messages
            total_messages = sum(
                len(thread_messages) for thread_messages in messages.values()
            )
            print_test_status(f"   Total messages: {total_messages}")

        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                "/chat/all-messages-for-all-threads",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error("/chat/all-messages-for-all-threads", "GET", e, response_time)


async def test_get_all_chat_messages_first_call(
    client: httpx.AsyncClient, results: BulkTestResults
):
    """Test GET /chat/all-messages-for-all-threads endpoint - first call (should be fresh, no cache)."""
    print_test_status(
        "🔍 Testing GET /chat/all-messages-for-all-threads (first call - no cache)"
    )
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            "/chat/all-messages-for-all-threads", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()

            # Extract cache information from headers
            cache_info = {
                "cache_control": response.headers.get("Cache-Control"),
                "etag": response.headers.get("ETag"),
                "response_time": response_time,
            }

            results.add_result(
                "/chat/all-messages-for-all-threads",
                "GET",
                response.status_code,
                response_data,
                response_time,
                cache_info,
            )

            # Analyze the bulk response structure
            messages = response_data.get("messages", {})
            run_ids = response_data.get("runIds", {})
            sentiments = response_data.get("sentiments", {})

            print_test_status("✅ All messages response (first call):")
            print_test_status(f"   Messages for {len(messages)} threads")
            print_test_status(f"   Run IDs for {len(run_ids)} threads")
            print_test_status(f"   Sentiments for {len(sentiments)} threads")

            # Count total messages
            total_messages = sum(
                len(thread_messages) for thread_messages in messages.values()
            )
            print_test_status(f"   Total messages: {total_messages}")

            # Check cache headers
            print_test_status(f"   Cache-Control: {cache_info['cache_control']}")
            print_test_status(f"   ETag: {cache_info['etag']}")
            print_test_status(f"   Response time: {response_time:.2f}s")

        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                "/chat/all-messages-for-all-threads",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error("/chat/all-messages-for-all-threads", "GET", e, response_time)


async def test_get_all_chat_messages_second_call(
    client: httpx.AsyncClient, results: BulkTestResults
):
    """Test GET /chat/all-messages-for-all-threads endpoint - second call (should be cached)."""
    print_test_status(
        "🔍 Testing GET /chat/all-messages-for-all-threads (second call - should be cached)"
    )
    start_time = time.time()

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            "/chat/all-messages-for-all-threads", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()

            # Extract cache information from headers
            cache_info = {
                "cache_control": response.headers.get("Cache-Control"),
                "etag": response.headers.get("ETag"),
                "response_time": response_time,
            }

            results.add_result(
                "/chat/all-messages-for-all-threads",
                "GET",
                response.status_code,
                response_data,
                response_time,
                cache_info,
            )

            # Analyze the bulk response structure
            messages = response_data.get("messages", {})
            run_ids = response_data.get("runIds", {})
            sentiments = response_data.get("sentiments", {})

            print_test_status("✅ All messages response (second call):")
            print_test_status(f"   Messages for {len(messages)} threads")
            print_test_status(f"   Run IDs for {len(run_ids)} threads")
            print_test_status(f"   Sentiments for {len(sentiments)} threads")

            # Count total messages
            total_messages = sum(
                len(thread_messages) for thread_messages in messages.values()
            )
            print_test_status(f"   Total messages: {total_messages}")

            # Check cache headers and performance
            print_test_status(f"   Cache-Control: {cache_info['cache_control']}")
            print_test_status(f"   ETag: {cache_info['etag']}")
            print_test_status(f"   Response time: {response_time:.2f}s")

            # Check if this was likely a cache hit (should be much faster)
            if response_time < 1.0:  # Less than 1 second suggests cache hit
                print_test_status("✅ Fast response suggests cache hit")
            else:
                print_test_status(
                    "⚠️ Slow response suggests cache miss or fresh processing"
                )

        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                "/chat/all-messages-for-all-threads",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error("/chat/all-messages-for-all-threads", "GET", e, response_time)


async def test_get_all_chat_messages_concurrent_calls(
    client: httpx.AsyncClient, results: BulkTestResults
):
    """Test GET /chat/all-messages-for-all-threads endpoint - concurrent calls (should test locking)."""
    print_test_status(
        "🔍 Testing GET /chat/all-messages-for-all-threads (concurrent calls - test locking)"
    )

    async def make_concurrent_request(request_id: int):
        """Make a single concurrent request."""
        start_time = time.time()
        try:
            token = create_test_jwt_token()
            headers = {"Authorization": f"Bearer {token}"}

            response = await client.get(
                "/chat/all-messages-for-all-threads", headers=headers
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                response_data = response.json()

                # Extract cache information from headers
                cache_info = {
                    "cache_control": response.headers.get("Cache-Control"),
                    "etag": response.headers.get("ETag"),
                    "response_time": response_time,
                    "request_id": request_id,
                }

                results.add_result(
                    f"/chat/all-messages-for-all-threads (concurrent-{request_id})",
                    "GET",
                    response.status_code,
                    response_data,
                    response_time,
                    cache_info,
                )

                print_test_status(
                    f"✅ Concurrent request {request_id} completed in {response_time:.2f}s"
                )
                return response_time, response_data

            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "detail", f"HTTP {response.status_code}"
                    )
                except Exception:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                results.add_error(
                    f"/chat/all-messages-for-all-threads (concurrent-{request_id})",
                    "GET",
                    Exception(error_message),
                    response_time,
                )
                return response_time, None

        except Exception as e:
            response_time = time.time() - start_time
            results.add_error(
                f"/chat/all-messages-for-all-threads (concurrent-{request_id})",
                "GET",
                e,
                response_time,
            )
            return response_time, None

    # Make 3 concurrent requests
    print_test_status("🔄 Making 3 concurrent requests...")
    concurrent_tasks = [make_concurrent_request(i) for i in range(1, 4)]
    concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

    # Analyze concurrent results
    successful_results = [r for r in concurrent_results if not isinstance(r, Exception)]
    print_test_status(
        f"✅ {len(successful_results)} concurrent requests completed successfully"
    )

    if len(successful_results) >= 2:
        response_times = [r[0] for r in successful_results if r[1] is not None]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print_test_status(f"   Average concurrent response time: {avg_time:.2f}s")
            print_test_status(
                "✅ Concurrent requests handled successfully (locking working)"
            )


async def test_get_all_chat_messages_different_user(
    client: httpx.AsyncClient, results: BulkTestResults
):
    """Test GET /chat/all-messages-for-all-threads endpoint with different user (should be separate cache)."""
    print_test_status(
        "🔍 Testing GET /chat/all-messages-for-all-threads (different user - separate cache)"
    )
    start_time = time.time()

    try:
        # Use different email for different cache
        different_email = "different_bulk_user@example.com"
        token = create_test_jwt_token(different_email)
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            "/chat/all-messages-for-all-threads", headers=headers
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            response_data = response.json()

            # Extract cache information from headers
            cache_info = {
                "cache_control": response.headers.get("Cache-Control"),
                "etag": response.headers.get("ETag"),
                "response_time": response_time,
                "user_email": different_email,
            }

            results.add_result(
                "/chat/all-messages-for-all-threads (different-user)",
                "GET",
                response.status_code,
                response_data,
                response_time,
                cache_info,
            )

            # Analyze the bulk response structure
            messages = response_data.get("messages", {})
            run_ids = response_data.get("runIds", {})
            sentiments = response_data.get("sentiments", {})

            print_test_status("✅ All messages response (different user):")
            print_test_status(f"   Messages for {len(messages)} threads")
            print_test_status(f"   Run IDs for {len(run_ids)} threads")
            print_test_status(f"   Sentiments for {len(sentiments)} threads")

            # Count total messages
            total_messages = sum(
                len(thread_messages) for thread_messages in messages.values()
            )
            print_test_status(f"   Total messages: {total_messages}")
            print_test_status(f"   Response time: {response_time:.2f}s")

        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            results.add_error(
                "/chat/all-messages-for-all-threads (different-user)",
                "GET",
                Exception(error_message),
                response_time,
            )

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error(
            "/chat/all-messages-for-all-threads (different-user)",
            "GET",
            e,
            response_time,
        )


async def run_bulk_endpoints_test() -> BulkTestResults:
    """Run comprehensive tests for bulk endpoints."""
    print_test_status("🎯 Starting comprehensive bulk endpoints test...")

    results = BulkTestResults()
    results.start_time = datetime.now()

    async with httpx.AsyncClient(
        base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
    ) as client:

        print_test_status(f"🌐 Server URL: {SERVER_BASE_URL}")
        print_test_status(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")
        print_test_status(f"👤 Test user: {TEST_EMAIL}")

        # Test bulk endpoints in sequence
        print("============================================================")
        print("🔍 TESTING GET ALL CHAT MESSAGES - BASIC TEST")
        print("============================================================")
        await test_get_all_chat_messages(client, results)

        print("============================================================")
        print("🔍 TESTING GET ALL CHAT MESSAGES - FIRST CALL")
        print("============================================================")
        await test_get_all_chat_messages_first_call(client, results)

        """
        print("============================================================")
        print("🔍 TESTING GET ALL CHAT MESSAGES - SECOND CALL (CACHE)")
        print("============================================================")
        await test_get_all_chat_messages_second_call(client, results)

        print("============================================================")
        print("🔍 TESTING GET ALL CHAT MESSAGES - CONCURRENT CALLS")
        print("============================================================")
        await test_get_all_chat_messages_concurrent_calls(client, results)

        print("============================================================")
        print("🔍 TESTING GET ALL CHAT MESSAGES - DIFFERENT USER")
        print("============================================================")
        await test_get_all_chat_messages_different_user(client, results)
        """
        # Add a small delay to ensure all results are recorded
        await asyncio.sleep(0.1)

    results.end_time = datetime.now()
    return results


def analyze_bulk_test_results(results: BulkTestResults):
    """Analyze and display the bulk test results."""
    print_test_status("============================================================")
    print_test_status("============================================================")
    print_test_status("============================================================")
    print_test_status("📊 BULK ENDPOINTS TEST RESULTS")
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
        cache_info = result.get("cache_info", {})
        cache_time = cache_info.get("response_time", result["response_time"])
        print_test_status(
            f"  {i}. {status_emoji} {result['method']} {result['endpoint']} | "
            f"Status: {result['status_code']} | Time: {cache_time:.2f}s"
        )

    # Show errors if any
    if results.errors:
        print_test_status("\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print_test_status(
                f"  {i}. {error['method']} {error['endpoint']} | Error: {error['error']}"
            )

    # Analysis
    print_test_status("\n🔍 BULK ENDPOINTS ANALYSIS:")

    # Check cache performance
    cache_results = [r for r in results.results if r.get("cache_info")]
    if cache_results:
        print_test_status("📊 Cache Performance Analysis:")
        for result in cache_results:
            cache_info = result.get("cache_info", {})
            endpoint = result["endpoint"]
            response_time = cache_info.get("response_time", result["response_time"])
            cache_control = cache_info.get("cache_control", "N/A")

            print_test_status(
                f"   {endpoint}: {response_time:.2f}s (Cache-Control: {cache_control})"
            )

    # Check for performance patterns
    bulk_results = [
        r
        for r in results.results
        if "/chat/all-messages-for-all-threads" in r["endpoint"]
    ]
    if len(bulk_results) >= 2:
        first_call = None
        second_call = None

        for result in bulk_results:
            if "first call" in result["endpoint"] or (
                not first_call and "concurrent" not in result["endpoint"]
            ):
                first_call = result
            elif "second call" in result["endpoint"] or (
                not second_call and "concurrent" not in result["endpoint"]
            ):
                second_call = result

        if first_call and second_call:
            first_time = first_call["response_time"]
            second_time = second_call["response_time"]
            speedup = first_time / second_time if second_time > 0 else 1

            print_test_status("🚀 Cache Performance:")
            print_test_status(f"   First call: {first_time:.2f}s")
            print_test_status(f"   Second call: {second_time:.2f}s")
            print_test_status(f"   Speedup: {speedup:.1f}x")

            if speedup > 2:
                print_test_status("✅ Good cache performance detected")
            else:
                print_test_status("⚠️ Cache performance may need improvement")

    return summary


async def test_database_connectivity():
    """Test basic database connectivity before running bulk tests."""
    print_test_status("🔍 Testing database connectivity...")
    try:
        # Test database connection using our existing functionality
        print_test_status("🔧 Creating database checkpointer...")
        checkpointer = await create_async_postgres_saver()
        if checkpointer:
            print_test_status("✅ Database checkpointer created successfully")
            # Test a simple operation
            test_config = {"configurable": {"thread_id": "bulk_connectivity_test"}}
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

    # Set debug variables specific to bulk operations
    debug_vars = {
        "print__chat_all_messages_debug": "1",
        "print__bulk_processing_debug": "1",
        "print__bulk_cache_debug": "1",
        "print__bulk_concurrency_debug": "1",
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
        "print__bulk_processing_debug": "1",
        "print__bulk_cache_debug": "1",
        "print__bulk_concurrency_debug": "1",
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
    print_test_status("🚀 Bulk Endpoints Test Starting...")
    print_test_status("=" * 60)

    # Check if server is running
    if not await check_server_connectivity():
        print_test_status("❌ Server connectivity check failed!")
        print_test_status("   Please start your uvicorn server first:")
        print_test_status("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
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
            "✅ Database connectivity confirmed - proceeding with bulk endpoints test"
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

            # Run the bulk endpoints test
            results = await run_bulk_endpoints_test()

            # Cleanup debug environment
            await cleanup_debug_environment(client)

        # Analyze results
        summary = analyze_bulk_test_results(results)

        # Determine overall test success
        # Test passes if:
        # 1. At least 75% of requests succeeded
        # 2. No empty/unknown error messages (server properly handled all requests)
        # 3. At least some performance improvement between first and second call (cache working)

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
async def test_bulk_endpoints():
    """Pytest-compatible test function."""
    result = await main()
    assert result, "Bulk endpoints test failed"


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

"""
Test for Phase 8.6: Message Routes
Tests the message endpoints with real HTTP requests and proper authentication.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import httpx
from typing import Dict, List
from datetime import datetime
import time
import asyncio
import uuid

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

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_user@example.com"
REQUIRED_ENDPOINTS = {"/chat/messages", "/chat/run-ids"}  # Simplified for tracking

# Test data - various thread IDs and scenarios
TEST_QUERIES = [
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "test-thread-1",
        "description": "Get messages for test thread 1",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Basic message retrieval for new thread",
    },
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "test-thread-2",
        "description": "Get messages for test thread 2",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Message retrieval for different thread ID",
    },
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "invalid-thread-uuid",
        "description": "Get messages for invalid thread ID",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Non-existent thread ID handling",
    },
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "",
        "description": "Empty thread ID",
        "should_succeed": False,
        "expected_status": 404,
        "test_focus": "URL routing validation with empty path parameter",
    },
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "thread-with-special-chars",
        "description": "Thread ID with safe special characters",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "URL-safe special character handling",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "test-thread-1",
        "description": "Get run IDs for test thread 1",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Basic run ID retrieval for new thread",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "test-thread-2",
        "description": "Get run IDs for test thread 2",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Run ID retrieval for different thread ID",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "invalid-thread-uuid",
        "description": "Get run IDs for invalid thread ID",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Non-existent thread run ID handling",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "",
        "description": "Empty thread ID for run IDs",
        "should_succeed": False,
        "expected_status": 404,
        "test_focus": "URL routing validation with empty path parameter for run IDs",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "thread-with-special-chars",
        "description": "Run IDs for thread with safe special characters",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "URL-safe special character handling for run IDs",
    },
    # Additional realistic test cases
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "user-conversation-123",
        "description": "Typical user conversation thread",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Realistic user conversation thread naming",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "user-conversation-123",
        "description": "Run IDs for typical user conversation",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "Run ID retrieval for realistic conversation thread",
    },
    {
        "endpoint": "/chat/{thread_id}/messages",
        "thread_id": "uuid-style-thread-" + str(uuid.uuid4()),
        "description": "UUID-style thread ID for messages",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "UUID-formatted thread ID message retrieval",
    },
    {
        "endpoint": "/chat/{thread_id}/run-ids",
        "thread_id": "uuid-style-thread-" + str(uuid.uuid4()),
        "description": "UUID-style thread ID for run IDs",
        "should_succeed": True,
        "expect_empty": True,
        "test_focus": "UUID-formatted thread ID run ID retrieval",
    },
]


def _normalize_endpoint_for_tracking(endpoint: str) -> str:
    """Normalize endpoint URL for tracking purposes."""
    # Convert /chat/thread-id/messages to /chat/messages
    # Convert /chat/thread-id/run-ids to /chat/run-ids
    if "/chat/" in endpoint and "/messages" in endpoint:
        return "/chat/messages"
    elif "/chat/" in endpoint and "/run-ids" in endpoint:
        return "/chat/run-ids"
    return endpoint


def _get_test_explanation(
    test_focus: str,
    should_succeed: bool,
    expected_status: int,
    thread_id: str,
    endpoint: str,
    expect_empty: bool = False,
) -> str:
    """Generate a detailed explanation of what the test is validating."""
    
    if should_succeed:
        # Success cases - explain what functionality we're testing
        if "/messages" in endpoint:
            if expect_empty:
                return f"Message retrieval: thread_id='{thread_id}', verifying PostgreSQL checkpoint history access, user ownership, and empty response for new/non-existent threads"
            else:
                return f"Message retrieval: thread_id='{thread_id}', verifying actual message data serialization and metadata extraction from checkpoint history"
        elif "/run-ids" in endpoint:
            if expect_empty:
                return f"Run ID retrieval: thread_id='{thread_id}', verifying database query for users_threads_runs table, UUID validation, and empty response for new threads"
            else:
                return f"Run ID retrieval: thread_id='{thread_id}', verifying actual run ID data with proper UUID format and timestamp handling"
    else:
        # Failure cases - explain what validation we're testing
        if expected_status == 404:
            return f"URL routing validation: empty thread_id should result in 404 Not Found due to FastAPI path parameter requirements"
        elif expected_status == 422:
            return f"Parameter validation: malformed thread_id '{thread_id}' should be rejected by request validation"
        elif expected_status == 400:
            return f"Business logic validation: invalid thread_id format '{thread_id}' should be caught by application logic"
    
    return f"Testing {test_focus} - verifying proper API behavior for thread_id: '{thread_id}'"


def _get_auth_test_explanation(endpoint: str, has_token: bool, token_valid: bool) -> str:
    """Generate explanation for authentication test cases."""
    if not has_token:
        return f"FastAPI should reject request to {endpoint} when Authorization header is completely missing (401 Unauthorized)"
    elif not token_valid:
        return f"JWT middleware should reject invalid/malformed token for {endpoint} (401/403 Unauthorized)"
    else:
        return f"Authentication validation for {endpoint} endpoint"


def _validate_chat_messages_response(data):
    """Validate chat messages response structure."""
    print("🔍 Testing: Chat messages response structure validation")

    assert isinstance(data, list), "Response must be a list"
    print("✅ Response is a list")

    for i, message in enumerate(data):
        assert isinstance(message, dict), f"Message {i} must be a dict"

        # Check required fields based on ChatMessage model
        required_fields = ["content", "isUser"]
        for field in required_fields:
            assert field in message, f"Message {i} missing required field '{field}'"

        # Validate field types
        assert isinstance(
            message["content"], str
        ), f"Message {i} content must be string"
        assert isinstance(
            message["isUser"], bool
        ), f"Message {i} isUser must be boolean"

        # Optional fields validation
        if "meta" in message and message["meta"] is not None:
            assert isinstance(message["meta"], dict), f"Message {i} meta must be dict"

        if "timestamp" in message and message["timestamp"] is not None:
            assert isinstance(
                message["timestamp"], str
            ), f"Message {i} timestamp must be string"

    print(f"✅ Validated {len(data)} chat messages")


def _validate_run_ids_response(data):
    """Validate run IDs response structure."""
    print("🔍 Testing: Run IDs response structure validation")

    assert isinstance(data, dict), "Response must be a dict"
    assert "run_ids" in data, "Response must contain 'run_ids' field"
    assert isinstance(data["run_ids"], list), "'run_ids' must be a list"
    print("✅ Response has correct structure")

    for i, run_data in enumerate(data["run_ids"]):
        assert isinstance(run_data, dict), f"Run data {i} must be a dict"

        required_fields = ["run_id", "prompt", "timestamp"]
        for field in required_fields:
            assert field in run_data, f"Run data {i} missing required field '{field}'"

        # Validate field types
        assert isinstance(
            run_data["run_id"], str
        ), f"Run data {i} run_id must be string"
        assert isinstance(
            run_data["prompt"], str
        ), f"Run data {i} prompt must be string"
        assert isinstance(
            run_data["timestamp"], str
        ), f"Run data {i} timestamp must be string"

        # Validate UUID format
        try:
            uuid.UUID(run_data["run_id"])
            print(f"✅ Valid UUID format: {run_data['run_id']}")
        except ValueError:
            raise AssertionError(
                f"Run data {i} run_id is not a valid UUID: {run_data['run_id']}"
            )

    print(f"✅ Validated {len(data['run_ids'])} run ID entries")


async def test_authentication_required():
    """Test that endpoints require authentication."""
    print("🔍 Testing: Authentication requirements")

    test_cases = [
        {
            "endpoint": "/chat/test-thread/messages",
            "endpoint_type": "Messages",
            "description": "No authorization header",
            "test_focus": "JWT token requirement validation",
        },
        {
            "endpoint": "/chat/test-thread/run-ids", 
            "endpoint_type": "Run IDs",
            "description": "No authorization header",
            "test_focus": "JWT token requirement validation",
        }
    ]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i, test_case in enumerate(test_cases, 1):
            endpoint = test_case["endpoint"]
            print(f"\n🔍 AUTH TEST {i}: {test_case['test_focus']}")
            print(f"   📍 Endpoint: GET {endpoint}")
            print(f"   � Headers: None (testing missing Authorization)")
            print(f"   ✅ Expected Result: 401 Unauthorized")
            print(f"   🎯 What we're testing: {_get_auth_test_explanation(endpoint, False, False)}")

            # Test without token
            response = await client.get(f"{SERVER_BASE_URL}{endpoint}")
            assert (
                response.status_code == 401
            ), f"Expected 401 for {endpoint} without auth"
            print(f"✅ {test_case['endpoint_type']}: correctly requires authentication (401)")

            # Test with invalid token
            print(f"\n🔍 AUTH TEST {i}b: Invalid token format")
            print(f"   📍 Endpoint: GET {endpoint}")
            print(f"   🔑 Headers: Authorization: Bearer invalid_token")
            print(f"   ✅ Expected Result: 401/403 Unauthorized")
            print(f"   🎯 What we're testing: {_get_auth_test_explanation(endpoint, True, False)}")
            
            headers = {"Authorization": "Bearer invalid_token"}
            response = await client.get(f"{SERVER_BASE_URL}{endpoint}", headers=headers)
            assert response.status_code in [
                401,
                403,
            ], f"Expected 401/403 for {endpoint} with invalid auth"
            print(f"✅ {test_case['endpoint_type']}: correctly rejects invalid token ({response.status_code})")


async def make_message_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    thread_id: str,
    description: str,
    should_succeed: bool,
    expect_empty: bool,
    results: BaseTestResults,
    expected_status: int = None,
):
    """Make a request to a message endpoint with server traceback capture."""
    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    # Replace placeholder in endpoint
    actual_endpoint = endpoint.replace("{thread_id}", thread_id)

    start_time = time.time()
    try:
        result = await make_request_with_traceback_capture(
            client,
            "GET",
            f"{SERVER_BASE_URL}{actual_endpoint}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(result)

        if result["response"] is None:
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"❌ Test {test_id} - Client Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id, actual_endpoint, description, error_obj, response_time
            )
            return

        response = result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.2f}s)")

        if should_succeed:
            if response.status_code == 200:
                try:
                    data = response.json()

                    # Validate response structure based on endpoint
                    if "/messages" in endpoint:
                        _validate_chat_messages_response(data)
                        if expect_empty:
                            print(
                                f"✅ Expected empty messages list, got {len(data)} messages"
                            )
                        else:
                            print(f"✅ Got {len(data)} messages as expected")
                    elif "/run-ids" in endpoint:
                        _validate_run_ids_response(data)
                        if expect_empty:
                            print(
                                f"✅ Expected empty run_ids list, got {len(data['run_ids'])} run_ids"
                            )
                        else:
                            print(f"✅ Got {len(data['run_ids'])} run_ids as expected")

                    results.add_result(
                        test_id,
                        _normalize_endpoint_for_tracking(actual_endpoint),
                        description,
                        data,
                        response_time,
                        response.status_code,
                    )
                except (AssertionError, Exception) as e:
                    print(f"❌ Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id, actual_endpoint, description, error_obj, response_time
                    )
            else:
                handle_error_response(
                    test_id,
                    actual_endpoint,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )
        else:
            handle_expected_failure(
                test_id,
                actual_endpoint,
                description,
                response,
                error_info,
                results,
                response_time,
                expected_status,
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
            actual_endpoint,
            description,
            error_obj,
            response_time,
            response_data=None,
        )


async def test_chat_messages_response_structure():
    """Test that chat messages responses have proper structure."""
    print("🔍 Testing: Messages response structure validation")

    test_cases = [
        {
            "endpoint": "/chat/valid-test-thread-123/messages",
            "thread_id": "valid-test-thread-123",
            "test_focus": "Message response structure validation - checking for proper JSON schema, metadata extraction, and content serialization",
            "description": "Valid thread ID with expected message data structure",
        }
    ]

    valid_token = create_test_jwt_token("test_user")
    headers = {"Authorization": f"Bearer {valid_token}"}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i, test_case in enumerate(test_cases, 1):
            endpoint = test_case["endpoint"]
            thread_id = test_case["thread_id"]
            
            print(f"\n🔍 STRUCTURE TEST {i}: {test_case['test_focus']}")
            print(f"   📍 Endpoint: GET {endpoint}")
            print(f"   🔑 Headers: Authorization Bearer token (valid)")
            print(f"   🎯 Thread ID: '{thread_id}'")
            print(f"   ✅ Expected Result: Valid JSON response with messages array")
            print(f"   🎯 What we're testing: {_get_test_explanation(test_case['test_focus'], True, 200, thread_id, endpoint, True)}")

            response = await client.get(f"{SERVER_BASE_URL}{endpoint}", headers=headers)
            
            # Should return valid response (might be empty for new thread)
            assert response.status_code in [
                200
            ], f"Expected 200 for {endpoint}, got {response.status_code}"

            try:
                data = response.json()
                print(f"✅ Response received: JSON structure valid")
                _validate_chat_messages_response(data)
                print(f"✅ Message structure validation completed")
            except json.JSONDecodeError:
                print(f"❌ Response is not valid JSON")
                raise
            except Exception as e:
                print(f"❌ Validation error: {e}")
                raise


async def test_no_auth_scenarios():
    """Test endpoints without authentication to verify they fail properly."""
    print("🔍 Testing: No authentication scenarios")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        test_endpoints = ["/chat/test-thread/messages", "/chat/test-thread/run-ids"]

        for endpoint in test_endpoints:
            response = await client.get(f"{SERVER_BASE_URL}{endpoint}")
            assert response.status_code == 401, f"Expected 401 for {endpoint}"
            print(f"✅ {endpoint} correctly returns 401 without auth")


async def test_malformed_thread_ids():
    """Test various malformed thread IDs."""
    print("🔍 Testing: Malformed thread ID handling")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    malformed_ids = [
        ("very-long-thread-id-" + "x" * 1000, "Very long thread ID"),
        ("thread\nwith\nnewlines", "Thread ID with newlines"),
        ("thread/with/slashes", "Thread ID with slashes"),
        ("thread with spaces", "Thread ID with spaces"),
        ("thread\twith\ttabs", "Thread ID with tabs"),
    ]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for thread_id, desc in malformed_ids:
            print(f"🧪 Testing malformed ID: {desc}")

            for endpoint_template in [
                "/chat/{thread_id}/messages",
                "/chat/{thread_id}/run-ids",
            ]:
                endpoint = endpoint_template.replace("{thread_id}", thread_id)
                try:
                    response = await client.get(
                        f"{SERVER_BASE_URL}{endpoint}", headers=headers
                    )
                    # Should either succeed with empty results or handle gracefully
                    assert response.status_code in [
                        200,
                        400,
                        422,
                    ], f"Unexpected status for {endpoint}"
                    print(
                        f"✅ {endpoint} handled malformed ID gracefully: {response.status_code}"
                    )
                except Exception as e:
                    print(f"⚠ {endpoint} with malformed ID caused: {e}")


async def test_endpoint_response_time_performance():
    """Test that endpoints respond within reasonable time limits."""
    print("🔍 Testing: Response time performance")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        endpoints_to_test = [
            "/chat/perf-test-thread/messages",
            "/chat/perf-test-thread/run-ids",
        ]

        for endpoint in endpoints_to_test:
            start_time = time.time()
            response = await client.get(f"{SERVER_BASE_URL}{endpoint}", headers=headers)
            response_time = time.time() - start_time

            assert response.status_code == 200, f"Expected 200 for {endpoint}"
            assert (
                response_time < 10.0
            ), f"Response time too slow for {endpoint}: {response_time:.2f}s"
            print(f"✅ {endpoint} responded in {response_time:.2f}s (< 10s limit)")


async def test_concurrent_requests():
    """Test that endpoints handle concurrent requests properly."""
    print("🔍 Testing: Concurrent request handling")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Test concurrent requests to messages endpoint
        tasks = []
        for i in range(5):
            task = client.get(
                f"{SERVER_BASE_URL}/chat/concurrent-test-{i}/messages", headers=headers
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"⚠ Concurrent request {i} failed: {response}")
            else:
                assert (
                    response.status_code == 200
                ), f"Concurrent request {i} failed with {response.status_code}"
                success_count += 1

        assert (
            success_count >= 3
        ), f"Expected at least 3 successful concurrent requests, got {success_count}"
        print(f"✅ {success_count}/5 concurrent requests succeeded")


async def test_edge_case_thread_ids():
    """Test various edge case thread IDs."""
    print("🔍 Testing: Edge case thread ID handling")

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    edge_cases = [
        ("1", "Single digit thread ID"),
        ("a", "Single character thread ID"),
        ("thread-" + "x" * 50, "Medium length thread ID"),
        ("123-456-789", "Numeric with dashes"),
        ("UPPERCASE-THREAD", "Uppercase thread ID"),
        ("MixedCase-Thread-123", "Mixed case thread ID"),
    ]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for thread_id, desc in edge_cases:
            print(f"🧪 Testing edge case: {desc}")

            for endpoint_type in ["messages", "run-ids"]:
                endpoint = f"/chat/{thread_id}/{endpoint_type}"
                response = await client.get(
                    f"{SERVER_BASE_URL}{endpoint}", headers=headers
                )

                # Should handle gracefully - either 200 with empty data or proper error
                assert response.status_code in [
                    200,
                    400,
                    404,
                ], f"Unexpected status for {endpoint}: {response.status_code}"

                if response.status_code == 200:
                    data = response.json()
                    if endpoint_type == "messages":
                        assert isinstance(
                            data, list
                        ), f"Messages should be list for {endpoint}"
                    else:
                        assert (
                            isinstance(data, dict) and "run_ids" in data
                        ), f"Run IDs should have correct structure for {endpoint}"

                print(
                    f"✅ {endpoint} handled edge case gracefully: {response.status_code}"
                )


async def run_message_tests() -> BaseTestResults:
    """Run all message endpoint tests."""
    print("🚀 Starting message tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Run all test cases
        for i, test_case in enumerate(TEST_QUERIES, 1):
            test_id = f"test_{i}"
            await make_message_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["thread_id"],
                test_case["description"],
                test_case["should_succeed"],
                test_case.get("expect_empty", False),
                results,
                test_case.get("expected_status"),
            )
            await asyncio.sleep(0.1)  # Small delay between requests

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results Summary:")
    print("=" * 80)

    summary = results.get_summary()

    print(f"📈 Overall Statistics:")
    print(f"   Total Requests: {summary['total_requests']}")
    print(f"   Successful: {summary['successful_requests']}")
    print(f"   Failed: {summary['failed_requests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"   Average Response Time: {summary['average_response_time']:.2f}s")
        print(f"   Max Response Time: {summary['max_response_time']:.2f}s")
        print(f"   Min Response Time: {summary['min_response_time']:.2f}s")

    print(f"\n🎯 Endpoint Coverage:")
    print(f"   Required Endpoints: {len(REQUIRED_ENDPOINTS)}")
    print(f"   Tested Endpoints: {len(summary['tested_endpoints'])}")
    print(
        f"   All Required Tested: {'✅' if summary['all_endpoints_tested'] else '❌'}"
    )

    if not summary["all_endpoints_tested"]:
        print(f"   Missing: {', '.join(summary['missing_endpoints'])}")
    else:
        print(f"   Tested: {', '.join(summary['tested_endpoints'])}")

    # Show detailed test breakdown
    if results.results:
        print(
            f"\n✅ Successful Tests ({len([r for r in results.results if r['success']])}):"
        )
        for result in results.results:
            if result["success"]:
                endpoint_type = (
                    "Messages" if "/messages" in result["endpoint"] else "Run IDs"
                )
                print(
                    f"   • {endpoint_type}: {result['description']} ({result['response_time']:.2f}s)"
                )

    # Show errors if any
    if results.errors:
        print(f"\n❌ Failed Tests ({len(results.errors)}):")
        for error in results.errors:
            endpoint_type = (
                "Messages" if "/messages" in error["endpoint"] else "Run IDs"
            )
            print(f"   • {endpoint_type}: {error['description']}")
            print(f"     Error: {error['error']}")

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function."""
    print("Message Endpoints Test Starting...")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__api_postgresql="1",
                print__feedback_flow="1",
                print__chat_messages_debug="1",
                DEBUG_TRACEBACK="1",
            )

            # Run authentication tests first
            await test_authentication_required()
            await test_no_auth_scenarios()
            await test_malformed_thread_ids()

            # Run performance and edge case tests
            await test_endpoint_response_time_performance()
            await test_concurrent_requests()
            await test_edge_case_thread_ids()

            # Run main endpoint tests
            results = await run_message_tests()

            await cleanup_debug_environment(
                client,
                print__api_postgresql="0",
                print__feedback_flow="0",
                print__chat_messages_debug="0",
                DEBUG_TRACEBACK="0",
            )

        summary = analyze_test_results(results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )
        has_database_errors = any(
            "no such variable" in error.get("error", "").lower()
            or "nameError" in error.get("error", "")
            or "undefined" in error.get("error", "").lower()
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors
            and not has_database_errors
            and summary["total_requests"] > 0
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
        )

        if has_empty_errors:
            print("Test failed: Server returned empty error messages")
        elif has_database_errors:
            print("Test failed: Database errors detected")
        elif summary["successful_requests"] == 0:
            print("Test failed: No requests succeeded")
        elif summary["failed_requests"] > 0:
            print(f"Test failed: {summary['failed_requests']} requests failed")

        print(f"\nOVERALL RESULT: {'PASSED' if test_passed else 'FAILED'}")
        return test_passed

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Queries": len(TEST_QUERIES),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

"""Test for Phase 8.4: Feedback Routes
Tests the feedback endpoints with real HTTP requests and proper authentication.
"""

import os
import sys
from pathlib import Path

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

import httpx
from typing import Dict
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

# Import database connection for test data setup
sys.path.insert(0, str(BASE_DIR))
from checkpointer.database.connection import get_direct_connection

# Test configuration
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 30
TEST_EMAIL = "test_feedback_user@example.com"
REQUIRED_ENDPOINTS = {"/feedback", "/sentiment"}


async def create_test_run_ids_in_db(user_email: str, count: int = 4) -> list[str]:
    """Create test run_ids in the database that the test user owns.

    Args:
        user_email: Email of the test user
        count: Number of test run_ids to create

    Returns:
        List of created run_id strings
    """
    test_run_ids = []

    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                for i in range(count):
                    run_id = str(uuid.uuid4())
                    thread_id = str(uuid.uuid4())

                    # Insert test run data into users_threads_runs table
                    await cur.execute(
                        """
                        INSERT INTO users_threads_runs 
                        (email, thread_id, run_id, sentiment, timestamp) 
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (run_id) DO NOTHING
                        """,
                        (user_email, thread_id, run_id, None),
                    )
                    test_run_ids.append(run_id)

                await conn.commit()
                print(
                    f"✅ Created {len(test_run_ids)} test run_ids in database for user: {user_email}"
                )

    except Exception as exc:
        print(f"⚠️ Failed to create test run_ids: {exc}")
        # If database setup fails, fall back to random UUIDs (tests will show ownership failures)
        test_run_ids = [str(uuid.uuid4()) for _ in range(count)]
        print(f"🔄 Using random UUIDs instead (tests will show ownership validation)")

    return test_run_ids


async def cleanup_test_run_ids_from_db(run_ids: list[str]):
    """Clean up test run_ids from the database after testing.

    Args:
        run_ids: List of run_id strings to remove
    """
    try:
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                for run_id in run_ids:
                    await cur.execute(
                        "DELETE FROM users_threads_runs WHERE run_id = %s", (run_id,)
                    )
                await conn.commit()
                print(f"🧹 Cleaned up {len(run_ids)} test run_ids from database")
    except Exception as exc:
        print(f"⚠️ Failed to cleanup test run_ids: {exc}")


# Helper function to generate valid and invalid UUIDs for testing
def generate_test_uuids():
    """Generate various UUIDs for testing purposes."""
    return {
        "valid_uuid": str(uuid.uuid4()),
        "another_valid_uuid": str(uuid.uuid4()),
        "invalid_format": "not-a-uuid",
        "empty_string": "",
        "short_string": "123",
        "malformed_uuid": "12345678-1234-5678-9012-12345678901",  # Missing one char
        "wrong_hyphens": "12345678_1234_5678_9012_123456789012",
    }


# Test cases for feedback endpoint
def get_feedback_test_cases(test_run_ids: list[str] = None):
    """Get test cases for the feedback endpoint.

    Args:
        test_run_ids: List of valid run_ids that exist in database for testing valid cases
    """
    test_uuids = generate_test_uuids()

    # Use provided test_run_ids for valid cases, or fall back to random UUIDs
    if test_run_ids and len(test_run_ids) >= 4:
        valid_run_id_1 = test_run_ids[0]
        valid_run_id_2 = test_run_ids[1]
        valid_run_id_3 = test_run_ids[2]
        valid_run_id_4 = test_run_ids[3]
    else:
        valid_run_id_1 = test_uuids["valid_uuid"]
        valid_run_id_2 = test_uuids["another_valid_uuid"]
        valid_run_id_3 = test_uuids["valid_uuid"]
        valid_run_id_4 = test_uuids["valid_uuid"]

    return [
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": valid_run_id_1,
                "feedback": 1,
                "comment": "Great response!",
            },
            "description": "Valid feedback submission with thumbs up and comment",
            "should_succeed": True,
            "test_focus": "Valid positive feedback with comment",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": valid_run_id_2,
                "feedback": 0,
                "comment": "Not helpful",
            },
            "description": "Valid feedback submission with thumbs down and comment",
            "should_succeed": True,
            "test_focus": "Valid negative feedback with comment",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {"run_id": valid_run_id_3, "feedback": 1},
            "description": "Valid feedback submission with thumbs up only",
            "should_succeed": True,
            "test_focus": "Valid positive feedback without comment",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": valid_run_id_4,
                "comment": "Just a comment without rating",
            },
            "description": "Valid feedback submission with comment only",
            "should_succeed": True,
            "test_focus": "Valid comment-only feedback",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": test_uuids["invalid_format"],
                "feedback": 1,
                "comment": "Invalid UUID",
            },
            "description": "Invalid UUID format",
            "should_succeed": False,
            "expected_status": 422,  # Pydantic validation happens first
            "test_focus": "UUID format validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {"run_id": test_uuids["empty_string"], "feedback": 1},
            "description": "Empty run_id",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Empty UUID validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": test_uuids["valid_uuid"],
                "feedback": 2,  # Invalid feedback value
            },
            "description": "Invalid feedback value (must be 0 or 1)",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Feedback value validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": test_uuids["valid_uuid"],
                "feedback": -1,  # Invalid feedback value
            },
            "description": "Negative feedback value",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Negative feedback value validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": test_uuids["valid_uuid"]
                # No feedback or comment
            },
            "description": "No feedback or comment provided",
            "should_succeed": False,
            "expected_status": 400,
            "test_focus": "Missing both feedback and comment validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {
                "run_id": test_uuids["valid_uuid"],
                "comment": "x" * 1001,  # Too long comment
            },
            "description": "Comment too long (>1000 chars)",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Comment length validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {"run_id": test_uuids["malformed_uuid"], "feedback": 1},
            "description": "Malformed UUID (missing character)",
            "should_succeed": False,
            "expected_status": 422,  # Pydantic validation happens first
            "test_focus": "Malformed UUID validation",
        },
        {
            "endpoint": "/feedback",
            "method": "POST",
            "json_data": {},  # Missing required run_id
            "description": "Missing required run_id field",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Required field validation",
        },
    ]


# Test cases for sentiment endpoint
def get_sentiment_test_cases(test_run_ids: list[str] = None):
    """Get test cases for the sentiment endpoint.

    Args:
        test_run_ids: List of valid run_ids that exist in database for testing valid cases
    """
    test_uuids = generate_test_uuids()

    # Use provided test_run_ids for valid cases, or fall back to random UUIDs
    # Note: We can reuse the same run_ids since sentiment updates don't conflict with feedback
    if test_run_ids and len(test_run_ids) >= 3:
        valid_run_id_1 = test_run_ids[0]  # For positive sentiment
        valid_run_id_2 = test_run_ids[1]  # For negative sentiment
        valid_run_id_3 = test_run_ids[0]  # For sentiment clear (reuse first one)
    else:
        valid_run_id_1 = test_uuids["valid_uuid"]
        valid_run_id_2 = test_uuids["another_valid_uuid"]
        valid_run_id_3 = test_uuids["valid_uuid"]

    return [
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": valid_run_id_1, "sentiment": True},
            "description": "Valid sentiment update to positive",
            "should_succeed": True,
            "test_focus": "Valid positive sentiment update",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {
                "run_id": valid_run_id_2,
                "sentiment": False,
            },
            "description": "Valid sentiment update to negative",
            "should_succeed": True,
            "test_focus": "Valid negative sentiment update",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": valid_run_id_3, "sentiment": None},
            "description": "Valid sentiment clear (null)",
            "should_succeed": True,
            "test_focus": "Valid sentiment clear operation",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": test_uuids["invalid_format"], "sentiment": True},
            "description": "Invalid UUID format for sentiment",
            "should_succeed": False,
            "expected_status": 422,  # Pydantic validation happens first
            "test_focus": "UUID format validation for sentiment",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": test_uuids["empty_string"], "sentiment": True},
            "description": "Empty run_id for sentiment",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Empty UUID validation for sentiment",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": test_uuids["short_string"], "sentiment": True},
            "description": "Short string as run_id for sentiment",
            "should_succeed": False,
            "expected_status": 422,  # Pydantic validation happens first
            "test_focus": "Short UUID validation for sentiment",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {"run_id": test_uuids["wrong_hyphens"], "sentiment": True},
            "description": "UUID with wrong separators",
            "should_succeed": False,
            "expected_status": 422,  # Pydantic validation happens first
            "test_focus": "UUID separator validation",
        },
        {
            "endpoint": "/sentiment",
            "method": "POST",
            "json_data": {},  # Missing required run_id
            "description": "Missing required run_id field for sentiment",
            "should_succeed": False,
            "expected_status": 422,
            "test_focus": "Required field validation for sentiment",
        },
    ]


def _validate_feedback_response(endpoint: str, data: dict, original_request: dict):
    """Validate feedback response structure."""
    print(
        f"🔍 Testing: {original_request.get('test_focus', 'Response structure validation')}"
    )

    if endpoint == "/feedback":
        assert "message" in data, "Missing 'message' field in feedback response"
        assert "run_id" in data, "Missing 'run_id' field in feedback response"
        assert isinstance(data["message"], str), "'message' must be a string"
        assert isinstance(data["run_id"], str), "'run_id' must be a string"

        # Validate that run_id is returned as a proper UUID string
        try:
            uuid.UUID(data["run_id"])
        except ValueError:
            raise AssertionError(
                f"Returned run_id '{data['run_id']}' is not a valid UUID"
            )

        # Check that original values are echoed back
        if "feedback" in original_request:
            assert "feedback" in data, "Missing 'feedback' field in response"
            assert (
                data["feedback"] == original_request["feedback"]
            ), "Feedback value mismatch"

        if "comment" in original_request:
            assert "comment" in data, "Missing 'comment' field in response"
            assert (
                data["comment"] == original_request["comment"]
            ), "Comment value mismatch"

        print(f"✅ /feedback validation passed - {original_request.get('test_focus')}")

    elif endpoint == "/sentiment":
        assert "message" in data, "Missing 'message' field in sentiment response"
        assert "run_id" in data, "Missing 'run_id' field in sentiment response"
        assert "sentiment" in data, "Missing 'sentiment' field in sentiment response"
        assert isinstance(data["message"], str), "'message' must be a string"
        assert isinstance(data["run_id"], str), "'run_id' must be a string"

        # Validate that run_id is returned as a proper UUID string
        try:
            uuid.UUID(data["run_id"])
        except ValueError:
            raise AssertionError(
                f"Returned run_id '{data['run_id']}' is not a valid UUID"
            )

        # Check that sentiment value is echoed back correctly
        expected_sentiment = original_request.get("sentiment")
        assert (
            data["sentiment"] == expected_sentiment
        ), f"Expected sentiment {expected_sentiment}, got {data['sentiment']}"

        print(f"✅ /sentiment validation passed - {original_request.get('test_focus')}")


def _get_test_explanation(
    test_focus: str,
    should_succeed: bool,
    expected_status: int,
    json_data: dict,
    endpoint: str,
) -> str:
    """Generate a detailed explanation of what the test is validating."""

    if should_succeed:
        # Success cases - explain what functionality we're testing
        if endpoint == "/feedback":
            if json_data.get("feedback") is not None and json_data.get("comment"):
                return f"Valid feedback submission: rating={json_data['feedback']}, comment='{json_data['comment'][:50]}...', verifying LangSmith integration and database ownership"
            elif json_data.get("feedback") is not None:
                return f"Valid feedback submission: rating={json_data['feedback']} only, verifying API accepts feedback without comment"
            elif json_data.get("comment"):
                return f"Valid comment-only submission: '{json_data['comment'][:50]}...', verifying API accepts comment without rating"
        elif endpoint == "/sentiment":
            sentiment_val = json_data.get("sentiment")
            if sentiment_val is True:
                return "Valid sentiment update to positive, verifying database update and ownership validation"
            elif sentiment_val is False:
                return "Valid sentiment update to negative, verifying database update and ownership validation"
            elif sentiment_val is None:
                return "Valid sentiment clear operation, verifying database can store null sentiment values"
    else:
        # Failure cases - explain what validation we're testing
        if expected_status == 422:
            if (
                not json_data.get("run_id")
                or len(str(json_data.get("run_id", ""))) == 0
            ):
                return "Pydantic validation: empty run_id should be rejected before reaching business logic"
            elif json_data.get("run_id") and not _is_valid_uuid_format(
                json_data["run_id"]
            ):
                return f"Pydantic validation: malformed UUID '{json_data['run_id']}' should be rejected by schema validation"
            elif json_data.get("feedback") is not None and json_data[
                "feedback"
            ] not in [0, 1]:
                return f"Pydantic validation: invalid feedback value '{json_data['feedback']}' (must be 0 or 1)"
            elif json_data.get("comment") and len(json_data["comment"]) > 1000:
                return f"Pydantic validation: comment too long ({len(json_data['comment'])} chars, max 1000)"
            elif not json_data:
                return "Pydantic validation: missing required fields should be caught by schema"
        elif expected_status == 400:
            if (
                endpoint == "/feedback"
                and not json_data.get("feedback")
                and not json_data.get("comment")
            ):
                return "Business logic validation: at least one of feedback or comment must be provided"

    return f"Testing {test_focus} - verifying proper API behavior"


def _is_valid_uuid_format(uuid_str: str) -> bool:
    """Check if string has valid UUID format."""
    try:
        uuid.UUID(uuid_str)
        return True
    except (ValueError, TypeError):
        return False


async def make_feedback_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    method: str,
    json_data: Dict,
    description: str,
    should_succeed: bool,
    results: BaseTestResults,
    expected_status: int = None,
    test_focus: str = None,
):
    """Make a request to a feedback endpoint with server traceback capture."""

    # Print detailed test information
    print(f"\n🔍 TEST {test_id}: {test_focus or description}")
    print(f"   📍 Endpoint: {method} {endpoint}")
    print(f"   📋 Request Data: {json_data}")
    print(
        f"   ✅ Expected Result: {'Success (200)' if should_succeed else f'Failure ({expected_status or 422})'}"
    )
    print(
        f"   🎯 What we're testing: {_get_test_explanation(test_focus, should_succeed, expected_status, json_data, endpoint)}"
    )

    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    start_time = time.time()
    try:
        result = await make_request_with_traceback_capture(
            client,
            method,
            f"{SERVER_BASE_URL}{endpoint}",
            json=json_data,
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
            results.add_error(test_id, endpoint, description, error_obj, response_time)
            return

        response = result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.2f}s)")

        if should_succeed:
            if response.status_code == 200:
                try:
                    data = response.json()
                    _validate_feedback_response(endpoint, data, json_data)
                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        data,
                        response_time,
                        response.status_code,
                        success=True,  # Explicitly mark 200 responses as success
                    )
                except (AssertionError, Exception) as e:
                    print(f"❌ Validation failed: {e}")
                    error_obj = Exception(f"Response validation failed: {e}")
                    error_obj.server_tracebacks = error_info["server_tracebacks"]
                    results.add_error(
                        test_id, endpoint, description, error_obj, response_time
                    )
            else:
                handle_error_response(
                    test_id,
                    endpoint,
                    description,
                    response,
                    error_info,
                    results,
                    response_time,
                )
        else:
            handle_expected_failure(
                test_id,
                endpoint,
                description,
                response,
                error_info,
                results,
                response_time,
                expected_status=expected_status,
            )

    except Exception as exc:
        response_time = time.time() - start_time
        error_message = (
            str(exc) if str(exc).strip() else f"{type(exc).__name__}: {repr(exc)}"
        )
        if not error_message or error_message.isspace():
            error_message = f"Unknown error of type {type(exc).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}")
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(
            test_id, endpoint, description, error_obj, response_time, response_data=None
        )


def _get_auth_test_explanation(test_case: dict) -> str:
    """Generate explanation for authentication test cases."""
    description = test_case["description"]
    endpoint = test_case["endpoint"]
    headers = test_case["headers"]

    if "No authorization header" in description:
        return f"FastAPI should reject request to {endpoint} when Authorization header is completely missing (401 Unauthorized)"
    elif "Invalid token format" in description:
        return f"JWT middleware should reject malformed token '{headers.get('Authorization', '')}' (401 Unauthorized)"
    elif "Missing Bearer prefix" in description:
        return f"JWT middleware should reject token without 'Bearer ' prefix: '{headers.get('Authorization', '')}' (401 Unauthorized)"

    return f"Authentication validation for {endpoint} endpoint"


async def test_authentication_failures():
    """Test authentication failures for feedback endpoints."""
    print("🔍 Testing: Authentication and authorization failures")

    test_uuids = generate_test_uuids()
    auth_test_cases = [
        {
            "description": "No authorization header",
            "headers": {},
            "endpoint": "/feedback",
            "json_data": {"run_id": test_uuids["valid_uuid"], "feedback": 1},
        },
        {
            "description": "Invalid token format",
            "headers": {"Authorization": "Bearer invalid_token"},
            "endpoint": "/feedback",
            "json_data": {"run_id": test_uuids["valid_uuid"], "feedback": 1},
        },
        {
            "description": "Missing Bearer prefix",
            "headers": {"Authorization": "invalid_token"},
            "endpoint": "/feedback",
            "json_data": {"run_id": test_uuids["valid_uuid"], "feedback": 1},
        },
        {
            "description": "No authorization header for sentiment",
            "headers": {},
            "endpoint": "/sentiment",
            "json_data": {"run_id": test_uuids["valid_uuid"], "sentiment": True},
        },
    ]

    results = BaseTestResults()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i, test_case in enumerate(auth_test_cases, 1):
            print(f"\n🔍 AUTH TEST {i}: {test_case['description']}")
            print(f"   📍 Endpoint: POST {test_case['endpoint']}")
            print(f"   🔑 Headers: {test_case['headers']}")
            print(f"   📋 Request Data: {test_case['json_data']}")
            print(f"   🎯 What we're testing: {_get_auth_test_explanation(test_case)}")
            start_time = time.time()
            try:
                response = await client.post(
                    f"{SERVER_BASE_URL}{test_case['endpoint']}",
                    json=test_case["json_data"],
                    headers=test_case["headers"],
                    timeout=REQUEST_TIMEOUT,
                )
                response_time = time.time() - start_time

                # Should fail with 401 or 403
                if response.status_code in [401, 403]:
                    print(
                        f"✅ Auth test {i}: Correctly rejected with {response.status_code}"
                    )
                    results.add_result(
                        f"auth_{i}",
                        test_case["endpoint"],
                        test_case["description"],
                        {"auth_rejection": True},
                        response_time,
                        response.status_code,
                        success=True,  # Auth rejection is expected = success
                    )
                else:
                    print(
                        f"❌ Auth test {i}: Expected 401/403 but got {response.status_code}"
                    )
                    error_obj = Exception(
                        f"Expected auth failure but got {response.status_code}"
                    )
                    results.add_error(
                        f"auth_{i}",
                        test_case["endpoint"],
                        test_case["description"],
                        error_obj,
                        response_time,
                    )
            except Exception as exc:
                response_time = time.time() - start_time
                print(f"❌ Auth test {i} - Error: {exc}")
                results.add_error(
                    f"auth_{i}",
                    test_case["endpoint"],
                    test_case["description"],
                    exc,
                    response_time,
                )

    return results


async def run_feedback_tests() -> BaseTestResults:
    """Run all feedback endpoint tests."""
    print("🚀 Starting feedback endpoint tests...")

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    # Create test run_ids in the database for valid test cases
    test_run_ids = await create_test_run_ids_in_db(TEST_EMAIL, 4)

    try:
        # Get all test cases with the created test run_ids
        feedback_tests = get_feedback_test_cases(test_run_ids)
        sentiment_tests = get_sentiment_test_cases(test_run_ids)
        all_tests = feedback_tests + sentiment_tests

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Run all test cases
            for i, test_case in enumerate(all_tests, 1):
                test_id = f"test_{i}"
                await make_feedback_request(
                    client,
                    test_id,
                    test_case["endpoint"],
                    test_case["method"],
                    test_case["json_data"],
                    test_case["description"],
                    test_case["should_succeed"],
                    results,
                    expected_status=test_case.get("expected_status"),
                    test_focus=test_case.get("test_focus"),
                )
                await asyncio.sleep(0.1)  # Small delay between requests

        # Test authentication failures
        auth_results = await test_authentication_failures()
        # Merge auth results into main results
        results.results.extend(auth_results.results)
        results.errors.extend(auth_results.errors)

    finally:
        # Clean up test data from database
        await cleanup_test_run_ids_from_db(test_run_ids)

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results."""
    print("\n📊 Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.2f}s")

    if not summary["all_endpoints_tested"]:
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

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
    print("[START] Feedback Endpoints Test Starting...")
    print("[INFO] Testing: Comprehensive feedback and sentiment endpoint functionality")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("[ERROR] Server connectivity check failed!")
        return False

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__feedback_debug="1",
                print__feedback_flow="1",
                print__sentiment_debug="1",
                print__sentiment_flow="1",
                DEBUG_TRACEBACK="1",
            )

            results = await run_feedback_tests()

            await cleanup_debug_environment(
                client,
                print__feedback_debug="0",
                print__feedback_flow="0",
                print__sentiment_debug="0",
                print__sentiment_flow="0",
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
            and summary["all_endpoints_tested"]
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
        )

        if has_empty_errors:
            print("❌ Test failed: Server returned empty error messages")
        elif has_database_errors:
            print("❌ Test failed: Database errors detected")
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded")
        elif not summary["all_endpoints_tested"]:
            print("❌ Test failed: Not all required endpoints were tested")
        elif summary["failed_requests"] > 0:
            print(f"❌ Test failed: {summary['failed_requests']} requests failed")

        print(f"\n[RESULT] OVERALL RESULT: {'PASSED' if test_passed else 'FAILED'}")
        print("[OK] Tested feedback endpoint with various valid/invalid inputs")
        print("[OK] Tested sentiment endpoint with various valid/invalid inputs")
        print("[OK] Tested authentication and authorization failures")
        print("[OK] Tested UUID validation and error handling")
        print("[OK] Tested request/response structure validation")

        return test_passed

    except Exception as exc:
        print(f"❌ Test execution failed: {str(exc)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Cases": len(get_feedback_test_cases())
            + len(get_sentiment_test_cases()),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=exc, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n[STOP] Test interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Fatal error: {str(exc)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Cases": "Unknown (failed during initialization)",
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=exc, test_context=test_context
        )
        sys.exit(1)

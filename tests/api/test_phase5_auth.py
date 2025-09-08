"""
Test for Phase 5: Authentication Functions (JWT and Dependencies)
Tests authentication module functions following the established project patterns.
"""

import os
import sys
import time
import traceback
from typing import Dict, Any
from datetime import datetime

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(BASE_DIR))
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    sys.path.insert(0, str(BASE_DIR))

# Import test helpers following Phase 8 patterns
from tests.helpers import (
    BaseTestResults,
    save_traceback_report,
)

# Test imports from extracted modules
try:
    from api.auth.jwt_auth import verify_google_jwt
    from api.dependencies.auth import get_current_user
    import jwt
    from fastapi import HTTPException

    print("✅ Successfully imported authentication functions")
except Exception as e:
    print(f"❌ Failed to import authentication functions: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)

# Test configuration following Phase 8 patterns
REQUIRED_FUNCTIONS = {"verify_google_jwt", "get_current_user"}

AUTH_VALIDATION_TESTS = [
    {
        "test_id": "AUTH_001",
        "function": "verify_google_jwt",
        "description": "Invalid token format - no dots",
        "token": "invalid_token",
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_002",
        "function": "verify_google_jwt",
        "description": "Invalid token format - two dots only",
        "token": "part1.part2",
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_003",
        "function": "verify_google_jwt",
        "description": "Invalid token format - empty parts",
        "token": "...",
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_004",
        "function": "verify_google_jwt",
        "description": "Valid test token (test mode only)",
        "token": "TEST_TOKEN",  # Will be replaced with actual test token
        "should_succeed": True,
        "test_mode_only": True,
    },
    {
        "test_id": "AUTH_005",
        "function": "verify_google_jwt",
        "description": "Expired test token",
        "token": "EXPIRED_TOKEN",  # Will be replaced with expired token
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_006",
        "function": "get_current_user",
        "description": "Missing authorization header",
        "auth_header": None,
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_007",
        "function": "get_current_user",
        "description": "Invalid authorization header format",
        "auth_header": "InvalidFormat",
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_008",
        "function": "get_current_user",
        "description": "Valid bearer token (test mode only)",
        "auth_header": "Bearer TEST_TOKEN",  # Will be replaced
        "should_succeed": True,
        "test_mode_only": True,
    },
    {
        "test_id": "AUTH_009",
        "function": "verify_google_jwt",
        "description": "Wrong audience in test token",
        "token": "WRONG_AUD_TOKEN",  # Will be replaced
        "should_succeed": False,
        "expected_status": 401,
    },
    {
        "test_id": "AUTH_010",
        "function": "get_current_user",
        "description": "Bearer token with expired test token",
        "auth_header": "Bearer EXPIRED_TOKEN",  # Will be replaced
        "should_succeed": False,
        "expected_status": 401,
    },
]


def create_test_token(
    audience: str = None,
    issuer: str = "test_issuer",
    email: str = "test@example.com",
    exp_minutes: int = 60,
) -> str:
    """Create a test JWT token for testing authentication."""
    if not audience:
        audience = os.getenv("GOOGLE_CLIENT_ID", "test_client_id")

    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": "test_user_123",
        "email": email,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "picture": "https://example.com/avatar.jpg",
        "iat": int(time.time()),
        "exp": int(time.time()) + (exp_minutes * 60),
    }

    # Use a simple secret for test tokens (not secure, but fine for testing)
    return jwt.encode(payload, "test_secret", algorithm="HS256")


def validate_auth_function(
    test_id: str,
    function_name: str,
    description: str,
    test_case: Dict[str, Any],
    should_succeed: bool,
    results: BaseTestResults,
):
    """Validate an authentication function with error tracking."""
    print(f"\n🔍 Test {test_id}: {description}")
    print(f"   📋 Testing: {function_name}()")
    print(f"   🎯 Expected: {'SUCCESS' if should_succeed else 'FAILURE'}")

    start_time = time.time()

    try:
        # Handle test mode requirements
        test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"
        if test_case.get("test_mode_only", False) and not test_mode:
            print(f"   ℹ️ SKIPPED: Test mode disabled")
            return

        # Prepare test data based on function
        if function_name == "verify_google_jwt":
            token = test_case.get("token", "")

            # Show what we're testing with
            if token == "TEST_TOKEN":
                token = create_test_token()
                print(f"   📥 Input: Valid test JWT token (created dynamically)")
            elif token == "EXPIRED_TOKEN":
                token = create_test_token(exp_minutes=-1)
                print(f"   📥 Input: Expired JWT token (exp: -1 min)")
            elif token == "WRONG_AUD_TOKEN":
                token = create_test_token(audience="wrong_audience")
                print(f"   📥 Input: JWT token with wrong audience")
            else:
                print(f"   📥 Input: '{token}' (invalid format)")

            # Call the function
            print(f"   🔧 Calling: verify_google_jwt(token)")
            result = verify_google_jwt(token)

        elif function_name == "get_current_user":
            auth_header = test_case.get("auth_header")

            # Show what we're testing with
            if auth_header is None:
                print(f"   📥 Input: None (missing authorization header)")
            elif auth_header == "Bearer TEST_TOKEN":
                test_token = create_test_token()
                auth_header = f"Bearer {test_token}"
                print(f"   📥 Input: 'Bearer <valid_test_token>' (created dynamically)")
            elif auth_header == "Bearer EXPIRED_TOKEN":
                expired_token = create_test_token(exp_minutes=-1)
                auth_header = f"Bearer {expired_token}"
                print(f"   📥 Input: 'Bearer <expired_token>' (exp: -1 min)")
            else:
                print(f"   📥 Input: '{auth_header}' (invalid format)")

            # Call the function
            print(f"   🔧 Calling: get_current_user(authorization='{auth_header}')")
            result = get_current_user(auth_header)

        else:
            raise ValueError(f"Unknown function: {function_name}")

        response_time = time.time() - start_time

        if should_succeed:
            # Expected success case
            print(f"   ✅ RESULT: SUCCESS ({response_time:.3f}s)")
            if function_name == "verify_google_jwt":
                assert isinstance(
                    result, dict
                ), "verify_google_jwt should return a dict"
                assert "email" in result, "Result should contain email"
                print(
                    f"   📤 Output: User info dict with email='{result.get('email', 'N/A')}'"
                )
            elif function_name == "get_current_user":
                assert isinstance(result, dict), "get_current_user should return a dict"
                assert "email" in result, "Result should contain email"
                print(
                    f"   📤 Output: User info dict with email='{result.get('email', 'N/A')}'"
                )

            results.add_result(
                test_id,
                function_name,
                description,
                {
                    "result": (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )
                },
                response_time,
                200,
            )
        else:
            # Should have failed but didn't
            print(f"   ❌ UNEXPECTED: Expected failure but got success")
            print(f"   📤 Output: {str(result)[:100]}...")
            results.add_error(
                test_id,
                function_name,
                description,
                Exception(
                    f"Expected failure but function succeeded with result: {result}"
                ),
                response_time,
            )

    except HTTPException as e:
        response_time = time.time() - start_time
        expected_status = test_case.get("expected_status", 401)

        if should_succeed:
            # Expected success but got HTTP error
            print(f"   ❌ UNEXPECTED: Expected success but got HTTP {e.status_code}")
            print(f"   📤 Output: HTTPException - {e.detail}")
            results.add_error(
                test_id,
                function_name,
                description,
                e,
                response_time,
                response_data={"status_code": e.status_code, "detail": e.detail},
            )
        else:
            # Expected failure - check if status code matches
            if e.status_code == expected_status:
                print(
                    f"   ✅ RESULT: Expected failure with HTTP {e.status_code} ({response_time:.3f}s)"
                )
                print(f"   📤 Output: HTTPException({e.status_code}) - {e.detail}")
                # Mark as successful since this is the expected outcome
                results.add_result(
                    test_id,
                    function_name,
                    description,
                    {"status_code": e.status_code, "detail": e.detail},
                    response_time,
                    200,  # Mark as success for expected failure
                )
            else:
                print(
                    f"   ❌ WRONG STATUS: Expected HTTP {expected_status} but got {e.status_code}"
                )
                print(f"   📤 Output: HTTPException({e.status_code}) - {e.detail}")
                results.add_error(
                    test_id,
                    function_name,
                    description,
                    Exception(
                        f"Expected HTTP {expected_status} but got {e.status_code}: {e.detail}"
                    ),
                    response_time,
                    response_data={"status_code": e.status_code, "detail": e.detail},
                )

    except Exception as e:
        response_time = time.time() - start_time
        if should_succeed:
            print(f"   ❌ UNEXPECTED: Unexpected error: {str(e)}")
            print(f"   📤 Output: {type(e).__name__} - {str(e)}")
        else:
            print(f"   ✅ RESULT: Expected failure with error ({response_time:.3f}s)")
            print(f"   📤 Output: {type(e).__name__} - {str(e)}")

        # For expected failures, this is success; for expected success, this is failure
        if should_succeed:
            results.add_error(test_id, function_name, description, e, response_time)
        else:
            # Mark as successful since this is the expected outcome
            results.add_result(
                test_id,
                function_name,
                description,
                {"error_type": type(e).__name__, "error_message": str(e)},
                response_time,
                200,  # Mark as success for expected failure
            )


def run_auth_validation_tests() -> BaseTestResults:
    """Run all authentication validation tests following Phase 8 patterns."""
    print("🚀 Starting authentication function validation tests...")

    # Show what we're going to test
    print("\n📋 Test Plan Overview:")
    print("=" * 60)

    verify_jwt_tests = [
        t for t in AUTH_VALIDATION_TESTS if t["function"] == "verify_google_jwt"
    ]
    get_user_tests = [
        t for t in AUTH_VALIDATION_TESTS if t["function"] == "get_current_user"
    ]

    print(f"🔐 verify_google_jwt() - {len(verify_jwt_tests)} tests:")
    for test in verify_jwt_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        mode = " (test mode only)" if test.get("test_mode_only") else ""
        print(f"   • {test['test_id']}: {test['description']}{mode} → {status}")

    print(f"\n👤 get_current_user() - {len(get_user_tests)} tests:")
    for test in get_user_tests:
        status = "✅ SUCCESS" if test["should_succeed"] else "❌ FAILURE"
        mode = " (test mode only)" if test.get("test_mode_only") else ""
        print(f"   • {test['test_id']}: {test['description']}{mode} → {status}")

    print("=" * 60)

    results = BaseTestResults(required_endpoints=REQUIRED_FUNCTIONS)
    results.start_time = datetime.now()

    # Run all test cases
    for test_case in AUTH_VALIDATION_TESTS:
        validate_auth_function(
            test_case["test_id"],
            test_case["function"],
            test_case["description"],
            test_case,
            test_case["should_succeed"],
            results,
        )

    results.end_time = datetime.now()
    return results


def analyze_auth_test_results(results: BaseTestResults):
    """Analyze and print authentication test results following Phase 8 patterns."""
    print("\n📊 Authentication Function Test Results:")

    summary = results.get_summary()

    print(
        f"Total: {summary['total_requests']}, Success: {summary['successful_requests']}, Failed: {summary['failed_requests']}"
    )
    print(f"Success Rate: {summary['success_rate']:.1f}%")

    if summary["successful_requests"] > 0:
        print(f"Avg Response Time: {summary['average_response_time']:.3f}s")

    # Check function coverage
    tested_functions = set()
    for test_case in AUTH_VALIDATION_TESTS:
        tested_functions.add(test_case["function"])

    missing_functions = REQUIRED_FUNCTIONS - tested_functions
    if missing_functions:
        print(f"❌ Missing function tests: {', '.join(missing_functions)}")
    else:
        print(f"✅ All required functions tested: {', '.join(REQUIRED_FUNCTIONS)}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors[:5]:  # Show first 5 errors
            print(
                f"  - Test {error.get('test_id', 'Unknown')}: {error.get('error', 'Unknown error')}"
            )
        if len(results.errors) > 5:
            print(f"  ... and {len(results.errors) - 5} more errors")

    # Environment info
    test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"
    print(f"\n🧪 Test Mode: {'ENABLED' if test_mode else 'DISABLED'}")
    if not test_mode:
        print("ℹ️ Some tests were skipped (enable with USE_TEST_TOKENS=1)")

    # Save traceback information (always save - empty file if no errors)
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


def main():
    """Main test execution function following Phase 8 patterns."""
    print("🚀 Phase 5 Authentication Function Tests Starting...")
    print(f"📂 BASE_DIR: {BASE_DIR}")
    print(f"🧪 USE_TEST_TOKENS: {os.getenv('USE_TEST_TOKENS', 'NOT_SET')}")
    print(f"🔑 GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID', 'NOT_SET')[:20]}...")
    print("=" * 80)

    try:
        # Run authentication validation tests
        results = run_auth_validation_tests()

        # Analyze results
        summary = analyze_auth_test_results(results)

        # Determine overall test success
        test_passed = (
            summary["total_requests"] > 0
            and summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
            and len(
                REQUIRED_FUNCTIONS
                - {test["function"] for test in AUTH_VALIDATION_TESTS}
            )
            == 0
        )

        # Handle partial test mode scenarios
        test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"
        if not test_mode:
            # In non-test mode, we expect some tests to be skipped
            # Success means no unexpected failures occurred
            actual_failures = [
                error
                for error in results.errors
                if not ("test mode" in error.get("error", "").lower())
            ]
            test_passed = (
                len(actual_failures) == 0 and summary["successful_requests"] > 0
            )

        if summary["total_requests"] == 0:
            print("❌ No tests were executed")
            test_passed = False
        elif summary["successful_requests"] == 0:
            print("❌ All tests failed")
            test_passed = False

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "BASE_DIR": str(BASE_DIR),
            "Test Mode": os.getenv("USE_TEST_TOKENS", "NOT_SET"),
            "Total Test Cases": len(AUTH_VALIDATION_TESTS),
            "Error Location": "main() function",
            "Error During": "Authentication function testing",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        test_result = main()
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        save_traceback_report(report_type="exception", exception=e)
        sys.exit(1)

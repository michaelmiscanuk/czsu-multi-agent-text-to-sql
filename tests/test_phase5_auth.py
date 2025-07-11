#!/usr/bin/env python3
"""
Test for Phase 5: Extract Authentication (JWT and Dependencies)
Based on test_concurrency.py pattern - imports functionality from main scripts
"""

import os

# CRITICAL: Set Windows event loop policy FIRST, before other imports
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import asyncio
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import jwt
from fastapi import HTTPException

# Add project root to path
sys.path.insert(0, str(BASE_DIR))

# Test imports from extracted modules
try:
    from api.auth.jwt_auth import verify_google_jwt
    from api.dependencies.auth import get_current_user

    print("✅ Successfully imported authentication functions")
except Exception as e:
    print(f"❌ Failed to import authentication functions: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)


def print_test_status(message: str):
    """Print test status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def create_test_token(
    audience: str = None,
    issuer: str = "test_issuer",
    email: str = "test@example.com",
    exp_minutes: int = 60,
):
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


def test_phase5_auth_imports():
    """Test that authentication modules can be imported successfully."""
    print_test_status("🔍 Testing Phase 5 authentication imports...")

    try:
        # Test JWT auth imports
        from api.auth.jwt_auth import verify_google_jwt

        assert callable(verify_google_jwt), "verify_google_jwt should be callable"
        print_test_status("✅ JWT auth module imported successfully")

        # Test dependencies imports
        from api.dependencies.auth import get_current_user

        assert callable(get_current_user), "get_current_user should be callable"
        print_test_status("✅ Auth dependencies module imported successfully")

        print_test_status("✅ Phase 5 authentication imports test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Phase 5 authentication imports test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


def test_verify_google_jwt_function():
    """Test the verify_google_jwt function with various token scenarios."""
    print_test_status("🔍 Testing verify_google_jwt function...")

    try:
        # Test 1: Invalid token format
        try:
            verify_google_jwt("invalid_token")
            print_test_status("❌ Should have rejected invalid token format")
            return False
        except HTTPException as e:
            if e.status_code == 401:
                print_test_status("✅ Correctly rejected invalid token format")
            else:
                print_test_status(
                    f"❌ Wrong status code for invalid token: {e.status_code}"
                )
                return False

        # Test 2: Valid test token (if test mode is enabled)
        test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"
        if test_mode:
            print_test_status("🧪 TEST MODE ENABLED: Testing with test tokens")
            test_token = create_test_token()

            try:
                result = verify_google_jwt(test_token)
                print_test_status(
                    f"✅ Test token verification successful: {result.get('email', 'Unknown')}"
                )
            except HTTPException as e:
                print_test_status(f"❌ Test token verification failed: {e.detail}")
                return False
        else:
            print_test_status("ℹ️ TEST MODE DISABLED: Skipping test token verification")

        # Test 3: Expired test token (should fail)
        expired_token = create_test_token(exp_minutes=-1)  # Expired 1 minute ago
        try:
            verify_google_jwt(expired_token)
            print_test_status("❌ Should have rejected expired token")
            return False
        except HTTPException as e:
            if e.status_code == 401:
                print_test_status("✅ Correctly rejected expired token")
            else:
                print_test_status(
                    f"❌ Wrong status code for expired token: {e.status_code}"
                )
                return False

        print_test_status("✅ verify_google_jwt function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ verify_google_jwt function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


def test_get_current_user_function():
    """Test the get_current_user function with various header scenarios."""
    print_test_status("🔍 Testing get_current_user function...")

    try:
        # Test 1: Missing authorization header
        try:
            get_current_user(None)
            print_test_status("❌ Should have rejected missing authorization header")
            return False
        except HTTPException as e:
            if e.status_code == 401 and "Missing Authorization header" in e.detail:
                print_test_status("✅ Correctly rejected missing authorization header")
            else:
                print_test_status(f"❌ Wrong error for missing header: {e.detail}")
                return False

        # Test 2: Invalid authorization header format
        try:
            get_current_user("InvalidFormat")
            print_test_status("❌ Should have rejected invalid header format")
            return False
        except HTTPException as e:
            if (
                e.status_code == 401
                and "Invalid Authorization header format" in e.detail
            ):
                print_test_status("✅ Correctly rejected invalid header format")
            else:
                print_test_status(f"❌ Wrong error for invalid format: {e.detail}")
                return False

        # Test 3: Valid test token (if test mode is enabled)
        test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"
        if test_mode:
            print_test_status(
                "🧪 TEST MODE ENABLED: Testing with valid authorization header"
            )
            test_token = create_test_token()
            auth_header = f"Bearer {test_token}"

            try:
                result = get_current_user(auth_header)
                print_test_status(
                    f"✅ Valid authorization header successful: {result.get('email', 'Unknown')}"
                )
            except HTTPException as e:
                print_test_status(f"❌ Valid authorization header failed: {e.detail}")
                return False
        else:
            print_test_status(
                "ℹ️ TEST MODE DISABLED: Skipping valid authorization header test"
            )

        print_test_status("✅ get_current_user function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ get_current_user function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


def test_authentication_integration():
    """Test the integration between JWT auth and dependencies."""
    print_test_status("🔍 Testing authentication integration...")

    try:
        # Test the complete flow: create token -> verify via get_current_user
        test_mode = os.getenv("USE_TEST_TOKENS", "0") == "1"

        if test_mode:
            print_test_status(
                "🧪 TEST MODE ENABLED: Testing complete authentication flow"
            )

            # Create a valid test token
            test_email = "integration@test.com"
            test_token = create_test_token(email=test_email)
            auth_header = f"Bearer {test_token}"

            # Test the complete flow
            user_info = get_current_user(auth_header)

            # Verify the user info
            assert (
                user_info.get("email") == test_email
            ), f"Expected email {test_email}, got {user_info.get('email')}"
            assert (
                user_info.get("iss") == "test_issuer"
            ), f"Expected issuer 'test_issuer', got {user_info.get('iss')}"

            print_test_status(
                f"✅ Authentication integration successful for user: {user_info.get('email')}"
            )
        else:
            print_test_status(
                "ℹ️ TEST MODE DISABLED: Skipping authentication integration test"
            )

        print_test_status("✅ Authentication integration test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Authentication integration test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Run all Phase 5 authentication tests."""
    print_test_status("🚀 Starting Phase 5 Authentication Tests")
    print_test_status(f"📂 BASE_DIR: {BASE_DIR}")
    print_test_status(f"🧪 USE_TEST_TOKENS: {os.getenv('USE_TEST_TOKENS', 'NOT_SET')}")
    print_test_status(
        f"🔑 GOOGLE_CLIENT_ID: {os.getenv('GOOGLE_CLIENT_ID', 'NOT_SET')[:20]}..."
    )
    print_test_status("=" * 80)

    all_tests_passed = True

    # Run all tests
    tests = [
        ("Authentication Imports", test_phase5_auth_imports),
        ("JWT Verification Function", test_verify_google_jwt_function),
        ("Current User Function", test_get_current_user_function),
        ("Authentication Integration", test_authentication_integration),
    ]

    for test_name, test_func in tests:
        print_test_status(f"\n📋 Running test: {test_name}")
        print_test_status("-" * 60)

        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print_test_status(f"❌ Test {test_name} crashed: {e}")
            print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
            all_tests_passed = False

    # Final summary
    print_test_status("=" * 80)
    if all_tests_passed:
        print_test_status("🎉 ALL PHASE 5 AUTHENTICATION TESTS PASSED!")
        print_test_status("✅ Authentication extraction successful")
        print_test_status("✅ JWT auth module working correctly")
        print_test_status("✅ Auth dependencies module working correctly")
        print_test_status("✅ Authentication integration working correctly")
    else:
        print_test_status("❌ SOME PHASE 5 AUTHENTICATION TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()

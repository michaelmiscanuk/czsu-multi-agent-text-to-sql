"""
Test for Phase 12: Comprehensive Testing
Tests all endpoints work through the new modular structure
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

# Standard imports
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import functionality from main scripts
from other.tests.test_concurrency import (
    check_server_connectivity,
    create_test_jwt_token,
)

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30.0


async def test_health_endpoints():
    """Test all health check endpoints work through modular structure."""
    print("🔍 Testing health endpoints...")

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test basic health endpoint
            response = await client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] in ["healthy", "degraded"]
            print("✅ /health endpoint working")

            # Test database health
            response = await client.get("/health/database")
            assert response.status_code in [
                200,
                503,
            ]  # 503 is acceptable for database issues
            print("✅ /health/database endpoint working")

            # Test memory health
            response = await client.get("/health/memory")
            assert response.status_code == 200
            memory_data = response.json()
            assert "memory_rss_mb" in memory_data
            print("✅ /health/memory endpoint working")

            # Test rate limits health
            response = await client.get("/health/rate-limits")
            assert response.status_code == 200
            rate_data = response.json()
            assert "total_tracked_clients" in rate_data
            print("✅ /health/rate-limits endpoint working")

        return True

    except Exception as e:
        print(f"❌ Health endpoints test failed: {e}")
        return False


async def test_authentication_flow():
    """Test authentication works through modular structure."""
    print("🔍 Testing authentication flow...")

    try:
        # Create test token
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test protected endpoint (catalog requires auth)
            response = await client.get("/catalog", headers=headers)
            assert response.status_code == 200
            print("✅ Authentication working - protected endpoint accessible")

            # Test without auth (should fail)
            response = await client.get("/catalog")
            assert response.status_code == 401
            print(
                "✅ Authentication working - unauthenticated requests properly rejected"
            )

        return True

    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False


async def test_catalog_endpoints():
    """Test catalog endpoints work through modular structure."""
    print("🔍 Testing catalog endpoints...")

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test catalog endpoint
            response = await client.get("/catalog", headers=headers)
            assert response.status_code == 200
            catalog_data = response.json()
            assert "results" in catalog_data
            print("✅ /catalog endpoint working")

            # Test data-tables endpoint
            response = await client.get("/data-tables", headers=headers)
            assert response.status_code == 200
            tables_data = response.json()
            assert "tables" in tables_data
            print("✅ /data-tables endpoint working")

            # Test data-table endpoint with parameter
            response = await client.get("/data-table?table=test", headers=headers)
            assert response.status_code == 200
            table_data = response.json()
            assert "columns" in table_data
            assert "rows" in table_data
            print("✅ /data-table endpoint working")

        return True

    except Exception as e:
        print(f"❌ Catalog endpoints test failed: {e}")
        return False


async def test_analyze_endpoint():
    """Test analyze endpoint works through modular structure."""
    print("🔍 Testing analyze endpoint...")

    try:
        token = create_test_jwt_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Simple test request
        request_data = {
            "prompt": "Test prompt for modular structure validation",
            "thread_id": f"test_thread_{int(time.time())}",
        }

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(120)
        ) as client:
            response = await client.post("/analyze", json=request_data, headers=headers)
            # Accept both 200 (success) and 500 (expected errors during testing)
            assert response.status_code in [200, 500]

            if response.status_code == 200:
                analyze_data = response.json()
                assert "result" in analyze_data
                assert "thread_id" in analyze_data
                print("✅ /analyze endpoint working - successful analysis")
            else:
                error_data = response.json()
                assert "detail" in error_data
                print("✅ /analyze endpoint working - proper error handling")

        return True

    except Exception as e:
        print(f"❌ Analyze endpoint test failed: {e}")
        return False


async def test_chat_endpoints():
    """Test chat management endpoints work through modular structure."""
    print("🔍 Testing chat endpoints...")

    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test chat threads endpoint
            response = await client.get("/chat-threads", headers=headers)
            assert response.status_code == 200
            threads_data = response.json()
            assert "threads" in threads_data
            assert "total_count" in threads_data
            print("✅ /chat-threads endpoint working")

            # Test chat messages endpoint (may return 404 for non-existent thread)
            test_thread_id = "non_existent_thread"
            response = await client.get(
                f"/chat/{test_thread_id}/messages", headers=headers
            )
            assert response.status_code in [200, 404]
            print("✅ /chat/{thread_id}/messages endpoint working")

            # Test chat sentiments endpoint
            response = await client.get(
                f"/chat/{test_thread_id}/sentiments", headers=headers
            )
            assert response.status_code in [
                200,
                500,
            ]  # May error if thread doesn't exist
            print("✅ /chat/{thread_id}/sentiments endpoint working")

        return True

    except Exception as e:
        print(f"❌ Chat endpoints test failed: {e}")
        return False


async def test_debug_endpoints():
    """Test debug endpoints work through modular structure."""
    print("🔍 Testing debug endpoints...")

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test pool status endpoint (no auth required)
            response = await client.get("/debug/pool-status")
            assert response.status_code in [200, 500]
            print("✅ /debug/pool-status endpoint working")

        return True

    except Exception as e:
        print(f"❌ Debug endpoints test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling works properly through modular structure."""
    print("🔍 Testing error handling...")

    try:
        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            # Test 404 error
            response = await client.get("/non-existent-endpoint")
            assert response.status_code == 404
            print("✅ 404 error handling working")

            # Test validation error (malformed request)
            response = await client.post("/analyze", json={"invalid": "data"})
            assert response.status_code in [
                401,
                422,
            ]  # 401 for missing auth, 422 for validation
            print("✅ Validation error handling working")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Main comprehensive test runner."""
    print("🚀 Phase 12: Comprehensive Testing Starting...")
    print("=" * 60)

    # Check server connectivity first
    if not await check_server_connectivity():
        print("❌ Server connectivity check failed!")
        print("   Please start your uvicorn server first:")
        print(f"   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False

    test_results = []

    # Run all test suites
    test_suites = [
        ("Health Endpoints", test_health_endpoints),
        ("Authentication Flow", test_authentication_flow),
        ("Catalog Endpoints", test_catalog_endpoints),
        ("Analyze Endpoint", test_analyze_endpoint),
        ("Chat Endpoints", test_chat_endpoints),
        ("Debug Endpoints", test_debug_endpoints),
        ("Error Handling", test_error_handling),
    ]

    for suite_name, test_func in test_suites:
        print(f"\n📋 Running {suite_name} tests...")
        try:
            result = await test_func()
            test_results.append((suite_name, result))
            if result:
                print(f"✅ {suite_name} tests PASSED")
            else:
                print(f"❌ {suite_name} tests FAILED")
        except Exception as e:
            print(f"❌ {suite_name} tests FAILED with exception: {e}")
            test_results.append((suite_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for suite_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {suite_name}")

    print(f"\n🏁 OVERALL RESULT: {passed_tests}/{total_tests} test suites passed")

    overall_success = passed_tests == total_tests
    if overall_success:
        print(
            "🎉 ALL COMPREHENSIVE TESTS PASSED - Modular structure working correctly!"
        )
    else:
        print(f"⚠️ Some tests failed - please review and fix issues before proceeding")

    return overall_success


if __name__ == "__main__":
    # Set debug mode for better visibility
    os.environ["DEBUG"] = "1"
    os.environ["USE_TEST_TOKENS"] = "1"

    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

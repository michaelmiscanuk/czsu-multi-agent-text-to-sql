"""
Test for Phase 8.1: Extract Health Routes
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
from datetime import datetime
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(BASE_DIR))

# Test imports from extracted modules
try:
    from api.routes.health import (
        database_health_check,
        health_check,
        memory_health_check,
        prepared_statements_health_check,
        rate_limit_health_check,
    )
    from api.routes.health import router as health_router

    print("✅ Successfully imported health route functions")
except Exception as e:
    print(f"❌ Failed to import health route functions: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)


def print_test_status(message: str):
    """Print test status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def test_phase8_health_imports():
    """Test that health routes can be imported successfully."""
    print_test_status("🔍 Testing Phase 8.1 health routes imports...")

    try:
        # Test router import
        from api.routes.health import router

        assert router is not None, "Health router should not be None"
        print_test_status("✅ Health router imported successfully")

        # Test individual function imports
        from api.routes.health import health_check

        assert callable(health_check), "health_check should be callable"
        print_test_status("✅ health_check function imported successfully")

        from api.routes.health import database_health_check

        assert callable(
            database_health_check
        ), "database_health_check should be callable"
        print_test_status("✅ database_health_check function imported successfully")

        from api.routes.health import memory_health_check

        assert callable(memory_health_check), "memory_health_check should be callable"
        print_test_status("✅ memory_health_check function imported successfully")

        from api.routes.health import rate_limit_health_check

        assert callable(
            rate_limit_health_check
        ), "rate_limit_health_check should be callable"
        print_test_status("✅ rate_limit_health_check function imported successfully")

        from api.routes.health import prepared_statements_health_check

        assert callable(
            prepared_statements_health_check
        ), "prepared_statements_health_check should be callable"
        print_test_status(
            "✅ prepared_statements_health_check function imported successfully"
        )

        print_test_status("✅ Phase 8.1 health routes imports test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Phase 8.1 health routes imports test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_health_check_function():
    """Test the main health check function."""
    print_test_status("🔍 Testing health_check function...")

    try:
        # Call the health check function directly
        result = await health_check()

        # Verify the result has expected structure
        if hasattr(result, "body"):
            # JSONResponse case
            import json

            result_data = json.loads(result.body.decode())
        else:
            # Direct dict case
            result_data = result

        assert "status" in result_data, "Health check result should have 'status' field"
        assert (
            "timestamp" in result_data
        ), "Health check result should have 'timestamp' field"
        assert "memory" in result_data, "Health check result should have 'memory' field"
        assert (
            "database" in result_data
        ), "Health check result should have 'database' field"

        print_test_status(f"✅ Health check status: {result_data.get('status')}")
        print_test_status(
            f"✅ Memory usage: {result_data.get('memory', {}).get('rss_mb', 'Unknown')}MB"
        )
        print_test_status(
            f"✅ Database status: {result_data.get('database', {}).get('healthy', 'Unknown')}"
        )

        print_test_status("✅ health_check function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ health_check function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_database_health_check_function():
    """Test the database health check function."""
    print_test_status("🔍 Testing database_health_check function...")

    try:
        # Call the database health check function directly
        result = await database_health_check()

        # Verify the result has expected structure
        if hasattr(result, "body"):
            # JSONResponse case
            import json

            result_data = json.loads(result.body.decode())
        else:
            # Direct dict case
            result_data = result

        assert (
            "timestamp" in result_data
        ), "Database health check result should have 'timestamp' field"
        assert (
            "checkpointer_available" in result_data
        ), "Database health check result should have 'checkpointer_available' field"

        print_test_status(
            f"✅ Checkpointer available: {result_data.get('checkpointer_available')}"
        )
        print_test_status(
            f"✅ Checkpointer type: {result_data.get('checkpointer_type', 'Unknown')}"
        )

        print_test_status("✅ database_health_check function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ database_health_check function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_memory_health_check_function():
    """Test the memory health check function."""
    print_test_status("🔍 Testing memory_health_check function...")

    try:
        # Call the memory health check function directly
        result = await memory_health_check()

        # Verify the result has expected structure
        assert (
            "status" in result
        ), "Memory health check result should have 'status' field"
        assert (
            "memory_rss_mb" in result
        ), "Memory health check result should have 'memory_rss_mb' field"
        assert (
            "timestamp" in result
        ), "Memory health check result should have 'timestamp' field"

        print_test_status(f"✅ Memory status: {result.get('status')}")
        print_test_status(f"✅ Memory RSS: {result.get('memory_rss_mb')}MB")
        print_test_status(f"✅ Memory threshold: {result.get('memory_threshold_mb')}MB")

        print_test_status("✅ memory_health_check function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ memory_health_check function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_rate_limit_health_check_function():
    """Test the rate limit health check function."""
    print_test_status("🔍 Testing rate_limit_health_check function...")

    try:
        # Call the rate limit health check function directly
        result = await rate_limit_health_check()

        # Verify the result has expected structure
        assert (
            "status" in result
        ), "Rate limit health check result should have 'status' field"
        assert (
            "total_tracked_clients" in result
        ), "Rate limit health check result should have 'total_tracked_clients' field"
        assert (
            "timestamp" in result
        ), "Rate limit health check result should have 'timestamp' field"

        print_test_status(f"✅ Rate limit status: {result.get('status')}")
        print_test_status(
            f"✅ Total tracked clients: {result.get('total_tracked_clients')}"
        )
        print_test_status(f"✅ Active clients: {result.get('active_clients')}")

        print_test_status("✅ rate_limit_health_check function test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ rate_limit_health_check function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def test_prepared_statements_health_check_function():
    """Test the prepared statements health check function."""
    print_test_status("🔍 Testing prepared_statements_health_check function...")

    try:
        # Call the prepared statements health check function directly
        result = await prepared_statements_health_check()

        # Verify the result has expected structure
        if hasattr(result, "body"):
            # JSONResponse case
            import json

            result_data = json.loads(result.body.decode())
        else:
            # Direct dict case
            result_data = result

        assert (
            "status" in result_data
        ), "Prepared statements health check result should have 'status' field"
        assert (
            "timestamp" in result_data
        ), "Prepared statements health check result should have 'timestamp' field"

        print_test_status(f"✅ Prepared statements status: {result_data.get('status')}")
        print_test_status(
            f"✅ Checkpointer status: {result_data.get('checkpointer_status', 'Unknown')}"
        )

        print_test_status("✅ prepared_statements_health_check function test PASSED")
        return True

    except Exception as e:
        print_test_status(
            f"❌ prepared_statements_health_check function test FAILED: {e}"
        )
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


def test_health_router_structure():
    """Test that health router has the correct route structure."""
    print_test_status("🔍 Testing health router structure...")

    try:
        from api.routes.health import router

        # Check router routes
        routes = router.routes
        route_paths = [route.path for route in routes]

        expected_paths = [
            "/health",
            "/health/database",
            "/health/memory",
            "/health/rate-limits",
            "/health/prepared-statements",
        ]

        for expected_path in expected_paths:
            assert (
                expected_path in route_paths
            ), f"Expected route {expected_path} not found in router"
            print_test_status(f"✅ Route {expected_path} found in router")

        print_test_status(f"✅ Router has {len(routes)} routes: {route_paths}")
        print_test_status("✅ Health router structure test PASSED")
        return True

    except Exception as e:
        print_test_status(f"❌ Health router structure test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False


async def main():
    """Run all Phase 8.1 health routes tests."""
    print_test_status("🚀 Starting Phase 8.1 Health Routes Tests")
    print_test_status(f"📂 BASE_DIR: {BASE_DIR}")
    print_test_status("=" * 80)

    all_tests_passed = True

    # Run all tests
    tests = [
        ("Health Routes Imports", test_phase8_health_imports),
        ("Health Router Structure", test_health_router_structure),
        ("Health Check Function", test_health_check_function),
        ("Database Health Check Function", test_database_health_check_function),
        ("Memory Health Check Function", test_memory_health_check_function),
        ("Rate Limit Health Check Function", test_rate_limit_health_check_function),
        (
            "Prepared Statements Health Check Function",
            test_prepared_statements_health_check_function,
        ),
    ]

    for test_name, test_func in tests:
        print_test_status(f"\n📋 Running test: {test_name}")
        print_test_status("-" * 60)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
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
        print_test_status("🎉 ALL PHASE 8.1 HEALTH ROUTES TESTS PASSED!")
        print_test_status("✅ Health routes extraction successful")
        print_test_status("✅ Health router working correctly")
        print_test_status("✅ Health check functions working correctly")
        print_test_status("✅ All health endpoints functional")
    else:
        print_test_status("❌ SOME PHASE 8.1 HEALTH ROUTES TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

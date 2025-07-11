#!/usr/bin/env python3
"""
Test for Phase 1: Setup and Preparation
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
from datetime import datetime, timedelta
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30.0


async def test_phase1_folder_structure():
    """Test Phase 1 folder structure setup by verifying all directories exist."""
    print("🔍 Testing Phase 1 folder structure...")

    # List of required directories
    required_dirs = [
        "api",
        "api/config",
        "api/utils",
        "api/models",
        "api/middleware",
        "api/auth",
        "api/exceptions",
        "api/dependencies",
        "api/routes",
        "tests",
    ]

    # Check each directory exists
    for dir_path in required_dirs:
        full_path = BASE_DIR / dir_path
        if not full_path.exists():
            print(f"❌ Directory missing: {dir_path}")
            return False
        elif not full_path.is_dir():
            print(f"❌ Path exists but is not a directory: {dir_path}")
            return False
        else:
            print(f"✅ Directory exists: {dir_path}")

    print("✅ All required directories exist")
    return True


async def test_phase1_init_files():
    """Test Phase 1 __init__.py files by verifying all packages can be imported."""
    print("🔍 Testing Phase 1 __init__.py files...")

    # List of required __init__.py files
    required_init_files = [
        "api/__init__.py",
        "api/config/__init__.py",
        "api/utils/__init__.py",
        "api/models/__init__.py",
        "api/middleware/__init__.py",
        "api/auth/__init__.py",
        "api/exceptions/__init__.py",
        "api/dependencies/__init__.py",
        "api/routes/__init__.py",
        "tests/__init__.py",
    ]

    # Check each __init__.py file exists
    for init_file in required_init_files:
        full_path = BASE_DIR / init_file
        if not full_path.exists():
            print(f"❌ __init__.py file missing: {init_file}")
            return False
        elif not full_path.is_file():
            print(f"❌ Path exists but is not a file: {init_file}")
            return False
        else:
            print(f"✅ __init__.py file exists: {init_file}")

    print("✅ All required __init__.py files exist")
    return True


async def test_phase1_package_imports():
    """Test Phase 1 package imports by attempting to import each package."""
    print("🔍 Testing Phase 1 package imports...")

    # List of packages to test import
    packages_to_import = [
        "api",
        "api.config",
        "api.utils",
        "api.models",
        "api.middleware",
        "api.auth",
        "api.exceptions",
        "api.dependencies",
        "api.routes",
        "tests",
    ]

    # Try to import each package
    for package in packages_to_import:
        try:
            __import__(package)
            print(f"✅ Successfully imported: {package}")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error importing {package}: {e}")
            return False

    print("✅ All packages imported successfully")
    return True


async def test_phase1_functionality():
    """Test Phase 1 functionality by running all setup validation tests."""
    print("🔍 Testing Phase 1 setup functionality...")

    # Run all Phase 1 tests
    tests = [
        ("Folder Structure", test_phase1_folder_structure),
        ("Init Files", test_phase1_init_files),
        ("Package Imports", test_phase1_package_imports),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            results.append((test_name, False))

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n📊 Phase 1 Test Summary:")
    print(f"   Tests passed: {passed}/{total}")

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    return passed == total


async def main():
    """Main test runner for Phase 1."""
    print("🚀 Starting Phase 1 tests...")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Test timestamp: {datetime.now().isoformat()}")

    # Run Phase 1 tests
    success = await test_phase1_functionality()

    if success:
        print("\n✅ Phase 1 tests completed successfully")
        print("🎉 API server refactoring folder structure is ready!")
        return True
    else:
        print("\n❌ Phase 1 tests failed")
        print("🔧 Please check the folder structure and __init__.py files")
        return False


if __name__ == "__main__":
    asyncio.run(main())

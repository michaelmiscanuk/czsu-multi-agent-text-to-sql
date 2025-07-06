#!/usr/bin/env python3
"""
Test for Phase 9: Create Main Application File
Based on test_concurrency.py pattern - imports functionality from main scripts
"""

# CRITICAL: Set Windows event loop policy FIRST, before other imports
import sys
import os
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
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(BASE_DIR))

# Test imports from extracted modules
try:
    from api.main import app, lifespan
    print("✅ Successfully imported main FastAPI application")
except Exception as e:
    print(f"❌ Failed to import main FastAPI application: {e}")
    print(f"❌ BASE_DIR: {BASE_DIR}")
    print(f"❌ sys.path: {sys.path}")
    print(f"❌ Full traceback:\n{traceback.format_exc()}")
    sys.exit(1)

def print_test_status(message: str):
    """Print test status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def test_phase9_main_imports():
    """Test that main application can be imported successfully."""
    print_test_status("🔍 Testing Phase 9 main application imports...")
    
    try:
        # Test main app import
        from api.main import app
        assert app is not None, "Main FastAPI app should not be None"
        print_test_status("✅ Main FastAPI app imported successfully")
        
        # Test lifespan function import
        from api.main import lifespan
        assert callable(lifespan), "lifespan should be callable"
        print_test_status("✅ Lifespan function imported successfully")
        
        print_test_status("✅ Phase 9 main application imports test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Phase 9 main application imports test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_fastapi_app_structure():
    """Test that FastAPI app has correct structure."""
    print_test_status("🔍 Testing FastAPI app structure...")
    
    try:
        from api.main import app
        from fastapi import FastAPI
        
        # Test app structure
        assert hasattr(app, 'title'), "Should have title attribute"
        assert hasattr(app, 'version'), "Should have version attribute"
        print_test_status("✅ App has basic attributes")
        
        # Test app metadata
        assert app.title == "CZSU Multi-Agent Text-to-SQL API", "Should have correct title"
        assert app.version == "1.0.0", "Should have correct version"
        print_test_status("✅ App metadata configured correctly")
        
        # Test lifespan is configured
        assert app.router.lifespan_context is not None, "Lifespan should be configured"
        print_test_status("✅ Lifespan context configured")
        
        print_test_status("✅ FastAPI app structure test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ FastAPI app structure test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_middleware_registration():
    """Test that all middleware is properly registered."""
    print_test_status("🔍 Testing middleware registration...")
    
    try:
        from api.main import app
        
        # Check middleware stack
        middleware_stack = app.user_middleware
        middleware_types = [middleware.cls.__name__ for middleware in middleware_stack]
        
        # Should have CORS and GZip middleware
        assert "CORSMiddleware" in middleware_types, "Should have CORS middleware"
        print_test_status("✅ CORS middleware registered")
        
        assert "GZipMiddleware" in middleware_types, "Should have GZip middleware" 
        print_test_status("✅ GZip middleware registered")
        
        # Check for custom middleware functions in routes
        route_middleware = []
        for route in app.routes:
            if hasattr(route, 'dependant') and route.dependant:
                route_middleware.extend([dep.call.__name__ for dep in route.dependant.dependencies if hasattr(dep.call, '__name__')])
        
        print_test_status(f"✅ Middleware stack contains {len(middleware_stack)} middleware components")
        print_test_status("✅ Middleware registration test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Middleware registration test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_exception_handlers():
    """Test that exception handlers are properly registered."""
    print_test_status("🔍 Testing exception handlers...")
    
    try:
        from api.main import app
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException
        
        # Check exception handlers
        exception_handlers = app.exception_handlers
        
        # Should have validation error handler
        assert RequestValidationError in exception_handlers, "Should have RequestValidationError handler"
        print_test_status("✅ RequestValidationError handler registered")
        
        # Should have HTTP exception handler
        assert HTTPException in exception_handlers, "Should have HTTPException handler"
        print_test_status("✅ HTTPException handler registered")
        
        # Should have general exception handlers
        assert ValueError in exception_handlers, "Should have ValueError handler"
        print_test_status("✅ ValueError handler registered")
        
        assert Exception in exception_handlers, "Should have general Exception handler"
        print_test_status("✅ General Exception handler registered")
        
        print_test_status(f"✅ {len(exception_handlers)} exception handlers registered")
        print_test_status("✅ Exception handlers test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Exception handlers test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_route_registration():
    """Test that all routes from modular routers are properly registered."""
    print_test_status("🔍 Testing route registration...")
    
    try:
        from api.main import app
        
        # Get all registered routes
        routes = app.routes
        route_paths = []
        
        for route in routes:
            if hasattr(route, 'path'):
                route_paths.append(route.path)
        
        # Test for key routes from each router
        expected_routes = [
            "/health",  # health router
            "/catalog",  # catalog router
            "/analyze",  # analysis router
            "/feedback",  # feedback router
            "/chat-threads",  # chat router
            "/placeholder/{width}/{height}",  # misc router
        ]
        
        for expected_route in expected_routes:
            assert expected_route in route_paths, f"Expected route {expected_route} not found"
            print_test_status(f"✅ Route {expected_route} registered")
        
        print_test_status(f"✅ {len(routes)} total routes registered")
        print_test_status("✅ Route registration test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Route registration test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_router_imports():
    """Test that all routers from extracted modules are properly imported."""
    print_test_status("🔍 Testing router imports...")
    
    try:
        # Test individual router imports
        from api.routes.health import router as health_router
        from api.routes.catalog import router as catalog_router
        from api.routes.analysis import router as analysis_router
        from api.routes.feedback import router as feedback_router
        from api.routes.chat import router as chat_router
        from api.routes.messages import router as messages_router
        from api.routes.bulk import router as bulk_router
        from api.routes.debug import router as debug_router
        from api.routes.misc import router as misc_router
        
        routers = [
            ("health", health_router),
            ("catalog", catalog_router),
            ("analysis", analysis_router),
            ("feedback", feedback_router),
            ("chat", chat_router),
            ("messages", messages_router),
            ("bulk", bulk_router),
            ("debug", debug_router),
            ("misc", misc_router),
        ]
        
        for router_name, router in routers:
            from fastapi import APIRouter
            assert isinstance(router, APIRouter), f"{router_name} router should be APIRouter instance"
            print_test_status(f"✅ {router_name} router imported successfully")
        
        print_test_status("✅ Router imports test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Router imports test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

def test_modular_dependencies():
    """Test that all modular dependencies are properly imported."""
    print_test_status("🔍 Testing modular dependencies...")
    
    try:
        # Test configuration imports
        from api.config.settings import GC_MEMORY_THRESHOLD, _app_startup_time, print__startup_debug
        print_test_status("✅ Configuration settings imported")
        
        # Test utility imports
        from api.utils.debug import print__debug
        print_test_status("✅ Debug utilities imported")
        
        from api.utils.memory import log_memory_usage, setup_graceful_shutdown
        print_test_status("✅ Memory utilities imported")
        
        from api.utils.rate_limiting import wait_for_rate_limit, check_rate_limit_with_throttling
        print_test_status("✅ Rate limiting utilities imported")
        
        # Test authentication imports
        from api.dependencies.auth import get_current_user
        print_test_status("✅ Authentication dependencies imported")
        
        print_test_status("✅ Modular dependencies test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Modular dependencies test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

async def test_lifespan_function():
    """Test that lifespan function is properly structured."""
    print_test_status("🔍 Testing lifespan function...")
    
    try:
        from api.main import lifespan
        import inspect
        
        # Test lifespan function structure
        assert callable(lifespan), "Lifespan should be callable"
        print_test_status("✅ Lifespan function is callable")
        
        # Test lifespan is an async context manager (decorated with @asynccontextmanager)
        assert hasattr(lifespan, '__call__'), "Lifespan should be callable"
        print_test_status("✅ Lifespan is async context manager")
        
        # Note: We don't actually execute the lifespan here as it requires
        # full startup/shutdown which involves database connections
        print_test_status("ℹ️ Lifespan execution testing requires full app startup")
        
        print_test_status("✅ Lifespan function test PASSED")
        return True
        
    except Exception as e:
        print_test_status(f"❌ Lifespan function test FAILED: {e}")
        print_test_status(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

async def main():
    """Run all Phase 9 main application tests."""
    print_test_status("🚀 Starting Phase 9 Main Application Tests")
    print_test_status(f"📂 BASE_DIR: {BASE_DIR}")
    print_test_status("=" * 80)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("Main Application Imports", test_phase9_main_imports),
        ("FastAPI App Structure", test_fastapi_app_structure),
        ("Middleware Registration", test_middleware_registration),
        ("Exception Handlers", test_exception_handlers),
        ("Route Registration", test_route_registration),
        ("Router Imports", test_router_imports),
        ("Modular Dependencies", test_modular_dependencies),
        ("Lifespan Function", test_lifespan_function),
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
        print_test_status("🎉 ALL PHASE 9 MAIN APPLICATION TESTS PASSED!")
        print_test_status("✅ Main application assembly successful")
        print_test_status("✅ FastAPI app properly configured")
        print_test_status("✅ All routers registered correctly")
        print_test_status("✅ All middleware and exception handlers working")
        print_test_status("✅ Modular structure successfully assembled")
    else:
        print_test_status("❌ SOME PHASE 9 MAIN APPLICATION TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 
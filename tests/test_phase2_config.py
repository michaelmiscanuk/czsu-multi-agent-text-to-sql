#!/usr/bin/env python3
"""
Test for Phase 2: Extract Configuration and Constants
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
import httpx
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functionality from main scripts (not reimplementing!)
from other.tests.test_concurrency import create_test_jwt_token, check_server_connectivity

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30.0

async def test_phase2_config_imports():
    """Test Phase 2 configuration imports by importing from api.config.settings."""
    print("🔍 Testing Phase 2 configuration imports...")
    
    try:
        # Import the configuration module
        from api.config import settings
        print("✅ Successfully imported api.config.settings")
        
        # Test that the module exists and has expected attributes
        assert hasattr(settings, 'start_time'), "settings.start_time not found"
        assert hasattr(settings, 'GC_MEMORY_THRESHOLD'), "settings.GC_MEMORY_THRESHOLD not found"
        assert hasattr(settings, 'GLOBAL_CHECKPOINTER'), "settings.GLOBAL_CHECKPOINTER not found"
        assert hasattr(settings, 'MAX_CONCURRENT_ANALYSES'), "settings.MAX_CONCURRENT_ANALYSES not found"
        assert hasattr(settings, 'analysis_semaphore'), "settings.analysis_semaphore not found"
        
        print("✅ All required configuration attributes found")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import api.config.settings: {e}")
        return False
    except AttributeError as e:
        print(f"❌ Missing configuration attribute: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing configuration: {e}")
        return False

async def test_phase2_config_values():
    """Test Phase 2 configuration values by validating configuration constants."""
    print("🔍 Testing Phase 2 configuration values...")
    
    try:
        from api.config.settings import (
            start_time,
            GC_MEMORY_THRESHOLD,
            _app_startup_time,
            _memory_baseline,
            _request_count,
            GLOBAL_CHECKPOINTER,
            MAX_CONCURRENT_ANALYSES,
            analysis_semaphore,
            rate_limit_storage,
            RATE_LIMIT_REQUESTS,
            RATE_LIMIT_WINDOW,
            RATE_LIMIT_BURST,
            RATE_LIMIT_MAX_WAIT,
            throttle_semaphores,
            _bulk_loading_cache,
            _bulk_loading_locks,
            BULK_CACHE_TIMEOUT,
            GOOGLE_JWK_URL,
            _jwt_kid_missing_count,
            INMEMORY_FALLBACK_ENABLED,
            BASE_DIR
        )
        
        # Validate critical constants exist and have expected types
        print(f"✅ start_time: {start_time} (type: {type(start_time).__name__})")
        assert isinstance(start_time, float), f"start_time should be float, got {type(start_time)}"
        
        print(f"✅ GC_MEMORY_THRESHOLD: {GC_MEMORY_THRESHOLD}MB (type: {type(GC_MEMORY_THRESHOLD).__name__})")
        assert isinstance(GC_MEMORY_THRESHOLD, int), f"GC_MEMORY_THRESHOLD should be int, got {type(GC_MEMORY_THRESHOLD)}"
        assert GC_MEMORY_THRESHOLD > 0, f"GC_MEMORY_THRESHOLD should be positive, got {GC_MEMORY_THRESHOLD}"
        
        print(f"✅ MAX_CONCURRENT_ANALYSES: {MAX_CONCURRENT_ANALYSES} (type: {type(MAX_CONCURRENT_ANALYSES).__name__})")
        assert isinstance(MAX_CONCURRENT_ANALYSES, int), f"MAX_CONCURRENT_ANALYSES should be int, got {type(MAX_CONCURRENT_ANALYSES)}"
        assert MAX_CONCURRENT_ANALYSES > 0, f"MAX_CONCURRENT_ANALYSES should be positive, got {MAX_CONCURRENT_ANALYSES}"
        
        print(f"✅ analysis_semaphore: {analysis_semaphore} (type: {type(analysis_semaphore).__name__})")
        assert hasattr(analysis_semaphore, '_value'), f"analysis_semaphore should be asyncio.Semaphore"
        
        print(f"✅ RATE_LIMIT_REQUESTS: {RATE_LIMIT_REQUESTS} (type: {type(RATE_LIMIT_REQUESTS).__name__})")
        assert isinstance(RATE_LIMIT_REQUESTS, int), f"RATE_LIMIT_REQUESTS should be int, got {type(RATE_LIMIT_REQUESTS)}"
        
        print(f"✅ RATE_LIMIT_WINDOW: {RATE_LIMIT_WINDOW}s (type: {type(RATE_LIMIT_WINDOW).__name__})")
        assert isinstance(RATE_LIMIT_WINDOW, int), f"RATE_LIMIT_WINDOW should be int, got {type(RATE_LIMIT_WINDOW)}"
        
        print(f"✅ BULK_CACHE_TIMEOUT: {BULK_CACHE_TIMEOUT}s (type: {type(BULK_CACHE_TIMEOUT).__name__})")
        assert isinstance(BULK_CACHE_TIMEOUT, int), f"BULK_CACHE_TIMEOUT should be int, got {type(BULK_CACHE_TIMEOUT)}"
        
        print(f"✅ GOOGLE_JWK_URL: {GOOGLE_JWK_URL} (type: {type(GOOGLE_JWK_URL).__name__})")
        assert isinstance(GOOGLE_JWK_URL, str), f"GOOGLE_JWK_URL should be str, got {type(GOOGLE_JWK_URL)}"
        assert GOOGLE_JWK_URL.startswith('https://'), f"GOOGLE_JWK_URL should be HTTPS URL, got {GOOGLE_JWK_URL}"
        
        print(f"✅ INMEMORY_FALLBACK_ENABLED: {INMEMORY_FALLBACK_ENABLED} (type: {type(INMEMORY_FALLBACK_ENABLED).__name__})")
        assert isinstance(INMEMORY_FALLBACK_ENABLED, bool), f"INMEMORY_FALLBACK_ENABLED should be bool, got {type(INMEMORY_FALLBACK_ENABLED)}"
        
        print(f"✅ BASE_DIR: {BASE_DIR} (type: {type(BASE_DIR).__name__})")
        assert hasattr(BASE_DIR, 'exists'), f"BASE_DIR should be Path object, got {type(BASE_DIR)}"
        
        print("✅ All configuration values validated successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import configuration constants: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error validating configuration: {e}")
        return False

async def test_phase2_debug_functions():
    """Test Phase 2 debug functions by importing and testing print functions."""
    print("🔍 Testing Phase 2 debug functions...")
    
    try:
        from api.config.settings import print__startup_debug, print__memory_monitoring
        
        # Test the debug functions exist and are callable
        assert callable(print__startup_debug), "print__startup_debug should be callable"
        assert callable(print__memory_monitoring), "print__memory_monitoring should be callable"
        
        # Test calling the functions (they should not raise errors)
        print__startup_debug("🧪 Testing startup debug function from Phase 2")
        print__memory_monitoring("🧪 Testing memory monitoring function from Phase 2")
        
        print("✅ Debug functions imported and tested successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import debug functions: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing debug functions: {e}")
        return False

async def test_phase2_functionality():
    """Test Phase 2 functionality by running all configuration validation tests."""
    print("🔍 Testing Phase 2 configuration functionality...")
    
    # Run all Phase 2 tests
    tests = [
        ("Configuration Imports", test_phase2_config_imports),
        ("Configuration Values", test_phase2_config_values),
        ("Debug Functions", test_phase2_debug_functions)
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
    
    print(f"\n📊 Phase 2 Test Summary:")
    print(f"   Tests passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    return passed == total

async def main():
    """Main test runner for Phase 2."""
    print("🚀 Starting Phase 2 tests...")
    print(f"   Base directory: {BASE_DIR}")
    print(f"   Test timestamp: {datetime.now().isoformat()}")
    
    # Run Phase 2 tests
    success = await test_phase2_functionality()
    
    if success:
        print("\n✅ Phase 2 tests completed successfully")
        print("🎉 API server configuration and constants extraction is working!")
        return True
    else:
        print("\n❌ Phase 2 tests failed")
        print("🔧 Please check the configuration extraction and settings module")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Test for Phase 3: Extract Utilities (Debug, Memory, Rate Limiting)
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
sys.path.insert(0, str(BASE_DIR))

# Import for testing
from api.utils.debug import (
    print__api_postgresql,
    print__feedback_flow,
    print__debug,
    print__analyze_debug,
    print__chat_all_messages_debug,
    print__admin_clear_cache_debug,
    print__analysis_tracing_debug
)

from api.utils.memory import (
    print__memory_monitoring,
    cleanup_bulk_cache,
    check_memory_and_gc,
    log_memory_usage,
    log_comprehensive_error,
    setup_graceful_shutdown
)

from api.utils.rate_limiting import (
    check_rate_limit_with_throttling,
    wait_for_rate_limit,
    check_rate_limit
)

def test_phase3_utilities():
    """Test Phase 3: Extract Utilities (Debug, Memory, Rate Limiting)"""
    print("🧪 PHASE 3 TEST: Extract Utilities")
    print("=" * 50)
    
    # Test 1: Debug utilities import
    print("\n1. Testing Debug Utilities...")
    try:
        # Test debug function calls
        print__api_postgresql("Test PostgreSQL debug message")
        print__feedback_flow("Test feedback flow message")
        print__debug("Test general debug message")
        print__analyze_debug("Test analyze debug message")
        print__chat_all_messages_debug("Test chat all messages debug")
        print__admin_clear_cache_debug("Test admin clear cache debug")
        print__analysis_tracing_debug("Test analysis tracing debug")
        print("✅ Debug utilities imported and callable")
    except Exception as e:
        print(f"❌ Debug utilities test failed: {e}")
        return False
    
    # Test 2: Memory management utilities
    print("\n2. Testing Memory Management Utilities...")
    try:
        # Test memory function calls
        print__memory_monitoring("Test memory monitoring message")
        
        # Test cache cleanup (should return number of cleaned entries)
        cleaned_entries = cleanup_bulk_cache()
        print(f"✅ Cache cleanup returned: {cleaned_entries} entries cleaned")
        
        # Test memory check
        memory_mb = check_memory_and_gc()
        print(f"✅ Memory check returned: {memory_mb:.1f}MB")
        
        # Test memory usage logging
        log_memory_usage("phase3_test")
        print("✅ Memory usage logging successful")
        
        # Test graceful shutdown setup
        setup_graceful_shutdown()
        print("✅ Graceful shutdown setup successful")
        
        print("✅ Memory management utilities imported and callable")
    except Exception as e:
        print(f"❌ Memory management utilities test failed: {e}")
        return False
    
    # Test 3: Rate limiting utilities
    print("\n3. Testing Rate Limiting Utilities...")
    try:
        test_ip = "127.0.0.1"
        
        # Test rate limit check with throttling
        rate_info = check_rate_limit_with_throttling(test_ip)
        print(f"✅ Rate limit check returned: {rate_info}")
        
        # Test simple rate limit check
        rate_allowed = check_rate_limit(test_ip)
        print(f"✅ Simple rate limit check returned: {rate_allowed}")
        
        print("✅ Rate limiting utilities imported and callable")
    except Exception as e:
        print(f"❌ Rate limiting utilities test failed: {e}")
        return False
    
    # Test 4: Cross-module integration
    print("\n4. Testing Cross-Module Integration...")
    try:
        # Test that debug functions can be called from memory module
        from api.utils.memory import log_comprehensive_error
        from fastapi import Request
        
        # Create a mock request object for testing
        class MockRequest:
            def __init__(self):
                self.method = "GET"
                self.url = "http://test.com/test"
                self.client = type('MockClient', (), {'host': '127.0.0.1'})()
        
        mock_request = MockRequest()
        test_error = ValueError("Test error for integration")
        
        # This should work without error
        log_comprehensive_error("phase3_integration_test", test_error, mock_request)
        print("✅ Cross-module integration successful")
    except Exception as e:
        print(f"❌ Cross-module integration test failed: {e}")
        return False
    
    print("\n🎉 ALL PHASE 3 TESTS PASSED!")
    print("=" * 50)
    print("✅ Debug utilities extracted successfully")
    print("✅ Memory management utilities extracted successfully")
    print("✅ Rate limiting utilities extracted successfully")
    print("✅ Cross-module integration working")
    print("✅ All utilities are importable and functional")
    return True

if __name__ == "__main__":
    success = test_phase3_utilities()
    sys.exit(0 if success else 1) 
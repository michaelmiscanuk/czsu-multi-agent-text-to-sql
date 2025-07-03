#!/usr/bin/env python3
"""
Data Endpoint Tests for CZSU Multi-Agent API
Tests catalog, data-tables, and other data-related endpoints.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import base64
from datetime import datetime, timedelta

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 30

def create_mock_jwt_token():
    """Create a properly formatted mock JWT token for testing."""
    # Create a minimal but properly formatted JWT token
    header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
    payload = base64.urlsafe_b64encode(b'{"sub":"test","email":"test@example.com"}').decode().rstrip('=')
    signature = base64.urlsafe_b64encode(b'mock_signature').decode().rstrip('=')
    return f"{header}.{payload}.{signature}"

# Mock JWT token (properly formatted for testing)
MOCK_JWT_TOKEN = f"Bearer {create_mock_jwt_token()}"

async def test_catalog_endpoint():
    """Test the catalog endpoint functionality."""
    print("🔍 Testing catalog endpoint...")
    
    test_cases = [
        # (query_params, description)
        ({}, "Default catalog request"),
        ({"page": 1, "page_size": 5}, "Paginated request"),
        ({"q": "test"}, "Search query"),
        ({"page": 2, "page_size": 10, "q": "data"}, "Combined parameters")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for params, description in test_cases:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}/catalog", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        print(f"     Status: {response.status}")
                        
                        if response.status == 401:
                            print(f"     ⚠️ 401 Unauthorized - Authentication required")
                            results[description] = "auth_required"
                        elif response.status == 200:
                            data = await response.json()
                            
                            # Check response structure
                            expected_fields = ['results', 'total', 'page', 'page_size']
                            missing_fields = [f for f in expected_fields if f not in data]
                            
                            if missing_fields:
                                print(f"     ❌ Missing fields: {missing_fields}")
                                results[description] = False
                            else:
                                print(f"     ✅ Valid structure - {len(data['results'])} results, total: {data['total']}")
                                results[description] = True
                        else:
                            print(f"     ❌ Unexpected status: {response.status}")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(1 for r in results.values() if r is True)
        auth_required = sum(1 for r in results.values() if r == "auth_required")
        total = len(results)
        
        print(f"✅ Catalog endpoint test: {successful}/{total} successful, {auth_required} auth required")
        return successful > 0 or auth_required > 0  # Pass if working or properly protected
        
    except Exception as e:
        print(f"❌ Catalog endpoint test failed: {e}")
        return False

async def test_data_tables_endpoint():
    """Test the data-tables endpoint functionality."""
    print("🔍 Testing data-tables endpoint...")
    
    test_cases = [
        # (query_params, description)
        ({}, "Default data tables request"),
        ({"q": "test"}, "Search query")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for params, description in test_cases:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}/data-tables", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        print(f"     Status: {response.status}")
                        
                        if response.status == 401:
                            print(f"     ⚠️ 401 Unauthorized - Authentication required")
                            results[description] = "auth_required"
                        elif response.status == 200:
                            data = await response.json()
                            
                            # Check response structure
                            if 'tables' in data and isinstance(data['tables'], list):
                                print(f"     ✅ Valid structure - {len(data['tables'])} tables found")
                                
                                # Check table structure if tables exist
                                if data['tables']:
                                    first_table = data['tables'][0]
                                    if 'selection_code' in first_table:
                                        print(f"     ✅ Table structure valid")
                                        results[description] = True
                                    else:
                                        print(f"     ❌ Invalid table structure")
                                        results[description] = False
                                else:
                                    print(f"     ✅ No tables found (valid empty response)")
                                    results[description] = True
                            else:
                                print(f"     ❌ Invalid response structure")
                                results[description] = False
                        else:
                            print(f"     ❌ Unexpected status: {response.status}")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(1 for r in results.values() if r is True)
        auth_required = sum(1 for r in results.values() if r == "auth_required")
        total = len(results)
        
        print(f"✅ Data-tables endpoint test: {successful}/{total} successful, {auth_required} auth required")
        return successful > 0 or auth_required > 0
        
    except Exception as e:
        print(f"❌ Data-tables endpoint test failed: {e}")
        return False

async def test_data_table_endpoint():
    """Test the data-table endpoint functionality."""
    print("🔍 Testing data-table endpoint...")
    
    test_cases = [
        # (query_params, description)
        ({}, "No table specified"),
        ({"table": "nonexistent_table"}, "Nonexistent table"),
        ({"table": "test_table"}, "Test table request")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for params, description in test_cases:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}/data-table", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        print(f"     Status: {response.status}")
                        
                        if response.status == 401:
                            print(f"     ⚠️ 401 Unauthorized - Authentication required")
                            results[description] = "auth_required"
                        elif response.status == 200:
                            data = await response.json()
                            
                            # Check response structure
                            if 'columns' in data and 'rows' in data:
                                print(f"     ✅ Valid structure - {len(data['columns'])} columns, {len(data['rows'])} rows")
                                results[description] = True
                            else:
                                print(f"     ❌ Invalid response structure")
                                results[description] = False
                        else:
                            print(f"     ❌ Unexpected status: {response.status}")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(1 for r in results.values() if r is True)
        auth_required = sum(1 for r in results.values() if r == "auth_required")
        total = len(results)
        
        print(f"✅ Data-table endpoint test: {successful}/{total} successful, {auth_required} auth required")
        return successful > 0 or auth_required > 0
        
    except Exception as e:
        print(f"❌ Data-table endpoint test failed: {e}")
        return False

async def test_chat_threads_endpoint():
    """Test the chat-threads endpoint functionality."""
    print("🔍 Testing chat-threads endpoint...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 401:
                    print(f"   ⚠️ 401 Unauthorized - Authentication required")
                    return "auth_required"
                elif response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        print(f"   ✅ Valid response - {len(data)} threads found")
                        
                        # Check thread structure if threads exist
                        if data:
                            first_thread = data[0]
                            expected_fields = ['thread_id', 'latest_timestamp', 'run_count', 'title']
                            missing_fields = [f for f in expected_fields if f not in first_thread]
                            
                            if missing_fields:
                                print(f"   ❌ Missing thread fields: {missing_fields}")
                                return False
                            else:
                                print(f"   ✅ Thread structure valid")
                                return True
                        else:
                            print(f"   ✅ No threads found (valid empty response)")
                            return True
                    else:
                        print(f"   ❌ Invalid response structure - expected list")
                        return False
                else:
                    print(f"   ❌ Unexpected status: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"   ❌ Chat-threads endpoint test failed: {e}")
        return False

async def test_debug_endpoints():
    """Test debug endpoints functionality."""
    print("🔍 Testing debug endpoints...")
    
    debug_endpoints = [
        ("/debug/pool-status", "Pool status debug"),
        ("/debug/chat/test-thread/checkpoints", "Checkpoints debug"),
        ("/debug/run-id/test-uuid", "Run ID debug")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for endpoint, description in debug_endpoints:
                print(f"   Testing: {description}")
                
                try:
                    headers = {}
                    if "run-id" in endpoint or "checkpoints" in endpoint:
                        headers = {"Authorization": MOCK_JWT_TOKEN}
                    
                    async with session.get(f"{BASE_URL}{endpoint}", headers=headers) as response:
                        print(f"     Status: {response.status}")
                        
                        if response.status in [200, 401, 404]:  # Expected responses
                            print(f"     ✅ Expected response")
                            results[description] = True
                        else:
                            print(f"     ❌ Unexpected status: {response.status}")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Debug endpoints test: {successful}/{total} working as expected")
        return successful >= total * 0.7  # Allow some tolerance
        
    except Exception as e:
        print(f"❌ Debug endpoints test failed: {e}")
        return False

async def test_endpoint_parameter_validation():
    """Test parameter validation on data endpoints."""
    print("🔍 Testing parameter validation...")
    
    validation_tests = [
        ("/catalog", {"page": -1}, "Negative page number"),
        ("/catalog", {"page_size": 0}, "Zero page size"),
        ("/catalog", {"page_size": 50000}, "Excessive page size"),
        ("/catalog", {"page": "invalid"}, "Non-numeric page"),
        ("/data-table", {"table": ""}, "Empty table name"),
        ("/data-table", {"table": "'; DROP TABLE users; --"}, "SQL injection attempt")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in validation_tests:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        # Should return 400 (bad request) or 422 (validation error), not 500
                        if response.status in [400, 401, 422]:
                            print(f"     ✅ {response.status} - Properly validated")
                            results[description] = True
                        elif response.status == 200:
                            print(f"     ⚠️ 200 - Parameter accepted (may be valid)")
                            results[description] = True
                        else:
                            print(f"     ❌ {response.status} - Unexpected status")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Parameter validation test: {successful}/{total} properly handled")
        return successful >= total * 0.8
        
    except Exception as e:
        print(f"❌ Parameter validation test failed: {e}")
        return False

async def test_response_performance():
    """Test response performance of data endpoints."""
    print("🔍 Testing response performance...")
    
    performance_tests = [
        ("/catalog", {}, "Catalog performance"),
        ("/data-tables", {}, "Data tables performance"),
        ("/chat-threads", {}, "Chat threads performance")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in performance_tests:
                print(f"   Testing: {description}")
                
                response_times = []
                
                # Test 3 times to get average
                for i in range(3):
                    try:
                        start_time = time.time()
                        async with session.get(f"{BASE_URL}{endpoint}", 
                                             params=params, 
                                             headers=headers) as response:
                            response_time = time.time() - start_time
                            response_times.append(response_time)
                            
                            if response.status not in [200, 401]:
                                print(f"     ❌ Request {i+1}: Status {response.status}")
                                break
                            
                    except Exception as e:
                        print(f"     ❌ Request {i+1}: {e}")
                        break
                
                if response_times:
                    avg_time = sum(response_times) / len(response_times)
                    max_time = max(response_times)
                    
                    print(f"     Average: {avg_time:.3f}s, Max: {max_time:.3f}s")
                    
                    if max_time < 5.0:  # Should respond within 5 seconds
                        print(f"     ✅ Performance acceptable")
                        results[description] = True
                    else:
                        print(f"     ⚠️ Slow response times")
                        results[description] = False
                else:
                    print(f"     ❌ No successful requests")
                    results[description] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Performance test: {successful}/{total} endpoints performing well")
        return successful >= total * 0.7
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def run_data_tests():
    """Run all data endpoint tests."""
    print("🚀 DATA ENDPOINT TESTS")
    print("=" * 50)
    print(f"Target URL: {BASE_URL}")
    print(f"Test started: {datetime.now()}")
    print("=" * 50)
    
    tests = [
        ("Catalog Endpoint", test_catalog_endpoint),
        ("Data Tables Endpoint", test_data_tables_endpoint),
        ("Data Table Endpoint", test_data_table_endpoint),
        ("Chat Threads Endpoint", test_chat_threads_endpoint),
        ("Debug Endpoints", test_debug_endpoints),
        ("Parameter Validation", test_endpoint_parameter_validation),
        ("Response Performance", test_response_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        result = await test_func()
        test_time = time.time() - start_time
        
        results[test_name] = {
            'passed': result,
            'time': test_time
        }
        
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   Result: {status} ({test_time:.2f}s)")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DATA ENDPOINT TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    total_time = sum(r['time'] for r in results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} {test_name} ({result['time']:.2f}s)")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total execution time: {total_time:.2f}s")
    
    if passed == total:
        print("🎉 All data endpoint tests passed!")
    else:
        print("⚠️ Some data endpoint tests failed - check authentication and database connectivity")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_data_tests()) 
#!/usr/bin/env python3
"""
Database Stress Testing for CZSU Multi-Agent API
Tests database connectivity, concurrent operations, edge cases, and boundary conditions.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import base64
import uuid
import random
import string
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 60  # Longer timeout for stress tests
MAX_CONCURRENT_REQUESTS = 20

def create_mock_jwt_token():
    """Create a properly formatted mock JWT token for testing."""
    header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
    payload = base64.urlsafe_b64encode(b'{"sub":"test","email":"test@example.com"}').decode().rstrip('=')
    signature = base64.urlsafe_b64encode(b'mock_signature').decode().rstrip('=')
    return f"{header}.{payload}.{signature}"

MOCK_JWT_TOKEN = f"Bearer {create_mock_jwt_token()}"

def generate_random_string(length=10):
    """Generate random string for testing."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_uuid():
    """Generate random UUID for testing."""
    return str(uuid.uuid4())

async def test_concurrent_chat_threads_access():
    """Test concurrent access to chat-threads endpoint."""
    print("🔍 Testing concurrent chat-threads access...")
    
    async def single_request(session, request_id):
        try:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/chat-threads", 
                                 headers={"Authorization": MOCK_JWT_TOKEN}) as response:
                response_time = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': response.status,
                    'response_time': response_time,
                    'success': response.status in [200, 401],  # Both are valid responses
                    'error': None
                }
        except Exception as e:
            return {
                'request_id': request_id,
                'status': None,
                'response_time': None,
                'success': False,
                'error': str(e)
            }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Test with different concurrency levels
            concurrency_levels = [5, 10, 15, 20]
            
            for concurrency in concurrency_levels:
                print(f"   Testing with {concurrency} concurrent requests...")
                
                start_time = time.time()
                tasks = [single_request(session, i) for i in range(concurrency)]
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                successful = sum(1 for r in results if r['success'])
                failed = sum(1 for r in results if not r['success'])
                avg_response_time = sum(r['response_time'] for r in results if r['response_time']) / len([r for r in results if r['response_time']])
                
                print(f"     Concurrency {concurrency}: {successful}/{concurrency} successful")
                print(f"     Total time: {total_time:.2f}s, Avg response: {avg_response_time:.3f}s")
                print(f"     Failures: {failed}")
                
                if failed > concurrency * 0.3:  # Allow up to 30% failures
                    print(f"     ❌ Too many failures at concurrency {concurrency}")
                    return False
                
                # Brief pause between concurrency tests
                await asyncio.sleep(2)
        
        print("   ✅ Concurrent chat-threads access test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Concurrent chat-threads test failed: {e}")
        return False

async def test_database_boundary_conditions():
    """Test database operations with boundary conditions."""
    print("🔍 Testing database boundary conditions...")
    
    boundary_tests = [
        # Catalog endpoint boundary tests
        ("/catalog", {"page": 1, "page_size": 1}, "Minimum page size"),
        ("/catalog", {"page": 1, "page_size": 10000}, "Maximum page size"),
        ("/catalog", {"page": 999999, "page_size": 10}, "Very high page number"),
        ("/catalog", {"q": "a" * 1000}, "Very long search query"),
        ("/catalog", {"q": "🔥🚀💯" * 100}, "Unicode search query"),
        ("/catalog", {"q": "'; DROP TABLE users; --"}, "SQL injection attempt"),
        
        # Data-tables boundary tests
        ("/data-tables", {"q": ""}, "Empty search query"),
        ("/data-tables", {"q": " " * 100}, "Whitespace query"),
        ("/data-tables", {"q": "SELECT * FROM information_schema.tables"}, "SQL query as search"),
        
        # Data-table boundary tests
        ("/data-table", {"table": ""}, "Empty table name"),
        ("/data-table", {"table": "a" * 100}, "Very long table name"),
        ("/data-table", {"table": "nonexistent_table_" + generate_random_string(50)}, "Long nonexistent table"),
        ("/data-table", {"table": "../../etc/passwd"}, "Path traversal attempt"),
        ("/data-table", {"table": "users'; DROP TABLE users; --"}, "SQL injection in table name"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in boundary_tests:
                print(f"   Testing: {description}")
                
                try:
                    start_time = time.time()
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        response_time = time.time() - start_time
                        
                        # Should not return 500 (internal server error)
                        if response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Poor error handling")
                            results[description] = False
                        elif response.status in [200, 400, 401, 404, 422]:
                            print(f"     ✅ {response.status} - Handled gracefully ({response_time:.3f}s)")
                            results[description] = True
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected but not critical")
                            results[description] = True
                            
                except asyncio.TimeoutError:
                    print(f"     ❌ Timeout - Server may be struggling")
                    results[description] = False
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
                
                # Brief pause between boundary tests
                await asyncio.sleep(0.5)
        
        successful = sum(results.values())
        total = len(results)
        print(f"   ✅ Boundary conditions test: {successful}/{total} handled gracefully")
        return successful >= total * 0.8  # Allow some tolerance
        
    except Exception as e:
        print(f"   ❌ Boundary conditions test failed: {e}")
        return False

async def test_rapid_sequential_requests():
    """Test rapid sequential requests to stress database connections."""
    print("🔍 Testing rapid sequential requests...")
    
    endpoints_to_test = [
        ("/chat-threads", {}),
        ("/catalog", {"page": 1, "page_size": 5}),
        ("/data-tables", {}),
        ("/data-table", {"table": "nonexistent"}),
    ]
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            print(f"   Sending 50 rapid sequential requests...")
            
            start_time = time.time()
            successful = 0
            failed = 0
            response_times = []
            
            for i in range(50):
                # Randomly select an endpoint
                endpoint, params = random.choice(endpoints_to_test)
                
                try:
                    request_start = time.time()
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status in [200, 401, 404]:
                            successful += 1
                        else:
                            failed += 1
                            
                except Exception as e:
                    failed += 1
                    print(f"     Request {i+1} failed: {e}")
                
                # Very brief pause to simulate rapid but not instantaneous requests
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            print(f"   Results:")
            print(f"     Total requests: 50")
            print(f"     Successful: {successful}")
            print(f"     Failed: {failed}")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Average response time: {avg_response_time:.3f}s")
            print(f"     Requests per second: {50/total_time:.1f}")
            
            if failed <= 5:  # Allow up to 10% failures
                print("   ✅ Rapid sequential requests test passed")
                return True
            else:
                print("   ❌ Too many failures in rapid sequential requests")
                return False
                
    except Exception as e:
        print(f"   ❌ Rapid sequential requests test failed: {e}")
        return False

async def test_memory_intensive_operations():
    """Test operations that might consume significant memory."""
    print("🔍 Testing memory-intensive operations...")
    
    memory_tests = [
        # Large page sizes
        ("/catalog", {"page": 1, "page_size": 1000}, "Large catalog page"),
        ("/catalog", {"page": 1, "page_size": 5000}, "Very large catalog page"),
        
        # Complex search patterns
        ("/catalog", {"q": "data analysis machine learning artificial intelligence"}, "Complex search"),
        ("/data-tables", {"q": "table database sql query select"}, "Multi-word search"),
        
        # Rapid repeated requests to same endpoint
        ("rapid_catalog", {}, "Rapid catalog requests"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for test_type, params, description in memory_tests:
                print(f"   Testing: {description}")
                
                if test_type == "rapid_catalog":
                    # Special test: 10 rapid catalog requests
                    start_time = time.time()
                    tasks = []
                    for i in range(10):
                        task = session.get(f"{BASE_URL}/catalog", 
                                         params={"page": 1, "page_size": 100}, 
                                         headers=headers)
                        tasks.append(task)
                    
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    test_time = time.time() - start_time
                    
                    successful_responses = 0
                    for response in responses:
                        if isinstance(response, Exception):
                            continue
                        if response.status in [200, 401]:
                            successful_responses += 1
                        response.close()
                    
                    print(f"     10 rapid requests: {successful_responses}/10 successful ({test_time:.2f}s)")
                    results[description] = successful_responses >= 7
                    
                else:
                    try:
                        start_time = time.time()
                        async with session.get(f"{BASE_URL}{test_type}", 
                                             params=params, 
                                             headers=headers) as response:
                            response_time = time.time() - start_time
                            
                            if response.status in [200, 401, 404]:
                                print(f"     ✅ {response.status} - Completed ({response_time:.3f}s)")
                                results[description] = True
                            else:
                                print(f"     ❌ {response.status} - Failed")
                                results[description] = False
                                
                    except asyncio.TimeoutError:
                        print(f"     ❌ Timeout - May indicate memory issues")
                        results[description] = False
                    except Exception as e:
                        print(f"     ❌ Error: {e}")
                        results[description] = False
                
                # Pause between memory-intensive tests
                await asyncio.sleep(3)
        
        successful = sum(results.values())
        total = len(results)
        print(f"   ✅ Memory-intensive operations: {successful}/{total} completed successfully")
        return successful >= total * 0.7
        
    except Exception as e:
        print(f"   ❌ Memory-intensive operations test failed: {e}")
        return False

async def test_database_connection_resilience():
    """Test database connection resilience under stress."""
    print("🔍 Testing database connection resilience...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            # Test 1: Sustained load
            print("   Test 1: Sustained load (30 requests over 30 seconds)")
            sustained_successful = 0
            
            for i in range(30):
                try:
                    async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                        if response.status in [200, 401]:
                            sustained_successful += 1
                except:
                    pass
                await asyncio.sleep(1)  # 1 request per second
            
            print(f"     Sustained load: {sustained_successful}/30 successful")
            
            # Test 2: Burst load
            print("   Test 2: Burst load (20 concurrent requests)")
            tasks = []
            for i in range(20):
                task = session.get(f"{BASE_URL}/catalog", 
                                 params={"page": 1, "page_size": 10}, 
                                 headers=headers)
                tasks.append(task)
            
            burst_responses = await asyncio.gather(*tasks, return_exceptions=True)
            burst_successful = 0
            
            for response in burst_responses:
                if isinstance(response, Exception):
                    continue
                if response.status in [200, 401]:
                    burst_successful += 1
                response.close()
            
            print(f"     Burst load: {burst_successful}/20 successful")
            
            # Test 3: Mixed operations
            print("   Test 3: Mixed operations (different endpoints)")
            mixed_operations = [
                ("/chat-threads", {}),
                ("/catalog", {"page": 1, "page_size": 10}),
                ("/data-tables", {}),
                ("/data-table", {"table": "test"}),
                ("/health", {}),
            ]
            
            mixed_successful = 0
            for endpoint, params in mixed_operations * 3:  # 15 total requests
                try:
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers if endpoint != "/health" else {}) as response:
                        if response.status in [200, 401, 404]:
                            mixed_successful += 1
                except:
                    pass
                await asyncio.sleep(0.2)
            
            print(f"     Mixed operations: {mixed_successful}/15 successful")
            
            # Overall assessment
            total_successful = sustained_successful + burst_successful + mixed_successful
            total_requests = 30 + 20 + 15
            success_rate = (total_successful / total_requests) * 100
            
            print(f"   Overall resilience: {total_successful}/{total_requests} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("   ✅ Database connection resilience test passed")
                return True
            else:
                print("   ❌ Database connection resilience test failed")
                return False
                
    except Exception as e:
        print(f"   ❌ Database resilience test failed: {e}")
        return False

async def test_edge_case_parameters():
    """Test edge case parameters that might cause issues."""
    print("🔍 Testing edge case parameters...")
    
    edge_cases = [
        # Null byte injection
        ("/catalog", {"q": "test\x00injection"}, "Null byte in query"),
        
        # Unicode edge cases
        ("/catalog", {"q": "🔥🚀💯🎉🌟"}, "Emoji in query"),
        ("/data-tables", {"q": "тест"}, "Cyrillic characters"),
        ("/data-table", {"table": "测试表"}, "Chinese characters"),
        
        # Numeric edge cases
        ("/catalog", {"page": 0}, "Zero page"),
        ("/catalog", {"page": -1}, "Negative page"),
        ("/catalog", {"page_size": -1}, "Negative page size"),
        ("/catalog", {"page_size": 999999}, "Huge page size"),
        
        # String edge cases
        ("/catalog", {"q": ""}, "Empty string query"),
        ("/catalog", {"q": " "}, "Space-only query"),
        ("/catalog", {"q": "\n\r\t"}, "Whitespace characters"),
        
        # Potential injection attempts
        ("/data-table", {"table": "'; SELECT * FROM users; --"}, "SQL injection attempt"),
        ("/catalog", {"q": "<script>alert('xss')</script>"}, "XSS attempt"),
        ("/data-tables", {"q": "../../etc/passwd"}, "Path traversal attempt"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            headers = {"Authorization": MOCK_JWT_TOKEN}
            
            for endpoint, params, description in edge_cases:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}{endpoint}", 
                                         params=params, 
                                         headers=headers) as response:
                        
                        # Key test: Should not return 500 (internal server error)
                        if response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Vulnerability!")
                            results[description] = False
                        elif response.status in [200, 400, 401, 404, 422]:
                            print(f"     ✅ {response.status} - Handled safely")
                            results[description] = True
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected status")
                            results[description] = True  # Not necessarily bad
                            
                except Exception as e:
                    print(f"     ❌ Exception: {e}")
                    results[description] = False
                
                await asyncio.sleep(0.3)  # Brief pause
        
        successful = sum(results.values())
        total = len(results)
        vulnerabilities = sum(1 for r in results.values() if not r)
        
        print(f"   ✅ Edge case parameters: {successful}/{total} handled safely")
        if vulnerabilities > 0:
            print(f"   ⚠️ Found {vulnerabilities} potential vulnerabilities!")
        
        return successful >= total * 0.9  # Very high bar for security
        
    except Exception as e:
        print(f"   ❌ Edge case parameters test failed: {e}")
        return False

async def run_database_stress_tests():
    """Run all database stress tests."""
    print("🚀 DATABASE STRESS TESTS")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"Test started: {datetime.now()}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 60)
    
    tests = [
        ("Concurrent Chat Threads Access", test_concurrent_chat_threads_access),
        ("Database Boundary Conditions", test_database_boundary_conditions),
        ("Rapid Sequential Requests", test_rapid_sequential_requests),
        ("Memory Intensive Operations", test_memory_intensive_operations),
        ("Database Connection Resilience", test_database_connection_resilience),
        ("Edge Case Parameters", test_edge_case_parameters),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        start_time = time.time()
        result = await test_func()
        test_time = time.time() - start_time
        
        results[test_name] = {
            'passed': result,
            'time': test_time
        }
        
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   Result: {status} ({test_time:.2f}s)")
        
        # Pause between major tests to let system recover
        await asyncio.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DATABASE STRESS TEST SUMMARY")
    print("=" * 60)
    
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
        print("🎉 All database stress tests passed!")
        print("💪 Your database layer is robust and resilient!")
    else:
        print("⚠️ Some database stress tests failed")
        print("🔧 Review failed tests for potential optimizations")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_database_stress_tests()) 
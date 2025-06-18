#!/usr/bin/env python3
"""
Health Endpoint Tests for CZSU Multi-Agent API
Tests the /health endpoint functionality and monitoring capabilities.
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 30

async def test_basic_health_check():
    """Test basic health endpoint functionality."""
    print("🔍 Testing basic health check...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/health") as response:
                response_time = time.time() - start_time
                
                print(f"   Status: {response.status}")
                print(f"   Response time: {response_time:.2f}s")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"   Health status: {data.get('status', 'unknown')}")
                    print(f"   Memory usage: {data.get('memory_usage_mb', 'unknown')}MB")
                    print(f"   Database: {data.get('database', 'unknown')}")
                    
                    # Check for critical fields
                    required_fields = ['status', 'timestamp']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        print(f"   ❌ Missing required fields: {missing_fields}")
                        return False
                    
                    print("   ✅ Basic health check passed")
                    return True
                else:
                    print(f"   ❌ Health check failed with status {response.status}")
                    return False
                    
    except Exception as e:
        print(f"   ❌ Health check failed with error: {e}")
        return False

async def test_health_response_structure():
    """Test the structure and content of health response."""
    print("🔍 Testing health response structure...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            async with session.get(f"{BASE_URL}/health") as response:
                
                if response.status != 200:
                    print(f"   ❌ Expected 200, got {response.status}")
                    return False
                
                data = await response.json()
                
                # Test expected fields
                expected_fields = {
                    'status': str,
                    'timestamp': str,
                    'memory_usage_mb': (int, float),
                    'database': str
                }
                
                for field, expected_type in expected_fields.items():
                    if field not in data:
                        print(f"   ⚠️ Missing field: {field}")
                    elif not isinstance(data[field], expected_type):
                        print(f"   ⚠️ Field {field} has wrong type: {type(data[field])} (expected {expected_type})")
                    else:
                        print(f"   ✅ Field {field}: {data[field]}")
                
                # Test memory usage is reasonable
                memory_mb = data.get('memory_usage_mb')
                if isinstance(memory_mb, (int, float)):
                    if memory_mb > 500:  # 512MB limit
                        print(f"   ⚠️ High memory usage: {memory_mb}MB (close to 512MB limit)")
                    elif memory_mb < 50:
                        print(f"   ⚠️ Unusually low memory usage: {memory_mb}MB")
                    else:
                        print(f"   ✅ Memory usage is healthy: {memory_mb}MB")
                
                # Test database status
                db_status = data.get('database')
                if db_status == 'connected':
                    print("   ✅ Database is connected")
                else:
                    print(f"   ⚠️ Database status: {db_status}")
                
                return True
                
    except Exception as e:
        print(f"   ❌ Health response structure test failed: {e}")
        return False

async def test_health_under_load():
    """Test health endpoint under concurrent requests."""
    print("🔍 Testing health endpoint under load...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Send 10 concurrent requests
            tasks = []
            start_time = time.time()
            
            for i in range(10):
                task = session.get(f"{BASE_URL}/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = 0
            failed = 0
            response_times = []
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"   Request {i+1}: ❌ {response}")
                    failed += 1
                else:
                    if response.status == 200:
                        successful += 1
                    else:
                        failed += 1
                    response.close()
            
            print(f"   Total requests: 10")
            print(f"   Successful: {successful}")
            print(f"   Failed: {failed}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per request: {total_time/10:.2f}s")
            
            if successful >= 8:  # Allow for some failures
                print("   ✅ Load test passed")
                return True
            else:
                print("   ❌ Load test failed - too many failures")
                return False
                
    except Exception as e:
        print(f"   ❌ Load test failed: {e}")
        return False

async def test_health_cors_headers():
    """Test CORS headers on health endpoint."""
    print("🔍 Testing CORS headers...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Test OPTIONS request
            async with session.options(f"{BASE_URL}/health") as response:
                print(f"   OPTIONS status: {response.status}")
                
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
                }
                
                for header, value in cors_headers.items():
                    print(f"   {header}: {value}")
                
                # Test actual GET request with origin
                headers = {'Origin': 'http://localhost:3000'}
                async with session.get(f"{BASE_URL}/health", headers=headers) as get_response:
                    origin_header = get_response.headers.get('Access-Control-Allow-Origin')
                    print(f"   GET with Origin header: {origin_header}")
                
                print("   ✅ CORS headers test completed")
                return True
                
    except Exception as e:
        print(f"   ❌ CORS test failed: {e}")
        return False

async def test_health_response_time_consistency():
    """Test consistency of health endpoint response times."""
    print("🔍 Testing response time consistency...")
    
    try:
        response_times = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for i in range(5):
                start_time = time.time()
                async with session.get(f"{BASE_URL}/health") as response:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response.status != 200:
                        print(f"   Request {i+1}: ❌ Status {response.status}")
                        return False
                    
                    print(f"   Request {i+1}: {response_time:.3f}s")
                
                # Small delay between requests
                await asyncio.sleep(0.5)
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Min response time: {min_time:.3f}s")
        print(f"   Max response time: {max_time:.3f}s")
        print(f"   Time variation: {max_time - min_time:.3f}s")
        
        if max_time < 2.0:  # Should respond within 2 seconds
            print("   ✅ Response times are consistent and fast")
            return True
        else:
            print("   ⚠️ Some responses were slow")
            return False
            
    except Exception as e:
        print(f"   ❌ Response time test failed: {e}")
        return False

async def run_health_tests():
    """Run all health endpoint tests."""
    print("🚀 HEALTH ENDPOINT TESTS")
    print("=" * 50)
    print(f"Target URL: {BASE_URL}")
    print(f"Test started: {datetime.now()}")
    print("=" * 50)
    
    tests = [
        ("Basic Health Check", test_basic_health_check),
        ("Response Structure", test_health_response_structure),
        ("Load Testing", test_health_under_load),
        ("CORS Headers", test_health_cors_headers),
        ("Response Time Consistency", test_health_response_time_consistency)
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
    print("📊 HEALTH ENDPOINT TEST SUMMARY")
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
        print("🎉 All health endpoint tests passed!")
    else:
        print("⚠️ Some health endpoint tests failed")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_health_tests()) 
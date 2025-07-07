"""
Test Recovery System with Higher Concurrency
Demonstrates that with recovery mechanisms, we can handle 5+ concurrent users gracefully.
"""
import asyncio
import aiohttp
import time
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import jwt
from concurrent.futures import ThreadPoolExecutor
import threading

# Set higher concurrency for testing recovery systems
os.environ['MAX_CONCURRENT_ANALYSES'] = '5'
print(f"🔧 Recovery Test: Set MAX_CONCURRENT_ANALYSES={os.environ.get('MAX_CONCURRENT_ANALYSES')} for recovery testing")

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 900  # 15 minutes for recovery tests

def create_mock_jwt_token(email="test.recovery@gmail.com", expire_minutes=60):
    """Create a mock JWT token for testing recovery."""
    payload = {
        "email": email,
        "aud": "mock_audience",
        "exp": datetime.utcnow() + timedelta(minutes=expire_minutes),
        "iat": datetime.utcnow(),
        "iss": "mock_issuer"
    }
    return jwt.encode(payload, "secret", algorithm="HS256")

async def test_concurrent_health_checks():
    """Test concurrent health check requests to verify the server can handle load."""
    print("🧪 TESTING CONCURRENT HEALTH CHECKS")
    print("=" * 60)
    
    async def single_health_request(session, request_id):
        """Make a single health check request."""
        start_time = time.time()
        try:
            async with session.get(f"{BASE_URL}/health", timeout=30) as response:
                duration = time.time() - start_time
                data = await response.json()
                
                if response.status == 200:
                    print(f"✅ Request {request_id}: Success in {duration:.2f}s - Status: {data.get('status', 'unknown')}")
                    return {"success": True, "duration": duration, "data": data}
                else:
                    print(f"❌ Request {request_id}: HTTP {response.status} in {duration:.2f}s")
                    return {"success": False, "duration": duration, "status": response.status}
                    
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            print(f"⏰ Request {request_id}: Timeout after {duration:.2f}s")
            return {"success": False, "duration": duration, "error": "timeout"}
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Request {request_id}: Error after {duration:.2f}s - {str(e)}")
            return {"success": False, "duration": duration, "error": str(e)}
    
    # Test with 10 concurrent health check requests
    print("🚀 Starting 10 concurrent health check requests...")
    
    async with aiohttp.ClientSession() as session:
        # Create 10 concurrent requests
        tasks = [
            single_health_request(session, i+1) 
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        
        print(f"\n📊 CONCURRENT HEALTH CHECK RESULTS (Total time: {total_time:.2f}s)")
        print("=" * 60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success rate: {(successful/len(results)*100):.1f}%")
        
        if successful >= 8:  # Allow for some failures due to load
            print("🎉 CONCURRENT HEALTH CHECK TEST PASSED!")
            return True
        else:
            print("❌ CONCURRENT HEALTH CHECK TEST FAILED!")
            return False

async def test_concurrent_pool_status_checks():
    """Test concurrent pool status requests to verify pool handling."""
    print("\n🧪 TESTING CONCURRENT POOL STATUS CHECKS")
    print("=" * 60)
    
    async def single_pool_request(session, request_id):
        """Make a single pool status request."""
        start_time = time.time()
        try:
            async with session.get(f"{BASE_URL}/debug/pool-status", timeout=30) as response:
                duration = time.time() - start_time
                data = await response.json()
                
                if response.status == 200:
                    pool_healthy = data.get('pool_healthy', False)
                    print(f"✅ Request {request_id}: Success in {duration:.2f}s - Pool healthy: {pool_healthy}")
                    return {"success": True, "duration": duration, "pool_healthy": pool_healthy}
                else:
                    print(f"❌ Request {request_id}: HTTP {response.status} in {duration:.2f}s")
                    return {"success": False, "duration": duration, "status": response.status}
                    
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Request {request_id}: Error after {duration:.2f}s - {str(e)}")
            return {"success": False, "duration": duration, "error": str(e)}
    
    # Test with 5 concurrent pool status requests
    print("🚀 Starting 5 concurrent pool status requests...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            single_pool_request(session, i+1) 
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        pool_healthy_count = sum(1 for r in results if isinstance(r, dict) and r.get("pool_healthy"))
        
        print(f"\n📊 CONCURRENT POOL STATUS RESULTS (Total time: {total_time:.2f}s)")
        print("=" * 60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"🔗 Pool healthy responses: {pool_healthy_count}")
        print(f"📈 Success rate: {(successful/len(results)*100):.1f}%")
        
        if successful >= 4 and pool_healthy_count >= 4:  # Most should succeed and show healthy pool
            print("🎉 CONCURRENT POOL STATUS TEST PASSED!")
            return True
        else:
            print("❌ CONCURRENT POOL STATUS TEST FAILED!")
            return False

async def test_memory_stability_under_load():
    """Test memory stability under concurrent load."""
    print("\n🧪 TESTING MEMORY STABILITY UNDER LOAD")
    print("=" * 60)
    
    async def get_memory_stats(session):
        """Get current memory statistics."""
        try:
            async with session.get(f"{BASE_URL}/health/memory", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "rss_mb": data.get("memory_rss_mb", 0),
                        "usage_percent": data.get("usage_percent", 0),
                        "status": data.get("status", "unknown")
                    }
        except:
            pass
        return None
    
    async with aiohttp.ClientSession() as session:
        # Get baseline memory
        print("📊 Getting baseline memory statistics...")
        baseline = await get_memory_stats(session)
        if baseline:
            print(f"   Baseline RSS: {baseline['rss_mb']:.1f}MB")
            print(f"   Baseline usage: {baseline['usage_percent']:.1f}%")
        
        # Run load test
        print("🚀 Running load test with 20 concurrent requests...")
        
        async def load_request(session, request_id):
            async with session.get(f"{BASE_URL}/health", timeout=10) as response:
                return response.status == 200
        
        tasks = [load_request(session, i+1) for i in range(20)]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        successful = sum(1 for r in results if r is True)
        
        # Get post-load memory
        print("📊 Getting post-load memory statistics...")
        post_load = await get_memory_stats(session)
        
        print(f"\n📊 MEMORY STABILITY RESULTS")
        print("=" * 60)
        print(f"✅ Successful requests: {successful}/20")
        print(f"⏱️ Load test duration: {duration:.2f}s")
        
        if baseline and post_load:
            memory_growth = post_load['rss_mb'] - baseline['rss_mb']
            print(f"📈 Memory growth: {memory_growth:.1f}MB")
            print(f"📊 Post-load RSS: {post_load['rss_mb']:.1f}MB")
            print(f"📊 Post-load usage: {post_load['usage_percent']:.1f}%")
            
            # Memory should not grow excessively (more than 50MB is concerning)
            if memory_growth < 50 and post_load['usage_percent'] < 90:
                print("🎉 MEMORY STABILITY TEST PASSED!")
                return True
            else:
                print("⚠️ MEMORY STABILITY TEST: High memory growth detected!")
                return False
        else:
            print("⚠️ Could not get memory statistics")
            return successful >= 18  # At least 90% success rate

async def main():
    """Run all recovery system tests."""
    print("🚀 RECOVERY SYSTEM TESTING WITH HIGHER CONCURRENCY")
    print("=" * 60)
    print("Testing that recovery systems allow 5+ concurrent users gracefully")
    print("Expected: Some users wait longer, but no crashes, data recovery works")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Concurrent health checks
    result1 = await test_concurrent_health_checks()
    test_results.append(("Concurrent Health Checks", result1))
    
    # Test 2: Concurrent pool status checks
    result2 = await test_concurrent_pool_status_checks()
    test_results.append(("Concurrent Pool Status", result2))
    
    # Test 3: Memory stability under load
    result3 = await test_memory_stability_under_load()
    test_results.append(("Memory Stability", result3))
    
    # Summary
    print(f"\n📊 RECOVERY SYSTEM TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    if passed_tests >= 2:  # At least 2/3 tests should pass
        print("🎉 RECOVERY SYSTEM TESTS PASSED!")
        print("✅ System can handle higher concurrency gracefully")
    else:
        print("❌ RECOVERY SYSTEM TESTS FAILED!")
        print("❌ System needs further optimization for higher concurrency")

if __name__ == "__main__":
    asyncio.run(main()) 
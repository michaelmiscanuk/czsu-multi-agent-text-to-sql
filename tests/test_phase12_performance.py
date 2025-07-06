#!/usr/bin/env python3
"""
Test for Phase 12: Performance Testing
Based on test_concurrency.py pattern - tests application startup time and performance
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

# Standard imports
import asyncio
import time
import httpx
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functionality from main scripts
from other.tests.test_concurrency import create_test_jwt_token, check_server_connectivity

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 60.0

class PerformanceResults:
    """Class to track performance test results."""
    
    def __init__(self):
        self.startup_time = None
        self.memory_baseline = None
        self.memory_after_requests = None
        self.request_times = []
        self.concurrent_performance = None
        
    def add_request_time(self, endpoint: str, response_time: float):
        """Add a request timing result."""
        self.request_times.append({
            "endpoint": endpoint,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self):
        """Get performance summary."""
        if self.request_times:
            avg_time = sum(r["response_time"] for r in self.request_times) / len(self.request_times)
            max_time = max(r["response_time"] for r in self.request_times)
            min_time = min(r["response_time"] for r in self.request_times)
        else:
            avg_time = max_time = min_time = 0
            
        return {
            "startup_time": self.startup_time,
            "memory_baseline_mb": self.memory_baseline,
            "memory_after_requests_mb": self.memory_after_requests,
            "memory_growth_mb": (self.memory_after_requests - self.memory_baseline) if both_exist(self.memory_baseline, self.memory_after_requests) else None,
            "total_requests": len(self.request_times),
            "avg_response_time": avg_time,
            "max_response_time": max_time,
            "min_response_time": min_time,
            "concurrent_performance": self.concurrent_performance
        }

def both_exist(a, b):
    """Helper to check if both values exist."""
    return a is not None and b is not None

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return None

async def test_application_startup_time():
    """Test application startup time with modular structure."""
    print("🔍 Testing application startup time...")
    
    try:
        # Measure time to get first successful health check
        start_time = time.time()
        
        # Wait for server to be ready (with timeout)
        max_wait = 30  # seconds
        while time.time() - start_time < max_wait:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    response = await client.get(f"{SERVER_BASE_URL}/health")
                    if response.status_code == 200:
                        startup_time = time.time() - start_time
                        print(f"✅ Application startup time: {startup_time:.2f}s")
                        return startup_time
            except:
                await asyncio.sleep(1)
        
        print("❌ Application did not start within timeout")
        return None
        
    except Exception as e:
        print(f"❌ Startup time test failed: {e}")
        return None

async def test_memory_usage_patterns():
    """Test memory usage patterns with modular structure."""
    print("🔍 Testing memory usage patterns...")
    
    try:
        # Get baseline memory
        baseline_memory = get_memory_usage()
        print(f"📊 Baseline memory: {baseline_memory:.1f}MB" if baseline_memory else "📊 Could not measure baseline memory")
        
        # Make several requests to test memory behavior
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        async with httpx.AsyncClient(base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            # Make multiple requests to different endpoints
            endpoints = [
                "/health",
                "/health/memory", 
                "/catalog",
                "/data-tables",
                "/chat-threads"
            ]
            
            for endpoint in endpoints:
                response = await client.get(endpoint, headers=headers if endpoint != "/health" else None)
                print(f"📊 {endpoint}: {response.status_code}")
                await asyncio.sleep(0.5)  # Small delay between requests
        
        # Get memory after requests
        final_memory = get_memory_usage()
        print(f"📊 Memory after requests: {final_memory:.1f}MB" if final_memory else "📊 Could not measure final memory")
        
        if baseline_memory and final_memory:
            growth = final_memory - baseline_memory
            print(f"📊 Memory growth: {growth:.1f}MB")
            
            # Check if memory growth is reasonable (less than 100MB for basic requests)
            if growth < 100:
                print("✅ Memory usage is reasonable")
                return {"baseline": baseline_memory, "final": final_memory, "growth": growth}
            else:
                print("⚠️ High memory growth detected")
                return {"baseline": baseline_memory, "final": final_memory, "growth": growth}
        else:
            print("⚠️ Could not measure memory properly")
            return {"baseline": baseline_memory, "final": final_memory, "growth": None}
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return None

async def test_concurrent_request_handling():
    """Test concurrent request handling performance."""
    print("🔍 Testing concurrent request handling...")
    
    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        async def make_request(endpoint: str, client: httpx.AsyncClient):
            """Make a single request and measure time."""
            start_time = time.time()
            try:
                response = await client.get(endpoint, headers=headers)
                response_time = time.time() - start_time
                return {
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "endpoint": endpoint,
                    "status_code": 0,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Test concurrent requests to health endpoints
        concurrent_start = time.time()
        
        async with httpx.AsyncClient(base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            # Make 5 concurrent requests
            tasks = [
                make_request("/health", client),
                make_request("/health/memory", client),
                make_request("/health/database", client),
                make_request("/health/rate-limits", client),
                make_request("/debug/pool-status", client)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_time = time.time() - concurrent_start
        
        # Analyze results
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        total_requests = len(results)
        avg_response_time = sum(r.get("response_time", 0) for r in results if isinstance(r, dict)) / total_requests
        
        print(f"📊 Concurrent test: {successful_requests}/{total_requests} successful")
        print(f"📊 Total concurrent time: {concurrent_time:.2f}s")
        print(f"📊 Average response time: {avg_response_time:.2f}s")
        
        performance_data = {
            "total_time": concurrent_time,
            "successful_requests": successful_requests,
            "total_requests": total_requests,
            "avg_response_time": avg_response_time,
            "success_rate": (successful_requests / total_requests) * 100
        }
        
        if successful_requests >= 4:  # At least 80% success rate
            print("✅ Concurrent request handling working well")
        else:
            print("⚠️ Some issues with concurrent request handling")
            
        return performance_data
        
    except Exception as e:
        print(f"❌ Concurrent request test failed: {e}")
        return None

async def test_endpoint_response_times():
    """Test individual endpoint response times."""
    print("🔍 Testing endpoint response times...")
    
    results = PerformanceResults()
    
    try:
        token = create_test_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test endpoints with expected performance characteristics
        test_endpoints = [
            ("/health", None),  # Should be very fast
            ("/health/memory", None),  # Should be fast
            ("/catalog", headers),  # Should be moderate
            ("/data-tables", headers),  # Should be moderate
            ("/chat-threads", headers)  # Should be moderate
        ]
        
        async with httpx.AsyncClient(base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            for endpoint, request_headers in test_endpoints:
                start_time = time.time()
                try:
                    response = await client.get(endpoint, headers=request_headers)
                    response_time = time.time() - start_time
                    
                    results.add_request_time(endpoint, response_time)
                    
                    status = "✅" if response.status_code == 200 else "⚠️"
                    print(f"{status} {endpoint}: {response_time:.3f}s (status: {response.status_code})")
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    results.add_request_time(endpoint, response_time)
                    print(f"❌ {endpoint}: {response_time:.3f}s (error: {str(e)[:50]})")
                
                # Small delay between requests
                await asyncio.sleep(0.2)
        
        return results
        
    except Exception as e:
        print(f"❌ Response time test failed: {e}")
        return results

async def main():
    """Main performance test runner."""
    print("🚀 Phase 12: Performance Testing Starting...")
    print("="*60)
    
    # Check server connectivity first
    if not await check_server_connectivity():
        print("❌ Server connectivity check failed!")
        print("   Please start your uvicorn server first:")
        print(f"   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    
    print("✅ Server is accessible - running performance tests...")
    
    performance_results = PerformanceResults()
    
    # Test 1: Application startup time
    print(f"\n📋 Test 1: Application Startup Time")
    startup_time = await test_application_startup_time()
    performance_results.startup_time = startup_time
    
    # Test 2: Memory usage patterns  
    print(f"\n📋 Test 2: Memory Usage Patterns")
    memory_data = await test_memory_usage_patterns()
    if memory_data:
        performance_results.memory_baseline = memory_data.get("baseline")
        performance_results.memory_after_requests = memory_data.get("final")
    
    # Test 3: Endpoint response times
    print(f"\n📋 Test 3: Endpoint Response Times")
    response_time_results = await test_endpoint_response_times()
    performance_results.request_times = response_time_results.request_times
    
    # Test 4: Concurrent request handling
    print(f"\n📋 Test 4: Concurrent Request Handling")
    concurrent_data = await test_concurrent_request_handling()
    performance_results.concurrent_performance = concurrent_data
    
    # Performance Analysis
    print("\n" + "="*60)
    print("📊 PERFORMANCE TEST RESULTS")
    print("="*60)
    
    summary = performance_results.get_summary()
    
    # Startup time analysis
    if summary["startup_time"]:
        startup_status = "✅ GOOD" if summary["startup_time"] < 10 else "⚠️ SLOW" if summary["startup_time"] < 30 else "❌ TOO SLOW"
        print(f"🚀 Startup Time: {summary['startup_time']:.2f}s {startup_status}")
    
    # Memory analysis
    if summary["memory_growth_mb"]:
        memory_status = "✅ GOOD" if summary["memory_growth_mb"] < 50 else "⚠️ HIGH" if summary["memory_growth_mb"] < 100 else "❌ EXCESSIVE"
        print(f"💾 Memory Growth: {summary['memory_growth_mb']:.1f}MB {memory_status}")
    
    # Response time analysis
    if summary["avg_response_time"]:
        response_status = "✅ GOOD" if summary["avg_response_time"] < 1.0 else "⚠️ SLOW" if summary["avg_response_time"] < 3.0 else "❌ TOO SLOW"
        print(f"⚡ Avg Response Time: {summary['avg_response_time']:.3f}s {response_status}")
    
    # Concurrent performance analysis
    if summary["concurrent_performance"]:
        concurrent_data = summary["concurrent_performance"]
        success_rate = concurrent_data.get("success_rate", 0)
        concurrent_status = "✅ GOOD" if success_rate >= 80 else "⚠️ ISSUES" if success_rate >= 60 else "❌ POOR"
        print(f"🔄 Concurrent Success Rate: {success_rate:.1f}% {concurrent_status}")
    
    # Overall assessment
    issues = 0
    if summary["startup_time"] and summary["startup_time"] > 30:
        issues += 1
    if summary["memory_growth_mb"] and summary["memory_growth_mb"] > 100:
        issues += 1
    if summary["avg_response_time"] and summary["avg_response_time"] > 3.0:
        issues += 1
    if summary["concurrent_performance"] and summary["concurrent_performance"].get("success_rate", 0) < 80:
        issues += 1
    
    print(f"\n🏁 OVERALL PERFORMANCE: ", end="")
    if issues == 0:
        print("🎉 EXCELLENT - All performance metrics are good")
        overall_success = True
    elif issues <= 1:
        print("✅ GOOD - Minor performance issues detected")
        overall_success = True
    else:
        print(f"⚠️ NEEDS ATTENTION - {issues} performance issues detected")
        overall_success = False
    
    return overall_success

if __name__ == "__main__":
    # Set debug mode for better visibility
    os.environ['DEBUG'] = '1'
    os.environ['USE_TEST_TOKENS'] = '1'
    
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 
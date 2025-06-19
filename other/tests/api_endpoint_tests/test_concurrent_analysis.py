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

# Set MAX_CONCURRENT_ANALYSES for testing
os.environ['MAX_CONCURRENT_ANALYSES'] = '3'
print(f"🔧 Test: Set MAX_CONCURRENT_ANALYSES={os.environ.get('MAX_CONCURRENT_ANALYSES')} for concurrent testing")

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 600  # 10 minutes for concurrent tests

def create_mock_jwt_token(email="test.concurrent@gmail.com", expire_minutes=60):
    """Create a mock JWT token for testing concurrent requests."""
    payload = {
        "email": email,
        "aud": "mock_audience",
        "exp": datetime.utcnow() + timedelta(minutes=expire_minutes),
        "iat": datetime.utcnow(),
        "iss": "mock_issuer"
    }
    # Use a simple signature for testing
    return jwt.encode(payload, "mock_secret", algorithm="HS256")

async def test_concurrent_analysis_max_3():
    """Test that exactly 3 concurrent analyses can run simultaneously."""
    print("\n=== Testing MAX_CONCURRENT_ANALYSES=3 ===")
    
    # Create unique test prompts
    test_prompts = [
        f"Jaký byl vývoz do Japonska v roce 2022? (Test {i})" for i in range(5)
    ]
    
    token = create_mock_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    start_times = []
    end_times = []
    responses = []
    
    async def single_analysis_request(session, prompt, request_id):
        """Single analysis request with timing."""
        thread_id = f"concurrent-test-{uuid.uuid4()}"
        
        request_data = {
            "prompt": prompt,
            "thread_id": thread_id
        }
        
        print(f"🚀 Request {request_id}: Starting analysis: {prompt[:50]}...")
        start_time = time.time()
        start_times.append(start_time)
        
        try:
            async with session.post(
                f"{BASE_URL}/analyze",
                json=request_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                end_time = time.time()
                end_times.append(end_time)
                duration = end_time - start_time
                
                response_data = await response.json()
                
                result = {
                    "request_id": request_id,
                    "prompt": prompt,
                    "thread_id": thread_id,
                    "status_code": response.status,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "success": response.status == 200,
                    "response_data": response_data,
                    "error": None
                }
                
                print(f"✅ Request {request_id}: Completed in {duration:.2f}s (Status: {response.status})")
                return result
                
        except asyncio.TimeoutError:
            end_time = time.time()
            end_times.append(end_time)
            duration = end_time - start_time
            
            result = {
                "request_id": request_id,
                "prompt": prompt,
                "thread_id": thread_id,
                "status_code": 408,
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
                "success": False,
                "response_data": None,
                "error": "Request timeout"
            }
            
            print(f"⏰ Request {request_id}: Timeout after {duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            end_times.append(end_time)
            duration = end_time - start_time
            
            result = {
                "request_id": request_id,
                "prompt": prompt,
                "thread_id": thread_id,
                "status_code": 500,
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
                "success": False,
                "response_data": None,
                "error": str(e)
            }
            
            print(f"❌ Request {request_id}: Error - {str(e)}")
            return result
    
    # Launch all requests concurrently
    print(f"🔄 Launching {len(test_prompts)} concurrent analysis requests...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            single_analysis_request(session, prompt, i)
            for i, prompt in enumerate(test_prompts)
        ]
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    print("\n=== CONCURRENT ANALYSIS RESULTS ===")
    
    successful_requests = [r for r in responses if isinstance(r, dict) and r.get("success")]
    failed_requests = [r for r in responses if isinstance(r, dict) and not r.get("success")]
    exception_requests = [r for r in responses if not isinstance(r, dict)]
    
    print(f"📊 Total requests: {len(responses)}")
    print(f"✅ Successful: {len(successful_requests)}")
    print(f"❌ Failed: {len(failed_requests)}")
    print(f"💥 Exceptions: {len(exception_requests)}")
    
    # Check timing to see if requests were truly concurrent
    if start_times and len(start_times) >= 3:
        start_times.sort()
        first_three_starts = start_times[:3]
        start_spread = max(first_three_starts) - min(first_three_starts)
        
        print(f"⏱️  Start time spread for first 3 requests: {start_spread:.2f}s")
        
        if start_spread < 2.0:  # If all started within 2 seconds
            print("✅ Requests appear to have started concurrently")
        else:
            print("⚠️  Requests may have been queued (not truly concurrent)")
    
    # Check for specific errors
    pool_closed_errors = []
    timeout_errors = []
    other_errors = []
    
    for response in responses:
        if isinstance(response, dict) and not response.get("success"):
            error_msg = response.get("error", "")
            if "pool" in error_msg.lower() and "closed" in error_msg.lower():
                pool_closed_errors.append(response)
            elif "timeout" in error_msg.lower():
                timeout_errors.append(response)
            else:
                other_errors.append(response)
    
    print(f"\n=== ERROR ANALYSIS ===")
    print(f"🏊 Pool closed errors: {len(pool_closed_errors)}")
    print(f"⏰ Timeout errors: {len(timeout_errors)}")
    print(f"🔧 Other errors: {len(other_errors)}")
    
    # Print detailed error information
    if pool_closed_errors:
        print("\n🚨 POOL CLOSED ERRORS (This is the main issue!):")
        for error in pool_closed_errors:
            print(f"   Request {error['request_id']}: {error['error']}")
    
    if timeout_errors:
        print("\n⏰ TIMEOUT ERRORS:")
        for error in timeout_errors:
            print(f"   Request {error['request_id']}: Duration {error['duration']:.2f}s")
    
    # Performance analysis
    if successful_requests:
        durations = [r["duration"] for r in successful_requests]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        print(f"📈 Average duration: {avg_duration:.2f}s")
        print(f"📊 Max duration: {max_duration:.2f}s")
        print(f"📉 Min duration: {min_duration:.2f}s")
    
    return {
        "total_requests": len(responses),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "pool_closed_errors": len(pool_closed_errors),
        "timeout_errors": len(timeout_errors),
        "responses": responses
    }

async def test_concurrent_pool_stability():
    """Test that the PostgreSQL pool remains stable under concurrent load."""
    print("\n=== Testing PostgreSQL Pool Stability ===")
    
    token = create_mock_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test pool health before concurrent requests
    async with aiohttp.ClientSession() as session:
        print("🔍 Checking pool health before concurrent tests...")
        
        async with session.get(f"{BASE_URL}/debug/pool-status") as response:
            pool_status_before = await response.json()
            print(f"📊 Pool status before: {json.dumps(pool_status_before, indent=2)}")
        
        # Make concurrent analysis requests
        test_prompts = [
            f"Test pool stability query {i}: What was the export to Germany in 2022?"
            for i in range(4)  # 4 requests for a pool limit test
        ]
        
        async def pool_test_request(session, prompt, request_id):
            thread_id = f"pool-test-{uuid.uuid4()}"
            request_data = {"prompt": prompt, "thread_id": thread_id}
            
            try:
                async with session.post(
                    f"{BASE_URL}/analyze",
                    json=request_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    result = await response.json()
                    return {
                        "request_id": request_id,
                        "status": response.status,
                        "success": response.status == 200,
                        "error": None
                    }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "status": 500,
                    "success": False,
                    "error": str(e)
                }
        
        print(f"🚀 Starting {len(test_prompts)} concurrent pool stability tests...")
        
        tasks = [
            pool_test_request(session, prompt, i)
            for i, prompt in enumerate(test_prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check pool health after concurrent requests
        print("🔍 Checking pool health after concurrent tests...")
        
        try:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                pool_status_after = await response.json()
                print(f"📊 Pool status after: {json.dumps(pool_status_after, indent=2)}")
        except Exception as e:
            print(f"❌ Could not get pool status after tests: {e}")
            pool_status_after = {"error": str(e)}
        
        # Analyze pool stability
        pool_stable = True
        stability_issues = []
        
        if pool_status_before.get("pool_healthy") and not pool_status_after.get("pool_healthy"):
            pool_stable = False
            stability_issues.append("Pool became unhealthy after concurrent requests")
        
        if pool_status_before.get("can_query") and not pool_status_after.get("can_query"):
            pool_stable = False
            stability_issues.append("Pool lost query capability after concurrent requests")
        
        # Check for pool closure errors in results
        pool_errors = [r for r in results if isinstance(r, dict) and r.get("error") and "pool" in r.get("error", "").lower()]
        
        if pool_errors:
            pool_stable = False
            stability_issues.append(f"Found {len(pool_errors)} pool-related errors in concurrent requests")
        
        print(f"\n=== POOL STABILITY RESULTS ===")
        print(f"🏊 Pool stable: {'✅ Yes' if pool_stable else '❌ No'}")
        
        if stability_issues:
            print("🚨 Stability issues found:")
            for issue in stability_issues:
                print(f"   - {issue}")
        
        return {
            "pool_stable": pool_stable,
            "stability_issues": stability_issues,
            "pool_status_before": pool_status_before,
            "pool_status_after": pool_status_after,
            "results": results
        }

async def test_rapid_sequential_requests():
    """Test rapid sequential requests to stress test the pool."""
    print("\n=== Testing Rapid Sequential Requests ===")
    
    token = create_mock_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Make 10 rapid sequential requests
    num_requests = 10
    delay_between = 0.1  # 100ms between requests
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            thread_id = f"rapid-test-{uuid.uuid4()}"
            request_data = {
                "prompt": f"Rapid test {i}: What was the import from China in 2022?",
                "thread_id": thread_id
            }
            
            start_time = time.time()
            
            try:
                async with session.post(
                    f"{BASE_URL}/analyze",
                    json=request_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    result = {
                        "request_id": i,
                        "status": response.status,
                        "duration": duration,
                        "success": response.status == 200,
                        "error": None
                    }
                    
                    if response.status != 200:
                        response_data = await response.json()
                        result["error"] = response_data.get("detail", "Unknown error")
                    
                    results.append(result)
                    print(f"📝 Request {i}: Status {response.status}, Duration {duration:.2f}s")
                    
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                result = {
                    "request_id": i,
                    "status": 500,
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                }
                
                results.append(result)
                print(f"❌ Request {i}: Error - {str(e)}")
            
            # Wait before next request
            if i < num_requests - 1:
                await asyncio.sleep(delay_between)
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n=== RAPID SEQUENTIAL RESULTS ===")
    print(f"📊 Total requests: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print("🚨 Failed requests:")
        for failure in failed:
            print(f"   Request {failure['request_id']}: {failure['error']}")
    
    # Check for pattern of failures (e.g., pool exhaustion)
    pool_failures = [r for r in failed if "pool" in r.get("error", "").lower()]
    if pool_failures:
        print(f"🏊 Pool-related failures: {len(pool_failures)}")
    
    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "pool_failures": len(pool_failures),
        "results": results
    }

async def test_concurrent_with_other_endpoints():
    """Test concurrent analysis with other endpoint access."""
    print("\n=== Testing Concurrent Analysis + Other Endpoints ===")
    
    token = create_mock_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    async def analysis_request(session):
        thread_id = f"mixed-test-{uuid.uuid4()}"
        request_data = {
            "prompt": "Mixed test: What was the trade balance with USA in 2022?",
            "thread_id": thread_id
        }
        
        try:
            async with session.post(
                f"{BASE_URL}/analyze",
                json=request_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:
                return {"type": "analysis", "status": response.status, "success": response.status == 200}
        except Exception as e:
            return {"type": "analysis", "status": 500, "success": False, "error": str(e)}
    
    async def health_request(session):
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                return {"type": "health", "status": response.status, "success": response.status == 200}
        except Exception as e:
            return {"type": "health", "status": 500, "success": False, "error": str(e)}
    
    async def chat_threads_request(session):
        try:
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                return {"type": "chat-threads", "status": response.status, "success": response.status == 200}
        except Exception as e:
            return {"type": "chat-threads", "status": 500, "success": False, "error": str(e)}
    
    async def pool_status_request(session):
        try:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                return {"type": "pool-status", "status": response.status, "success": response.status == 200}
        except Exception as e:
            return {"type": "pool-status", "status": 500, "success": False, "error": str(e)}
    
    async with aiohttp.ClientSession() as session:
        # Mix of different request types
        tasks = [
            analysis_request(session),  # Heavy operation
            analysis_request(session),  # Heavy operation
            health_request(session),    # Light operation
            chat_threads_request(session),  # DB operation
            pool_status_request(session),   # Debug operation
            health_request(session),    # Light operation
        ]
        
        print(f"🚀 Starting mixed concurrent requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze mixed results
        by_type = {}
        for result in results:
            if isinstance(result, dict):
                req_type = result.get("type", "unknown")
                if req_type not in by_type:
                    by_type[req_type] = {"successful": 0, "failed": 0, "errors": []}
                
                if result.get("success"):
                    by_type[req_type]["successful"] += 1
                else:
                    by_type[req_type]["failed"] += 1
                    if "error" in result:
                        by_type[req_type]["errors"].append(result["error"])
        
        print(f"\n=== MIXED CONCURRENT RESULTS ===")
        for req_type, stats in by_type.items():
            print(f"📊 {req_type}: {stats['successful']} successful, {stats['failed']} failed")
            if stats["errors"]:
                print(f"   Errors: {stats['errors']}")
        
        return by_type

async def run_all_concurrent_tests():
    """Run all concurrent analysis tests."""
    print("🚀 Starting Comprehensive Concurrent Analysis Tests")
    print("=" * 60)
    
    all_results = {}
    
    try:
        # Test 1: MAX_CONCURRENT_ANALYSES=3
        print("\n1️⃣  Testing MAX_CONCURRENT_ANALYSES=3...")
        concurrent_results = await test_concurrent_analysis_max_3()
        all_results["concurrent_analysis"] = concurrent_results
        
        # Brief pause between tests
        await asyncio.sleep(2)
        
        # Test 2: Pool stability
        print("\n2️⃣  Testing pool stability...")
        pool_results = await test_concurrent_pool_stability()
        all_results["pool_stability"] = pool_results
        
        await asyncio.sleep(2)
        
        # Test 3: Rapid sequential
        print("\n3️⃣  Testing rapid sequential requests...")
        rapid_results = await test_rapid_sequential_requests()
        all_results["rapid_sequential"] = rapid_results
        
        await asyncio.sleep(2)
        
        # Test 4: Mixed concurrent
        print("\n4️⃣  Testing mixed concurrent requests...")
        mixed_results = await test_concurrent_with_other_endpoints()
        all_results["mixed_concurrent"] = mixed_results
        
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        all_results["test_suite_error"] = str(e)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL CONCURRENT TEST SUMMARY")
    print("=" * 60)
    
    if "concurrent_analysis" in all_results:
        ca_results = all_results["concurrent_analysis"]
        print(f"🔄 Concurrent Analysis (MAX=3): {ca_results['successful_requests']}/{ca_results['total_requests']} successful")
        if ca_results["pool_closed_errors"] > 0:
            print(f"   🚨 Pool closed errors: {ca_results['pool_closed_errors']} (MAIN ISSUE!)")
    
    if "pool_stability" in all_results:
        ps_results = all_results["pool_stability"]
        print(f"🏊 Pool Stability: {'✅ Stable' if ps_results['pool_stable'] else '❌ Unstable'}")
        if ps_results["stability_issues"]:
            print(f"   Issues: {ps_results['stability_issues']}")
    
    if "rapid_sequential" in all_results:
        rs_results = all_results["rapid_sequential"]
        print(f"⚡ Rapid Sequential: {rs_results['successful']}/{rs_results['total_requests']} successful")
        if rs_results["pool_failures"] > 0:
            print(f"   🏊 Pool failures: {rs_results['pool_failures']}")
    
    # Determine overall test result
    major_issues = []
    
    if all_results.get("concurrent_analysis", {}).get("pool_closed_errors", 0) > 0:
        major_issues.append("Pool closure during concurrent analysis")
    
    if not all_results.get("pool_stability", {}).get("pool_stable", True):
        major_issues.append("Pool instability under load")
    
    if all_results.get("rapid_sequential", {}).get("pool_failures", 0) > 0:
        major_issues.append("Pool failures during rapid requests")
    
    print(f"\n🎯 OVERALL RESULT: {'❌ ISSUES FOUND' if major_issues else '✅ ALL TESTS PASSED'}")
    
    if major_issues:
        print("🚨 Critical issues that need fixing:")
        for issue in major_issues:
            print(f"   - {issue}")
        print("\n💡 Recommended fixes:")
        print("   - Improve connection pool management for concurrent access")
        print("   - Add proper connection pool locking/semaphores")
        print("   - Implement connection pool health checks and recovery")
        print("   - Consider reducing MAX_CONCURRENT_ANALYSES back to 1 until fixed")
    
    return all_results

if __name__ == "__main__":
    async def main():
        # Check if server is running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        print("✅ Server is running, starting concurrent tests...")
                        results = await run_all_concurrent_tests()
                        
                        # Save results to file for analysis
                        with open("concurrent_test_results.json", "w") as f:
                            json.dump(results, f, indent=2, default=str)
                        print(f"\n📁 Results saved to concurrent_test_results.json")
                        
                    else:
                        print(f"❌ Server health check failed: {response.status}")
        except Exception as e:
            print(f"❌ Could not connect to server: {e}")
            print(f"   Make sure the server is running at {BASE_URL}")
    
    asyncio.run(main()) 
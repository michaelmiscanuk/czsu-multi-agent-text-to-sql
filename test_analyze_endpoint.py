#!/usr/bin/env python3
"""
Comprehensive Test for /analyze Endpoint
Tests the POST /analyze endpoint to debug 500 Internal Server Error issues.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import uuid
import base64
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000"  # Change this to your server URL
PRODUCTION_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 600  # 10 minutes timeout for analysis

def create_mock_jwt_token(email="test.analyze@gmail.com", expire_minutes=60):
    """Create a properly formatted mock JWT token for testing."""
    now = datetime.utcnow()
    payload = {
        "email": email,
        "aud": "mock_audience",
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
        "iat": int(now.timestamp()),
        "iss": "mock_issuer",
        "sub": "test_user_123",
        "name": "Test User",
        "email_verified": True
    }
    # Create a proper JWT structure
    header = base64.urlsafe_b64encode(json.dumps({"typ": "JWT", "alg": "HS256"}).encode()).decode().rstrip('=')
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    signature = base64.urlsafe_b64encode(b'mock_signature_for_testing').decode().rstrip('=')
    
    return f"{header}.{payload_b64}.{signature}"

async def test_server_health(base_url: str) -> bool:
    """Test if the server is running and healthy."""
    print(f"🔍 Testing server health at {base_url}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Test basic health
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Server health: {health_data}")
                    
                    # Test database health
                    try:
                        async with session.get(f"{base_url}/health/database", timeout=aiohttp.ClientTimeout(total=15)) as db_response:
                            if db_response.status == 200:
                                db_health = await db_response.json()
                                print(f"✅ Database health: {db_health}")
                            else:
                                print(f"⚠️ Database health check failed: {db_response.status}")
                    except Exception as e:
                        print(f"⚠️ Database health check error: {e}")
                    
                    return True
                else:
                    print(f"❌ Server health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Server health check error: {e}")
        return False

async def test_analyze_endpoint_basic(base_url: str) -> Dict[str, Any]:
    """Test basic functionality of the /analyze endpoint."""
    print(f"\n🔍 Testing basic /analyze endpoint functionality")
    
    test_cases = [
        {
            "name": "Simple query",
            "prompt": "What was the total population of Czech Republic in 2022?",
            "thread_id": f"test-basic-{uuid.uuid4().hex[:8]}"
        },
        {
            "name": "Trade statistics query", 
            "prompt": "What were the exports to Germany in 2022?",
            "thread_id": f"test-trade-{uuid.uuid4().hex[:8]}"
        },
        {
            "name": "Housing statistics query",
            "prompt": "How many residential buildings were completed in Prague in 2022?",
            "thread_id": f"test-housing-{uuid.uuid4().hex[:8]}"
        }
    ]
    
    results = {}
    token = create_mock_jwt_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for test_case in test_cases:
                print(f"\n   🔍 Testing: {test_case['name']}")
                
                payload = {
                    "prompt": test_case["prompt"],
                    "thread_id": test_case["thread_id"]
                }
                
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{base_url}/analyze",
                        json=payload,
                        headers=headers
                    ) as response:
                        duration = time.time() - start_time
                        
                        print(f"     Response status: {response.status}")
                        print(f"     Response time: {duration:.2f}s")
                        
                        if response.status == 200:
                            response_data = await response.json()
                            
                            # Validate response structure
                            required_fields = ["prompt", "result", "thread_id", "run_id"]
                            missing_fields = [field for field in required_fields if field not in response_data]
                            
                            if missing_fields:
                                print(f"     ❌ Missing fields: {missing_fields}")
                                results[test_case["name"]] = {
                                    "status": "failed",
                                    "error": f"Missing fields: {missing_fields}",
                                    "duration": duration
                                }
                            else:
                                print(f"     ✅ Success - Run ID: {response_data.get('run_id', 'N/A')}")
                                print(f"     📊 Result preview: {response_data['result'][:100]}...")
                                
                                results[test_case["name"]] = {
                                    "status": "success",
                                    "run_id": response_data.get("run_id"),
                                    "duration": duration,
                                    "result_length": len(response_data.get("result", "")),
                                    "queries_count": len(response_data.get("queries_and_results", []))
                                }
                        else:
                            error_text = await response.text()
                            print(f"     ❌ Failed with status {response.status}")
                            print(f"     📝 Error response: {error_text}")
                            
                            results[test_case["name"]] = {
                                "status": "failed",
                                "error": f"HTTP {response.status}: {error_text}",
                                "duration": duration
                            }
                            
                except asyncio.TimeoutError:
                    duration = time.time() - start_time
                    print(f"     ⏰ Request timed out after {duration:.2f}s")
                    results[test_case["name"]] = {
                        "status": "timeout",
                        "error": "Request timeout",
                        "duration": duration
                    }
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"     ❌ Request error: {e}")
                    results[test_case["name"]] = {
                        "status": "error",
                        "error": str(e),
                        "duration": duration
                    }
                
                # Small delay between requests
                await asyncio.sleep(1)
    
    except Exception as e:
        print(f"❌ Test session error: {e}")
        results["session_error"] = {"status": "error", "error": str(e)}
    
    return results

async def test_analyze_endpoint_edge_cases(base_url: str) -> Dict[str, Any]:
    """Test edge cases and validation for the /analyze endpoint."""
    print(f"\n🔍 Testing /analyze endpoint edge cases")
    
    edge_cases = [
        {
            "name": "Empty prompt",
            "payload": {"prompt": "", "thread_id": "test-empty"},
            "expected_status": [400, 422]  # Validation error
        },
        {
            "name": "Missing prompt",
            "payload": {"thread_id": "test-missing-prompt"},
            "expected_status": [400, 422]  # Validation error
        },
        {
            "name": "Missing thread_id",
            "payload": {"prompt": "Test prompt"},
            "expected_status": [400, 422]  # Validation error
        },
        {
            "name": "Empty thread_id",
            "payload": {"prompt": "Test prompt", "thread_id": ""},
            "expected_status": [400, 422]  # Validation error
        },
        {
            "name": "Very long prompt",
            "payload": {"prompt": "A" * 15000, "thread_id": "test-long"},
            "expected_status": [400, 422]  # Validation error - exceeds max length
        },
        {
            "name": "Very long thread_id",
            "payload": {"prompt": "Test prompt", "thread_id": "A" * 200},
            "expected_status": [400, 422]  # Validation error - exceeds max length
        },
        {
            "name": "Special characters in prompt",
            "payload": {"prompt": "Test with émojis 🚀 and spëcial chars: àáâãäå", "thread_id": "test-special"},
            "expected_status": [200]  # Should work
        },
        {
            "name": "SQL injection attempt",
            "payload": {"prompt": "'; DROP TABLE users; --", "thread_id": "test-sql-injection"},
            "expected_status": [200]  # Should handle safely
        }
    ]
    
    results = {}
    token = create_mock_jwt_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            for case in edge_cases:
                print(f"\n   🔍 Testing: {case['name']}")
                
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{base_url}/analyze",
                        json=case["payload"],
                        headers=headers
                    ) as response:
                        duration = time.time() - start_time
                        
                        print(f"     Response status: {response.status}")
                        
                        if response.status in case["expected_status"]:
                            print(f"     ✅ Expected status received")
                            
                            if response.status == 200:
                                response_data = await response.json()
                                results[case["name"]] = {
                                    "status": "success", 
                                    "duration": duration,
                                    "run_id": response_data.get("run_id")
                                }
                            else:
                                error_data = await response.json()
                                results[case["name"]] = {
                                    "status": "expected_error",
                                    "duration": duration,
                                    "error": error_data
                                }
                        else:
                            error_text = await response.text()
                            print(f"     ❌ Unexpected status {response.status}")
                            print(f"     📝 Error: {error_text}")
                            
                            results[case["name"]] = {
                                "status": "unexpected_status",
                                "expected": case["expected_status"],
                                "actual": response.status,
                                "error": error_text,
                                "duration": duration
                            }
                            
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"     ❌ Request error: {e}")
                    results[case["name"]] = {
                        "status": "error",
                        "error": str(e),
                        "duration": duration
                    }
                
                await asyncio.sleep(0.5)
    
    except Exception as e:
        print(f"❌ Edge case test session error: {e}")
        results["session_error"] = {"status": "error", "error": str(e)}
    
    return results

async def test_analyze_endpoint_authentication(base_url: str) -> Dict[str, Any]:
    """Test authentication scenarios for the /analyze endpoint."""
    print(f"\n🔍 Testing /analyze endpoint authentication")
    
    auth_cases = [
        {
            "name": "No Authorization header",
            "headers": {"Content-Type": "application/json"},
            "expected_status": [401]
        },
        {
            "name": "Invalid Bearer token",
            "headers": {
                "Authorization": "Bearer invalid_token",
                "Content-Type": "application/json"
            },
            "expected_status": [401]
        },
        {
            "name": "Malformed Authorization header",
            "headers": {
                "Authorization": "NotBearer token",
                "Content-Type": "application/json"
            },
            "expected_status": [401]
        },
        {
            "name": "Expired token",
            "headers": {
                "Authorization": f"Bearer {create_mock_jwt_token(expire_minutes=-10)}",
                "Content-Type": "application/json"
            },
            "expected_status": [401]
        },
        {
            "name": "Valid token",
            "headers": {
                "Authorization": f"Bearer {create_mock_jwt_token()}",
                "Content-Type": "application/json"
            },
            "expected_status": [200, 408, 500]  # Success, timeout, or server error (not auth error)
        }
    ]
    
    results = {}
    test_payload = {
        "prompt": "Test authentication prompt",
        "thread_id": f"auth-test-{uuid.uuid4().hex[:8]}"
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            for case in auth_cases:
                print(f"\n   🔍 Testing: {case['name']}")
                
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{base_url}/analyze",
                        json=test_payload,
                        headers=case["headers"]
                    ) as response:
                        duration = time.time() - start_time
                        
                        print(f"     Response status: {response.status}")
                        
                        if response.status in case["expected_status"]:
                            print(f"     ✅ Expected status received")
                            results[case["name"]] = {
                                "status": "success",
                                "response_status": response.status,
                                "duration": duration
                            }
                        else:
                            error_text = await response.text()
                            print(f"     ❌ Unexpected status {response.status}")
                            results[case["name"]] = {
                                "status": "unexpected_status",
                                "expected": case["expected_status"],
                                "actual": response.status,
                                "error": error_text,
                                "duration": duration
                            }
                            
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"     ❌ Request error: {e}")
                    results[case["name"]] = {
                        "status": "error",
                        "error": str(e),
                        "duration": duration
                    }
                
                await asyncio.sleep(0.5)
    
    except Exception as e:
        print(f"❌ Authentication test session error: {e}")
        results["session_error"] = {"status": "error", "error": str(e)}
    
    return results

async def test_analyze_endpoint_stress(base_url: str, num_requests: int = 3) -> Dict[str, Any]:
    """Test stress scenarios with multiple concurrent requests."""
    print(f"\n🔍 Testing /analyze endpoint stress ({num_requests} concurrent requests)")
    
    results = {}
    token = create_mock_jwt_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    async def single_stress_request(session, request_id):
        payload = {
            "prompt": f"Stress test query {request_id}: What was the population in 2022?",
            "thread_id": f"stress-{request_id}-{uuid.uuid4().hex[:8]}"
        }
        
        start_time = time.time()
        
        try:
            async with session.post(
                f"{base_url}/analyze",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes per request
            ) as response:
                duration = time.time() - start_time
                
                result = {
                    "request_id": request_id,
                    "status_code": response.status,
                    "duration": duration,
                    "success": response.status == 200
                }
                
                if response.status == 200:
                    response_data = await response.json()
                    result["run_id"] = response_data.get("run_id")
                    result["result_length"] = len(response_data.get("result", ""))
                else:
                    result["error"] = await response.text()
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            return {
                "request_id": request_id,
                "status_code": None,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"   🚀 Starting {num_requests} concurrent requests...")
            
            start_time = time.time()
            tasks = [single_stress_request(session, i) for i in range(num_requests)]
            stress_results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            successful = sum(1 for r in stress_results if r["success"])
            failed = len(stress_results) - successful
            avg_duration = sum(r["duration"] for r in stress_results) / len(stress_results)
            
            print(f"   📊 Stress test results:")
            print(f"     Total requests: {num_requests}")
            print(f"     Successful: {successful}")
            print(f"     Failed: {failed}")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Average duration: {avg_duration:.2f}s")
            
            results = {
                "total_requests": num_requests,
                "successful": successful,
                "failed": failed,
                "total_time": total_time,
                "average_duration": avg_duration,
                "detailed_results": stress_results
            }
    
    except Exception as e:
        print(f"❌ Stress test session error: {e}")
        results = {"status": "error", "error": str(e)}
    
    return results

def print_summary(all_results: Dict[str, Any]):
    """Print a comprehensive summary of all test results."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    for test_type, results in all_results.items():
        if test_type == "server_health":
            continue
            
        print(f"\n📋 {test_type.upper().replace('_', ' ')}")
        print("-" * 40)
        
        if isinstance(results, dict) and "status" not in results:
            # This is a collection of test results
            for test_name, test_result in results.items():
                if isinstance(test_result, dict):
                    total_tests += 1
                    status = test_result.get("status", "unknown")
                    
                    if status in ["success", "expected_error"]:
                        passed_tests += 1
                        status_icon = "✅"
                    else:
                        status_icon = "❌"
                    
                    duration = test_result.get("duration", 0)
                    print(f"  {status_icon} {test_name}: {status} ({duration:.2f}s)")
                    
                    if "error" in test_result:
                        print(f"      Error: {test_result['error']}")
        else:
            # This is a single test result
            total_tests += 1
            if results.get("successful", 0) > 0:
                passed_tests += 1
                print(f"  ✅ Stress test: {results.get('successful', 0)}/{results.get('total_requests', 0)} successful")
            else:
                print(f"  ❌ Stress test failed")
    
    print("\n" + "="*80)
    print(f"🎯 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("="*80)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The /analyze endpoint is working correctly.")
    elif passed_tests > total_tests * 0.8:
        print("⚠️ Most tests passed, but some issues detected. Check the details above.")
    else:
        print("🚨 SIGNIFICANT ISSUES DETECTED! The /analyze endpoint needs attention.")
    
    print("\n💡 DEBUGGING TIPS:")
    print("- Check server logs for detailed error messages")
    print("- Verify database connection health")
    print("- Monitor memory usage during analysis")
    print("- Check authentication token validity")
    print("- Ensure required environment variables are set")

async def main():
    """Main test function."""
    print("🚀 COMPREHENSIVE /analyze ENDPOINT TESTING")
    print("=" * 80)
    print(f"⏰ Test started at: {datetime.now().isoformat()}")
    
    # Determine which URL to test
    test_local = True
    test_production = False
    base_url = BASE_URL  # Use local variable to avoid scope issues
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--production":
            test_production = True
            test_local = False
        elif sys.argv[1] == "--both":
            test_production = True
            test_local = True
        elif sys.argv[1].startswith("http"):
            base_url = sys.argv[1]
            test_local = True
            test_production = False
    
    urls_to_test = []
    if test_local:
        urls_to_test.append(("LOCAL", base_url))
    if test_production:
        urls_to_test.append(("PRODUCTION", PRODUCTION_URL))
    
    for env_name, url in urls_to_test:
        print(f"\n🌐 TESTING {env_name} ENVIRONMENT: {url}")
        print("=" * 80)
        
        # Test server health first
        server_healthy = await test_server_health(url)
        if not server_healthy:
            print(f"❌ {env_name} server is not healthy, skipping tests")
            continue
        
        all_results = {
            "server_health": server_healthy,
            "basic_functionality": await test_analyze_endpoint_basic(url),
            "edge_cases": await test_analyze_endpoint_edge_cases(url),
            "authentication": await test_analyze_endpoint_authentication(url),
            "stress_test": await test_analyze_endpoint_stress(url, 2)  # 2 concurrent requests
        }
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyze_endpoint_test_results_{env_name.lower()}_{timestamp}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\n📁 Results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        # Print summary
        print_summary(all_results)
    
    print(f"\n⏰ All tests completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    print("Usage:")
    print("  python test_analyze_endpoint.py                    # Test local server")
    print("  python test_analyze_endpoint.py --production       # Test production server") 
    print("  python test_analyze_endpoint.py --both             # Test both servers")
    print("  python test_analyze_endpoint.py http://custom:8000 # Test custom URL")
    print()
    
    asyncio.run(main()) 
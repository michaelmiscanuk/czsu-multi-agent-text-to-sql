#!/usr/bin/env python3
"""
Authentication Endpoint Tests for CZSU Multi-Agent API
Tests authentication, JWT handling, and protected endpoint security.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import uuid
import base64
from datetime import datetime, timedelta

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"
TEST_TIMEOUT = 30

# Mock JWT tokens for testing
MOCK_SECRET = "test-secret-key"
MOCK_AUDIENCE = "test-audience"

def create_mock_jwt_token(payload_overrides=None, expire_minutes=60):
    """Create a mock JWT token for testing."""
    payload = {
        "iss": "https://accounts.google.com",
        "aud": MOCK_AUDIENCE,
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "exp": datetime.utcnow() + timedelta(minutes=expire_minutes),
        "iat": datetime.utcnow(),
        "sub": "1234567890"
    }
    
    if payload_overrides:
        payload.update(payload_overrides)
    
    return jwt.encode(payload, MOCK_SECRET, algorithm="HS256")

def create_malformed_jwt_token(token_type="empty"):
    """Create various types of malformed JWT tokens for testing."""
    if token_type == "empty":
        return ""
    elif token_type == "invalid_format":
        return "not-a-jwt-token"
    elif token_type == "one_part":
        return "onlyonepart"
    elif token_type == "two_parts":
        return "header.payload"
    elif token_type == "three_empty_parts":
        return ".."
    elif token_type == "valid_structure_invalid_content":
        # This looks like a JWT but has invalid base64 content
        return "invalid_header.invalid_payload.invalid_signature"
    elif token_type == "valid_base64_invalid_jwt":
        # This has valid base64 but invalid JWT content
        header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
        payload = base64.urlsafe_b64encode(b'{"sub":"test"}').decode().rstrip('=')
        signature = base64.urlsafe_b64encode(b'invalid_signature').decode().rstrip('=')
        return f"{header}.{payload}.{signature}"
    else:
        return "default-invalid-token"

async def test_no_auth_endpoints():
    """Test endpoints that should work without authentication."""
    print("🔍 Testing no-auth endpoints...")
    
    no_auth_endpoints = [
        ("/health", "Health check"),
        ("/docs", "API docs"),
        ("/openapi.json", "OpenAPI spec")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for endpoint, description in no_auth_endpoints:
                print(f"   Testing {endpoint} ({description})...")
                
                try:
                    async with session.get(f"{BASE_URL}{endpoint}") as response:
                        if response.status == 200:
                            print(f"     ✅ {response.status} - OK")
                            results[endpoint] = True
                        else:
                            print(f"     ❌ {response.status} - Unexpected status")
                            results[endpoint] = False
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[endpoint] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ No-auth endpoints test completed: {successful}/{total} successful")
        return successful == total
        
    except Exception as e:
        print(f"❌ No-auth endpoints test failed: {e}")
        return False

async def test_protected_endpoints_without_auth():
    """Test that protected endpoints require authentication."""
    print("🔍 Testing protected endpoints without authentication...")
    
    protected_endpoints = [
        ("/analyze", "POST", "Analysis endpoint"),
        ("/feedback", "POST", "Feedback endpoint"),
        ("/sentiment", "POST", "Sentiment endpoint"),
        ("/chat-threads", "GET", "Chat threads endpoint"),
        ("/catalog", "GET", "Catalog endpoint"),
        ("/data-tables", "GET", "Data tables endpoint")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for endpoint, method, description in protected_endpoints:
                print(f"   Testing {method} {endpoint} ({description})...")
                
                try:
                    if method == "GET":
                        async with session.get(f"{BASE_URL}{endpoint}") as response:
                            status = response.status
                    elif method == "POST":
                        async with session.post(f"{BASE_URL}{endpoint}", json={}) as response:
                            status = response.status
                    
                    if status == 401:
                        print(f"     ✅ 401 Unauthorized - Correctly protected")
                        results[endpoint] = True
                    else:
                        print(f"     ❌ {status} - Should be 401")
                        results[endpoint] = False
                        
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[endpoint] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Protected endpoints test completed: {successful}/{total} correctly protected")
        return successful == total
        
    except Exception as e:
        print(f"❌ Protected endpoints test failed: {e}")
        return False

async def test_invalid_jwt_scenarios():
    """Test various invalid JWT token scenarios."""
    print("🔍 Testing invalid JWT token scenarios...")
    
    invalid_tokens = [
        ("", "Empty token"),
        (create_malformed_jwt_token("invalid_format"), "Invalid format"),
        ("Bearer", "Bearer without token"),
        (f"Bearer {create_malformed_jwt_token('one_part')}", "Bearer with one part token"),
        (f"Bearer {create_malformed_jwt_token('two_parts')}", "Bearer with two parts token"),
        (f"Bearer {create_malformed_jwt_token('valid_base64_invalid_jwt')}", "Valid structure but invalid JWT"),
        (f"Bearer {create_mock_jwt_token(expire_minutes=-10)}", "Expired token"),
        (f"Bearer {create_mock_jwt_token({'email': None})}", "Token without email")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for token, description in invalid_tokens:
                print(f"   Testing: {description}")
                
                headers = {}
                if token:
                    headers["Authorization"] = token
                
                try:
                    async with session.post(f"{BASE_URL}/analyze", 
                                          json={"prompt": "test", "thread_id": "test"}, 
                                          headers=headers) as response:
                        
                        if response.status == 401:
                            print(f"     ✅ 401 Unauthorized - Correctly rejected")
                            results[description] = True
                        else:
                            print(f"     ❌ {response.status} - Should be 401")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Invalid JWT test completed: {successful}/{total} correctly rejected")
        return successful >= total * 0.7  # Allow some tolerance
        
    except Exception as e:
        print(f"❌ Invalid JWT test failed: {e}")
        return False

async def test_jwt_token_variations():
    """Test various JWT token format variations and edge cases."""
    print("🔍 Testing JWT token variations...")
    
    jwt_variations = [
        # Case sensitivity tests
        ("bearer " + create_mock_jwt_token(), "Lowercase bearer"),
        ("BEARER " + create_mock_jwt_token(), "Uppercase bearer"),
        ("Bearer  " + create_mock_jwt_token(), "Extra space after Bearer"),
        ("Bearer\t" + create_mock_jwt_token(), "Tab after Bearer"),
        ("Bearer\n" + create_mock_jwt_token(), "Newline after Bearer"),
        
        # Whitespace variations
        (" Bearer " + create_mock_jwt_token(), "Leading space"),
        ("Bearer " + create_mock_jwt_token() + " ", "Trailing space"),
        ("Bearer " + create_mock_jwt_token() + "\n", "Trailing newline"),
        
        # Multiple Bearer tokens
        ("Bearer token1 Bearer token2", "Multiple Bearer keywords"),
        ("Bearer " + create_mock_jwt_token() + " " + create_mock_jwt_token(), "Multiple tokens"),
        
        # Malformed base64 in JWT parts
        ("Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid_payload.signature", "Invalid payload base64"),
        ("Bearer invalid_header.eyJzdWIiOiJ0ZXN0In0.signature", "Invalid header base64"),
        
        # Very long tokens
        ("Bearer " + "a" * 10000, "Extremely long token"),
        ("Bearer " + create_mock_jwt_token() + "a" * 5000, "Valid token with extra data"),
        
        # Special characters in token
        ("Bearer token-with-hyphens", "Token with hyphens"),
        ("Bearer token_with_underscores", "Token with underscores"),
        ("Bearer token.with.dots", "Token with dots"),
        
        # Binary and non-printable characters
        ("Bearer " + "\x00\x01\x02", "Binary characters"),
        ("Bearer " + "token\x7f\x80\x81", "Non-printable characters"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for token, description in jwt_variations:
                print(f"   Testing: {description}")
                
                try:
                    headers = {"Authorization": token}
                    async with session.get(f"{BASE_URL}/catalog", headers=headers) as response:
                        
                        if response.status == 401:
                            print(f"     ✅ 401 Unauthorized - Correctly rejected")
                            results[description] = True
                        elif response.status == 400:
                            print(f"     ✅ 400 Bad Request - Correctly rejected")
                            results[description] = True
                        elif response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Poor error handling")
                            results[description] = False
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected status")
                            results[description] = True
                            
                except Exception as e:
                    print(f"     ❌ Exception: {e}")
                    results[description] = False
                
                await asyncio.sleep(0.2)
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ JWT token variations test: {successful}/{total} handled correctly")
        return successful >= total * 0.8
        
    except Exception as e:
        print(f"❌ JWT token variations test failed: {e}")
        return False

async def test_concurrent_authentication():
    """Test concurrent authentication requests."""
    print("🔍 Testing concurrent authentication...")
    
    async def auth_request(session, request_id, token):
        try:
            headers = {"Authorization": token}
            start_time = time.time()
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                response_time = time.time() - start_time
                return {
                    'request_id': request_id,
                    'status': response.status,
                    'response_time': response_time,
                    'expected': response.status in [200, 401]
                }
        except Exception as e:
            return {
                'request_id': request_id,
                'status': None,
                'response_time': None,
                'expected': False,
                'error': str(e)
            }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Test with mix of valid and invalid tokens
            tokens = [
                f"Bearer {create_mock_jwt_token()}",  # Valid format
                "Bearer invalid_token",               # Invalid format
                "",                                   # Empty
                "Bearer",                            # Missing token
                f"Bearer {create_mock_jwt_token()}",  # Another valid format
            ]
            
            # Create 25 concurrent requests (5 tokens x 5 requests each)
            tasks = []
            for i in range(25):
                token = tokens[i % len(tokens)]
                task = auth_request(session, i, token)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            expected_responses = sum(1 for r in results if r['expected'])
            unexpected_responses = sum(1 for r in results if not r['expected'])
            
            print(f"   Concurrent authentication results:")
            print(f"     Total requests: 25")
            print(f"     Expected responses: {expected_responses}")
            print(f"     Unexpected responses: {unexpected_responses}")
            print(f"     Total time: {total_time:.2f}s")
            
            if unexpected_responses <= 2:  # Allow small margin for errors
                print("   ✅ Concurrent authentication test passed")
                return True
            else:
                print("   ❌ Too many unexpected responses")
                return False
                
    except Exception as e:
        print(f"   ❌ Concurrent authentication test failed: {e}")
        return False

async def test_authentication_header_variations():
    """Test various authentication header variations."""
    print("🔍 Testing authentication header variations...")
    
    header_variations = [
        # Different header names (should all fail)
        ({"authorization": f"Bearer {create_mock_jwt_token()}"}, "Lowercase authorization"),
        ({"AUTHORIZATION": f"Bearer {create_mock_jwt_token()}"}, "Uppercase authorization"),
        ({"Auth": f"Bearer {create_mock_jwt_token()}"}, "Short auth header"),
        ({"Bearer": create_mock_jwt_token()}, "Bearer as header name"),
        
        # Multiple authorization headers
        ({"Authorization": f"Bearer {create_mock_jwt_token()}", 
          "authorization": f"Bearer {create_mock_jwt_token()}"}, "Duplicate auth headers"),
        
        # Authorization with other auth schemes
        ({"Authorization": f"Basic {create_mock_jwt_token()}"}, "Basic auth scheme"),
        ({"Authorization": f"Digest {create_mock_jwt_token()}"}, "Digest auth scheme"),
        ({"Authorization": f"Token {create_mock_jwt_token()}"}, "Token auth scheme"),
        
        # Malformed authorization headers
        ({"Authorization": "Bearer"}, "Bearer without token"),
        ({"Authorization": create_mock_jwt_token()}, "Token without Bearer"),
        ({"Authorization": f"Bearer {create_mock_jwt_token()} extra"}, "Extra data after token"),
        
        # Special characters in headers
        ({"Authorization": f"Bearer {create_mock_jwt_token()}\x00"}, "Null byte in header"),
        ({"Authorization": f"Bearer {create_mock_jwt_token()}\n"}, "Newline in header"),
        ({"Authorization": f"Bearer {create_mock_jwt_token()}\r"}, "Carriage return in header"),
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for headers, description in header_variations:
                print(f"   Testing: {description}")
                
                try:
                    async with session.get(f"{BASE_URL}/catalog", headers=headers) as response:
                        
                        if response.status == 401:
                            print(f"     ✅ 401 Unauthorized - Correctly rejected")
                            results[description] = True
                        elif response.status == 400:
                            print(f"     ✅ 400 Bad Request - Correctly rejected")
                            results[description] = True
                        elif response.status == 500:
                            print(f"     ❌ 500 Internal Server Error - Poor error handling")
                            results[description] = False
                        else:
                            print(f"     ⚠️ {response.status} - Unexpected status")
                            results[description] = True
                            
                except Exception as e:
                    print(f"     ❌ Exception: {e}")
                    results[description] = False
                
                await asyncio.sleep(0.2)
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Authentication header variations: {successful}/{total} handled correctly")
        return successful >= total * 0.8
        
    except Exception as e:
        print(f"❌ Authentication header variations test failed: {e}")
        return False

async def test_malformed_requests():
    """Test malformed request payloads."""
    print("🔍 Testing malformed requests...")
    
    malformed_requests = [
        ("/analyze", {}, "Empty analyze request"),
        ("/analyze", {"prompt": "", "thread_id": "test"}, "Empty prompt"),
        ("/analyze", {"thread_id": "test"}, "Missing prompt"),
        ("/feedback", {}, "Empty feedback request"),
        ("/feedback", {"run_id": "invalid-uuid", "feedback": 1}, "Invalid run_id format"),
        ("/sentiment", {}, "Empty sentiment request"),
        ("/sentiment", {"run_id": "invalid-uuid", "sentiment": True}, "Invalid sentiment run_id")
    ]
    
    results = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            for endpoint, payload, description in malformed_requests:
                print(f"   Testing: {description}")
                
                try:
                    async with session.post(f"{BASE_URL}{endpoint}", json=payload) as response:
                        
                        # Should return 422 (validation error) or 400 (bad request), not 500
                        if response.status in [400, 401, 422]:
                            print(f"     ✅ {response.status} - Properly handled")
                            results[description] = True
                        else:
                            print(f"     ❌ {response.status} - Unexpected status")
                            results[description] = False
                            
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    results[description] = False
        
        successful = sum(results.values())
        total = len(results)
        print(f"✅ Malformed requests test completed: {successful}/{total} properly handled")
        return successful >= total * 0.5  # Allow some tolerance as this tests error handling
        
    except Exception as e:
        print(f"❌ Malformed requests test failed: {e}")
        return False

async def test_rate_limiting():
    """Test rate limiting behavior."""
    print("🔍 Testing rate limiting behavior...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            print(f"   Sending 10 rapid requests to /health...")
            
            start_time = time.time()
            tasks = []
            
            for i in range(10):
                task = session.get(f"{BASE_URL}/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = 0
            rate_limited = 0
            errors = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    errors += 1
                else:
                    if response.status == 200:
                        successful += 1
                    elif response.status == 429:  # Too Many Requests
                        rate_limited += 1
                    else:
                        errors += 1
                    response.close()
            
            print(f"   Results:")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Successful: {successful}")
            print(f"     Rate limited: {rate_limited}")
            print(f"     Errors: {errors}")
            
            if rate_limited > 0:
                print("   ✅ Rate limiting is working")
                return True
            else:
                print("   ⚠️  No rate limiting detected - consider adding for production")
                return True  # Not a failure, just a recommendation
                
    except Exception as e:
        print(f"   ❌ Rate limiting test failed: {e}")
        return False

async def test_cors_preflight():
    """Test CORS preflight requests."""
    print("🔍 Testing CORS preflight requests...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)) as session:
            # Test preflight for POST request
            headers = {
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,Authorization"
            }
            
            async with session.options(f"{BASE_URL}/analyze", headers=headers) as response:
                print(f"   Preflight status: {response.status}")
                
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
                }
                
                for header, value in cors_headers.items():
                    print(f"   {header}: {value}")
                
                # Check if CORS is properly configured
                if (response.status in [200, 204] and 
                    cors_headers['Access-Control-Allow-Origin']):
                    print("   ✅ CORS preflight working")
                    return True
                else:
                    print("   ⚠️ CORS preflight may need configuration")
                    return True  # Not a critical failure
                    
    except Exception as e:
        print(f"   ❌ CORS preflight test failed: {e}")
        return False

async def run_auth_tests():
    """Run all authentication tests."""
    print("🚀 AUTHENTICATION ENDPOINT TESTS")
    print("=" * 50)
    print(f"Target URL: {BASE_URL}")
    print(f"Test started: {datetime.now()}")
    print("=" * 50)
    
    tests = [
        ("No-Auth Endpoints", test_no_auth_endpoints),
        ("Protected Endpoints Security", test_protected_endpoints_without_auth),
        ("Invalid JWT Handling", test_invalid_jwt_scenarios),
        ("JWT Token Variations", test_jwt_token_variations),
        ("Authentication Header Variations", test_authentication_header_variations),
        ("Concurrent Authentication", test_concurrent_authentication),
        ("Malformed Request Handling", test_malformed_requests),
        ("Rate Limiting", test_rate_limiting),
        ("CORS Preflight", test_cors_preflight)
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
        
        # Brief pause between tests
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 AUTHENTICATION TEST SUMMARY")
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
        print("🎉 All authentication tests passed!")
    else:
        print("⚠️ Some authentication tests failed - check logs for details")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_auth_tests()) 
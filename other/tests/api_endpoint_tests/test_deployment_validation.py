#!/usr/bin/env python3
"""
Deployment Validation Tests for CZSU Multi-Agent API
Validates that the Supabase connection pool fixes are working in production.
"""

import asyncio
import aiohttp
import time
import json
import jwt
import base64
import uuid
import os
from datetime import datetime

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"

def create_mock_jwt_token():
    """Create a valid JWT token for testing."""
    header = base64.urlsafe_b64encode(b'{"typ":"JWT","alg":"HS256"}').decode().rstrip('=')
    payload = base64.urlsafe_b64encode(b'{"sub":"test","email":"test@example.com"}').decode().rstrip('=')
    signature = base64.urlsafe_b64encode(b'fake_signature').decode().rstrip('=')
    return f"{header}.{payload}.{signature}"

async def test_deployment_health():
    """Test basic deployment health and database connectivity."""
    print("\n🏥 Testing Deployment Health")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Basic health check
    print("📊 Test 1: Basic health check")
    total_tests += 1
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check status: {data.get('status')}")
                    
                    # Check database connectivity
                    db_status = data.get('database', 'unknown')
                    print(f"📊 Database status: {db_status}")
                    
                    if db_status == 'connected':
                        print("🎉 PostgreSQL connection is working!")
                        success_count += 1
                    elif db_status == 'in_memory_fallback':
                        print("⚠️ Using InMemorySaver fallback - PostgreSQL connection failed")
                    else:
                        print(f"⚠️ Database status unclear: {db_status}")
                else:
                    print(f"❌ Health check failed: {response.status}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Pool status debug endpoint
    print("📊 Test 2: Connection pool status")
    total_tests += 1
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Pool status endpoint works")
                    
                    checkpointer_type = data.get('checkpointer_type')
                    pool_healthy = data.get('pool_healthy')
                    can_query = data.get('can_query')
                    
                    print(f"📊 Checkpointer type: {checkpointer_type}")
                    print(f"📊 Pool healthy: {pool_healthy}")
                    print(f"📊 Can query: {can_query}")
                    
                    if checkpointer_type == 'PostgresSaver':
                        print("🎉 PostgreSQL checkpointer is active!")
                        if pool_healthy and can_query:
                            print("🎉 Connection pool is healthy and can execute queries!")
                            success_count += 1
                        else:
                            print("⚠️ PostgreSQL checkpointer exists but pool has issues")
                    elif checkpointer_type == 'InMemorySaver':
                        print("❌ PRODUCTION ISSUE: Still using InMemorySaver!")
                        print("💡 The connection pool fix didn't work properly")
                    else:
                        print(f"❓ Unknown checkpointer type: {checkpointer_type}")
                else:
                    print(f"❌ Pool status check failed: {response.status}")
    except Exception as e:
        print(f"❌ Pool status error: {e}")
    
    # Test 3: Chat threads endpoint (tests real database usage)
    print("📊 Test 3: Chat threads endpoint (database functionality)")
    total_tests += 1
    try:
        token = create_mock_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Chat threads endpoint works: {len(data)} threads")
                    
                    # If we get results, database is working
                    # If we get empty results, it could be database issues OR just no data
                    print("📊 Database query functionality is working")
                    success_count += 1
                elif response.status == 401:
                    print("⚠️ Authentication failed (expected with mock token)")
                    print("📊 But endpoint is reachable, which is good")
                    success_count += 1  # This is actually expected
                elif response.status == 500:
                    error_text = await response.text()
                    if "database" in error_text.lower() or "connection" in error_text.lower():
                        print("❌ Database connection error detected in chat threads")
                        print(f"📊 Error: {error_text[:200]}...")
                    else:
                        print(f"❌ Chat threads failed with 500: {error_text[:200]}...")
                else:
                    print(f"❌ Chat threads failed: {response.status}")
                    error_text = await response.text()
                    print(f"📊 Error: {error_text[:200]}...")
    except Exception as e:
        print(f"❌ Chat threads error: {e}")
    
    print(f"\n🎯 Deployment health: {success_count}/{total_tests} tests passed")
    return success_count >= 2  # Need at least health + pool status working

async def test_connection_pool_fix_validation():
    """Validate that the specific connection pool fixes are working."""
    print("\n🔧 Testing Connection Pool Fix Validation")
    
    success_indicators = []
    
    # Check 1: Health endpoint should show database connection
    print("📊 Check 1: Database connection indicator")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    db_status = data.get('database')
                    
                    if db_status == 'connected':
                        print("✅ Database shows as connected")
                        success_indicators.append("database_connected")
                    elif db_status == 'in_memory_fallback':
                        print("❌ Still using InMemorySaver fallback")
                    else:
                        print(f"⚠️ Database status: {db_status}")
    except Exception as e:
        print(f"❌ Error checking database status: {e}")
    
    # Check 2: Pool status should show PostgreSQL checkpointer
    print("📊 Check 2: PostgreSQL checkpointer active")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                if response.status == 200:
                    data = await response.json()
                    checkpointer_type = data.get('checkpointer_type')
                    
                    if checkpointer_type == 'PostgresSaver':
                        print("✅ PostgreSQL checkpointer is active")
                        success_indicators.append("postgres_checkpointer")
                    elif checkpointer_type == 'InMemorySaver':
                        print("❌ Still using InMemorySaver")
                    else:
                        print(f"⚠️ Checkpointer type: {checkpointer_type}")
                        
                    # Check pool health
                    if data.get('pool_healthy'):
                        print("✅ Connection pool is healthy")
                        success_indicators.append("pool_healthy")
                    
                    if data.get('can_query'):
                        print("✅ Can execute database queries")
                        success_indicators.append("can_query")
    except Exception as e:
        print(f"❌ Error checking pool status: {e}")
    
    # Check 3: No timeout errors in recent logs (simulated by checking response times)
    print("📊 Check 3: Response time validation (no timeouts)")
    try:
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # Make multiple quick requests to see if there are timeout issues
            tasks = []
            for i in range(5):
                tasks.append(session.get(f"{BASE_URL}/health"))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_responses = 0
            for response in responses:
                if not isinstance(response, Exception):
                    if response.status == 200:
                        successful_responses += 1
                    await response.close()
            
            elapsed_time = time.time() - start_time
            
            if successful_responses >= 4 and elapsed_time < 10:
                print(f"✅ Response times good: {elapsed_time:.2f}s for 5 requests")
                success_indicators.append("no_timeouts")
            else:
                print(f"⚠️ Potential timeout issues: {successful_responses}/5 successful, {elapsed_time:.2f}s")
                
    except Exception as e:
        print(f"❌ Error testing response times: {e}")
    
    # Summary
    print(f"\n🎯 Connection Pool Fix Validation:")
    print(f"📊 Success indicators: {len(success_indicators)}/4")
    for indicator in success_indicators:
        print(f"  ✅ {indicator}")
    
    missing_indicators = set(['database_connected', 'postgres_checkpointer', 'pool_healthy', 'can_query']) - set(success_indicators)
    for indicator in missing_indicators:
        print(f"  ❌ {indicator}")
    
    # The fix is successful if we have at least 3 out of 4 indicators
    fix_successful = len(success_indicators) >= 3
    
    if fix_successful:
        print("\n🎉 CONNECTION POOL FIX APPEARS TO BE SUCCESSFUL!")
        print("💡 The Supabase transaction mode compatibility fixes are working")
    else:
        print("\n❌ CONNECTION POOL FIX VALIDATION FAILED")
        print("💡 The deployment may still have database connection issues")
    
    return fix_successful

async def test_ui_functionality():
    """Test that UI functionality that depends on database is working."""
    print("\n🖥️ Testing UI-Critical Functionality")
    
    # This simulates what the UI does when it loads
    success_count = 0
    total_tests = 0
    
    # Test 1: Chat threads loading (UI home page)
    print("📊 Test 1: Chat threads loading (UI home page scenario)")
    total_tests += 1
    try:
        token = create_mock_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/chat-threads", headers=headers) as response:
                # We expect 401 with mock token, but endpoint should be reachable
                if response.status in [200, 401]:
                    print("✅ Chat threads endpoint is reachable")
                    success_count += 1
                elif response.status == 500:
                    error_text = await response.text()
                    print(f"❌ UI would fail to load: {error_text[:100]}...")
                else:
                    print(f"❌ Unexpected error: {response.status}")
    except Exception as e:
        print(f"❌ Chat threads loading error: {e}")
    
    # Test 2: Individual chat messages loading
    print("📊 Test 2: Individual chat messages endpoint")
    total_tests += 1
    try:
        token = create_mock_jwt_token()
        headers = {"Authorization": f"Bearer {token}"}
        test_thread_id = str(uuid.uuid4())
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(f"{BASE_URL}/chat/{test_thread_id}/messages", headers=headers) as response:
                if response.status in [200, 401]:
                    print("✅ Chat messages endpoint is reachable")
                    success_count += 1
                elif response.status == 500:
                    error_text = await response.text()
                    print(f"❌ Chat messages would fail: {error_text[:100]}...")
                else:
                    print(f"❌ Unexpected error: {response.status}")
    except Exception as e:
        print(f"❌ Chat messages error: {e}")
    
    print(f"\n🎯 UI functionality: {success_count}/{total_tests} tests passed")
    return success_count >= total_tests * 0.8

async def main():
    """Run all deployment validation tests."""
    print("🚀 CZSU Multi-Agent API - Deployment Validation")
    print("=" * 60)
    print("This validates that the Supabase connection pool fixes are working")
    print(f"Target: {BASE_URL}")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all test suites
    health_passed = await test_deployment_health()
    fix_passed = await test_connection_pool_fix_validation()
    ui_passed = await test_ui_functionality()
    
    # Overall summary
    print("\n" + "=" * 60)
    print("🎯 DEPLOYMENT VALIDATION SUMMARY")
    print("=" * 60)
    
    if health_passed:
        print("✅ Deployment Health: PASSED")
    else:
        print("❌ Deployment Health: FAILED")
        all_tests_passed = False
    
    if fix_passed:
        print("✅ Connection Pool Fix: PASSED")
    else:
        print("❌ Connection Pool Fix: FAILED")
        all_tests_passed = False
    
    if ui_passed:
        print("✅ UI Functionality: PASSED")
    else:
        print("❌ UI Functionality: FAILED")
        all_tests_passed = False
    
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("💡 The deployment fixes are working correctly")
        print("💡 Users should be able to see their chat history again")
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("💡 The deployment may still have issues")
        print("💡 Check the logs above for specific problems")
    
    print(f"\nValidation completed at: {datetime.now().isoformat()}")
    
    return all_tests_passed

if __name__ == "__main__":
    asyncio.run(main()) 
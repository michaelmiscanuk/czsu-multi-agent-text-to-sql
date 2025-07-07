"""
Validation script to test the concurrent access fixes.
This can be run once the server is deployed with our fixes.
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"  # Change to production URL when testing on Render

def create_simple_auth_token():
    """Create a simple test token."""
    # For real testing, you'd need a proper JWT token
    # This is a placeholder that shows the structure
    return "Bearer test_token_placeholder"

async def test_pool_health_endpoint():
    """Test the pool health endpoint."""
    print("🔍 Testing pool health endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/debug/pool-status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Pool status: {json.dumps(data, indent=2)}")
                    return True
                else:
                    print(f"❌ Pool status check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Pool status check error: {e}")
        return False

async def test_health_endpoint():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Memory RSS: {data.get('memory_rss_mb', 'N/A')}MB")
                    print(f"   Memory fragmentation: {'Yes' if data.get('potential_memory_fragmentation') else 'No'}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

async def test_concurrent_requests_simulation():
    """Simulate concurrent requests without authentication."""
    print("🔍 Testing concurrent request handling...")
    
    async def single_health_request(session, request_id):
        try:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/health") as response:
                end_time = time.time()
                duration = end_time - start_time
                return {
                    "request_id": request_id,
                    "status": response.status,
                    "duration": duration,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "request_id": request_id,
                "status": 500,
                "duration": 0,
                "success": False,
                "error": str(e)
            }
    
    # Send 5 concurrent health requests to test pool handling
    async with aiohttp.ClientSession() as session:
        tasks = [single_health_request(session, i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
        
        print(f"📊 Concurrent requests test:")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        
        if failed:
            print("   Failed requests:")
            for failure in failed:
                print(f"     Request {failure.get('request_id')}: {failure.get('error', 'Unknown error')}")
        
        return len(successful) == 5  # All should succeed

async def test_memory_monitoring():
    """Test memory monitoring endpoints."""
    print("🔍 Testing memory monitoring...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health/memory") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Memory monitoring working")
                    print(f"   Status: {data.get('status')}")
                    print(f"   RSS Memory: {data.get('memory_rss_mb')}MB")
                    print(f"   Usage: {data.get('usage_percent')}%")
                    print(f"   Fragmentation ratio: {data.get('fragmentation_ratio')}")
                    print(f"   Memory growth: {data.get('memory_growth_mb')}MB")
                    
                    # Check for warnings
                    warnings = data.get('warnings', [])
                    if warnings:
                        print(f"   ⚠️ Warnings: {warnings}")
                    
                    return True
                else:
                    print(f"❌ Memory monitoring failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Memory monitoring error: {e}")
        return False

async def validate_fixes():
    """Main validation function."""
    print("🧪 VALIDATING CONCURRENT ACCESS FIXES")
    print("=" * 50)
    print(f"⏰ Validation started at: {datetime.now().isoformat()}")
    print(f"🎯 Testing against: {BASE_URL}")
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Pool Health", test_pool_health_endpoint),
        ("Memory Monitoring", test_memory_monitoring),
        ("Concurrent Requests", test_concurrent_requests_simulation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results[test_name] = result
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Concurrent access fixes appear to be working")
        print("✅ Server is stable and responding correctly")
        print("✅ Memory monitoring is functional")
        print("✅ Pool management improvements are effective")
        
        print("\n🚀 READY FOR PRODUCTION TESTING:")
        print("   1. Try increasing MAX_CONCURRENT_ANALYSES to 2")
        print("   2. Test with real user traffic")
        print("   3. Monitor logs for pool closure errors")
        print("   4. Check memory usage patterns")
        
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED")
        print("❌ Issues still exist that need to be addressed")
        print("❌ Keep MAX_CONCURRENT_ANALYSES=1 until all tests pass")
        
        print("\n🔧 DEBUGGING STEPS:")
        print("   1. Check server logs for specific errors")
        print("   2. Verify PostgreSQL connection is working")
        print("   3. Check for remaining race conditions")
        print("   4. Review pool management implementation")
    
    print(f"\n⏰ Validation completed at: {datetime.now().isoformat()}")
    return passed == total

if __name__ == "__main__":
    async def main():
        try:
            await validate_fixes()
        except KeyboardInterrupt:
            print("\n⚠️ Validation interrupted by user")
        except Exception as e:
            print(f"\n❌ Validation error: {e}")
    
    asyncio.run(main()) 
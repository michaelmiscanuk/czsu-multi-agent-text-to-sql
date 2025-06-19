#!/usr/bin/env python3
"""
Comprehensive API Endpoint Test Runner
Runs all endpoint test suites and provides detailed reporting.
"""

import asyncio
import time
import sys
from datetime import datetime

# Import all test modules
from test_health_endpoint import run_health_tests
from test_auth_endpoints import run_auth_tests
from test_data_endpoints import run_data_tests

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"

async def run_comprehensive_tests():
    """Run all test suites and provide comprehensive reporting."""
    
    print("🚀 CZSU MULTI-AGENT API COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Target URL: {BASE_URL}")
    print(f"Test suite started: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print("=" * 80)
    
    # Define test suites
    test_suites = [
        ("Health Endpoint Tests", run_health_tests, "Tests basic health monitoring and service status"),
        ("Authentication Tests", run_auth_tests, "Tests JWT authentication and security measures"),
        ("Data Endpoint Tests", run_data_tests, "Tests data access and catalog functionality")
    ]
    
    overall_start_time = time.time()
    suite_results = {}
    
    # Run each test suite
    for suite_name, test_func, description in test_suites:
        print(f"\n{'='*20} {suite_name.upper()} {'='*20}")
        print(f"Description: {description}")
        print(f"Started: {datetime.now()}")
        print("-" * 80)
        
        suite_start_time = time.time()
        
        try:
            result = await test_func()
            suite_time = time.time() - suite_start_time
            
            suite_results[suite_name] = {
                'passed': result,
                'time': suite_time,
                'error': None
            }
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{suite_name} Result: {status} ({suite_time:.2f}s)")
            
        except Exception as e:
            suite_time = time.time() - suite_start_time
            suite_results[suite_name] = {
                'passed': False,
                'time': suite_time,
                'error': str(e)
            }
            
            print(f"\n{suite_name} Result: ❌ ERROR ({suite_time:.2f}s)")
            print(f"Error: {e}")
    
    total_time = time.time() - overall_start_time
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)
    
    passed_suites = sum(1 for r in suite_results.values() if r['passed'])
    total_suites = len(suite_results)
    success_rate = (passed_suites / total_suites) * 100 if total_suites > 0 else 0
    
    print(f"Overall Results:")
    print(f"  • Test Suites Passed: {passed_suites}/{total_suites}")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Total Execution Time: {total_time:.2f}s")
    print(f"  • Average Time per Suite: {total_time/total_suites:.2f}s")
    
    print(f"\nDetailed Results:")
    for suite_name, result in suite_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        error_info = f" (Error: {result['error']})" if result['error'] else ""
        print(f"  {status} {suite_name}: {result['time']:.2f}s{error_info}")
    
    # Service health assessment
    print(f"\n🏥 SERVICE HEALTH ASSESSMENT:")
    
    health_passed = suite_results.get("Health Endpoint Tests", {}).get('passed', False)
    auth_passed = suite_results.get("Authentication Tests", {}).get('passed', False)
    data_passed = suite_results.get("Data Endpoint Tests", {}).get('passed', False)
    
    if health_passed:
        print("  ✅ Service is responding and healthy")
    else:
        print("  ❌ Service health issues detected")
    
    if auth_passed:
        print("  ✅ Authentication and security working properly")
    else:
        print("  ⚠️  Authentication issues detected")
    
    if data_passed:
        print("  ✅ Data endpoints functioning correctly")
    else:
        print("  ⚠️  Data access issues detected")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if passed_suites == total_suites:
        print("  🎉 All tests passed! Your API is functioning excellently.")
        print("  📈 Consider adding performance monitoring for production.")
        print("  🔒 Ensure proper rate limiting is in place for production traffic.")
    else:
        print("  🔧 Some tests failed - review the detailed results above.")
        
        if not health_passed:
            print("  🚨 Priority: Fix health endpoint issues first")
        
        if not auth_passed:
            print("  🔐 Priority: Review authentication and JWT handling")
        
        if not data_passed:
            print("  📊 Priority: Check database connectivity and data access")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    fastest_suite = min(suite_results.items(), key=lambda x: x[1]['time'])
    slowest_suite = max(suite_results.items(), key=lambda x: x[1]['time'])
    
    print(f"  ⚡ Fastest Suite: {fastest_suite[0]} ({fastest_suite[1]['time']:.2f}s)")
    print(f"  🐌 Slowest Suite: {slowest_suite[0]} ({slowest_suite[1]['time']:.2f}s)")
    
    if total_time > 60:
        print(f"  ⚠️  Total test time is high ({total_time:.1f}s) - consider optimizing")
    else:
        print(f"  ✅ Total test time is acceptable ({total_time:.1f}s)")
    
    # Final status
    print(f"\n" + "=" * 80)
    if passed_suites == total_suites:
        print("🎉 COMPREHENSIVE TEST SUITE: ALL TESTS PASSED!")
        print("🚀 Your API is ready for production traffic!")
    else:
        print("⚠️  COMPREHENSIVE TEST SUITE: SOME ISSUES DETECTED")
        print("🔧 Please review and fix the failing tests before production deployment.")
    
    print(f"Test suite completed: {datetime.now()}")
    print("=" * 80)
    
    return passed_suites == total_suites

async def run_quick_health_check():
    """Run a quick health check to verify basic connectivity."""
    print("🏥 QUICK HEALTH CHECK")
    print("=" * 40)
    
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            start_time = time.time()
            async with session.get(f"{BASE_URL}/health") as response:
                response_time = time.time() - start_time
                
                print(f"Service URL: {BASE_URL}")
                print(f"Response Status: {response.status}")
                print(f"Response Time: {response_time:.3f}s")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Service Status: {data.get('status', 'unknown')}")
                    print(f"Memory Usage: {data.get('memory_usage_mb', 'unknown')}MB")
                    print("✅ Service is responding correctly")
                    return True
                else:
                    print("❌ Service is not responding correctly")
                    return False
                    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("🔧 Please check if the service is running and accessible")
        return False

if __name__ == "__main__":
    async def main():
        # First do a quick health check
        print("Starting comprehensive API testing...\n")
        
        health_ok = await run_quick_health_check()
        
        if not health_ok:
            print("\n⚠️  Basic connectivity failed. Continuing with full test suite anyway...")
            print("   (Some tests may fail due to connectivity issues)\n")
        else:
            print("\n✅ Basic connectivity confirmed. Proceeding with full test suite...\n")
        
        # Run comprehensive tests
        success = await run_comprehensive_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    
    asyncio.run(main()) 
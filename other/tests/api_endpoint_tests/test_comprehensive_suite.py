#!/usr/bin/env python3
"""
Comprehensive API Test Suite
Runs all test categories including stress tests and provides detailed reporting.
"""

import asyncio
import time
import sys
from datetime import datetime

# Import all test modules
from test_health_endpoint import run_health_tests
from test_auth_endpoints import run_auth_tests
from test_data_endpoints import run_data_tests
from test_database_stress import run_database_stress_tests

# Configuration
BASE_URL = "https://czsu-multi-agent-text-to-sql.onrender.com"

async def run_comprehensive_test_suite():
    """Run all test suites with comprehensive reporting."""
    
    print("🚀 CZSU MULTI-AGENT API COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Target URL: {BASE_URL}")
    print(f"Test suite started: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print("=" * 80)
    
    # Define test suites in order of importance
    test_suites = [
        ("Health Endpoint Tests", run_health_tests, "Basic service health and monitoring", "critical"),
        ("Authentication Tests", run_auth_tests, "JWT authentication and security measures", "critical"),
        ("Data Endpoint Tests", run_data_tests, "Data access and catalog functionality", "important"),
        ("Database Stress Tests", run_database_stress_tests, "Database resilience and edge cases", "performance")
    ]
    
    overall_start_time = time.time()
    suite_results = {}
    
    # Run each test suite
    for suite_name, test_func, description, priority in test_suites:
        print(f"\n{'='*20} {suite_name.upper()} {'='*20}")
        print(f"Description: {description}")
        print(f"Priority: {priority.upper()}")
        print(f"Started: {datetime.now()}")
        print("-" * 80)
        
        suite_start_time = time.time()
        
        try:
            result = await test_func()
            suite_time = time.time() - suite_start_time
            
            suite_results[suite_name] = {
                'passed': result,
                'time': suite_time,
                'error': None,
                'priority': priority
            }
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{suite_name} Result: {status} ({suite_time:.2f}s)")
            
        except Exception as e:
            suite_time = time.time() - suite_start_time
            suite_results[suite_name] = {
                'passed': False,
                'time': suite_time,
                'error': str(e),
                'priority': priority
            }
            
            print(f"\n{suite_name} Result: ❌ ERROR ({suite_time:.2f}s)")
            print(f"Error: {e}")
        
        # Pause between major test suites
        if suite_name != list(test_suites)[-1][0]:  # Not the last suite
            print(f"\n⏸️  Pausing 10 seconds before next test suite...")
            await asyncio.sleep(10)
    
    total_time = time.time() - overall_start_time
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUITE FINAL REPORT")
    print("=" * 80)
    
    # Overall statistics
    passed_suites = sum(1 for r in suite_results.values() if r['passed'])
    total_suites = len(suite_results)
    success_rate = (passed_suites / total_suites) * 100 if total_suites > 0 else 0
    
    print(f"🎯 OVERALL RESULTS:")
    print(f"  • Test Suites Passed: {passed_suites}/{total_suites}")
    print(f"  • Success Rate: {success_rate:.1f}%")
    print(f"  • Total Execution Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"  • Average Time per Suite: {total_time/total_suites:.2f}s")
    
    # Results by priority
    priority_results = {}
    for suite_name, result in suite_results.items():
        priority = result['priority']
        if priority not in priority_results:
            priority_results[priority] = {'passed': 0, 'total': 0}
        priority_results[priority]['total'] += 1
        if result['passed']:
            priority_results[priority]['passed'] += 1
    
    print(f"\n📈 RESULTS BY PRIORITY:")
    for priority in ['critical', 'important', 'performance']:
        if priority in priority_results:
            stats = priority_results[priority]
            rate = (stats['passed'] / stats['total']) * 100
            status_icon = "🔴" if rate < 50 else "🟡" if rate < 80 else "🟢"
            print(f"  {status_icon} {priority.upper()}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for suite_name, result in suite_results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        priority_icon = {"critical": "🔥", "important": "⚡", "performance": "💪"}.get(result['priority'], "📊")
        error_info = f" (Error: {result['error']})" if result['error'] else ""
        print(f"  {status} {priority_icon} {suite_name}: {result['time']:.2f}s{error_info}")
    
    # Service health assessment
    print(f"\n🏥 SERVICE HEALTH ASSESSMENT:")
    
    health_passed = suite_results.get("Health Endpoint Tests", {}).get('passed', False)
    auth_passed = suite_results.get("Authentication Tests", {}).get('passed', False)
    data_passed = suite_results.get("Data Endpoint Tests", {}).get('passed', False)
    stress_passed = suite_results.get("Database Stress Tests", {}).get('passed', False)
    
    if health_passed:
        print("  ✅ Service is responding and healthy")
    else:
        print("  🔴 Service health issues detected - CRITICAL")
    
    if auth_passed:
        print("  ✅ Authentication and security working properly")
    else:
        print("  🔴 Authentication issues detected - CRITICAL")
    
    if data_passed:
        print("  ✅ Data endpoints functioning correctly")
    else:
        print("  🟡 Data access issues detected - needs attention")
    
    if stress_passed:
        print("  ✅ Database layer is robust and resilient")
    else:
        print("  🟡 Some database stress tests failed - consider optimizations")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    fastest_suite = min(suite_results.items(), key=lambda x: x[1]['time'])
    slowest_suite = max(suite_results.items(), key=lambda x: x[1]['time'])
    
    print(f"  ⚡ Fastest Suite: {fastest_suite[0]} ({fastest_suite[1]['time']:.2f}s)")
    print(f"  🐌 Slowest Suite: {slowest_suite[0]} ({slowest_suite[1]['time']:.2f}s)")
    
    if total_time > 300:  # 5 minutes
        print(f"  ⚠️  Total test time is high ({total_time/60:.1f} minutes) - expected for comprehensive testing")
    else:
        print(f"  ✅ Total test time is reasonable ({total_time/60:.1f} minutes)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    critical_failed = sum(1 for r in suite_results.values() if not r['passed'] and r['priority'] == 'critical')
    
    if passed_suites == total_suites:
        print("  🎉 ALL TESTS PASSED! Your API is production-ready!")
        print("  📈 Consider setting up continuous monitoring")
        print("  🔒 Ensure proper rate limiting for production traffic")
        print("  📊 Monitor performance metrics in production")
    elif critical_failed == 0:
        print("  ✅ All critical tests passed - API core functionality is solid")
        if not data_passed:
            print("  🔧 Priority: Fix data endpoint issues")
        if not stress_passed:
            print("  🔧 Consider: Optimize database performance for high load")
    else:
        print("  🚨 CRITICAL ISSUES DETECTED - Address these before production:")
        if not health_passed:
            print("    🔥 Fix health endpoint issues immediately")
        if not auth_passed:
            print("    🔥 Fix authentication and JWT handling immediately")
    
    # Production readiness assessment
    print(f"\n🚀 PRODUCTION READINESS ASSESSMENT:")
    
    critical_pass_rate = priority_results.get('critical', {}).get('passed', 0) / priority_results.get('critical', {}).get('total', 1) * 100
    
    if critical_pass_rate == 100:
        print("  🟢 READY FOR PRODUCTION")
        print("    • All critical systems are functioning")
        print("    • Security measures are in place")
        print("    • Service is healthy and responsive")
    elif critical_pass_rate >= 50:
        print("  🟡 NEEDS ATTENTION BEFORE PRODUCTION")
        print("    • Some critical issues need to be resolved")
        print("    • Review failed critical tests")
    else:
        print("  🔴 NOT READY FOR PRODUCTION")
        print("    • Multiple critical systems are failing")
        print("    • Requires immediate attention")
    
    # Final status
    print(f"\n" + "=" * 80)
    if passed_suites == total_suites:
        print("🎉 COMPREHENSIVE TEST SUITE: ALL TESTS PASSED!")
        print("🚀 Your API is ready for production deployment!")
    elif critical_failed == 0:
        print("✅ COMPREHENSIVE TEST SUITE: CORE FUNCTIONALITY SOLID")
        print("🔧 Some optimizations recommended but not blocking")
    else:
        print("⚠️  COMPREHENSIVE TEST SUITE: CRITICAL ISSUES DETECTED")
        print("🔧 Please resolve critical issues before production deployment")
    
    print(f"Test suite completed: {datetime.now()}")
    print(f"Total duration: {total_time/60:.1f} minutes")
    print("=" * 80)
    
    return critical_failed == 0  # Success if no critical failures

async def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("💨 QUICK SMOKE TEST")
    print("=" * 40)
    
    smoke_tests = [
        ("/health", "Health check"),
        ("/docs", "API documentation"),
    ]
    
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for endpoint, description in smoke_tests:
                print(f"Testing {endpoint} ({description})...")
                
                try:
                    start_time = time.time()
                    async with session.get(f"{BASE_URL}{endpoint}") as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            print(f"  ✅ {response.status} OK ({response_time:.3f}s)")
                        else:
                            print(f"  ❌ {response.status} ({response_time:.3f}s)")
                            return False
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    return False
            
            print("✅ Smoke test passed - proceeding with comprehensive tests")
            return True
                    
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("Starting comprehensive API testing suite...\n")
        
        # Quick smoke test first
        smoke_ok = await run_quick_smoke_test()
        
        if not smoke_ok:
            print("\n⚠️  Smoke test failed. Service may be down.")
            print("   Continuing with comprehensive tests anyway...\n")
        else:
            print("\n✅ Smoke test passed. Starting comprehensive testing...\n")
        
        # Run comprehensive tests
        success = await run_comprehensive_test_suite()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    
    asyncio.run(main()) 
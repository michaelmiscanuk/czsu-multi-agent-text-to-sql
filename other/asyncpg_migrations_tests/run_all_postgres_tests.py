#!/usr/bin/env python3
"""
PostgreSQL Test Runner
Runs all PostgreSQL tests with various configurations and options
"""

import asyncio
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

class TestRunner:
    """Test runner for PostgreSQL tests"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def run_test_script(self, script_path: str, description: str) -> Dict[str, Any]:
        """Run a test script and capture results"""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {description}")
        print(f"📄 Script: {script_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print("STDERR:", result.stderr)
            
            test_result = {
                'script': script_path,
                'description': description,
                'success': success,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} {description} ({duration:.2f}s)")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ TIMEOUT {description} ({duration:.2f}s)")
            
            return {
                'script': script_path,
                'description': description,
                'success': False,
                'duration': duration,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 ERROR {description}: {e}")
            
            return {
                'script': script_path,
                'description': description,
                'success': False,
                'duration': duration,
                'return_code': -2,
                'stdout': '',
                'stderr': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_tests(self, test_suite: str = "all") -> bool:
        """Run all PostgreSQL tests"""
        
        # Define test suites
        test_suites = {
            "quick": [
                ("test_postgres_quick.py", "Quick Unit Tests")
            ],
            "comprehensive": [
                ("test_postgres_quick.py", "Quick Unit Tests"),
                ("test_postgres_comprehensive.py", "Comprehensive Test Suite")
            ],
            "stress": [
                ("test_postgres_quick.py", "Quick Unit Tests"),
                ("test_postgres_stress.py", "Stress Test Suite")
            ],
            "all": [
                ("test_postgres_quick.py", "Quick Unit Tests"),
                ("test_postgres_comprehensive.py", "Comprehensive Test Suite"),
                ("test_postgres_stress.py", "Stress Test Suite")
            ]
        }
        
        if test_suite not in test_suites:
            print(f"❌ Unknown test suite: {test_suite}")
            print(f"Available suites: {', '.join(test_suites.keys())}")
            return False
        
        tests_to_run = test_suites[test_suite]
        
        print(f"🚀 PostgreSQL Test Runner")
        print(f"📋 Test Suite: {test_suite}")
        print(f"🔢 Tests to run: {len(tests_to_run)}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check environment first
        if not self.check_environment():
            print("❌ Environment check failed. Cannot run tests.")
            return False
        
        # Run tests
        for script_path, description in tests_to_run:
            if not os.path.exists(script_path):
                print(f"⚠️ Skipping {script_path} - file not found")
                continue
            
            result = self.run_test_script(script_path, description)
            self.results.append(result)
        
        # Print summary
        return self.print_summary()
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        print(f"\n🔍 Environment Check")
        print("-" * 30)
        
        required_vars = ['user', 'password', 'host', 'port', 'dbname']
        missing_vars = []
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                # Don't print password
                display_value = "***" if var == 'password' else value
                print(f"✅ {var}: {display_value}")
            else:
                print(f"❌ {var}: Not set")
                missing_vars.append(var)
        
        if missing_vars:
            print(f"\n❌ Missing required environment variables: {missing_vars}")
            print("💡 Make sure your .env file is properly configured")
            return False
        
        print("✅ Environment check passed")
        return True
    
    def print_summary(self) -> bool:
        """Print test summary"""
        total_duration = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏰ Total Duration: {total_duration:.2f}s")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "📈 Success Rate: 0%")
        
        # Individual test results
        print(f"\n📋 Individual Results:")
        for result in self.results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['description']} ({result['duration']:.2f}s)")
        
        # Failed tests details
        if failed_tests > 0:
            print(f"\n❌ Failed Tests Details:")
            for result in self.results:
                if not result['success']:
                    print(f"\n  📄 {result['description']}:")
                    print(f"     Return Code: {result['return_code']}")
                    if result['stderr']:
                        print(f"     Error: {result['stderr'][:200]}...")
        
        # Save results
        results_file = f"all_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_duration': total_duration,
                    'summary': {
                        'total': total_tests,
                        'passed': passed_tests,
                        'failed': failed_tests,
                        'success_rate': (passed_tests/total_tests)*100 if total_tests > 0 else 0
                    },
                    'results': self.results
                }, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save results file: {e}")
        
        # Final verdict
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            print("✅ PostgreSQL implementation is working correctly")
            return True
        else:
            print(f"\n💥 {failed_tests} TEST(S) FAILED 💥")
            print("❌ Please check the implementation and fix the issues")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PostgreSQL Test Runner')
    parser.add_argument(
        '--suite', 
        choices=['quick', 'comprehensive', 'stress', 'all'],
        default='all',
        help='Test suite to run (default: all)'
    )
    parser.add_argument(
        '--list-tests',
        action='store_true',
        help='List available tests and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_tests:
        print("Available Test Suites:")
        print("  quick:        Quick unit tests only")
        print("  comprehensive: Quick + comprehensive tests")
        print("  stress:       Quick + stress tests")
        print("  all:          All tests (quick + comprehensive + stress)")
        return 0
    
    runner = TestRunner()
    success = runner.run_all_tests(args.suite)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
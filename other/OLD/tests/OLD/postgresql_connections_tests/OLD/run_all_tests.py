"""
Comprehensive Test Runner for PostgreSQL Connection Pool Fix

This script runs all test suites to validate the PostgreSQL connection pool fix
and provides a comprehensive summary of the results.

Windows Unicode Fix: All Unicode characters replaced with ASCII equivalents
to prevent UnicodeEncodeError on Windows console.
"""

import asyncio
import os
import sys
import subprocess
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width)

def print_section(title: str, width: int = 80):
    """Print a section header."""
    print(f"\n{'=' * 20} {title} {'=' * (width - len(title) - 22)}")

def run_test_script(script_path: str, script_name: str) -> Tuple[bool, str, float]:
    """Run a test script and return success status, output, and duration."""
    print(f"\n>> Running {script_name}...")
    print(f"   Script: {script_path}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        duration = time.time() - start_time
        
        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout
        
        if success:
            print(f"[OK] {script_name} completed successfully in {duration:.2f}s")
        else:
            print(f"[X] {script_name} failed after {duration:.2f}s")
            print(f"Exit code: {result.returncode}")
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[TIMEOUT] {script_name} timed out after {duration:.2f}s")
        return False, "Test timed out", duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"[ERROR] {script_name} failed with exception: {e}")
        return False, f"Exception: {str(e)}", duration

def check_prerequisites():
    """Check if all prerequisites are met."""
    print_section("Prerequisites Check")
    
    prerequisites_met = True
    
    # Check Python version
    python_version = sys.version_info
    print(f"[P] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("[X] Python 3.8+ required")
        prerequisites_met = False
    else:
        print("[OK] Python version OK")
    
    # Check required packages
    required_packages = [
        "asyncio",
        "psycopg",
        "psycopg_pool", 
        "langgraph",
        "dotenv"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} available")
        except ImportError:
            print(f"[X] {package} not available")
            prerequisites_met = False
    
    # Check for environment file
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"[OK] Environment file ({env_file}) found")
    else:
        print(f"[WARN] Environment file ({env_file}) not found")
        print("   Make sure database connection variables are set")
    
    # Check test scripts exist
    test_scripts = [
        "test_postgres_basic_connection.py",
        "test_langgraph_checkpointer.py", 
        "test_multiuser_scenarios.py",
        "test_integration_with_existing_system.py"
    ]
    
    for script in test_scripts:
        if os.path.exists(script):
            print(f"[OK] Test script {script} found")
        else:
            print(f"[X] Test script {script} missing")
            prerequisites_met = False
    
    return prerequisites_met

def generate_summary_report(test_results: Dict, total_duration: float):
    """Generate a comprehensive summary report."""
    print_header("COMPREHENSIVE TEST SUMMARY REPORT")
    
    # Overall statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"[STATS] OVERALL STATISTICS")
    print(f"  Total Test Suites: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    print(f"  Total Duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
    
    # Individual test results
    print(f"\n[RESULTS] INDIVIDUAL TEST RESULTS")
    print("-" * 80)
    
    for test_name, result in test_results.items():
        status = "[OK] PASSED" if result["success"] else "[X] FAILED"
        duration = result["duration"]
        print(f"{test_name:<40}: {status:<12} ({duration:.2f}s)")
    
    # Detailed failure analysis
    failed_tests_details = {name: result for name, result in test_results.items() if not result["success"]}
    
    if failed_tests_details:
        print(f"\n[FAILURES] FAILURE ANALYSIS")
        print("-" * 80)
        
        for test_name, result in failed_tests_details.items():
            print(f"\n[X] {test_name}:")
            print(f"   Duration: {result['duration']:.2f}s")
            
            # Extract key error information from output
            output_lines = result["output"].split('\n')
            error_lines = [line for line in output_lines if '[X]' in line or 'ERROR' in line.upper() or 'FAILED' in line.upper()]
            
            if error_lines:
                print("   Key errors:")
                for error_line in error_lines[:5]:  # Show first 5 errors
                    print(f"     {error_line.strip()}")
                if len(error_lines) > 5:
                    print(f"     ... and {len(error_lines) - 5} more errors")
            else:
                print("   No specific error messages found in output")
    
    # Performance analysis
    print(f"\n[PERF] PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    durations = [result["duration"] for result in test_results.values()]
    avg_duration = sum(durations) / len(durations)
    max_duration = max(durations)
    min_duration = min(durations)
    
    slowest_test = max(test_results.items(), key=lambda x: x[1]["duration"])
    fastest_test = min(test_results.items(), key=lambda x: x[1]["duration"])
    
    print(f"  Average test duration: {avg_duration:.2f}s")
    print(f"  Slowest test: {slowest_test[0]} ({slowest_test[1]['duration']:.2f}s)")
    print(f"  Fastest test: {fastest_test[0]} ({fastest_test[1]['duration']:.2f}s)")
    
    # Recommendations
    print(f"\n[RECOMMENDATIONS] RECOMMENDATIONS")
    print("-" * 80)
    
    if passed_tests == total_tests:
        print("[SUCCESS] All tests passed! Your PostgreSQL connection pool fix is working correctly.")
        print("[OK] Your system is ready for production use.")
        print("[FIXED] The 'connection is closed' error should now be resolved.")
        print("\n[NEXT] Next steps:")
        print("  1. Deploy the fix to your production environment")
        print("  2. Monitor the system for any connection-related issues")
        print("  3. Keep these test scripts for future validation")
    else:
        print("[WARN] Some tests failed. Please address the following:")
        print("\n[TROUBLESHOOT] Troubleshooting steps:")
        print("  1. Check PostgreSQL server connectivity")
        print("  2. Verify environment variables and credentials")
        print("  3. Review failed test logs for specific issues")
        print("  4. Ensure all required Python packages are installed")
        print("  5. Check PostgreSQL configuration and permissions")
        
        # Specific failure recommendations
        if "test_postgres_basic_connection.py" in failed_tests_details:
            print("\n[WARN] Basic connection test failed:")
            print("  - Verify PostgreSQL server is running")
            print("  - Check network connectivity and firewall settings")
            print("  - Validate credentials in .env file")
        
        if "test_langgraph_checkpointer.py" in failed_tests_details:
            print("\n[WARN] LangGraph checkpointer test failed:")
            print("  - Check LangGraph package version")
            print("  - Verify AsyncPostgresSaver functionality")
            print("  - Review PostgreSQL table creation permissions")
        
        if "test_multiuser_scenarios.py" in failed_tests_details:
            print("\n[WARN] Multi-user test failed:")
            print("  - Check PostgreSQL max_connections setting")
            print("  - Review connection pool configuration")
            print("  - Verify concurrent access handling")
        
        if "test_integration_with_existing_system.py" in failed_tests_details:
            print("\n[WARN] Integration test failed:")
            print("  - Verify postgres_checkpointer.py exists")
            print("  - Check module import paths")
            print("  - Review existing implementation compatibility")

def save_detailed_logs(test_results: Dict, timestamp: str):
    """Save detailed test logs to files."""
    print_section("Saving Detailed Logs")
    
    # Create logs directory if it doesn't exist
    logs_dir = "test_logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Save individual test logs
    for test_name, result in test_results.items():
        log_filename = f"{logs_dir}/{test_name}_{timestamp}.log"
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Duration: {result['duration']:.2f}s\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("OUTPUT:\n")
            f.write(result['output'])
        
        print(f"  [LOG] {test_name}.log")
    
    # Save summary log
    summary_filename = f"{logs_dir}/summary_{timestamp}.log"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE TEST SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Tests: {len(test_results)}\n")
        f.write(f"Passed: {sum(1 for r in test_results.values() if r['success'])}\n")
        f.write(f"Failed: {sum(1 for r in test_results.values() if not r['success'])}\n")
        f.write("\nDETAILED RESULTS:\n")
        for test_name, result in test_results.items():
            status = "PASSED" if result["success"] else "FAILED"
            f.write(f"{test_name}: {status} ({result['duration']:.2f}s)\n")
    
    print(f"  [LOG] summary.log")
    print(f"Logs saved to {logs_dir}/ directory")

def main():
    """Main test runner function."""
    print_header("PostgreSQL Connection Pool Fix - Test Suite")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n[X] Prerequisites not met. Please fix the issues above before running tests.")
        sys.exit(1)
    
    print("\n[OK] All prerequisites met. Starting test execution...")
    
    # Define test scripts to run
    test_scripts = [
        ("test_postgres_basic_connection.py", "Basic PostgreSQL Connection"),
        ("test_langgraph_checkpointer.py", "LangGraph Checkpointer"),
        ("test_multiuser_scenarios.py", "Multi-User Scenarios"),
        ("test_integration_with_existing_system.py", "Integration with Existing System")
    ]
    
    # Run all tests
    test_results = {}
    total_start_time = time.time()
    
    for script_path, script_name in test_scripts:
        print_section(f"Test: {script_name}")
        success, output, duration = run_test_script(script_path, script_name)
        
        test_results[script_path] = {
            "success": success,
            "output": output,
            "duration": duration,
            "name": script_name
        }
        
        # Print immediate result
        status = "[OK] PASSED" if success else "[X] FAILED"
        print(f"[RESULT] Result: {status} (Duration: {duration:.2f}s)")
        
        if not success:
            print("[WARN] Test failed, but continuing with remaining tests...")
    
    total_duration = time.time() - total_start_time
    
    # Generate comprehensive report
    generate_summary_report(test_results, total_duration)
    
    # Save logs with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_detailed_logs(test_results, timestamp)
    
    # Print final status
    passed_count = sum(1 for result in test_results.values() if result["success"])
    total_count = len(test_results)
    
    if passed_count == total_count:
        print_header("[SUCCESS] ALL TESTS PASSED!", char="*")
        print("[OK] PostgreSQL connection pool fix is working correctly")
        print("[OK] Your system is ready for production use")
        print("[FIXED] The 'connection is closed' error should now be resolved")
        exit_code = 0
    else:
        print_header("[WARN] SOME TESTS FAILED", char="!")
        print("[X] Please review the detailed analysis above")
        print("[ACTION] Address the identified issues before deploying to production")
        exit_code = 1
    
    print(f"\nTest execution completed in {total_duration:.2f} seconds")
    print(f"Summary: {passed_count}/{total_count} tests passed")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test runner failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 
#!/usr/bin/env python3
"""
Concurrency test for the /analyze endpoint.
Tests concurrent requests to ensure proper database connection handling
and PostgreSQL checkpointer stability under load.

This test uses real functionality from the main scripts without hardcoding values.
"""

import asyncio
import sys
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List
import json

# Add the root directory to Python path to import from main scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import FastAPI testing utilities
import pytest
import httpx
from fastapi.testclient import TestClient
from fastapi import Depends

# Import ASGI test utilities for async testing
from httpx import ASGITransport

# Import main application and dependencies from our existing scripts
from api_server import app, get_current_user, AnalyzeRequest
from my_agent.utils.postgres_checkpointer import (
    check_postgres_env_vars, 
    get_db_config,
    print__analysis_tracing_debug,
    create_async_postgres_saver,
    close_async_postgres_saver
)

# Test configuration
TEST_EMAIL = "test_user@example.com"
TEST_PROMPTS = [
    "Porovnej narust poctu lidi v Brne a Praze v poslednich letech.",
    "Kontrastuj uroven zivota v Brne a v Praze."
]

class ConcurrencyTestResults:
    """Class to track and analyze concurrency test results."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        self.errors: List[Dict[str, Any]] = []
    
    def add_result(self, thread_id: str, prompt: str, response_data: Dict, 
                   response_time: float, status_code: int):
        """Add a test result."""
        result = {
            "thread_id": thread_id,
            "prompt": prompt,
            "response_data": response_data,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "success": status_code == 200
        }
        self.results.append(result)
        print(f"✅ Result added: Thread {thread_id}, Status {status_code}, Time {response_time:.2f}s")
    
    def add_error(self, thread_id: str, prompt: str, error: Exception, 
                  response_time: float = None):
        """Add an error result."""
        error_info = {
            "thread_id": thread_id,
            "prompt": prompt,
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_info)
        print(f"❌ Error added: Thread {thread_id}, Error: {str(error)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        total_requests = len(self.results) + len(self.errors)
        successful_requests = len([r for r in self.results if r['success']])
        failed_requests = len(self.errors) + len([r for r in self.results if not r['success']])
        
        if self.results:
            avg_response_time = sum(r['response_time'] for r in self.results) / len(self.results)
            max_response_time = max(r['response_time'] for r in self.results)
            min_response_time = min(r['response_time'] for r in self.results)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        total_test_time = None
        if self.start_time and self.end_time:
            total_test_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "average_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "total_test_time": total_test_time,
            "errors": self.errors,
            "concurrent_requests_completed": successful_requests >= 2
        }

def create_mock_user():
    """Create a mock user object for testing."""
    return {
        "email": TEST_EMAIL,
        "name": "Test User",
        "sub": "test_user_123",
        "aud": "test_audience",
        "exp": int(time.time()) + 3600  # Expires in 1 hour
    }

def override_get_current_user():
    """Override the get_current_user dependency for testing."""
    return create_mock_user()

def setup_test_environment():
    """Set up the test environment and check prerequisites."""
    print("🔧 Setting up test environment...")
    
    # Check if PostgreSQL environment variables are set
    if not check_postgres_env_vars():
        print("❌ PostgreSQL environment variables are not properly configured!")
        print("Required variables: host, port, dbname, user, password")
        print("Current config:", get_db_config())
        return False
    
    print("✅ PostgreSQL environment variables are configured")
    
    # Override the authentication dependency for testing
    app.dependency_overrides[get_current_user] = override_get_current_user
    print("✅ Authentication dependency overridden for testing")
    
    return True

async def make_analyze_request(client: httpx.AsyncClient, thread_id: str, 
                              prompt: str, results: ConcurrencyTestResults):
    """Make a single analyze request and record the result."""
    print(f"🚀 Starting request for thread {thread_id}")
    start_time = time.time()
    
    try:
        # Create the request payload using the same structure as the main app
        request_data = {
            "prompt": prompt,
            "thread_id": thread_id
        }
        
        # Make the request
        response = await client.post("/analyze", json=request_data)
        response_time = time.time() - start_time
        
        print(f"📝 Thread {thread_id} - Status: {response.status_code}, Time: {response_time:.2f}s")
        
        # Parse response
        if response.status_code == 200:
            response_data = response.json()
            results.add_result(thread_id, prompt, response_data, response_time, response.status_code)
        else:
            # Handle non-200 responses
            try:
                error_data = response.json()
            except:
                error_data = {"error": response.text}
            results.add_result(thread_id, prompt, error_data, response_time, response.status_code)
        
    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Thread {thread_id} - Error: {str(e)}, Time: {response_time:.2f}s")
        results.add_error(thread_id, prompt, e, response_time)

async def run_concurrency_test() -> ConcurrencyTestResults:
    """Run the main concurrency test with 2 simultaneous requests."""
    print("🎯 Starting concurrency test with 2 simultaneous requests...")
    
    results = ConcurrencyTestResults()
    results.start_time = datetime.now()
    
    # Create test client using ASGITransport for proper async testing
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), 
        base_url="http://test"
    ) as client:
        # Generate unique thread IDs for the test
        thread_id_1 = f"test_thread_{uuid.uuid4().hex[:8]}"
        thread_id_2 = f"test_thread_{uuid.uuid4().hex[:8]}"
        
        print(f"📋 Test threads: {thread_id_1}, {thread_id_2}")
        print(f"📋 Test prompts: {TEST_PROMPTS}")
        
        # Create concurrent tasks
        tasks = [
            make_analyze_request(client, thread_id_1, TEST_PROMPTS[0], results),
            make_analyze_request(client, thread_id_2, TEST_PROMPTS[1], results)
        ]
        
        # Run tasks concurrently
        print("⚡ Executing concurrent requests...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Add a small delay to ensure all results are recorded
        await asyncio.sleep(0.1)
    
    results.end_time = datetime.now()
    return results

def analyze_concurrency_results(results: ConcurrencyTestResults):
    """Analyze and display the concurrency test results."""
    print("\n" + "="*60)
    print("📊 CONCURRENCY TEST RESULTS ANALYSIS")
    print("="*60)
    
    summary = results.get_summary()
    
    print(f"🔢 Total Requests: {summary['total_requests']}")
    print(f"✅ Successful: {summary['successful_requests']}")
    print(f"❌ Failed: {summary['failed_requests']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['total_test_time']:
        print(f"⏱️  Total Test Time: {summary['total_test_time']:.2f}s")
    
    if summary['successful_requests'] > 0:
        print(f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"🏆 Best Response Time: {summary['min_response_time']:.2f}s")
        print(f"🐌 Worst Response Time: {summary['max_response_time']:.2f}s")
    
    print(f"🎯 Concurrent Requests Completed: {'✅ YES' if summary['concurrent_requests_completed'] else '❌ NO'}")
    
    # Show individual results
    print("\n📋 Individual Request Results:")
    for i, result in enumerate(results.results, 1):
        status_emoji = "✅" if result['success'] else "❌"
        print(f"  {i}. {status_emoji} Thread: {result['thread_id'][:12]}... | "
              f"Status: {result['status_code']} | Time: {result['response_time']:.2f}s")
        if 'run_id' in result.get('response_data', {}):
            print(f"     Run ID: {result['response_data']['run_id']}")
    
    # Show errors if any
    if results.errors:
        print("\n❌ Errors Encountered:")
        for i, error in enumerate(results.errors, 1):
            print(f"  {i}. Thread: {error['thread_id'][:12]}... | Error: {error['error']}")
    
    # Connection pool and database analysis
    print("\n🔍 CONCURRENCY ANALYSIS:")
    if summary['concurrent_requests_completed']:
        print("✅ Both requests completed - database connection handling appears stable")
        if summary['max_response_time'] - summary['min_response_time'] < 2.0:
            print("✅ Response times are consistent - good concurrent performance")
        else:
            print("⚠️  Response times vary significantly - possible connection contention")
    else:
        print("❌ Concurrent requests failed - potential database connection issues")
    
    return summary

async def test_database_connectivity():
    """Test basic database connectivity before running concurrency tests."""
    print("🔍 Testing database connectivity...")
    
    try:
        # Test database connection using our existing functionality
        from my_agent.utils.postgres_checkpointer import create_async_postgres_saver, close_async_postgres_saver
        
        print("🔧 Creating database checkpointer...")
        checkpointer = await create_async_postgres_saver()
        
        if checkpointer:
            print("✅ Database checkpointer created successfully")
            
            # Test a simple operation
            test_config = {"configurable": {"thread_id": "connectivity_test"}}
            test_result = await checkpointer.aget(test_config)
            print("✅ Database connectivity test passed")
            
            await close_async_postgres_saver()
            print("✅ Database connection closed properly")
            return True
        else:
            print("❌ Failed to create database checkpointer")
            return False
            
    except Exception as e:
        print(f"❌ Database connectivity test failed: {str(e)}")
        return False

def cleanup_test_environment():
    """Clean up the test environment."""
    print("🧹 Cleaning up test environment...")
    
    # Remove the dependency override
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]
        print("✅ Authentication dependency override removed")

async def main():
    """Main test execution function."""
    print("🚀 PostgreSQL Concurrency Test Starting...")
    print("="*60)
    
    # Setup test environment
    if not setup_test_environment():
        print("❌ Test environment setup failed!")
        return False
    
    try:
        # Test database connectivity first
        if not await test_database_connectivity():
            print("❌ Database connectivity test failed!")
            return False
        
        print("✅ Database connectivity confirmed - proceeding with concurrency test")
        
        # Run the concurrency test
        results = await run_concurrency_test()
        
        # Analyze results
        summary = analyze_concurrency_results(results)
        
        # Determine overall test success
        test_passed = (
            summary['concurrent_requests_completed'] and 
            summary['success_rate'] >= 50  # At least 50% success rate
        )
        
        print(f"\n🏁 OVERALL TEST RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        
        return test_passed
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False
        
    finally:
        cleanup_test_environment()

# Test runner for pytest
@pytest.mark.asyncio
async def test_analyze_endpoint_concurrency():
    """Pytest-compatible test function."""
    result = await main()
    assert result, "Concurrency test failed"

if __name__ == "__main__":
    # Direct execution
    import sys
    
    # Set debug mode for better visibility
    os.environ['print__analysis_tracing_debug'] = '0'
    os.environ['print__analyze_debug'] = '1'
    
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
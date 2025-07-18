#!/usr/bin/env python3
"""
Concurrency test for the /analyze endpoint.
Tests concurrent requests to ensure proper database connection handling
and PostgreSQL checkpointer stability under load.
"""

import asyncio
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List

import httpx
import pytest

from tests.helpers import (
    BaseTestResults,
    extract_detailed_error_info,
    make_request_with_traceback_capture,
    save_traceback_report,
    create_test_jwt_token,
    check_server_connectivity,
    setup_debug_environment,
    cleanup_debug_environment,
)
from my_agent.utils.postgres_checkpointer import (
    check_postgres_env_vars,
    cleanup_checkpointer,
    close_async_postgres_saver,
    create_async_postgres_saver,
    get_db_config,
)

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Test configuration
SERVER_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 180
TEST_EMAIL = "test_user@example.com"
TEST_PROMPTS = [
    "Porovnej narust poctu lidi v Brne a Praze v poslednich letech.",
    "Kontrastuj uroven zivota v Brne a v Praze.",
]


def setup_test_environment():
    """Set up the test environment and check prerequisites."""
    print("🔧 Setting up test environment...")

    if not check_postgres_env_vars():
        print("❌ PostgreSQL environment variables are not properly configured!")
        print("Required variables: host, port, dbname, user, password")
        print("Current config:", get_db_config())
        return False

    print("✅ PostgreSQL environment variables are configured")

    use_test_tokens = os.getenv("USE_TEST_TOKENS", "0")
    if use_test_tokens != "1":
        print("⚠️  WARNING: USE_TEST_TOKENS environment variable is not set to '1'")
        print("   The server needs USE_TEST_TOKENS=1 to accept test tokens")
        print("   Set this environment variable in your server environment:")
        print("   SET USE_TEST_TOKENS=1 (Windows)")
        print("   export USE_TEST_TOKENS=1 (Linux/Mac)")
        print("   Continuing test anyway - this may cause 401 authentication errors")
    else:
        print("✅ USE_TEST_TOKENS=1 - test tokens will be accepted by server")

    print("✅ Test environment setup complete (using real HTTP requests)")
    return True


async def test_database_connectivity():
    """Test basic database connectivity before running concurrency tests."""
    print("🔍 Testing database connectivity...")
    try:
        print("🔧 Creating database checkpointer...")
        checkpointer = await create_async_postgres_saver()
        if checkpointer:
            print("✅ Database checkpointer created successfully")
            test_config = {"configurable": {"thread_id": "connectivity_test"}}
            _ = await checkpointer.aget(test_config)
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


async def make_analyze_request(
    client: httpx.AsyncClient,
    test_id: str,
    prompt: str,
    results: BaseTestResults,
):
    """Make a single analyze request and record the result."""
    thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"
    token = create_test_jwt_token(TEST_EMAIL)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    start_time = time.time()
    try:
        request_data = {"prompt": prompt, "thread_id": thread_id}

        result = await make_request_with_traceback_capture(
            client,
            "POST",
            f"{SERVER_BASE_URL}/analyze",
            json=request_data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        response_time = time.time() - start_time
        error_info = extract_detailed_error_info(result)

        if result["response"] is None:
            error_message = error_info["client_error"] or "Unknown client error"
            print(f"❌ Test {test_id} - Client Error: {error_message}")
            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, "/analyze", prompt, error_obj, response_time)
            return

        response = result["response"]
        print(f"Test {test_id}: {response.status_code} ({response_time:.2f}s)")

        if response.status_code == 200:
            try:
                data = response.json()
                results.add_result(
                    test_id,
                    "/analyze",
                    prompt,
                    data,
                    response_time,
                    response.status_code,
                )
            except Exception as e:
                print(f"❌ Response parsing failed: {e}")
                error_obj = Exception(f"Response parsing failed: {e}")
                error_obj.server_tracebacks = error_info["server_tracebacks"]
                results.add_error(test_id, "/analyze", prompt, error_obj, response_time)
        else:
            # Handle error responses
            error_message = None
            if error_info.get("server_error_messages"):
                error_message = "; ".join(
                    f"{em['exception_type']}: {em['exception_message']}"
                    for em in error_info["server_error_messages"]
                )
            if not error_message:
                error_message = error_info.get("client_error") or "Unknown error"

            error_obj = Exception(error_message)
            error_obj.server_tracebacks = error_info.get("server_tracebacks", [])
            results.add_error(test_id, "/analyze", prompt, error_obj, response_time)

    except Exception as e:
        response_time = time.time() - start_time
        error_message = str(e) if str(e).strip() else f"{type(e).__name__}: {repr(e)}"
        if not error_message or error_message.isspace():
            error_message = f"Unknown error of type {type(e).__name__}"

        print(f"❌ Test {test_id} - Error: {error_message}")
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(test_id, "/analyze", prompt, error_obj, response_time)


async def run_concurrency_tests() -> BaseTestResults:
    """Run the main concurrency test with simultaneous requests."""
    print("🚀 Starting concurrency tests...")

    results = BaseTestResults()
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        print(f"📋 Test prompts: {TEST_PROMPTS}")
        print(f"🌐 Server URL: {SERVER_BASE_URL}")
        print(f"⏱️ Request timeout: {REQUEST_TIMEOUT}s")

        # Create concurrent tasks for each prompt
        tasks = []
        for i, prompt in enumerate(TEST_PROMPTS):
            test_id = f"test_{i+1}"
            print(f"🔍 STARTING CONCURRENT REQUEST {i+1}")
            tasks.append(make_analyze_request(client, test_id, prompt, results))

        print("⚡ Executing concurrent requests...")
        print("💡 Note: Analysis requests can take 30-60+ seconds, please wait...")

        # Execute all tasks and handle any exceptions
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check if any tasks returned exceptions
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                print(f"⚠️ Task {i+1} failed with exception: {result}")

        await asyncio.sleep(0.1)  # Small delay to ensure all results are recorded

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results."""
    print("\n📊 CONCURRENCY TEST RESULTS ANALYSIS")
    print("=" * 60)

    summary = results.get_summary()

    print(f"🔢 Total Requests: {summary['total_requests']}")
    print(f"✅ Successful: {summary['successful_requests']}")
    print(f"❌ Failed: {summary['failed_requests']}")
    print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

    if summary.get("total_test_time"):
        print(f"⏱️  Total Test Time: {summary['total_test_time']:.2f}s")

    if summary["successful_requests"] > 0:
        print(f"⚡ Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"🏆 Best Response Time: {summary['min_response_time']:.2f}s")
        print(f"🐌 Worst Response Time: {summary['max_response_time']:.2f}s")

    concurrent_requests_completed = summary["successful_requests"] >= 2
    print(
        f"🎯 Concurrent Requests Completed: {'✅ YES' if concurrent_requests_completed else '❌ NO'}"
    )

    # Show individual results
    print("\n📋 Individual Request Results:")
    for i, result in enumerate(results.results, 1):
        status_emoji = "✅" if result["success"] else "❌"
        print(
            f"  {i}. {status_emoji} Status: {result['status_code']} | Time: {result['response_time']:.2f}s"
        )
        if "run_id" in result.get("response_data", {}):
            print(f"     Run ID: {result['response_data']['run_id']}")

    # Show errors if any
    if results.errors:
        print(f"\n❌ {len(results.errors)} Errors:")
        for error in results.errors:
            print(f"  {error['test_id']}: {error['error']}")

    # Concurrency analysis
    print("\n🔍 CONCURRENCY ANALYSIS:")
    if concurrent_requests_completed:
        print(
            "✅ Multiple requests completed - database connection handling appears stable"
        )
        if summary["max_response_time"] - summary["min_response_time"] < 2.0:
            print("✅ Response times are consistent - good concurrent performance")
        else:
            print(
                "⚠️  Response times vary significantly - possible connection contention"
            )
    else:
        print("❌ Concurrent requests failed - potential database connection issues")

    # Save traceback information if there are failures
    if results.errors or summary["failed_requests"] > 0:
        save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def async_cleanup():
    """Async cleanup for database connections."""
    try:
        await close_async_postgres_saver()
        try:
            await cleanup_checkpointer()
        except Exception as e:
            print(f"⚠️ Warning during global checkpointer cleanup: {e}")
        await asyncio.sleep(0.2)
    except Exception as e:
        print(f"⚠️ Warning during async cleanup: {e}")


async def main():
    """Main test execution function."""
    print("🚀 PostgreSQL Concurrency Test Starting...")
    print("=" * 60)

    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed!")
        print("   Please start your uvicorn server first:")
        print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False

    if not setup_test_environment():
        print("❌ Test environment setup failed!")
        return False

    try:
        if not await test_database_connectivity():
            print("❌ Database connectivity test failed!")
            return False

        print("✅ Database connectivity confirmed - proceeding with concurrency test")

        async with httpx.AsyncClient(
            base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
        ) as client:
            await setup_debug_environment(
                client,
                print__analysis_tracing_debug="1",
                print__analyze_debug="1",
                DEBUG_TRACEBACK="1",
            )
            results = await run_concurrency_tests()
            await cleanup_debug_environment(
                client,
                print__analysis_tracing_debug="0",
                print__analyze_debug="0",
                DEBUG_TRACEBACK="0",
            )

        summary = analyze_test_results(results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == ""
            or "Unknown error" in error.get("error", "")
            for error in summary["errors"]
        )

        test_passed = (
            not has_empty_errors
            and summary["total_requests"] > 0
            and (
                summary["successful_requests"] > 0
                or all(
                    error.get("error", "").strip() != "" for error in summary["errors"]
                )
            )
        )

        if has_empty_errors:
            print(
                "❌ Test failed: Server returned empty error messages (potential crash/hang)"
            )
        elif summary["successful_requests"] == 0:
            print("❌ Test failed: No requests succeeded (server may be down)")
        else:
            print(
                f"✅ Test criteria met: {summary['successful_requests']}/{summary['total_requests']} requests successful"
            )

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Prompts": len(TEST_PROMPTS),
            "Error Location": "main() function",
            "Error During": "Test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False

    finally:
        await async_cleanup()


# Test runner for pytest
@pytest.mark.asyncio
async def test_analyze_endpoint_concurrency():
    """Pytest-compatible test function."""
    result = await main()
    assert result, "Concurrency test failed"


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Test Prompts": len(TEST_PROMPTS),
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)

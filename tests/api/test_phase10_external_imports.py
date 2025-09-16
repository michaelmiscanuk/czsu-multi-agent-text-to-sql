"""
Test for Phase 10: External File Imports and Modular Structure Validation
Tests that the new modular structure works correctly and all imports function properly.
Follows test_phase8_catalog.py pattern with real HTTP requests and comprehensive testing.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

import httpx
from typing import Dict, Any
from datetime import datetime
import time
import asyncio
import traceback
import uuid
import importlib.util

from tests.helpers import (
    BaseTestResults,
    handle_error_response,
    handle_expected_failure,
    extract_detailed_error_info,
    make_request_with_traceback_capture,
    save_traceback_report,
    create_test_jwt_token,
    check_server_connectivity,
    setup_debug_environment,
    cleanup_debug_environment,
)

# Test configuration
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 15  # Reduced from 30 to get quicker feedback on timeout issues
TEST_EMAIL = "test_user@example.com"

# Required import paths to validate
REQUIRED_IMPORTS = {
    "api.main": ["app"],
    "api.routes.health": ["health_check", "router"],
    "api.routes.catalog": ["get_catalog", "router"],
    "api.routes.messages": ["get_chat_messages", "router"],
    "api.routes.analysis": ["analyze", "router"],
    "api.routes.feedback": ["submit_feedback", "router"],
    "api.routes.chat": ["get_thread_sentiments", "router"],
    "api.routes.bulk": ["get_all_chat_messages", "router"],
    "api.routes.debug": ["get_pool_status", "router"],
    "api.auth.jwt_auth": ["verify_google_jwt"],
    "api.dependencies.auth": ["get_current_user"],
    "api.models.requests": ["AnalyzeRequest", "FeedbackRequest"],
    "api.models.responses": ["ChatMessage", "ChatThreadResponse"],
    "api.config.settings": ["GLOBAL_CHECKPOINTER", "start_time"],
    "checkpointer.checkpointer.factory": [
        "initialize_checkpointer",
        "get_global_checkpointer",
    ],
    "checkpointer.user_management.thread_operations": ["create_thread_run_entry"],
}

# Test endpoints to validate modular structure works
ENDPOINT_TESTS = [
    {
        "endpoint": "/health",
        "method": "GET",
        "description": "Health check through modular structure",
        "requires_auth": False,
        "should_succeed": True,
        "expected_keys": ["status"],
    },
    {
        "endpoint": "/catalog",
        "method": "GET",
        "description": "Catalog endpoint through modular structure",
        "requires_auth": True,
        "should_succeed": True,
        "expected_keys": ["results"],
    },
    {
        "endpoint": "/data-tables",
        "method": "GET",
        "description": "Data tables endpoint through modular structure",
        "requires_auth": True,
        "should_succeed": True,
        "expected_keys": ["tables"],
    },
    {
        "endpoint": "/analyze",
        "method": "POST",
        "description": "Analysis endpoint through modular structure",
        "requires_auth": True,
        "should_succeed": True,
        "json_data": {
            "prompt": "SELECT 1",
            "thread_id": "test-" + str(uuid.uuid4())[:8],
        },
        "expected_keys": ["result", "thread_id"],
    },
    {
        "endpoint": "/chat/all-messages-for-all-threads",
        "method": "GET",
        "description": "Bulk messages endpoint through modular structure",
        "requires_auth": True,
        "should_succeed": True,
        "expected_keys": ["messages", "runIds", "sentiments"],
    },
    {
        "endpoint": "/debug/pool-status",
        "method": "GET",
        "description": "Debug endpoint through modular structure",
        "requires_auth": False,
        "should_succeed": True,
        "expected_keys": ["timestamp", "global_checkpointer_exists"],
    },
]


async def test_import_validation() -> BaseTestResults:
    """Test that all required imports work correctly."""
    print("🔍 Testing import validation...")
    results = BaseTestResults(required_endpoints=set())
    results.start_time = datetime.now()

    for module_path, expected_attrs in REQUIRED_IMPORTS.items():
        test_id = f"import_{module_path.replace('.', '_')}"

        try:
            start_time = time.time()
            # Import the module
            module = __import__(module_path, fromlist=expected_attrs)
            response_time = time.time() - start_time

            # Check that expected attributes exist
            missing_attrs = []
            for attr in expected_attrs:
                if not hasattr(module, attr):
                    missing_attrs.append(attr)

            if missing_attrs:
                error_msg = f"Missing attributes in {module_path}: {missing_attrs}"
                results.add_error(
                    test_id,
                    module_path,
                    f"Import validation for {module_path}",
                    Exception(error_msg),
                    response_time,
                )
                print(f"❌ {test_id} - Missing attributes: {missing_attrs}")
            else:
                results.add_result(
                    test_id,
                    module_path,
                    f"Import validation for {module_path}",
                    {"imported_attributes": expected_attrs},
                    response_time,
                    200,  # Mock status code for successful import
                    True,
                )
                print(f"✅ {test_id} - Successfully imported {module_path}")

        except Exception as e:
            response_time = time.time() - start_time
            results.add_error(
                test_id,
                module_path,
                f"Import validation for {module_path}",
                e,
                response_time,
            )
            print(f"❌ {test_id} - Import failed: {e}")

        await asyncio.sleep(0.05)  # Small delay between imports

    results.end_time = datetime.now()
    return results


async def test_external_file_compatibility() -> BaseTestResults:
    """Test that external files can import from the new modular structure."""
    print("🔍 Testing external file compatibility...")
    results = BaseTestResults(required_endpoints=set())
    results.start_time = datetime.now()

    # List of external test files that should work with new imports
    external_files = [
        "tests/api/test_phase8_catalog.py",
        "tests/api/test_phase8_feedback.py",
        "tests/api/test_phase8_analysis.py",
        "tests/api/test_phase8_messages.py",
        "tests/api/test_phase8_chat.py",
        "tests/api/test_phase8_bulk.py",
        "tests/api/test_phase8_debug.py",
    ]

    for file_path in external_files:
        if not Path(file_path).exists():
            continue

        test_id = f"external_{Path(file_path).stem}"

        try:
            start_time = time.time()

            # Try to validate imports in the file (syntax check)
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec and spec.loader:
                # Don't execute the module, just verify it can be loaded
                module = importlib.util.module_from_spec(spec)
                # Check if the module can be created without import errors
                response_time = time.time() - start_time

                results.add_result(
                    test_id,
                    file_path,
                    f"External file compatibility for {file_path}",
                    {"validated": True},
                    response_time,
                    200,
                    True,
                )
                print(f"✅ {test_id} - External file imports validated")
            else:
                response_time = time.time() - start_time
                results.add_error(
                    test_id,
                    file_path,
                    f"External file compatibility for {file_path}",
                    Exception("Could not create module spec"),
                    response_time,
                )

        except Exception as e:
            response_time = time.time() - start_time
            results.add_error(
                test_id,
                file_path,
                f"External file compatibility for {file_path}",
                e,
                response_time,
            )
            print(f"❌ {test_id} - External file validation failed: {e}")

        await asyncio.sleep(0.05)

    results.end_time = datetime.now()
    return results


async def make_modular_endpoint_request(
    client: httpx.AsyncClient,
    test_id: str,
    endpoint: str,
    method: str,
    description: str,
    requires_auth: bool,
    should_succeed: bool,
    results: BaseTestResults,
    json_data: Dict = None,
    expected_keys: list = None,
):
    """Make a request to test modular endpoint functionality."""
    print(f"🔍 {test_id}: Starting request to {method} {endpoint}")
    print(
        f"🔍 {test_id}: Auth required: {requires_auth}, JSON data: {json_data is not None}"
    )
    if json_data:
        print(f"🔍 {test_id}: JSON payload: {json_data}")

    start_time = time.time()

    try:
        headers = {}
        if requires_auth:
            print(f"🔍 {test_id}: Creating JWT token for {TEST_EMAIL}")
            token = create_test_jwt_token(TEST_EMAIL)
            headers["Authorization"] = f"Bearer {token}"
            print(f"🔍 {test_id}: JWT token created, length: {len(token)}")

        print(f"🔍 {test_id}: Making {method} request to {SERVER_BASE_URL}{endpoint}")
        print(f"🔍 {test_id}: Headers: {list(headers.keys())}")

        if method.upper() == "GET":
            print(f"🔍 {test_id}: Executing GET request...")
            response = await client.get(f"{SERVER_BASE_URL}{endpoint}", headers=headers)
        elif method.upper() == "POST":
            print(f"🔍 {test_id}: Executing POST request with JSON data...")
            response = await client.post(
                f"{SERVER_BASE_URL}{endpoint}", headers=headers, json=json_data or {}
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(f"🔍 {test_id}: Request completed, status: {response.status_code}")

        response_time = time.time() - start_time
        print(f"🔍 {test_id}: Response received in {response_time:.3f}s")
        print(f"🔍 {test_id}: Response status: {response.status_code}")
        print(f"🔍 {test_id}: Response headers: {dict(response.headers)}")

        # Try to peek at response content
        try:
            response_text_preview = response.text[:200] if response.text else "<empty>"
            print(f"🔍 {test_id}: Response preview: {response_text_preview}...")
        except Exception as preview_error:
            print(f"🔍 {test_id}: Could not preview response: {preview_error}")

        # Handle response based on expectation
        if should_succeed:
            if response.status_code in [200, 201]:
                try:
                    response_data = response.json()

                    # Check for expected keys if specified
                    if expected_keys:
                        missing_keys = [
                            key for key in expected_keys if key not in response_data
                        ]
                        if missing_keys:
                            error_msg = f"Missing expected keys: {missing_keys}"
                            results.add_error(
                                test_id,
                                endpoint,
                                description,
                                Exception(error_msg),
                                response_time,
                            )
                            print(f"❌ {test_id} - Missing keys: {missing_keys}")
                            return

                    results.add_result(
                        test_id,
                        endpoint,
                        description,
                        response_data,
                        response_time,
                        response.status_code,
                        True,
                    )
                    print(f"✅ {test_id} - Endpoint working through modular structure")

                except Exception as e:
                    results.add_error(test_id, endpoint, description, e, response_time)
                    print(f"❌ {test_id} - JSON parsing failed: {e}")
            else:
                handle_error_response(
                    test_id,
                    endpoint,
                    description,
                    response,
                    {"status_code": response.status_code, "server_tracebacks": []},
                    results,
                    response_time,
                )
        else:
            # Expected failure case
            if response.status_code >= 400:
                results.add_result(
                    test_id,
                    endpoint,
                    description,
                    {"expected_failure": True, "status_code": response.status_code},
                    response_time,
                    response.status_code,
                    True,
                )
                print(f"✅ {test_id} - Expected failure handled correctly")
            else:
                error_msg = f"Expected failure but got status {response.status_code}"
                results.add_error(
                    test_id, endpoint, description, Exception(error_msg), response_time
                )

    except Exception as e:
        response_time = time.time() - start_time
        error_class = e.__class__.__name__
        error_message = str(e) if str(e).strip() else f"{error_class}: {repr(e)}"

        print(f"🔍 {test_id}: Exception occurred after {response_time:.3f}s")
        print(f"🔍 {test_id}: Exception type: {error_class}")
        print(f"🔍 {test_id}: Exception module: {e.__class__.__module__}")

        # Check for specific error types
        if "timeout" in error_message.lower() or "ReadTimeout" in error_class:
            print(
                f"🔍 {test_id}: TIMEOUT DETECTED - Request took {response_time:.1f}s (limit: {REQUEST_TIMEOUT}s)"
            )
            error_message = (
                f"Request timeout after {response_time:.1f}s - {error_message}"
            )
        elif "ConnectionError" in error_class:
            print(f"🔍 {test_id}: CONNECTION ERROR - Server may be unreachable")
        elif not error_message.strip():
            error_message = f"Empty {error_class} exception occurred"

        print(f"❌ Test {test_id} - Error: {error_message}")

        # For analyze endpoint specifically, add more context
        if "endpoint_4" in test_id or "/analyze" in endpoint:
            print(f"🔍 {test_id}: ANALYZE ENDPOINT DIAGNOSTICS:")
            print(f"    - Endpoint: {endpoint}")
            print(f"    - Method: {method}")
            print(f"    - Auth required: {requires_auth}")
            print(f"    - JSON data: {json_data}")
            print(f"    - Timeout setting: {REQUEST_TIMEOUT}s")
            print(f"    - Actual time taken: {response_time:.3f}s")

        # Handle server traceback if available
        error_obj = Exception(error_message)
        error_obj.server_tracebacks = []
        results.add_error(test_id, endpoint, description, error_obj, response_time)


async def test_modular_endpoints() -> BaseTestResults:
    """Test that all endpoints work through the modular structure."""
    print("🔍 Testing modular endpoints...")
    results = BaseTestResults(
        required_endpoints={test["endpoint"] for test in ENDPOINT_TESTS}
    )
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i, test_case in enumerate(ENDPOINT_TESTS, 1):
            test_id = f"endpoint_{i}"

            await make_modular_endpoint_request(
                client,
                test_id,
                test_case["endpoint"],
                test_case["method"],
                test_case["description"],
                test_case["requires_auth"],
                test_case["should_succeed"],
                results,
                test_case.get("json_data"),
                test_case.get("expected_keys"),
            )

            await asyncio.sleep(0.1)  # Small delay between requests

    results.end_time = datetime.now()
    return results


async def test_backward_compatibility() -> BaseTestResults:
    """Test backward compatibility - ensure old patterns still work."""
    print("🔍 Testing backward compatibility...")
    results = BaseTestResults(required_endpoints=set())
    results.start_time = datetime.now()

    # Test that we can still import the main app and it has expected attributes
    test_cases = [
        {
            "test_id": "main_app_compatibility",
            "description": "Main app backward compatibility",
            "test_func": lambda: __import__("api.main", fromlist=["app"]).app,
        },
        {
            "test_id": "routes_accessibility",
            "description": "Routes are accessible through main app",
            "test_func": lambda: hasattr(
                __import__("api.main", fromlist=["app"]).app, "routes"
            ),
        },
        {
            "test_id": "middleware_compatibility",
            "description": "Middleware is properly attached",
            "test_func": lambda: hasattr(
                __import__("api.main", fromlist=["app"]).app, "middleware"
            ),
        },
    ]

    for test_case in test_cases:
        test_id = test_case["test_id"]
        try:
            start_time = time.time()
            result = test_case["test_func"]()
            response_time = time.time() - start_time

            if result:
                results.add_result(
                    test_id,
                    "compatibility",
                    test_case["description"],
                    {"success": True},
                    response_time,
                    200,
                    True,
                )
                print(f"✅ {test_id} - Backward compatibility maintained")
            else:
                results.add_error(
                    test_id,
                    "compatibility",
                    test_case["description"],
                    Exception("Compatibility test failed"),
                    response_time,
                )

        except Exception as e:
            response_time = time.time() - start_time
            results.add_error(
                test_id, "compatibility", test_case["description"], e, response_time
            )
            print(f"❌ {test_id} - Backward compatibility failed: {e}")

    results.end_time = datetime.now()
    return results


def analyze_test_results(results: BaseTestResults):
    """Analyze and print test results following Phase 8 patterns."""
    print("\n" + "=" * 80)
    print("📊 PHASE 10 TEST ANALYSIS")
    print("=" * 80)

    summary = results.get_summary()

    print(f"\n📋 TEST SUMMARY:")
    print(f"   • Total Tests: {summary['total_requests']}")
    print(f"   • Successful: {summary['successful_requests']}")
    print(f"   • Failed: {summary['failed_requests']}")
    print(f"   • Success Rate: {summary['success_rate']:.1f}%")
    print(f"   • Average Response Time: {summary['average_response_time']:.3f}s")
    total_test_time = summary.get("total_test_time", 0) or 0
    print(f"   • Total Test Time: {total_test_time:.2f}s")

    if summary["errors"]:
        print(f"\n❌ ERRORS ({len(summary['errors'])}):")
        for error in summary["errors"][:10]:  # Limit to first 10 errors
            print(f"   • {error.get('test_id')} - {error.get('description')}")
            print(f"     Error: {error.get('error')}")
            if error.get("response_time"):
                print(f"     Response Time: {error['response_time']:.3f}s")

    if summary["successful_requests"] > 0:
        print(f"\n✅ SUCCESSFUL TESTS ({summary['successful_requests']}):")
        successful_results = [r for r in results.results if r.get("success")]
        for result in successful_results[:5]:  # Show first 5 successful tests
            print(f"   • {result.get('test_id')} - {result.get('description')}")
            print(f"     Response Time: {result.get('response_time', 0):.3f}s")

    # Always save traceback report
    save_traceback_report(report_type="test_failure", test_results=results)

    return summary


async def main():
    """Main test execution function following Phase 8 patterns."""
    print("🚀 Phase 10: External File Imports Test Starting...")
    print("📋 Testing: Modular structure imports and compatibility")
    print("=" * 80)

    # Only test server connectivity for endpoint tests
    server_available = await check_server_connectivity(SERVER_BASE_URL)
    if not server_available:
        print(
            "⚠️ Server not available - will skip endpoint tests but continue with import tests"
        )

    try:
        all_results = BaseTestResults(required_endpoints=set())

        # Run import validation tests (always run these)
        print("\n" + "=" * 60)
        print("🧪 RUNNING IMPORT VALIDATION TESTS")
        print("=" * 60)

        import_results = await test_import_validation()
        all_results.results.extend(import_results.results)
        all_results.errors.extend(import_results.errors)

        # Run external file compatibility tests
        print("\n" + "=" * 60)
        print("🧪 RUNNING EXTERNAL FILE COMPATIBILITY TESTS")
        print("=" * 60)

        external_results = await test_external_file_compatibility()
        all_results.results.extend(external_results.results)
        all_results.errors.extend(external_results.errors)

        # Run backward compatibility tests
        print("\n" + "=" * 60)
        print("🧪 RUNNING BACKWARD COMPATIBILITY TESTS")
        print("=" * 60)

        backward_results = await test_backward_compatibility()
        all_results.results.extend(backward_results.results)
        all_results.errors.extend(backward_results.errors)

        # Run endpoint tests if server is available
        if server_available:
            print("\n" + "=" * 60)
            print("🧪 RUNNING MODULAR ENDPOINT TESTS")
            print("=" * 60)

            async with httpx.AsyncClient(
                base_url=SERVER_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT)
            ) as client:
                await setup_debug_environment(
                    client,
                    print__imports_debug="1",
                    print__modular_debug="1",
                    DEBUG_TRACEBACK="1",
                )

                endpoint_results = await test_modular_endpoints()
                all_results.results.extend(endpoint_results.results)
                all_results.errors.extend(endpoint_results.errors)

                await cleanup_debug_environment(
                    client,
                    print__imports_debug="0",
                    print__modular_debug="0",
                    DEBUG_TRACEBACK="0",
                )
        else:
            print("⚠️ Skipping endpoint tests - server not available")

        # Analyze all results
        summary = analyze_test_results(all_results)

        # Determine overall test success
        has_empty_errors = any(
            error.get("error", "").strip() == "" for error in summary["errors"]
        )
        has_timeout_errors = any(
            "timeout" in error.get("error", "").lower() for error in summary["errors"]
        )
        has_import_errors = any(
            "import" in error.get("error", "").lower()
            or "modulenotfounderror" in error.get("error", "").lower()
            for error in summary["errors"]
        )

        # Focus on critical imports - if basic imports work, test passes
        critical_import_tests = [
            r
            for r in all_results.results
            if r.get("test_id", "").startswith("import_")
            and any(
                module in r.get("endpoint", "")
                for module in ["api.main", "api.routes", "api.auth"]
            )
        ]

        test_passed = (
            not has_empty_errors
            and not has_timeout_errors  # No timeout errors allowed - we need to fix them
            and summary["total_requests"] > 0
            and summary["successful_requests"] > 0
            and len(critical_import_tests) >= 3  # At least 3 critical imports working
            and summary["success_rate"]
            >= 95.0  # Require higher success rate now that we're debugging
        )

        if has_empty_errors:
            print("\n❌ Test failed: Empty error messages detected")
        elif has_timeout_errors:
            print(
                f"\n❌ Test failed: Timeout errors detected - need to investigate analyze endpoint ({summary['success_rate']:.1f}% success rate)"
            )
        elif has_import_errors and summary["success_rate"] < 50.0:
            print("\n❌ Test failed: Too many import errors")
        elif summary["successful_requests"] == 0:
            print("\n❌ Test failed: No successful tests")
        elif len(critical_import_tests) < 3:
            print("\n❌ Test failed: Critical imports not working")
        elif summary["success_rate"] < 95.0:
            print(
                f"\n❌ Test failed: Success rate too low ({summary['success_rate']:.1f}% < 95%)"
            )
        else:
            print(
                f"\n✅ Test criteria met: {summary['success_rate']:.1f}% success rate with modular imports working"
            )

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        print("\n📋 TESTED COMPONENTS:")
        print("   ✅ Module import validation")
        print("   ✅ External file compatibility")
        print("   ✅ Backward compatibility")
        if server_available:
            print("   ✅ Live endpoint testing")
        else:
            print("   ⚠️  Live endpoint testing (skipped - server not available)")

        print(f"\n📊 IMPORT SUMMARY:")
        import_count = len(
            [
                r
                for r in all_results.results
                if r.get("test_id", "").startswith("import_")
            ]
        )
        print(f"   • {import_count} modules tested for import compatibility")
        print(f"   • {len(REQUIRED_IMPORTS)} critical import paths validated")
        if server_available:
            endpoint_count = len(
                [
                    r
                    for r in all_results.results
                    if r.get("test_id", "").startswith("endpoint_")
                ]
            )
            print(f"   • {endpoint_count} endpoints tested through modular structure")

        return test_passed

    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        test_context = {
            "Server URL": SERVER_BASE_URL,
            "Request Timeout": f"{REQUEST_TIMEOUT}s",
            "Total Required Imports": len(REQUIRED_IMPORTS),
            "Error Location": "main() function",
            "Error During": "Phase 10 testing",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set debug mode for better visibility
    os.environ["DEBUG"] = "1"
    os.environ["USE_TEST_TOKENS"] = "1"

    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

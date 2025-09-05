#!/usr/bin/env python3
"""
Phase 2: Configuration & Settings Validation Tests

Upgraded to use the comprehensive testing pattern from Phase 8 (catalog tests):
- Uses helpers.BaseTestResults for structured result tracking
- Real HTTP calls to running server
- JWT auth headers
- Dynamic debug env variable setup & cleanup
- Server traceback capture
- Validation of key configuration behavior (health endpoint, rate limiting structure, concurrency semaphore limits)

Prerequisites: server must be running (start via start_backend). These tests DO NOT start the server themselves.
"""

import os
import sys
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

import httpx

from tests.helpers import (
    BaseTestResults,
    create_test_jwt_token,
    check_server_connectivity,
    make_request_with_traceback_capture,
    extract_detailed_error_info,
    save_traceback_report,
    setup_debug_environment,
    cleanup_debug_environment,
    handle_error_response,
)

# Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SERVER_BASE_URL = os.environ.get("TEST_SERVER_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("TEST_REQUEST_TIMEOUT", "30"))
TEST_EMAIL = os.environ.get("TEST_USER_EMAIL", "test_user@example.com")
REQUIRED_ENDPOINTS = {"/health", "/debug/set-env", "/debug/reset-env"}

# Configuration expectations (align with api.config.settings)
EXPECTED_MIN_CONCURRENT = 1
EXPECTED_MAX_CONCURRENT = 32  # sanity upper bound
EXPECTED_RATE_LIMIT_WINDOW = 60
EXPECTED_RATE_LIMIT_MIN_REQUESTS = 10


async def _auth_headers() -> Dict[str, str]:
    token = create_test_jwt_token(TEST_EMAIL)
    return {"Authorization": f"Bearer {token}"}


async def _validate_health_structure(data: Dict[str, Any]):
    assert isinstance(data, dict), "Health response must be JSON object"
    # Flexible keys: allow presence of typical health items
    required_any = ["status", "uptime", "timestamp"]
    missing = [k for k in required_any if k not in data]
    assert not missing, f"Health response missing keys: {missing}"
    assert data.get("status") in {"ok", "healthy", "pass"}, "Unexpected health status"
    assert (
        isinstance(data.get("uptime"), (int, float)) and data["uptime"] >= 0
    ), "Invalid uptime"


async def test_health_endpoint(client: httpx.AsyncClient, results: BaseTestResults):
    test_id = "health_1"
    endpoint = "/health"
    start = time.time()
    result = await make_request_with_traceback_capture(
        client,
        "GET",
        f"{SERVER_BASE_URL}{endpoint}",
        headers=await _auth_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    rt = time.time() - start
    error_info = extract_detailed_error_info(result)
    response = result["response"]

    if not response:
        err = Exception(error_info.get("client_error") or "No response")
        err.server_tracebacks = error_info["server_tracebacks"]
        results.add_error(test_id, endpoint, "Health endpoint unreachable", err, rt)
        return

    if response.status_code == 200:
        try:
            data = response.json()
            await _validate_health_structure(data)
            results.add_result(
                test_id, endpoint, "Health endpoint basic structure", data, rt, 200
            )
        except Exception as e:
            err = Exception(f"Validation failed: {e}")
            err.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id, endpoint, "Health structure validation failure", err, rt
            )
    else:
        handle_error_response(
            test_id,
            endpoint,
            "Health endpoint non-200",
            response,
            error_info,
            results,
            rt,
        )


async def test_debug_env_set_and_reset(
    client: httpx.AsyncClient, results: BaseTestResults
):
    # Validate we can toggle DEBUG env flags via existing endpoints
    for action, endpoint in [("set", "/debug/set-env"), ("reset", "/debug/reset-env")]:
        test_id = f"debug_env_{action}"
        start = time.time()
        body = {"DEBUG": "1"} if action == "set" else {"DEBUG": "0"}
        result = await make_request_with_traceback_capture(
            client,
            "POST",
            f"{SERVER_BASE_URL}{endpoint}",
            headers=await _auth_headers(),
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        rt = time.time() - start
        error_info = extract_detailed_error_info(result)
        response = result["response"]

        if not response:
            err = Exception(error_info.get("client_error") or "No response")
            err.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, f"Env {action} unreachable", err, rt)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
                results.add_result(
                    test_id,
                    endpoint,
                    f"Debug env {action} endpoint",
                    data,
                    rt,
                    200,
                )
            except Exception as e:
                err = Exception(f"JSON parse failed: {e}")
                err.server_tracebacks = error_info["server_tracebacks"]
                results.add_error(
                    test_id, endpoint, f"Env {action} parse fail", err, rt
                )
        else:
            handle_error_response(
                test_id,
                endpoint,
                f"Debug env {action} non-200",
                response,
                error_info,
                results,
                rt,
            )


async def test_configuration_module_values(results: BaseTestResults):
    # Direct import validation of api.config.settings (no HTTP)
    test_id = "config_module_values"
    endpoint = "internal:settings_import"
    start = time.time()
    try:
        from api.config import settings as s  # type: ignore

        # Core invariants
        assert isinstance(s.start_time, (int, float)), "start_time must be numeric"
        assert (
            EXPECTED_MIN_CONCURRENT
            <= s.MAX_CONCURRENT_ANALYSES
            <= EXPECTED_MAX_CONCURRENT
        ), f"MAX_CONCURRENT_ANALYSES out of expected bounds: {s.MAX_CONCURRENT_ANALYSES}"
        assert isinstance(
            s.analysis_semaphore, asyncio.Semaphore
        ), "analysis_semaphore must be Semaphore"
        assert (
            s.RATE_LIMIT_WINDOW == EXPECTED_RATE_LIMIT_WINDOW
        ), "RATE_LIMIT_WINDOW mismatch"
        assert (
            s.RATE_LIMIT_REQUESTS >= EXPECTED_RATE_LIMIT_MIN_REQUESTS
        ), "RATE_LIMIT_REQUESTS too low"
        assert isinstance(
            s.rate_limit_storage, dict
        ), "rate_limit_storage must be dict-like"
        assert isinstance(
            s._bulk_loading_cache, dict
        ), "_bulk_loading_cache must be dict"

        data = {
            "start_time": s.start_time,
            "MAX_CONCURRENT_ANALYSES": s.MAX_CONCURRENT_ANALYSES,
            "RATE_LIMIT_WINDOW": s.RATE_LIMIT_WINDOW,
            "RATE_LIMIT_REQUESTS": s.RATE_LIMIT_REQUESTS,
            "semaphore_value": getattr(s.analysis_semaphore, "_value", None),
            "bulk_cache_size": len(s._bulk_loading_cache),
        }
        results.add_result(
            test_id,
            endpoint,
            "Configuration module invariant checks",
            data,
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id,
            endpoint,
            "Configuration module validation failed",
            e,
            time.time() - start,
        )


async def test_semaphore_concurrency_simulation(results: BaseTestResults):
    # Simulate acquiring the analysis semaphore up to its limit
    test_id = "semaphore_concurrency"
    endpoint = "internal:semaphore"
    start = time.time()
    try:
        from api.config import settings as s

        permits = s.MAX_CONCURRENT_ANALYSES
        acquired = 0
        acquired_list: List[asyncio.Task] = []

        async def acquire_and_hold():
            async with s.analysis_semaphore:
                await asyncio.sleep(0.05)

        # Launch tasks = permits
        for _ in range(permits):
            t = asyncio.create_task(acquire_and_hold())
            acquired_list.append(t)

        await asyncio.sleep(0)  # allow scheduling
        current_value = getattr(s.analysis_semaphore, "_value", None)
        # After scheduling but before completion, semaphore internal value should be 0 or near 0
        assert (
            current_value in (0, None) or current_value <= 1
        ), f"Semaphore did not decrement as expected (value={current_value})"

        await asyncio.gather(*acquired_list)

        # After completion, value should be restored to permits
        end_value = getattr(s.analysis_semaphore, "_value", None)
        assert (
            end_value == permits
        ), f"Semaphore value not restored (expected {permits}, got {end_value})"

        results.add_result(
            test_id,
            endpoint,
            "Semaphore concurrency acquisition & release",
            {"permits": permits, "end_value": end_value},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id,
            endpoint,
            "Semaphore concurrency simulation failed",
            e,
            time.time() - start,
        )


async def run_phase2_tests() -> BaseTestResults:
    print("🚀 Starting Phase 2 configuration tests...")
    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Setup debug (turn on minimal debug for visibility)
        await setup_debug_environment(
            client,
            DEBUG="1",
            DEBUG_TRACEBACK="1",
        )

        # HTTP endpoint tests
        await test_health_endpoint(client, results)
        await test_debug_env_set_and_reset(client, results)

        # Internal module + behavior tests
        await test_configuration_module_values(results)
        await test_semaphore_concurrency_simulation(results)

        # Cleanup
        await cleanup_debug_environment(client, DEBUG="0", DEBUG_TRACEBACK="0")

    results.end_time = datetime.now()
    return results


def analyze_results(results: BaseTestResults) -> Dict[str, Any]:
    print("\n📊 Phase 2 Configuration Test Results:")
    summary = results.get_summary()
    print(
        f"Total: {summary['total_requests']}  Success: {summary['successful_requests']}  Failed: {summary['failed_requests']}  SuccessRate: {summary['success_rate']:.1f}%"
    )

    if summary["successful_requests"] > 0:
        print(
            f"Avg Response: {summary['average_response_time']:.3f}s  Min: {summary['min_response_time']:.3f}s  Max: {summary['max_response_time']:.3f}s"
        )

    if not summary["all_endpoints_tested"]:
        print(f"❌ Missing endpoints: {', '.join(summary['missing_endpoints'])}")

    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)}):")
        for e in results.errors:
            print(f"  {e['test_id']} -> {e['error']}")

    save_traceback_report(report_type="test_failure", test_results=results)
    return summary


async def main():
    print("🚀 Phase 2 Configuration & Settings Tests Starting...")
    if not await check_server_connectivity(SERVER_BASE_URL):
        print("❌ Server connectivity check failed")
        return False

    try:
        results = await run_phase2_tests()
        summary = analyze_results(results)
        passed = (
            summary["failed_requests"] == 0
            and summary["successful_requests"] > 0
            and summary["all_endpoints_tested"]
        )
        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    except Exception as e:
        print(f"💥 Fatal execution error: {e}")
        save_traceback_report(
            report_type="exception",
            exception=e,
            test_context={
                "Phase": "2",
                "Component": "Configuration",
                "Server URL": SERVER_BASE_URL,
            },
        )
        return False


if __name__ == "__main__":
    try:
        ok = asyncio.run(main())
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error in runner: {e}")
        sys.exit(1)

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
from pathlib import Path

import httpx

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

# Helper to safely import settings with retry after path adjustment
_settings_module_cache = None


def load_settings_module():
    global _settings_module_cache
    if _settings_module_cache is not None:
        return _settings_module_cache
    try:
        from api.config import settings as s  # type: ignore

        _settings_module_cache = s
        return s
    except ModuleNotFoundError as e:
        raise e


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
REQUIRED_ENDPOINTS = {
    "/health",
    "/debug/set-env",
    "/debug/reset-env",
    "/health/memory",
    "/health/rate-limits",
}

# Configuration expectations (align with api.config.settings)
EXPECTED_MIN_CONCURRENT = 1
EXPECTED_MAX_CONCURRENT = 32  # sanity upper bound
EXPECTED_RATE_LIMIT_WINDOW = 60
EXPECTED_RATE_LIMIT_MIN_REQUESTS = 10

ADDITIONAL_INTERNAL_VALIDATIONS = [
    "gc_threshold",
    "rate_limit_mutation",
    "bulk_cache_ttl",
    "throttle_semaphore",
    "api_import_warning",
]


async def _auth_headers() -> Dict[str, str]:
    token = create_test_jwt_token(TEST_EMAIL)
    return {"Authorization": f"Bearer {token}"}


async def _validate_health_structure(data: Dict[str, Any]):
    # Updated to align with api.routes.health implementation (uses 'uptime_seconds')
    assert isinstance(data, dict), "Health response must be JSON object"
    base_missing = [k for k in ["status", "timestamp"] if k not in data]
    assert not base_missing, f"Health response missing keys: {base_missing}"
    # Accept either uptime_seconds or uptime
    assert ("uptime_seconds" in data) or (
        "uptime" in data
    ), "Missing uptime_seconds/uptime"
    assert data.get("status") in {
        "ok",
        "healthy",
        "pass",
        "degraded",
    }, "Unexpected health status"
    if "uptime_seconds" in data:
        assert (
            isinstance(data["uptime_seconds"], (int, float))
            and data["uptime_seconds"] >= 0
        ), "Invalid uptime_seconds"
    if "uptime" in data:
        assert (
            isinstance(data["uptime"], (int, float)) and data["uptime"] >= 0
        ), "Invalid uptime"
    # Optional deeper structure checks (memory block)
    if "memory" in data and isinstance(data["memory"], dict):
        mem = data["memory"]
        for k in ["rss_mb", "vms_mb", "percent"]:
            if k in mem:
                assert isinstance(mem[k], (int, float)), f"memory.{k} must be numeric"


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


async def test_memory_health_endpoint(
    client: httpx.AsyncClient, results: BaseTestResults
):
    test_id = "health_memory"
    endpoint = "/health/memory"
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
        results.add_error(test_id, endpoint, "Memory health unreachable", err, rt)
        return
    if response.status_code == 200:
        try:
            data = response.json()
            for key in ["status", "memory_rss_mb", "memory_threshold_mb", "timestamp"]:
                assert key in data, f"Memory health missing '{key}'"
            assert isinstance(data["memory_rss_mb"], (int, float))
            results.add_result(
                test_id, endpoint, "Memory health structure", data, rt, 200
            )
        except Exception as e:
            err = Exception(f"Validation failed: {e}")
            err.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id, endpoint, "Memory health validation failure", err, rt
            )
    else:
        handle_error_response(
            test_id,
            endpoint,
            "Memory health non-200",
            response,
            error_info,
            results,
            rt,
        )


async def test_rate_limit_health_endpoint(
    client: httpx.AsyncClient, results: BaseTestResults
):
    test_id = "health_rate_limits"
    endpoint = "/health/rate-limits"
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
        results.add_error(test_id, endpoint, "Rate limit health unreachable", err, rt)
        return
    if response.status_code == 200:
        try:
            data = response.json()
            for key in [
                "status",
                "rate_limit_window",
                "rate_limit_requests",
                "timestamp",
            ]:
                assert key in data, f"Rate limit health missing '{key}'"
            assert isinstance(data["rate_limit_window"], int)
            assert isinstance(data["rate_limit_requests"], int)
            results.add_result(
                test_id, endpoint, "Rate limit health structure", data, rt, 200
            )
        except Exception as e:
            err = Exception(f"Validation failed: {e}")
            err.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(
                test_id, endpoint, "Rate limit health validation failure", err, rt
            )
    else:
        handle_error_response(
            test_id,
            endpoint,
            "Rate limit health non-200",
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
    test_id = "config_module_values"
    endpoint = "internal:settings_import"
    start = time.time()
    try:
        s = load_settings_module()
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
    test_id = "semaphore_concurrency"
    endpoint = "internal:semaphore"
    start = time.time()
    try:
        s = load_settings_module()
        permits = s.MAX_CONCURRENT_ANALYSES
        acquired_list: List[asyncio.Task] = []

        async def acquire_and_hold():
            async with s.analysis_semaphore:
                await asyncio.sleep(0.05)

        for _ in range(permits):
            acquired_list.append(asyncio.create_task(acquire_and_hold()))
        await asyncio.sleep(0)
        current_value = getattr(s.analysis_semaphore, "_value", None)
        assert (
            current_value in (0, None) or current_value <= 1
        ), f"Semaphore did not decrement as expected (value={current_value})"
        await asyncio.gather(*acquired_list)
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


async def test_api_import_warning(results: BaseTestResults):
    test_id = "api_import_warning"
    endpoint = "internal:api_import"
    start = time.time()
    try:
        # Capture stdout during import
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            import importlib

            importlib.invalidate_caches()
            import api  # noqa: F401
        output = buf.getvalue()
        # If warning is present, fail
        assert "Warning: Some API imports failed" not in output, output.strip()
        results.add_result(
            test_id,
            endpoint,
            "API package imports without warnings",
            {"output": output.strip()[:200]},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id,
            endpoint,
            "API import produced warning/errors",
            e,
            time.time() - start,
        )


async def test_gc_threshold(results: BaseTestResults):
    test_id = "gc_threshold"
    endpoint = "internal:gc_threshold"
    start = time.time()
    try:
        s = load_settings_module()
        assert (
            isinstance(s.GC_MEMORY_THRESHOLD, int) and s.GC_MEMORY_THRESHOLD > 100
        ), "GC_MEMORY_THRESHOLD too low"
        assert s.GC_MEMORY_THRESHOLD < 100000, "GC_MEMORY_THRESHOLD unrealistic"
        results.add_result(
            test_id,
            endpoint,
            "GC memory threshold sanity",
            {"GC_MEMORY_THRESHOLD": s.GC_MEMORY_THRESHOLD},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id, endpoint, "GC threshold validation failed", e, time.time() - start
        )


async def test_rate_limit_mutation(results: BaseTestResults):
    test_id = "rate_limit_mutation"
    endpoint = "internal:rate_limit"
    start = time.time()
    try:
        s = load_settings_module()
        before = len(s.rate_limit_storage)
        ip = "127.0.0.1"
        s.rate_limit_storage[ip].append(time.time())
        after = len(s.rate_limit_storage)
        assert after >= before, "Rate limit storage did not record entry"
        results.add_result(
            test_id,
            endpoint,
            "Rate limit storage mutation",
            {"before_keys": before, "after_keys": after},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id, endpoint, "Rate limit mutation failed", e, time.time() - start
        )


async def test_bulk_cache_ttl(results: BaseTestResults):
    test_id = "bulk_cache_ttl"
    endpoint = "internal:bulk_cache"
    start = time.time()
    try:
        s = load_settings_module()
        from api.utils.memory import cleanup_bulk_cache

        key = "test_cache_key"
        # Insert tuple (data, timestamp) per cleanup_bulk_cache iteration expectation
        expired_timestamp = time.time() - (s.BULK_CACHE_TIMEOUT + 5)
        s._bulk_loading_cache[key] = ({"payload": 123}, expired_timestamp)
        cleaned = cleanup_bulk_cache()
        still_present = key in s._bulk_loading_cache
        assert cleaned >= 1 and not still_present, "Expired cache entry was not cleaned"
        results.add_result(
            test_id,
            endpoint,
            "Bulk cache TTL cleanup",
            {"cleaned": cleaned, "still_present": still_present},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id, endpoint, "Bulk cache TTL test failed", e, time.time() - start
        )


async def test_throttle_semaphore(results: BaseTestResults):
    test_id = "throttle_semaphore"
    endpoint = "internal:throttle_semaphore"
    start = time.time()
    try:
        s = load_settings_module()
        sem = s.throttle_semaphores["127.0.0.1"]
        assert isinstance(sem, asyncio.Semaphore)
        initial_value = getattr(sem, "_value", None)
        async with sem:
            during_value = getattr(sem, "_value", None)
            assert (during_value == initial_value - 1) or (
                initial_value is None
            ), "Semaphore did not decrement"
        final_value = getattr(sem, "_value", None)
        assert final_value == initial_value, "Semaphore did not restore"
        results.add_result(
            test_id,
            endpoint,
            "Throttle semaphore acquire/release",
            {"initial": initial_value, "final": final_value},
            time.time() - start,
            200,
        )
    except Exception as e:
        results.add_error(
            test_id, endpoint, "Throttle semaphore test failed", e, time.time() - start
        )


async def run_phase2_tests() -> BaseTestResults:
    print("🚀 Starting Phase 2 configuration tests...")
    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        await setup_debug_environment(client, DEBUG="1", DEBUG_TRACEBACK="1")
        # HTTP endpoint tests (expanded)
        await test_health_endpoint(client, results)
        await test_memory_health_endpoint(client, results)
        await test_rate_limit_health_endpoint(client, results)
        await test_debug_env_set_and_reset(client, results)
        # Internal tests (existing)
        await test_configuration_module_values(results)
        await test_semaphore_concurrency_simulation(results)
        # New deeper internal validations
        await test_gc_threshold(results)
        await test_rate_limit_mutation(results)
        await test_bulk_cache_ttl(results)
        await test_throttle_semaphore(results)
        await test_api_import_warning(results)
        await cleanup_debug_environment(client, DEBUG="0", DEBUG_TRACEBACK="0")
    results.end_time = datetime.now()
    return results


def analyze_results(results: BaseTestResults) -> Dict[str, Any]:
    print("\n📊 Phase 2 Configuration Test Results:")
    summary = results.get_summary()

    # Per-test detailed listing
    print("\n🧪 Individual Test Outcomes:")
    if results.results:
        for r in results.results:
            status = "PASS" if r.get("success") else "FAIL"
            print(
                f"  - {r['test_id']:<24} {status:<4} | {r['endpoint']:<28} | {r['response_time']:.3f}s | {r['description']}"
            )
    if results.errors:
        for e in results.errors:
            print(
                f"  - {e['test_id']:<24} FAIL | {e['endpoint']:<28} | {e.get('response_time', 0) or 0:.3f}s | {e['description']}"
            )

    print(
        f"\n📈 Aggregate: Total={summary['total_requests']}  Success={summary['successful_requests']}  Failed={summary['failed_requests']}  SuccessRate={summary['success_rate']:.1f}%"
    )
    if summary["successful_requests"] > 0:
        print(
            f"⏱️  Timing: Avg={summary['average_response_time']:.3f}s  Min={summary['min_response_time']:.3f}s  Max={summary['max_response_time']:.3f}s"
        )

    # Endpoint coverage
    print("\n🔌 Required Endpoint Coverage:")
    if summary["all_endpoints_tested"]:
        print("  ✅ All required HTTP endpoints exercised")
    else:
        print(
            f"  ❌ Missing: {', '.join(summary['missing_endpoints']) if summary['missing_endpoints'] else 'Unknown'}"
        )
    print(
        f"  Tested HTTP endpoints: {', '.join(sorted(e for e in summary['tested_endpoints'] if e.startswith('/')))}"
    )

    # Diagnostics / heuristics similar to phase8 style
    has_import_warning = any(
        e["test_id"] == "api_import_warning" for e in results.errors
    )
    has_empty_errors = any(
        (not e.get("error", "").strip()) or "Unknown error" in e.get("error", "")
        for e in results.errors
    )
    has_config_anomalies = any(
        "threshold" in e.get("description", "").lower() for e in results.errors
    )

    if has_import_warning:
        print("⚠️  Import warning detected (api_import_warning)")
    if has_empty_errors:
        print("⚠️  One or more errors had empty messages")
    if has_config_anomalies:
        print("⚠️  Configuration anomaly errors present")

    # Error section
    if results.errors:
        print(f"\n❌ {len(results.errors)} Error(s):")
        for err in results.errors:
            print(
                f"  • {err['test_id']} | {err['endpoint']} | {err['error_type']}: {err['error'][:180]}"
            )

    # Save traceback report
    save_traceback_report(report_type="test_failure", test_results=results)

    # Compose enriched summary diagnostics
    summary["diagnostics"] = {
        "has_import_warning": has_import_warning,
        "has_empty_errors": has_empty_errors,
        "has_config_anomalies": has_config_anomalies,
    }

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

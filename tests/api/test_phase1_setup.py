"""
Phase 1: Setup & Preparation Validation

Enhanced test suite that now:
 1. Verifies critical folder structure & __init__ packages
 2. Performs live HTTP checks against core endpoints (health + all API routes)
 3. Validates response schema for selected routes
 4. Captures server tracebacks & timing (pattern aligned with phase8 tests)
 5. Produces aggregated summary & traceback report
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Union

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv

load_dotenv()

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

import asyncio
import httpx

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Imported testing utilities (pattern reused from phase8)
# ---------------------------------------------------------------------------
try:
    from tests.helpers import (
        BaseTestResults,
        extract_detailed_error_info,
        make_request_with_traceback_capture,
        save_traceback_report,
        check_server_connectivity,
        create_test_jwt_token,
    )
except Exception as e:  # Fallback minimal shim if helpers import fails
    print(f"⚠️ Failed importing advanced helpers: {e}. Using basic fallback.")

    class BaseTestResults:  # type: ignore
        def __init__(self, required_endpoints: set = None):
            self.results = []
            self.errors = []
            self.start_time = None
            self.end_time = None
            self.required_endpoints = required_endpoints or set()

        def add_result(
            self,
            test_id,
            endpoint,
            description,
            response_data,
            response_time,
            status_code,
        ):
            self.results.append(
                {
                    "test_id": test_id,
                    "endpoint": endpoint,
                    "description": description,
                    "response_data": response_data,
                    "response_time": response_time,
                    "status_code": status_code,
                    "timestamp": datetime.now().isoformat(),
                    "success": status_code == 200,
                }
            )

        def add_error(
            self,
            test_id,
            endpoint,
            description,
            error,
            response_time=None,
            response_data=None,
        ):
            self.errors.append(
                {
                    "test_id": test_id,
                    "endpoint": endpoint,
                    "description": description,
                    "error": str(error),
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                    "response_data": response_data,
                }
            )

        def get_summary(self):
            total = len(self.results) + len(self.errors)
            success = len([r for r in self.results if r.get("success")])
            failed = total - success
            avg_time = (
                sum(r["response_time"] for r in self.results if r.get("response_time"))
                / max(success, 1)
                if self.results
                else 0
            )
            tested_endpoints = set(
                r["endpoint"] for r in self.results if r.get("success")
            )
            missing = (
                list(self.required_endpoints - tested_endpoints)
                if self.required_endpoints
                else []
            )
            return {
                "total_requests": total,
                "successful_requests": success,
                "failed_requests": failed,
                "success_rate": (success / total * 100) if total else 0,
                "average_response_time": avg_time,
                "all_endpoints_tested": len(missing) == 0,
                "missing_endpoints": missing,
                "errors": self.errors,
            }

    async def make_request_with_traceback_capture(client, method, url, **kwargs):  # type: ignore
        try:
            resp = await client.request(method, url, **kwargs)
            return {
                "response": resp,
                "success": resp.status_code == 200,
                "server_tracebacks": [],
            }
        except Exception as e:  # noqa
            return {
                "response": None,
                "success": False,
                "client_exception": e,
                "server_tracebacks": [],
            }

    def extract_detailed_error_info(result):  # type: ignore
        return {
            "has_server_errors": False,
            "has_client_errors": not result.get("success"),
            "server_tracebacks": [],
            "server_error_messages": [],
            "client_error": (
                str(result.get("client_exception"))
                if result.get("client_exception")
                else None
            ),
        }

    async def check_server_connectivity(base_url: str, timeout: float = 5.0) -> bool:  # type: ignore
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                await c.get(base_url + "/health")
            return True
        except Exception:
            return False

    def save_traceback_report(**kwargs):  # type: ignore
        return None

    def create_test_jwt_token(email: str = "test@example.com"):  # type: ignore
        return "dummy.token.value"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("TEST_REQUEST_TIMEOUT", "30"))
TEST_EMAIL = os.environ.get("TEST_EMAIL", "test_user@example.com")

# ---------------------------------------------------------------------------
# Local (non-HTTP) validation tests
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    "api",
    "api/config",
    "api/utils",
    "api/models",
    "api/middleware",
    "api/auth",
    "api/exceptions",
    "api/dependencies",
    "api/routes",
    "tests",
]

REQUIRED_INIT_FILES = [
    "api/__init__.py",
    "api/config/__init__.py",
    "api/utils/__init__.py",
    "api/models/__init__.py",
    "api/middleware/__init__.py",
    "api/auth/__init__.py",
    "api/exceptions/__init__.py",
    "api/dependencies/__init__.py",
    "api/routes/__init__.py",
    "tests/__init__.py",
]


async def _local_folder_structure_check(results: BaseTestResults):
    """Validate folder structure exists."""
    missing = []
    for d in REQUIRED_DIRS:
        p = BASE_DIR / d
        if not p.exists() or not p.is_dir():
            missing.append(d)
    if missing:
        results.add_error(
            "local-structure",
            "local:folder-structure",
            "Required directories missing",
            Exception(f"Missing directories: {missing}"),
        )
    else:
        results.add_result(
            "local-structure",
            "local:folder-structure",
            "All required directories present",
            {"checked": len(REQUIRED_DIRS)},
            0.0,
            200,
        )


async def _local_init_files_check(results: BaseTestResults):
    missing = []
    for f in REQUIRED_INIT_FILES:
        p = BASE_DIR / f
        if not p.exists() or not p.is_file():
            missing.append(f)
    if missing:
        results.add_error(
            "local-init",
            "local:init-files",
            "Required __init__.py files missing",
            Exception(f"Missing init files: {missing}"),
        )
    else:
        results.add_result(
            "local-init",
            "local:init-files",
            "All required __init__.py files present",
            {"checked": len(REQUIRED_INIT_FILES)},
            0.0,
            200,
        )


async def _local_import_check(results: BaseTestResults):
    packages = [
        "api",
        "api.config",
        "api.utils",
        "api.models",
        "api.middleware",
        "api.auth",
        "api.exceptions",
        "api.dependencies",
        "api.routes",
        "tests",
    ]
    try:
        for pkg in packages:
            __import__(pkg)
        results.add_result(
            "local-imports",
            "local:imports",
            "All packages importable",
            {"imported": len(packages)},
            0.0,
            200,
        )
    except Exception as e:
        results.add_error(
            "local-imports",
            "local:imports",
            "Package import failure",
            e,
        )


# ---------------------------------------------------------------------------
# HTTP Endpoint Validators (extended)
# ---------------------------------------------------------------------------
Validator = Callable[[str, Dict[str, Any]], None]


def validate_health(endpoint: str, data: Dict[str, Any]):
    assert "status" in data, "Missing status"
    assert "timestamp" in data, "Missing timestamp"
    assert "memory" in data, "Missing memory section"
    assert "database" in data, "Missing database section"
    assert isinstance(data["database"], dict)


def validate_health_database(endpoint: str, data: Dict[str, Any]):
    assert "timestamp" in data
    assert "checkpointer_available" in data
    assert "checkpointer_type" in data


def validate_health_memory(endpoint: str, data: Dict[str, Any]):
    for key in [
        "status",
        "memory_rss_mb",
        "memory_threshold_mb",
        "memory_usage_percent",
        "timestamp",
    ]:
        assert key in data, f"Missing {key}"


def validate_rate_limits(endpoint: str, data: Dict[str, Any]):
    for key in [
        "status",
        "total_tracked_clients",
        "rate_limit_window",
        "rate_limit_requests",
        "timestamp",
    ]:
        assert key in data, f"Missing {key}"


def validate_prepared_statements(endpoint: str, data: Dict[str, Any]):
    assert "status" in data
    # Depending on DB availability we accept either healthy/degraded/unhealthy
    assert data["status"] in {"healthy", "degraded", "unhealthy"}


def validate_catalog(endpoint: str, data: Dict[str, Any]):
    assert "results" in data, "Missing results"
    assert "total" in data and isinstance(data["total"], int)
    assert "page" in data and isinstance(data["page"], int)
    assert "page_size" in data and isinstance(data["page_size"], int)
    assert isinstance(data["results"], list)


def validate_data_tables(endpoint: str, data: Dict[str, Any]):
    assert "tables" in data and isinstance(data["tables"], list)
    if data["tables"]:
        sample = data["tables"][0]
        assert isinstance(sample, dict)
        # selection_code optional validation
        if "selection_code" in sample:
            assert isinstance(sample["selection_code"], str)


def validate_data_table(endpoint: str, data: Dict[str, Any]):
    assert "columns" in data and isinstance(data["columns"], list)
    assert "rows" in data and isinstance(data["rows"], list)


# Generic light validators for endpoints with variable data
def validate_generic_dict(endpoint: str, data: Dict[str, Any]):
    assert isinstance(data, dict), "Response not a JSON object"


def validate_placeholder_svg(endpoint: str, raw_text: str):  # special non-JSON
    assert "<svg" in raw_text.lower(), "SVG tag missing in placeholder response"


# Feedback/Sentiment simple validators (structure may vary)
def validate_feedback(endpoint: str, data: Dict[str, Any]):  # optional
    if isinstance(data, dict):
        # Accept presence of run_id / feedback status or message keys
        pass


def validate_sentiment(endpoint: str, data: Dict[str, Any]):
    if isinstance(data, dict):
        pass


# Debug endpoints: just ensure dict if JSON body
validate_debug = validate_generic_dict

# ---------------------------------------------------------------------------
# HTTP TEST DEFINITIONS (ALL ENDPOINTS)
# ---------------------------------------------------------------------------
# Each test entry supports:
# method: HTTP method
# endpoint: path template (may include {thread_id} etc.)
# path_params: dict for formatting template
# params: query params
# json: JSON body for POST
# expect_status: list of acceptable status codes
# validator: function accepting (endpoint, data) for JSON responses
# expect_json: bool (default True) - if False, treat as text response
# description: human readable description
# skip_auth: if True, do not include Authorization header
#
DUMMY_THREAD_ID = os.environ.get("TEST_THREAD_ID", "dummy-thread")
DUMMY_RUN_ID = os.environ.get("TEST_RUN_ID", "dummy-run")
DUMMY_WIDTH = 120
DUMMY_HEIGHT = 80

HTTP_TESTS: List[Dict[str, Any]] = [
    # Health endpoints
    {
        "method": "GET",
        "endpoint": "/health",
        "description": "General health check",
        "validator": validate_health,
        "expect_status": [200, 503],
    },
    {
        "method": "GET",
        "endpoint": "/health/database",
        "description": "Database health",
        "validator": validate_health_database,
        "expect_status": [200, 503],
    },
    {
        "method": "GET",
        "endpoint": "/health/memory",
        "description": "Memory health",
        "validator": validate_health_memory,
        "expect_status": [200],
    },
    {
        "method": "GET",
        "endpoint": "/health/rate-limits",
        "description": "Rate limits health",
        "validator": validate_rate_limits,
        "expect_status": [200],
    },
    {
        "method": "GET",
        "endpoint": "/health/prepared-statements",
        "description": "Prepared statements health",
        "validator": validate_prepared_statements,
        "expect_status": [200, 503],
    },
    # Catalog endpoints
    {
        "method": "GET",
        "endpoint": "/catalog",
        "description": "Catalog page 1",
        "params": {"page": 1, "page_size": 5},
        "validator": validate_catalog,
        "expect_status": [200],
    },
    {
        "method": "GET",
        "endpoint": "/data-tables",
        "description": "List data tables",
        "validator": validate_data_tables,
        "expect_status": [200],
    },
    {
        "method": "GET",
        "endpoint": "/data-table",
        "description": "Empty table query",
        "validator": validate_data_table,
        "expect_status": [200],
    },
    # Analysis
    {
        "method": "POST",
        "endpoint": "/analyze",
        "description": "Analyze minimal payload",
        "json": {"prompt": "test", "mode": "quick"},
        "validator": validate_generic_dict,
        "expect_status": [200, 400, 422],
    },
    # Bulk (removed 500 to always treat 5xx as failure)
    {
        "method": "GET",
        "endpoint": "/chat/all-messages-for-all-threads",
        "description": "Bulk messages",
        "validator": validate_generic_dict,
        "expect_status": [200, 503],
    },
    # Chat/messages endpoints with thread_id
    {
        "method": "GET",
        "endpoint": "/chat/{thread_id}/messages",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Thread messages",
        "validator": validate_generic_dict,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "GET",
        "endpoint": "/chat/{thread_id}/run-ids",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Thread run IDs",
        "validator": validate_generic_dict,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "GET",
        "endpoint": "/chat/{thread_id}/sentiments",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Thread sentiments",
        "validator": validate_generic_dict,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "GET",
        "endpoint": "/chat/all-messages-for-one-thread/{thread_id}",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Messages for one thread",
        "validator": validate_generic_dict,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "GET",
        "endpoint": "/chat-threads",
        "description": "Chat threads list",
        "params": {"page": 1, "limit": 5},
        "validator": validate_generic_dict,
        "expect_status": [200, 400],
    },
    {
        "method": "DELETE",
        "endpoint": "/chat/{thread_id}",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Delete thread",
        "validator": validate_generic_dict,
        "expect_status": [200, 404, 400, 422],
    },
    # Feedback endpoints
    {
        "method": "POST",
        "endpoint": "/feedback",
        "description": "Submit feedback",
        "json": {"run_id": DUMMY_RUN_ID, "feedback": True, "comment": "test"},
        "validator": validate_feedback,
        "expect_status": [200, 400, 401, 404, 422],
    },
    {
        "method": "POST",
        "endpoint": "/sentiment",
        "description": "Update sentiment",
        "json": {"run_id": DUMMY_RUN_ID, "sentiment": 1},
        "validator": validate_sentiment,
        "expect_status": [200, 400, 401, 404, 422],
    },
    # Debug endpoints
    {
        "method": "GET",
        "endpoint": "/debug/chat/{thread_id}/checkpoints",
        "path_params": {"thread_id": DUMMY_THREAD_ID},
        "description": "Debug checkpoints",
        "validator": validate_debug,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "GET",
        "endpoint": "/debug/pool-status",
        "description": "Debug pool status",
        "validator": validate_debug,
        "expect_status": [200, 503],
    },
    {
        "method": "GET",
        "endpoint": "/debug/run-id/{run_id}",
        "path_params": {"run_id": DUMMY_RUN_ID},
        "description": "Debug run id",
        "validator": validate_debug,
        "expect_status": [200, 404, 400, 422],
    },
    {
        "method": "POST",
        "endpoint": "/admin/clear-cache",
        "description": "Admin clear cache",
        "validator": validate_debug,
        "expect_status": [200, 401, 403],
    },
    {
        "method": "POST",
        "endpoint": "/admin/clear-prepared-statements",
        "description": "Admin clear prepared statements",
        "validator": validate_debug,
        "expect_status": [200, 401, 403],
    },
    {
        "method": "POST",
        "endpoint": "/debug/set-env",
        "description": "Set debug env",
        "json": {"TEST_FLAG": "1"},
        "validator": validate_debug,
        "expect_status": [200, 400, 401, 422],
    },
    {
        "method": "POST",
        "endpoint": "/debug/reset-env",
        "description": "Reset debug env",
        "json": {"TEST_FLAG": "1"},
        "validator": validate_debug,
        "expect_status": [200, 400, 401, 422],
    },
    # Misc placeholder (non-JSON)
    {
        "method": "GET",
        "endpoint": "/placeholder/{width}/{height}",
        "path_params": {"width": DUMMY_WIDTH, "height": DUMMY_HEIGHT},
        "description": "Placeholder SVG",
        "expect_status": [200],
        "expect_json": False,
        "validator_text": validate_placeholder_svg,
    },
]

# Update REQUIRED_ENDPOINTS to base endpoint patterns (without dynamic values)
REQUIRED_ENDPOINTS = set(
    {
        "/health",
        "/health/database",
        "/health/memory",
        "/health/rate-limits",
        "/health/prepared-statements",
        "/catalog",
        "/data-tables",
        "/data-table",
        "/analyze",
        "/chat/all-messages-for-all-threads",
        "/chat/{thread_id}/messages",
        "/chat/{thread_id}/run-ids",
        "/chat/{thread_id}/sentiments",
        "/chat/all-messages-for-one-thread/{thread_id}",
        "/chat-threads",
        "/chat/{thread_id}",
        "/feedback",
        "/sentiment",
        "/debug/chat/{thread_id}/checkpoints",
        "/debug/pool-status",
        "/debug/run-id/{run_id}",
        "/admin/clear-cache",
        "/admin/clear-prepared-statements",
        "/debug/set-env",
        "/debug/reset-env",
        "/placeholder/{width}/{height}",
    }
)


# ---------------------------------------------------------------------------
# HTTP Request Execution (generalized)
# ---------------------------------------------------------------------------
async def _run_http_endpoint_tests(results: BaseTestResults):
    token = create_test_jwt_token(TEST_EMAIL)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for idx, test_case in enumerate(HTTP_TESTS, start=1):
            method = test_case.get("method", "GET").upper()
            endpoint_template = test_case["endpoint"]
            path_params = test_case.get("path_params", {}) or {}
            try:
                endpoint = endpoint_template.format(**path_params)
            except KeyError as ke:
                print(f"❌ [{idx}] Path param missing for {endpoint_template}: {ke}")
                results.add_error(
                    f"ep-{idx}", endpoint_template, test_case.get("description"), ke
                )
                continue
            description = test_case.get("description", endpoint)
            params = test_case.get("params")
            json_body = test_case.get("json")
            expect_status: List[int] = test_case.get("expect_status", [200])
            expect_json = test_case.get("expect_json", True)
            validator: Optional[Validator] = test_case.get("validator")
            validator_text = test_case.get("validator_text")  # for non-json
            skip_auth = test_case.get("skip_auth", False)

            url = f"{SERVER_BASE_URL}{endpoint}"
            start = time.time()
            try:
                req_headers = {} if skip_auth else headers.copy()
                result = await make_request_with_traceback_capture(
                    client,
                    method,
                    url,
                    headers=req_headers,
                    params=params,
                    json=json_body,
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = time.time() - start
                error_info = extract_detailed_error_info(result)
                response = result.get("response")
                if response is None:
                    msg = error_info.get("client_error") or "No response received"
                    print(f"❌ [{idx}] {method} {endpoint} - Client error: {msg}")
                    err = Exception(msg)
                    results.add_error(
                        f"ep-{idx}", endpoint_template, description, err, elapsed
                    )
                    continue

                status = response.status_code
                # Treat any 5xx as failure, regardless of expect_status
                if status >= 500:
                    ok_status = False
                else:
                    ok_status = status in expect_status
                raw_text: Optional[str] = None
                data: Union[Dict[str, Any], List[Any], None] = None
                if expect_json:
                    try:
                        data = response.json()
                    except Exception:
                        data = None
                else:
                    raw_text = response.text

                if ok_status:
                    validation_failed = False
                    if expect_json and validator and isinstance(data, dict):
                        try:
                            validator(endpoint, data)  # type: ignore
                        except AssertionError as ae:
                            validation_failed = True
                            print(
                                f"❌ [{idx}] {method} {endpoint} validation failed: {ae}"
                            )
                            results.add_error(
                                f"ep-{idx}",
                                endpoint_template,
                                description,
                                ae,
                                elapsed,
                                response_data=data,
                            )
                    elif not expect_json and validator_text and raw_text is not None:
                        try:
                            validator_text(endpoint, raw_text)  # type: ignore
                        except AssertionError as ae:
                            validation_failed = True
                            print(
                                f"❌ [{idx}] {method} {endpoint} text validation failed: {ae}"
                            )
                            results.add_error(
                                f"ep-{idx}", endpoint_template, description, ae, elapsed
                            )

                    if not validation_failed:
                        results.add_result(
                            f"ep-{idx}",
                            endpoint_template,
                            description,
                            data if data is not None else {"raw": raw_text},
                            elapsed,
                            status,
                        )
                        print(
                            f"✅ [{idx}] {method} {endpoint} ({status}) {elapsed:.2f}s"
                        )
                else:
                    print(
                        f"❌ [{idx}] {method} {endpoint} unexpected status {status} (expected one of {expect_status} and not 5xx)"
                    )
                    # Treat unexpected as error
                    results.add_error(
                        f"ep-{idx}",
                        endpoint_template,
                        description,
                        Exception(f"HTTP {status}"),
                        elapsed,
                        response_data=data if data is not None else {"raw": raw_text},
                    )
            except Exception as e:
                elapsed = time.time() - start
                print(f"💥 [{idx}] {method} {endpoint} fatal error: {e}")
                results.add_error(
                    f"ep-{idx}", endpoint_template, description, e, elapsed
                )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_phase1_tests() -> BaseTestResults:
    print("🚀 Starting Phase 1 (Setup & Endpoint) tests...")

    if not await check_server_connectivity(SERVER_BASE_URL):
        print(
            "⚠️ Initial connectivity to /health failed (server may still start later)."
        )

    results = BaseTestResults(required_endpoints=REQUIRED_ENDPOINTS)
    results.start_time = datetime.now()

    await _local_folder_structure_check(results)
    await _local_init_files_check(results)
    await _local_import_check(results)

    if await check_server_connectivity(SERVER_BASE_URL):
        await _run_http_endpoint_tests(results)
    else:
        results.add_error(
            "connectivity",
            "http:server",
            "Server connectivity failed",
            Exception("Server unreachable"),
        )

    results.end_time = datetime.now()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def analyze_results(results: BaseTestResults) -> Dict[str, Any]:
    summary = results.get_summary()
    # Print improved headline listing coverage
    covered = {r["endpoint"] for r in results.results}
    missing = REQUIRED_ENDPOINTS - covered
    print("\n📊 Phase 1 Summary")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Success: {summary['successful_requests']}")
    print(f"  Failed: {summary['failed_requests']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    if missing:
        print(f"  Missing Endpoints (no success record): {sorted(missing)}")
    if summary["successful_requests"]:
        print(f"  Avg Response Time: {summary['average_response_time']:.2f}s")
    if not summary.get("all_endpoints_tested", True):
        print(f"  Missing endpoints: {summary.get('missing_endpoints')}")

    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)}):")
        for err in results.errors[:20]:  # limit output
            print(f"  - {err.get('test_id')} {err.get('endpoint')}: {err.get('error')}")
            if err.get("response_time") is not None:
                print(f"      ⏱ {err['response_time']:.2f}s")

    # Always write traceback report (empty if no errors)
    try:
        save_traceback_report(report_type="test_failure", test_results=results)
    except Exception as e:
        print(f"⚠️ Failed to write traceback report: {e}")

    return summary


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
async def main():
    results = await run_phase1_tests()
    summary = analyze_results(results)
    server_reachable = any(
        r for r in results.results if r["endpoint"].startswith("/health")
    )
    passed = summary["failed_requests"] == 0 and (
        not server_reachable or summary.get("all_endpoints_tested", True)
    )
    print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


if __name__ == "__main__":
    try:
        exit_code = 0 if asyncio.run(main()) else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
        sys.exit(1)

"""Test helpers and utilities for the test suite."""

import logging
import traceback
import inspect
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv

load_dotenv()

# CLEANED_TRACEBACK: 1=cleaned traceback (default), 0=full traceback
CLEANED_TRACEBACK = int(os.environ.get("CLEANED_TRACEBACK", "1"))


class BaseTestResults:
    """Base class to track and analyze endpoint test results."""

    def __init__(self, required_endpoints: set = None):
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        self.errors: List[Dict[str, Any]] = []
        self.required_endpoints = required_endpoints or set()

    def add_result(
        self,
        test_id: str,
        endpoint: str,
        description: str,
        response_data: Dict,
        response_time: float,
        status_code: int,
    ):
        """Add a test result."""
        result = {
            "test_id": test_id,
            "endpoint": endpoint,
            "description": description,
            "response_data": response_data,
            "response_time": response_time,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            # Both success and validation errors are valid outcomes
            "success": status_code in [200, 422],
        }
        self.results.append(result)

    def add_error(
        self,
        test_id: str,
        endpoint: str,
        description: str,
        error: Exception,
        response_time: float = None,
        response_data: dict = None,
    ):
        """Add an error result."""
        error_info = {
            "test_id": test_id,
            "endpoint": endpoint,
            "description": description,
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            # Store server response data (may include traceback)
            "error_obj": error,
            "response_data": response_data,
        }
        self.errors.append(error_info)
        print(f"❌ Test {test_id} failed: {str(error)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of test results."""
        total_requests = len(self.results) + len(self.errors)
        successful_requests = len([r for r in self.results if r["success"]])
        failed_requests = len(self.errors) + len(
            [r for r in self.results if not r["success"]]
        )

        if self.results:
            avg_response_time = sum(r["response_time"] for r in self.results) / len(
                self.results
            )
            max_response_time = max(r["response_time"] for r in self.results)
            min_response_time = min(r["response_time"] for r in self.results)
        else:
            avg_response_time = max_response_time = min_response_time = 0

        total_test_time = None
        if self.start_time and self.end_time:
            total_test_time = (self.end_time - self.start_time).total_seconds()

        tested_endpoints = set(r["endpoint"] for r in self.results if r["success"])

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            ),
            "average_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "total_test_time": total_test_time,
            "errors": self.errors,
            "all_endpoints_tested": tested_endpoints.issuperset(
                self.required_endpoints
            ),
            "tested_endpoints": tested_endpoints,
            "missing_endpoints": self.required_endpoints - tested_endpoints,
        }


def handle_error_response(
    test_id: str,
    endpoint: str,
    description: str,
    response,
    error_info: dict,
    results,  # Can be BaseTestResults or any compatible results object
    response_time: float,
):
    """Handle error responses for expected success cases."""
    try:
        error_data = response.json()
        error_message = error_data.get(
            "detail", f"HTTP {response.status_code}: {response.text}"
        )
    except Exception:
        error_data = None
        error_message = f"HTTP {response.status_code}: {response.text}"

    print(f"❌ Test {test_id} - Expected success but got HTTP {response.status_code}")
    error_obj = Exception(f"Expected success but got: {error_message}")
    error_obj.server_tracebacks = error_info["server_tracebacks"]
    results.add_error(
        test_id,
        endpoint,
        description,
        error_obj,
        response_time,
        response_data=error_data,
    )


def handle_expected_failure(
    test_id: str,
    endpoint: str,
    description: str,
    response,
    error_info: dict,
    results,  # Can be BaseTestResults or any compatible results object
    response_time: float,
    expected_status: int = None,
):
    """Handle responses for expected failure cases."""
    if expected_status:
        # Use specific expected status code
        if response.status_code == expected_status:
            print(f"✅ Test {test_id} - Correctly failed with HTTP {expected_status}")
            data = {"expected_failure": True, "status_code": expected_status}
            results.add_result(
                test_id,
                endpoint,
                description,
                data,
                response_time,
                response.status_code,
            )
        elif response.status_code == 200:
            print(
                f"❌ Test {test_id} - Expected HTTP {expected_status} but got success"
            )
            error_obj = Exception(
                f"Expected HTTP {expected_status} but request succeeded"
            )
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)
        else:
            print(
                f"❌ Test {test_id} - Expected HTTP {expected_status} but got HTTP {response.status_code}"
            )
            error_obj = Exception(
                f"Expected HTTP {expected_status} but got HTTP {response.status_code}"
            )
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)
    else:
        # Default behavior - expect validation error (422)
        if response.status_code == 422:  # Validation error
            print(f"✅ Test {test_id} - Correctly failed with validation error")
            data = {"validation_error": True}
            results.add_result(
                test_id,
                endpoint,
                description,
                data,
                response_time,
                response.status_code,
            )
        elif response.status_code == 200:
            print(f"❌ Test {test_id} - Expected validation error but got success")
            error_obj = Exception("Expected validation error but request succeeded")
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)
        else:
            print(
                f"❌ Test {test_id} - Expected validation error but got HTTP {response.status_code}"
            )
            error_obj = Exception(
                f"Expected validation error but got HTTP {response.status_code}"
            )
            error_obj.server_tracebacks = error_info["server_tracebacks"]
            results.add_error(test_id, endpoint, description, error_obj, response_time)


async def check_server_connectivity(
    base_url: str = "http://localhost:8000", timeout: float = 10.0
) -> bool:
    """Check if the server is running and accessible."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Server is accessible")
                return True
            else:
                print(f"❌ Server responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False


async def setup_debug_environment(client: httpx.AsyncClient, **debug_vars) -> bool:
    """Setup debug environment with configurable debug variables."""
    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = await client.post("/debug/set-env", headers=headers, json=debug_vars)
        return response.status_code == 200
    except Exception:
        return False


async def cleanup_debug_environment(client: httpx.AsyncClient, **debug_vars) -> bool:
    """Reset debug environment with configurable debug variables."""
    token = create_test_jwt_token()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = await client.post(
            "/debug/reset-env", headers=headers, json=debug_vars
        )
        return response.status_code == 200
    except Exception:
        return False


def create_test_jwt_token(email="test_user@example.com"):
    """Create a simple test JWT token for authentication."""
    try:
        import jwt

        google_client_id = (
            "722331814120-9kdm64s2mp9cq8kig0mvrluf1eqkso74.apps.googleusercontent.com"
        )
        payload = {
            "email": email,
            "aud": google_client_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "test_issuer",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
        }
        token = jwt.encode(payload, "test_secret", algorithm="HS256")
        return token
    except ImportError:
        return "test_token_placeholder"


class ServerLogCapture:
    """Captures server-side logs and tracebacks during testing."""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.tracebacks: List[Dict[str, Any]] = []

    def add_log_entry(self, level: str, message: str, exc_info=None):
        """Add a log entry with optional exception info."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "exc_info": exc_info,
        }
        self.logs.append(entry)

        if exc_info and level in ["ERROR", "CRITICAL"]:
            tb_lines = traceback.format_exception(*exc_info)
            self.tracebacks.append(
                {
                    "timestamp": entry["timestamp"],
                    "level": level,
                    "message": message,
                    "traceback": "".join(tb_lines),
                    "exception_type": (
                        exc_info[0].__name__ if exc_info[0] else "Unknown"
                    ),
                    "exception_message": str(exc_info[1]) if exc_info[1] else "Unknown",
                }
            )

    def get_captured_tracebacks(self) -> List[Dict[str, Any]]:
        return self.tracebacks

    def get_captured_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def clear(self):
        self.logs.clear()
        self.tracebacks.clear()


class CustomLogHandler(logging.Handler):
    """Custom log handler that captures logs and tracebacks."""

    def __init__(self, log_capture: ServerLogCapture):
        super().__init__()
        self.log_capture = log_capture

    def emit(self, record):
        try:
            message = self.format(record)
            exc_info = record.exc_info if record.exc_info else None
            self.log_capture.add_log_entry(
                level=record.levelname, message=message, exc_info=exc_info
            )
        except Exception:
            pass


@contextmanager
def capture_server_logs():
    """Context manager to capture server-side logs and tracebacks."""
    log_capture = ServerLogCapture()
    custom_handler = CustomLogHandler(log_capture)
    custom_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    loggers_to_capture = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "starlette",
        "app",
        "root",
        "",
    ]
    original_handlers = {}

    try:
        for logger_name in loggers_to_capture:
            logger = logging.getLogger(logger_name)
            original_handlers[logger_name] = logger.handlers.copy()
            logger.addHandler(custom_handler)
            if logger.level > logging.DEBUG:
                logger.setLevel(logging.DEBUG)
        yield log_capture
    finally:
        for logger_name, handlers in original_handlers.items():
            logger = logging.getLogger(logger_name)
            logger.removeHandler(custom_handler)


async def make_request_with_traceback_capture(
    client: httpx.AsyncClient, method: str, url: str, **kwargs
) -> Dict[str, Any]:
    """Make an HTTP request and capture any server-side tracebacks."""
    with capture_server_logs() as log_capture:
        try:
            response = await client.request(method, url, **kwargs)
            return {
                "response": response,
                "server_logs": log_capture.get_captured_logs(),
                "server_tracebacks": log_capture.get_captured_tracebacks(),
                "success": True,
            }
        except Exception as e:
            return {
                "response": None,
                "server_logs": log_capture.get_captured_logs(),
                "server_tracebacks": log_capture.get_captured_tracebacks(),
                "client_exception": e,
                "success": False,
            }


def extract_detailed_error_info(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed error information from a request result."""
    error_info = {
        "has_server_errors": bool(result["server_tracebacks"]),
        "has_client_errors": not result["success"]
        or bool(result.get("client_exception")),
        "server_tracebacks": result["server_tracebacks"],
        "server_error_messages": [],
        "client_error": (
            str(result.get("client_exception", ""))
            if result.get("client_exception")
            else None
        ),
        "http_status": result["response"].status_code if result["response"] else None,
        "response_body": None,
    }

    for tb in result["server_tracebacks"]:
        error_info["server_error_messages"].append(
            {
                "timestamp": tb["timestamp"],
                "level": tb["level"],
                "message": tb["message"],
                "exception_type": tb["exception_type"],
                "exception_message": tb["exception_message"],
            }
        )

    if result["response"]:
        try:
            error_info["response_body"] = result["response"].text
        except Exception:
            error_info["response_body"] = "<Could not decode response body>"

    return error_info


def _create_report_header(
    report_type: str, test_file_name: str, timestamp: str
) -> List[str]:
    return [
        "=" * 100,
        f"{report_type.upper()} REPORT",
        f"Generated: {timestamp}",
        f"Test File: {test_file_name}",
        "=" * 100,
        "",
    ]


def _clean_traceback(traceback_text: str) -> str:
    """Clean up verbose traceback by removing redundant layers and focusing on core error."""
    if not traceback_text:
        return traceback_text

    lines = traceback_text.split("\n")
    cleaned_lines = []

    # Find app code references
    app_code_indices = []
    for i, line in enumerate(lines):
        if (
            "api\\routes\\" in line
            or "api/routes/" in line
            or "my_agent/" in line
            or "czsu-multi-agent-text-to-sql" in line
            and "site-packages" not in line
        ):
            app_code_indices.append(i)

    if app_code_indices:
        start_idx = max(0, app_code_indices[-1] - 3)
        error_line_idx = None
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if (
                (": name " in line and "is not defined" in line)
                or ("Error:" in line)
                or ("Exception:" in line)
                or (line.endswith("Error") and not line.startswith("  "))
            ):
                error_line_idx = i
                break

        if error_line_idx:
            context_start = max(0, error_line_idx - 10)
            context_end = min(len(lines), error_line_idx + 3)
            cleaned_lines = lines[context_start:context_end]
        else:
            cleaned_lines = lines[start_idx : start_idx + 15]
    else:
        # Look for core error message
        for i, line in enumerate(lines):
            if (
                "NameError:" in line
                or "TypeError:" in line
                or "ValueError:" in line
                or "AttributeError:" in line
            ):
                context_start = max(0, i - 5)
                context_end = min(len(lines), i + 3)
                cleaned_lines = lines[context_start:context_end]
                break

    if not cleaned_lines:
        # Remove "During handling" sections
        meaningful_lines = []
        skip_section = False
        for line in lines:
            if (
                "During handling of the above exception, another exception occurred:"
                in line
            ):
                skip_section = True
                continue
            elif (
                line.strip() and not line.startswith(" ") and not line.startswith("\t")
            ):
                skip_section = False
            if not skip_section:
                meaningful_lines.append(line)
        cleaned_lines = meaningful_lines[-20:] if meaningful_lines else lines[-20:]

    # Remove empty lines at beginning and end
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def _write_report_to_file(file_path: Path, content_lines: List[str], report_type: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content_lines))
        return True
    except Exception as e:
        print(f"❌ Failed to save {report_type.lower()} report: {e}")
        return False


def save_traceback_report(
    report_type: str = "test_failure",
    test_results: Any = None,
    exception: Exception = None,
    server_tracebacks: Optional[List[Dict[str, Any]]] = None,
    test_context: Optional[Dict[str, Any]] = None,
):
    """Save traceback reports. Creates empty file if no errors, detailed report if errors exist."""
    # Get file path
    frame = inspect.currentframe()
    try:
        test_file_name = Path(frame.f_back.f_code.co_filename).name
    except:
        test_file_name = "unknown_test_file.py"
    finally:
        del frame

    traceback_dir = Path("tests/traceback_errors")
    traceback_dir.mkdir(exist_ok=True)
    traceback_file = (
        traceback_dir / f"{Path(test_file_name).stem}_{report_type}_report.txt"
    )

    # Check for any errors/failures
    has_errors = (
        exception
        or server_tracebacks
        or (test_results and getattr(test_results, "errors", None))
        or (
            test_results
            and getattr(test_results, "results", None)
            and any(not r.get("success", True) for r in test_results.results)
        )
    )

    # Empty file if no errors, detailed report if errors
    if not has_errors:
        traceback_file.write_text("", encoding="utf-8")
        return True

    # Build detailed report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content_lines = _create_report_header(report_type, test_file_name, timestamp)

    if test_context:
        content_lines.extend(["TEST CONTEXT:", "-" * 30])
        for key, value in test_context.items():
            content_lines.append(f"{key}: {value}")
        content_lines.extend(["", ""])

    if report_type == "test_failure" and test_results:
        _add_test_failure_content(content_lines, test_results)
    elif report_type == "exception" and exception:
        _add_exception_content(content_lines, exception)
    elif report_type == "server_traceback" and server_tracebacks:
        _add_server_traceback_content(content_lines, server_tracebacks)

    if test_results and hasattr(test_results, "get_summary"):
        _add_test_summary(content_lines, test_results.get_summary())

    return _write_report_to_file(traceback_file, content_lines, report_type)


def _add_test_failure_content(content_lines: List[str], test_results: Any):
    """Add test failure specific content to the report."""
    if hasattr(test_results, "errors") and test_results.errors:
        content_lines.extend(
            [f"TOTAL FAILED TESTS: {len(test_results.errors)}", "=" * 100, ""]
        )

        for i, error in enumerate(test_results.errors, 1):
            content_lines.extend(
                [
                    f"FAILED TEST #{i}",
                    "=" * 60,
                    f"Test ID: {error.get('test_id', 'Unknown')}",
                    f"Endpoint: {error.get('endpoint', 'Unknown')}",
                    f"Description: {error.get('description', 'Unknown')}",
                    f"Error Type: {error.get('error_type', 'Unknown')}",
                    f"Timestamp: {error.get('timestamp', 'Unknown')}",
                    "",
                ]
            )

            if error.get("response_time"):
                content_lines.extend(
                    [f"Response Time: {error['response_time']:.2f}s", ""]
                )

            # Add server traceback from response_data
            has_response_traceback = False
            if "response_data" in error and isinstance(error["response_data"], dict):
                traceback_text = error["response_data"].get("traceback")
                if traceback_text:
                    has_response_traceback = True
                    if CLEANED_TRACEBACK == 1:
                        cleaned_traceback = _clean_traceback(traceback_text)
                        content_lines.extend(
                            [
                                "SERVER TRACEBACK (CLEANED):",
                                "-" * 40,
                                cleaned_traceback,
                                "",
                            ]
                        )
                    else:
                        content_lines.extend(
                            [
                                "SERVER TRACEBACK (FULL ORIGINAL):",
                                "-" * 40,
                                traceback_text,
                                "",
                            ]
                        )

            server_tracebacks = error.get("server_tracebacks", [])
            if server_tracebacks and not has_response_traceback:
                content_lines.extend(["SERVER-SIDE TRACEBACKS:", "-" * 40, ""])
                for j, tb in enumerate(server_tracebacks, 1):
                    content_lines.extend(
                        [
                            f"SERVER TRACEBACK #{j}:",
                            f"Timestamp: {tb.get('timestamp', 'Unknown')}",
                            f"Level: {tb.get('level', 'Unknown')}",
                            f"Exception Type: {tb.get('exception_type', 'Unknown')}",
                            f"Exception Message: {tb.get('exception_message', 'Unknown')}",
                            "",
                            "FULL SERVER TRACEBACK:",
                            "-" * 25,
                            tb.get("traceback", "No traceback available"),
                            "",
                            "~" * 60,
                            "",
                        ]
                    )

            if i < len(test_results.errors):
                content_lines.extend(["~" * 80, ""])

    # Handle failed results
    if hasattr(test_results, "results"):
        failed_results = [r for r in test_results.results if not r.get("success", True)]
        if failed_results:
            content_lines.extend(
                [
                    "",
                    "FAILED RESULT DETAILS:",
                    "=" * 60,
                    f"Total Failed Results: {len(failed_results)}",
                    "",
                ]
            )

            for i, result in enumerate(failed_results, 1):
                content_lines.extend(
                    [
                        f"FAILED RESULT #{i}",
                        "-" * 40,
                        f"Test ID: {result.get('test_id', 'Unknown')}",
                        f"Endpoint: {result.get('endpoint', 'Unknown')}",
                        f"Description: {result.get('description', 'Unknown')}",
                        f"Status Code: {result.get('status_code', 'Unknown')}",
                        f"Response Time: {result.get('response_time', 'Unknown')}s",
                        f"Timestamp: {result.get('timestamp', 'Unknown')}",
                        "",
                    ]
                )

                if result.get("response_data"):
                    content_lines.extend(
                        ["RESPONSE DATA:", "-" * 20, str(result["response_data"]), ""]
                    )

                if i < len(failed_results):
                    content_lines.extend(["~" * 60, ""])


def _add_exception_content(content_lines: List[str], exception: Exception):
    """Add exception specific content to the report."""
    content_lines.extend(
        [
            "EXCEPTION INFORMATION:",
            "-" * 40,
            f"Exception Type: {type(exception).__name__}",
            f"Exception Message: {str(exception)}",
            "",
            "FULL TRACEBACK:",
            "-" * 30,
        ]
    )
    tb_lines = traceback.format_exception(
        type(exception), exception, exception.__traceback__
    )
    content_lines.extend(tb_lines)


def _add_server_traceback_content(
    content_lines: List[str], server_tracebacks: List[Dict[str, Any]]
):
    """Add server traceback specific content to the report."""
    if server_tracebacks:
        content_lines.extend(
            [f"TOTAL SERVER TRACEBACKS: {len(server_tracebacks)}", "=" * 100, ""]
        )

        for i, tb in enumerate(server_tracebacks, 1):
            content_lines.extend(
                [
                    f"SERVER TRACEBACK #{i}",
                    "=" * 60,
                    f"Timestamp: {tb.get('timestamp', 'Unknown')}",
                    f"Level: {tb.get('level', 'Unknown')}",
                    f"Message: {tb.get('message', 'Unknown')}",
                    f"Exception Type: {tb.get('exception_type', 'Unknown')}",
                    f"Exception Message: {tb.get('exception_message', 'Unknown')}",
                    "",
                    "FULL SERVER TRACEBACK:",
                    "-" * 40,
                    tb.get("traceback", "No traceback available"),
                    "",
                ]
            )

            if i < len(server_tracebacks):
                content_lines.extend(["~" * 80, ""])
    else:
        content_lines.extend(
            [
                "NO SERVER TRACEBACKS CAPTURED",
                "=" * 50,
                "This could mean:",
                "- No server errors occurred",
                "- Server logging was not properly captured",
                "- Errors occurred but were not logged at the expected level",
                "",
            ]
        )


def _add_test_summary(content_lines: List[str], summary: Dict[str, Any]):
    """Add test summary to the report."""
    content_lines.extend(
        [
            "",
            "TEST SUMMARY:",
            "=" * 40,
            f"Total Requests: {summary.get('total_requests', 'Unknown')}",
            f"Successful Requests: {summary.get('successful_requests', 'Unknown')}",
            f"Failed Requests: {summary.get('failed_requests', 'Unknown')}",
            f"Success Rate: {summary.get('success_rate', 'Unknown')}%",
            f"Average Response Time: {summary.get('average_response_time', 'Unknown')}s",
            f"Total Test Time: {summary.get('total_test_time', 'Unknown')}s",
            "",
        ]
    )

    if summary.get("missing_endpoints"):
        content_lines.extend(
            [f"Missing Endpoints: {', '.join(summary['missing_endpoints'])}", ""]
        )

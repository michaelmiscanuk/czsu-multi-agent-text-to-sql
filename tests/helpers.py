#!/usr/bin/env python3
"""
Test helpers and utilities for the test suite.
"""

import asyncio
import io
import logging
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from unittest.mock import patch

import httpx


class ServerLogCapture:
    """Captures server-side logs and tracebacks during testing."""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.tracebacks: List[Dict[str, Any]] = []
        self.original_handlers = {}
        self.log_buffer = io.StringIO()
        self.error_buffer = io.StringIO()
        
    def add_log_entry(self, level: str, message: str, exc_info=None):
        """Add a log entry with optional exception info."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "exc_info": exc_info
        }
        
        self.logs.append(entry)
        
        # If this is an error with exception info, capture the traceback
        if exc_info and level in ["ERROR", "CRITICAL"]:
            tb_lines = traceback.format_exception(*exc_info)
            self.tracebacks.append({
                "timestamp": entry["timestamp"],
                "level": level,
                "message": message,
                "traceback": "".join(tb_lines),
                "exception_type": exc_info[0].__name__ if exc_info[0] else "Unknown",
                "exception_message": str(exc_info[1]) if exc_info[1] else "Unknown"
            })
            
    def get_captured_tracebacks(self) -> List[Dict[str, Any]]:
        """Get all captured tracebacks."""
        return self.tracebacks
        
    def get_captured_logs(self) -> List[Dict[str, Any]]:
        """Get all captured logs."""
        return self.logs
        
    def clear(self):
        """Clear all captured data."""
        self.logs.clear()
        self.errors.clear()
        self.tracebacks.clear()
        self.log_buffer.seek(0)
        self.log_buffer.truncate(0)
        self.error_buffer.seek(0)
        self.error_buffer.truncate(0)


class CustomLogHandler(logging.Handler):
    """Custom log handler that captures logs and tracebacks."""
    
    def __init__(self, log_capture: ServerLogCapture):
        super().__init__()
        self.log_capture = log_capture
        
    def emit(self, record):
        try:
            # Format the message
            message = self.format(record)
            
            # Capture exception info if present
            exc_info = None
            if record.exc_info:
                exc_info = record.exc_info
                
            # Add to log capture
            self.log_capture.add_log_entry(
                level=record.levelname,
                message=message,
                exc_info=exc_info
            )
            
        except Exception:
            # Don't let logging errors break the application
            pass


@contextmanager
def capture_server_logs():
    """Context manager to capture server-side logs and tracebacks."""
    log_capture = ServerLogCapture()
    custom_handler = CustomLogHandler(log_capture)
    
    # Configure the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    custom_handler.setFormatter(formatter)
    
    # Get all relevant loggers
    loggers_to_capture = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "starlette",
        "app",
        "root",
        "",  # Root logger
    ]
    
    original_handlers = {}
    
    try:
        # Add our custom handler to all relevant loggers
        for logger_name in loggers_to_capture:
            logger = logging.getLogger(logger_name)
            original_handlers[logger_name] = logger.handlers.copy()
            logger.addHandler(custom_handler)
            
            # Ensure we capture all levels
            if logger.level > logging.DEBUG:
                logger.setLevel(logging.DEBUG)
        
        yield log_capture
        
    finally:
        # Restore original handlers
        for logger_name, handlers in original_handlers.items():
            logger = logging.getLogger(logger_name)
            logger.removeHandler(custom_handler)


async def make_request_with_traceback_capture(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Make an HTTP request and capture any server-side tracebacks.
    
    Returns a dictionary with:
    - response: The HTTP response object
    - server_logs: List of captured server logs
    - server_tracebacks: List of captured server tracebacks
    """
    
    with capture_server_logs() as log_capture:
        try:
            response = await client.request(method, url, **kwargs)
            
            return {
                "response": response,
                "server_logs": log_capture.get_captured_logs(),
                "server_tracebacks": log_capture.get_captured_tracebacks(),
                "success": True
            }
            
        except Exception as e:
            return {
                "response": None,
                "server_logs": log_capture.get_captured_logs(),
                "server_tracebacks": log_capture.get_captured_tracebacks(),
                "client_exception": e,
                "success": False
            }


def extract_detailed_error_info(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract detailed error information from a request result.
    
    Args:
        result: Result from make_request_with_traceback_capture
        
    Returns:
        Dictionary with detailed error information
    """
    error_info = {
        "has_server_errors": False,
        "has_client_errors": False,
        "server_tracebacks": [],
        "server_error_messages": [],
        "client_error": None,
        "http_status": None,
        "response_body": None
    }
    
    # Check for client-side errors
    if not result["success"] or result.get("client_exception"):
        error_info["has_client_errors"] = True
        error_info["client_error"] = str(result.get("client_exception", "Unknown client error"))
    
    # Check for server-side errors
    if result["server_tracebacks"]:
        error_info["has_server_errors"] = True
        error_info["server_tracebacks"] = result["server_tracebacks"]
        
        # Extract error messages from tracebacks
        for tb in result["server_tracebacks"]:
            error_info["server_error_messages"].append({
                "timestamp": tb["timestamp"],
                "level": tb["level"],
                "message": tb["message"],
                "exception_type": tb["exception_type"],
                "exception_message": tb["exception_message"]
            })
    
    # Get HTTP response info
    if result["response"]:
        error_info["http_status"] = result["response"].status_code
        try:
            error_info["response_body"] = result["response"].text
        except Exception:
            error_info["response_body"] = "<Could not decode response body>"
    
    return error_info


def save_test_failures_traceback(
    test_file_name: str,
    test_results: Any,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save traceback information for failed tests to a file.
    
    Args:
        test_file_name: Name of the test file (e.g., "test_phase8_catalog.py")
        test_results: Test results object containing errors
        additional_info: Optional additional information to include
    """
    # Create traceback_errors directory if it doesn't exist
    traceback_dir = Path("tests/traceback_errors")
    traceback_dir.mkdir(exist_ok=True)
    
    # Generate filename based on test file name
    base_name = Path(test_file_name).stem  # Remove .py extension
    traceback_file = traceback_dir / f"{base_name}_traceback_failures.txt"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare content to write
    content_lines = []
    
    # Header
    content_lines.extend([
        "=" * 100,
        f"TEST FAILURE TRACEBACK REPORT",
        f"Generated: {timestamp}",
        f"Test File: {test_file_name}",
        "=" * 100,
        ""
    ])
    
    # Add additional info if provided
    if additional_info:
        content_lines.extend([
            "ADDITIONAL TEST INFORMATION:",
            "-" * 50,
        ])
        for key, value in additional_info.items():
            content_lines.append(f"{key}: {value}")
        content_lines.extend(["", ""])
    
    # Check if test_results has errors
    if hasattr(test_results, 'errors') and test_results.errors:
        content_lines.extend([
            f"TOTAL FAILED TESTS: {len(test_results.errors)}",
            "=" * 100,
            ""
        ])
        
        # Process each error
        for i, error in enumerate(test_results.errors, 1):
            content_lines.extend([
                f"FAILED TEST #{i}",
                "=" * 60,
                f"Test ID: {error.get('test_id', 'Unknown')}",
                f"Endpoint: {error.get('endpoint', 'Unknown')}",
                f"Description: {error.get('description', 'Unknown')}",
                f"Error Type: {error.get('error_type', 'Unknown')}",
                f"Timestamp: {error.get('timestamp', 'Unknown')}",
                ""
            ])
            
            if error.get('response_time'):
                content_lines.append(f"Response Time: {error['response_time']:.2f}s")
                content_lines.append("")
            
            # Add the actual error message
            content_lines.extend([
                "ERROR MESSAGE:",
                "-" * 30,
                str(error.get('error', 'No error message available')),
                ""
            ])

            # Add server traceback from response_data if present
            traceback_text = None
            if 'response_data' in error and isinstance(error['response_data'], dict):
                traceback_text = error['response_data'].get('traceback')
            if traceback_text:
                content_lines.extend([
                    "SERVER TRACEBACK FROM RESPONSE:",
                    "-" * 40,
                    traceback_text,
                    ""
                ])
            
            # Check if this error has detailed server traceback info
            if hasattr(error, 'server_tracebacks') or 'server_tracebacks' in error:
                server_tracebacks = error.get('server_tracebacks', [])
                if server_tracebacks:
                    content_lines.extend([
                        "SERVER-SIDE TRACEBACKS:",
                        "-" * 40,
                        ""
                    ])
                    
                    for j, tb in enumerate(server_tracebacks, 1):
                        content_lines.extend([
                            f"SERVER TRACEBACK #{j}:",
                            f"Timestamp: {tb.get('timestamp', 'Unknown')}",
                            f"Level: {tb.get('level', 'Unknown')}",
                            f"Exception Type: {tb.get('exception_type', 'Unknown')}",
                            f"Exception Message: {tb.get('exception_message', 'Unknown')}",
                            "",
                            "FULL SERVER TRACEBACK:",
                            "-" * 25,
                            tb.get('traceback', 'No traceback available'),
                            "",
                            "~" * 60,
                            ""
                        ])
            
            # Add separator between tests
            if i < len(test_results.errors):
                content_lines.extend([
                    "~" * 80,
                    ""
                ])
    
    # Check if test_results has failed results (different from errors)
    if hasattr(test_results, 'results'):
        failed_results = [r for r in test_results.results if not r.get('success', True)]
        if failed_results:
            content_lines.extend([
                "",
                "FAILED RESULT DETAILS:",
                "=" * 60,
                f"Total Failed Results: {len(failed_results)}",
                ""
            ])
            
            for i, result in enumerate(failed_results, 1):
                content_lines.extend([
                    f"FAILED RESULT #{i}",
                    "-" * 40,
                    f"Test ID: {result.get('test_id', 'Unknown')}",
                    f"Endpoint: {result.get('endpoint', 'Unknown')}",
                    f"Description: {result.get('description', 'Unknown')}",
                    f"Status Code: {result.get('status_code', 'Unknown')}",
                    f"Response Time: {result.get('response_time', 'Unknown')}s",
                    f"Timestamp: {result.get('timestamp', 'Unknown')}",
                    ""
                ])
                
                # Add response data if available
                if result.get('response_data'):
                    content_lines.extend([
                        "RESPONSE DATA:",
                        "-" * 20,
                        str(result['response_data']),
                        ""
                    ])
                
                # Add separator between results
                if i < len(failed_results):
                    content_lines.extend([
                        "~" * 60,
                        ""
                    ])
    
    # Add summary information
    if hasattr(test_results, 'get_summary'):
        summary = test_results.get_summary()
        content_lines.extend([
            "",
            "TEST SUMMARY:",
            "=" * 40,
            f"Total Requests: {summary.get('total_requests', 'Unknown')}",
            f"Successful Requests: {summary.get('successful_requests', 'Unknown')}",
            f"Failed Requests: {summary.get('failed_requests', 'Unknown')}",
            f"Success Rate: {summary.get('success_rate', 'Unknown')}%",
            f"Average Response Time: {summary.get('average_response_time', 'Unknown')}s",
            f"Total Test Time: {summary.get('total_test_time', 'Unknown')}s",
            ""
        ])
        
        if summary.get('missing_endpoints'):
            content_lines.extend([
                f"Missing Endpoints: {', '.join(summary['missing_endpoints'])}",
                ""
            ])
    
    # Add footer
    content_lines.extend([
        "=" * 100,
        f"End of traceback report for {test_file_name}",
        f"Generated: {timestamp}",
        "=" * 100
    ])
    
    # Write to file
    try:
        with open(traceback_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        print(f"\n📝 Traceback report saved to: {traceback_file}")
        print(f"📊 Report contains {len(test_results.errors) if hasattr(test_results, 'errors') else 0} error(s)")
        
    except Exception as e:
        print(f"❌ Failed to save traceback report: {e}")
        print(f"❌ Attempted to write to: {traceback_file}")


def save_exception_traceback(
    test_file_name: str,
    exception: Exception,
    test_context: Optional[Dict[str, Any]] = None
):
    """
    Save exception traceback information to a file.
    
    Args:
        test_file_name: Name of the test file
        exception: The exception that occurred
        test_context: Optional context information about the test
    """
    # Create traceback_errors directory if it doesn't exist
    traceback_dir = Path("tests/traceback_errors")
    traceback_dir.mkdir(exist_ok=True)
    
    # Generate filename based on test file name
    base_name = Path(test_file_name).stem
    traceback_file = traceback_dir / f"{base_name}_exception_traceback.txt"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare content to write
    content_lines = []
    
    # Header
    content_lines.extend([
        "=" * 100,
        f"EXCEPTION TRACEBACK REPORT",
        f"Generated: {timestamp}",
        f"Test File: {test_file_name}",
        "=" * 100,
        ""
    ])
    
    # Add test context if provided
    if test_context:
        content_lines.extend([
            "TEST CONTEXT:",
            "-" * 30,
        ])
        for key, value in test_context.items():
            content_lines.append(f"{key}: {value}")
        content_lines.extend(["", ""])
    
    # Add exception information
    content_lines.extend([
        "EXCEPTION INFORMATION:",
        "-" * 40,
        f"Exception Type: {type(exception).__name__}",
        f"Exception Message: {str(exception)}",
        "",
        "FULL TRACEBACK:",
        "-" * 30,
    ])
    
    # Add full traceback
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    content_lines.extend(tb_lines)
    
    # Add footer
    content_lines.extend([
        "",
        "=" * 100,
        f"End of exception traceback for {test_file_name}",
        f"Generated: {timestamp}",
        "=" * 100
    ])
    
    # Write to file
    try:
        with open(traceback_file, 'w', encoding='utf-8') as f:
            f.write(''.join(content_lines))
        
        print(f"\n📝 Exception traceback saved to: {traceback_file}")
        
    except Exception as e:
        print(f"❌ Failed to save exception traceback: {e}")
        print(f"❌ Attempted to write to: {traceback_file}")


def save_server_traceback_report(
    test_file_name: str,
    test_results: Any,
    server_tracebacks: List[Dict[str, Any]],
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save a comprehensive server traceback report.
    
    Args:
        test_file_name: Name of the test file
        test_results: Test results object
        server_tracebacks: List of captured server tracebacks
        additional_info: Optional additional information
    """
    # Create traceback_errors directory if it doesn't exist
    traceback_dir = Path("tests/traceback_errors")
    traceback_dir.mkdir(exist_ok=True)
    
    # Generate filename based on test file name
    base_name = Path(test_file_name).stem
    traceback_file = traceback_dir / f"{base_name}_server_tracebacks.txt"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare content to write
    content_lines = []
    
    # Header
    content_lines.extend([
        "=" * 100,
        f"SERVER TRACEBACK REPORT",
        f"Generated: {timestamp}",
        f"Test File: {test_file_name}",
        "=" * 100,
        ""
    ])
    
    # Add additional info if provided
    if additional_info:
        content_lines.extend([
            "ADDITIONAL TEST INFORMATION:",
            "-" * 50,
        ])
        for key, value in additional_info.items():
            content_lines.append(f"{key}: {value}")
        content_lines.extend(["", ""])
    
    # Add server tracebacks
    if server_tracebacks:
        content_lines.extend([
            f"TOTAL SERVER TRACEBACKS: {len(server_tracebacks)}",
            "=" * 100,
            ""
        ])
        
        for i, tb in enumerate(server_tracebacks, 1):
            content_lines.extend([
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
                tb.get('traceback', 'No traceback available'),
                ""
            ])
            
            # Add separator between tracebacks
            if i < len(server_tracebacks):
                content_lines.extend([
                    "~" * 80,
                    ""
                ])
    else:
        content_lines.extend([
            "NO SERVER TRACEBACKS CAPTURED",
            "=" * 50,
            "This could mean:",
            "- No server errors occurred",
            "- Server logging was not properly captured",
            "- Errors occurred but were not logged at the expected level",
            ""
        ])
    
    # Add test summary if available
    if hasattr(test_results, 'get_summary'):
        summary = test_results.get_summary()
        content_lines.extend([
            "",
            "TEST SUMMARY:",
            "=" * 40,
            f"Total Requests: {summary.get('total_requests', 'Unknown')}",
            f"Successful Requests: {summary.get('successful_requests', 'Unknown')}",
            f"Failed Requests: {summary.get('failed_requests', 'Unknown')}",
            f"Success Rate: {summary.get('success_rate', 'Unknown')}%",
            ""
        ])
    
    # Add footer
    content_lines.extend([
        "=" * 100,
        f"End of server traceback report for {test_file_name}",
        f"Generated: {timestamp}",
        "=" * 100
    ])
    
    # Write to file
    try:
        with open(traceback_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        print(f"\n📝 Server traceback report saved to: {traceback_file}")
        print(f"📊 Report contains {len(server_tracebacks)} server traceback(s)")
        
    except Exception as e:
        print(f"❌ Failed to save server traceback report: {e}")
        print(f"❌ Attempted to write to: {traceback_file}")

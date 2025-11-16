"""
MODULE_DESCRIPTION: Exception Handlers - Centralized Error Processing for FastAPI

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module provides centralized exception handling for the CZSU Multi-Agent
Text-to-SQL API. It defines custom exception handlers that intercept errors
throughout the application and convert them into properly formatted JSON responses
with appropriate HTTP status codes.

The handlers ensure consistent error response formats across all endpoints,
enhanced debugging for authentication failures, and proper error logging.

Exception Handler Types:
    1. validation_exception_handler: Pydantic validation errors → 422 responses
    2. http_exception_handler: HTTP exceptions → JSON with debug tracing
    3. value_error_handler: ValueError → 400 Bad Request
    4. general_exception_handler: Unexpected errors → 500 Internal Server Error

Key Benefits:
    - Consistent error response format across all endpoints
    - Enhanced debugging for authentication (401) errors
    - Prevents sensitive information leakage
    - Comprehensive logging for troubleshooting
    - Client-friendly error messages

===================================================================================
KEY FEATURES
===================================================================================

1. Pydantic Validation Error Handling
   - Intercepts RequestValidationError from Pydantic models
   - Returns 422 Unprocessable Entity (standard for validation)
   - Includes detailed field-level error information
   - Client-friendly error messages

2. HTTP Exception Processing
   - Handles Starlette/FastAPI HTTPException
   - Enhanced debugging for 401 Unauthorized errors
   - Logs request context (URL, method, headers, IP)
   - Returns JSON responses with proper status codes

3. ValueError Handling
   - Converts ValueError to 400 Bad Request
   - Useful for business logic validation
   - Clean error messages

4. Catch-All Exception Handler
   - Handles unexpected exceptions
   - Returns 500 Internal Server Error
   - Logs full error details
   - Prevents application crashes

5. Authentication Error Debugging
   - Special handling for 401 errors
   - Logs full request context
   - Tracks client IP for security
   - Full traceback for diagnosis

===================================================================================
EXCEPTION HANDLERS
===================================================================================

Handler 1: validation_exception_handler
    Purpose: Handle Pydantic validation errors

    Trigger:
        - Invalid request body fields
        - Missing required fields
        - Type validation failures
        - Custom validator failures

    Response:
        Status: 422 Unprocessable Entity
        Body: {
            "detail": "Validation error",
            "errors": [
                {
                    "loc": ["body", "field_name"],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }

    Example:
        POST /analyze with missing "prompt" field
        → 422 with detailed field errors

Handler 2: http_exception_handler
    Purpose: Handle HTTP exceptions with enhanced debugging

    Special Handling for 401 Errors:
        - Logs detailed request information
        - Tracks client IP address
        - Includes full traceback
        - Request URL, method, headers

    Response:
        Status: exc.status_code (401, 404, 500, etc.)
        Body: {
            "detail": "<error message>"
        }

    Debug Output (401 errors):
        🚨 HTTP 401 UNAUTHORIZED: <detail>
        🚨 HTTP 401 TRACE: Request URL: /analyze
        🚨 HTTP 401 TRACE: Request method: POST
        🚨 HTTP 401 TRACE: Request headers: {...}
        🚨 HTTP 401 TRACE: Exception detail: <detail>
        🚨 HTTP 401 TRACE: Full traceback: ...
        🚨 HTTP 401 CLIENT: IP address: 192.168.1.100

Handler 3: value_error_handler
    Purpose: Convert ValueError to 400 Bad Request

    Trigger:
        - Business logic validation failures
        - Invalid parameter values
        - Data processing errors

    Response:
        Status: 400 Bad Request
        Body: {
            "detail": "<error message>"
        }

    Example:
        raise ValueError("Invalid thread_id format")
        → 400 with error message

Handler 4: general_exception_handler
    Purpose: Catch-all for unexpected exceptions

    Trigger:
        - Unhandled exceptions
        - Programming errors
        - Library errors
        - Database connection failures

    Response:
        Status: 500 Internal Server Error
        Body: {
            "detail": "Internal server error"
        }

    Logging:
        - Exception type and message
        - Does NOT include full traceback (security)
        - Logs to console for debugging

===================================================================================
AUTHENTICATION ERROR DEBUGGING (401)
===================================================================================

Enhanced Debugging for 401 Errors:
    Why: Authentication errors are critical security events

    Logged Information:
        1. Error detail message
        2. Request URL (which endpoint failed)
        3. Request method (GET, POST, etc.)
        4. Request headers (including Authorization)
        5. Full exception traceback
        6. Client IP address

    Debug Functions Used:
        - print__analyze_debug(): Main debug output
        - print__analysis_tracing_debug(): Detailed tracing

    Environment Control:
        - ANALYZE_DEBUG=1: Enable debug logging
        - ANALYSIS_TRACING_DEBUG=1: Enable trace logging

Security Considerations:
    - Authorization headers logged only in debug mode
    - Full tracebacks only in development
    - Client IP tracked for security audits
    - Error messages generic to prevent info disclosure

===================================================================================
ERROR RESPONSE FORMAT
===================================================================================

Standard JSON Response:
    {
        "detail": "<error message>"
    }

Validation Error Response:
    {
        "detail": "Validation error",
        "errors": [
            {
                "loc": ["body", "field_name"],
                "msg": "field required",
                "type": "value_error.missing"
            }
        ]
    }

HTTP Status Codes:
    - 400 Bad Request: ValueError, invalid input
    - 401 Unauthorized: Authentication failures
    - 404 Not Found: Resource not found
    - 422 Unprocessable Entity: Validation errors
    - 500 Internal Server Error: Unexpected errors

===================================================================================
INTEGRATION WITH FASTAPI
===================================================================================

Registration in main.py:
    from api.exceptions.handlers import (
        validation_exception_handler,
        http_exception_handler,
        value_error_handler,
        general_exception_handler
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler
    )
    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler
    )
    app.add_exception_handler(
        ValueError,
        value_error_handler
    )
    app.add_exception_handler(
        Exception,
        general_exception_handler
    )

Handler Priority:
    - Most specific first (RequestValidationError)
    - HTTP exceptions second
    - Specific exceptions (ValueError) third
    - General Exception last (catch-all)

===================================================================================
DEBUG UTILITIES
===================================================================================

Debug Functions:
    print__debug(msg)
        - General purpose debug logging
        - Controlled by DEBUG environment variable

    print__analyze_debug(msg)
        - Analysis-specific debug logging
        - Controlled by ANALYZE_DEBUG environment variable
        - Used for authentication errors

    print__analysis_tracing_debug(msg)
        - Detailed tracing information
        - Controlled by ANALYSIS_TRACING_DEBUG environment variable
        - Used for request context logging

Logging Patterns:
    - Use emojis for visual distinction (🚨, 🔍, ✅, ❌)
    - Include context information
    - Full tracebacks for diagnosis
    - Structured information (URL, method, etc.)

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Information Disclosure Prevention
   - Generic error messages to clients
   - Detailed errors only in logs
   - No stack traces in production responses
   - No sensitive data in error messages

2. Authentication Monitoring
   - Track failed authentication attempts
   - Log client IP addresses
   - Full request context for security analysis
   - Detect brute force attacks

3. Header Logging
   - Authorization headers logged only in debug mode
   - Enable only in development
   - Disable in production to prevent token leakage

4. Rate Limiting Integration
   - Failed requests count toward rate limits
   - Prevents error-based attacks
   - Protects against enumeration attacks

===================================================================================
ERROR HANDLING BEST PRACTICES
===================================================================================

1. Use Specific Exceptions
   - Raise ValueError for business logic errors
   - Raise HTTPException for HTTP-specific errors
   - Let Pydantic handle validation automatically

2. Provide Context
   - Include relevant information in error messages
   - Use f-strings for dynamic messages
   - Example: f"Thread {thread_id} not found"

3. Log Appropriately
   - Log all exceptions for debugging
   - Include request context
   - Use appropriate log levels

4. Test Error Paths
   - Test each exception handler
   - Verify response format and status codes
   - Check logging output

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Overhead:
    - Minimal performance impact
    - Exception handlers only called on errors
    - JSON serialization is fast
    - Logging controlled by environment variables

Memory Usage:
    - Negligible per-request overhead
    - Traceback objects released after logging
    - No persistent storage

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Mock request objects
    - Mock exception objects
    - Verify response format and status codes
    - Test logging output

Test Examples:
    def test_validation_error_handler():
        from pydantic import ValidationError
        exc = RequestValidationError([...])
        response = await validation_exception_handler(request, exc)
        assert response.status_code == 422
        assert "errors" in response.body

    def test_http_exception_handler():
        exc = HTTPException(status_code=404, detail="Not found")
        response = await http_exception_handler(request, exc)
        assert response.status_code == 404
        assert response.body["detail"] == "Not found"

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection
    - traceback: Exception tracebacks

Third-Party:
    - fastapi: Request, RequestValidationError, JSONResponse
    - starlette.exceptions: HTTPException
    - dotenv: Environment variable loading

Internal:
    - api.utils.debug: Debug logging functions

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Structured Logging
   - JSON-formatted logs
   - Integration with log aggregation services
   - Structured error codes

2. Error Reporting
   - Sentry or similar error tracking
   - Automatic error categorization
   - Alerting for critical errors

3. Rate Limit Integration
   - Track error rates per IP
   - Block IPs with high error rates
   - Automatic threat detection

4. Custom Error Codes
   - Application-specific error codes
   - Error code documentation
   - Client-friendly error handling

===================================================================================
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import traceback

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import debug functions from utils
from api.utils.debug import (
    print__analysis_tracing_debug,
    print__analyze_debug,
    print__debug,
)

# ==============================================================================
# EXCEPTION HANDLERS
# ==============================================================================


async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with proper 422 status code.

    This handler intercepts RequestValidationError exceptions raised by Pydantic
    during request body/query parameter validation. It converts them into
    properly formatted JSON responses with detailed field-level error information.

    Args:
        _request: The FastAPI Request object (unused but required by signature)
        exc: The RequestValidationError exception from Pydantic

    Returns:
        JSONResponse with status 422 and detailed validation errors

    Response Format:
        {
            "detail": "Validation error",
            "errors": [
                {
                    "loc": ["body", "field_name"],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }
    """
    # Log validation error for debugging
    print__debug(f"Validation error: {exc.errors()}")

    # Return 422 Unprocessable Entity with detailed error information
    return JSONResponse(
        status_code=422, content={"detail": "Validation error", "errors": exc.errors()}
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with comprehensive debugging for 401 errors.

    This handler processes all HTTPException instances raised throughout the API,
    converting them into properly formatted JSON responses. It provides enhanced
    debugging for authentication failures (401 errors) with full request context.

    Args:
        request: The FastAPI Request object containing request context
        exc: The HTTPException (from Starlette/FastAPI)

    Returns:
        JSONResponse with the exception's status code and detail message

    Response Format:
        {
            "detail": "<error message from exception>"
        }

    Enhanced Debugging:
        - 401 errors: Full request context, headers, IP tracking
        - 4xx errors: Request URL, method, traceback
    """
    # =======================================================================
    # AUTHENTICATION ERROR DEBUGGING (401)
    # =======================================================================

    # Enhanced debugging for 401 errors since these are authentication-related
    # Logs full request context to diagnose authentication failures
    if exc.status_code == 401:
        print__analyze_debug(f"🚨 HTTP 401 UNAUTHORIZED: {exc.detail}")
        print__analysis_tracing_debug(f"🚨 HTTP 401 TRACE: Request URL: {request.url}")
        print__analysis_tracing_debug(
            f"🚨 HTTP 401 TRACE: Request method: {request.method}"
        )
        print__analysis_tracing_debug(
            f"🚨 HTTP 401 TRACE: Request headers: {dict(request.headers)}"
        )
        print__analysis_tracing_debug(
            f"🚨 HTTP 401 TRACE: Exception detail: {exc.detail}"
        )
        print__analysis_tracing_debug(
            f"🚨 HTTP 401 TRACE: Full traceback:\n{traceback.format_exc()}"
        )

        # Log client IP for security analysis and debugging
        client_ip = request.client.host if request.client else "unknown"
        print__analyze_debug(f"🚨 HTTP 401 CLIENT: IP address: {client_ip}")

    # =======================================================================
    # OTHER HTTP ERROR DEBUGGING (4xx, 5xx)
    # =======================================================================

    # Debug prints for other HTTP exceptions (less detailed than 401)
    elif exc.status_code >= 400:
        print__analyze_debug(f"🚨 HTTP {exc.status_code} ERROR: {exc.detail}")
        print__analysis_tracing_debug(
            f"🚨 HTTP {exc.status_code} TRACE: Request URL: {request.url}"
        )
        print__analysis_tracing_debug(
            f"🚨 HTTP {exc.status_code} TRACE: Request method: {request.method}"
        )
        print__analysis_tracing_debug(
            f"🚨 HTTP {exc.status_code} TRACE: Full traceback:\n{traceback.format_exc()}"
        )

    # =======================================================================
    # RETURN JSON RESPONSE
    # =======================================================================

    # Return JSON response with original status code and detail message
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def value_error_handler(_request: Request, exc: ValueError):
    """Handle ValueError exceptions as 400 Bad Request.

    This handler converts ValueError exceptions (typically from business logic
    validation) into 400 Bad Request responses with the error message.

    Args:
        _request: The FastAPI Request object (unused but required by signature)
        exc: The ValueError exception

    Returns:
        JSONResponse with status 400 and error message

    Response Format:
        {
            "detail": "<error message from ValueError>"
        }

    Example:
        raise ValueError("Invalid thread_id format")
        → 400 Bad Request with detail message
    """
    # Log ValueError for debugging
    print__debug(f"ValueError: {str(exc)}")

    # Return 400 Bad Request with error message
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def general_exception_handler(_request: Request, exc: Exception):
    """Handle unexpected exceptions with 500 Internal Server Error.

    This is a catch-all handler for any exceptions not handled by more specific
    handlers. It logs the error details and returns a generic error message to
    prevent sensitive information disclosure.

    Args:
        _request: The FastAPI Request object (unused but required by signature)
        exc: The unexpected Exception

    Returns:
        JSONResponse with status 500 and generic error message

    Response Format:
        {
            "detail": "Internal server error"
        }

    Security Note:
        - Does NOT include exception details in response (prevents info disclosure)
        - Logs full error details for debugging
        - Generic message to clients
    """
    # Log unexpected error with type and message for debugging
    print__debug(f"Unexpected error: {type(exc).__name__}: {str(exc)}")

    # Return 500 Internal Server Error with generic message
    # (do not expose internal error details to clients for security)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

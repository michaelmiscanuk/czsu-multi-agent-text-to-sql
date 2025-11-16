"""
MODULE_DESCRIPTION: API Helper Functions - Error Response Formatting and Debug Utilities

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module provides utility functions for error handling and response formatting
in the CZSU Multi-Agent Text-to-SQL API. The primary function enables detailed
error debugging by including full exception tracebacks in JSON responses when
debug mode is enabled.

This is particularly useful during development and troubleshooting, allowing
developers to see the complete stack trace directly in API responses without
having to check server logs.

Key Functionality:
    - Conditional traceback inclusion based on environment variable
    - Formatted JSON error responses with debug information
    - Optional run_id tracking for error correlation
    - Fallback mechanism for production environments

===================================================================================
KEY FEATURES
===================================================================================

1. Environment-Controlled Debug Mode
   - DEBUG_TRACEBACK=1: Include full tracebacks in responses
   - DEBUG_TRACEBACK=0 or unset: Return None (caller handles fallback)
   - Safe for production (disabled by default)

2. Full Exception Information
   - Exception type name
   - Exception message
   - Complete stack trace
   - Source code context

3. Run ID Integration
   - Optional run_id parameter
   - Links errors to specific analysis runs
   - Helps correlate errors with user actions
   - Consistent with successful response format

4. Flexible Status Codes
   - Configurable HTTP status code
   - Default: 500 Internal Server Error
   - Supports custom codes (400, 422, etc.)

5. Fallback Pattern
   - Returns None when debug mode disabled
   - Caller implements production error handling
   - Prevents accidental info disclosure

===================================================================================
HELPER FUNCTIONS
===================================================================================

Function: traceback_json_response(e, status_code=500, run_id=None)
    Purpose: Create JSON error response with optional traceback

    Parameters:
        e (Exception): The exception that occurred
            - Any Python exception instance
            - Used to extract type, message, and traceback

        status_code (int): HTTP status code for the response
            - Default: 500 (Internal Server Error)
            - Common values: 400, 422, 500
            - Allows caller to specify appropriate code

        run_id (str | None): Optional run ID to include in response
            - UUID string identifying the analysis run
            - Helps correlate errors with specific requests
            - None if not applicable

    Returns:
        JSONResponse | None:
            - JSONResponse with error details if DEBUG_TRACEBACK=1
            - None if DEBUG_TRACEBACK not set (caller handles fallback)

    Response Format (debug mode):
        {
            "detail": "<exception message>",
            "traceback": "<full stack trace>",
            "run_id": "<uuid>"  # Optional
        }

Usage Example:
    try:
        # Some operation that might fail
        result = process_data()
    except Exception as e:
        # Try to create debug response
        debug_response = traceback_json_response(e, status_code=500, run_id=run_id)

        if debug_response:
            # Debug mode enabled - return detailed error
            return debug_response
        else:
            # Production mode - return generic error
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

===================================================================================
DEBUG MODE CONTROL
===================================================================================

Environment Variable: DEBUG_TRACEBACK
    Values:
        "1": Enable traceback in responses
        "0" or unset: Disable traceback (production)

    Setting:
        .env file: DEBUG_TRACEBACK=1
        Environment: export DEBUG_TRACEBACK=1
        Command line: DEBUG_TRACEBACK=1 python main.py

Security Implications:
    Enabled (DEBUG_TRACEBACK=1):
        - Exposes internal code structure
        - Shows file paths and line numbers
        - Reveals implementation details
        - USE ONLY IN DEVELOPMENT

    Disabled (default):
        - Safe for production
        - No information disclosure
        - Generic error messages only

===================================================================================
TRACEBACK FORMATTING
===================================================================================

Traceback Content:
    traceback.format_exception(type(e), e, e.__traceback__)

    Includes:
        - Exception type (e.g., ValueError, KeyError)
        - Exception message
        - Stack trace with file names
        - Line numbers
        - Source code context

Example Traceback:
    Traceback (most recent call last):
      File "api/routes/analyze.py", line 45, in analyze
        result = process_query(prompt)
      File "my_agent/graph.py", line 123, in process_query
        sql = generate_sql(query)
      File "my_agent/nodes.py", line 78, in generate_sql
        raise ValueError("Invalid query format")
    ValueError: Invalid query format

String Formatting:
    tb_str = "".join(traceback.format_exception(...))

    - format_exception returns list of strings
    - "".join() concatenates into single string
    - Preserves newlines and formatting
    - Ready for JSON serialization

===================================================================================
RUN ID INTEGRATION
===================================================================================

Purpose of run_id:
    - Links errors to specific analysis runs
    - Enables error correlation in logs
    - Helps users report issues
    - Consistent with successful response format

When to Include:
    - Analysis endpoints (/analyze)
    - Any operation with run_id tracking
    - Feedback submission errors
    - Chat message operations

When to Omit:
    - Generic utility operations
    - Health checks
    - Authentication errors
    - Operations without run tracking

Example with run_id:
    {
        "detail": "Database connection failed",
        "traceback": "Traceback (most recent call last)...",
        "run_id": "550e8400-e29b-41d4-a716-446655440000"
    }

===================================================================================
FALLBACK PATTERN
===================================================================================

Why Return None:
    - Allows caller to implement production behavior
    - Prevents accidental debug info disclosure
    - Flexible error handling strategy
    - Explicit opt-in for debug mode

Caller Pattern:
    response = traceback_json_response(e, 500, run_id)
    if response:
        return response  # Debug mode
    else:
        # Production mode - return generic error
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

Alternative Pattern:
    response = traceback_json_response(e, 500, run_id)
    if not response:
        # Build production response
        response = JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "run_id": run_id  # Still include run_id
            }
        )
    return response

===================================================================================
USE CASES
===================================================================================

1. Analysis Endpoint Errors:
    @router.post("/analyze")
    async def analyze(request: AnalyzeRequest):
        try:
            result = await process_analysis(request)
            return result
        except Exception as e:
            response = traceback_json_response(e, 500, request.run_id)
            return response or JSONResponse(
                status_code=500,
                content={"detail": "Analysis failed"}
            )

2. Validation Errors:
    try:
        validate_input(data)
    except ValueError as e:
        response = traceback_json_response(e, 400)
        return response or JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )

3. Database Errors:
    try:
        result = db.query(sql)
    except DatabaseError as e:
        response = traceback_json_response(e, 500, run_id)
        return response or JSONResponse(
            status_code=500,
            content={"detail": "Database error"}
        )

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Production Deployment
   - ALWAYS disable DEBUG_TRACEBACK in production
   - Tracebacks expose internal structure
   - Can reveal security vulnerabilities
   - Helps attackers understand system

2. Information Disclosure
   - File paths reveal directory structure
   - Line numbers help locate code
   - Error messages may contain sensitive data
   - Stack traces show control flow

3. Safe Practices
   - Use environment-specific .env files
   - .env.production should NOT set DEBUG_TRACEBACK
   - .env.development can set DEBUG_TRACEBACK=1
   - Never commit DEBUG_TRACEBACK=1 to production config

4. Error Logging
   - Still log full tracebacks to server logs
   - Server logs are secure
   - Don't expose logs to clients
   - Use proper log aggregation

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Overhead:
    - Traceback formatting: ~1-5ms
    - JSON serialization: ~0.5ms
    - Total: <10ms (negligible for error path)

Memory Usage:
    - Traceback string: ~1-10 KB
    - Temporary during response creation
    - Released after response sent
    - No persistent storage

When to Avoid:
    - Not a performance concern (only on errors)
    - Errors are exceptional path
    - Debug mode only enabled in development
    - No optimization needed

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Test with DEBUG_TRACEBACK=1
    - Test with DEBUG_TRACEBACK=0
    - Verify traceback content
    - Verify status code handling
    - Test run_id inclusion

Test Examples:
    def test_traceback_enabled():
        os.environ["DEBUG_TRACEBACK"] = "1"
        exc = ValueError("test error")
        response = traceback_json_response(exc, 400, "test-run-id")

        assert response is not None
        assert response.status_code == 400
        content = json.loads(response.body)
        assert "traceback" in content
        assert "run_id" in content

    def test_traceback_disabled():
        os.environ["DEBUG_TRACEBACK"] = "0"
        exc = ValueError("test error")
        response = traceback_json_response(exc)

        assert response is None

===================================================================================
INTEGRATION PATTERNS
===================================================================================

Consistent Error Handling:
    # Define in route handler
    try:
        result = operation()
        return {"result": result, "run_id": run_id}
    except Exception as e:
        # Try debug response
        debug_response = traceback_json_response(e, 500, run_id)
        if debug_response:
            return debug_response

        # Fallback to production response
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Operation failed",
                "run_id": run_id  # Maintain consistency
            }
        )

Centralized Error Handler:
    def handle_error(e: Exception, status_code: int, run_id: str = None):
        \"\"\"Centralized error handling with debug support.\"\"\"
        debug_response = traceback_json_response(e, status_code, run_id)
        if debug_response:
            return debug_response

        return JSONResponse(
            status_code=status_code,
            content={
                "detail": str(e),
                "run_id": run_id
            }
        )

    # Usage
    try:
        result = operation()
    except Exception as e:
        return handle_error(e, 500, run_id)

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variable access
    - traceback: Exception traceback formatting

Third-Party:
    - fastapi.responses: JSONResponse class

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Structured Error Codes
   - Application-specific error codes
   - Error code documentation
   - Client-friendly error categorization

2. Enhanced Debug Information
   - Request context (headers, body)
   - User information
   - System state at error time
   - Performance metrics

3. Error Reporting Integration
   - Automatic Sentry reporting
   - Error aggregation
   - Duplicate detection
   - Alerting thresholds

4. Conditional Detail Levels
   - VERBOSE, NORMAL, MINIMAL modes
   - Different detail levels per environment
   - Per-endpoint debug configuration

===================================================================================
"""

# API helper functions for error handling and response formatting
import os
import traceback

from fastapi.responses import JSONResponse


# ==============================================================================
# ERROR RESPONSE HELPERS
# ==============================================================================


def traceback_json_response(e, status_code=500, run_id=None):
    """Create a JSON response with traceback information when in debug mode.

    This function generates detailed error responses including full exception
    tracebacks when DEBUG_TRACEBACK environment variable is set to "1". In
    production mode (debug disabled), it returns None to allow the caller to
    implement safe error handling without exposing internal details.

    Args:
        e: The exception that occurred (any Python exception instance)
        status_code: HTTP status code for the response (default: 500)
        run_id: Optional run ID to include in the response for error correlation

    Returns:
        JSONResponse with error details and traceback if DEBUG_TRACEBACK=1,
        None otherwise (caller should handle fallback to production error response)

    Response Format (debug mode):
        {
            "detail": "<exception message>",
            "traceback": "<full stack trace with file paths and line numbers>",
            "run_id": "<uuid>"  # Optional, included if provided
        }

    Security Warning:
        Only enable DEBUG_TRACEBACK=1 in development environments. Tracebacks
        expose internal code structure, file paths, and implementation details
        that should not be revealed in production.

    Example:
        try:
            result = process_data()
        except Exception as e:
            response = traceback_json_response(e, 500, run_id)
            if response:
                return response  # Debug mode - detailed error
            else:
                return JSONResponse(  # Production mode - safe error
                    status_code=500,
                    content={"detail": "Internal server error"}
                )
    """
    # =======================================================================
    # CHECK DEBUG MODE
    # =======================================================================
    # Only include tracebacks when DEBUG_TRACEBACK=1 (development mode)
    # Production should NOT set this variable (security risk)
    if os.environ.get("DEBUG_TRACEBACK") == "1":
        # =======================================================================
        # FORMAT EXCEPTION TRACEBACK
        # =======================================================================
        # Extract full stack trace from exception
        # Includes: exception type, message, file paths, line numbers, code context
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        # =======================================================================
        # BUILD ERROR RESPONSE CONTENT
        # =======================================================================
        # Prepare response content with exception details
        response_content = {
            "detail": str(e),  # Human-readable error message
            "traceback": tb_str,  # Full stack trace for debugging
        }

        # Include run_id if provided to maintain consistency with successful responses
        # Helps correlate errors with specific analysis runs
        if run_id:
            response_content["run_id"] = run_id

        # =======================================================================
        # RETURN DEBUG RESPONSE
        # =======================================================================
        # Return JSONResponse with detailed error information
        return JSONResponse(
            status_code=status_code,
            content=response_content,
        )

    # =======================================================================
    # PRODUCTION MODE - RETURN NONE
    # =======================================================================
    # Debug mode disabled - return None so caller can handle fallback
    # Prevents accidental exposure of internal details
    return None  # Caller should handle fallback to safe production response

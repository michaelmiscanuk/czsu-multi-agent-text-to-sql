"""API helper functions for error handling and response formatting."""

import os
import traceback

from fastapi.responses import JSONResponse


def traceback_json_response(e, status_code=500, run_id=None):
    """Create a JSON response with traceback information when in debug mode.

    Args:
        e: The exception that occurred
        status_code: HTTP status code for the response (default: 500)
        run_id: Optional run ID to include in the response

    Returns:
        JSONResponse with error details and traceback if DEBUG_TRACEBACK=1,
        None otherwise (caller should handle fallback)
    """
    if os.environ.get("DEBUG_TRACEBACK") == "1":
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        response_content = {
            "detail": str(e),
            "traceback": tb_str,
        }
        # Include run_id if provided to maintain consistency with successful responses
        if run_id:
            response_content["run_id"] = run_id
        return JSONResponse(
            status_code=status_code,
            content=response_content,
        )
    return None  # Caller should handle fallback

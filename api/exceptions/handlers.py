"""FastAPI exception handlers for error processing and response formatting."""

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

# ============================================================
# EXCEPTION HANDLERS
# ============================================================


async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with proper 422 status code."""
    print__debug(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422, content={"detail": "Validation error", "errors": exc.errors()}
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with comprehensive debugging for 401 errors."""

    # Enhanced debugging for 401 errors since these are authentication-related
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

        # Log client IP for debugging
        client_ip = request.client.host if request.client else "unknown"
        print__analyze_debug(f"🚨 HTTP 401 CLIENT: IP address: {client_ip}")

    # Debug prints for other HTTP exceptions
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

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def value_error_handler(_request: Request, exc: ValueError):
    """Handle ValueError exceptions as 400 Bad Request."""
    print__debug(f"ValueError: {str(exc)}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def general_exception_handler(_request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    print__debug(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

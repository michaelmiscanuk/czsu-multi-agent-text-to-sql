"""FastAPI application for the CZSU multi-agent text-to-SQL system.

This module provides the main FastAPI application with routes for analysis,
chat, catalog, and other functionality for the Czech Statistical Office data system.
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

    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Add the root directory to Python path for imports to work
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Standard imports
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

import psutil
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import configuration and globals
from api.config.settings import (
    _APP_STARTUP_TIME,
    _MEMORY_BASELINE,
    _REQUEST_COUNT,
    throttle_semaphores,
)

# Load memory-related variables directly from environment
GC_MEMORY_THRESHOLD = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))
MEMORY_PROFILER_ENABLED = os.environ.get("MEMORY_PROFILER_ENABLED", "0") == "1"
MEMORY_PROFILER_INTERVAL = int(os.environ.get("MEMORY_PROFILER_INTERVAL", "30"))
MEMORY_PROFILER_TOP_STATS = int(os.environ.get("MEMORY_PROFILER_TOP_STATS", "10"))

# Import authentication
# from api.dependencies.auth import get_current_user

# Import debug functions
from api.utils.debug import (
    print__analysis_tracing_debug,
    print__analyze_debug,
    print__debug,
    print__startup_debug,
)

# Import memory utilities
from api.utils.memory import (
    log_comprehensive_error,
    log_memory_usage,
    setup_graceful_shutdown,
    start_memory_profiler,
    start_memory_cleanup,
    stop_memory_profiler,
    stop_memory_cleanup,
)

# Import rate limiting utilities
from api.utils.rate_limiting import (
    check_rate_limit_with_throttling,
    wait_for_rate_limit,
)

# Import database functions
sys.path.insert(0, str(BASE_DIR))
from api.routes.analysis import router as analysis_router
from api.routes.bulk import router as bulk_router
from api.routes.catalog import router as catalog_router
from api.routes.chat import router as chat_router
from api.routes.debug import router as debug_router
from api.routes.feedback import router as feedback_router

# Import all route routers
from api.routes.health import router as health_router
from api.routes.messages import router as messages_router
from api.routes.misc import router as misc_router
from api.routes.stop import router as stop_router
from checkpointer.checkpointer.factory import (
    initialize_checkpointer,
    cleanup_checkpointer,
)
from api.routes.root import router as root_router


# Lifespan management function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _APP_STARTUP_TIME, _MEMORY_BASELINE
    _APP_STARTUP_TIME = datetime.now()

    print__startup_debug("🚀 FastAPI application starting up...")
    from api.utils.memory import print__memory_monitoring

    print__memory_monitoring(
        f"Application startup initiated at {_APP_STARTUP_TIME.isoformat()}"
    )
    log_memory_usage("app_startup")

    # ROUTE REGISTRATION MONITORING: Track all routes that get registered
    # This prevents the exact issue described in the "Needle in a haystack" article
    print__memory_monitoring(
        "🔍 Monitoring route registrations to prevent memory leaks..."
    )

    # Setup graceful shutdown handlers
    setup_graceful_shutdown()

    await initialize_checkpointer()

    # Set memory baseline after initialization
    if _MEMORY_BASELINE is None:
        try:
            process = psutil.Process()
            _MEMORY_BASELINE = process.memory_info().rss / 1024 / 1024
            print__memory_monitoring(
                f"Memory baseline established: {_MEMORY_BASELINE:.1f}MB RSS"
            )
        except:
            pass

    if MEMORY_PROFILER_ENABLED:
        try:
            start_memory_profiler(
                interval=MEMORY_PROFILER_INTERVAL,
                top_stats=MEMORY_PROFILER_TOP_STATS,
            )
        except Exception as profiler_error:
            print__memory_monitoring(
                f"⚠️ Failed to start memory profiler: {profiler_error}"
            )

    # Start periodic memory cleanup (controlled by MEMORY_CLEANUP env var)
    # This runs as a background asyncio task that forces memory to be returned to the OS
    # It helps prevent heap and anonymous memory from staying at peak levels when app is idle
    cleanup_task = (
        start_memory_cleanup()
    )  # Runs every MEMORY_CLEANUP_INTERVAL seconds (from .env)
    if cleanup_task:
        print__memory_monitoring("✅ Memory cleanup task started successfully")
    else:
        print__memory_monitoring(
            "ℹ️ Memory cleanup task not started (disabled or error)"
        )

    log_memory_usage("app_ready")
    print__startup_debug("✅ FastAPI application ready to serve requests")

    yield

    # Shutdown
    print__startup_debug("🛑 FastAPI application shutting down...")
    print__memory_monitoring(
        f"Application ran for {datetime.now() - _APP_STARTUP_TIME}"
    )

    if MEMORY_PROFILER_ENABLED:
        try:
            await stop_memory_profiler()
        except Exception as profiler_error:
            print__memory_monitoring(
                f"⚠️ Failed to stop memory profiler: {profiler_error}"
            )

    # Stop memory cleanup task
    try:
        await stop_memory_cleanup()
    except Exception as cleanup_error:
        print__memory_monitoring(
            f"⚠️ Failed to stop memory cleanup task: {cleanup_error}"
        )

    # Log final memory statistics
    if _MEMORY_BASELINE:
        try:
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024
            total_growth = final_memory - _MEMORY_BASELINE
            print__memory_monitoring(
                f"Final memory stats: Started={_MEMORY_BASELINE:.1f}MB, "
                f"Final={final_memory:.1f}MB, Growth={total_growth:.1f}MB"
            )
            if (
                total_growth > GC_MEMORY_THRESHOLD
            ):  # More than threshold growth - app will restart soon
                print__memory_monitoring(
                    "🚨 SIGNIFICANT MEMORY GROWTH DETECTED - investigate for leaks!"
                )
        except:
            pass

    await cleanup_checkpointer()


# Create the FastAPI application
app = FastAPI(
    title="CZSU Multi-Agent Text-to-SQL API",
    description="""Multi-agent system for converting natural language queries to SQL and retrieving data from the Czech Statistical Office (CZSU) and complex PDF containing statistical summaries of Czech Economy using an AI chatbot.
    
## Features
- 🤖 AI-powered natural language to SQL conversion
- 💬 Multi-threaded conversation management
- 📊 Statistical data catalog browsing
- 👍 User feedback and sentiment tracking
- 🔍 Query execution and result retrieval

## Authentication
All endpoints (except `/health` and `/docs`) require Bearer token authentication.
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "email": "michael.miscanuk@google.com",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {
            "url": "https://www.multiagent-texttosql-prototype.online/api",
            "description": "Production server",
        },
    ],
    responses={
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "content": {
                "application/json": {
                    "example": {"detail": "User email not found in token"}
                }
            },
        },
        422: {
            "description": "Validation Error - Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Validation error",
                        "errors": [
                            {
                                "loc": ["body", "prompt"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ],
                    }
                }
            },
        },
        429: {
            "description": "Rate Limit Exceeded - Too many requests",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Please wait 5.0s before retrying.",
                        "retry_after": 5,
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error"}}
            },
        },
    },
)

# Monitor all route registrations (including middleware and CORS)
from api.utils.memory import print__memory_monitoring

print__memory_monitoring("[CORS] Registering CORS middleware...")
# Note: Route registration monitoring happens at runtime to avoid import-time global variable access

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print__memory_monitoring("[GZIP] Registering GZip middleware...")
# Add GZip compression to reduce response sizes and memory usage
app.add_middleware(GZipMiddleware, minimum_size=1000)


# RATE LIMITING MIDDLEWARE
@app.middleware("http")
async def throttling_middleware(request: Request, call_next):
    """Throttling middleware that makes requests wait instead of rejecting them."""

    # Skip throttling for health checks and static endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"

    # Use semaphore to limit concurrent requests per IP
    semaphore = throttle_semaphores[client_ip]

    async with semaphore:
        # Try to wait for rate limit instead of immediately rejecting
        if not await wait_for_rate_limit(client_ip):
            # Only reject if we can't wait (wait time too long or max attempts exceeded)
            rate_info = check_rate_limit_with_throttling(client_ip)
            log_comprehensive_error(
                "rate_limit_exceeded_after_wait",
                Exception(
                    f"Rate limit exceeded for IP: {client_ip} after waiting. Burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, Window: {rate_info['window_count']}/{rate_info['window_limit']}"
                ),
                request,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Please wait {rate_info['suggested_wait']:.1f}s before retrying.",
                    "retry_after": max(rate_info["suggested_wait"], 1),
                    "burst_usage": f"{rate_info['burst_count']}/{rate_info['burst_limit']}",
                    "window_usage": f"{rate_info['window_count']}/{rate_info['window_limit']}",
                },
                headers={"Retry-After": str(max(int(rate_info["suggested_wait"]), 1))},
            )

        return await call_next(request)


# Enhanced middleware to monitor memory patterns and detect leaks
@app.middleware("http")
async def simplified_memory_monitoring_middleware(request: Request, call_next):
    """Simplified memory monitoring middleware."""
    global _REQUEST_COUNT

    _REQUEST_COUNT += 1

    # Only check memory for heavy operations
    request_path = request.url.path
    if any(
        path in request_path
        for path in ["/analyze", "/chat/all-messages-for-all-threads"]
    ):
        log_memory_usage(f"before_{request_path.replace('/', '_')}")

    response = await call_next(request)

    # Check memory after heavy operations
    if any(
        path in request_path
        for path in ["/analyze", "/chat/all-messages-for-all-threads"]
    ):
        log_memory_usage(f"after_{request_path.replace('/', '_')}")

    return response


# EXCEPTION HANDLERS
# Global exception handlers for proper error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with proper 422 status code.

    Uses jsonable_encoder to avoid JSON serialization errors (observed 500 during dumps).
    Falls back to minimal structure if encoding fails.
    """
    print__debug(f"Validation error: {exc.errors()}")
    try:
        payload = {"detail": "Validation error", "errors": exc.errors()}
        return JSONResponse(status_code=422, content=jsonable_encoder(payload))
    except Exception as encoding_error:  # Fallback in case of non-serializable content
        print__debug(
            f"🚨 Validation encoding failure: {type(encoding_error).__name__}: {encoding_error}"  # noqa: E501
        )
        # Provide simplified error list of messages only
        simple_errors = [
            {"msg": e.get("msg"), "loc": e.get("loc"), "type": e.get("type")}
            for e in (exc.errors() if hasattr(exc, "errors") else [])
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": simple_errors,
                "note": "Simplified due to serialization issue",
            },
        )


@app.exception_handler(StarletteHTTPException)
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions as 400 Bad Request."""
    print__debug(f"ValueError: {str(exc)}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    import os

    if os.getenv("DEBUG_TRACEBACK", "0") == "1":
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print__debug(
            f"Unexpected error (with traceback): {type(exc).__name__}: {str(exc)}\n{tb}"
        )
        return JSONResponse(
            status_code=500, content={"detail": str(exc), "traceback": tb}
        )
    print__debug(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ROUTE REGISTRATION
# Register all route routers
print__memory_monitoring("[ROUTES] Registering route routers...")

app.include_router(root_router, tags=["Root"])
app.include_router(health_router, tags=["Health & Monitoring"])
app.include_router(catalog_router, tags=["Data Catalog"])
app.include_router(analysis_router, tags=["Query Analysis"])
app.include_router(feedback_router, tags=["Feedback & Sentiment"])
app.include_router(chat_router, tags=["Chat & Threads"])
app.include_router(messages_router, tags=["Messages"])
app.include_router(bulk_router, tags=["Bulk Operations"])
app.include_router(debug_router, tags=["Debug & Admin"])
app.include_router(misc_router, tags=["Utilities"])
app.include_router(stop_router, tags=["Execution Control"])

print__memory_monitoring("[SUCCESS] All route routers registered successfully")

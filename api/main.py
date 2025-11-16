"""CZSU Multi-Agent Text-to-SQL FastAPI Backend Application

This module serves as the main entry point for the FastAPI backend application that powers
the Czech Statistical Office (CZSU) multi-agent text-to-SQL system. It orchestrates a
sophisticated AI-powered natural language to SQL conversion system with multi-threaded
conversation management, comprehensive memory monitoring, and production-grade error handling.

The application combines LangGraph-based multi-agent workflows with FastAPI's modern async
capabilities to provide a robust API for querying statistical databases and PDF-based
knowledge bases through natural language.
"""

MODULE_DESCRIPTION = r"""CZSU Multi-Agent Text-to-SQL FastAPI Backend Application

This module serves as the main entry point for the FastAPI backend application that powers
the Czech Statistical Office (CZSU) multi-agent text-to-SQL system. It orchestrates a 
sophisticated AI-powered natural language to SQL conversion system with multi-threaded 
conversation management, comprehensive memory monitoring, and production-grade error handling.

The application combines LangGraph-based multi-agent workflows with FastAPI's modern async 
capabilities to provide a robust API for querying statistical databases and PDF-based 
knowledge bases through natural language.

Key Features:
-------------
1. Multi-Agent AI System:
   - LangGraph-based agent orchestration for complex query understanding
   - Natural language to SQL query translation
   - Multi-step reasoning with specialized agents
   - Context-aware query refinement and validation
   - Integration with ChromaDB for semantic search over PDF documentation
   - Support for both SQLite (Turso) and cloud database backends

2. Production-Ready Infrastructure:
   - Asynchronous request handling with FastAPI
   - PostgreSQL-based checkpointing for conversation state persistence
   - Thread-safe multi-user conversation management
   - Graceful shutdown handling with resource cleanup
   - Comprehensive error handling with detailed logging
   - Rate limiting with intelligent throttling (wait instead of reject)
   - CORS and Brotli compression middleware

3. Memory Management & Monitoring:
   - Real-time memory profiling with configurable intervals
   - Automatic memory leak detection and alerting
   - Periodic garbage collection with configurable thresholds
   - Memory baseline tracking from startup
   - Per-request memory usage monitoring for heavy operations
   - RSS (Resident Set Size) tracking with growth analysis
   - Optional memory profiler with top statistics reporting
   - Automatic memory cleanup background tasks

4. Windows Compatibility:
   - Explicit Windows event loop policy configuration
   - psycopg compatibility fixes for Windows async operations
   - Proper path handling for cross-platform support

5. Authentication & Security:
   - Bearer token authentication for all protected endpoints
   - User email extraction from JWT tokens
   - IP-based rate limiting with burst and window limits
   - Configurable throttling with semaphores per client
   - Request validation with proper error responses

6. API Route Organization:
   - Health & Monitoring: System status, pool diagnostics, memory stats
   - Data Catalog: Browse CZSU datasets, selections, and metadata
   - Query Analysis: Natural language to SQL conversion and execution
   - Chat & Threads: Multi-threaded conversation management
   - Messages: Retrieve conversation history
   - Feedback: User feedback collection and sentiment tracking
   - Bulk Operations: Batch processing capabilities
   - Debug & Admin: Internal diagnostics and debugging tools
   - Execution Control: Query interruption and cancellation

7. Error Handling & Debugging:
   - Custom exception handlers for validation, HTTP, and unexpected errors
   - Detailed 401 authentication error tracing
   - Configurable traceback output for development
   - Comprehensive error logging with request context
   - Memory-aware error reporting with leak detection

8. Observability & Diagnostics:
   - Request counting and tracking
   - Memory usage logging at startup, per-request, and shutdown
   - Application uptime tracking
   - Memory growth detection with threshold alerts
   - Route registration monitoring to prevent memory leaks
   - Connection pool status monitoring

Architecture Components:
-----------------------
1. Application Lifecycle:
   - Lifespan context manager for startup/shutdown orchestration
   - Early environment variable loading with dotenv
   - PostgreSQL checkpointer initialization at startup
   - Memory baseline establishment
   - Background task management (profiler, cleanup)
   - Graceful resource cleanup on shutdown

2. Middleware Stack (Order Matters):
   - CORS: Allow cross-origin requests (development: *, production: specific)
   - Brotli: Response compression for reduced bandwidth
   - Throttling: Rate limiting with intelligent wait-instead-of-reject
   - Memory Monitoring: Track memory usage for heavy operations

3. Database Integration:
   - PostgreSQL for conversation state checkpointing
   - Turso SQLite Cloud for CZSU statistical data
   - ChromaDB for semantic search over PDF documentation
   - Connection pooling with status monitoring

4. Route Structure:
   - Root: Basic API information and welcome message
   - Health: /health, /debug/pool-status, /debug/memory-status
   - Catalog: /catalog/datasets, /catalog/selections, /catalog/metadata
   - Analysis: /analyze (main NL-to-SQL endpoint)
   - Chat: /chat/threads, /chat/messages, /chat/all-messages-for-all-threads
   - Feedback: /feedback (submit user feedback)
   - Bulk: /bulk/* (batch operations)
   - Debug: /debug/* (internal diagnostics)
   - Stop: /stop (cancel query execution)

Configuration & Environment:
---------------------------
Required Environment Variables:
- Database credentials (PostgreSQL, Turso)
- LLM API keys (OpenAI, Anthropic, or others)
- Authentication tokens and secrets
- CORS allowed origins for production

Memory Management Variables:
- GC_MEMORY_THRESHOLD: Memory growth threshold for alerts (default: 1900 MB)
- MEMORY_PROFILER_ENABLED: Enable memory profiler (0/1, default: 0)
- MEMORY_PROFILER_INTERVAL: Profiler snapshot interval (default: 30s)
- MEMORY_PROFILER_TOP_STATS: Number of top memory consumers to report (default: 10)
- MEMORY_CLEANUP_INTERVAL: Background cleanup interval (default: 300s)
- DEBUG_TRACEBACK: Include tracebacks in error responses (0/1, default: 0)

Rate Limiting Variables:
- RATE_LIMIT_BURST: Maximum burst requests per client (default: 5)
- RATE_LIMIT_WINDOW: Window size for rate limiting (default: 60s)
- THROTTLE_MAX_CONCURRENT: Max concurrent requests per IP (default: 3)
- THROTTLE_MAX_WAIT: Maximum wait time before rejecting (default: 30s)

Startup Sequence:
----------------
1. Windows Event Loop Policy Setup:
   - Must be FIRST to fix psycopg compatibility on Windows
   - Sets WindowsSelectorEventLoopPolicy before any async imports

2. Environment Loading:
   - Load .env file for configuration
   - Establish BASE_DIR for project root
   - Add BASE_DIR to sys.path for imports

3. Configuration Import:
   - Load global settings and constants
   - Initialize throttle semaphores dictionary
   - Set memory management variables

4. Application Initialization:
   - Create FastAPI app with lifespan manager
   - Configure OpenAPI documentation
   - Set API metadata and contact info

5. Middleware Registration:
   - CORS for cross-origin support
   - Brotli for response compression
   - Rate limiting with throttling
   - Memory monitoring for heavy operations

6. Exception Handler Registration:
   - RequestValidationError: 422 responses
   - StarletteHTTPException: HTTP errors with detailed 401 tracing
   - ValueError: 400 Bad Request
   - General Exception: 500 Internal Server Error

7. Route Registration:
   - Import all route modules
   - Register routers with tags
   - Monitor route additions to prevent leaks

8. Lifespan Startup:
   - Record startup timestamp
   - Initialize PostgreSQL checkpointer
   - Establish memory baseline
   - Start memory profiler (if enabled)
   - Start background cleanup task

9. Ready to Serve:
   - Log memory usage
   - Report startup success
   - Begin handling requests

Shutdown Sequence:
-----------------
1. Shutdown Signal:
   - Catch SIGTERM/SIGINT
   - Initiate graceful shutdown

2. Resource Cleanup:
   - Stop memory profiler
   - Stop background cleanup task
   - Close database connections
   - Cleanup checkpointer resources

3. Final Reporting:
   - Calculate application uptime
   - Report final memory statistics
   - Compare to baseline for leak detection
   - Alert if memory growth exceeds threshold

Memory Management Strategy:
--------------------------
The application implements a sophisticated memory management system to prevent
memory leaks and optimize resource usage in long-running production environments.

1. Memory Baseline:
   - Established at startup after initialization
   - Used as reference for growth calculations
   - Helps identify abnormal memory accumulation

2. Real-Time Monitoring:
   - Per-request tracking for heavy operations (/analyze, /chat/all-messages-*)
   - RSS (Resident Set Size) measurement before and after
   - Memory growth alerts when exceeding thresholds

3. Memory Profiler (Optional):
   - Uses tracemalloc for detailed memory snapshots
   - Configurable snapshot intervals
   - Reports top memory consumers
   - Helps identify leak sources in development

4. Background Cleanup:
   - Periodic garbage collection
   - Forces memory return to OS
   - Prevents heap fragmentation
   - Runs at configurable intervals

5. Route Registration Monitoring:
   - Tracks all route additions at startup
   - Prevents "needle in a haystack" memory leak pattern
   - Ensures routes aren't duplicated on reload

6. Leak Detection:
   - Compares final memory to baseline at shutdown
   - Alerts if growth exceeds threshold (default: 1900 MB)
   - Provides detailed statistics for investigation

Rate Limiting Strategy:
----------------------
The application uses a two-tier rate limiting approach:

1. Burst Protection:
   - Short-term limit for rapid requests
   - Prevents API abuse and DDoS
   - Default: 5 requests in quick succession

2. Window-Based Limiting:
   - Longer-term request limits
   - Default: 60 requests per 60-second window
   - Prevents sustained overuse

3. Intelligent Throttling:
   - Instead of rejecting immediately, requests wait
   - Uses asyncio semaphores per client IP
   - Max concurrent requests configurable
   - Only rejects if wait time exceeds threshold

4. Exempt Endpoints:
   - /health, /docs, /openapi.json
   - /debug/pool-status
   - Essential for monitoring and documentation

Error Handling Philosophy:
-------------------------
The application prioritizes detailed error information while maintaining security:

1. Development Mode:
   - Full tracebacks included in responses
   - Detailed memory statistics
   - Comprehensive logging

2. Production Mode:
   - Generic error messages
   - No internal details exposed
   - Log all errors for investigation

3. Special Cases:
   - 401 errors: Enhanced debugging for authentication issues
   - 422 errors: Detailed validation errors with field information
   - 429 errors: Rate limit info with retry-after headers
   - 500 errors: Generic message, full logging internally

4. Error Context:
   - Request URL, method, headers
   - Client IP address
   - Memory state at error time
   - Full traceback in logs

API Documentation:
-----------------
The application automatically generates OpenAPI documentation at:
- /docs: Swagger UI (interactive documentation)
- /redoc: ReDoc (alternative documentation view)
- /openapi.json: Raw OpenAPI specification

Documentation includes:
- All endpoint descriptions
- Request/response schemas
- Authentication requirements
- Error response examples
- Rate limiting information

Usage Example:
-------------
# Development server startup
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production server with Gunicorn
gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000

# Docker container
docker-compose up backend

# Query the API
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show me population data for Prague", "thread_id": "user123"}'

Dependencies:
------------
Core Framework:
- fastapi: Modern async web framework
- uvicorn: ASGI server
- starlette: ASGI toolkit (used by FastAPI)
- pydantic: Data validation

Database:
- psycopg[binary]: PostgreSQL driver with Windows async support
- sqlalchemy: ORM for database operations
- chromadb: Vector database for semantic search

AI/ML:
- langchain: LLM orchestration framework
- langgraph: Agent workflow management
- openai: OpenAI API client
- anthropic: Anthropic API client (optional)

Utilities:
- python-dotenv: Environment variable management
- psutil: System and process utilities
- python-multipart: Form data parsing
- python-jose: JWT token handling
- tenacity: Retry logic with exponential backoff

API Endpoints Summary:
---------------------
GET  /                          - API welcome message
GET  /health                    - Health check (no auth required)
GET  /debug/pool-status         - Connection pool diagnostics
GET  /debug/memory-status       - Memory usage statistics
GET  /catalog/datasets          - List available CZSU datasets
GET  /catalog/selections/{id}   - Get selections for dataset
POST /analyze                   - Convert NL query to SQL and execute
POST /chat/threads              - Create new conversation thread
GET  /chat/threads/{thread_id}  - Get thread messages
GET  /chat/all-messages-*       - Bulk message retrieval (admin)
POST /feedback                  - Submit user feedback
POST /stop                      - Cancel query execution
GET  /debug/*                   - Internal debugging endpoints

Performance Considerations:
--------------------------
1. Memory:
   - Baseline: ~200-400 MB after startup
   - Per request: +10-50 MB (depends on query complexity)
   - Alert threshold: 1900 MB growth from baseline
   - Background cleanup prevents accumulation

2. Database:
   - Connection pooling for PostgreSQL
   - Efficient checkpointing to minimize overhead
   - ChromaDB caching for repeated queries

3. Response Time:
   - Simple queries: <1 second
   - Complex multi-agent queries: 5-15 seconds
   - Bulk operations: Varies by size

4. Concurrency:
   - Async request handling
   - Per-client throttling prevents overload
   - Background tasks don't block requests

Troubleshooting:
---------------
1. Memory Growth:
   - Check MEMORY_PROFILER output for leak sources
   - Verify GC_MEMORY_THRESHOLD is appropriate
   - Ensure MEMORY_CLEANUP_INTERVAL is set
   - Monitor memory-status endpoint

2. Rate Limiting:
   - Check client IP in logs
   - Adjust RATE_LIMIT_* variables
   - Verify exempt endpoints list
   - Monitor throttle semaphore usage

3. Authentication Errors:
   - Enable DEBUG_TRACEBACK for details
   - Verify JWT token format
   - Check token expiration
   - Review 401 error traces in logs

4. Database Connection:
   - Monitor pool-status endpoint
   - Check PostgreSQL connection string
   - Verify checkpointer initialization
   - Review connection pool settings

5. Windows Async Issues:
   - Ensure event loop policy is set FIRST
   - Verify psycopg[binary] installation
   - Check for asyncio compatibility

Production Deployment:
---------------------
Recommended setup:
- Use Gunicorn with uvicorn workers
- Set appropriate memory limits
- Enable memory profiler for initial monitoring
- Configure proper CORS origins
- Use environment variables for secrets
- Set up health check monitoring
- Configure log aggregation
- Implement container restart policies

Security Checklist:
- ✓ Bearer token authentication
- ✓ Rate limiting per client IP
- ✓ CORS configured for production
- ✓ No sensitive data in responses
- ✓ Validation on all inputs
- ✓ Proper error handling
- ✓ Secure environment variable management

Monitoring Recommendations:
- Health check endpoint for uptime
- Memory status for resource usage
- Pool status for database health
- Log aggregation for error tracking
- Request count for traffic analysis
- Rate limit metrics for abuse detection

Notes:
------
- Windows event loop policy MUST be set before any async imports
- Memory baseline is established AFTER initialization, not at import time
- Rate limiting uses wait-instead-of-reject for better UX
- Heavy operations are monitored for memory usage
- Route registration is monitored to prevent memory leaks
- Background cleanup helps prevent memory accumulation in long-running processes
- Graceful shutdown ensures proper resource cleanup

Version History:
---------------
- 1.0.0: Initial production release with multi-agent system
- Enhanced memory management and monitoring
- Improved rate limiting with intelligent throttling
- Production-ready error handling and logging"""

# ==============================================================================
# CRITICAL WINDOWS COMPATIBILITY SETUP
# ==============================================================================
# MUST BE FIRST: Set Windows event loop policy before ANY other imports
# This fixes psycopg[binary] async compatibility issues on Windows platforms
# Without this, PostgreSQL checkpointer operations will fail with event loop errors
# Reference: https://docs.python.org/3/library/asyncio-policy.html#asyncio.WindowsSelectorEventLoopPolicy
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==============================================================================
# ENVIRONMENT VARIABLES LOADING
# ==============================================================================
# Load .env file early to ensure all configuration is available before imports
# This allows modules to access environment variables during their initialization
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# PROJECT ROOT DIRECTORY CONFIGURATION
# ==============================================================================
# Establish BASE_DIR as the project root (parent of api/ directory)
# This is used for relative imports and file path resolution
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]  # Go up one level from api/main.py
except NameError:
    # Fallback for interactive environments (Jupyter, REPL)
    BASE_DIR = Path(os.getcwd())

# Add the root directory to Python path for imports to work
# This enables imports like "from checkpointer import ..." and "from my_agent import ..."
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ==============================================================================
# STANDARD LIBRARY AND THIRD-PARTY IMPORTS
# ==============================================================================
import traceback  # For detailed error logging and debugging
from contextlib import asynccontextmanager  # For lifespan management
from datetime import datetime  # For timestamp tracking

import psutil  # For system/process monitoring (memory, CPU)
from fastapi import FastAPI, Request  # Core FastAPI framework
from fastapi.exceptions import RequestValidationError  # Pydantic validation errors
from fastapi.responses import JSONResponse  # JSON response formatting
from fastapi.encoders import jsonable_encoder  # Safe JSON encoding
from starlette.exceptions import HTTPException as StarletteHTTPException  # HTTP errors

# Import middleware setup functions
from api.middleware.cors import setup_cors_middleware, setup_brotli_middleware
from api.middleware.rate_limiting import setup_throttling_middleware
from api.middleware.memory_monitoring import setup_memory_monitoring_middleware

# ==============================================================================
# CONFIGURATION AND GLOBAL VARIABLES
# ==============================================================================
# Import shared global state and configuration from settings module
from api.config.settings import (
    _APP_STARTUP_TIME,  # Timestamp when application started (for uptime calculation)
    _MEMORY_BASELINE,  # Initial memory usage after startup (for leak detection)
)

# ==============================================================================
# MEMORY MANAGEMENT CONFIGURATION
# ==============================================================================
# Load memory-related variables directly from environment
# These control the memory monitoring and cleanup behavior
GC_MEMORY_THRESHOLD = int(
    os.environ.get("GC_MEMORY_THRESHOLD", "1900")
)  # MB growth before alert
MEMORY_PROFILER_ENABLED = (
    os.environ.get("MEMORY_PROFILER_ENABLED", "0") == "1"
)  # Enable tracemalloc profiler
MEMORY_PROFILER_INTERVAL = int(
    os.environ.get("MEMORY_PROFILER_INTERVAL", "30")
)  # Snapshot interval (seconds)
MEMORY_PROFILER_TOP_STATS = int(
    os.environ.get("MEMORY_PROFILER_TOP_STATS", "10")
)  # Top N memory consumers

# ==============================================================================
# AUTHENTICATION UTILITIES
# ==============================================================================
# Authentication is handled via Bearer tokens in route dependencies
# Uncomment below if centralized auth dependency is needed
# from api.dependencies.auth import get_current_user

# ==============================================================================
# DEBUG AND LOGGING UTILITIES
# ==============================================================================
# Import debug printing functions for different contexts
from api.utils.debug import (
    print__analysis_tracing_debug,  # Detailed query analysis tracing
    print__analyze_debug,  # Analysis endpoint debugging
    print__debug,  # General debug output
    print__startup_debug,  # Application startup/shutdown logging
)

# ==============================================================================
# MEMORY MANAGEMENT UTILITIES
# ==============================================================================
# Import memory monitoring, profiling, and cleanup functions
from api.utils.memory import (
    log_memory_usage,  # Log current memory usage with label
    print__memory_monitoring,  # Print memory monitoring messages
    setup_graceful_shutdown,  # Register signal handlers for clean shutdown
    start_memory_profiler,  # Start tracemalloc-based memory profiler
    start_memory_cleanup,  # Start background memory cleanup task
    stop_memory_profiler,  # Stop and report memory profiler statistics
    stop_memory_cleanup,  # Stop background cleanup task
)

# ==============================================================================
# ROUTE ROUTERS AND CHECKPOINTER IMPORTS
# ==============================================================================
# Ensure BASE_DIR is in path for checkpointer imports
sys.path.insert(0, str(BASE_DIR))

# Import all API route routers
# Each router handles a specific functional area of the API
from api.routes.analysis import (
    router as analysis_router,
)  # NL-to-SQL conversion and execution
from api.routes.bulk import router as bulk_router  # Batch operations
from api.routes.catalog import router as catalog_router  # CZSU data catalog browsing
from api.routes.chat import router as chat_router  # Multi-threaded conversations
from api.routes.debug import router as debug_router  # Internal diagnostics
from api.routes.feedback import router as feedback_router  # User feedback collection
from api.routes.health import router as health_router  # Health checks and monitoring
from api.routes.messages import router as messages_router  # Message retrieval
from api.routes.misc import router as misc_router  # Miscellaneous utilities
from api.routes.stop import router as stop_router  # Query execution control
from api.routes.root import router as root_router  # Root endpoint

# Import PostgreSQL checkpointer for conversation state persistence
from checkpointer.checkpointer.factory import (
    initialize_checkpointer,  # Setup PostgreSQL connection and tables
    cleanup_checkpointer,  # Close connections and cleanup resources
)


# ==============================================================================
# APPLICATION LIFESPAN MANAGEMENT
# ==============================================================================
# Context manager that handles startup and shutdown operations
# This ensures proper resource initialization and cleanup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifecycle: startup initialization and shutdown cleanup.

    Startup sequence:
    1. Record startup timestamp for uptime tracking
    2. Setup graceful shutdown handlers (SIGTERM, SIGINT)
    3. Initialize PostgreSQL checkpointer for conversation state
    4. Establish memory baseline for leak detection
    5. Start memory profiler (if enabled)
    6. Start background memory cleanup task

    Shutdown sequence:
    1. Stop memory profiler and report final statistics
    2. Stop background cleanup task
    3. Cleanup checkpointer and close database connections
    4. Log final memory statistics and detect leaks
    """
    # ==========================================================================
    # STARTUP SEQUENCE
    # ==========================================================================
    # pylint: disable=global-statement
    global _APP_STARTUP_TIME, _MEMORY_BASELINE
    _APP_STARTUP_TIME = datetime.now()

    print__startup_debug("🚀 FastAPI application starting up...")

    print__memory_monitoring(
        f"Application startup initiated at {_APP_STARTUP_TIME.isoformat()}"
    )
    log_memory_usage("app_startup")

    # ROUTE REGISTRATION MONITORING: Track all routes that get registered
    # This prevents the "needle in a haystack" memory leak pattern where routes
    # are accidentally registered multiple times during hot reloads or improper setup
    # Reference: Common FastAPI memory leak patterns in production
    print__memory_monitoring(
        "🔍 Monitoring route registrations to prevent memory leaks..."
    )

    # Setup graceful shutdown handlers for SIGTERM and SIGINT
    # Ensures proper cleanup when container/process is terminated
    setup_graceful_shutdown()

    # Initialize PostgreSQL checkpointer for LangGraph conversation state persistence
    # Creates necessary tables and establishes connection pool
    await initialize_checkpointer()

    # Set memory baseline after initialization
    # This captures the "normal" memory footprint after all imports and setup
    # Later comparisons against this baseline help detect memory leaks
    if _MEMORY_BASELINE is None:
        try:
            process = psutil.Process()
            _MEMORY_BASELINE = process.memory_info().rss / 1024 / 1024  # Convert to MB
            print__memory_monitoring(
                f"Memory baseline established: {_MEMORY_BASELINE:.1f}MB RSS"
            )
        except Exception:  # pylint: disable=broad-except
            pass

    if MEMORY_PROFILER_ENABLED:
        try:
            start_memory_profiler(
                interval=MEMORY_PROFILER_INTERVAL,
                top_stats=MEMORY_PROFILER_TOP_STATS,
            )
        except Exception as profiler_error:  # pylint: disable=broad-except
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

    yield  # Application runs here, serving requests

    # ==========================================================================
    # SHUTDOWN SEQUENCE
    # ==========================================================================
    print__startup_debug("🛑 FastAPI application shutting down...")
    print__memory_monitoring(
        f"Application ran for {datetime.now() - _APP_STARTUP_TIME}"
    )

    if MEMORY_PROFILER_ENABLED:
        try:
            await stop_memory_profiler()
        except Exception as profiler_error:  # pylint: disable=broad-except
            print__memory_monitoring(
                f"⚠️ Failed to stop memory profiler: {profiler_error}"
            )

    # Stop memory cleanup task
    try:
        await stop_memory_cleanup()
    except Exception as cleanup_error:  # pylint: disable=broad-except
        print__memory_monitoring(
            f"⚠️ Failed to stop memory cleanup task: {cleanup_error}"
        )

    # Log final memory statistics and detect potential memory leaks
    # Compare final memory usage against baseline to identify abnormal growth
    if _MEMORY_BASELINE:
        try:
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024  # Current RSS in MB
            total_growth = final_memory - _MEMORY_BASELINE
            print__memory_monitoring(
                f"Final memory stats: Started={_MEMORY_BASELINE:.1f}MB, "
                f"Final={final_memory:.1f}MB, Growth={total_growth:.1f}MB"
            )
            # Alert if memory growth exceeds threshold (default: 1900 MB)
            # This indicates a potential memory leak requiring investigation
            if total_growth > GC_MEMORY_THRESHOLD:
                print__memory_monitoring(
                    "🚨 SIGNIFICANT MEMORY GROWTH DETECTED - investigate for leaks!"
                )
        except Exception:  # pylint: disable=broad-except
            pass

    # Cleanup PostgreSQL checkpointer: close connections and release resources
    await cleanup_checkpointer()


# ==============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ==============================================================================
# Create the main FastAPI application instance with OpenAPI configuration
# This serves as the central application object for all routes and middleware
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

# ==============================================================================
# MIDDLEWARE REGISTRATION
# ==============================================================================
# Setup CORS and compression using wrapper functions from api.middleware.cors
# These functions provide:
# - Environment-based CORS configuration (production-ready)
# - Brotli compression
# - Logging and monitoring integration

# Setup CORS middleware for cross-origin request handling
setup_cors_middleware(app)

# Setup Brotli middleware for response compression
setup_brotli_middleware(app)

# Setup throttling middleware for intelligent rate limiting with wait-instead-of-reject
setup_throttling_middleware(app)

# Setup memory monitoring middleware for tracking usage in heavy operations
setup_memory_monitoring_middleware(app)


# ==============================================================================
# EXCEPTION HANDLERS
# ==============================================================================
# Global exception handlers for comprehensive error handling across all endpoints
# These ensure consistent error responses and proper HTTP status codes


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with proper 422 status code.

    Triggered when request data doesn't match expected Pydantic models.
    Common causes:
    - Missing required fields
    - Wrong data types
    - Invalid enum values
    - Constraint violations

    Uses jsonable_encoder to avoid JSON serialization errors (observed 500 during dumps).
    Falls back to minimal structure if encoding fails.
    """
    print__debug(f"Validation error: {exc.errors()}")
    try:
        payload = {"detail": "Validation error", "errors": exc.errors()}
        return JSONResponse(status_code=422, content=jsonable_encoder(payload))
    except Exception as encoding_error:  # pylint: disable=broad-except
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
    """Handle HTTP exceptions with comprehensive debugging for 401 errors.

    Special handling for authentication errors (401):
    - Detailed logging of request context
    - Headers inspection
    - Client IP tracking
    - Full traceback

    This helps diagnose authentication issues in production.
    """

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
async def value_error_handler(_request: Request, exc: ValueError):
    """Handle ValueError exceptions as 400 Bad Request.

    ValueErrors typically indicate invalid input data that passed
    Pydantic validation but failed business logic validation.
    """
    print__debug(f"ValueError: {str(exc)}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    """Handle unexpected exceptions (500 Internal Server Error).

    Catches all unhandled exceptions to prevent application crashes.

    Behavior:
    - Development (DEBUG_TRACEBACK=1): Include full traceback in response
    - Production (DEBUG_TRACEBACK=0): Generic error message, log details internally

    This ensures the application never crashes and always returns a response.
    """
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


# ==============================================================================
# ROUTE REGISTRATION
# ==============================================================================
# Register all route routers with the FastAPI application
# Tags are used for OpenAPI documentation organization
# Order doesn't affect routing, but affects documentation display
print__memory_monitoring("[ROUTES] Registering route routers...")

app.include_router(root_router, tags=["Root"])  # GET / - Welcome message
app.include_router(
    health_router, tags=["Health & Monitoring"]
)  # GET /health, /debug/pool-status, etc.
app.include_router(
    catalog_router, tags=["Data Catalog"]
)  # GET /catalog/* - Browse CZSU data
app.include_router(
    analysis_router, tags=["Query Analysis"]
)  # POST /analyze - NL-to-SQL conversion
app.include_router(
    feedback_router, tags=["Feedback & Sentiment"]
)  # POST /feedback - User feedback
app.include_router(
    chat_router, tags=["Chat & Threads"]
)  # POST /chat/threads, GET /chat/threads/*
app.include_router(
    messages_router, tags=["Messages"]
)  # GET /chat/messages/* - Message retrieval
app.include_router(
    bulk_router, tags=["Bulk Operations"]
)  # POST /bulk/* - Batch operations
app.include_router(
    debug_router, tags=["Debug & Admin"]
)  # GET /debug/* - Internal diagnostics
app.include_router(misc_router, tags=["Utilities"])  # Miscellaneous utility endpoints
app.include_router(
    stop_router, tags=["Execution Control"]
)  # POST /stop - Cancel execution

print__memory_monitoring("[SUCCESS] All route routers registered successfully")

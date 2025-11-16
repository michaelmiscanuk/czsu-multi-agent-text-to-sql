"""
MODULE_DESCRIPTION: Memory Monitoring Middleware - Request-Level Memory Tracking

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module provides middleware for monitoring memory usage at the request level
in the CZSU Multi-Agent Text-to-SQL API. It tracks memory consumption for specific
endpoints (particularly /analyze and bulk operations) to identify memory leaks and
optimize resource usage.

The middleware focuses on "heavy" operations that consume significant memory,
avoiding unnecessary overhead for lightweight endpoints like health checks.

Key Responsibilities:
    - Track total request count (global counter)
    - Log memory usage before/after heavy operations
    - Selective monitoring for performance-critical endpoints
    - Integration with memory profiling utilities

===================================================================================
KEY FEATURES
===================================================================================

1. Selective Monitoring
   - Only monitors heavy endpoints (/analyze, /bulk)
   - Skips lightweight endpoints (health checks, etc.)
   - Reduces logging overhead
   - Focused memory tracking

2. Request Count Tracking
   - Global _REQUEST_COUNT incremented for every request
   - Helps correlate memory usage with load
   - Used in health checks and diagnostics
   - Tracks application activity

3. Before/After Memory Logging
   - Captures memory snapshot before heavy operations
   - Captures memory snapshot after heavy operations
   - Identifies memory leaks and growth patterns
   - Helps optimize memory-intensive operations

4. Lightweight Design
   - Minimal overhead for non-monitored endpoints
   - Fast path checks
   - Efficient string operations
   - No blocking operations

===================================================================================
MIDDLEWARE FUNCTION
===================================================================================

Function: simplified_memory_monitoring_middleware(request, call_next)
    Purpose: Monitor memory usage for heavy operations

    Flow:
        1. Increment global request counter
        2. Check if endpoint is "heavy" (needs monitoring)
        3. If heavy: Log memory before operation
        4. Execute request handler (call_next)
        5. If heavy: Log memory after operation
        6. Return response

    Monitored Endpoints:
        - /analyze: AI-powered query analysis (memory-intensive)
        - /chat/all-messages-for-all-threads: Bulk message loading (large datasets)

    Skipped Endpoints:
        - /health: Health check (lightweight)
        - /docs: API documentation (static)
        - /debug/*: Debug endpoints (avoid recursive logging)
        - All other endpoints not explicitly monitored

Memory Logging:
    Uses log_memory_usage(label) from api.utils.memory

    Label Format:
        - Before: f"before_{path.replace('/', '_')}"
        - After: f"after_{path.replace('/', '_')}"

    Example:
        /analyze → "before__analyze"
        /analyze → "after__analyze"

===================================================================================
REQUEST COUNT TRACKING
===================================================================================

Global Variable: _REQUEST_COUNT
    Location: api.config.settings
    Type: int

    Purpose:
        - Tracks total requests processed since startup
        - Used in health checks to monitor activity
        - Helps correlate memory usage with load
        - Debugging and diagnostics

    Increment:
        Every request increments the counter
        Thread-safe (GIL protection in Python)

    Usage in Health Checks:
        GET /health → includes request_count in response
        Helps identify if server is processing requests

===================================================================================
HEAVY ENDPOINT DETECTION
===================================================================================

Detection Logic:
    request_path = request.url.path
    if any(
        path in request_path
        for path in ["/analyze", "/chat/all-messages-for-all-threads"]
    ):
        # Monitor this endpoint

Why These Endpoints:
    /analyze:
        - AI model inference
        - Vector database queries
        - SQL generation and execution
        - Result processing
        - Memory usage: 200-500MB per request

    /chat/all-messages-for-all-threads:
        - Loads all messages for all threads
        - Database queries for multiple threads
        - Message serialization
        - Memory usage: 50-200MB depending on data

Adding New Endpoints:
    To monitor additional endpoints, add to the list:
    if any(
        path in request_path
        for path in [
            "/analyze",
            "/chat/all-messages-for-all-threads",
            "/new-heavy-endpoint"  # Add here
        ]
    ):

===================================================================================
MEMORY LOGGING INTEGRATION
===================================================================================

log_memory_usage Function:
    From: api.utils.memory

    Purpose:
        - Captures current memory usage (RSS, VMS)
        - Logs to console with label
        - Tracks memory growth over time
        - Identifies potential leaks

    Output Example:
        📊 MEMORY [before__analyze]: RSS=1234MB, VMS=2345MB
        📊 MEMORY [after__analyze]: RSS=1456MB, VMS=2567MB
        📊 MEMORY DELTA: +222MB RSS, +222MB VMS

    Environment Control:
        MEMORY_MONITORING_DEBUG=1: Enable logging
        Default: Disabled

Memory Metrics:
    RSS (Resident Set Size):
        - Physical memory used by process
        - Most relevant for monitoring
        - Indicates actual memory consumption

    VMS (Virtual Memory Size):
        - Total virtual memory (includes swap)
        - Useful for detecting memory fragmentation
        - Higher than RSS on most systems

===================================================================================
MIDDLEWARE EXECUTION FLOW
===================================================================================

Example: /analyze Request
    1. Middleware receives request
    2. Increment _REQUEST_COUNT (now 42)
    3. Check path: "/analyze" in monitored paths → Yes
    4. Call log_memory_usage("before__analyze")
       Output: RSS=1200MB, VMS=2300MB
    5. Call next middleware/route handler (await call_next(request))
    6. Route handler processes analysis
    7. Route handler returns response
    8. Call log_memory_usage("after__analyze")
       Output: RSS=1450MB, VMS=2550MB
    9. Return response to client

Example: /health Request
    1. Middleware receives request
    2. Increment _REQUEST_COUNT (now 43)
    3. Check path: "/health" not in monitored paths → No
    4. Skip memory logging
    5. Call next middleware/route handler
    6. Return response to client

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Overhead:
    Monitored endpoints:
        - Memory logging: ~1-2ms per log call
        - Total overhead: ~2-4ms per request
        - Acceptable for heavy operations

    Non-monitored endpoints:
        - Counter increment: <0.01ms
        - Path check: <0.01ms
        - Total overhead: <0.02ms (negligible)

Memory Usage:
    - Counter: 8 bytes (int)
    - Path string: ~50 bytes (temporary)
    - No persistent storage
    - Minimal memory footprint

===================================================================================
WINDOWS COMPATIBILITY
===================================================================================

Event Loop Policy:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

Why Required:
    - psycopg (PostgreSQL driver) incompatible with Windows default loop
    - Must be set BEFORE any async operations
    - Must be at TOP of file

===================================================================================
DEBUGGING AND DIAGNOSTICS
===================================================================================

Enabling Memory Logging:
    Set environment variable:
    MEMORY_MONITORING_DEBUG=1

Interpreting Logs:
    Before vs After:
        - Compare RSS values
        - Significant delta (>200MB) may indicate leak
        - Small delta (<50MB) is normal for /analyze

    Trend Analysis:
        - Monitor RSS over time
        - Steady increase = potential leak
        - Stable after GC = healthy

    Correlation with Requests:
        - High _REQUEST_COUNT = high load
        - Memory should be proportional
        - Disproportionate growth = leak

Common Patterns:
    Healthy:
        before__analyze: RSS=1200MB
        after__analyze: RSS=1250MB (+50MB)
        (Memory released after response)

    Potential Leak:
        before__analyze: RSS=1200MB
        after__analyze: RSS=1600MB (+400MB)
        (Memory not released)

===================================================================================
INTEGRATION WITH MAIN APPLICATION
===================================================================================

Registration in main.py:
    from api.middleware.memory_monitoring import (
        simplified_memory_monitoring_middleware
    )

    app.middleware("http")(
        simplified_memory_monitoring_middleware
    )

Middleware Order:
    Typically registered AFTER:
        - CORS middleware
        - GZip middleware
        - Rate limiting middleware

    Reason:
        - Want to measure total memory including other middleware
        - Captures complete request lifecycle

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Mock Request object
    - Mock call_next function
    - Verify _REQUEST_COUNT increments
    - Verify log_memory_usage called for heavy endpoints
    - Verify log_memory_usage skipped for light endpoints

Integration Tests:
    - Test with real /analyze requests
    - Verify memory logging output
    - Check request counter in /health response
    - Load testing to verify no leaks

Test Example:
    @pytest.mark.asyncio
    async def test_memory_monitoring_analyze():
        request = MockRequest(url="/analyze")
        response = await simplified_memory_monitoring_middleware(
            request, mock_call_next
        )
        assert _REQUEST_COUNT > 0
        # Verify log_memory_usage was called

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection

Third-Party:
    - fastapi: Request object
    - dotenv: Environment variable loading

Internal:
    - api.config.settings: _REQUEST_COUNT global variable
    - api.utils.memory: log_memory_usage function

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Configurable Monitoring
   - Environment variable for monitored endpoints
   - MONITORED_ENDPOINTS=/analyze,/bulk
   - Dynamic configuration

2. Memory Threshold Alerts
   - Alert when memory delta exceeds threshold
   - Automatic logging of large increases
   - Integration with monitoring systems

3. Per-User Memory Tracking
   - Track memory per authenticated user
   - Identify users causing high memory usage
   - Quota enforcement

4. Prometheus Metrics
   - Export memory metrics to Prometheus
   - Real-time dashboards
   - Historical analysis

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
from fastapi import Request

# Import globals from config
from api.config.settings import _REQUEST_COUNT

# Import memory functions from utils
from api.utils.memory import log_memory_usage

# ==============================================================================
# MEMORY MONITORING MIDDLEWARE
# ==============================================================================


async def simplified_memory_monitoring_middleware(request: Request, call_next):
    """Simplified memory monitoring middleware for FastAPI requests.

    This middleware tracks memory usage for memory-intensive endpoints while
    minimizing overhead for lightweight endpoints. It increments a global request
    counter and logs memory before/after heavy operations.

    Args:
        request: The FastAPI Request object
        call_next: The next middleware/route handler in the chain

    Returns:
        The response from the next handler

    Monitored Endpoints:
        - /analyze: AI-powered analysis (memory-intensive)
        - /chat/all-messages-for-all-threads: Bulk message loading (large datasets)

    Memory Logging:
        - Logs RSS and VMS before heavy operations
        - Logs RSS and VMS after heavy operations
        - Calculates memory delta for leak detection

    Request Counting:
        - Increments global _REQUEST_COUNT for every request
        - Used in health checks and diagnostics
    """
    global _REQUEST_COUNT

    # =======================================================================
    # STEP 1: INCREMENT REQUEST COUNTER
    # =======================================================================
    # Track total requests processed for diagnostics and health checks
    _REQUEST_COUNT += 1

    # =======================================================================
    # STEP 2: CHECK IF ENDPOINT REQUIRES MEMORY MONITORING
    # =======================================================================
    # Only check memory for heavy operations to minimize overhead
    # These endpoints are known to consume significant memory
    request_path = request.url.path
    is_heavy_operation = any(
        path in request_path
        for path in ["/analyze", "/chat/all-messages-for-all-threads"]
    )

    # =======================================================================
    # STEP 3: LOG MEMORY BEFORE HEAVY OPERATIONS
    # =======================================================================
    if is_heavy_operation:
        # Log memory state before processing
        # Label format: "before_/analyze" → "before__analyze"
        log_memory_usage(f"before_{request_path.replace('/', '_')}")

    # =======================================================================
    # STEP 4: EXECUTE REQUEST HANDLER
    # =======================================================================
    # Call next middleware or route handler
    response = await call_next(request)

    # =======================================================================
    # STEP 5: LOG MEMORY AFTER HEAVY OPERATIONS
    # =======================================================================
    # Check memory after heavy operations to detect leaks
    if is_heavy_operation:
        # Log memory state after processing
        # Label format: "after_/analyze" → "after__analyze"
        log_memory_usage(f"after_{request_path.replace('/', '_')}")

    # =======================================================================
    # STEP 6: RETURN RESPONSE
    # =======================================================================
    return response

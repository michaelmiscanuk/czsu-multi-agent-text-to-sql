"""
MODULE_DESCRIPTION: Rate Limiting Middleware - Request Throttling and DDoS Protection

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module implements rate limiting middleware for the CZSU Multi-Agent Text-to-SQL
API, protecting the service from abuse, DDoS attacks, and resource exhaustion.

Instead of immediately rejecting requests that exceed rate limits, this middleware
implements a "wait-and-retry" strategy that attempts to delay requests until capacity
is available. Only if waiting exceeds the maximum wait time does it reject the request.

Key Features:
    - Per-IP rate limiting (sliding window algorithm)
    - Semaphore-based concurrency control (max 8 concurrent per IP)
    - Wait-and-retry strategy (up to 5 seconds)
    - Selective endpoint exemption (health checks, docs)
    - Comprehensive error logging and monitoring

===================================================================================
KEY FEATURES
===================================================================================

1. Wait-and-Retry Strategy
   - Attempts to wait for rate limit availability
   - Maximum wait time: 5 seconds (configurable)
   - Only rejects after exhausting wait attempts
   - Better user experience than instant rejection

2. Per-IP Concurrency Control
   - Semaphore limits concurrent requests per IP
   - Default: 8 concurrent requests per IP
   - Prevents single IP from monopolizing resources
   - Fair resource allocation

3. Sliding Window Rate Limiting
   - Tracks request timestamps per IP
   - Configurable window (default: 60 seconds)
   - Configurable limits (default: 100 + 20 burst)
   - Automatic cleanup of old timestamps

4. Endpoint Exemption
   - Health checks bypassed (/health)
   - API documentation bypassed (/docs, /openapi.json)
   - Debug endpoints bypassed (/debug/pool-status)
   - Reduces unnecessary overhead

5. Comprehensive Error Logging
   - Logs rate limit violations
   - Tracks client IP and request details
   - Integration with error logging utilities
   - Monitoring and alerting support

===================================================================================
MIDDLEWARE FUNCTION
===================================================================================

Function: throttling_middleware(request, call_next)
    Purpose: Enforce rate limits and concurrency controls

    Flow:
        1. Check if endpoint is exempted
        2. Extract client IP address
        3. Acquire semaphore for IP (limits concurrency)
        4. Wait for rate limit availability
        5. If wait successful: Process request
        6. If wait failed: Return 429 Too Many Requests
        7. Release semaphore

    Exempted Endpoints:
        - /health: Health checks (monitoring)
        - /docs: API documentation
        - /openapi.json: OpenAPI schema
        - /debug/pool-status: Debug endpoint

    Response on Rejection:
        Status: 429 Too Many Requests
        Body: {
            "detail": "Rate limit exceeded. Please wait Xs before retrying.",
            "retry_after": X,
            "burst_usage": "Y/Z",
            "window_usage": "A/B"
        }
        Headers: {
            "Retry-After": "X"
        }

===================================================================================
RATE LIMITING STRATEGY
===================================================================================

Sliding Window Algorithm:
    1. Store timestamps of recent requests per IP
    2. Filter timestamps within the window (last 60s)
    3. Count valid timestamps
    4. Allow if count < RATE_LIMIT_REQUESTS + RATE_LIMIT_BURST
    5. Wait if count >= limit but < max wait time
    6. Reject if wait time exceeds maximum

Configuration:
    RATE_LIMIT_REQUESTS: 100 requests
    RATE_LIMIT_WINDOW: 60 seconds
    RATE_LIMIT_BURST: 20 additional requests
    RATE_LIMIT_MAX_WAIT: 5 seconds

Example Calculation:
    IP has made 110 requests in last 60 seconds
    Limit: 100 + 20 burst = 120 total
    Current: 110 requests
    Remaining: 10 requests
    Action: Allow request

Wait-and-Retry Logic:
    Uses wait_for_rate_limit(client_ip) function:
        - Calculates suggested wait time
        - Sleeps if wait time < RATE_LIMIT_MAX_WAIT
        - Returns True if wait successful
        - Returns False if wait exceeds maximum

===================================================================================
CONCURRENCY CONTROL
===================================================================================

Semaphore Strategy:
    Each IP has dedicated semaphore (auto-created)
    Limit: 8 concurrent requests per IP

    Purpose:
        - Prevents single IP from monopolizing resources
        - Fair allocation across users
        - Protects against slow loris attacks

Semaphore Storage:
    throttle_semaphores = defaultdict(
        lambda: asyncio.Semaphore(8)
    )

    Key: Client IP address
    Value: asyncio.Semaphore with limit 8
    Auto-creates semaphore for new IPs

Execution Flow:
    async with semaphore:
        # Only 8 requests from this IP can be here
        # Wait for rate limit
        # Process request
    # Semaphore released, next request can proceed

===================================================================================
WAIT-AND-RETRY IMPLEMENTATION
===================================================================================

Function: wait_for_rate_limit(client_ip)
    From: api.utils.rate_limiting

    Purpose:
        - Check if rate limit allows request
        - If not, calculate wait time
        - Sleep if wait time reasonable
        - Return success/failure

    Flow:
        1. Check current rate limit status
        2. If allowed: Return True immediately
        3. If blocked: Calculate suggested wait time
        4. If wait time < RATE_LIMIT_MAX_WAIT:
           - Sleep for suggested time
           - Re-check rate limit
           - Return result
        5. If wait time >= RATE_LIMIT_MAX_WAIT:
           - Return False (give up)

Benefits:
    - Better UX (requests wait instead of instant fail)
    - Smoother traffic flow
    - Reduces client-side retry logic
    - Handles burst traffic gracefully

===================================================================================
ERROR HANDLING AND LOGGING
===================================================================================

Rejection Logging:
    log_comprehensive_error(
        "rate_limit_exceeded_after_wait",
        Exception(error_msg),
        request
    )

    Logged Information:
        - Client IP address
        - Burst usage (X/Y)
        - Window usage (A/B)
        - Suggested wait time
        - Request URL and method

Response Details:
    detail: Human-readable error message
    retry_after: Seconds to wait before retry
    burst_usage: Current/limit for burst
    window_usage: Current/limit for window

Example Response:
    {
        "detail": "Rate limit exceeded. Please wait 3.2s before retrying.",
        "retry_after": 4,
        "burst_usage": "25/20",
        "window_usage": "125/100"
    }

===================================================================================
EXEMPTED ENDPOINTS
===================================================================================

Bypassed Paths:
    - /health: Monitoring and health checks
    - /docs: Swagger UI documentation
    - /openapi.json: OpenAPI schema
    - /debug/pool-status: Database pool monitoring

Reason for Exemption:
    - Health checks must always work (monitoring)
    - Documentation should be freely accessible
    - Debug endpoints for troubleshooting
    - No rate limiting overhead for these paths

Adding New Exemptions:
    if request.url.path in [
        "/health",
        "/docs",
        "/openapi.json",
        "/debug/pool-status",
        "/new-exempted-path"  # Add here
    ]:
        return await call_next(request)

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
INTEGRATION WITH MAIN APPLICATION
===================================================================================

Registration in main.py:
    from api.middleware.rate_limiting import throttling_middleware

    app.middleware("http")(throttling_middleware)

Middleware Order:
    Typically registered AFTER:
        - CORS middleware
        - Brotli middleware

    Typically registered BEFORE:
        - Memory monitoring
        - Authentication (applies to all requests)

===================================================================================
PERFORMANCE CHARACTERISTICS
===================================================================================

Overhead:
    Exempted endpoints: <0.01ms (path check only)
    Rate-limited endpoints:
        - Within limits: ~0.5-1ms (timestamp check)
        - With wait: 100ms - 5000ms (sleep time)
        - Rejected: ~0.5-1ms (check + response)

Memory Usage:
    - Semaphore per IP: ~200 bytes
    - Timestamp list per IP: ~50 bytes per timestamp
    - Typical: 1-5 KB per active IP
    - Auto-cleanup prevents growth

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. DDoS Protection
   - Rate limiting prevents request flooding
   - Per-IP limits prevent single-source attacks
   - Concurrency limits prevent resource exhaustion

2. IP Spoofing
   - Uses request.client.host (trusted by reverse proxy)
   - Configure X-Forwarded-For in production
   - Validate IP addresses

3. Bypass Prevention
   - No authentication bypass for rate limits
   - Applies to all non-exempted endpoints
   - Cannot be disabled per-request

4. Information Disclosure
   - Error messages don't reveal internal state
   - Generic "rate limit exceeded" message
   - Retry-After header guides clients

===================================================================================
MONITORING AND ALERTING
===================================================================================

Metrics to Track:
    - Rate limit rejections per IP
    - Total rejections per time period
    - Average wait times
    - Semaphore saturation (all slots full)

Alerting Triggers:
    - High rejection rate (>10% of requests)
    - Single IP with many rejections (attack)
    - All semaphore slots full (capacity issue)
    - Increasing average wait times (load issue)

Log Analysis:
    grep "rate_limit_exceeded_after_wait" logs
    - Identify problematic IPs
    - Track rejection patterns
    - Tune rate limit settings

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Test exempted endpoints bypass
    - Test rate limit enforcement
    - Test wait-and-retry logic
    - Test 429 response format

Load Tests:
    - Simulate burst traffic
    - Test concurrent request limits
    - Verify semaphore behavior
    - Check wait times under load

Test Examples:
    def test_exempted_endpoint():
        # Health check should bypass rate limiting
        for i in range(200):  # Exceed limit
            response = client.get("/health")
            assert response.status_code == 200

    def test_rate_limit_enforcement():
        # Exceed rate limit
        for i in range(150):
            response = client.post("/analyze", ...)
        # Should eventually get 429

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection

Third-Party:
    - fastapi: Request, JSONResponse
    - dotenv: Environment variable loading

Internal:
    - api.config.settings: throttle_semaphores
    - api.utils.memory: log_comprehensive_error
    - api.utils.rate_limiting: check_rate_limit_with_throttling, wait_for_rate_limit

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Per-User Rate Limits
   - Different limits for authenticated users
   - Tiered limits (free vs premium)
   - User-specific burst allowances

2. Adaptive Rate Limiting
   - Adjust limits based on server load
   - Increase limits when idle
   - Decrease limits under high load

3. Redis-Based Storage
   - Distributed rate limiting
   - Share limits across multiple API instances
   - Horizontal scaling support

4. Advanced Monitoring
   - Real-time rate limit dashboards
   - Automatic IP blocking for abuse
   - Machine learning anomaly detection

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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import globals from config
from api.config.settings import throttle_semaphores

# Import memory functions from utils
from api.utils.memory import log_comprehensive_error, print__memory_monitoring

# Import rate limiting functions from utils
from api.utils.rate_limiting import (
    check_rate_limit_with_throttling,
    wait_for_rate_limit,
)

# ==============================================================================
# RATE LIMITING MIDDLEWARE
# ==============================================================================


async def throttling_middleware(request: Request, call_next):
    """Throttling middleware that makes requests wait instead of rejecting them.

    This middleware implements a wait-and-retry strategy for rate limiting instead
    of immediately rejecting requests that exceed limits. It uses per-IP semaphores
    to control concurrency and attempts to wait for rate limit capacity before
    giving up and returning a 429 error.

    Args:
        request: The FastAPI Request object
        call_next: The next middleware/route handler in the chain

    Returns:
        The response from the next handler, or a 429 JSONResponse if rate limit
        exceeded and wait time exceeded maximum

    Exempted Endpoints:
        - /health: Health checks (monitoring)
        - /docs: API documentation
        - /openapi.json: OpenAPI schema
        - /debug/pool-status: Database pool status

    Rate Limit Response (429):
        {
            "detail": "Rate limit exceeded. Please wait Xs before retrying.",
            "retry_after": X,
            "burst_usage": "Y/Z",
            "window_usage": "A/B"
        }
    """
    # =======================================================================
    # STEP 1: CHECK FOR EXEMPTED ENDPOINTS
    # =======================================================================
    # Skip throttling for health checks and static endpoints
    # These endpoints must always be accessible for monitoring and documentation
    if request.url.path in ["/health", "/docs", "/openapi.json", "/debug/pool-status"]:
        return await call_next(request)

    # =======================================================================
    # STEP 2: EXTRACT CLIENT IP ADDRESS
    # =======================================================================
    # Get client IP for per-IP rate limiting and concurrency control
    # Falls back to "unknown" if client info not available
    client_ip = request.client.host if request.client else "unknown"

    # =======================================================================
    # STEP 3: ACQUIRE SEMAPHORE FOR CONCURRENCY CONTROL
    # =======================================================================
    # Use semaphore to limit concurrent requests per IP
    # Auto-creates new semaphore for new IPs (defaultdict)
    # Maximum 8 concurrent requests per IP
    semaphore = throttle_semaphores[client_ip]

    async with semaphore:
        # =======================================================================
        # STEP 4: WAIT FOR RATE LIMIT AVAILABILITY
        # =======================================================================
        # Try to wait for rate limit instead of immediately rejecting
        # Only reject if we can't wait (wait time too long or max attempts exceeded)
        if not await wait_for_rate_limit(client_ip):
            # =======================================================================
            # STEP 5: RATE LIMIT EXCEEDED - PREPARE REJECTION RESPONSE
            # =======================================================================
            # Only reject if we can't wait (wait time too long or max attempts exceeded)
            rate_info = check_rate_limit_with_throttling(client_ip)
            error_msg = (
                f"Rate limit exceeded for IP: {client_ip} after waiting. "
                f"Burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, "
                f"Window: {rate_info['window_count']}/{rate_info['window_limit']}"
            )

            # Log comprehensive error with request context
            log_comprehensive_error(
                "rate_limit_exceeded_after_wait",
                Exception(error_msg),
                request,
            )

            # =======================================================================
            # STEP 6: RETURN 429 TOO MANY REQUESTS RESPONSE
            # =======================================================================
            # Prepare detailed error response with usage statistics and retry guidance
            response_content = {
                "detail": (
                    f"Rate limit exceeded. Please wait "
                    f"{rate_info['suggested_wait']:.1f}s before retrying."
                ),
                "retry_after": max(rate_info["suggested_wait"], 1),
                "burst_usage": f"{rate_info['burst_count']}/{rate_info['burst_limit']}",
                "window_usage": f"{rate_info['window_count']}/{rate_info['window_limit']}",
            }
            return JSONResponse(
                status_code=429,
                content=response_content,
                headers={"Retry-After": str(max(int(rate_info["suggested_wait"]), 1))},
            )

        # =======================================================================
        # STEP 7: RATE LIMIT OK - PROCESS REQUEST
        # =======================================================================
        # Rate limit check passed (or wait was successful)
        # Proceed with request processing
        return await call_next(request)


# ==============================================================================
# MIDDLEWARE SETUP FUNCTION
# ==============================================================================


def setup_throttling_middleware(app: FastAPI):
    """Setup rate limiting middleware for the FastAPI application.

    Implements intelligent throttling that makes requests wait instead of
    immediately rejecting them when rate limits are exceeded.

    Features:
    - Per-IP rate limiting (sliding window algorithm)
    - Semaphore-based concurrency control (max 8 concurrent per IP)
    - Wait-and-retry strategy (up to 5 seconds)
    - Selective endpoint exemption (health checks, docs)

    Args:
        app: The FastAPI application instance

    Exempted Endpoints:
        - /health: Health checks (monitoring)
        - /docs: API documentation
        - /openapi.json: OpenAPI schema
        - /debug/pool-status: Database pool monitoring
    """
    print__memory_monitoring("📋 Registering rate limiting middleware...")
    app.middleware("http")(throttling_middleware)
    print__memory_monitoring("✅ Rate limiting middleware registered successfully")

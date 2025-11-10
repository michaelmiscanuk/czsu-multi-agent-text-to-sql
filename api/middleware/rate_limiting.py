"""Rate limiting middleware for FastAPI application."""

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
from fastapi.responses import JSONResponse

# Import globals from config
from api.config.settings import throttle_semaphores

# Import memory functions from utils
from api.utils.memory import log_comprehensive_error

# Import rate limiting functions from utils
from api.utils.rate_limiting import (
    check_rate_limit_with_throttling,
    wait_for_rate_limit,
)

# ============================================================
# RATE LIMITING MIDDLEWARE
# ============================================================


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
            error_msg = (
                f"Rate limit exceeded for IP: {client_ip} after waiting. "
                f"Burst: {rate_info['burst_count']}/{rate_info['burst_limit']}, "
                f"Window: {rate_info['window_count']}/{rate_info['window_limit']}"
            )
            log_comprehensive_error(
                "rate_limit_exceeded_after_wait",
                Exception(error_msg),
                request,
            )
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

        return await call_next(request)

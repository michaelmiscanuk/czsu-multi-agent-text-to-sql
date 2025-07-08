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
from api.config.settings import _request_count

# Import memory functions from utils
from api.utils.memory import log_memory_usage

# ============================================================
# MEMORY MONITORING MIDDLEWARE
# ============================================================


async def simplified_memory_monitoring_middleware(request: Request, call_next):
    """Simplified memory monitoring middleware."""
    global _request_count

    _request_count += 1

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

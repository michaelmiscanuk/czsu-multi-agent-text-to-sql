"""
Routes package for the API server.

This package contains FastAPI route handlers for health checks,
analysis, feedback, chat management, and other endpoints
for the CZSU Multi-Agent Text-to-SQL application.
"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Routes module initialization
from .health import router as health_router
from .chat import router as chat_router
from .catalog import router as catalog_router
from .messages import router as messages_router
from .feedback import router as feedback_router
from .analysis import router as analysis_router
from .bulk import router as bulk_router
from .debug import router as debug_router
from .misc import router as misc_router
from .stop import router as stop_router

# Export all routers for easy import
__all__ = [
    "health_router",
    "chat_router",
    "catalog_router",
    "messages_router",
    "feedback_router",
    "analysis_router",
    "bulk_router",
    "debug_router",
    "misc_router",
    "stop_router",
]

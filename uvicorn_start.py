#!/usr/bin/env python3
"""
CZSU Multi-Agent Text-to-SQL API Server
Uvicorn start script - uses modular FastAPI app from api.main
"""

from api.main import app
import os
import sys

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Import the modular FastAPI app

# For development server
if __name__ == "__main__":
    import uvicorn

    # More robust reload configuration for Windows
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "my_agent"],  # Specify directories to watch
        reload_delay=0.25,  # Add small delay to prevent multiple reloads
        log_level="info",
        use_colors=True,
        access_log=True
    )

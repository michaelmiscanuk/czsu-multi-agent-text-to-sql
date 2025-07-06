#!/usr/bin/env python3
"""
CZSU Multi-Agent Text-to-SQL API Server
Uvicorn start script - uses modular FastAPI app from api.main
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

# Import the modular FastAPI app
from api.main import app

# For development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
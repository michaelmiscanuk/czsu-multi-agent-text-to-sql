#!/usr/bin/env python3
"""
Custom uvicorn startup script that ensures proper event loop policy for PostgreSQL on Windows
"""

import sys
import os

# Load .env file early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use environment variables as-is

# CRITICAL: Set Windows event loop policy BEFORE any other imports
if sys.platform == "win32":
    import asyncio
    print("🔧 Uvicorn Startup: Setting WindowsSelectorEventLoopPolicy for PostgreSQL compatibility")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"✅ Event loop policy set to: {type(asyncio.get_event_loop_policy()).__name__}")

# Now import uvicorn and start the server
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting uvicorn with SelectorEventLoop...")
    
    # Start uvicorn with the correct event loop policy already set
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    ) 
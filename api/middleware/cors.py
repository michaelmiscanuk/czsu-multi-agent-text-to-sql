"""CORS and compression middleware setup for FastAPI application."""

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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Import memory monitoring functions from utils
from api.utils.memory import print__memory_monitoring

# ============================================================
# MIDDLEWARE SETUP - CORS AND GZIP
# ============================================================


def setup_cors_middleware(app: FastAPI):
    """Setup CORS middleware for the FastAPI application."""

    # Monitor route registrations (including middleware and CORS)
    print__memory_monitoring("📋 Registering CORS middleware...")
    # Note: Route registration monitoring happens at runtime to avoid import-time global variable access

    # Allow CORS for local frontend dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_gzip_middleware(app: FastAPI):
    """Setup GZip compression middleware for the FastAPI application."""

    print__memory_monitoring("📋 Registering GZip middleware...")
    # Add GZip compression to reduce response sizes and memory usage
    app.add_middleware(GZipMiddleware, minimum_size=1000)

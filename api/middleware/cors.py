"""
MODULE_DESCRIPTION: CORS and Compression Middleware - Cross-Origin and Performance Setup

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module configures two essential middleware components for the CZSU Multi-Agent
Text-to-SQL API:

1. CORS (Cross-Origin Resource Sharing) Middleware:
   - Enables frontend applications from different origins to access the API
   - Critical for React/Next.js frontends running on different ports/domains
   - Configures allowed origins, methods, headers, and credentials

2. GZip Compression Middleware:
   - Reduces response payload sizes for better performance
   - Compresses JSON responses before sending to clients
   - Reduces bandwidth usage and improves load times
   - Especially beneficial for large bulk message responses

Both middleware components are registered during application startup and process
every HTTP request/response pair.

===================================================================================
KEY FEATURES
===================================================================================

1. CORS Configuration
   - Wildcard origin allowance (*) for development flexibility
   - Credential support for authenticated requests
   - All HTTP methods allowed (GET, POST, PUT, DELETE, etc.)
   - All headers allowed (Authorization, Content-Type, etc.)

2. GZip Compression
   - Automatic response compression
   - Configurable minimum size threshold (1000 bytes)
   - Transparent to application code
   - Browser-compatible compression

3. Memory Monitoring Integration
   - Logs middleware registration events
   - Tracks application initialization progress
   - Helps diagnose startup issues

4. Production-Ready Configuration
   - Easy to adjust for production security
   - Environment-based configuration support
   - Documented security considerations

===================================================================================
CORS MIDDLEWARE
===================================================================================

Function: setup_cors_middleware(app: FastAPI)
    Purpose: Configure CORS middleware for cross-origin requests

    Configuration:
        allow_origins: ["*"]
            - Allows requests from any origin
            - Development-friendly
            - PRODUCTION WARNING: Restrict to specific domains

        allow_credentials: True
            - Allows cookies and Authorization headers
            - Required for JWT authentication
            - Enables authenticated cross-origin requests

        allow_methods: ["*"]
            - Allows all HTTP methods
            - GET, POST, PUT, DELETE, PATCH, OPTIONS
            - Supports full RESTful API operations

        allow_headers: ["*"]
            - Allows all custom headers
            - Authorization, Content-Type, etc.
            - Flexible for various client needs

Production Security Recommendations:
    - Replace allow_origins=["*"] with specific domains:
      allow_origins=[
          "https://app.example.com",
          "https://www.example.com"
      ]

    - Consider restricting methods if not all are needed:
      allow_methods=["GET", "POST", "PUT", "DELETE"]

    - Limit headers to those actually used:
      allow_headers=["Authorization", "Content-Type"]

CORS Flow:
    1. Browser sends preflight OPTIONS request
    2. CORS middleware intercepts request
    3. Checks origin against allow_origins
    4. Returns appropriate CORS headers
    5. Browser allows/blocks subsequent request
    6. Actual request proceeds if allowed

Example Scenario:
    Frontend: http://localhost:3000 (Next.js dev server)
    Backend: http://localhost:8000 (FastAPI)

    Without CORS:
        Browser blocks requests due to different origins

    With CORS:
        Middleware adds headers allowing cross-origin access
        Browser permits requests to proceed

===================================================================================
GZIP COMPRESSION MIDDLEWARE
===================================================================================

Function: setup_gzip_middleware(app: FastAPI)
    Purpose: Enable automatic response compression

    Configuration:
        minimum_size: 1000
            - Only compress responses >= 1000 bytes
            - Avoids overhead for small responses
            - Optimal balance for performance

    Compression Algorithm:
        - Uses standard GZip compression
        - Browser-compatible (all modern browsers)
        - Transparent decompression by client

Compression Benefits:
    1. Bandwidth Reduction:
       - JSON responses: 70-90% size reduction
       - Large bulk messages: Significant savings
       - Lower network costs

    2. Faster Load Times:
       - Smaller payloads = faster transfer
       - Especially beneficial on slow connections
       - Better user experience

    3. Memory Efficiency:
       - Compressed responses use less memory
       - Helps with large result sets
       - Reduces memory pressure

Example Compression:
    Original JSON response: 50 KB
    Compressed response: 10 KB (80% reduction)
    Transfer time on 1 Mbps: 400ms → 80ms

When Compression Occurs:
    - Response size >= minimum_size (1000 bytes)
    - Client supports gzip (Accept-Encoding: gzip)
    - Content-Type is compressible (application/json, text/html)

When Compression Skipped:
    - Response < 1000 bytes (overhead not worth it)
    - Client doesn't support gzip (rare)
    - Response already compressed
    - Streaming responses

===================================================================================
MEMORY MONITORING INTEGRATION
===================================================================================

Logging Function: print__memory_monitoring(msg)
    Purpose: Track middleware registration during startup

    Environment Control:
        - MEMORY_MONITORING_DEBUG=1: Enable logging
        - Disabled by default

    Usage:
        print__memory_monitoring("📋 Registering CORS middleware...")
        print__memory_monitoring("📋 Registering GZip middleware...")

    Benefits:
        - Tracks initialization progress
        - Helps diagnose startup issues
        - Monitors middleware order
        - Debugging tool for configuration

Middleware Order:
    Middleware is executed in REVERSE order of registration:
        1. Last registered = First executed
        2. First registered = Last executed

    Current Order:
        Registration: CORS → GZip
        Execution: GZip → CORS → Route → CORS → GZip

    Why This Order:
        - CORS needs to process before route handlers
        - GZip compresses final responses
        - Works correctly for both orders

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
    - Only applies on Windows platform

===================================================================================
USAGE IN MAIN APPLICATION
===================================================================================

Integration in main.py:
    from api.middleware.cors import (
        setup_cors_middleware,
        setup_gzip_middleware
    )

    app = FastAPI()

    # Setup middleware
    setup_cors_middleware(app)
    setup_gzip_middleware(app)

    # Register routes
    app.include_router(routes)

Startup Sequence:
    1. Create FastAPI application
    2. Setup CORS middleware
    3. Setup GZip middleware
    4. Setup other middleware (rate limiting, memory monitoring)
    5. Register exception handlers
    6. Register routes
    7. Start application

===================================================================================
PERFORMANCE IMPACT
===================================================================================

CORS Overhead:
    - Minimal: ~0.1ms per request
    - Header checks and additions
    - Only OPTIONS requests have slight overhead

GZip Overhead:
    - Compression: 5-20ms for large responses
    - Decompression: Handled by browser (transparent)
    - Net benefit: Faster transfer time outweighs compression time

Memory Usage:
    - CORS: Negligible
    - GZip: Temporary buffer during compression
    - Released immediately after compression

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. CORS Security
   - Wildcard (*) allows any origin (development only)
   - Production: Restrict to trusted domains
   - Prevents unauthorized cross-origin access
   - Credential support requires careful origin control

2. GZip Security
   - BREACH attack: Compression + secrets = vulnerability
   - Mitigation: Don't compress responses with secrets
   - CSRF tokens should not be compressed
   - Generally safe for JSON APIs

3. Header Validation
   - All headers allowed (flexible but less secure)
   - Production: Restrict to required headers
   - Prevents header injection attacks

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

CORS Testing:
    - Test cross-origin requests from different origins
    - Verify preflight OPTIONS requests
    - Check CORS headers in responses
    - Test with credentials (cookies, auth headers)

GZip Testing:
    - Send Accept-Encoding: gzip header
    - Verify Content-Encoding: gzip in response
    - Check response size reduction
    - Test with various payload sizes

Integration Tests:
    from fastapi.testclient import TestClient

    def test_cors_headers():
        response = client.options("/analyze")
        assert "access-control-allow-origin" in response.headers

    def test_gzip_compression():
        headers = {"Accept-Encoding": "gzip"}
        response = client.get("/bulk", headers=headers)
        assert response.headers.get("content-encoding") == "gzip"

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection

Third-Party:
    - fastapi: FastAPI, CORSMiddleware, GZipMiddleware
    - dotenv: Environment variable loading

Internal:
    - api.utils.memory: Memory monitoring functions

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Environment-Based CORS
   - Different origins for dev/staging/production
   - Automatic configuration based on environment
   - Example: ALLOWED_ORIGINS env variable

2. Compression Tuning
   - Configurable compression level
   - Different minimum sizes per endpoint
   - Brotli compression for better ratios

3. CORS Presets
   - Predefined configurations for common scenarios
   - Development, staging, production presets
   - Easy switching via environment

4. Monitoring
   - Track compression ratios
   - Monitor CORS rejection rates
   - Performance metrics

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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Import memory monitoring functions from utils
from api.utils.memory import print__memory_monitoring

# ==============================================================================
# MIDDLEWARE SETUP - CORS AND GZIP
# ==============================================================================


def setup_cors_middleware(app: FastAPI):
    """Setup CORS middleware for the FastAPI application.

    Configures Cross-Origin Resource Sharing (CORS) to allow requests from
    frontend applications running on different origins (domains/ports).

    Args:
        app: The FastAPI application instance

    Configuration:
        - allow_origins: ["*"] - Allows requests from any origin (DEV ONLY)
        - allow_credentials: True - Enables cookies and auth headers
        - allow_methods: ["*"] - Allows all HTTP methods
        - allow_headers: ["*"] - Allows all headers

    Production Security:
        Replace allow_origins=["*"] with specific trusted domains:
        allow_origins=[
            "https://app.example.com",
            "https://www.example.com"
        ]
    """
    # =======================================================================
    # LOG MIDDLEWARE REGISTRATION
    # =======================================================================

    # Monitor route registrations (including middleware and CORS)
    print__memory_monitoring("📋 Registering CORS middleware...")
    # Note: Route registration monitoring happens at runtime to avoid import-time global variable access

    # =======================================================================
    # ADD CORS MIDDLEWARE
    # =======================================================================

    # Allow CORS for local frontend dev
    # SECURITY WARNING: allow_origins=["*"] is for development only
    # In production, restrict to specific trusted domains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production to specific domains
        allow_credentials=True,  # Required for JWT auth with cookies/headers
        allow_methods=["*"],  # Allows GET, POST, PUT, DELETE, PATCH, OPTIONS
        allow_headers=["*"],  # Allows Authorization, Content-Type, etc.
    )


def setup_gzip_middleware(app: FastAPI):
    """Setup GZip compression middleware for the FastAPI application.

    Enables automatic GZip compression for responses to reduce bandwidth usage
    and improve load times. Only compresses responses larger than the minimum
    size threshold.

    Args:
        app: The FastAPI application instance

    Configuration:
        - minimum_size: 1000 bytes - Only compress responses >= 1KB

    Benefits:
        - Reduces response size by 70-90% for JSON
        - Faster transfer times
        - Lower bandwidth costs
        - Transparent to clients (browsers auto-decompress)
    """
    # =======================================================================
    # LOG MIDDLEWARE REGISTRATION
    # =======================================================================
    print__memory_monitoring("📋 Registering GZip middleware...")

    # =======================================================================
    # ADD GZIP COMPRESSION MIDDLEWARE
    # =======================================================================

    # Add GZip compression to reduce response sizes and memory usage
    # Only compresses responses >= 1000 bytes (overhead not worth it for smaller)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

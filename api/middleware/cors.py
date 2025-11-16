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

2. Brotli Compression Middleware:
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

2. Brotli Compression
   - Automatic response compression
   - Configurable minimum size threshold (1000 bytes)
   - Transparent to application code
   - Modern browser-compatible compression

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
      allow_headers=["Authorization", "Content-Type", "Accept", "Accept-Encoding", "Accept-Language", "Cache-Control", "X-Requested-With"]

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
COMPRESSION MIDDLEWARE (BROTLI)
===================================================================================

Function: setup_brotli_middleware(app: FastAPI)
    Purpose: Enable automatic response compression with Brotli

    Configuration:
        minimum_size: 1000
            - Only compress responses >= 1000 bytes
            - Avoids overhead for small responses
            - Optimal balance for performance

    Compression Algorithm:
        - Uses Brotli compression
        - Modern browser-compatible (all current browsers)
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
    - Client supports brotli (Accept-Encoding: br)
    - Content-Type is compressible (application/json, text/html)

When Compression Skipped:
    - Response < 1000 bytes (overhead not worth it)
    - Client doesn't support brotli (older browsers fall back to uncompressed)
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
        print__memory_monitoring("📋 Registering Brotli middleware...")

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
        Registration: CORS → Brotli
        Execution: Brotli → CORS → Route → CORS → Brotli

    Why This Order:
        - CORS needs to process before route handlers
        - Brotli compresses final responses
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
        setup_brotli_middleware
    )

    app = FastAPI()

    # Setup middleware
    setup_cors_middleware(app)
    setup_brotli_middleware(app)

    # Register routes
    app.include_router(routes)

Startup Sequence:
    1. Create FastAPI application
    2. Setup CORS middleware
    3. Setup Brotli compression middleware
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

Compression Overhead:
    - Brotli: 10-30ms for large responses
    - Decompression: Handled by browser (transparent)
    - Net benefit: Faster transfer time outweighs compression time

Memory Usage:
    - CORS: Negligible
    - Compression: Temporary buffer during compression
    - Released immediately after compression

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. CORS Security
   - Environment-based origins (production-ready)
   - Specific allowed methods and headers
   - Prevents unauthorized cross-origin access
   - Credential support with controlled origins

2. Compression Security
   - BREACH attack: Compression + secrets = vulnerability
   - Mitigation: Don't compress responses with secrets
   - CSRF tokens should not be compressed
   - Generally safe for JSON APIs
   - Brotli has same security considerations as GZip

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

CORS Testing:
    - Test cross-origin requests from different origins
    - Verify preflight OPTIONS requests
    - Check CORS headers in responses
    - Test with credentials (cookies, auth headers)

Compression Testing:
    - Send Accept-Encoding: br header
    - Verify Content-Encoding: br in response
    - Check response size reduction (expect 70-90% for JSON)
    - Test with various payload sizes
    - Test fallback behavior for clients without Brotli support

Integration Tests:
    from fastapi.testclient import TestClient

    def test_cors_headers():
        response = client.options("/analyze")
        assert "access-control-allow-origin" in response.headers

    def test_brotli_compression():
        headers = {"Accept-Encoding": "br"}
        response = client.get("/bulk", headers=headers)
        assert response.headers.get("content-encoding") == "br"

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection

Third-Party:
    - fastapi: FastAPI, CORSMiddleware
    - brotli-asgi: BrotliMiddleware
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
from brotli_asgi import BrotliMiddleware

# Import memory monitoring functions from utils
from api.utils.memory import print__memory_monitoring

# ==============================================================================
# MIDDLEWARE SETUP - CORS AND BROTLI
# ==============================================================================


def setup_cors_middleware(app: FastAPI):
    """Setup CORS middleware for the FastAPI application.

    Configures Cross-Origin Resource Sharing (CORS) to allow requests from
    frontend applications running on different origins (domains/ports).

    Reads allowed origins from CORS_ALLOWED_ORIGINS environment variable.
    Format: Comma-separated list of URLs
    Example: https://example.com,https://app.example.com,http://localhost:3000

    Args:
        app: The FastAPI application instance

    Configuration:
        - allow_origins: From CORS_ALLOWED_ORIGINS env var (secure)
        - allow_credentials: True - Enables cookies and auth headers
        - allow_methods: Specific methods only - GET, POST, PUT, DELETE, OPTIONS
        - allow_headers: ["*"] - Allows all headers for maximum flexibility

    Security:
        Uses environment-based configuration for allowed origins.
        Defaults to localhost URLs if CORS_ALLOWED_ORIGINS not set.
    """
    # =======================================================================
    # LOG MIDDLEWARE REGISTRATION
    # =======================================================================

    # Monitor route registrations (including middleware and CORS)
    print__memory_monitoring("📋 Registering CORS middleware...")
    # Note: Route registration monitoring happens at runtime to avoid import-time global variable access

    # =======================================================================
    # GET ALLOWED ORIGINS FROM ENVIRONMENT
    # =======================================================================

    # Get allowed origins from environment variable
    # Format: Comma-separated list of URLs
    allowed_origins_str = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000",  # Default for development
    )

    # Split by comma and strip whitespace
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

    print__memory_monitoring(f"📋 CORS allowed origins: {allowed_origins}")

    # =======================================================================
    # ADD CORS MIDDLEWARE
    # =======================================================================

    # Add CORS middleware with environment-based configuration
    # Production-ready: Uses specific allowed origins from .env
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # From CORS_ALLOWED_ORIGINS env var
        allow_credentials=True,  # Required for JWT auth with cookies/headers
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specific methods
        allow_headers=["*"],  # Allow all headers for maximum flexibility
    )


def setup_brotli_middleware(app: FastAPI):
    """Setup Brotli compression middleware for the FastAPI application.

    Brotli provides 15-20% better compression than GZip and is supported by
    all modern browsers. For clients that don't support Brotli, responses
    will be sent uncompressed.

    Only compresses responses larger than the minimum size threshold.

    Args:
        app: The FastAPI application instance

    Configuration:
        - minimum_size: 1000 bytes - Only compress responses >= 1KB

    Benefits:
        - 15-20% better compression than GZip
        - Reduces response size by 70-90% for JSON
        - Faster transfer times
        - Lower bandwidth costs
        - Transparent to clients (browsers auto-decompress)
        - Supported by all modern browsers (Chrome, Firefox, Safari, Edge)

    Compression Behavior:
        1. Client sends Accept-Encoding: br header
        2. Server compresses response with Brotli
        3. No compression if client doesn't support Brotli
        4. No compression if response < 1000 bytes
    """
    # =======================================================================
    # LOG MIDDLEWARE REGISTRATION
    # =======================================================================
    print__memory_monitoring("📋 Registering Brotli compression middleware...")

    # =======================================================================
    # ADD COMPRESSION MIDDLEWARE
    # =======================================================================

    # Add Brotli compression (15-20% better than GZip)
    # Modern browsers support Brotli (Chrome, Firefox, Safari, Edge)
    # Only compresses responses >= 1000 bytes (overhead not worth it for smaller)
    app.add_middleware(BrotliMiddleware, minimum_size=1000)

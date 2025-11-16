"""
MODULE_DESCRIPTION: API Configuration Settings - Global State and Application Constants

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module serves as the central configuration hub for the CZSU Multi-Agent
Text-to-SQL API. It defines global constants, shared state variables, and
configuration parameters that are accessed throughout the application.

The module manages:
    - Application startup tracking (uptime, baseline metrics)
    - Global checkpointer for conversation state persistence
    - Concurrency controls (semaphores for rate limiting)
    - Rate limiting configuration and storage
    - Cache management for bulk operations
    - JWT authentication settings
    - Memory management thresholds
    - Windows compatibility settings

Design Principle:
    Centralized configuration allows consistent behavior across all modules
    and simplifies configuration changes without modifying business logic.

===================================================================================
KEY FEATURES
===================================================================================

1. Application Lifecycle Tracking
   - Startup time recording for uptime calculations
# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

# =======================================================================
# APPLICATION LIFECYCLE TRACKING
# =======================================================================

# Application startup time for uptime tracking
# =======================================================================
# MEMORY MANAGEMENT SETTINGS
# =======================================================================

# Read GC memory threshold from environment with default fallback
# MOVED TO memory.py: GC_MEMORY_THRESHOLD = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))  # 1900MB for 2GB memory allocation

# MEMORY LEAK PREVENTION: Simplified global tracking
# Reserved for future enhanced startup tracking
_APP_STARTUP_TIME = None

# Baseline memory at startup for leak detection
# Would store RSS memory value when application starts
_MEMORY_BASELINE = None  # RSS memory at startup

# Total requests processed since startup
# Incremented by memory monitoring middleware
# =======================================================================
# CONCURRENCY CONTROL
# =======================================================================

# Add a semaphore to limit concurrent analysis requests
# Prevents too many AI operations running simultaneously
# Higher value = faster but more memory usage
# Lower value = slower but safer memory profile
MAX_CONCURRENT_ANALYSES = int(
    os.environ.get("MAX_CONCURRENT_ANALYSES", "3")
)  # Read from .env with fallback to 3

# Semaphore enforcing MAX_CONCURRENT_ANALYSES limit
# Shared across all /analyze requests
# Blocks when limit reached, releases after completion
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# =======================================================================
# RATE LIMITING
# =======================================================================

# RATE LIMITING: Global rate limiting storage
# Stores request timestamps per client IP for sliding window algorithm
# Key: Client IP address, Value: List of Unix timestamps
rate_limit_storage = defaultdict(list)

# Maximum requests allowed per window (per IP)
RATE_LIMIT_REQUESTS = 100  # requests per window

# Time window in seconds for rate limiting
RATE_LIMIT_WINDOW = 60  # 60 seconds window

# Additional burst allowance for temporary spikes
RATE_LIMIT_BURST = 20  # burst limit for rapid requests

# Maximum seconds to wait before giving up on rate limit
RATE_LIMIT_MAX_WAIT = 5  # maximum seconds to wait before giving up

# =======================================================================
# THROTTLING
# =======================================================================

# Throttling semaphores per IP to limit concurrent requests
# Auto-creates semaphore for new IPs with consistent limit
# Prevents single user from monopolizing resources
throttle_semaphores = defaultdict(
    lambda: asyncio.Semaphore(8)
)  # Max 8 concurrent requests per IP

# =======================================================================
# BULK LOADING CACHE
# =======================================================================

# Global cache for bulk loading to prevent repeated calls
# Key: f"bulk_messages_{user_email}"
# Value: (result_dict, timestamp)
_bulk_loading_cache = {}

# Per-user locks for bulk loading operations
# Prevents duplicate bulk loads for same user
# Key: User email, Value: asyncio.Lock
_bulk_loading_locks = defaultdict(asyncio.Lock)

# Cache TTL in seconds
# Balance between freshness and performance
BULK_CACHE_TIMEOUT = 120  # Cache timeout in seconds

# =======================================================================
# JWT AUTHENTICATION
# =======================================================================

# Google's public key certificates URL for JWT verification
# Standard Google OAuth2 endpoint
GOOGLE_JWK_URL = "https://www.googleapis.com/oauth2/v3/certs"

# Global counter for tracking JWT 'kid' missing events to reduce log spam
# Incremented when JWT token lacks 'kid' field
# Used to throttle repeated error logging
_JWT_KID_MISSING_COUNT = 0ulk message retrieval
   - Per-user locking mechanism
   - Configurable TTL (Time To Live)
   - Prevents duplicate bulk loads

6. JWT Authentication Configuration
   - Google JWK URL for public key retrieval
   - Token verification settings
   - Missing 'kid' tracking to reduce log spam

7. Fallback Mechanisms
   - InMemorySaver fallback when PostgreSQL unavailable
   - Graceful degradation support
   - Environment-based feature toggles

===================================================================================
GLOBAL VARIABLES
===================================================================================

Application Lifecycle:
    start_time (float)
        - Unix timestamp when application started
        - Used for uptime calculations in /health endpoint
        - Initialized at module import time

    _APP_STARTUP_TIME (None)
        - Reserved for future enhanced startup tracking
        - Currently unused placeholder

    _MEMORY_BASELINE (None)
        - Reserved for baseline memory tracking
        - Would store RSS memory at startup for leak detection

    _REQUEST_COUNT (int)
        - Total number of requests processed
        - Incremented by memory monitoring middleware
        - Used for monitoring and diagnostics

Checkpointer:
    GLOBAL_CHECKPOINTER (None | AsyncPostgresSaver)
        - Shared checkpointer instance for all requests
        - Initialized during application startup
        - Provides conversation state persistence
        - None until initialized by startup event

    INMEMORY_FALLBACK_ENABLED (bool)
        - Whether to use InMemorySaver if PostgreSQL fails
        - Controlled by InMemorySaver_fallback env var
        - Default: True (enabled)
        - Production: Consider disabling for consistency

Concurrency Control:
    MAX_CONCURRENT_ANALYSES (int)
        - Maximum concurrent /analyze operations
        - Default: 3 (configurable via env)
        - Prevents memory exhaustion from AI operations
        - Higher = faster but more memory usage

    analysis_semaphore (asyncio.Semaphore)
        - Semaphore enforcing MAX_CONCURRENT_ANALYSES limit
        - Shared across all /analyze requests
        - Blocks when limit reached
        - Automatically releases after completion

    throttle_semaphores (defaultdict[str, asyncio.Semaphore])
        - Per-IP semaphores for throttling
        - Key: Client IP address
        - Value: Semaphore with limit 8 (max 8 concurrent per IP)
        - Prevents single user from monopolizing resources

Rate Limiting:
    rate_limit_storage (defaultdict[str, list])
        - Stores request timestamps per client IP
        - Key: Client IP address
        - Value: List of Unix timestamps (recent requests)
        - Cleaned up automatically by middleware

    RATE_LIMIT_REQUESTS (int)
        - Max requests allowed per window
        - Default: 100 requests
        - Applied per client IP

    RATE_LIMIT_WINDOW (int)
        - Time window in seconds
        - Default: 60 seconds
        - Sliding window algorithm

    RATE_LIMIT_BURST (int)
        - Additional burst allowance
        - Default: 20 requests
        - Allows temporary spikes

    RATE_LIMIT_MAX_WAIT (int)
        - Maximum seconds to wait before giving up
        - Default: 5 seconds
        - Prevents indefinite blocking

Bulk Loading Cache:
    _bulk_loading_cache (dict)
        - Caches bulk message loading results
        - Key: f"bulk_messages_{user_email}"
        - Value: (result_dict, timestamp)
        - TTL: BULK_CACHE_TIMEOUT seconds

    _bulk_loading_locks (defaultdict[str, asyncio.Lock])
        - Per-user locks for bulk loading
        - Key: User email
        - Value: asyncio.Lock
        - Prevents duplicate bulk loads for same user

    BULK_CACHE_TIMEOUT (int)
        - Cache TTL in seconds
        - Default: 120 seconds (2 minutes)
        - Balance between freshness and performance

JWT Authentication:
    GOOGLE_JWK_URL (str)
        - URL for Google's public key certificates
        - Used to verify Google-issued JWT tokens
        - Standard Google OAuth2 endpoint

    _JWT_KID_MISSING_COUNT (int)
        - Counter for JWT 'kid' missing events
        - Reduces log spam from repeated errors
        - Reset periodically or on successful verification

===================================================================================
CONFIGURATION PATTERNS
===================================================================================

Environment Variable Loading:
    - Load .env file early (before other imports)
    - Use os.environ.get() with defaults
    - Type casting for numeric values
    - Boolean parsing ("1" = True, "0" = False)

Example:
    MAX_CONCURRENT_ANALYSES = int(
        os.environ.get("MAX_CONCURRENT_ANALYSES", "3")
    )

Semaphore Creation:
    analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
    - Created at module import time
    - Value from environment variable
    - Shared across all requests

DefaultDict Pattern:
    throttle_semaphores = defaultdict(
        lambda: asyncio.Semaphore(8)
    )
    - Auto-creates semaphore for new IPs
    - Consistent limit (8) for all IPs
    - No manual initialization needed

Cache Key Format:
    cache_key = f"bulk_messages_{user_email}"
    - Consistent naming convention
    - Scoped by user for isolation
    - Prevents collisions

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

Symptoms Without:
    - RuntimeError: Event loop closed
    - Connection deadlocks
    - Unpredictable async behavior

===================================================================================
MEMORY MANAGEMENT
===================================================================================

Commented-Out Variables:
    - GC_MEMORY_THRESHOLD: Moved to memory.py
    - MEMORY_PROFILER_ENABLED: Moved to memory.py
    - MEMORY_PROFILER_INTERVAL: Moved to memory.py
    - MEMORY_PROFILER_TOP_STATS: Moved to memory.py

Reason for Move:
    - Better organization (all memory logic in one place)
    - Reduces circular import risks
    - Cleaner separation of concerns

Remaining Memory Tracking:
    - _MEMORY_BASELINE: Reserved for future use
    - _REQUEST_COUNT: Tracks total requests
    - _APP_STARTUP_TIME: Reserved for enhanced tracking

===================================================================================
CONCURRENCY LIMITS
===================================================================================

Analysis Semaphore:
    Purpose: Prevent too many concurrent AI operations
    Default: 3 concurrent analyses
    Tuning:
        - Higher (5-10): Faster but more memory
        - Lower (1-2): Slower but safer
        - Monitor memory usage to optimize

Throttle Semaphore:
    Purpose: Limit concurrent requests per IP
    Default: 8 concurrent requests per IP
    Benefits:
        - Prevents single user from monopolizing
        - Fair resource allocation
        - DDoS protection

===================================================================================
RATE LIMITING STRATEGY
===================================================================================

Algorithm: Sliding Window
    - Track timestamps of recent requests
    - Count requests within RATE_LIMIT_WINDOW
    - Allow if count < RATE_LIMIT_REQUESTS + RATE_LIMIT_BURST
    - Wait or reject based on RATE_LIMIT_MAX_WAIT

Storage Cleanup:
    - Old timestamps removed automatically
    - Timestamps older than window discarded
    - Prevents memory growth

Burst Handling:
    - RATE_LIMIT_BURST allows temporary spikes
    - Example: 100 req/60s + 20 burst = up to 120 in short period
    - Smooths legitimate traffic spikes

===================================================================================
CACHE MANAGEMENT
===================================================================================

Bulk Loading Cache Strategy:
    1. Check cache: If hit and not expired, return cached data
    2. Acquire lock: Prevent duplicate processing
    3. Double-check cache: Another request may have completed
    4. Process: Load data if still not cached
    5. Cache result: Store with timestamp
    6. Release lock: Allow subsequent requests

Lock Pattern (Double-Check):
    if cache_key in cache:
        return cache[cache_key]  # Fast path

    async with lock:
        if cache_key in cache:
            return cache[cache_key]  # Another request completed

        result = await expensive_operation()
        cache[cache_key] = (result, time.time())
        return result

TTL Management:
    - BULK_CACHE_TIMEOUT defines max age
    - Cleanup on access (lazy cleanup)
    - Periodic cleanup via /health/memory endpoint

===================================================================================
SECURITY CONSIDERATIONS
===================================================================================

1. Rate Limiting
   - Protects against abuse and DDoS
   - Per-IP tracking for fairness
   - Configurable thresholds

2. Concurrency Limits
   - Prevents resource exhaustion attacks
   - Memory protection
   - CPU protection

3. Cache Isolation
   - Per-user caches (no cross-user leakage)
   - User email scoping
   - Separate locks per user

4. JWT Configuration
   - Google's official JWK URL
   - Public key verification
   - Standard OAuth2 flow

===================================================================================
PERFORMANCE TUNING
===================================================================================

Analysis Concurrency:
    - Increase MAX_CONCURRENT_ANALYSES for more throughput
    - Monitor memory usage (typically 200-500MB per analysis)
    - Consider server specs (2GB RAM → max 3-4 concurrent)

Rate Limits:
    - Increase RATE_LIMIT_REQUESTS for higher-volume clients
    - Decrease for tighter control
    - Adjust RATE_LIMIT_WINDOW for different time scales

Cache Timeout:
    - Increase BULK_CACHE_TIMEOUT for more cache hits
    - Decrease for fresher data
    - Balance between performance and freshness

===================================================================================
MONITORING AND OBSERVABILITY
===================================================================================

Metrics to Track:
    - _REQUEST_COUNT: Total requests
    - start_time: Application uptime
    - len(rate_limit_storage): Active clients
    - len(_bulk_loading_cache): Cached users
    - analysis_semaphore._value: Available analysis slots

Health Checks:
    - Monitor semaphore availability
    - Check cache size
    - Track request counts
    - Verify checkpointer initialization

Alerting:
    - Cache growing unbounded → Memory leak
    - Semaphore always full → Increase limit
    - High rate limit rejections → Adjust limits

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Test environment variable parsing
    - Test default values
    - Test semaphore creation
    - Test cache key format

Integration Tests:
    - Test concurrent analysis limiting
    - Test rate limiting behavior
    - Test cache TTL expiration
    - Test per-IP throttling

Mock Testing:
    - Mock environment variables
    - Mock asyncio.Semaphore
    - Mock time.time() for cache tests

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - os: Environment variables
    - sys: Platform detection
    - time: Startup time, cache timestamps
    - collections.defaultdict: Auto-initializing dicts
    - asyncio: Semaphores, locks, event loop policy

Third-Party:
    - dotenv: Environment variable loading

===================================================================================
MIGRATION NOTES
===================================================================================

Memory Settings Moved:
    Previously in this file, now in api/utils/memory.py:
    - GC_MEMORY_THRESHOLD
    - MEMORY_PROFILER_ENABLED
    - MEMORY_PROFILER_INTERVAL
    - MEMORY_PROFILER_TOP_STATS

Impact:
    - Import from api.utils.memory if needed
    - No functional changes
    - Better code organization

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Distributed Caching
   - Move _bulk_loading_cache to Redis
   - Share across multiple API instances
   - Horizontal scaling support

2. Dynamic Configuration
   - Runtime configuration updates
   - No restart required
   - Admin API for tuning

3. Advanced Rate Limiting
   - Per-user rate limits (not just IP)
   - Tiered limits (free vs premium)
   - Geographic-based limits

4. Enhanced Monitoring
   - Prometheus metrics export
   - Real-time dashboards
   - Automatic anomaly detection

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

import asyncio

# Standard imports
import time
from collections import defaultdict

# ============================================================
# CONFIGURATION AND CONSTANTS
# ============================================================

# Application startup time for uptime tracking
start_time = time.time()

# Read InMemorySaver fallback configuration from environment
INMEMORY_FALLBACK_ENABLED = os.environ.get("InMemorySaver_fallback", "1") == "1"

# Read GC memory threshold from environment with default fallback
# MOVED TO memory.py: GC_MEMORY_THRESHOLD = int(os.environ.get("GC_MEMORY_THRESHOLD", "1900"))  # 1900MB for 2GB memory allocation

# MEMORY LEAK PREVENTION: Simplified global tracking
_APP_STARTUP_TIME = None
_MEMORY_BASELINE = None  # RSS memory at startup
_REQUEST_COUNT = 0  # Track total requests processed

# Tracemalloc-based memory profiler configuration
# MOVED TO memory.py: MEMORY_PROFILER_ENABLED = os.environ.get("MEMORY_PROFILER_ENABLED", "0") == "1"
# MOVED TO memory.py: MEMORY_PROFILER_INTERVAL = int(os.environ.get("MEMORY_PROFILER_INTERVAL", "30"))
# MOVED TO memory.py: MEMORY_PROFILER_TOP_STATS = int(os.environ.get("MEMORY_PROFILER_TOP_STATS", "10"))

# Global shared checkpointer for conversation memory across API requests
# This ensures that conversation state is preserved between frontend requests using PostgreSQL
GLOBAL_CHECKPOINTER = None

# Add a semaphore to limit concurrent analysis requests
MAX_CONCURRENT_ANALYSES = int(
    os.environ.get("MAX_CONCURRENT_ANALYSES", "3")
)  # Read from .env with fallback to 3
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

# RATE LIMITING: Global rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # 60 seconds window
RATE_LIMIT_BURST = 20  # burst limit for rapid requests
RATE_LIMIT_MAX_WAIT = 5  # maximum seconds to wait before giving up

# Throttling semaphores per IP to limit concurrent requests
throttle_semaphores = defaultdict(
    lambda: asyncio.Semaphore(8)
)  # Max 8 concurrent requests per IP

# Global cache for bulk loading to prevent repeated calls
_bulk_loading_cache = {}
_bulk_loading_locks = defaultdict(asyncio.Lock)
BULK_CACHE_TIMEOUT = 120  # Cache timeout in seconds

GOOGLE_JWK_URL = "https://www.googleapis.com/oauth2/v3/certs"

# Global counter for tracking JWT 'kid' missing events to reduce log spam
_JWT_KID_MISSING_COUNT = 0

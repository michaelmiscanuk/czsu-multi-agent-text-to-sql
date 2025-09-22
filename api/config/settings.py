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
GC_MEMORY_THRESHOLD = int(
    os.environ.get("GC_MEMORY_THRESHOLD", "1900")
)  # 1900MB for 2GB memory allocation

# MEMORY LEAK PREVENTION: Simplified global tracking
_app_startup_time = None
_memory_baseline = None  # RSS memory at startup
_request_count = 0  # Track total requests processed

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
_jwt_kid_missing_count = 0

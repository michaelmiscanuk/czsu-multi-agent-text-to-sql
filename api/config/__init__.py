"""
Configuration package for the API server.

This package contains settings, constants, and configuration management
for the CZSU Multi-Agent Text-to-SQL application.
"""

# Import key configuration items for easier access
from .settings import (  # Environment variables; Application constants; Checkpointer; Concurrency settings; Rate limiting; Bulk loading cache; JWT settings; Debug functions
    BASE_DIR,
    BULK_CACHE_TIMEOUT,
    GLOBAL_CHECKPOINTER,
    GOOGLE_JWK_URL,
    INMEMORY_FALLBACK_ENABLED,
    MAX_CONCURRENT_ANALYSES,
    RATE_LIMIT_BURST,
    RATE_LIMIT_MAX_WAIT,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    _APP_STARTUP_TIME,
    _bulk_loading_cache,
    _bulk_loading_locks,
    _JWT_KID_MISSING_COUNT,
    _MEMORY_BASELINE,
    _REQUEST_COUNT,
    analysis_semaphore,
    rate_limit_storage,
    start_time,
    throttle_semaphores,
)

__all__ = [
    "INMEMORY_FALLBACK_ENABLED",
    "BASE_DIR",
    "start_time",
    # "GC_MEMORY_THRESHOLD",  # MOVED TO memory.py
    "_APP_STARTUP_TIME",
    "_MEMORY_BASELINE",
    "_REQUEST_COUNT",
    "GLOBAL_CHECKPOINTER",
    "MAX_CONCURRENT_ANALYSES",
    "analysis_semaphore",
    "rate_limit_storage",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW",
    "RATE_LIMIT_BURST",
    "RATE_LIMIT_MAX_WAIT",
    "throttle_semaphores",
    "_bulk_loading_cache",
    "_bulk_loading_locks",
    "BULK_CACHE_TIMEOUT",
    "GOOGLE_JWK_URL",
    "_JWT_KID_MISSING_COUNT",
]

"""
Configuration package for the API server.

This package contains settings, constants, and configuration management
for the CZSU Multi-Agent Text-to-SQL application.
"""

# Import key configuration items for easier access
from .settings import (
    # Environment variables
    INMEMORY_FALLBACK_ENABLED,
    BASE_DIR,
    
    # Application constants
    start_time,
    GC_MEMORY_THRESHOLD,
    _app_startup_time,
    _memory_baseline,
    _request_count,
    
    # Checkpointer
    GLOBAL_CHECKPOINTER,
    
    # Concurrency settings
    MAX_CONCURRENT_ANALYSES,
    analysis_semaphore,
    
    # Rate limiting
    rate_limit_storage,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    RATE_LIMIT_BURST,
    RATE_LIMIT_MAX_WAIT,
    throttle_semaphores,
    
    # Bulk loading cache
    _bulk_loading_cache,
    _bulk_loading_locks,
    BULK_CACHE_TIMEOUT,
    
    # JWT settings
    GOOGLE_JWK_URL,
    _jwt_kid_missing_count,
    
    # Debug functions
    print__startup_debug,
    print__memory_monitoring
)

__all__ = [
    'INMEMORY_FALLBACK_ENABLED',
    'BASE_DIR',
    'start_time',
    'GC_MEMORY_THRESHOLD',
    '_app_startup_time',
    '_memory_baseline',
    '_request_count',
    'GLOBAL_CHECKPOINTER',
    'MAX_CONCURRENT_ANALYSES',
    'analysis_semaphore',
    'rate_limit_storage',
    'RATE_LIMIT_REQUESTS',
    'RATE_LIMIT_WINDOW',
    'RATE_LIMIT_BURST',
    'RATE_LIMIT_MAX_WAIT',
    'throttle_semaphores',
    '_bulk_loading_cache',
    '_bulk_loading_locks',
    'BULK_CACHE_TIMEOUT',
    'GOOGLE_JWK_URL',
    '_jwt_kid_missing_count',
    'print__startup_debug',
    'print__memory_monitoring'
] 
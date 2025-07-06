"""
Utility functions package for the API server.

This package contains debug utilities, memory management, rate limiting,
and other utility functions for the CZSU Multi-Agent Text-to-SQL application.
"""

# Debug utilities
from .debug import (
    print__api_postgresql,
    print__feedback_flow,
    print__token_debug,
    print__sentiment_flow,
    print__debug,
    print__analyze_debug,
    print__chat_all_messages_debug,
    print__feedback_debug,
    print__sentiment_debug,
    print__chat_threads_debug,
    print__chat_messages_debug,
    print__delete_chat_debug,
    print__chat_sentiments_debug,
    print__catalog_debug,
    print__data_tables_debug,
    print__data_table_debug,
    print__chat_thread_id_checkpoints_debug,
    print__debug_pool_status_debug,
    print__chat_thread_id_run_ids_debug,
    print__debug_run_id_debug,
    print__admin_clear_cache_debug,
    print__analysis_tracing_debug
)

# Memory management utilities
from .memory import (
    print__memory_monitoring,
    cleanup_bulk_cache,
    check_memory_and_gc,
    log_memory_usage,
    log_comprehensive_error,
    setup_graceful_shutdown,
    perform_deletion_operations
)

# Rate limiting utilities
from .rate_limiting import (
    check_rate_limit_with_throttling,
    wait_for_rate_limit,
    check_rate_limit
)

# Export all utilities for easy access
__all__ = [
    # Debug utilities
    'print__api_postgresql',
    'print__feedback_flow',
    'print__token_debug',
    'print__sentiment_flow',
    'print__debug',
    'print__analyze_debug',
    'print__chat_all_messages_debug',
    'print__feedback_debug',
    'print__sentiment_debug',
    'print__chat_threads_debug',
    'print__chat_messages_debug',
    'print__delete_chat_debug',
    'print__chat_sentiments_debug',
    'print__catalog_debug',
    'print__data_tables_debug',
    'print__data_table_debug',
    'print__chat_thread_id_checkpoints_debug',
    'print__debug_pool_status_debug',
    'print__chat_thread_id_run_ids_debug',
    'print__debug_run_id_debug',
    'print__admin_clear_cache_debug',
    'print__analysis_tracing_debug',
    
    # Memory management utilities
    'print__memory_monitoring',
    'cleanup_bulk_cache',
    'check_memory_and_gc',
    'log_memory_usage',
    'log_comprehensive_error',
    'setup_graceful_shutdown',
    'perform_deletion_operations',
    
    # Rate limiting utilities
    'check_rate_limit_with_throttling',
    'wait_for_rate_limit',
    'check_rate_limit'
] 
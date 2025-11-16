import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "Tests - CZSU Multi-Agent Backend": {
        "Summary": "Comprehensive test suite covering API routes, database operations, and utility functions with automated validation and reporting.",
        "1. Test Infrastructure (tests/helpers.py)": {
            "BaseTestResults Class": {
                "Purpose": "Track and analyze endpoint test results with structured data collection",
                "Details": [
                    "add_result() - Record successful test with response data, status, timing",
                    "add_error() - Record test failures with traceback and error details",
                    "get_summary() - Generate comprehensive test metrics (success rate, timing, coverage)",
                    "Required endpoints tracking - Verify all endpoints tested",
                    "Response time aggregation - avg/max/min",
                    "Timestamp tracking for all operations",
                ],
                "Summary": "Core test results aggregation and analysis framework.",
            },
            "Helper Functions": {
                "Request Handling": {
                    "Details": [
                        "make_request_with_traceback_capture() - HTTP request with server-side error capture",
                        "handle_error_response() - Process unexpected error responses",
                        "handle_expected_failure() - Validate expected failure cases (422, 401, etc.)",
                        "extract_detailed_error_info() - Parse server tracebacks and error details",
                    ],
                    "Summary": "HTTP request utilities with comprehensive error tracking.",
                },
                "Authentication": {
                    "Details": [
                        "create_test_jwt_token() - Generate test JWT for Google OAuth simulation",
                        "Supports NextAuth.js id_token format",
                        "Configurable email and expiration",
                    ],
                    "Summary": "Test authentication token generation.",
                },
                "Server Control": {
                    "Details": [
                        "check_server_connectivity() - Verify server is running before tests",
                        "setup_debug_environment() - Configure debug flags via API",
                        "cleanup_debug_environment() - Reset debug state after tests",
                    ],
                    "Summary": "Server health and environment management.",
                },
                "Reporting": {
                    "Details": [
                        "save_traceback_report() - Write detailed error reports to tests/traceback_errors/",
                        "Cleaned vs full traceback modes (CLEANED_TRACEBACK env var)",
                        "Empty file if no errors, detailed report otherwise",
                        "Server-side and client-side error sections",
                        "Test summary with metrics",
                    ],
                    "Summary": "Automated error reporting and traceback analysis.",
                },
                "Summary": "Reusable test utilities for validation, auth, and reporting.",
            },
            "Summary": "Shared testing infrastructure providing result tracking, error capture, and reporting utilities.",
        },
        "2. API Route Tests (tests/api/test_phase*.py)": {
            "Common Structure (All test_phase* Files)": {
                "Test Organization": {
                    "Details": [
                        "Windows event loop policy set FIRST (asyncio.WindowsSelectorEventLoopPolicy)",
                        "Base directory resolution and sys.path setup",
                        "Import from tests.helpers (BaseTestResults, create_test_jwt_token, etc.)",
                        "REQUIRED_ENDPOINTS set definition",
                        "TEST_CASES or similar data structures",
                        "Async test functions with descriptive names",
                        "run_phase*_tests() orchestrator function",
                        "analyze_results() for test summary and reporting",
                        "main() async entry point with exit codes",
                    ],
                    "Summary": "Consistent test file structure across all phase tests.",
                },
                "Test Execution Pattern": {
                    "Details": [
                        "Setup: Load environment, create HTTP client, check server",
                        "Debug environment configuration (optional)",
                        "Execute individual test cases in sequence or parallel",
                        "Collect results in BaseTestResults instance",
                        "Cleanup: Reset debug environment, close connections",
                        "Analyze and report results",
                        "Save traceback report",
                        "Return pass/fail status",
                    ],
                    "Summary": "Standardized test lifecycle management.",
                },
                "Validation Pattern": {
                    "Details": [
                        "Response structure validation (assert required fields)",
                        "Type validation (isinstance checks)",
                        "Value range validation (min/max bounds)",
                        "UUID format validation for run_id/thread_id",
                        "Status code expectations (200, 400, 422, 401, etc.)",
                        "Expected failure handling (validation errors, auth failures)",
                    ],
                    "Summary": "Comprehensive response validation framework.",
                },
                "Summary": "Unified testing patterns ensuring consistency and maintainability.",
            },
            "Phase 1: Setup & Preparation (test_phase1_setup.py)": {
                "Purpose": "Validate project structure, imports, and all API endpoints availability",
                "Local Validation Tests": {
                    "Details": [
                        "Folder structure check - api/, api/config/, api/utils/, etc.",
                        "__init__.py files presence in all packages",
                        "Package import validation (api, api.config, api.utils, etc.)",
                        "No HTTP calls - pure filesystem and import checks",
                    ],
                    "Summary": "Verify project folder structure and Python package setup.",
                },
                "HTTP Endpoint Tests": {
                    "Coverage": [
                        "/health - General health with memory, database, uptime",
                        "/health/database - Database connection status",
                        "/health/memory - Memory usage metrics",
                        "/health/rate-limits - Rate limiting configuration",
                        "/health/prepared-statements - PostgreSQL prepared statement health",
                        "/catalog - CZSU data catalog browsing",
                        "/data-tables - List available data tables",
                        "/data-table - Query data table",
                        "/analyze - Natural language to SQL conversion",
                        "/chat/* - Thread messages, run IDs, sentiments, delete",
                        "/feedback, /sentiment - User feedback submission",
                        "/debug/* - Debug endpoints (checkpoints, pool-status, run-id)",
                        "/admin/* - Admin operations (clear cache, clear prepared statements)",
                        "/placeholder/{width}/{height} - Placeholder SVG generation",
                    ],
                    "Validation": [
                        "Response structure validation per endpoint",
                        "Expected status codes (200, 400, 422, etc.)",
                        "JSON vs non-JSON response handling",
                        "Path parameter substitution (thread_id, run_id, etc.)",
                    ],
                    "Summary": "Comprehensive endpoint availability and basic validation.",
                },
                "Test Cases": {
                    "Total Endpoints": "26+ endpoints tested",
                    "Validation Functions": "Specific validators per endpoint type",
                    "Summary": "All-inclusive API surface area validation.",
                },
                "Summary": "Foundation test ensuring project structure and all endpoints are accessible.",
            },
            "Phase 2: Configuration & Settings (test_phase2_config.py)": {
                "Purpose": "Validate configuration module and settings behavior",
                "HTTP Endpoint Tests": {
                    "Details": [
                        "test_health_endpoint() - Health response structure validation",
                        "test_memory_health_endpoint() - Memory metrics validation",
                        "test_rate_limit_health_endpoint() - Rate limit config validation",
                        "test_debug_env_set_and_reset() - Debug environment control",
                    ],
                    "Summary": "Health and debug endpoint configuration validation.",
                },
                "Internal Configuration Tests": {
                    "test_configuration_module_values": [
                        "Verify start_time is numeric",
                        "MAX_CONCURRENT_ANALYSES within expected range (1-32)",
                        "analysis_semaphore is asyncio.Semaphore",
                        "RATE_LIMIT_WINDOW = 60 seconds",
                        "RATE_LIMIT_REQUESTS >= 10",
                        "rate_limit_storage and _bulk_loading_cache existence",
                    ],
                    "test_semaphore_concurrency_simulation": [
                        "Acquire all semaphore permits",
                        "Verify semaphore decrements correctly",
                        "Verify semaphore restores after release",
                        "Simulates concurrent analysis requests",
                    ],
                    "test_gc_threshold": [
                        "GC_MEMORY_THRESHOLD validation (> 100MB, < 100GB)",
                        "Reasonable memory limit check",
                    ],
                    "test_rate_limit_mutation": [
                        "Rate limit storage mutation test",
                        "Verify storage records new entries",
                    ],
                    "test_bulk_cache_ttl": [
                        "Bulk cache expiration mechanism",
                        "cleanup_bulk_cache() removes expired entries",
                    ],
                    "test_throttle_semaphore": [
                        "Per-IP throttle semaphore acquire/release",
                        "Verify semaphore state changes",
                    ],
                    "test_api_import_warning": [
                        "Capture stdout during API import",
                        "Verify no import warnings",
                    ],
                    "Summary": "Deep internal configuration and behavior validation.",
                },
                "Test Cases": {
                    "HTTP Tests": "4 health/debug endpoints",
                    "Internal Tests": "7 configuration behavior tests",
                    "Summary": "11 total tests covering configuration and settings.",
                },
                "Summary": "Validate application configuration, limits, and internal state management.",
            },
            "Phase 3: Utilities (test_phase3_utilities.py)": {
                "Purpose": "Test utility functions for debug, memory, and rate limiting",
                "Debug Utilities Tests": {
                    "Functions Tested": [
                        "print__debug - Basic debug logging with env var control",
                        "print__api_postgresql - PostgreSQL debug logging",
                        "print__feedback_flow - Feedback flow debug logging",
                        "print__memory_monitoring - Memory monitoring debug",
                        "print__token_debug - Token debug logging",
                        "print__sentiment_flow - Sentiment flow debug",
                        "print__analyze_debug - Analyze endpoint debug",
                        "print__chat_all_messages_debug - Chat messages debug",
                    ],
                    "Validation": [
                        "Environment variable activation (e.g., print__debug=1)",
                        "Output capture and verification",
                        "stdout.flush() confirmation",
                    ],
                    "Summary": "Environment-controlled debug print functions validation.",
                },
                "Memory Utilities Tests": {
                    "Functions Tested": [
                        "check_memory_and_gc() - Memory check with GC trigger",
                        "cleanup_bulk_cache() - Cache cleanup returning count",
                        "log_memory_usage() - Memory logging with context",
                        "setup_graceful_shutdown() - Signal handler setup",
                        "log_comprehensive_error() - Error logging with request context",
                    ],
                    "Validation": [
                        "Return type validation (float, int, None)",
                        "Value range checks (memory in MB)",
                        "Mock request handling",
                    ],
                    "Summary": "Memory management and monitoring utilities validation.",
                },
                "Rate Limiting Utilities Tests": {
                    "Functions Tested": [
                        "check_rate_limit_with_throttling() - Detailed rate limit info",
                        "check_rate_limit() - Simple boolean check",
                        "wait_for_rate_limit() - Async wait with retry",
                    ],
                    "Validation": [
                        "Response structure (allowed, suggested_wait, burst_count, etc.)",
                        "Burst behavior testing",
                        "Async timeout scenarios",
                        "Per-IP rate limit isolation",
                    ],
                    "Summary": "Rate limiting logic and waiting mechanisms validation.",
                },
                "Edge Cases & Performance": {
                    "Edge Cases": [
                        "Empty messages in debug functions",
                        "Very long messages",
                        "Null/invalid inputs",
                    ],
                    "Performance": [
                        "Debug output performance",
                        "Memory function performance",
                        "Rate limiting response time",
                    ],
                    "Summary": "Boundary condition and performance testing.",
                },
                "Test Cases": {
                    "Debug Tests": "8 debug print functions",
                    "Memory Tests": "7 memory management tests",
                    "Rate Limiting Tests": "6 rate limiting tests",
                    "Summary": "21+ utility function tests with edge cases.",
                },
                "Summary": "Comprehensive validation of debug, memory, and rate limiting utilities.",
            },
            "Phase 4: Models (test_phase4_models.py)": {
                "Purpose": "Validate Pydantic request/response models",
                "Details": [
                    "Request models: AnalyzeRequest, FeedbackRequest, SentimentRequest",
                    "Response models: ChatThreadResponse, PaginatedChatThreadsResponse, ChatMessage",
                    "Field validation (UUID format, min/max length, type checking)",
                    "Optional vs required field handling",
                ],
                "Summary": "Pydantic model structure and validation testing.",
            },
            "Phase 5: Auth (test_phase5_auth.py)": {
                "Purpose": "Validate authentication and JWT token verification",
                "Details": [
                    "verify_google_jwt() function testing",
                    "Google OAuth token validation",
                    "NextAuth.js id_token support (missing 'kid')",
                    "JWT format validation (3 parts, base64)",
                    "Audience (GOOGLE_CLIENT_ID) verification",
                    "Clock skew handling (5-minute leeway)",
                    "Test token support (USE_TEST_TOKENS=1)",
                    "get_current_user() dependency testing",
                ],
                "Summary": "JWT authentication and Google OAuth integration validation.",
            },
            "Phase 6: Exceptions (test_phase6_exceptions.py)": {
                "Purpose": "Validate global exception handlers",
                "Details": [
                    "RequestValidationError handler (422 responses)",
                    "StarletteHTTPException handler (enhanced 401 debugging)",
                    "ValueError handler (400 responses)",
                    "General Exception handler (500 with optional traceback)",
                    "JSON serialization error fallback",
                ],
                "Summary": "Exception handling and error response validation.",
            },
            "Phase 7: Middleware (test_phase7_middleware.py)": {
                "Purpose": "Validate middleware stack behavior",
                "Middleware Tested": {
                    "CORS": [
                        "allow_origins=['*'] validation",
                        "allow_credentials=True verification",
                        "allow_methods and allow_headers",
                    ],
                    "Brotli": [
                        "Compression for responses > 1000 bytes",
                        "Content-Encoding header verification",
                    ],
                    "Rate Limiting": [
                        "Per-IP rate limiting",
                        "Throttling with wait (not immediate 429)",
                        "Exclusion of /health, /docs, /debug/pool-status",
                        "429 response with retry_after header",
                    ],
                    "Memory Monitoring": [
                        "Selective monitoring (/analyze, /chat/all-messages-*)",
                        "log_memory_usage() before and after",
                        "Request count increment",
                    ],
                    "Summary": "All middleware layers validated through actual requests.",
                },
                "Summary": "Middleware functionality and execution order validation.",
            },
            "Phase 8: Route Groups (test_phase8_*.py)": {
                "Phase 8.1: Analysis Routes (test_phase8_analysis.py)": {
                    "Purpose": "Test /analyze endpoint for NL-to-SQL conversion",
                    "Test Cases": {
                        "Valid Queries": [
                            "Basic population query",
                            "Table structure analysis",
                            "Data query analysis",
                            "Dataset explanation",
                            "SQL generation",
                        ],
                        "Invalid Requests": [
                            "Empty prompt (should fail 422)",
                            "Empty thread_id (should fail 422)",
                            "Missing prompt field",
                            "Missing thread_id field",
                            "Null prompt value",
                            "Null thread_id value",
                        ],
                        "Edge Cases": [
                            "Very long prompt (9999 chars)",
                            "Complex philosophical query",
                        ],
                    },
                    "Validation": [
                        "_validate_response_structure() checks required fields",
                        "prompt, result, thread_id, run_id presence and types",
                        "UUID format validation for run_id",
                        "Optional fields: queries_and_results, datasets_used, sql, etc.",
                    ],
                    "Summary": "Natural language analysis endpoint with LangGraph integration testing.",
                },
                "Phase 8.2: Health Routes (test_phase8_health.py)": {
                    "Endpoints Tested": [
                        "/health - Overall health with memory, database, uptime",
                        "/health/database - checkpointer_available, checkpointer_type",
                        "/health/memory - memory_rss_mb, threshold, usage_percent",
                        "/health/rate-limits - window, requests, tracked_clients",
                        "/health/prepared-statements - prepared statement count, status",
                    ],
                    "Validation": [
                        "Response structure per endpoint",
                        "Status values (healthy, degraded, unhealthy)",
                        "Numeric field validation (memory, thresholds)",
                    ],
                    "Summary": "Health monitoring endpoints validation.",
                },
                "Phase 8.3: Catalog Routes (test_phase8_catalog.py)": {
                    "Endpoints Tested": [
                        "/catalog - Browse CZSU data catalog with pagination",
                        "/data-tables - List available data tables",
                        "/data-table - Query specific table data",
                    ],
                    "Validation": [
                        "Pagination parameters (page, page_size)",
                        "Search query (q parameter)",
                        "results, total, page, page_size fields",
                        "selection_code and description validation",
                    ],
                    "Summary": "CZSU data catalog browsing and querying validation.",
                },
                "Phase 8.4: Feedback Routes (test_phase8_feedback.py)": {
                    "Endpoints Tested": [
                        "/feedback - Submit user feedback (run_id, feedback 0/1, comment)",
                        "/sentiment - Update sentiment (run_id, sentiment bool or null)",
                    ],
                    "Test Setup": [
                        "create_test_run_ids() - Create owned run_ids in database",
                        "cleanup_test_run_ids() - Remove test data after tests",
                    ],
                    "Test Cases": {
                        "Valid Feedback": [
                            "Positive feedback (1) with comment",
                            "Negative feedback (0) with comment",
                            "Feedback without comment",
                        ],
                        "Invalid Feedback": [
                            "Missing run_id (422)",
                            "Invalid run_id format (422)",
                            "Non-existent run_id (404)",
                            "Missing feedback field (422)",
                        ],
                        "Sentiment Tests": [
                            "Positive sentiment (1)",
                            "Negative sentiment (0)",
                            "Neutral sentiment (null)",
                            "Missing/invalid run_id",
                        ],
                        "Auth Tests": [
                            "Missing Authorization header (401)",
                            "Invalid token (401)",
                            "Malformed token (401)",
                        ],
                    },
                    "Validation": [
                        "LangSmith feedback submission",
                        "Database sentiment update",
                        "run_id ownership verification",
                    ],
                    "Summary": "User feedback and sentiment tracking validation.",
                },
                "Phase 8.5: Chat Routes (test_phase8_chat.py)": {
                    "Endpoints Tested": [
                        "GET /chat/{thread_id}/sentiments - Get thread sentiments",
                        "GET /chat-threads - List user threads with pagination",
                        "GET /chat/all-messages-for-one-thread/{thread_id} - Get all messages",
                        "DELETE /chat/{thread_id} - Delete thread checkpoints",
                    ],
                    "Validation": [
                        "Thread ownership verification",
                        "Pagination (page, limit parameters)",
                        "Message structure validation",
                        "Sentiment data format",
                    ],
                    "Summary": "Multi-threaded conversation management validation.",
                },
                "Phase 8.6: Messages Routes (test_phase8_messages.py)": {
                    "Endpoints Tested": [
                        "GET /chat/{thread_id}/messages - Get thread messages",
                        "GET /chat/{thread_id}/run-ids - Get run IDs for thread",
                    ],
                    "Test Cases": {
                        "Authentication": [
                            "Valid token - successful access",
                            "Missing token - 401 error",
                            "Invalid token - 401 error",
                        ],
                        "Thread IDs": [
                            "Valid thread_id",
                            "Malformed thread_id",
                            "Non-existent thread_id",
                            "Edge case thread IDs (very long, special chars)",
                        ],
                        "Performance": [
                            "Response time < 5 seconds",
                            "Concurrent request handling",
                        ],
                    },
                    "Validation": [
                        "Message structure (id, threadId, user, createdAt, etc.)",
                        "Run IDs array format",
                    ],
                    "Summary": "Thread message retrieval and run ID listing validation.",
                },
                "Phase 8.7: Bulk Routes (test_phase8_bulk.py)": {
                    "Endpoints Tested": [
                        "GET /chat/all-messages-for-all-threads - Bulk load all messages",
                    ],
                    "Test Cases": {
                        "First Call": [
                            "Cache miss - slow response expected",
                            "Data loaded from database",
                        ],
                        "Second Call": [
                            "Cache hit - faster response",
                            "Same data returned",
                        ],
                        "Concurrent Calls": [
                            "Multiple simultaneous requests",
                            "Lock prevents duplicate loading",
                        ],
                        "Different User": [
                            "User isolation",
                            "Separate cache per user",
                        ],
                    },
                    "Validation": [
                        "Caching behavior (BULK_CACHE_TIMEOUT)",
                        "Lock mechanism (_bulk_loading_locks)",
                        "Performance improvement on cached requests",
                    ],
                    "Summary": "Bulk message loading with caching validation.",
                },
                "Phase 8.8: Debug Routes (test_phase8_debug.py)": {
                    "Endpoints Tested": [
                        "GET /debug/chat/{thread_id}/checkpoints - View checkpoints",
                        "GET /debug/pool-status - PostgreSQL pool diagnostics",
                        "GET /debug/run-id/{run_id} - Run ID details",
                        "POST /admin/clear-cache - Clear bulk cache",
                        "POST /admin/clear-prepared-statements - Clear PG statements",
                        "POST /debug/set-env - Set debug environment variables",
                        "POST /debug/reset-env - Reset debug environment variables",
                    ],
                    "Validation": [
                        "Debug data structure validation",
                        "Admin endpoint authorization",
                        "Pool status metrics",
                    ],
                    "Summary": "Debug and admin endpoint validation.",
                },
                "Phase 8.9: Misc Routes (test_phase8_misc.py)": {
                    "Endpoints Tested": [
                        "GET /placeholder/{width}/{height} - SVG placeholder generation",
                    ],
                    "Test Cases": [
                        "Standard dimensions (120x80, 300x200, 640x480)",
                        "Edge cases (1x1, 9999x9999)",
                        "Error handling (invalid dimensions)",
                    ],
                    "Validation": [
                        "SVG content verification (<svg tag)",
                        "Width and height attributes",
                        "viewBox attribute",
                    ],
                    "Summary": "Placeholder image generation validation.",
                },
                "Summary": "Comprehensive route testing covering all API endpoint categories.",
            },
            "Phase 9: Main Application (test_phase9_main.py)": {
                "Purpose": "Test main application entry point and integration",
                "Tests": {
                    "test_application_startup": [
                        "FastAPI app import",
                        "App instance attributes (title, version, openapi_url)",
                        "Lifespan context manager",
                        "Route registration",
                    ],
                    "test_middleware_functionality": [
                        "CORS headers in responses",
                        "Brotli compression",
                        "Rate limiting behavior",
                        "Memory monitoring",
                    ],
                },
                "Summary": "Main application structure and middleware integration validation.",
            },
            "Phase 10: External Imports (test_phase10_external_imports.py)": {
                "Purpose": "Validate external dependencies and compatibility",
                "Tests": [
                    "test_import_validation - Critical package imports",
                    "test_external_file_compatibility - External file integrations",
                    "test_modular_endpoints - Modular endpoint structure",
                    "test_backward_compatibility - Version compatibility",
                ],
                "Summary": "External dependency and compatibility validation.",
            },
            "Phase 12: Performance (test_phase12_performance.py)": {
                "Purpose": "Performance benchmarking and optimization validation",
                "Tests": {
                    "test_application_startup_time": [
                        "Startup time measurement",
                        "Initialization performance",
                    ],
                    "test_memory_usage_patterns": [
                        "Memory baseline establishment",
                        "Memory growth patterns",
                        "GC effectiveness",
                    ],
                    "test_concurrent_request_handling": [
                        "Concurrent request throughput",
                        "Resource contention",
                    ],
                    "test_concurrent_analysis_performance": [
                        "MAX_CONCURRENT_ANALYSES semaphore behavior",
                        "Analysis request queueing",
                    ],
                    "test_endpoint_response_times": [
                        "Response time benchmarks per endpoint",
                        "SLA compliance",
                    ],
                    "test_load_performance": [
                        "Sustained load handling",
                        "Performance degradation",
                    ],
                    "test_cache_performance": [
                        "Cache hit rates",
                        "Cache speedup measurements",
                    ],
                    "test_database_connection_performance": [
                        "Connection pool efficiency",
                        "Query execution times",
                    ],
                    "test_authentication_performance": [
                        "JWT verification overhead",
                        "Token validation speed",
                    ],
                },
                "Summary": "Comprehensive performance benchmarking and stress testing.",
            },
            "Summary": "Extensive API route testing with 12 test phases covering structure, functionality, and performance.",
        },
        "3. Database/Checkpointer Tests (tests/database/)": {
            "test_checkpointer_checkpointer.py": {
                "Purpose": "Test core checkpointer factory and lifecycle management",
                "Components Tested": {
                    "Factory Functions": [
                        "create_async_postgres_saver() - Create checkpointer instance",
                        "close_async_postgres_saver() - Clean shutdown",
                        "get_global_checkpointer() - Singleton access",
                        "initialize_checkpointer() - Startup initialization",
                        "cleanup_checkpointer() - Shutdown cleanup",
                    ],
                    "Health Checks": [
                        "check_pool_health_and_recreate() - Pool health validation and recovery",
                    ],
                },
                "Test Cases": [
                    "Checkpointer creation and initialization",
                    "Global checkpointer singleton behavior",
                    "Health check and pool recreation",
                    "Graceful shutdown and cleanup",
                    "Error handling during creation",
                ],
                "Summary": "Core checkpointer creation, lifecycle, and health management.",
            },
            "test_checkpointer_database.py": {
                "Purpose": "Test database layer: connections, configuration, and basic operations",
                "Components Tested": {
                    "Connection Management": [
                        "get_connection_string() - Build PostgreSQL connection string",
                        "get_connection_kwargs() - Connection parameters",
                        "get_direct_connection() - Direct async connection",
                    ],
                    "Configuration": [
                        "get_db_config() - Retrieve database configuration",
                        "check_postgres_env_vars() - Validate environment variables",
                    ],
                },
                "Test Cases": [
                    "Connection string construction",
                    "Direct connection creation and cleanup",
                    "Configuration retrieval and validation",
                    "Environment variable presence",
                ],
                "Summary": "Database connection and configuration foundation.",
            },
            "test_checkpointer_error_handling.py": {
                "Purpose": "Test error handling mechanisms and retry logic",
                "Components Tested": {
                    "Prepared Statement Errors": [
                        "is_prepared_statement_error() - Error detection",
                        "clear_prepared_statements() - Error recovery",
                    ],
                    "Retry Decorators": [
                        "retry_on_prepared_statement_error - Automatic retry wrapper",
                    ],
                },
                "Test Cases": [
                    "Prepared statement error detection",
                    "Automatic statement clearing",
                    "Retry behavior with exponential backoff",
                    "Max retry limits",
                    "Error propagation after max retries",
                ],
                "Summary": "Error detection, recovery, and retry mechanism validation.",
            },
            "test_checkpointer_overall.py": {
                "Purpose": "Integration tests combining all checkpointer components",
                "Test Categories": {
                    "Database Foundation": [
                        "Connection string generation",
                        "Direct connection creation",
                        "Configuration retrieval",
                    ],
                    "Table Management": [
                        "setup_checkpointer_with_autocommit() - Create checkpointer tables",
                        "setup_users_threads_runs_table() - Create user management tables",
                        "table_exists() - Table existence validation",
                    ],
                    "Connection Pool": [
                        "modern_psycopg_pool() - Pool creation",
                        "cleanup_all_pools() - Pool cleanup",
                        "force_close_modern_pools() - Forced shutdown",
                    ],
                    "Checkpointer Management": [
                        "initialize_checkpointer() - Full initialization",
                        "cleanup_checkpointer() - Full cleanup",
                        "check_pool_health_and_recreate() - Health and recovery",
                    ],
                    "End-to-End Workflows": [
                        "Complete initialization sequence",
                        "User thread creation and querying",
                        "Sentiment tracking",
                        "Complete cleanup sequence",
                    ],
                },
                "Test Cases": "50+ integration tests combining components",
                "Summary": "Comprehensive integration testing of all database components working together.",
            },
            "test_checkpointer_user_management.py": {
                "Purpose": "Test user management: thread operations and sentiment tracking",
                "Components Tested": {
                    "Thread Operations": [
                        "create_thread_run_entry() - Create thread run",
                        "get_user_chat_threads() - List user threads",
                        "get_user_chat_threads_count() - Count user threads",
                        "delete_user_thread_entries() - Delete thread",
                    ],
                    "Sentiment Tracking": [
                        "update_thread_run_sentiment() - Update sentiment",
                        "get_thread_run_sentiments() - Retrieve sentiments",
                    ],
                },
                "Test Cases": {
                    "Sentiment Tests": [
                        "Update sentiment (positive, negative, neutral)",
                        "Retrieve sentiments for thread",
                        "Handle non-existent run_id",
                    ],
                    "Thread Tests": [
                        "Create thread run entry",
                        "List threads with pagination",
                        "Count threads",
                        "Delete thread entries",
                    ],
                    "Integration": [
                        "Combined thread and sentiment workflows",
                    ],
                },
                "Summary": "User thread management and sentiment tracking validation.",
            },
            "test_checkpointer_stress.py": {
                "Purpose": "Stress testing for performance, concurrency, and resource limits",
                "Stress Tests": {
                    "test_concurrent_connections": [
                        "Create multiple concurrent connections",
                        "Measure connection time and resource usage",
                        "Verify pool limits enforcement",
                    ],
                    "test_bulk_operations": [
                        "Bulk thread creation",
                        "Bulk sentiment updates",
                        "Measure throughput",
                    ],
                    "test_connection_pool_exhaustion": [
                        "Exhaust pool connections",
                        "Verify queuing behavior",
                        "Test recovery after exhaustion",
                    ],
                    "test_memory_leak": [
                        "Sustained operations over time",
                        "Memory growth tracking",
                        "Detect memory leaks",
                    ],
                    "test_error_recovery": [
                        "Simulate database errors",
                        "Verify automatic recovery",
                        "Connection recreation",
                    ],
                    "test_performance_benchmarks": [
                        "Checkpoint creation speed",
                        "Thread query performance",
                        "Sentiment update speed",
                    ],
                },
                "Metrics Collected": [
                    "Memory usage (before/after, peak)",
                    "Operation throughput (ops/second)",
                    "Response times (avg, min, max)",
                    "Concurrent operation count",
                    "Error rates",
                ],
                "Summary": "Performance benchmarking and stress testing under heavy load.",
            },
            "test_database_connection.py": {
                "Purpose": "Test basic database connectivity with multiple drivers",
                "Tests": {
                    "test_sync_connection": [
                        "psycopg (sync) connection",
                        "Basic query execution",
                        "Connection close",
                    ],
                    "test_async_connection_asyncpg": [
                        "asyncpg connection",
                        "Async query execution",
                        "Async close",
                    ],
                    "test_async_connection_psycopg": [
                        "psycopg (async) connection",
                        "Async query with psycopg",
                        "Async close",
                    ],
                    "test_connection_pooling": [
                        "Connection pool creation",
                        "Pool connection acquisition",
                        "Pool cleanup",
                    ],
                },
                "Summary": "Basic database connectivity validation with multiple driver options.",
            },
            "test_ssl_fix.py": {
                "Purpose": "Test SSL connection fixes and secure connections",
                "Tests": [
                    "SSL connection parameter validation",
                    "Secure connection establishment",
                    "Certificate verification",
                ],
                "Summary": "SSL/TLS connection security validation.",
            },
            "Summary": "Comprehensive database and checkpointer testing covering connections, operations, stress, and error handling.",
        },
        "4. Other Tests (tests/other/)": {
            "test_memory_cleanup.py": {
                "Purpose": "Test memory cleanup mechanisms and garbage collection",
                "Tests": {
                    "test_manual_cleanup": [
                        "Manual gc.collect() invocation",
                        "Memory release verification",
                        "malloc_trim() if available",
                    ],
                    "test_periodic_cleanup": [
                        "Background cleanup task",
                        "Periodic cleanup interval",
                        "Memory trend analysis",
                    ],
                },
                "Metrics": [
                    "Memory before/after cleanup (RSS MB)",
                    "Objects collected count",
                    "Cleanup effectiveness",
                ],
                "Summary": "Memory management and cleanup effectiveness validation.",
            },
            "test_fastmcp_integration.py": {
                "Purpose": "Test FastMCP server integration and connectivity",
                "Tests": [
                    "FastMCP server connection",
                    "Tool invocation",
                    "Response handling",
                ],
                "Summary": "MCP (Model Context Protocol) server integration validation.",
            },
            "test_chromadb_switching.py": {
                "Purpose": "Test ChromaDB cloud/local switching and client creation",
                "Tests": {
                    "test_environment_config": [
                        "CHROMADB_MODE validation (cloud/local)",
                        "Environment variable presence",
                    ],
                    "test_client_creation": [
                        "Cloud client creation with auth",
                        "Local client creation",
                        "Client switching logic",
                    ],
                    "test_collection_access": [
                        "Collection retrieval",
                        "Query execution",
                        "Error handling",
                    ],
                },
                "Summary": "ChromaDB configuration and client switching validation.",
            },
            "azure_ai_translator_test.py": {
                "Purpose": "Test Azure AI Translator service integration",
                "Details": [
                    "Translation API calls",
                    "Language detection",
                    "Error handling",
                ],
                "Summary": "Azure AI translation service validation.",
            },
            "azure_ai_language_detect_test.py": {
                "Purpose": "Test Azure AI language detection service",
                "Details": [
                    "Language detection API",
                    "Confidence scores",
                    "Multi-language support",
                ],
                "Summary": "Azure AI language detection validation.",
            },
            "Summary": "Specialized tests for memory management, MCP integration, ChromaDB, and Azure AI services.",
        },
        "5. Test Concurrency & Performance (tests/api/)": {
            "test_concurrency.py": {
                "Purpose": "Test concurrent request handling and thread safety",
                "Tests": [
                    "Concurrent analysis requests",
                    "Semaphore limiting behavior",
                    "Thread safety validation",
                    "Resource contention handling",
                ],
                "Summary": "Concurrency and thread safety validation.",
            },
            "Summary": "Dedicated concurrency testing for thread-safe operations.",
        },
        "6. Test Execution Patterns": {
            "Async Test Pattern": {
                "Structure": [
                    "async def test_* functions",
                    "asyncio.run(test_function())",
                    "Proper event loop management",
                ],
                "Summary": "Asynchronous test execution with proper event loop handling.",
            },
            "Server Dependency": {
                "Requirement": "Running server at SERVER_BASE_URL (default: http://localhost:8000)",
                "Validation": "check_server_connectivity() before tests",
                "Failure Handling": "Skip HTTP tests if server unavailable",
                "Summary": "Tests require live server instance.",
            },
            "Database Dependency": {
                "Requirement": "PostgreSQL database configured via environment variables",
                "Validation": "Database connectivity check before database tests",
                "Fallback": "Mock implementations for import failures",
                "Summary": "Database tests require configured PostgreSQL instance.",
            },
            "Environment Setup": {
                "Pattern": [
                    "load_dotenv() at module level",
                    "Environment variable validation",
                    "Debug environment control via API",
                    "Cleanup after tests",
                ],
                "Summary": "Environment configuration and cleanup management.",
            },
            "Result Reporting": {
                "Pattern": [
                    "BaseTestResults aggregation",
                    "analyze_results() summary generation",
                    "save_traceback_report() for failures",
                    "Console output with emoji indicators",
                    "Exit code 0 (pass) or 1 (fail)",
                ],
                "Summary": "Standardized result reporting and exit codes.",
            },
            "Summary": "Common patterns for test execution, dependencies, and reporting.",
        },
        "7. Test Coverage Summary": {
            "API Routes": {
                "Coverage": "26+ endpoints across 12 test phases",
                "Categories": "Health, Analysis, Catalog, Chat, Feedback, Messages, Bulk, Debug, Admin, Misc",
                "Summary": "Complete API surface area coverage.",
            },
            "Database/Checkpointer": {
                "Coverage": "8 test files covering all database operations",
                "Categories": "Connections, Operations, Error Handling, User Management, Stress Testing, Integration",
                "Summary": "Comprehensive database layer validation.",
            },
            "Utilities": {
                "Coverage": "Debug (8 functions), Memory (5 functions), Rate Limiting (3 functions)",
                "Categories": "Debug Logging, Memory Management, Rate Limiting",
                "Summary": "Utility function validation with edge cases.",
            },
            "Infrastructure": {
                "Coverage": "Auth, Models, Exceptions, Middleware, Configuration, Performance",
                "Summary": "Core infrastructure component validation.",
            },
            "Integration": {
                "Coverage": "MCP Server, ChromaDB, Azure AI Services",
                "Summary": "External service integration validation.",
            },
            "Total Test Files": "35+ test files",
            "Total Test Cases": "500+ individual test cases",
            "Test Categories": "API Routes, Database, Utilities, Infrastructure, Integration, Performance",
            "Summary": "Extensive test coverage across all application layers and components.",
        },
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="CZSU Multi-Agent Backend Testing Mindmap v3")
        graph.attr(rankdir="LR")  # Left to right layout for horizontal mindmap
        graph.attr(
            "graph", fontsize="10", ranksep="0.8", nodesep="0.5"
        )  # Tighter spacing

    colors = [
        "lightblue",
        "lightgreen",
        "lightyellow",
        "lightpink",
        "lightcyan",
        "wheat",
        "lavender",
    ]

    for key, value in mindmap_dict.items():
        node_id = f"{parent}_{key}" if parent else key
        # Sanitize node_id more thoroughly
        node_id = (
            node_id.replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "_")
            .replace("/", "_")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("@", "")
            .replace("{", "")
            .replace("}", "")
            .replace(",", "")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "")
            .replace("!", "")
            .replace("?", "")
            .replace("&", "")
            .replace("#", "")
            .replace("*", "")
            .replace("+", "")
        )

        # Set node color based on level
        color = colors[min(level, len(colors) - 1)]

        if isinstance(value, dict):
            # This is a branch node
            graph.node(node_id, key, shape="box", style="filled", fillcolor=color)
            if parent:
                graph.edge(parent, node_id)
            create_mindmap_graph(value, graph, node_id, level + 1)
        elif isinstance(value, list):
            # This is a leaf node with multiple items
            graph.node(node_id, key, shape="ellipse", style="filled", fillcolor=color)
            if parent:
                graph.edge(parent, node_id)
            for item in value:
                # Sanitize item_id more thoroughly
                sanitized_item = (
                    str(item)
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(".", "_")
                    .replace("/", "_")
                    .replace("'", "")
                    .replace('"', "")
                    .replace(":", "")
                    .replace("@", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace(",", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("|", "")
                    .replace("!", "")
                    .replace("?", "")
                    .replace("&", "")
                    .replace("#", "")
                    .replace("*", "")
                    .replace("+", "")[:50]
                )  # Limit length
                item_id = f"{node_id}_{sanitized_item}"
                graph.node(item_id, str(item), shape="plaintext", fontsize="8")
                graph.edge(node_id, item_id)
        else:
            # Single leaf node
            graph.node(node_id, str(value), shape="plaintext", fontsize="8")
            if parent:
                graph.edge(parent, node_id)

    return graph


def main():
    """Generate and save the mindmap visualization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    graph = create_mindmap_graph(mindmap)

    # Save as PNG
    png_path = os.path.join(script_dir, script_name)
    graph.render(png_path, format="png", cleanup=True)
    print(f"Mindmap saved as '{png_path}.png'")

    # Also save as PDF for better quality
    pdf_path = os.path.join(script_dir, script_name)
    graph.render(pdf_path, format="pdf", cleanup=True)
    print(f"Mindmap saved as '{pdf_path}.pdf'")

    # Print text representation
    print("\nText-based Mindmap:")
    print_mindmap_text(mindmap)


def print_mindmap_text(mindmap_dict, prefix=""):
    """Print a text-based representation of the mindmap in a vertical tree format."""
    keys = list(mindmap_dict.keys())
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + key)

        value = mindmap_dict[key]
        if isinstance(value, dict):
            extension = "    " if is_last else "│   "
            print_mindmap_text(value, prefix + extension)
        elif isinstance(value, list):
            for j, item in enumerate(value):
                is_last_sub = j == len(value) - 1
                sub_connector = "└── " if is_last_sub else "├── "
                sub_extension = "    " if is_last else "│   "
                print(prefix + sub_extension + sub_connector + str(item))


if __name__ == "__main__":
    main()

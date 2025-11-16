import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "FastAPI + Uvicorn Backend": {
        "1. Application Entry Point (api/main.py)": {
            "Critical Windows Setup": {
                "Details": [
                    "asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy)",
                    "MUST be FIRST before any imports to fix psycopg async",
                    "Fixes PostgreSQL checkpointer event loop issues on Windows",
                ],
                "Summary": "Windows event loop policy setup for psycopg[binary] compatibility.",
            },
            "Environment Loading": {
                "Details": [
                    "load_dotenv() loads .env file early",
                    "BASE_DIR = Path(__file__).resolve().parents[1]",
                    "sys.path.insert(0, str(BASE_DIR)) for imports",
                ],
                "Summary": "Environment variables and project root setup before imports.",
            },
            "FastAPI Instance Creation": {
                "Details": [
                    "app = FastAPI(title='CZSU Multi-Agent Text-to-SQL API')",
                    "version='1.0.0'",
                    "lifespan=lifespan context manager",
                    "OpenAPI documentation at /docs, /redoc, /openapi.json",
                    "Contact: michael.miscanuk@google.com",
                ],
                "Summary": "Main FastAPI application instance with OpenAPI configuration.",
            },
            "Lifespan Management": {
                "Startup Sequence": {
                    "Details": [
                        "Record _APP_STARTUP_TIME for uptime tracking",
                        "setup_graceful_shutdown() for SIGTERM/SIGINT handlers",
                        "await initialize_checkpointer() for PostgreSQL",
                        "Establish _MEMORY_BASELINE after init (RSS in MB)",
                        "start_memory_profiler() if MEMORY_PROFILER_ENABLED=1",
                        "start_memory_cleanup() background task",
                        "log_memory_usage('app_ready')",
                    ],
                    "Summary": "Application startup initialization with checkpointer and memory tracking.",
                },
                "Shutdown Sequence": {
                    "Details": [
                        "await stop_memory_profiler() if enabled",
                        "await stop_memory_cleanup() background task",
                        "Calculate application uptime",
                        "Compare final memory vs _MEMORY_BASELINE",
                        "Alert if memory growth > GC_MEMORY_THRESHOLD (default 1900MB)",
                        "await cleanup_checkpointer() to close DB connections",
                    ],
                    "Summary": "Graceful shutdown with memory leak detection and resource cleanup.",
                },
                "Summary": "Async context manager for startup/shutdown lifecycle management.",
            },
            "Summary": "Main entry point handling Windows compatibility, environment, and lifecycle.",
        },
        "2. Middleware Stack (Order Matters)": {
            "CORS Middleware": {
                "Details": [
                    "CORSMiddleware from fastapi.middleware.cors",
                    "allow_origins=['*'] (TODO: restrict in production)",
                    "allow_credentials=True for cookies/auth headers",
                    "allow_methods=['*'] all HTTP methods",
                    "allow_headers=['*'] all headers",
                    "File: api/middleware/cors.py + api/main.py",
                ],
                "Summary": "Cross-origin resource sharing for frontend API access.",
            },
            "GZip Middleware": {
                "Details": [
                    "GZipMiddleware from fastapi.middleware.gzip",
                    "minimum_size=1000 bytes before compression",
                    "Reduces bandwidth and improves performance",
                    "File: api/middleware/cors.py + api/main.py",
                ],
                "Summary": "Response compression for payloads > 1000 bytes.",
            },
            "Rate Limiting Middleware": {
                "Details": [
                    "@app.middleware('http') throttling_middleware",
                    "Skips /health, /docs, /openapi.json, /debug/pool-status",
                    "Per-client IP semaphore limiting (throttle_semaphores)",
                    "check_rate_limit_with_throttling(client_ip)",
                    "wait_for_rate_limit() - waits instead of rejecting",
                    "RATE_LIMIT_BURST=20, RATE_LIMIT_WINDOW=60s",
                    "RATE_LIMIT_REQUESTS=100 per window",
                    "Returns 429 with retry_after header if exceeded",
                    "File: api/main.py + api/utils/rate_limiting.py",
                ],
                "Summary": "Intelligent throttling that waits for rate limits instead of rejecting.",
            },
            "Memory Monitoring Middleware": {
                "Details": [
                    "@app.middleware('http') simplified_memory_monitoring_middleware",
                    "Tracks memory for /analyze and /chat/all-messages-* only",
                    "log_memory_usage('before_<path>') and 'after_<path>'",
                    "Increments global _REQUEST_COUNT",
                    "Avoids overhead on every request",
                    "File: api/main.py + api/middleware/memory_monitoring.py",
                ],
                "Summary": "Memory usage tracking for heavy operations to detect leaks.",
            },
            "Summary": "Layered middleware for CORS, compression, rate limiting, and monitoring.",
        },
        "3. Exception Handlers": {
            "Validation Exception Handler": {
                "Details": [
                    "@app.exception_handler(RequestValidationError)",
                    "Returns 422 status code with error details",
                    "Uses jsonable_encoder to avoid JSON serialization errors",
                    "Falls back to simplified errors if encoding fails",
                    "File: api/main.py + api/exceptions/handlers.py",
                ],
                "Summary": "Handles Pydantic validation errors with proper 422 responses.",
            },
            "HTTP Exception Handler": {
                "Details": [
                    "@app.exception_handler(StarletteHTTPException)",
                    "Enhanced debugging for 401 authentication errors",
                    "Logs request URL, method, headers, client IP",
                    "Full traceback for debugging auth issues",
                    "Generic handling for other HTTP errors",
                    "File: api/main.py + api/exceptions/handlers.py",
                ],
                "Summary": "Comprehensive HTTP exception handling with detailed 401 debugging.",
            },
            "ValueError Handler": {
                "Details": [
                    "@app.exception_handler(ValueError)",
                    "Returns 400 Bad Request",
                    "For business logic validation failures",
                    "File: api/main.py",
                ],
                "Summary": "Handles ValueError as 400 Bad Request.",
            },
            "General Exception Handler": {
                "Details": [
                    "@app.exception_handler(Exception)",
                    "Returns 500 Internal Server Error",
                    "Includes traceback if DEBUG_TRACEBACK=1",
                    "Generic 'Internal server error' in production",
                    "Prevents application crashes",
                    "File: api/main.py",
                ],
                "Summary": "Catch-all handler for unexpected exceptions.",
            },
            "Summary": "Global exception handlers for consistent error responses.",
        },
        "4. Configuration & Settings (api/config/settings.py)": {
            "Global Variables": {
                "Details": [
                    "_APP_STARTUP_TIME: datetime for uptime tracking",
                    "_MEMORY_BASELINE: RSS memory at startup (MB)",
                    "_REQUEST_COUNT: total requests processed",
                    "GLOBAL_CHECKPOINTER: shared AsyncPostgresSaver instance",
                    "analysis_semaphore: limits concurrent /analyze requests",
                    "throttle_semaphores: defaultdict per-IP semaphores",
                    "rate_limit_storage: defaultdict for rate limit tracking",
                    "_bulk_loading_cache: cache for bulk operations",
                    "_bulk_loading_locks: asyncio.Lock per cache key",
                ],
                "Summary": "Shared global state for lifecycle, memory, concurrency, and caching.",
            },
            "Environment Constants": {
                "Details": [
                    "INMEMORY_FALLBACK_ENABLED: fallback to MemorySaver",
                    "MAX_CONCURRENT_ANALYSES: semaphore limit (default 3)",
                    "RATE_LIMIT_REQUESTS=100, RATE_LIMIT_WINDOW=60",
                    "RATE_LIMIT_BURST=20, RATE_LIMIT_MAX_WAIT=5",
                    "BULK_CACHE_TIMEOUT=120 seconds",
                    "GOOGLE_JWK_URL for JWT verification",
                    "_JWT_KID_MISSING_COUNT: counter for NextAuth tokens",
                ],
                "Summary": "Configuration constants loaded from environment variables.",
            },
            "Summary": "Centralized configuration and global state management.",
        },
        "5. Authentication (api/auth/ + api/dependencies/)": {
            "JWT Token Verification (api/auth/jwt_auth.py)": {
                "Details": [
                    "verify_google_jwt(token) validates Google OAuth tokens",
                    "Checks JWT format (3 parts separated by dots)",
                    "Validates 'kid' field in header for standard tokens",
                    "Supports NextAuth.js id_token (missing 'kid')",
                    "Uses Google tokeninfo endpoint for NextAuth",
                    "JWKS from https://www.googleapis.com/oauth2/v3/certs",
                    "Verifies audience matches GOOGLE_CLIENT_ID",
                    "5-minute leeway for clock sync (leeway=300)",
                    "Test token support if USE_TEST_TOKENS=1",
                    "Returns decoded payload with user info",
                ],
                "Summary": "Google OAuth JWT verification with NextAuth.js support and clock sync handling.",
            },
            "Auth Dependency (api/dependencies/auth.py)": {
                "Details": [
                    "get_current_user(authorization: str = Header(None))",
                    "Extracts 'Bearer <token>' from Authorization header",
                    "Calls verify_google_jwt(token)",
                    "Returns user_info dict with email, name, etc.",
                    "Raises HTTPException 401 if missing/invalid",
                    "Enhanced debug logging with print__token_debug",
                ],
                "Summary": "FastAPI dependency for extracting and verifying user from JWT token.",
            },
            "Summary": "JWT authentication with Google OAuth and NextAuth.js support.",
        },
        "6. Request/Response Models (api/models/)": {
            "Request Models (api/models/requests.py)": {
                "Details": [
                    "AnalyzeRequest: prompt, thread_id, run_id (optional UUID)",
                    "FeedbackRequest: run_id, feedback (0/1), comment",
                    "SentimentRequest: run_id, sentiment (bool or null)",
                    "All use Pydantic BaseModel with validators",
                    "UUID format validation for run_id fields",
                    "Min/max length constraints on strings",
                ],
                "Summary": "Pydantic request models with validation for API endpoints.",
            },
            "Response Models (api/models/responses.py)": {
                "Details": [
                    "ChatThreadResponse: thread_id, latest_timestamp, run_count, title, full_prompt",
                    "PaginatedChatThreadsResponse: threads[], total_count, page, limit, has_more",
                    "ChatMessage: id, threadId, user, createdAt, prompt, final_answer, etc.",
                    "Includes queries_and_results, datasets_used, top_chunks",
                    "sql_query, error, isLoading, run_id, followup_prompts",
                ],
                "Summary": "Pydantic response models for structured API responses.",
            },
            "Summary": "Type-safe request/response models with Pydantic validation.",
        },
        "7. API Routes (api/routes/)": {
            "Root Route (api/routes/root.py)": {
                "Details": [
                    "GET / - Welcome message and API info",
                    "Returns API title, version, documentation links",
                ],
                "Summary": "Basic API information endpoint.",
            },
            "Health Routes (api/routes/health.py)": {
                "Details": [
                    "GET /health - Overall health check",
                    "GET /health/database - Database connection health",
                    "GET /health/memory - Memory usage and cache info",
                    "GET /health/rate-limits - Rate limit status",
                    "GET /health/prepared-statements - PostgreSQL prepared statements",
                    "Returns status, memory stats, uptime, database health",
                    "Tests AsyncPostgresSaver with aget_tuple()",
                    "Runs gc.collect() and reports objects collected",
                    "Calculates estimated_max_threads based on memory",
                ],
                "Summary": "Comprehensive health check endpoints for monitoring.",
            },
            "Analysis Route (api/routes/analysis.py)": {
                "Details": [
                    "POST /analyze - Main NL-to-SQL conversion endpoint",
                    "Requires authentication via get_current_user",
                    "Uses analysis_semaphore to limit concurrent requests",
                    "Generates run_id (UUID) if not provided",
                    "Registers execution with cancellation system",
                    "Calls analysis_main() from main.py (LangGraph)",
                    "Creates thread_run_entry in database",
                    "Extracts metadata from checkpoint state",
                    "Handles cancellation via is_cancelled() checks",
                    "Unregisters execution on completion",
                    "Returns final_answer, sql_query, datasets_used, etc.",
                ],
                "Summary": "Natural language to SQL conversion with LangGraph integration.",
            },
            "Catalog Route (api/routes/catalog.py)": {
                "Details": [
                    "GET /catalog - Browse CZSU data catalog",
                    "Requires authentication",
                    "Paginated results with page, page_size, q (search query)",
                    "Searches selection_descriptions.db SQLite database",
                    "Filters by selection_code or extended_description",
                    "Returns selection codes and descriptions",
                    "ORDER BY selection_code with LIMIT/OFFSET",
                ],
                "Summary": "Searchable catalog of CZSU statistical datasets.",
            },
            "Chat Routes (api/routes/chat.py)": {
                "Details": [
                    "POST /chat/threads - Create new thread",
                    "GET /chat/threads - List user's threads (paginated)",
                    "GET /chat/threads/{thread_id} - Get single thread messages",
                    "DELETE /chat/threads/{thread_id} - Delete thread",
                    "GET /chat/all-messages-for-all-threads - Bulk load all threads",
                    "GET /chat/all-messages-for-one-thread/{thread_id} - Load single thread",
                    "Uses checkpointer to extract messages from state",
                    "Security check: verify user owns thread before access",
                    "Caching with _bulk_loading_cache for performance",
                    "Returns ChatMessage[] with metadata",
                ],
                "Summary": "Multi-threaded conversation management with caching.",
            },
            "Feedback Route (api/routes/feedback.py)": {
                "Details": [
                    "POST /feedback - Submit user feedback",
                    "Requires authentication",
                    "Accepts run_id, feedback (0/1), comment",
                    "Submits to LangSmith via Client.create_feedback()",
                    "Updates sentiment in database via update_thread_run_sentiment()",
                    "Returns confirmation with run_id",
                ],
                "Summary": "User feedback collection with LangSmith integration.",
            },
            "Messages Route (api/routes/messages.py)": {
                "Details": [
                    "GET /chat/messages/{thread_id} - Get thread messages",
                    "Requires authentication",
                    "Retrieves messages from checkpointer state",
                    "Returns formatted ChatMessage list",
                ],
                "Summary": "Message retrieval for specific threads.",
            },
            "Stop Route (api/routes/stop.py)": {
                "Details": [
                    "POST /stop-execution - Cancel running analysis",
                    "Requires authentication",
                    "Accepts thread_id and run_id",
                    "Calls request_cancellation() from cancellation.py",
                    "Returns success or not_found status",
                ],
                "Summary": "Execution control for cancelling running queries.",
            },
            "Debug Route (api/routes/debug.py)": {
                "Details": [
                    "GET /debug/pool-status - PostgreSQL pool diagnostics",
                    "GET /debug/memory-status - Memory usage stats",
                    "Internal diagnostics for development",
                ],
                "Summary": "Debug endpoints for internal diagnostics.",
            },
            "Bulk Route (api/routes/bulk.py)": {
                "Details": [
                    "POST /bulk/* - Batch operations",
                    "Bulk message loading and processing",
                ],
                "Summary": "Batch processing endpoints.",
            },
            "Misc Route (api/routes/misc.py)": {
                "Details": [
                    "Miscellaneous utility endpoints",
                    "Additional helper functions",
                ],
                "Summary": "Utility endpoints for various operations.",
            },
            "Summary": "Organized API routes for analysis, catalog, chat, feedback, and control.",
        },
        "8. Utility Functions (api/utils/)": {
            "Memory Management (api/utils/memory.py)": {
                "Details": [
                    "log_memory_usage(context) - Log RSS memory with label",
                    "check_memory_and_gc() - Check threshold and trigger cleanup",
                    "force_release_memory() - gc.collect() + malloc_trim(0)",
                    "cleanup_bulk_cache() - Remove expired cache entries",
                    "start_memory_profiler() - tracemalloc snapshots",
                    "stop_memory_profiler() - Report top memory consumers",
                    "start_memory_cleanup() - Background cleanup task",
                    "stop_memory_cleanup() - Stop background task",
                    "setup_graceful_shutdown() - SIGTERM/SIGINT handlers",
                    "log_comprehensive_error() - Detailed error logging",
                    "_get_uvicorn_logger() - Get uvicorn.error logger",
                    "GC_MEMORY_THRESHOLD=1900MB default",
                    "MEMORY_PROFILER_ENABLED, INTERVAL, TOP_STATS from env",
                    "MEMORY_CLEANUP_ENABLED=1, INTERVAL=60s",
                    "Uses psutil for process memory info",
                    "malloc_trim support via ctypes.CDLL('libc.so.6')",
                ],
                "Summary": "Comprehensive memory monitoring, profiling, and cleanup utilities.",
            },
            "Rate Limiting (api/utils/rate_limiting.py)": {
                "Details": [
                    "check_rate_limit_with_throttling(client_ip) - Returns throttle info dict",
                    "wait_for_rate_limit(client_ip) - Async wait with retry",
                    "check_rate_limit(client_ip) - Simple boolean check",
                    "Uses rate_limit_storage defaultdict",
                    "RATE_LIMIT_BURST=20, WINDOW=60, REQUESTS=100",
                    "RATE_LIMIT_MAX_WAIT=5 seconds",
                    "Cleans old entries from storage",
                    "Returns suggested_wait, burst_count, window_count",
                ],
                "Summary": "Rate limiting with intelligent wait-instead-of-reject strategy.",
            },
            "Debug Functions (api/utils/debug.py)": {
                "Details": [
                    "30+ debug print functions controlled by env vars",
                    "print__analyze_debug, print__memory_monitoring, print__token_debug",
                    "print__feedback_flow, print__sentiment_flow, print__chat_*_debug",
                    "All check env var before printing (e.g., print__debug=1)",
                    "sys.stdout.flush() for immediate output",
                    "Enables granular debug logging per module",
                ],
                "Summary": "Environment-controlled debug logging functions for all modules.",
            },
            "Cancellation (api/utils/cancellation.py)": {
                "Details": [
                    "register_execution(thread_id, run_id) - Track running execution",
                    "request_cancellation(thread_id, run_id) - Set cancelled flag",
                    "is_cancelled(thread_id, run_id) - Check if cancelled",
                    "unregister_execution(thread_id, run_id) - Remove from registry",
                    "cleanup_old_entries() - Remove entries > 30 minutes",
                    "get_active_count() - Count active executions",
                    "_cancellation_registry: Dict[(thread_id, run_id), {cancelled, timestamp}]",
                ],
                "Summary": "Execution cancellation system for multi-user query control.",
            },
            "Summary": "Utility functions for memory, rate limiting, debugging, and cancellation.",
        },
        "9. Helper Functions (api/helpers.py)": {
            "Details": [
                "traceback_json_response(e, status_code, run_id) - Error response",
                "Returns JSONResponse with traceback if DEBUG_TRACEBACK=1",
                "Includes run_id in response if provided",
                "Fallback returns None for caller to handle",
            ],
            "Summary": "Error handling helper for consistent JSON error responses.",
        },
        "10. Uvicorn Server (uvicorn_start.py)": {
            "Uvicorn Configuration": {
                "Details": [
                    "uvicorn.run('api.main:app')",
                    "host='0.0.0.0', port=8000",
                    "reload=True for development",
                    "reload_dirs=['api', 'my_agent'] watch directories",
                    "reload_delay=0.25 to prevent multiple reloads",
                    "log_level='info', use_colors=True",
                    "access_log=True for request logging",
                    "Windows event loop policy set before import",
                ],
                "Summary": "Uvicorn development server with hot reload configuration.",
            },
            "Production Deployment": {
                "Details": [
                    "Railway: python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}",
                    "Gunicorn option: gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker",
                    "pyproject.toml: uvicorn[standard]>=0.24.0",
                    "railway.toml: RAILPACK builder with uvicorn start command",
                ],
                "Summary": "Production deployment with Uvicorn/Gunicorn on Railway.",
            },
            "Summary": "ASGI server startup for development and production.",
        },
        "11. Memory Management Strategy": {
            "Memory Baseline": {
                "Details": [
                    "Established at startup after initialization",
                    "process.memory_info().rss / 1024 / 1024 (MB)",
                    "Used as reference for growth calculations",
                    "Stored in _MEMORY_BASELINE global variable",
                ],
                "Summary": "Reference memory footprint after initialization.",
            },
            "Real-Time Monitoring": {
                "Details": [
                    "Per-request tracking for /analyze, /chat/all-messages-*",
                    "log_memory_usage() before and after heavy operations",
                    "RSS measurement and growth analysis",
                    "Alerts when exceeding GC_MEMORY_THRESHOLD (1900MB)",
                ],
                "Summary": "Track memory for heavy operations to detect leaks.",
            },
            "Memory Profiler (Optional)": {
                "Details": [
                    "Uses tracemalloc for detailed snapshots",
                    "MEMORY_PROFILER_ENABLED=1 to enable",
                    "MEMORY_PROFILER_INTERVAL=30 seconds",
                    "MEMORY_PROFILER_TOP_STATS=10 top consumers",
                    "Reports file paths and allocation sizes",
                    "Compares snapshots for growth detection",
                ],
                "Summary": "Optional tracemalloc-based profiler for leak detection.",
            },
            "Background Cleanup": {
                "Details": [
                    "Periodic gc.collect() + malloc_trim(0)",
                    "MEMORY_CLEANUP_ENABLED=1 by default",
                    "MEMORY_CLEANUP_INTERVAL=60 seconds",
                    "Forces memory return to OS",
                    "Prevents heap fragmentation",
                    "Runs as asyncio background task",
                ],
                "Summary": "Automated garbage collection and memory release to OS.",
            },
            "Leak Detection": {
                "Details": [
                    "Compare final memory vs _MEMORY_BASELINE at shutdown",
                    "Alert if growth > GC_MEMORY_THRESHOLD (1900MB)",
                    "Detailed statistics for investigation",
                    "Route registration monitoring to prevent duplicates",
                ],
                "Summary": "Shutdown memory comparison for leak identification.",
            },
            "Summary": "Multi-layered memory management for long-running production deployment.",
        },
        "12. Deployment Configuration": {
            "Railway Deployment (railway.toml)": {
                "Details": [
                    "builder = 'RAILPACK'",
                    "buildCommand: uv install, unzip_files.py, rm *.zip",
                    "startCommand: python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}",
                    "RAILPACK_DEPLOY_APT_PACKAGES = 'libsqlite3-0'",
                    "restartPolicyType = 'ON_FAILURE'",
                    "restartPolicyMaxRetries = 5",
                    "limitOverride: memoryBytes = 4000000000 (4GB)",
                    "sleepApplication = true when inactive",
                    "multiRegionConfig: europe-west4",
                ],
                "Summary": "Railway deployment with RAILPACK builder and 4GB memory.",
            },
            "Dependencies (pyproject.toml)": {
                "Details": [
                    "fastapi>=0.104.1, uvicorn[standard]>=0.24.0",
                    "starlette==0.45.3, pydantic>=2.5.0",
                    "langchain, langgraph, langsmith ecosystem",
                    "psycopg[binary,pool]>=3.1.13 for PostgreSQL",
                    "chromadb>=0.4.18 for vector search",
                    "cohere==5.15.0 for reranking",
                    "PyJWT[crypto]>=2.8.0 for authentication",
                    "psutil>=5.9.0 for memory monitoring",
                    "python-dotenv>=1.0.0 for env management",
                ],
                "Summary": "FastAPI stack with LangChain, PostgreSQL, and ChromaDB.",
            },
            "Summary": "Production deployment on Railway with comprehensive dependency management.",
        },
        "Summary": "Production-ready FastAPI application with Uvicorn ASGI server, comprehensive middleware, authentication, memory management, and Railway deployment.",
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="CZSU Multi-Agent Backend Mindmap v3")
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
                    item.replace(" ", "_")
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
                )
                item_id = f"{node_id}_{sanitized_item}"
                graph.node(item_id, item, shape="plaintext", fontsize="8")
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
                print(prefix + sub_extension + sub_connector + item)


if __name__ == "__main__":
    main()

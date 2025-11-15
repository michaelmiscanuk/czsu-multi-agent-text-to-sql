import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "CZSU Multi-Agent Text-to-SQL Backend": {
        "1. FastAPI Application": {
            "Configuration": {
                "Environment bootstrapping (api/main.py)": {
                    "Details": [
                        "BASE_DIR = Path(__file__).resolve().parents[1]",
                        "sys.path.insert(0, str(BASE_DIR)) to enable absolute imports",
                        "load_dotenv() before other module imports",
                        "WindowsSelectorEventLoopPolicy applied on win32",
                    ],
                    "Summary": "Sets up Python paths, environment variables, and event loop policy so imports and async behavior are consistent across platforms.",
                },
                "Runtime settings (api/config/settings.py)": {
                    "Details": [
                        "start_time = time.time() for uptime tracking",
                        "INMEMORY_FALLBACK_ENABLED via InMemorySaver_fallback env flag",
                        "_app_startup_time/_memory_baseline/_request_count trackers",
                        "GLOBAL_CHECKPOINTER placeholder shared across requests",
                        "MAX_CONCURRENT_ANALYSES with analysis_semaphore",
                        "throttle_semaphores per IP (max 8 concurrent requests)",
                        "rate_limit_storage with RATE_LIMIT_* constants",
                        "_bulk_loading_cache, _bulk_loading_locks, BULK_CACHE_TIMEOUT",
                        "GOOGLE_JWK_URL and _jwt_kid_missing_count",
                    ],
                    "Summary": "Centralizes runtime counters, concurrency primitives, caching layers, and JWT configuration driven by environment variables.",
                },
                "Memory controls (api/main.py)": {
                    "Details": [
                        "GC_MEMORY_THRESHOLD env default = 1900MB",
                        "MEMORY_PROFILER_ENABLED flag with interval/top stats",
                        "MEMORY_PROFILER_INTERVAL and MEMORY_PROFILER_TOP_STATS",
                    ],
                    "Summary": "Captures memory profiler flags and thresholds that downstream utilities use for monitoring and alerts.",
                },
                "Lifespan management (lifespan() in api/main.py)": {
                    "Startup sequence": {
                        "Details": [
                            "_app_startup_time = datetime.now()",
                            "setup_graceful_shutdown()",
                            "await initialize_checkpointer()",
                            "log_memory_usage('app_startup')",
                            "Establish _memory_baseline via psutil",
                        ],
                        "Summary": "Performs instrumentation and checkpointer initialization before the application begins accepting requests.",
                    },
                    "Background tasks": {
                        "Details": [
                            "start_memory_profiler() when enabled",
                            "cleanup_task = start_memory_cleanup()",
                            "print__memory_monitoring() for route registration",
                        ],
                        "Summary": "Starts optional profiler and cleanup loops while documenting monitoring configuration.",
                    },
                    "Shutdown sequence": {
                        "Details": [
                            "await stop_memory_profiler() when enabled",
                            "await stop_memory_cleanup()",
                            "Log final RSS growth vs _memory_baseline",
                            "await cleanup_checkpointer()",
                        ],
                        "Summary": "Stops background workers, logs memory differences, and tears down the checkpointer cleanly.",
                    },
                    "Summary": "Coordinates startup and shutdown orchestration so checkpointer and memory services remain healthy.",
                },
                "Summary": "Prepares the runtime environment, settings, and lifecycle hooks so the FastAPI service starts cleanly and stays healthy.",
            },
            "Middleware": {
                "CORS": {
                    "CORSMiddleware": {
                        "Details": [
                            "allow_origins=['*']",
                            "allow_credentials=True",
                            "allow_methods=['*']",
                            "allow_headers=['*']",
                        ],
                        "Summary": "Allows any origin while supporting credentials so the frontend and tooling can integrate without friction.",
                    },
                    "Summary": "Governs cross-origin access policies for external client applications.",
                },
                "GZip": {
                    "GZipMiddleware": {
                        "Details": [
                            "minimum_size=1000",
                            "Compress JSON/api responses over threshold",
                        ],
                        "Summary": "Applies gzip compression when payloads exceed 1KB to reduce bandwidth and latency.",
                    },
                    "Summary": "Adds transparent response compression for sizable payloads.",
                },
                "Throttling": {
                    "@app.middleware('http') throttling_middleware()": {
                        "Details": [
                            "Skip /health, /docs, /openapi.json, /debug/pool-status",
                            "Resolve client_ip and acquire throttle_semaphores[client_ip]",
                            "await wait_for_rate_limit(client_ip)",
                            "Return JSONResponse(429) with rate info when exceeded",
                        ],
                        "Summary": "Gates incoming requests using per-IP semaphores and returns 429 responses whenever concurrency budgets are exhausted.",
                    },
                    "Rate limiting helpers": {
                        "Details": [
                            "check_rate_limit_with_throttling() for metrics",
                            "wait_for_rate_limit(client_ip) with retry + sleep",
                            "log_comprehensive_error('rate_limit_exceeded_after_wait', ...)",
                        ],
                        "Summary": "Implements wait-and-retry semantics plus detailed telemetry when clients exceed rate limits.",
                    },
                    "Summary": "Controls per-IP concurrency and rate limits to protect downstream services.",
                },
                "Memory monitoring": {
                    "@app.middleware('http') simplified_memory_monitoring_middleware()": {
                        "Details": [
                            "Increment _request_count",
                            "log_memory_usage() before/after heavy endpoints",
                            "Targets '/analyze' and '/chat/all-messages-for-all-threads'",
                        ],
                        "Summary": "Increments counters and logs memory snapshots around heavy chat and analysis routes.",
                    },
                    "Summary": "Observes request volume and memory usage for the heaviest endpoints.",
                },
                "Summary": "Adds cross-cutting request middleware like CORS, compression, throttling, and memory tracking.",
            },
            "Routes": {
                "Root (api/routes/root.py)": {
                    "Details": [
                        "@router.get('/') -> api_root() returns API metadata map",
                    ],
                    "Summary": "Provides a top-level metadata document that highlights available API route groups.",
                },
                "Health (api/routes/health.py)": {
                    "Details": [
                        "@router.get('/health') -> health_check() with memory stats",
                        "@router.get('/health/database') -> database_health_check()",
                        "@router.get('/health/memory') -> memory_health_check()",
                        "@router.get('/health/rate-limits') -> rate_limit_health_check()",
                        "@router.get('/health/prepared-statements') -> prepared_statements_health_check()",
                    ],
                    "Summary": "Surfaces system health metrics including memory, rate limits, and database connectivity checks.",
                },
                "Catalog (api/routes/catalog.py)": {
                    "Details": [
                        "@router.get('/catalog') -> get_catalog() paginates selection_descriptions",
                        "@router.get('/data-tables') -> get_data_tables() lists sqlite tables",
                        "@router.get('/data-table') -> get_data_table() returns columns/rows",
                    ],
                    "Summary": "Serves dataset catalog metadata and table previews sourced from local SQLite selections.",
                },
                "Analysis (api/routes/analysis.py)": {
                    "Details": [
                        "@router.post('/analyze') -> analyze() orchestrates analysis_main",
                        "Uses analysis_semaphore for concurrency control",
                        "register_execution()/unregister_execution() for cancellation",
                        "cancellable_analysis() polls is_cancelled() every 0.5s",
                        "Fallback to InMemorySaver when prepared statements fail",
                    ],
                    "Summary": "Runs the analysis pipeline with concurrency guards, cancellation hooks, and prepared statement fallbacks.",
                },
                "Feedback (api/routes/feedback.py)": {
                    "Details": [
                        "@router.post('/feedback') -> submit_feedback() stores LangSmith feedback",
                        "@router.post('/sentiment') -> submit_sentiment() updates run sentiment",
                    ],
                    "Summary": "Accepts explicit feedback and sentiment updates to feed LangSmith monitoring.",
                },
                "Chat (api/routes/chat.py)": {
                    "Details": [
                        "@router.get('/chat/{thread_id}/sentiments') -> get_thread_sentiments()",
                        "@router.get('/chat-threads') -> get_chat_threads() paginates threads",
                        "@router.delete('/chat/{thread_id}') -> delete_chat_thread()",
                        "@router.get('/chat/all-messages-for-one-thread/{thread_id}') -> get_all_messages_for_thread()",
                        "Helper: get_thread_messages_with_metadata() reused by other routes",
                    ],
                    "Summary": "Manages chat thread lifecycle, retrieval, sentiment insight, and deletion workflows.",
                },
                "Messages (api/routes/messages.py)": {
                    "Details": [
                        "@router.get('/chat/{thread_id}/messages') -> get_chat_messages()",
                        "@router.get('/chat/{thread_id}/run-ids') -> get_message_run_ids()",
                    ],
                    "Summary": "Provides per-thread message history along with the run identifiers that generated them.",
                },
                "Bulk (api/routes/bulk.py)": {
                    "Details": [
                        "@router.get('/chat/all-messages-for-all-threads') -> get_all_chat_messages() with caching",
                    ],
                    "Summary": "Returns cached multi-thread chat transcripts for operational support tooling.",
                },
                "Debug (api/routes/debug.py)": {
                    "Details": [
                        "@router.get('/debug/chat/{thread_id}/checkpoints') -> debug_checkpoints()",
                        "@router.get('/debug/pool-status') -> debug_pool_status()",
                        "@router.get('/debug/run-id/{run_id}') -> debug_run_id()",
                        "@router.post('/admin/clear-cache') -> clear_cache()",
                        "@router.post('/admin/clear-prepared-statements') -> clear_prepared_statements()",
                        "@router.post('/debug/set-env') and '/debug/reset-env'",
                    ],
                    "Summary": "Offers administrative diagnostics for checkpoints, connection pools, caches, and environment toggles.",
                },
                "Misc (api/routes/misc.py)": {
                    "Details": [
                        "@router.get('/placeholder/{width}/{height}') -> get_placeholder_image()",
                        "@router.get('/initial-followup-prompts') -> get_initial_followup_prompts()",
                    ],
                    "Summary": "Delivers helper utilities including placeholder imagery and default follow-up prompts.",
                },
                "Stop (api/routes/stop.py)": {
                    "Details": [
                        "StopExecutionRequest model with thread_id/run_id",
                        "@router.post('/stop-execution') -> stop_execution()",
                        "Invokes request_cancellation() and get_active_count()",
                    ],
                    "Summary": "Enables clients to request graceful cancellation of in-flight analysis runs.",
                },
                "Summary": "Exposes domain-specific HTTP endpoints for health monitoring, catalog browsing, analysis, chat operations, and administrative tooling.",
            },
            "Models": {
                "Requests (api/models/requests.py)": {
                    "AnalyzeRequest": {
                        "Details": [
                            "prompt: str (1-10000 chars) validated via field_validator",
                            "thread_id: str (1-100 chars) stripped and validated",
                            "run_id optional UUID string with format check",
                        ],
                        "Summary": "Validates prompt content, thread identifiers, and optional run IDs before triggering analysis.",
                    },
                    "FeedbackRequest": {
                        "Details": [
                            "run_id must be UUID string",
                            "feedback Optional[int] bounded between 0 and 1",
                            "comment Optional[str] trimmed and limited to 1000 chars",
                        ],
                        "Summary": "Enforces UUID and range constraints on user feedback payloads.",
                    },
                    "SentimentRequest": {
                        "Details": [
                            "run_id UUID validation",
                            "sentiment Optional[bool] used to set/clear sentiment",
                        ],
                        "Summary": "Validates sentiment toggles associated with an existing LangSmith run.",
                    },
                    "Summary": "Defines request payload schemas for analysis, feedback, and sentiment operations.",
                },
                "Responses (api/models/responses.py)": {
                    "Details": [
                        "ChatThreadResponse with latest_timestamp, run_count, title, full_prompt",
                        "PaginatedChatThreadsResponse for thread pagination metadata",
                        "ChatMessage schema with datasets_used, sql_query, followup_prompts",
                    ],
                    "Summary": "Shapes outbound chat responses and pagination metadata consumed by the frontend.",
                },
                "Summary": "Maintains Pydantic schemas that validate inbound payloads and structure outbound responses.",
            },
            "Dependencies": {
                "Auth (api/dependencies/auth.py)": {
                    "Details": [
                        "get_current_user() expects 'Authorization: Bearer <token>'",
                        "Splits header, validates format, extracts JWT",
                        "verify_google_jwt(token) performs signature check",
                        "Returns user_info dict (email) or raises HTTPException(401)",
                        "Logs via print__token_debug and log_comprehensive_error()",
                    ],
                    "Summary": "Enforces Google-signed JWT authentication before protected routes execute.",
                },
                "Summary": "Provides shared dependency hooks such as authentication for protected endpoints.",
            },
            "Utils": {
                "Cancellation (api/utils/cancellation.py)": {
                    "Details": [
                        "register_execution(thread_id, run_id) seeds registry",
                        "request_cancellation() flips 'cancelled' flag",
                        "is_cancelled() polled by cancellable_analysis()",
                        "unregister_execution() removes completed runs",
                        "cleanup_old_entries() purges stale entries",
                        "get_active_count() reports tracked executions",
                    ],
                    "Summary": "Tracks active runs and exposes cancellation primitives for long-running analyses.",
                },
                "Debug logging (api/utils/debug.py)": {
                    "Details": [
                        "print__analysis_tracing_debug()/print__analyze_debug()",
                        "print__debug(), print__startup_debug(), print__token_debug()",
                        "print__checkpointers_debug(), print__feedback_flow(), etc.",
                    ],
                    "Summary": "Provides targeted debugging print helpers to trace execution flows and data states.",
                },
                "Memory management (api/utils/memory.py)": {
                    "Details": [
                        "log_memory_usage(label) using psutil",
                        "setup_graceful_shutdown() signal handlers",
                        "start_memory_profiler()/stop_memory_profiler() with tracemalloc",
                        "start_memory_cleanup()/stop_memory_cleanup() background task",
                        "log_comprehensive_error(event, exc, request=None)",
                        "print__memory_monitoring() helper outputs",
                    ],
                    "Summary": "Offers logging, profiler control, and cleanup utilities around process memory usage.",
                },
                "Rate limiting (api/utils/rate_limiting.py)": {
                    "Details": [
                        "check_rate_limit_with_throttling(client_ip)",
                        "wait_for_rate_limit(client_ip) async retries",
                        "check_rate_limit(client_ip) simple boolean variant",
                    ],
                    "Summary": "Implements throttling primitives leveraged by the HTTP middleware layer.",
                },
                "Summary": "Collects reusable helpers that support cancellation, debugging, memory tracking, and throttling.",
            },
            "Exception Handlers (api/main.py)": {
                "RequestValidationError": {
                    "Details": [
                        "jsonable_encoder(exc.errors()) -> JSONResponse(422)",
                        "Fallback to simplified list when encoding fails",
                    ],
                    "Summary": "Formats validation errors consistently while keeping payloads JSON serializable.",
                },
                "StarletteHTTPException": {
                    "Details": [
                        "Detailed logging for 401 and other 4xx/5xx",
                        "Returns JSONResponse with exc.detail",
                    ],
                    "Summary": "Captures context for HTTP errors before relaying the status and detail to clients.",
                },
                "ValueError": {
                    "Details": [
                        "JSONResponse(status_code=400, detail=str(exc))",
                    ],
                    "Summary": "Maps explicit value parsing issues to clear 400 responses.",
                },
                "Exception": {
                    "Details": [
                        "Optional traceback when DEBUG_TRACEBACK=1",
                        "JSONResponse(status_code=500, detail='Internal server error')",
                    ],
                    "Summary": "Provides a fallback handler for unexpected errors with optional tracebacks for debugging.",
                },
                "Summary": "Customizes error responses and logging for validation issues, HTTP exceptions, and unexpected failures.",
            },
            "Summary": "Coordinates configuration, middleware, routes, data models, utilities, and persistence for the FastAPI backend.",
        },
        "2. LangGraph Orchestration": {
            "Entry Point (main.py)": {
                "Purpose": {
                    "Details": [
                        "Main orchestration function for CZSU multi-agent text-to-SQL analysis",
                        "Serves as central coordinator from NL question to formatted answer",
                        "Supports both CLI and API execution modes",
                        "Handles error recovery, memory management, and observability",
                    ],
                    "Summary": "Primary entry point that coordinates the entire analysis workflow using LangGraph.",
                },
                "Execution Modes": {
                    "CLI Mode": {
                        "Details": [
                            "Direct execution with command-line arguments",
                            "Optional --thread_id flag for continuing conversations",
                            "Optional --run_id flag for LangSmith tracing",
                            "Uses DEFAULT_PROMPT if none provided",
                        ],
                        "Summary": "Command-line interface for testing and development with argument parsing.",
                    },
                    "API Mode": {
                        "Details": [
                            "Called from FastAPI endpoints (api/main.py)",
                            "All parameters provided programmatically",
                            "Thread IDs managed by API for multi-user isolation",
                            "Checkpointer shared across requests for efficiency",
                        ],
                        "Summary": "Programmatic invocation from FastAPI with parameter-based control.",
                    },
                    "Summary": "Flexible execution supporting both interactive CLI and production API deployment.",
                },
                "Main Function Workflow": {
                    "1. Parameter Resolution": {
                        "Details": [
                            "Parse CLI args or use provided parameters",
                            "Generate thread_id if not provided (format: data_analysis_{8_hex})",
                            "Generate run_id for LangSmith tracing if needed",
                        ],
                        "Summary": "Resolves execution parameters from CLI or API call context.",
                    },
                    "2. Memory Baseline": {
                        "Details": [
                            "Record initial RSS memory usage via psutil",
                            "Force garbage collection for clean baseline",
                            "Enable leak detection through before/after comparison",
                        ],
                        "Summary": "Establishes memory monitoring baseline for leak detection.",
                    },
                    "3. Checkpointer Initialization": {
                        "Details": [
                            "Attempt PostgreSQL-based AsyncPostgresSaver",
                            "Fallback to InMemorySaver if PostgreSQL unavailable",
                            "Enables conversation memory and resumability",
                        ],
                        "Summary": "Initializes persistent or in-memory checkpointer for state management.",
                    },
                    "4. Graph Creation": {
                        "Details": [
                            "Initialize LangGraph workflow with checkpointer",
                            "Graph contains 17 nodes across 6 processing stages",
                            "Nodes: rewrite, retrieve, generate, reflect, format, save",
                        ],
                        "Summary": "Creates compiled LangGraph workflow with all nodes and edges.",
                    },
                    "5. State Preparation": {
                        "Details": [
                            "Check for existing conversation state",
                            "New: Initialize complete state with all 13 fields",
                            "Continuing: Reset iteration-specific fields only",
                            "Generate initial follow-up prompts for new conversations",
                        ],
                        "Summary": "Prepares input state based on conversation context (new vs continuing).",
                    },
                    "6. Graph Execution": {
                        "Details": [
                            "Invoke LangGraph with state and configuration",
                            "Automatic checkpointing after each node",
                            "Parallel retrieval of database + PDF sources",
                            "Iterative SQL generation with reflection (up to MAX_ITERATIONS)",
                        ],
                        "Summary": "Executes graph workflow with automatic state persistence and parallel processing.",
                    },
                    "7. Memory Monitoring": {
                        "Details": [
                            "Track memory growth during execution",
                            "Trigger emergency GC if exceeds GC_MEMORY_THRESHOLD (1900MB)",
                            "Log warnings for suspicious memory patterns",
                        ],
                        "Summary": "Monitors memory usage and triggers cleanup when thresholds exceeded.",
                    },
                    "8. Result Processing": {
                        "Details": [
                            "Extract final answer from messages list (last AI message)",
                            "Filter selection codes to only those used in queries",
                            "Serialize PDF chunks (Document objects) to JSON",
                            "Generate dataset URLs for frontend navigation",
                        ],
                        "Summary": "Processes graph output into API-friendly serialized format.",
                    },
                    "9. Final Cleanup": {
                        "Details": [
                            "Force final garbage collection",
                            "Monitor total memory retention",
                            "Log warnings if growth exceeds 100MB",
                            "Return serialized result",
                        ],
                        "Summary": "Performs final cleanup and returns structured result dictionary.",
                    },
                    "Summary": "Nine-step orchestration workflow from initialization to result delivery.",
                },
                "Helper Functions": {
                    "extract_table_names_from_sql()": {
                        "Details": [
                            "Parses SQL queries to identify table names",
                            "Handles FROM and JOIN clauses",
                            "Strips comments and normalizes whitespace",
                            "Returns deduplicated uppercase table names",
                        ],
                        "Summary": "Extracts dataset table names from SQL for attribution purposes.",
                    },
                    "get_used_selection_codes()": {
                        "Details": [
                            "Filters top_selection_codes to only those in queries",
                            "Parses all SQL queries via extract_table_names_from_sql()",
                            "Returns intersection of candidates and actually used codes",
                        ],
                        "Summary": "Identifies which dataset selection codes were actually queried.",
                    },
                    "generate_initial_followup_prompts()": {
                        "Details": [
                            "Generates 5 diverse starter suggestions for new conversations",
                            "Uses timestamp-based pseudo-random seed",
                            "Template-based with topic pools (regions, metrics, topics)",
                            "Ensures variety across conversation starts",
                        ],
                        "Summary": "Creates dynamic follow-up prompt suggestions for user guidance.",
                    },
                    "Summary": "Utility functions supporting SQL parsing, dataset filtering, and prompt generation.",
                },
                "Error Handling": {
                    "Retry Decorators": {
                        "Details": [
                            "@retry_on_ssl_connection_error(max_retries=3)",
                            "@retry_on_prepared_statement_error(max_retries=3)",
                            "Exponential backoff between attempts",
                        ],
                        "Summary": "Automatic retry logic for transient PostgreSQL connection errors.",
                    },
                    "Summary": "Comprehensive error handling with retry mechanisms for database issues.",
                },
                "Configuration": {
                    "Environment Variables": {
                        "Details": [
                            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",
                            "POSTGRES_CONNECTION_STRING for checkpointer",
                            "GC_MEMORY_THRESHOLD (default: 1900MB)",
                            "MAX_QUERIES_LIMIT_FOR_REFLECT (default: 10)",
                        ],
                        "Summary": "Environment-driven configuration for Azure, PostgreSQL, and memory limits.",
                    },
                    "Constants": {
                        "Details": [
                            "DEFAULT_PROMPT for testing/development",
                            "MAX_ITERATIONS from nodes.py (default: 1)",
                        ],
                        "Summary": "Hardcoded constants for workflow behavior and testing.",
                    },
                    "Summary": "Configuration via environment variables and constants.",
                },
                "Windows Compatibility": {
                    "Details": [
                        "asyncio.WindowsSelectorEventLoopPolicy set on win32",
                        "Ensures psycopg3 compatibility on Windows",
                        "Applied before any async operations",
                    ],
                    "Summary": "Special handling for Windows async event loop compatibility.",
                },
                "Summary": "Main entry point orchestrating LangGraph execution with memory management, error handling, and dual execution modes.",
            },
            "Graph Definition (my_agent/agent.py)": {
                "Purpose": {
                    "Details": [
                        "Defines LangGraph StateGraph structure and execution flow",
                        "Orchestrates multi-agent text-to-SQL analysis workflow",
                        "Manages state transitions and conditional routing",
                        "Implements checkpointing for workflow resumability",
                    ],
                    "Summary": "Core graph architecture defining node connections and workflow execution logic.",
                },
                "Graph Architecture": {
                    "Phase 1: Query Preprocessing": {
                        "Details": [
                            "START → rewrite_prompt → summarize_messages_rewrite",
                            "Converts conversational questions to standalone queries",
                            "Summarizes conversation history for token management",
                            "Prepares optimized query for parallel retrieval",
                        ],
                        "Summary": "Rewrites user query and summarizes conversation context.",
                    },
                    "Phase 2: Parallel Retrieval": {
                        "Branch A - Database Selections": {
                            "Details": [
                                "retrieve_similar_selections_hybrid_search",
                                "rerank_table_descriptions (Cohere reranking)",
                                "relevant_selections (top-k filtering)",
                            ],
                            "Summary": "Hybrid search + reranking for dataset selection retrieval.",
                        },
                        "Branch B - PDF Documentation": {
                            "Details": [
                                "retrieve_similar_chunks_hybrid_search",
                                "rerank_chunks (Cohere reranking)",
                                "relevant_chunks (threshold filtering)",
                            ],
                            "Summary": "Hybrid search + reranking for PDF chunk retrieval.",
                        },
                        "Summary": "Dual parallel branches for database and PDF retrieval with reranking.",
                    },
                    "Phase 3: Synchronization & Routing": {
                        "Details": [
                            "[relevant_selections, relevant_chunks] → post_retrieval_sync",
                            "route_after_sync() determines next step:",
                            "IF top_selection_codes → get_schema (database queries)",
                            "ELIF chromadb_missing → END (error)",
                            "ELSE → format_answer (PDF-only response)",
                        ],
                        "Summary": "Synchronizes parallel branches and routes based on available data sources.",
                    },
                    "Phase 4: Query Loop": {
                        "Details": [
                            "get_schema → generate_query → summarize_messages_query",
                            "route_after_query() controls iteration:",
                            "IF iteration < MAX_ITERATIONS → reflect",
                            "ELSE → format_answer (force answer)",
                            "Reflection cycle: reflect → summarize_messages_reflect",
                            "route_after_reflect() decision:",
                            "IF decision == 'improve' → generate_query (loop back)",
                            "ELIF decision == 'answer' → format_answer",
                        ],
                        "Summary": "Iterative SQL generation with optional reflection for improvement.",
                    },
                    "Phase 5: Answer Finalization": {
                        "Details": [
                            "format_answer → summarize_messages_format",
                            "generate_followup_prompts → submit_final_answer",
                            "save → cleanup_resources → END",
                            "Synthesizes SQL results + PDF chunks",
                            "Generates followup prompts",
                        ],
                        "Summary": "Formats final answer, generates suggestions, saves, and cleans up.",
                    },
                    "Summary": "Five-phase workflow with parallel retrieval, conditional routing, and iterative improvement.",
                },
                "Node Summary": {
                    "Details": [
                        "Total: 17 nodes across 6 functional categories",
                        "Preprocessing: rewrite_prompt",
                        "Retrieval: 2 hybrid search nodes (selections, chunks)",
                        "Reranking: 2 rerank nodes (table_descriptions, chunks)",
                        "Filtering: 2 relevant nodes (selections, chunks)",
                        "Synchronization: post_retrieval_sync",
                        "Query: get_schema, generate_query",
                        "Reflection: reflect",
                        "Formatting: format_answer, generate_followup_prompts, submit_final_answer",
                        "Persistence: save, cleanup_resources",
                        "Memory: 4 summarize_messages instances (rewrite/query/reflect/format)",
                    ],
                    "Summary": "17 specialized nodes handling retrieval, generation, reflection, and formatting.",
                },
                "Conditional Routing": {
                    "route_after_sync()": {
                        "Details": [
                            "Routes based on available data sources",
                            "Checks top_selection_codes for database availability",
                            "Checks chromadb_missing flag for errors",
                            "Returns: 'get_schema' | 'format_answer' | END",
                        ],
                        "Summary": "Post-retrieval routing based on data source availability.",
                    },
                    "route_after_query()": {
                        "Details": [
                            "Controls reflection loop iteration",
                            "Checks iteration count vs MAX_ITERATIONS",
                            "Returns: 'reflect' | 'format_answer'",
                        ],
                        "Summary": "Iteration control preventing infinite reflection loops.",
                    },
                    "route_after_reflect()": {
                        "Details": [
                            "Routes based on reflection decision",
                            "Checks reflection_decision: 'improve' | 'answer'",
                            "Returns: 'generate_query' | 'format_answer'",
                        ],
                        "Summary": "Reflection-based routing for query improvement or answer finalization.",
                    },
                    "Summary": "Three routing functions externalized to my_agent/utils/routers.py for clarity.",
                },
                "create_graph() Function": {
                    "Details": [
                        "Main factory function for graph instantiation",
                        "Accepts optional checkpointer (defaults to InMemorySaver)",
                        "Initializes StateGraph(DataAnalysisState)",
                        "Adds all 17 nodes",
                        "Defines edges (sequential and conditional)",
                        "Compiles graph with checkpointer",
                        "Returns compiled graph ready for execution",
                    ],
                    "Summary": "Factory function that constructs and compiles the complete workflow graph.",
                },
                "Design Principles": {
                    "Details": [
                        "Parallel execution for efficiency (dual retrieval)",
                        "Conditional routing based on available data",
                        "Controlled iteration with MAX_ITERATIONS prevention",
                        "State checkpointing for workflow resumption",
                        "Token management via automatic message summarization",
                        "Resource cleanup for connection management",
                        "Modular architecture with separated routing logic",
                        "Absolute imports with my_agent prefix",
                    ],
                    "Summary": "Eight key design principles ensuring efficiency, safety, and maintainability.",
                },
                "Summary": "Graph definition module implementing multi-phase workflow with parallel retrieval, conditional routing, and state management.",
            },
            "State Management (my_agent/utils/state.py)": {
                "Purpose": {
                    "Details": [
                        "Defines state schema for LangGraph workflow",
                        "Implements TypedDict-based state structure",
                        "Manages 15 fields tracking data flow through system",
                        "Provides custom reducers for memory efficiency",
                    ],
                    "Summary": "State schema and reducers for workflow data management.",
                },
                "DataAnalysisState TypedDict": {
                    "Input Fields": {
                        "Details": [
                            "prompt: str - Original user question",
                        ],
                        "Summary": "User-provided input field.",
                    },
                    "Query Processing": {
                        "Details": [
                            "rewritten_prompt: str - Standalone search-optimized question",
                        ],
                        "Summary": "Processed query for retrieval optimization.",
                    },
                    "Conversation Management": {
                        "Details": [
                            "messages: List[BaseMessage] - Always [summary, last_message]",
                            "Uses default replacement behavior",
                            "Summarization in summarize_messages_node before replacement",
                        ],
                        "Summary": "Token-efficient message storage with 2-message structure.",
                    },
                    "Workflow Control": {
                        "Details": [
                            "iteration: int - Loop counter (default: 0)",
                            "reflection_decision: str - 'improve' | 'answer'",
                            "chromadb_missing: bool - Error flag",
                        ],
                        "Summary": "Control fields for iteration tracking and error flagging.",
                    },
                    "Database Selection Retrieval": {
                        "Details": [
                            "hybrid_search_results: List[Document] - Raw search results",
                            "most_similar_selections: List[Tuple[str, float]] - Reranked with scores",
                            "top_selection_codes: List[str] - Final top-k codes",
                        ],
                        "Summary": "Three-stage database selection retrieval state.",
                    },
                    "PDF Chunk Retrieval": {
                        "Details": [
                            "hybrid_search_chunks: List[Document] - Raw PDF search results",
                            "most_similar_chunks: List[Tuple[Document, float]] - Reranked chunks",
                            "top_chunks: List[Document] - Filtered above threshold",
                        ],
                        "Summary": "Three-stage PDF chunk retrieval state.",
                    },
                    "Query Execution": {
                        "Details": [
                            "queries_and_results: Annotated[List[Tuple[str, str]], limited_queries_reducer]",
                            "List of (SQL_query, result_string) tuples",
                            "Custom reducer limits to recent N entries",
                        ],
                        "Summary": "Query history with custom limiting reducer.",
                    },
                    "Output": {
                        "Details": [
                            "final_answer: str - Formatted answer synthesizing all sources",
                            "followup_prompts: List[str] - Suggested follow-up questions",
                        ],
                        "Summary": "Final output fields for answer and suggestions.",
                    },
                    "Summary": "15-field TypedDict organizing workflow state across input, processing, retrieval, execution, and output stages.",
                },
                "Custom Reducers": {
                    "limited_queries_reducer": {
                        "Details": [
                            "Limits queries_and_results to latest N entries",
                            "Default: MAX_QUERIES_LIMIT_FOR_REFLECT = 10",
                            "Prevents memory/token overflow in reflection loops",
                            "Combines existing (left) + new (right) queries",
                            "Returns combined[-MAX_QUERIES_LIMIT_FOR_REFLECT:]",
                        ],
                        "Summary": "Custom reducer preventing query history overflow by limiting to recent entries.",
                    },
                    "Summary": "Single custom reducer for query history management.",
                },
                "State Lifecycle": {
                    "Details": [
                        "Initial: prompt, messages=[], iteration=0",
                        "After rewrite: rewritten_prompt populated",
                        "After retrieval: hybrid_search_results, hybrid_search_chunks populated",
                        "After reranking: most_similar_selections, most_similar_chunks populated",
                        "After filtering: top_selection_codes, top_chunks populated",
                        "After query: queries_and_results appended",
                        "After reflection: reflection_decision, iteration updated",
                        "After formatting: final_answer, followup_prompts populated",
                    ],
                    "Summary": "State evolves through eight stages from input to formatted output.",
                },
                "Memory Efficiency Features": {
                    "Details": [
                        "Always 2 messages: [summary, last_message]",
                        "Limited query history via custom reducer",
                        "Automatic summarization prevents token overflow",
                        "Bounded memory usage in long conversations",
                    ],
                    "Summary": "Multiple strategies ensuring memory-efficient state management.",
                },
                "Summary": "TypedDict-based state schema with 15 fields and custom reducers for memory-efficient workflow tracking.",
            },
            "Routing Logic (my_agent/utils/routers.py)": {
                "Purpose": {
                    "Details": [
                        "Contains all conditional routing logic",
                        "Determines next node based on current state",
                        "Externalized from agent.py for modularity",
                        "Three routing functions for three decision points",
                    ],
                    "Summary": "Dedicated module for workflow routing decisions.",
                },
                "Routing Functions": {
                    "route_after_sync()": {
                        "Details": [
                            "Called after parallel retrieval synchronization",
                            "Checks state['top_selection_codes'] for database availability",
                            "Returns 'get_schema' if selections found",
                            "Returns END if chromadb_missing flag set",
                            "Returns 'format_answer' if only PDF chunks available",
                        ],
                        "Summary": "Routes based on available data sources after retrieval.",
                    },
                    "route_after_query()": {
                        "Details": [
                            "Called after query generation",
                            "Checks state['iteration'] vs MAX_ITERATIONS",
                            "Returns 'reflect' if iteration < MAX_ITERATIONS",
                            "Returns 'format_answer' if max iterations reached",
                        ],
                        "Summary": "Controls reflection loop iteration limit.",
                    },
                    "route_after_reflect()": {
                        "Details": [
                            "Called after reflection node",
                            "Checks state['reflection_decision']",
                            "Returns 'generate_query' if decision == 'improve'",
                            "Returns 'format_answer' if decision == 'answer'",
                        ],
                        "Summary": "Routes based on reflection decision to improve or answer.",
                    },
                    "Summary": "Three routing functions for post-sync, post-query, and post-reflect decisions.",
                },
                "Summary": "Modular routing logic determining workflow paths based on state conditions.",
            },
            "LangGraph Configuration (langgraph.json)": {
                "Details": [
                    "project_name: 'data_analysis_langgraph'",
                    "version: '0.1.0'",
                    "entry_point: 'main.py'",
                    "python_version: '>=3.9,<3.12'",
                    "requirements_file: 'requirements.txt'",
                    "graphs.agent: 'my_agent.agent:create_graph'",
                ],
                "Summary": "LangGraph project configuration specifying entry point and graph location.",
            },
            "Supporting Utilities": {
                "Models (my_agent/utils/models.py)": {
                    "Azure OpenAI Models": {
                        "Details": [
                            "get_azure_llm_gpt_4o(temperature=0.0)",
                            "get_azure_llm_gpt_4o_mini(temperature=0.0)",
                            "get_azure_llm_gpt_4o_4_1(temperature=0.0)",
                            "Test variants with _test suffix",
                        ],
                        "Summary": "Factory functions for Azure OpenAI GPT models with configurable temperature.",
                    },
                    "Ollama Models": {
                        "Details": [
                            "get_ollama_llm() for local LLM usage",
                            "get_ollama_llm_test() test variant",
                        ],
                        "Summary": "Local Ollama model factory functions.",
                    },
                    "Embedding Models": {
                        "Details": [
                            "get_azure_embedding_model()",
                            "get_langchain_azure_embedding_model()",
                            "Test variants available",
                        ],
                        "Summary": "Azure embedding model factory functions for vector search.",
                    },
                    "Summary": "LLM and embedding model factory functions for Azure and Ollama.",
                },
                "Tools (my_agent/utils/tools.py)": {
                    "Details": [
                        "finish_gathering() - Signals query gathering completion",
                        "LocalSQLiteQueryInput - Pydantic model for tool input",
                        "LocalSQLiteQueryTool - BaseTool for local SQLite queries",
                        "get_sqlite_tools() - Returns list of available tools",
                    ],
                    "Summary": "MCP-compatible tools for SQL query execution against local SQLite.",
                },
                "Helpers (my_agent/utils/helpers.py)": {
                    "Details": [
                        "load_schema(state) - Loads SQLite schema for selections",
                        "translate_to_english(text) - Czech to English translation",
                        "detect_language(text) - Language detection (Czech/English)",
                    ],
                    "Summary": "Helper functions for schema loading, translation, and language detection.",
                },
                "Summary": "Supporting utilities for models, tools, and helper functions used by graph nodes.",
            },
            "Server Startup (uvicorn_start.py)": {
                "Details": [
                    "Imports FastAPI app from api.main",
                    "Sets WindowsSelectorEventLoopPolicy for Windows",
                    "Loads environment variables early via load_dotenv()",
                    "Runs uvicorn with reload enabled",
                    "Watches api/ and my_agent/ directories",
                    "Listens on 0.0.0.0:8000",
                ],
                "Summary": "Uvicorn startup script for development server with hot reload.",
            },
            "Summary": "LangGraph orchestration layer coordinating workflow execution, state management, and conditional routing for multi-agent text-to-SQL analysis.",
        },
        "3. External Integrations": {
            "Checkpointer Integration": {
                "Factory (checkpointer/checkpointer/factory.py)": {
                    "Details": [
                        "initialize_checkpointer() bootstraps AsyncPostgresSaver",
                        "cleanup_checkpointer() closes pools",
                        "get_global_checkpointer() caches saver instance",
                        "create_async_postgres_saver() handles retries + pooling",
                        "Fallback to MemorySaver when Postgres unavailable",
                    ],
                    "Summary": "Initializes and manages AsyncPostgresSaver with retry logic and fallback.",
                },
                "Checkpoint Utilities": {
                    "Details": [
                        "checkpointer.user_management.thread_operations.create_thread_run_entry()",
                        "checkpointer.user_management.sentiment_tracking.update_thread_run_sentiment()",
                        "checkpointer.database.connection.get_direct_connection()",
                    ],
                    "Summary": "Helper operations for thread runs, sentiments, and database access.",
                },
                "Summary": "LangGraph persistence layer using PostgreSQL (referenced, detailed mapping later).",
            },
            "Data Stores": {
                "SQLite Databases": {
                    "Details": [
                        "data/czsu_data.db - Main CZSU statistical tables",
                        "metadata/llm_selection_descriptions/selection_descriptions.db",
                    ],
                    "Summary": "Local SQLite databases for CZSU data and selection metadata.",
                },
                "Vector Stores": {
                    "Details": [
                        "metadata/czsu_chromadb/ - ChromaDB for hybrid search",
                        "Stores dataset descriptions and PDF chunks",
                        "Supports semantic + BM25 hybrid search",
                    ],
                    "Summary": "ChromaDB vector store for retrieval-augmented generation.",
                },
                "Summary": "Persistent datasets and vector assets powering catalog and retrieval.",
            },
            "LangSmith Tracing": {
                "Details": [
                    "Run ID tracking for observability",
                    "Thread ID for conversation tracking",
                    "Configuration metadata capture",
                    "Input/output logging for debugging",
                    "Feedback and sentiment tracking integration",
                ],
                "Summary": "Observability platform for LLM call tracing and debugging.",
            },
            "Azure OpenAI": {
                "Details": [
                    "AZURE_OPENAI_API_KEY for authentication",
                    "AZURE_OPENAI_ENDPOINT for API calls",
                    "GPT-4o, GPT-4o-mini model support",
                    "text-embedding-3-large for embeddings",
                ],
                "Summary": "Azure OpenAI service integration for LLM and embedding calls.",
            },
            "Cohere Reranking": {
                "Details": [
                    "Reranks database selection search results",
                    "Reranks PDF chunk search results",
                    "Provides relevance scores for filtering",
                ],
                "Summary": "Cohere API for reranking hybrid search results.",
            },
            "Summary": "External service integrations including checkpointer, data stores, tracing, LLM, and reranking.",
        },
        "Summary": "Comprehensive backend architecture spanning FastAPI API layer, LangGraph orchestration, and external integrations.",
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

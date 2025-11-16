import os

from graphviz import Digraph

# Define the mindmap structure for Checkpointer subsystem
mindmap = {
    "CZSU Checkpointer Subsystem": {
        "1. Core Architecture": {
            "Purpose": {
                "Details": [
                    "Provides persistent conversation memory using PostgreSQL",
                    "Implements LangGraph's AsyncPostgresSaver for state persistence",
                    "Enables multi-turn dialogues with context preservation",
                    "Manages user session tracking and thread ownership",
                    "Fallback to InMemorySaver when PostgreSQL unavailable",
                ],
                "Summary": "PostgreSQL-based checkpointing for LangGraph conversation state persistence and user session management.",
            },
            "Integration Points": {
                "main.py Integration": {
                    "Details": [
                        "Lines 40-47: Checkpointer initialization via get_global_checkpointer()",
                        "Line 46: Graph creation with checkpointer parameter",
                        "Line 49: Thread ID-based configuration for state retrieval",
                        "Lines 50-54: Existing state detection for conversation continuation",
                        "Line 58: Graph execution with checkpointer for automatic state saving",
                    ],
                    "Summary": "Main.py uses checkpointer for conversation state management and graph execution.",
                },
                "State Management": {
                    "Details": [
                        "New conversations: Full state initialization with 13 fields",
                        "Continuing conversations: Partial state update with reset fields",
                        "Thread-based isolation for concurrent users",
                        "Automatic checkpointing after each LangGraph node",
                        "Message summarization to prevent token overflow",
                    ],
                    "Summary": "Sophisticated state management with new/continuing conversation handling.",
                },
                "Summary": "Integrates with main.py for graph execution and state persistence across conversation turns.",
            },
            "Summary": "PostgreSQL-based persistent memory system for multi-turn conversations with user session tracking.",
        },
        "2. Configuration Layer (config.py)": {
            "Connection Timeouts": {
                "Details": [
                    "CONNECT_TIMEOUT = 90 (initial connection establishment)",
                    "TCP_USER_TIMEOUT = 240000ms (TCP-level timeout for concurrent ops)",
                    "KEEPALIVES_IDLE = 300 (5 min before first keepalive)",
                    "KEEPALIVES_INTERVAL = 30 (30s between keepalives)",
                    "KEEPALIVES_COUNT = 3 (failed keepalives before disconnect)",
                ],
                "Summary": "Cloud-optimized timeouts and keepalive settings for reliable connectivity.",
            },
            "Connection Pool Settings": {
                "Details": [
                    "DEFAULT_POOL_MIN_SIZE = 5 (increased for higher concurrency)",
                    "DEFAULT_POOL_MAX_SIZE = 25 (supports more concurrent connections)",
                    "DEFAULT_POOL_TIMEOUT = 180 (3 min pool timeout for concurrent testing)",
                    "DEFAULT_MAX_IDLE = 600 (10 min idle timeout for long operations)",
                    "DEFAULT_MAX_LIFETIME = 3600 (60 min max connection lifetime)",
                ],
                "Summary": "Aggressive pool sizing optimized for high concurrency and cloud databases.",
            },
            "Retry Configuration": {
                "Details": [
                    "DEFAULT_MAX_RETRIES = 2 (standard retry attempts)",
                    "CHECKPOINTER_CREATION_MAX_RETRIES = 2 (creation-specific retries)",
                    "Exponential backoff for SSL connection errors (2^attempt, max 30s)",
                    "Prepared statement cleanup on retry",
                    "Global checkpointer recreation on persistent failures",
                ],
                "Summary": "Comprehensive retry strategy for error recovery with exponential backoff.",
            },
            "Display and UI Constants": {
                "Details": [
                    "USER_MESSAGE_PREVIEW_LENGTH = 50 (log preview length)",
                    "AI_MESSAGE_PREVIEW_LENGTH = 100 (AI response preview)",
                    "THREAD_TITLE_MAX_LENGTH = 47 (thread title truncation)",
                    "MAX_DEBUG_MESSAGES_DETAILED = 6 (detailed message logging limit)",
                    "DEBUG_CHECKPOINT_LOG_INTERVAL = 5 (log every Nth checkpoint)",
                ],
                "Summary": "UI and logging display constants for readability and performance.",
            },
            "Configuration Functions": {
                "get_db_config()": {
                    "Details": [
                        "Extracts PostgreSQL config from environment variables",
                        "Returns dict: user, password, host, port, dbname",
                        "Port defaults to 5432 if not provided",
                        "Provides debug logging for verification",
                        "Used by connection string and pool creation",
                    ],
                    "Summary": "Centralized database configuration extraction from environment.",
                },
                "check_postgres_env_vars()": {
                    "Details": [
                        "Validates all required env vars: host, port, dbname, user, password",
                        "Returns bool: True if complete, False if missing",
                        "Reports missing variables for troubleshooting",
                        "Used during initialization to fail fast on misconfiguration",
                        "Supports automated deployment validation",
                    ],
                    "Summary": "Environment variable validation for fail-fast configuration checking.",
                },
                "Summary": "Configuration helper functions for database setup and validation.",
            },
            "Summary": "Centralized configuration with cloud-optimized timeouts, aggressive pooling, and comprehensive retry settings.",
        },
        "3. Global State (globals.py)": {
            "Global Variables": {
                "Details": [
                    "_GLOBAL_CHECKPOINTER = None (AsyncPostgresSaver instance or None)",
                    "_CONNECTION_STRING_CACHE = None (cached connection string)",
                    "_CHECKPOINTER_INIT_LOCK = None (async lock for initialization)",
                    "Lazy lock initialization to avoid event loop issues",
                    "Thread-safe access via double-checked locking pattern",
                ],
                "Summary": "Global state management for shared checkpointer instance and connection caching.",
            },
            "Type Definitions": {
                "Details": [
                    "TypeVar definitions for generic typing",
                    "BASE_DIR calculation for project root",
                    "Handles both normal execution and special cases (Jupyter, REPL)",
                    "Type hints for better code documentation",
                ],
                "Summary": "Type definitions and base directory configuration for module organization.",
            },
            "Summary": "Global state variables and type definitions for checkpointer instance management.",
        },
        "4. Database Layer": {
            "connection.py": {
                "get_connection_string()": {
                    "Details": [
                        "Generates PostgreSQL connection string with cloud optimization",
                        "Unique app name: czsu_langgraph_{pid}_{thread}_{time}_{random}",
                        "SSL mode required for secure cloud connections",
                        "Includes timeout and keepalive parameters",
                        "Global caching to ensure consistent app names",
                    ],
                    "Summary": "Cloud-optimized connection string generation with unique application naming.",
                },
                "get_connection_kwargs()": {
                    "Details": [
                        "autocommit=False for better cloud database compatibility",
                        "prepare_threshold=None to disable prepared statements",
                        "Prevents prepared statement conflicts and memory leaks",
                        "Based on LangGraph docs and cloud best practices",
                        "Used by all connection creation for consistency",
                    ],
                    "Summary": "Standardized connection parameters optimized for cloud databases and concurrent load.",
                },
                "check_connection_health()": {
                    "Details": [
                        "Performs SELECT 1 query to verify connection health",
                        "Used as callback for psycopg connection pools",
                        "Catches all exceptions to avoid breaking pool operations",
                        "Prevents 'SSL connection closed unexpectedly' errors",
                        "Essential for long-running apps with concurrent access",
                    ],
                    "Summary": "Connection health verification for pool management and error prevention.",
                },
                "get_direct_connection()": {
                    "Details": [
                        "Async context manager for direct database connections",
                        "Used for users_threads_runs table operations",
                        "Separate from AsyncPostgresSaver pool",
                        "Ensures connection cleanup via context management",
                        "Applies standard connection kwargs",
                    ],
                    "Summary": "Direct connection context manager for custom table operations.",
                },
                "Summary": "Connection string generation, health checking, and direct connection management.",
            },
            "pool_manager.py": {
                "cleanup_all_pools()": {
                    "Details": [
                        "Gracefully closes AsyncPostgresSaver pool",
                        "Resets _GLOBAL_CHECKPOINTER to None",
                        "Forces garbage collection for memory cleanup",
                        "Handles cleanup errors gracefully",
                        "Used during error recovery and shutdown",
                    ],
                    "Summary": "Graceful connection pool cleanup with proper resource deallocation.",
                },
                "force_close_modern_pools()": {
                    "Details": [
                        "Aggressive cleanup for troubleshooting scenarios",
                        "Calls cleanup_all_pools() then additional cleanup",
                        "Clears _CONNECTION_STRING_CACHE to force regeneration",
                        "Resets global state for clean restart",
                        "More thorough than standard cleanup",
                    ],
                    "Summary": "Aggressive pool cleanup for error recovery and restart scenarios.",
                },
                "modern_psycopg_pool()": {
                    "Details": [
                        "Async context manager for psycopg connection pools",
                        "Uses recommended approach to avoid deprecation warnings",
                        "open=False to explicitly control pool lifecycle",
                        "Automatic pool closure via context management",
                        "Applies all pool configuration constants",
                    ],
                    "Summary": "Modern context manager approach for connection pool management.",
                },
                "Summary": "Connection pool lifecycle management with graceful and aggressive cleanup strategies.",
            },
            "table_setup.py": {
                "setup_checkpointer_with_autocommit()": {
                    "Details": [
                        "Creates LangGraph checkpoint tables via AsyncPostgresSaver.setup()",
                        "Uses dedicated autocommit connection to avoid transaction conflicts",
                        "Prevents 'CREATE INDEX CONCURRENTLY cannot run inside transaction' errors",
                        "Creates temporary connection pool for setup",
                        "Idempotent - safe to call multiple times",
                    ],
                    "Summary": "LangGraph checkpoint table creation with autocommit to avoid transaction conflicts.",
                },
                "setup_users_threads_runs_table()": {
                    "Details": [
                        "Creates custom application table for user session tracking",
                        "Schema: id, email, thread_id, run_id, prompt, timestamp, sentiment",
                        "Indexes: email, thread_id, email+thread_id composite",
                        "Unique constraint on run_id for deduplication",
                        "Enables conversation ownership and sentiment tracking",
                    ],
                    "Summary": "Custom table creation for user-thread associations and conversation metadata.",
                },
                "table_exists()": {
                    "Details": [
                        "Checks information_schema.tables for table existence",
                        "Returns bool for conditional table creation",
                        "Used to skip setup if tables already exist",
                        "Prevents duplicate creation attempts",
                    ],
                    "Summary": "Table existence verification for idempotent setup operations.",
                },
                "Summary": "Database schema setup for both LangGraph and custom application tables.",
            },
            "Summary": "Complete database layer: connections, pools, and table management.",
        },
        "5. Error Handling Layer": {
            "prepared_statements.py": {
                "is_prepared_statement_error()": {
                    "Details": [
                        "Detects prepared statement conflicts via error message patterns",
                        "Patterns: 'prepared statement', 'does not exist', '_pg3_', '_pg_'",
                        "Case-insensitive pattern matching",
                        "Covers psycopg2 and psycopg3 naming conventions",
                        "Used by retry decorators for conditional retry",
                    ],
                    "Summary": "Error pattern detection for prepared statement conflicts.",
                },
                "clear_prepared_statements()": {
                    "Details": [
                        "Queries pg_prepared_statements system catalog",
                        "Deallocates all _pg3_* and _pg_* prepared statements",
                        "Uses unique cleanup connection to avoid conflicts",
                        "Limits detailed logging to first 3 statements",
                        "Non-fatal - continues on individual statement failures",
                    ],
                    "Summary": "Comprehensive prepared statement cleanup for error recovery.",
                },
                "Summary": "Prepared statement conflict detection and cleanup mechanisms.",
            },
            "retry_decorators.py": {
                "is_ssl_connection_error()": {
                    "Details": [
                        "Detects SSL connection failures via error patterns",
                        "Patterns: 'ssl connection closed', 'connection reset', 'broken pipe'",
                        "Checks both error message and error type",
                        "Covers various SSL-related error scenarios",
                        "Used by SSL retry decorator for conditional retry",
                    ],
                    "Summary": "SSL connection error detection for retry logic.",
                },
                "retry_on_ssl_connection_error()": {
                    "Details": [
                        "Decorator for automatic SSL error retry with exponential backoff",
                        "Default max_retries=3 with delay: min(2^attempt, 30s)",
                        "Closes and recreates connection pool on SSL errors",
                        "Clears global checkpointer state for fresh connections",
                        "Only retries on confirmed SSL connection errors",
                    ],
                    "Summary": "SSL connection error recovery with pool recreation and exponential backoff.",
                },
                "retry_on_prepared_statement_error()": {
                    "Details": [
                        "Decorator for automatic prepared statement error retry",
                        "Default max_retries=2 with prepared statement cleanup",
                        "Calls clear_prepared_statements() before retry",
                        "Recreates global checkpointer on persistent errors",
                        "Includes full traceback logging for debugging",
                    ],
                    "Summary": "Prepared statement error recovery with cleanup and checkpointer recreation.",
                },
                "Summary": "Retry decorators for SSL and prepared statement error recovery with comprehensive logging.",
            },
            "Summary": "Robust error handling with automatic retry, cleanup, and recovery mechanisms.",
        },
        "6. Checkpointer Management": {
            "factory.py": {
                "create_async_postgres_saver()": {
                    "Details": [
                        "Creates AsyncPostgresSaver with connection pool approach",
                        "Clears existing state first to avoid conflicts",
                        "Opens pool with health checking via check_connection_health",
                        "Conditionally runs setup based on table existence",
                        "Tests checkpointer with 'setup_test' thread",
                        "Decorated with @retry_on_ssl_connection_error and @retry_on_prepared_statement_error",
                    ],
                    "Summary": "Robust AsyncPostgresSaver creation with retry logic and health checking.",
                },
                "close_async_postgres_saver()": {
                    "Details": [
                        "Closes AsyncPostgresSaver connection pool",
                        "Clears _GLOBAL_CHECKPOINTER to None",
                        "Handles pool closing errors gracefully",
                        "Used during cleanup and recreation scenarios",
                    ],
                    "Summary": "AsyncPostgresSaver cleanup and resource deallocation.",
                },
                "get_global_checkpointer()": {
                    "Details": [
                        "Unified access point for global checkpointer instance",
                        "Lazy initialization with async lock for thread safety",
                        "Double-checked locking pattern to prevent race conditions",
                        "Runs health check before returning checkpointer",
                        "Decorated with retry decorators for error recovery",
                    ],
                    "Summary": "Thread-safe singleton accessor with lazy initialization and health checking.",
                },
                "initialize_checkpointer()": {
                    "Details": [
                        "Application startup initialization function",
                        "Creates checkpointer using create_async_postgres_saver()",
                        "Falls back to InMemorySaver on PostgreSQL failure",
                        "Provides comprehensive debug logging",
                        "Called during FastAPI application startup",
                    ],
                    "Summary": "Application startup checkpointer initialization with fallback mechanism.",
                },
                "cleanup_checkpointer()": {
                    "Details": [
                        "Application shutdown cleanup function",
                        "Uses force_close_modern_pools() for thorough cleanup",
                        "Handles AsyncPostgresSaver and other types differently",
                        "Clears global state on shutdown",
                        "Called during FastAPI application shutdown",
                    ],
                    "Summary": "Application shutdown checkpointer cleanup with comprehensive resource release.",
                },
                "Summary": "Complete checkpointer lifecycle: creation, access, initialization, and cleanup.",
            },
            "health.py": {
                "check_pool_health_and_recreate()": {
                    "Details": [
                        "Acquires connection from pool with 10s timeout",
                        "Runs SELECT 1 health check query",
                        "Detects SSL/connection issues via error patterns",
                        "Forces pool closure and checkpointer recreation on failure",
                        "Returns bool: True if healthy, False if recreated",
                    ],
                    "Summary": "Pool health monitoring with automatic recreation on SSL/connection failures.",
                },
                "Summary": "Connection pool health monitoring and automatic recovery.",
            },
            "Summary": "Comprehensive checkpointer lifecycle management with health monitoring and error recovery.",
        },
        "7. User Management Layer": {
            "thread_operations.py": {
                "create_thread_run_entry()": {
                    "Details": [
                        "Creates entry in users_threads_runs table",
                        "Parameters: email, thread_id, prompt, run_id (auto-generated if None)",
                        "ON CONFLICT DO UPDATE for idempotency",
                        "Returns run_id even on database failure (fallback UUID)",
                        "Decorated with @retry_on_prepared_statement_error",
                    ],
                    "Summary": "Thread run entry creation with conflict handling and fallback.",
                },
                "get_user_chat_threads()": {
                    "Details": [
                        "Retrieves user's threads with pagination (limit, offset)",
                        "Aggregates: latest_timestamp, run_count per thread",
                        "Extracts first prompt for thread title generation",
                        "Truncates titles to THREAD_TITLE_MAX_LENGTH + '...'",
                        "Returns list of dicts: thread_id, title, timestamp, run_count",
                    ],
                    "Summary": "Paginated user thread retrieval with title generation from first prompt.",
                },
                "get_user_chat_threads_count()": {
                    "Details": [
                        "Counts distinct thread_ids for user",
                        "Used for pagination calculations",
                        "Returns 0 on error to prevent API crashes",
                        "Decorated with retry logic",
                    ],
                    "Summary": "Total thread count for user with error-safe fallback.",
                },
                "delete_user_thread_entries()": {
                    "Details": [
                        "Deletes all users_threads_runs entries for user+thread",
                        "Counts entries before deletion for verification",
                        "Returns dict: deleted_count, message, thread_id, user_email",
                        "Provides detailed logging and traceback on errors",
                        "Used for thread deletion functionality",
                    ],
                    "Summary": "Thread entry deletion with count verification and detailed feedback.",
                },
                "Summary": "Comprehensive thread management: creation, retrieval, counting, and deletion.",
            },
            "sentiment_tracking.py": {
                "update_thread_run_sentiment()": {
                    "Details": [
                        "Updates sentiment field by run_id",
                        "Sentiment: bool (True=positive, False=negative, None=no feedback)",
                        "Returns bool: True if updated, False on error",
                        "Used for user feedback tracking",
                        "Decorated with @retry_on_prepared_statement_error",
                    ],
                    "Summary": "User feedback sentiment update by run identifier.",
                },
                "get_thread_run_sentiments()": {
                    "Details": [
                        "Retrieves all sentiments for email+thread_id where sentiment IS NOT NULL",
                        "Returns dict: {run_id: sentiment}",
                        "Empty dict on error to prevent crashes",
                        "Used for displaying feedback history",
                    ],
                    "Summary": "Thread sentiment retrieval for feedback history display.",
                },
                "Summary": "User sentiment tracking for feedback collection and analysis.",
            },
            "Summary": "User session management: thread operations and sentiment tracking.",
        },
        "8. Windows Compatibility": {
            "Event Loop Policy": {
                "Details": [
                    "Windows detection: sys.platform == 'win32'",
                    "Sets WindowsSelectorEventLoopPolicy for psycopg compatibility",
                    "Default ProactorEventLoop doesn't support all psycopg features",
                    "SelectorEventLoop provides better database connection compatibility",
                    "Configured in main.py before any async operations",
                ],
                "Summary": "Windows-specific event loop configuration for PostgreSQL driver compatibility.",
            },
            "Summary": "Windows platform compatibility configuration for async PostgreSQL operations.",
        },
        "9. Debugging and Observability": {
            "Debug Logging": {
                "Details": [
                    "print__checkpointers_debug() from api.utils.debug",
                    "Numbered step logging (e.g., '212 - DB CONFIG START')",
                    "Full traceback logging on errors",
                    "Operation-specific prefixes (SSL_RETRY, POOL HEALTH CHECK, etc.)",
                    "Conditional detailed logging (first N items) for performance",
                ],
                "Summary": "Comprehensive debug logging with numbered steps and operation prefixes.",
            },
            "Error Tracking": {
                "Details": [
                    "Full traceback capture via traceback.format_exc()",
                    "Error pattern detection for specialized handling",
                    "Retry attempt counting and logging",
                    "Pool health check failure tracking",
                    "Cleanup operation error logging (non-fatal)",
                ],
                "Summary": "Detailed error tracking with tracebacks and retry attempt monitoring.",
            },
            "Performance Monitoring": {
                "Details": [
                    "Connection pool health checks with timing",
                    "Checkpointer creation timing via debug logs",
                    "Table setup operation timing",
                    "Prepared statement cleanup counts",
                    "Memory cleanup via garbage collection",
                ],
                "Summary": "Performance monitoring through debug logs and operation timing.",
            },
            "Summary": "Extensive debugging and observability through comprehensive logging and error tracking.",
        },
        "10. Main.py Checkpointer Usage": {
            "Initialization Flow": {
                "Details": [
                    "Line 40: Check for provided checkpointer parameter",
                    "Lines 41-44: Try PostgreSQL via get_global_checkpointer()",
                    "Lines 45-47: Fallback to InMemorySaver on failure",
                    "Line 46: Create LangGraph graph with checkpointer",
                    "Checkpointer enables conversation memory and resumability",
                ],
                "Summary": "Checkpointer initialization with PostgreSQL primary and InMemorySaver fallback.",
            },
            "State Management": {
                "Lines 50-54: Existing State Detection": {
                    "Details": [
                        "graph.aget_state() queries checkpointer for existing state",
                        "Checks for messages to determine continuation",
                        "is_continuing_conversation flag drives state initialization",
                        "Error handling treats failures as new conversation",
                    ],
                    "Summary": "Existing state detection from checkpointer for conversation continuation.",
                },
                "Lines 55-57: State Preparation": {
                    "Details": [
                        "Continuing: Partial state update (reset iteration fields)",
                        "New: Full state initialization (all 13 DataAnalysisState fields)",
                        "New conversations get generate_initial_followup_prompts()",
                        "Checkpointer merges partial updates with existing state",
                    ],
                    "Summary": "State preparation based on conversation type (new vs continuing).",
                },
                "Summary": "Sophisticated state management for new and continuing conversations.",
            },
            "Graph Execution": {
                "Details": [
                    "Line 58: graph.ainvoke(input_state, config=config)",
                    "config contains thread_id for checkpointer state key",
                    "Automatic checkpointing after each LangGraph node",
                    "State persisted to PostgreSQL throughout execution",
                    "Enables resumability and error recovery",
                ],
                "Summary": "Graph execution with automatic state persistence via checkpointer.",
            },
            "Memory Management": {
                "Details": [
                    "Lines 37-39: Baseline memory measurement before execution",
                    "Lines 59-67: Post-execution memory monitoring and emergency GC",
                    "Lines 68-72: Final cleanup with memory growth warnings",
                    "GC_MEMORY_THRESHOLD (1900MB) for emergency cleanup trigger",
                    "Memory growth typically 50-150MB per analysis",
                ],
                "Summary": "Memory monitoring and garbage collection to prevent leaks.",
            },
            "Summary": "Main.py uses checkpointer for initialization, state management, and graph execution with memory monitoring.",
        },
        "Summary": "Comprehensive PostgreSQL-based checkpointing system with robust error handling, user management, and Windows compatibility.",
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="CZSU Checkpointer Subsystem Mindmap")
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
    # pdf_path = os.path.join(script_dir, script_name)
    # graph.render(pdf_path, format="pdf", cleanup=True)
    # print(f"Mindmap saved as '{pdf_path}.pdf'")

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

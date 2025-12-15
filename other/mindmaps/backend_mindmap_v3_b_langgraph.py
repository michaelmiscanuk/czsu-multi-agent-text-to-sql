import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "CZSU Multi-Agent Text-to-SQL Backend": {
        "2. LangGraph Orchestration": {
            "Entry Point (main.py)": {
                "Purpose": {
                    "Details": [
                        "Main orchestration function for CZSU multi-agent text-to-SQL analysis",
                        "Serves as central coordinator from NL question to formatted answer",
                        "Supports both CLI and API execution modes",
                        "Handles error recovery, memory management, and observability",
                        "Converts natural language questions to SQL, executes queries, and synthesizes answers",
                        "Combines database results with PDF documentation for comprehensive answers",
                    ],
                    "Summary": "Primary entry point orchestrating LangGraph workflow with memory management and dual execution modes.",
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
                            "Parse CLI args (argparse) or use API-provided parameters",
                            "Generate thread_id if not provided (format: data_analysis_{8_hex})",
                            "Generate run_id for LangSmith tracing if not provided",
                            "Escape curly braces in prompt for safe f-string formatting",
                        ],
                        "Summary": "Resolves execution parameters from CLI or API call context.",
                    },
                    "2. Memory Baseline": {
                        "Details": [
                            "Record initial RSS memory usage via psutil.Process()",
                            "Force garbage collection (gc.collect()) for clean baseline",
                            "Enable leak detection through before/after comparison",
                            "Track memory_before in MB for monitoring",
                        ],
                        "Summary": "Establishes memory monitoring baseline for leak detection during execution.",
                    },
                    "3. Checkpointer Initialization": {
                        "Details": [
                            "Attempt PostgreSQL-based AsyncPostgresSaver via get_global_checkpointer()",
                            "Fallback to InMemorySaver if PostgreSQL unavailable",
                            "Enables conversation memory and workflow resumability",
                            "Checkpointer can be provided externally (API mode) or created (CLI mode)",
                        ],
                        "Summary": "Initializes persistent or in-memory checkpointer for state management.",
                    },
                    "4. Graph Creation": {
                        "Details": [
                            "Initialize LangGraph workflow via create_graph(checkpointer)",
                            "Graph contains 17 nodes across 6 processing stages",
                            "Returns compiled StateGraph ready for execution",
                        ],
                        "Summary": "Creates compiled LangGraph workflow with all nodes and edges.",
                    },
                    "5. State Preparation": {
                        "New Conversations": {
                            "Details": [
                                "Initialize complete state with all 13 required fields",
                                "Set prompt, iteration=0, empty lists for results",
                                "Initialize messages with HumanMessage(content=prompt)",
                                "Generate initial follow-up prompts via generate_initial_followup_prompts()",
                                "Template-based prompt generation with pseudo-random seed",
                            ],
                            "Summary": "Full state initialization for new conversations with starter suggestions.",
                        },
                        "Continuing Conversations": {
                            "Details": [
                                "Check existing state via graph.aget_state()",
                                "Reset iteration-specific fields only (iteration=0, queries_and_results=[])",
                                "Preserve conversation history (messages maintained by checkpointer)",
                                "Reset retrieval results to force fresh search",
                            ],
                            "Summary": "Partial state update preserving conversation context.",
                        },
                        "Summary": "Prepares input state based on conversation context (new vs continuing).",
                    },
                    "6. Graph Execution": {
                        "Details": [
                            "Invoke LangGraph: result = await graph.ainvoke(input_state, config)",
                            "config = {'configurable': {'thread_id': thread_id}, 'run_id': run_id}",
                            "Automatic checkpointing after each node",
                            "Parallel retrieval of database selections + PDF chunks",
                            "Iterative SQL generation with reflection (up to MAX_ITERATIONS)",
                            "Graph executes: rewrite → retrieve (parallel) → generate → reflect → format → save",
                        ],
                        "Summary": "Executes graph workflow with automatic state persistence and parallel processing.",
                    },
                    "7. Memory Monitoring": {
                        "Details": [
                            "Track memory growth during execution (memory_after_graph - memory_before)",
                            "Trigger emergency GC if exceeds GC_MEMORY_THRESHOLD (default: 1900MB)",
                            "Log warnings for suspicious memory patterns",
                            "Calculate memory freed by emergency GC",
                        ],
                        "Summary": "Monitors memory usage and triggers cleanup when thresholds exceeded.",
                    },
                    "8. Result Processing": {
                        "Details": [
                            "Extract final_answer from messages[-1].content (last AI message)",
                            "Extract queries_and_results: List[(SQL_query, result_data)]",
                            "Filter selection codes via get_used_selection_codes()",
                            "Parse SQL with extract_table_names_from_sql() to find used tables",
                            "Serialize PDF chunks (Document objects) to JSON-compatible format",
                            "Generate dataset URLs: /datasets/{selection_code}",
                            "Extract followup_prompts from graph result",
                        ],
                        "Summary": "Processes graph output into API-friendly serialized format.",
                    },
                    "9. Final Cleanup": {
                        "Details": [
                            "Force final garbage collection (gc.collect())",
                            "Monitor total memory retention (memory_final - memory_before)",
                            "Log warnings if total growth exceeds 100MB",
                            "Return serialized result dictionary with 10 fields",
                        ],
                        "Summary": "Performs final cleanup and returns structured result dictionary.",
                    },
                    "Summary": "Nine-step orchestration workflow from initialization to result delivery.",
                },
                "Helper Functions": {
                    "extract_table_names_from_sql(sql_query: str)": {
                        "Details": [
                            "Parses SQL queries to identify all table names",
                            "Handles FROM clauses (comma-separated tables)",
                            "Handles all JOIN types (INNER, LEFT, RIGHT, FULL)",
                            "Strips SQL comments (line and block comments)",
                            "Normalizes whitespace for consistent parsing",
                            "Regex patterns for quoted/unquoted table names",
                            "Returns deduplicated uppercase table names",
                        ],
                        "Summary": "Extracts dataset table names from SQL for attribution purposes.",
                    },
                    "get_used_selection_codes(queries_and_results, top_selection_codes)": {
                        "Details": [
                            "Filters top_selection_codes to only those in queries",
                            "Parses all SQL queries via extract_table_names_from_sql()",
                            "Compares table names against candidate selection codes",
                            "Returns intersection of candidates and actually used codes",
                            "Ensures frontend displays only relevant datasets",
                        ],
                        "Summary": "Identifies which dataset selection codes were actually queried.",
                    },
                    "generate_initial_followup_prompts()": {
                        "Details": [
                            "Generates 5 diverse starter suggestions for new conversations",
                            "Uses timestamp-based pseudo-random seed (int(time.time() * 1000) % 1000000)",
                            "Template-based generation with 15+ prompt templates",
                            "Topic pools: regions, metrics, topics, periods, indicators, etc.",
                            "Fills placeholders: {region}, {metric}, {topic}, {period}, etc.",
                            "Ensures variety across conversation starts",
                            "Prevents duplicate templates in single generation",
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
                        "Azure OpenAI": {
                            "Details": [
                                "AZURE_OPENAI_API_KEY - Authentication key",
                                "AZURE_OPENAI_ENDPOINT - API endpoint URL",
                                "AZURE_OPENAI_API_VERSION - API version (default: 2024-12-01-preview)",
                            ],
                            "Summary": "Azure OpenAI service configuration.",
                        },
                        "PostgreSQL Checkpointer": {
                            "Details": [
                                "POSTGRES_CONNECTION_STRING - PostgreSQL connection for checkpointer",
                                "Format: postgresql://user:pass@host:port/dbname",
                            ],
                            "Summary": "PostgreSQL configuration for persistent state.",
                        },
                        "Memory Management": {
                            "Details": [
                                "GC_MEMORY_THRESHOLD - Memory growth threshold in MB (default: 1900)",
                                "Triggers emergency garbage collection when exceeded",
                            ],
                            "Summary": "Memory leak prevention threshold.",
                        },
                        "Workflow Control": {
                            "Details": [
                                "MAX_ITERATIONS - Maximum reflection iterations (default: 1)",
                                "MAX_QUERIES_LIMIT_FOR_REFLECT - Query history limit (default: 10)",
                            ],
                            "Summary": "Workflow iteration and history limits.",
                        },
                        "MCP Server": {
                            "Details": [
                                "MCP_SERVER_URL - Remote MCP server URL for SQLite tools",
                                "USE_LOCAL_SQLITE_FALLBACK - Enable local fallback (default: '1')",
                            ],
                            "Summary": "MCP server configuration with fallback control.",
                        },
                        "Azure Translator": {
                            "Details": [
                                "TRANSLATOR_TEXT_SUBSCRIPTION_KEY - Azure Translator API key",
                                "TRANSLATOR_TEXT_REGION - Azure region (e.g., 'westeurope')",
                                "TRANSLATOR_TEXT_ENDPOINT - Translator API endpoint URL",
                            ],
                            "Summary": "Azure Translator service for language detection and translation.",
                        },
                        "Summary": "Environment-driven configuration for Azure, PostgreSQL, memory limits, and workflow control.",
                    },
                    "Constants": {
                        "Details": [
                            "DEFAULT_PROMPT - Default test prompt (Czech fuel production question)",
                            "MAX_ITERATIONS - Imported from nodes.py (default: 1)",
                            "MAX_QUERIES_LIMIT_FOR_REFLECT - Defined in state.py (default: 10)",
                            "BASE_DIR - Project root directory via Path(__file__).resolve().parents[2]",
                            "DB_PATH - BASE_DIR / 'data' / 'czsu_data.db'",
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
                        "Creates directed graph with controlled cycles (reflection loop)",
                        "Supports dual retrieval sources (database + PDF documentation)",
                    ],
                    "Summary": "Core graph architecture defining node connections and workflow execution logic.",
                },
                "Graph Architecture": {
                    "Phase 1: Query Preprocessing": {
                        "Details": [
                            "START → rewrite_prompt → summarize_messages_rewrite",
                            "rewrite_prompt: Converts conversational questions to standalone queries",
                            "Handles pronoun resolution and topic change detection",
                            "summarize_messages_rewrite: Token management via conversation summarization",
                            "Prepares optimized query for parallel retrieval",
                        ],
                        "Summary": "Rewrites user query and summarizes conversation context.",
                    },
                    "Phase 2: Parallel Retrieval": {
                        "Branch A - Database Selections": {
                            "Details": [
                                "retrieve_similar_selections_hybrid_search",
                                "Hybrid search: semantic (ChromaDB) + BM25 with configurable weighting",
                                "rerank_table_descriptions: Cohere reranking for relevance",
                                "relevant_selections: Top-k filtering with threshold",
                                "Outputs: top_selection_codes (dataset identifiers)",
                            ],
                            "Summary": "Hybrid search + reranking for dataset selection retrieval.",
                        },
                        "Branch B - PDF Documentation": {
                            "Details": [
                                "retrieve_similar_chunks_hybrid_search",
                                "Hybrid search: semantic (ChromaDB) + BM25 for PDF chunks",
                                "rerank_chunks: Cohere reranking for chunk relevance",
                                "relevant_chunks: Threshold filtering for quality",
                                "Outputs: top_chunks (Document objects with metadata)",
                            ],
                            "Summary": "Hybrid search + reranking for PDF chunk retrieval.",
                        },
                        "Synchronization": {
                            "Details": [
                                "Both branches feed into post_retrieval_sync node",
                                "Waits for both parallel branches to complete",
                                "No data modification, just synchronization point",
                            ],
                            "Summary": "Synchronizes parallel retrieval branches before routing.",
                        },
                        "Summary": "Dual parallel branches for database and PDF retrieval with reranking.",
                    },
                    "Phase 3: Synchronization & Routing": {
                        "Details": [
                            "[relevant_selections, relevant_chunks] → post_retrieval_sync",
                            "Conditional routing via route_after_sync() from my_agent.utils.routers",
                            "IF top_selection_codes found → get_schema (proceed with database)",
                            "ELIF chromadb_missing flag set → END (error: no ChromaDB)",
                            "ELSE → format_answer (PDF-only response, no database data)",
                        ],
                        "Summary": "Synchronizes parallel branches and routes based on available data sources.",
                    },
                    "Phase 4: Query Loop (Optional Reflection)": {
                        "Schema Loading": {
                            "Details": [
                                "get_schema node loads SQLite schema metadata",
                                "Queries selection_descriptions.db for extended_description",
                                "Formats schemas with dataset identifier prefix",
                                "Schema includes: table names, columns, types, distinct values",
                            ],
                            "Summary": "Loads database schema for SQL generation context.",
                        },
                        "Query Generation": {
                            "Details": [
                                "generate_query: Agentic SQL generation with tool calling",
                                "LLM autonomously decides when/how many times to execute queries",
                                "Uses sqlite_query tool (MCP or local fallback)",
                                "Appends (query, result) tuples to queries_and_results",
                                "Handles Czech diacritics, JOINs, CELKEM row filtering",
                            ],
                            "Summary": "Generates and executes SQL queries iteratively.",
                        },
                        "Summarization": {
                            "Details": [
                                "summarize_messages_query: Token management after query execution",
                                "Keeps messages to [summary, last_message] structure",
                            ],
                            "Summary": "Manages token limits after query generation.",
                        },
                        "Iteration Control": {
                            "Details": [
                                "Conditional routing via route_after_query()",
                                "IF iteration < MAX_ITERATIONS (default: 1) → reflect",
                                "ELSE → format_answer (force answer at iteration limit)",
                            ],
                            "Summary": "Controls reflection loop to prevent infinite cycles.",
                        },
                        "Reflection Cycle": {
                            "Details": [
                                "reflect: Analyzes query results for completeness",
                                "LLM decides: 'improve' (better query needed) or 'answer' (sufficient data)",
                                "Provides specific feedback for query improvement",
                                "Updates reflection_decision field",
                                "summarize_messages_reflect: Token management",
                                "route_after_reflect() routes based on decision:",
                                "IF decision == 'improve' → generate_query (loop back)",
                                "ELIF decision == 'answer' → format_answer",
                            ],
                            "Summary": "Self-correction cycle for query improvement.",
                        },
                        "Summary": "Iterative SQL generation with optional reflection for improvement.",
                    },
                    "Phase 5: Answer Finalization": {
                        "Details": [
                            "format_answer: Synthesizes SQL results + PDF chunks + selection descriptions",
                            "Generates bilingual answers (matches query language)",
                            "Provides source attribution (database vs documentation)",
                            "summarize_messages_format: Token management",
                            "generate_followup_prompts: LLM-generated suggestions based on answer context",
                            "submit_final_answer: Packages answer for user delivery",
                            "save: Optional file save (configurable)",
                            "cleanup_resources: Explicit ChromaDB client cleanup",
                            "END",
                        ],
                        "Summary": "Formats final answer, generates suggestions, saves, and cleans up.",
                    },
                    "Summary": "Five-phase workflow with parallel retrieval, conditional routing, and iterative improvement.",
                },
                "Node Summary": {
                    "Details": [
                        "Total: 17 nodes across 6 functional categories",
                        "Preprocessing: rewrite_prompt (1 node)",
                        "Retrieval: retrieve_similar_selections_hybrid_search, retrieve_similar_chunks_hybrid_search (2 nodes)",
                        "Reranking: rerank_table_descriptions, rerank_chunks (2 nodes)",
                        "Filtering: relevant_selections, relevant_chunks (2 nodes)",
                        "Synchronization: post_retrieval_sync (1 node)",
                        "Query Execution: get_schema, generate_query (2 nodes)",
                        "Reflection: reflect (1 node)",
                        "Answer Generation: format_answer, generate_followup_prompts, submit_final_answer (3 nodes)",
                        "Persistence: save, cleanup_resources (2 nodes)",
                        "Memory Management: summarize_messages (4 instances at different points: rewrite/query/reflect/format)",
                        "Node implementations: my_agent/utils/nodes.py (NOT detailed in mindmap per user request)",
                    ],
                    "Summary": "17 specialized nodes handling retrieval, generation, reflection, and formatting.",
                },
                "Conditional Routing": {
                    "route_after_sync() - Post-Retrieval Router": {
                        "Details": [
                            "Location: my_agent/utils/routers.py",
                            "Routes based on available data sources",
                            "Checks: state.get('top_selection_codes') presence",
                            "Checks: state.get('chromadb_missing') error flag",
                            "Returns: 'get_schema' | 'format_answer' | END",
                            "Logic: Database found → schema, ChromaDB error → END, PDF only → format",
                        ],
                        "Summary": "Post-retrieval routing based on data source availability.",
                    },
                    "route_after_query() - Iteration Control": {
                        "Details": [
                            "Location: my_agent/utils/routers.py",
                            "Controls reflection loop iteration",
                            "Checks: state.get('iteration', 0) vs MAX_ITERATIONS (default: 1)",
                            "Returns: 'reflect' | 'format_answer'",
                            "Prevents infinite reflection loops",
                        ],
                        "Summary": "Iteration control preventing infinite reflection loops.",
                    },
                    "route_after_reflect() - Decision Router": {
                        "Details": [
                            "Location: my_agent/utils/routers.py",
                            "Routes based on reflection decision",
                            "Checks: state.get('reflection_decision', 'improve')",
                            "Returns: 'generate_query' | 'format_answer'",
                            "Logic: 'improve' → query again, 'answer' → finalize",
                        ],
                        "Summary": "Reflection-based routing for query improvement or answer finalization.",
                    },
                    "Summary": "Three routing functions externalized to my_agent/utils/routers.py for clarity.",
                },
                "create_graph() Function": {
                    "Details": [
                        "Main factory function in my_agent/agent.py",
                        "Accepts optional checkpointer parameter (defaults to InMemorySaver)",
                        "Initializes: graph = StateGraph(DataAnalysisState)",
                        "Adds all 17 nodes via graph.add_node()",
                        "Defines sequential edges via graph.add_edge()",
                        "Defines conditional edges via graph.add_conditional_edges()",
                        "Edge types: START edge, parallel edges, sync edges, conditional edges, END edge",
                        "Compiles: compiled_graph = graph.compile(checkpointer=checkpointer)",
                        "Returns compiled graph ready for ainvoke()",
                    ],
                    "Summary": "Factory function that constructs and compiles the complete workflow graph.",
                },
                "Design Principles": {
                    "Details": [
                        "1. Parallel execution for efficiency (dual retrieval branches)",
                        "2. Conditional routing based on available data (externalized to routers module)",
                        "3. Controlled iteration with MAX_ITERATIONS prevention (env configurable)",
                        "4. State checkpointing for workflow resumption (PostgreSQL or InMemory)",
                        "5. Token management via automatic message summarization (4 summarization points)",
                        "6. Resource cleanup for connection management (explicit cleanup_resources node)",
                        "7. Modular architecture with separated routing logic (my_agent.utils.routers)",
                        "8. Absolute imports with my_agent prefix for clarity",
                        "9. Debug tracing integrated (print__analysis_tracing_debug)",
                        "10. Error handling with graceful degradation (fallbacks at each level)",
                    ],
                    "Summary": "Ten key design principles ensuring efficiency, safety, and maintainability.",
                },
                "Summary": "Graph definition module implementing multi-phase workflow with parallel retrieval, conditional routing, and state management.",
            },
            "State Management (my_agent/utils/state.py)": {
                "Purpose": {
                    "Details": [
                        "Defines state schema for LangGraph workflow via TypedDict",
                        "Implements DataAnalysisState with 15 fields tracking data flow",
                        "Provides custom reducers for memory efficiency (limited_queries_reducer)",
                        "Enables type safety and IDE autocomplete",
                        "Manages conversational memory with bounded token usage",
                    ],
                    "Summary": "State schema and reducers for workflow data management (see state.py for full field details).",
                },
                "Key State Fields": {
                    "Input & Processing": {
                        "Details": [
                            "prompt: str - Original user question (unchanged throughout)",
                            "rewritten_prompt: str - Standalone search-optimized question",
                        ],
                        "Summary": "Input and processed query fields.",
                    },
                    "Conversation Management": {
                        "Details": [
                            "messages: List[BaseMessage] - Always [summary, last_message]",
                            "Uses default replacement behavior (not append)",
                            "Summarization in summarize_messages_node before replacement",
                            "Prevents token overflow in long conversations",
                        ],
                        "Summary": "Token-efficient message storage with 2-message structure.",
                    },
                    "Workflow Control": {
                        "Details": [
                            "iteration: int - Loop counter (default: 0, max: MAX_ITERATIONS)",
                            "reflection_decision: str - 'improve' | 'answer' from reflect node",
                            "chromadb_missing: bool - Error flag for missing ChromaDB",
                        ],
                        "Summary": "Control fields for iteration tracking and error flagging.",
                    },
                    "Database Selection Retrieval": {
                        "Details": [
                            "hybrid_search_results: List[Document] - Raw hybrid search results",
                            "most_similar_selections: List[Tuple[str, float]] - Reranked with Cohere scores",
                            "top_selection_codes: List[str] - Final top-k codes passing threshold",
                        ],
                        "Summary": "Three-stage database selection retrieval state.",
                    },
                    "PDF Chunk Retrieval": {
                        "Details": [
                            "hybrid_search_chunks: List[Document] - Raw PDF search results",
                            "most_similar_chunks: List[Tuple[Document, float]] - Reranked chunks with scores",
                            "top_chunks: List[Document] - Filtered above relevance threshold",
                        ],
                        "Summary": "Three-stage PDF chunk retrieval state.",
                    },
                    "Query Execution": {
                        "Details": [
                            "queries_and_results: Annotated[List[Tuple[str, str]], limited_queries_reducer]",
                            "List of (SQL_query, result_string) tuples",
                            "Custom reducer limits to recent MAX_QUERIES_LIMIT_FOR_REFLECT entries (default: 10)",
                            "Prevents memory/token overflow in reflection loops",
                        ],
                        "Summary": "Query history with custom limiting reducer.",
                    },
                    "Output": {
                        "Details": [
                            "final_answer: str - Formatted answer synthesizing all sources",
                            "followup_prompts: List[str] - Suggested follow-up questions (5 prompts)",
                        ],
                        "Summary": "Final output fields for answer and suggestions.",
                    },
                    "Summary": "15-field TypedDict organizing workflow state across input, processing, retrieval, execution, and output stages.",
                },
                "Custom Reducers": {
                    "limited_queries_reducer": {
                        "Details": [
                            "Limits queries_and_results to latest N entries",
                            "N = MAX_QUERIES_LIMIT_FOR_REFLECT (env var, default: 10)",
                            "Prevents memory/token overflow in reflection loops",
                            "Combines existing (left) + new (right) queries",
                            "Returns: combined[-MAX_QUERIES_LIMIT_FOR_REFLECT:]",
                            "Ensures reflection context stays bounded",
                        ],
                        "Summary": "Custom reducer preventing query history overflow by limiting to recent entries.",
                    },
                    "Summary": "Single custom reducer for query history management.",
                },
                "Memory Efficiency Features": {
                    "Details": [
                        "Always 2 messages: [summary (SystemMessage), last_message]",
                        "Limited query history via custom reducer (max 10 entries)",
                        "Automatic summarization at 4 points (rewrite/query/reflect/format)",
                        "Bounded memory usage in long conversations",
                        "No unbounded list growth (all lists have limits)",
                    ],
                    "Summary": "Multiple strategies ensuring memory-efficient state management.",
                },
                "Summary": "TypedDict-based state schema with 15 fields and custom reducers for memory-efficient workflow tracking (detailed implementation in state.py).",
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
                    "description: 'LangGraph application for data analysis'",
                    "entry_point: 'main.py'",
                    "dependencies.python_version: '>=3.9,<3.12'",
                    "dependencies.requirements_file: 'requirements.txt'",
                    "graphs.agent: 'my_agent.agent:create_graph'",
                ],
                "Summary": "LangGraph project configuration specifying entry point and graph location.",
            },
            "Package Initialization (my_agent/__init__.py)": {
                "Details": [
                    "AGENT_TYPE = 'basic1' - Agent type selection constant",
                    "Conditional import based on AGENT_TYPE",
                    "from .agent import create_graph (for AGENT_TYPE='basic1')",
                    "__all__ = ['create_graph'] - Public API export",
                    "Hides internal configuration from external imports",
                ],
                "Summary": "Package initialization with agent type selection and clean public API.",
            },
            "Supporting Utilities": {
                "Models (my_agent/utils/models.py)": {
                    "Azure OpenAI Chat Models": {
                        "Details": [
                            "get_azure_openai_chat_llm(deployment_name='gpt-4o__test1', model_name='gpt-4o', openai_api_version='2024-05-01-preview', temperature=0.0) - GPT-4o via AzureChatOpenAI",
                            "Deployment: gpt-4o__test1, Model: gpt-4o",
                            "API version: 2024-05-01-preview",
                            "get_azure_openai_chat_llm(deployment_name='gpt-4o-mini-mimi2', model_name='gpt-4o-mini', openai_api_version='2024-05-01-preview', temperature=0.0) - GPT-4o-mini",
                            "Deployment: gpt-4o-mini-mimi2, Model: gpt-4o-mini",
                            "get_azure_openai_chat_llm(deployment_name='gpt-4.1___test1', model_name='gpt-4.1', openai_api_version='2024-05-01-preview', temperature=0.0) - GPT-4.1",
                            "Deployment: gpt-4.1___test1, Model: gpt-4.1",
                            "All support sync (invoke) and async (ainvoke) operations",
                            "Configured via AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY",
                        ],
                        "Summary": "Factory functions for Azure OpenAI GPT models with configurable temperature.",
                    },
                    "Ollama Local Models": {
                        "Details": [
                            "get_ollama_llm(model_name='llama3.2:1b', base_url='http://localhost:11434/v1')",
                            "Uses ChatOpenAI with local OLLAMA endpoint",
                            "Supports: llama3.2, smollm, qwen:7b, etc.",
                            "Dummy API key ('ollama-local-dummy-key') required but not used",
                            "Test function: get_ollama_llm_test()",
                        ],
                        "Summary": "Local Ollama model factory functions for development/testing.",
                    },
                    "Azure Embedding Models": {
                        "Details": [
                            "get_azure_embedding_model() - Returns AzureOpenAI client",
                            "API version: 2024-12-01-preview",
                            "get_langchain_azure_embedding_model(model_name='text-embedding-3-large__test1')",
                            "Returns: AzureOpenAIEmbeddings for LangChain integration",
                            "Used for ChromaDB vector search",
                        ],
                        "Summary": "Azure embedding model factory functions for vector search.",
                    },
                    "Summary": "LLM and embedding model factory functions for Azure and Ollama with sync/async support.",
                },
                "Tools (my_agent/utils/tools.py)": {
                    "MCP SQLite Tools": {
                        "Details": [
                            "get_sqlite_tools() - Async function returning List[BaseTool]",
                            "Primary: MultiServerMCPClient with streamable_http transport",
                            "Connects to remote MCP server: MCP_SERVER_URL (env var)",
                            "Transport: 'streamable_http' for remote FastMCP servers",
                            "Returns: Tools from MCP server (sqlite_query tool)",
                            "Fallback: LocalSQLiteQueryTool if MCP unavailable",
                        ],
                        "Summary": "MCP-compatible SQLite query tools with remote/local fallback.",
                    },
                    "Local SQLite Fallback": {
                        "Details": [
                            "LocalSQLiteQueryInput - Pydantic model for tool input schema",
                            "LocalSQLiteQueryTool - BaseTool implementation",
                            "Executes queries against local czsu_data.db",
                            "Database path: BASE_DIR / 'data' / 'czsu_data.db'",
                            "Supports both sync (_run) and async (_arun) execution",
                            "Fallback enabled via USE_LOCAL_SQLITE_FALLBACK (env var, default: '1')",
                        ],
                        "Summary": "Local SQLite tool implementation for MCP fallback.",
                    },
                    "Control Tools": {
                        "Details": [
                            "@tool finish_gathering() - Signals data gathering completion",
                            "LLM calls this to indicate sufficient data collected",
                            "Used in agentic query execution workflow",
                        ],
                        "Summary": "Control tool for LLM-driven query execution flow.",
                    },
                    "Summary": "MCP-compatible tools for SQL query execution with fallback mechanism and agentic control.",
                },
                "Helpers (my_agent/utils/helpers.py)": {
                    "Schema Loading": {
                        "Details": [
                            "load_schema(state) - Async function loading database schema",
                            "Queries: metadata/llm_selection_descriptions/selection_descriptions.db",
                            "Fetches: extended_description for each selection_code",
                            "Schema includes: table names, columns, types, distinct values, descriptions",
                            "Format: 'Dataset: {code}.\\n{extended_description}'",
                            "Delimiter: '\\n**************\\n' between multiple schemas",
                            "Returns: Concatenated schema string or error message",
                        ],
                        "Summary": "Helper function loading SQLite schema metadata for SQL generation.",
                    },
                    "Translation": {
                        "Details": [
                            "translate_text(text, target_language='en') - Async Azure Translator API call",
                            "Endpoint: /translate?api-version=3.0&to={target_language}",
                            "Uses: TRANSLATOR_TEXT_SUBSCRIPTION_KEY, TRANSLATOR_TEXT_REGION, TRANSLATOR_TEXT_ENDPOINT",
                            "Headers: Ocp-Apim-Subscription-Key, Ocp-Apim-Subscription-Region, X-ClientTraceId",
                            "Runs synchronous requests.post in thread pool executor (async-safe)",
                            "Returns: Translated text in target language",
                        ],
                        "Summary": "Azure Translator API integration for multilingual translation.",
                    },
                    "Language Detection": {
                        "Details": [
                            "detect_language(text) - Async Azure Translator API call",
                            "Endpoint: /detect?api-version=3.0",
                            "Same authentication as translate_text",
                            "Runs synchronous requests.post in thread pool executor",
                            "Returns: Language code (e.g., 'cs', 'en', 'de')",
                            "Used for bilingual answer generation (match query language)",
                        ],
                        "Summary": "Azure Translator API integration for language detection.",
                    },
                    "Summary": "Helper functions for schema loading, translation, and language detection with async Azure API calls.",
                },
                "Routing Logic (my_agent/utils/routers.py)": {
                    "Purpose": {
                        "Details": [
                            "Contains all conditional routing logic for graph",
                            "Determines next node based on current state",
                            "Externalized from agent.py for modularity",
                            "Three routing functions for three decision points",
                            "Imports MAX_ITERATIONS from nodes module",
                        ],
                        "Summary": "Dedicated module for workflow routing decisions.",
                    },
                    "route_after_sync(state)": {
                        "Details": [
                            "Called after parallel retrieval synchronization",
                            "Checks: state.get('top_selection_codes') for database availability",
                            "Checks: state.get('chromadb_missing') error flag",
                            "Returns: 'get_schema' if selections found",
                            "Returns: END if chromadb_missing flag set",
                            "Returns: 'format_answer' if only PDF chunks available",
                            "Debug logging via print__analysis_tracing_debug",
                        ],
                        "Summary": "Routes based on available data sources after retrieval.",
                    },
                    "route_after_query(state)": {
                        "Details": [
                            "Called after query generation",
                            "Checks: state.get('iteration', 0) vs MAX_ITERATIONS",
                            "Returns: 'reflect' if iteration < MAX_ITERATIONS",
                            "Returns: 'format_answer' if max iterations reached",
                            "Prevents infinite reflection loops",
                        ],
                        "Summary": "Controls reflection loop iteration limit.",
                    },
                    "route_after_reflect(state)": {
                        "Details": [
                            "Called after reflection node",
                            "Checks: state.get('reflection_decision', 'improve')",
                            "Returns: 'generate_query' if decision == 'improve'",
                            "Returns: 'format_answer' if decision == 'answer'",
                            "Implements LLM-driven decision routing",
                        ],
                        "Summary": "Routes based on reflection decision to improve or answer.",
                    },
                    "Summary": "Modular routing logic determining workflow paths based on state conditions (detailed in routers.py).",
                },
                "Summary": "Supporting utilities for models, tools, helpers, and routing logic used by graph nodes.",
            },
            "Server Startup (uvicorn_start.py)": {
                "Details": [
                    "Imports FastAPI app from api.main",
                    "Sets WindowsSelectorEventLoopPolicy for Windows compatibility (CRITICAL: before any async imports)",
                    "Loads environment variables early via load_dotenv()",
                    "Development server configuration:",
                    "  - Host: 0.0.0.0 (all interfaces)",
                    "  - Port: 8000",
                    "  - Reload: True (hot reload enabled)",
                    "  - reload_dirs: ['api', 'my_agent'] (watch specific directories)",
                    "  - reload_delay: 0.25 (prevent multiple reloads)",
                    "  - Log level: info with colors and access log",
                    "Runs: uvicorn.run('api.main:app', ...)",
                ],
                "Summary": "Uvicorn startup script for development server with hot reload and Windows compatibility.",
            },
            "Summary": "LangGraph orchestration layer coordinating workflow execution, state management, and conditional routing for multi-agent text-to-SQL analysis.",
        },
        "3. External Integrations": {
            "Checkpointer Integration": {
                "Note": {
                    "Details": [
                        "Full checkpointer details in 'checkpointer' folder (NOT included per user request)",
                        "Factory: checkpointer/checkpointer/factory.py",
                        "Functions: initialize_checkpointer(), get_global_checkpointer(), create_async_postgres_saver()",
                        "Utilities: checkpointer/user_management/, checkpointer/database/",
                    ],
                    "Summary": "PostgreSQL-based persistent checkpointing (detailed mapping in separate checkpointer mindmap).",
                },
                "Summary": "LangGraph persistence layer using PostgreSQL (excluded from this mindmap per user request).",
            },
            "Data Stores": {
                "SQLite Databases": {
                    "Details": [
                        "data/czsu_data.db - Main CZSU statistical tables with Czech data",
                        "metadata/llm_selection_descriptions/selection_descriptions.db - Dataset metadata",
                        "Extended schema descriptions for SQL generation",
                        "Accessed via sqlite3.connect() or MCP tools",
                    ],
                    "Summary": "Local SQLite databases for CZSU data and selection metadata.",
                },
                "Vector Stores": {
                    "Details": [
                        "metadata/czsu_chromadb/ - ChromaDB for hybrid search",
                        "Collections: dataset descriptions and PDF chunks",
                        "Supports semantic + BM25 hybrid search with configurable weights",
                        "Used by retrieve_similar_selections and retrieve_similar_chunks nodes",
                        "Client cleanup in cleanup_resources_node",
                    ],
                    "Summary": "ChromaDB vector store for retrieval-augmented generation.",
                },
                "Summary": "Persistent datasets and vector assets powering data retrieval and generation.",
            },
            "LangSmith Tracing": {
                "Details": [
                    "Run ID tracking for observability (UUID per execution)",
                    "Thread ID for conversation tracking (persistent across turns)",
                    "Configuration metadata capture (thread_id, run_id in config)",
                    "Input/output logging for debugging (all LLM calls traced)",
                    "Enabled via run_id parameter in graph.ainvoke()",
                    "Environment: LANGCHAIN_API_KEY, LANGCHAIN_PROJECT",
                ],
                "Summary": "Observability platform for LLM call tracing and debugging.",
            },
            "Azure OpenAI": {
                "Details": [
                    "AZURE_OPENAI_API_KEY for authentication",
                    "AZURE_OPENAI_ENDPOINT for API calls",
                    "Models: GPT-4o, GPT-4o-mini, GPT-4.1 support",
                    "Embeddings: text-embedding-3-large for vector search",
                    "API version: 2024-05-01-preview (chat), 2024-12-01-preview (embeddings)",
                    "Used across all nodes requiring LLM calls",
                ],
                "Summary": "Azure OpenAI service integration for LLM and embedding calls.",
            },
            "Cohere Reranking": {
                "Details": [
                    "Reranks database selection search results in rerank_table_descriptions_node",
                    "Reranks PDF chunk search results in rerank_chunks_node",
                    "Provides relevance scores (0.0-1.0) for filtering",
                    "Improves retrieval quality after hybrid search",
                    "Used in dual retrieval branches (selections + chunks)",
                ],
                "Summary": "Cohere API for reranking hybrid search results.",
            },
            "Azure Translator": {
                "Details": [
                    "Language detection: detect_language(text) in helpers.py",
                    "Translation: translate_text(text, target_language='en') in helpers.py",
                    "API version: 3.0",
                    "Used for: PDF chunk retrieval (Czech query → English docs)",
                    "Used for: Bilingual answer generation (match query language)",
                    "Endpoints: /detect, /translate",
                ],
                "Summary": "Azure Cognitive Services Translator for language detection and multilingual translation.",
            },
            "MCP Server (Optional)": {
                "Details": [
                    "Remote MCP server for SQLite tools (streamable_http transport)",
                    "MultiServerMCPClient from langchain_mcp_adapters",
                    "URL: MCP_SERVER_URL environment variable",
                    "Fallback: Local SQLite tool if MCP unavailable",
                    "Used in generate_query_node for SQL execution",
                ],
                "Summary": "Optional Model Context Protocol server for remote SQLite queries.",
            },
            "Summary": "External service integrations including checkpointer, data stores, tracing, LLM, reranking, translation, and MCP.",
        },
        "Summary": "Comprehensive LangGraph-based backend architecture for CZSU multi-agent text-to-SQL analysis. Main entry (main.py) orchestrates workflow, LangGraph graph (my_agent/agent.py) defines 17-node execution flow with parallel retrieval and reflection loop, state management (my_agent/utils/state.py) tracks 15 fields with custom reducers, routing logic (my_agent/utils/routers.py) provides conditional flow control, and supporting utilities (models, tools, helpers) enable LLM calls, SQL execution, and data processing. External integrations include PostgreSQL checkpointing, ChromaDB vector search, Azure OpenAI, Cohere reranking, Azure Translator, and optional MCP server. Excludes: FastAPI layer, checkpointer details, tests, and node implementations (per user request).",
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

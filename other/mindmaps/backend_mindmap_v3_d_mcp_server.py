import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "MCP Server (Model Context Protocol) - SQLite Query Service": {
        "1. MCP Server Implementation (czsu_mcp_server_sqlite/)": {
            "FastMCP Server Core (main.py)": {
                "Server Initialization": {
                    "Details": [
                        "FastMCP framework version >=2.0.0",
                        "Server name: 'CZSU-SQLite-Server'",
                        "Single tool exposed: sqlite_query",
                        "SSE (Server-Sent Events) transport for HTTP",
                        "Zero-config deployment ready",
                        "Module-level mcp object export",
                    ],
                    "Summary": "FastMCP server instance with streamable HTTP transport for MCP protocol.",
                },
                "Environment Configuration": {
                    "Details": [
                        "PORT - Server port (default: 8100)",
                        "DATABASE_TYPE - 'sqlitecloud' or 'turso' (default: turso)",
                        "SQLITE_CLOUD_CONNECTION_STRING - SQLite Cloud connection",
                        "TURSO_CONNECTION_STRING - Turso libSQL connection",
                        "DEBUG - Enable debug logging (default: 0)",
                        "load_dotenv() for .env file loading",
                    ],
                    "Summary": "Environment-based configuration supporting multiple database backends.",
                },
                "Database Connectivity": {
                    "SQLite Cloud Backend": {
                        "Details": [
                            "get_sqlitecloud_connection() factory function",
                            "Uses sqlitecloud>=0.0.80 package",
                            "Connection string format: sqlitecloud://host:port/db?apikey=xxx",
                            "Cloud-hosted database (no local file needed)",
                            "Automatic connection pooling",
                        ],
                        "Summary": "SQLite Cloud connectivity for cloud-hosted database access.",
                    },
                    "Turso Backend": {
                        "Details": [
                            "get_turso_connection() factory function",
                            "Uses libsql>=0.1.11 package",
                            "Parse connection string with urllib.parse",
                            "Extract auth_token from query parameters",
                            "Distributed SQLite with edge replication",
                        ],
                        "Summary": "Turso libSQL connectivity for distributed edge database.",
                    },
                    "Connection Factory": {
                        "Details": [
                            "get_db_connection() - unified interface",
                            "Selects backend based on DATABASE_TYPE env var",
                            "Validates configuration on startup",
                            "Raises ValueError for unsupported types",
                            "Connection error handling with detailed messages",
                        ],
                        "Summary": "Abstracted database connection factory supporting multiple backends.",
                    },
                    "Summary": "Multi-backend database connectivity with automatic failover and validation.",
                },
                "Summary": "Standalone FastMCP server providing SQLite query access via MCP protocol.",
            },
            "MCP Tool Definition (@mcp.tool decorator)": {
                "sqlite_query Tool": {
                    "Tool Signature": {
                        "Details": [
                            "@mcp.tool() decorator for automatic registration",
                            "async def sqlite_query(query: str, ctx: Context) -> str",
                            "Parameter: query - SQL query string (SELECT only)",
                            "Returns: String representation of results",
                            "Context object for logging and metadata",
                        ],
                        "Summary": "Type-safe async tool with FastMCP decorator and context support.",
                    },
                    "Tool Implementation": {
                        "Details": [
                            "await ctx.info() for logging query execution",
                            "asyncio.to_thread() for blocking DB calls",
                            "_execute_query(q) sync function wrapper",
                            "Automatic result formatting (single value, list, JSON)",
                            "Returns 'No results found' for empty sets",
                            "Database-specific cursor handling (turso vs sqlitecloud)",
                        ],
                        "Summary": "Async query execution with thread pool and automatic result formatting.",
                    },
                    "Result Formatting": {
                        "Details": [
                            "Single value: str(result[0][0])",
                            "Multiple rows: json.dumps(result, ensure_ascii=False)",
                            "Empty result: 'No results found'",
                            "Czech character support with ensure_ascii=False",
                            "Logging: result summary and full result",
                        ],
                        "Summary": "Intelligent result formatting for different query types with UTF-8 support.",
                    },
                    "Summary": "Main MCP tool for executing SQL queries with context logging and formatting.",
                },
                "Tool Documentation": {
                    "Details": [
                        "Comprehensive docstring with args and returns",
                        "Usage examples in docstring",
                        "Note about read-only operations",
                        "Automatically exposed in MCP protocol schema",
                        "FastMCP auto-generates OpenAPI documentation",
                    ],
                    "Summary": "Self-documenting tool with examples and constraints.",
                },
                "Summary": "Single @mcp.tool() decorated function for SQL query execution.",
            },
            "Custom Routes (@mcp.custom_route)": {
                "Health Check Endpoint": {
                    "Details": [
                        "@mcp.custom_route('/health', methods=['GET'])",
                        "Tests database connectivity with SELECT 1",
                        "Returns JSON: {status, database, database_type}",
                        "Status codes: 200 (healthy), 503 (unhealthy)",
                        "Handles ValueError, ConnectionError, ImportError",
                        "Database-specific health check logic",
                    ],
                    "Summary": "Health monitoring endpoint for uptime checks and database validation.",
                },
                "Summary": "Custom HTTP routes for monitoring and diagnostics.",
            },
            "Server Startup Logic": {
                "Local Testing Mode": {
                    "Details": [
                        "if __name__ == '__main__': block",
                        "Ignored by FastMCP Cloud (uses mcp object directly)",
                        "Print startup banner with configuration",
                        "Test database connection before starting",
                        "mcp.run(transport='sse', host='0.0.0.0', port=PORT)",
                        "SSE transport for HTTP-based MCP protocol",
                    ],
                    "Summary": "Local development server with SSE transport and connection validation.",
                },
                "Summary": "Dual-mode startup: local testing and cloud deployment.",
            },
            "Summary": "Complete MCP server implementation with tool definition, routes, and startup.",
        },
        "2. Integration with LangGraph Agent": {
            "Tool Acquisition (my_agent/utils/tools.py)": {
                "get_sqlite_tools Function": {
                    "Primary Strategy - Remote MCP": {
                        "Details": [
                            "async def get_sqlite_tools() -> List[BaseTool]",
                            "Check MCP_SERVER_URL environment variable",
                            "Use MultiServerMCPClient from langchain_mcp_adapters",
                            "Transport: 'streamable_http' (not 'streamable-http')",
                            "Config: {'sqlite': {'transport': 'streamable_http', 'url': MCP_SERVER_URL}}",
                            "await client.get_tools() to retrieve MCP tools",
                            "Returns LangChain-compatible BaseTool instances",
                            "Logs connection success and available tool names",
                        ],
                        "Summary": "Official LangChain MCP adapter pattern for remote tool retrieval.",
                    },
                    "Fallback Strategy - Local SQLite": {
                        "Details": [
                            "Activated when MCP_SERVER_URL not set or connection fails",
                            "USE_LOCAL_SQLITE_FALLBACK env var (default: 1)",
                            "LocalSQLiteQueryTool(BaseTool) class",
                            "name='sqlite_query' (same as remote)",
                            "Direct SQLite connection to DB_PATH (data/czsu_data.db)",
                            "Identical interface to remote tool",
                            "Logs: '💾 Using local SQLite fallback'",
                        ],
                        "Summary": "Automatic fallback to local database when MCP server unavailable.",
                    },
                    "Error Handling": {
                        "Details": [
                            "Try-except for ConnectionError, RuntimeError, ValueError",
                            "Detailed error logging with SQLITE_TOOL_ID",
                            "Fallback disabled raises ConnectionError",
                            "RuntimeError if both strategies fail",
                            "Debug logging at each decision point",
                        ],
                        "Summary": "Comprehensive error handling with graceful degradation.",
                    },
                    "Summary": "Tool acquisition with automatic MCP/local fallback strategy.",
                },
                "LocalSQLiteQueryTool Class": {
                    "Details": [
                        "Inherits from LangChain BaseTool",
                        "args_schema: LocalSQLiteQueryInput (Pydantic model)",
                        "_run(query) - synchronous execution",
                        "_arun(query) - async wrapper (calls _run)",
                        "sqlite3.connect(DB_PATH) for local queries",
                        "Same result formatting as remote tool",
                        "ToolException on errors",
                    ],
                    "Summary": "LangChain-compatible local SQLite tool for fallback mode.",
                },
                "finish_gathering Tool": {
                    "Details": [
                        "@tool decorator from LangChain",
                        "def finish_gathering() - signals data gathering complete",
                        "Returns: 'Data gathering finished.'",
                        "Used in agentic loop to stop tool calling",
                        "Appended to tools list in generate_query_node",
                    ],
                    "Summary": "Control tool for terminating agentic query iteration.",
                },
                "Summary": "Tool acquisition module with remote/local strategy and control tools.",
            },
            "Tool Usage in generate_query_node": {
                "Node Setup": {
                    "Details": [
                        "tools = await get_sqlite_tools() - acquire MCP tools",
                        "sqlite_tool = next((tool for tool in tools if tool.name == 'sqlite_query'), None)",
                        "Validation: raise error if sqlite_query not found",
                        "tools.append(finish_gathering) - add control tool",
                        "llm_with_tools = llm.bind_tools(tools) - bind to Azure GPT-4o",
                        "MAX_TOOL_ITERATIONS limit (env configurable)",
                    ],
                    "Summary": "MCP tool integration with LLM tool binding and validation.",
                },
                "Agentic Loop Execution": {
                    "Details": [
                        "while tool_call_count < MAX_TOOL_ITERATIONS:",
                        "llm_response = await llm_with_tools.ainvoke(conversation_messages)",
                        "Check llm_response.tool_calls for tool invocations",
                        "Extract tool_name, tool_args, tool_call_id",
                        "Execute sqlite_tool.ainvoke({'query': sql_query})",
                        "Handle TextContent wrapper format with regex",
                        "Store (sql_query, result_text) in new_queries_and_results",
                        "Append ToolMessage to conversation_messages",
                        "LLM sees results and decides if more queries needed",
                    ],
                    "Summary": "Iterative agentic pattern with tool calling and result feedback.",
                },
                "Error Recovery": {
                    "Details": [
                        "Try-except around sqlite_tool.ainvoke()",
                        "Store errors in queries_and_results: (query, 'Error: ...')",
                        "Send error as ToolMessage back to LLM",
                        "LLM can retry with corrected query",
                        "Comprehensive logging with print__nodes_debug",
                        "GENERATE_QUERY_ID for log tracking",
                    ],
                    "Summary": "Error handling with LLM feedback for self-correction.",
                },
                "Finish Condition": {
                    "Details": [
                        "LLM calls finish_gathering tool when satisfied",
                        "Execute: await finish_gathering.ainvoke({})",
                        "Set finished=True and break loop",
                        "Alternative: LLM returns no tool_calls",
                        "Fallback: MAX_TOOL_ITERATIONS reached",
                    ],
                    "Summary": "Multiple exit strategies for agentic loop termination.",
                },
                "Summary": "Core agent node using MCP tools in iterative query execution pattern.",
            },
            "Summary": "Complete integration of MCP tools into LangGraph agentic workflow.",
        },
        "3. Deployment & Configuration": {
            "Standalone Architecture": {
                "Details": [
                    "czsu_mcp_server_sqlite/ is independent folder",
                    "NO imports from parent project",
                    "Own pyproject.toml with minimal dependencies",
                    "Can be moved to separate Git repository",
                    "Designed for microservice deployment",
                ],
                "Summary": "Self-contained MCP server with zero parent project dependencies.",
            },
            "Dependency Management": {
                "Production Dependencies": {
                    "Details": [
                        "fastmcp>=2.0.0 - MCP server framework (includes FastAPI, Uvicorn)",
                        "python-dotenv>=1.0.0 - Environment variable loading",
                        "starlette>=0.45.0 - Web framework (included with FastMCP)",
                        "libsql>=0.1.11 - Turso database client",
                        "sqlitecloud>=0.0.80 - SQLite Cloud client",
                        "docutils>=0.20.0,<0.22.0 - Fix for metadata issue",
                        "Total: 6 dependencies (vs 70+ in main project)",
                    ],
                    "Summary": "Minimal dependency footprint for focused MCP server functionality.",
                },
                "Development Dependencies": {
                    "Details": [
                        "pytest>=7.0.0 - Testing framework",
                        "httpx>=0.25.0 - HTTP client for testing",
                        "Install with: uv pip install .[dev]",
                    ],
                    "Summary": "Optional development tools for testing MCP server.",
                },
                "Package Management": {
                    "Details": [
                        "pyproject.toml (preferred) - PEP 621 compliant",
                        "requirements.txt (legacy) - backward compatibility",
                        "uv pip install . - recommended (10-100x faster)",
                        "pip install . - traditional fallback",
                        "FastMCP Cloud auto-detects pyproject.toml",
                    ],
                    "Summary": "Modern package management with uv support and cloud compatibility.",
                },
                "Summary": "Lightweight dependency management aligned with modern Python standards.",
            },
            "Local Development Setup": {
                "Setup Script (setup.bat)": {
                    "Details": [
                        "uv venv --python 3.11.9 - create virtual environment",
                        "uv pip install . - install dependencies",
                        "uv pip install .[dev] - install dev dependencies",
                        "copy .env.example .env - create environment file",
                        "Creates .vscode/settings.json for Python interpreter",
                        "Checks and validates environment setup",
                    ],
                    "Summary": "Automated setup script for local development environment.",
                },
                "Local Testing": {
                    "Details": [
                        "python main.py - start server on localhost:8100",
                        "curl http://localhost:8100/health - test health endpoint",
                        "FastMCP Client for tool testing",
                        "SSE transport for HTTP-based MCP protocol",
                        "Startup banner with configuration details",
                    ],
                    "Summary": "Local server startup with health checks and MCP client testing.",
                },
                "Summary": "Complete local development workflow with automated setup.",
            },
            "FastMCP Cloud Deployment": {
                "Platform Features": {
                    "Details": [
                        "Official managed platform for MCP servers",
                        "Free during beta period",
                        "Zero-config deployment (no railway.toml, Dockerfile, etc.)",
                        "Auto-detects dependencies from pyproject.toml",
                        "GitHub integration for CI/CD",
                        "Automatic HTTPS with SSL certificates",
                        "Server URL: https://project-name.fastmcp.app/mcp",
                    ],
                    "Summary": "Managed MCP hosting platform with automatic deployment pipeline.",
                },
                "Deployment Process": {
                    "Details": [
                        "1. Push code to GitHub repository",
                        "2. Sign in to fastmcp.cloud with GitHub",
                        "3. Create project with name and repo selection",
                        "4. Set entrypoint: main.py:mcp (critical!)",
                        "5. FastMCP Cloud auto-builds and deploys",
                        "6. Assigns URL: https://project-name.fastmcp.app",
                        "7. Update main app .env: MCP_SERVER_URL=...",
                    ],
                    "Summary": "7-step deployment process with automatic build and URL assignment.",
                },
                "Auto-Deployment Features": {
                    "Details": [
                        "Push to main branch triggers auto-redeploy",
                        "Pull requests create preview deployments",
                        "Manual redeploy via dashboard",
                        "Build logs in real-time",
                        "Deployment history with rollback",
                        "Environment variable management",
                    ],
                    "Summary": "Continuous deployment with GitHub integration and preview environments.",
                },
                "Monitoring & Diagnostics": {
                    "Details": [
                        "FastMCP Cloud dashboard for logs and status",
                        "Health endpoint monitoring",
                        "Build status and deployment history",
                        "Real-time server logs",
                        "Resource usage metrics",
                    ],
                    "Summary": "Comprehensive monitoring and diagnostics via cloud dashboard.",
                },
                "Summary": "Production deployment on FastMCP Cloud with zero-config CI/CD.",
            },
            "Environment Configuration": {
                "Main Project .env": {
                    "Details": [
                        "MCP_SERVER_URL - Remote MCP server URL",
                        "USE_LOCAL_SQLITE_FALLBACK - Enable local fallback (default: 1)",
                        "Example: MCP_SERVER_URL=https://project-name.fastmcp.app/mcp",
                        "Empty MCP_SERVER_URL triggers local mode",
                    ],
                    "Summary": "Main application configuration for MCP server connection.",
                },
                "MCP Server .env": {
                    "Details": [
                        "PORT - Server port (default: 8100)",
                        "DATABASE_TYPE - 'sqlitecloud' or 'turso'",
                        "SQLITE_CLOUD_CONNECTION_STRING - Cloud DB connection",
                        "TURSO_CONNECTION_STRING - Turso connection",
                        "DEBUG - Enable debug logging",
                    ],
                    "Summary": "MCP server configuration for database backend and logging.",
                },
                "Summary": "Environment-based configuration for flexible deployment modes.",
            },
            "Summary": "Complete deployment strategy with local, cloud, and hybrid modes.",
        },
        "4. MCP Protocol & Transport": {
            "Model Context Protocol (MCP)": {
                "Protocol Purpose": {
                    "Details": [
                        "Standard protocol for AI tool integration",
                        "Defines tool discovery, invocation, and result format",
                        "Language-agnostic specification",
                        "Enables AI agents to use external tools",
                        "Anthropic-initiated open standard",
                        "Specification: https://spec.modelcontextprotocol.io/",
                    ],
                    "Summary": "Standardized protocol for AI tool integration and invocation.",
                },
                "MCP Components": {
                    "Details": [
                        "Server: Exposes tools via MCP protocol",
                        "Client: Discovers and invokes tools",
                        "Tool: Function with schema and handler",
                        "Transport: Communication layer (SSE, stdio, HTTP)",
                        "Resource: Data source or capability",
                    ],
                    "Summary": "Core MCP protocol components for tool ecosystem.",
                },
                "Summary": "Open protocol standard for AI agent tool integration.",
            },
            "FastMCP Framework": {
                "Framework Features": {
                    "Details": [
                        "Python framework for building MCP servers",
                        "Based on FastAPI and Starlette",
                        "@mcp.tool() decorator for tool definition",
                        "@mcp.custom_route() for HTTP endpoints",
                        "Automatic schema generation from type hints",
                        "Built-in error handling and validation",
                        "SSE transport for web-based deployments",
                        "Version: >=2.0.0",
                    ],
                    "Summary": "Python MCP server framework with FastAPI foundation and decorators.",
                },
                "Tool Definition Pattern": {
                    "Details": [
                        "@mcp.tool() decorator automatically registers tools",
                        "Type hints for parameter validation",
                        "Context object for logging and metadata",
                        "Async/await support for I/O operations",
                        "Automatic OpenAPI documentation generation",
                        "Pydantic models for complex input schemas",
                    ],
                    "Summary": "Decorator-based tool definition with type safety and auto-documentation.",
                },
                "Summary": "FastAPI-based framework simplifying MCP server development.",
            },
            "Transport Layer": {
                "SSE (Server-Sent Events)": {
                    "Details": [
                        "HTTP-based unidirectional streaming",
                        "Transport: 'streamable_http' in MultiServerMCPClient",
                        "Used by FastMCP servers for web deployment",
                        "Compatible with standard HTTP infrastructure",
                        "No WebSocket complexity",
                        "Browser-compatible (can be tested in browser)",
                    ],
                    "Summary": "HTTP streaming transport for web-based MCP servers.",
                },
                "Transport Configuration": {
                    "Details": [
                        "Server: mcp.run(transport='sse')",
                        "Client: MultiServerMCPClient({'transport': 'streamable_http', 'url': ...})",
                        "Note: Use 'streamable_http' (underscore) not 'streamable-http' (hyphen)",
                        "Automatic protocol negotiation",
                        "Compatible with FastMCP Cloud deployment",
                    ],
                    "Summary": "SSE transport configuration for server and client communication.",
                },
                "Summary": "HTTP-based streaming transport layer for MCP protocol.",
            },
            "LangChain MCP Adapters": {
                "Integration Library": {
                    "Details": [
                        "Package: langchain-mcp-adapters>=0.1.12",
                        "from langchain_mcp_adapters.client import MultiServerMCPClient",
                        "Converts MCP tools to LangChain BaseTool instances",
                        "Handles protocol details automatically",
                        "Multi-server support in single client",
                        "Async tool invocation",
                    ],
                    "Summary": "Official LangChain adapter for MCP protocol integration.",
                },
                "MultiServerMCPClient": {
                    "Details": [
                        "Connects to multiple MCP servers simultaneously",
                        "Config: dict with server names and connection details",
                        "await client.get_tools() retrieves all tools",
                        "Returns LangChain-compatible tool objects",
                        "Automatic transport selection based on config",
                        "Error handling for connection failures",
                    ],
                    "Summary": "Multi-server MCP client for LangChain integration.",
                },
                "Summary": "LangChain integration library for MCP protocol support.",
            },
            "Summary": "MCP protocol implementation with FastMCP framework and LangChain adapters.",
        },
        "5. Database Backends": {
            "SQLite Cloud": {
                "Platform Features": {
                    "Details": [
                        "Cloud-hosted SQLite database",
                        "No local file management needed",
                        "Connection string format: sqlitecloud://host:port/db?apikey=xxx",
                        "API key authentication",
                        "Package: sqlitecloud>=0.0.80",
                        "Fully compatible SQLite dialect",
                    ],
                    "Summary": "Managed SQLite cloud hosting with API key authentication.",
                },
                "Connection Implementation": {
                    "Details": [
                        "import sqlitecloud",
                        "connection = sqlitecloud.connect(SQLITE_CLOUD_CONNECTION_STRING)",
                        "with connection: cursor = connection.cursor()",
                        "cursor.execute(query) / cursor.fetchall()",
                        "Automatic connection cleanup",
                        "Error handling for import and connection",
                    ],
                    "Summary": "SQLite Cloud client library for cloud database access.",
                },
                "Summary": "Cloud-hosted SQLite backend with managed infrastructure.",
            },
            "Turso (libSQL)": {
                "Platform Features": {
                    "Details": [
                        "Distributed SQLite with edge replication",
                        "libSQL fork of SQLite optimized for edge",
                        "Connection string with embedded auth token",
                        "Multi-region data distribution",
                        "Package: libsql>=0.1.11",
                        "SQLite compatibility maintained",
                    ],
                    "Summary": "Edge-optimized distributed SQLite with global replication.",
                },
                "Connection Implementation": {
                    "Details": [
                        "import libsql",
                        "Parse connection string with urllib.parse",
                        "Extract auth_token from query parameters",
                        "connection = libsql.connect(url, auth_token=auth_token)",
                        "cursor = connection.execute(query)",
                        "result = cursor.fetchall()",
                    ],
                    "Summary": "Turso libSQL client with connection string parsing and auth.",
                },
                "Summary": "Distributed SQLite backend with edge computing support.",
            },
            "Database Abstraction": {
                "Details": [
                    "get_db_connection() factory function",
                    "DATABASE_TYPE env var selects backend",
                    "Identical query interface for both backends",
                    "Backend-specific cursor handling",
                    "Connection error validation",
                    "ImportError for missing packages",
                ],
                "Summary": "Unified database interface supporting multiple SQLite backends.",
            },
            "Summary": "Multi-backend database support with cloud and edge deployment options.",
        },
        "6. Documentation & Testing": {
            "Documentation Files": {
                "README.md": {
                    "Details": [
                        "Quick start guide",
                        "Local development setup",
                        "FastMCP Cloud deployment instructions",
                        "API endpoints documentation",
                        "Environment variables reference",
                        "Troubleshooting guide",
                    ],
                    "Summary": "Main documentation with setup and deployment guides.",
                },
                "DEPLOYMENT.md": {
                    "Details": [
                        "Pre-deployment checklist",
                        "Step-by-step FastMCP Cloud deployment",
                        "Testing and verification procedures",
                        "Troubleshooting common issues",
                        "Auto-deployment features",
                        "Monitoring and rollback strategies",
                    ],
                    "Summary": "Comprehensive deployment guide with troubleshooting.",
                },
                "PACKAGE_MANAGEMENT.md": {
                    "Details": [
                        "Dependency management with pyproject.toml",
                        "uv vs pip installation methods",
                        "Comparison with main project dependencies",
                        "Updating dependencies guide",
                        "FastMCP Cloud auto-detection",
                    ],
                    "Summary": "Package management guide comparing uv and pip workflows.",
                },
                "Summary": "Complete documentation covering setup, deployment, and maintenance.",
            },
            "Testing Strategy": {
                "Local Testing": {
                    "Details": [
                        "python main.py - start local server",
                        "curl /health - verify server health",
                        "FastMCP Client integration tests",
                        "Manual SQL query testing",
                        "Database connectivity validation",
                    ],
                    "Summary": "Local testing workflow with health checks and client integration.",
                },
                "Integration Testing": {
                    "Details": [
                        "tests/other/test_fastmcp_integration.py",
                        "Test MCP server connection",
                        "Test tool discovery and invocation",
                        "Verify result formatting",
                        "pytest framework for automated tests",
                    ],
                    "Summary": "Automated integration tests for MCP client-server interaction.",
                },
                "Production Verification": {
                    "Details": [
                        "Health endpoint monitoring",
                        "Main app connection tests",
                        "Query execution validation",
                        "Fallback mechanism testing",
                        "FastMCP Cloud dashboard logs",
                    ],
                    "Summary": "Production environment verification and monitoring.",
                },
                "Summary": "Multi-level testing strategy from local to production deployment.",
            },
            "Summary": "Comprehensive documentation and testing for reliable MCP server operation.",
        },
        "7. Error Handling & Resilience": {
            "Connection Resilience": {
                "Automatic Fallback": {
                    "Details": [
                        "Primary: Remote MCP server via MultiServerMCPClient",
                        "Fallback: Local SQLite database via LocalSQLiteQueryTool",
                        "USE_LOCAL_SQLITE_FALLBACK env var controls fallback",
                        "Seamless transition without application restart",
                        "Identical tool interface for both modes",
                        "Logging indicates active mode (remote/local)",
                    ],
                    "Summary": "Automatic failover from remote MCP to local SQLite database.",
                },
                "Connection Validation": {
                    "Details": [
                        "Health check before server startup",
                        "Test database query (SELECT 1)",
                        "Environment variable validation",
                        "ImportError handling for missing packages",
                        "ConnectionError for unreachable databases",
                        "Detailed error messages with troubleshooting hints",
                    ],
                    "Summary": "Pre-startup validation and runtime health monitoring.",
                },
                "Summary": "Multi-level connection resilience with automatic failover.",
            },
            "Query Error Handling": {
                "Server-Side": {
                    "Details": [
                        "Try-except around database query execution",
                        "Return error message to client (not exception)",
                        "Logging with await ctx.info() and ctx.error()",
                        "Database-specific error handling",
                        "Connection cleanup in finally block",
                    ],
                    "Summary": "Graceful error handling with client-friendly error messages.",
                },
                "Client-Side (LangGraph)": {
                    "Details": [
                        "Try-except around sqlite_tool.ainvoke()",
                        "Store errors in queries_and_results list",
                        "Send error as ToolMessage to LLM",
                        "LLM can retry with corrected query",
                        "Comprehensive logging for debugging",
                        "Error tracking with run_id and thread_id",
                    ],
                    "Summary": "Error recovery with LLM feedback loop for self-correction.",
                },
                "Summary": "End-to-end error handling from server to LLM-based recovery.",
            },
            "Operational Safeguards": {
                "Details": [
                    "Read-only database access (no writes/deletes)",
                    "Query result size limits",
                    "Connection timeout handling",
                    "Resource cleanup with context managers",
                    "Graceful degradation when MCP unavailable",
                    "Health monitoring for early issue detection",
                ],
                "Summary": "Multiple safeguards for safe and reliable operation.",
            },
            "Summary": "Comprehensive error handling and resilience strategies.",
        },
        "8. Integration Points": {
            "Main Application Integration": {
                "Environment Configuration": {
                    "Details": [
                        "MCP_SERVER_URL in main project .env",
                        "USE_LOCAL_SQLITE_FALLBACK for fallback control",
                        "No code changes needed to switch modes",
                        "Empty MCP_SERVER_URL triggers local mode",
                        "FastMCP Cloud URL format: https://project.fastmcp.app/mcp",
                    ],
                    "Summary": "Environment-driven configuration for deployment flexibility.",
                },
                "Tool Discovery": {
                    "Details": [
                        "await get_sqlite_tools() in generate_query_node",
                        "MultiServerMCPClient.get_tools() for remote",
                        "Returns LangChain BaseTool instances",
                        "Tool name: 'sqlite_query' (consistent across modes)",
                        "Validation: check sqlite_query tool exists",
                    ],
                    "Summary": "Automatic tool discovery and validation in agent initialization.",
                },
                "LLM Tool Binding": {
                    "Details": [
                        "llm.bind_tools(tools) for Azure GPT-4o",
                        "LLM receives tool schema automatically",
                        "Tool calls in llm_response.tool_calls",
                        "Extract tool_name, tool_args, tool_call_id",
                        "Execute via tool.ainvoke(args)",
                    ],
                    "Summary": "LangChain tool binding for LLM tool calling capability.",
                },
                "Summary": "Seamless integration into LangGraph agent workflow.",
            },
            "Data Flow": {
                "Query Execution Flow": {
                    "Details": [
                        "1. User prompt → generate_query_node",
                        "2. LLM generates SQL query",
                        "3. LLM calls sqlite_query tool",
                        "4. Tool routed to MCP server (or local)",
                        "5. Database executes query",
                        "6. Results formatted and returned",
                        "7. ToolMessage sent back to LLM",
                        "8. LLM analyzes results, decides next action",
                        "9. Repeat or finish_gathering",
                    ],
                    "Summary": "Complete data flow from user prompt to query results.",
                },
                "State Management": {
                    "Details": [
                        "queries_and_results list in DataAnalysisState",
                        "Each iteration appends (query, result) tuples",
                        "LLM sees conversation_messages history",
                        "ToolMessage preserves tool_call_id for tracking",
                        "State persisted in PostgreSQL checkpointer",
                    ],
                    "Summary": "Stateful query history for iterative refinement.",
                },
                "Summary": "End-to-end data flow with state management and iteration.",
            },
            "Summary": "Complete integration architecture from environment to execution.",
        },
        "Summary": "Production-ready MCP server providing SQLite query capabilities via Model Context Protocol, integrated with LangGraph agent through official LangChain adapters, supporting cloud and local deployment modes with automatic failover.",
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="CZSU MCP Server Mindmap v3")
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

"""MCP server implementation for SQLite queries.

This module implements a Model Context Protocol (MCP) client that provides
a SQLite query tool for data analysis. It supports two modes:

1. Remote MCP Server Mode (Primary):
   - Connects to a remote FastMCP server (e.g., deployed on FastMCP Cloud)
   - Uses FastMCP Client with SSE transport
   - Configured via MCP_SERVER_URL environment variable

2. Local SQLite Fallback Mode (Secondary):
   - Direct SQLite access when remote server is unavailable
   - Uses local database file
   - Automatic fallback with logging

Configuration (via .env):
    MCP_SERVER_URL: URL of the remote MCP server (e.g., http://localhost:8100/mcp or https://your-project.fastmcp.app/mcp)
    USE_LOCAL_SQLITE_FALLBACK: Enable/disable fallback to local SQLite (1/0)
"""

import os
import sqlite3
from pathlib import Path
from typing import List

from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

# Get base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Import debug functions from utils
from api.utils.debug import print__tools_debug

# Database path (for local fallback)
DB_PATH = BASE_DIR / "data" / "czsu_data.db"

# Debug constant for tool ID
SQLITE_TOOL_ID = 21  # Static ID for SQLiteQueryTool

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").strip()
USE_LOCAL_SQLITE_FALLBACK = int(
    os.getenv("USE_LOCAL_SQLITE_FALLBACK", "1").split("#")[0].strip()
)


class SQLiteQueryInput(BaseModel):
    """Input schema for SQLite query tool."""

    query: str = Field(description="SQL query to execute against the SQLite database")


class SQLiteQueryTool(BaseTool):
    """Tool for executing SQL queries against a SQLite database.

    Supports two modes:
    1. Remote MCP Server: FastMCP Client connection to remote server
    2. Local SQLite: Direct database access (fallback)
    """

    name: str = "sqlite_query"
    description: str = (
        "Execute SQL query on the SQLite database. Input should be a valid SQL query string."
    )
    args_schema: type[BaseModel] = SQLiteQueryInput

    # Class variable to track which mode is being used
    _using_remote_mcp: bool = False

    async def _execute_via_remote_mcp_async(self, query: str) -> str:
        """Execute query via remote FastMCP server."""
        try:
            # Import FastMCP Client
            from fastmcp import Client

            print__tools_debug(
                f"{SQLITE_TOOL_ID}: Using REMOTE FastMCP server at: {MCP_SERVER_URL}"
            )

            # Create FastMCP client with SSE transport
            client = Client(MCP_SERVER_URL)

            # Connect and execute query
            async with client:
                # Call the sqlite_query tool on the remote server
                result = await client.call_tool("sqlite_query", {"query": query})
                return str(result.data) if hasattr(result, "data") else str(result)

        except Exception as e:
            raise Exception(f"Remote FastMCP request failed: {str(e)}")

    def _execute_via_local_sqlite(self, query: str) -> str:
        """Execute query via local SQLite database."""
        print__tools_debug(
            f"{SQLITE_TOOL_ID}: Using LOCAL SQLite fallback at: {DB_PATH}"
        )

        # Execute SQL query directly
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()

        # Format the result
        if not result:
            result_value = "No results found"
        elif len(result) == 1 and len(result[0]) == 1:
            result_value = str(result[0][0])
        else:
            result_value = str(result)

        return result_value

    def _run(self, query: str) -> str:
        """Execute the SQL query and return results (synchronous wrapper)."""
        # Import asyncio to run async code
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async version
        return loop.run_until_complete(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute the SQL query asynchronously."""
        try:
            # Debug print query
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: ====================================="
            )
            print__tools_debug(f"{SQLITE_TOOL_ID}: Executing query:")
            print__tools_debug(f"{SQLITE_TOOL_ID}: {query}")
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: ====================================="
            )

            # Try remote MCP server first (if configured)
            if MCP_SERVER_URL:
                try:
                    result_value = await self._execute_via_remote_mcp_async(query)
                    self._using_remote_mcp = True
                except Exception as e:
                    print__tools_debug(
                        f"{SQLITE_TOOL_ID}: Remote FastMCP failed: {str(e)}"
                    )

                    # Fall back to local SQLite if enabled
                    if USE_LOCAL_SQLITE_FALLBACK:
                        print__tools_debug(
                            f"{SQLITE_TOOL_ID}: Falling back to LOCAL SQLite..."
                        )
                        result_value = self._execute_via_local_sqlite(query)
                        self._using_remote_mcp = False
                    else:
                        raise ToolException(
                            f"Remote MCP server unavailable and fallback disabled: {str(e)}"
                        )
            else:
                # No remote server configured, use local SQLite
                print__tools_debug(
                    f"{SQLITE_TOOL_ID}: No MCP_SERVER_URL configured, using LOCAL SQLite"
                )
                result_value = self._execute_via_local_sqlite(query)
                self._using_remote_mcp = False

            # Debug print result
            print__tools_debug(f"{SQLITE_TOOL_ID}: Query result:")
            print__tools_debug(f"{SQLITE_TOOL_ID}: {result_value}")
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: ====================================="
            )

            return result_value

        except Exception as e:
            raise ToolException(f"Query error: {str(e)}")


async def create_mcp_server() -> List[BaseTool]:
    """Create and configure the MCP server with tools.

    This function creates the SQLite query tool and logs which mode
    (remote MCP or local SQLite) will be used based on configuration.

    Returns:
        A list of LangChain tools that can be used with the MCP server
    """
    # Log configuration on startup
    if MCP_SERVER_URL:
        print__tools_debug(f"{SQLITE_TOOL_ID}: =====================================")
        print__tools_debug(f"{SQLITE_TOOL_ID}: 🌐 MCP SERVER CONFIGURED")
        print__tools_debug(f"{SQLITE_TOOL_ID}: Remote MCP URL: {MCP_SERVER_URL}")
        print__tools_debug(
            f"{SQLITE_TOOL_ID}: Fallback enabled: {bool(USE_LOCAL_SQLITE_FALLBACK)}"
        )
        print__tools_debug(f"{SQLITE_TOOL_ID}: =====================================")
        print(f"🌐 Using REMOTE MCP server at: {MCP_SERVER_URL}")
        if USE_LOCAL_SQLITE_FALLBACK:
            print(f"   (with local SQLite fallback)")
    else:
        print__tools_debug(f"{SQLITE_TOOL_ID}: =====================================")
        print__tools_debug(f"{SQLITE_TOOL_ID}: 💾 LOCAL SQLITE MODE")
        print__tools_debug(f"{SQLITE_TOOL_ID}: Database path: {DB_PATH}")
        print__tools_debug(f"{SQLITE_TOOL_ID}: =====================================")
        print(f"💾 Using LOCAL SQLite database at: {DB_PATH}")

    # Create and return the SQLite query tool
    return [SQLiteQueryTool()]

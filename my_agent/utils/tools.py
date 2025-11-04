# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sqlite3
from pathlib import Path
from typing import List

from langchain_core.tools import tool
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

# Official LangChain MCP adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Get base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Import debug functions
from api.utils.debug import print__tools_debug

# Database path (for local fallback)
DB_PATH = BASE_DIR / "data" / "czsu_data.db"

# Debug constant for tool ID
SQLITE_TOOL_ID = 21  # Static ID for SQLite tools

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").strip()
USE_LOCAL_SQLITE_FALLBACK = os.getenv("USE_LOCAL_SQLITE_FALLBACK", "1").split("#")[
    0
].strip().strip('"').lower() in ("1", "true", "yes")


# ==============================================================================
# LANGCHAIN TOOLS
# ==============================================================================
@tool
def finish_gathering():
    """Call this tool when you have gathered sufficient data to answer the user's question."""
    return "Data gathering finished."


# ==============================================================================
# LOCAL SQLITE FALLBACK TOOL
# ==============================================================================
class LocalSQLiteQueryInput(BaseModel):
    """Input schema for local SQLite query tool."""

    query: str = Field(
        description="SQL query to execute against the local SQLite database"
    )


class LocalSQLiteQueryTool(BaseTool):
    """Local SQLite query tool (fallback when MCP server is unavailable)."""

    name: str = "sqlite_query"
    description: str = (
        "Execute SQL query on the local SQLite database. Input should be a valid SQL query string."
    )
    args_schema: type[BaseModel] = LocalSQLiteQueryInput

    def _execute_query(self, query: str) -> str:
        """Execute query against local SQLite database."""
        print__tools_debug(f"{SQLITE_TOOL_ID}: 💾 Using LOCAL SQLite: {DB_PATH}")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()

        # Format result (same as remote server)
        if not result:
            return "No results found"
        elif len(result) == 1 and len(result[0]) == 1:
            return str(result[0][0])
        else:
            return str(result)

    def _run(self, query: str) -> str:
        """Execute SQL query synchronously."""
        try:
            print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")
            print__tools_debug(f"{SQLITE_TOOL_ID}: Executing local query: {query}")
            print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")

            result = self._execute_query(query)

            print__tools_debug(f"{SQLITE_TOOL_ID}: ✅ Local result: {result}")
            print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")

            return result
        except (sqlite3.Error, OSError) as e:
            raise ToolException(f"Local SQLite query error: {str(e)}") from e

    async def _arun(self, query: str) -> str:
        """Execute SQL query asynchronously."""
        return self._run(query)


# ==============================================================================
# MCP TOOLS WITH FALLBACK
# ==============================================================================
async def get_sqlite_tools() -> List[BaseTool]:
    """Get SQLite query tools using official LangChain MCP adapters with local fallback.

    This function implements the official LangChain MCP pattern:
    1. Primary: Try to connect to remote MCP server using MultiServerMCPClient
    2. Fallback: Use local SQLite tool if remote MCP server is unavailable

    Returns:
        List of LangChain tools for SQL querying
    """
    print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")
    print__tools_debug(f"{SQLITE_TOOL_ID}: 🔧 INITIALIZING SQLITE TOOLS")
    print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")

    # Try MCP server first (if configured)
    if MCP_SERVER_URL:
        try:
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: 🌐 Attempting MCP connection: {MCP_SERVER_URL}"
            )

            # Use official LangChain MCP adapters with streamable_http transport
            # Note: Use "streamable_http" (underscore) not "streamable-http" (hyphen)
            client = MultiServerMCPClient(
                {
                    "sqlite": {
                        "transport": "streamable_http",  # Streamable HTTP for remote FastMCP servers
                        "url": MCP_SERVER_URL,
                    }
                }
            )

            # Get tools from MCP server
            tools = await client.get_tools()

            print__tools_debug(
                f"{SQLITE_TOOL_ID}: ✅ MCP server connected successfully"
            )
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: 📝 Available tools: {[tool.name for tool in tools]}"
            )
            print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")
            print(f"🌐 SQLite Tools: Remote MCP server at {MCP_SERVER_URL}")
            print(f"   📝 Tools loaded: {[tool.name for tool in tools]}")

            return tools

        except (ConnectionError, RuntimeError, ValueError) as e:
            print__tools_debug(
                f"{SQLITE_TOOL_ID}: ❌ MCP server connection failed: {str(e)}"
            )

            if USE_LOCAL_SQLITE_FALLBACK:
                print__tools_debug(f"{SQLITE_TOOL_ID}: ↩️  Falling back to local SQLite")
            else:
                print__tools_debug(
                    f"{SQLITE_TOOL_ID}: ❌ Fallback disabled, raising error"
                )
                raise ConnectionError(
                    f"MCP server unavailable and fallback disabled: {str(e)}"
                ) from e

    # Use local SQLite fallback
    if not MCP_SERVER_URL or USE_LOCAL_SQLITE_FALLBACK:
        if not MCP_SERVER_URL:
            print__tools_debug(f"{SQLITE_TOOL_ID}: 💾 No MCP_SERVER_URL configured")

        print__tools_debug(f"{SQLITE_TOOL_ID}: 💾 Using local SQLite fallback")
        print__tools_debug(f"{SQLITE_TOOL_ID}: 📁 Database path: {DB_PATH}")
        print__tools_debug(f"{SQLITE_TOOL_ID}: {'='*50}")
        print(f"💾 SQLite Tools: Local database at {DB_PATH}")

        return [LocalSQLiteQueryTool()]

    # Should not reach here
    raise RuntimeError("No SQLite tools available (both MCP and local fallback failed)")

"""MCP server implementation for SQLite queries.

This module implements a Model Context Protocol (MCP) server that provides
a SQLite query tool for data analysis.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult as MCPToolResult
from langchain.tools import BaseTool
from langchain_core.tools import ToolException

# Get base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Database path
DB_PATH = BASE_DIR / "data" / "czsu_data.db"

class SQLiteQueryTool(BaseTool):
    """Tool for executing SQL queries against a SQLite database."""
    
    name: str = "sqlite_query"
    description: str = "Execute SQL query on the SQLite database"
    
    def _run(self, query: str) -> str:
        """Execute the SQL query and return results."""
        try:
            # Execute SQL query
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
            
        except Exception as e:
            raise ToolException(f"Query error: {str(e)}")
    
    async def _arun(self, query: str) -> str:
        """Execute the SQL query asynchronously."""
        return self._run(query)

async def create_mcp_server() -> List[BaseTool]:
    """Create and configure the MCP server with tools.
    
    Returns:
        A list of LangChain tools that can be used with the MCP server
    """
    # Create and return the SQLite query tool
    return [SQLiteQueryTool()]

if __name__ == "__main__":
    # Create and run the server
    server = FastMCP("sqlite_server")
    server.run() 
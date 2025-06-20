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

# Debug constant for tool ID
SQLITE_TOOL_ID = 21  # Static ID for SQLiteQueryTool

def print__mcp_debug(msg: str) -> None:
    """Print MCP debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[MCP-DEBUG] {msg}")
        import sys
        sys.stdout.flush()

class SQLiteQueryTool(BaseTool):
    """Tool for executing SQL queries against a SQLite database."""
    
    name: str = "sqlite_query"
    description: str = "Execute SQL query on the SQLite database"
    
    def _run(self, query: str) -> str:
        """Execute the SQL query and return results."""
        try:
            # Debug print query
            print__mcp_debug(f"🗄️ {SQLITE_TOOL_ID}: =====================================")
            print__mcp_debug(f"⚡ {SQLITE_TOOL_ID}: Executing query:")
            print__mcp_debug(f"📝 {SQLITE_TOOL_ID}: {query}")
            print__mcp_debug(f"🗄️ {SQLITE_TOOL_ID}: =====================================")

            # Execute SQL query
            with sqlite3.connect(DB_PATH) as conn:
                results = conn.execute(query).fetchall()
                
                if not results:
                    result_value = "No results found."
                else:
                    # Get column names
                    column_names = [description[0] for description in conn.description]
                    
                    # Format results
                    formatted_results = []
                    for row in results:
                        row_dict = dict(zip(column_names, row))
                        formatted_results.append(row_dict)
                    
                    result_value = f"Found {len(formatted_results)} result(s):\n" + str(formatted_results)

            # Debug print result
            print__mcp_debug(f"📊 {SQLITE_TOOL_ID}: Query result:")
            print__mcp_debug(f"✅ {SQLITE_TOOL_ID}: {result_value}")
            print__mcp_debug(f"🗄️ {SQLITE_TOOL_ID}: =====================================")
            
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

class AgentMCPServer:
    """MCP Server that exposes agent functionality."""
    
    def __init__(self, agent_executor):
        """Initialize the MCP server with agent executor."""
        self.agent_executor = agent_executor
        self.server = FastMCP("czsu-multi-agent-text-to-sql")
        self._setup_handlers()
        print__mcp_debug("🚀 MCP Server initialized")
        
    def _setup_handlers(self):
        """Setup MCP server handlers for different message types."""
        print__mcp_debug("🔧 Setting up MCP handlers")
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[BaseTool]:
            """List available tools for MCP clients."""
            print__mcp_debug("📋 MCP Client requested tool list")
            
            return [
                SQLiteQueryTool()
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[MCPToolResult]:
            """Handle tool execution requests from MCP clients."""
            print__mcp_debug(f"🔧 MCP Tool execution request: {name}")
            print__mcp_debug(f"📥 MCP Arguments: {arguments}")
            
            if name == "analyze_czech_statistical_data":
                try:
                    question = arguments.get("question", "")
                    thread_id = arguments.get("thread_id")
                    
                    print__mcp_debug(f"🔍 MCP Processing question: {question[:100]}...")
                    
                    if not question:
                        error_msg = "Question parameter is required"
                        print__mcp_debug(f"❌ MCP Error: {error_msg}")
                        return [MCPToolResult(type="text", text=f"Error: {error_msg}")]
                    
                    # Create configuration for agent execution
                    config = {"configurable": {"thread_id": thread_id or f"mcp_{uuid.uuid4()}"}}
                    
                    print__mcp_debug(f"🤖 MCP Invoking agent with thread_id: {config['configurable']['thread_id']}")
                    
                    # Execute the agent
                    result = await self.agent_executor.ainvoke(
                        {"prompt": question},
                        config=config
                    )
                    
                    # Extract the final answer
                    final_answer = result.get("final_answer", "No answer generated")
                    
                    print__mcp_debug(f"✅ MCP Analysis completed successfully")
                    print__mcp_debug(f"📄 MCP Result length: {len(str(final_answer))} characters")
                    
                    return [MCPToolResult(type="text", text=str(final_answer))]
                    
                except Exception as e:
                    error_msg = f"Error during analysis: {str(e)}"
                    print__mcp_debug(f"❌ MCP Tool execution error: {error_msg}")
                    return [MCPToolResult(type="text", text=error_msg)]
            else:
                error_msg = f"Unknown tool: {name}"
                print__mcp_debug(f"❌ MCP Unknown tool requested: {name}")
                return [MCPToolResult(type="text", text=error_msg)]
    
    async def run_stdio(self):
        """Run the MCP server using stdio transport."""
        print__mcp_debug("🔄 Starting MCP server with stdio transport")
        
        await self.server.run()

def create_mcp_server(agent_executor):
    """Factory function to create MCP server instance."""
    print__mcp_debug("🏭 Creating MCP server instance")
    return AgentMCPServer(agent_executor)

async def main():
    """Main entry point for MCP server."""
    try:
        print__mcp_debug("🚀 Starting CZSU Multi-Agent Text-to-SQL MCP Server")
        
        # Create the agent
        from my_agent import create_graph
        from my_agent.utils.postgres_checkpointer import get_postgres_checkpointer
        
        print__mcp_debug("🔧 Initializing PostgreSQL checkpointer")
        checkpointer = await get_postgres_checkpointer()
        
        print__mcp_debug("🤖 Creating agent graph")
        agent_executor = create_graph(checkpointer)
        
        print__mcp_debug("🔧 Creating MCP server")
        mcp_server = create_mcp_server(agent_executor)
        
        print__mcp_debug("🎬 Starting MCP server...")
        await mcp_server.run_stdio()
        
    except KeyboardInterrupt:
        print__mcp_debug("⏹️ MCP Server interrupted by user")
    except Exception as e:
        print__mcp_debug(f"❌ MCP Server error: {e}")
        import traceback
        print__mcp_debug(f"🔍 MCP Error traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Create and run the server
    print__mcp_debug(f"{SQLITE_TOOL_ID}: =====================================")
    print__mcp_debug(f"{SQLITE_TOOL_ID}: STARTING MCP SERVER")
    asyncio.run(main()) 
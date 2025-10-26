"""
Quick test script for FastMCP server and client integration.

This script tests:
1. MCP server starts correctly
2. Client can connect via FastMCP
3. Tools are listed properly
4. sqlite_query tool works
"""

import asyncio
from fastmcp import Client


async def test_mcp_connection():
    """Test connection to FastMCP server."""

    # Configuration
    server_url = "http://localhost:8100/sse"  # Important: /sse path is required for SSE transport!

    print("=" * 60)
    print("🧪 Testing FastMCP Server Connection")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print()

    try:
        # Create FastMCP client
        print("1. Creating FastMCP client...")
        client = Client(server_url)

        async with client:
            print("✓ Connected successfully!")
            print()

            # Test ping
            print("2. Testing server ping...")
            await client.ping()
            print("✓ Server is responsive!")
            print()

            # List tools
            print("3. Listing available tools...")
            tools = await client.list_tools()
            print(f"✓ Found {len(tools)} tool(s):")
            for tool in tools:
                print(f"   - {tool.name}: {tool.description}")
            print()

            # Call sqlite_query tool
            print("4. Testing sqlite_query tool...")
            query = "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5"
            print(f"   Query: {query}")

            result = await client.call_tool("sqlite_query", {"query": query})
            print(f"✓ Query executed successfully!")
            print(f"   Result: {result.data if hasattr(result, 'data') else result}")
            print()

            # Try another query
            print("5. Testing count query...")
            count_query = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            print(f"   Query: {count_query}")

            result = await client.call_tool("sqlite_query", {"query": count_query})
            print(f"✓ Query executed successfully!")
            print(f"   Result: {result.data if hasattr(result, 'data') else result}")
            print()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Test failed: {str(e)}")
        print("=" * 60)
        print()
        print("Make sure:")
        print("1. MCP server is running: cd czsu_mcp_server_sqlite && python main.py")
        print(f"2. Server is accessible at: {server_url}")
        print("3. Database file exists in czsu_mcp_server_sqlite/data/czsu_data.db")


if __name__ == "__main__":
    print()
    print("FastMCP Integration Test")
    print()
    print("Prerequisites:")
    print("- MCP server must be running on http://localhost:8100/sse")
    print("- Start it with: cd czsu_mcp_server_sqlite && python main.py")
    print()

    asyncio.run(test_mcp_connection())

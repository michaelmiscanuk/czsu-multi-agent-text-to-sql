"""
Simple test script for MCP tools - tests cloud and local separately.

Tests:
1. Cloud MCP connection (no fallback)
2. Local SQLite connection

Usage:
    python test_mcp_simple.py
"""

import os
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))


async def test_cloud_mcp():
    """Test cloud MCP connection without fallback."""
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Cloud MCP Connection (No Fallback)")
    print("=" * 60)

    # Set environment for cloud-only mode
    os.environ["USE_LOCAL_SQLITE_FALLBACK"] = "0"

    # Reload tools module to pick up new environment
    import importlib
    from my_agent.utils import tools

    importlib.reload(tools)

    try:
        print("📡 Attempting cloud MCP connection...")
        tools_list = await tools.get_sqlite_tools()

        print(f"✅ Got {len(tools_list)} tool(s)")
        for tool in tools_list:
            print(f"   - {tool.name}: {tool.__class__.__name__}")

        # Test with a simple query
        print("\n📊 Testing query: SELECT 1")
        sqlite_tool = tools_list[0]

        # MCP tools use dict input
        if "StructuredTool" in sqlite_tool.__class__.__name__:
            result = await sqlite_tool.ainvoke({"query": "SELECT 1"})
        else:
            result = await sqlite_tool.ainvoke("SELECT 1")

        print(f"   Result: {result}")

        # Test table listing
        print(
            "\n📋 Testing query: SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"
        )
        if "StructuredTool" in sqlite_tool.__class__.__name__:
            result = await sqlite_tool.ainvoke(
                {"query": "SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"}
            )
        else:
            result = await sqlite_tool.ainvoke(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"
            )

        print(f"   Result: {result}")

        print("\n✅ CLOUD MCP: PASS")
        return True

    except (ImportError, AttributeError, RuntimeError, ConnectionError) as e:
        print(f"\n❌ CLOUD MCP: FAIL - {e}")
        return False


async def test_local_sqlite():
    """Test local SQLite connection."""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Local SQLite Connection")
    print("=" * 60)

    # Set environment for local-only mode
    os.environ["MCP_SERVER_URL"] = ""
    os.environ["USE_LOCAL_SQLITE_FALLBACK"] = "1"

    # Reload tools module to pick up new environment
    import importlib
    from my_agent.utils import tools

    importlib.reload(tools)

    try:
        print("💾 Using local SQLite database...")
        tools_list = await tools.get_sqlite_tools()

        print(f"✅ Got {len(tools_list)} tool(s)")
        for tool in tools_list:
            print(f"   - {tool.name}: {tool.__class__.__name__}")

        # Test with a simple query
        print("\n📊 Testing query: SELECT 1")
        sqlite_tool = tools_list[0]
        result = await sqlite_tool.ainvoke("SELECT 1")
        print(f"   Result: {result}")

        # Test table listing
        print(
            "\n📋 Testing query: SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"
        )
        result = await sqlite_tool.ainvoke(
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 3"
        )
        print(f"   Result: {result}")

        print("\n✅ LOCAL SQLITE: PASS")
        return True

    except (ImportError, AttributeError, RuntimeError, ConnectionError) as e:
        print(f"\n❌ LOCAL SQLITE: FAIL - {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 MCP Tools Test Suite")
    print("=" * 60)

    # Run tests
    cloud_result = await test_cloud_mcp()
    local_result = await test_local_sqlite()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Cloud MCP: {'✅ PASS' if cloud_result else '❌ FAIL'}")
    print(f"Local SQLite: {'✅ PASS' if local_result else '❌ FAIL'}")
    print("=" * 60)

    if cloud_result and local_result:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

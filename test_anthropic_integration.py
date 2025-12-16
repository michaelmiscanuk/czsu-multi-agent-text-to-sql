"""Test script to verify Anthropic model integration in nodes.py

This script tests the get_configured_llm helper function with different model types
to ensure proper LLM initialization and tool binding configuration, including
the new tools parameter functionality.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Import the helper function
from my_agent.utils.nodes import get_configured_llm


# Define a simple test tool
@tool
def test_calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def test_model_configuration(model_type: str):
    """Test LLM configuration for a specific model type"""
    print(f"\n{'='*60}")
    print(f"Testing model_type: {model_type}")
    print(f"{'='*60}")

    try:
        # Test without tools
        llm, use_bind_tools = get_configured_llm(model_type)
        print(f"✅ LLM initialized successfully (without tools)")
        print(f"   LLM type: {type(llm).__name__}")
        print(f"   Use bind_tools: {use_bind_tools}")

        # Test with tools
        llm_with_tools, use_bind_tools = get_configured_llm(
            model_type, tools=[test_calculator]
        )
        print(f"✅ LLM initialized successfully (with tools)")
        print(f"   LLM type: {type(llm_with_tools).__name__}")
        print(f"   Use bind_tools: {use_bind_tools}")
        if use_bind_tools:
            print(f"   Tools bound to LLM: Yes")
        else:
            print(f"   Tools bound to LLM: No (will pass to ainvoke)")

        # Test a simple invocation
        response = llm.invoke("Say 'Hello' in one word")
        print(f"   Test response: {response.content[:50]}...")

        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


async def test_model_configuration_async(model_type: str):
    """Test async LLM configuration for a specific model type"""
    print(f"\n{'='*60}")
    print(f"Testing ASYNC model_type: {model_type}")
    print(f"{'='*60}")

    try:
        # Test with tools
        llm_with_tools, use_bind_tools = get_configured_llm(
            model_type, tools=[test_calculator]
        )
        print(f"✅ LLM initialized successfully")
        print(f"   LLM type: {type(llm_with_tools).__name__}")
        print(f"   Use bind_tools: {use_bind_tools}")

        # Test async invocation
        response = await llm_with_tools.ainvoke("Say 'Hello' in one word")
        print(f"   Test async response: {response.content[:50]}...")

        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run tests for all supported model types"""
    print("\n" + "=" * 60)
    print("TESTING LLM CONFIGURATION HELPER FUNCTION")
    print("WITH TOOL BINDING SUPPORT")
    print("=" * 60)

    # Test each model type
    results = {}

    # Test Azure OpenAI
    if os.getenv("AZURE_OPENAI_API_KEY"):
        results["azureopenai"] = test_model_configuration("azureopenai")
    else:
        print("\n⚠️  Skipping Azure OpenAI (no API key)")
        results["azureopenai"] = None

    # Test Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        results["anthropic"] = test_model_configuration("anthropic")
    else:
        print("\n⚠️  Skipping Anthropic (no API key)")
        results["anthropic"] = None

    # Test Gemini
    if os.getenv("GOOGLE_API_KEY"):
        results["gemini"] = test_model_configuration("gemini")
    else:
        print("\n⚠️  Skipping Gemini (no API key)")
        results["gemini"] = None

    # Test OLLAMA (local, no API key needed)
    print("\n⚠️  Skipping OLLAMA (requires local server)")
    results["ollama"] = None

    # Test async functionality with Anthropic
    print("\n" + "=" * 60)
    print("TESTING ASYNC FUNCTIONALITY WITH TOOLS")
    print("=" * 60)
    if os.getenv("ANTHROPIC_API_KEY"):
        asyncio.run(test_model_configuration_async("anthropic"))
    else:
        print("\n⚠️  Skipping async test (no Anthropic API key)")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for model, result in results.items():
        if result is True:
            print(f"✅ {model}: PASSED")
        elif result is False:
            print(f"❌ {model}: FAILED")
        else:
            print(f"⚠️  {model}: SKIPPED")


if __name__ == "__main__":
    main()

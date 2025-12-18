"""
Test script for different LLM models using sqlite_query tool.

This script tests how different models (Azure OpenAI, Anthropic Claude, Google Gemini, OLLAMA)
perform when using the sqlite_query tool to answer the question "How many people live in Prague?"

Setup mimics the generate_query_node from nodes.py with simplified agentic loop.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from pathlib import Path

# CRITICAL: Set Windows event loop policy FIRST, before other imports
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Resolve base directory (project root)
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:  # Fallback if __file__ not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# Make project root importable
sys.path.insert(0, str(BASE_DIR))

# Import required modules
from my_agent.utils.helpers import get_configured_llm
from my_agent.utils.tools import get_sqlite_tools
from my_agent.utils.helpers import load_schema
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


async def test_model_with_tool(model_type: str):
    """Test a specific model with the sqlite_query tool."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_type.upper()}")
    print(f"{'='*60}")

    try:
        # Get sqlite tools (same as generate_query_node)
        tools = await get_sqlite_tools()
        sqlite_tool = next(
            (tool for tool in tools if tool.name == "sqlite_query"), None
        )
        if not sqlite_tool:
            print(f"❌ sqlite_query tool not found for {model_type}")
            return

        # Add finish_gathering tool
        # from my_agent.utils.tools import finish_gathering

        # tools.append(finish_gathering)

        # Get configured LLM with tools (same as generate_query_node)
        llm_with_tools, use_bind_tools = get_configured_llm(
            model_type=model_type, tools=tools
        )

        # Print the exact model being used
        model_name = getattr(llm_with_tools, "model_name", "Unknown")
        deployment_name = getattr(llm_with_tools, "deployment_name", None)
        if deployment_name:
            model_name = f"{deployment_name} ({model_name})"
        print(f"🔧 Using model: {model_name}")
        print(
            f"🔧 Tool binding method: {'bind_tools()' if use_bind_tools else 'direct tools parameter'}"
        )

        # Load schema (same as generate_query_node)
        schema_data = await load_schema(
            {"top_selection_codes": []}
        )  # Empty for testing

        # Build system prompt (simplified version from generate_query_node)
        system_prompt = f"""
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect.
Your task is to translate the user's natural-language question into SQLite SQL queries using the sqlite_query tool.

TOOL USAGE - CRITICAL INSTRUCTIONS:
- You have access to the sqlite_query tool that executes SQLITE SQL queries on the database
- You can call this tool up to 3 times to gather all necessary information
- You can use some preparatory queries first to examine data structure and understand what information is available
- After each tool call, you'll see the results and can decide if you need more data
- IMPORTANT: When you have sufficient information to answer the user's question, provide your final answer directly. Do not call any more tools.
- Use the tool iteratively to refine your understanding and gather comprehensive data

IMPORTANT Schema Details:
{schema_data}

IMPORTANT notes about SQL query generation:
- Limit the output to at most 10 rows using LIMIT unless the user specifies otherwise
- Select only the necessary columns, never all columns.
- Use appropriate SQL aggregation functions when needed (e.g., SUM, AVG)
- Column to Aggregate or extract numeric values is always called "value"!
- Do NOT modify the database.
- Always examine the schema to understand how the data are laid out.
- If you are not sure with column names, call the tool with this query to get the table schema: PRAGMA table_info(table_name)
- Be careful about how you ALIAS the Column names.
- ALWAYS INCLUDE COLUMN "ukazatel" or similar metric columns in SELECT and GROUP BY clauses if present.

VERIFICATION AND COMPLETION:
- After executing queries, review the results to ensure they answer the user's question
- If the results are incomplete or unclear, execute additional queries to gather more information
- When you have sufficient data, provide your final answer directly without calling any more tools
"""

        # Create initial messages (same as generate_query_node)
        human_prompt = f"""
**************
User question: How many people live in Prague?

**************
Schemas:
{schema_data}
"""

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        initial_messages = prompt_template.format_messages()

        conversation_messages = list(initial_messages)
        new_queries_and_results = []
        tool_call_count = 0
        max_tool_iterations = 3  # Simplified for testing

        print(f"🤖 Starting agentic loop for {model_type}...")

        # Simplified agentic loop (similar to generate_query_node but limited)
        while tool_call_count < max_tool_iterations:
            tool_call_count += 1
            print(f"🔄 Tool iteration {tool_call_count}/{max_tool_iterations}")

            # Invoke LLM
            try:
                if use_bind_tools:
                    llm_response = await llm_with_tools.ainvoke(conversation_messages)
                else:
                    llm_response = await llm_with_tools.ainvoke(
                        conversation_messages, tools=tools
                    )
                print(f"💬 LLM Response: {llm_response.content}")
            except Exception as e:
                print(f"❌ LLM error for {model_type}: {str(e)}")
                break

            # Check if LLM wants to use tools
            if not llm_response.tool_calls:
                print(f"✅ {model_type} finished gathering data (no more tool calls)")
                break

            # Process tool calls
            conversation_messages.append(llm_response)

            for tool_call in llm_response.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", str(asyncio.get_event_loop().time()))

                # if tool_name == "finish_gathering":
                #     print(
                #         f"🎯 {model_type} called finish_gathering - data gathering complete"
                #     )
                #     # Create completion message
                #     completion_message = AIMessage(
                #         content=f"Data gathering complete. {len(new_queries_and_results)} queries executed.",
                #         id="query_result",
                #     )
                #     conversation_messages.append(completion_message)
                #     break

                if "query" in tool_args:
                    sql_query = tool_args["query"]
                    print(f"⚡ {model_type} executing SQL: {sql_query}")

                    try:
                        tool_result = await sqlite_tool.ainvoke({"query": sql_query})
                        result_text = str(tool_result)

                        # Handle potential wrapper formats
                        import re

                        text_content_match = re.match(
                            r"^\[TextContent\(type='text', text='(.+)', annotations=None\)\]$",
                            result_text,
                            re.DOTALL,
                        )
                        if text_content_match:
                            result_text = text_content_match.group(1)

                        print(f"✅ Query result: {result_text[:200]}...")
                        new_queries_and_results.append((sql_query, result_text))

                        tool_message = type(
                            "ToolMessage",
                            (),
                            {"content": result_text, "tool_call_id": tool_call_id},
                        )()
                        conversation_messages.append(tool_message)

                    except Exception as e:
                        error_msg = f"Error executing query: {str(e)}"
                        print(f"❌ Query error: {error_msg}")
                        new_queries_and_results.append((sql_query, f"Error: {str(e)}"))

                        tool_message = type(
                            "ToolMessage",
                            (),
                            {"content": error_msg, "tool_call_id": tool_call_id},
                        )()
                        conversation_messages.append(tool_message)
                else:
                    print(f"⚠️ Unknown tool call: {tool_name}")

            # Check if finished
            # if any(
            #     tc.get("name") == "finish_gathering" for tc in llm_response.tool_calls
            # ):
            #     break

        # Print results
        print(f"\n📊 {model_type.upper()} RESULTS:")
        print(f"   Queries executed: {len(new_queries_and_results)}")
        for i, (query, result) in enumerate(new_queries_and_results, 1):
            print(f"   Query {i}: {query}")
            print(f"   Result {i}: {result[:300]}{'...' if len(result) > 300 else ''}")
            print()

    except Exception as e:
        print(f"❌ Error testing {model_type}: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    """Run tests for all models."""
    print("🚀 Starting LLM Tool Testing")
    print("Question: How many people live in Prague?")
    print("Models to test: azureopenai, anthropic, gemini, ollama, xai, mistral")

    models_to_test = [
        # "azureopenai",
        # "anthropic",
        # "gemini",
        # "ollama",
        # "xai",
        "mistral",
    ]

    for model_type in models_to_test:
        await test_model_with_tool(model_type)

    print(f"\n{'='*60}")
    print("✅ Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

"""Graph node implementations for the data analysis workflow.

This module defines all the node functions used in the LangGraph workflow,
including schema loading, query generation, execution, and result formatting.
"""

#==============================================================================
# IMPORTS
#==============================================================================
import os
import json
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, TypedDict
import ast  # used for syntax validation of generated pandas expressions

from .state import DataAnalysisState
from .tools import DEBUG_MODE, PandasQueryTool, SQLiteQueryTool
from langgraph.prebuilt import ToolNode, tools_condition

#==============================================================================
# CONSTANTS & CONFIGURATION
#==============================================================================
# Static IDs for easier debug‑tracking
GET_SCHEMA_ID = 3
QUERY_GEN_ID = 4
CHECK_QUERY_ID = 5
EXECUTE_QUERY_ID = 6
SUBMIT_FINAL_ID = 7
SAVE_RESULT_ID = 8
SHOULD_CONTINUE_ID = 9

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
MAX_ITERATIONS = 2  # Reduced from 3 to prevent excessive looping
FORMAT_ANSWER_ID = 10  # Add to CONSTANTS section
ROUTE_DECISION_ID = 11  # ID for routing decision function

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    if DEBUG_MODE:
        print(msg)

def get_azure_llm(temperature=0.0):
    """Get an instance of Azure OpenAI LLM with standard configuration.
    
    Args:
        temperature (float): Temperature setting for generation randomness
        
    Returns:
        AzureChatOpenAI: Configured LLM instance
    """
    return AzureChatOpenAI(
        deployment_name='gpt-4o__test1',
        model_name='gpt-4o',
        openai_api_version='2024-05-01-preview',
        temperature=temperature,
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )

async def load_schema():
    """Load the schema metadata from the JSON file asynchronously.
    
    While file operations are synchronous, this function provides an async
    interface for consistency with the async workflow.
    
    Returns:
        dict: The schema metadata
    """
    schema_path = BASE_DIR / "metadata" / "OBY01PDT01_metadata.json"
    with schema_path.open('r', encoding='utf-8') as f:
        return json.load(f)

#==============================================================================
# NODE FUNCTIONS
#==============================================================================
async def get_schema_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Get schema details for relevant columns."""
    debug_print(f"{GET_SCHEMA_ID}: Enter get_schema_node")
    schema = await load_schema()
    msg = AIMessage(content=f"Schema details: {json.dumps(schema, ensure_ascii=False, indent=2)}")
    return {"messages": [msg]}  # Reducer will append this to existing messages

async def reflect_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Reflect on the current state and provide feedback for next query.
    
    This node analyzes the current state of messages and queries to provide detailed
    feedback about what information is missing or what needs to be adjusted.
    It always returns to query_gen for another iteration.
    """
    debug_print(f"{ROUTE_DECISION_ID}: Enter reflect_node")
    
    llm = get_azure_llm(temperature=0.0)
    
    # Format messages for context
    messages_text = "\n\n".join([
        f"{msg.type}: {msg.content}" 
        for msg in state["messages"]
    ])
    
    # Format queries and results
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
        for i, (query, result) in enumerate(state["queries_and_results"])
    )
    
    system_prompt = """
You are a data analysis reflection agent. Your task is to analyze the current state and provide
detailed feedback to guide the next query. You should:

1. Review the original question and all messages in the conversation
2. Analyze all executed queries and their results
3. Provide detailed feedback about:
   - What specific information is still missing
   - What kind of query would help get this information
   - How to adjust the query approach
   - Any patterns or insights that could be useful

For comparison questions, ensure we have data for all entities being compared.
For trend analysis, ensure we have data across all relevant time periods.
For distribution questions, ensure we have complete coverage of all categories.

Your response should be detailed and specific, helping guide the next query.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original question: {question}\n\nConversation history:\n{messages}\n\nCurrent queries and results:\n{results}\n\nWhat feedback can you provide to guide the next query?")
    ])
    
    result = await llm.ainvoke(
        prompt.format_messages(
            question=state["prompt"],
            messages=messages_text,
            results=queries_results_text
        )
    )
    
    # Add reflection to messages
    return {"messages": [result]}

async def query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate pandas query based on question and schema."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_node")
    
    llm = get_azure_llm(temperature=0.0)
    
    # Create fresh tool instance
    sqlite_tool = SQLiteQueryTool()
    llm_with_tools = llm.bind_tools([sqlite_tool])
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect. Your task is to translate the user's natural-language question into a SQL query and execute it using the sqlite_query tool.

To accomplish this:
1. Read and analyze the provided inputs:
- User prompt (in Czech or English)
- Schema metadata (containing Czech column names and values)
- Previous messages in the conversation
- Any feedback from the reflection agent

2. Process the prompt by:
- Identifying key terms in either language
- Matching terms to their Czech equivalents in the schema
- Handling Czech diacritics and special characters
- Converting geographical names between languages and similar concepts

3. Construct an appropriate SQL query by:
- Using exact column names from the schema (can be Czech or English)
- Matching user prompt terms to correct data values
- Ensuring proper string matching for Czech characters
- GENERATING A NEW QUERY THAT PROVIDES ADDITIONAL INFORMATION that is not already present in the previously executed queries

4. Use the sqlite_query tool to execute the query.

5. Always answer in the language of the prompt and preserve Czech characters.
6. Numeric outputs must be plain digits with NO thousands separators.

When generating the query:
- Return ONLY the SQL expression that answers the question.
- Limit the output to at most 5 rows using LIMIT unless the user specifies otherwise.
- Select only the necessary columns, never all columns.
- Use appropriate SQL aggregation functions when needed (e.g., SUM, AVG).
- Do NOT modify the database.
- NEVER invent data that is not present in the dataset.

IMPORTANT: You must use the sqlite_query tool to execute queries. Direct query execution is not allowed. Always format your response as a tool call using the sqlite_query tool.

=== IMPORTANT RULE ===
Return ONLY the final SQL query that should be executed, with nothing else around it. Do NOT wrap it in ```sql``` fences and do NOT add explanations.

USE only one TABLE in FROM clause called 'OBY01PDT01'
"""
    
    # Format messages for context
    messages_text = "\n\n".join([
        f"{msg.type}: {msg.content}" 
        for msg in state["messages"]
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {prompt}\nSchema: {schema}\nPrevious messages:\n{messages}")
    ])
    
    schema = await load_schema()
    result = await llm_with_tools.ainvoke(prompt.format_messages(
        prompt=state["prompt"],
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
        messages=messages_text
    ))
    
    # Execute tool and store results
    new_messages = []
    new_queries = []
    
    if hasattr(result, 'additional_kwargs') and 'tool_calls' in result.additional_kwargs:
        tool_calls = result.additional_kwargs['tool_calls']
        for tool_call in tool_calls:
            if tool_call['type'] == 'function' and tool_call['function']['name'] == 'sqlite_query':
                tool_args = json.loads(tool_call['function']['arguments'])
                query = tool_args['query']
                
                try:
                    # Execute the query using the SQLite tool
                    tool_result = sqlite_tool._run(query)
                    debug_print(f"{QUERY_GEN_ID}: Successfully executed query: {query}")
                    debug_print(f"{QUERY_GEN_ID}: Query result: {tool_result}")
                    # Store the query and its string result
                    new_queries.append((query, tool_result))
                except Exception as e:
                    error_msg = f"Error executing query: {str(e)}"
                    debug_print(f"{QUERY_GEN_ID}: {error_msg}")
                    # Store the query with error message
                    new_queries.append((query, f"Error: {str(e)}"))
                    # Add error message to state
                    new_messages.append(AIMessage(content=error_msg))
    
    new_messages.append(result)
    debug_print(f"{QUERY_GEN_ID}: Current state of queries_and_results: {new_queries}")
    
    # Return updated state without routing decision
    return {
        "prompt": state["prompt"],
        "messages": new_messages,
        "iteration": state["iteration"],
        "queries_and_results": new_queries
    }

async def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Submit the final answer to the user."""
    debug_print(f"{SUBMIT_FINAL_ID}: Enter submit_final_answer_node")
    debug_print(f"{SUBMIT_FINAL_ID}: Final answer prepared")
    return state

async def save_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Save the result to a file."""
    debug_print(f"{SAVE_RESULT_ID}: Enter save_node")
    
    # Get the final answer from the last message
    final_answer = state["messages"][-1].content
    
    result_path = BASE_DIR / "analysis_results.txt"
    result_obj = {
        "prompt": state["prompt"],
        "result": final_answer,
        "queries_and_results": [{"query": q, "result": r} for q, r in state["queries_and_results"]]
    }
    
    with result_path.open("a", encoding='utf-8') as f:
        f.write(f"Prompt: {state['prompt']}\n")
        f.write(f"Result: {final_answer}\n")
        f.write("Queries and Results:\n")
        for query, result in state["queries_and_results"]:
            f.write(f"  Query: {query}\n")
            f.write(f"  Result: {result}\n")
        f.write("----------------------------------------------------------------------------\n")
    
    json_result_path = BASE_DIR / "analysis_results.json"
    existing_results = []
    if json_result_path.exists():
        try:
            with json_result_path.open("r", encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
    
    existing_results.append(result_obj)
    with json_result_path.open("w", encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    debug_print(f"{SAVE_RESULT_ID}: ✅ Result saved to {result_path} and {json_result_path}")
    return state

async def format_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Format the query result into a natural language answer.
    
    This node works only with queries_and_results to generate the final analysis.
    """
    debug_print(f"{FORMAT_ANSWER_ID}: Enter format_answer_node")
    
    llm = get_azure_llm(temperature=0.1)
    
    # Format results for better readability
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
        for i, (query, result) in enumerate(state["queries_and_results"])
    )
    
    system_prompt = """
You are a bilingual (Czech/English) data analysis assistant. Your task is to analyze the query results 
and provide a complete answer based ONLY on the provided query results.

Your response should:
1. Consider all query results as part of the complete answer
2. Make clear comparisons when values are being compared
3. Maintain numeric precision from all results
4. Be concise but complete
5. Answer in the same language as the original question
"""
    
    formatted_prompt = f"Question: {state['prompt']}\n\nQueries and Results:\n{queries_results_text}\n\nPlease provide a complete analysis."
    
    chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    result = await llm.ainvoke(
        chain.format_messages(input=formatted_prompt)
    )
    
    debug_print(f"{FORMAT_ANSWER_ID}: Analysis completed")
    
    return {"messages": [result]}

async def increment_iteration_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Increment the iteration counter and return updated state."""
    debug_print(f"{ROUTE_DECISION_ID}: Incrementing iteration counter")
    return {"iteration": state.get("iteration", 0) + 1}
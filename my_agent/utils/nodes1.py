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
from typing import Literal
import ast  # used for syntax validation of generated pandas expressions

from .state import DataAnalysisState
from .tools import DEBUG_MODE, PandasQueryTool
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
BASE_DIR = Path(__file__).resolve().parents[2]
MAX_ITERATIONS = 3  # prevent infinite correction loops
FORMAT_ANSWER_ID = 10  # Add to CONSTANTS section

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
    print("AAAAAAAAAAAAAA")
    debug_print(f"{GET_SCHEMA_ID}: Enter get_schema_node")
    schema = await load_schema()
    msg = AIMessage(content=f"Schema details: {json.dumps(schema, ensure_ascii=False, indent=2)}")
    state.messages.append(msg)
    debug_print(f"{GET_SCHEMA_ID}: Schema details appended")
    return state

async def query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate pandas query based on question and schema."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_node")
    llm = get_azure_llm(temperature=0.0)
    
    # Create fresh tool instance
    llm_with_tools = llm.bind_tools([PandasQueryTool()])
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in pandas. Your task is to translate the user's natural-language question into a pandas query and execute it using the pandas_query tool.

To accomplish this:
1. Read and analyze the provided inputs:
- User prompt (in Czech or English)
- Schema metadata (containing Czech column names and values)

2. Process the prompt by:
- Identifying key terms in either language
- Matching terms to their Czech equivalents in the schema
- Handling Czech diacritics and special characters
- Converting geographical names between languages and similar concepts

3. Construct an appropriate pandas query by:
- Using exact column names from the schema (can be Czech or English)
- Matching user prompt terms to correct data values (the schema contains a list of unique values in specific columns)
- Ensuring proper string matching for Czech characters
- Carefully examining dimensional unique values in the schema, especially for columns that may contain both totals and detailed records (e.g., "CZ, Region" may include "Czech Republic" and "Regions")

4. Use the pandas_query tool to execute the query.

5. Always answer in the language of the prompt and preserve Czech characters.
6. Numeric outputs must be plain digits with NO thousands separators.

When generating the query:
- Return ONLY the pandas expression that answers the question.
- Limit the output to at most 5 rows unless the user specifies otherwise.
- Select only the necessary columns, never all columns.
- Use the appropriate pandas function for aggregation results (e.g., sum, mean).
- Do NOT modify the CSV file.
- NEVER invent data that is not present in the dataset.

IMPORTANT: You must use the pandas_query tool to execute queries. Direct query execution is not allowed. Always format your response as a tool call using the pandas_query tool.

=== IMPORTANT RULE ===
Return ONLY the final pandas expression that should be executed, with nothing else around it. Do NOT wrap it in ```python``` fences and do NOT add explanations.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {prompt}\nSchema: {schema}")
    ])
    
    schema = await load_schema()
    result = await llm_with_tools.ainvoke(prompt.format_messages(prompt=state.prompt, 
                                                          schema=json.dumps(schema, ensure_ascii=False, indent=2)))
    
    # Store structured result in state if it's a tool response
    if hasattr(result, 'additional_kwargs') and 'tool_calls' in result.additional_kwargs:
        try:
            tool_result = json.loads(result.content)
            state.queries_and_results.append(tool_result)
        except json.JSONDecodeError:
            pass
    
    state.messages.append(result)
    debug_print(f"{QUERY_GEN_ID}: Query generated")
    return state

async def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Submit the final answer to the user."""
    debug_print(f"{SUBMIT_FINAL_ID}: Enter submit_final_answer_node")
    
    # Get the formatted answer from the last message
    last_message_content = state.messages[-1].content
    
    # Add the final answer to the messages
    state.messages.append(AIMessage(content=last_message_content))
    debug_print(f"{SUBMIT_FINAL_ID}: Final answer prepared")
    return state

async def save_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Save the result to a file."""
    debug_print(f"{SAVE_RESULT_ID}: Enter save_node")
    
    # Get the final answer from the last message
    final_answer = state.messages[-1].content
    
    result_path = BASE_DIR / "analysis_results.txt"
    result_obj = {
        "prompt": state.prompt,
        "result": final_answer,
        "queries_and_results": state.queries_and_results
    }
    
    with result_path.open("a", encoding='utf-8') as f:
        f.write(f"Prompt: {state.prompt}\n")
        f.write(f"Result: {final_answer}\n")
        f.write("Queries and Results:\n")
        for qr in state.queries_and_results:
            f.write(f"  Query: {qr['query']}\n")
            f.write(f"  Result: {qr['result']}\n")
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
    """Node: Format the query result into a natural language answer."""
    debug_print(f"{FORMAT_ANSWER_ID}: Enter format_answer_node")
    
    # Debug print current state of queries_and_results
    debug_print(f"{FORMAT_ANSWER_ID}: Number of queries and results: {len(state.queries_and_results)}")
    
    llm = get_azure_llm(temperature=0.1)
    
    system_prompt = """
You are a bilingual (Czech/English) data analysis assistant. Your task is to analyze the query results and provide a complete answer.
You will receive:
1. The original question
2. A list of executed queries and their results

Your response should:
1. Consider all query results as part of the complete answer
2. Make clear comparisons when values are being compared
3. Maintain numeric precision from all results
4. Be concise but complete
5. Use the same language as the original question"""
    
    # Format results for better readability
    if state.queries_and_results:
        queries_results_text = ""
        for qr in state.queries_and_results:
            queries_results_text += f"Query: {qr['query']}\n"
            queries_results_text += f"Result: {qr['result']}\n\n"
    else:
        queries_results_text = "No query results available."
        
    chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nQueries and Results:\n{results}\n\nPlease provide a complete analysis.")
    ])
    
    result = await llm.ainvoke(
        chain.format_messages(
            question=state.prompt,
            results=queries_results_text
        )
    )
    
    debug_print(f"{FORMAT_ANSWER_ID}: Analysis completed")
    debug_print(f"{FORMAT_ANSWER_ID}: Formatted input sent to LLM:")
    debug_print(f"{FORMAT_ANSWER_ID}: Question: {state.prompt}")
    debug_print(f"{FORMAT_ANSWER_ID}: Results provided:\n{queries_results_text}")
    
    state.messages.append(result)
    return state
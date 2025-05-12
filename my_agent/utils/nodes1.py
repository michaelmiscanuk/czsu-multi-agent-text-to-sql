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
from .tools import PandasQueryTool, DEBUG_MODE
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

def load_schema():
    """Load the schema metadata from the JSON file.
    
    Returns:
        dict: The schema metadata
    """
    schema_path = BASE_DIR / "metadata" / "OBY01PDT01_metadata.json"
    with schema_path.open('r', encoding='utf-8') as f:
        return json.load(f)

#==============================================================================
# NODE FUNCTIONS
#==============================================================================
def get_schema_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Get schema details for relevant columns."""
    debug_print(f"{GET_SCHEMA_ID}: Enter get_schema_node")
    schema = load_schema()
    msg = AIMessage(content=f"Schema details: {json.dumps(schema, ensure_ascii=False, indent=2)}")
    state.messages.append(msg)
    debug_print(f"{GET_SCHEMA_ID}: Schema details appended")
    return state

def query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate pandas query based on question and schema."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_node")
    llm = get_azure_llm(temperature=0.0)
    
    # Create PandasQueryTool instance and bind to LLM
    pandas_tool = PandasQueryTool()
    llm_with_tools = llm.bind_tools([pandas_tool])
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in pandas.
Your task is to translate the user's natural‑language question into a pandas query and execute it.

To accomplish this:
1. Analyze the schema and question
2. Construct an appropriate pandas query
3. Use the pandas_query tool to execute the query
4. Return the result

IMPORTANT: You must use the pandas_query tool to execute queries. Direct query execution is not allowed.
Always format your response as a tool call using the pandas_query tool.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {prompt}\nSchema: {schema}")
    ])
    
    schema = load_schema()
    result = llm_with_tools.invoke(prompt.format_messages(prompt=state.prompt, 
                                                        schema=json.dumps(schema, ensure_ascii=False, indent=2)))
    state.messages.append(result)
    debug_print(f"{QUERY_GEN_ID}: Query generated")
    return state

def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Submit the final answer to the user."""
    debug_print(f"{SUBMIT_FINAL_ID}: Enter submit_final_answer_node")
    
    # Extract the answer from the previous message
    last_message_content = state.messages[-1].content
    
    #--------------------------------------------------------------------------
    # Format final answer
    #--------------------------------------------------------------------------
    # Check if this is a query result that needs formatting
    if last_message_content.startswith("Query result:"):
        # Extract the query result
        query_result = last_message_content[len("Query result:"):].strip()
        
        # Create a more user-friendly answer using the prompt and query result
        formatted_answer = f"Based on your question '{state.prompt}', the answer is: {query_result}"
        state.result = formatted_answer
    else:
        # Just use the existing answer
        state.result = last_message_content
    
    # Add the final answer to the messages
    state.messages.append(AIMessage(content=f"Final answer: {state.result}"))
    debug_print(f"{SUBMIT_FINAL_ID}: Final answer prepared")
    return state

def save_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Save the result to a file."""
    debug_print(f"{SAVE_RESULT_ID}: Enter save_node")
    result_path = BASE_DIR / "analysis_results.txt"
    result_obj = {
        "prompt": state.prompt,
        "result": state.result
    }
    with result_path.open("a", encoding='utf-8') as f:
        f.write(f"Prompt: {state.prompt}\n")
        f.write(f"Result: {state.result}\n")
        f.write(f"----------------------------------------------------------------------------\n")
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
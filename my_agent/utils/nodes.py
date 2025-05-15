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
MAX_ITERATIONS = 3  # prevent infinite correction loops
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
    sqlite_tool = SQLiteQueryTool()
    llm_with_tools = llm.bind_tools([sqlite_tool])
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect. Your task is to translate the user's natural-language question into a SQL query and execute it using the sqlite_query tool.

To accomplish this:
1. Read and analyze the provided inputs:
- User prompt (in Czech or English)
- Schema metadata (containing Czech column names and values)
- Previously executed queries and their results

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
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {prompt}\nSchema: {schema}\nPreviously executed queries and results:\n{current_queries}")
    ])
    
    # Format current queries for context
    if state.queries_and_results:
        current_queries = "\n\n".join([
            f"Query {i+1}:\n{q}\n\nResult {i+1}:\n{r}" 
            for i, (q, r) in enumerate(state.queries_and_results)
        ])
    else:
        current_queries = "No queries have been executed yet."
    
    schema = await load_schema()
    result = await llm_with_tools.ainvoke(prompt.format_messages(
        prompt=state.prompt,
        schema=json.dumps(schema, ensure_ascii=False, indent=2),
        current_queries=current_queries
    ))
    
    # Execute tool and store results
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
                    state.queries_and_results.append((query, tool_result))
                except Exception as e:
                    debug_print(f"{QUERY_GEN_ID}: Error executing query: {str(e)}")
                    # Store the query with error message
                    state.queries_and_results.append((query, f"Error: {str(e)}"))
    
    state.messages.append(result)
    debug_print(f"{QUERY_GEN_ID}: Current state of queries_and_results: {state.queries_and_results}")
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
        "queries_and_results": [{"query": q, "result": r} for q, r in state.queries_and_results]
    }
    
    with result_path.open("a", encoding='utf-8') as f:
        f.write(f"Prompt: {state.prompt}\n")
        f.write(f"Result: {final_answer}\n")
        f.write("Queries and Results:\n")
        for query, result in state.queries_and_results:
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
    """Node: Format the query result into a natural language answer."""
    debug_print(f"{FORMAT_ANSWER_ID}: Enter format_answer_node")
    
    # Debug print current state of queries_and_results
    debug_print(f"{FORMAT_ANSWER_ID}: Number of queries and results: {len(state.queries_and_results)}")
    
    # Debug print current state of queries_and_results
    debug_print(f"{FORMAT_ANSWER_ID}: queries and results: {(state.queries_and_results)}")
    
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
"""
    
    # Format results for better readability
    queries_results_text = "No query results available."
    if state.queries_and_results:
        queries_results_text = "\n\n".join(
            f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
            for i, (query, result) in enumerate(state.queries_and_results)
        )
    
    formatted_prompt = f"Question: {state.prompt}\n\nQueries and Results:\n{queries_results_text}\n\nPlease provide a complete analysis."
    
    chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    result = await llm.ainvoke(
        chain.format_messages(input=formatted_prompt)
    )
    
    debug_print(f"{FORMAT_ANSWER_ID}: Analysis completed")
    debug_print(f"{FORMAT_ANSWER_ID}: Formatted input sent to LLM:")
    debug_print(f"{FORMAT_ANSWER_ID}: Question: {state.prompt}")
    debug_print(f"{FORMAT_ANSWER_ID}: Results provided:\n{queries_results_text}")
    
    state.messages.append(result)
    return state

async def route_after_query(state: DataAnalysisState) -> Literal["query_again", "format_answer"]:
    """Determine whether to run another query or proceed to formatting the answer.
    
    This routing function uses an LLM to analyze the current query results and 
    determine if more queries are needed to fully answer the user's question.
    
    Args:
        state: The current workflow state
        
    Returns:
        str: Either "query_again" to execute another query or "format_answer" to proceed
    """
    debug_print(f"{ROUTE_DECISION_ID}: Enter route_after_query")
    
    # Hard limit on number of queries to prevent infinite loops
    if len(state.queries_and_results) >= 3:
        debug_print(f"{ROUTE_DECISION_ID}: Query limit reached, proceeding to format answer")
        return "format_answer"
    
    llm = get_azure_llm(temperature=0.0)
    
    # Format current queries and results for the LLM
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
        for i, (query, result) in enumerate(state.queries_and_results)
    )
    
    # Prepare the decision prompt
    system_prompt = """
You are a data analysis advisor. Your task is to determine whether additional queries 
are needed to fully answer a user's question based on the current query results.

You should answer "query_again" if:
1. The current results are incomplete or insufficient to answer the original question
2. For comparison questions, you need data about all entities being compared
3. The original question has multiple parts that haven't all been addressed
4. The data is from the wrong time period or geographical area

You should answer "format_answer" if:
1. All necessary data to answer the question has been collected
2. For comparison questions, you have data for all entities being compared
3. Additional queries would be redundant or unnecessary

CRITICAL INSTRUCTION: Answer ONLY with "query_again" or "format_answer", nothing else.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original question: {question}\n\nCurrent queries and results:\n{results}\n\nDo we need another query to fully answer the original question? Answer 'query_again' or 'format_answer'.")
    ])
    
    # Get decision from the LLM
    try:
        result = await llm.ainvoke(
            prompt.format_messages(
                question=state.prompt,
                results=queries_results_text
            )
        )
        
        # Extract decision from the LLM's response
        decision = result.content.lower().strip()
        
        debug_print(f"{ROUTE_DECISION_ID}: LLM decision: '{decision}'")
        
        # Default to format_answer if anything unexpected is returned
        if "query_again" in decision:
            debug_print(f"{ROUTE_DECISION_ID}: Routing to generate another query")
            return "query_again"
        else:
            debug_print(f"{ROUTE_DECISION_ID}: Routing to format final answer")
            return "format_answer"
        
    except Exception as e:
        debug_print(f"{ROUTE_DECISION_ID}: Error in decision making: {e}")
        # Default to format_answer on error to avoid getting stuck
        return "format_answer"
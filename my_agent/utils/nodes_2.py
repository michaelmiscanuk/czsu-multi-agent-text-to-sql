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
from .models import get_azure_llm

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

def query_gen_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate pandas query based on question and schema."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_gen_node")
    llm = get_azure_llm(temperature=0.0)
    
    #--------------------------------------------------------------------------
    # Prompt template
    #--------------------------------------------------------------------------
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in pandas.
Your task is to translate the user's natural‑language question into an exact pandas expression,
execute it (via the pandas_query tool), and deliver the numerical answer.

1. Understand the prompt and the provided schema (Czech column names & values).
2. Map Czech/English terms, handle diacritics, and beware of "total / region" rows.
3. Build a pandas query using the exact column names and matching data values.
4. Execute & validate: if errors arise, refine the query until it works.
5. Always answer in the language of the prompt and preserve Czech characters.
6. Numeric outputs must be plain digits with NO thousands separators.

When generating the query:
- Return ONLY the pandas expression that answers the question.
- Limit to at most 5 rows unless the user asked otherwise.
- Never select all columns, only those needed.
- If the result is an aggregation (sum, mean …) use the proper pandas function.
- Do NOT modify the CSV file.
- NEVER invent data that is not in the dataset.

=== IMPORTANT RULE ===
If you call the tool (check_query) the *content* of your assistant message MUST be
just the pandas expression (e.g. df[(df["col"]=="val")]["value"]).  
No markdown fences, no comments, no extra prose.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {prompt}\nSchema: {schema}")
    ])
    
    #--------------------------------------------------------------------------
    # Generate query with LLM
    #--------------------------------------------------------------------------
    schema = load_schema()
    result = llm.invoke(prompt.format_messages(prompt=state.prompt, schema=json.dumps(schema, ensure_ascii=False, indent=2)))
    state.messages.append(AIMessage(content=result.content, tool_calls=[{"name": "check_query", "args": {"query": result.content}, "id": "tool_query_check"}]))
    debug_print(f"{QUERY_GEN_ID}: Query generated and appended")
    return state

def check_query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Double-check the pandas query for common mistakes."""
    debug_print(f"{CHECK_QUERY_ID}: Enter check_query_node")
    llm = get_azure_llm(temperature=0.0)
    
    #--------------------------------------------------------------------------
    # Prompt template
    #--------------------------------------------------------------------------
    system_prompt = """
You are a pandas expert. Double check the pandas query for common mistakes, including:
- Data type mismatch
- Properly quoting column names
- Using the correct number of arguments for functions
- Using the proper columns for filtering and aggregation

If there are mistakes, rewrite the query. If not, just reproduce the original query.

=== IMPORTANT RULE ===
Return ONLY the final pandas expression that should be executed, with
nothing else around it. Do NOT wrap it in ```python``` fences and do
NOT add explanations.
"""
    last_query = state.messages[-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Query to check: {last_query}")
    ])
    
    #--------------------------------------------------------------------------
    # Check query with LLM
    #--------------------------------------------------------------------------
    result = llm.invoke(prompt.format_messages())
    state.messages.append(AIMessage(content=result.content, tool_calls=[{"name": "execute_query", "args": {"query": result.content}, "id": "tool_execute_query"}]))
    debug_print(f"{CHECK_QUERY_ID}: Checked query appended")
    return state


def execute_query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Execute the pandas query and return the result.
    
    This node is responsible for safely executing the generated pandas query against 
    the dataset. It includes several safety mechanisms:
    1. Increments iteration counter to prevent infinite loops
    2. Validates syntax before execution to catch errors early
    3. Uses a sandboxed execution environment via the PandasQueryTool
    4. Provides detailed error messages for debugging
    
    The execution result (or error) is added to the conversation history for
    downstream nodes to process.
    
    Args:
        state: Current workflow state containing the query to execute
        
    Returns:
        Updated state with execution results or error information
    """
    debug_print(f"{EXECUTE_QUERY_ID}: Enter execute_query_node (iteration {state.iteration})")
    # Increment iteration to track progress and prevent infinite loops
    state.iteration += 1
    pandas_tool = PandasQueryTool()

    # The assistant message contains only the pandas expression without formatting
    query_expression = state.messages[-1].content.strip()

    #--------------------------------------------------------------------------
    # Validate and execute query
    #--------------------------------------------------------------------------
    # Early syntax validation prevents executing malformed queries
    # This provides more helpful error messages and improves security
    try:
        ast.parse(query_expression, mode="eval")
    except SyntaxError as syntax_error:
        error_message = f"Invalid pandas expression received: {syntax_error}"
        state.messages.append(AIMessage(content=f"Error: {error_message}"))
        debug_print(f"{EXECUTE_QUERY_ID}: {error_message}")
        return state

    debug_print(f"{EXECUTE_QUERY_ID}: Executing query: {query_expression}")
    try:
        # Execution happens in the PandasQueryTool which provides sandboxing
        # for better security and consistent error handling
        query_result = pandas_tool._run(query_expression)
        state.messages.append(AIMessage(content=f"Query result: {query_result}"))
        debug_print(f"{EXECUTE_QUERY_ID}: Query succeeded")
    except Exception as execution_error:
        # Capture and format execution errors for the correction step
        state.messages.append(AIMessage(content=f"Error: {str(execution_error)}"))
        debug_print(f"{EXECUTE_QUERY_ID}: Query failed – {execution_error}")
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

#==============================================================================
# FLOW CONTROL FUNCTIONS
#==============================================================================

def should_continue(state: DataAnalysisState) -> Literal["submit_final_answer", "correct_query"]:
    """Decide whether to continue the workflow or submit the final answer.
    
    This function serves as a critical decision point in the workflow that prevents
    infinite loops and determines the next step based on the current state.
    
    The logic implements several important safeguards:
    1. Hard limit on iterations to prevent endless correction cycles
    2. Success detection to short-circuit when we have valid results
    3. Always routes errors to correction rather than regeneration
    
    This careful routing is essential for maintaining workflow stability
    and preventing common failure modes in LLM-based systems.
    
    Args:
        state: Current workflow state
        
    Returns:
        String indicating which node should be executed next
    """
    debug_print(f"{SHOULD_CONTINUE_ID}: Deciding next step (iteration {state.iteration})")
    
    # Safety check - prevent infinite loops by enforcing a maximum iteration count
    # This is critical for production stability and cost control
    if state.iteration >= MAX_ITERATIONS:
        debug_print(f"{SHOULD_CONTINUE_ID}: Max iterations reached")
        state.result = f"Max iterations reached. Last message: {state.messages[-1].content}"
        return "submit_final_answer"
    
    # Check if the last message contains "Query result:" prefix, indicating successful execution
    # This determines if we can proceed to final answer or need another correction cycle
    if "Query result:" in state.messages[-1].content:
        debug_print(f"{SHOULD_CONTINUE_ID}: Query result detected, submitting final answer")
        return "submit_final_answer"
    
    # For all other cases (especially errors), go to correction rather than regeneration
    # This avoids repetitive failures by letting the model fix its previous attempt
    # rather than starting from scratch and potentially making the same mistake
    debug_print(f"{SHOULD_CONTINUE_ID}: Proceeding to correct query")
    return "correct_query"

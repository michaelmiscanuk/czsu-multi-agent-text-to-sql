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
from .tools import PandasQueryTool, SQLiteQueryTool
from langgraph.prebuilt import ToolNode, tools_condition
import os

# Get debug mode from environment variable
DEBUG_MODE = os.environ.get('MY_AGENT_DEBUG', '0') == '1'

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
    # Always check environment variable directly to respect runtime changes
    if os.environ.get('MY_AGENT_DEBUG', '0') == '1':
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


async def query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate pandas query based on question and schema."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_node")
    
    llm = get_azure_llm(temperature=0.0)
    
    # Create fresh tool instance
    sqlite_tool = SQLiteQueryTool()
    llm_with_tools = llm.bind_tools([sqlite_tool])
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect. 
Your task is to translate the user's natural-language question into a SQL query and execute it using the sqlite_query tool.

To accomplish this, follow these steps:

1. Read and analyze the provided inputs:
- User prompt (can be in Czech or English)
- Read schema carefully to you can understand how the data are laid, 
    layout can be non standard, but you have a loot of information there.
- Previous messages in the conversation
- Any feedback from the reflection agent, if any.

2. Process the prompt by:
- Identifying key terms in either language
- Matching terms to their Czech equivalents in the schema
- Handling Czech diacritics and special characters
- Converting concepts between languages

3. Construct an appropriate SQL query by:
- Using exact column names from the schema (can be Czech or English)
- Matching user prompt terms to correct data values
- Ensuring proper string matching for Czech characters
- GENERATING A NEW QUERY THAT PROVIDES ADDITIONAL INFORMATION that is not already present in the previously executed queries

4. Use the sqlite_query tool to execute the query.

6. Numeric outputs must be plain digits with NO thousands separators.

Important Schema Details:
- "dimensions" key contains several other keys, which are columns in the table.
-- Each of them contains "values" key with list of distinct values in that column.
-- If there is a column of type "metric", it means that it is a column that contains names of metrics, not values - it can be used in WHERE clause to filter.
- Then there is key "value_column" with name of the column that contains the values for metric names, they can be used in aggregations, like sum, etc.

HERE IS THE MOST IMPORTANT PART:
Always read carefully all distinct values of dimensions, 
and do some thinking to choose the best ones to fit our question. 
Your can use LIKE and regex to filter them, if it is necessary by our question. 
For example, if user asks about "female", but dimensional value are "start_period_female" and "end_period_female", just filter for %female%, if it makes sense.

IMPORTANT note about total records: 
The dataset contains statistical records that include total rows for certain dimension values. 
These total rows, which may have been generated by SQL clauses such as GROUP BY WITH TOTALS or GROUP BY ROLLUP, 
should be ignored in calculations. 
For instance, if the data includes regions within a republic, 
there may also be rows representing total values for the entire republic, 
further split by dimensions like male/female. When performing analyses such as distribution by regions, 
including these total records will result in percentage values being inaccurately halved. 
Additionally, failing to exclude these totals during summarization will lead to double counting. 
Always calculate using only the relevant data and separate pieces, ensuring accuracy in statistical results.

When generating the query:
- Return ONLY the SQL expression that answers the question.
- Limit the output to at most 5 rows using LIMIT unless the user specifies otherwise - but first think if you dont need to group it somehow so it returns reasonable 5 rows.
- Select only the necessary columns, never all columns.
- Use appropriate SQL aggregation functions when needed (e.g., SUM, AVG).
- Do NOT modify the database.

USE only one TABLE in FROM clause called 'OBY01PDT01'.

=== IMPORTANT RULE ===
Return ONLY the final SQL query that should be executed, with nothing else around it. 
Do NOT wrap it in code fences and do NOT add explanations.

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

async def reflect_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Reflect on the current state and provide feedback for next query or decide to answer.
    
    This node analyzes the current state of messages and queries to provide detailed
    feedback about what information is missing or what needs to be adjusted.
    It can now decide to either continue iterating ("improve") or proceed to answer formatting ("answer").
    The decision is stored in the 'reflection_decision' field of the state.
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
You are a data analysis reflection agent.
Your task is to analyze the current state and provide detailed feedback to guide the next query.

You must also decide if there is now enough information to answer the user's question.

If you believe the current queries and results are sufficient to answer the user's question,
return a line at the END of your response exactly as:
    DECISION: answer
If you believe more queries or improvements are needed,
return a line at the END of your response exactly as:
    DECISION: improve

Your process:
  1. Review the original question and all messages in the conversation.
  2. Analyze all executed queries and their results.
  3. Provide detailed feedback about:
      - What specific information is still missing
      - What kind of SQL query would help get this information
      - Any patterns or insights that could be useful

Guidelines:
  - For comparison questions, ensure we have data for all entities being compared.
  - For trend analysis, ensure we have data across all relevant time periods.
  - For distribution questions, ensure we have complete coverage of all categories.

Your response should be detailed and specific, helping guide the next query.

MOST IMPORTANT: 
If improvement will be needed - Imagine it is a chatbot and you are now playing a role of a HUMAN giving instructions to the LLM about 
how to improve the SQL QUERY - so phrase it like instructions.

REMEMBER: Always end your response with either 'DECISION: answer' or 'DECISION: improve' on its own line.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original question: {question}\n\nConversation history:\n{messages}\n\nCurrent queries and results:\n{results}\n\nWhat feedback can you provide to guide the next query? Should we answer now or improve further?")
    ])
    
    result = await llm.ainvoke(
        prompt.format_messages(
            question=state["prompt"],
            messages=messages_text,
            results=queries_results_text
        )
    )
    
    # Parse the LLM's response for the decision marker
    content = result.content if hasattr(result, 'content') else str(result)
    if "DECISION: answer" in content:
        reflection_decision = "answer"
    else:
        reflection_decision = "improve"
    
    # Add reflection to messages and set the decision in state
    return {"messages": [result], "reflection_decision": reflection_decision}

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
You are a bilingual (Czech/English) data analyst. Respond strictly using provided SQL results:

1. **Data Rules**:
   - Use ONLY provided data (no external knowledge)
   - Preserve exact numbers (no rounding/formatting)
   
2. **Response Rules**:
   - Match question's language
   - Synthesize all data into one direct answer
   - Compare values when relevant
   - Highlight patterns if asked
   - Note contradictions if found

3. **Style Rules**:
   - No query/results references
   - No filler phrases
   - No unsupported commentary
   - Logical structure (e.g., highest-to-lowest)

Good: "X is 1234567 while Y is 7654321"
Bad: "The query shows X is 1,234,567"
"""
    
    # formatted_prompt = f"Question: {state['prompt']}\n\nQueries and Results:\n{queries_results_text}\n\nPlease answer the question based on the queries and results."
    formatted_prompt = f"Question: {state['prompt']}\n\nContext: \n{state['messages']}\n\nPlease answer the question based on the queries and results."
    
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
    debug_print(f"{ROUTE_DECISION_ID}: Enter increment_iteration_node")
    return {"iteration": state.get("iteration", 0) + 1}

async def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Submit the final answer to the user."""
    debug_print(f"{SUBMIT_FINAL_ID}: Enter submit_final_answer_node")
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

"""Graph node implementations for the data analysis workflow.

This module defines all the node functions used in the LangGraph workflow,
including schema loading, query generation, execution, and result formatting.
"""

#==============================================================================
# IMPORTS
#==============================================================================
import os
import json
import sqlite3
from pathlib import Path
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os

import sqlite3
import chromadb

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
RETRIEVE_NODE_ID = 20
RELEVANT_NODE_ID = 21
HYBRID_SEARCH_NODE_ID = 22
RERANK_NODE_ID = 23

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    
    
from .mcp_server import create_mcp_server
from .state import DataAnalysisState
from metadata.create_and_load_chromadb import (
    get_langchain_chroma_vectorstore,
    cohere_rerank,
    hybrid_search
)
from my_agent.utils.models import get_azure_llm_gpt_4o, get_azure_llm_gpt_4o_mini, get_ollama_llm


# Configurable iteration limit to prevent excessive looping
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', '1'))  # Configurable via environment variable, default 2
FORMAT_ANSWER_ID = 10  # Add to CONSTANTS section
ROUTE_DECISION_ID = 11  # ID for routing decision function
REFLECT_NODE_ID = 12
INCREMENT_ITERATION_ID = 13
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
def debug_print(msg: str) -> None:
    """Print debug messages when debug mode is enabled.
    
    Args:
        msg: The message to print
    """
    # Always check environment variable directly to respect runtime changes
    debug_mode = os.environ.get('MY_AGENT_DEBUG', '0')
    if debug_mode == '1':
        print(f"[DEBUG] {msg}")
        # Flush immediately to ensure output appears
        import sys
        sys.stdout.flush()

async def load_schema(state=None):
    """Load the schema metadata from the SQLite database based on top_selection_codes in state."""
    if state and state.get("top_selection_codes"):
        selection_codes = state["top_selection_codes"]
        db_path = BASE_DIR / "metadata" / "llm_selection_descriptions" / "selection_descriptions.db"
        schemas = []
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            for selection_code in selection_codes:
                cursor.execute(
                    """
                    SELECT extended_description FROM selection_descriptions
                    WHERE selection_code = ? AND extended_description IS NOT NULL AND extended_description != ''
                    """,
                    (selection_code,)
                )
                row = cursor.fetchone()
                if row:
                    schemas.append(f"Dataset: {selection_code}.\n" + row[0])
                else:
                    schemas.append(f"No schema found for selection_code {selection_code}.")
        except Exception as e:
            schemas.append(f"Error loading schema from DB: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
        return "\n**************\n".join(schemas)
    # fallback
    return "No selection_code provided in state."

#==============================================================================
# NODE FUNCTIONS
#==============================================================================
async def rewrite_query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Rewrite the user's prompt using an LLM, using only the summary and the current prompt.
    The messages list is always set to [summary, rewritten_message].
    """
    debug_print("REWRITE: Enter rewrite_query_node (simplified)")
    llm = get_azure_llm_gpt_4o(temperature=0.0)
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    prompt_text = state["prompt"]
    system_prompt = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a vector database.

CRITICAL RULES:
1. Your PRIMARY task is to rephrase the FOLLOW UP QUESTION, not to continue or repeat topics from chat history
2. Use the chat history ONLY to resolve pronouns and references in the follow up question
3. If the follow up question changes the topic or corrects something ("but I meant..."), follow the NEW topic from the follow up question
4. If you do not see any chat history, return the follow up question as is
5. Always preserve the user's intent from the follow up question
6. The rewritten question MUST NOT introduce, suggest, or add any information, examples, or details that were not explicitly present in the original question or chat history. Do not expand with examples or specifics unless they were directly asked for.
7. If a user provides an instruction instead of a question, combine the instruction with the original question or lastest summary context and rewrite the whole thing as a standalone question which is a followup based on latest summary discussion.

VECTOR SEARCH OPTIMIZATION:
7. Expand brief/vague questions into more detailed, searchable queries
8. Add relevant context terms that would help vector search find related documents
9. Include specific domains, locations, time periods, or categories when implied
10. Use complete sentences with clear subject-verb-object structure
11. Add synonyms or related terms that might appear in documents

EXAMPLES:

Example 1 - No history:
```
Summary of conversation so far: (empty)
Original question: How is Prague's population?
Standalone Question: How is Prague's population?
```

Example 2 - Reference resolution:
```
Summary of conversation so far: 
User asked about Prague's population and received information that Prague has 1.3 million inhabitants in 2023.
Original question: What about hotels there?
Standalone Question: What is the number and capacity of hotels in Prague?
```

Example 3 - Topic correction (MOST IMPORTANT):
```
Summary of conversation so far:
User asked about Prague's population and received information that Prague has 1.3 million inhabitants in 2023.
Original question: but I meant hotels
Standalone Question: How many hotels are there in Prague and what is their capacity?
```

Example 4 - Year reference with expansion:
```
Summary of conversation so far:
User asked about Prague's population in 2023 and received information that Prague had 1.3 million inhabitants in 2023.
Original question: what about 2024?
Standalone Question: What was Prague's population in 2024 compared to 2023?
```

Example 5 - Brief question expansion:
```
Summary of conversation so far: (empty)
Original question: hotels
Standalone Question: What is the number of hotels, hotel capacity, and accommodation facilities available?
```

Example 6 - Vague question improvement:
```
Summary of conversation so far:
User asked about Prague tourism and received information that Prague is a popular tourist destination with various attractions and services.
Original question: trends
Standalone Question: What are the current tourism trends, visitor statistics, and development patterns in Prague?
```

Example 7 - Vector search optimization (IMPORTANT):
```
Summary of conversation so far:
User asked about current population in Pilsen. Received data for 2023 showing that in Pilsen region the total population change per 1,000 inhabitants is 13.08862768. However, this data does not answer the question about actual population of Pilsen city. Need to get specific population number for Pilsen city in 2023, not just regional statistics.
Original question: but I meant hotels
Standalone Question: What is the total number of hotels and accommodation facilities in Pilsen, including their capacity in rooms and beds and occupancy statistics?
```

IMPORTANT: 
- If the follow up question is a correction, clarification, or topic change, prioritize the NEW intent over the chat history topic
- For brief questions (1-2 words), expand them into complete, searchable questions with relevant context
- Always maintain the original language and core intent while making the question more search-friendly

Now process this conversation:
"""
    human_prompt = f"Summary of conversation so far:\n{summary.content}\n\nOriginal question: {prompt_text}\nStandalone Question:"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    result = await llm.ainvoke(prompt.format_messages())
    rewritten_prompt = result.content.strip()
    debug_print(f"REWRITE: Rewritten prompt: {rewritten_prompt}")
    if not hasattr(result, "id") or not result.id:
        result.id = "rewrite_query"
    messages = [summary, result]
    return {
        "rewritten_prompt": rewritten_prompt,
        "messages": messages
    }

async def get_schema_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Get schema details for relevant columns. Messages list is always [summary, last_message]."""
    debug_print(f"{GET_SCHEMA_ID}: Enter get_schema_node")
    schema = await load_schema(state)
    msg = AIMessage(content=f"Schema details: {schema}", id="schema_details")
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    messages = [summary, msg]
    return {"messages": messages}


async def query_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Generate SQL query based on question and schema. Messages list is always [summary, last_message]."""
    debug_print(f"{QUERY_GEN_ID}: Enter query_node")
    
    # Log current state for debugging
    current_iteration = state.get("iteration", 0)
    existing_queries = state.get("queries_and_results", [])
    debug_print(f"{QUERY_GEN_ID}: Iteration {current_iteration}, existing queries count: {len(existing_queries)}")
    
    # Check for potential query loops by examining recent queries
    if len(existing_queries) >= 3:
        recent_queries = [q for q, r in existing_queries[-3:]]
        debug_print(f"{QUERY_GEN_ID}: Recent queries: {recent_queries}")
    
    llm = get_azure_llm_gpt_4o(temperature=0.0)
    # llm = get_ollama_llm("qwen:7b")
    tools = await create_mcp_server()
    sqlite_tool = next(tool for tool in tools if tool.name == "sqlite_query")
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    last_message = messages[1] if len(messages) > 1 else None
    
    # Skip last message if it's schema details to avoid duplication
    # (schema is already included separately in the prompt)
    if last_message and hasattr(last_message, 'id') and last_message.id == "schema_details":
        last_message_content = ""
    else:
        last_message_content = last_message.content if last_message else ""
    
    # Load schema before building the prompt
    schema = await load_schema(state)
    
    system_prompt = """
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect. 
Your task is to translate the user's natural-language question into a SQLITE SQL query and execute it using the sqlite_query tool.

To accomplish this, follow these steps:

=== OUTPUT - MOST IMPORTANT AND CRITICAL RULE FOR IT TO WORK!!!!!!!! ===
- Return ONLY the final SQLITE SQL QUERY that should be executed, with nothing else around it. 
- Do NOT wrap it in code fences, like ``` ``` and do NOT add explanations, only the SQLITE query.
- Do NOT add any other text or comments, only the SQLITE query.
- REMEMBER, only SQLITE SQL QUERY is allowed to be returned, nothing else, or tool called with it will fail.

1. Read and analyze the provided inputs:
- User prompt (can be in Czech or English)
- Read provided schemas carefully to you can understand how the data are laid, 
    layout can be non standard, but you have a loot of information there.
- Read Previous summary of the conversation
- Read The last message in the conversation
- Read Any feedback from the reflection agent, often in last message.

2. Process the prompt by:
- Identifying key terms in either language
- Matching terms to their Czech equivalents in the schema
- Handling Czech diacritics and special characters
- Converting concepts between languages

3. Construct an appropriate SQLITE SQL query by:
- Choosing the correct dataset and its schema
- Using exact column names from the schema provided (can be Czech or English), always use backticks around column names, like `Druh vlastnictví` = "Bytové družstvo";
- Matching user prompt terms to correct dimension values provided as a distinct list of values
- Ensuring proper string matching for Czech characters
- Generating a NEW SQLITE QUERY THAT PROVIDES ADDITIONAL INFORMATION that is not already present in the previously executed queries

4. Use the sqlite_query tool to execute the query.

6. Numeric outputs must be plain digits with NO thousands separators.

7. Be mindful about technical statistical terms - for example if someone asks about momentum, result should be some kind of rate of change.

Important Schema Details for one dataset:
- "dimensions" key contains several other keys, which are columns in the table
- Each of those columns under "dimensions" key contain "values" key with list of distinct values in that column
- If there is a column of type "metric / ukazatel", it means that it is a column that contains names of metrics, not values - it can be used in WHERE clause for filtering
- Column "value" is always the column that contains the numeric values for the metrics, it can be used in aggregations, like sum, etc.

HERE IS THE MOST IMPORTANT PART:
- Always read carefully all distinct values of dimensions, and do some thinking to choose the best ones to fit our question
- Use use LIKE or regex when filtering the dimensional values, if it is necessary for our question
- For example, if user asks about "female", but dimensional value are "start_period_female" and "end_period_female", just filter for %female%, if it makes sense

IMPORTANT notes about TOTAL records: 
- The dataset contains statistical records that include TOTAL ROWS for certain dimension values.
- These total rows, which may have been generated by SQL clauses such as GROUP BY WITH TOTALS or GROUP BY ROLLUP, should be ignored in calculations. 
- For instance, if the data includes regions within a republic, there may also be rows representing total values for the entire republic, which can be further split by dimensions like male/female. 
- When performing analyses such as distribution by regions, including these total records will result in percentage values being inaccurately halved. 
- Additionally, failing to exclude these totals during summarization will lead to double counting. 
- Always calculate using only the relevant data and separate pieces (excluding the rows with TOTALS), ensuring accuracy in statistical results.

IMPORTANT notes about SQL query generation:
- Return ONLY the SQL expression that answers the question.
- Limit the output to at most 10 rows using LIMIT unless the user specifies otherwise - but first think if you dont need to group it somehow so it returns reasonable 10 rows.
- Select only the necessary columns, never all columns.
- Use appropriate SQL aggregation functions when needed (e.g., SUM, AVG) 
- but always look carefully at the schema and distinct categorical values if your aggregations makes sence by this dimension or meric values.
- Column to Aggregate or extract numeric values is always called "value"! Never use different one or assume how its called.
- Do NOT modify the database.
- Always examine the ALL Schema to see how the data are laid out - column names and its concrete dimensional values. 
- If you are not sure with column names, call the tool with this query to get the table schema with column names: PRAGMA table_info(EP801) where EP801 is the table name.

=== OUTPUT - MOST IMPORTANT AND CRITICAL RULE FOR IT TO WORK!!!!!!!! ===
- Return ONLY the final SQLITE SQL QUERY that should be executed, with nothing else around it. 
- Do NOT wrap it in code fences, like ``` ``` and do NOT add explanations, only the SQLITE query.
- Do NOT add any other text or comments, only the SQLITE query.
- REMEMBER, only SQLITE SQL QUERY is allowed to be returned, nothing else, or tool called with it will fail.

"""
    # Build human prompt conditionally to avoid empty "Last message:" section
    human_prompt_parts = [
        f"User question: {state.get('rewritten_prompt') or state['prompt']}",
        f"Schema: {schema}",
        f"Summary of conversation:\n{summary.content}"
    ]
    
    if last_message_content:
        human_prompt_parts.append(f"Last message:\n{last_message_content}")
    
    human_prompt = "\n".join(human_prompt_parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    result = await llm.ainvoke(prompt.format_messages())
    new_queries = []
    query = result.content.strip()
    
    debug_print(f"{QUERY_GEN_ID}: Generated query: {query}")
    
    try:
        tool_result = await sqlite_tool.ainvoke({"query": query})
        if isinstance(tool_result, Exception):
            error_msg = f"Error executing query: {str(tool_result)}"
            debug_print(f"{QUERY_GEN_ID}: {error_msg}")
            new_queries.append((query, f"Error: {str(tool_result)}"))
            last_message = AIMessage(content=error_msg)
        else:
            debug_print(f"{QUERY_GEN_ID}: Successfully executed query: {query}")
            debug_print(f"{QUERY_GEN_ID}: Query result: {tool_result}")
            new_queries.append((query, tool_result))
            # Format the last message to include both query and result
            formatted_content = f"Query:\n{query}\n\nResult:\n{tool_result}"
            last_message = AIMessage(content=formatted_content, id="query_result")
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        debug_print(f"{QUERY_GEN_ID}: {error_msg}")
        new_queries.append((query, f"Error: {str(e)}"))
        last_message = AIMessage(content=error_msg)
    debug_print(f"{QUERY_GEN_ID}: Current state of queries_and_results: {new_queries}")
    messages = [summary, last_message]
    return {
        "rewritten_prompt": state.get("rewritten_prompt"),
        "messages": messages,
        "iteration": state["iteration"],
        "queries_and_results": new_queries
    }

async def reflect_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Reflect on the current state and provide feedback for next query or decide to answer.
    Messages list is always [summary, last_message].
    """
    debug_print(f"{REFLECT_NODE_ID}: Enter reflect_node")
    
    # Check current iteration and total queries to prevent excessive looping
    current_iteration = state.get("iteration", 0)
    total_queries = len(state.get("queries_and_results", []))
    
    debug_print(f"{REFLECT_NODE_ID}: Current iteration: {current_iteration}, Total queries: {total_queries}")
    
    # Force answer if we've hit iteration limit or have too many queries
    if current_iteration >= MAX_ITERATIONS:
        debug_print(f"{REFLECT_NODE_ID}: Forcing answer due to iteration limit ({current_iteration} >= {MAX_ITERATIONS})")
        # Create a simple reflection message
        messages = state.get("messages", [])
        summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
        from langchain_core.messages import AIMessage
        result = AIMessage(content="Maximum iterations reached. Proceeding to answer with available data.", id="reflect_forced")
        return {
            "messages": [summary, result],
            "reflection_decision": "answer",
            "iteration": current_iteration
        }
    
    llm = get_azure_llm_gpt_4o_mini(temperature=0.0)
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    last_message = messages[1] if len(messages) > 1 else None
    last_message_content = last_message.content if last_message else ""
    
    # Limit the queries_results_text to prevent token overflow
    queries_and_results = state.get("queries_and_results", [])
    
    # Only include the last few queries to prevent token overflow
    max_queries_for_reflection = 5  # Show only last 5 queries in reflection
    recent_queries = queries_and_results[-max_queries_for_reflection:] if len(queries_and_results) > max_queries_for_reflection else queries_and_results
    
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
        for i, (query, result) in enumerate(recent_queries)
    )
    
    # Add summary if we're showing limited queries
    if len(queries_and_results) > max_queries_for_reflection:
        total_queries_note = f"\n[Note: Showing last {max_queries_for_reflection} of {len(queries_and_results)} total queries]"
        queries_results_text = total_queries_note + "\n\n" + queries_results_text
    
    debug_print(f"{REFLECT_NODE_ID}: Processing {len(recent_queries)} queries for reflection (total: {len(queries_and_results)})")
    
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
  1. Review the original question and the summary of the conversation so far.
  2. Analyze all executed queries and their results.
  3. Provide detailed feedback about:
      - What specific information is still missing
      - What kind of SQL query would help get this information
      - Any patterns or insights that could be useful

Guidelines:
  - For comparison questions, ensure we have data for all entities being compared.
  - For trend analysis, ensure we have data across all relevant time periods.
  - For distribution questions, ensure we have complete coverage of all categories.
  - If you see repetitive or very similar queries, strongly consider answering with current data.

Your response should be detailed and specific, helping guide the next query. 
But it also must be to the point and not too long, max 400 words.

MOST IMPORTANT: 
If improvement will be needed - Imagine it is a chatbot and you are now playing a role of a HUMAN giving instructions to the LLM about 
how to improve the SQL QUERY - so phrase it like instructions.

REMEMBER: Always end your response with either 'DECISION: answer' or 'DECISION: improve' on its own line.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original question: {question}\n\nSummary of conversation:\n{summary}\nLast message:\n{last_message}\n\nCurrent queries and results:\n{results}\n\nWhat feedback can you provide to guide the next query? Should we answer now or improve further?")
    ])
    result = await llm.ainvoke(
        prompt.format_messages(
            question=state.get("rewritten_prompt") or state["prompt"],
            summary=summary.content,
            last_message=last_message_content,
            results=queries_results_text
        )
    )
    content = result.content if hasattr(result, 'content') else str(result)
    if "DECISION: answer" in content:
        reflection_decision = "answer"
        debug_print(f"{REFLECT_NODE_ID}: Decision: answer")
    else:
        reflection_decision = "improve"
        # Increment iteration when deciding to improve
        current_iteration += 1
        debug_print(f"{REFLECT_NODE_ID}: Decision: improve (iteration will be: {current_iteration})")
    if not hasattr(result, "id") or not result.id:
        result.id = "reflect"
    if last_message:
        messages = [summary, result]
    else:
        messages = [summary]
    return {
        "messages": messages,
        "reflection_decision": reflection_decision,
        "iteration": current_iteration
    }

async def format_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Format the query result into a natural language answer. Messages list is always [summary, last_message]."""
    debug_print(f"{FORMAT_ANSWER_ID}: Enter format_answer_node")
    llm = get_azure_llm_gpt_4o_mini(temperature=0.1)
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}" 
        for i, (query, result) in enumerate(state["queries_and_results"])
    )
    system_prompt = """
You are a bilingual (Czech/English) data analyst. Respond strictly using provided SQL results:

1. **Data Rules**:
   - Use ONLY provided data (no external knowledge)
   - Always read in details the QUERY and match it again with user question - if it does make sense.
   - Never format any numbers, use plain digits, no separators, no markdown etc.
   
2. **Response Rules**:
   - Match question's language
   - Synthesize all data into one direct answer
   - Compare values when relevant
   - Highlight patterns if asked
   - Note contradictions if found
   - Never hallucinate, if you are not sure about the answer or if the answer is not in the results, just say so.
   - Be careful not to say that something was 0 when you got no results from SQL.
   - Again read carefully the question, and provide answer using the QUERIES and its RESULTS, only if those answer the question. For example if question is about cinemas, dont answer about houses.
   

3. **Style Rules**:
   - No query/results references
   - No filler phrases
   - No unsupported commentary
   - Logical structure (e.g., highest-to-lowest)
   - Make output more structured, instead of making one long sentence.

Example regarding numeric output:
Good: "X is 1234567 while Y is 7654321"
Bad: "The query shows X is 1,234,567"
"""
    formatted_prompt = f"Question: {state.get('rewritten_prompt') or state['prompt']}\n\nContext: \n{state['queries_and_results']}\n\nPlease answer the question based on the queries and results."
    chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    result = await llm.ainvoke(
        chain.format_messages(input=formatted_prompt)
    )
    debug_print(f"{FORMAT_ANSWER_ID}: Analysis completed")
    
    # Extract the final answer content
    final_answer_content = result.content if hasattr(result, 'content') else str(result)
    debug_print(f"{FORMAT_ANSWER_ID}: Final answer: {final_answer_content[:100]}...")
    
    # Update messages state (existing logic)
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    if not hasattr(result, "id") or not result.id:
        result.id = "format_answer"
    messages = [summary, result]
    
    # Return both messages and final_answer states
    return {
        "messages": messages,
        "final_answer": final_answer_content
    }

async def increment_iteration_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Increment the iteration counter and return updated state."""
    debug_print(f"{INCREMENT_ITERATION_ID}: Enter increment_iteration_node")
    return {"iteration": state.get("iteration", 0) + 1}

async def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Submit the final answer to the user."""
    debug_print(f"{SUBMIT_FINAL_ID}: Enter submit_final_answer_node")
    return state

async def save_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Save the result to a file."""
    debug_print(f"{SAVE_RESULT_ID}: Enter save_node")
    # Get the final answer from the last message
    final_answer = state["messages"][-1].content if state.get("messages") and len(state["messages"]) > 1 else ""
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

async def retrieve_similar_selections_hybrid_search_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Perform hybrid search on ChromaDB to retrieve initial candidate documents. Returns hybrid search results as Document objects."""
    debug_print(f"{HYBRID_SEARCH_NODE_ID}: Enter retrieve_similar_selections_hybrid_search_node")
    query = state.get("rewritten_prompt") or state["prompt"]
    n_results = state.get("n_results", 60)  # Increased from 20 to 60 to capture more relevant documents

    debug_print(f"{HYBRID_SEARCH_NODE_ID}: Query: {query}")
    debug_print(f"{HYBRID_SEARCH_NODE_ID}: Requested n_results: {n_results}")

    # Check if ChromaDB directory exists
    chroma_db_dir = BASE_DIR / "metadata" / "czsu_chromadb"
    if not chroma_db_dir.exists() or not chroma_db_dir.is_dir():
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: ChromaDB directory not found at {chroma_db_dir}")
        return {"hybrid_search_results": [], "chromadb_missing": True}

    try:
        chroma_vectorstore = get_langchain_chroma_vectorstore(
            collection_name=CHROMA_COLLECTION_NAME,
            chroma_db_path=str(CHROMA_DB_PATH),
            embedding_model_name=EMBEDDING_DEPLOYMENT
        )
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: ChromaDB vectorstore initialized")
        
        hybrid_results = hybrid_search(chroma_vectorstore._collection, query, n_results=n_results)
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: Retrieved {len(hybrid_results)} hybrid search results")
        
        # Convert dict results to Document objects for compatibility
        from langchain_core.documents import Document
        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(
                page_content=result['document'],
                metadata=result['metadata']
            )
            hybrid_docs.append(doc)
        
        # Debug: Show detailed hybrid search results
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: Detailed hybrid search results:")
        for i, doc in enumerate(hybrid_docs[:10], 1):  # Show first 10
            selection = doc.metadata.get('selection') if doc.metadata else 'N/A'
            content_preview = doc.page_content[:100].replace('\n', ' ') if hasattr(doc, 'page_content') else 'N/A'
            debug_print(f"{HYBRID_SEARCH_NODE_ID}: #{i}: {selection} | Content: {content_preview}...")
        
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: All selection codes: {[doc.metadata.get('selection') for doc in hybrid_docs]}")
        return {"hybrid_search_results": hybrid_docs}
    except Exception as e:
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: Error in hybrid search: {e}")
        import traceback
        debug_print(f"{HYBRID_SEARCH_NODE_ID}: Traceback: {traceback.format_exc()}")
        return {"hybrid_search_results": []}

async def rerank_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Rerank hybrid search results using Cohere rerank model. Returns selection codes and Cohere rerank scores."""
    # Force debug mode on for this node to ensure visibility
    original_debug = os.environ.get('MY_AGENT_DEBUG', '0')
    os.environ['MY_AGENT_DEBUG'] = '1'
    
    debug_print(f"🔥🔥🔥 {RERANK_NODE_ID}: ===== RERANK NODE EXECUTING ===== 🔥🔥🔥")
    debug_print(f"{RERANK_NODE_ID}: Enter rerank_node")
    query = state.get("rewritten_prompt") or state["prompt"]
    hybrid_results = state.get("hybrid_search_results", [])
    n_results = state.get("n_results", 60)  # Increased from 20 to 60 to match hybrid search

    debug_print(f"{RERANK_NODE_ID}: Query: {query}")
    debug_print(f"{RERANK_NODE_ID}: Number of hybrid results received: {len(hybrid_results)}")
    debug_print(f"{RERANK_NODE_ID}: Requested n_results: {n_results}")

    # Check if we have hybrid search results to rerank
    if not hybrid_results:
        debug_print(f"{RERANK_NODE_ID}: No hybrid search results to rerank")
        os.environ['MY_AGENT_DEBUG'] = original_debug  # Restore debug setting
        return {"most_similar_selections": []}

    # Debug: Show input to rerank
    debug_print(f"{RERANK_NODE_ID}: Input hybrid results for reranking:")
    for i, doc in enumerate(hybrid_results[:10], 1):  # Show first 10
        selection = doc.metadata.get('selection') if doc.metadata else 'N/A'
        content_preview = doc.page_content[:100].replace('\n', ' ') if hasattr(doc, 'page_content') else 'N/A'
        debug_print(f"{RERANK_NODE_ID}: #{i}: {selection} | Content: {content_preview}...")

    try:
        debug_print(f"{RERANK_NODE_ID}: Calling cohere_rerank with {len(hybrid_results)} documents")
        reranked = cohere_rerank(query, hybrid_results, top_n=n_results)
        debug_print(f"{RERANK_NODE_ID}: Cohere returned {len(reranked)} reranked results")
        
        most_similar = []
        for i, (doc, res) in enumerate(reranked, 1):
            selection_code = doc.metadata.get("selection") if doc.metadata else None
            score = res.relevance_score
            most_similar.append((selection_code, score))
            # Debug: Show detailed rerank results
            if i <= 10:  # Show top 10 results
                debug_print(f"{RERANK_NODE_ID}: Rerank #{i}: {selection_code} | Score: {score:.6f}")
        
        debug_print(f"🎯🎯🎯 {RERANK_NODE_ID}: FINAL RERANK OUTPUT: {most_similar[:5]} 🎯🎯🎯")
        os.environ['MY_AGENT_DEBUG'] = original_debug  # Restore debug setting
        return {"most_similar_selections": most_similar}
    except Exception as e:
        debug_print(f"{RERANK_NODE_ID}: Error in reranking: {e}")
        import traceback
        debug_print(f"{RERANK_NODE_ID}: Traceback: {traceback.format_exc()}")
        os.environ['MY_AGENT_DEBUG'] = original_debug  # Restore debug setting
        return {"most_similar_selections": []}

async def relevant_selections_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Select the top 3 reranked selections if their Cohere relevance score exceeds the threshold (0.0005)."""
    debug_print(f"{RELEVANT_NODE_ID}: Enter relevant_selections_node")
    SIMILARITY_THRESHOLD = 0.0005  # Minimum Cohere rerank score required
    most_similar = state.get("most_similar_selections", [])
    # Select up to 3 top selections above threshold
    top_selection_codes = [sel for sel, score in most_similar if sel is not None and score is not None and score >= SIMILARITY_THRESHOLD][:3]
    debug_print(f"{RELEVANT_NODE_ID}: top_selection_codes: {top_selection_codes}")
    return {"top_selection_codes": top_selection_codes}

async def summarize_messages_node(state: DataAnalysisState) -> DataAnalysisState:
    """Node: Summarize the conversation so far, always setting messages to [summary, last_message]."""
    debug_print("SUMMARY: Enter summarize_messages_node")
    llm = get_azure_llm_gpt_4o_mini(temperature=0.0)
    messages = state.get("messages", [])
    summary = messages[0] if messages and isinstance(messages[0], SystemMessage) else SystemMessage(content="")
    last_message = messages[1] if len(messages) > 1 else None
    prev_summary = summary.content
    last_message_content = last_message.content if last_message else ""
    debug_print(f"SUMMARY: prev_summary: '{prev_summary}'")
    debug_print(f"SUMMARY: last_message_content: '{last_message_content}'")
    if not prev_summary and not last_message_content:
        debug_print("SUMMARY: Skipping summarization (no previous summary or last message).")
        return {"messages": [summary] if not last_message else [summary, last_message]}
    system_prompt = """
You are a conversation summarization agent.
Your job is to maintain a concise, cumulative summary of a data analysis conversation between a 
user and an AI assistant. 
Each time you are called, you receive the previous summary (which may be empty) and 
the latest message. Update the summary to include any new information, decisions, or 
context from the latest message. The summary should be suitable for 
providing context to an LLM in future queries. 
Be concise but do not omit important details. 
Do not include any meta-commentary or formatting, just the summary text."""
    human_prompt = f"Previous summary:\n{prev_summary}\n\nLatest message:\n{last_message_content}\n\nUpdate the summary to include the latest message."
    debug_print(f"SUMMARY: human_prompt: {human_prompt}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    result = await llm.ainvoke(prompt.format_messages())
    new_summary = result.content.strip()
    debug_print(f"SUMMARY: Updated summary: {new_summary}")
    summary_msg = SystemMessage(content=new_summary)
    if last_message:
        messages = [summary_msg, last_message]
    else:
        messages = [summary_msg]
    debug_print(f"SUMMARY: New messages: {[getattr(m, 'content', None) for m in messages]}")
    return {"messages": messages}


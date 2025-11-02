module_description = r"""Node Implementations for LangGraph Multi-Agent Text-to-SQL Workflow

This module implements all node functions used in the LangGraph workflow, defining the complete
processing pipeline from query rewriting through parallel retrieval, SQL generation, reflection,
answer formatting, and resource cleanup. Each node is an async function that receives and returns
a DataAnalysisState dictionary.

The module contains 16 node functions plus 2 helper functions, organized into logical processing
stages that handle query optimization, dual-source retrieval (database + PDF), SQL execution,
iterative improvement, and answer synthesis.

Architecture Overview:
=====================
Node functions are organized into 6 processing stages:
1. Query Preprocessing (rewrite_prompt_node, summarize_messages_node)
2. Parallel Retrieval - Database Selections (retrieve, rerank, relevant)
3. Parallel Retrieval - PDF Chunks (retrieve, rerank, relevant)
4. SQL Generation & Execution (get_schema, generate_query_node)
5. Reflection & Improvement (reflect_node)
6. Answer Finalization (format_answer, submit_final_answer, save, cleanup)

Key Features:
============
1. Query Rewriting & Optimization:
   - LLM-powered conversational context resolution
   - Pronoun and reference replacement
   - Topic change detection ("but I meant...")
   - Brief question expansion (e.g., "hotels" → detailed query)
   - Vector search optimization with synonyms
   - Bilingual support (Czech/English)
   - Escape handling for f-string safety

2. Dual-Source Parallel Retrieval:
   - Database selections: ChromaDB hybrid search on dataset descriptions
   - PDF chunks: Hybrid search on parsed documentation content
   - Independent parallel execution paths
   - Shared hybrid search algorithm (semantic + BM25)
   - Cohere reranking for both sources
   - Configurable result counts and thresholds

3. Schema Loading & Agentic SQL Generation:
   - Dynamic schema retrieval from SQLite metadata DB
   - Schema truncation for token efficiency
   - MCP (Model Context Protocol) tool integration with remote/local fallback
   - Agentic tool calling pattern: LLM autonomously decides when to execute queries
   - LLM can call sqlite_query tool multiple times (up to MAX_TOOL_ITERATIONS)
   - Iterative data gathering: LLM examines results and decides if more queries needed
   - Comprehensive SQL generation prompts with bilingual support
   - Czech character and diacritics handling
   - JOIN optimization and aggregation guidance
   - CELKEM (total) row handling instructions

4. Agentic Reflection & Self-Correction:
   - Analyzes SQL query results for completeness
   - Decides "improve" (generate better query) or "answer" (sufficient data)
   - Provides specific feedback for query improvement
   - Handles missing data, errors, and ambiguous results
   - Iteration limit enforcement (MAX_ITERATIONS)
   - Limited query history to prevent token overflow

5. Multi-Source Answer Synthesis:
   - Combines SQL results and PDF chunks into single answer
   - Bilingual response generation (matches query language)
   - Markdown formatting support
   - Clear source attribution (SQL vs PDF)
   - Numerical data presentation without separators
   - Fallback to PDF-only when no database results

6. Memory & Resource Management:
   - Message summarization at key points
   - ChromaDB client cleanup with garbage collection
   - Minimal checkpoint state (5 essential fields only)
   - Explicit memory release after retrieval
   - Token-efficient state management

Processing Stages & Nodes:
==========================

Stage 1: Query Preprocessing
----------------------------
Nodes: rewrite_prompt_node, summarize_messages_node

rewrite_prompt_node:
- Input: state["prompt"], state["messages"] (summary + last message)
- Output: rewritten_prompt, updated messages
- LLM: Azure GPT-4o (temp=0.0)
- Purpose: Converts conversational questions into standalone, search-optimized queries
- Key Logic:
  * Resolves pronouns/references using conversation summary
  * Detects topic corrections ("but I meant X")
  * Expands vague queries with domain context
  * Adds synonyms for better vector search
  * Preserves original language
  * Escapes curly braces for f-string safety

summarize_messages_node:
- Input: state["messages"] (may be long list)
- Output: Updated messages [summary (SystemMessage), last_message]
- LLM: Azure GPT-4o-mini (temp=0.0)
- Purpose: Maintains bounded memory by summarizing conversation history
- Key Logic:
  * Combines previous summary with latest message
  * Generates cumulative summary of conversation
  * Always returns 2-message structure
  * Enables token-efficient context preservation
- Called at 4 points: after rewrite, query, reflect, format

Stage 2: Database Selection Retrieval
-------------------------------------
Nodes: retrieve_similar_selections_hybrid_search_node, rerank_table_descriptions_node, relevant_selections_node

retrieve_similar_selections_hybrid_search_node:
- Input: rewritten_prompt, n_results (default: 20)
- Output: hybrid_search_results (List[Document])
- Purpose: Hybrid search on dataset selection descriptions
- Key Logic:
  * Checks ChromaDB directory existence (local) or uses cloud
  * Initializes ChromaDB client and collection
  * Calls hybrid_search() (semantic + BM25 weighted combination)
  * Converts dict results to Document objects
  * Memory cleanup: Explicitly closes client and runs gc.collect()
  * Sets chromadb_missing flag if directory not found

rerank_table_descriptions_node:
- Input: hybrid_search_results, rewritten_prompt, n_results (default: 20)
- Output: most_similar_selections (List[Tuple[str, float]])
- Purpose: Rerank with Cohere multilingual model for better relevance
- Key Logic:
  * Calls cohere_rerank() with query and documents
  * Extracts selection_code and relevance_score from results
  * Returns list of (selection_code, score) tuples
  * Debug logging shows top 10 reranked results

relevant_selections_node:
- Input: most_similar_selections
- Output: top_selection_codes (List[str]), clears intermediate results
- Purpose: Filter by threshold and select top 3 selection codes
- Key Logic:
  * Filters by SIMILARITY_THRESHOLD (0.0005)
  * Takes top 3 codes above threshold
  * Clears hybrid_search_results and most_similar_selections
  * Sets final_answer="No Relevant Selections Found" if empty

Stage 3: PDF Chunk Retrieval
----------------------------
Nodes: retrieve_similar_chunks_hybrid_search_node, rerank_chunks_node, relevant_chunks_node

retrieve_similar_chunks_hybrid_search_node:
- Input: rewritten_prompt, n_results (default: 15)
- Output: hybrid_search_chunks (List[Document])
- Purpose: Hybrid search on parsed PDF documentation
- Key Logic:
  * Translates query to English using Azure Translator API
  * Checks PDF ChromaDB directory existence or uses cloud
  * Calls pdf_hybrid_search() on PDF collection
  * Converts results to Document objects
  * Memory cleanup: Closes client and runs gc.collect()
  * Returns empty list if PDF_FUNCTIONALITY_AVAILABLE=False

rerank_chunks_node:
- Input: hybrid_search_chunks, rewritten_prompt, n_results (default: 5)
- Output: most_similar_chunks (List[Tuple[Document, float]])
- Purpose: Rerank PDF chunks with Cohere
- Key Logic:
  * Calls pdf_cohere_rerank() with query and documents
  * Returns list of (Document, relevance_score) tuples
  * Debug logging shows top N reranked chunks

relevant_chunks_node:
- Input: most_similar_chunks
- Output: top_chunks (List[Document]), clears intermediate results
- Purpose: Filter PDF chunks by relevance threshold
- Key Logic:
  * Filters by PDF_RELEVANCE_THRESHOLD (0.01)
  * Keeps all chunks above threshold (no top-k limit)
  * Clears hybrid_search_chunks and most_similar_chunks
  * Debug shows chunk count and previews

Stage 4: SQL Generation & Execution
-----------------------------------
Nodes: get_schema_node, generate_query_node
Helper: load_schema()

get_schema_node:
- Input: top_selection_codes
- Output: Updated messages with schema details
- Purpose: Load database schema for selected tables
- Key Logic:
  * Calls load_schema() helper function
  * Creates AIMessage with schema content
  * Preserves summary, replaces last_message
  * Sets message id="schema_details" for tracking

load_schema() helper:
- Input: top_selection_codes (from state)
- Output: Formatted schema string
- Purpose: Retrieve extended descriptions from metadata DB
- Key Logic:
  * Connects to selection_descriptions.db
  * Queries for extended_description by selection_code
  * Joins multiple schemas with separator
  * Returns "No schema found" if code missing

generate_query_node:
- Input: messages (with schema), rewritten_prompt, top_selection_codes, iteration, queries_and_results
- Output: Updated queries_and_results, messages, iteration
- LLM: Azure GPT-4o (temp=0.0)
- MCP Tool: sqlite_query
- Purpose: Generate and execute SQL query
- Key Logic:
  * Loads schema again for prompt context
  * Builds comprehensive SQL generation system prompt:
    - Bilingual instructions (Czech/English)
    - Schema structure explanation (dimensions, metrics, values)
    - CELKEM (total) row handling warnings
    - Czech character matching guidelines
    - Column aliasing best practices
    - CRITICAL: Output only raw SQL query, no code fences
  * Skips schema_details message to avoid duplication
  * Invokes LLM to generate SQL query
  * Executes query using sqlite_query MCP tool
  * Handles execution errors gracefully
  * Appends (query, result) to queries_and_results
  * Creates AIMessage with query + result
  * Debug logging for query and result

Stage 5: Reflection & Improvement
---------------------------------
Nodes: reflect_node

reflect_node:
- Input: queries_and_results, messages, iteration, rewritten_prompt
- Output: reflection_decision ("improve" | "answer"), updated messages, iteration
- LLM: Azure GPT-4o-mini (temp=0.0)
- Purpose: Analyze results and decide next action
- Key Logic:
  * Forces "answer" if iteration >= MAX_ITERATIONS
  * Limits query history to last 5 for reflection (token efficiency)
  * Builds reflection prompt with:
    - Original question
    - Conversation summary
    - Recent queries and results
    - Guidelines for completeness (comparisons, trends, distributions)
  * LLM analyzes and provides feedback
  * Extracts DECISION from response: "answer" or "improve"
  * Increments iteration if "improve" chosen
  * Returns reflection message with decision

Stage 6: Answer Finalization
----------------------------
Nodes: format_answer_node, submit_final_answer_node, save_node, cleanup_resources_node

format_answer_node:
- Input: queries_and_results, top_chunks, rewritten_prompt
- Output: final_answer, updated messages, top_chunks (preserved)
- LLM: Azure GPT-4o-mini (temp=0.1)
- Purpose: Synthesize multi-source answer with markdown formatting
- Key Logic:
  * Builds SQL context from queries_and_results
  * Builds PDF context from top_chunks (max 10 chunks)
  * Separate sections for SQL and PDF data
  * Comprehensive system prompt:
    - Use ONLY provided data (SQL + PDF)
    - Match question's language
    - No number formatting (plain digits)
    - Markdown formatting (bullets, headings, tables)
    - Clear source attribution
    - Combine SQL and PDF insights
  * Generates formatted answer
  * Stores in final_answer field
  * Preserves top_chunks for frontend display
  * Debug logging for PDF chunk usage

submit_final_answer_node:
- Input: final_answer, messages, queries_and_results, top_chunks, top_selection_codes
- Output: Explicitly preserved final_answer and essential state
- Purpose: Ensure final_answer persists through checkpointing
- Key Logic:
  * Explicitly returns final_answer from state
  * Preserves messages, queries_and_results, top_chunks, top_selection_codes
  * Debug logging for final_answer length and preview
  * Ensures answer reaches API/frontend

save_node:
- Input: prompt, queries_and_results, final_answer, full state
- Output: Minimal checkpoint state (5 essential fields)
- Purpose: Save results to file and create minimal checkpoint
- Key Logic:
  * Optionally saves to text file (if SAVE_TO_FILE_TXT_JSONL=1)
  * Appends to JSONL for memory efficiency
  * Creates minimal checkpoint with only:
    - prompt
    - queries_and_results
    - most_similar_selections
    - most_similar_chunks
    - final_answer
    - messages (for API compatibility)
  * Dramatically reduces checkpoint size
  * Debug logging for checkpoint fields

cleanup_resources_node:
- Input: full state
- Output: Essential state fields only
- Purpose: Final memory cleanup and garbage collection
- Key Logic:
  * Creates state_copy with 6 essential fields:
    - prompt, final_answer, queries_and_results
    - messages, top_chunks, top_selection_codes
  * Runs aggressive garbage collection (2 passes)
  * Catches circular references
  * Ensures ChromaDB resources released
  * Debug logging for GC object counts

Helper Functions:
================

translate_to_english():
- Input: text (any language)
- Output: English translation
- Purpose: Translate PDF search queries to English
- API: Azure Translator API
- Key Logic:
  * Loads API credentials from environment
  * Constructs translation endpoint with params
  * Runs synchronous request in thread pool
  * Returns translated text
  * Used before PDF chunk retrieval

Constants & Configuration:
==========================

Debug IDs (for tracing):
- GET_SCHEMA_ID = 3
- GENERATE_QUERY_ID = 4
- SUBMIT_FINAL_ID = 7
- SAVE_RESULT_ID = 8
- RETRIEVE_NODE_ID = 20
- RELEVANT_NODE_ID = 21
- HYBRID_SEARCH_NODE_ID = 22
- RERANK_TABLE_DESCRIPTIONS_NODE_ID = 23
- RETRIEVE_CHUNKS_NODE_ID = 24
- RERANK_CHUNKS_NODE_ID = 25
- RELEVANT_CHUNKS_NODE_ID = 26
- FORMAT_ANSWER_ID = 10
- REFLECT_NODE_ID = 12

Workflow Control:
- MAX_ITERATIONS: 1 (default, env configurable)
- SAVE_TO_FILE_TXT_JSONL: 0 (file saving disabled by default)

Paths:
- BASE_DIR: Project root (auto-detected from __file__ or cwd)
- CHROMA_DB_PATH: metadata/czsu_chromadb
- CHROMA_COLLECTION_NAME: "czsu_selections_chromadb"
- PDF_CHROMA_DB_PATH: data/pdf_chromadb_llamaparse
- PDF_COLLECTION_NAME: "pdf_document_collection"

Retrieval Configuration:
- Database selections:
  * SELECTIONS_HYBRID_SEARCH_DEFAULT_RESULTS: 20
  * SQL_RELEVANCE_THRESHOLD: 0.0005 (Cohere score)
  * Top selection codes: 3
- PDF chunks:
  * PDF_HYBRID_SEARCH_DEFAULT_RESULTS: 15
  * PDF_N_TOP_CHUNKS: 5 (for debug display)
  * PDF_RELEVANCE_THRESHOLD: 0.01 (Cohere score)

LLM Models:
- Query rewriting: Azure GPT-4o (temp=0.0)
- Query generation: Azure GPT-4o (temp=0.0)
- Reflection: Azure GPT-4o-mini (temp=0.0)
- Answer formatting: Azure GPT-4o-mini (temp=0.1)
- Message summarization: Azure GPT-4o-mini (temp=0.0)

Embedding Model:
- EMBEDDING_DEPLOYMENT: "text-embedding-3-large__test1"

Node Execution Patterns:
========================

Memory Management Pattern:
-------------------------
All retrieval nodes follow this pattern:
1. Initialize ChromaDB client
2. Get collection
3. Perform search operation
4. Convert results to required format
5. Explicitly set client/collection to None
6. Delete client reference
7. Run gc.collect()
8. Return results

Example:
```python
client = get_chromadb_client(path, collection_name)
collection = client.get_collection(name)
results = hybrid_search(collection, query, n_results)
# ... process results ...
collection = None
del client
gc.collect()
return {"results": processed_results}
```

Message Structure Pattern:
--------------------------
All nodes maintain [summary, last_message] structure:
```python
summary = (
    messages[0] if messages and isinstance(messages[0], SystemMessage)
    else SystemMessage(content="")
)
last_message = messages[1] if len(messages) > 1 else None
# ... process ...
return {"messages": [summary, new_message]}
```

State Update Pattern:
--------------------
Nodes return partial state updates:
```python
return {
    "field1": new_value1,
    "field2": new_value2,
    # LangGraph merges with existing state
}
```

Clearing Intermediate Data Pattern:
----------------------------------
After processing, clear intermediate results:
```python
return {
    "final_results": processed_data,
    "intermediate_results": [],  # Clear to save memory
}
```

Error Handling Patterns:
========================

ChromaDB Missing:
----------------
```python
if not chroma_db_dir.exists() or not chroma_db_dir.is_dir():
    return {"hybrid_search_results": [], "chromadb_missing": True}
```

SQL Execution Error:
-------------------
```python
try:
    tool_result = await sqlite_tool.ainvoke({"query": query})
    if isinstance(tool_result, Exception):
        new_queries = [(query, f"Error: {str(tool_result)}")]
        last_message = AIMessage(content=error_msg)
except Exception as e:
    new_queries = [(query, f"Error: {str(e)}")]
    last_message = AIMessage(content=error_msg)
```

Graceful Degradation:
--------------------
```python
if not PDF_FUNCTIONALITY_AVAILABLE:
    return {"hybrid_search_chunks": []}
```

Debug Logging System:
====================
Three debug functions with unique IDs:
- print__nodes_debug(): Node-level operations
- print__chromadb_debug(): ChromaDB/retrieval operations
- print__analysis_tracing_debug(): Graph-level tracing

Example usage:
```python
print__nodes_debug(f"🧠 {GENERATE_QUERY_ID}: Generated query: {query}")
print__nodes_debug(f"✅ {GENERATE_QUERY_ID}: Successfully executed query")
print__nodes_debug(f"❌ {GENERATE_QUERY_ID}: Error: {error_msg}")
```

Emoji conventions:
- 🧠: LLM operations
- 🔍: Search/retrieval
- 🔄: Reranking/iteration
- 🎯: Selection/filtering
- 📊: Results
- 📄: PDF operations
- 💾: Schema/save operations
- 📝: Summarization
- 🎨: Formatting
- 📤: Submit
- 🧹: Cleanup
- ✅: Success
- ❌: Error

Integration Points:
==================
This module integrates with:
1. my_agent/utils/state.py: DataAnalysisState TypedDict
2. my_agent/utils/models.py: LLM client getters
3. my_agent/utils/mcp_server.py: MCP tool creation
4. metadata/create_and_load_chromadb__04.py: Hybrid search, cohere_rerank
5. data/pdf_to_chromadb.py: PDF hybrid search, pdf_cohere_rerank
6. metadata/chromadb_client_factory.py: ChromaDB client management
7. api/utils/debug.py: Debug logging functions

Usage Example:
=============
Nodes are called automatically by LangGraph based on graph structure:

```python
# Define in agent.py
graph.add_node("rewrite_prompt", rewrite_prompt_node)
graph.add_node("generate_query", generate_query_node)
# ... etc

# Graph executes nodes based on edges
graph.add_edge("rewrite_prompt", "summarize_messages_rewrite")
graph.add_edge("generate_query", "reflect")

# Nodes receive state and return updates
result = await rewrite_prompt_node(state)
# Returns: {"rewritten_prompt": "...", "messages": [...]}
```

Performance Considerations:
==========================
1. Memory Management:
   - Explicit client cleanup after retrieval
   - Garbage collection after ChromaDB operations
   - Minimal checkpoint state (5 fields vs full 15)
   - Limited query history (10 most recent)
   - Capped message summarization points (4 total)

2. Token Efficiency:
   - Message summarization at key points
   - Schema truncation for large datasets
   - Limited query context in reflection (last 5)
   - Limited PDF chunks in formatting (max 10)
   - Removing intermediate results after use

3. API Efficiency:
   - Batch operations where possible
   - Reusing clients within nodes
   - Async operations throughout
   - Connection pooling via factories

4. Parallel Execution:
   - Database and PDF retrieval run simultaneously
   - Independent branches until synchronization
   - No blocking between parallel paths

Quality & Validation:
====================
1. Query Generation:
   - Comprehensive SQL generation instructions
   - Czech character handling guidance
   - CELKEM (total) row warnings
   - Aggregation safety checks
   - Output format validation (raw SQL only)

2. Reflection:
   - Completeness checks (comparisons, trends, distributions)
   - Pattern detection (repetitive queries)
   - Specific improvement suggestions
   - Iteration limit enforcement

3. Answer Formatting:
   - Multi-source synthesis (SQL + PDF)
   - Source attribution required
   - Data validation (query vs question match)
   - No hallucination policy
   - Markdown formatting standards

See my_agent/agent.py for graph structure connecting these nodes.
See my_agent/utils/state.py for DataAnalysisState schema.
"""

"""Graph node implementations for the data analysis workflow.

This module defines all the node functions used in the LangGraph workflow,
including schema loading, query generation, execution, and result formatting.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import uuid
import json
import logging
from pathlib import Path
import gc
import traceback
import re

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool


# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Static IDs for easier debug‑tracking
GET_SCHEMA_ID = 3
GENERATE_QUERY_ID = 4
CHECK_QUERY_ID = 5
EXECUTE_QUERY_ID = 6
SUBMIT_FINAL_ID = 7
SAVE_RESULT_ID = 8
SHOULD_CONTINUE_ID = 9
RETRIEVE_NODE_ID = 20
RELEVANT_NODE_ID = 21
HYBRID_SEARCH_NODE_ID = 22
RERANK_TABLE_DESCRIPTIONS_NODE_ID = 23
# New PDF chunk node IDs
RETRIEVE_CHUNKS_NODE_ID = 24
RERANK_CHUNKS_NODE_ID = 25
RELEVANT_CHUNKS_NODE_ID = 26

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
    print(f"🔍 BASE_DIR calculated from __file__: {BASE_DIR}")
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
    print(f"🔍 BASE_DIR calculated from cwd: {BASE_DIR}")

SAVE_TO_FILE_TXT_JSONL = 0

print(f"🔍 Current working directory: {Path.cwd()}")

# Import debug functions from utils
from api.utils.debug import print__nodes_debug, print__analysis_tracing_debug

# Import helper functions
from .helpers import load_schema, translate_to_english, detect_language

# PDF chunk functionality imports
from data.pdf_to_chromadb import CHROMA_DB_PATH as PDF_CHROMA_DB_PATH
from data.pdf_to_chromadb import COLLECTION_NAME as PDF_COLLECTION_NAME
from data.pdf_to_chromadb import cohere_rerank as pdf_cohere_rerank
from data.pdf_to_chromadb import hybrid_search as pdf_hybrid_search
from metadata.create_and_load_chromadb__04 import (
    cohere_rerank,
    hybrid_search,
)
from metadata.chromadb_client_factory import (
    get_chromadb_client,
)
from my_agent.utils.models import (
    get_azure_llm_gpt_4o,
    get_azure_llm_gpt_4o_mini,
    get_azure_llm_gpt_4o_4_1,
)
from metadata.chromadb_client_factory import should_use_cloud
from .mcp_server import create_mcp_server
from .state import DataAnalysisState


# ==============================================================================
# HELPER TOOLS
# ==============================================================================
@tool
def finish_gathering():
    """Call this tool when you have gathered sufficient data to answer the user's question."""
    return "Data gathering finished."


# Set up logger
logger = logging.getLogger(__name__)

PDF_FUNCTIONALITY_AVAILABLE = True

# Configurable iteration limit to prevent excessive looping
MAX_ITERATIONS = int(
    os.environ.get("MAX_ITERATIONS", "1")
)  # Configurable via environment variable, default 2

# Maximum number of tool calls the LLM can make in generate_query_node (agentic pattern)
MAX_TOOL_ITERATIONS = int(
    os.environ.get("MAX_TOOL_ITERATIONS", "5")
)  # Default: LLM can call sqlite_query tool up to 5 times

FORMAT_ANSWER_ID = 10  # Add to CONSTANTS section
POST_RETRIEVAL_SYNC_ID = 11  # ID for post-retrieval synchronization function
REFLECT_NODE_ID = 12
INCREMENT_ITERATION_ID = 13
CHROMA_DB_PATH = BASE_DIR / "metadata" / "czsu_chromadb"
CHROMA_COLLECTION_NAME = "czsu_selections_chromadb"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large__test1"

# PDF Chunk Processing Configuration
PDF_HYBRID_SEARCH_DEFAULT_RESULTS = (
    20  # Number of chunks to retrieve from PDF hybrid search
)
PDF_N_TOP_CHUNKS = (
    5  # Number of top chunks to keep in top_chunks state and show in debug
)
SQL_RELEVANCE_THRESHOLD = 0.0005  # Minimum relevance score for SQL selections
PDF_RELEVANCE_THRESHOLD = 0.0005  # Minimum relevance score for PDF chunks

# Database Selections Processing Configuration
SELECTIONS_HYBRID_SEARCH_DEFAULT_RESULTS = (
    20  # Number of selections to retrieve from hybrid search
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


# ==============================================================================
# NODE FUNCTIONS
# ==============================================================================
async def rewrite_prompt_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that rewrites user prompts for optimal vector search and standalone context.

    This node converts conversational questions into standalone, search-optimized queries by resolving
    pronouns, detecting topic changes, expanding brief questions, and adding relevant context terms.
    It uses Azure GPT-4o to maintain the original language while improving searchability.

    The rewriting process handles:
    - Pronoun resolution using conversation summary
    - Topic corrections ("but I meant...")
    - Brief question expansion ("hotels" → detailed query)
    - Vector search optimization with synonyms
    - Escape handling for f-string safety

    Args:
        state (DataAnalysisState): Workflow state containing prompt and message history.

    Returns:
        DataAnalysisState: Updated state with 'rewritten_prompt' and messages [summary, rewritten_message].

    Key Steps:
        1. Extract original prompt and conversation summary
        2. Call Azure GPT-4o with rewriting instructions
        3. Generate standalone, search-optimized query
        4. Escape curly braces for f-string safety
        5. Return rewritten prompt and updated messages
    """
    print__nodes_debug("🧠 REWRITE: Enter rewrite_prompt_node (simplified)")

    prompt_text = state["prompt"]
    print(f"🧠 REWRITE: Original prompt: {prompt_text}")
    messages = state.get("messages", [])
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )

    llm = get_azure_llm_gpt_4o(temperature=0.0)

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
    # FIX: Escape curly braces in content to prevent f-string parsing errors
    summary.content.replace("{", "{{").replace("}", "}}")
    prompt_text.replace("{", "{{").replace("}", "}}")

    human_prompt = "Summary of conversation so far:\n{summary_content}\n\nOriginal question: {prompt_text}\nStandalone Question:"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    result = await llm.ainvoke(
        prompt.format_messages(summary_content=summary.content, prompt_text=prompt_text)
    )
    rewritten_prompt = result.content.strip()
    # FIX: Escape curly braces in rewritten_prompt to prevent f-string parsing errors
    rewritten_prompt_escaped = rewritten_prompt.replace("{", "{{").replace("}", "}}")
    print__nodes_debug(f"🚀 REWRITE: Rewritten prompt: {rewritten_prompt_escaped}")
    if not hasattr(result, "id") or not result.id:
        result.id = "rewrite_prompt"

    return {"rewritten_prompt": rewritten_prompt, "messages": [summary, result]}


async def followup_prompts_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that generates follow-up prompt suggestions for continued data exploration.

    This node uses Azure GPT-4o-mini with high temperature (1.0) to generate 3 diverse, interesting
    follow-up prompts based on the conversation summary. Prompts are context-aware and help users
    discover related insights about Czech Statistical Office data.

    The generated prompts cover different aspects like economy, population, finance, etc., and are
    designed to be directly usable as natural language queries for the system.

    Args:
        state (DataAnalysisState): Workflow state containing conversation summary.

    Returns:
        DataAnalysisState: Updated state with 'followup_prompts' containing list of 3 prompt strings.

    Key Steps:
        1. Extract conversation summary from messages
        2. Call Azure GPT-4o-mini with creative temperature (1.0)
        3. Generate 3 diverse, relevant prompts
        4. Parse and validate prompt list
        5. Return maximum 3 prompts
    """
    print__nodes_debug("💡 FOLLOWUP_PROMPTS: Enter followup_prompts_node")

    messages = state.get("messages", [])
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )
    summary_content = summary.content

    print__nodes_debug(f"💡 FOLLOWUP_PROMPTS: Summary content: '{summary_content}'")

    # Use LLM with temperature 1.0 for creative prompt generation
    llm = get_azure_llm_gpt_4o_mini(temperature=1.0)

    system_prompt = """
You are a prompt generation assistant for a Czech Statistical Office data analysis system.

About our data:
Summary data on Czechia provides selected data from individual areas with a focus on the real economy, 
monetary and fiscal indicators. They gather data from the CZSO as well as data from other institutions, 
such as the Czech National Bank, the Ministry of Finance and others.

Your task: Generate exactly 3 diverse interesting and useful follow-up prompts based on the conversation summary that users might use to continue exploring the data.

Important guidelines:
- Prompts don't have to be questions - they can be statements, commands, or other types of intents, but structured in a way that can be directly provided to LLM.
- Be concise and to the point - prompts should be brief
- Each prompt should be on a new line
- Don't number the prompts
- If there's a conversation summary, make prompts relevant to the topics discussed
- If the summary is empty, generate general prompts covering different aspects (economy, population, finance, etc.)
- Make them natural and user-friendly
- Ensure prompts are diverse and explore different angles or related topics
"""

    human_prompt = "Conversation summary:\n{summary_content}\n\nGenerate 3 diverse most relevant and most interesting follow-up prompts for the user to continue exploring the data."
    print__nodes_debug("💡 FOLLOWUP_PROMPTS: Calling LLM to generate prompts")

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    result = await llm.ainvoke(
        prompt.format_messages(summary_content=summary_content or "(empty)")
    )
    generated_text = result.content.strip()
    print__nodes_debug(
        f"💡 FOLLOWUP_PROMPTS: LLM returned {len(generated_text)} characters"
    )

    # Parse the generated prompts (split by newlines and filter empty lines)
    followup_prompts = [
        line.strip() for line in generated_text.split("\n") if line.strip()
    ]
    print__nodes_debug(
        f"💡 FOLLOWUP_PROMPTS: Parsed {len(followup_prompts)} prompts from LLM response"
    )

    # Return maximum 3 prompts
    final_prompts = followup_prompts[:3]
    print__nodes_debug(
        f"💡 FOLLOWUP_PROMPTS: Returning {len(final_prompts)} follow-up prompts"
    )
    for i, p in enumerate(final_prompts, 1):
        print__nodes_debug(f"💡 FOLLOWUP_PROMPTS:   {i}. {p}")

    return {"followup_prompts": final_prompts}


async def get_schema_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that retrieves database schema for relevant dataset selections.

    This node loads schema information for the top dataset selections identified by the retrieval
    process. The schema includes table names, column names, data types, and distinct values for
    categorical dimensions, which are essential for SQL query generation.

    Args:
        state (DataAnalysisState): Workflow state containing top_selection_codes and messages.

    Returns:
        DataAnalysisState: Updated state with messages [summary, schema_details_message].

    Key Steps:
        1. Extract top_selection_codes from state
        2. Load schema using load_schema helper function
        3. Create AIMessage with schema details
        4. Return updated messages preserving summary
    """
    print__nodes_debug(f"💾 {GET_SCHEMA_ID}: Enter get_schema_node")

    top_selection_codes = state.get("top_selection_codes")
    messages = state.get("messages", [])
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )

    schema = await load_schema({"top_selection_codes": top_selection_codes})
    msg = AIMessage(content=f"Schema details: {schema}", id="schema_details")

    return {"messages": [summary, msg]}


async def generate_query_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that generates and executes SQLite queries using agentic tool calling pattern.

    This node implements an agentic workflow where the LLM autonomously decides when to execute
    SQL queries and when it has gathered sufficient data. The LLM can call the sqlite_query tool
    multiple times (up to MAX_TOOL_ITERATIONS) to iteratively gather information.

    The node uses Azure GPT-4o with tool binding, allowing the LLM to:
    - Generate SQL queries based on user questions
    - Execute queries using the MCP (Model Context Protocol) sqlite_query tool
    - Examine results and decide if more queries are needed
    - Iteratively refine data collection until sufficient information is gathered

    The MCP tool automatically handles remote/local server fallback, providing robust query execution
    with comprehensive error handling and logging.

    Args:
        state (DataAnalysisState): Workflow state containing prompt, schema, messages, and iteration info.

    Returns:
        DataAnalysisState: Updated state with messages, queries_and_results, iteration, and rewritten_prompt.

    Key Steps:
        1. Extract state variables (current_iteration, existing_queries_and_results, messages, rewritten_prompt, original_prompt, selected_codes)
        2. Check for potential query loops by examining recent queries
        3. Set up LLM and MCP tools (get Azure GPT-4o model, create MCP server, find sqlite_tool, add finish_gathering tool)
        4. Extract conversation summary and last message (get summary_message, last_message, determine last_message_content)
        5. Load schema for selected datasets using load_schema helper
        6. Build SQL generation prompt with system and human messages (define system_prompt, build human_prompt_parts, prepare template_vars)
        7. Bind tools to LLM and initialize conversation (bind tools, create prompt_template, format initial_messages, set conversation_messages)
        8. Enter agentic loop while tool_call_count < MAX_TOOL_ITERATIONS (invoke LLM, check for tool calls, execute tools, append results)
        9. After loop: Create completion message and return updated state with messages, iteration, and queries_and_results

    System Prompt Key Points:
    - Bilingual Data Query Specialist proficient in Czech and English, expert in SQL with SQLite dialect
    - Task: Translate natural-language questions into SQLite queries using sqlite_query tool
    - Tool usage: Access to sqlite_query tool, can call up to MAX_TOOL_ITERATIONS times, use preparatory queries, call finish_gathering when sufficient data gathered
    - Process prompt: Read user prompt, schemas, summary, last message; identify key terms, match to schema, handle Czech diacritics, convert concepts
    - Construct queries: Use provided schemas, exact column names with backticks, match to distinct values, ensure proper string matching for Czech characters
    - Iterative tool use: Call tool with queries, examine results, decide if more data needed, continue until comprehensive data obtained
    - Numeric outputs: Plain digits with no thousands separators
    - Technical terms: Mindful of statistical terms like momentum as rate of change
    - Schema details: Dataset blocks with descriptions, columns, distinct values, hierarchical organization; "value" column contains numeric metrics
    - TOTAL records (CELKEM): Ignore in calculations to avoid double counting, exclude totals when summing or aggregating
    - SQL generation: Limit to 10 rows, select necessary columns, use aggregations carefully, include metric columns in SELECT/GROUP BY, do not modify database, use PRAGMA for schema if unsure, alias columns appropriately
    - Verification: Review results, execute additional queries if incomplete, stop when sufficient data gathered without further tool calls
    - Examples: Dataset descriptions, sample queries for various SQL techniques
    """

    print__nodes_debug(f"🧠 {GENERATE_QUERY_ID}: Enter generate_query_node")

    # Key Step 1: Extract state variables (current_iteration, existing_queries_and_results, messages, rewritten_prompt, original_prompt, selected_codes)
    current_iteration = state.get("iteration", 0)
    existing_queries_and_results = state.get("queries_and_results", [])
    messages = state.get("messages", [])
    rewritten_prompt = state.get("rewritten_prompt")
    original_prompt = state["prompt"]
    selected_codes = state.get("top_selection_codes")

    print__nodes_debug(
        f"🔄 {GENERATE_QUERY_ID}: Iteration {current_iteration}, existing queries count: {len(existing_queries_and_results)}"
    )
    # Key Step 2: Check for potential query loops by examining recent queries
    if len(existing_queries_and_results) >= 3:
        recent_queries = [query for query, _ in existing_queries_and_results[-3:]]
        print__nodes_debug(f"🔄 {GENERATE_QUERY_ID}: Recent queries: {recent_queries}")

    # Key Step 3: Set up LLM and MCP tools (get Azure GPT-4o model, create MCP server, find sqlite_tool, add finish_gathering tool)
    llm = get_azure_llm_gpt_4o_4_1(temperature=0.0)
    tools = await create_mcp_server()
    sqlite_tool = next((tool for tool in tools if tool.name == "sqlite_query"), None)
    if not sqlite_tool:
        error_msg = "sqlite_query tool not found in MCP server"
        print__nodes_debug(f"❌ {GENERATE_QUERY_ID}: {error_msg}")
        return {
            "messages": messages,
            "queries_and_results": existing_queries_and_results
            + [("ERROR", error_msg)],
            "iteration": current_iteration,
        }

    # Add finish gathering tool
    tools.append(finish_gathering)

    # Key Step 4: Extract conversation summary and last message (get summary_message, last_message, determine last_message_content)
    summary_message = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )
    last_message = messages[1] if len(messages) > 1 else None

    # Skip last message if it's schema details to avoid duplication in prompt
    last_message_content = ""
    if (
        last_message
        and hasattr(last_message, "id")
        and last_message.id == "schema_details"
    ):
        last_message_content = ""
    else:
        last_message_content = last_message.content if last_message else ""

    # Key Step 5: Load schema for selected datasets using load_schema helper
    schema_data = await load_schema({"top_selection_codes": selected_codes})

    # Key Step 6: Build SQL generation prompt with system and human messages (define system_prompt, build human_prompt_parts, prepare template_vars)
    system_prompt = f"""
You are a Bilingual Data Query Specialist proficient in both Czech and English and an expert in SQL with SQLite dialect. 
Your task is to translate the user's natural-language question into SQLite SQL queries using the sqlite_query tool.

=== TOOL USAGE - CRITICAL INSTRUCTIONS ===
- You have access to the sqlite_query tool that executes SQLITE SQL queries on the database
- You can call this tool up to {MAX_TOOL_ITERATIONS} times to gather all necessary information
- You can use some preparatory queries first to examine data structure and understand what information is available
- After each tool call, you'll see the results and can decide if you need more data
- IMPORTANT: When you have sufficient information to answer the user's question, call the finish_gathering tool with no arguments. Do not generate any text response. The data you gathered will be used later to format the final answer.
- Use the tool iteratively to refine your understanding and gather comprehensive data
- If the last message contains reflection feedback indicating that more queries are needed to answer the question, you MUST call the sqlite_query tool to gather additional data. Do not provide an answer based on existing summary data when reflection feedback requests more queries.

- Do not ask the user for additional information; provide the best answer possible based on available data.

1. Read and analyze the provided inputs:
- User prompt (can be in Czech or English)
- Read provided schemas carefully so you can understand how the data are laid out, 
    layout can be non standard, but you have a lot of information there.
- Read the previous summary of the conversation
- Read the last message in the conversation
- Read any feedback from the reflection agent, often in the last message and take it into account.

2. Process the prompt by:
- Identifying key terms in either language
- Matching terms to their Czech equivalents in the schema
- Handling Czech diacritics and special characters
- Converting concepts between languages

3. Construct appropriate SQLite SQL queries by:
- You are provided with the schemas of the 3 most relevant datasets; choose and use any of these datasets to construct your queries
- Using exact column names from the schema provided (can be Czech or English), always use backticks around column names, like `Druh vlastnictví` = "Bytové družstvo";
- Matching user prompt terms to correct dimension values provided as a distinct list of values
- Ensuring proper string matching for Czech characters
- Each query should provide NEW information that is not already present in previously executed queries

4. Iteratively use the sqlite_query tool:
- Call the tool with each SQL query you construct
- Examine the results after each execution
- Determine if additional queries are needed to fully answer the question
- Continue calling the tool until you have comprehensive data
- When satisfied with the information gathered, provide your final response summarizing the findings

6. Numeric outputs must be plain digits with NO thousands separators.

7. Be mindful about technical statistical terms - for example if someone asks about momentum, results should be some kind of rate of change.

Important Schema Details for the provided datasets:
- Each dataset block starts with "Dataset: [CODE]" where CODE is the table name (e.g., ZAM06AT1)
- Each dataset includes:
  -- A description of what the dataset focuses on (employment, job applicants, self-employed persons, etc.)
  -- A list of columns with Czech names (e.g., "Ukazatel", "Pohlaví", "Roky", "Věkové skupiny")
  -- For each column, the available distinct values are explicitly listed in quotes - use them exactly as written - you must match user queries to these exact values
  -- Hierarchical organization explanation showing how data is structured (geographic, temporal, demographic dimensions)
- Column "value" is always the column that contains the numeric values for the metrics, it can be used in aggregations, like sum, etc. - but don't aggregate without thinking if it makes sense, always read full record, often it is not necessary to aggregate when record contains 'celkem' or 'total' in one of the columns.
- Columns like "Ukazatel" or similar contain metric names and can be used in WHERE clauses for filtering

HERE IS THE MOST IMPORTANT PART:
- Always read carefully all available values for each column in the relevant datasets
- Choose the dataset that best matches the user's question based on the description and available data
- Use LIKE '%term%' when the user's term is a substring of available values
- For age groups, regions, or other categorical data, find the closest matching available values
- If user asks for "total" or "all", look for "Celkem" values in the data

IMPORTANT notes about TOTAL records (CELKEM): 
- The dataset contains statistical records that include TOTAL ROWS for certain dimension values.
- These total rows, which may have been generated by SQL clauses such as GROUP BY WITH TOTALS or GROUP BY ROLLUP, should be ignored in calculations, so be careful if a value says "celkem", which means "total" in Czech.
- For instance, if the data includes regions within a republic, there may also be rows representing total values (CELKEM) for the entire republic, which can be further split by dimensions like male/female. 
- When performing analyses such as distribution by regions, including these total records will result in percentage values being inaccurately halved. 
- Additionally, failing to exclude these totals (CELKEM) during summarization will lead to double counting. 
- Always calculate using only the relevant data and separate pieces (excluding the rows with TOTALS (CELKEM)), ensuring accuracy in statistical results.

IMPORTANT notes about SQL query generation:
- Limit the output to at most 10 rows using LIMIT unless the user specifies otherwise - but first think if you don't need to group it somehow so it returns reasonable 10 rows.
- Select only the necessary columns, never all columns.
- Use appropriate SQL aggregation functions when needed (e.g., SUM, AVG) 
- but always look carefully at the schema and distinct categorical values if your aggregations make sense by this dimension or metric values.
- Column to Aggregate or extract numeric values is always called "value"! Never use different one or assume how it's called.
- Do NOT modify the database.
- Always examine the ALL Schema to see how the data are laid out - column names and their concrete dimensional values. 
- If you are not sure with column names, call the tool with this query to get the table schema with column names: PRAGMA table_info(EP801) where EP801 is the table name. Then use it to generate correct query and use tool to get results.
- Be careful about how you ALIAS (AS Clause) the Column names to make sense of the data - based on what you use in where or group by clause.
- ALWAYS INCLUDE COLUMN "metric" or "ukazatel" columns in SELECT and GROUP BY clauses if present - it will provide additional information about meaning of 'value' column in the result.

=== VERIFICATION AND COMPLETION ===
- After executing queries, review the results to ensure they answer the user's question
- If the results are incomplete or unclear, execute additional queries to gather more information
- When you have sufficient data, STOP immediately without generating any text response
- Simply do not make any more tool calls - the system will automatically format your gathered data into an answer

=== EXAMPLE DATASET AND SAMPLE QUERIES ===
Below is an example of how dataset schemas are provided and how to construct effective SQL queries:

EXAMPLE DATASET DESCRIPTION:
************************
Dataset: ZAM06AT1.
This dataset focuses on the employment rate by age group. The data is collected at the yearly level and covers various territory types, including the state, NUTS2 regions, and individual regions within the Czech Republic. The dataset includes the following columns: "Ukazatel", "Pohlaví", "Roky", "ČR, Reg. soudržnosti, Kraje", and "Věkové skupiny".

For "Ukazatel", the available value is "Míra zaměstnanosti (%)".

For "Pohlaví", the available value is "Celkem".

For "Roky", the available value is "2023".

For "ČR, Reg. soudržnosti, Kraje", the available values are "Česko", "Praha", "Střední Čechy".

For "Věkové skupiny", the available values are "15 až 19", "20 až 24".

The data is organized hierarchically, with "ČR, Reg. soudržnosti, Kraje" representing the geographic dimension, "Roky" representing the temporal dimension, and "Věkové skupiny" representing the demographic dimension. The metric "Ukazatel" provides the employment rate as a percentage, and "Pohlaví" specifies the gender category, which in this dataset is aggregated as "Celkem".
************************

EXAMPLE DATA ROWS:
"Ukazatel","Pohlaví","Roky","ČR, Reg. soudržnosti, Kraje","Věkové skupiny","value"
"Míra zaměstnanosti (%)","Celkem","2024","Česko","15 až 19","5.0298978485"
"Míra zaměstnanosti (%)","Celkem","2024","Praha","15 až 19","5.4763366729"
"Míra zaměstnanosti (%)","Celkem","2024","Praha","20 až 24","45.029916104"

EXAMPLE SQL QUERIES FOR THIS DATASET:
Below are practical query examples showcasing different SQL techniques:

1. Simple WHERE filtering - Getting total for Czech Republic:
   SELECT `Ukazatel`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `ČR, Reg. soudržnosti, Kraje` = "Česko" 
   LIMIT 10;
   -- Note: "Česko" represents the total for Czech Republic. When such total rows exist,
   -- you don't need to sum over individual regions to avoid double counting.
   -- Always include Ukazatel to provide context about what the value represents.

2. Filtering specific age group:
   SELECT `Ukazatel`, `ČR, Reg. soudržnosti, Kraje`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `Věkové skupiny` = "20 až 24" 
   LIMIT 10;

3. Using LIKE for partial matching:
   SELECT `Ukazatel`, `ČR, Reg. soudržnosti, Kraje`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `ČR, Reg. soudržnosti, Kraje` LIKE "%Praha%" 
   LIMIT 10;

4. Multiple WHERE conditions with AND:
   SELECT `Ukazatel`, `ČR, Reg. soudržnosti, Kraje`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `Věkové skupiny` = "25 až 29" 
   AND `ČR, Reg. soudržnosti, Kraje` = "Praha";

5. Using IN clause for multiple age groups:
   SELECT `Ukazatel`, `ČR, Reg. soudržnosti, Kraje`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `Věkové skupiny` IN ("20 až 24", "25 až 29") 
   AND `ČR, Reg. soudržnosti, Kraje` = "Česko";

6. GROUP BY with SUM (use with caution - avoid when totals exist and user asks about totals):
   SELECT `Ukazatel`, `Věkové skupiny`, SUM(`value`) as total_employment_rate 
   FROM ZAM06AT1 
   WHERE `ČR, Reg. soudržnosti, Kraje` NOT LIKE "%Česko%" 
   AND `ČR, Reg. soudržnosti, Kraje` NOT LIKE "%soudržnosti%" 
   GROUP BY `Ukazatel`, `Věkové skupiny` 
   LIMIT 10;
   -- Note: We exclude "Česko" and cohesion regions (soudržnosti) to avoid totals.
   -- When using GROUP BY, always include Ukazatel in both SELECT and GROUP BY.

7. Ordering results by value:
   SELECT `Ukazatel`, `ČR, Reg. soudržnosti, Kraje`, `Věkové skupiny`, `value` 
   FROM ZAM06AT1 
   WHERE `Věkové skupiny` = "20 až 24" 
   ORDER BY `value` DESC 
   LIMIT 10;

8. Finding average employment rate by age group:
   SELECT `Ukazatel`, `Věkové skupiny`, AVG(`value`) as avg_employment_rate 
   FROM ZAM06AT1 
   WHERE `ČR, Reg. soudržnosti, Kraje` LIKE "%kraj" 
   GROUP BY `Ukazatel`, `Věkové skupiny` 
   ORDER BY avg_employment_rate DESC;
   -- Note: Using "kraj" suffix to filter only individual regions, excluding totals.
   -- Including Ukazatel in GROUP BY ensures the metric name is preserved in results.

Remember: Always examine the schema to understand:
- What columns contain categorical dimensions vs. metrics
- What values are available for filtering
- Whether total/aggregate rows ("Celkem", "Česko") exist that should be used instead of summing
- How to properly match user queries to exact column values

"""
    # Build human prompt conditionally to avoid empty sections
    human_prompt_parts = [
        "\n**************\nUser question: {user_question}",
        "\n**************\nSchemas:\n{schema}",
        "\n**************\nSummary of conversation:\n{summary_content}",
    ]

    if last_message_content:
        human_prompt_parts.append(
            "\n**************\nLast message:\n{last_message_content}"
        )

    human_prompt = "\n".join(human_prompt_parts)

    # Prepare template variables
    template_vars = {
        "user_question": rewritten_prompt or original_prompt,
        "schema": schema_data,
        "summary_content": summary_message.content,
    }

    if last_message_content:
        template_vars["last_message_content"] = last_message_content

    # Key Step 7: Bind tools to LLM and initialize conversation (bind tools, create prompt_template, format initial_messages, set conversation_messages)
    llm_with_tools = llm.bind_tools(tools)

    # Build initial messages for the conversation
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    initial_messages = prompt_template.format_messages(**template_vars)

    # Enter agentic loop: LLM generates queries and calls tool iteratively
    conversation_messages = list(initial_messages)
    new_queries_and_results = []
    tool_call_count = 0
    finished = False

    print__nodes_debug(
        f"🔄 {GENERATE_QUERY_ID}: Starting agentic loop (max iterations: {MAX_TOOL_ITERATIONS})"
    )

    # Key Step 8: Enter agentic loop while tool_call_count < MAX_TOOL_ITERATIONS (invoke LLM, check for tool calls, execute tools, append results)
    while tool_call_count < MAX_TOOL_ITERATIONS:
        tool_call_count += 1
        print__nodes_debug(
            f"🔄 {GENERATE_QUERY_ID}: Tool iteration {tool_call_count}/{MAX_TOOL_ITERATIONS}"
        )

        # Invoke LLM (may return tool calls or signal completion)
        llm_response = await llm_with_tools.ainvoke(conversation_messages)

        # Check if LLM wants to use tools
        if not llm_response.tool_calls:
            # No more tool calls - LLM decided it has enough data
            print__nodes_debug(
                f"✅ {GENERATE_QUERY_ID}: LLM finished gathering data (no more tool calls)"
            )
            break

        # LLM wants to call tools - add its message to conversation
        conversation_messages.append(llm_response)
        print__nodes_debug(
            f"🔧 {GENERATE_QUERY_ID}: LLM requested {len(llm_response.tool_calls)} tool call(s)"
        )

        # Execute each tool call
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", str(uuid.uuid4()))

            print__nodes_debug(
                f"🔧 {GENERATE_QUERY_ID}: Executing tool '{tool_name}' with call_id: {tool_call_id}"
            )

            if tool_name == "finish_gathering":
                # Execute finish tool
                tool_result = await finish_gathering.ainvoke({})
                result_text = str(tool_result)
                tool_message = ToolMessage(
                    content=result_text, tool_call_id=tool_call_id
                )
                conversation_messages.append(tool_message)
                finished = True
                break
            elif "query" in tool_args:
                sql_query = tool_args["query"]
                print__nodes_debug(f"⚡ {GENERATE_QUERY_ID}: SQL Query: {sql_query}")

                # Execute the query using MCP tool
                try:
                    tool_result = await sqlite_tool.ainvoke({"query": sql_query})

                    # Extract text from potential wrapper formats
                    result_text = str(tool_result)
                    # Handle TextContent wrapper if present
                    text_content_match = re.match(
                        r"^\[TextContent\(type='text', text='(.+)', annotations=None\)\]$",
                        result_text,
                        re.DOTALL,
                    )
                    if text_content_match:
                        result_text = text_content_match.group(1)

                    print__nodes_debug(
                        f"✅ {GENERATE_QUERY_ID}: Query executed successfully"
                    )
                    print__nodes_debug(f"📊 {GENERATE_QUERY_ID}: Result: {result_text}")

                    # Store query and result
                    new_queries_and_results.append((sql_query, result_text))

                    # Create tool message with result
                    tool_message = ToolMessage(
                        content=result_text, tool_call_id=tool_call_id
                    )
                    conversation_messages.append(tool_message)

                except Exception as e:
                    error_msg = f"Error executing query: {str(e)}"
                    print__nodes_debug(f"❌ {GENERATE_QUERY_ID}: {error_msg}")
                    logger.error("❌ %s: %s", GENERATE_QUERY_ID, error_msg)

                    # Store error as result
                    new_queries_and_results.append((sql_query, f"Error: {str(e)}"))

                    # Send error back to LLM
                    tool_message = ToolMessage(
                        content=error_msg, tool_call_id=tool_call_id
                    )
                    conversation_messages.append(tool_message)
            else:
                # Tool call without required query parameter
                error_msg = "Tool call missing 'query' parameter"
                print__nodes_debug(f"❌ {GENERATE_QUERY_ID}: {error_msg}")
                # Log error but don't append to results or conversation to avoid confusion

        if finished:
            break

        # Continue loop so LLM can see results and decide if more queries are needed

    else:
        print__nodes_debug(
            f"⚠️ {GENERATE_QUERY_ID}: Max tool iterations ({MAX_TOOL_ITERATIONS}) reached"
        )

    # Key Step 9: After loop: Create completion message and return updated state with messages, iteration, and queries_and_results
    # Create completion message
    completion_message = AIMessage(
        content=f"Data gathering complete. {len(new_queries_and_results)} queries executed.",
        id="query_result",
    )

    print__nodes_debug(
        f"🔄 {GENERATE_QUERY_ID}: Agentic loop complete. Total queries executed: {len(new_queries_and_results)}"
    )
    print__nodes_debug(
        f"🔄 {GENERATE_QUERY_ID}: Current state of queries_and_results: {new_queries_and_results}"
    )

    # Update state with new queries and results
    return {
        "rewritten_prompt": rewritten_prompt,
        "messages": [summary_message, completion_message],
        "iteration": current_iteration,
        "queries_and_results": new_queries_and_results,
    }


async def reflect_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that analyzes query results and decides whether to improve or answer.

    This is the agentic reflection component that implements self-correction and iterative improvement.
    It analyzes executed queries and their results, determines if sufficient data exists to answer the
    user's question, and provides specific feedback for query improvement if needed.

    The node enforces iteration limits (MAX_ITERATIONS) and uses Azure GPT-4o-mini to make decisions.
    It returns either "answer" (proceed to formatting) or "improve" (generate better query) along with
    detailed feedback for the query generation node.

    Args:
        state (DataAnalysisState): Workflow state containing queries_and_results, messages, and iteration count.

    Returns:
        DataAnalysisState: Updated state with reflection_decision, messages, and incremented iteration.

    Key Steps:
        1. Check iteration limit and force answer if exceeded
        2. Extract recent queries (last 5) to prevent token overflow
        3. Call Azure GPT-4o-mini with reflection prompt
        4. Parse decision ("answer" or "improve")
        5. Increment iteration counter if improving
        6. Return decision and feedback message
    """
    print__nodes_debug(f"💭 {REFLECT_NODE_ID}: Enter reflect_node")

    current_iteration = state.get("iteration", 0)
    queries_and_results = state.get("queries_and_results", [])
    messages = state.get("messages", [])
    rewritten_prompt = state.get("rewritten_prompt")
    prompt = state["prompt"]

    total_queries = len(queries_and_results)

    print__nodes_debug(
        f"🧠 {REFLECT_NODE_ID}: Current iteration: {current_iteration}, Total queries: {total_queries}"
    )

    # Force answer if we've hit iteration limit or have too many queries
    if current_iteration >= MAX_ITERATIONS:
        print__nodes_debug(
            f"🔄 {REFLECT_NODE_ID}: Forcing answer due to iteration limit ({current_iteration} >= {MAX_ITERATIONS})"
        )
        # Create a simple reflection message
        summary = (
            messages[0]
            if messages and isinstance(messages[0], SystemMessage)
            else SystemMessage(content="")
        )

        result = AIMessage(
            content="Maximum iterations reached. Proceeding to answer with available data.",
            id="reflect_forced",
        )
        return {
            "messages": [summary, result],
            "reflection_decision": "answer",
            "iteration": current_iteration,
        }

    llm = get_azure_llm_gpt_4o_mini(temperature=0.0)
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )
    last_message = messages[1] if len(messages) > 1 else None
    last_message_content = last_message.content if last_message else ""

    # Limit the queries_results_text to prevent token overflow
    # Only include the last few queries to prevent token overflow
    max_queries_for_reflection = 5  # Show only last 5 queries in reflection
    recent_queries = (
        queries_and_results[-max_queries_for_reflection:]
        if len(queries_and_results) > max_queries_for_reflection
        else queries_and_results
    )

    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}"
        for i, (query, result) in enumerate(recent_queries)
    )

    # Add summary if we're showing limited queries
    if len(queries_and_results) > max_queries_for_reflection:
        total_queries_note = f"\n[Note: Showing last {max_queries_for_reflection} of {len(queries_and_results)} total queries]"
        queries_results_text = total_queries_note + "\n\n" + queries_results_text

    print__nodes_debug(
        f"🧠 {REFLECT_NODE_ID}: Processing {len(recent_queries)} queries for reflection (total: {len(queries_and_results)})"
    )

    system_prompt = """
You are a data analysis reflection agent.
Your task is to analyze the current state and provide detailed feedback to guide the next query generation.

NOTE: The query generation agent uses an agentic tool calling pattern, where it can execute multiple SQL queries 
iteratively. Your feedback will help it understand what additional information is needed.

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
  - Remember that the query agent can execute multiple queries, so you can request comprehensive data gathering.

Your response should be detailed and specific, helping guide the next query. 
But it also must be to the point and not too long, max 400 words.

MOST IMPORTANT: 
If improvement will be needed - Imagine it is a chatbot and you are now playing a role of a HUMAN giving instructions to the LLM about 
what additional information to gather using SQL queries - so phrase it like instructions.

REMEMBER: Always end your response with either 'DECISION: answer' or 'DECISION: improve' on its own line.
"""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Original question: {question}\n\nSummary of conversation:\n{summary}\nLast message:\n{last_message}\n\nCurrent queries and results:\n{results}\n\nWhat feedback can you provide to guide the next query? Should we answer now or improve further?",
            ),
        ]
    )
    result = await llm.ainvoke(
        prompt_template.format_messages(
            question=rewritten_prompt or prompt,
            summary=summary.content,
            last_message=last_message_content,
            results=queries_results_text,
        )
    )
    content = result.content if hasattr(result, "content") else str(result)
    if "DECISION: answer" in content:
        reflection_decision = "answer"
        print__nodes_debug(f"✅ {REFLECT_NODE_ID}: Decision: answer")
    else:
        reflection_decision = "improve"
        # Increment iteration when deciding to improve
        current_iteration += 1
        print__nodes_debug(
            f"🔄 {REFLECT_NODE_ID}: Decision: improve (iteration will be: {current_iteration})"
        )

    if not hasattr(result, "id") or not result.id:
        result.id = "reflect"

    new_messages = [summary, result] if last_message else [summary]

    return {
        "messages": new_messages,
        "reflection_decision": reflection_decision,
        "iteration": current_iteration,
    }


async def format_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that formats SQL results and PDF chunks into natural language answers.

    This node synthesizes data from multiple sources (SQL query results and PDF document chunks)
    into a coherent, bilingual answer. It uses Azure GPT-4o-mini to generate markdown-formatted
    responses that match the user's query language (Czech/English) and follow strict data-only rules.

    The node handles:
    - Multi-source data synthesis (SQL + PDF)
    - Source attribution and clear separation
    - Markdown formatting with tables and lists
    - Number formatting without separators
    - Language matching (Czech/English)
    - Preservation of PDF chunks for frontend display

    Args:
        state (DataAnalysisState): Workflow state containing queries_and_results, top_chunks, and prompt.

    Returns:
        DataAnalysisState: Updated state with final_answer, messages, and preserved top_chunks.

    Key Steps:
        1. Extract SQL results and PDF chunks from state
        2. Build separate context sections for SQL and PDF data
        3. Call Azure GPT-4o-mini with synthesis prompt
        4. Generate markdown-formatted answer
        5. Preserve PDF chunks for frontend
        6. Return final_answer and updated messages
    """
    print__nodes_debug(f"🎨 {FORMAT_ANSWER_ID}: Enter format_answer_node")

    queries_and_results = state.get("queries_and_results", [])
    top_chunks = state.get("top_chunks", [])
    rewritten_prompt = state.get("rewritten_prompt")
    prompt = state["prompt"]
    messages = state.get("messages", [])

    # Detect the language of the original prompt
    detected_language = await detect_language(prompt)
    print__nodes_debug(f"🌍 {FORMAT_ANSWER_ID}: Detected language: {detected_language}")

    # Add debug logging for PDF chunks
    print__nodes_debug(f"📄 {FORMAT_ANSWER_ID}: PDF chunks count: {len(top_chunks)}")
    if top_chunks:
        print__nodes_debug(
            f"📄 {FORMAT_ANSWER_ID}: First chunk preview: {top_chunks[0].page_content[:100] if hasattr(top_chunks[0], 'page_content') else str(top_chunks[0])[:100]}..."
        )

    llm = get_azure_llm_gpt_4o_mini(temperature=0.1)

    # Prepare SQL queries and results context
    queries_results_text = "\n\n".join(
        f"Query {i+1}:\n{query}\nResult {i+1}:\n{result}"
        for i, (query, result) in enumerate(queries_and_results)
    )

    # Prepare PDF chunks context separately
    pdf_chunks_text = ""
    if top_chunks:
        print__nodes_debug(
            f"📄 {FORMAT_ANSWER_ID}: Including {len(top_chunks)} PDF chunks in context"
        )
        chunks_content = []
        for i, chunk in enumerate(
            top_chunks[:10], 1
        ):  # Limit to top 10 chunks to prevent token overflow
            source = (
                chunk.metadata.get("source", "unknown") if chunk.metadata else "unknown"
            )
            content = (
                chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            )
            chunks_content.append(f"PDF Source {i} ({source}):\n{content}")
        pdf_chunks_text = "\n\n".join(chunks_content)
    else:
        print__nodes_debug(
            f"📄 {FORMAT_ANSWER_ID}: No PDF chunks available for context"
        )

    system_prompt = f"""
You are a bilingual (Czech/English) data analyst. Respond strictly using provided SQL results and PDF document context:

1. **Data Rules**:
   - Use ONLY provided data (SQL results and PDF document content)
   - Always read in details the QUERY and match it again with user question - if it does make sense.
   - Column 'value' can have different meaning, to understand what number means, you need to read carefully all values in each column on the same record.
   - Never format any numbers, use plain digits, no separators etc.
   - If PDF document context is provided, use it to enrich your answer with additional relevant information
   
2. **Response Rules**:
   - IMPORTANT: Your response MUST be in this language: {detected_language}
   - IMPORTANT: Translate ALL content to {detected_language}, including data labels, column names, table headers, and any Czech terms in the data
   - Synthesize all data (SQL + PDF) into one comprehensive answer
   - Provide as much as possible of the RELEVANT details from SQL and PDF data that are relevant to the user's prompt.
   - Compare values when relevant
   - Highlight patterns and trends if applicable
   - Note contradictions if found
   - Never hallucinate, always check the meaning of the data and if you are not sure about the answer or if the answer is not in the results, just say so.
   - Be careful not to say that something was 0 when you got no results from SQL.
   - Again read carefully the question, and provide answer using the QUERIES and its RESULTS, only if those answer the question. For example if question is about cinemas, dont answer about houses.
   - Be sure that your answer makes sense with regard of data and is gramatically coherent and meaningful.
   - When using PDF context, clearly indicate what information comes from PDF sources vs SQL data
   - Dont mention anything general if user asks for something specific, like dont mention general imports to whole country, if user asks about import from one country.
   
3. **Style Rules**:
   - No query/results references in final answer
   - No filler phrases
   - No unsupported commentary
   - Logical structure (e.g., highest-to-lowest)
   - Make output more structured, instead of making one long sentence.
   - If using both SQL and PDF data, organize the answer to show how they complement each other

4. **Output Format**:
   - Format as MARKDOWN!
    - Use bullet points for lists, format statistical data nicely as tables or in bullets
    - Use headings for sections
    - Use tables for structured data
    
Example regarding numeric output:
Good: "X is 1234567 while Y is 7654321"
Bad: "The query shows X is 1,234,567"


"""

    # Build the formatted prompt with separate sections for SQL and PDF data
    formatted_prompt_parts = ["Question: {question}"]

    # Add SQL data section if available
    if queries_and_results:
        formatted_prompt_parts.append("SQL Data Context:\n{sql_context}")

    # Add PDF data section if available
    if pdf_chunks_text:
        formatted_prompt_parts.append("PDF Document Context:\n{pdf_context}")

    # Add instruction
    if queries_and_results and pdf_chunks_text:
        instruction = "Please answer the question based on both the SQL queries/results and the PDF document context provided."
    elif queries_and_results:
        instruction = (
            "Please answer the question based on the SQL queries and results provided."
        )
    elif pdf_chunks_text:
        instruction = (
            "Please answer the question based on the PDF document context provided."
        )
    else:
        instruction = "No data context available to answer the question."

    language_stress = f"   - IMPORTANT: Translate ALL content to {detected_language}, including data labels, column names, table headers, and any Czech terms in the data"
    formatted_prompt_parts.append(instruction)
    formatted_prompt_parts.append(language_stress)
    formatted_prompt = "\n\n".join(formatted_prompt_parts)

    # Prepare template variables
    template_vars = {"question": rewritten_prompt or prompt}

    if queries_and_results:
        template_vars["sql_context"] = queries_results_text

    if pdf_chunks_text:
        template_vars["pdf_context"] = pdf_chunks_text

    chain = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", formatted_prompt)]
    )
    result = await llm.ainvoke(chain.format_messages(**template_vars))
    print__nodes_debug(f"✅ {FORMAT_ANSWER_ID}: Analysis completed")

    # Extract the final answer content
    final_answer_content = result.content if hasattr(result, "content") else str(result)
    # FIX: Escape curly braces in final_answer_content to prevent f-string parsing errors
    final_answer_preview = (
        final_answer_content[:100].replace("{", "{{").replace("}", "}}")
    )
    print__nodes_debug(
        f"📄 {FORMAT_ANSWER_ID}: Final answer: {final_answer_preview}..."
    )

    # Update messages state (existing logic)
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )
    if not hasattr(result, "id") or not result.id:
        result.id = "format_answer"

    # Add final debug logging
    print__nodes_debug(
        f"📄 {FORMAT_ANSWER_ID}: Preserving {len(top_chunks)} PDF chunks for frontend"
    )

    return {
        "messages": [summary, result],
        "final_answer": final_answer_content,
        "top_chunks": top_chunks,  # Preserve chunks for frontend instead of clearing them
    }


async def increment_iteration_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that increments the iteration counter for loop control.

    This simple utility node increments the iteration counter to track the number of query-reflect
    cycles executed. It's used in the improvement loop to enforce MAX_ITERATIONS limit and prevent
    infinite loops.

    Args:
        state (DataAnalysisState): Workflow state containing iteration count.

    Returns:
        DataAnalysisState: Updated state with incremented iteration counter.
    """
    print__nodes_debug(f"🔄 {INCREMENT_ITERATION_ID}: Enter increment_iteration_node")

    current_iteration = state.get("iteration", 0)
    return {"iteration": current_iteration + 1}


async def submit_final_answer_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that prepares and preserves final answer for user delivery.

    This node ensures the final_answer field is properly preserved along with all essential state
    fields needed for the frontend API response. It acts as a state validation checkpoint before
    saving and cleanup operations.

    The node explicitly preserves:
    - final_answer (primary response)
    - messages (conversation history)
    - queries_and_results (SQL execution history)
    - top_chunks (PDF document excerpts)
    - top_selection_codes (relevant datasets)
    - followup_prompts (suggestions for continued exploration)

    Args:
        state (DataAnalysisState): Workflow state containing final_answer and supporting data.

    Returns:
        DataAnalysisState: Validated state with all essential fields explicitly preserved.
    """
    print__nodes_debug(f"📤 {SUBMIT_FINAL_ID}: Enter submit_final_answer_node")

    # Ensure final_answer is properly preserved in the state
    final_answer = state.get("final_answer", "")

    print__nodes_debug(
        f"📤 {SUBMIT_FINAL_ID}: Final answer length: {len(final_answer)} characters"
    )
    # FIX: Escape curly braces in final_answer to prevent f-string parsing errors
    final_answer_preview = (
        (final_answer[:100] if final_answer else "")
        .replace("{", "{{")
        .replace("}", "}}")
    )
    print__nodes_debug(
        f"📤 {SUBMIT_FINAL_ID}: Final answer preview: {final_answer_preview}..."
    )

    # Return state with final_answer explicitly preserved
    return {
        "final_answer": final_answer,
        # Preserve other important state
        "messages": state.get("messages", []),
        "queries_and_results": state.get("queries_and_results", []),
        "top_chunks": state.get("top_chunks", []),
        "top_selection_codes": state.get("top_selection_codes", []),
        "followup_prompts": state.get(
            "followup_prompts", []
        ),  # CRITICAL: Preserve follow-up prompts
    }


async def save_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that saves results to files and creates minimal database checkpoint.

    This node performs dual persistence:
    1. File persistence: Appends results to analysis_results.txt (human-readable) and
       analysis_results.jsonl (machine-readable) for historical analysis
    2. Database checkpoint: Creates minimal state snapshot with only 7 essential fields
       to dramatically reduce PostgreSQL storage requirements

    The minimal checkpoint approach reduces state size from potentially megabytes (full state
    with embeddings, intermediate results) to kilobytes (essential fields only), solving
    memory and storage issues in production.

    Args:
        state (DataAnalysisState): Complete workflow state with all intermediate data.

    Returns:
        DataAnalysisState: Minimal checkpoint state with only essential fields for database storage.

    Key Steps:
        1. Extract final_answer, prompt, and queries_and_results
        2. Save to text file (if enabled)
        3. Append to JSONL file (if enabled)
        4. Create minimal checkpoint with 7 essential fields
        5. Return minimal state for database persistence
    """
    print__nodes_debug(f"💾 {SAVE_RESULT_ID}: Enter save_node")

    prompt = state["prompt"]
    queries_and_results = state.get("queries_and_results", [])

    # FIXED: Use final_answer directly from state instead of extracting from messages
    final_answer = state.get("final_answer", "")
    # FIX: Escape curly braces in final_answer to prevent f-string parsing errors
    final_answer_preview = (
        (final_answer[:100] if final_answer else "EMPTY")
        .replace("{", "{{")
        .replace("}", "}}")
    )
    print__nodes_debug(
        f"💾 {SAVE_RESULT_ID}: Final answer from state: '{final_answer_preview}'..."
    )

    result_path = BASE_DIR / "analysis_results.txt"
    result_obj = {
        "prompt": prompt,
        "result": final_answer,
        "queries_and_results": [
            {"query": q, "result": r} for q, r in queries_and_results
        ],
    }

    if SAVE_TO_FILE_TXT_JSONL:
        # Stream write to text file (no memory issues)
        with result_path.open("a", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Result: {final_answer}\n")
            f.write("Queries and Results:\n")
            for query, result in queries_and_results:
                f.write(f"  Query: {query}\n")
                f.write(f"  Result: {result}\n")
            f.write(
                "----------------------------------------------------------------------------\n"
            )

        # Append to a JSONL (JSON Lines) file for memory efficiency
        json_result_path = BASE_DIR / "analysis_results.jsonl"

        try:
            # Simply append one JSON object per line (no loading existing file)
            with json_result_path.open("a", encoding="utf-8") as f:

                f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            print__nodes_debug(
                f"✅ {SAVE_RESULT_ID}: ✅ Result saved to {result_path} and {json_result_path}"
            )
        except Exception as e:
            logger.error("❌ %s: ⚠️ Error saving JSON: %s", SAVE_RESULT_ID, e)
    else:
        print__nodes_debug(
            f"💾 {SAVE_RESULT_ID}: File saving disabled (SAVE_TO_FILE_TXT_JSONL = {SAVE_TO_FILE_TXT_JSONL})"
        )

    # MINIMAL CHECKPOINT STATE: Return only essential fields for checkpointing
    # This dramatically reduces database storage from full state to just these fields
    minimal_checkpoint_state = {
        "prompt": state.get("prompt", ""),
        "queries_and_results": state.get("queries_and_results", []),
        "most_similar_selections": state.get("most_similar_selections", []),
        "most_similar_chunks": state.get("most_similar_chunks", []),
        "final_answer": final_answer,  # Now correctly uses the final_answer from state
        # Keep messages for API compatibility but don't store large intermediate state
        "messages": state.get("messages", []),
        "followup_prompts": state.get("followup_prompts", []),
    }

    print__nodes_debug(
        f"💾 {SAVE_RESULT_ID}: Created minimal checkpoint with {len(minimal_checkpoint_state)} essential fields"
    )
    print__nodes_debug(
        f"💾 {SAVE_RESULT_ID}: Checkpoint fields: {list(minimal_checkpoint_state.keys())}"
    )
    # FIX: Escape curly braces in the final debug message as well
    final_answer_debug = (
        (final_answer[:100] if final_answer else "EMPTY")
        .replace("{", "{{")
        .replace("}", "}}")
    )
    print__nodes_debug(
        f"💾 {SAVE_RESULT_ID}: Final answer being stored: '{final_answer_debug}'..."
    )

    return minimal_checkpoint_state


async def retrieve_similar_selections_hybrid_search_node(
    state: DataAnalysisState,
) -> DataAnalysisState:
    """LangGraph node that performs hybrid search on ChromaDB to retrieve similar dataset selections.

    This node executes a hybrid search combining semantic similarity and BM25 keyword matching
    to find the most relevant dataset selections from the ChromaDB vector database. It supports
    both local and cloud ChromaDB configurations and includes memory cleanup to prevent resource leaks.

    The search query is extracted from the state (preferring 'rewritten_prompt' over 'prompt').
    Results are converted to Document objects for downstream processing in the LangGraph workflow.

    Args:
        state (DataAnalysisState): Workflow state containing query and configuration parameters.

    Returns:
        DataAnalysisState: Updated state with 'hybrid_search_results' containing a list of Document objects
                          representing the retrieved dataset selections. May also include 'chromadb_missing'
                          flag if local ChromaDB is unavailable.

    Key Steps:
        1. Extract query and n_results from state (default: SELECTIONS_HYBRID_SEARCH_DEFAULT_RESULTS)
        2. Check ChromaDB availability (local directory or cloud)
        3. Initialize ChromaDB client and collection
        4. Perform hybrid search using hybrid_search function
        5. Convert results to Document objects with metadata
        6. Clean up ChromaDB resources and force garbage collection
        7. Return results or empty list on error
    """
    print__nodes_debug(
        f"🔍 {HYBRID_SEARCH_NODE_ID}: Enter retrieve_similar_selections_hybrid_search_node"
    )

    query = state.get("rewritten_prompt") or state["prompt"]
    n_results = state.get("n_results", SELECTIONS_HYBRID_SEARCH_DEFAULT_RESULTS)

    print__nodes_debug(f"🔍 {HYBRID_SEARCH_NODE_ID}: Query: {query}")
    print__nodes_debug(f"🔍 {HYBRID_SEARCH_NODE_ID}: Requested n_results: {n_results}")

    # Check if ChromaDB directory exists (only when using local ChromaDB)

    use_cloud = should_use_cloud()

    if not use_cloud:
        # Only check for local directory when not using cloud
        chroma_db_dir = BASE_DIR / "metadata" / "czsu_chromadb"
        print__nodes_debug(
            f"🔍 {HYBRID_SEARCH_NODE_ID}: Checking local ChromaDB at: {chroma_db_dir}"
        )
        print__nodes_debug(
            f"🔍 {HYBRID_SEARCH_NODE_ID}: ChromaDB exists: {chroma_db_dir.exists()}"
        )
        print__nodes_debug(
            f"🔍 {HYBRID_SEARCH_NODE_ID}: ChromaDB is_dir: {chroma_db_dir.is_dir() if chroma_db_dir.exists() else 'N/A'}"
        )

        if not chroma_db_dir.exists() or not chroma_db_dir.is_dir():
            print__nodes_debug(
                f"📄 {HYBRID_SEARCH_NODE_ID}: ChromaDB directory not found at {chroma_db_dir}"
            )
            return {"hybrid_search_results": [], "chromadb_missing": True}

        print__nodes_debug(
            f"🔍 {HYBRID_SEARCH_NODE_ID}: Local ChromaDB found! Resetting chromadb_missing flag"
        )
    else:
        print__nodes_debug(
            f"🌐 {HYBRID_SEARCH_NODE_ID}: Using Chroma Cloud (skipping local directory check)"
        )

    try:
        # Use the same method as the test script to get ChromaDB collection directly

        client = get_chromadb_client(
            local_path=CHROMA_DB_PATH, collection_name=CHROMA_COLLECTION_NAME
        )
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print__nodes_debug(
            f"📊 {HYBRID_SEARCH_NODE_ID}: ChromaDB collection initialized"
        )

        hybrid_results = hybrid_search(collection, query, n_results=n_results)
        print__nodes_debug(
            f"📊 {HYBRID_SEARCH_NODE_ID}: Retrieved {len(hybrid_results)} hybrid search results"
        )

        # Convert dict results to Document objects for compatibility

        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(page_content=result["document"], metadata=result["metadata"])
            hybrid_docs.append(doc)

        # Debug: Show detailed hybrid search results
        print__nodes_debug(
            f"📄 {HYBRID_SEARCH_NODE_ID}: Detailed hybrid search results:"
        )
        for i, doc in enumerate(hybrid_docs[:10], 1):  # Show first 10
            selection = doc.metadata.get("selection") if doc.metadata else "N/A"
            content_preview = (
                doc.page_content[:100].replace("\n", " ")
                if hasattr(doc, "page_content")
                else "N/A"
            )
            print__nodes_debug(
                f"📄 {HYBRID_SEARCH_NODE_ID}: #{i}: {selection} | Content: {content_preview}..."
            )

        print__nodes_debug(
            f"📄 {HYBRID_SEARCH_NODE_ID}: All selection codes: {[doc.metadata.get('selection') for doc in hybrid_docs]}"
        )

        # MEMORY CLEANUP: Explicitly close ChromaDB resources
        print__nodes_debug(
            f"🧹 {HYBRID_SEARCH_NODE_ID}: Cleaning up ChromaDB client resources"
        )
        collection = None  # Clear collection reference
        del client  # Explicitly delete client
        gc.collect()  # Force garbage collection to release memory
        print__nodes_debug(f"✅ {HYBRID_SEARCH_NODE_ID}: ChromaDB resources released")

        return {"hybrid_search_results": hybrid_docs}
    except Exception as e:
        logger.error("❌ %s: Error in hybrid search: %s", HYBRID_SEARCH_NODE_ID, e)

        logger.error(
            "📄 %s: Traceback: %s", HYBRID_SEARCH_NODE_ID, traceback.format_exc()
        )
        return {"hybrid_search_results": []}


async def rerank_table_descriptions_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that reranks dataset selection hybrid search results using Cohere.

    This node applies Cohere's multilingual rerank model to improve the quality of dataset selection
    retrieval. It takes hybrid search results (semantic + BM25) and reorders them based on semantic
    relevance to the query, returning selection codes with Cohere relevance scores.

    Cohere reranking significantly improves precision by understanding query semantics beyond simple
    keyword matching, especially important for Czech language queries and domain-specific terminology.

    Args:
        state (DataAnalysisState): Workflow state containing query and hybrid_search_results.

    Returns:
        DataAnalysisState: Updated state with most_similar_selections as list of (selection_code, score) tuples.

    Key Steps:
        1. Extract query and hybrid search results
        2. Call cohere_rerank with documents and query
        3. Extract selection codes and relevance scores
        4. Return ranked list of (code, score) pairs
        5. Log top results for debugging
    """

    print__nodes_debug(
        f"🔥🔥🔥 🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: ===== RERANK NODE EXECUTING ===== 🔥🔥🔥"
    )
    print__nodes_debug(
        f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Enter rerank_table_descriptions_node"
    )

    query = state.get("rewritten_prompt") or state["prompt"]
    hybrid_results = state.get("hybrid_search_results", [])
    n_results = state.get("n_results", 20)

    print__nodes_debug(f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Query: {query}")
    print__nodes_debug(
        f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Number of hybrid results received: {len(hybrid_results)}"
    )
    print__nodes_debug(
        f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Requested n_results: {n_results}"
    )

    # Check if we have hybrid search results to rerank
    if not hybrid_results:
        print__nodes_debug(
            f"📄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: No hybrid search results to rerank"
        )
        return {"most_similar_selections": []}

    # Debug: Show input to rerank
    print__nodes_debug(
        f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Input hybrid results for reranking:"
    )
    for i, doc in enumerate(hybrid_results[:10], 1):  # Show first 10
        selection = doc.metadata.get("selection") if doc.metadata else "N/A"
        content_preview = (
            doc.page_content[:100].replace("\n", " ")
            if hasattr(doc, "page_content")
            else "N/A"
        )
        print__nodes_debug(
            f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: #{i}: {selection} | Content: {content_preview}..."
        )

    try:
        print__nodes_debug(
            f"🔄 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Calling cohere_rerank with {len(hybrid_results)} documents"
        )
        reranked = cohere_rerank(query, hybrid_results, top_n=n_results)
        print__nodes_debug(
            f"📊 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Cohere returned {len(reranked)} reranked results"
        )

        most_similar = []
        for i, (doc, res) in enumerate(reranked, 1):
            selection_code = doc.metadata.get("selection") if doc.metadata else None
            score = res.relevance_score
            most_similar.append((selection_code, score))
            # Debug: Show detailed rerank results
            if i <= 10:  # Show top 10 results
                print__nodes_debug(
                    f"🎯🎯🎯 🎯 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: Rerank #{i}: {selection_code} | Score: {score:.6f}"
                )

        print__nodes_debug(
            f"🎯🎯🎯 🎯🎯🎯 {RERANK_TABLE_DESCRIPTIONS_NODE_ID}: FINAL RERANK OUTPUT: {most_similar[:5]} 🎯🎯🎯"
        )

        return {"most_similar_selections": most_similar}
    except Exception as e:
        logger.error(
            "❌ %s: Error in reranking: %s", RERANK_TABLE_DESCRIPTIONS_NODE_ID, e
        )

        logger.error(
            "📄 %s: Traceback: %s",
            RERANK_TABLE_DESCRIPTIONS_NODE_ID,
            traceback.format_exc(),
        )
        return {"most_similar_selections": []}


async def relevant_selections_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that filters reranked selections by relevance threshold and selects top 3.

    This node implements quality control by filtering dataset selections based on Cohere relevance
    scores. Only selections exceeding SQL_RELEVANCE_THRESHOLD (0.0005) are retained, with a maximum
    of 3 selections to prevent information overload and maintain schema size for SQL generation.

    If no selections pass the threshold, the node sets a special final_answer indicating no relevant
    data was found, allowing the workflow to skip SQL generation and proceed directly to answer formatting.

    Args:
        state (DataAnalysisState): Workflow state containing most_similar_selections with scores.

    Returns:
        DataAnalysisState: Updated state with top_selection_codes (max 3), cleared intermediate results.

    Key Steps:
        1. Apply SQL_RELEVANCE_THRESHOLD (0.0005) filter
        2. Select up to 3 top selections
        3. Clear intermediate hybrid search state
        4. Set special final_answer if no selections pass
        5. Return filtered selection codes
    """
    print__nodes_debug(f"🎯 {RELEVANT_NODE_ID}: Enter relevant_selections_node")

    most_similar = state.get("most_similar_selections", [])

    # Select up to 3 top selections above threshold
    top_selection_codes = [
        sel
        for sel, score in most_similar
        if sel is not None and score is not None and score >= SQL_RELEVANCE_THRESHOLD
    ][:3]
    print__nodes_debug(
        f"🎯 {RELEVANT_NODE_ID}: top_selection_codes: {top_selection_codes}"
    )

    result = {
        "top_selection_codes": top_selection_codes,
        "hybrid_search_results": [],
        "most_similar_selections": [],
    }

    # If no selections pass the threshold, set final_answer for frontend
    if not top_selection_codes:
        print__nodes_debug(
            f"📄 {RELEVANT_NODE_ID}: No selections passed the threshold - setting final_answer"
        )
        result["final_answer"] = "No Relevant Selections Found"

    return result


async def summarize_messages_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that maintains bounded conversation memory through cumulative summarization.

    This node implements memory management by compressing conversation history into a concise summary.
    It uses Azure GPT-4o-mini to update an existing summary with the latest message, maintaining
    bounded context size while preserving important conversation details.

    The summarization strategy prevents token overflow in long conversations while ensuring that
    context-dependent queries (pronouns, references) can still be resolved in query rewriting.

    The node always maintains a 2-message structure: [summary (SystemMessage), last_message],
    which is the canonical message format throughout the workflow.

    Args:
        state (DataAnalysisState): Workflow state containing messages list.

    Returns:
        DataAnalysisState: Updated state with messages as [new_summary, last_message].

    Key Steps:
        1. Extract previous summary and last message
        2. Skip if both are empty
        3. Call Azure GPT-4o-mini to generate updated summary
        4. Create new SystemMessage with summary
        5. Return 2-message structure [summary, last_message]
    """
    print__nodes_debug("📝 SUMMARY: Enter summarize_messages_node")

    messages = state.get("messages", [])
    summary = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else SystemMessage(content="")
    )
    last_message = messages[1] if len(messages) > 1 else None
    prev_summary = summary.content
    last_message_content = last_message.content if last_message else ""

    print__nodes_debug(f"📝 SUMMARY: prev_summary: '{prev_summary}'")
    print__nodes_debug(f"📝 SUMMARY: last_message_content: '{last_message_content}'")

    if not prev_summary and not last_message_content:
        print__nodes_debug(
            "📝 SUMMARY: Skipping summarization (no previous summary or last message)."
        )
        return {"messages": [summary] if not last_message else [summary, last_message]}

    llm = get_azure_llm_gpt_4o_mini(temperature=0.0)

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

    human_prompt = "Previous summary:\n{prev_summary}\n\nLatest message:\n{last_message_content}\n\nUpdate the summary to include the latest message."
    print__nodes_debug(f"📝 SUMMARY: human_prompt template: {human_prompt}")

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)]
    )
    result = await llm.ainvoke(
        prompt.format_messages(
            prev_summary=prev_summary, last_message_content=last_message_content
        )
    )
    new_summary = result.content.strip()
    print__nodes_debug(f"📝 SUMMARY: Updated summary: {new_summary}")

    summary_msg = SystemMessage(content=new_summary)
    new_messages = [summary_msg, last_message] if last_message else [summary_msg]

    print__nodes_debug(
        f"📝 SUMMARY: New messages: {[getattr(m, 'content', None) for m in new_messages]}"
    )

    return {"messages": new_messages}


# ==============================================================================
# PDF CHUNK NODES
# ==============================================================================
async def retrieve_similar_chunks_hybrid_search_node(
    state: DataAnalysisState,
) -> DataAnalysisState:
    """LangGraph node that performs hybrid search on PDF ChromaDB to retrieve similar document chunks.

    This node executes a hybrid search combining semantic similarity and BM25 keyword matching
    to find the most relevant PDF document chunks from the ChromaDB vector database. It first
    translates the query to English for better search performance, then supports both local and
    cloud ChromaDB configurations with memory cleanup to prevent resource leaks.

    The search query is extracted from the state (preferring 'rewritten_prompt' over 'prompt'),
    translated to English, and used to retrieve relevant PDF chunks. Results are converted to
    Document objects for downstream processing in the LangGraph workflow.

    Args:
        state (DataAnalysisState): Workflow state containing query and configuration parameters.

    Returns:
        DataAnalysisState: Updated state with 'hybrid_search_chunks' containing a list of Document objects
                          representing the retrieved PDF chunks. Returns empty list if PDF functionality
                          is unavailable or ChromaDB is not accessible.

    Key Steps:
        1. Check if PDF functionality is available
        2. Extract and translate query to English using Azure Translator
        3. Extract n_results from state (default: PDF_HYBRID_SEARCH_DEFAULT_RESULTS)
        4. Check PDF ChromaDB availability (local directory or cloud)
        5. Initialize PDF ChromaDB client and collection
        6. Perform hybrid search using pdf_hybrid_search function
        7. Convert results to Document objects with metadata
        8. Clean up ChromaDB resources and force garbage collection
        9. Return results or empty list on error
    """
    print__nodes_debug(
        f"🔍 {RETRIEVE_CHUNKS_NODE_ID}: Enter retrieve_similar_chunks_hybrid_search_node"
    )

    if not PDF_FUNCTIONALITY_AVAILABLE:
        print__nodes_debug(
            f"📄 {RETRIEVE_CHUNKS_NODE_ID}: PDF functionality not available"
        )
        return {"hybrid_search_chunks": []}

    query_original_language = state.get("rewritten_prompt") or state["prompt"]

    # Translate query to English using Azure Translator
    query = await translate_to_english(query_original_language)

    print__nodes_debug(
        f"🔄 {RETRIEVE_CHUNKS_NODE_ID}: Original query: '{query_original_language}' -> Translated query: '{query}'"
    )

    n_results = state.get("n_results", PDF_HYBRID_SEARCH_DEFAULT_RESULTS)

    print__nodes_debug(f"🔄 {RETRIEVE_CHUNKS_NODE_ID}: Query: {query}")
    print__nodes_debug(
        f"🔄 {RETRIEVE_CHUNKS_NODE_ID}: Requested n_results: {n_results}"
    )

    # Check if PDF ChromaDB directory exists (only when using local ChromaDB)

    use_cloud = should_use_cloud()

    if not use_cloud:
        # Only check for local directory when not using cloud
        if not PDF_CHROMA_DB_PATH.exists() or not PDF_CHROMA_DB_PATH.is_dir():
            print__nodes_debug(
                f"📄 {RETRIEVE_CHUNKS_NODE_ID}: PDF ChromaDB directory not found at {PDF_CHROMA_DB_PATH}"
            )
            return {"hybrid_search_chunks": []}
        print__nodes_debug(
            f"🔍 {RETRIEVE_CHUNKS_NODE_ID}: Local PDF ChromaDB found at {PDF_CHROMA_DB_PATH}"
        )
    else:
        print__nodes_debug(
            f"🌐 {RETRIEVE_CHUNKS_NODE_ID}: Using Chroma Cloud for PDF chunks"
        )

    try:
        # Use the PDF ChromaDB collection directly with cloud/local support

        client = get_chromadb_client(
            local_path=PDF_CHROMA_DB_PATH, collection_name=PDF_COLLECTION_NAME
        )
        collection = client.get_collection(name=PDF_COLLECTION_NAME)
        print__nodes_debug(
            f"📊 {RETRIEVE_CHUNKS_NODE_ID}: PDF ChromaDB collection initialized"
        )

        hybrid_results = pdf_hybrid_search(collection, query, n_results=n_results)
        print__nodes_debug(
            f"📊 {RETRIEVE_CHUNKS_NODE_ID}: Retrieved {len(hybrid_results)} PDF hybrid search results"
        )

        # Convert dict results to Document objects for compatibility

        hybrid_docs = []
        for result in hybrid_results:
            doc = Document(page_content=result["document"], metadata=result["metadata"])
            hybrid_docs.append(doc)

        # Debug: Show detailed hybrid search results
        print__nodes_debug(
            f"📄 {RETRIEVE_CHUNKS_NODE_ID}: Detailed PDF hybrid search results:"
        )
        for i, doc in enumerate(
            hybrid_docs[:PDF_N_TOP_CHUNKS], 1
        ):  # Show first few results
            source = doc.metadata.get("source") if doc.metadata else "N/A"
            content_preview = (
                doc.page_content[:100].replace("\n", " ")
                if hasattr(doc, "page_content")
                else "N/A"
            )
            print__nodes_debug(
                f"📄 {RETRIEVE_CHUNKS_NODE_ID}: #{i}: {source} | Content: {content_preview}..."
            )

        # MEMORY CLEANUP: Explicitly close ChromaDB resources
        print__nodes_debug(
            f"🧹 {RETRIEVE_CHUNKS_NODE_ID}: Cleaning up PDF ChromaDB client resources"
        )
        collection = None  # Clear collection reference
        del client  # Explicitly delete client
        gc.collect()  # Force garbage collection to release memory
        print__nodes_debug(
            f"✅ {RETRIEVE_CHUNKS_NODE_ID}: PDF ChromaDB resources released"
        )

        return {"hybrid_search_chunks": hybrid_docs}
    except Exception as e:
        logger.error(
            "❌ %s: Error in PDF hybrid search: %s", RETRIEVE_CHUNKS_NODE_ID, e
        )

        logger.error(
            "📄 %s: Traceback: %s", RETRIEVE_CHUNKS_NODE_ID, traceback.format_exc()
        )
        return {"hybrid_search_chunks": []}


async def rerank_chunks_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that reranks PDF chunk hybrid search results using Cohere.

    This node applies Cohere's multilingual rerank model to improve the quality of PDF chunk
    retrieval. It takes hybrid search results (semantic + BM25) from PDF documents and reorders
    them based on semantic relevance to the query, returning document objects with Cohere scores.

    PDF chunks are typically English text from documentation, so the reranking helps bridge the
    language gap when queries are in Czech by understanding cross-lingual semantic similarity.

    Args:
        state (DataAnalysisState): Workflow state containing query and hybrid_search_chunks.

    Returns:
        DataAnalysisState: Updated state with most_similar_chunks as list of (Document, score) tuples.

    Key Steps:
        1. Check PDF functionality availability
        2. Extract query and hybrid search chunks
        3. Call pdf_cohere_rerank with documents and query
        4. Extract documents and relevance scores
        5. Return ranked list of (document, score) pairs
        6. Log top results for debugging
    """
    print__nodes_debug(f"🔄 {RERANK_CHUNKS_NODE_ID}: Enter rerank_chunks_node")

    if not PDF_FUNCTIONALITY_AVAILABLE:
        print__nodes_debug(
            f"📄 {RERANK_CHUNKS_NODE_ID}: PDF functionality not available"
        )
        return {"most_similar_chunks": []}

    query = state.get("rewritten_prompt") or state["prompt"]
    hybrid_results = state.get("hybrid_search_chunks", [])
    n_results = state.get("n_results", PDF_N_TOP_CHUNKS)

    print__nodes_debug(f"🔄 {RERANK_CHUNKS_NODE_ID}: Query: {query}")
    print__nodes_debug(
        f"🔄 {RERANK_CHUNKS_NODE_ID}: Number of PDF hybrid results received: {len(hybrid_results)}"
    )
    print__nodes_debug(f"🔄 {RERANK_CHUNKS_NODE_ID}: Requested n_results: {n_results}")

    # Check if we have hybrid search results to rerank
    if not hybrid_results:
        print__nodes_debug(
            f"📄 {RERANK_CHUNKS_NODE_ID}: No PDF hybrid search results to rerank"
        )
        return {"most_similar_chunks": []}

    # Debug: Show input to rerank
    print__nodes_debug(
        f"🔄 {RERANK_CHUNKS_NODE_ID}: Input PDF hybrid results for reranking:"
    )
    for i, doc in enumerate(
        hybrid_results[:PDF_N_TOP_CHUNKS], 1
    ):  # Show first few results
        source = doc.metadata.get("source") if doc.metadata else "N/A"
        content_preview = (
            doc.page_content[:100].replace("\n", " ")
            if hasattr(doc, "page_content")
            else "N/A"
        )
        print__nodes_debug(
            f"🔄 {RERANK_CHUNKS_NODE_ID}: #{i}: {source} | Content: {content_preview}..."
        )

    try:
        print__nodes_debug(
            f"🔄 {RERANK_CHUNKS_NODE_ID}: Calling PDF cohere_rerank with {len(hybrid_results)} documents"
        )
        reranked = pdf_cohere_rerank(query, hybrid_results, top_n=n_results)
        print__nodes_debug(
            f"📄 {RERANK_CHUNKS_NODE_ID}: PDF Cohere returned {len(reranked)} reranked results"
        )

        most_similar = []
        for i, (doc, res) in enumerate(reranked, 1):
            score = res.relevance_score
            most_similar.append((doc, score))
            # Debug: Show detailed rerank results
            if i <= PDF_N_TOP_CHUNKS:  # Show top few results
                source = doc.metadata.get("source") if doc.metadata else "unknown"
                print__nodes_debug(
                    f"🎯🎯🎯 🎯 {RERANK_CHUNKS_NODE_ID}: PDF Rerank #{i}: {source} | Score: {score:.6f}"
                )

        print__nodes_debug(
            f"🎯🎯🎯 🎯🎯🎯 {RERANK_CHUNKS_NODE_ID}: FINAL PDF RERANK OUTPUT: {len(most_similar)} chunks 🎯🎯🎯"
        )

        return {"most_similar_chunks": most_similar}
    except Exception as e:
        logger.error("❌ %s: Error in PDF reranking: %s", RERANK_CHUNKS_NODE_ID, e)

        logger.error(
            "📄 %s: Traceback: %s", RERANK_CHUNKS_NODE_ID, traceback.format_exc()
        )
        return {"most_similar_chunks": []}


async def relevant_chunks_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that filters reranked PDF chunks by relevance threshold.

    This node implements quality control for PDF document retrieval by filtering chunks based on
    Cohere relevance scores. Only chunks exceeding PDF_RELEVANCE_THRESHOLD are retained for
    inclusion in the final answer synthesis.

    Unlike dataset selections (which have a max of 3), PDF chunks don't have an explicit limit
    beyond the threshold filter, though upstream configuration (PDF_N_TOP_CHUNKS) controls the
    maximum number of chunks retrieved.

    Args:
        state (DataAnalysisState): Workflow state containing most_similar_chunks with scores.

    Returns:
        DataAnalysisState: Updated state with top_chunks (filtered Documents), cleared intermediate results.

    Key Steps:
        1. Apply PDF_RELEVANCE_THRESHOLD filter
        2. Extract Document objects from (doc, score) tuples
        3. Clear intermediate hybrid search state
        4. Log chunks that passed threshold
        5. Return filtered PDF chunks
    """
    print__nodes_debug(f"🎯 {RELEVANT_CHUNKS_NODE_ID}: Enter relevant_chunks_node")
    SIMILARITY_THRESHOLD = PDF_RELEVANCE_THRESHOLD  # Threshold for PDF chunk relevance

    most_similar = state.get("most_similar_chunks", [])

    # Select chunks above threshold
    top_chunks = [
        doc
        for doc, score in most_similar
        if score is not None and score >= SIMILARITY_THRESHOLD
    ]
    print__nodes_debug(
        f"📄 {RELEVANT_CHUNKS_NODE_ID}: top_chunks: {len(top_chunks)} chunks passed threshold {SIMILARITY_THRESHOLD}"
    )

    # Debug: Show what passed
    for i, chunk in enumerate(top_chunks[:PDF_N_TOP_CHUNKS], 1):
        source = chunk.metadata.get("source") if chunk.metadata else "unknown"
        content_preview = (
            chunk.page_content[:100].replace("\n", " ")
            if hasattr(chunk, "page_content")
            else "N/A"
        )
        print__nodes_debug(
            f"📄 {RELEVANT_CHUNKS_NODE_ID}: Chunk #{i}: {source} | Content: {content_preview}..."
        )

    return {
        "top_chunks": top_chunks,
        "hybrid_search_chunks": [],
        "most_similar_chunks": [],
    }


async def post_retrieval_sync_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph synchronization node that waits for parallel retrieval branches to complete.

    This node serves as a barrier/join point in the LangGraph workflow, ensuring both parallel
    retrieval paths (database selections and PDF chunks) have completed before proceeding to
    SQL generation and answer formatting.

    The node is a pass-through that returns state unchanged but provides a synchronization point
    for the graph execution engine to coordinate parallel branches.

    Args:
        state (DataAnalysisState): Workflow state after parallel retrieval completion.

    Returns:
        DataAnalysisState: Unchanged state (pass-through for synchronization).
    """

    print__analysis_tracing_debug(
        "90 - SYNC NODE: Both selection and chunk branches completed"
    )
    return state  # Pass through state unchanged


async def cleanup_resources_node(state: DataAnalysisState) -> DataAnalysisState:
    """LangGraph node that performs aggressive memory cleanup and garbage collection.

    This node runs at the very end of the graph execution to force garbage collection and release
    memory from ChromaDB clients, embeddings, and large intermediate data structures. It solves
    memory leak issues in production deployments by explicitly clearing references and running
    multiple GC passes to handle circular references.

    The node returns a minimal state copy containing only essential fields needed for the API
    response, discarding all intermediate processing artifacts.

    Args:
        state (DataAnalysisState): Complete workflow state with all processing artifacts.

    Returns:
        DataAnalysisState: Minimal state copy with only essential fields for API response.

    Key Steps:
        1. Create state copy with only essential response fields
        2. Run first garbage collection pass
        3. Run second GC pass to catch circular references
        4. Log collection statistics
        5. Return minimal state for API delivery
    """

    CLEANUP_NODE_ID = 99
    print__nodes_debug(f"🧹 {CLEANUP_NODE_ID}: Enter cleanup_resources_node")

    # Clear large intermediate data structures that are no longer needed
    state_copy = {
        "prompt": state.get("prompt", ""),
        "final_answer": state.get("final_answer", ""),
        "queries_and_results": state.get("queries_and_results", []),
        "messages": state.get("messages", []),
        "top_chunks": state.get("top_chunks", []),
        "top_selection_codes": state.get("top_selection_codes", []),
        "followup_prompts": state.get("followup_prompts", []),
    }

    # Force garbage collection multiple times to ensure cleanup
    print__nodes_debug(f"🧹 {CLEANUP_NODE_ID}: Running aggressive garbage collection")
    collected = gc.collect()
    print__nodes_debug(
        f"✅ {CLEANUP_NODE_ID}: First GC pass collected {collected} objects"
    )

    # Run GC again to catch circular references
    collected = gc.collect()
    print__nodes_debug(
        f"✅ {CLEANUP_NODE_ID}: Second GC pass collected {collected} objects"
    )

    print__nodes_debug(
        f"✅ {CLEANUP_NODE_ID}: Cleanup complete, memory should be released"
    )

    return state_copy

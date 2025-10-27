"""Main entry point for the data analysis application.

This module provides the main execution logic for the data analysis
application, handling command line arguments and invoking the LangGraph workflow.

The system is designed to support both interactive use (as a library) and
command-line execution with configurable prompts, making it versatile for
different deployment scenarios.
"""

import asyncio

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys

# Configure asyncio event loop policy for Windows compatibility with psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import gc
import os
import re
import uuid
from pathlib import Path
from typing import List

import psutil
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

from my_agent import create_graph
from my_agent.utils.nodes import MAX_ITERATIONS
from my_agent.utils.models import get_azure_llm_gpt_4o_mini
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
    retry_on_ssl_connection_error,
)
from checkpointer.checkpointer.factory import get_global_checkpointer
from my_agent.utils.state import DataAnalysisState

# Robust BASE_DIR logic for project root
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import debug functions from utils
from api.utils.debug import (
    print__analysis_tracing_debug,
    print__main_debug,
    print__memory_debug,
)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Default prompt if none provided
# DEFAULT_PROMPT = "Did Prague have more residents than Central Bohemia at the start of 2024?"
# DEFAULT_PROMPT = "Can you compare number of man and number of woman in prague and in plzen? Create me a bar chart with this data."
# DEFAULT_PROMPT = "How much did Prague's population grow from start to end of Q3?"
# DEFAULT_PROMPT = "What was South Bohemia's population change rate per month?"
# DEFAULT_PROMPT = "Tell me a joke"
# DEFAULT_PROMPT = "Is there some very interesting trend in my data?"
# DEFAULT_PROMPT = "tell me about how many people were in prague at 2024 and compare it with whole republic data? Pak mi dej distribuci kazdeho regionu, v procentech."
# DEFAULT_PROMPT = "tell me about people in prague, compare, contrast, what is interesting, provide trends."
# DEFAULT_PROMPT = "What was the maximum female population recorded in any region?"
# DEFAULT_PROMPT = "List regions where the absolute difference between male and female population changes was greater than 3000, and indicate whether men or women changed more"
# DEFAULT_PROMPT = "What is the average population rate of change for regions with more than 1 million residents?"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy v Praze"
# DEFAULT_PROMPT = "Jaky obor ma nejvyssi prumerne mzdy?"
# DEFAULT_PROMPT = """
# This table contains information about wages and salaries across different industries. It includes data on average wages categorized by economic sectors or industries.

# Available columns:

# industry (odvětví): distinct values include manufacturing, IT, construction, healthcare, education, etc.

# average_wage (průměrná mzda): numerical values representing monthly or annual averages

# year: distinct values may include 2020, 2021, 2022, etc.

# measurement_unit: e.g., CZK, EUR, USD per month/year

# The table allows comparison of wage levels across different economic sectors.
# """
DEFAULT_PROMPT = "Jaká byla výroba kapalných paliv z ropy v Česku v roce 2023?"
# DEFAULT_PROMPT = "Jaký byl podíl osob používajících internet v Česku ve věku 16 a vice v roce 2023?"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def extract_table_names_from_sql(sql_query: str) -> List[str]:
    """Extract table names from SQL query FROM clauses.

    Args:
        sql_query: The SQL query string

    Returns:
        List of table names found in FROM clauses
    """
    # Remove comments and normalize whitespace
    sql_clean = re.sub(r"--.*?(?=\n|$)", "", sql_query, flags=re.MULTILINE)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)
    sql_clean = " ".join(sql_clean.split())

    # Pattern to match FROM clause with table names
    # This handles: FROM table_name, FROM schema.table_name, FROM "table_name", etc.
    from_pattern = r'\bFROM\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1(?:\s*,\s*(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\3)*'

    table_names = []
    matches = re.finditer(from_pattern, sql_clean, re.IGNORECASE)

    for match in matches:
        # Extract the main table name (group 2)
        if match.group(2):
            table_names.append(match.group(2).upper())
        # Extract additional table names if comma-separated (group 4)
        if match.group(4):
            table_names.append(match.group(4).upper())

    # Also handle JOIN clauses
    join_pattern = r'\bJOIN\s+(["\']?)([a-zA-Z_][a-zA-Z0-9_]*)\1'
    join_matches = re.finditer(join_pattern, sql_clean, re.IGNORECASE)

    for match in join_matches:
        if match.group(2):
            table_names.append(match.group(2).upper())

    return list(set(table_names))  # Remove duplicates


def get_used_selection_codes(
    queries_and_results: list, top_selection_codes: List[str]
) -> List[str]:
    """Filter top_selection_codes to only include those actually used in queries.

    Args:
        queries_and_results: List of (query, result) tuples
        top_selection_codes: List of all candidate selection codes

    Returns:
        List of selection codes that were actually used in the queries
    """
    if not queries_and_results or not top_selection_codes:
        return []

    # Extract all table names used in queries
    used_table_names = set()
    for query, _ in queries_and_results:
        if query:
            table_names = extract_table_names_from_sql(query)
            used_table_names.update(table_names)

    # Filter selection codes to only include those that match used table names
    used_selection_codes = []
    for selection_code in top_selection_codes:
        if selection_code.upper() in used_table_names:
            used_selection_codes.append(selection_code)

    return used_selection_codes


def generate_initial_followup_prompts() -> List[str]:
    """Generate initial follow-up prompt suggestions for new conversations using dynamic templates.

    This function generates diverse starter suggestions using pre-defined templates
    filled with random selections to ensure variety. These prompts will be displayed
    to users when they start a new chat, giving them ideas for questions they can ask
    about Czech Statistical Office data.

    Returns:
        List[str]: A list of dynamically generated suggested follow-up prompts for the user
    """
    print__main_debug(
        "🎯 PROMPT GEN: Starting dynamic template-based prompt generation"
    )

    # Generate dynamic prompts based on current timestamp to ensure variety
    import random
    import time

    # Use timestamp as seed for pseudo-randomness
    seed = int(time.time() * 1000) % 1000000
    random.seed(seed)

    # Pool of diverse prompt templates
    prompt_templates = [
        "What are the population trends in {region}?",
        "Show me employment statistics by {category}.",
        "Compare {metric} growth across different years.",
        "What are the latest statistics on {topic}?",
        "How has {indicator} changed in recent {period}?",
        "What are the {type} rates in {location}?",
        "Show me data about {subject} from {source}.",
        "What trends can you see in {area} statistics?",
        "Compare {metric} between {group1} and {group2}.",
        "What are the current {indicator} figures for {region}?",
        "Tell me about {topic} in {location}.",
        "Show me {subject} statistics for {period}.",
        "What is the {indicator} situation in {region}?",
        "Compare {metric} across {group1} and {group2}.",
        "What are the trends in {area} data?",
    ]

    # Fill in the templates with random selections
    regions = [
        "Prague",
        "Czech Republic",
        "major cities",
        "different regions",
        "Brno",
    ]
    categories = [
        "region",
        "industry",
        "age group",
        "education level",
        "sector",
    ]
    metrics = [
        "GDP",
        "employment",
        "population",
        "export",
        "import",
        "wage",
    ]
    topics = [
        "crime rates",
        "healthcare spending",
        "education levels",
        "housing prices",
        "migration",
        "birth rates",
    ]
    periods = [
        "years",
        "quarters",
        "months",
        "decades",
        "recent years",
    ]
    types = [
        "unemployment",
        "inflation",
        "birth",
        "migration",
        "divorce",
    ]
    locations = [
        "Prague",
        "Brno",
        "Czech Republic",
        "major regions",
    ]
    subjects = [
        "agricultural production",
        "industrial output",
        "tourism numbers",
        "energy consumption",
        "trade balance",
    ]
    sources = [
        "government reports",
        "statistical surveys",
        "economic indicators",
        "census data",
        "official statistics",
    ]
    areas = [
        "labor market",
        "demographic",
        "economic",
        "environmental",
        "social",
        "health",
    ]
    indicators = [
        "unemployment",
        "inflation",
        "GDP growth",
        "population",
        "wage growth",
        "export growth",
    ]
    group1_group2 = [
        ("urban and rural areas", "rural areas"),
        ("men and women", "women"),
        ("young and old", "older population"),
        ("public and private sector", "private companies"),
        ("domestic and foreign", "foreign companies"),
        ("large and small enterprises", "small businesses"),
    ]

    # Generate 5 unique prompts
    generated_prompts = []
    used_templates = set()

    while len(generated_prompts) < 5:
        template = random.choice(prompt_templates)
        if template in used_templates:
            continue
        used_templates.add(template)

        # Fill in template variables
        prompt = template
        if "{region}" in prompt:
            prompt = prompt.replace("{region}", random.choice(regions))
        if "{category}" in prompt:
            prompt = prompt.replace("{category}", random.choice(categories))
        if "{metric}" in prompt:
            prompt = prompt.replace("{metric}", random.choice(metrics))
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(topics))
        if "{period}" in prompt:
            prompt = prompt.replace("{period}", random.choice(periods))
        if "{type}" in prompt:
            prompt = prompt.replace("{type}", random.choice(types))
        if "{location}" in prompt:
            prompt = prompt.replace("{location}", random.choice(locations))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(subjects))
        if "{source}" in prompt:
            prompt = prompt.replace("{source}", random.choice(sources))
        if "{area}" in prompt:
            prompt = prompt.replace("{area}", random.choice(areas))
        if "{indicator}" in prompt:
            prompt = prompt.replace("{indicator}", random.choice(indicators))
        if "{group1} and {group2}" in prompt:
            g1, g2 = random.choice(group1_group2)
            prompt = prompt.replace("{group1}", g1).replace("{group2}", g2)

        generated_prompts.append(prompt)

    final_prompts = generated_prompts
    print__main_debug(f"🎲 Generated {len(final_prompts)} dynamic prompts")
    for i, p in enumerate(final_prompts, 1):
        print__main_debug(f"   {i}. {p}")

    return final_prompts


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
@retry_on_ssl_connection_error(max_retries=3)
@retry_on_prepared_statement_error(max_retries=3)
async def main(prompt=None, thread_id=None, checkpointer=None, run_id=None):
    """Main entry point for the application.

    This async function serves as the central coordinator for the data analysis process.
    It handles prompt acquisition from different sources (function parameter,
    command line, or default), initializes tracing for observability, and
    executes the LangGraph workflow. A thread ID is generated to allow
    tracking of each analysis run independently.

    Args:
        prompt (str, optional): The analysis prompt to process. If None and script is run
                               directly, will attempt to get from command line args.
        thread_id (str, optional): The conversation thread ID for memory. If None and script is run
                                   directly, a new thread ID will be generated.
        checkpointer (optional): External checkpointer instance for shared memory. If None,
                                creates a new InMemorySaver instance.
        run_id (str, optional): The run ID for LangSmith tracing. If None, will generate one.

    Returns:
        dict: A dictionary containing the prompt, result, and thread_id for downstream
              processing or API responses.
    """
    print__analysis_tracing_debug("29 - MAIN ENTRY: main() function entry point")
    print__main_debug("29 - MAIN ENTRY: main() function entry point")

    # Handle prompt sourcing - command line args have priority over defaults
    # This allows flexibility in how the application is used
    if prompt is None and __name__ == "__main__":
        print__analysis_tracing_debug(
            "30 - COMMAND LINE ARGS: Processing command line arguments"
        )
        parser = argparse.ArgumentParser(description="Run data analysis with LangGraph")
        parser.add_argument(
            "prompt",
            nargs="?",
            default=DEFAULT_PROMPT,
            help=f'Analysis prompt (default: "{DEFAULT_PROMPT}")',
        )
        parser.add_argument(
            "--thread_id",
            type=str,
            default=None,
            help="Conversation thread ID for memory",
        )
        parser.add_argument(
            "--run_id", type=str, default=None, help="Run ID for LangSmith tracing"
        )
        args = parser.parse_args()
        prompt = args.prompt
        thread_id = args.thread_id
        run_id = args.run_id

    # Ensure we always have a valid prompt to avoid None-type errors downstream
    if prompt is None:
        print__analysis_tracing_debug("31 - DEFAULT PROMPT: Using default prompt")
        prompt = DEFAULT_PROMPT
    else:
        print__analysis_tracing_debug(
            f"32 - PROMPT PROVIDED: Using provided prompt (length: {len(prompt)})"
        )

    # Use a thread_id for short-term memory (thread-level persistence)
    if thread_id is None:
        thread_id = f"data_analysis_{uuid.uuid4().hex[:8]}"
        print__analysis_tracing_debug(
            f"33 - THREAD ID GENERATED: Generated new thread_id {thread_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"34 - THREAD ID PROVIDED: Using provided thread_id {thread_id}"
        )

    # Generate run_id if not provided (for command-line usage)
    if run_id is None:
        run_id = str(uuid.uuid4())
        print__analysis_tracing_debug(
            f"35 - RUN ID GENERATED: Generated new run_id {run_id}"
        )
    else:
        print__analysis_tracing_debug(
            f"36 - RUN ID PROVIDED: Using provided run_id {run_id}"
        )

    # Initialize tracing for debugging and performance monitoring
    # This is crucial for production deployments to track execution paths
    # instrument(project_name="LangGraph_czsu-multi-agent-text-to-sql", framework=Framework.LANGGRAPH)

    print__analysis_tracing_debug("37 - MEMORY MONITORING: Starting memory monitoring")
    # MEMORY LEAK PREVENTION: Track memory before and after analysis
    # Memory monitoring before analysis
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    print__memory_debug(f"🔍 MEMORY: Starting analysis with {memory_before:.1f}MB RSS")
    print__analysis_tracing_debug(
        f"38 - MEMORY BASELINE: Memory before analysis: {memory_before:.1f}MB RSS"
    )

    # Force garbage collection before starting
    collected = gc.collect()
    print__memory_debug(f"🧹 MEMORY: Pre-analysis GC collected {collected} objects")

    print__analysis_tracing_debug("40 - CHECKPOINTER SETUP: Setting up checkpointer")
    # Create the LangGraph execution graph with standard AsyncPostgresSaver
    # We use interrupt_after=['save'] and minimal state in save_node to optimize storage
    if checkpointer is None:
        try:
            print__analysis_tracing_debug(
                "41 - POSTGRES CHECKPOINTER: Attempting to get PostgreSQL checkpointer"
            )
            checkpointer = await get_global_checkpointer()
            print__analysis_tracing_debug(
                "42 - POSTGRES SUCCESS: PostgreSQL checkpointer obtained"
            )
        except Exception as e:
            print__analysis_tracing_debug(
                f"43 - POSTGRES FAILED: Failed to initialize PostgreSQL checkpointer - {str(e)}"
            )
            print__main_debug(f"⚠️ Failed to initialize PostgreSQL checkpointer: {e}")
            # Fallback to InMemorySaver to ensure application still works
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
            print__analysis_tracing_debug(
                "44 - INMEMORY FALLBACK: Using InMemorySaver fallback"
            )
            print__main_debug("⚠️ Using InMemorySaver fallback")
    else:
        print__analysis_tracing_debug(
            f"45 - CHECKPOINTER PROVIDED: Using provided checkpointer ({type(checkpointer).__name__})"
        )

    print__analysis_tracing_debug(
        "46 - GRAPH CREATION: Creating LangGraph execution graph"
    )
    graph = create_graph(checkpointer=checkpointer)
    print__analysis_tracing_debug(
        "47 - GRAPH CREATED: LangGraph execution graph created successfully"
    )

    # FIX: Escape curly braces in prompt to prevent f-string parsing errors
    prompt_escaped = prompt.replace("{", "{{").replace("}", "}}")
    print__main_debug(
        f"🚀 Processing prompt: {prompt_escaped} (thread_id={thread_id}, run_id={run_id})"
    )
    print__analysis_tracing_debug(
        f"48 - PROCESSING START: Processing prompt with thread_id={thread_id}, run_id={run_id}"
    )

    # Configuration for thread-level persistence and LangSmith tracing
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id}
    print__analysis_tracing_debug(
        "49 - CONFIG SETUP: Configuration for thread-level persistence and LangSmith tracing"
    )

    print__analysis_tracing_debug("50 - STATE CHECK: Checking for existing state")
    # Check if there's existing state for this thread to determine if this is a new or continuing conversation
    try:
        existing_state = await graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        is_continuing_conversation = (
            existing_state
            and existing_state.values
            and existing_state.values.get("messages")
            and len(existing_state.values.get("messages", [])) > 0
        )
        print__main_debug(f"🔍 Found existing state: {existing_state is not None}")
        print__analysis_tracing_debug(
            f"51 - STATE CHECK RESULT: Found existing state: {existing_state is not None}"
        )
        if existing_state and existing_state.values:
            msg_count = len(existing_state.values.get("messages", []))
            print__main_debug(f"📋 Message count: {msg_count}")
            print__analysis_tracing_debug(
                f"52 - MESSAGE COUNT: Message count: {msg_count}"
            )
        print__main_debug(f"🔀 Continuing conversation: {is_continuing_conversation}")
        print__analysis_tracing_debug(
            f"53 - CONVERSATION TYPE: Continuing conversation: {is_continuing_conversation}"
        )
    except Exception as e:
        print__main_debug(f"❌ Error checking existing state: {e}")
        print__analysis_tracing_debug(
            f"54 - STATE CHECK ERROR: Error checking existing state - {str(e)}"
        )
        is_continuing_conversation = False

    print__analysis_tracing_debug("55 - STATE PREPARATION: Preparing input state")
    # Prepare input state based on whether this is a new or continuing conversation
    if is_continuing_conversation:
        print__analysis_tracing_debug(
            "56 - CONTINUING CONVERSATION: Preparing state for continuing conversation"
        )
        # For continuing conversations, pass only the fields that need to be updated
        # The checkpointer will merge this with the existing state
        # CRITICAL FIX: Also reset rewritten_prompt and queries to prevent double execution
        input_state = {
            "prompt": prompt,
            "rewritten_prompt": None,  # Critical: reset to force fresh rewrite
            "iteration": 0,  # Reset for new question
            "queries_and_results": [],  # Critical: reset queries to prevent using old ones
            "followup_prompts": [],  # Reset follow-up prompts for new question
            "final_answer": "",  # Reset final answer
            # Reset retrieval results to force fresh search
            "hybrid_search_results": [],
            "most_similar_selections": [],
            "top_selection_codes": [],
            "hybrid_search_chunks": [],
            "most_similar_chunks": [],
            "top_chunks": [],
        }
    else:
        print__analysis_tracing_debug(
            "57 - NEW CONVERSATION: Preparing state for new conversation"
        )
        # Generate initial follow-up prompts for new conversations
        initial_followup_prompts = generate_initial_followup_prompts()
        print__main_debug(
            f"💡 Generated {len(initial_followup_prompts)} initial follow-up prompts for new conversation"
        )

        # For new conversations, initialize with COMPLETE state including ALL fields from DataAnalysisState
        # CRITICAL FIX: All state fields must be initialized for checkpointing to work properly
        input_state = {
            # Basic conversation fields
            "prompt": prompt,
            "rewritten_prompt": None,
            "messages": [
                SystemMessage(content=""),
                AIMessage(content=""),
            ],  # Initialize for new conversation
            "iteration": 0,
            "queries_and_results": [],
            "chromadb_missing": False,
            "final_answer": "",  # Initialize final_answer field
            # MISSING FIELDS - These were causing checkpoint storage issues
            "reflection_decision": "",  # Last decision from reflection node
            "hybrid_search_results": [],  # Intermediate hybrid search results before reranking
            "most_similar_selections": [],  # List of (selection_code, cohere_rerank_score) after reranking
            "top_selection_codes": [],  # List of top N selection codes
            # PDF chunk functionality states
            "hybrid_search_chunks": [],  # Intermediate hybrid search results for PDF chunks
            "most_similar_chunks": [],  # List of (document, cohere_rerank_score) after reranking PDF chunks
            "top_chunks": [],  # List of top N PDF chunks that passed relevance threshold
            # Follow-up prompts functionality
            "followup_prompts": initial_followup_prompts,  # Pre-populated with initial suggestions
        }

    print__analysis_tracing_debug("58 - GRAPH EXECUTION: Starting LangGraph execution")
    print__main_debug(f"🚀 About to call graph.ainvoke() with thread_id={thread_id}, run_id={run_id}")
    print__main_debug(f"🚀 Input state keys: {list(input_state.keys())}")
    
    # Execute the graph with checkpoint configuration and run_id for LangSmith tracing
    # Checkpoints allow resuming execution if interrupted and maintaining conversation memory
    result = await graph.ainvoke(input_state, config=config)
    
    print__main_debug(f"✅ graph.ainvoke() completed for thread_id={thread_id}, run_id={run_id}")

    print__analysis_tracing_debug(
        "59 - GRAPH EXECUTION COMPLETE: LangGraph execution completed"
    )
    # MEMORY LEAK PREVENTION: Monitor memory after graph execution
    memory_after_graph = process.memory_info().rss / 1024 / 1024
    memory_growth_graph = memory_after_graph - memory_before
    print__memory_debug(
        f"🔍 MEMORY: After graph execution: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )
    print__memory_debug(
        f"60 - MEMORY CHECK: Memory after graph: {memory_after_graph:.1f}MB RSS (growth: {memory_growth_graph:.1f}MB)"
    )

    if memory_growth_graph > float(os.environ.get("GC_MEMORY_THRESHOLD", "1900")):
        print__memory_debug(
            f"⚠️ MEMORY: Suspicious growth detected: {memory_growth_graph:.1f}MB during graph execution!"
        )
        print__analysis_tracing_debug(
            f"61 - MEMORY WARNING: Suspicious memory growth detected: {memory_growth_graph:.1f}MB"
        )

        print__memory_debug(
            f"🚨 MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - implementing emergency cleanup"
        )
        print__analysis_tracing_debug(
            f"62 - MEMORY EMERGENCY: {memory_growth_graph:.1f}MB growth - emergency cleanup"
        )

        # Emergency garbage collection
        collected = gc.collect()
        print__memory_debug(f"🧹 MEMORY: Emergency GC collected {collected} objects")
        print__analysis_tracing_debug(
            f"63 - EMERGENCY GC: Emergency GC collected {collected} objects"
        )

        # Check memory after emergency GC
        memory_after_gc = process.memory_info().rss / 1024 / 1024
        freed_by_gc = memory_after_graph - memory_after_gc
        print__memory_debug(
            f"🧹 MEMORY: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )
        print__memory_debug(
            f"64 - EMERGENCY GC RESULT: Emergency GC freed {freed_by_gc:.1f}MB, current: {memory_after_gc:.1f}MB"
        )

        # Update memory tracking
        memory_after_graph = memory_after_gc
        memory_growth_graph = memory_after_graph - memory_before

    print__analysis_tracing_debug("65 - RESULT PROCESSING: Processing graph result")
    # Log details about the result to understand memory usage
    try:
        result_size = len(str(result)) / 1024 if result else 0  # Size in KB
        print__memory_debug(f"🔍 MEMORY: Result object size: {result_size:.1f}KB")
        print__analysis_tracing_debug(
            f"66 - RESULT SIZE: Result object size: {result_size:.1f}KB"
        )
    except:
        print__memory_debug(f"🔍 MEMORY: Could not determine result size")

    print__analysis_tracing_debug(
        "68 - FINAL CLEANUP: Starting final cleanup and monitoring"
    )
    # MEMORY LEAK PREVENTION: Final cleanup and monitoring before return
    try:
        # Final garbage collection to clean up any temporary objects from graph execution
        collected = gc.collect()
        print__memory_debug(
            f"🧹 MEMORY: Final cleanup GC collected {collected} objects"
        )
        print__analysis_tracing_debug(
            f"69 - FINAL GC: Final cleanup GC collected {collected} objects"
        )

        # Final memory check
        memory_final = process.memory_info().rss / 1024 / 1024
        total_growth = memory_final - memory_before

        print__memory_debug(
            f"🔍 MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )
        print__analysis_tracing_debug(
            f"70 - FINAL MEMORY: Final memory: {memory_final:.1f}MB RSS (total growth: {total_growth:.1f}MB)"
        )

        # Warn about high memory retention patterns
        if total_growth > 100:  # More than 100MB total growth
            print__memory_debug(
                f"⚠️ MEMORY WARNING: High memory retention ({total_growth:.1f}MB) detected!"
            )
            print__memory_debug(
                f"💡 MEMORY: Consider investigating LangGraph nodes for memory leaks"
            )
            print__analysis_tracing_debug(
                f"71 - HIGH MEMORY WARNING: High memory retention ({total_growth:.1f}MB) detected!"
            )

    except Exception as memory_error:
        print__memory_debug(
            f"⚠️ MEMORY: Error during final memory check: {memory_error}"
        )
        print__memory_debug(
            f"72 - MEMORY ERROR: Error during final memory check - {str(memory_error)}"
        )

    print__analysis_tracing_debug(
        "73 - RESULT EXTRACTION: Extracting values from graph result"
    )
    # Extract values from the graph result dictionary
    # The graph now uses a messages list: [summary (SystemMessage), last_message (AIMessage)]
    queries_and_results = result["queries_and_results"]
    final_answer = (
        result["messages"][-1].content
        if result.get("messages") and len(result["messages"]) > 1
        else ""
    )

    # Use top_selection_codes for dataset reference (use first if available)
    top_selection_codes = result.get("top_selection_codes", [])
    sql_query = queries_and_results[-1][0] if queries_and_results else None
    followup_prompts = result.get("followup_prompts", [])

    print__analysis_tracing_debug(
        f"74 - SELECTION CODES: Processing {len(top_selection_codes)} selection codes"
    )
    print__analysis_tracing_debug(
        f"74a - FOLLOWUP PROMPTS: Extracted {len(followup_prompts)} follow-up prompts from graph result"
    )
    # Filter to only include selection codes actually used in queries
    used_selection_codes = get_used_selection_codes(
        queries_and_results, top_selection_codes
    )
    print__analysis_tracing_debug(
        f"75 - USED CODES: {len(used_selection_codes)} selection codes actually used"
    )

    dataset_url = None
    if used_selection_codes:
        dataset_url = f"/datasets/{used_selection_codes[0]}"
        print__analysis_tracing_debug(
            f"76 - DATASET URL: Generated dataset URL: {dataset_url}"
        )

    print__analysis_tracing_debug(
        "77 - TOP CHUNKS SERIALIZATION: Converting top_chunks to JSON-serializable format"
    )
    # Convert the result to a JSON-serializable format
    # Convert top_chunks (Document objects) to JSON-serializable format
    top_chunks_serialized = []
    if result.get("top_chunks"):
        chunk_count = len(result["top_chunks"])
        print__main_debug(f"📦 main.py - Found {chunk_count} top_chunks to serialize")
        print__analysis_tracing_debug(
            f"78 - CHUNKS FOUND: Found {chunk_count} top_chunks to serialize"
        )
        for i, chunk in enumerate(result["top_chunks"]):
            chunk_data = {
                "content": (
                    chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                ),
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
            }
            top_chunks_serialized.append(chunk_data)
            if i == 0:  # Log first chunk for debugging
                content_preview = chunk_data["content"][:100]
                print__main_debug(
                    f"🔍 main.py - First chunk content preview: {content_preview}..."
                )
                print__analysis_tracing_debug(
                    f"79 - FIRST CHUNK: First chunk content preview: {content_preview}..."
                )
    else:
        print__main_debug("⚠️ main.py - No top_chunks found in result")
        print__analysis_tracing_debug("80 - NO CHUNKS: No top_chunks found in result")

    print__analysis_tracing_debug(
        "81 - RESULT SERIALIZATION: Creating serializable result dictionary"
    )
    serializable_result = {
        "prompt": prompt,
        "result": final_answer,
        "queries_and_results": queries_and_results,
        "thread_id": thread_id,
        "top_selection_codes": used_selection_codes,  # Return only codes actually used in queries
        "iteration": result.get("iteration", 0),
        "max_iterations": MAX_ITERATIONS,
        "sql": sql_query,
        "datasetUrl": dataset_url,
        "top_chunks": top_chunks_serialized,  # Add serialized PDF chunks for frontend
        "followup_prompts": followup_prompts,  # Add follow-up prompts from graph state
    }

    print__main_debug(
        f"📦 main.py - Serializable result includes {len(top_chunks_serialized)} top_chunks"
    )
    print__main_debug(
        f"💡 main.py - Serializable result includes {len(followup_prompts)} followup_prompts"
    )
    print__analysis_tracing_debug(
        f"82 - SERIALIZATION COMPLETE: Serializable result includes {len(top_chunks_serialized)} top_chunks and {len(followup_prompts)} followup_prompts"
    )
    print__analysis_tracing_debug("83 - MAIN EXIT: main() function returning result")

    return serializable_result


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# Note: This block is commented out to prevent Railway from auto-executing this file.
# Railway's RAILPACK builder was detecting main.py as an entry point and running it
# instead of executing the startCommand (uvicorn).
#
# To run the analysis CLI manually, use:
#   python -m asyncio -c "from main import main; import asyncio; asyncio.run(main())"
# Or create a separate CLI script that imports and calls main()
#
# if __name__ == "__main__":
#     asyncio.run(main())

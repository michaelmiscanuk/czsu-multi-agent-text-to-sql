import os
from pathlib import Path
import sys

# Handle base directory path
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from my_agent.utils.state import DataAnalysisState
from my_agent.utils.nodes import (
    get_schema_node,
    query_node,
    reflect_node,
    format_answer_node,
    submit_final_answer_node,
    save_node,
    retrieve_similar_selections_node,
    relevant_selections_node,
    rewrite_query_node
)
import asyncio

# Initialize complete state with all required properties
state = DataAnalysisState({
    "prompt": "What is the population of Prague in 2023?",
    "rewritten_prompt": None,
    "messages": [],
    "iteration": 0,
    "queries_and_results": [],
    "reflection_decision": None,
    "most_similar_selections": [],
    "top_selection_codes": [],
    "chromadb_missing": False
})

# Call node and print result
result = asyncio.run(rewrite_query_node(state))
print("Input state:", state)
print("\nOutput state:", result)
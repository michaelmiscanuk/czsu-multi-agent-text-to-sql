"""
Simple test to access LangSmith trace outputs.
Uses the exact same logic as pairwise_compare.py
"""

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURE THIS
# ============================================================================
TRACE_ID = "779a0cfd-9d41-4922-b32f-d717a50049f7"  # Your trace ID from LangSmith
NODE_NAME = "format_answer"  # Node you want to check
STATE_KEY = "final_answer"  # State field you want to extract

# ============================================================================
# EXTRACTION FUNCTION (same as pairwise_compare.py)
# ============================================================================


def get_output(trace_id: str, node_name: str, state_key: str) -> str:
    """Get output from specified node and state."""
    client = Client()

    # Get all runs in trace
    all_runs = list(client.list_runs(trace_id=trace_id))

    # Find the target node
    for r in all_runs:
        if r.name == node_name and r.outputs:
            if state_key in r.outputs:
                return str(r.outputs[state_key])

    return ""


# ============================================================================
# TEST
# ============================================================================

print(f"Looking for node: {NODE_NAME}")
print(f"Looking for state: {STATE_KEY}")
print()

result = get_output(TRACE_ID, NODE_NAME, STATE_KEY)

if result:
    preview = result[:300] + "..." if len(result) > 300 else result
    print(f"✅ Found {STATE_KEY} in node {NODE_NAME}")
    print(f"   Length: {len(result)} chars")
    print(f"   Content: {preview}")
else:
    print(f"❌ Could not find {STATE_KEY} in node {NODE_NAME}")
    print(f"   Check if node name and state key are correct")

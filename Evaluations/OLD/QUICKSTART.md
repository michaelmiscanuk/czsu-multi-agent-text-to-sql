# Quick Start Guide - Pairwise Comparison with LangGraph

## Problem Fixed
Previously, pairwise comparison showed empty responses because we were trying to access `runs[0].outputs.get("output")`, but LangGraph stores the final answer in a specific node's state, not the root run outputs.

## Solution Summary

### 1. Added Configuration (Top of pairwise_compare.py)
```python
# Which node/state to extract
TARGET_NODE_NAME = "format_answer"
TARGET_STATE_KEY = "final_answer"

# Debug mode
DEBUG_OUTPUT_EXTRACTION = True
```

### 2. Added Smart Extraction Function
The new `extract_output_from_run()` function:
- First checks root outputs (fast)
- Then searches child runs for your target node
- Falls back to common keys if needed
- Prints debug info when enabled

### 3. Updated Evaluator
```python
# Old (doesn't work with LangGraph):
pred_a = runs[0].outputs.get("output", "")

# New (works with LangGraph):
pred_a = extract_output_from_run(runs[0], client)
```

## How to Use

### Step 1: Test with One Trace First
```bash
# 1. Open test_langsmith_trace_access.py
# 2. Copy a trace ID from LangSmith UI
# 3. Update TEST_TRACE_ID in the script
# 4. Run it
python test_langsmith_trace_access.py
```

This shows you:
- All available nodes in your graph
- Where each state field is stored
- What keys to use for TARGET_NODE_NAME and TARGET_STATE_KEY

### Step 2: Run Pairwise Comparison
```bash
# Already configured for your setup:
# TARGET_NODE_NAME = "format_answer"
# TARGET_STATE_KEY = "final_answer"
# DEBUG_OUTPUT_EXTRACTION = True

python pairwise_compare.py
```

Watch the console output - it will show you exactly what it's finding and extracting.

### Step 3: Verify Results
The debug output will show:
```
🔍 Extracting output from run: 019b40c5...
   ✅ Found 'final_answer' in node 'format_answer': The average consumer price...
   
📊 Extraction Results:
   Response A length: 1234 chars
   Response B length: 1456 chars
```

If you see empty responses, the test script will help you find the correct node/state names.

## Common Configurations

### Compare final answers (default):
```python
TARGET_NODE_NAME = "format_answer"
TARGET_STATE_KEY = "final_answer"
```

### Compare messages:
```python
TARGET_NODE_NAME = "summarize_messages_format"
TARGET_STATE_KEY = "messages"
TARGET_MESSAGE_INDEX = 0  # First message
```

### Compare any other state:
Use the test script to find the right node and state key!

## Files Changed
1. ✅ `pairwise_compare.py` - Main comparison script (updated)
2. ✅ `test_langsmith_trace_access.py` - Testing/exploration script (new)
3. ✅ `README_OUTPUT_EXTRACTION.md` - Detailed docs (new)
4. ✅ `QUICKSTART.md` - This file (new)

## Troubleshooting
- **Empty responses**: Run test script first
- **Wrong content**: Check node names in test script output
- **Errors**: Enable DEBUG_OUTPUT_EXTRACTION to see what's happening

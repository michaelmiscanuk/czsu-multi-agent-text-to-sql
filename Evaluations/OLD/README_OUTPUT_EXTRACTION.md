# LangSmith Output Extraction for Pairwise Comparison

## Problem

When comparing LangGraph traces in LangSmith, the final outputs are not directly available in the root run's `outputs` field. LangGraph stores outputs in specific node executions, so we need to access child runs to extract the correct state values.

## Solution

### 1. Test Script: `test_langsmith_trace_access.py`

Use this script to explore trace structure and find where your outputs are stored:

```bash
# 1. Copy a trace ID from LangSmith UI (from the Run tab)
# 2. Edit the script and set TEST_TRACE_ID
# 3. Run the script
python test_langsmith_trace_access.py
```

The script will:
- Show all nodes in your trace
- Display the structure of inputs/outputs for each node
- Help you identify where `final_answer` or other states are stored
- Provide recommendations for accessing the data

### 2. Updated Pairwise Comparison: `pairwise_compare.py`

#### Configuration Section (Top of File)

```python
# Configure which node/state to compare
TARGET_NODE_NAME = "format_answer"  # Node name from LangGraph
TARGET_STATE_KEY = "final_answer"   # State field to extract

# Enable debug prints
DEBUG_OUTPUT_EXTRACTION = True
```

#### Common Configurations

**Compare final answers:**
```python
TARGET_NODE_NAME = "format_answer"
TARGET_STATE_KEY = "final_answer"
```

**Compare first message in messages state:**
```python
TARGET_NODE_NAME = "summarize_messages_format"
TARGET_STATE_KEY = "messages"
TARGET_MESSAGE_INDEX = 0  # 0 for first message, 1 for second
```

**Compare from any other node:**
```python
TARGET_NODE_NAME = "your_node_name"  # Check test script output
TARGET_STATE_KEY = "your_state_key"  # e.g., "queries_and_results", "top_chunks"
```

### 3. How It Works

The extraction logic follows this hierarchy:

1. **Root outputs** (fast path): Check if the state key exists directly in run.outputs
2. **Child runs** (LangGraph pattern): Search all child runs for the target node name, then extract the state key
3. **Fallback keys**: Try common keys like "output", "result", "answer", "response"

#### Debug Output Example

When `DEBUG_OUTPUT_EXTRACTION = True`, you'll see:

```
🔍 Extracting output from run: 019b40c5-8543-7f22-8619-828a65fe06aa
   Run name: LangGraph
   Trace ID: 019b40c5-8543-7f22-8619-828a65fe06aa
   Root outputs keys: ['prompt', 'messages', 'iteration']
   ⚠️ 'final_answer' not in root outputs, searching child runs...
   Found 45 total runs in trace
   Available nodes: ['format_answer', 'generate_query', 'rewrite_prompt', ...]
   Found 1 runs for node 'format_answer'
   Checking run abc123..., outputs keys: ['final_answer', 'messages']
   ✅ Found 'final_answer' in node 'format_answer': The average consumer price...
```

### 4. Workflow

1. **Test with one trace:**
   ```bash
   python test_langsmith_trace_access.py
   ```

2. **Configure pairwise comparison:**
   - Edit `TARGET_NODE_NAME` and `TARGET_STATE_KEY` based on test results
   - Set `DEBUG_OUTPUT_EXTRACTION = True` for first run

3. **Run comparison:**
   ```bash
   python pairwise_compare.py
   ```

4. **Check debug output** to verify correct extraction

5. **Disable debug** once working:
   ```python
   DEBUG_OUTPUT_EXTRACTION = False
   ```

### 5. Troubleshooting

**Problem: Both responses are empty**
- Run test script first to explore trace structure
- Check that TARGET_NODE_NAME matches your graph node names (see agent.py)
- Check that TARGET_STATE_KEY matches your state fields (see state.py)

**Problem: Responses are cut off or incomplete**
- Check if the state value is a list or complex object
- You might need to add custom handling in `extract_output_from_run()`

**Problem: Getting wrong content**
- Verify the node name using the test script
- Some nodes might execute multiple times - the function returns the first match

### 6. Advanced: Custom Extraction

If you need custom extraction logic, modify the `extract_output_from_run()` function:

```python
def extract_output_from_run(run: Run, client: Client) -> str:
    # ... existing code ...
    
    # Add your custom logic here
    if TARGET_STATE_KEY == "my_custom_field":
        # Handle special case
        pass
    
    return extracted_value
```

## Files Modified

1. ✅ `test_langsmith_trace_access.py` - New testing script
2. ✅ `pairwise_compare.py` - Updated with:
   - Configuration variables at top
   - `extract_output_from_run()` function
   - Debug printing
   - Better fallback logic

## Next Steps

After testing:
1. Set `DEBUG_OUTPUT_EXTRACTION = False`
2. Run on your full experiment pairs
3. Check LangSmith UI for pairwise comparison results

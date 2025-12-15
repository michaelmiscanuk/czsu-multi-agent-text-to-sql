# Quick Start Guide - Unused Functions Detector

## 🚀 Quick Usage

```bash
python find_unused_functions.py
```

That's it! The script will automatically analyze your `czsu-multi-agent-text-to-sql` project.

## 📊 Latest Analysis Results

**Date**: Run on your machine
**Total functions analyzed**: 991
**Unused functions found**: 32 (all high confidence 95-100%)

## 🎯 High Confidence Unused Functions (100%)

### Category 1: Debug Functions (Safe to Remove)
These are debug print functions that are not currently used:

- `print__chat_messages_debug` - api\utils\debug.py:206
- `print__data_table_debug` - api\utils\debug.py:276
- `print__chat_thread_id_checkpoints_debug` - api\utils\debug.py:290
- `print__debug_pool_status_debug` - api\utils\debug.py:304
- `print__chat_thread_id_run_ids_debug` - api\utils\debug.py:318
- `print__debug_run_id_debug` - api\utils\debug.py:332
- `print__admin_clear_cache_debug` - api\utils\debug.py:346

**Action**: Verify these aren't needed for debugging, then remove.

### Category 2: Legacy/Deprecated Functions
Functions marked as legacy or kept for backward compatibility:

- `retry_node` - Evaluations\LangSmith_Evaluation\langsmith_evaluate_selection_retrieval.py:241
  - Doc: "Legacy retry function - kept for backward compatibility but now uses n..."
- `save_parsed_text_to_file_legacy` - data\helpers.py:220
- `load_parsed_text_from_file_legacy` - data\helpers.py:239

**Action**: Safe to remove if backward compatibility not needed.

### Category 3: Cleanup/Maintenance Functions
- `cleanup_old_entries` - api\utils\cancellation.py:85
  - Doc: "Remove old entries from the registry to prevent memory leaks"

**Action**: Check if this should be called periodically. If not needed, remove.

### Category 4: Test LLM Functions
Test functions for LLM models (not used in production):

- `get_azure_llm_gpt_4o_test` - my_agent\utils\models.py:39 (REMOVED - refactored into get_azure_openai_chat_llm)
- `get_azure_llm_gpt_4o_mini_test` - my_agent\utils\models.py:71 (REMOVED - refactored into get_azure_openai_chat_llm)
- `get_ollama_llm_test` - my_agent\utils\models.py:143
- `get_azure_embedding_model_test` - my_agent\utils\models.py:184
- `get_langchain_azure_embedding_model_test` - my_agent\utils\models.py:220

**Action**: Keep for manual testing or remove if not needed.

### Category 5: PDF Processing Functions
Unused PDF processing functions (possibly old implementations):

- `process_pdf_to_chromadb` - Multiple files
- `process_pdf_pages_to_chunks` - Multiple files
- `convert_html_tables_to_markdown` - data\pdf_to_chromadb__llamaparse_tables_in_yaml.py:964

**Action**: Check if these are old implementations. If so, remove.

### Category 6: ChromaDB Functions
- `get_or_create_chromadb_collection` - metadata\chromadb_client_factory.py:207

**Action**: Verify not used, then remove.

### Category 7: Data Processing
- `save_to_sqlite` - metadata\llm_selection_descriptions\dynamic_parallel_dataframe_llm_processor__03.py:382

**Action**: Check if needed for data pipeline, otherwise remove.

### Category 8: Other Functions
- `_arun` - my_agent\utils\tools.py:104 (async method)
- `http_request_pattern` - other\2.py:38
- `add_failed_request` - tests\api\test_phase12_performance.py:130
- `record_concurrent_result` - tests\database\test_checkpointer_stress.py:300
- `emit` - tests\helpers.py:352
- `download_from_gdrive` - unzip_files.py:25

**Action**: Review each individually.

## ✅ Recommended Removal Order

### Phase 1: Debug Functions (Lowest Risk)
Remove all unused debug print functions first:
```python
# These 7 functions in api\utils\debug.py
print__chat_messages_debug
print__data_table_debug
print__chat_thread_id_checkpoints_debug
print__debug_pool_status_debug
print__chat_thread_id_run_ids_debug
print__debug_run_id_debug
print__admin_clear_cache_debug
```

### Phase 2: Test Functions
Remove test LLM functions if not needed:
```python
# These 3 functions in my_agent\utils\models.py (2 were refactored)
# get_azure_llm_gpt_4o_test (REMOVED - refactored into get_azure_openai_chat_llm)
# get_azure_llm_gpt_4o_mini_test (REMOVED - refactored into get_azure_openai_chat_llm)
get_ollama_llm_test
get_azure_embedding_model_test
get_langchain_azure_embedding_model_test
```

### Phase 3: Legacy Functions
Remove legacy file I/O functions:
```python
# These 2 functions in data\helpers.py
save_parsed_text_to_file_legacy
load_parsed_text_from_file_legacy
```

### Phase 4: Review Remaining
Carefully review the remaining 18 functions individually.

## 🛡️ Safety Checklist for Each Function

Before removing ANY function:

1. **Search entire project**:
   ```
   Ctrl+Shift+F in VS Code
   Search for: function_name
   ```

2. **Check git history**:
   ```bash
   git log -p --all -S "function_name"
   ```

3. **Check for string references**:
   - Search for the function name in quotes
   - Look for `getattr(obj, "function_name")`

4. **Run tests after removal**:
   ```bash
   pytest
   # or your test command
   ```

5. **Commit incrementally**:
   ```bash
   git add .
   git commit -m "Remove unused function: function_name"
   ```

## ⚙️ Customizing the Analysis

### Change Confidence Threshold

Edit `find_unused_functions.py`:

```python
CONFIG = {
    "min_confidence": 95,  # Only show very high confidence
    # or
    "min_confidence": 85,  # Show more candidates
}
```

### Exclude More Directories

```python
CONFIG = {
    "exclude_patterns": [
        "**/*_OLD/**",  # Already excluded
        "**/examples/**",  # Add this
        "**/scripts/**",  # Add this
    ],
}
```

### Add Framework Decorators

If you have custom decorators that mean a function is "used":

```python
CONFIG = {
    "framework_decorators": {
        # ... existing ones ...
        "my_custom_decorator",  # Add yours here
    },
}
```

## 📝 What the Script Catches

✅ **Detects as USED** (will NOT be flagged):
- FastAPI route handlers (`@app.get`, `@router.post`)
- pytest fixtures (`@pytest.fixture`)
- Test functions (`test_*`)
- Event handlers (`@app.on_event`)
- Magic methods (`__init__`, `__str__`)
- Functions in `__all__` exports
- Functions with string references (`getattr(obj, "func")`)

✅ **Detects as UNUSED** (will be flagged):
- Functions never called
- Functions only defined but not used
- Legacy/deprecated functions
- Debug functions not actively used
- Old implementations replaced by new ones

## 🔧 Troubleshooting

### "Project directory not found"
Make sure `czsu-multi-agent-text-to-sql` folder is in the same directory as `find_unused_functions.py`.

### "Syntax error in file"
Some files have syntax errors and will be skipped. This is normal. The script will analyze all valid Python files.

### Too few results
The script is conservative. It may miss some unused functions to avoid false positives. This is by design.

### Function I know is unused is not listed
Check the confidence threshold. Lower it to 85% to see more candidates:
```python
CONFIG = {"min_confidence": 85}
```

## 💡 Pro Tips

1. **Start with debug functions**: These are safest to remove
2. **Remove one at a time**: Commit after each removal
3. **Run tests frequently**: After each removal, run your test suite
4. **Keep test functions**: Even if marked as unused, keep test helper functions
5. **Document removals**: In commit messages, explain why function was removed
6. **Create a branch**: 
   ```bash
   git checkout -b cleanup/remove-unused-functions
   ```

## 📈 Expected Impact

Removing 32 unused functions will:
- **Reduce codebase size** by ~500-1000 lines (estimated)
- **Improve code maintainability** (less code to understand)
- **Reduce confusion** for new developers
- **Improve IDE performance** (fewer symbols to index)
- **No impact on functionality** if done correctly

## ⚠️ Warning Signs

**STOP** removing if:
- ❌ Tests start failing
- ❌ Application crashes on startup
- ❌ Features stop working
- ❌ You're not 100% sure about a function

**When in doubt, keep the function!**

## 📞 Need Help?

If you're unsure about a specific function:

1. Check the function's docstring
2. Search the entire project for its name
3. Check git history: `git log -p --all -S "function_name"`
4. Ask in code review
5. **When in doubt, DON'T remove it**

---

**Remember**: This is a tool to **assist** you, not to automatically remove functions. Always review manually before removing anything!

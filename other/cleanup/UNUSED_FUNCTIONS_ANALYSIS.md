# Unused Functions Analysis

## Overview

The `find_unused_functions.py` script performs comprehensive static analysis to detect unused functions in your Python project with maximum confidence, avoiding false positives that could break your application.

## Features

### 🎯 High Confidence Detection
- **AST-based analysis**: Uses Python's Abstract Syntax Tree for precise function detection
- **Confidence scoring**: Each unused function gets a 0-100% confidence score
- **Smart filtering**: Only reports functions with ≥90% confidence by default

### 🛡️ Framework-Aware Analysis
The script recognizes functions that are used by frameworks even if not called directly:

- **FastAPI/Flask**: Route decorators (`@app.get`, `@router.post`, etc.)
- **pytest**: Test functions and fixtures (`@pytest.fixture`, `test_*`)
- **LangGraph**: Node and tool decorators
- **Event handlers**: `@app.on_event`, middleware functions
- **Pydantic**: Validators (`@validator`, `@root_validator`)
- **asyncio**: Context managers (`@asynccontextmanager`)
- **Click**: CLI commands (`@click.command`)

### 🔍 Dynamic Usage Detection
Identifies functions called dynamically:

- `getattr(obj, "function_name")`
- `hasattr(obj, "function_name")`
- String-based imports
- `__all__` exports in `__init__.py`

### 🎨 Smart Categorization
Automatically recognizes:

- **Magic methods**: `__init__`, `__str__`, etc. (always marked as used)
- **Test functions**: `test_*`, `setup_*`, `teardown_*`
- **Event handlers**: `on_*`, `_on_*`, `handle_*`, `callback_*`
- **Private functions**: `_function` (higher tolerance for being unused)
- **Public API functions**: Functions in `__init__.py` (likely part of API)

## Usage

### Basic Usage

```bash
python find_unused_functions.py
```

This will analyze the `czsu-multi-agent-text-to-sql` directory and generate a report.

### Configuration

Edit the `CONFIG` dictionary at the top of the script to customize:

```python
CONFIG = {
    # Project directory name (must be in same directory as script)
    "project_name": "czsu-multi-agent-text-to-sql",
    
    # Minimum confidence to report (0-100)
    "min_confidence": 90,
    
    # Files to exclude
    "exclude_patterns": [
        "**/*_OLD/**",  # Exclude old versions
        "**/__pycache__/**",
        # Add more patterns as needed
    ],
    
    # Additional framework decorators to recognize
    "framework_decorators": {
        # Add custom decorators here
    },
}
```

## Understanding the Report

### Confidence Levels

The script uses a sophisticated scoring system:

- **100%**: Function is defined but never referenced anywhere
- **95-99%**: Very likely unused, minimal signs of usage
- **90-94%**: Likely unused, but some ambiguous indicators
- **Below 90%**: Not reported (too much uncertainty)

### Confidence Reduction Factors

The script reduces confidence when it finds:

- ✅ **Framework decorators** (-100%, confidence = 0%)
- ✅ **Magic methods** (-100%, confidence = 0%)
- ✅ **Exported in `__all__`** (-100%, confidence = 0%)
- ✅ **Special naming patterns** (-100%, confidence = 0%)
- ✅ **Used in other files** (-40% for 1 usage, -80% for 2+ usages)
- ✅ **String-based references** (-50%)
- ✅ **Has decorators** (-20% per decorator, up to -60%)
- ✅ **In `__init__.py` and public** (-30%)

### Sample Report

```
🔴 HIGH CONFIDENCE (95-100%): 5 functions
────────────────────────────────────────────────────────────────────────────────

📍 old_helper_function
   File: api/utils/helpers.py:45
   Confidence: 98%
   Flags: private
   Doc: This function was replaced by new_helper_function

📍 unused_validation
   File: api/models/validators.py:120
   Confidence: 95%
   Doc: Validates user input (DEPRECATED)

🟡 MEDIUM CONFIDENCE (90-94%): 3 functions
────────────────────────────────────────────────────────────────────────────────

📍 debug_print_state
   File: my_agent/utils/debug.py:78
   Confidence: 92%
   Flags: private, method
```

## How It Avoids False Positives

### 1. FastAPI Routes
```python
@app.get("/health")  # ← Decorator detected
async def health_check():  # ✅ Marked as USED (confidence = 0%)
    return {"status": "ok"}
```

### 2. Event Handlers
```python
@app.on_event("startup")  # ← Framework decorator
async def startup():  # ✅ Marked as USED
    pass
```

### 3. pytest Fixtures
```python
@pytest.fixture  # ← pytest decorator detected
def database_session():  # ✅ Marked as USED
    return Session()
```

### 4. Dynamic Usage
```python
# In code:
handler = getattr(self, "on_message")  # ← String reference detected

# In another file:
def on_message(self):  # ✅ Marked as USED (string reference found)
    pass
```

### 5. Test Functions
```python
def test_user_creation():  # ✅ Marked as USED (test_ prefix)
    assert create_user()
```

### 6. Magic Methods
```python
class MyClass:
    def __init__(self):  # ✅ Marked as USED (magic method)
        pass
```

## Recommended Workflow

1. **Run the analysis**:
   ```bash
   python find_unused_functions.py
   ```

2. **Review HIGH confidence functions first** (95-100%):
   - These are most likely truly unused
   - Search your project for the function name to verify
   - Check git history to understand why it exists

3. **For each function**:
   - Read the docstring and code
   - Search entire project: `Ctrl+Shift+F` in VS Code
   - Check if it's imported anywhere
   - Review git blame to see last changes

4. **Remove incrementally**:
   ```bash
   # Remove one function
   git add -p
   git commit -m "Remove unused function: old_helper_function"
   
   # Run tests
   pytest
   
   # If tests pass, continue with next function
   ```

5. **Review MEDIUM confidence functions** (90-94%):
   - These need more careful review
   - May have subtle usage patterns
   - Consider keeping if uncertain

## What the Script Does NOT Detect

The script is conservative and will NOT flag as unused:

- ❌ Functions with ANY framework decorators
- ❌ Functions starting with `test_`, `setup_`, `on_`, `handle_`, `callback_`
- ❌ Magic methods (`__init__`, `__str__`, etc.)
- ❌ Functions exported in `__all__`
- ❌ Functions referenced in strings (getattr, hasattr, etc.)
- ❌ Functions in `__init__.py` that look like public API
- ❌ Functions with decorators (suspected framework integration)

## Limitations

### String-Based Imports
```python
# May not detect:
module = __import__("my_module")
module.my_function()  # Function might be flagged as unused
```

**Mitigation**: The script scans for `__import__` patterns, but complex dynamic imports may be missed.

### Metaclasses and Descriptors
```python
class Meta(type):
    def validate(self):  # May be flagged as unused
        pass  # But actually called by metaclass magic
```

**Mitigation**: Functions in metaclasses are typically in base classes with special names - review carefully.

### Callback Registration
```python
# In one file:
registry.register("handler", my_callback)

# In another file:
def my_callback():  # Might be flagged as unused
    pass
```

**Mitigation**: The script looks for string references, but complex registration patterns may need manual review.

## Advanced Configuration

### Add Custom Framework Decorators

```python
CONFIG = {
    "framework_decorators": {
        # Add your custom decorators
        "app.get", "app.post",  # Already included
        "my_framework.route",   # Add custom
        "custom_decorator",     # Add custom
    },
}
```

### Adjust Confidence Threshold

```python
CONFIG = {
    "min_confidence": 95,  # Only show very high confidence (stricter)
    # or
    "min_confidence": 85,  # Show more candidates (less strict)
}
```

### Exclude Specific Directories

```python
CONFIG = {
    "exclude_patterns": [
        "**/migrations/**",    # Database migrations
        "**/legacy/**",        # Legacy code
        "**/examples/**",      # Example code
        "**/*_OLD/**",        # Old versions
    ],
}
```

## Troubleshooting

### "Project directory not found"
- Ensure `czsu-multi-agent-text-to-sql` is in the same directory as the script
- Or modify `CONFIG["project_name"]` to match your directory name

### "Syntax error in file"
- Some files may have syntax errors and will be skipped
- Check the warnings in the output
- Fix syntax errors or exclude those files

### Too many false positives
- Increase `min_confidence` to 95 or higher
- Add framework-specific decorators to `framework_decorators`
- Review and add patterns to `special_function_patterns`

### Too few results
- Lower `min_confidence` to 85 or 80
- Review the analysis output for confidence scores
- Check if functions are genuinely used

## Safety Checklist

Before removing any function:

- [ ] Searched entire project for function name
- [ ] Checked if function is in `__all__` exports
- [ ] Reviewed git history for context
- [ ] Read function docstring and code
- [ ] Confirmed no string-based references
- [ ] Verified not used by framework/decorators
- [ ] Run full test suite after removal
- [ ] Committed changes incrementally

## Examples of Safe Removals

### Example 1: Deprecated Helper
```python
# SAFE TO REMOVE (confidence: 98%)
def old_calculate_total(items):
    """DEPRECATED: Use calculate_sum instead"""
    return sum(item.price for item in items)
```

### Example 2: Debug Function
```python
# SAFE TO REMOVE (confidence: 95%)
def _debug_print_state(state):
    """Debug helper - not used in production"""
    print(f"State: {state}")
```

### Example 3: Unused Import
```python
# SAFE TO REMOVE (confidence: 100%)
from typing import Optional

# Function never used
def format_optional_value(val: Optional[str]) -> str:
    return val or "N/A"
```

## Conclusion

This script provides a **high-confidence, conservative approach** to detecting unused functions. It errs on the side of caution, preferring false negatives (missing some unused functions) over false positives (flagging functions that are actually used).

**Always verify manually before removing any function**, especially:
- Functions in core modules
- Functions that might be called by frameworks
- Functions with unclear purposes
- Functions in `__init__.py` files

Remember: **It's better to keep an unused function than to break your application!**

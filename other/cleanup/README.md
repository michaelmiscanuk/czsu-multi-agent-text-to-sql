# Unused Functions Detection Tools

## 🎯 Overview

This directory contains **portable, project-agnostic** tools for detecting and removing unused functions in Python projects.

## ✨ Key Features

- **🔍 Automatic Project Root Detection**: No hardcoded project names
- **📦 Portable**: Drop these scripts into any Python project
- **🛡️ Safe**: Conservative analysis with confidence scoring
- **🎨 Framework-Aware**: Recognizes FastAPI, pytest, LangGraph, etc.

## 📂 Files

- **`find_unused_functions.py`**: Main analysis tool
- **`remove_unused_functions.py`**: Interactive removal helper
- **`README.md`**: This file

## 🚀 Quick Start

### 1. Run Analysis

```bash
python find_unused_functions.py
```

The script will:
1. **Auto-detect project root** by searching for markers (`.git`, `pyproject.toml`, `setup.py`, etc.)
2. Analyze all Python files in the project
3. Generate a report of unused functions with confidence scores

### 2. Review Results

The script reports functions with ≥90% confidence of being unused, categorized by:
- **HIGH confidence (95-100%)**: Very safe to remove
- **MEDIUM confidence (90-94%)**: Review carefully

### 3. Remove Functions (Optional)

```bash
python remove_unused_functions.py
```

Interactive helper that:
- Shows function code
- Searches for references
- Runs tests after removal
- Commits changes via git

## 🎯 How Project Root Detection Works

The scripts automatically find your project root by searching **upward** from the script location for these markers (in order):

1. `.git` directory (Git repository)
2. `pyproject.toml` (Modern Python project)
3. `setup.py` (Traditional Python package)
4. `setup.cfg` (Python package config)
5. `requirements.txt` (Python dependencies)
6. `Pipfile` (Pipenv project)
7. `poetry.lock` (Poetry project)
8. `package.json` (Node.js/mixed projects)
9. `Cargo.toml` (Rust/mixed projects)

**Example**: If you place the script at:
```
my-project/
├── .git/              ← Root marker found here!
├── pyproject.toml
├── src/
│   └── main.py
└── tools/
    └── cleanup/
        └── find_unused_functions.py  ← Script runs from here
```

The script will:
1. Start at `tools/cleanup/`
2. Search upward: `tools/cleanup/` → `tools/` → `my-project/`
3. Find `.git` in `my-project/`
4. Use `my-project/` as project root ✓

## 📍 Where to Place These Scripts

You can place these scripts **anywhere** in your project:

### Option 1: Project Tools Directory (Recommended)
```
my-project/
├── tools/
│   └── find_unused_functions.py
└── src/
```

### Option 2: Cleanup Subdirectory
```
my-project/
├── scripts/
│   └── cleanup/
│       └── find_unused_functions.py
└── src/
```

### Option 3: Root Directory
```
my-project/
├── find_unused_functions.py
└── src/
```

**All work the same way!** The script auto-detects the project root.

## 🔧 Configuration

Edit the `CONFIG` dictionary at the top of `find_unused_functions.py`:

```python
CONFIG = {
    # File patterns to include
    "include_patterns": ["**/*.py"],
    
    # File patterns to exclude
    "exclude_patterns": [
        "**/__pycache__/**",
        "**/venv/**",
        "**/build/**",
        # Add your patterns here
    ],
    
    # Directories to exclude completely
    "exclude_dirs": {
        "__pycache__",
        ".git",
        "venv",
        # Add your directories here
    },
    
    # Minimum confidence to report (0-100)
    "min_confidence": 90,
}
```

## 🎓 Usage Examples

### Analyze Current Project
```bash
# From anywhere in the project
python path/to/find_unused_functions.py
```

### Analyze Different Project
```bash
# Copy script to other project
cp find_unused_functions.py /path/to/other-project/tools/
cd /path/to/other-project/tools/
python find_unused_functions.py
```

### Change Confidence Threshold
Edit the script:
```python
CONFIG = {
    "min_confidence": 95,  # Only show very high confidence
}
```

## 🛡️ What the Script Detects

### ✅ Marked as USED (won't be flagged)
- FastAPI routes: `@app.get`, `@router.post`
- pytest fixtures: `@pytest.fixture`
- Test functions: `test_*`
- Event handlers: `@app.on_event`
- Magic methods: `__init__`, `__str__`
- Functions in `__all__` exports
- Functions with string references: `getattr(obj, "func")`
- LangGraph tools: `@tool`
- Pydantic validators: `@validator`

### 🚨 Marked as UNUSED (will be flagged)
- Functions never called
- Legacy/deprecated functions
- Debug functions not actively used
- Old implementations replaced by new ones

## 📊 Sample Output

```
🚀 Unused Functions Detector
================================================================================
✓ Found project root marker: .git
✓ Project root: /home/user/my-project
📁 Project: /home/user/my-project
📝 Config: Min confidence = 90%
🔍 Analyzing project: /home/user/my-project
================================================================================
✓ Found 145 Python files

📋 Extracting function definitions and usages...
✓ Found 1036 function definitions

🔎 Analyzing string-based function references...
✓ Found 46 string references

🎯 Calculating unused functions with confidence scores...

================================================================================
📊 UNUSED FUNCTIONS REPORT
================================================================================

⚠️  Found 39 potentially unused functions

🔴 HIGH CONFIDENCE (95-100%): 39 functions
--------------------------------------------------------------------------------

📍 old_helper_function
   File: src/utils/helpers.py:45
   Confidence: 100%
   Doc: DEPRECATED - use new_helper instead

... (more functions)

================================================================================
📈 SUMMARY
================================================================================
Total functions analyzed: 1036
High confidence unused (≥95%): 39
Medium confidence unused (90-94%): 0
Total potentially unused: 39
```

## 🔍 Troubleshooting

### "No project root markers found"
**Solution**: Add a `.git` directory or `pyproject.toml` to your project root.

The script will still work (using script directory as root), but may not analyze the full project.

### Script analyzing itself
This is **normal** if you place the script inside the project. The script's own visitor methods will appear as "unused" (they're called by Python's AST framework dynamically).

**Solution**: Add the script's directory to `exclude_patterns` if you don't want it analyzed.

### Wrong project root detected
Check for markers in parent directories. The script searches upward and stops at the first marker found.

**Solution**: Ensure the correct directory has a clear marker (`.git`, `pyproject.toml`, etc.)

## 📝 Notes

- The scripts are **read-only** by default (analysis only)
- `remove_unused_functions.py` requires manual confirmation before removing anything
- Always review functions manually before removing
- Run tests after each removal
- Commit changes incrementally

## 🎉 Benefits

- ✅ **Portable**: Works in any Python project
- ✅ **No configuration needed**: Auto-detects project structure
- ✅ **Safe**: Conservative analysis prevents false positives
- ✅ **Smart**: Recognizes framework patterns
- ✅ **Fast**: Analyzes ~1000 functions in seconds

## 📚 For More Information

See the documentation files in the parent directory for detailed explanations:
- `UNUSED_FUNCTIONS_ANALYSIS.md` - Complete technical documentation
- `QUICK_START.md` - Quick reference guide
- `README_UNUSED_FUNCTIONS.md` - Detailed usage guide

---

**Version**: 1.0.0  
**Updated**: Made fully portable with automatic project root detection

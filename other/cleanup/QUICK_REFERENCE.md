# 🚀 Quick Reference

## One-Line Commands

```bash
# Analyze current project (from anywhere)
python find_unused_functions.py

# Interactive removal helper
python remove_unused_functions.py
```

## How It Works

```
Script Location: /project/tools/cleanup/find_unused_functions.py
                              ↓
                    [Search upward for markers]
                              ↓
            .git, pyproject.toml, setup.py, etc.
                              ↓
                    [Found at /project/]
                              ↓
              ✓ Use /project/ as root
                              ↓
               Analyze all *.py files
                              ↓
            Report unused functions
```

## Project Root Markers (in order)

1. `.git` ← Git repository
2. `pyproject.toml` ← Modern Python
3. `setup.py` ← Traditional Python
4. `setup.cfg` ← Package config
5. `requirements.txt` ← Dependencies
6. `Pipfile` ← Pipenv
7. `poetry.lock` ← Poetry
8. `package.json` ← Node.js (mixed)
9. `Cargo.toml` ← Rust (mixed)

## Confidence Levels

```
100% ████████████ SAFE - No references found
 95% ██████████░  SAFE - Minimal indicators
 90% ████████░░░  REVIEW - Check carefully
<90% ██████░░░░░  KEEP - Likely used
```

## What Gets Flagged as USED

✅ Framework decorators (`@app.get`, `@router.post`)
✅ pytest functions (`test_*`, `@pytest.fixture`)
✅ Magic methods (`__init__`, `__str__`)
✅ Event handlers (`on_*`, `handle_*`)
✅ Exported functions (`__all__`)
✅ Dynamic references (`getattr(obj, "func")`)
✅ LangGraph tools (`@tool`)

## What Gets Flagged as UNUSED

🚨 Never called functions
🚨 Legacy/deprecated code
🚨 Debug functions not in use
🚨 Old implementations

## Configuration

Edit `CONFIG` in `find_unused_functions.py`:

```python
CONFIG = {
    "min_confidence": 90,  # Adjust threshold
    "exclude_patterns": [
        "**/__pycache__/**",
        "**/venv/**",
        # Add your patterns
    ],
}
```

## Workflow

```
1. python find_unused_functions.py
   ↓
2. Review HIGH confidence (95-100%)
   ↓
3. Search project for each function (Ctrl+Shift+F)
   ↓
4. Remove function manually or use helper
   ↓
5. Run tests: pytest
   ↓
6. Commit: git commit -m "Remove unused: func"
   ↓
7. Repeat for next function
```

## Safety Rules

✓ Review manually before removing
✓ Remove one function at a time
✓ Run tests after each removal
✓ Commit incrementally
✓ When in doubt, DON'T remove

## Common Issues

**"No project root markers found"**
→ Add `.git` or `pyproject.toml` to project

**Script analyzing itself**
→ Normal if script is in project
→ Add script dir to `exclude_patterns`

**Wrong root detected**
→ Check for markers in parent dirs
→ Move/remove incorrect markers

## Output Example

```
🚀 Unused Functions Detector
✓ Found project root marker: .git
✓ Project root: /home/user/my-project
📁 Project: /home/user/my-project
📝 Config: Min confidence = 90%

... analysis ...

📊 UNUSED FUNCTIONS REPORT
⚠️  Found 39 potentially unused functions

🔴 HIGH CONFIDENCE (95-100%): 39 functions

📍 old_helper
   File: src/utils/helpers.py:45
   Confidence: 100%
```

## Tips

💡 Start with debug functions (safest)
💡 Remove incrementally
💡 Keep test helpers
💡 Document why you removed functions
💡 Create a cleanup branch

## Files

- `find_unused_functions.py` - Main analysis
- `remove_unused_functions.py` - Interactive helper
- `README.md` - Full documentation
- `CHANGELOG.md` - Version history
- `QUICK_REFERENCE.md` - This file

---

**Print this for quick reference!**

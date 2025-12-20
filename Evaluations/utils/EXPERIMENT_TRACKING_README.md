# Experiment Tracking and Resume Feature

## Overview

The parallel evaluation script now supports tracking experiment executions and resuming incomplete runs. This is useful when evaluations are interrupted due to network issues, rate limits, or other failures.

## Key Features

1. **Execution Tracking**: Each run gets a unique execution ID with timestamp
2. **Experiment Persistence**: Pre-generated experiment names ensure consistency
3. **Resume Capability**: Continue from where you left off
4. **Auto-Skip Completed**: Automatically skips models that completed successfully
5. **JSON Configuration**: All execution data stored in JSON file alongside script

## Configuration

At the top of `001_evaluate_output_correctness_iterate_models_parallel.py`:

```python
# EXECUTION MODE CONFIGURATION
EXECUTION_MODE = "new"  # or "resume"
RESUME_EXECUTION_ID = None  # or specific ID like "exec_2025-12-20_103045_a1b2c3d4"
```

### Modes

- **`"new"`**: Start a fresh evaluation run
  - Creates new execution ID
  - Generates new experiment names with UUID suffixes
  - Evaluates all models in `MODELS_TO_EVALUATE`

- **`"resume"`**: Continue a previous run
  - Uses `RESUME_EXECUTION_ID` or auto-resumes latest
  - Skips models with status "completed"
  - Re-runs models with status "failed" or "in_progress"

## Experiment Naming

Experiments are named with this format:
```
judge_{judge_model_id}__Node_{node_name}__Model_{model_id}-{short_uuid}
```

Example:
```
judge_azureopenai_gpt-4o__Node_format_answer_node__Model_mistral_mistral-large-2512-a1b2c3d4
```

The UUID suffix ensures uniqueness while maintaining readability. This exact name is used in both:
- The tracking JSON file
- LangSmith experiments

## JSON Structure

The configuration file (`001_evaluate_output_correctness_iterate_models_parallel.json`) stores:

```json
{
  "executions": {
    "exec_2025-12-20_103045_a1b2c3d4": {
      "execution_id": "exec_2025-12-20_103045_a1b2c3d4",
      "timestamp": "2025-12-20T10:30:45.123456",
      "status": "partial",
      "config": {
        "node_name": "format_answer_node",
        "dataset_name": "001d_golden_dataset__...",
        "judge_model_id": "azureopenai_gpt-4o",
        "max_concurrency": 1
      },
      "models": {
        "mistral_mistral-large-2512": {
          "status": "completed",
          "experiment_name": "judge_azureopenai_gpt-4o__Node_format_answer_node__Model_mistral_mistral-large-2512-a1b2c3d4",
          "examples_completed": 30,
          "error": null
        }
      }
    }
  },
  "latest_execution_id": "exec_2025-12-20_103045_a1b2c3d4"
}
```

### Status Values

**Execution Status:**
- `"in_progress"`: Currently running
- `"completed"`: All models succeeded
- `"partial"`: Some models failed or incomplete

**Model Status:**
- `"pending"`: Not started yet
- `"in_progress"`: Currently running
- `"completed"`: Successfully finished all examples
- `"failed"`: Encountered error during evaluation

## Usage Examples

### Example 1: New Run

```python
EXECUTION_MODE = "new"
RESUME_EXECUTION_ID = None
```

Run the script:
```bash
python 001_evaluate_output_correctness_iterate_models_parallel.py
```

Output:
```
================================================================================
EVALUATION SUITE - Subprocess Isolation
Mode: NEW
Node: format_answer_node
...
🆕 Creating new execution...
📂 Execution ID: exec_2025-12-20_140530_f7a8b9c0
...
```

### Example 2: Auto-Resume Latest

If a run was interrupted, resume from latest:

```python
EXECUTION_MODE = "resume"
RESUME_EXECUTION_ID = None  # Auto-resume latest
```

Output:
```
Mode: RESUME
...
📂 Resuming execution: exec_2025-12-20_140530_f7a8b9c0
   Timestamp: 2025-12-20T14:05:30.123456
   Status: partial

🔄 Models to evaluate/resume: 3
  • mistral_devstral-2512 [failed, 15 examples]
  • azureopenai_gpt-4o-mini [in_progress, 20 examples]
  • gemini_gemini-3-pro-preview [pending, 0 examples]
```

### Example 3: Resume Specific Execution

To resume a specific execution (not the latest):

```python
EXECUTION_MODE = "resume"
RESUME_EXECUTION_ID = "exec_2025-12-19_083000_a1b2c3d4"
```

## How It Works

### First Run (New Mode)

1. Script creates execution ID: `exec_2025-12-20_140530_f7a8b9c0`
2. For each model, generates experiment name with UUID:
   - `judge_azureopenai_gpt-4o__Node_format_answer_node__Model_mistral_mistral-large-2512-a1b2c3d4`
3. Saves to JSON with status "pending"
4. Passes experiment name to subprocess via environment variable
5. Subprocess uses `experiment=<name>` parameter in `aevaluate`
6. LangSmith creates experiment with that exact name
7. On completion, updates JSON with status "completed"

### Resume Run

1. Loads execution from JSON by ID
2. Filters models where status != "completed"
3. Reuses the same experiment names from JSON
4. Passes to subprocess which uses `experiment=<name>`
5. LangSmith continues adding results to existing experiment
6. Updates JSON with new status

## Benefits

1. **No Duplicate Work**: Completed models are skipped automatically
2. **Network Resilience**: Can stop and resume if connection drops
3. **Predictable Names**: Experiment names are known before execution
4. **Easy Tracking**: All execution history in one JSON file
5. **LangSmith Integration**: Seamlessly continues existing experiments

## Troubleshooting

### "No execution to resume"
```
❌ No execution to resume. Run with EXECUTION_MODE='new' first.
```
**Solution**: Run with `EXECUTION_MODE = "new"` first to create an execution.

### "All models already completed"
```
✅ All models already completed!
```
**Solution**: This is success! All models in that execution finished. Start a new run if needed.

### Finding Execution IDs

Check the JSON file to see all past executions:
```python
from pathlib import Path
import sys

# Add Evaluations to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(Path("001_evaluate_output_correctness_iterate_models_parallel.json"))
for exec_data in tracker.get_all_executions():
    print(f"{exec_data['execution_id']}: {exec_data['status']}")
```

## Advanced: Programmatic Access

```python
from pathlib import Path
import sys

# Add Evaluations to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_tracker import ExperimentTracker

# Load tracker
config_file = Path("001_evaluate_output_correctness_iterate_models_parallel.json")
tracker = ExperimentTracker(config_file)

# Get latest execution
latest_id = tracker.get_latest_execution_id()
execution = tracker.get_execution(latest_id)

# Check model status
for model_id, model_data in execution['models'].items():
    print(f"{model_id}: {model_data['status']} ({model_data['examples_completed']} examples)")

# Get incomplete models
incomplete = tracker.get_incomplete_models(latest_id)
print(f"Models to resume: {list(incomplete.keys())}")
```

## Files Created

- `001_evaluate_output_correctness_iterate_models_parallel.json` - Execution tracking (auto-created)
- `001_evaluate_output_correctness_iterate_models_parallel.json.example` - Example structure (reference only)
- `Evaluations/utils/experiment_tracker.py` - Helper module for tracking (shared utility)

## Notes

- JSON file is created automatically on first run
- Each parallel evaluation script gets its own JSON file (named after the script)
- Experiment names in JSON exactly match those in LangSmith
- The 8-character UUID suffix provides ~4 billion unique combinations
- Old executions remain in JSON for historical tracking

# Resume Capability for Parallel Evaluations

## Overview

This system enables resuming interrupted parallel LangSmith evaluations by continuing the same experiments. When a network drop or other failure interrupts execution, you can resume and:
- **Continue existing experiments** (if they have experiment_name stored)
- **Create fresh experiments** (if no experiment was created yet)

## Key Concepts

### LangSmith Experiment Naming

- **experiment_prefix**: Clean name you provide (e.g., `"judge-gpt-4o__node-sql__model-gpt-4o-mini"`)
- **experiment_name**: Full name LangSmith creates by appending timestamp (e.g., `"judge-gpt-4o__node-sql__model-gpt-4o-mini-2024-12-20-14-30-22"`)
- **experiment_id**: UUID that LangSmith generates (e.g., `"a3f4b8c9-1234-5678-90ab-cdef12345678"`)

### True Resume Capability

✅ **Resume Existing**: If experiment_name exists in JSON → continues that exact experiment in LangSmith  
✅ **Create New**: If no experiment_name (wasn't created yet) → creates fresh experiment with prefix

The system intelligently decides:
1. Check if model has `experiment_name` stored in JSON
2. If yes: pass `experiment=<name>` to LangSmith → **continues same experiment**
3. If no: pass `experiment_prefix=<prefix>` → **creates new experiment**

## How It Works

### Execution Tracking

All execution state is stored in `001_evaluate_output_correctness_iterate_models_parallel.json`:

```json
{
  "executions": {
    "exec_2024-12-20_143022_a3f4b8c9": {
      "timestamp": "2024-12-20T14:30:22",
      "status": "partial",
      "config": {
        "node_name": "sql_execution_db",
        "dataset_name": "output_correctness_eval_dataset",
        "judge_model_id": "openai/gpt-4o",
        "max_concurrency": 5
      },
      "models": {
        "openai/gpt-4o": {
          "status": "completed",
          "experiment_prefix": "judge-openai-gpt-4o__node-sql-execution-db__model-openai-gpt-4o",
          "experiment_name": "judge-openai-gpt-4o__node-sql-execution-db__model-openai-gpt-4o-2024-12-20-143022",
          "experiment_id": "a3f4b8c9-1234-5678-90ab-cdef12345678",
          "examples_completed": 30,
          "error": null
        },
        "anthropic/claude-3-5-sonnet-20241022": {
          "status": "failed",
          "experiment_prefix": "judge-openai-gpt-4o__node-sql-execution-db__model-anthropic-claude-3-5-sonnet-20241022",
          "experiment_name": null,
          "experiment_id": null,
          "examples_completed": 0,
          "error": "ConnectionError: Network timeout"
        }
      }
    }
  },
  "latest_execution_id": "exec_2024-12-20_143022_a3f4b8c9"
}
```

### LangSmith Experiment Flow

#### New Execution (First Run)

1. **Generate clean prefix**:
   ```python
   prefix = "judge-openai-gpt-4o__node-sql-execution-db__model-openai-gpt-4o"
   ```

2. **Subprocess receives prefix**:
   ```python
   EXPERIMENT_PREFIX = os.environ.get("EVAL_EXPERIMENT_PREFIX")
   ```

3. **Create experiment with prefix**:
   ```python
   experiment_results = await aevaluate(
       target_fn,
       data=DATASET_NAME,
       evaluators=[correctness],
       experiment_prefix=prefix,  # LangSmith adds timestamp
   )
   ```

4. **LangSmith generates full name**:
   ```
   judge-openai-gpt-4o__node-sql-execution-db__model-openai-gpt-4o-2024-12-20-143022
   ```

5. **Subprocess captures and outputs metadata**:
   ```python
   experiment_name = experiment_results.experiment_name
   experiment_id = experiment_results.experiment_id
   print(f"EXPERIMENT_NAME: {experiment_name}", file=sys.stderr)
   print(f"EXPERIMENT_ID: {experiment_id}", file=sys.stderr)
   ```

6. **Parent parses stderr and updates JSON**:
   ```python
   tracker.update_model_experiment_metadata(
       execution_id, model_id, experiment_name, experiment_id
   )
   ```

#### Resume Execution

1. **Parent reads incomplete models from JSON**:
   ```python
   incomplete = tracker.get_incomplete_models(execution_id)
   # Returns: {"anthropic/claude-3-5-sonnet": "judge-...-2024-12-20-143022"}  # Has name
   #       or {"openai/gpt-4o": "judge-gpt-4o__node-sql__model-gpt-4o"}  # No name yet
   ```

2. **Subprocess receives experiment_name OR experiment_prefix**:
   ```python
   EXPERIMENT_NAME = os.environ.get("EVAL_EXPERIMENT_NAME")  # If exists
   EXPERIMENT_PREFIX = os.environ.get("EVAL_EXPERIMENT_PREFIX")  # If no name yet
   ```

3. **For existing experiments, filter already-evaluated examples**:
   ```python
   async def get_unevaluated_examples(client, experiment_name, dataset_name):
       # Get all examples from dataset
       all_examples = list(client.list_examples(dataset_name=dataset_name))
       
       # Get existing runs from experiment
       existing_runs = list(client.list_runs(project_name=experiment_name))
       
       # Extract IDs of already-evaluated examples
       evaluated_ids = {run.reference_example_id for run in existing_runs 
                       if run.reference_example_id}
       
       # Return only unevaluated examples
       return [ex for ex in all_examples if ex.id not in evaluated_ids]
   ```

4. **Two scenarios**:

   **A. Resume existing experiment** (has experiment_name):
   ```python
   # Filter to only unevaluated examples
   unevaluated = await get_unevaluated_examples(client, EXPERIMENT_NAME, DATASET_NAME)
   
   experiment_results = await aevaluate(
       target_fn,
       data=unevaluated,  # Only unevaluated examples!
       evaluators=[correctness],
       experiment=EXPERIMENT_NAME,  # Exact name continues experiment
   )
   ```
   → LangSmith **continues adding runs** to existing experiment (only for unevaluated examples)

   **B. Create fresh experiment** (no experiment_name yet):
   ```python
   experiment_results = await aevaluate(
       target_fn,
       data=DATASET_NAME,  # Full dataset for new experiments
       evaluators=[correctness],
       experiment_prefix="judge-gpt-4o__node-sql__model-gpt-4o",
   )
   ```
   → LangSmith **creates new experiment** with timestamp

5. **Capture metadata** (same as new execution)
6. **Update JSON** with experiment details

## Configuration Modes

### Mode: "new"

```python
EXECUTION_MODE = "new"
```

- Creates fresh execution ID
- Generates experiment prefixes for all models
- LangSmith creates brand new experiments
- JSON stores all metadata

### Mode: "resume"

```python
EXECUTION_MODE = "resume"
RESUME_EXECUTION_ID = None  # Auto-detects latest, or specify ID
```

- Uses existing execution ID (latest or specified)
- Gets incomplete models from JSON
- Creates NEW experiments (with same prefix) for incomplete models
- LangSmith generates new timestamps/IDs
- JSON updated with new experiment metadata

## Usage

### First Run

1. **Set mode to "new"**:
   ```python
   EXECUTION_MODE = "new"
   ```

2. **Run script**:
   ```bash
   python 001_evaluate_output_correctness_iterate_models_parallel.py
   ```

3. **If interrupted, check JSON**:
   - Look for latest execution_id
   - Check which models completed vs failed
   - Note experiment metadata for completed models

### Resume After Interruption

1. **Set mode to "resume"**:
   ```python
   EXECUTION_MODE = "resume"
   RESUME_EXECUTION_ID = None  # Uses latest execution
   # or
   RESUME_EXECUTION_ID = "exec_2024-12-20_143022_a3f4b8c9"  # Specific execution
   ```

2. **Run script**:
   ```bash
   python 001_evaluate_output_correctness_iterate_models_parallel.py
   ```

3. **Script intelligently resumes**:
   - **Models with experiment_name**: Continues the SAME LangSmith experiment
   - **Models without experiment_name**: Creates NEW experiments
   - LangSmith adds results to appropriate experiment
   - JSON updated with complete experiment metadata

## Example Scenario

### Initial Run (Interrupted)

```json
{
  "models": {
    "openai/gpt-4o": {
      "status": "completed",
      "experiment_prefix": "judge-gpt-4o__node-sql__model-gpt-4o",
      "experiment_name": "judge-gpt-4o__node-sql__model-gpt-4o-2024-12-20-143022",
      "experiment_id": "abc-123",
      "examples_completed": 30
    },
    "anthropic/claude": {
      "status": "in_progress",
      "experiment_prefix": "judge-gpt-4o__node-sql__model-claude",
      "experiment_name": "judge-gpt-4o__node-sql__model-claude-2024-12-20-143025",
      "experiment_id": "def-456",
      "examples_completed": 15  // Interrupted at 15/30
    },
    "mistral/mixtral": {
      "status": "pending",
      "experiment_prefix": "judge-gpt-4o__node-sql__model-mixtral",
      "experiment_name": null,  // Never started
      "experiment_id": null,
      "examples_completed": 0
    }
  }
}
```

### Resume Mode

```python
EXECUTION_MODE = "resume"
```

**What happens:**
1. **openai/gpt-4o**: Skipped (already completed ✓)
2. **anthropic/claude**: **Continues experiment "...claude-2024-12-20-143025"** (adds remaining 15 examples)
3. **mistral/mixtral**: **Creates NEW experiment** with prefix (full 30 examples)

## Key Files

### [001_evaluate_output_correctness_iterate_models_parallel.py](001_evaluate_output_correctness_iterate_models_parallel.py)
- Main orchestrator
- Manages execution modes
- Runs subprocesses in parallel
- Parses experiment metadata from stderr
- Updates JSON tracker

### [run_single_evaluation.py](run_single_evaluation.py)
- Subprocess worker
- Receives experiment prefix
- Runs single model evaluation  
- Outputs experiment_name and experiment_id to stderr

### [Evaluations/utils/experiment_tracker.py](../../utils/experiment_tracker.py)
- ExperimentTracker class
- JSON persistence
- Execution/model status management
- Experiment metadata storage

## Important Notes

### Example Filtering on Resume

**Critical**: LangSmith does NOT automatically skip already-evaluated examples when continuing an experiment. Without explicit filtering, passing the full dataset with `experiment=<name>` would **re-evaluate all examples**, including ones already completed.

Our implementation prevents this by:

1. **Querying existing runs**: `client.list_runs(project_name=experiment_name)` 
2. **Extracting evaluated IDs**: Each run has `reference_example_id` linking it to a dataset example
3. **Filtering the dataset**: Only pass examples whose IDs aren't in the evaluated set
4. **Passing filtered data**: `aevaluate(data=unevaluated_examples, experiment=name)`

This ensures true resume capability:
- ✅ Already-evaluated examples are **skipped**
- ✅ Only remaining examples are **processed**
- ✅ All runs appear in the **same experiment**
- ✅ No wasted computation or duplicate runs

**Example Output When Resuming**:
```
RESUMING: judge-gpt-4o__node-sql__model-claude-2024-12-20-143025
Dataset: 30 total examples
Already evaluated: 15 examples
Remaining to evaluate: 15 examples
```

### LangSmith Naming Convention

- **Clean prefix without dates**: `"judge-gpt-4o__node-sql__model-gpt-4o-mini"`
- **LangSmith adds timestamp**: `"...-2024-12-20-143022"`
- **Model names may contain dates**: e.g., `"claude-3-5-sonnet-20241022"` (that's the model's version date, not our timestamp)
- **No double dates**: We removed our UUID/timestamp since LangSmith adds its own

### LangSmith Experiment Metadata

- **experiment_name**: Human-readable string with timestamp
- **experiment_id**: UUID for programmatic access
- **ls_experiment_id**: System parameter (UUID), automatically added to all runs in that experiment
- Both name and ID can be used to reference experiments in LangSmith API

### Subprocess Communication

- **stdout**: Script success/failure signal (`"SUCCESS"`)
- **stderr**: Experiment metadata output
  - `"EXPERIMENT_PREFIX: <prefix>"`
  - `"EXPERIMENT_NAME: <full_name>"`
  - `"EXPERIMENT_ID: <uuid>"`
- **Return code**: 0 for success, non-zero for failure

### Error Handling

- **Network interruptions**: 
  - If experiment was created: Resume continues that experiment
  - If experiment wasn't created: Creates fresh experiment
- **Subprocess crashes**: Error stored in JSON, other models unaffected
- **JSON corruption**: Manual fix required, or start new execution

## Troubleshooting

### Double Dates in Experiment Names

**Fixed**: We now use clean prefixes. LangSmith adds the timestamp, so you won't see double dates like before.

### Models Re-running Despite "completed" Status

**Cause**: EXECUTION_MODE still set to "new"

**Fix**: Change to `EXECUTION_MODE = "resume"`

### Can I Continue An Interrupted Experiment?

**Yes!** If the experiment was created before interruption (has `experiment_name` in JSON), resume mode will continue that exact experiment.

**Important**: LangSmith does NOT automatically filter already-evaluated examples when continuing an experiment. Our implementation adds intelligent filtering:

1. **Fetches existing runs** from the experiment using `client.list_runs(project_name=experiment_name)`
2. **Extracts evaluated example IDs** from `run.reference_example_id`
3. **Filters dataset** to only include unevaluated examples
4. **Passes filtered examples** to `aevaluate()` with `experiment=<name>` parameter

This prevents re-evaluation of already-completed examples and ensures true resume capability without wasted work.

If the experiment wasn't created yet (no `experiment_name`), a fresh experiment will be created with the same prefix and will evaluate all examples in the dataset.

### JSON Not Updating

**Cause**: Multiple concurrent writes, or permission issues

**Fix**: Ensure single instance running, check file permissions

## LangSmith Documentation Reference

From official LangSmith docs:

- `experiment_prefix`: Parameter for creating NEW experiments - LangSmith appends timestamp/UUID
- `experiment`: Parameter for continuing EXISTING experiments - pass experiment name or ID
- `experiment_name`: Full name returned in `ExperimentResults`  
- `experiment_id` / `ls_experiment_id`: UUID returned in `ExperimentResults`
- **Resume capability**: Pass `experiment=<name_or_id>` with target function to add more runs to existing experiment
- **Add evaluators**: Pass `experiment=<name_or_id>` WITHOUT target function to add evaluators to completed experiment

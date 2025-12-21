# Example Filtering Implementation for Resume Capability

## Problem Discovered

After studying the LangSmith documentation (especially `langsmith_evaluation_reference.txt` and `langsmith_async_client_reference.txt`), I discovered a critical issue:

**LangSmith does NOT automatically filter already-evaluated examples when continuing an experiment.**

When you pass `experiment=<name_or_id>` to continue an existing experiment, LangSmith simply adds new runs to that experiment. If you pass the full dataset again, it will **re-evaluate ALL examples**, including those already completed.

## Solution Implemented

Added intelligent filtering in [run_single_evaluation.py](run_single_evaluation.py):

### New Function: `get_unevaluated_examples()`

```python
async def get_unevaluated_examples(client: Client, experiment_name: str, dataset_name: str):
    """Get examples that haven't been evaluated yet in the experiment.
    
    Args:
        client: LangSmith client
        experiment_name: Name or ID of the experiment
        dataset_name: Name of the dataset
        
    Returns:
        List of Example objects that haven't been evaluated
    """
    # 1. Get all examples from the dataset
    all_examples = list(client.list_examples(dataset_name=dataset_name))
    
    # 2. Get all runs from the experiment
    try:
        existing_runs = list(client.list_runs(project_name=experiment_name))
    except Exception as e:
        print(f"Could not fetch existing runs: {e}", file=sys.stderr, flush=True)
        # If we can't fetch runs, assume no examples evaluated yet
        return all_examples
    
    # 3. Get set of example IDs that have already been evaluated
    evaluated_example_ids = {run.reference_example_id for run in existing_runs 
                            if run.reference_example_id}
    
    # 4. Filter to only unevaluated examples
    unevaluated_examples = [ex for ex in all_examples 
                           if ex.id not in evaluated_example_ids]
    
    # 5. Log statistics
    print(f"Dataset: {len(all_examples)} total examples", file=sys.stderr, flush=True)
    print(f"Already evaluated: {len(evaluated_example_ids)} examples", file=sys.stderr, flush=True)
    print(f"Remaining to evaluate: {len(unevaluated_examples)} examples", file=sys.stderr, flush=True)
    
    return unevaluated_examples
```

### Modified `run_evaluation()` Function

**Resume Mode** (when `EXPERIMENT_NAME` is set):

```python
if EXPERIMENT_NAME:
    # RESUME MODE: Continue existing experiment, filtering already-evaluated examples
    print(f"RESUMING: {EXPERIMENT_NAME}", file=sys.stderr, flush=True)
    
    # Get only unevaluated examples
    unevaluated_examples = await get_unevaluated_examples(client, EXPERIMENT_NAME, DATASET_NAME)
    
    if not unevaluated_examples:
        print("All examples already evaluated!", file=sys.stderr, flush=True)
        # Still need to return experiment metadata
        existing_project = client.read_project(project_name=EXPERIMENT_NAME)
        print(f"EXPERIMENT_NAME: {existing_project.name}", file=sys.stderr, flush=True)
        print(f"EXPERIMENT_ID: {existing_project.id}", file=sys.stderr, flush=True)
        return
    
    experiment_results = await aevaluate(
        target_fn,
        data=unevaluated_examples,  # Only pass unevaluated examples!
        evaluators=[correctness],
        max_concurrency=MAX_CONCURRENCY,
        experiment=EXPERIMENT_NAME,  # Use exact name/ID to continue
    )
```

**New Mode** (when `EXPERIMENT_PREFIX` is set):

```python
else:
    # NEW MODE: Create new experiment with prefix
    if not EXPERIMENT_PREFIX:
        from Evaluations.utils.experiment_tracker import generate_experiment_prefix
        experiment_prefix = generate_experiment_prefix(JUDGE_MODEL_ID, NODE_NAME, MODEL_ID)
    else:
        experiment_prefix = EXPERIMENT_PREFIX
    
    print(f"CREATING: {experiment_prefix}", file=sys.stderr, flush=True)
    experiment_results = await aevaluate(
        target_fn,
        data=DATASET_NAME,  # Use full dataset for new experiments
        evaluators=[correctness],
        max_concurrency=MAX_CONCURRENCY,
        experiment_prefix=experiment_prefix,  # LangSmith appends timestamp/UUID
    )
```

## How It Works

### Step-by-Step Flow

1. **Subprocess started in resume mode** with `EVAL_EXPERIMENT_NAME` environment variable set

2. **Fetch all dataset examples**:
   ```python
   all_examples = list(client.list_examples(dataset_name="my_dataset"))
   # Result: [Example(id=uuid1, ...), Example(id=uuid2, ...), ...]
   ```

3. **Fetch existing runs from experiment**:
   ```python
   existing_runs = list(client.list_runs(project_name="my_experiment"))
   # Result: [Run(id=run1, reference_example_id=uuid1), ...]
   ```

4. **Extract evaluated example IDs**:
   ```python
   evaluated_ids = {uuid1, uuid5, uuid7, ...}  # IDs from existing runs
   ```

5. **Filter to unevaluated examples**:
   ```python
   unevaluated = [ex for ex in all_examples if ex.id not in evaluated_ids]
   # Only examples whose IDs are NOT in evaluated_ids
   ```

6. **Pass filtered examples to aevaluate**:
   ```python
   await aevaluate(
       target_fn,
       data=unevaluated,  # Only 15 examples instead of 30!
       experiment=experiment_name,  # Continues same experiment
   )
   ```

### Example Scenario

**Dataset**: 30 examples (IDs: `ex1`, `ex2`, ..., `ex30`)

**Initial Run**: Evaluated 15 examples (`ex1` - `ex15`) before network interruption

**Existing Experiment**:
- Name: `"judge-gpt-4o__node-sql__model-claude-2024-12-20-143025"`
- Runs: 15 runs with `reference_example_id` in {`ex1`, ..., `ex15`}

**Resume Execution**:
1. Fetch all 30 examples from dataset
2. Fetch 15 existing runs from experiment
3. Extract evaluated IDs: {`ex1`, ..., `ex15`}
4. Filter examples: [`ex16`, `ex17`, ..., `ex30`] (15 remaining)
5. Pass only 15 unevaluated examples to `aevaluate()`

**Result**:
- Only evaluates `ex16` - `ex30` (15 new examples)
- All 30 runs appear in same experiment
- No duplicate work
- True resume capability achieved

## Benefits

✅ **No Wasted Computation**: Already-evaluated examples are never re-run  
✅ **Unified Experiment**: All runs (old + new) appear in the same LangSmith experiment  
✅ **Accurate Progress Tracking**: Console shows exactly how many examples remain  
✅ **Graceful Completion**: If all examples done, returns metadata without running evaluations  
✅ **Error Resilience**: If can't fetch runs, assumes nothing evaluated (safe fallback)

## LangSmith API Details

### Client Methods Used

1. **`client.list_examples(dataset_name=str)`**:
   - Returns: `Iterator[Example]`
   - Each Example has: `id`, `inputs`, `outputs`, `dataset_id`
   - Purpose: Get all examples in dataset

2. **`client.list_runs(project_name=str)`**:
   - Returns: `Iterator[Run]`  
   - Each Run has: `id`, `reference_example_id`, `inputs`, `outputs`, `error`
   - Purpose: Get all runs in experiment (project)
   - Note: `reference_example_id` links run back to dataset example

3. **`client.read_project(project_name=str)`**:
   - Returns: `TracerSession` (project/experiment object)
   - Has: `name`, `id`, `created_at`, `metadata`
   - Purpose: Get experiment metadata when all examples already evaluated

### aevaluate() Parameters

- **`data=<dataset_name>`**: LangSmith fetches all examples from named dataset
- **`data=<list[Example]>`**: Use specific list of Example objects (our filtered list!)
- **`experiment=<name_or_id>`**: Continue existing experiment by name or UUID
- **`experiment_prefix=<prefix>`**: Create new experiment (LangSmith adds timestamp/UUID)

## Testing Recommendations

### Test Case 1: Fresh Start
- Run with new experiment prefix
- Interrupt after 10/30 examples
- Verify 10 runs in LangSmith experiment

### Test Case 2: Resume Partial
- Resume interrupted experiment
- Should show: "Already evaluated: 10 examples, Remaining: 20 examples"
- Verify 30 total runs in same experiment (10 old + 20 new)

### Test Case 3: Resume Complete
- Resume already-completed experiment
- Should show: "All examples already evaluated!"
- Should NOT create new runs

### Test Case 4: Network Failure During Fetch
- Simulate failure when fetching existing runs
- Should log error and evaluate all examples (safe fallback)
- Should continue adding to experiment

## Future Enhancements

Consider adding:

1. **Incremental Progress Updates**: Track progress in JSON after each example
2. **Example-Level Retry**: Retry individual failed examples rather than whole model
3. **Parallel Example Filtering**: If dataset is huge, parallelize the filtering
4. **Cache Evaluated IDs**: Store evaluated IDs in JSON to avoid repeated API calls

## Related Documentation

- [RESUME_CAPABILITY.md](RESUME_CAPABILITY.md): Complete resume system documentation
- [langsmith_evaluation_reference.txt](../../langsmith_evaluation_reference.txt): Official LangSmith evaluation API docs
- [langsmith_async_client_reference.txt](../../langsmith_async_client_reference.txt): Official LangSmith async client API docs

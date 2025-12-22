# Extract Pairwise Comparison Results from LangSmith

## Overview

This script (`extract_pairwise_results_from_langsmith.py`) extracts pairwise comparison results directly from LangSmith by querying the feedback data from comparative evaluation runs. It generates a CSV file with the correct win/loss/tie statistics that match what you see in the LangSmith UI.

## Problem It Solves

The original pairwise comparison script (`pairwise_compare_more_experiments.py`) generates and runs pairwise comparisons, but there may be cases where:
- You want to re-extract results without re-running expensive evaluations
- You need to verify the results match what's shown in LangSmith UI
- The original CSV has incorrect aggregations or missing data

This extraction script reads the **actual feedback data** stored in LangSmith and produces a corrected CSV.

## How It Works

1. **Finds Pairwise Experiments**: Queries LangSmith to find all comparative/pairwise experiments linked to your dataset
2. **Extracts Feedback**: For each pairwise experiment, reads the feedback scores from all runs
3. **Aggregates Scores**: Counts how many times each experiment won based on preference scores
4. **Generates CSV**: Creates a CSV file with the same format as the original but with correct numbers

## Configuration

Edit the script and set:

```python
# Specify dataset by name (recommended)
DATASET_NAME = "001d_golden_dataset__output_correctness__simple_QA_from_SQL__manually_chosen_questions"

# OR specify by UUID
DATASET_ID = None  # Set to your dataset UUID if preferred
```

### How to Find Your Dataset

**Option 1: By Name**
- Open LangSmith
- Go to "Datasets & Experiments" → "Datasets"
- Copy the exact dataset name

**Option 2: By UUID**
- Open your dataset in LangSmith
- Copy the UUID from the URL: `https://smith.langchain.com/o/.../datasets/<UUID>`

## Usage

1. **Configure the script** with your dataset name or ID

2. **Make sure your Python environment is activated:**
   ```bash
   .venv\Scripts\activate
   ```

3. **Run the script:**
   ```bash
   python Evaluations\what_is_evaluated\output_pairwise_comparison\extract_pairwise_results_from_langsmith.py
   ```

4. **Check the output:**
   - The script will print progress and statistics
   - A CSV file will be created: `pairwise_results_extracted_from_langsmith_YYYYMMDD_HHMMSS.csv`

## Output Format

The generated CSV has these columns:

| Column | Description |
|--------|-------------|
| `pair_number` | Sequential pair number (1, 2, 3, ...) |
| `pairwise_experiment_id` | LangSmith experiment ID |
| `pairwise_experiment_name` | Name of the pairwise experiment |
| `experiment_a_name` | Name of first experiment being compared |
| `experiment_b_name` | Name of second experiment being compared |
| `winner` | "A", "B", or "TIE" |
| `a_wins` | Number of times experiment A won |
| `b_wins` | Number of times experiment B won |
| `ties` | Number of tied comparisons |
| `total_comparisons` | Total number of examples compared |
| `a_preference_score` | Percentage preference for A (%) |
| `b_preference_score` | Percentage preference for B (%) |

## Matching LangSmith UI

The scores in the CSV should match what you see in:
- LangSmith UI → Datasets → [Your Dataset] → "Pairwise Experiments" tab
- The preference scores shown for each experiment pair

If you see "4, 25" in the UI for a pair, the CSV should show:
- `a_wins: 4`
- `b_wins: 25`
- `total_comparisons: 29` (or 30 if there were ties)

## Troubleshooting

### No pairwise experiments found
- **Cause**: No comparative evaluations have been run for this dataset
- **Solution**: Run `pairwise_compare_more_experiments.py` first to create pairwise comparisons

### Wrong dataset
- **Cause**: Dataset name or ID doesn't match
- **Solution**: Double-check the exact dataset name (case-sensitive) or UUID in LangSmith

### Missing feedback
- **Cause**: Pairwise evaluations ran but feedback wasn't recorded
- **Solution**: Re-run the evaluations with the correct evaluator configuration

### Scores don't match UI
- **Cause**: The extraction logic may need adjustment based on your feedback structure
- **Solution**: Enable `DEBUG = True` in the script to see detailed extraction info

## Debug Mode

Set `DEBUG = True` in the script to see:
- Detailed extraction process
- Feedback scores for individual comparisons
- Why certain runs were included/excluded

## Related Scripts

- `pairwise_compare_more_experiments.py` - Generates and runs pairwise comparisons
- `pairwise_comparison_analysis_and_score_based_ranking.py` - Analyzes results and ranks models
- `pairwise_comparison_analysis_bradley_terry.py` - Advanced statistical ranking (Bradley-Terry model)

## API Used

This script uses the LangSmith Python SDK:
- `client.read_dataset()` - Get dataset info
- `client.list_examples()` - List dataset examples
- `client.list_runs()` - Find runs linked to dataset
- `client.list_feedback()` - Extract feedback scores
- `client.read_project()` - Get experiment metadata

## Requirements

- `langsmith` Python package
- `python-dotenv` for environment variables
- LangSmith API key configured in `.env`

## Environment Variables

Make sure your `.env` file contains:
```
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=your_project_name
```

## Example Output

```
===============================================================================
📥 EXTRACT PAIRWISE COMPARISON RESULTS FROM LANGSMITH
===============================================================================

✓ Found dataset: 001d_golden_dataset__output_correctness__simple_QA_from_SQL__manually_chosen_questions (abc-123-def)

📦 Dataset ID: abc-123-def

🔍 Searching for pairwise experiments linked to dataset abc-123-def...
   Found 45 runs referencing dataset examples
   ✓ Found pairwise experiment: pairwise_comparison_pair_1
   ✓ Found pairwise experiment: pairwise_comparison_pair_2
   ...

✓ Found 88 pairwise experiments

📊 Processing pairwise experiment: pairwise_comparison_pair_1
   Found 30 root runs
   Results: A=4, B=25, Total=30

   ✓ Pair 1: A=4, B=25, Ties=1, Total=30

...

💾 Results saved to CSV: pairwise_results_extracted_from_langsmith_20251222_115030.csv

===============================================================================
✅ EXTRACTION COMPLETE
===============================================================================

📊 Extracted 88 pairwise comparison results
💾 Results saved to: pairwise_results_extracted_from_langsmith_20251222_115030.csv
```

## Notes

- This script **does not run any evaluations** - it only extracts existing results
- The extraction is much faster than re-running pairwise comparisons
- You can run this script multiple times without affecting your data
- The script respects LangSmith API rate limits with built-in retry logic

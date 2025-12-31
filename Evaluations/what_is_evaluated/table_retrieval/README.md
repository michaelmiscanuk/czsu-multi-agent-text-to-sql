# Table Retrieval Evaluation Scripts

This folder contains 3 evaluation scripts that test different retrieval methods for table/selection retrieval from ChromaDB.

## Scripts Overview

### 001a_evaluate_semantic_search.py
**Method**: Simple semantic search using embeddings only
- Uses ChromaDB's `.query()` method with Azure OpenAI embeddings
- No keyword matching (BM25)
- Pure similarity-based retrieval

**Configuration**:
- `TOP_N = 3` - Number of top results to check in top_n evaluator
- `N_RESULTS = 20` - Number of results to retrieve

### 001b_evaluate_hybrid_search.py
**Method**: Hybrid search (Semantic + BM25)
- Combines semantic similarity with keyword matching
- Uses the `hybrid_search()` function from `metadata/create_and_load_chromadb__04.py`
- Weighted combination: semantic (0.85) + BM25 (0.15)

**Configuration**:
- `TOP_N = 3` - Number of top results to check in top_n evaluator
- `N_RESULTS = 20` - Number of results to retrieve

### 001c_evaluate_rerank.py
**Method**: Full pipeline (Hybrid search + Cohere reranking)
- First performs hybrid search
- Then reranks results using Cohere's multilingual rerank model
- Most sophisticated method with best expected performance

**Configuration**:
- `TOP_N = 3` - Number of top results to check in top_n evaluator
- `N_RESULTS = 20` - Number of results from hybrid search
- `TOP_K_RERANK = 20` - Number of results to rerank

## Evaluators

Each script uses 2 evaluators:

### 1. top_1_correct
Checks if the correct table code is ranked #1 (first position) in the results.
- Returns `True` if the expected table is the top result
- Returns `False` otherwise

### 2. table_in_top_n
Checks if the correct table code appears anywhere in the top N results.
- Configurable N (default: 3)
- Returns `True` if the expected table is in top N
- Returns `False` otherwise

## Dataset

All scripts use the same golden dataset:
- **Dataset name**: `langsmith_create_golden_dataset_table_retrieval`
- **Content**: 30 question-answer pairs
- **Format**: Each example has:
  - Input: `{"question": "What was the average consumer price...?"}`
  - Output: `{"answer": "CEN0101DT01"}` (table code)

## ChromaDB Configuration

All scripts automatically use **Cloud ChromaDB** by default through the `chromadb_client_factory`:
- Reads `CHROMA_USE_CLOUD` from `.env` (defaults to `"false"`)
- If you want to ensure cloud usage, set: `CHROMA_USE_CLOUD=true` in your `.env` file
- The factory handles cloud/local switching automatically

## Running the Scripts

Each script can be run independently:

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Run semantic search evaluation
python Evaluations\what_is_evaluated\table_retrieval\001a_evaluate_semantic_search.py

# Run hybrid search evaluation  
python Evaluations\what_is_evaluated\table_retrieval\001b_evaluate_hybrid_search.py

# Run rerank evaluation
python Evaluations\what_is_evaluated\table_retrieval\001c_evaluate_rerank.py
```

## Customization

To adjust the TOP_N parameter, edit the configuration section in each script:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_N = 3  # Change this value to test different top-N thresholds
```

## Expected Results

The scripts will:
1. Load the golden dataset from LangSmith
2. Run each question through the retrieval method
3. Evaluate with both evaluators
4. Create an experiment in LangSmith with results
5. Print summary statistics

Output includes:
- Per-question evaluation results
- Overall accuracy for top_1_correct
- Overall accuracy for table_in_top_n
- Experiment link in LangSmith for detailed analysis

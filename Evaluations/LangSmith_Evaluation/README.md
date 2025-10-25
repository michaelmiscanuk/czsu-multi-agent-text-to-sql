# LangSmith Evaluation Scripts

This directory contains evaluation scripts for testing different components of the selection retrieval system.

## Scripts Overview

### 1. `langsmith_evaluate_hybrid_search_only.py`
**Purpose**: Tests only the hybrid search functionality (retrieve_similar_selections_hybrid_search_node)

**What it evaluates**:
- Hybrid search combining semantic similarity and BM25 retrieval
- Document retrieval without reranking
- Returns raw Document objects with metadata

**Key evaluators**:
- `selection_correct_hybrid`: Checks if top-1 hybrid result matches expected selection
- `selection_in_top_n_hybrid`: Checks if expected selection is in top-N hybrid results

**Use case**: Testing the effectiveness of the base retrieval system before reranking

### 2. `langsmith_evaluate_selection_retrieval.py`
**Purpose**: Tests the complete pipeline (hybrid search + rerank)

**What it evaluates**:
- Full two-node sequence: `retrieve_similar_selections_hybrid_search_node` → `rerank_table_descriptions_node`
- Hybrid search followed by Cohere rerank model
- Returns selection codes with rerank scores

**Key evaluators**:
- `selection_correct`: Checks if top-1 reranked result matches expected selection
- `selection_in_top_n`: Checks if expected selection is in top-N reranked results

**Use case**: Testing the complete retrieval and reranking pipeline

## Experiment Prefixes

- Hybrid search only: `"hybrid-search-only"`
- Full pipeline: `"full-pipeline-hybrid-rerank"`

## Usage

Run the hybrid search evaluation:
```bash
python Evaluations/LangSmith_Evaluation/langsmith_evaluate_hybrid_search_only.py
```

Run the full pipeline evaluation:
```bash
python Evaluations/LangSmith_Evaluation/langsmith_evaluate_selection_retrieval.py
```

## Key Differences

| Aspect | Hybrid Search Only | Full Pipeline |
|--------|-------------------|---------------|
| **Nodes tested** | `retrieve_similar_selections_hybrid_search_node` | `retrieve_similar_selections_hybrid_search_node` + `rerank_table_descriptions_node` |
| **Output format** | Document objects with metadata | (selection_code, score) tuples |
| **Scoring method** | Hybrid search ranking | Cohere rerank scores |
| **Debug focus** | Retrieval effectiveness | End-to-end pipeline performance |
| **Use case** | Component testing | Production pipeline testing | 
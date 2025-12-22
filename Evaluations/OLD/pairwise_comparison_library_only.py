"""
Pairwise Comparison Analysis - Library-Based Methods Only

This script uses ONLY external libraries for ranking (NO manual calculations).

Methods implemented using choix library:
1. Bradley-Terry Model - Maximum likelihood via MM algorithm
2. Luce Spectral Ranking - Eigenvector-based method
3. Rank Centrality - PageRank-like algorithm

Required:
    pip install choix numpy
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import choix

    CHOIX_AVAILABLE = True
except ImportError:
    CHOIX_AVAILABLE = False
    print("❌ ERROR: 'choix' library is required!")
    print("   Install with: pip install choix")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = Path(
    "Evaluations/what_is_evaluated/output_pairwise_comparison/pairwise_compare_more_experiments_20251222_110500.csv"
)

# ============================================================================
# DATA LOADING
# ============================================================================


def load_pairwise_data(csv_path: Path) -> Tuple[Dict, List[str]]:
    """Load pairwise comparison data from CSV."""
    match_results = {}
    models_set = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            exp_a = row["experiment_a_name"]
            exp_b = row["experiment_b_name"]
            a_wins = int(row["a_wins"])
            b_wins = int(row["b_wins"])
            total = int(row["total_comparisons"])
            ties = total - a_wins - b_wins

            models_set.add(exp_a)
            models_set.add(exp_b)

            match_results[(exp_a, exp_b)] = {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
            }

    model_names = sorted(list(models_set))
    return match_results, model_names


def prepare_choix_data(
    match_results: Dict, model_names: List[str]
) -> List[Tuple[int, int]]:
    """Convert match results to choix format (winner-loser pairs)."""
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}
    data = []

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        # Add A wins
        for _ in range(result["a_wins"]):
            data.append((idx_a, idx_b))

        # Add B wins
        for _ in range(result["b_wins"]):
            data.append((idx_b, idx_a))

        # Add ties (split evenly)
        ties = result["ties"]
        if ties > 0:
            for _ in range(ties // 2):
                data.append((idx_a, idx_b))
                data.append((idx_b, idx_a))

    return data


# ============================================================================
# RANKING METHODS (using choix library)
# ============================================================================


def bradley_terry_model(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """
    Bradley-Terry Model using choix.ilsr_pairwise

    Maximum likelihood estimation via Minorization-Maximization algorithm.
    Accounts for opponent strength in pairwise comparisons.
    """
    data = prepare_choix_data(match_results, model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Compute parameters using iterative algorithm
    params = choix.ilsr_pairwise(len(model_names), data, alpha=0.01)

    # Convert to ratings (exp transform for interpretability)
    ratings = {model: np.exp(params[model_to_idx[model]]) for model in model_names}

    return ratings


def luce_spectral_ranking(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """
    Luce Spectral Ranking using choix.lsr_pairwise

    Eigenvector-based method that computes the principal eigenvector
    of the pairwise comparison matrix.
    """
    data = prepare_choix_data(match_results, model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Compute spectral ranking
    params = choix.lsr_pairwise(len(model_names), data, alpha=0.01)

    ratings = {model: params[model_to_idx[model]] for model in model_names}

    return ratings


def rank_centrality_method(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """
    Rank Centrality using choix.rank_centrality

    PageRank-like algorithm that interprets pairwise comparisons
    as transitions in a Markov chain.
    """
    data = prepare_choix_data(match_results, model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Compute rank centrality
    params = choix.rank_centrality(len(model_names), data, alpha=0.01)

    ratings = {model: params[model_to_idx[model]] for model in model_names}

    return ratings


# ============================================================================
# OUTPUT
# ============================================================================


def print_ranking_table(method_name: str, ratings: Dict[str, float]) -> Dict[str, int]:
    """Print clean ranking table and return rank mapping."""
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Column widths
    width_line = 130
    width_rank = 6
    width_model = 105
    width_rating = 18

    print("\n" + "=" * width_line)
    print(f"🏆 {method_name}")
    print("=" * width_line)

    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'Rating':<{width_rating}}"
    )
    print(header)
    print("-" * width_line)

    # Store ranks for each model
    ranks = {}
    for rank, (model, rating) in enumerate(ranking, 1):
        row = (
            f"{rank:<{width_rank}} "
            f"{model:<{width_model}} "
            f"{rating:<{width_rating}.6f}"
        )
        print(row)
        ranks[model] = rank

    print("=" * width_line)

    return ranks


def print_average_ranking_table(all_ranks: List[Dict[str, int]]):
    """Print table with average ranks across all methods."""
    # Calculate average rank for each model
    model_names = all_ranks[0].keys()
    avg_ranks = {}

    for model in model_names:
        ranks = [method_ranks[model] for method_ranks in all_ranks]
        avg_ranks[model] = sum(ranks) / len(ranks)

    # Sort by average rank (ascending - lower is better)
    ranking = sorted(avg_ranks.items(), key=lambda x: x[1])

    # Column widths
    width_line = 130
    width_rank = 6
    width_model = 95
    width_avg_rank = 12
    width_ranks = 16

    print("\n" + "=" * width_line)
    print("🏆 AVERAGE RANKING ACROSS ALL METHODS")
    print("=" * width_line)

    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'Avg Rank':<{width_avg_rank}} "
        f"{'Individual Ranks':<{width_ranks}}"
    )
    print(header)
    print("-" * width_line)

    for rank, (model, avg_rank) in enumerate(ranking, 1):
        # Get individual ranks from each method
        individual_ranks = [str(method_ranks[model]) for method_ranks in all_ranks]
        ranks_str = ", ".join(individual_ranks)

        row = (
            f"{rank:<{width_rank}} "
            f"{model:<{width_model}} "
            f"{avg_rank:<{width_avg_rank}.2f} "
            f"{ranks_str:<{width_ranks}}"
        )
        print(row)

    print("=" * width_line)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all library-based ranking methods."""

    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    if not CHOIX_AVAILABLE:
        print("❌ ERROR: 'choix' library required but not installed!")
        print("   Install with: pip install choix")
        return

    print(f"\n📂 Reading pairwise comparison results:")
    print(f"   {CSV_PATH}")

    # Load data
    match_results, model_names = load_pairwise_data(CSV_PATH)

    print(f"\n📊 Data loaded:")
    print(f"   Models: {len(model_names)}")
    print(f"   Matchups: {len(match_results)}")

    print("\n" + "=" * 130)
    print("🎯 LIBRARY-BASED RANKING METHODS (choix)")
    print("=" * 130)

    # Store ranks from each method
    all_ranks = []

    # Method 1: Bradley-Terry
    print("\n[1/3] Bradley-Terry Model...")
    ratings_1 = bradley_terry_model(match_results, model_names)
    ranks_1 = print_ranking_table(
        "METHOD 1: BRADLEY-TERRY MODEL (choix.ilsr_pairwise)", ratings_1
    )
    all_ranks.append(ranks_1)

    # Method 2: Luce Spectral
    print("\n[2/3] Luce Spectral Ranking...")
    ratings_2 = luce_spectral_ranking(match_results, model_names)
    ranks_2 = print_ranking_table(
        "METHOD 2: LUCE SPECTRAL RANKING (choix.lsr_pairwise)", ratings_2
    )
    all_ranks.append(ranks_2)

    # Method 3: Rank Centrality
    print("\n[3/3] Rank Centrality...")
    ratings_3 = rank_centrality_method(match_results, model_names)
    ranks_3 = print_ranking_table(
        "METHOD 3: RANK CENTRALITY (choix.rank_centrality)", ratings_3
    )
    all_ranks.append(ranks_3)

    # Display average ranking
    print("\n[SUMMARY] Computing Average Ranking...")
    print_average_ranking_table(all_ranks)

    print("\n" + "=" * 130)
    print("✅ All rankings computed successfully!")
    print("=" * 130)

    print("\n📚 ABOUT THESE METHODS:")
    print("   All use the 'choix' library - professional implementations of")
    print("   choice models based on Luce's axiom.")
    print()
    print("   • Bradley-Terry: ML estimation via iterative algorithm")
    print("     - Ratings around 1.0 (scaled exponentially)")
    print("     - Higher values = stronger models")
    print()
    print("   • Luce Spectral Ranking: Principal eigenvector computation")
    print("     - Raw log-scale ratings (can be positive or negative)")
    print("     - Centered around 0, relative differences matter")
    print()
    print("   • Rank Centrality: Markov chain stationary distribution")
    print("     - Also log-scale, very similar to Luce Spectral")
    print("     - Both are spectral methods, often give nearly identical results")
    print()
    print("   ⚠️  NOTE: Luce Spectral & Rank Centrality produce similar rankings")
    print("       because both are spectral algorithms on the same comparison matrix.")
    print("       The minor differences come from slight algorithmic variations.")
    print()
    print("   💡 Rating scales differ by method - compare RANKS, not raw values!")
    print()


if __name__ == "__main__":
    main()

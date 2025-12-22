"""
Pairwise Comparison Analysis using Bradley-Terry Model

The Bradley-Terry model estimates the "strength" of each model by analyzing
pairwise comparison results. Unlike simple win counting, this method:
- Accounts for the strength of opponents
- Uses an iterative algorithm to find optimal ratings
- Provides more nuanced rankings

The algorithm works by:
1. Starting with equal ratings for all models (e.g., 1.0)
2. Iteratively updating each rating based on wins and losses
3. Continuing until ratings converge (stabilize)

Formula for each iteration:
    rating[i] = total_wins[i] / sum_of_all_opponents_strength
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = Path(
    "Evaluations\what_is_evaluated\output_pairwise_comparison\pairwise_compare_more_experiments_20251222_110500.csv"
)

# Bradley-Terry algorithm parameters
MAX_ITERATIONS = 100  # Maximum number of iterations
CONVERGENCE_THRESHOLD = 0.0001  # Stop when changes are smaller than this

# ============================================================================
# BRADLEY-TERRY ALGORITHM
# ============================================================================


def load_pairwise_results(csv_path: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load pairwise comparison results from CSV.

    Returns:
        Dictionary mapping each model to its opponents and match results:
        {
            "model_a": {
                "model_b": {"wins": 5, "losses": 3, "ties": 2},
                "model_c": {"wins": 7, "losses": 1, "ties": 2},
            },
            ...
        }
    """
    results = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            exp_a = row["experiment_a_name"]
            exp_b = row["experiment_b_name"]
            a_wins = int(row["a_wins"])
            b_wins = int(row["b_wins"])
            total = int(row["total_comparisons"])
            ties = total - a_wins - b_wins

            # Store results for both models
            results[exp_a][exp_b] = {"wins": a_wins, "losses": b_wins, "ties": ties}
            results[exp_b][exp_a] = {"wins": b_wins, "losses": a_wins, "ties": ties}

    return dict(results)


def calculate_bradley_terry_ratings(
    results: Dict[str, Dict[str, Dict]],
) -> Dict[str, float]:
    """
    Calculate Bradley-Terry ratings using iterative algorithm.

    The algorithm:
    1. Initialize all ratings to 1.0
    2. For each model, calculate new rating:
       new_rating = total_wins / sum(comparisons / (my_rating + opponent_rating))
    3. Normalize ratings so they sum to number of models
    4. Repeat until convergence

    Args:
        results: Pairwise comparison results

    Returns:
        Dictionary mapping model names to their Bradley-Terry ratings
    """
    # Step 1: Initialize ratings (start with 1.0 for everyone)
    models = list(results.keys())
    ratings = {model: 1.0 for model in models}

    print(f"\n🔄 Running Bradley-Terry algorithm...")
    print(f"   Models: {len(models)}")
    print(f"   Starting ratings: all = 1.0")
    print(f"   Convergence threshold: {CONVERGENCE_THRESHOLD}")

    # Step 2: Iterate until convergence
    for iteration in range(MAX_ITERATIONS):
        old_ratings = ratings.copy()
        new_ratings = {}

        # Update rating for each model
        for model in models:
            # Count total wins (ties count as 0.5 wins)
            total_wins = 0.0
            for opponent, match in results[model].items():
                total_wins += match["wins"] + 0.5 * match["ties"]

            # Calculate denominator: sum of (comparisons / (my_rating + opponent_rating))
            denominator = 0.0
            for opponent, match in results[model].items():
                total_comparisons = match["wins"] + match["losses"] + match["ties"]
                denominator += total_comparisons / (
                    old_ratings[model] + old_ratings[opponent]
                )

            # Calculate new rating
            if denominator > 0:
                new_ratings[model] = total_wins / denominator
            else:
                new_ratings[model] = old_ratings[model]

        # Step 3: Normalize ratings (so they sum to number of models)
        total = sum(new_ratings.values())
        if total > 0:
            scale_factor = len(models) / total
            ratings = {
                model: rating * scale_factor for model, rating in new_ratings.items()
            }
        else:
            ratings = new_ratings

        # Step 4: Check for convergence
        max_change = max(abs(ratings[model] - old_ratings[model]) for model in models)

        if iteration % 10 == 0 or iteration < 5:
            print(f"   Iteration {iteration + 1}: max change = {max_change:.6f}")

        if max_change < CONVERGENCE_THRESHOLD:
            print(f"   ✅ Converged after {iteration + 1} iterations!")
            break
    else:
        print(f"   ⚠️ Stopped at maximum iterations ({MAX_ITERATIONS})")

    return ratings


def calculate_statistics(results: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Calculate win-loss statistics for each model."""
    stats = {}

    for model, opponents in results.items():
        wins = sum(match["wins"] for match in opponents.values())
        losses = sum(match["losses"] for match in opponents.values())
        ties = sum(match["ties"] for match in opponents.values())
        total = wins + losses + ties

        stats[model] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total": total,
            "win_rate": wins / total if total > 0 else 0.0,
            "points": wins + 0.5 * ties,
            "point_rate": (wins + 0.5 * ties) / total if total > 0 else 0.0,
        }

    return stats


# ============================================================================
# REPORTING
# ============================================================================


def print_bradley_terry_report(
    ratings: Dict[str, float],
    stats: Dict[str, Dict],
    results: Dict[str, Dict[str, Dict]],
) -> None:
    """Print comprehensive Bradley-Terry ranking report."""

    # Sort by Bradley-Terry rating (descending)
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Calculate column widths
    width_line = 170
    width_rank = 6
    width_model = 100
    width_rating = 12
    width_record = 18
    width_win_rate = 11
    width_point_rate = 11

    print("\n" + "=" * width_line)
    print("📊 BRADLEY-TERRY MODEL RANKING ANALYSIS")
    print("=" * width_line)

    # Summary
    total_models = len(ratings)
    total_comparisons = sum(s["total"] for s in stats.values()) // 2

    print(f"\n📈 Summary:")
    print(f"   Total models: {total_models}")
    print(f"   Total pairwise comparisons: {total_comparisons}")
    print(f"   Comparisons per model: {total_comparisons * 2 // total_models}")

    # Explain Bradley-Terry rating
    print(f"\n💡 About Bradley-Terry Rating:")
    print(f"   - Higher rating = stronger model")
    print(f"   - Average rating = 1.0")
    print(f"   - Rating accounts for opponent strength (not just wins)")
    print(f"   - Rating of 2.0 means twice as strong as average")

    # Detailed ranking table
    print("\n" + "=" * width_line)
    print("🏆 RANKING (by Bradley-Terry rating)")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'BT Rating':<{width_rating}} "
        f"{'W-L-T':<{width_record}} "
        f"{'Win Rate':<{width_win_rate}} "
        f"{'Point Rate':<{width_point_rate}}"
    )
    print(header)
    print("-" * width_line)

    # Data rows
    for rank, (model, rating) in enumerate(ranking, 1):
        s = stats[model]
        w, l, t = s["wins"], s["losses"], s["ties"]
        record = f"{w}-{l}-{t}"

        row = (
            f"{rank:<{width_rank}} "
            f"{model:<{width_model}} "
            f"{rating:<{width_rating}.3f} "
            f"{record:<{width_record}} "
            f"{s['win_rate']:<{width_win_rate}.2%} "
            f"{s['point_rate']:<{width_point_rate}.2%}"
        )
        print(row)

    print("=" * width_line)

    # Detailed individual statistics
    print("\n📋 Individual Statistics:")
    print("-" * width_line)

    for rank, (model, rating) in enumerate(ranking, 1):
        s = stats[model]
        print(f"\n{rank}. {model}")
        print(f"   Bradley-Terry Rating: {rating:.3f}")
        print(
            f"   Record: {s['wins']}-{s['losses']}-{s['ties']} ({s['win_rate']:.1%} wins)"
        )
        print(f"   Points: {s['points']:.1f} / {s['total']} ({s['point_rate']:.2%})")

        # Show comparison with average
        if rating > 1.0:
            print(f"   💪 {(rating / 1.0):.2f}x stronger than average")
        elif rating < 1.0:
            print(f"   📉 {(1.0 / rating):.2f}x weaker than average")
        else:
            print(f"   📊 Average strength")

    print("\n" + "=" * width_line)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run Bradley-Terry model analysis and ranking."""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    print(f"\n📂 Reading pairwise comparison results from:")
    print(f"   {CSV_PATH}")

    # Load pairwise results
    results = load_pairwise_results(CSV_PATH)

    # Calculate Bradley-Terry ratings
    ratings = calculate_bradley_terry_ratings(results)

    # Calculate additional statistics
    stats = calculate_statistics(results)

    # Print report
    print_bradley_terry_report(ratings, stats, results)

    print("\n✅ Bradley-Terry analysis complete!")


if __name__ == "__main__":
    main()

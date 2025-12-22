"""
Pairwise Comparison Analysis and Ranking

Calculates win-loss record ranking from pairwise comparison results.
Uses simple win counting method where:
- Win = 1 point
- Tie = 0.5 points
- Loss = 0 points
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

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def calculate_win_loss_ranking(csv_path: Path) -> Dict[str, Dict]:
    """
    Calculate ranking based on win-loss records from pairwise comparisons.

    This implements a simple win counting method where each experiment
    accumulates points based on pairwise comparison results.
    Uses pre-aggregated a_wins and b_wins from CSV.

    Returns:
        Dictionary with experiment names as keys and stats as values
    """
    # Initialize stats tracking
    stats = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_comparisons": 0,
            "points": 0.0,
        }
    )

    # Read CSV and accumulate statistics
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            exp_a = row["experiment_a_name"]
            exp_b = row["experiment_b_name"]
            a_wins = int(row["a_wins"])
            b_wins = int(row["b_wins"])
            total_comps = int(row["total_comparisons"])

            # Calculate ties: total - wins_a - wins_b
            ties = total_comps - a_wins - b_wins

            # Update experiment A stats
            stats[exp_a]["wins"] += a_wins
            stats[exp_a]["losses"] += b_wins
            stats[exp_a]["ties"] += ties
            stats[exp_a]["total_comparisons"] += total_comps
            stats[exp_a]["points"] += (a_wins * 1.0) + (ties * 0.5)

            # Update experiment B stats
            stats[exp_b]["wins"] += b_wins
            stats[exp_b]["losses"] += a_wins
            stats[exp_b]["ties"] += ties
            stats[exp_b]["total_comparisons"] += total_comps
            stats[exp_b]["points"] += (b_wins * 1.0) + (ties * 0.5)

    # Calculate win rate for each experiment
    for exp_name, exp_stats in stats.items():
        total = exp_stats["total_comparisons"]
        if total > 0:
            exp_stats["win_rate"] = exp_stats["wins"] / total
            exp_stats["point_rate"] = exp_stats["points"] / total
        else:
            exp_stats["win_rate"] = 0.0
            exp_stats["point_rate"] = 0.0

    return dict(stats)


def print_ranking_report(stats: Dict[str, Dict]) -> None:
    """Print comprehensive ranking report with statistics."""

    # Sort by points (descending), then by wins, then by win_rate
    ranking = sorted(
        stats.items(),
        key=lambda x: (x[1]["points"], x[1]["wins"], x[1]["win_rate"]),
        reverse=True,
    )

    print("\n" + "=" * 150)
    print("📊 PAIRWISE COMPARISON RANKING ANALYSIS")
    print("=" * 150)

    # Summary statistics
    total_experiments = len(stats)
    total_comparisons = (
        sum(s["total_comparisons"] for s in stats.values()) // 2
    )  # Divide by 2 as each comparison is counted twice

    print(f"\n📈 Summary:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Total pairwise comparisons: {total_comparisons}")
    print(
        f"   Comparisons per experiment: {total_comparisons * 2 // total_experiments}"
    )

    # Detailed ranking table
    print("\n" + "=" * 150)
    print("🏆 RANKING (by points)")
    print("=" * 150)
    print(
        f"{'Rank':<6} {'Experiment Name':<110} {'W-L-T':<12} {'Points':<10} {'Rate':<8}"
    )
    print("-" * 150)

    for rank, (exp_name, exp_stats) in enumerate(ranking, 1):
        w = exp_stats["wins"]
        l = exp_stats["losses"]
        t = exp_stats["ties"]
        points = exp_stats["points"]
        rate = exp_stats["point_rate"]

        print(f"{rank:<6} {exp_name:<110} {w}-{l}-{t:<8} {points:<10.1f} {rate:<8.2%}")

    print("=" * 150)

    # Win matrix summary
    print("\n📋 Individual Statistics:")
    print("-" * 150)

    for rank, (exp_name, exp_stats) in enumerate(ranking, 1):
        print(f"\n{rank}. {exp_name}")
        print(
            f"   Points: {exp_stats['points']:.1f} / {exp_stats['total_comparisons']}"
        )
        print(f"   Wins: {exp_stats['wins']} ({exp_stats['win_rate']:.1%})")
        print(f"   Losses: {exp_stats['losses']}")
        print(f"   Ties: {exp_stats['ties']}")
        print(f"   Point rate: {exp_stats['point_rate']:.2%}")


def main():
    """Run pairwise comparison analysis and ranking."""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    print(f"\n📂 Reading pairwise comparison results from:")
    print(f"   {CSV_PATH}")

    # Calculate rankings
    stats = calculate_win_loss_ranking(CSV_PATH)

    # Print report
    print_ranking_report(stats)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

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
    "Evaluations/what_is_evaluated/output_pairwise_comparison/pairwise_compare_more_experiments_20260102_234014.csv"
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


def save_ranking_to_csv(
    stats: Dict[str, Dict], csv_path: Path, algorithm_name: str
) -> None:
    """Save ranking report to CSV file with semicolon delimiter."""

    # Sort by points (descending), then by wins, then by win_rate
    ranking = sorted(
        stats.items(),
        key=lambda x: (x[1]["points"], x[1]["wins"], x[1]["win_rate"]),
        reverse=True,
    )

    # Generate output filename based on algorithm name
    timestamp = (
        csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
    )  # Extract timestamp
    output_filename = f"ranking_{algorithm_name}_{timestamp}.csv"
    output_path = csv_path.parent / output_filename

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")

        # Write header
        writer.writerow(
            [
                "Rank",
                "Experiment Name",
                "Wins",
                "Losses",
                "Ties",
                "Points",
                "Win Rate",
                "Point Rate",
            ]
        )

        # Write data rows
        for rank, (exp_name, exp_stats) in enumerate(ranking, 1):
            writer.writerow(
                [
                    rank,
                    exp_name,
                    exp_stats["wins"],
                    exp_stats["losses"],
                    exp_stats["ties"],
                    f"{exp_stats['points']:.1f}",
                    f"{exp_stats['win_rate']:.4f}",
                    f"{exp_stats['point_rate']:.4f}",
                ]
            )

    print(f"\n💾 Ranking saved to: {output_path}")


def print_ranking_report(stats: Dict[str, Dict]) -> None:
    """Print comprehensive ranking report with statistics."""

    # Sort by points (descending), then by wins, then by win_rate
    ranking = sorted(
        stats.items(),
        key=lambda x: (x[1]["points"], x[1]["wins"], x[1]["win_rate"]),
        reverse=True,
    )

    print("\n" + "=" * 165)
    print("🏆 RANKING (by points)")
    print("=" * 165)

    # Detailed ranking table header
    print(f"{'Rank':<6} {'Experiment Name':<105} {'W-L-T':<20} {'Points':<15} {'Rate'}")
    print("-" * 165)

    for rank, (exp_name, exp_stats) in enumerate(ranking, 1):
        w = exp_stats["wins"]
        l = exp_stats["losses"]
        t = exp_stats["ties"]
        points = exp_stats["points"]
        rate = exp_stats["point_rate"]

        # Shorten experiment name for better display
        display_name = exp_name.replace(
            "judge-azureopenai-gpt-4.1__node-format_answer_node__model-", ""
        )

        # Format W-L-T with better spacing
        wlt_str = f"{w}-{l}-{t}"
        wlt_display = f"{wlt_str:<15} {points:>6.1f}     {rate:>6.2%}"

        print(f"{rank:<6} {display_name:<105} {wlt_display}")

        # Add spacing after every 3 rows for better readability
        if rank % 3 == 0 and rank < len(ranking):
            print()

    print("=" * 165)


def main():
    """Run pairwise comparison analysis and ranking."""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    print("\n📂 Reading pairwise comparison results from:")
    print(f"   {CSV_PATH}")

    # Calculate rankings
    stats = calculate_win_loss_ranking(CSV_PATH)

    # Save ranking to CSV
    save_ranking_to_csv(stats, CSV_PATH, "simple_win_counting")

    # Print report
    print_ranking_report(stats)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

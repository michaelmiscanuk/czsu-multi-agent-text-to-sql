"""
Pairwise Comparison Strength Analysis

This script analyzes the STRENGTH of differences between models in pairwise comparisons.
It calculates multiple sophisticated metrics to measure:
1. Ranking Consistency (Standard Deviation & Coefficient of Variation)
2. Ranking Confidence (based on method agreement)
3. Statistical Separation (distance from other models)
4. Effect Size (Cohen's d for pairwise differences)
5. Dominance Score (how often model beats others)

These metrics help identify which models have:
- Strong, consistent performance across methods
- Weak or inconsistent performance
- High uncertainty in rankings
- Statistical significance in differences

Based on statistical methods including:
- Standard deviation and coefficient of variation for consistency
- Cohen's d effect size for pairwise comparisons
- Hedges' g with small sample correction
- Dominance analysis for competitive strength
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigs
from scipy import stats

try:
    import choix

    CHOIX_AVAILABLE = True
except ImportError:
    CHOIX_AVAILABLE = False
    print("⚠️ Warning: 'choix' library not available. Install with: pip install choix")

try:
    import trueskill

    TRUESKILL_AVAILABLE = True
except ImportError:
    TRUESKILL_AVAILABLE = False
    print(
        "⚠️ Warning: 'trueskill' library not available. Install with: pip install trueskill"
    )

try:
    import glicko2

    GLICKO2_AVAILABLE = True
except ImportError:
    GLICKO2_AVAILABLE = False
    print(
        "⚠️ Warning: 'glicko2' library not available. Install with: pip install glicko2"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = Path(
    "Evaluations/what_is_evaluated/output_pairwise_comparison/pairwise_compare_more_experiments_20260102_234014.csv"
)

# ============================================================================
# DATA LOADING (reusing from original script)
# ============================================================================


def load_pairwise_data(csv_path: Path) -> Tuple[Dict, Dict, List[str]]:
    """Load pairwise comparison data from CSV."""
    match_results = {}
    model_stats = defaultdict(
        lambda: {"wins": 0, "losses": 0, "ties": 0, "opponents": []}
    )
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

            model_stats[exp_a]["wins"] += a_wins
            model_stats[exp_a]["losses"] += b_wins
            model_stats[exp_a]["ties"] += ties
            model_stats[exp_a]["opponents"].append(exp_b)

            model_stats[exp_b]["wins"] += b_wins
            model_stats[exp_b]["losses"] += a_wins
            model_stats[exp_b]["ties"] += ties
            model_stats[exp_b]["opponents"].append(exp_a)

    model_names = sorted(list(models_set))
    return match_results, dict(model_stats), model_names


# ============================================================================
# IMPORT RANKING METHODS (simplified versions)
# ============================================================================


def simple_win_counting(model_stats: Dict) -> Dict[str, float]:
    """Win=1, Tie=0.5, Loss=0."""
    ratings = {}
    for model, stats in model_stats.items():
        points = stats["wins"] + 0.5 * stats["ties"]
        ratings[model] = points
    return ratings


def elo_rating(
    match_results: Dict, model_names: List[str], k_factor: int = 32
) -> Dict[str, float]:
    """Elo rating system."""
    ratings = {model: 1500.0 for model in model_names}

    for (model_a, model_b), result in match_results.items():
        for _ in range(result["a_wins"]):
            ratings[model_a], ratings[model_b] = update_elo(
                ratings[model_a], ratings[model_b], 1.0, k_factor
            )
        for _ in range(result["b_wins"]):
            ratings[model_a], ratings[model_b] = update_elo(
                ratings[model_a], ratings[model_b], 0.0, k_factor
            )
        for _ in range(result["ties"]):
            ratings[model_a], ratings[model_b] = update_elo(
                ratings[model_a], ratings[model_b], 0.5, k_factor
            )

    return ratings


def update_elo(
    rating_a: float, rating_b: float, score_a: float, k: int
) -> Tuple[float, float]:
    """Update Elo ratings after a match."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1 - score_a) - expected_b)

    return new_rating_a, new_rating_b


def colley_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """Colley Matrix method."""
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    C = np.zeros((n, n))
    b = np.ones(n)

    games_played = {model: 0 for model in model_names}
    win_diff = {model: 0 for model in model_names}

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total_games = result["a_wins"] + result["b_wins"] + result["ties"]
        games_played[model_a] += total_games
        games_played[model_b] += total_games

        win_diff[model_a] += result["a_wins"] - result["b_wins"]
        win_diff[model_b] += result["b_wins"] - result["a_wins"]

        C[idx_a, idx_b] -= total_games
        C[idx_b, idx_a] -= total_games

    for model in model_names:
        idx = model_to_idx[model]
        C[idx, idx] = 2 + games_played[model]
        b[idx] = 1 + win_diff[model] / 2

    ratings_array = np.linalg.solve(C, b)
    ratings = {model: ratings_array[model_to_idx[model]] * 100 for model in model_names}

    return ratings


# ============================================================================
# STRENGTH METRICS CALCULATION
# ============================================================================


def calculate_ranking_consistency(
    all_ranks: List[Dict[str, int]], model_names: List[str]
) -> Dict[str, Dict]:
    """
    Calculate consistency metrics for each model's ranking across methods.

    Metrics:
    - Standard Deviation: Lower = more consistent
    - Coefficient of Variation: Normalized consistency (SD / Mean)
    - Rank Range: Max rank - Min rank
    - Inter-Quartile Range: Spread of middle 50% of ranks
    """
    consistency_metrics = {}

    for model in model_names:
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]

        if ranks:
            ranks_array = np.array(ranks)
            mean_rank = np.mean(ranks_array)
            std_rank = np.std(ranks_array, ddof=1)  # Sample standard deviation

            # Coefficient of Variation (lower = more consistent)
            cv = (std_rank / mean_rank * 100) if mean_rank > 0 else 0

            # Range
            rank_range = np.max(ranks_array) - np.min(ranks_array)

            # Inter-Quartile Range
            q75, q25 = np.percentile(ranks_array, [75, 25])
            iqr = q75 - q25

            consistency_metrics[model] = {
                "mean_rank": mean_rank,
                "std_dev": std_rank,
                "coefficient_variation": cv,
                "rank_range": rank_range,
                "iqr": iqr,
                "min_rank": np.min(ranks_array),
                "max_rank": np.max(ranks_array),
            }
        else:
            consistency_metrics[model] = {
                "mean_rank": float("inf"),
                "std_dev": 0,
                "coefficient_variation": 0,
                "rank_range": 0,
                "iqr": 0,
                "min_rank": 0,
                "max_rank": 0,
            }

    return consistency_metrics


def calculate_confidence_score(
    all_ranks: List[Dict[str, int]], model_names: List[str]
) -> Dict[str, float]:
    """
    Calculate confidence score based on method agreement.

    Higher score = more methods agree on similar ranking.
    Uses mode frequency and clustering of ranks.
    """
    confidence_scores = {}

    for model in model_names:
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]

        if ranks:
            # Mode frequency (how often most common rank appears)
            rank_counts = Counter(ranks)
            mode_count = max(rank_counts.values())
            mode_frequency = mode_count / len(ranks)

            # Rank clustering (proportion within ±1 of median)
            median_rank = np.median(ranks)
            within_one = sum(1 for r in ranks if abs(r - median_rank) <= 1)
            clustering_score = within_one / len(ranks)

            # Combined confidence (0-100 scale)
            confidence = (mode_frequency * 0.5 + clustering_score * 0.5) * 100
            confidence_scores[model] = confidence
        else:
            confidence_scores[model] = 0.0

    return confidence_scores


def calculate_separation_strength(
    all_ranks: List[Dict[str, int]], model_names: List[str]
) -> Dict[str, float]:
    """
    Calculate how well-separated each model is from others.

    Higher score = larger gap between this model and adjacent models.
    Uses average rank distance to nearest neighbors.
    """
    separation_scores = {}

    # Get average ranks for all models
    avg_ranks = {}
    for model in model_names:
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]
        avg_ranks[model] = np.mean(ranks) if ranks else float("inf")

    # Sort models by average rank
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])

    for i, (model, avg_rank) in enumerate(sorted_models):
        # Distance to previous and next model
        distances = []

        if i > 0:
            distances.append(abs(avg_rank - sorted_models[i - 1][1]))

        if i < len(sorted_models) - 1:
            distances.append(abs(sorted_models[i + 1][1] - avg_rank))

        # Average separation (normalized by total models)
        avg_separation = np.mean(distances) if distances else 0
        normalized_separation = avg_separation / len(model_names) * 100

        separation_scores[model] = normalized_separation

    return separation_scores


def calculate_effect_size_cohens_d(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """
    Calculate Cohen's d effect size for each model vs all others.

    Cohen's d measures the standardized difference between two means:
    - Small effect: d = 0.2
    - Medium effect: d = 0.5
    - Large effect: d = 0.8

    Higher absolute value = stronger effect.
    """
    effect_sizes = {}

    for model in model_names:
        # Collect all pairwise win rates for this model
        win_rates = []

        for (model_a, model_b), result in match_results.items():
            if model_a == model:
                total = result["a_wins"] + result["b_wins"] + result["ties"]
                if total > 0:
                    win_rate = (result["a_wins"] + 0.5 * result["ties"]) / total
                    win_rates.append(win_rate)
            elif model_b == model:
                total = result["a_wins"] + result["b_wins"] + result["ties"]
                if total > 0:
                    win_rate = (result["b_wins"] + 0.5 * result["ties"]) / total
                    win_rates.append(win_rate)

        if len(win_rates) > 1:
            # Calculate mean and std of win rates
            mean_win_rate = np.mean(win_rates)
            std_win_rate = np.std(win_rates, ddof=1)

            # Cohen's d comparing to neutral (0.5 = equal performance)
            if std_win_rate > 0:
                cohens_d = (mean_win_rate - 0.5) / std_win_rate
            else:
                cohens_d = 0.0

            effect_sizes[model] = cohens_d
        else:
            effect_sizes[model] = 0.0

    return effect_sizes


def calculate_dominance_score(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """
    Calculate dominance score (percentage of pairwise matchups won).

    Accounts for strength of opponents using weighted scoring.
    """
    dominance_scores = {}

    # First pass: calculate raw win rates
    raw_win_rates = {}
    for model in model_names:
        wins = 0
        total_matchups = 0

        for (model_a, model_b), result in match_results.items():
            if model_a == model:
                total = result["a_wins"] + result["b_wins"] + result["ties"]
                wins += result["a_wins"] + 0.5 * result["ties"]
                total_matchups += total
            elif model_b == model:
                total = result["a_wins"] + result["b_wins"] + result["ties"]
                wins += result["b_wins"] + 0.5 * result["ties"]
                total_matchups += total

        raw_win_rates[model] = wins / total_matchups if total_matchups > 0 else 0

    # Second pass: weighted dominance considering opponent strength
    for model in model_names:
        weighted_score = 0
        total_weight = 0

        for (model_a, model_b), result in match_results.items():
            opponent = None
            model_wins = 0
            total = result["a_wins"] + result["b_wins"] + result["ties"]

            if model_a == model:
                opponent = model_b
                model_wins = result["a_wins"] + 0.5 * result["ties"]
            elif model_b == model:
                opponent = model_a
                model_wins = result["b_wins"] + 0.5 * result["ties"]

            if opponent and total > 0:
                # Weight by opponent's strength (higher weight for beating strong opponents)
                opponent_strength = raw_win_rates.get(opponent, 0.5)
                weight = 0.5 + opponent_strength  # Weight between 0.5 and 1.5

                win_rate = model_wins / total
                weighted_score += win_rate * weight
                total_weight += weight

        dominance_scores[model] = (
            (weighted_score / total_weight * 100) if total_weight > 0 else 0
        )

    return dominance_scores


def calculate_strength_category(
    consistency_metrics: Dict,
    confidence_scores: Dict,
    effect_sizes: Dict,
    dominance_scores: Dict,
) -> Dict[str, str]:
    """
    Categorize each model's strength based on multiple metrics.

    Categories:
    - DOMINANT: High dominance, high confidence, large effect size
    - STRONG: Good metrics across the board
    - MODERATE: Average performance
    - WEAK: Below average metrics
    - INCONSISTENT: High variance in rankings
    """
    categories = {}

    for model in consistency_metrics.keys():
        cv = consistency_metrics[model]["coefficient_variation"]
        confidence = confidence_scores.get(model, 0)
        effect_size = abs(effect_sizes.get(model, 0))
        dominance = dominance_scores.get(model, 50)

        # Decision tree for categorization
        if cv > 30:
            category = "INCONSISTENT"
        elif dominance >= 70 and confidence >= 70 and effect_size >= 0.8:
            category = "DOMINANT"
        elif dominance >= 60 and confidence >= 60:
            category = "STRONG"
        elif dominance <= 40 and confidence <= 40:
            category = "WEAK"
        else:
            category = "MODERATE"

        categories[model] = category

    return categories


# ============================================================================
# INTEGRATED ANALYSIS
# ============================================================================


def run_all_ranking_methods(
    match_results: Dict, model_names: List[str]
) -> List[Dict[str, int]]:
    """Run a subset of key ranking methods and return ranks."""
    all_ranks = []

    # Method 1: Simple Win Counting
    ratings_1 = simple_win_counting(
        {
            model: {
                "wins": sum(
                    r["a_wins"] if m == model else r["b_wins"]
                    for (m, _), r in match_results.items()
                    if m == model or _ == model
                ),
                "losses": sum(
                    r["b_wins"] if m == model else r["a_wins"]
                    for (m, _), r in match_results.items()
                    if m == model or _ == model
                ),
                "ties": sum(
                    r["ties"]
                    for (m, _), r in match_results.items()
                    if m == model or _ == model
                ),
                "opponents": [],
            }
            for model in model_names
        }
    )
    ranks_1 = {
        model: rank
        for rank, (model, _) in enumerate(
            sorted(ratings_1.items(), key=lambda x: x[1], reverse=True), 1
        )
    }
    all_ranks.append(ranks_1)

    # Method 2: Elo Rating
    ratings_2 = elo_rating(match_results, model_names)
    ranks_2 = {
        model: rank
        for rank, (model, _) in enumerate(
            sorted(ratings_2.items(), key=lambda x: x[1], reverse=True), 1
        )
    }
    all_ranks.append(ranks_2)

    # Method 3: Colley Method
    ratings_3 = colley_method(match_results, model_names)
    ranks_3 = {
        model: rank
        for rank, (model, _) in enumerate(
            sorted(ratings_3.items(), key=lambda x: x[1], reverse=True), 1
        )
    }
    all_ranks.append(ranks_3)

    return all_ranks


def clean_model_name(model: str) -> str:
    """Clean model name by removing prefix and suffix."""
    prefix = "judge-azureopenai-gpt-4.1__node-format_answer_node__model-"

    cleaned = model
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]

    if "-" in cleaned:
        parts = cleaned.rsplit("-", 1)
        if len(parts[1]) <= 10 and parts[1].isalnum():
            cleaned = parts[0]

    return cleaned


def save_strength_analysis_to_csv(
    model_names: List[str],
    consistency_metrics: Dict,
    confidence_scores: Dict,
    separation_scores: Dict,
    effect_sizes: Dict,
    dominance_scores: Dict,
    strength_categories: Dict,
    csv_path: Path,
) -> None:
    """Save comprehensive strength analysis to CSV with semicolon delimiter."""

    # Generate output filename
    timestamp = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
    output_filename = f"ranking_strength_analysis_{timestamp}.csv"
    output_path = csv_path.parent / output_filename

    # Prepare data sorted by dominance score (descending)
    model_data = []
    for model in model_names:
        model_data.append(
            {
                "model": model,
                "dominance": dominance_scores.get(model, 0),
                "confidence": confidence_scores.get(model, 0),
                "effect_size": effect_sizes.get(model, 0),
                "separation": separation_scores.get(model, 0),
                "mean_rank": consistency_metrics[model]["mean_rank"],
                "std_dev": consistency_metrics[model]["std_dev"],
                "cv": consistency_metrics[model]["coefficient_variation"],
                "rank_range": consistency_metrics[model]["rank_range"],
                "category": strength_categories.get(model, "UNKNOWN"),
            }
        )

    # Sort by dominance score (descending)
    model_data.sort(key=lambda x: x["dominance"], reverse=True)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(
            [
                "Rank",
                "Model Name",
                "Strength Category",
                "Dominance Score",
                "Confidence Score",
                "Effect Size (d)",
                "Separation",
                "Mean Rank",
                "Std Dev",
                "CV %",
                "Rank Range",
            ]
        )

        # Write data rows
        for rank, data in enumerate(model_data, 1):
            writer.writerow(
                [
                    rank,
                    data["model"],
                    data["category"],
                    f"{data['dominance']:.2f}",
                    f"{data['confidence']:.2f}",
                    f"{data['effect_size']:.3f}",
                    f"{data['separation']:.2f}",
                    f"{data['mean_rank']:.2f}",
                    f"{data['std_dev']:.3f}",
                    f"{data['cv']:.2f}",
                    int(data["rank_range"]),
                ]
            )

    print(f"\n💾 Strength analysis saved to: {output_path}")


def print_strength_analysis_report(
    model_names: List[str],
    consistency_metrics: Dict,
    confidence_scores: Dict,
    separation_scores: Dict,
    effect_sizes: Dict,
    dominance_scores: Dict,
    strength_categories: Dict,
) -> None:
    """Print comprehensive strength analysis report."""

    # Prepare data sorted by dominance score
    model_data = []
    for model in model_names:
        model_data.append(
            {
                "model": model,
                "dominance": dominance_scores.get(model, 0),
                "confidence": confidence_scores.get(model, 0),
                "effect_size": effect_sizes.get(model, 0),
                "separation": separation_scores.get(model, 0),
                "mean_rank": consistency_metrics[model]["mean_rank"],
                "std_dev": consistency_metrics[model]["std_dev"],
                "cv": consistency_metrics[model]["coefficient_variation"],
                "category": strength_categories.get(model, "UNKNOWN"),
            }
        )

    model_data.sort(key=lambda x: x["dominance"], reverse=True)

    # Print report
    width_line = 180
    print("\n" + "=" * width_line)
    print("🎯 STRENGTH OF DIFFERENCES ANALYSIS")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<6} "
        f"{'Model Name':<40} "
        f"{'Category':<14} "
        f"{'Dom.':<7} "
        f"{'Conf.':<7} "
        f"{'Effect':<8} "
        f"{'Sep.':<7} "
        f"{'Mean':<7} "
        f"{'StdDev':<8} "
        f"{'CV%':<7}"
    )
    print(header)
    print("-" * width_line)

    # Data rows
    for rank, data in enumerate(model_data, 1):
        cleaned_name = clean_model_name(data["model"])

        row = (
            f"{rank:<6} "
            f"{cleaned_name:<40} "
            f"{data['category']:<14} "
            f"{data['dominance']:<7.2f} "
            f"{data['confidence']:<7.2f} "
            f"{data['effect_size']:<8.3f} "
            f"{data['separation']:<7.2f} "
            f"{data['mean_rank']:<7.2f} "
            f"{data['std_dev']:<8.3f} "
            f"{data['cv']:<7.2f}"
        )
        print(row)

    print("=" * width_line)

    # Legend
    print("\n📊 METRICS EXPLANATION:")
    print("   Dom. (Dominance): % of matchups won (weighted by opponent strength)")
    print("   Conf. (Confidence): Agreement between ranking methods (0-100)")
    print(
        "   Effect: Cohen's d effect size vs neutral (|0.2|=small, |0.5|=medium, |0.8|=large)"
    )
    print("   Sep. (Separation): Statistical distance from adjacent models")
    print("   Mean: Average rank across all methods")
    print("   StdDev: Standard deviation of ranks (consistency)")
    print("   CV%: Coefficient of Variation (normalized consistency)")

    print("\n🏆 STRENGTH CATEGORIES:")
    print("   DOMINANT: Top performer with consistent, strong results")
    print("   STRONG: Above-average performance across metrics")
    print("   MODERATE: Average performance")
    print("   WEAK: Below-average performance")
    print("   INCONSISTENT: High variability in rankings across methods")

    # Category breakdown
    category_counts = Counter(data["category"] for data in model_data)
    print("\n📈 DISTRIBUTION:")
    for category in ["DOMINANT", "STRONG", "MODERATE", "WEAK", "INCONSISTENT"]:
        count = category_counts.get(category, 0)
        pct = count / len(model_data) * 100 if model_data else 0
        print(f"   {category:<14}: {count:>2} models ({pct:>5.1f}%)")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run comprehensive strength analysis."""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return

    print(f"\n📂 Reading pairwise comparison results from:")
    print(f"   {CSV_PATH}")

    # Load data
    match_results, model_stats, model_names = load_pairwise_data(CSV_PATH)

    print(f"\n📊 Loaded data:")
    print(f"   Models: {len(model_names)}")
    print(f"   Pairwise matchups: {len(match_results)}")

    print("\n" + "=" * 170)
    print("🔬 COMPUTING STRENGTH METRICS")
    print("=" * 170)

    # Run ranking methods to get ranks
    print("\n[1/7] Running ranking methods...")
    all_ranks = run_all_ranking_methods(match_results, model_names)

    # Calculate strength metrics
    print("[2/7] Calculating ranking consistency...")
    consistency_metrics = calculate_ranking_consistency(all_ranks, model_names)

    print("[3/7] Calculating confidence scores...")
    confidence_scores = calculate_confidence_score(all_ranks, model_names)

    print("[4/7] Calculating separation strength...")
    separation_scores = calculate_separation_strength(all_ranks, model_names)

    print("[5/7] Calculating effect sizes...")
    effect_sizes = calculate_effect_size_cohens_d(match_results, model_names)

    print("[6/7] Calculating dominance scores...")
    dominance_scores = calculate_dominance_score(match_results, model_names)

    print("[7/7] Categorizing strength levels...")
    strength_categories = calculate_strength_category(
        consistency_metrics, confidence_scores, effect_sizes, dominance_scores
    )

    # Save to CSV
    save_strength_analysis_to_csv(
        model_names,
        consistency_metrics,
        confidence_scores,
        separation_scores,
        effect_sizes,
        dominance_scores,
        strength_categories,
        CSV_PATH,
    )

    # Print report
    print_strength_analysis_report(
        model_names,
        consistency_metrics,
        confidence_scores,
        separation_scores,
        effect_sizes,
        dominance_scores,
        strength_categories,
    )

    print("\n" + "=" * 170)
    print("✅ Strength analysis complete!")
    print("=" * 170)
    print("\n")


if __name__ == "__main__":
    main()

"""
Validation Script for Pairwise Comparison Data

This script validates the quality and suitability of pairwise comparison data
for ranking methods. It checks:
1. Completeness - whether all pairs have been compared
2. Balance - whether comparisons per pair are consistent
3. Statistical power - whether sample sizes are adequate
4. Transitivity - whether there are cycles (A>B>C>A patterns)
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = Path(
    "Evaluations/what_is_evaluated/output_pairwise_comparison/"
    "pairwise_compare_more_experiments_20251222_110500.csv"
)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def load_and_analyze_structure(
    csv_path: Path,
) -> Tuple[Set[str], List[Tuple[str, str]], List[int], Dict]:
    """
    Load CSV and extract basic structure information.

    Returns:
        - models: Set of all model names
        - matchups: List of (model_a, model_b) tuples
        - comparisons_per_pair: List of comparison counts per matchup
        - win_matrix: Dictionary of win results
    """
    models = set()
    matchups = []
    comparisons_per_pair = []
    win_matrix = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            exp_a = row["experiment_a_name"]
            exp_b = row["experiment_b_name"]
            total = int(row["total_comparisons"])
            a_wins = int(row["a_wins"])
            b_wins = int(row["b_wins"])
            ties = total - a_wins - b_wins  # Calculate ties from total

            models.add(exp_a)
            models.add(exp_b)
            matchups.append((exp_a, exp_b))
            comparisons_per_pair.append(total)
            win_matrix[(exp_a, exp_b)] = {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "total": total,
            }

    return models, matchups, comparisons_per_pair, win_matrix


def check_completeness(models: Set[str], matchups: List[Tuple[str, str]]) -> Dict:
    """Check if all possible pairs have been compared."""
    n_models = len(models)
    n_matchups = len(matchups)
    expected_matchups = n_models * (n_models - 1) // 2

    completeness = {
        "n_models": n_models,
        "n_matchups": n_matchups,
        "expected_matchups": expected_matchups,
        "is_complete": n_matchups == expected_matchups,
        "missing_matchups": expected_matchups - n_matchups,
    }

    return completeness


def check_balance(comparisons_per_pair: List[int]) -> Dict:
    """Check if comparison counts are balanced across matchups."""
    min_comp = min(comparisons_per_pair)
    max_comp = max(comparisons_per_pair)
    mean_comp = np.mean(comparisons_per_pair)
    std_comp = np.std(comparisons_per_pair)
    cv = std_comp / mean_comp if mean_comp > 0 else 0  # Coefficient of variation

    balance = {
        "min_comparisons": min_comp,
        "max_comparisons": max_comp,
        "mean_comparisons": mean_comp,
        "std_comparisons": std_comp,
        "coefficient_of_variation": cv,
        "is_balanced": cv < 0.1,  # CV < 0.1 indicates good balance
    }

    return balance


def check_statistical_power(comparisons_per_pair: List[int]) -> Dict:
    """Check if sample sizes are adequate for reliable rankings."""
    min_comp = min(comparisons_per_pair)

    if min_comp >= 30:
        power_level = "excellent"
        is_adequate = True
    elif min_comp >= 20:
        power_level = "good"
        is_adequate = True
    elif min_comp >= 10:
        power_level = "adequate"
        is_adequate = True
    else:
        power_level = "insufficient"
        is_adequate = False

    power = {
        "min_sample_size": min_comp,
        "power_level": power_level,
        "is_adequate": is_adequate,
        "n_below_10": sum(1 for c in comparisons_per_pair if c < 10),
        "n_below_20": sum(1 for c in comparisons_per_pair if c < 20),
        "n_below_30": sum(1 for c in comparisons_per_pair if c < 30),
    }

    return power


def find_cycles(win_matrix: Dict, models: Set[str]) -> Tuple[int, List]:
    """
    Detect cycles (transitivity violations) in pairwise comparisons.

    A cycle exists when A>B, B>C, but C>A (rock-paper-scissors pattern).
    """
    # Build dominance graph (who beats whom)
    dominates = defaultdict(set)

    for (model_a, model_b), results in win_matrix.items():
        a_wins = results["a_wins"]
        b_wins = results["b_wins"]

        if a_wins > b_wins:
            dominates[model_a].add(model_b)
        elif b_wins > a_wins:
            dominates[model_b].add(model_a)
        # Ties don't establish dominance

    # Find 3-cycles (smallest cycles)
    cycles_found = []
    cycle_count = 0

    models_list = list(models)
    for i, a in enumerate(models_list):
        for b in dominates[a]:
            for c in dominates[b]:
                if a in dominates[c] and a != c and b != c:
                    cycle = (a, b, c)
                    # Avoid counting same cycle multiple times
                    if cycle not in cycles_found:
                        cycles_found.append(cycle)
                        cycle_count += 1

    return cycle_count, cycles_found


def check_win_rate_distribution(win_matrix: Dict, models: Set[str]) -> Dict:
    """Analyze the distribution of win rates across models."""
    model_win_rates = {}

    for model in models:
        total_wins = 0
        total_comparisons = 0

        for (model_a, model_b), results in win_matrix.items():
            if model_a == model:
                total_wins += results["a_wins"]
                total_comparisons += results["total"]
            elif model_b == model:
                total_wins += results["b_wins"]
                total_comparisons += results["total"]

        win_rate = total_wins / total_comparisons if total_comparisons > 0 else 0
        model_win_rates[model] = win_rate

    win_rates = list(model_win_rates.values())

    distribution = {
        "min_win_rate": min(win_rates),
        "max_win_rate": max(win_rates),
        "mean_win_rate": np.mean(win_rates),
        "std_win_rate": np.std(win_rates),
        "range": max(win_rates) - min(win_rates),
        "is_discriminative": (max(win_rates) - min(win_rates))
        > 0.2,  # At least 20% range
    }

    return distribution, model_win_rates


# ============================================================================
# REPORTING
# ============================================================================


def clean_model_name(model: str) -> str:
    """Clean model name by removing prefix and suffix."""
    prefix = "judge-azureopenai-gpt-4.1__node-format_answer_node__model-"

    cleaned = model
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]

    if "-" in cleaned:
        parts = cleaned.rsplit("-", 1)
        if len(parts[-1]) == 8:  # Likely a hash
            cleaned = parts[0]

    return cleaned


def print_validation_report(
    completeness: Dict,
    balance: Dict,
    power: Dict,
    cycles: Tuple[int, List],
    distribution: Dict,
    model_win_rates: Dict,
):
    """Print comprehensive validation report."""

    width = 100

    print("\n" + "=" * width)
    print("PAIRWISE COMPARISON DATA VALIDATION REPORT")
    print("=" * width)
    print(f"\n📂 Data source: {CSV_PATH}")

    # 1. COMPLETENESS
    print("\n" + "-" * width)
    print("1️⃣  COMPLETENESS CHECK")
    print("-" * width)
    print(f"   Models analyzed: {completeness['n_models']}")
    print(f"   Actual matchups: {completeness['n_matchups']}")
    print(
        f"   Expected matchups (full round-robin): {completeness['expected_matchups']}"
    )

    if completeness["is_complete"]:
        print(f"   ✅ COMPLETE: All possible pairs have been compared")
    else:
        print(f"   ⚠️  INCOMPLETE: Missing {completeness['missing_matchups']} matchups")
        print(f"      → Impact: Some ranking methods may be biased")
        print(f"      → Recommendation: Complete missing comparisons if possible")

    # 2. BALANCE
    print("\n" + "-" * width)
    print("2️⃣  COMPARISON BALANCE")
    print("-" * width)
    print(f"   Min comparisons per matchup: {balance['min_comparisons']}")
    print(f"   Max comparisons per matchup: {balance['max_comparisons']}")
    print(f"   Mean comparisons: {balance['mean_comparisons']:.1f}")
    print(f"   Standard deviation: {balance['std_comparisons']:.1f}")
    print(f"   Coefficient of variation: {balance['coefficient_of_variation']:.2%}")

    if balance["is_balanced"]:
        print(f"   ✅ BALANCED: Low variance in comparison counts (CV < 10%)")
    else:
        print(f"   ⚠️  UNBALANCED: High variance in comparison counts (CV ≥ 10%)")
        print(f"      → Impact: Some matchups have more weight than others")
        print(f"      → Recommendation: Consider weighted ranking methods")

    # 3. STATISTICAL POWER
    print("\n" + "-" * width)
    print("3️⃣  STATISTICAL POWER")
    print("-" * width)
    print(f"   Minimum sample size: {power['min_sample_size']}")
    print(f"   Power level: {power['power_level'].upper()}")
    print(f"   Matchups with n < 10: {power['n_below_10']}")
    print(f"   Matchups with n < 20: {power['n_below_20']}")
    print(f"   Matchups with n < 30: {power['n_below_30']}")

    if power["is_adequate"]:
        print(f"   ✅ ADEQUATE: Sample sizes support reliable rankings")
    else:
        print(f"   ❌ INSUFFICIENT: Sample sizes too small for reliable rankings")
        print(f"      → Impact: High uncertainty in rankings")
        print(f"      → Recommendation: Collect more comparisons per matchup")

    # 4. TRANSITIVITY (CYCLES)
    print("\n" + "-" * width)
    print("4️⃣  TRANSITIVITY CHECK")
    print("-" * width)

    cycle_count, cycles_found = cycles
    print(f"   Cycles detected: {cycle_count}")

    if cycle_count == 0:
        print(f"   ✅ TRANSITIVE: No cycles found (clean hierarchy)")
    elif cycle_count <= 3:
        print(f"   ⚠️  MINOR CYCLES: Few transitivity violations detected")
        print(f"      → Impact: Rankings mostly reliable with minor ambiguities")
    else:
        print(f"   ⚠️  MULTIPLE CYCLES: Several transitivity violations detected")
        print(f"      → Impact: No clear hierarchy, rankings may be ambiguous")
        print(f"      → Recommendation: Use consensus methods (average rankings)")

    if cycles_found and cycle_count <= 5:
        print(f"\n   Examples of cycles found:")
        for i, (a, b, c) in enumerate(cycles_found[:5], 1):
            a_clean = clean_model_name(a)
            b_clean = clean_model_name(b)
            c_clean = clean_model_name(c)
            print(f"      {i}. {a_clean} > {b_clean} > {c_clean} > {a_clean}")

    # 5. WIN RATE DISTRIBUTION
    print("\n" + "-" * width)
    print("5️⃣  WIN RATE DISTRIBUTION")
    print("-" * width)
    print(f"   Minimum win rate: {distribution['min_win_rate']:.1%}")
    print(f"   Maximum win rate: {distribution['max_win_rate']:.1%}")
    print(f"   Mean win rate: {distribution['mean_win_rate']:.1%}")
    print(f"   Standard deviation: {distribution['std_win_rate']:.1%}")
    print(f"   Range: {distribution['range']:.1%}")

    if distribution["is_discriminative"]:
        print(f"   ✅ DISCRIMINATIVE: Wide range of performance (≥20% spread)")
        print(f"      → Rankings will show clear differences between models")
    else:
        print(f"   ⚠️  LOW DISCRIMINATION: Narrow range of performance (<20% spread)")
        print(f"      → Impact: Models are very similar, rankings may be sensitive")
        print(f"      → Recommendation: Focus on top-tier consensus, not exact order")

    # Top and Bottom Performers
    sorted_models = sorted(model_win_rates.items(), key=lambda x: x[1], reverse=True)

    print(f"\n   Top 3 performers by win rate:")
    for i, (model, win_rate) in enumerate(sorted_models[:3], 1):
        print(f"      {i}. {clean_model_name(model)}: {win_rate:.1%}")

    print(f"\n   Bottom 3 performers by win rate:")
    for i, (model, win_rate) in enumerate(sorted_models[-3:], 1):
        print(f"      {i}. {clean_model_name(model)}: {win_rate:.1%}")

    # 6. OVERALL ASSESSMENT
    print("\n" + "=" * width)
    print("6️⃣  OVERALL ASSESSMENT")
    print("=" * width)

    checks_passed = sum(
        [
            completeness["is_complete"],
            balance["is_balanced"],
            power["is_adequate"],
            cycle_count <= 3,
            distribution["is_discriminative"],
        ]
    )

    print(f"\n   Quality checks passed: {checks_passed}/5")

    if checks_passed >= 4:
        print(f"\n   ✅ EXCELLENT: Data is highly suitable for ranking methods")
        print(f"      → All ranking methods should produce meaningful results")
        print(f"      → High confidence in consensus rankings")
    elif checks_passed >= 3:
        print(f"\n   ✅ GOOD: Data is suitable for ranking methods")
        print(f"      → Most ranking methods should work well")
        print(f"      → Focus on consensus methods for robustness")
    elif checks_passed >= 2:
        print(f"\n   ⚠️  FAIR: Data has some limitations")
        print(f"      → Rankings are meaningful but interpret with caution")
        print(f"      → Use multiple methods and check for agreement")
        print(f"      → Consider addressing identified issues")
    else:
        print(f"\n   ❌ POOR: Data has significant limitations")
        print(f"      → Rankings may be unreliable")
        print(f"      → Strongly recommend improving data quality")
        print(f"      → Address critical issues before ranking")

    # Recommendations
    print("\n" + "-" * width)
    print("💡 RECOMMENDATIONS")
    print("-" * width)

    recommendations = []

    if not completeness["is_complete"]:
        recommendations.append("• Complete missing pairwise comparisons")

    if not balance["is_balanced"]:
        recommendations.append("• Balance comparison counts across matchups")
        recommendations.append("• Or use weighted ranking methods")

    if not power["is_adequate"]:
        recommendations.append("• Increase sample size per matchup (target n≥20)")

    if cycle_count > 3:
        recommendations.append("• Focus on consensus methods (average rankings)")
        recommendations.append("• Consider ensemble approaches")

    if not distribution["is_discriminative"]:
        recommendations.append("• Models are similar - focus on top-tier grouping")
        recommendations.append("• Report confidence intervals for rankings")

    if not recommendations:
        recommendations.append("• Data quality is excellent - proceed with ranking!")
        recommendations.append("• Use multiple methods to validate consensus")
        recommendations.append("• Consider bootstrap analysis for confidence intervals")

    for rec in recommendations:
        print(f"   {rec}")

    print("\n" + "=" * width)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run validation analysis on pairwise comparison data."""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"\n❌ ERROR: CSV file not found at {CSV_PATH}")
        print(f"   Please check the file path and try again.")
        return

    print("\n🔍 Starting validation analysis...")

    # Load data
    models, matchups, comparisons_per_pair, win_matrix = load_and_analyze_structure(
        CSV_PATH
    )

    # Run validation checks
    completeness = check_completeness(models, matchups)
    balance = check_balance(comparisons_per_pair)
    power = check_statistical_power(comparisons_per_pair)
    cycles = find_cycles(win_matrix, models)
    distribution, model_win_rates = check_win_rate_distribution(win_matrix, models)

    # Print report
    print_validation_report(
        completeness, balance, power, cycles, distribution, model_win_rates
    )

    print("\n✅ Validation analysis complete!\n")


if __name__ == "__main__":
    main()

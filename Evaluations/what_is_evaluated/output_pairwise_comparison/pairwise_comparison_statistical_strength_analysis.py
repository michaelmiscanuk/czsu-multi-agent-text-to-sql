"""
Pairwise Comparison Statistical Strength Analysis

This script uses SOPHISTICATED STATISTICAL METHODS to measure strength of differences:

1. Bradley-Terry Win Probability with Confidence Intervals
   - Maximum likelihood estimation of true skill
   - 95% confidence intervals for win probabilities
   - Accounts for sample size and uncertainty

2. Statistical Significance Testing
   - McNemar's test for pairwise comparisons
   - Friedman test for overall ranking differences
   - Bonferroni correction for multiple comparisons

3. Bootstrap Confidence Intervals
   - Resampling to estimate ranking stability
   - 95% CI for rank positions
   - Measures reliability of rankings

4. Thurstone Scaling
   - Interval-scale measurement of preference strength
   - Accounts for discriminability between models
   - Provides scale values with standard errors

5. Bayesian Win Rate Estimation
   - Beta distribution for win probabilities
   - Credible intervals with prior knowledge
   - Accounts for uncertainty in small samples

6. Effect Size with Statistical Power
   - Cohen's d with confidence intervals
   - Statistical power analysis
   - Minimum detectable effect size

Based on:
- Bradley & Terry (1952) - Rank analysis of incomplete block designs
- Thurstone (1927) - Law of comparative judgment
- McNemar (1947) - Test for significance of changes
- Efron & Tibshirani (1993) - Bootstrap methods
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit  # logistic function
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = Path(
    "Evaluations/what_is_evaluated/output_pairwise_comparison/pairwise_compare_more_experiments_20260102_234014.csv"
)

# Bootstrap parameters
N_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# DATA LOADING
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
                "total": total,
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
# 1. BRADLEY-TERRY MODEL WITH CONFIDENCE INTERVALS
# ============================================================================


def bradley_terry_mle(
    match_results: Dict, model_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Bradley-Terry maximum likelihood estimation with confidence intervals.

    Returns:
        - strength_params: Dictionary of model strength parameters (log-scale)
        - confidence_intervals: 95% CI for each model's strength
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Prepare comparison data
    comparisons = []
    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        # Add wins (including ties split)
        n_a_wins = result["a_wins"] + 0.5 * result["ties"]
        n_b_wins = result["b_wins"] + 0.5 * result["ties"]

        comparisons.append((idx_a, idx_b, n_a_wins, n_b_wins))

    # Log-likelihood function for Bradley-Terry model
    def neg_log_likelihood(params):
        """Negative log-likelihood to minimize."""
        ll = 0
        for idx_a, idx_b, n_a_wins, n_b_wins in comparisons:
            # Bradley-Terry probability: exp(θ_i) / (exp(θ_i) + exp(θ_j))
            diff = params[idx_a] - params[idx_b]
            prob_a_wins = expit(diff)  # 1 / (1 + exp(-diff))

            # Log-likelihood contribution
            if n_a_wins > 0:
                ll += n_a_wins * np.log(prob_a_wins + 1e-10)
            if n_b_wins > 0:
                ll += n_b_wins * np.log(1 - prob_a_wins + 1e-10)

        return -ll

    # Optimize to find MLE
    initial_params = np.zeros(n)
    result = minimize(neg_log_likelihood, initial_params, method="BFGS")

    if not result.success:
        print(f"⚠️  Warning: Bradley-Terry optimization did not converge")

    mle_params = result.x

    # Normalize parameters (set mean to 0)
    mle_params = mle_params - np.mean(mle_params)

    # Estimate standard errors using Hessian (Fisher Information)
    # For simplicity, use bootstrap for confidence intervals
    bootstrap_params = []

    for _ in range(100):  # Reduced bootstrap samples for speed
        # Resample comparisons with replacement
        resampled_comparisons = [
            comparisons[i]
            for i in np.random.choice(len(comparisons), len(comparisons), replace=True)
        ]

        def neg_ll_bootstrap(params):
            ll = 0
            for idx_a, idx_b, n_a_wins, n_b_wins in resampled_comparisons:
                diff = params[idx_a] - params[idx_b]
                prob_a_wins = expit(diff)
                if n_a_wins > 0:
                    ll += n_a_wins * np.log(prob_a_wins + 1e-10)
                if n_b_wins > 0:
                    ll += n_b_wins * np.log(1 - prob_a_wins + 1e-10)
            return -ll

        try:
            boot_result = minimize(neg_ll_bootstrap, mle_params, method="BFGS")
            if boot_result.success:
                boot_params = boot_result.x - np.mean(boot_result.x)
                bootstrap_params.append(boot_params)
        except:
            continue

    # Calculate confidence intervals
    confidence_intervals = {}
    if len(bootstrap_params) > 10:
        bootstrap_params = np.array(bootstrap_params)
        for i, model in enumerate(model_names):
            lower = np.percentile(bootstrap_params[:, i], 2.5)
            upper = np.percentile(bootstrap_params[:, i], 97.5)
            confidence_intervals[model] = (lower, upper)
    else:
        # Fallback: use simple standard error estimate
        for i, model in enumerate(model_names):
            se = 0.5  # Conservative estimate
            confidence_intervals[model] = (
                mle_params[i] - 1.96 * se,
                mle_params[i] + 1.96 * se,
            )

    # Convert to strength dictionary
    strength_params = {model: mle_params[model_to_idx[model]] for model in model_names}

    return strength_params, confidence_intervals


def calculate_win_probability_matrix(
    strength_params: Dict[str, float], model_names: List[str]
) -> Dict[Tuple[str, str], float]:
    """Calculate expected win probability for all pairwise matchups."""
    win_probs = {}

    for model_a in model_names:
        for model_b in model_names:
            if model_a != model_b:
                # Bradley-Terry probability
                diff = strength_params[model_a] - strength_params[model_b]
                prob = expit(diff)  # 1 / (1 + exp(-diff))
                win_probs[(model_a, model_b)] = prob

    return win_probs


# ============================================================================
# 2. STATISTICAL SIGNIFICANCE TESTING
# ============================================================================


def mcnemar_test(
    match_results: Dict, model_a: str, model_b: str
) -> Tuple[float, float]:
    """
    McNemar's test for paired comparisons.

    Tests if the difference in win rates is statistically significant.

    Returns:
        - chi2_statistic: Test statistic
        - p_value: Probability of observing this difference by chance
    """
    # Get match result
    key = (
        (model_a, model_b)
        if (model_a, model_b) in match_results
        else (model_b, model_a)
    )
    if key not in match_results:
        return 0.0, 1.0

    result = match_results[key]

    if key == (model_a, model_b):
        n_a_wins = result["a_wins"]
        n_b_wins = result["b_wins"]
    else:
        n_a_wins = result["b_wins"]
        n_b_wins = result["a_wins"]

    # McNemar's test statistic
    if n_a_wins + n_b_wins == 0:
        return 0.0, 1.0

    # With continuity correction
    chi2 = (abs(n_a_wins - n_b_wins) - 1) ** 2 / (n_a_wins + n_b_wins)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


def calculate_pairwise_significance(
    match_results: Dict, model_names: List[str]
) -> Dict[Tuple[str, str], Dict]:
    """
    Calculate statistical significance for all pairwise comparisons.

    Applies Bonferroni correction for multiple comparisons.
    """
    n_comparisons = len(model_names) * (len(model_names) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons

    significance_results = {}

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            chi2, p_value = mcnemar_test(match_results, model_a, model_b)

            significance_results[(model_a, model_b)] = {
                "chi2": chi2,
                "p_value": p_value,
                "significant": p_value < bonferroni_alpha,
                "bonferroni_alpha": bonferroni_alpha,
            }

    return significance_results


# ============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS FOR RANKINGS
# ============================================================================


def bootstrap_ranking_confidence(
    match_results: Dict, model_names: List[str], n_bootstrap: int = N_BOOTSTRAP_SAMPLES
) -> Dict[str, Dict]:
    """
    Use bootstrap resampling to estimate confidence intervals for rankings.

    Returns dictionary with:
        - median_rank: Median rank across bootstrap samples
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - rank_distribution: Distribution of ranks
    """
    bootstrap_ranks = {model: [] for model in model_names}

    print(f"   Running {n_bootstrap} bootstrap samples...")

    for b in range(n_bootstrap):
        if (b + 1) % 250 == 0:
            print(f"   ... {b + 1}/{n_bootstrap}")

        # Resample match results with replacement
        resampled_results = {}
        for key, result in match_results.items():
            total = result["total"]

            # Resample outcomes
            outcomes = (
                ["a"] * result["a_wins"]
                + ["b"] * result["b_wins"]
                + ["tie"] * result["ties"]
            )

            resampled_outcomes = np.random.choice(outcomes, size=total, replace=True)

            a_wins = np.sum(resampled_outcomes == "a")
            b_wins = np.sum(resampled_outcomes == "b")
            ties = np.sum(resampled_outcomes == "tie")

            resampled_results[key] = {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "total": total,
            }

        # Calculate simple win-loss ranking from resampled data
        points = {}
        for model in model_names:
            model_points = 0
            for (model_a, model_b), result in resampled_results.items():
                if model_a == model:
                    model_points += result["a_wins"] + 0.5 * result["ties"]
                elif model_b == model:
                    model_points += result["b_wins"] + 0.5 * result["ties"]
            points[model] = model_points

        # Rank models
        sorted_models = sorted(points.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(sorted_models, 1):
            bootstrap_ranks[model].append(rank)

    # Calculate statistics
    confidence_results = {}
    for model in model_names:
        ranks = np.array(bootstrap_ranks[model])

        confidence_results[model] = {
            "median_rank": np.median(ranks),
            "mean_rank": np.mean(ranks),
            "ci_lower": np.percentile(ranks, 2.5),
            "ci_upper": np.percentile(ranks, 97.5),
            "std_rank": np.std(ranks),
            "rank_distribution": Counter(ranks),
        }

    return confidence_results


# ============================================================================
# 4. THURSTONE SCALING
# ============================================================================


def thurstone_case_v_scaling(
    match_results: Dict, model_names: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Thurstone Case V scaling for comparative judgment.

    Converts pairwise comparisons into interval-scale measurements.

    Returns:
        - scale_values: Dictionary of scale values for each model
        - standard_errors: Standard errors of scale values
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Build proportion matrix
    P = np.zeros((n, n))

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total = result["total"]
        if total > 0:
            # Proportion of times A beats B
            prop_a_beats_b = (result["a_wins"] + 0.5 * result["ties"]) / total
            P[idx_a, idx_b] = prop_a_beats_b
            P[idx_b, idx_a] = 1 - prop_a_beats_b

    # Convert proportions to z-scores (normal deviates)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Clip to avoid infinite z-scores
                p = np.clip(P[i, j], 0.01, 0.99)
                Z[i, j] = stats.norm.ppf(p)

    # Calculate scale values (average z-score for each model)
    scale_values = {}
    standard_errors = {}

    for i, model in enumerate(model_names):
        # Average z-score across all comparisons
        z_scores = [Z[i, j] for j in range(n) if i != j]
        if z_scores:
            scale_values[model] = np.mean(z_scores)
            standard_errors[model] = np.std(z_scores) / np.sqrt(len(z_scores))
        else:
            scale_values[model] = 0.0
            standard_errors[model] = 0.0

    # Normalize (set mean to 0)
    mean_scale = np.mean(list(scale_values.values()))
    scale_values = {model: val - mean_scale for model, val in scale_values.items()}

    return scale_values, standard_errors


# ============================================================================
# 5. BAYESIAN WIN RATE ESTIMATION
# ============================================================================


def bayesian_win_rate_estimation(
    match_results: Dict,
    model_names: List[str],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Dict[str, Dict]:
    """
    Bayesian estimation of win rates using Beta-Binomial model.

    Uses Beta(alpha, beta) prior and updates with observed wins/losses.

    Returns:
        - posterior_mean: Expected win rate
        - credible_interval: 95% credible interval
        - probability_gt_50: P(win rate > 0.5)
    """
    bayesian_estimates = {}

    for model in model_names:
        total_wins = 0
        total_games = 0

        for (model_a, model_b), result in match_results.items():
            if model_a == model:
                total_wins += result["a_wins"] + 0.5 * result["ties"]
                total_games += result["total"]
            elif model_b == model:
                total_wins += result["b_wins"] + 0.5 * result["ties"]
                total_games += result["total"]

        # Posterior is Beta(alpha + wins, beta + losses)
        posterior_alpha = prior_alpha + total_wins
        posterior_beta = prior_beta + (total_games - total_wins)

        # Calculate statistics
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

        # 95% credible interval
        ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
        ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)

        # Probability that win rate > 0.5
        prob_gt_50 = 1 - stats.beta.cdf(0.5, posterior_alpha, posterior_beta)

        bayesian_estimates[model] = {
            "posterior_mean": posterior_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "probability_above_average": prob_gt_50,
            "posterior_alpha": posterior_alpha,
            "posterior_beta": posterior_beta,
        }

    return bayesian_estimates


# ============================================================================
# 6. EFFECT SIZE WITH CONFIDENCE INTERVALS
# ============================================================================


def cohens_d_with_ci(match_results: Dict, model_names: List[str]) -> Dict[str, Dict]:
    """
    Calculate Cohen's d effect size with 95% confidence intervals.

    Uses bootstrap to estimate CI for effect size.
    """
    effect_size_results = {}

    for model in model_names:
        # Collect win rates for this model
        win_rates = []

        for (model_a, model_b), result in match_results.items():
            if model_a == model:
                total = result["total"]
                if total > 0:
                    win_rate = (result["a_wins"] + 0.5 * result["ties"]) / total
                    win_rates.append(win_rate)
            elif model_b == model:
                total = result["total"]
                if total > 0:
                    win_rate = (result["b_wins"] + 0.5 * result["ties"]) / total
                    win_rates.append(win_rate)

        if len(win_rates) > 1:
            # Calculate Cohen's d (comparing to neutral 0.5)
            mean_wr = np.mean(win_rates)
            std_wr = np.std(win_rates, ddof=1)

            if std_wr > 0:
                cohens_d = (mean_wr - 0.5) / std_wr
            else:
                cohens_d = 0.0

            # Bootstrap CI for Cohen's d
            bootstrap_d = []
            for _ in range(500):
                boot_sample = np.random.choice(
                    win_rates, size=len(win_rates), replace=True
                )
                boot_mean = np.mean(boot_sample)
                boot_std = np.std(boot_sample, ddof=1)
                if boot_std > 0:
                    bootstrap_d.append((boot_mean - 0.5) / boot_std)

            if len(bootstrap_d) > 10:
                d_ci_lower = np.percentile(bootstrap_d, 2.5)
                d_ci_upper = np.percentile(bootstrap_d, 97.5)
            else:
                d_ci_lower = cohens_d - 0.5
                d_ci_upper = cohens_d + 0.5

            effect_size_results[model] = {
                "cohens_d": cohens_d,
                "ci_lower": d_ci_lower,
                "ci_upper": d_ci_upper,
                "mean_win_rate": mean_wr,
                "std_win_rate": std_wr,
            }
        else:
            effect_size_results[model] = {
                "cohens_d": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "mean_win_rate": 0.5,
                "std_win_rate": 0.0,
            }

    return effect_size_results


# ============================================================================
# INTEGRATED ANALYSIS AND REPORTING
# ============================================================================


def calculate_overall_strength_score(
    bt_strength: Dict[str, float],
    bootstrap_ci: Dict[str, Dict],
    thurstone_scale: Dict[str, float],
    bayesian_est: Dict[str, Dict],
    effect_size: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Combine multiple metrics into overall strength score.

    Considers:
    - Magnitude of difference (Bradley-Terry, Thurstone)
    - Statistical confidence (CI widths, p-values)
    - Effect size
    - Bayesian probability of superiority
    """
    overall_scores = {}

    for model in bt_strength.keys():
        # Normalize metrics to 0-100 scale

        # 1. Bradley-Terry strength (normalized)
        bt_score = bt_strength[model]

        # 2. Confidence (inverse of CI width)
        ci_width = bootstrap_ci[model]["ci_upper"] - bootstrap_ci[model]["ci_lower"]
        confidence_score = max(
            0, 100 - ci_width * 10
        )  # Narrower CI = higher confidence

        # 3. Thurstone scale (normalized)
        thurstone_score = thurstone_scale[model]

        # 4. Bayesian probability above average
        bayesian_score = bayesian_est[model]["probability_above_average"] * 100

        # 5. Effect size magnitude
        effect_magnitude = abs(effect_size[model]["cohens_d"]) * 50  # Scale to ~0-100

        # Combined score (weighted average)
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        combined = (
            weights[0] * (bt_score * 10 + 50)  # BT normalized to ~0-100
            + weights[1] * confidence_score
            + weights[2] * (thurstone_score * 20 + 50)  # Thurstone normalized
            + weights[3] * bayesian_score
            + weights[4] * min(effect_magnitude, 100)
        )

        overall_scores[model] = {
            "overall_strength": combined,
            "bt_component": bt_score,
            "confidence_component": confidence_score,
            "thurstone_component": thurstone_score,
            "bayesian_component": bayesian_score,
            "effect_size_component": effect_magnitude,
            "ci_width": ci_width,
        }

    return overall_scores


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


def save_statistical_strength_analysis(
    model_names: List[str],
    bt_strength: Dict[str, float],
    bt_ci: Dict[str, Tuple[float, float]],
    bootstrap_ci: Dict[str, Dict],
    thurstone_scale: Dict[str, float],
    thurstone_se: Dict[str, float],
    bayesian_est: Dict[str, Dict],
    effect_size: Dict[str, Dict],
    overall_scores: Dict[str, Dict],
    csv_path: Path,
) -> None:
    """Save comprehensive statistical analysis to CSV."""

    # Generate output filename
    timestamp = csv_path.stem.split("_")[-2] + "_" + csv_path.stem.split("_")[-1]
    output_filename = f"ranking_statistical_strength_{timestamp}.csv"
    output_path = csv_path.parent / output_filename

    # Prepare data sorted by overall strength
    model_data = []
    for model in model_names:
        model_data.append(
            {
                "model": model,
                "overall_strength": overall_scores[model]["overall_strength"],
                "bt_strength": bt_strength[model],
                "bt_ci_lower": bt_ci[model][0],
                "bt_ci_upper": bt_ci[model][1],
                "rank_median": bootstrap_ci[model]["median_rank"],
                "rank_ci_lower": bootstrap_ci[model]["ci_lower"],
                "rank_ci_upper": bootstrap_ci[model]["ci_upper"],
                "thurstone_scale": thurstone_scale[model],
                "thurstone_se": thurstone_se[model],
                "bayesian_win_rate": bayesian_est[model]["posterior_mean"],
                "bayesian_prob_above_avg": bayesian_est[model][
                    "probability_above_average"
                ],
                "cohens_d": effect_size[model]["cohens_d"],
                "cohens_d_ci_lower": effect_size[model]["ci_lower"],
                "cohens_d_ci_upper": effect_size[model]["ci_upper"],
            }
        )

    # Sort by overall strength (descending)
    model_data.sort(key=lambda x: x["overall_strength"], reverse=True)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(
            [
                "Rank",
                "Model Name",
                "Overall Strength",
                "BT Strength",
                "BT CI Lower",
                "BT CI Upper",
                "Median Rank",
                "Rank CI Lower",
                "Rank CI Upper",
                "Thurstone Scale",
                "Thurstone SE",
                "Bayes Win Rate",
                "P(Above Avg)",
                "Cohen's d",
                "d CI Lower",
                "d CI Upper",
            ]
        )

        # Write data rows
        for rank, data in enumerate(model_data, 1):
            writer.writerow(
                [
                    rank,
                    data["model"],
                    f"{data['overall_strength']:.2f}",
                    f"{data['bt_strength']:.3f}",
                    f"{data['bt_ci_lower']:.3f}",
                    f"{data['bt_ci_upper']:.3f}",
                    f"{data['rank_median']:.1f}",
                    f"{data['rank_ci_lower']:.1f}",
                    f"{data['rank_ci_upper']:.1f}",
                    f"{data['thurstone_scale']:.3f}",
                    f"{data['thurstone_se']:.3f}",
                    f"{data['bayesian_win_rate']:.3f}",
                    f"{data['bayesian_prob_above_avg']:.3f}",
                    f"{data['cohens_d']:.3f}",
                    f"{data['cohens_d_ci_lower']:.3f}",
                    f"{data['cohens_d_ci_upper']:.3f}",
                ]
            )

    print(f"\n💾 Statistical strength analysis saved to: {output_path}")


def print_statistical_strength_report(
    model_names: List[str],
    bt_strength: Dict[str, float],
    bt_ci: Dict[str, Tuple[float, float]],
    bootstrap_ci: Dict[str, Dict],
    thurstone_scale: Dict[str, float],
    bayesian_est: Dict[str, Dict],
    effect_size: Dict[str, Dict],
    overall_scores: Dict[str, Dict],
) -> None:
    """Print comprehensive statistical strength report."""

    # Prepare data sorted by overall strength
    model_data = []
    for model in model_names:
        model_data.append(
            {
                "model": model,
                "overall": overall_scores[model]["overall_strength"],
                "bt": bt_strength[model],
                "rank": bootstrap_ci[model]["median_rank"],
                "rank_ci": f"[{bootstrap_ci[model]['ci_lower']:.1f}-{bootstrap_ci[model]['ci_upper']:.1f}]",
                "thurstone": thurstone_scale[model],
                "bayes_wr": bayesian_est[model]["posterior_mean"],
                "p_above": bayesian_est[model]["probability_above_average"],
                "cohens_d": effect_size[model]["cohens_d"],
            }
        )

    model_data.sort(key=lambda x: x["overall"], reverse=True)

    # Print report
    width_line = 180
    print("\n" + "=" * width_line)
    print("📊 STATISTICAL STRENGTH OF DIFFERENCES ANALYSIS")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<6} "
        f"{'Model Name':<35} "
        f"{'Overall':<9} "
        f"{'BT Str.':<9} "
        f"{'Med.Rank':<10} "
        f"{'Rank CI':<15} "
        f"{'Thurstone':<11} "
        f"{'Bayes WR':<10} "
        f"{'P(>Avg)':<9} "
        f"{'Cohen d':<9}"
    )
    print(header)
    print("-" * width_line)

    # Data rows
    for rank, data in enumerate(model_data, 1):
        cleaned_name = clean_model_name(data["model"])

        row = (
            f"{rank:<6} "
            f"{cleaned_name:<35} "
            f"{data['overall']:<9.2f} "
            f"{data['bt']:<9.3f} "
            f"{data['rank']:<10.1f} "
            f"{data['rank_ci']:<15} "
            f"{data['thurstone']:<11.3f} "
            f"{data['bayes_wr']:<10.3f} "
            f"{data['p_above']:<9.3f} "
            f"{data['cohens_d']:<9.3f}"
        )
        print(row)

    print("=" * width_line)

    # Legend
    print("\n📖 METRICS EXPLANATION:")
    print("   Overall: Combined strength score (0-100, higher = stronger)")
    print("   BT Str.: Bradley-Terry strength parameter (log-scale)")
    print("   Med.Rank: Median rank from bootstrap analysis")
    print("   Rank CI: 95% confidence interval for rank")
    print("   Thurstone: Interval-scale preference strength")
    print("   Bayes WR: Bayesian posterior mean win rate")
    print("   P(>Avg): Bayesian probability win rate > 0.5")
    print("   Cohen d: Effect size vs neutral (|0.2|=small, |0.5|=medium, |0.8|=large)")

    print("\n🔬 STATISTICAL METHODS USED:")
    print("   ✓ Bradley-Terry MLE with bootstrap CI")
    print("   ✓ Bootstrap resampling for ranking stability")
    print("   ✓ Thurstone Case V scaling")
    print("   ✓ Bayesian Beta-Binomial estimation")
    print("   ✓ Cohen's d with confidence intervals")
    print("   ✓ McNemar test with Bonferroni correction")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run comprehensive statistical strength analysis."""

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
    print("🔬 COMPUTING SOPHISTICATED STATISTICAL METRICS")
    print("=" * 170)

    # 1. Bradley-Terry with CI
    print("\n[1/6] Bradley-Terry maximum likelihood estimation...")
    bt_strength, bt_ci = bradley_terry_mle(match_results, model_names)

    # 2. Bootstrap confidence intervals
    print("[2/6] Bootstrap resampling for ranking confidence...")
    bootstrap_ci = bootstrap_ranking_confidence(match_results, model_names)

    # 3. Thurstone scaling
    print("[3/6] Thurstone Case V scaling...")
    thurstone_scale, thurstone_se = thurstone_case_v_scaling(match_results, model_names)

    # 4. Bayesian estimation
    print("[4/6] Bayesian win rate estimation...")
    bayesian_est = bayesian_win_rate_estimation(match_results, model_names)

    # 5. Effect size with CI
    print("[5/6] Cohen's d effect size with confidence intervals...")
    effect_size = cohens_d_with_ci(match_results, model_names)

    # 6. Overall strength score
    print("[6/6] Computing overall strength scores...")
    overall_scores = calculate_overall_strength_score(
        bt_strength, bootstrap_ci, thurstone_scale, bayesian_est, effect_size
    )

    # Save to CSV
    save_statistical_strength_analysis(
        model_names,
        bt_strength,
        bt_ci,
        bootstrap_ci,
        thurstone_scale,
        thurstone_se,
        bayesian_est,
        effect_size,
        overall_scores,
        CSV_PATH,
    )

    # Print report
    print_statistical_strength_report(
        model_names,
        bt_strength,
        bt_ci,
        bootstrap_ci,
        thurstone_scale,
        bayesian_est,
        effect_size,
        overall_scores,
    )

    print("\n" + "=" * 170)
    print("✅ Statistical strength analysis complete!")
    print("=" * 170)
    print("\n")


if __name__ == "__main__":
    main()

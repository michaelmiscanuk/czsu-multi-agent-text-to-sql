"""
Pairwise Comparison Analysis using Multiple Ranking Methods

This script implements 15 different ranking methods for pairwise comparisons:
1. Simple Win Counting - basic points system (Win=1, Tie=0.5, Loss=0)
2. Bradley-Terry Model - accounts for opponent strength (using choix library)
3. Luce Spectral Ranking - eigenvector-based method (using choix library)
4. Rank Centrality - PageRank-like algorithm (using choix library)
5. Elo Rating - dynamic rating system from chess
6. Colley Matrix - linear algebra approach
7. Massey Method - least squares rating system
8. Glicko-2 Rating - Enhanced Elo with rating deviation
9. TrueSkill - Bayesian ranking (Microsoft's system)
10. Copeland Method - Pairwise wins minus losses
11. Borda Count - Positional voting adapted for pairwise
12. Keener Method - Iterative strength-weighted ranking
13. Offense-Defense Rating - Separate offensive/defensive strengths
14. Markov Rank - Stationary distribution of random walks
15. Recursive Buchholz - Strength of schedule based ranking

Required libraries:
    pip install choix numpy scipy trueskill glicko2
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigs

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
    "Evaluations\what_is_evaluated\output_pairwise_comparison\pairwise_compare_more_experiments_20251222_110500.csv"
)

# ============================================================================
# DATA LOADING
# ============================================================================


def load_pairwise_data(csv_path: Path) -> Tuple[Dict, Dict, List[str]]:
    """
    Load pairwise comparison data from CSV.

    Returns:
        - match_results: Dict mapping (model_a, model_b) to results
        - model_stats: Dict with win/loss/tie counts per model
        - model_names: Sorted list of all model names
    """
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

            # Store match results
            match_results[(exp_a, exp_b)] = {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
            }

            # Update model stats
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
# METHOD 1: SIMPLE WIN COUNTING
# ============================================================================


def simple_win_counting(model_stats: Dict) -> Dict[str, float]:
    """Win=1, Tie=0.5, Loss=0. Sum all points."""
    ratings = {}
    for model, stats in model_stats.items():
        points = stats["wins"] + 0.5 * stats["ties"]
        ratings[model] = points
    return ratings


# ============================================================================
# METHOD 2: BRADLEY-TERRY MODEL (using choix library)
# ============================================================================


def bradley_terry_choix(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """Bradley-Terry model using choix library."""
    if not CHOIX_AVAILABLE:
        return {model: 0.0 for model in model_names}

    # Create model index mapping
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Prepare data for choix (list of comparisons)
    data = []
    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        # Add wins for A
        for _ in range(result["a_wins"]):
            data.append((idx_a, idx_b))

        # Add wins for B
        for _ in range(result["b_wins"]):
            data.append((idx_b, idx_a))

        # Add ties (split as half-wins for each)
        ties = result["ties"]
        if ties > 0:
            # For ties, add half to each side (approximate)
            for _ in range(ties // 2):
                data.append((idx_a, idx_b))
                data.append((idx_b, idx_a))

    # Compute Bradley-Terry parameters using MM algorithm
    params = choix.ilsr_pairwise(len(model_names), data, alpha=0.01)

    # Convert to ratings (exponential of parameters)
    ratings = {model: np.exp(params[model_to_idx[model]]) for model in model_names}

    return ratings


# ============================================================================
# METHOD 3: LUCE SPECTRAL RANKING (using choix library)
# ============================================================================


def luce_spectral_ranking(
    match_results: Dict, model_names: List[str]
) -> Dict[str, float]:
    """Luce Spectral Ranking - eigenvector-based method."""
    if not CHOIX_AVAILABLE:
        return {model: 0.0 for model in model_names}

    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Prepare data
    data = []
    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        for _ in range(result["a_wins"]):
            data.append((idx_a, idx_b))
        for _ in range(result["b_wins"]):
            data.append((idx_b, idx_a))
        for _ in range(result["ties"] // 2):
            data.append((idx_a, idx_b))
            data.append((idx_b, idx_a))

    # Compute Luce Spectral Ranking
    params = choix.lsr_pairwise(len(model_names), data, alpha=0.01)

    ratings = {model: params[model_to_idx[model]] for model in model_names}

    return ratings


# ============================================================================
# METHOD 4: RANK CENTRALITY (using choix library)
# ============================================================================


def rank_centrality(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """Rank Centrality - similar to PageRank algorithm."""
    if not CHOIX_AVAILABLE:
        return {model: 0.0 for model in model_names}

    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Prepare data
    data = []
    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        for _ in range(result["a_wins"]):
            data.append((idx_a, idx_b))
        for _ in range(result["b_wins"]):
            data.append((idx_b, idx_a))
        for _ in range(result["ties"] // 2):
            data.append((idx_a, idx_b))
            data.append((idx_b, idx_a))

    # Compute Rank Centrality
    params = choix.rank_centrality(len(model_names), data, alpha=0.01)

    ratings = {model: params[model_to_idx[model]] for model in model_names}

    return ratings


# ============================================================================
# METHOD 5: ELO RATING
# ============================================================================


def elo_rating(
    match_results: Dict, model_names: List[str], k_factor: int = 32
) -> Dict[str, float]:
    """
    Elo rating system.

    Args:
        k_factor: Sensitivity parameter (higher = more volatile ratings)
    """
    # Initialize all models with rating 1500
    ratings = {model: 1500.0 for model in model_names}

    # Process each match
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


# ============================================================================
# METHOD 6: COLLEY MATRIX
# ============================================================================


def colley_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Colley Matrix method - linear algebra approach.

    Rating is based on wins/losses adjusted for number of games.
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Initialize Colley matrix (diagonal = 2 + games played)
    # and b vector (1 + (wins - losses) / 2)
    C = np.zeros((n, n))
    b = np.ones(n)

    # Track games played and win differential
    games_played = {model: 0 for model in model_names}
    win_diff = {model: 0 for model in model_names}

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total_games = result["a_wins"] + result["b_wins"] + result["ties"]
        games_played[model_a] += total_games
        games_played[model_b] += total_games

        # Win differential (ties count as 0)
        win_diff[model_a] += result["a_wins"] - result["b_wins"]
        win_diff[model_b] += result["b_wins"] - result["a_wins"]

        # Off-diagonal elements: -1 for each game played
        C[idx_a, idx_b] -= total_games
        C[idx_b, idx_a] -= total_games

    # Set diagonal and b vector
    for model in model_names:
        idx = model_to_idx[model]
        C[idx, idx] = 2 + games_played[model]
        b[idx] = 1 + win_diff[model] / 2

    # Solve C * r = b
    ratings_array = np.linalg.solve(C, b)

    # Convert to ratings (scale to be more interpretable)
    ratings = {model: ratings_array[model_to_idx[model]] * 100 for model in model_names}

    return ratings


# ============================================================================
# METHOD 7: MASSEY METHOD
# ============================================================================


def massey_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Massey rating method - least squares approach.

    Minimizes the sum of squared prediction errors.
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Initialize Massey matrix and point differential vector
    M = np.zeros((n, n))
    p = np.zeros(n)

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total_games = result["a_wins"] + result["b_wins"] + result["ties"]
        point_diff = result["a_wins"] - result["b_wins"]  # Positive if A wins more

        # Update diagonal
        M[idx_a, idx_a] += total_games
        M[idx_b, idx_b] += total_games

        # Update off-diagonal
        M[idx_a, idx_b] -= total_games
        M[idx_b, idx_a] -= total_games

        # Update point differential
        p[idx_a] += point_diff
        p[idx_b] -= point_diff

    # Massey matrix is singular, so we replace last row with constraint that sum(ratings) = 0
    M[-1, :] = 1
    p[-1] = 0

    # Solve M * r = p
    ratings_array = np.linalg.solve(M, p)

    # Convert to ratings (shift so minimum is 0 and scale for readability)
    min_rating = ratings_array.min()
    ratings = {
        model: (ratings_array[model_to_idx[model]] - min_rating) * 10
        for model in model_names
    }

    return ratings


# ============================================================================
# METHOD 8: GLICKO-2 RATING
# ============================================================================


def glicko2_rating(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Glicko-2 rating system - enhanced Elo with rating deviation.

    Accounts for uncertainty in ratings through rating deviation (RD).
    """
    if not GLICKO2_AVAILABLE:
        # Fallback to simple implementation
        ratings = {model: 1500.0 for model in model_names}
        rd = {model: 350.0 for model in model_names}

        for (model_a, model_b), result in match_results.items():
            for _ in range(result["a_wins"]):
                ratings[model_a], ratings[model_b] = update_glicko2_simple(
                    ratings[model_a], ratings[model_b], 1.0, rd[model_a], rd[model_b]
                )
            for _ in range(result["b_wins"]):
                ratings[model_a], ratings[model_b] = update_glicko2_simple(
                    ratings[model_a], ratings[model_b], 0.0, rd[model_a], rd[model_b]
                )
            for _ in range(result["ties"]):
                ratings[model_a], ratings[model_b] = update_glicko2_simple(
                    ratings[model_a], ratings[model_b], 0.5, rd[model_a], rd[model_b]
                )
        return ratings

    # Use glicko2 library
    players = {model: glicko2.Player() for model in model_names}

    for (model_a, model_b), result in match_results.items():
        for _ in range(result["a_wins"]):
            players[model_a].update_player(
                [players[model_b].rating], [players[model_b].rd], [1]
            )
            players[model_b].update_player(
                [players[model_a].rating], [players[model_a].rd], [0]
            )
        for _ in range(result["b_wins"]):
            players[model_a].update_player(
                [players[model_b].rating], [players[model_b].rd], [0]
            )
            players[model_b].update_player(
                [players[model_a].rating], [players[model_a].rd], [1]
            )
        for _ in range(result["ties"]):
            players[model_a].update_player(
                [players[model_b].rating], [players[model_b].rd], [0.5]
            )
            players[model_b].update_player(
                [players[model_a].rating], [players[model_a].rd], [0.5]
            )

    ratings = {model: players[model].rating for model in model_names}
    return ratings


def update_glicko2_simple(
    rating_a: float, rating_b: float, score_a: float, rd_a: float, rd_b: float
) -> Tuple[float, float]:
    """Simplified Glicko-2 update (fallback when library unavailable)."""
    q = np.log(10) / 400
    g_rd = 1 / np.sqrt(1 + 3 * q**2 * rd_b**2 / np.pi**2)
    expected_a = 1 / (1 + 10 ** (-g_rd * (rating_a - rating_b) / 400))

    d2 = 1 / (q**2 * g_rd**2 * expected_a * (1 - expected_a))
    new_rating_a = rating_a + (q / (1 / rd_a**2 + 1 / d2)) * g_rd * (
        score_a - expected_a
    )
    new_rating_b = rating_b + (q / (1 / rd_b**2 + 1 / d2)) * g_rd * (
        (1 - score_a) - (1 - expected_a)
    )

    return new_rating_a, new_rating_b


# ============================================================================
# METHOD 9: TRUESKILL
# ============================================================================


def trueskill_rating(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    TrueSkill - Bayesian ranking system developed by Microsoft.

    Uses Gaussian distributions to model player skill with uncertainty.
    """
    if not TRUESKILL_AVAILABLE:
        # Fallback to Elo-like system
        return elo_rating(match_results, model_names, k_factor=32)

    # Initialize TrueSkill environment and players
    env = trueskill.TrueSkill(draw_probability=0.1)
    players = {model: env.create_rating() for model in model_names}

    for (model_a, model_b), result in match_results.items():
        # Process wins for A
        for _ in range(result["a_wins"]):
            new_a, new_b = env.rate_1vs1(players[model_a], players[model_b])
            players[model_a] = new_a
            players[model_b] = new_b

        # Process wins for B
        for _ in range(result["b_wins"]):
            new_b, new_a = env.rate_1vs1(players[model_b], players[model_a])
            players[model_a] = new_a
            players[model_b] = new_b

        # Process ties
        for _ in range(result["ties"]):
            new_a, new_b = env.rate_1vs1(players[model_a], players[model_b], drawn=True)
            players[model_a] = new_a
            players[model_b] = new_b

    # Use conservative skill estimate (mu - 3*sigma)
    ratings = {model: env.expose(players[model]) for model in model_names}
    return ratings


# ============================================================================
# METHOD 10: COPELAND METHOD
# ============================================================================


def copeland_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Copeland method - counts pairwise wins minus losses.

    Simple but effective tournament ranking method.
    """
    copeland_scores = {model: 0.0 for model in model_names}

    for (model_a, model_b), result in match_results.items():
        a_wins = result["a_wins"]
        b_wins = result["b_wins"]

        if a_wins > b_wins:
            copeland_scores[model_a] += 1
            copeland_scores[model_b] -= 1
        elif b_wins > a_wins:
            copeland_scores[model_b] += 1
            copeland_scores[model_a] -= 1
        # Ties contribute 0

    return copeland_scores


# ============================================================================
# METHOD 11: BORDA COUNT
# ============================================================================


def borda_count_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Borda count - positional voting method adapted for pairwise comparisons.

    Each model gets points based on how many opponents it beats.
    """
    borda_points = {model: 0.0 for model in model_names}

    # For each model, count how many opponents it beats
    for (model_a, model_b), result in match_results.items():
        total_a = result["a_wins"]
        total_b = result["b_wins"]
        ties = result["ties"]

        # Award points proportional to win rate in this matchup
        if total_a + total_b + ties > 0:
            a_score = (total_a + 0.5 * ties) / (total_a + total_b + ties)
            b_score = (total_b + 0.5 * ties) / (total_a + total_b + ties)

            borda_points[model_a] += a_score
            borda_points[model_b] += b_score

    return borda_points


# ============================================================================
# METHOD 12: KEENER METHOD
# ============================================================================


def keener_method(
    match_results: Dict, model_names: List[str], max_iter: int = 100
) -> Dict[str, float]:
    """
    Keener method - iterative strength-weighted ranking.

    Uses a nonlinear perturbation to handle teams with no wins/losses.
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Build win matrix
    wins = np.zeros((n, n))
    games = np.zeros((n, n))

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total = result["a_wins"] + result["b_wins"] + result["ties"]
        wins[idx_a, idx_b] = result["a_wins"] + 0.5 * result["ties"]
        wins[idx_b, idx_a] = result["b_wins"] + 0.5 * result["ties"]
        games[idx_a, idx_b] = total
        games[idx_b, idx_a] = total

    # Keener's method with perturbation function h(x) = x + epsilon
    epsilon = 0.01
    ratings = np.ones(n)

    for _ in range(max_iter):
        old_ratings = ratings.copy()
        new_ratings = np.zeros(n)

        for i in range(n):
            numerator = 0
            denominator = 0
            for j in range(n):
                if i != j and games[i, j] > 0:
                    score_ij = wins[i, j] / games[i, j] if games[i, j] > 0 else 0.5
                    h_score = score_ij + epsilon  # Perturbation
                    numerator += games[i, j] * h_score * old_ratings[j]
                    denominator += games[i, j] * old_ratings[j]

            if denominator > 0:
                new_ratings[i] = numerator / denominator
            else:
                new_ratings[i] = 1.0

        # Normalize
        if new_ratings.sum() > 0:
            ratings = new_ratings / new_ratings.sum() * n

        # Check convergence
        if np.max(np.abs(ratings - old_ratings)) < 1e-6:
            break

    return {model: ratings[model_to_idx[model]] for model in model_names}


# ============================================================================
# METHOD 13: OFFENSE-DEFENSE RATING
# ============================================================================


def offense_defense_rating(
    match_results: Dict, model_names: List[str], max_iter: int = 100
) -> Dict[str, float]:
    """
    Offense-Defense rating - separate offensive and defensive strengths.

    Final rating is geometric mean of offense and defense ratings.
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Initialize ratings
    offense = np.ones(n)
    defense = np.ones(n)

    for iteration in range(max_iter):
        old_offense = offense.copy()
        old_defense = defense.copy()

        new_offense = np.zeros(n)
        new_defense = np.zeros(n)

        for (model_a, model_b), result in match_results.items():
            idx_a = model_to_idx[model_a]
            idx_b = model_to_idx[model_b]

            total = result["a_wins"] + result["b_wins"] + result["ties"]
            if total == 0:
                continue

            # Offense rating: ability to score against opponent's defense
            new_offense[idx_a] += result["a_wins"] / (old_defense[idx_b] + 1e-6)
            new_offense[idx_b] += result["b_wins"] / (old_defense[idx_a] + 1e-6)

            # Defense rating: ability to prevent opponent from scoring
            new_defense[idx_a] += result["b_wins"] / (old_offense[idx_b] + 1e-6)
            new_defense[idx_b] += result["a_wins"] / (old_offense[idx_a] + 1e-6)

        # Normalize
        if new_offense.sum() > 0:
            offense = new_offense / new_offense.mean()
        if new_defense.sum() > 0:
            defense = 1.0 / (
                new_defense / new_defense.mean()
            )  # Lower is better for defense

        # Check convergence
        if (
            np.max(np.abs(offense - old_offense)) < 1e-6
            and np.max(np.abs(defense - old_defense)) < 1e-6
        ):
            break

    # Combined rating: geometric mean of offense and defense
    ratings = {
        model: np.sqrt(offense[model_to_idx[model]] * defense[model_to_idx[model]])
        for model in model_names
    }

    return ratings


# ============================================================================
# METHOD 14: MARKOV RANK
# ============================================================================


def markov_rank_method(match_results: Dict, model_names: List[str]) -> Dict[str, float]:
    """
    Markov Rank - stationary distribution of random walks on comparison graph.

    Similar to PageRank but specialized for pairwise comparisons.
    """
    n = len(model_names)
    model_to_idx = {model: idx for idx, model in enumerate(model_names)}

    # Build transition matrix
    P = np.zeros((n, n))

    for (model_a, model_b), result in match_results.items():
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]

        total = result["a_wins"] + result["b_wins"] + result["ties"]
        if total == 0:
            continue

        # Probability of transitioning from loser to winner
        a_win_prob = (result["a_wins"] + 0.5 * result["ties"]) / total
        b_win_prob = (result["b_wins"] + 0.5 * result["ties"]) / total

        P[idx_b, idx_a] += a_win_prob  # Transition from b to a when a wins
        P[idx_a, idx_b] += b_win_prob  # Transition from a to b when b wins

    # Normalize rows (make it a proper transition matrix)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = P / row_sums

    # Add damping factor (like PageRank)
    damping = 0.85
    P = damping * P + (1 - damping) / n * np.ones((n, n))

    # Find stationary distribution (principal eigenvector)
    eigenvalues, eigenvectors = eigs(P.T, k=1, which="LM")
    stationary = np.abs(eigenvectors[:, 0].real)
    stationary = stationary / stationary.sum()

    ratings = {model: stationary[model_to_idx[model]] * 100 for model in model_names}
    return ratings


# ============================================================================
# METHOD 15: RECURSIVE BUCHHOLZ
# ============================================================================


def recursive_buchholz_method(
    match_results: Dict, model_names: List[str], depth: int = 3
) -> Dict[str, float]:
    """
    Recursive Buchholz - strength of schedule based ranking.

    Recursively considers the strength of opponents (and opponents' opponents).
    """
    n = len(model_names)

    # Start with simple win rates
    ratings = simple_win_counting(
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

    # Iteratively refine ratings based on opponent strength
    for iteration in range(depth):
        new_ratings = {}

        for model in model_names:
            # Get all opponents
            opponents = set()
            for (model_a, model_b), result in match_results.items():
                if model_a == model:
                    opponents.add(model_b)
                elif model_b == model:
                    opponents.add(model_a)

            # Buchholz score: sum of opponent ratings
            if opponents:
                buchholz_score = sum(ratings.get(opp, 0) for opp in opponents) / len(
                    opponents
                )
                # Combine own performance with opponent strength
                own_score = ratings.get(model, 0)
                new_ratings[model] = 0.7 * own_score + 0.3 * buchholz_score
            else:
                new_ratings[model] = ratings.get(model, 0)

        ratings = new_ratings

    return ratings


# ============================================================================
# REPORTING
# ============================================================================


def clean_model_name(model: str) -> str:
    """
    Clean model name by removing prefix and suffix.

    Removes:
    - Prefix: "judge-azureopenai-gpt-4.1__node-format_answer_node__model-"
    - Suffix: Everything after the last "-" (e.g., "-6873e5c9")
    """
    prefix = "judge-azureopenai-gpt-4.1__node-format_answer_node__model-"

    # Remove prefix if present
    cleaned = model
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :]

    # Remove suffix after last "-"
    if "-" in cleaned:
        parts = cleaned.rsplit("-", 1)
        # Check if the last part looks like a hash (alphanumeric, typically 8 chars)
        if len(parts[1]) <= 10 and parts[1].isalnum():
            cleaned = parts[0]

    return cleaned


def print_ranking_table(
    method_name: str, ratings: Dict[str, float], model_stats: Dict
) -> Dict[str, int]:
    """Print a ranking table for a specific method and return rank mapping."""

    # Sort by rating (descending) - each method has its own rating calculation
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    # Column widths
    width_line = 130
    width_rank = 6
    width_model = 105
    width_rating = 18

    print("\n" + "=" * width_line)
    print(f"🏆 {method_name}")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'Rating':<{width_rating}}"
    )
    print(header)
    print("-" * width_line)

    # Data rows and collect ranks
    ranks = {}
    for rank, (model, rating) in enumerate(ranking, 1):
        cleaned_name = clean_model_name(model)
        row = (
            f"{rank:<{width_rank}} "
            f"{cleaned_name:<{width_model}} "
            f"{rating:<{width_rating}.3f}"
        )
        print(row)
        ranks[model] = rank

    print("=" * width_line)
    return ranks


def print_average_ranking_table(
    all_ranks: List[Dict[str, int]], model_names: List[str]
):
    """Print average ranking across all methods."""

    # Calculate average rank and mode for each model
    avg_ranks = {}
    mode_ranks = {}
    mode_percentages = {}

    for model in model_names:
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]
        if ranks:
            avg_ranks[model] = sum(ranks) / len(ranks)

            # Calculate mode (most frequent rank)
            from collections import Counter

            rank_counts = Counter(ranks)
            mode_rank, mode_count = rank_counts.most_common(1)[0]
            mode_ranks[model] = mode_rank
            mode_percentages[model] = (mode_count / len(ranks)) * 100
        else:
            avg_ranks[model] = float("inf")  # Models without ranks go to bottom
            mode_ranks[model] = 0
            mode_percentages[model] = 0.0

    # Sort by average rank (ascending - lower is better)
    ranking = sorted(avg_ranks.items(), key=lambda x: x[1])

    # Column widths
    width_line = 180
    width_rank = 6
    width_model = 20
    width_avg_rank = 10
    width_mode = 6
    width_mode_pct = 10
    width_individual = 50

    print("\n" + "=" * width_line)
    print("🎯 AVERAGE RANKING ACROSS ALL METHODS")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'Avg Rank':<{width_avg_rank}} "
        f"{'Mode':<{width_mode}} "
        f"{'Mode %':<{width_mode_pct}} "
        f"{'Individual Ranks'}"
    )
    print(header)
    print("-" * width_line)

    # Data rows
    for overall_rank, (model, avg_rank) in enumerate(ranking, 1):
        mode = mode_ranks[model]
        mode_pct = mode_percentages[model]

        # Get individual ranks for this model
        individual = [
            str(method_ranks[model])
            for method_ranks in all_ranks
            if model in method_ranks
        ]
        individual_str = ", ".join(individual)

        cleaned_name = clean_model_name(model)

        row = (
            f"{overall_rank:<{width_rank}} "
            f"{cleaned_name:<{width_model}} "
            f"{avg_rank:<{width_avg_rank}.2f} "
            f"{mode:<{width_mode}d} "
            f"{mode_pct:<{width_mode_pct}.1f} "
            f"{individual_str}"
        )
        print(row)

    print("=" * width_line)
    print("\n💡 Lower average rank = better overall performance")
    print(f"   Mode = Most frequent rank across {len(all_ranks)} methods")
    print("   Mode % = Percentage of methods that assigned this rank")


def print_average_ranking_without_outliers(
    all_ranks: List[Dict[str, int]], model_names: List[str]
):
    """Print average ranking with outliers removed based on mode distance."""

    # Calculate average rank with outlier removal
    avg_ranks_clean = {}
    mode_ranks = {}
    removed_counts = {}
    outlier_values = {}

    for model in model_names:
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]
        if ranks:
            # Calculate mode (most frequent rank)
            from collections import Counter

            rank_counts = Counter(ranks)
            mode_rank, _ = rank_counts.most_common(1)[0]
            mode_ranks[model] = mode_rank

            # Find the rank value furthest from mode
            if len(set(ranks)) > 1:  # Only remove outliers if there's variation
                distances = {rank: abs(rank - mode_rank) for rank in set(ranks)}
                max_distance = max(distances.values())
                outlier_value = [
                    rank for rank, dist in distances.items() if dist == max_distance
                ][0]

                # Remove all instances of the outlier value
                cleaned_ranks = [r for r in ranks if r != outlier_value]
                removed_counts[model] = ranks.count(outlier_value)
                outlier_values[model] = outlier_value
            else:
                cleaned_ranks = ranks
                removed_counts[model] = 0
                outlier_values[model] = None

            # Calculate average from cleaned ranks
            if cleaned_ranks:
                avg_ranks_clean[model] = sum(cleaned_ranks) / len(cleaned_ranks)
            else:
                avg_ranks_clean[model] = sum(ranks) / len(ranks)  # Fallback
        else:
            avg_ranks_clean[model] = float("inf")
            mode_ranks[model] = 0
            removed_counts[model] = 0
            outlier_values[model] = None

    # Sort by average rank (ascending - lower is better)
    ranking = sorted(avg_ranks_clean.items(), key=lambda x: x[1])

    # Column widths
    width_line = 190
    width_rank = 5
    width_model = 40
    width_avg_rank = 9
    width_mode = 5
    width_removed = 8
    width_outlier = 8
    width_individual = 50

    print("\n" + "=" * width_line)
    print("🎯 AVERAGE RANKING (OUTLIERS REMOVED)")
    print("=" * width_line)

    # Header
    header = (
        f"{'Rank':<{width_rank}} "
        f"{'Model Name':<{width_model}} "
        f"{'Avg Rank':<{width_avg_rank}} "
        f"{'Mode':<{width_mode}} "
        f"{'Removed':<{width_removed}} "
        f"{'Outlier':<{width_outlier}} "
        f"{'Remaining Ranks'}"
    )
    print(header)
    print("-" * width_line)

    # Data rows
    for overall_rank, (model, avg_rank) in enumerate(ranking, 1):
        mode = mode_ranks[model]
        removed = removed_counts[model]
        outlier = outlier_values[model]
        outlier_str = str(outlier) if outlier is not None else "-"

        # Get individual ranks for this model
        ranks = [
            method_ranks[model] for method_ranks in all_ranks if model in method_ranks
        ]

        # Get cleaned ranks (after outlier removal)
        if len(set(ranks)) > 1:
            rank_counts = Counter(ranks)
            mode_rank, _ = rank_counts.most_common(1)[0]
            distances = {rank: abs(rank - mode_rank) for rank in set(ranks)}
            max_distance = max(distances.values())
            outlier_value = [
                rank for rank, dist in distances.items() if dist == max_distance
            ][0]
            cleaned_ranks = [r for r in ranks if r != outlier_value]
        else:
            cleaned_ranks = ranks

        individual_str = ", ".join(str(r) for r in cleaned_ranks)
        cleaned_name = clean_model_name(model)

        row = (
            f"{overall_rank:<{width_rank}} "
            f"{cleaned_name:<{width_model}} "
            f"{avg_rank:<{width_avg_rank}.2f} "
            f"{mode:<{width_mode}d} "
            f"{removed:<{width_removed}d} "
            f"{outlier_str:<{width_outlier}} "
            f"{individual_str}"
        )
        print(row)

    print("=" * width_line)
    print("\n💡 Outlier removal: The rank value furthest from the mode is removed")
    print(f"   This helps eliminate extreme rankings that disagree with the consensus")
    print("   Removed = Number of outlier values removed")
    print("   Outlier = The specific rank value that was removed (furthest from mode)")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Run all ranking methods and display results."""

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

    if not CHOIX_AVAILABLE:
        print(f"\n⚠️  WARNING: 'choix' library not installed.")
        print(f"   Methods 2-4 will be skipped.")
        print(f"   Install with: pip install choix")

    if not TRUESKILL_AVAILABLE:
        print(f"\n⚠️  WARNING: 'trueskill' library not installed.")
        print(f"   Method 9 (TrueSkill) will use Elo fallback.")
        print(f"   Install with: pip install trueskill")

    if not GLICKO2_AVAILABLE:
        print(f"\n⚠️  WARNING: 'glicko2' library not installed.")
        print(f"   Method 8 (Glicko-2) will use simplified implementation.")
        print(f"   Install with: pip install glicko2")

    print("\n" + "=" * 170)
    print("🎯 COMPUTING RANKINGS WITH 15 DIFFERENT METHODS")
    print("=" * 170)

    # Store ranks from all methods
    all_ranks = []

    # Method 1: Simple Win Counting
    print("\n[1/15] Computing Simple Win Counting...")
    ratings_1 = simple_win_counting(model_stats)
    ranks_1 = print_ranking_table(
        "METHOD 1: SIMPLE WIN COUNTING", ratings_1, model_stats
    )
    all_ranks.append(ranks_1)

    # Method 2: Bradley-Terry Model
    if CHOIX_AVAILABLE:
        print("\n[2/15] Computing Bradley-Terry Model...")
        ratings_2 = bradley_terry_choix(match_results, model_names)
        ranks_2 = print_ranking_table(
            "METHOD 2: BRADLEY-TERRY MODEL (choix)", ratings_2, model_stats
        )
        all_ranks.append(ranks_2)
    else:
        print("\n[2/15] Skipping Bradley-Terry Model (choix not available)")

    # Method 3: Luce Spectral Ranking
    if CHOIX_AVAILABLE:
        print("\n[3/15] Computing Luce Spectral Ranking...")
        ratings_3 = luce_spectral_ranking(match_results, model_names)
        ranks_3 = print_ranking_table(
            "METHOD 3: LUCE SPECTRAL RANKING (choix)", ratings_3, model_stats
        )
        all_ranks.append(ranks_3)
    else:
        print("\n[3/15] Skipping Luce Spectral Ranking (choix not available)")

    # Method 4: Rank Centrality
    if CHOIX_AVAILABLE:
        print("\n[4/15] Computing Rank Centrality...")
        ratings_4 = rank_centrality(match_results, model_names)
        ranks_4 = print_ranking_table(
            "METHOD 4: RANK CENTRALITY (choix)", ratings_4, model_stats
        )
        all_ranks.append(ranks_4)
    else:
        print("\n[4/15] Skipping Rank Centrality (choix not available)")

    # Method 5: Elo Rating
    print("\n[5/15] Computing Elo Rating...")
    ratings_5 = elo_rating(match_results, model_names)
    ranks_5 = print_ranking_table("METHOD 5: ELO RATING SYSTEM", ratings_5, model_stats)
    all_ranks.append(ranks_5)

    # Method 6: Colley Method
    print("\n[6/15] Computing Colley Matrix Method...")
    ratings_6 = colley_method(match_results, model_names)
    ranks_6 = print_ranking_table(
        "METHOD 6: COLLEY MATRIX METHOD", ratings_6, model_stats
    )
    all_ranks.append(ranks_6)

    # Method 7: Massey Method
    print("\n[7/15] Computing Massey Method...")
    ratings_7 = massey_method(match_results, model_names)
    ranks_7 = print_ranking_table("METHOD 7: MASSEY METHOD", ratings_7, model_stats)
    all_ranks.append(ranks_7)

    # Method 8: Glicko-2
    print("\n[8/15] Computing Glicko-2 Rating...")
    ratings_8 = glicko2_rating(match_results, model_names)
    ranks_8 = print_ranking_table("METHOD 8: GLICKO-2 RATING", ratings_8, model_stats)
    all_ranks.append(ranks_8)

    # Method 9: TrueSkill
    print("\n[9/15] Computing TrueSkill...")
    ratings_9 = trueskill_rating(match_results, model_names)
    ranks_9 = print_ranking_table("METHOD 9: TRUESKILL", ratings_9, model_stats)
    all_ranks.append(ranks_9)

    # Method 10: Copeland
    print("\n[10/15] Computing Copeland Method...")
    ratings_10 = copeland_method(match_results, model_names)
    ranks_10 = print_ranking_table(
        "METHOD 10: COPELAND METHOD", ratings_10, model_stats
    )
    all_ranks.append(ranks_10)

    # Method 11: Borda Count
    print("\n[11/15] Computing Borda Count...")
    ratings_11 = borda_count_method(match_results, model_names)
    ranks_11 = print_ranking_table("METHOD 11: BORDA COUNT", ratings_11, model_stats)
    all_ranks.append(ranks_11)

    # Method 12: Keener
    print("\n[12/15] Computing Keener Method...")
    ratings_12 = keener_method(match_results, model_names)
    ranks_12 = print_ranking_table("METHOD 12: KEENER METHOD", ratings_12, model_stats)
    all_ranks.append(ranks_12)

    # Method 13: Offense-Defense
    print("\n[13/15] Computing Offense-Defense Rating...")
    ratings_13 = offense_defense_rating(match_results, model_names)
    ranks_13 = print_ranking_table(
        "METHOD 13: OFFENSE-DEFENSE RATING", ratings_13, model_stats
    )
    all_ranks.append(ranks_13)

    # Method 14: Markov Rank
    print("\n[14/15] Computing Markov Rank...")
    ratings_14 = markov_rank_method(match_results, model_names)
    ranks_14 = print_ranking_table("METHOD 14: MARKOV RANK", ratings_14, model_stats)
    all_ranks.append(ranks_14)

    # Method 15: Recursive Buchholz
    print("\n[15/15] Computing Recursive Buchholz...")
    ratings_15 = recursive_buchholz_method(match_results, model_names)
    ranks_15 = print_ranking_table(
        "METHOD 15: RECURSIVE BUCHHOLZ", ratings_15, model_stats
    )
    all_ranks.append(ranks_15)

    # Display average ranking
    print("\n[SUMMARY] Computing Average Ranking...")
    print_average_ranking_table(all_ranks, model_names)

    # Display average ranking without outliers
    print("\n[SUMMARY] Computing Average Ranking (Outliers Removed)...")
    print_average_ranking_without_outliers(all_ranks, model_names)

    print("\n" + "=" * 170)
    print("✅ All ranking methods computed successfully!")
    print("=" * 170)

    # Summary of methods
    print("\n📚 METHODS SUMMARY:")
    print("   1. Simple Win Counting: Basic point system (W=1, T=0.5, L=0)")
    print(
        "   2. Bradley-Terry: Maximum likelihood estimation accounting for opponent strength"
    )
    print("   3. Luce Spectral: Eigenvector-based ranking from spectral graph theory")
    print("   4. Rank Centrality: PageRank-like algorithm on comparison graph")
    print("   5. Elo Rating: Dynamic rating system popularized by chess")
    print("   6. Colley Matrix: Linear algebra approach with bias-free formulation")
    print("   7. Massey Method: Least squares minimization of prediction errors")
    print("   8. Glicko-2: Enhanced Elo with rating deviation (uncertainty)")
    print("   9. TrueSkill: Bayesian skill rating from Microsoft Research")
    print("   10. Copeland: Simple pairwise wins minus losses count")
    print("   11. Borda Count: Positional voting adapted for pairwise comparisons")
    print("   12. Keener: Iterative strength-weighted ranking with perturbation")
    print("   13. Offense-Defense: Separate offensive and defensive strength ratings")
    print(
        "   14. Markov Rank: Stationary distribution of random walks (PageRank variant)"
    )
    print(
        "   15. Recursive Buchholz: Strength of schedule with recursive opponent consideration"
    )
    print("\n")


if __name__ == "__main__":
    main()

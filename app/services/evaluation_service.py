"""
Evaluation service layer.

Provides both aggregate metrics and the raw prediction arrays
required by the Model Evaluation tab (Brier decomposition,
calibration plot, baselines, Log5 baseline, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from elo_lab.workflows.evaluate_models import evaluate_models
from elo_lab.evaluation.log5 import compute_log5_predictions
from elo_lab.evaluation.calibration_report import full_calibration_report


# ---------------------------------------------------------------------------
# Public API – Elo backtest results
# ---------------------------------------------------------------------------

def get_model_metrics(output_dir: str = "outputs") -> pd.DataFrame:
    """
    Return the current model evaluation summary
    (one row per backtest model with accuracy / log_loss / brier).
    """
    return evaluate_models(output_dir=output_dir)


def get_backtest_path(
    model: str,
    output_dir: str = "outputs",
) -> Optional[Path]:
    """
    Return the Path to the backtest CSV for a given model name,
    or None if the file does not exist.
    """
    path = Path(output_dir) / f"backtest_{model}.csv"
    return path if path.is_file() else None


def get_model_predictions(
    model: str,
    output_dir: str = "outputs",
) -> Optional[pd.DataFrame]:
    """
    Load the raw backtest predictions for a single model.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing at least the columns ``p_home`` and ``actual``.
    """
    path = get_backtest_path(model, output_dir=output_dir)
    if path is None:
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required = {"p_home", "actual"}
    if not required.issubset(df.columns):
        return None

    return df


def get_prediction_arrays(
    model: str,
    output_dir: str = "outputs",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper that returns (probs, actuals) as NumPy arrays.
    """
    df = get_model_predictions(model, output_dir=output_dir)
    if df is None:
        return None

    probs = df["p_home"].to_numpy(dtype=float)
    actuals = df["actual"].to_numpy(dtype=float)
    return probs, actuals


def list_available_models(output_dir: str = "outputs") -> list[str]:
    """
    Return a sorted list of model names that have a backtest CSV.
    """
    output = Path(output_dir)
    if not output.is_dir():
        return []

    models = []
    for p in sorted(output.glob("backtest_*.csv")):
        name = p.stem.replace("backtest_", "", 1)
        models.append(name)
    return models


# ---------------------------------------------------------------------------
# Log5 baseline (computed from original schedule CSVs)
# ---------------------------------------------------------------------------

def get_log5_arrays(
    sport: str = "NFL",
    data_dir: str = "data",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Log5 probabilities on the full schedule for the given sport.

    Returns (probs, actuals) or None if the schedule cannot be loaded.
    """
    sport = sport.upper()
    if sport == "NFL":
        path = Path(data_dir) / "nfl_games.csv"
        date_col: str | list[str] = ["season", "week"]
    elif sport == "NHL":
        path = Path(data_dir) / "nhl_games.csv"
        date_col = "date"
    else:
        return None

    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required = {"home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        return None

    probs, actuals = compute_log5_predictions(
        df,
        home_col="home_team",
        away_col="away_team",
        home_score_col="home_score",
        away_score_col="away_score",
        date_col=date_col,
    )
    return probs, actuals


def get_log5_report(
    sport: str = "NFL",
    data_dir: str = "data",
) -> Optional[dict]:
    """
    Full calibration report for the Log5 baseline on the given sport.
    """
    arrays = get_log5_arrays(sport=sport, data_dir=data_dir)
    if arrays is None:
        return None
    probs, actuals = arrays
    home_win_rate = float(np.mean(actuals))
    return full_calibration_report(
        probs,
        actuals,
        n_bins=10,
        baselines={"home_win_rate": home_win_rate, "coin_flip": 0.5},
    )


# ---------------------------------------------------------------------------
# Parameter-matched historical evaluation (completed seasons only)
# ---------------------------------------------------------------------------

def score_config_on_season(
    sport: str,
    season: str,
    config: dict,
    mean_elo: float = 1500.0,
) -> dict:
    """
    Walk actual games of a completed season once with the given model config.

    Returns accuracy / log_loss / brier plus prediction arrays.
    Suitable for Model Comparison when the user is projecting an upcoming season.
    """
    import numpy as np
    import pandas as pd

    try:
        from elo_lab.workflows.simulate_season import (
            _load_season_schedule,
            filter_regular_season,
            is_playoff_row,
        )
        from elo_lab.neutral_sites import exclude_neutral_sites
        from elo_lab.engine import run_game, compute_pregame
    except ImportError as e:
        return {"error": f"imports failed: {e}"}

    df = _load_season_schedule(sport, str(season))
    if df is None or df.empty:
        return {"error": f"No schedule for {sport} {season}"}

    # Prefer regular season for cleaner scoring; fall back to all rows with scores
    reg = filter_regular_season(df, sport=sport)
    if reg is not None and not reg.empty:
        df = reg

    # Same neutral-site exclusion as backtest / grid search / snapshot.
    df = exclude_neutral_sites(df, sport=sport)

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        return {"error": f"No completed games with scores for {sport} {season}"}

    ratings = {}
    probs = []
    actuals = []

    for _, game in played.iterrows():
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        try:
            hs, as_ = float(game["home_score"]), float(game["away_score"])
        except (TypeError, ValueError):
            continue
        if hs == as_:
            continue

        ratings.setdefault(home, mean_elo)
        ratings.setdefault(away, mean_elo)

        try:
            pre = compute_pregame(
                home_elo=float(ratings[home]),
                away_elo=float(ratings[away]),
                context={
                    "season": season,
                    "sport": sport,
                    "home_team": home,
                    "away_team": away,
                },
                config=config,
            )
            p_home = float(pre.get("p_home", 0.5))
        except Exception:
            # fallback logistic
            diff = float(ratings[home]) - float(ratings[away])
            p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        actual = 1.0 if hs > as_ else 0.0
        probs.append(p_home)
        actuals.append(actual)

        try:
            result = run_game(
                home_elo=float(ratings[home]),
                away_elo=float(ratings[away]),
                context={
                    "season": season,
                    "sport": sport,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "actual": int(actual),
                },
                config=config,
            )
            ratings[home] = float(result["home_elo_post"])
            ratings[away] = float(result["away_elo_post"])
        except Exception:
            # still update with simple Elo if run_game fails
            k = float(config.get("k", 20))
            delta = k * (actual - p_home)
            ratings[home] = float(ratings[home]) + delta
            ratings[away] = float(ratings[away]) - delta

    if not probs:
        return {"error": f"No scorable games for {sport} {season}"}

    probs_a = np.asarray(probs, dtype=float)
    actuals_a = np.asarray(actuals, dtype=float)
    preds = (probs_a >= 0.5).astype(int)
    acc = float(np.mean(preds == actuals_a))
    p_clip = np.clip(probs_a, 1e-15, 1.0 - 1e-15)
    log_loss = float(
        -np.mean(actuals_a * np.log(p_clip) + (1 - actuals_a) * np.log(1 - p_clip))
    )
    brier = float(np.mean((probs_a - actuals_a) ** 2))

    return {
        "sport": sport,
        "season": str(season),
        "n_games": int(len(probs_a)),
        "accuracy": acc,
        "log_loss": log_loss,
        "brier": brier,
        "probs": probs_a,
        "actuals": actuals_a,
        "label": f"Current params on {season}",
    }

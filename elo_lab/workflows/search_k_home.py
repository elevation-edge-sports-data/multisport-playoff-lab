#!/usr/bin/env python3
"""
Joint grid search over k and home-advantage Elo points.

Uses decisive, non-neutral regular-season games only. Elevation Edge is
off. Margin of victory is off, so the grid is only k and home advantage.

Home field is placed on ``pregame_pipeline``. The public defaults store a
home-advantage value but do not set that pipeline, so this search does
not change them. Each season starts every team at 1500. Ratings do not
carry across seasons.

Score is the lab's log loss and Brier score (lower is better), plus
accuracy. The best pair is the lowest log loss. Brier breaks ties.

``--extended`` re-searches NFL and NBA on a wider lattice. NHL stays at
the pair the first search selected (k=15, home advantage 25) and is not
fit again. Production defaults are not written.

Usage (from the repo root):

    python -m elo_lab.workflows.search_k_home
    python -m elo_lab.workflows.search_k_home --out outputs/k_home_search.csv
    python -m elo_lab.workflows.search_k_home --extended
    python -m elo_lab.workflows.search_k_home --nfl-k
    python -m elo_lab.workflows.search_k_home --elevation
    python -m elo_lab.workflows.search_k_home --elevation-holdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

def _prepare_import_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    for entry in (root, root / "app"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


REPO_ROOT = _prepare_import_path()

from elo_lab.engine import run_game  # noqa: E402
from elo_lab.evaluation.metrics import accuracy, brier, log_loss  # noqa: E402
from elo_lab.neutral_sites import exclude_neutral_sites  # noqa: E402
from elo_lab.workflows.generate_default_sims import default_config  # noqa: E402
from elo_lab.workflows.simulate_season import (  # noqa: E402
    _load_season_schedule,
    filter_regular_season,
)
from app.services.initial_ratings_service import get_available_seasons  # noqa: E402

SPORTS = ("NHL", "NFL", "NBA")

# Modest k lattice. The sport's current default k is added when it is
# not already on this list (NHL 12, NBA 22) so that pair is scored too.
K_GRID = (10, 15, 20, 25, 30)

# Home-advantage steps around the current default, in Elo points.
_HFA_OFFSETS = (-20, -10, 0, 10, 20)

# Wider lattice after the first search sat on a boundary for NFL and NBA.
# NHL is not in this map. It stays at PREVIOUS_BEST["NHL"].
EXTENDED_GRIDS = {
    "NFL": ((25, 30, 35, 40), (20, 25, 30, 35, 45)),
    "NBA": ((25, 30, 35), (35, 40, 45, 50, 55)),
}

# (k, home advantage) from the first search. NFL and NBA are the
# comparison points on the extended grid. NHL is held, not re-searched.
PREVIOUS_BEST = {
    "NHL": (15.0, 25.0),
    "NFL": (30.0, 35.0),
    "NBA": (30.0, 45.0),
}

# NFL follow-up: home advantage stays in a narrow band, k moves past 40.
# Compared with the extended-grid winner, not the first-search pair.
NFL_K_FOLLOWUP = ((35, 40, 45, 50), (25, 30, 35))
NFL_PRIOR_EXTENDED = (40.0, 30.0)

# Best (k, home advantage) from the joint searches. Elevation was off.
# NFL uses the latest k follow-up. NHL and NBA were not revised after that.
LOCKED_PAIRS = {
    "NHL": (15.0, 25.0),
    "NFL": (50.0, 30.0),
    "NBA": (30.0, 45.0),
}
# Elevation coefficients chosen with home advantage left at LOCKED_PAIRS.
RECOMMENDED_ELEVATION = {
    "NHL": 4.0,
    "NFL": 10.0,
    "NBA": 10.0,
}
# Elo points per 1,000 ft. Same lattice the dashboard optimizer publishes.
# Zero is the k + home-advantage baseline.
ELEVATION_GRID = (0, 2, 4, 6, 8, 10)
# One-dimensional home-advantage check once elevation is locked.
HOME_RECHECK = {
    "NHL": (15, 20, 25, 30, 35),
    "NFL": (20, 25, 30, 35, 40),
    "NBA": (35, 40, 45, 50, 55),
}
# A home-advantage change is kept only when log loss improves by this much.
# The same bar decides whether elevation earns a place on a held-out season.
HOME_MOVE_MIN_LOG_LOSS = 0.001

Game = Tuple[str, str, str, float, float]
MEAN_ELO = 1500.0


def current_pair(sport: str) -> Tuple[float, float]:
    """(k, home advantage) stored on the public default config."""
    config = default_config(sport)
    home = float(config["adjustments"]["home_field"]["value"])
    return float(config["k"]), home


def k_values_for(sport: str) -> list:
    default_k, _ = current_pair(sport)
    values = {float(k) for k in K_GRID}
    values.add(float(default_k))
    return sorted(values)


def home_values_for(sport: str) -> list:
    _, default_home = current_pair(sport)
    return [float(default_home + offset) for offset in _HFA_OFFSETS]


def grid_position(k: float, home: float, k_values, home_values) -> str:
    """``interior`` when both knobs sit strictly inside the searched lattice."""
    ks = [float(v) for v in k_values]
    homes = [float(v) for v in home_values]
    on_k_edge = float(k) == min(ks) or float(k) == max(ks)
    on_h_edge = float(home) == min(homes) or float(home) == max(homes)
    if on_k_edge or on_h_edge:
        return "boundary"
    return "interior"


def k_axis_position(k: float, k_values) -> str:
    """Where ``k`` sits on its own axis: interior, or a lower/upper edge."""
    ks = [float(v) for v in k_values]
    value = float(k)
    if value == max(ks):
        return "upper boundary"
    if value == min(ks):
        return "lower boundary"
    return "interior"


def fitting_config(k: float, home_advantage: float, elevation: float = 0.0) -> dict:
    """Config that applies this k and home advantage.

    Margin of victory stays off. Elevation Edge is on only when
    ``elevation`` is not zero. The coefficient is Elo points per 1,000 ft.
    Home field is on the pregame pipeline so the engine adds it.
    """
    pregame = ["home_field"]
    elevation_on = float(elevation) != 0.0
    if elevation_on:
        pregame.append("elevation_edge")
    return {
        "k": float(k),
        "pregame_pipeline": pregame,
        "postgame_pipeline": [],
        "adjustments": {
            "home_field": {"enabled": True, "value": float(home_advantage)},
            "margin_of_victory": {"enabled": False, "scale": 1.0},
            "elevation_edge": {
                "enabled": elevation_on,
                "value": float(elevation),
            },
        },
    }


def load_fitting_games(sport: str) -> pd.DataFrame:
    """Decisive non-neutral regular-season games, one block per season.

    Seasons stay in schedule order. Inside a season, games are ordered by
    week, then date, then file order. Unplayed rows and ties are omitted.
    """
    frames = []
    for season in get_available_seasons(sport):
        schedule = _load_season_schedule(sport, str(season))
        if schedule is None or schedule.empty:
            continue
        regular = filter_regular_season(schedule, sport=sport)
        regular = exclude_neutral_sites(regular, sport=sport)
        if regular is None or regular.empty:
            continue
        block = regular.copy()
        block["season"] = str(season)
        block["_order"] = np.arange(len(block))
        frames.append(block)
    if not frames:
        return pd.DataFrame(
            columns=["season", "home_team", "away_team", "home_score", "away_score"]
        )

    games = pd.concat(frames, ignore_index=True)
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games = games[games["home_score"] != games["away_score"]].copy()
    if "week" not in games.columns:
        games["week"] = pd.NA
    if "date" not in games.columns:
        games["date"] = pd.NA
    games["_week"] = pd.to_numeric(games["week"], errors="coerce")
    games["_date"] = pd.to_datetime(games["date"], errors="coerce")

    ordered = []
    for _, part in games.groupby("season", sort=False):
        ordered.append(
            part.sort_values(["_week", "_date", "_order"], na_position="last")
        )
    out = pd.concat(ordered, ignore_index=True)
    return out.drop(columns=["_order", "_week", "_date"])


def _records(games: pd.DataFrame) -> list:
    rows: list = []
    for season, home, away, hs, as_ in zip(
        games["season"],
        games["home_team"],
        games["away_team"],
        games["home_score"],
        games["away_score"],
    ):
        rows.append((str(season), str(home).strip(), str(away).strip(), float(hs), float(as_)))
    return rows


def score_games(
    games: Sequence[Game],
    k: float,
    home_advantage: float,
    elevation: float = 0.0,
    sport: Optional[str] = None,
) -> dict:
    """Walk ``games`` with one parameter triple. Reset Elo each season.

    ``sport`` is required for Elevation Edge. Without it the adjustment
    looks up NFL elevations.
    """
    config = fitting_config(k, home_advantage, elevation=elevation)
    ratings: dict = {}
    season_now: Optional[str] = None
    probs = []
    actuals = []
    for season, home, away, hs, as_ in games:
        if season != season_now:
            ratings = {}
            season_now = season
        ratings.setdefault(home, MEAN_ELO)
        ratings.setdefault(away, MEAN_ELO)
        actual = 1 if hs > as_ else 0
        context = {
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "actual": actual,
        }
        if sport:
            context["sport"] = sport.upper()
        result = run_game(
            home_elo=float(ratings[home]),
            away_elo=float(ratings[away]),
            context=context,
            config=config,
        )
        probs.append(float(result["p_home"]))
        actuals.append(actual)
        ratings[home] = float(result["home_elo_post"])
        ratings[away] = float(result["away_elo_post"])

    p = np.asarray(probs, dtype=float)
    y = np.asarray(actuals, dtype=float)
    return {
        "k": float(k),
        "home_advantage": float(home_advantage),
        "elevation": float(elevation),
        "log_loss": float(np.mean(log_loss(p, y))),
        "brier": float(np.mean(brier(p, y))),
        "accuracy": float(accuracy(p, y)),
        "n_games": int(len(y)),
    }


def search_sport(
    sport: str,
    games: Optional[pd.DataFrame] = None,
    k_values: Optional[Sequence[float]] = None,
    home_values: Optional[Sequence[float]] = None,
    extra_pairs: Optional[Sequence[Tuple[float, float]]] = None,
    previous_pair: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Score a k × home-advantage grid for one sport.

    ``extra_pairs`` are scored for comparison and left off the grid, so
    they do not compete for rank. The current default and the previous
    best are flagged when a scored pair matches them.
    """
    sport_u = sport.upper()
    frame = load_fitting_games(sport_u) if games is None else games
    records = _records(frame)
    if not records:
        return pd.DataFrame()
    counts = frame.groupby("season", sort=False).size()
    print(
        "  "
        + ", ".join(f"{season}={int(n)}" for season, n in counts.items()),
        flush=True,
    )

    grid_ks = list(k_values) if k_values is not None else k_values_for(sport_u)
    grid_hs = list(home_values) if home_values is not None else home_values_for(sport_u)
    grid_pairs = [(float(k), float(h)) for k in grid_ks for h in grid_hs]
    extra = []
    for pair in extra_pairs or []:
        pair_f = (float(pair[0]), float(pair[1]))
        if pair_f not in grid_pairs:
            extra.append(pair_f)

    default_k, default_home = current_pair(sport_u)
    if previous_pair is not None:
        previous = (float(previous_pair[0]), float(previous_pair[1]))
    else:
        previous = PREVIOUS_BEST.get(sport_u)
    n_seasons = int(frame["season"].nunique())
    rows = []
    for k, home in grid_pairs + extra:
        scored = score_games(records, k, home)
        on_grid = (k, home) in grid_pairs
        scored.update({
            "sport": sport_u,
            "n_seasons": n_seasons,
            "on_grid": on_grid,
            "is_default": bool(k == default_k and home == default_home),
            "is_previous_best": bool(previous is not None and (k, home) == previous),
            "is_held": False,
            "grid_position": grid_position(k, home, grid_ks, grid_hs) if on_grid else "",
            "k_position": k_axis_position(k, grid_ks) if on_grid else "",
        })
        rows.append(scored)

    out = pd.DataFrame(rows)
    grid = out[out["on_grid"]].sort_values(
        ["log_loss", "brier", "k", "home_advantage"],
        ascending=[True, True, True, True],
    ).copy()
    grid["rank"] = np.arange(1, len(grid) + 1)
    off = out[~out["on_grid"]].copy()
    off["rank"] = pd.NA
    return pd.concat([grid, off], ignore_index=True)


def score_held_pair(sport: str, k: float, home: float) -> pd.DataFrame:
    """Score one pair and mark it held. It is not a grid result."""
    sport_u = sport.upper()
    frame = load_fitting_games(sport_u)
    records = _records(frame)
    counts = frame.groupby("season", sort=False).size()
    print(
        "  "
        + ", ".join(f"{season}={int(n)}" for season, n in counts.items()),
        flush=True,
    )
    default_k, default_home = current_pair(sport_u)
    previous = PREVIOUS_BEST.get(sport_u)
    scored = score_games(records, k, home)
    scored.update({
        "sport": sport_u,
        "n_seasons": int(frame["season"].nunique()),
        "on_grid": False,
        "is_default": bool(float(k) == default_k and float(home) == default_home),
        "is_previous_best": bool(
            previous is not None and (float(k), float(home)) == previous
        ),
        "is_held": True,
        "grid_position": "",
        "rank": pd.NA,
    })
    return pd.DataFrame([scored])


def _format_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in ("log_loss", "brier", "accuracy"):
        show[col] = show[col].map(lambda v: f"{v:.6f}")
    show["k"] = show["k"].map(lambda v: f"{v:g}")
    show["home_advantage"] = show["home_advantage"].map(lambda v: f"{v:g}")
    cols = ["rank", "k", "home_advantage", "log_loss", "brier", "accuracy", "is_default"]
    return show[cols].to_string(index=False)


def _candidate_rows(block: pd.DataFrame) -> pd.DataFrame:
    """Grid rows only, best log loss first. A held pair is not a candidate."""
    rows = block
    if "on_grid" in rows.columns:
        rows = rows[rows["on_grid"]]
    if rows.empty:
        return rows
    return rows.sort_values(
        ["log_loss", "brier", "k", "home_advantage"],
        ascending=[True, True, True, True],
    )


def summarize(results: pd.DataFrame, top_n: int = 5) -> str:
    """Text report: best pair, nearby pairs, and the gap versus the default pair."""
    lines = []
    present = [sport for sport in SPORTS if sport in set(results["sport"])]
    for sport in present:
        block = results[results["sport"] == sport]
        if block.empty:
            lines.append(f"{sport}: no scored games")
            continue
        if "is_held" in block.columns and bool(block["is_held"].all()):
            held = block.iloc[0]
            lines.append(
                f"{sport}: held at k={held['k']:g}, home={held['home_advantage']:g} "
                f"(not re-searched) on {int(held['n_games'])} games / "
                f"{int(held['n_seasons'])} seasons"
            )
            lines.append(
                f"  log loss {held['log_loss']:.6f}  brier {held['brier']:.6f}  "
                f"accuracy {held['accuracy']:.4f}"
            )
            lines.append("")
            continue
        candidates = _candidate_rows(block)
        best = candidates.iloc[0]
        default_rows = block[block["is_default"]] if "is_default" in block.columns else block.iloc[0:0]
        lines.append(
            f"{sport}: best k={best['k']:g}, home={best['home_advantage']:g} "
            f"on {int(best['n_games'])} games / {int(best['n_seasons'])} seasons"
        )
        if "grid_position" in best.index and best["grid_position"]:
            lines.append(f"  grid position: {best['grid_position']}")
        if "k_position" in best.index and best["k_position"]:
            lines.append(f"  k position: {best['k_position']}")
        lines.append(
            f"  log loss {best['log_loss']:.6f}  brier {best['brier']:.6f}  "
            f"accuracy {best['accuracy']:.4f}"
        )
        if "is_previous_best" in block.columns and block["is_previous_best"].any():
            previous = block[block["is_previous_best"]].iloc[0]
            lines.append(
                f"  previous best k={previous['k']:g}, home={previous['home_advantage']:g}: "
                f"log loss {previous['log_loss']:.6f}  brier {previous['brier']:.6f}"
            )
            lines.append(
                f"  vs previous best: "
                f"log loss {previous['log_loss'] - best['log_loss']:+.6f}  "
                f"brier {previous['brier'] - best['brier']:+.6f}  "
                f"(positive means the new pair scores lower)"
            )
        if not default_rows.empty:
            default = default_rows.iloc[0]
            if pd.notna(default.get("rank")):
                where = f"rank {int(default['rank'])}"
            else:
                where = "off this grid"
            lines.append(
                f"  default k={default['k']:g}, home={default['home_advantage']:g} "
                f"({where}): "
                f"log loss {default['log_loss']:.6f}  brier {default['brier']:.6f}  "
                f"accuracy {default['accuracy']:.4f}"
            )
            lines.append(
                f"  vs default: "
                f"log loss {default['log_loss'] - best['log_loss']:+.6f}  "
                f"brier {default['brier'] - best['brier']:+.6f}  "
                f"(positive means the best pair scores lower)"
            )
        brier_rows = candidates.sort_values(["brier", "log_loss"])
        brier_best = brier_rows.iloc[0]
        if (brier_best["k"], brier_best["home_advantage"]) != (best["k"], best["home_advantage"]):
            lines.append(
                f"  lowest brier on the grid is k={brier_best['k']:g}, "
                f"home={brier_best['home_advantage']:g} "
                f"(brier {brier_best['brier']:.6f})"
            )
        nearby = candidates.head(top_n)
        lines.append("  top pairs:")
        lines.append(_format_table(nearby))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_search(sports: Iterable[str] = SPORTS) -> pd.DataFrame:
    frames = []
    for sport in sports:
        print(f"Searching {sport}...", flush=True)
        frames.append(search_sport(sport))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_extended_search() -> pd.DataFrame:
    """NFL and NBA on the wider grids. NHL is scored once at its held pair."""
    frames = []
    held_k, held_home = PREVIOUS_BEST["NHL"]
    print(
        f"NHL held at k={held_k:g}, home={held_home:g} (not re-searched)...",
        flush=True,
    )
    frames.append(score_held_pair("NHL", held_k, held_home))
    for sport, (k_values, home_values) in EXTENDED_GRIDS.items():
        print(f"Searching {sport} extended grid...", flush=True)
        frames.append(search_sport(
            sport,
            k_values=k_values,
            home_values=home_values,
            extra_pairs=[current_pair(sport)],
        ))
    return pd.concat(frames, ignore_index=True)


def run_nfl_k_followup() -> pd.DataFrame:
    """NFL only. Home advantage stays in {25, 30, 35}. k runs through 50."""
    k_values, home_values = NFL_K_FOLLOWUP
    print("Searching NFL k follow-up...", flush=True)
    return search_sport(
        "NFL",
        k_values=k_values,
        home_values=home_values,
        extra_pairs=[current_pair("NFL")],
        previous_pair=NFL_PRIOR_EXTENDED,
    )


def adopt_recheck_home(locked_home: float, recheck_home: float, log_loss_gain: float) -> bool:
    """Keep a new home advantage only when log loss improves enough."""
    if float(recheck_home) == float(locked_home):
        return False
    return float(log_loss_gain) >= HOME_MOVE_MIN_LOG_LOSS


def _rank_scored(rows: list) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["log_loss", "brier", "elevation", "home_advantage"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_elevation_search() -> pd.DataFrame:
    """Grid elevation at the locked (k, home) pair, then re-check home advantage.

    Elevation 0 is the k + home-advantage baseline. A new home value is
    kept only when ``adopt_recheck_home`` says the log-loss gain is material.
    """
    frames = []
    for sport, (k, home) in LOCKED_PAIRS.items():
        print(
            f"Elevation search {sport} at k={k:g}, home={home:g}...",
            flush=True,
        )
        schedule = load_fitting_games(sport)
        records = _records(schedule)
        counts = schedule.groupby("season", sort=False).size()
        print(
            "  " + ", ".join(f"{season}={int(n)}" for season, n in counts.items()),
            flush=True,
        )
        n_seasons = int(schedule["season"].nunique())
        elev_rows = []
        for elevation in ELEVATION_GRID:
            scored = score_games(records, k, home, elevation=elevation, sport=sport)
            scored.update({
                "sport": sport,
                "stage": "elevation",
                "n_seasons": n_seasons,
            })
            elev_rows.append(scored)
        elev_df = _rank_scored(elev_rows)
        best_elevation = float(elev_df.iloc[0]["elevation"])
        print(f"  best elevation {best_elevation:g}", flush=True)

        print(f"  home re-check at elevation {best_elevation:g}...", flush=True)
        recheck_rows = []
        for home_value in HOME_RECHECK[sport]:
            scored = score_games(
                records, k, home_value, elevation=best_elevation, sport=sport,
            )
            scored.update({
                "sport": sport,
                "stage": "home_recheck",
                "n_seasons": n_seasons,
            })
            recheck_rows.append(scored)
        recheck_df = _rank_scored(recheck_rows)
        recheck_home = float(recheck_df.iloc[0]["home_advantage"])
        locked_at_elev = recheck_df[recheck_df["home_advantage"] == float(home)].iloc[0]
        gain = float(locked_at_elev["log_loss"] - recheck_df.iloc[0]["log_loss"])
        changed = adopt_recheck_home(home, recheck_home, gain)
        final_home = recheck_home if changed else float(home)
        final = recheck_df[recheck_df["home_advantage"] == final_home].iloc[0]
        baseline = elev_df[elev_df["elevation"] == 0.0].iloc[0]
        recommendation = {
            "sport": sport,
            "stage": "recommended",
            "k": float(k),
            "home_advantage": final_home,
            "elevation": best_elevation,
            "log_loss": float(final["log_loss"]),
            "brier": float(final["brier"]),
            "accuracy": float(final["accuracy"]),
            "n_games": int(final["n_games"]),
            "n_seasons": n_seasons,
            "home_changed": changed,
            "locked_home": float(home),
            "recheck_home": recheck_home,
            "recheck_log_loss_gain": gain,
            "baseline_log_loss": float(baseline["log_loss"]),
            "baseline_brier": float(baseline["brier"]),
            "rank": 1,
        }
        frames.extend([elev_df, recheck_df, pd.DataFrame([recommendation])])
    return pd.concat(frames, ignore_index=True)


def summarize_elevation(results: pd.DataFrame) -> str:
    """Per sport: best elevation, home re-check, and the recommended triple."""
    lines = []
    for sport in LOCKED_PAIRS:
        block = results[results["sport"] == sport]
        if block.empty:
            lines.append(f"{sport}: no scored games")
            continue
        recommended = block[block["stage"] == "recommended"].iloc[0]
        changed = bool(recommended["home_changed"])
        lines.append(
            f"{sport}: recommended k={recommended['k']:g}, "
            f"home={recommended['home_advantage']:g}, "
            f"elevation={recommended['elevation']:g}"
        )
        if recommended["elevation"] == max(ELEVATION_GRID):
            lines.append("  elevation is on the upper edge of the grid")
        elif recommended["elevation"] == 0:
            lines.append("  best elevation coefficient is 0")
        if changed:
            lines.append(
                f"  home re-check moved {recommended['locked_home']:g} -> "
                f"{recommended['recheck_home']:g} "
                f"(log-loss gain {recommended['recheck_log_loss_gain']:.6f})"
            )
        else:
            lines.append(
                f"  home re-check best {recommended['recheck_home']:g} "
                f"(log-loss gain {recommended['recheck_log_loss_gain']:.6f}); "
                f"kept {recommended['locked_home']:g}"
            )
        lines.append(
            f"  recommended log loss {recommended['log_loss']:.6f}  "
            f"brier {recommended['brier']:.6f}  "
            f"accuracy {recommended['accuracy']:.4f}"
        )
        lines.append(
            f"  k+home baseline (elevation 0): "
            f"log loss {recommended['baseline_log_loss']:.6f}  "
            f"brier {recommended['baseline_brier']:.6f}"
        )
        lines.append(
            f"  vs baseline: "
            f"log loss {recommended['baseline_log_loss'] - recommended['log_loss']:+.6f}  "
            f"brier {recommended['baseline_brier'] - recommended['brier']:+.6f}  "
            f"(positive means the triple scores lower)"
        )
        elev = block[block["stage"] == "elevation"].sort_values("rank")
        recheck = block[block["stage"] == "home_recheck"].sort_values("rank")
        lines.append("  elevation grid:")
        lines.append(_format_elevation_table(elev))
        lines.append("  home re-check:")
        lines.append(_format_elevation_table(recheck))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_elevation_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in ("log_loss", "brier", "accuracy"):
        show[col] = show[col].map(lambda v: f"{float(v):.6f}")
    show["k"] = show["k"].map(lambda v: f"{float(v):g}")
    show["home_advantage"] = show["home_advantage"].map(lambda v: f"{float(v):g}")
    show["elevation"] = show["elevation"].map(lambda v: f"{float(v):g}")
    cols = ["rank", "k", "home_advantage", "elevation", "log_loss", "brier", "accuracy"]
    return show[cols].to_string(index=False)


def historical_eval_season(sport: str) -> str:
    """Season the dashboard scores: the file before the latest schedule.

    That is the most recent completed season when the newest file is the
    season in progress. Those games were also inside the fitting sample.
    """
    seasons = get_available_seasons(sport)
    if len(seasons) < 2:
        raise ValueError(f"{sport} does not have a prior season to score")
    return str(seasons[-2])


def _holdout_specs(sport: str) -> list:
    """Default, k+home with elevation off, and the recommended triple."""
    default_k, default_home = current_pair(sport)
    k, home = LOCKED_PAIRS[sport]
    elevation = RECOMMENDED_ELEVATION[sport]
    return [
        ("default", default_k, default_home, 0.0),
        ("k_home", k, home, 0.0),
        ("triple", k, home, elevation),
    ]


def run_elevation_holdout() -> pd.DataFrame:
    """Score three configs with ``score_config_on_season`` on one season each."""
    from app.services.evaluation_service import score_config_on_season

    rows = []
    for sport in LOCKED_PAIRS:
        season = historical_eval_season(sport)
        print(f"Holdout {sport} season {season}...", flush=True)
        for label, k, home, elevation in _holdout_specs(sport):
            scored = score_config_on_season(
                sport,
                season,
                fitting_config(k, home, elevation=elevation),
            )
            if "error" in scored:
                raise RuntimeError(f"{sport} {season} {label}: {scored['error']}")
            rows.append({
                "sport": sport,
                "season": season,
                "model": label,
                "k": float(k),
                "home_advantage": float(home),
                "elevation": float(elevation),
                "n_games": int(scored["n_games"]),
                "log_loss": float(scored["log_loss"]),
                "brier": float(scored["brier"]),
                "accuracy": float(scored["accuracy"]),
            })
    return pd.DataFrame(rows)


def summarize_holdout(results: pd.DataFrame) -> str:
    """Deltas on the eval season, and whether elevation clears the keep bar."""
    lines = []
    for sport in LOCKED_PAIRS:
        block = results[results["sport"] == sport]
        if block.empty:
            lines.append(f"{sport}: no scored games")
            continue
        by_model = {row["model"]: row for _, row in block.iterrows()}
        default = by_model["default"]
        pair = by_model["k_home"]
        triple = by_model["triple"]
        elev_ll = float(pair["log_loss"] - triple["log_loss"])
        elev_br = float(pair["brier"] - triple["brier"])
        keep = elev_ll >= HOME_MOVE_MIN_LOG_LOSS
        lines.append(
            f"{sport} {triple['season']}: {int(triple['n_games'])} games "
            f"(most recent completed season; these games were in the fit)"
        )
        lines.append(
            f"  default k={default['k']:g} home={default['home_advantage']:g} "
            f"elevation=0: log loss {default['log_loss']:.6f}  "
            f"brier {default['brier']:.6f}  accuracy {default['accuracy']:.4f}"
        )
        lines.append(
            f"  k+home k={pair['k']:g} home={pair['home_advantage']:g} "
            f"elevation=0: log loss {pair['log_loss']:.6f}  "
            f"brier {pair['brier']:.6f}  accuracy {pair['accuracy']:.4f}"
        )
        lines.append(
            f"  triple k={triple['k']:g} home={triple['home_advantage']:g} "
            f"elevation={triple['elevation']:g}: log loss {triple['log_loss']:.6f}  "
            f"brier {triple['brier']:.6f}  accuracy {triple['accuracy']:.4f}"
        )
        lines.append(
            f"  triple vs k+home: log loss {elev_ll:+.6f}  brier {elev_br:+.6f}  "
            f"(positive means elevation scores lower)"
        )
        lines.append(
            f"  triple vs default: "
            f"log loss {default['log_loss'] - triple['log_loss']:+.6f}  "
            f"brier {default['brier'] - triple['brier']:+.6f}"
        )
        lines.append(
            f"  k+home vs default: "
            f"log loss {default['log_loss'] - pair['log_loss']:+.6f}  "
            f"brier {default['brier'] - pair['brier']:+.6f}"
        )
        if keep:
            lines.append(
                f"  keep elevation: yes (log-loss gain {elev_ll:.6f} "
                f">= {HOME_MOVE_MIN_LOG_LOSS:.3f})"
            )
        else:
            lines.append(
                f"  keep elevation: no (log-loss gain {elev_ll:.6f} "
                f"is below {HOME_MOVE_MIN_LOG_LOSS:.3f})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Grid search k and home advantage on non-neutral regular-season games."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--extended",
        action="store_true",
        help="NFL and NBA wider grids. NHL stays at k=15, home advantage 25.",
    )
    mode.add_argument(
        "--nfl-k",
        action="store_true",
        help="NFL only. k in {35, 40, 45, 50}, home advantage in {25, 30, 35}.",
    )
    mode.add_argument(
        "--elevation",
        action="store_true",
        help="Elevation Edge grid at the locked (k, home) pairs, then a home re-check.",
    )
    mode.add_argument(
        "--elevation-holdout",
        action="store_true",
        help="Score default, k+home, and the elevation triple on the latest completed season.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default depends on the mode: outputs/k_home_search.csv, "
             "outputs/k_home_search_extended.csv, outputs/k_home_search_nfl_k.csv, "
             "outputs/elevation_search.csv, or outputs/elevation_holdout.csv)",
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        default=list(SPORTS),
        help="Sports to search (default: NHL NFL NBA). Ignored with --extended.",
    )
    parser.add_argument("--top", type=int, default=5, help="Nearby pairs to print")
    args = parser.parse_args(argv)

    import os

    os.chdir(REPO_ROOT)
    if args.extended:
        results = run_extended_search()
        default_out = REPO_ROOT / "outputs" / "k_home_search_extended.csv"
    elif args.nfl_k:
        results = run_nfl_k_followup()
        default_out = REPO_ROOT / "outputs" / "k_home_search_nfl_k.csv"
    elif args.elevation:
        results = run_elevation_search()
        default_out = REPO_ROOT / "outputs" / "elevation_search.csv"
    elif args.elevation_holdout:
        results = run_elevation_holdout()
        default_out = REPO_ROOT / "outputs" / "elevation_holdout.csv"
    else:
        results = run_search(sport.upper() for sport in args.sports)
        default_out = REPO_ROOT / "outputs" / "k_home_search.csv"
    if args.out is None:
        args.out = default_out
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print()
    if args.elevation:
        print(summarize_elevation(results))
    elif args.elevation_holdout:
        print(summarize_holdout(results))
    else:
        print(summarize(results, top_n=args.top))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

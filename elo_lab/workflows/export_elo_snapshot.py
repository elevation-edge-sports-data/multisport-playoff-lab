#!/usr/bin/env python3
"""
Write a static Elo snapshot for downstream consumers.

Ratings are the current post-locked-games Elo: the same warm-up a default
simulation uses, then real target-season results only. Games without a
decisive score are not sampled. When the target slate has no locked games,
the ratings are that warm-up (the ratings a default run starts from).

``home_advantage`` is the Elo ``compute_pregame`` adds to the home team
under ``generate_default_sims.default_config``. That config names
``home_field`` on ``pregame_pipeline`` and leaves Elevation Edge off, so
the applied delta is the stored home advantage (NHL 25, NFL 30, NBA 45).

Usage (from the repo root):

    python -m elo_lab.workflows.export_elo_snapshot
    python -m elo_lab.workflows.export_elo_snapshot --out outputs/elo_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Import path: `python -m` from the repo root, or a direct script path.
# `app/` is on the path so `metadata` resolves the same way the dashboard does.
# ---------------------------------------------------------------------------
def _prepare_import_path() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[2]
    for entry in (root, root / "app"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


REPO_ROOT = _prepare_import_path()

from elo_lab.engine import compute_pregame, run_game  # noqa: E402
from elo_lab.neutral_sites import exclude_neutral_sites  # noqa: E402
from elo_lab.workflows.generate_default_sims import default_config  # noqa: E402
from elo_lab.workflows.simulate_season import (  # noqa: E402
    _build_name_to_abbr,
    _load_season_schedule,
    describe_schedule_lock,
    filter_regular_season,
    parse_locked_scores,
    warm_elo_from_actual_history,
)

SPORTS = ("NFL", "NHL", "NBA")
DEFAULT_OUT = REPO_ROOT / "outputs" / "elo_snapshot.json"

# Same warm-up knobs as run_simulation / generate_default_sims.
_INTER_SEASON_REGRESSION = 0.35
_PLAYOFF_K_MULTIPLIER = 1.75

_CACHE: Dict[str, object] = {"stamp": None, "payload": None}


def applied_home_advantage(config: dict, sport: str = "NFL") -> float:
    """Elo points compute_pregame adds to the home side under ``config``."""
    pre = compute_pregame(
        1500.0,
        1500.0,
        {"sport": sport, "home_team": "HOME", "away_team": "AWAY"},
        config,
    )
    return float(pre["home_elo_adjusted"] - pre["home_elo_pre"])


def _configured_home_field(config: dict) -> Optional[float]:
    adj = (config.get("adjustments") or {}).get("home_field") or {}
    if "value" not in adj:
        return None
    try:
        return float(adj["value"])
    except (TypeError, ValueError):
        return None


def _data_stamp() -> str:
    """Change when slate status or a season CSV changes, so a cached snapshot drops."""
    parts = []
    status = REPO_ROOT / "data" / "slate_status.json"
    if status.is_file():
        parts.append(f"slate:{status.stat().st_mtime_ns}")
    for sport in SPORTS:
        folder = REPO_ROOT / "data" / sport.lower()
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            parts.append(f"{path.name}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


def _default_window(sport: str) -> Tuple[dict, Optional[str], Optional[str], dict]:
    """Target season, warm-up start, and prior used by a default simulation."""
    from app.services.initial_ratings_service import (
        get_available_seasons,
        get_initial_ratings,
        get_simulate_from_options,
        get_simulatable_seasons,
    )

    config = default_config(sport)
    seasons = get_simulatable_seasons(sport)
    season = seasons[-1] if seasons else None
    from_options = get_simulate_from_options(sport, target_season=season)
    from_season = from_options[-1] if from_options else None

    initial: dict = {}
    if from_season is not None and from_season in get_available_seasons(sport):
        try:
            initial = get_initial_ratings(
                sport,
                season=from_season,
                rating_source="regular_season",
                rating_basis="record",
                apply_regression=True,
                regression_strength=0.25,
            )
        except Exception:
            initial = {}
    return config, season, from_season, initial


def _history_seasons(sport: str, season: str, from_season: Optional[str]) -> list:
    """Seasons walked before the target. Matches run_simulation's window."""
    from app.services.initial_ratings_service import get_available_seasons

    all_seasons = get_available_seasons(sport)
    if not all_seasons or str(season) not in all_seasons:
        return []
    target_idx = all_seasons.index(str(season))
    max_history = 2
    if from_season is not None and from_season in all_seasons:
        from_idx = all_seasons.index(from_season)
        max_history = max(1, target_idx - from_idx)
    history = all_seasons[:target_idx]
    if max_history > 0:
        history = history[-max_history:]
    return history


def _sort_schedule(schedule):
    """Same sort keys simulate_season uses before it walks games."""
    sort_cols = []
    if "season" in schedule.columns:
        sort_cols.append("season")
    if "week" in schedule.columns:
        sort_cols.append("week")
    if "date" in schedule.columns and not sort_cols:
        sort_cols.append("date")
    if sort_cols:
        return schedule.sort_values(sort_cols)
    return schedule


def _replay_locked(schedule, config: dict, initial_ratings: dict, sport: str) -> dict:
    """Update Elo on decisive locked games only. Unplayed games stay unsampled.

    Neutral-site games are not sampled, same rule as the warm-up.
    """
    schedule = exclude_neutral_sites(schedule, sport=sport)
    schedule_teams = (
        pd.concat([schedule["home_team"], schedule["away_team"]])
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    initial = dict(initial_ratings or {})
    upper_map = {str(k).upper(): k for k in initial}
    ratings: Dict[str, float] = {}
    for team in schedule_teams:
        if team in initial:
            ratings[team] = float(initial[team])
        elif team.upper() in upper_map:
            ratings[team] = float(initial[upper_map[team.upper()]])
        else:
            ratings[team] = 1500.0
    for team, elo in initial.items():
        ratings.setdefault(team, float(elo))

    for _, game in schedule.iterrows():
        locked = parse_locked_scores(game)
        if locked is None:
            continue
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        ratings.setdefault(home, 1500.0)
        ratings.setdefault(away, 1500.0)
        home_score, away_score = locked
        actual = 1 if home_score > away_score else 0
        week = game["week"] if "week" in game.index else None
        result = run_game(
            home_elo=float(ratings[home]),
            away_elo=float(ratings[away]),
            context={
                "season": game.get("season") if hasattr(game, "get") else None,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "actual": actual,
                "sport": sport,
                "locked": True,
            },
            config=config,
        )
        ratings[home] = float(result["home_elo_post"])
        ratings[away] = float(result["away_elo_post"])
    return ratings


def _project_teams(sport: str, ratings: dict) -> Dict[str, float]:
    """Keep current metadata abbreviations. Direct abbr keys win over name aliases."""
    from metadata import load_teams

    teams = load_teams(sport) or {}
    name_map = _build_name_to_abbr(sport)
    by_abbr: Dict[str, float] = {}
    for key, elo in ratings.items():
        text = str(key).strip()
        abbr = name_map.get(text) or name_map.get(text.upper()) or text
        if abbr in teams:
            by_abbr[abbr] = float(elo)
    for key, elo in ratings.items():
        text = str(key).strip()
        if text in teams:
            by_abbr[text] = float(elo)
        elif text.upper() in teams:
            by_abbr[text.upper()] = float(elo)
    for abbr in teams:
        by_abbr.setdefault(abbr, 1500.0)
    return {abbr: round(float(by_abbr[abbr]), 4) for abbr in sorted(teams)}


def _sport_block(sport: str) -> dict:
    config, season, from_season, initial = _default_window(sport)
    applied = applied_home_advantage(config, sport)
    stored = _configured_home_field(config)

    ratings = dict(initial)
    if season is not None:
        ratings = warm_elo_from_actual_history(
            sport=sport,
            seasons=_history_seasons(sport, season, from_season),
            config=config,
            initial_ratings=initial,
            inter_season_regression=_INTER_SEASON_REGRESSION,
            playoff_k_multiplier=_PLAYOFF_K_MULTIPLIER,
            include_playoffs=True,
        )
        target = _load_season_schedule(sport, str(season))
        if target is not None and not target.empty:
            target = filter_regular_season(target, sport=sport)
            target = _sort_schedule(target)
            ratings = _replay_locked(target, config, ratings, sport)
            lock = describe_schedule_lock(target)
        else:
            lock = {"n_games": 0, "n_locked": 0, "status": "upcoming"}
    else:
        lock = {"n_games": 0, "n_locked": 0, "status": "upcoming"}

    n_locked = int(lock.get("n_locked", 0))
    n_games = int(lock.get("n_games", 0))
    if n_locked > 0:
        origin = (
            f"post-locked {sport} {season} ({n_locked}/{n_games} games locked) "
            f"after the default-simulation warm-up from {from_season}"
        )
    else:
        origin = (
            f"default warm-up entering {sport} {season} "
            f"({n_locked}/{n_games} games locked); unplayed games are not sampled"
        )
    stored_text = "unset" if stored is None else f"{stored:g}"
    source = (
        f"{origin}. home_advantage is the Elo compute_pregame adds under the "
        f"public default config (stored home_field {stored_text}, applied {applied:g})."
    )
    return {
        "home_advantage": applied,
        "teams": _project_teams(sport, ratings),
        "source": source,
    }


def build_elo_snapshot(*, force: bool = False) -> dict:
    """NFL, NHL, and NBA ratings plus the home advantage the default config applies."""
    stamp = _data_stamp()
    cached = _CACHE.get("payload")
    if not force and cached is not None and _CACHE.get("stamp") == stamp:
        return cached  # type: ignore[return-value]
    payload = {
        "as_of": date.today().isoformat(),
        "sports": {sport.lower(): _sport_block(sport) for sport in SPORTS},
    }
    _CACHE["stamp"] = stamp
    _CACHE["payload"] = payload
    return payload


def snapshot_json_text(*, force: bool = False) -> str:
    return json.dumps(build_elo_snapshot(force=force), indent=2) + "\n"


def snapshot_json_bytes(*, force: bool = False) -> bytes:
    return snapshot_json_text(force=force).encode("utf-8")


def write_snapshot(path: Optional[Path] = None, *, force: bool = False) -> Path:
    """Overwrite ``path`` with the snapshot. Default is ``outputs/elo_snapshot.json``."""
    out = Path(path) if path is not None else DEFAULT_OUT
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(snapshot_json_text(force=force), encoding="utf-8")
    return out


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write current NFL, NHL, and NBA Elo ratings to a JSON snapshot."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path (default: outputs/elo_snapshot.json)",
    )
    args = parser.parse_args(argv)

    # Season files are read as data/<sport>/... relative to the repo root.
    import os

    os.chdir(REPO_ROOT)
    out = write_snapshot(args.out, force=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    print(f"Wrote {out}")
    for sport, block in payload["sports"].items():
        print(
            f"  {sport}: {len(block['teams'])} teams, "
            f"home_advantage={block['home_advantage']}"
        )


if __name__ == "__main__":
    main()

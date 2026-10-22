"""k / home-advantage search uses non-neutral regular-season games only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for entry in (ROOT, APP):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from elo_lab.engine import compute_pregame  # noqa: E402
from elo_lab.workflows.generate_default_sims import default_config  # noqa: E402
from elo_lab.workflows.search_k_home import (  # noqa: E402
    EXTENDED_GRIDS,
    NFL_K_FOLLOWUP,
    NFL_PRIOR_EXTENDED,
    PREVIOUS_BEST,
    adopt_recheck_home,
    current_pair,
    historical_eval_season,
    fitting_config,
    grid_position,
    k_axis_position,
    load_fitting_games,
    score_games,
)


def test_public_defaults_match_the_locked_calibration():
    assert current_pair("NHL") == (15.0, 25.0)
    assert current_pair("NBA") == (30.0, 45.0)
    assert current_pair("NFL") == (50.0, 30.0)
    for sport, home in (("NHL", 25), ("NFL", 30), ("NBA", 45)):
        shipped = default_config(sport)
        assert shipped["pregame_pipeline"] == ["home_field"]
        assert shipped["postgame_pipeline"] == []
        assert shipped["adjustments"]["home_field"]["value"] == home
        assert shipped["adjustments"]["elevation_edge"]["enabled"] is False


def test_fitting_config_applies_home_advantage_and_not_elevation():
    plain = compute_pregame(1500, 1500, {"home_team": "A", "away_team": "B"}, fitting_config(20, 0))
    bumped = compute_pregame(1500, 1500, {"home_team": "A", "away_team": "B"}, fitting_config(20, 55))
    assert plain["p_home"] == 0.5
    assert bumped["home_elo_adjusted"] == 1555
    assert bumped["p_home"] > plain["p_home"]
    assert fitting_config(20, 55)["pregame_pipeline"] == ["home_field"]
    assert fitting_config(20, 55)["postgame_pipeline"] == []
    assert fitting_config(20, 55)["adjustments"]["elevation_edge"]["enabled"] is False
    edged = fitting_config(20, 55, elevation=10)
    assert edged["pregame_pipeline"] == ["home_field", "elevation_edge"]
    assert edged["adjustments"]["elevation_edge"]["enabled"] is True
    assert edged["adjustments"]["margin_of_victory"]["enabled"] is False
    plain_den = compute_pregame(
        1500, 1500, {"home_team": "DEN", "away_team": "MIA", "sport": "NFL"}, fitting_config(20, 0),
    )
    boosted = compute_pregame(
        1500, 1500,
        {"home_team": "DEN", "away_team": "MIA", "sport": "NFL"},
        fitting_config(20, 0, elevation=10),
    )
    # Denver 5280 ft, Miami 10 ft. Boost = 10 * 5270 / 1000.
    assert boosted["home_elo_adjusted"] == plain_den["home_elo_adjusted"] + 52.7
    assert boosted["p_home"] > plain_den["p_home"]
    assert adopt_recheck_home(30, 30, 0.01) is False
    assert adopt_recheck_home(30, 25, 0.0004) is False
    assert adopt_recheck_home(30, 25, 0.001) is True


def test_historical_eval_season_is_the_one_before_the_latest_file():
    # Dashboard scores the season before the newest schedule file.
    assert historical_eval_season("NFL") == "2025"
    assert historical_eval_season("NHL") == "2026"
    assert historical_eval_season("NBA") == "2026"


def test_nfl_fitting_games_are_regular_season_and_not_neutral():
    games = load_fitting_games("NFL")
    week = games["week"]
    # Playoff labels do not survive the regular-season filter.
    assert not week.astype(str).str.lower().isin(
        ["wildcard", "division", "confchamp", "superbowl"]
    ).any()
    week_num = pd.to_numeric(week, errors="coerce")
    assert week_num.notna().all()
    assert (week_num <= 18).all()
    if "neutral_site" in games.columns:
        assert not games["neutral_site"].any()

    def has(season, week_value, clubs):
        part = games[games["season"].astype(str).eq(season)]
        hit = part["week"].map(lambda v: str(v) in {str(week_value), f"{week_value}.0"})
        pairs = [
            frozenset({h, a}) == clubs
            for h, a, keep in zip(part["home_team"], part["away_team"], hit)
            if keep
        ]
        return any(pairs)

    # 2025 Brazil game is neutral. The Week 15 rematch is a home game.
    assert not has("2025", 1, frozenset({"LAC", "KC"}))
    assert has("2025", 15, frozenset({"LAC", "KC"}))


def test_extended_grid_marks_interior_and_boundary():
    assert PREVIOUS_BEST["NHL"] == (15.0, 25.0)
    assert "NHL" not in EXTENDED_GRIDS
    nfl_k, nfl_h = EXTENDED_GRIDS["NFL"]
    nba_k, nba_h = EXTENDED_GRIDS["NBA"]
    assert nfl_k == (25, 30, 35, 40)
    assert nfl_h == (20, 25, 30, 35, 45)
    assert nba_k == (25, 30, 35)
    assert nba_h == (35, 40, 45, 50, 55)
    assert grid_position(30, 30, nfl_k, nfl_h) == "interior"
    assert grid_position(40, 30, nfl_k, nfl_h) == "boundary"
    assert grid_position(30, 45, nfl_k, nfl_h) == "boundary"
    assert grid_position(30, 45, nba_k, nba_h) == "interior"
    assert grid_position(35, 45, nba_k, nba_h) == "boundary"
    assert grid_position(25, 35, nba_k, nba_h) == "boundary"


def test_nfl_k_followup_grid_and_k_axis():
    k_values, home_values = NFL_K_FOLLOWUP
    assert k_values == (35, 40, 45, 50)
    assert home_values == (25, 30, 35)
    assert NFL_PRIOR_EXTENDED == (40.0, 30.0)
    assert k_axis_position(40, k_values) == "interior"
    assert k_axis_position(45, k_values) == "interior"
    assert k_axis_position(35, k_values) == "lower boundary"
    assert k_axis_position(50, k_values) == "upper boundary"
    assert current_pair("NFL") == (50.0, 30.0)


def test_home_advantage_changes_the_score_on_a_short_slate():
    games = [
        ("2024", "H", "A", 1.0, 0.0),
        ("2024", "H", "A", 1.0, 0.0),
        ("2024", "A", "H", 0.0, 1.0),
    ]
    low = score_games(games, 20, 0)
    high = score_games(games, 20, 80)
    assert low["n_games"] == 3
    assert low["log_loss"] != high["log_loss"]
    assert low["brier"] != high["brier"]

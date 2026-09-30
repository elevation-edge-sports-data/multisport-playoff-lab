"""Invariants of the current public engine API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for entry in (ROOT, APP):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from elo_lab.adjustments import elevation_edge
from elo_lab.engine import update_ratings, win_probability
from metadata import get_elevation_ft


def test_equal_elo_win_probability_is_one_half():
    assert win_probability(1500, 1500) == pytest.approx(0.5)


def test_home_win_is_zero_sum_when_mov_is_off():
    # Margin of victory off keeps the multiplier at 1, so the K step
    # raises the winner and lowers the loser by the same amount.
    home, away = 1600.0, 1480.0
    p_home = win_probability(home, away)
    new_home, new_away = update_ratings(
        home_elo=home,
        away_elo=away,
        actual=1,
        p_home=p_home,
        k=20,
        multiplier=1.0,
    )
    assert new_home > home
    assert new_away < away
    assert new_home + new_away == pytest.approx(home + away)


def test_elevation_edge_boost_is_zero_when_away_is_higher():
    """Away higher than home => boost 0.

    Expected boost = scale * max(0, home_ft - away_ft) / 1000.
    """
    feet = get_elevation_ft("NBA")
    home_team, away_team = "MIA", "DEN"
    home_ft = float(feet[home_team])
    away_ft = float(feet[away_team])
    scale = 15.0
    expected = scale * max(0.0, home_ft - away_ft) / 1000.0

    state = {
        "home_elo": 1500.0,
        "away_elo": 1500.0,
        "context": {
            "sport": "NBA",
            "home_team": home_team,
            "away_team": away_team,
        },
        "config": {
            "adjustments": {
                "elevation_edge": {"enabled": True, "value": scale},
            },
        },
    }
    out = elevation_edge(state)

    assert away_ft > home_ft
    assert expected == 0.0
    assert out["home_elo"] == pytest.approx(1500.0)

"""95% intervals for playoff proportions, and the odds table stays numeric."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for entry in (ROOT, APP):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from elo_lab.playoffs.uncertainty import (  # noqa: E402
    MAX_PLAYOFF_SIMS,
    format_percent,
    normal_interval,
    playoff_sample_size,
)


def test_normal_interval_is_p_plus_minus_1_96_se():
    p, n = 0.423, 250
    se = math.sqrt(p * (1.0 - p) / n)
    lo, hi = normal_interval(p, n)
    assert lo == pytest.approx(p - 1.96 * se)
    assert hi == pytest.approx(p + 1.96 * se)


def test_normal_interval_clips_to_unit_interval():
    lo, hi = normal_interval(0.99, 25)
    assert lo > 0.0
    assert hi == 1.0
    lo0, hi0 = normal_interval(0.01, 25)
    assert lo0 == 0.0
    assert hi0 < 1.0
    assert normal_interval(0.0, 100) == (0.0, 0.0)
    assert normal_interval(1.0, 100) == (1.0, 1.0)


def test_format_percent_keeps_point_and_shows_range():
    # p=0.5, n=100 → SE=0.05 → ±9.8 percentage points.
    assert format_percent(50.0, 100) == "50.0% (40.2\u201359.8)"
    assert format_percent(50.0, None) == "50.0%"
    assert format_percent(50.0, 0) == "50.0%"
    clipped = format_percent(99.0, 25)
    assert clipped.startswith("99.0% (")
    assert clipped.endswith("\u2013100.0)")


def test_playoff_sample_size_prefers_explicit_count_and_caps_legacy_draws():
    assert playoff_sample_size({"n_playoff_sims": 100, "n_sims": 5000}) == 100
    raw = pd.DataFrame({"sim_id": list(range(1000))})
    assert playoff_sample_size({"raw_sim_results": raw}) == MAX_PLAYOFF_SIMS
    assert playoff_sample_size({"n_sims": 250}) == 250
    assert playoff_sample_size({}) is None


def test_odds_frame_sorts_on_numeric_point_estimates():
    from app.components.moneypuck_viz import (
        _probability_cell_inner,
        build_playoff_odds_frame,
    )

    probs = {
        "KC": {
            "Wild Card": 0.8,
            "Divisional": 0.5,
            "Conference": 0.3,
            "Super Bowl": 0.2,
            "Champion": 0.1,
        },
        "BUF": {
            "Wild Card": 0.4,
            "Divisional": 0.2,
            "Conference": 0.1,
            "Super Bowl": 0.05,
            "Champion": 0.02,
        },
    }
    df = build_playoff_odds_frame("NFL", probs)
    assert df["Make Playoffs"].dtype.kind == "f"
    assert df["Win Super Bowl"].dtype.kind == "f"
    assert list(df["Make Playoffs"]) == sorted(df["Make Playoffs"], reverse=True)
    assert df["Win Super Bowl"].iloc[0] == pytest.approx(10.0)

    cell = _probability_cell_inner(42.3, 250)
    assert "42.3%" in cell
    assert "\u2013" in cell
    assert cell.index("42.3%") < cell.index("\u2013")
    assert _probability_cell_inner(42.3, None) == "42.3%"

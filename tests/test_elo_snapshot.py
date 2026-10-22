"""JSON Elo snapshot: schema, abbreviations, and applied home advantage."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for entry in (ROOT, APP):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from elo_lab.workflows.export_elo_snapshot import (  # noqa: E402
    applied_home_advantage,
    build_elo_snapshot,
)
from elo_lab.workflows.generate_default_sims import default_config  # noqa: E402
from metadata import load_teams  # noqa: E402


def test_snapshot_matches_compute_pregame_and_metadata_abbreviations():
    snapshot = build_elo_snapshot(force=True)

    assert snapshot["as_of"] == date.today().isoformat()
    assert list(snapshot["sports"]) == ["nfl", "nhl", "nba"]

    for sport in ("NFL", "NHL", "NBA"):
        block = snapshot["sports"][sport.lower()]
        assert set(block) == {"home_advantage", "teams", "source"}
        assert block["home_advantage"] == applied_home_advantage(default_config(sport), sport)
        assert set(block["teams"]) == set(load_teams(sport))
        assert all(isinstance(value, float) for value in block["teams"].values())

    # Locked slates move ratings off a flat 1500. NBA stays on the warm-up path
    # whether or not a few target-season games have a score.
    assert len(set(snapshot["sports"]["nfl"]["teams"].values())) > 1
    assert len(set(snapshot["sports"]["nhl"]["teams"].values())) > 1
    assert len(set(snapshot["sports"]["nba"]["teams"].values())) > 1
    assert "post-locked" in snapshot["sports"]["nfl"]["source"]
    assert "warm-up" in snapshot["sports"]["nba"]["source"]
    assert snapshot["sports"]["nhl"]["home_advantage"] == 25.0
    assert snapshot["sports"]["nfl"]["home_advantage"] == 35.0
    assert snapshot["sports"]["nba"]["home_advantage"] == 45.0

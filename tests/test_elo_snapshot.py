"""JSON Elo snapshot: schema, abbreviations, and applied home advantage."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for entry in (ROOT, APP):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from elo_lab import __version__  # noqa: E402
from elo_lab.workflows.export_elo_snapshot import (  # noqa: E402
    applied_home_advantage,
    build_elo_snapshot,
    snapshot_json_text,
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

    generated = datetime.fromisoformat(snapshot["generated_at"])
    assert generated.tzinfo is not None
    assert generated.utcoffset() == timezone.utc.utcoffset(None)
    assert snapshot["version"] == f"v{__version__}"
    assert snapshot["n_teams"] == {
        sport: len(block["teams"]) for sport, block in snapshot["sports"].items()
    }
    assert set(snapshot["lock_status"]) == set(snapshot["sports"])
    for sport, note in snapshot["lock_status"].items():
        assert set(note) == {"season", "n_locked", "n_games", "status"}
        assert isinstance(note["season"], str) and note["season"]
        assert 0 <= note["n_locked"] <= note["n_games"]
        assert note["status"] in {"upcoming", "in_progress", "final"}

    # CLI write and the dashboard download both serialize this same builder.
    exported = json.loads(snapshot_json_text())
    assert set(exported) == {
        "as_of",
        "generated_at",
        "version",
        "n_teams",
        "lock_status",
        "sports",
    }
    assert exported["sports"] == snapshot["sports"]
    assert exported["n_teams"] == snapshot["n_teams"]
    assert exported["lock_status"] == snapshot["lock_status"]
    assert exported["version"] == snapshot["version"]
    datetime.fromisoformat(exported["generated_at"])

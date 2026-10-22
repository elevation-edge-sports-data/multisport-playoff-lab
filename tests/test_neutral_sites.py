"""Neutral-site rows are flagged from source flags and dropped on Elo walks."""

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

from elo_lab.neutral_sites import (  # noqa: E402
    NFL_INTERNATIONAL_GAMES,
    arena_is_neutral,
    exclude_neutral_sites,
    neutral_mask,
    notes_mark_neutral,
)
from elo_lab.workflows.simulate_season import (  # noqa: E402
    _load_season_schedule,
    filter_regular_season,
    warm_elo_from_actual_history,
)

DATA = ROOT / "data"


def _nfl_source(season: str) -> pd.DataFrame:
    return pd.read_csv(DATA / "nfl" / f"nfl_{season}.csv")


def _row_teams(row) -> frozenset:
    if "Winner/tie" in row.index:
        return frozenset({str(row["Winner/tie"]).strip(), str(row["Loser/tie"]).strip()})
    return frozenset({str(row["VisTm"]).strip(), str(row["HomeTm"]).strip()})


def test_explicit_helpers():
    assert notes_mark_neutral("Neutral")
    assert notes_mark_neutral("at O2 Arena (Prague CZ)")
    assert notes_mark_neutral("at Mexico City Mexico")
    assert not notes_mark_neutral("NBA Cup")
    assert not notes_mark_neutral("Play-In Game")
    assert not notes_mark_neutral("No point for regulation tie for Minnesota (pulled goalie in OT).")
    assert arena_is_neutral("T-Mobile Arena")
    assert arena_is_neutral("Mexico City Arena")
    assert not arena_is_neutral("TD Garden")
    assert not arena_is_neutral("Moody Center")
    assert not arena_is_neutral("Alamodome")


def test_hall_of_fame_week_label_is_neutral_and_not_in_these_files():
    frame = pd.DataFrame(
        {
            "Week": ["Hall of Fame"],
            "Winner/tie": ["Buffalo Bills"],
            "Unnamed: 5": [""],
            "Loser/tie": ["Chicago Bears"],
        }
    )
    assert bool(neutral_mask(frame, "NFL").iloc[0])
    for path in (DATA / "nfl").glob("nfl_*.csv"):
        weeks = pd.read_csv(path, usecols=["Week"])["Week"].astype(str).str.lower()
        assert not weeks.str.contains("hall of fame").any()


def test_nfl_source_flags_super_bowl_and_international_only():
    listed = {}
    for date, team_a, team_b, _venue in NFL_INTERNATIONAL_GAMES:
        listed.setdefault(date[:4], []).append(frozenset({team_a, team_b}))

    for season, games in listed.items():
        frame = _nfl_source(season)
        mask = neutral_mask(frame, "NFL", season=season)
        super_bowls = frame["Week"].astype(str).str.lower().eq("superbowl").sum() if "Week" in frame.columns else 0
        assert int(mask.sum()) == len(games) + int(super_bowls)
        flagged = [
            _row_teams(row)
            for _, row in frame.loc[mask].iterrows()
        ]
        for pair in games:
            assert pair in flagged
        # A division rematch of an international pairing stays in the fit.
        if season == "2025":
            week15 = frame["Week"].astype(str).eq("15")
            chiefs = frame.apply(_row_teams, axis=1).map(lambda p: p == frozenset({
                "Los Angeles Chargers", "Kansas City Chiefs",
            }))
            assert not bool(mask[week15 & chiefs].any())
            assert bool(mask[frame["Week"].astype(str).str.lower().eq("superbowl")].all())


def test_nfl_2026_return_game_is_not_neutral():
    frame = _nfl_source("2026")
    mask = neutral_mask(frame, "NFL", season="2026")
    week = frame["Week"].astype(str)
    pair = frame.apply(_row_teams, axis=1)
    rams = pair.map(lambda p: p == frozenset({"San Francisco 49ers", "Los Angeles Rams"}))
    assert bool(mask[week.eq("1") & rams].all())
    assert not bool(mask[week.eq("14") & rams].any())


def test_nfl_games_csv_drops_2025_neutrals_and_keeps_the_rematch():
    frame = pd.read_csv(DATA / "nfl_games.csv")
    kept = exclude_neutral_sites(frame, sport="NFL")
    assert len(frame) - len(kept) == 8

    def pair_week(df, week, clubs):
        hit = df["week"].astype(int).eq(week)
        clubs_ok = [
            frozenset({h, a}) == clubs
            for h, a in zip(df.loc[hit, "home_team"], df.loc[hit, "away_team"])
        ]
        return any(clubs_ok)

    chargers = frozenset({"LAC", "KC"})
    assert pair_week(frame, 1, chargers)
    assert not pair_week(kept, 1, chargers)
    assert pair_week(kept, 15, chargers)
    assert pair_week(frame, 22, frozenset({"SEA", "NE"}))
    assert not pair_week(kept, 22, frozenset({"SEA", "NE"}))
    assert (kept["week"].astype(int) == 21).any()


def test_run_backtest_does_not_score_neutral_games(monkeypatch):
    seen = []

    def fake_run_game(**kwargs):
        seen.append(kwargs["context"])
        return {"home_elo_post": 1500.0, "away_elo_post": 1500.0, "p_home": 0.5}

    monkeypatch.setattr("elo_lab.workflows.run_backtest.run_game", fake_run_game)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *args, **kwargs: None)
    from elo_lab.workflows.run_backtest import run_backtest

    result = run_backtest({"k": 20, "adjustments": {}}, "neutral_filter_check")
    assert (result["week"].astype(int) == 22).sum() == 0
    assert not any(
        frozenset({ctx["home_team"], ctx["away_team"]}) == frozenset({"LAC", "KC"})
        and int(ctx["week"]) == 1
        for ctx in seen
    )
    assert any(
        frozenset({ctx["home_team"], ctx["away_team"]}) == frozenset({"LAC", "KC"})
        and int(ctx["week"]) == 15
        for ctx in seen
    )


def test_nhl_at_venue_notes_only():
    frame = pd.read_csv(DATA / "nhl" / "nhl_2024.csv")
    mask = neutral_mask(frame, "NHL")
    notes = frame["Notes"].fillna("").astype(str)
    assert int(mask.sum()) == int(notes.str.match(r"(?i)^\s*at\s+\S").sum())
    pulled = notes.str.startswith("No point")
    assert pulled.any()
    assert not bool(mask[pulled].any())
    prague = pd.read_csv(DATA / "nhl" / "nhl_2025.csv")
    prague_mask = neutral_mask(prague, "NHL")
    assert int(prague_mask.sum()) == 6
    assert prague.loc[prague_mask, "Notes"].astype(str).str.startswith("at ").all()


def test_nba_arena_list_and_neutral_token():
    raw = pd.read_csv(DATA / "nba" / "raw" / "nba_2025.csv", dtype=str)
    mask = neutral_mask(raw, "NBA")
    arena = raw["Arena"].fillna("")
    assert set(arena[mask].unique()) == {"Mexico City Arena", "Accor Arena", "T-Mobile Arena"}
    assert not bool(mask[arena.eq("Moody Center")].any())
    cup_home = raw["Notes"].fillna("").str.contains("NBA Cup") & ~arena.eq("T-Mobile Arena")
    assert cup_home.any()
    assert not bool(mask[cup_home].any())

    upcoming = pd.read_csv(DATA / "nba" / "nba_2027.csv", dtype=str)
    upcoming_mask = neutral_mask(upcoming, "NBA")
    assert int(upcoming_mask.sum()) == 3
    assert upcoming.loc[upcoming_mask, "Notes"].eq("Neutral").all()


def test_loader_carries_flag_from_raw_arena():
    schedule = _load_season_schedule("NBA", "2025")
    assert "neutral_site" in schedule.columns
    assert int(schedule["neutral_site"].sum()) == 6
    schedule_2026 = _load_season_schedule("NFL", "2026")
    rams = schedule_2026["home_team"].isin(["SF", "LAR"]) & schedule_2026["away_team"].isin(["SF", "LAR"])
    week = schedule_2026["week"].astype(str)
    assert bool(schedule_2026.loc[rams & week.eq("1"), "neutral_site"].all())
    assert not bool(schedule_2026.loc[rams & week.eq("14"), "neutral_site"].any())


def test_warm_elo_skips_neutral_rows(monkeypatch):
    calls = []

    def fake_run_game(**kwargs):
        calls.append(kwargs["context"])
        return {"home_elo_post": 1500.0, "away_elo_post": 1500.0, "p_home": 0.5}

    monkeypatch.setattr("elo_lab.workflows.simulate_season.run_game", fake_run_game)
    warm_elo_from_actual_history(
        "NFL",
        ["2025"],
        config={"k": 20, "adjustments": {}},
        include_playoffs=True,
    )
    schedule = _load_season_schedule("NFL", "2025")
    played = schedule.dropna(subset=["home_score", "away_score"])
    decisive = played[played["home_score"] != played["away_score"]]
    expected = int((~decisive["neutral_site"]).sum())
    assert len(calls) == expected
    assert expected < len(decisive)


def test_replay_locked_skips_played_international(monkeypatch):
    from elo_lab.workflows.export_elo_snapshot import _replay_locked

    calls = []

    def fake_run_game(**kwargs):
        calls.append(kwargs["context"])
        return {"home_elo_post": 1500.0, "away_elo_post": 1500.0, "p_home": 0.5}

    monkeypatch.setattr("elo_lab.workflows.export_elo_snapshot.run_game", fake_run_game)
    schedule = filter_regular_season(_load_season_schedule("NFL", "2026"), sport="NFL")
    _replay_locked(schedule, {"k": 20, "adjustments": {}}, {}, "NFL")
    assert not any(
        frozenset({ctx["home_team"], ctx["away_team"]}) == frozenset({"SF", "LAR"})
        and str(ctx["week"]) == "1"
        for ctx in calls
    )
    assert any(
        frozenset({ctx["home_team"], ctx["away_team"]}) == frozenset({"NE", "SEA"})
        for ctx in calls
    )


def test_score_config_on_season_skips_nba_neutrals():
    from app.services.evaluation_service import score_config_on_season

    schedule = filter_regular_season(_load_season_schedule("NBA", "2025"), sport="NBA")
    played = schedule.dropna(subset=["home_score", "away_score"])
    decisive = played[played["home_score"] != played["away_score"]]
    expected = int((~decisive["neutral_site"]).sum())
    result = score_config_on_season("NBA", "2025", {"k": 20, "adjustments": {}})
    assert "error" not in result
    assert result["n_games"] == expected
    assert expected < len(decisive)

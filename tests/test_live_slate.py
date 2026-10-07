"""Live slate ingestion: schema match, apply scores, no network."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elo_lab.workflows.live_slate import (
    CompletedGame,
    SlateReport,
    apply_completed_games,
    detect_schema,
    fetch_completed_for_csv,
    fetch_nba_scores,
    parse_espn_events,
    parse_nba_cdn_scoreboard,
    parse_nflverse_rows,
    parse_nhl_score_payload,
    season_overrides_for_refresh,
    update_all_current,
    update_season_scores,
)
from elo_lab.workflows.simulate_season import describe_schedule_lock, normalize_schedule


def _nfl_upcoming() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Week": [1, 1, 2],
            "Day": ["Wed", "Thu", "Sun"],
            "Unnamed: 2": ["September 9", "September 10", "September 20"],
            "VisTm": ["New England Patriots", "San Francisco 49ers", "Carolina Panthers"],
            "Pts": [pd.NA, pd.NA, pd.NA],
            "Unnamed: 5": ["@", "@", "@"],
            "HomeTm": ["Seattle Seahawks", "Los Angeles Rams", "Atlanta Falcons"],
            "Pts.1": [pd.NA, pd.NA, pd.NA],
            "Time": ["8:20 PM", "8:35 PM", "1:00 PM"],
        }
    )


def test_detect_nfl_upcoming_schema():
    schema = detect_schema(_nfl_upcoming())
    assert schema["away"] == "VisTm"
    assert schema["home"] == "HomeTm"
    assert schema["away_pts"] == "Pts"
    assert schema["home_pts"] == "Pts.1"
    assert schema["week"] == "Week"


def test_apply_locks_matching_week_and_skips_unplayed():
    games = [
        CompletedGame("NFL", "New England Patriots", "Seattle Seahawks", 10, 13, week=1),
        CompletedGame("NFL", "San Francisco 49ers", "Los Angeles Rams", 17, 14, week=1),
    ]
    out, applied, unmatched, already = apply_completed_games(_nfl_upcoming(), games)
    assert len(applied) == 2
    assert unmatched == []
    assert already == 0
    assert float(out.loc[0, "Pts"]) == 10
    assert float(out.loc[0, "Pts.1"]) == 13
    assert pd.isna(out.loc[2, "Pts"])
    canon = normalize_schedule(out, sport="NFL")
    lock = describe_schedule_lock(canon)
    assert lock["n_locked"] == 2
    assert lock["status"] == "in_progress"


def test_apply_does_not_overwrite_unless_asked():
    df = _nfl_upcoming()
    df.loc[0, "Pts"] = 10
    df.loc[0, "Pts.1"] = 13
    games = [CompletedGame("NFL", "New England Patriots", "Seattle Seahawks", 99, 99, week=1)]
    out, applied, unmatched, already = apply_completed_games(df, games, overwrite=False)
    assert applied == []
    assert already == 1
    assert float(out.loc[0, "Pts"]) == 10


def test_parse_espn_final_only():
    payload = {
        "content": {
            "sbData": {
                "week": {"number": 1},
                "events": [
                    {
                        "date": "2026-09-10T00:20Z",
                        "week": {"number": 1},
                        "status": {"type": {"completed": True, "state": "post", "name": "STATUS_FINAL"}},
                        "competitions": [
                            {
                                "status": {"type": {"completed": True, "state": "post", "name": "STATUS_FINAL"}},
                                "competitors": [
                                    {"homeAway": "home", "score": "13", "team": {"displayName": "Seattle Seahawks"}},
                                    {"homeAway": "away", "score": "10", "team": {"displayName": "New England Patriots"}},
                                ],
                            }
                        ],
                    },
                    {
                        "date": "2026-09-20T17:00Z",
                        "week": {"number": 2},
                        "status": {"type": {"completed": False, "state": "pre", "name": "STATUS_SCHEDULED"}},
                        "competitions": [
                            {
                                "status": {"type": {"completed": False, "state": "pre"}},
                                "competitors": [
                                    {"homeAway": "home", "score": "0", "team": {"displayName": "Atlanta Falcons"}},
                                    {"homeAway": "away", "score": "0", "team": {"displayName": "Carolina Panthers"}},
                                ],
                            }
                        ],
                    },
                ],
            }
        }
    }
    games = parse_espn_events("NFL", payload)
    assert len(games) == 1
    assert games[0].home == "Seattle Seahawks"
    assert games[0].away_score == 10
    assert games[0].home_score == 13
    assert games[0].week == 1


def test_parse_nflverse_maps_la_to_rams_and_skips_blank():
    rows = [
        {
            "season": "2026",
            "game_type": "REG",
            "week": "1",
            "gameday": "2026-09-09",
            "away_team": "NE",
            "away_score": "10",
            "home_team": "SEA",
            "home_score": "13",
        },
        {
            "season": "2026",
            "game_type": "REG",
            "week": "1",
            "gameday": "2026-09-10",
            "away_team": "SF",
            "away_score": "27",
            "home_team": "LA",
            "home_score": "7",
        },
        {
            "season": "2026",
            "game_type": "REG",
            "week": "2",
            "gameday": "2026-09-20",
            "away_team": "CAR",
            "away_score": "",
            "home_team": "ATL",
            "home_score": "",
        },
        {
            "season": "2026",
            "game_type": "PRE",
            "week": "0",
            "away_team": "NE",
            "away_score": "14",
            "home_team": "SEA",
            "home_score": "7",
        },
    ]
    games = parse_nflverse_rows(rows, 2026)
    assert len(games) == 2
    assert games[0].away == "New England Patriots"
    assert games[1].home == "Los Angeles Rams"
    assert games[1].away_score == 27


def test_parse_nhl_official_skips_preseason_and_unplayed():
    payload = {
        "games": [
            {
                "gameType": 1,
                "gameState": "FINAL",
                "gameDate": "2026-09-19",
                "awayTeam": {"abbrev": "DAL", "score": 2, "name": {"default": "Stars"}},
                "homeTeam": {"abbrev": "STL", "score": 1, "name": {"default": "Blues"}},
            },
            {
                "gameType": 2,
                "gameState": "FUT",
                "gameDate": "2026-09-29",
                "awayTeam": {"abbrev": "FLA", "name": {"default": "Panthers"}},
                "homeTeam": {"abbrev": "CAR", "name": {"default": "Hurricanes"}},
            },
            {
                "gameType": 2,
                "gameState": "FINAL",
                "gameDate": "2026-09-29",
                "awayTeam": {"abbrev": "FLA", "score": 3, "name": {"default": "Panthers"}},
                "homeTeam": {"abbrev": "CAR", "score": 2, "name": {"default": "Hurricanes"}},
            },
        ]
    }
    games = parse_nhl_score_payload(payload)
    assert len(games) == 1
    assert games[0].away == "Florida Panthers"
    assert games[0].home == "Carolina Hurricanes"
    assert games[0].away_score == 3


def test_parse_nba_cdn_finals_only():
    payload = {
        "scoreboard": {
            "gameDate": "2025-10-21",
            "games": [
                {
                    "gameStatus": 3,
                    "gameStatusText": "Final",
                    "awayTeam": {
                        "teamTricode": "HOU",
                        "teamCity": "Houston",
                        "teamName": "Rockets",
                        "score": 124,
                    },
                    "homeTeam": {
                        "teamTricode": "OKC",
                        "teamCity": "Oklahoma City",
                        "teamName": "Thunder",
                        "score": 125,
                    },
                },
                {
                    "gameStatus": 2,
                    "gameStatusText": "Q3 5:12",
                    "awayTeam": {
                        "teamTricode": "DET",
                        "teamCity": "Detroit",
                        "teamName": "Pistons",
                        "score": 70,
                    },
                    "homeTeam": {
                        "teamTricode": "BOS",
                        "teamCity": "Boston",
                        "teamName": "Celtics",
                        "score": 80,
                    },
                },
            ],
        }
    }
    games = parse_nba_cdn_scoreboard(payload)
    assert len(games) == 1
    assert games[0].away == "Houston Rockets"
    assert games[0].home == "Oklahoma City Thunder"
    assert games[0].home_score == 125
    assert games[0].date == "2025-10-21"


def test_parse_espn_nba_final_only():
    payload = {
        "events": [
            {
                "date": "2025-10-21T02:00Z",
                "competitions": [
                    {
                        "status": {
                            "type": {
                                "completed": True,
                                "state": "post",
                                "name": "STATUS_FINAL",
                            }
                        },
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "125",
                                "team": {"displayName": "Oklahoma City Thunder"},
                            },
                            {
                                "homeAway": "away",
                                "score": "124",
                                "team": {"displayName": "Houston Rockets"},
                            },
                        ],
                    }
                ],
            },
            {
                "date": "2025-10-22T00:00Z",
                "competitions": [
                    {
                        "status": {
                            "type": {
                                "completed": False,
                                "state": "in",
                                "name": "STATUS_IN_PROGRESS",
                            }
                        },
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "40",
                                "team": {"displayName": "Boston Celtics"},
                            },
                            {
                                "homeAway": "away",
                                "score": "38",
                                "team": {"displayName": "Detroit Pistons"},
                            },
                        ],
                    }
                ],
            },
        ]
    }
    games = parse_espn_events("NBA", payload)
    assert len(games) == 1
    assert games[0].away == "Houston Rockets"
    assert games[0].home_score == 125


def test_fetch_nba_scores_uses_espn_when_cdn_fails(monkeypatch):
    import urllib.error
    from datetime import date

    from elo_lab.workflows import live_slate as live_slate_mod

    def _cdn_down():
        raise urllib.error.URLError("cdn unavailable")

    final = CompletedGame(
        "NBA",
        "Houston Rockets",
        "Oklahoma City Thunder",
        124,
        125,
        date="2025-10-21",
    )
    monkeypatch.setattr(live_slate_mod, "fetch_nba_cdn_today", _cdn_down)
    monkeypatch.setattr(live_slate_mod, "fetch_espn_nba_dates", lambda days: [final])

    games = fetch_nba_scores([date(2025, 10, 21)])
    assert len(games) == 1
    assert games[0].away == "Houston Rockets"
    assert games[0].home_score == 125
    assert "key" not in (live_slate_mod.fetch_nba_scores.last_source or "").lower()


def test_nba_refresh_does_not_require_env_key(monkeypatch):
    from elo_lab.workflows import live_slate as live_slate_mod

    assert not hasattr(live_slate_mod, "_balldontlie_key")

    def _no_network(*_args, **_kwargs):
        raise AssertionError("unit test must not hit the network")

    monkeypatch.setattr(live_slate_mod, "_http_json", _no_network)
    monkeypatch.setattr(live_slate_mod, "fetch_nba_cdn_today", lambda: [])
    monkeypatch.setattr(live_slate_mod, "fetch_espn_nba_dates", lambda days: [])

    df = pd.DataFrame(
        {
            "Date": ["2025-10-21"],
            "Visitor/Neutral": ["Houston Rockets"],
            "PTS": [pd.NA],
            "Home/Neutral": ["Oklahoma City Thunder"],
            "PTS.1": [pd.NA],
        }
    )
    games = fetch_completed_for_csv("NBA", "2026", df)
    assert games == []


def test_update_season_scores_writes_csv():
    import tempfile
    from elo_lab.workflows import live_slate as live_slate_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data" / "nfl"
        data_dir.mkdir(parents=True)
        path = data_dir / "nfl_2026.csv"
        _nfl_upcoming().to_csv(path, index=False)
        original_status = live_slate_mod.SLATE_STATUS_PATH
        live_slate_mod.SLATE_STATUS_PATH = tmp_path / "data" / "slate_status.json"
        try:
            games = [
                CompletedGame("NFL", "New England Patriots", "Seattle Seahawks", 10, 13, week=1)
            ]
            report = update_season_scores("NFL", "2026", root=tmp_path, games=games)
            assert report.updated == 1
            assert report.n_locked == 1
            written = pd.read_csv(path)
            assert float(written.loc[0, "Pts"]) == 10
            status = json.loads((tmp_path / "data" / "slate_status.json").read_text())
            assert status["NFL:2026"]["n_locked"] == 1
        finally:
            live_slate_mod.SLATE_STATUS_PATH = original_status


def test_feedback_line_lists_lock_and_newly_locked():
    report = SlateReport(
        sport="NFL",
        season="2026",
        path="",
        updated=2,
        n_games=272,
        n_locked=16,
        status="in_progress",
    )
    assert (
        report.feedback_line()
        == "NFL 2026 · in progress · 16/272 games locked · 2 newly locked"
    )
    report.error = "feed down"
    assert report.feedback_line().endswith("· feed down")


def test_season_overrides_keep_each_sport_on_its_own_season():
    assert season_overrides_for_refresh("NFL", "2026", None) == {"NFL": "2026"}
    mixed = season_overrides_for_refresh(
        "nhl",
        "2026",
        {"NFL": "2025", "NBA": "2027", "NHL": "1999"},
    )
    assert mixed == {"NHL": "2026", "NFL": "2025", "NBA": "2027"}
    active_only = season_overrides_for_refresh("NFL", "2026", {"NFL": "2024"})
    assert active_only == {"NFL": "2026"}


def _write_mini_slate(root: Path, sport: str, season: str, scored: bool = False) -> Path:
    data_dir = root / "data" / sport.lower()
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = _nfl_upcoming()
    if scored:
        frame.loc[0, "Pts"] = 10
        frame.loc[0, "Pts.1"] = 13
    path = data_dir / f"{sport.lower()}_{season}.csv"
    frame.to_csv(path, index=False)
    return path


def test_update_all_current_refreshes_each_sport_on_its_own_season(monkeypatch):
    from elo_lab.workflows import live_slate as live_slate_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nfl_path = _write_mini_slate(tmp_path, "NFL", "2026")
        nhl_path = _write_mini_slate(tmp_path, "NHL", "2027", scored=True)
        _write_mini_slate(tmp_path, "NBA", "2027")
        original_status = live_slate_mod.SLATE_STATUS_PATH
        live_slate_mod.SLATE_STATUS_PATH = tmp_path / "data" / "slate_status.json"

        def fake_default(sport):
            return {"NFL": "1999", "NHL": "2027", "NBA": "2027"}[sport.upper()]

        def fake_fetch(sport, season, df):
            sport_u = sport.upper()
            if sport_u == "NBA":
                raise RuntimeError("nba feed down")
            if sport_u == "NFL":
                return [
                    CompletedGame(
                        "NFL", "New England Patriots", "Seattle Seahawks", 10, 13, week=1
                    )
                ]
            if sport_u == "NHL":
                return [
                    CompletedGame(
                        "NHL", "New England Patriots", "Seattle Seahawks", 99, 99, week=1
                    )
                ]
            return []

        monkeypatch.setattr(live_slate_mod, "default_target_season", fake_default)
        monkeypatch.setattr(live_slate_mod, "fetch_completed_for_csv", fake_fetch)
        try:
            reports = update_all_current(root=tmp_path, seasons={"NFL": "2026"})
        finally:
            live_slate_mod.SLATE_STATUS_PATH = original_status

        assert [r.sport for r in reports] == ["NFL", "NHL", "NBA"]
        assert [r.season for r in reports] == ["2026", "2027", "2027"]
        assert reports[0].error is None
        assert reports[0].updated == 1
        assert (
            reports[0].feedback_line()
            == "NFL 2026 · in progress · 1/3 games locked · 1 newly locked"
        )
        written = pd.read_csv(nfl_path)
        assert float(written.loc[0, "Pts"]) == 10
        assert reports[1].error is None
        assert reports[1].updated == 0
        nhl_written = pd.read_csv(nhl_path)
        assert float(nhl_written.loc[0, "Pts"]) == 10
        assert reports[2].error == "nba feed down"
        assert "nba feed down" in reports[2].feedback_line()


def test_update_all_current_one_failure_does_not_abort(monkeypatch):
    from elo_lab.workflows import live_slate as live_slate_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for sport, season in (("NFL", "2026"), ("NHL", "2027"), ("NBA", "2027")):
            _write_mini_slate(tmp_path, sport, season)
        original_status = live_slate_mod.SLATE_STATUS_PATH
        live_slate_mod.SLATE_STATUS_PATH = tmp_path / "data" / "slate_status.json"
        calls = []
        real = live_slate_mod.update_season_scores

        def wrapped(sport, season, root=live_slate_mod.ROOT, overwrite=False, dry_run=False, games=None):
            calls.append((sport.upper(), overwrite))
            if sport.upper() == "NHL":
                raise RuntimeError("nhl down")
            return real(
                sport,
                season,
                root=root,
                overwrite=overwrite,
                dry_run=dry_run,
                games=[] if games is None else games,
            )

        monkeypatch.setattr(live_slate_mod, "update_season_scores", wrapped)
        try:
            reports = update_all_current(
                root=tmp_path,
                seasons={"NFL": "2026", "NHL": "2027", "NBA": "2027"},
            )
        finally:
            live_slate_mod.SLATE_STATUS_PATH = original_status

        assert [r.sport for r in reports] == ["NFL", "NHL", "NBA"]
        assert calls == [("NFL", False), ("NHL", False), ("NBA", False)]
        assert reports[0].error is None
        assert reports[1].error == "nhl down"
        assert reports[2].error is None


def test_update_all_current_missing_season_is_an_error_report(monkeypatch):
    from elo_lab.workflows import live_slate as live_slate_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_mini_slate(tmp_path, "NFL", "2026")
        _write_mini_slate(tmp_path, "NHL", "2027")
        original_status = live_slate_mod.SLATE_STATUS_PATH
        live_slate_mod.SLATE_STATUS_PATH = tmp_path / "data" / "slate_status.json"

        def fake_default(sport):
            if sport.upper() == "NBA":
                return None
            return {"NFL": "2026", "NHL": "2027"}[sport.upper()]

        monkeypatch.setattr(live_slate_mod, "default_target_season", fake_default)
        monkeypatch.setattr(live_slate_mod, "fetch_completed_for_csv", lambda *args, **kwargs: [])
        try:
            reports = update_all_current(root=tmp_path)
        finally:
            live_slate_mod.SLATE_STATUS_PATH = original_status

        assert [r.sport for r in reports] == ["NFL", "NHL", "NBA"]
        assert reports[0].error is None
        assert reports[1].error is None
        assert reports[2].error == "No season file for NBA"
        assert reports[2].feedback_line() == "NBA · No season file for NBA"

"""
Live slate: pull completed scores and lock them into the season CSV.

v15.1 sources
  NFL — nflverse schedules release (games.csv)
  NHL — official NHL web API (api-web.nhle.com)
  NBA — NBA CDN today's scoreboard, then ESPN scoreboard by date (no key)

Writes Pts / G columns on the existing per-season files so v14 lock logic
can see real results.

Usage:
    python -m elo_lab.workflows.live_slate --sport NFL --season 2026
    python -m elo_lab.workflows.live_slate --sport NBA --season 2026
    python -m elo_lab.workflows.live_slate --sport all
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from elo_lab.workflows.simulate_season import (
    describe_schedule_lock,
    filter_regular_season,
    format_lock_line,
    normalize_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
SLATE_STATUS_PATH = ROOT / "data" / "slate_status.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

NFLVERSE_SCHEDULES_CSV = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
)
NHL_WEB_API = "https://api-web.nhle.com/v1"
NBA_CDN_SCOREBOARD = (
    "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
)
ESPN_NBA_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
NBA_CDN_HEADERS = {
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Feed abbreviations that differ from dashboard metadata
ABBR_ALIASES = {
    "NFL": {"LA": "LAR", "WSH": "WAS", "JAC": "JAX"},
    "NHL": {"SJ": "SJS", "NJ": "NJD", "TB": "TBL", "LA": "LAK", "UTAH": "UTA", "ARI": "UTA"},
    "NBA": {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "PHO": "PHX", "WSH": "WAS", "BKN": "BKN", "BRK": "BKN"},
}

DEFAULT_SOURCE = {
    "NFL": "nflverse",
    "NHL": "nhl-web-api",
    "NBA": "nba-cdn+espn",
}

NAME_ALIASES = {
    "washington football team": "washington commanders",
    "washington redskins": "washington commanders",
    "utah hockey club": "utah mammoth",
    "arizona coyotes": "utah mammoth",
    "phoenix coyotes": "utah mammoth",
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "la rams": "los angeles rams",
    "la chargers": "los angeles chargers",
    "ny jets": "new york jets",
    "ny giants": "new york giants",
    "ny rangers": "new york rangers",
    "ny islanders": "new york islanders",
    "ny knicks": "new york knicks",
    "golden state": "golden state warriors",
}


@dataclass
class CompletedGame:
    sport: str
    away: str
    home: str
    away_score: float
    home_score: float
    week: Optional[int] = None
    date: Optional[str] = None
    status: str = "final"


@dataclass
class SlateReport:
    sport: str
    season: str
    path: str
    fetched: int = 0
    updated: int = 0
    already_locked: int = 0
    unmatched: int = 0
    n_games: int = 0
    n_locked: int = 0
    status: str = "upcoming"
    fetched_at: str = ""
    source: str = ""
    applied: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_games: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def lock_line(self) -> str:
        return format_lock_line(
            self.sport,
            self.season,
            {"n_games": self.n_games, "n_locked": self.n_locked, "status": self.status},
        )

    def feedback_line(self) -> str:
        """One sidebar line: lock summary, newly locked count, and any error."""
        if self.error and not self.n_games and not self.updated and not self.fetched:
            season = f" {self.season}" if self.season else ""
            return f"{self.sport}{season} · {self.error}"
        line = f"{self.lock_line()} · {self.updated} newly locked"
        if self.error:
            return f"{line} · {self.error}"
        return line

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lock_line"] = self.lock_line()
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def season_csv_path(sport: str, season: str, root: Path = ROOT) -> Path:
    sport_l = sport.lower()
    return root / "data" / sport_l / f"{sport_l}_{season}.csv"


def _norm_name(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace(".", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_ALIASES.get(s, s)


def _http_json(url: str, timeout: int = 20, headers: Optional[dict] = None) -> dict:
    h = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _http_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _load_team_meta(sport: str) -> dict:
    sport_u = sport.upper()
    app_dir = ROOT / "app"
    if str(app_dir) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(app_dir))
    try:
        from metadata import load_teams
        return load_teams(sport_u) or {}
    except Exception:
        pass
    try:
        if sport_u == "NFL":
            from metadata.nfl_teams import NFL_TEAMS
            return NFL_TEAMS
        if sport_u == "NHL":
            from metadata.nhl_teams import NHL_TEAMS
            return NHL_TEAMS
        if sport_u == "NBA":
            from metadata.nba_teams import NBA_TEAMS
            return NBA_TEAMS
    except Exception:
        pass
    return {}


def _full_team_name(sport: str, token: Any) -> str:
    raw = str(token or "").strip()
    if not raw:
        return raw
    teams = _load_team_meta(sport)
    alias = ABBR_ALIASES.get(sport.upper(), {})
    key = alias.get(raw.upper(), raw.upper())
    if key in teams:
        return teams[key].get("name") or raw
    if raw in teams:
        return teams[raw].get("name") or raw
    for abbr, info in teams.items():
        name = info.get("name") or ""
        if name.lower() == raw.lower():
            return name
    return raw


def _events_from_payload(payload: dict) -> List[dict]:
    content = payload.get("content", payload) if isinstance(payload, dict) else {}
    sb = content.get("sbData") or content.get("scoreboard") or content
    if isinstance(sb, dict):
        events = sb.get("events") or []
        if isinstance(events, list):
            return events
    return []


def parse_espn_events(sport: str, payload: dict) -> List[CompletedGame]:
    games: List[CompletedGame] = []
    for ev in _events_from_payload(payload):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed")) or str(status.get("state", "")).lower() == "post"
        if not completed:
            continue
        home = away = None
        hs = as_ = None
        for team in comp.get("competitors") or []:
            name = (team.get("team") or {}).get("displayName") or team.get("displayName")
            try:
                score = float(team.get("score"))
            except (TypeError, ValueError):
                score = None
            side = str(team.get("homeAway", "")).lower()
            if side == "home":
                home, hs = name, score
            elif side == "away":
                away, as_ = name, score
        if not home or not away or hs is None or as_ is None:
            continue
        week = None
        week_blob = ev.get("week") or (payload.get("content") or {}).get("sbData", {}).get("week")
        if isinstance(week_blob, dict):
            try:
                week = int(week_blob.get("number"))
            except (TypeError, ValueError):
                week = None
        elif week_blob is not None:
            try:
                week = int(week_blob)
            except (TypeError, ValueError):
                week = None
        games.append(
            CompletedGame(
                sport=sport.upper(),
                away=str(away),
                home=str(home),
                away_score=as_,
                home_score=hs,
                week=week,
                date=(ev.get("date") or "")[:10] or None,
                status=str(status.get("name") or "final"),
            )
        )
    return games


def fetch_nfl_week(year: int, week: int) -> List[CompletedGame]:
    """Compatibility wrapper: nflverse season file filtered to one week."""
    return [
        g for g in fetch_nflverse_season(year)
        if g.week == int(week)
    ]


def fetch_daily(sport: str, day: date) -> List[CompletedGame]:
    sport_u = sport.upper()
    if sport_u == "NHL":
        return fetch_nhl_official_date(day)
    if sport_u == "NBA":
        return fetch_nba_scores([day])
    return []


def parse_nflverse_rows(rows: Iterable[dict], season: int) -> List[CompletedGame]:
    games = []
    season_s = str(season)
    for row in rows:
        if str(row.get("season")) != season_s:
            continue
        game_type = str(row.get("game_type") or "REG").upper()
        if game_type not in {"REG", "WC", "DIV", "CON", "SB"}:
            continue
        away_s, home_s = row.get("away_score"), row.get("home_score")
        if away_s in (None, "", "NA") or home_s in (None, "", "NA"):
            continue
        try:
            away_score = float(away_s)
            home_score = float(home_s)
        except (TypeError, ValueError):
            continue
        week = None
        try:
            week = int(float(row.get("week")))
        except (TypeError, ValueError):
            week = None
        games.append(
            CompletedGame(
                sport="NFL",
                away=_full_team_name("NFL", row.get("away_team")),
                home=_full_team_name("NFL", row.get("home_team")),
                away_score=away_score,
                home_score=home_score,
                week=week,
                date=str(row.get("gameday") or "") or None,
                status="final",
            )
        )
    return games


def fetch_nflverse_season(season: int) -> List[CompletedGame]:
    text = _http_text(NFLVERSE_SCHEDULES_CSV)
    rows = csv.DictReader(io.StringIO(text))
    return parse_nflverse_rows(rows, season)


def _nhl_team_full(blob: dict) -> str:
    abbrev = blob.get("abbrev") or blob.get("teamAbbrev") or ""
    name = _full_team_name("NHL", abbrev) if abbrev else ""
    if name and name.upper() != str(abbrev).upper():
        return name
    place = (blob.get("placeName") or {}).get("default") if isinstance(blob.get("placeName"), dict) else None
    common = (blob.get("commonName") or blob.get("name") or {})
    if isinstance(common, dict):
        common = common.get("default")
    if place and common:
        return f"{place} {common}"
    return name or str(common or abbrev)


def parse_nhl_score_payload(payload: dict) -> List[CompletedGame]:
    games = []
    for g in payload.get("games") or []:
        game_type = g.get("gameType")
        # 1 preseason, 2 regular, 3 playoffs
        if game_type not in (2, 3, "2", "3"):
            continue
        state = str(g.get("gameState") or "").upper()
        if state not in {"FINAL", "OFF"}:
            continue
        away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}
        try:
            as_ = float(away.get("score"))
            hs = float(home.get("score"))
        except (TypeError, ValueError):
            continue
        games.append(
            CompletedGame(
                sport="NHL",
                away=_nhl_team_full(away),
                home=_nhl_team_full(home),
                away_score=as_,
                home_score=hs,
                date=str(g.get("gameDate") or "")[:10] or None,
                status=state.lower(),
            )
        )
    return games


def fetch_nhl_official_date(day: date) -> List[CompletedGame]:
    payload = _http_json(f"{NHL_WEB_API}/score/{day.isoformat()}")
    return parse_nhl_score_payload(payload)


def fetch_nhl_official_dates(days: Iterable[date]) -> List[CompletedGame]:
    out: List[CompletedGame] = []
    for day in days:
        try:
            out.extend(fetch_nhl_official_date(day))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(0.08)
    return out


def _nba_side_name(blob: dict) -> str:
    tricode = str(blob.get("teamTricode") or blob.get("teamAbbrev") or "").strip()
    named = _full_team_name("NBA", tricode) if tricode else ""
    if named and named.upper() != tricode.upper():
        return named
    city = str(blob.get("teamCity") or "").strip()
    team = str(blob.get("teamName") or "").strip()
    if city and team:
        return f"{city} {team}"
    return named or tricode


def _nba_game_is_final(game: dict) -> bool:
    status = game.get("gameStatus")
    try:
        if int(status) == 3:
            return True
    except (TypeError, ValueError):
        pass
    return "final" in str(game.get("gameStatusText") or "").lower()


def _iso_day(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return None


def parse_nba_cdn_scoreboard(payload: dict) -> List[CompletedGame]:
    """Finals from today's NBA CDN scoreboard. This feed is not a date archive."""
    board = payload.get("scoreboard") if isinstance(payload, dict) else None
    if not isinstance(board, dict):
        board = {}
    board_date = _iso_day(board.get("gameDate"))
    games: List[CompletedGame] = []
    for game in board.get("games") or []:
        if not isinstance(game, dict) or not _nba_game_is_final(game):
            continue
        away = game.get("awayTeam") or {}
        home = game.get("homeTeam") or {}
        try:
            away_score = float(away.get("score"))
            home_score = float(home.get("score"))
        except (TypeError, ValueError):
            continue
        game_date = (
            _iso_day(game.get("gameDate"))
            or _iso_day(game.get("gameTimeUTC"))
            or _iso_day(game.get("gameEt"))
            or board_date
        )
        games.append(
            CompletedGame(
                sport="NBA",
                away=_nba_side_name(away),
                home=_nba_side_name(home),
                away_score=away_score,
                home_score=home_score,
                date=game_date,
                status="final",
            )
        )
    return games


def fetch_nba_cdn_today() -> List[CompletedGame]:
    payload = _http_json(NBA_CDN_SCOREBOARD, headers=NBA_CDN_HEADERS)
    return parse_nba_cdn_scoreboard(payload)


def fetch_espn_nba_dates(days: List[date]) -> List[CompletedGame]:
    """ESPN public scoreboard, one date at a time. A bad date is skipped."""
    out: List[CompletedGame] = []
    for day in days:
        url = ESPN_NBA_SCOREBOARD + "?" + urllib.parse.urlencode(
            {"dates": day.strftime("%Y%m%d")}
        )
        try:
            payload = _http_json(url)
            out.extend(parse_espn_events("NBA", payload))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
            OSError,
        ):
            pass
        time.sleep(0.1)
    return out


def _nba_keep_cdn_game(game: CompletedGame, days: List[date], today: date) -> bool:
    raw = _iso_day(game.date)
    if raw is None:
        return False
    try:
        played = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return False
    allowed = set(days)
    allowed.add(today)
    allowed.add(today - timedelta(days=1))
    return played in allowed


def _nba_dedupe_key(game: CompletedGame) -> tuple:
    return ((game.date or "")[:10], _norm_name(game.away), _norm_name(game.home))


def fetch_nba_scores(days: List[date]) -> List[CompletedGame]:
    """CDN today, then ESPN for pending dates. No keys. Empty feeds return []."""
    today = datetime.now(timezone.utc).date()
    cdn_games: List[CompletedGame] = []
    try:
        cdn_games = [
            game
            for game in fetch_nba_cdn_today()
            if _nba_keep_cdn_game(game, days, today)
        ]
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ):
        cdn_games = []

    espn_games: List[CompletedGame] = []
    if days:
        try:
            espn_games = fetch_espn_nba_dates(list(days))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
            OSError,
        ):
            espn_games = []

    seen = set()
    out: List[CompletedGame] = []
    espn_contributed = False
    for game in list(cdn_games) + list(espn_games):
        key = _nba_dedupe_key(game)
        if key in seen:
            continue
        seen.add(key)
        if game not in cdn_games:
            espn_contributed = True
        out.append(game)
    if cdn_games and espn_contributed:
        fetch_nba_scores.last_source = "nba-cdn+espn"
    elif espn_contributed:
        fetch_nba_scores.last_source = "espn"
    elif cdn_games:
        fetch_nba_scores.last_source = "nba-cdn"
    else:
        fetch_nba_scores.last_source = "nba-cdn+espn"
    return out


fetch_nba_scores.last_source = "nba-cdn+espn"


def detect_schema(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    def first(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    if "VisTm" in df.columns and "HomeTm" in df.columns:
        pts = [c for c in cols if str(c).lower().startswith("pts")]
        return {
            "away": "VisTm",
            "home": "HomeTm",
            "away_pts": pts[0] if pts else "Pts",
            "home_pts": pts[1] if len(pts) > 1 else "Pts.1",
            "week": first("Week"),
            "date": first("Date", "Unnamed: 2"),
        }
    if "Visitor/Neutral" in df.columns and "Home/Neutral" in df.columns:
        pts = [c for c in cols if str(c) == "PTS" or str(c).startswith("PTS")]
        return {
            "away": "Visitor/Neutral",
            "home": "Home/Neutral",
            "away_pts": pts[0] if pts else "PTS",
            "home_pts": pts[1] if len(pts) > 1 else "PTS.1",
            "week": None,
            "date": first("Date"),
        }
    if "Visitor" in df.columns and "Home" in df.columns:
        gs = [c for c in cols if c == "G" or str(c).startswith("G")]
        return {
            "away": "Visitor",
            "home": "Home",
            "away_pts": gs[0] if gs else "G",
            "home_pts": gs[1] if len(gs) > 1 else "G.1",
            "week": None,
            "date": first("Date"),
        }
    raise ValueError(f"Unrecognized slate schema: {cols}")


def _row_has_scores(row, away_pts: str, home_pts: str) -> bool:
    try:
        a = row[away_pts]
        h = row[home_pts]
    except Exception:
        return False
    if pd.isna(a) or pd.isna(h):
        return False
    try:
        float(a)
        float(h)
        return True
    except (TypeError, ValueError):
        return False


def _parse_row_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.date()
    text = str(value).strip()
    for fmt in ("%B %d", "%b %d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt != "%Y-%m-%d":
                return None
            return dt.date()
        except ValueError:
            continue
    return None


def _find_row(df: pd.DataFrame, schema: dict, game: CompletedGame) -> Optional[int]:
    away_n = _norm_name(game.away)
    home_n = _norm_name(game.home)
    away_col, home_col = schema["away"], schema["home"]
    candidates = []
    for idx, row in df.iterrows():
        if _norm_name(row[away_col]) != away_n or _norm_name(row[home_col]) != home_n:
            continue
        score = 0
        if schema.get("week") and game.week is not None:
            try:
                if int(float(row[schema["week"]])) == int(game.week):
                    score += 5
                else:
                    continue
            except (TypeError, ValueError):
                pass
        if schema.get("date") and game.date:
            row_day = _parse_row_date(row[schema["date"]])
            if row_day is not None:
                try:
                    gday = datetime.strptime(game.date[:10], "%Y-%m-%d").date()
                    delta = abs((row_day - gday).days)
                    if delta == 0:
                        score += 4
                    elif delta == 1:
                        score += 1
                except ValueError:
                    pass
        if not _row_has_scores(row, schema["away_pts"], schema["home_pts"]):
            score += 2
        candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return int(candidates[0][1])


def apply_completed_games(
    df: pd.DataFrame,
    games: Iterable[CompletedGame],
    overwrite: bool = False,
) -> tuple[pd.DataFrame, List[dict], List[dict], int]:
    schema = detect_schema(df)
    out = df.copy()
    applied = []
    unmatched = []
    already = 0
    used_idx = set()
    for game in games:
        idx = _find_row(out, schema, game)
        if idx is None or idx in used_idx:
            unmatched.append(asdict(game))
            continue
        row = out.loc[idx]
        if _row_has_scores(row, schema["away_pts"], schema["home_pts"]) and not overwrite:
            already += 1
            used_idx.add(idx)
            continue
        out.at[idx, schema["away_pts"]] = game.away_score
        out.at[idx, schema["home_pts"]] = game.home_score
        used_idx.add(idx)
        applied.append(
            {
                "away": game.away,
                "home": game.home,
                "away_score": game.away_score,
                "home_score": game.home_score,
                "week": game.week,
                "date": game.date,
            }
        )
    return out, applied, unmatched, already


def _lock_from_path(sport: str, path: Path) -> dict:
    raw = pd.read_csv(path)
    canon = normalize_schedule(raw, sport=sport)
    canon = filter_regular_season(canon, sport=sport)
    return describe_schedule_lock(canon)


def _unique_csv_dates(df: pd.DataFrame, schema: dict) -> List[date]:
    col = schema.get("date")
    if not col or col not in df.columns:
        return []
    days = []
    for value in df[col].tolist():
        day = _parse_row_date(value)
        if day is not None:
            days.append(day)
    return sorted(set(days))


def _pending_dates(df: pd.DataFrame, schema: dict, today: date) -> List[date]:
    days = _unique_csv_dates(df, schema)
    pending = []
    for day in days:
        if day > today:
            continue
        mask_day = df[schema["date"]].map(lambda v: _parse_row_date(v) == day)
        subset = df.loc[mask_day]
        needs = True
        if not subset.empty:
            needs = any(
                not _row_has_scores(row, schema["away_pts"], schema["home_pts"])
                for _, row in subset.iterrows()
            )
        if needs or (today - day).days <= 2:
            pending.append(day)
    return pending[-40:]


def fetch_completed_for_csv(sport: str, season: str, df: pd.DataFrame) -> List[CompletedGame]:
    sport_u = sport.upper()
    schema = detect_schema(df)
    today = datetime.now(timezone.utc).date()

    if sport_u == "NFL":
        year = int(str(season)[:4])
        return fetch_nflverse_season(year)

    pending_days = _pending_dates(df, schema, today)
    if sport_u == "NHL":
        return fetch_nhl_official_dates(pending_days)

    if sport_u == "NBA":
        return fetch_nba_scores(pending_days)

    return []


def update_season_scores(
    sport: str,
    season: str,
    root: Path = ROOT,
    overwrite: bool = False,
    dry_run: bool = False,
    games: Optional[List[CompletedGame]] = None,
) -> SlateReport:
    path = season_csv_path(sport, season, root=root)
    report = SlateReport(
        sport=sport.upper(),
        season=str(season),
        path=str(path),
        fetched_at=_now_iso(),
        source=DEFAULT_SOURCE.get(sport.upper(), ""),
    )
    if not path.exists():
        report.error = f"Schedule file not found: {path}"
        return report

    df = pd.read_csv(path)
    try:
        fetched = list(games) if games is not None else fetch_completed_for_csv(sport, season, df)
    except Exception as exc:
        report.error = str(exc)
        fetched = []
    if sport.upper() == "NBA" and games is None:
        report.source = getattr(fetch_nba_scores, "last_source", None) or report.source
    report.fetched = len(fetched)
    updated_df, applied, unmatched, already = apply_completed_games(
        df, fetched, overwrite=overwrite
    )
    report.updated = len(applied)
    report.already_locked = already
    report.unmatched = len(unmatched)
    report.applied = applied
    report.unmatched_games = unmatched

    if not dry_run and applied:
        updated_df.to_csv(path, index=False)

    lock = _lock_from_path(sport, path)
    report.n_games = lock["n_games"]
    report.n_locked = lock["n_locked"]
    report.status = lock["status"]
    _write_status(report)
    return report


def _write_status(report: SlateReport) -> None:
    payload = {}
    if SLATE_STATUS_PATH.exists():
        try:
            payload = json.loads(SLATE_STATUS_PATH.read_text())
        except Exception:
            payload = {}
    key = f"{report.sport}:{report.season}"
    payload[key] = {
        "fetched_at": report.fetched_at,
        "n_locked": report.n_locked,
        "n_games": report.n_games,
        "status": report.status,
        "updated": report.updated,
        "source": report.source,
        "error": report.error,
    }
    SLATE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SLATE_STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def read_slate_status(sport: str, season: str) -> Optional[dict]:
    if not SLATE_STATUS_PATH.exists():
        return None
    try:
        payload = json.loads(SLATE_STATUS_PATH.read_text())
    except Exception:
        return None
    return payload.get(f"{sport.upper()}:{season}")


def default_target_season(sport: str) -> Optional[str]:
    try:
        from app.services.initial_ratings_service import get_simulatable_seasons
    except ImportError:
        try:
            from services.initial_ratings_service import get_simulatable_seasons
        except ImportError:
            get_simulatable_seasons = None
    if get_simulatable_seasons:
        seasons = get_simulatable_seasons(sport)
        if seasons:
            return seasons[-1]
    sport_l = sport.lower()
    files = sorted((ROOT / "data" / sport_l).glob(f"{sport_l}_*.csv"))
    if files:
        return files[-1].stem.split("_")[-1]
    return None


def season_overrides_for_refresh(
    active_sport: str,
    active_season: Optional[str],
    remembered: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Season keys for an all-sports refresh.

    The open sport uses the sidebar season. Each other sport uses its own
    remembered season when one is stored. Sports with no entry are omitted
    so ``update_all_current`` can apply that sport's target season.
    """
    active = str(active_sport or "").upper()
    overrides: Dict[str, str] = {}
    if active in ("NFL", "NHL", "NBA") and active_season:
        overrides[active] = str(active_season)
    if remembered:
        for name, value in remembered.items():
            key = str(name).upper()
            if key not in ("NFL", "NHL", "NBA") or key == active or not value:
                continue
            overrides.setdefault(key, str(value))
    return overrides


def update_all_current(
    root: Path = ROOT,
    dry_run: bool = False,
    seasons: Optional[Dict[str, Optional[str]]] = None,
) -> List[SlateReport]:
    """Refresh NFL, NHL, and NBA.

    ``seasons`` overrides the season key per sport. Omitted sports use
    ``default_target_season``. A sport with no season is reported with an
    error and is not fetched. One failure does not stop the others.
    Existing scores stay put; ``overwrite`` is not enabled here.
    """
    overrides: Dict[str, str] = {}
    if seasons:
        for key, value in seasons.items():
            if value:
                overrides[str(key).upper()] = str(value)

    reports: List[SlateReport] = []
    for sport in ("NFL", "NHL", "NBA"):
        season = overrides.get(sport) or default_target_season(sport)
        if not season:
            reports.append(
                SlateReport(
                    sport=sport,
                    season="",
                    path="",
                    fetched_at=_now_iso(),
                    source=DEFAULT_SOURCE.get(sport, ""),
                    error=f"No season file for {sport}",
                )
            )
            continue
        try:
            reports.append(
                update_season_scores(sport, season, root=root, dry_run=dry_run)
            )
        except Exception as exc:
            reports.append(
                SlateReport(
                    sport=sport,
                    season=str(season),
                    path=str(season_csv_path(sport, season, root=root)),
                    fetched_at=_now_iso(),
                    source=DEFAULT_SOURCE.get(sport, ""),
                    error=str(exc),
                )
            )
    return reports


def upcoming_and_locked_tables(sport: str, season: str, root: Path = ROOT, limit: int = 12):
    path = season_csv_path(sport, season, root=root)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_csv(path)
    schema = detect_schema(raw)
    locked_rows = []
    open_rows = []
    for _, row in raw.iterrows():
        item = {
            "Away": row[schema["away"]],
            "Home": row[schema["home"]],
        }
        if schema.get("week"):
            item["Week"] = row[schema["week"]]
        if schema.get("date"):
            item["Date"] = row[schema["date"]]
        if _row_has_scores(row, schema["away_pts"], schema["home_pts"]):
            item["Score"] = f"{int(float(row[schema['away_pts']]))}-{int(float(row[schema['home_pts']]))}"
            locked_rows.append(item)
        else:
            open_rows.append(item)
    locked = pd.DataFrame(locked_rows)
    upcoming = pd.DataFrame(open_rows)
    if not locked.empty:
        locked = locked.tail(limit)
    if not upcoming.empty:
        upcoming = upcoming.head(limit)
    return locked, upcoming


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh live scores into season CSVs")
    parser.add_argument("--sport", default="all", help="NFL, NHL, NBA, or all")
    parser.add_argument("--season", default=None, help="Season key matching the CSV suffix")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sports = ["NFL", "NHL", "NBA"] if args.sport.lower() == "all" else [args.sport.upper()]
    code = 0
    for sport in sports:
        season = args.season or default_target_season(sport)
        if not season:
            print(f"{sport}: no season file found")
            code = 1
            continue
        try:
            report = update_season_scores(
                sport, season, overwrite=args.overwrite, dry_run=args.dry_run
            )
        except Exception as exc:
            print(f"{sport}: {exc}")
            code = 1
            continue
        print(report.lock_line())
        print(
            f"  fetched={report.fetched} updated={report.updated} "
            f"already={report.already_locked} unmatched={report.unmatched}"
        )
        if report.error:
            print(f"  error: {report.error}")
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Neutral-site filter for Elo fitting.

Backtests, parameter search, historical parameter scoring, and Elo
snapshot updates skip games that were not played in either club's
building. Home-field Elo is not fit on those rows. This module does
not change K, home-field, or any other model parameter.

The Monte Carlo slate still contains the games. Only the walks that
update Elo from completed results drop them.

Detection uses an explicit flag or venue list already in the data.
Kickoff time is not used.

NFL
    The Pro Football Reference column between ``Winner/tie`` and
    ``Loser/tie`` is ``N`` for a neutral site. In these CSVs that mark
    is the Super Bowl. A Week label of Super Bowl or Hall of Fame is
    the same exclusion (Hall of Fame is not in the current files).
    International Series rows are stored as ordinary home/away games,
    with no stadium column, so they are listed in
    ``NFL_INTERNATIONAL_GAMES`` (date, both clubs, venue).
    ``data/nfl_games.csv`` has no flag. Those rows match the season
    file on season, week, and the two clubs. Playoff week numbers in
    that file are Wild Card 19, Divisional 20, Conference 21, Super
    Bowl 22.

NHL
    Hockey-Reference ``Notes`` that start with ``at `` name the
    building. That covers the Global Series and outdoor games (Stadium
    Series, Winter Classic, Heritage Classic). Any other note stays.

NBA
    A ``Notes`` token ``Neutral``, a Basketball-Reference ``at …`` city
    note, or an arena in ``NBA_NEUTRAL_ARENAS``. The arena list is
    Global Games buildings plus T-Mobile Arena, where the NBA Cup
    knockout is played. NBA Cup group games stay; they are in the home
    arena. Play-in games stay. Spurs home games at the Alamodome or
    Moody Center stay; those buildings are not on the list.
    Processed schedule files drop ``Arena``. The loader copies it back
    from ``data/nba/raw`` when the processed file has none.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Buildings that are not an NBA club's home court.
# T-Mobile Arena hosts the NBA Cup semis and final in Las Vegas.
NBA_NEUTRAL_ARENAS = frozenset(
    {
        "Mexico City Arena",
        "Accor Arena",
        "Uber Arena",
        "The O2 Arena",
        "O2 Arena",
        "T-Mobile Arena",
    }
)

# Week text on the PFR schedule. Hall of Fame is included so a future
# file that carries the preseason game is excluded without a new rule.
_NEUTRAL_WEEK_LABELS = frozenset(
    {
        "superbowl",
        "super bowl",
        "hall of fame",
        "halloffame",
        "hall of fame game",
    }
)

# nfl_games.csv numbers the postseason this way. The season CSV uses labels.
_PLAYOFF_WEEK_NUM = {
    "wildcard": 19,
    "wild card": 19,
    "division": 20,
    "divisional": 20,
    "confchamp": 21,
    "conference": 21,
    "superbowl": 22,
    "super bowl": 22,
}

# (date, team, team, venue). Team names match the season CSVs.
# These rows are neutral, and the CSVs do not mark them N.
NFL_INTERNATIONAL_GAMES: Tuple[Tuple[str, str, str, str], ...] = (
    ("2022-10-02", "Minnesota Vikings", "New Orleans Saints", "Tottenham Hotspur Stadium, London"),
    ("2022-10-09", "New York Giants", "Green Bay Packers", "Tottenham Hotspur Stadium, London"),
    ("2022-10-30", "Denver Broncos", "Jacksonville Jaguars", "Wembley Stadium, London"),
    ("2022-11-13", "Tampa Bay Buccaneers", "Seattle Seahawks", "Allianz Arena, Munich"),
    ("2022-11-21", "San Francisco 49ers", "Arizona Cardinals", "Estadio Azteca, Mexico City"),
    ("2023-10-01", "Jacksonville Jaguars", "Atlanta Falcons", "Wembley Stadium, London"),
    ("2023-10-08", "Jacksonville Jaguars", "Buffalo Bills", "Tottenham Hotspur Stadium, London"),
    ("2023-10-15", "Baltimore Ravens", "Tennessee Titans", "Tottenham Hotspur Stadium, London"),
    ("2023-11-05", "Kansas City Chiefs", "Miami Dolphins", "Deutsche Bank Park, Frankfurt"),
    ("2023-11-12", "Indianapolis Colts", "New England Patriots", "Deutsche Bank Park, Frankfurt"),
    ("2024-09-06", "Philadelphia Eagles", "Green Bay Packers", "Arena Corinthians, Sao Paulo"),
    ("2024-10-06", "Minnesota Vikings", "New York Jets", "Tottenham Hotspur Stadium, London"),
    ("2024-10-13", "Chicago Bears", "Jacksonville Jaguars", "Tottenham Hotspur Stadium, London"),
    ("2024-10-20", "Jacksonville Jaguars", "New England Patriots", "Wembley Stadium, London"),
    ("2024-11-10", "Carolina Panthers", "New York Giants", "Allianz Arena, Munich"),
    ("2025-09-05", "Los Angeles Chargers", "Kansas City Chiefs", "Arena Corinthians, Sao Paulo"),
    ("2025-09-28", "Pittsburgh Steelers", "Minnesota Vikings", "Croke Park, Dublin"),
    ("2025-10-05", "Minnesota Vikings", "Cleveland Browns", "Tottenham Hotspur Stadium, London"),
    ("2025-10-12", "Denver Broncos", "New York Jets", "Tottenham Hotspur Stadium, London"),
    ("2025-10-19", "Los Angeles Rams", "Jacksonville Jaguars", "Wembley Stadium, London"),
    ("2025-11-09", "Indianapolis Colts", "Atlanta Falcons", "Olympiastadion, Berlin"),
    ("2025-11-16", "Miami Dolphins", "Washington Commanders", "Santiago Bernabeu, Madrid"),
    ("2026-09-10", "San Francisco 49ers", "Los Angeles Rams", "Melbourne Cricket Ground"),
    ("2026-09-27", "Baltimore Ravens", "Dallas Cowboys", "Maracana Stadium, Rio de Janeiro"),
    ("2026-10-04", "Indianapolis Colts", "Washington Commanders", "Tottenham Hotspur Stadium, London"),
    ("2026-10-11", "Philadelphia Eagles", "Jacksonville Jaguars", "Tottenham Hotspur Stadium, London"),
    ("2026-10-18", "Houston Texans", "Jacksonville Jaguars", "Wembley Stadium, London"),
    ("2026-10-25", "Pittsburgh Steelers", "New Orleans Saints", "Stade de France, Paris"),
    ("2026-11-08", "Cincinnati Bengals", "Atlanta Falcons", "Santiago Bernabeu, Madrid"),
    ("2026-11-15", "New England Patriots", "Detroit Lions", "Allianz Arena, Munich"),
    ("2026-11-22", "Minnesota Vikings", "San Francisco 49ers", "Estadio Banorte, Mexico City"),
)

_NEUTRAL_TOKEN = re.compile(r"(?i)(?<![a-z])neutral(?![a-z])")
_AT_VENUE = re.compile(r"(?i)^\s*at\s+\S")
_MONTH_DAY = re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2}$")
_TEAM_PAIRS = (
    ("Winner/tie", "Loser/tie"),
    ("VisTm", "HomeTm"),
    ("Visitor/Neutral", "Home/Neutral"),
    ("Visitor", "Home"),
    ("away_team", "home_team"),
    ("Away", "Home"),
)


def arena_is_neutral(arena) -> bool:
    """True when ``arena`` is on ``NBA_NEUTRAL_ARENAS``."""
    text = "" if arena is None or (isinstance(arena, float) and pd.isna(arena)) else str(arena).strip()
    if not text or text.lower() == "nan":
        return False
    folded = text.casefold()
    return any(name.casefold() == folded for name in NBA_NEUTRAL_ARENAS)


def notes_mark_neutral(notes) -> bool:
    """True for an explicit Neutral token or a leading ``at <venue>`` note.

    NHL uses the ``at`` form for every non-arena building in these files.
    NBA uses it for Global Games city lines (``at Mexico City Mexico``).
    Other notes, including NBA Cup and Play-In, do not match.
    """
    if notes is None or (isinstance(notes, float) and pd.isna(notes)):
        return False
    text = str(notes).strip()
    if not text or text.lower() == "nan":
        return False
    if _NEUTRAL_TOKEN.search(text):
        return True
    return _AT_VENUE.match(text) is not None


def neutral_mask(
    df: pd.DataFrame,
    sport: str,
    season: Optional[str] = None,
) -> pd.Series:
    """Boolean series aligned to ``df``. True means the row is neutral-site."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=bool)
    sport_u = (sport or "").upper()
    mask = pd.Series(False, index=df.index)
    if "neutral_site" in df.columns:
        mask = mask | df["neutral_site"].map(_as_bool)

    if sport_u == "NFL":
        mask = mask | _nfl_marker_mask(df)
        mask = mask | _nfl_week_label_mask(df)
        mask = mask | _nfl_international_mask(df, season)
        # Season CSVs carry the flag forward as neutral_site. The combined
        # nfl_games.csv does not, so match it back to those season files.
        if (
            "home_team" in df.columns
            and "Winner/tie" not in df.columns
            and "VisTm" not in df.columns
            and "neutral_site" not in df.columns
        ):
            mask = mask | _nfl_games_join_mask(df)
    elif sport_u in ("NHL", "NBA"):
        mask = mask | _notes_mask(df)
        if sport_u == "NBA":
            mask = mask | _arena_mask(df)
    return mask.astype(bool)


def exclude_neutral_sites(
    df: pd.DataFrame,
    sport: str,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Drop neutral-site rows. Other rows and columns are unchanged."""
    if df is None or len(df) == 0:
        return df
    mask = neutral_mask(df, sport, season=season)
    if len(mask) != len(df):
        mask = pd.Series(mask.to_numpy(), index=df.index)
    kept = df.loc[~mask.fillna(False)].copy()
    return kept.reset_index(drop=True)


def attach_nba_arena(df: pd.DataFrame, season: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Copy ``Arena`` from ``data/nba/raw`` onto a processed NBA schedule.

    Processed files keep scores (including live-slate updates) but drop the
    arena. The raw export still has it. Rows that already name an arena
    are left alone. Scores are not copied.
    """
    if df is None or len(df) == 0 or "Arena" in df.columns:
        return df
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    raw_path = root / "nba" / "raw" / f"nba_{season}.csv"
    if not raw_path.is_file():
        return df
    raw = pd.read_csv(raw_path, dtype=str)
    if "Arena" not in raw.columns:
        return df
    raw_keys = _nba_game_key(raw)
    arena_by_key = (
        pd.Series(raw["Arena"].values, index=raw_keys)
        .groupby(level=0)
        .first()
    )
    out = df.copy()
    out["Arena"] = _nba_game_key(out).map(arena_by_key)
    return out


def _as_bool(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "t", "yes", "y")
    return bool(value)


def _notes_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for name in df.columns:
        if str(name).strip().lower() in ("notes", "note"):
            return df[name]
    return None


def _notes_mask(df: pd.DataFrame) -> pd.Series:
    notes = _notes_series(df)
    if notes is None:
        return pd.Series(False, index=df.index)
    return notes.map(notes_mark_neutral).astype(bool)


def _arena_mask(df: pd.DataFrame) -> pd.Series:
    if "Arena" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Arena"].map(arena_is_neutral).astype(bool)


def _nfl_location_column(df: pd.DataFrame) -> Optional[str]:
    if "Winner/tie" not in df.columns:
        return None
    columns = list(df.columns)
    idx = columns.index("Winner/tie")
    if idx + 1 < len(columns) and columns[idx + 1] != "Loser/tie":
        return columns[idx + 1]
    return None


def _nfl_marker_mask(df: pd.DataFrame) -> pd.Series:
    column = _nfl_location_column(df)
    if column is None:
        return pd.Series(False, index=df.index)
    text = df[column].map(lambda v: "" if pd.isna(v) else str(v).strip().upper())
    return text.eq("N")


def _nfl_week_label_mask(df: pd.DataFrame) -> pd.Series:
    if "Week" not in df.columns:
        return pd.Series(False, index=df.index)
    labels = df["Week"].map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
    return labels.isin(_NEUTRAL_WEEK_LABELS)


def _season_year(season: Optional[str]) -> Optional[int]:
    if season is None:
        return None
    match = re.search(r"\d{4}", str(season))
    if not match:
        return None
    return int(match.group(0))


def _parse_game_date(value, season_year: Optional[int]):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text[:10], errors="coerce")
    if _MONTH_DAY.match(text):
        if season_year is None:
            return pd.NaT
        for fmt in ("%B %d", "%b %d"):
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            year = season_year if parsed.month >= 7 else season_year + 1
            return pd.Timestamp(year=year, month=parsed.month, day=parsed.day)
        return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def _date_series(df: pd.DataFrame, season: Optional[str]) -> pd.Series:
    column = None
    for name in df.columns:
        if str(name).strip().lower() == "date":
            column = name
            break
    if column is None:
        for name in df.columns:
            sample = df[name].dropna().astype(str).str.strip().head(12)
            if len(sample) == 0:
                continue
            if sample.map(lambda v: _MONTH_DAY.match(v) is not None).mean() >= 0.5:
                column = name
                break
    year = _season_year(season)
    if column is None:
        return pd.Series(pd.NaT, index=df.index)
    return df[column].map(lambda v: _parse_game_date(v, year))


def _team_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    for left, right in _TEAM_PAIRS:
        if left in df.columns and right in df.columns:
            return left, right
    return None, None


@lru_cache(maxsize=1)
def _nfl_abbr_map() -> dict:
    try:
        from metadata import load_teams
    except ImportError:
        from app.metadata import load_teams
    mapping = {}
    for abbr, meta in load_teams("NFL").items():
        mapping[abbr.upper()] = abbr
        name = str(meta.get("name", "")).strip().lower()
        if name:
            mapping[name] = abbr
    return mapping


def _to_abbr(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    mapping = _nfl_abbr_map()
    return mapping.get(text) or mapping.get(text.upper()) or mapping.get(text.lower()) or text.upper()


def _pair_frame(df: pd.DataFrame) -> pd.Series:
    """frozenset of abbreviations per row, or None when a club is missing."""
    left_name, right_name = _team_columns(df)
    if left_name is None:
        return pd.Series([None] * len(df), index=df.index)

    def one(left, right):
        a = _to_abbr(left)
        b = _to_abbr(right)
        if not a or not b:
            return None
        return frozenset((a, b))

    return pd.Series(
        [one(l, r) for l, r in zip(df[left_name], df[right_name])],
        index=df.index,
    )


@lru_cache(maxsize=1)
def _international_keys() -> FrozenSet[Tuple[str, FrozenSet[str]]]:
    keys = []
    for date, team_a, team_b, _venue in NFL_INTERNATIONAL_GAMES:
        keys.append((date, frozenset((_to_abbr(team_a), _to_abbr(team_b)))))
    return frozenset(keys)


def _nfl_international_mask(df: pd.DataFrame, season: Optional[str]) -> pd.Series:
    dates = _date_series(df, season)
    pairs = _pair_frame(df)
    keys = _international_keys()
    flags = []
    for day, pair in zip(dates, pairs):
        if pair is None or pd.isna(day):
            flags.append(False)
            continue
        flags.append((pd.Timestamp(day).strftime("%Y-%m-%d"), pair) in keys)
    return pd.Series(flags, index=df.index)


def _week_number(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        compact = text.lower().replace(" ", "")
        if compact in _PLAYOFF_WEEK_NUM:
            return _PLAYOFF_WEEK_NUM[compact]
        return _PLAYOFF_WEEK_NUM.get(text.lower())


def _nfl_file_stamp() -> str:
    folder = DATA_DIR / "nfl"
    if not folder.is_dir():
        return ""
    parts = []
    for path in sorted(folder.glob("nfl_*.csv")):
        parts.append(f"{path.name}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


@lru_cache(maxsize=4)
def _nfl_neutral_keys_cached(stamp: str) -> FrozenSet[Tuple[str, int, FrozenSet[str]]]:
    del stamp  # cache key only; files are read from DATA_DIR
    keys = set()
    folder = DATA_DIR / "nfl"
    if not folder.is_dir():
        return frozenset()
    for path in sorted(folder.glob("nfl_*.csv")):
        season = path.stem.split("_")[-1]
        frame = pd.read_csv(path)
        mask = neutral_mask(frame, "NFL", season=season)
        if mask.empty or not bool(mask.any()):
            continue
        weeks = frame["Week"] if "Week" in frame.columns else pd.Series(pd.NA, index=frame.index)
        pairs = _pair_frame(frame)
        for idx in frame.index[mask.to_numpy()]:
            week = _week_number(weeks.loc[idx])
            pair = pairs.loc[idx]
            if week is None or pair is None:
                continue
            keys.add((str(season), int(week), pair))
    return frozenset(keys)


def _nfl_neutral_keys() -> FrozenSet[Tuple[str, int, FrozenSet[str]]]:
    return _nfl_neutral_keys_cached(_nfl_file_stamp())


def _nfl_games_join_mask(df: pd.DataFrame) -> pd.Series:
    """Match canonical rows (season, week, clubs) to neutral games in the season CSVs."""
    keys = _nfl_neutral_keys()
    if "season" not in df.columns or "week" not in df.columns:
        return pd.Series(False, index=df.index)
    pairs = _pair_frame(df)
    flags = []
    for season, week, pair in zip(df["season"], df["week"], pairs):
        week_num = _week_number(week)
        if pair is None or week_num is None or pd.isna(season):
            flags.append(False)
            continue
        year = _season_year(season)
        season_key = str(year) if year is not None else str(season).strip()
        flags.append((season_key, week_num, pair) in keys)
    return pd.Series(flags, index=df.index)


def _nba_game_key(df: pd.DataFrame) -> pd.Series:
    date = df["Date"].fillna("").astype(str).str.strip() if "Date" in df.columns else ""
    away_col = "Visitor/Neutral" if "Visitor/Neutral" in df.columns else None
    home_col = "Home/Neutral" if "Home/Neutral" in df.columns else None
    if away_col is None or home_col is None:
        return pd.Series([""] * len(df), index=df.index)
    away = df[away_col].fillna("").astype(str).str.strip().str.lower()
    home = df[home_col].fillna("").astype(str).str.strip().str.lower()
    return date + "|" + away + "|" + home

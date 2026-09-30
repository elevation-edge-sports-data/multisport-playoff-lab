"""
simulate_season.py – Version 10

v10: Per-team games_played; shared Monte Carlo for standings + Elo evolution
v14.0: Lock games that already have scores; sample only the rest

Multiseason + NBA support:
  - Guards max_week filter when schedule has no "week" column
  - NBA entry in local SPORT_CONFIGS (wins-based like NFL)
  - Safe team Elo lookup (fills missing teams at 1500)
  - Compatible with per-season schedule files under data/{sport}/
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

try:
    from elo_lab.engine import compute_pregame, run_game
except ImportError:
    def compute_pregame(home_elo, away_elo, context, config):
        diff = home_elo - away_elo
        return {"p_home": 1.0 / (1.0 + 10 ** (-diff / 400.0))}

    def run_game(home_elo, away_elo, context, config):
        k = (config or {}).get("k", 20)
        actual = context.get("actual", 0)
        p_home = 1.0 / (1.0 + 10 ** (-(home_elo - away_elo) / 400.0))
        delta = k * (actual - p_home)
        return {"home_elo_post": home_elo + delta, "away_elo_post": away_elo - delta}


# ==================== SPORT CONFIG ====================
# Local defaults – schedule_path is a fallback only.
# Preferred path resolution lives in app/services/simulation_service.py
# (per-season files under data/{sport}/).
SPORT_CONFIGS = {
    "NFL": {
        "schedule_path": "data/nfl_games.csv",
        "max_week": 18,
        "outcome_type": "wins",
        "ot_rate": 0.0,
        "points_per_win": 1,
        "points_per_ot_loss": 0,
    },
    "NHL": {
        "schedule_path": "data/nhl_games.csv",
        "max_week": None,
        "outcome_type": "points",
        "ot_rate": 0.23,
        "points_per_win": 2,
        "points_per_ot_loss": 1,
    },
    "NBA": {
        "schedule_path": "data/nba_games.csv",
        "max_week": None,
        "outcome_type": "wins",
        "ot_rate": 0.0,
        "points_per_win": 1,
        "points_per_ot_loss": 0,
    },
}

# ---------------------------------------------------------------------------
# Schedule normalizer – map Basketball-Reference / legacy schemas
# to canonical columns: home_team, away_team, home_score, away_score
# ---------------------------------------------------------------------------

def _build_name_to_abbr(sport: str) -> dict:
    """Map full team name (and abbr) -> abbr using metadata when available."""
    mapping = {}
    teams = None
    for import_path in (
        "metadata",
        "app.metadata",
    ):
        try:
            mod = __import__(import_path, fromlist=["load_teams"])
            teams = mod.load_teams(sport)
            break
        except Exception:
            continue

    if teams:
        for abbr, info in teams.items():
            mapping[abbr.upper()] = abbr
            mapping[abbr] = abbr
            name = info.get("name", "")
            if name:
                mapping[name] = abbr
                mapping[name.upper()] = abbr
                mapping[name.replace(" ", "").upper()] = abbr
    # Always layer playoff full-name maps (covers renames e.g. Utah Mammoth)
    try:
        sport_u = sport.upper()
        if sport_u == "NHL":
            from elo_lab.playoffs.nhl.adapter import NHL_TEAM_META, NHL_FULL_NAME_TO_ABBR
            for abbr in NHL_TEAM_META:
                mapping[abbr] = abbr
                mapping[abbr.upper()] = abbr
            for name, abbr in NHL_FULL_NAME_TO_ABBR.items():
                mapping[name] = abbr
                mapping[name.upper()] = abbr
            for name in ("Utah Mammoth", "Utah Hockey Club", "Arizona Coyotes", "Phoenix Coyotes"):
                mapping[name] = "UTA"
                mapping[name.upper()] = "UTA"
            mapping["ARI"] = "UTA"
        elif sport_u == "NFL":
            from elo_lab.playoffs.nfl.adapter import NFL_TEAM_META
            for abbr in NFL_TEAM_META:
                mapping[abbr] = abbr
                mapping[abbr.upper()] = abbr
        elif sport_u == "NBA":
            from elo_lab.playoffs.nba.adapter import NBA_TEAM_META
            for abbr in NBA_TEAM_META:
                mapping[abbr] = abbr
                mapping[abbr.upper()] = abbr
    except Exception:
        pass
    return mapping


def normalize_schedule(df: pd.DataFrame, sport: str = "NFL") -> pd.DataFrame:
    """
    Convert various schedule schemas into canonical columns:
      home_team, away_team, home_score, away_score  (+ season, week if present)

    Supported input formats:
      - Legacy combined: home_team / away_team / home_score / away_score
      - NBA B-R: Visitor/Neutral, PTS, Home/Neutral, PTS
      - NHL B-R: Visitor, G, Home, G
      - NFL B-R: Winner/tie, Loser/tie, Pts, Pts  (home unknown – treat winner as home)
    """
    df = df.copy()
    cols = {c.strip(): c for c in df.columns}
    lower = {c.lower().strip(): c for c in df.columns}

    name_map = _build_name_to_abbr(sport)

    def to_abbr(val):
        s = str(val).strip()
        if s in name_map:
            return name_map[s]
        if s.upper() in name_map:
            return name_map[s.upper()]
        return s  # leave as-is if unknown

    # Already canonical
    if "home_team" in df.columns and "away_team" in df.columns:
        for col in ("home_team", "away_team"):
            df[col] = df[col].map(to_abbr)
        return df

    # NBA: Visitor/Neutral, PTS, Home/Neutral, PTS  (duplicate PTS names)
    if "Visitor/Neutral" in df.columns and "Home/Neutral" in df.columns:
        # pandas renames duplicate PTS to PTS and PTS.1
        pts_cols = [c for c in df.columns if c == "PTS" or c.startswith("PTS")]
        away_pts = pts_cols[0] if pts_cols else None
        home_pts = pts_cols[1] if len(pts_cols) > 1 else None
        out = pd.DataFrame()
        out["away_team"] = df["Visitor/Neutral"].map(to_abbr)
        out["home_team"] = df["Home/Neutral"].map(to_abbr)
        if away_pts:
            out["away_score"] = pd.to_numeric(df[away_pts], errors="coerce")
        if home_pts:
            out["home_score"] = pd.to_numeric(df[home_pts], errors="coerce")
        if "Date" in df.columns:
            out["date"] = df["Date"]
        return out

    # NHL B-R: Visitor, G, Home, G
    if "Visitor" in df.columns and "Home" in df.columns:
        g_cols = [c for c in df.columns if c == "G" or c.startswith("G")]
        away_g = g_cols[0] if g_cols else None
        home_g = g_cols[1] if len(g_cols) > 1 else None
        out = pd.DataFrame()
        out["away_team"] = df["Visitor"].map(to_abbr)
        out["home_team"] = df["Home"].map(to_abbr)
        if away_g:
            out["away_score"] = pd.to_numeric(df[away_g], errors="coerce")
        if home_g:
            out["home_score"] = pd.to_numeric(df[home_g], errors="coerce")
        if "Date" in df.columns:
            out["date"] = df["Date"]
        return out

    # NFL B-R completed: Winner/tie, Loser/tie, optional @ marker, Pts, Pts
    if "Winner/tie" in df.columns and "Loser/tie" in df.columns:
        pts_cols = [c for c in df.columns if str(c).lower().startswith("pts")]
        w_pts = pts_cols[0] if pts_cols else None
        l_pts = pts_cols[1] if len(pts_cols) > 1 else None
        # Middle column between Winner and Loser is "@" when the winner was away
        win_idx = list(df.columns).index("Winner/tie")
        marker_c = None
        if win_idx + 1 < len(df.columns) and df.columns[win_idx + 1] != "Loser/tie":
            marker_c = df.columns[win_idx + 1]

        homes, aways, home_scores, away_scores = [], [], [], []
        for _, row in df.iterrows():
            w = row["Winner/tie"]
            l = row["Loser/tie"]
            if pd.isna(w) or pd.isna(l):
                continue
            at_away_win = False
            if marker_c is not None:
                at_away_win = str(row[marker_c]).strip() == "@"
            if at_away_win:
                home, away = l, w
            else:
                home, away = w, l
            hs = as_ = None
            if w_pts is not None and l_pts is not None:
                try:
                    pw = float(row[w_pts]) if pd.notna(row[w_pts]) else None
                    pl = float(row[l_pts]) if pd.notna(row[l_pts]) else None
                except (TypeError, ValueError):
                    pw = pl = None
                if pw is not None and pl is not None:
                    if at_away_win:
                        hs, as_ = pl, pw
                    else:
                        hs, as_ = pw, pl
            homes.append(to_abbr(home))
            aways.append(to_abbr(away))
            home_scores.append(hs)
            away_scores.append(as_)

        out = pd.DataFrame({
            "home_team": homes,
            "away_team": aways,
            "home_score": home_scores,
            "away_score": away_scores,
        })
        if "Week" in df.columns:
            # align only if same length – rebuild from kept rows is safer; skip week if mismatch
            if len(out) == len(df):
                out["week"] = df["Week"].values
        if "Date" in df.columns and len(out) == len(df):
            out["date"] = df["Date"].values
        return out

    # Upcoming / generic: explicit Away + Home (or Visitor + Home) without results yet
    away_c = None
    home_c = None
    for a_name, h_name in (
        ("Away", "Home"),
        ("away", "home"),
        ("Visitor", "Home"),
        ("Visitor/Neutral", "Home/Neutral"),
        ("Away Team", "Home Team"),
        ("away_team", "home_team"),
        ("VisTm", "HomeTm"),
        ("VisTm", "Home"),
        ("Visitor Team", "Home Team"),
    ):
        if a_name in df.columns and h_name in df.columns:
            away_c, home_c = a_name, h_name
            break
    # case-insensitive fallback
    if away_c is None:
        lower_map = {c.lower().strip(): c for c in df.columns}
        if "away" in lower_map and "home" in lower_map:
            away_c, home_c = lower_map["away"], lower_map["home"]
        elif "visitor" in lower_map and "home" in lower_map:
            away_c, home_c = lower_map["visitor"], lower_map["home"]
        elif "vistm" in lower_map and "hometm" in lower_map:
            away_c, home_c = lower_map["vistm"], lower_map["hometm"]
        elif "vistm" in lower_map and "home" in lower_map:
            away_c, home_c = lower_map["vistm"], lower_map["home"]

    if away_c and home_c:
        out = pd.DataFrame()
        out["away_team"] = df[away_c].map(to_abbr)
        out["home_team"] = df[home_c].map(to_abbr)
        # optional scores if present (including PFR Pts / Pts.1 beside VisTm / HomeTm)
        pts_cols = [c for c in df.columns if str(c).lower().startswith("pts")]
        for score_names, dest in (
            (("home_score", "Home Pts", "Pts Home", "Score.1"), "home_score"),
            (("away_score", "Away Pts", "Pts Away", "Score"), "away_score"),
        ):
            for sn in score_names:
                if sn in df.columns:
                    out[dest] = pd.to_numeric(df[sn], errors="coerce")
                    break
        # Positional: VisTm, Pts, ..., HomeTm, Pts.1
        if "away_score" not in out.columns or out["away_score"].isna().all():
            if len(pts_cols) >= 1:
                out["away_score"] = pd.to_numeric(df[pts_cols[0]], errors="coerce")
        if "home_score" not in out.columns or out["home_score"].isna().all():
            if len(pts_cols) >= 2:
                out["home_score"] = pd.to_numeric(df[pts_cols[1]], errors="coerce")
        if "home_score" not in out.columns:
            out["home_score"] = pd.NA
        if "away_score" not in out.columns:
            out["away_score"] = pd.NA
        if "Week" in df.columns:
            out["week"] = df["Week"]
        if "Date" in df.columns:
            out["date"] = df["Date"]
        # drop rows without two teams
        out = out.dropna(subset=["home_team", "away_team"])
        out = out[
            out["home_team"].astype(str).str.len().gt(0)
            & out["away_team"].astype(str).str.len().gt(0)
        ]
        return out.reset_index(drop=True)

    raise ValueError(
        f"Unrecognized schedule schema. Columns: {list(df.columns)}. "
        "Expected home_team/away_team, Winner/Loser, or Away/Home (upcoming) format."
    )






def filter_regular_season(schedule: pd.DataFrame, sport: str = "NFL") -> pd.DataFrame:
    """Keep regular-season games only (NFL week<=18; NHL/NBA before mid-April)."""
    if schedule is None or schedule.empty:
        return schedule
    df = schedule.copy()
    sport_u = sport.upper()
    if sport_u == "NFL" and "week" in df.columns:
        week_num = pd.to_numeric(df["week"], errors="coerce")
        return df[week_num.notna() & (week_num <= 18)].copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            max_year = int(dt.dt.year.max())
            cutoff = pd.Timestamp(year=max_year, month=4, day=16)
            return df[dt < cutoff].copy()
    cfg = SPORT_CONFIGS.get(sport_u, {})
    if cfg.get("max_week") and "week" in df.columns:
        return df[pd.to_numeric(df["week"], errors="coerce") <= cfg["max_week"]].copy()
    return df


def parse_game_scores(row):
    """
    Return (home_score, away_score) when both values are present and numeric.

    Ties are included. Missing or unparsable scores return None.
    """
    if row is None:
        return None
    try:
        hs = row["home_score"] if "home_score" in row.index else None
        as_ = row["away_score"] if "away_score" in row.index else None
    except Exception:
        hs = row.get("home_score") if hasattr(row, "get") else None
        as_ = row.get("away_score") if hasattr(row, "get") else None
    if hs is None or as_ is None or pd.isna(hs) or pd.isna(as_):
        return None
    try:
        return float(hs), float(as_)
    except (TypeError, ValueError):
        return None


def parse_locked_scores(row):
    """
    Return (home_score, away_score) when the game has a decisive result.

    Used for Elo updates. Ties have scores but no winner; they are still
    counted as locked by describe_schedule_lock via parse_game_scores.
    """
    scores = parse_game_scores(row)
    if scores is None:
        return None
    hs_f, as_f = scores
    if hs_f == as_f:
        return None
    return hs_f, as_f


def describe_schedule_lock(schedule: pd.DataFrame) -> dict:
    """
    Derive lock counts and status from a canonical schedule.

    Status:
      upcoming     — no games locked
      in_progress  — some but not all games locked
      final        — every game locked
    """
    if schedule is None or len(schedule) == 0:
        return {"n_games": 0, "n_locked": 0, "status": "upcoming"}
    n_games = int(len(schedule))
    n_locked = 0
    for _, row in schedule.iterrows():
        if parse_game_scores(row) is not None:
            n_locked += 1
    if n_locked <= 0:
        status = "upcoming"
    elif n_locked >= n_games:
        status = "final"
    else:
        status = "in_progress"
    return {"n_games": n_games, "n_locked": n_locked, "status": status}


def format_lock_line(sport: str, season, lock_info: dict) -> str:
    """Sidebar caption. 'final' is omitted from the line; upcoming / in progress are not."""
    n_locked = int(lock_info.get("n_locked", 0))
    n_games = int(lock_info.get("n_games", 0))
    status = lock_info.get("status", "upcoming")
    label = f"{sport} {season}"
    if status == "upcoming":
        return f"{label} · upcoming · {n_locked}/{n_games} games locked"
    if status == "in_progress":
        return f"{label} · in progress · {n_locked}/{n_games} games locked"
    return f"{label} · {n_locked}/{n_games} games locked"


def describe_target_season_lock(sport: str, season: str) -> dict:
    """Load the target regular-season schedule and return lock status."""
    df = _load_season_schedule(sport, season)
    if df is None or df.empty:
        return {"n_games": 0, "n_locked": 0, "status": "upcoming"}
    df = filter_regular_season(df, sport=sport)
    return describe_schedule_lock(df)


def is_playoff_row(row, sport: str = "NFL") -> bool:
    sport_u = sport.upper()
    if sport_u == "NFL":
        week = row.get("week") if hasattr(row, "get") else None
        if week is None or (isinstance(week, float) and pd.isna(week)):
            return False
        try:
            float(week)
            return False
        except (TypeError, ValueError):
            return True
    try:
        dt = pd.to_datetime(row.get("date") if hasattr(row, "get") else row["date"], errors="coerce")
        if pd.notna(dt) and (dt.month > 4 or (dt.month == 4 and dt.day >= 16)):
            return True
    except Exception:
        pass
    return False


def simulate_season(
    schedule_path=None,
    config=None,
    seed=42,
    initial_ratings=None,
    sport="NFL",
    schedule_df=None,
    regular_season_only=True,
):

    """
    Core season simulation with sport support (NHL points + OT, NFL/NBA wins).

    Games with both scores filled are locked (same outcome in every sim).
    Games with missing, unparsable, or tied scores are Monte Carlo sampled.
    """
    cfg = SPORT_CONFIGS.get(sport, SPORT_CONFIGS["NFL"])

    if schedule_df is not None:
        schedule = schedule_df.copy()
        if "home_team" not in schedule.columns or "away_team" not in schedule.columns:
            schedule = normalize_schedule(schedule, sport=sport)
    else:
        if schedule_path is None:
            schedule_path = cfg["schedule_path"]
        try:
            schedule = pd.read_csv(schedule_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Schedule file not found: {schedule_path}\n"
                "→ Prefer per-season files under data/{sport}/ "
                "(e.g. data/nba/nba_2025.csv). Combined CSVs are legacy."
            )
        schedule = normalize_schedule(schedule, sport=sport)

    if "season" in schedule.columns and schedule["season"].nunique() > 1:
        schedule = schedule[schedule["season"] == schedule["season"].iloc[0]]

    if regular_season_only:
        schedule = filter_regular_season(schedule, sport=sport)
    elif cfg.get("max_week") and "week" in schedule.columns:
        schedule = schedule[pd.to_numeric(schedule["week"], errors="coerce") <= cfg["max_week"]]

    sort_cols = []
    if "season" in schedule.columns:
        sort_cols.append("season")
    if "week" in schedule.columns:
        sort_cols.append("week")
    if "date" in schedule.columns and not sort_cols:
        sort_cols.append("date")
    if sort_cols:
        schedule = schedule.sort_values(sort_cols)
    else:
        schedule = schedule.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Collect every team that appears in this schedule
    schedule_teams = (
        pd.concat([schedule["home_team"], schedule["away_team"]])
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if initial_ratings is None:
        initial_ratings = {team: 1500.0 for team in schedule_teams}
    else:
        # Ensure every schedule team has an entry (prevents KeyError)
        initial_ratings = dict(initial_ratings)
        for team in schedule_teams:
            if team not in initial_ratings:
                # try case-insensitive match
                upper_map = {k.upper(): k for k in initial_ratings}
                if team.upper() in upper_map:
                    initial_ratings[team] = initial_ratings[upper_map[team.upper()]]
                else:
                    initial_ratings[team] = 1500.0

    team_elo = {t: float(initial_ratings[t]) for t in schedule_teams}
    # also keep any extra keys from initial_ratings
    for t, e in initial_ratings.items():
        if t not in team_elo:
            team_elo[t] = float(e)

    wins = {team: 0 for team in team_elo}
    points = {team: 0 for team in team_elo}
    losses = {team: 0 for team in team_elo}
    team_gp = {team: 0 for team in team_elo}  # per-team games played

    elo_history = []
    game_number = 0

    for _, game in schedule.iterrows():
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        week = game["week"] if "week" in game.index and pd.notna(game.get("week")) else (game_number // 2 + 1)
        game_number += 1

        # Safety – should never hit after the fill above
        if home not in team_elo:
            team_elo[home] = 1500.0
            wins[home] = losses[home] = points[home] = 0
        if away not in team_elo:
            team_elo[away] = 1500.0
            wins[away] = losses[away] = points[away] = 0

        pregame = compute_pregame(
            home_elo=team_elo[home],
            away_elo=team_elo[away],
            context={
                "season": game.get("season"),
                "week": week,
                "home_team": home,
                "away_team": away,
            },
            config=config,
        )

        p_home = pregame.get("p_home", 0.5)
        recorded = parse_game_scores(game)
        locked_decisive = parse_locked_scores(game)
        if recorded is not None:
            hs_f, as_f = recorded
            home_score, away_score = hs_f, as_f
            goes_to_ot = False
            if hs_f == as_f:
                # Locked tie: played, no winner, Elo unchanged (engine actual is 0/1).
                team_gp[home] = team_gp.get(home, 0) + 1
                team_gp[away] = team_gp.get(away, 0) + 1
                elo_history.append({
                    "team": home,
                    "elo": team_elo[home],
                    "wins": wins[home],
                    "points": points[home],
                    "week": week,
                    "game_number": game_number,
                    "games_played": team_gp[home],
                })
                elo_history.append({
                    "team": away,
                    "elo": team_elo[away],
                    "wins": wins[away],
                    "points": points[away],
                    "week": week,
                    "game_number": game_number,
                    "games_played": team_gp[away],
                })
                continue
            actual_home_win = 1 if hs_f > as_f else 0
        else:
            actual_home_win = int(rng.random() < p_home)
            home_score, away_score = float(actual_home_win), float(1 - actual_home_win)
            goes_to_ot = rng.random() < cfg.get("ot_rate", 0.0)

        result = run_game(
            home_elo=team_elo[home],
            away_elo=team_elo[away],
            context={
                "season": game.get("season"),
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "actual": actual_home_win,
                "locked": locked_decisive is not None,
            },
            config=config,
        )

        team_elo[home] = result["home_elo_post"]
        team_elo[away] = result["away_elo_post"]

        team_gp[home] = team_gp.get(home, 0) + 1
        team_gp[away] = team_gp.get(away, 0) + 1

        if cfg.get("outcome_type") == "points":
            # NHL: 2 points for win, 1 for OT loss.
            # Locked games use regulation scoring (OT not recorded on the slate).
            if actual_home_win:
                wins[home] += 1
                points[home] += cfg.get("points_per_win", 2)
                if goes_to_ot:
                    points[away] += cfg.get("points_per_ot_loss", 1)
                else:
                    losses[away] += 1
            else:
                wins[away] += 1
                points[away] += cfg.get("points_per_win", 2)
                if goes_to_ot:
                    points[home] += cfg.get("points_per_ot_loss", 1)
                else:
                    losses[home] += 1
        else:
            # NFL / NBA: standard win/loss
            if actual_home_win:
                wins[home] += 1
                losses[away] += 1
            else:
                wins[away] += 1
                losses[home] += 1

        # Record after outcome update so cumulative wins/points are current
        elo_history.append({
            "team": home,
            "elo": team_elo[home],
            "wins": wins[home],
            "points": points[home],
            "week": week,
            "game_number": game_number,
            "games_played": team_gp[home],
        })
        elo_history.append({
            "team": away,
            "elo": team_elo[away],
            "wins": wins[away],
            "points": points[away],
            "week": week,
            "game_number": game_number,
            "games_played": team_gp[away],
        })

    # Only teams that appear in the schedule (avoids double-counting when
    # initial_ratings keys use abbreviations and the schedule uses full names).
    standing_teams = list(schedule_teams)
    standings = pd.DataFrame({
        "team": standing_teams,
        "wins": [wins.get(t, 0) for t in standing_teams],
        "losses": [losses.get(t, 0) for t in standing_teams],
        "points": [points.get(t, wins.get(t, 0)) for t in standing_teams],
        "elo": [team_elo.get(t, 1500.0) for t in standing_teams],
    })

    return standings, team_elo, pd.DataFrame(elo_history)


def _aggregate_elo_evolution(history_frames):
    """Aggregate per-sim Elo histories into mean / p05 / p95 by team and games_played."""
    if not history_frames:
        return pd.DataFrame(columns=["team", "games_played", "mean_elo", "p05_elo", "p95_elo"])

    history = pd.concat(history_frames, ignore_index=True)
    group_col = "games_played" if "games_played" in history.columns else "week"

    evolution = (
        history.groupby(["sim_id", "team", group_col])["elo"]
        .last()
        .reset_index()
        .groupby(["team", group_col])["elo"]
        .agg(
            mean_elo="mean",
            p05_elo=lambda x: x.quantile(0.05),
            p95_elo=lambda x: x.quantile(0.95),
        )
        .reset_index()
        .sort_values(["team", group_col])
    )
    return evolution.rename(columns={group_col: "games_played"})


def _aggregate_win_evolution(history_frames, metric: str = "wins"):
    """Aggregate per-sim cumulative wins (or points) into mean / p05 / p95 by team and games_played."""
    cols = ["team", "games_played", f"mean_{metric}", f"p05_{metric}", f"p95_{metric}"]
    if not history_frames:
        return pd.DataFrame(columns=cols)

    history = pd.concat(history_frames, ignore_index=True)
    if metric not in history.columns:
        return pd.DataFrame(columns=cols)

    group_col = "games_played" if "games_played" in history.columns else "week"

    evolution = (
        history.groupby(["sim_id", "team", group_col])[metric]
        .last()
        .reset_index()
        .groupby(["team", group_col])[metric]
        .agg(
            mean_val="mean",
            p05_val=lambda x: x.quantile(0.05),
            p95_val=lambda x: x.quantile(0.95),
        )
        .reset_index()
        .sort_values(["team", group_col])
    )
    evolution = evolution.rename(columns={
        group_col: "games_played",
        "mean_val": f"mean_{metric}",
        "p05_val": f"p05_{metric}",
        "p95_val": f"p95_{metric}",
    })
    return evolution



def _aggregate_season_evolution(history_frames):
    """Elo + cumulative wins/points evolution in one frame."""
    elo_df = _aggregate_elo_evolution(history_frames)
    wins_df = _aggregate_win_evolution(history_frames, metric="wins")
    points_df = _aggregate_win_evolution(history_frames, metric="points")

    out = elo_df
    for extra in (wins_df, points_df):
        if extra is None or extra.empty:
            continue
        keys = ["team", "games_played"]
        cols = [c for c in extra.columns if c not in keys]
        out = out.merge(extra[keys + cols], on=keys, how="left") if not out.empty else extra
    return out


def simulate_many_seasons(
    n_sims=500,
    schedule_path=None,
    config=None,
    seed=42,
    initial_ratings=None,
    sport="NFL",
    return_elo_evolution=False,
    return_season_results=False,
    progress_callback=None,
):
    """
    Monte Carlo regular-season simulations.

    Parameters
    ----------
    return_elo_evolution : bool
        If True, also return aggregated Elo trajectories from the *same*
        simulated game outcomes (so final Elo ranks align with wins/points).
    return_season_results : bool
        If True, also return a list of (standings_df, team_elo) for each sim.
        Used by the playoff path so playoffs can be run on the *same* outcomes
        without a second full regular-season pass.
    progress_callback : callable, optional
        Called as progress_callback(fraction: float) with fraction in [0, 1]
        approximately every 10% of sims (or more frequently for small n_sims).

    Returns
    -------
    results_df : pd.DataFrame
        Columns: sim_id, team, wins, points
    elo_evolution : pd.DataFrame (only if return_elo_evolution=True)
        Columns: team, games_played, mean_elo, p05_elo, p95_elo
    season_results : list (only if return_season_results=True)
        List of (standings_df, team_elo_dict) tuples, one per sim.
    """
    cfg = SPORT_CONFIGS.get(sport, SPORT_CONFIGS["NFL"])
    if schedule_path is None:
        schedule_path = cfg["schedule_path"]

    results = []
    history = []
    season_results = [] if return_season_results else None
    # Report at 10% increments (and 0 / 100)
    report_every = max(1, n_sims // 10)
    if progress_callback is not None:
        try:
            progress_callback(0.0)
        except Exception:
            pass
    for i in range(n_sims):
        standings, team_elo, elo_hist = simulate_season(
            schedule_path=schedule_path,
            config=config,
            seed=seed + i,
            initial_ratings=initial_ratings,
            sport=sport,
        )
        for _, row in standings.iterrows():
            results.append({
                "sim_id": i,
                "team": row["team"],
                "wins": int(row["wins"]),
                "points": int(row.get("points", row["wins"])),
            })
        if return_season_results:
            # Keep a shallow copy of the final Elo dict; standings is already a new DF
            season_results.append((standings, dict(team_elo)))
        if return_elo_evolution:
            elo_hist = elo_hist.copy()
            elo_hist["sim_id"] = i
            history.append(elo_hist)
        if progress_callback is not None and ((i + 1) % report_every == 0 or i + 1 == n_sims):
            try:
                progress_callback((i + 1) / n_sims)
            except Exception:
                pass

    results_df = pd.DataFrame(results)
    out = [results_df]
    if return_elo_evolution:
        out.append(_aggregate_season_evolution(history))
    if return_season_results:
        out.append(season_results)
    return out[0] if len(out) == 1 else tuple(out)


def simulate_elo_evolution(
    n_sims=500,
    schedule_path=None,
    config=None,
    seed=42,
    initial_ratings=None,
    sport="NFL",
):
    """
    Elo trajectories from the same Monte Carlo path as simulate_many_seasons.

    Prefer calling simulate_many_seasons(..., return_elo_evolution=True) when
    you also need standings, so wins and Elo share outcomes.
    """
    _, evolution = simulate_many_seasons(
        n_sims=n_sims,
        schedule_path=schedule_path,
        config=config,
        seed=seed,
        initial_ratings=initial_ratings,
        sport=sport,
        return_elo_evolution=True,
    )
    return evolution


def summarize_simulations(sim_results):
    agg_spec = {
        "median_wins": ("wins", "median"),
        "mean_wins": ("wins", "mean"),
    }
    if "points" in sim_results.columns:
        agg_spec["mean_points"] = ("points", "mean")
        agg_spec["median_points"] = ("points", "median")
    return sim_results.groupby("team").agg(**agg_spec).reset_index()


def win_distributions(sim_results):
    return sim_results


def _load_season_schedule(sport: str, season: str):
    """Load and normalize one season's schedule."""
    errors = []
    for c in [
        Path(f"data/{sport.lower()}/{sport.lower()}_{season}.csv"),
        Path(f"data/{sport.lower()}/{season}.csv"),
    ]:
        if c.exists():
            try:
                return normalize_schedule(pd.read_csv(c), sport=sport)
            except Exception as e:
                errors.append(f"{c}: {e}")
                continue
    if errors:
        raise FileNotFoundError(
            f"No usable schedule for {sport} {season}. "
            f"Tried files but normalize failed: {'; '.join(errors)}"
        )
    return None


def warm_elo_from_actual_history(
    sport: str,
    seasons,
    config=None,
    initial_ratings=None,
    inter_season_regression: float = 0.35,
    mean_elo: float = 1500.0,
    playoff_k_multiplier: float = 1.75,
    include_playoffs: bool = True,
):
    """
    Update Elo from actual results (regular season + optional playoffs).

    Prior releases started target-season Monte Carlo from a static/ranking prior
    only. This warm-up walks real scores first so recent form enters the ratings.
    """
    cfg_base = dict(config or {})
    base_k = float(cfg_base.get("k", 20))
    ratings = dict(initial_ratings) if initial_ratings else {}

    for s in seasons:
        df = _load_season_schedule(sport, s)
        if df is None or df.empty:
            continue
        if not include_playoffs:
            df = filter_regular_season(df, sport=sport)
        if df.empty:
            continue

        teams = (
            pd.concat([df["home_team"], df["away_team"]])
            .dropna().astype(str).str.strip().unique().tolist()
        )
        for team in teams:
            ratings.setdefault(team, mean_elo)

        for _, game in df.iterrows():
            home = str(game["home_team"]).strip()
            away = str(game["away_team"]).strip()
            hs, as_ = game.get("home_score"), game.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue
            try:
                hs_f, as_f = float(hs), float(as_)
            except (TypeError, ValueError):
                continue
            if hs_f == as_f:
                continue

            actual_home_win = 1 if hs_f > as_f else 0
            playoff = is_playoff_row(game, sport)
            k = base_k * (playoff_k_multiplier if playoff else 1.0)
            cfg = dict(cfg_base)
            cfg["k"] = k

            result = run_game(
                home_elo=float(ratings[home]),
                away_elo=float(ratings[away]),
                context={
                    "season": s,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs_f,
                    "away_score": as_f,
                    "actual": actual_home_win,
                    "is_playoff": playoff,
                },
                config=cfg,
            )
            ratings[home] = float(result["home_elo_post"])
            ratings[away] = float(result["away_elo_post"])

        if inter_season_regression and inter_season_regression > 0:
            reg = float(inter_season_regression)
            ratings = {
                team: (1.0 - reg) * e + reg * mean_elo
                for team, e in ratings.items()
            }

    return ratings


def simulate_many_seasons_multiyear(
    n_sims: int = 250,
    sport: str = "NHL",
    target_season: str = None,
    config=None,
    seed: int = 42,
    base_initial_ratings=None,
    return_elo_evolution: bool = True,
    return_season_results: bool = False,
    hybrid_warmup: bool = True,
    inter_season_regression: float = 0.35,
    playoff_k_multiplier: float = 1.75,
    include_playoffs_in_warmup: bool = True,
    max_history_seasons: int = 2,
    progress_callback=None,
):
    """
    Warm-up Elo on actual recent seasons, then Monte Carlo the target season only.

    Replaces the prior-only start used in published releases through 11.4.

    return_season_results : bool
        If True, also return list of (standings_df, team_elo) for each target
        sim so playoffs can reuse the same outcomes (no second regular-season pass).
    progress_callback : callable, optional
        Called as progress_callback(fraction: float) with fraction in [0, 1]
        approximately every 10% of the Monte Carlo target-season sims.
    """
    try:
        from app.services.initial_ratings_service import get_available_seasons
    except ImportError:
        from services.initial_ratings_service import get_available_seasons  # type: ignore

    all_seasons = get_available_seasons(sport)
    if not all_seasons:
        raise ValueError(f"No seasons available for {sport}")
    if target_season is None:
        target_season = all_seasons[-1]
    if target_season not in all_seasons:
        raise ValueError(f"target_season {target_season} not in {all_seasons}")

    target_idx = all_seasons.index(target_season)
    history_all = all_seasons[:target_idx]
    if max_history_seasons and max_history_seasons > 0:
        history_seasons = history_all[-max_history_seasons:]
    else:
        history_seasons = history_all

    start_ratings = dict(base_initial_ratings) if base_initial_ratings else {}

    warmed = warm_elo_from_actual_history(
        sport=sport,
        seasons=history_seasons,
        config=config,
        initial_ratings=start_ratings,
        inter_season_regression=inter_season_regression,
        playoff_k_multiplier=playoff_k_multiplier,
        include_playoffs=include_playoffs_in_warmup,
    )

    target_df = _load_season_schedule(sport, target_season)
    if target_df is None or target_df.empty:
        raise FileNotFoundError(f"No schedule for target {target_season}")
    target_df = filter_regular_season(target_df, sport=sport)

    results, history = [], []
    season_results = [] if return_season_results else None
    report_every = max(1, n_sims // 10)
    if progress_callback is not None:
        try:
            progress_callback(0.0)
        except Exception:
            pass
    for i in range(n_sims):
        standings, team_elo, elo_hist = simulate_season(
            schedule_df=target_df,
            config=config,
            seed=seed + i,
            initial_ratings=warmed,
            sport=sport,
            regular_season_only=False,
        )
        for _, row in standings.iterrows():
            results.append({
                "sim_id": i,
                "team": row["team"],
                "wins": int(row["wins"]),
                "points": int(row.get("points", row["wins"])),
            })
        if return_season_results:
            season_results.append((standings, dict(team_elo)))
        if return_elo_evolution and elo_hist is not None and not elo_hist.empty:
            h = elo_hist.copy()
            h["sim_id"] = i
            history.append(h)
        if progress_callback is not None and ((i + 1) % report_every == 0 or i + 1 == n_sims):
            try:
                progress_callback((i + 1) / n_sims)
            except Exception:
                pass

    results_df = pd.DataFrame(results)
    out = [results_df]
    if return_elo_evolution:
        out.append(_aggregate_season_evolution(history))
    if return_season_results:
        out.append(season_results)
    return out[0] if len(out) == 1 else tuple(out)

import pandas as pd
import tempfile
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Use the multi-sport playoff wrapper
# ---------------------------------------------------------------------------
from elo_lab.playoffs.uncertainty import MAX_PLAYOFF_SIMS
from elo_lab.workflows.simulate_with_playoffs import simulate_many_seasons

from elo_lab.workflows.simulate_season import (
    summarize_simulations,
    win_distributions,
)

try:
    from elo_lab.configuration.sport_configs import SPORT_CONFIGS
except ImportError:
    SPORT_CONFIGS = {}

try:
    from metadata.nhl_teams import NHL_TEAMS
except ImportError:
    NHL_TEAMS = {}

try:
    from metadata.nba_teams import NBA_TEAMS
except ImportError:
    NBA_TEAMS = {}

try:
    from metadata.nfl_teams import NFL_TEAMS
except ImportError:
    NFL_TEAMS = {}


def _team_meta_lookup(teams: dict) -> dict:
    """Map lowercase abbr and full name → metadata dict."""
    lookup = {}
    for abbr, data in (teams or {}).items():
        lookup[str(abbr).lower().strip()] = data
        name = data.get("name")
        if name:
            lookup[str(name).lower().strip()] = data
    return lookup


def _attach_conference_division(sim_data: pd.DataFrame, teams: dict) -> pd.DataFrame:
    lookup = _team_meta_lookup(teams)

    def get_meta(team_name):
        return lookup.get(str(team_name).lower().strip(), {})

    out = sim_data.copy()
    out["conference"] = out["team"].apply(lambda x: get_meta(x).get("conference"))
    out["division"] = out["team"].apply(lambda x: get_meta(x).get("division"))
    return out


def _resolve_schedule_path(sport: str, season=None) -> tuple:
    """
    Return (schedule_path, tmp_path_or_None).

    Preference (combined CSVs are temporary / being removed):
      1. Per-season file for the requested season under data/{sport}/
      2. Concatenate all per-season files under data/{sport}/ (optionally filter)
      3. Combined CSV (legacy fallback while visualizations still need it)
    """
    sport_l = sport.lower()
    season_dir = Path(f"data/{sport_l}")

    # 1. Specific per-season file
    if season is not None and season_dir.is_dir():
        candidates = [
            season_dir / f"{sport_l}_{season}.csv",
            season_dir / f"{season}.csv",
        ]
        for c in candidates:
            if c.exists():
                return str(c), None

    # 2. Build from all per-season files
    if season_dir.is_dir():
        files = sorted(season_dir.glob(f"{sport_l}_*.csv"))
        if not files:
            files = sorted(season_dir.glob("*.csv"))
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if "season" not in df.columns:
                    year = f.stem.split("_")[-1]
                    df["season"] = year
                frames.append(df)
            except Exception:
                continue
        if frames:
            combined_df = pd.concat(frames, ignore_index=True)
            if season is not None and "season" in combined_df.columns:
                combined_df = combined_df[
                    combined_df["season"].astype(str) == str(season)
                ]
            if not combined_df.empty:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False, newline=""
                )
                combined_df.to_csv(tmp.name, index=False)
                tmp.close()
                return tmp.name, tmp.name

    # 3. Legacy combined CSV
    combined = {
        "NHL": "data/nhl_games.csv",
        "NFL": "data/nfl_games.csv",
        "NBA": "data/nba_games.csv",
    }.get(sport, f"data/{sport_l}_games.csv")

    cfg = SPORT_CONFIGS.get(sport, {})
    if cfg.get("schedule_path"):
        combined = cfg["schedule_path"]

    if Path(combined).exists():
        if season is None:
            return combined, None
        try:
            df = pd.read_csv(combined)
            if "season" in df.columns:
                filtered = df[df["season"].astype(str) == str(season)]
                if not filtered.empty:
                    tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", delete=False, newline=""
                    )
                    filtered.to_csv(tmp.name, index=False)
                    tmp.close()
                    return tmp.name, tmp.name
        except Exception:
            pass
        return combined, None

    return combined, None


def calculate_achievement_probabilities(sim_results: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Regular-season achievement rates across Monte Carlo sims.

    Columns (rates later scaled to 0–100):
      make_playoffs, home_ice, first_in_division, first_in_conference, first_in_league

    ``home_ice`` is a generic key; UI labels it by sport:
      NHL — top 2 in division (home-ice style edge)
      NBA — top 4 in conference (1st-round home court)
      NFL — top 4 in conference (bye or host Wild Card)
    """
    empty = pd.DataFrame(columns=[
        "team", "make_playoffs", "home_ice",
        "first_in_division", "first_in_conference", "first_in_league",
    ])
    if sim_results.empty:
        return empty

    sport_u = sport.upper()
    all_results = []

    for sim_id in sim_results["sim_id"].unique():
        sim_data = sim_results[sim_results["sim_id"] == sim_id].copy()

        if sport_u == "NHL":
            sim_data = _attach_conference_division(sim_data, NHL_TEAMS)
            matched = sim_data.dropna(subset=["conference", "division"]).copy()
            if matched.empty:
                continue
            matched = matched.sort_values("points", ascending=False)

            qualified, home_adv, first_div, first_conf = set(), set(), set(), set()
            for conf in ("Eastern", "Western"):
                conf_teams = matched[matched["conference"] == conf]
                qualified.update(conf_teams.head(8)["team"].tolist())
                if not conf_teams.empty:
                    first_conf.add(conf_teams.iloc[0]["team"])
            for _, group in matched.groupby(["conference", "division"]):
                home_adv.update(group.head(2)["team"].tolist())
                if not group.empty:
                    first_div.add(group.iloc[0]["team"])
            first_league = matched.iloc[0]["team"] if not matched.empty else None

        elif sport_u == "NBA":
            sim_data = _attach_conference_division(sim_data, NBA_TEAMS)
            matched = sim_data.dropna(subset=["conference", "division"]).copy()
            if matched.empty:
                continue
            # Rank by wins (points column may be absent)
            sort_col = "wins" if "wins" in matched.columns else "points"
            matched = matched.sort_values(sort_col, ascending=False)

            qualified, home_adv, first_div, first_conf = set(), set(), set(), set()
            for conf in ("Eastern", "Western"):
                conf_teams = matched[matched["conference"] == conf]
                # Top 8 per conference make the playoffs (Play-In simplified away)
                qualified.update(conf_teams.head(8)["team"].tolist())
                # Seeds 1–4 have home court in the first round
                home_adv.update(conf_teams.head(4)["team"].tolist())
                if not conf_teams.empty:
                    first_conf.add(conf_teams.iloc[0]["team"])
            for _, group in matched.groupby(["conference", "division"]):
                if not group.empty:
                    first_div.add(group.iloc[0]["team"])
            first_league = matched.iloc[0]["team"] if not matched.empty else None

        elif sport_u == "NFL":
            sim_data = _attach_conference_division(sim_data, NFL_TEAMS)
            matched = sim_data.dropna(subset=["conference", "division"]).copy()
            if matched.empty:
                continue
            sort_col = "wins" if "wins" in matched.columns else "points"
            matched = matched.sort_values(sort_col, ascending=False)

            qualified, home_adv, first_div, first_conf = set(), set(), set(), set()
            for conf in ("AFC", "NFC"):
                conf_teams = matched[matched["conference"] == conf]
                # Division winners
                winners = []
                for _, group in conf_teams.groupby("division"):
                    if not group.empty:
                        w = group.iloc[0]["team"]
                        winners.append(w)
                        first_div.add(w)
                # Wild cards: best remaining records in conference
                remaining = conf_teams[~conf_teams["team"].isin(winners)]
                wild = remaining.head(3)["team"].tolist()
                qualified.update(winners)
                qualified.update(wild)
                # Seeds 1–4 (division winners by record) host or have a bye
                winners_sorted = conf_teams[conf_teams["team"].isin(winners)]
                home_adv.update(winners_sorted.head(4)["team"].tolist())
                if not conf_teams.empty:
                    first_conf.add(conf_teams.iloc[0]["team"])
            first_league = matched.iloc[0]["team"] if not matched.empty else None

        else:
            # Unknown sport — leave achievements empty for this sim
            continue

        sim_data["make_playoffs"] = sim_data["team"].isin(qualified)
        sim_data["home_ice"] = sim_data["team"].isin(home_adv)
        sim_data["first_in_division"] = sim_data["team"].isin(first_div)
        sim_data["first_in_conference"] = sim_data["team"].isin(first_conf)
        sim_data["first_in_league"] = sim_data["team"] == first_league

        all_results.append(sim_data[[
            "sim_id", "team", "make_playoffs", "home_ice",
            "first_in_division", "first_in_conference", "first_in_league",
        ]])

    if not all_results:
        return empty

    df = pd.concat(all_results, ignore_index=True)
    prob_df = df.groupby("team").agg(
        make_playoffs=("make_playoffs", "mean"),
        home_ice=("home_ice", "mean"),
        first_in_division=("first_in_division", "mean"),
        first_in_conference=("first_in_conference", "mean"),
        first_in_league=("first_in_league", "mean"),
    ).reset_index()

    for col in prob_df.columns:
        if col != "team":
            prob_df[col] = (prob_df[col] * 100).round(1)

    return prob_df


def run_simulation(
    config,
    n_sims,
    initial_ratings=None,
    sport="NFL",
    season=None,
    from_season=None,
    seed: int = 42,
    inter_season_regression: float = 0.35,
    progress_callback=None,
):
    """
    When season is set: warm Elo on actual recent seasons, then MC the target.
    When season is None: legacy single-pass path (prior-only style).

    from_season : optional start of the warm-up window (inclusive).
                  History runs from from_season through the season before
                  the target. If None, falls back to the previous default
                  of max 2 history seasons.
    seed : RNG seed for reproducible Monte Carlo draws.
    inter_season_regression : pull toward mean_elo after each warm-up season
                  (0.35 default; previously 0.67 which erased recent form).
    progress_callback : callable, optional
        Called as progress_callback(fraction: float) with fraction in [0, 1]
        during Monte Carlo work (approximately 10% steps).
    """
    tmp_path = None
    try:
        n_collected = int(n_sims)
        if season is not None:
            from elo_lab.workflows.simulate_season import simulate_many_seasons_multiyear
            try:
                from services.initial_ratings_service import get_available_seasons
            except ImportError:
                from app.services.initial_ratings_service import get_available_seasons

            # Map explicit from_season → max_history_seasons
            max_history = 2  # legacy default
            if from_season is not None:
                all_seasons = get_available_seasons(sport)
                if (
                    from_season in all_seasons
                    and str(season) in all_seasons
                ):
                    from_idx = all_seasons.index(from_season)
                    target_idx = all_seasons.index(str(season))
                    # number of seasons in [from_season, target)
                    max_history = max(1, target_idx - from_idx)

            # Single target-season Monte Carlo + collect per-sim standings/Elo
            # so playoffs reuse the exact same outcomes (no second full pass).
            def _multi_progress(frac):
                if progress_callback is not None:
                    try:
                        # Regular-season phase ≈ 0–85% of reported progress
                        progress_callback(0.85 * float(frac))
                    except Exception:
                        pass

            sim_results, elo_evolution, season_results = simulate_many_seasons_multiyear(
                n_sims=n_sims,
                sport=sport,
                target_season=str(season),
                config=config,
                seed=int(seed),
                base_initial_ratings=initial_ratings or {},
                return_elo_evolution=True,
                return_season_results=True,
                hybrid_warmup=True,
                inter_season_regression=float(inter_season_regression),
                playoff_k_multiplier=1.75,
                include_playoffs_in_warmup=True,
                max_history_seasons=max_history,
                progress_callback=_multi_progress,
            )

            # Playoffs from the collected target-season outcomes only
            n_collected = len(season_results) if season_results else 0
            _, playoff_probs, _ = simulate_many_seasons(
                n_sims=n_sims,  # ignored when season_results is supplied
                config=config,
                seed=int(seed),
                sport=sport,
                season_results=season_results,
                progress_callback=progress_callback,
            )
        else:
            schedule_path, tmp_path = _resolve_schedule_path(sport, season)
            sim_results, playoff_probs, elo_evolution = simulate_many_seasons(
                n_sims=n_sims,
                schedule_path=schedule_path,
                config=config,
                seed=int(seed),
                initial_ratings=initial_ratings,
                sport=sport,
                progress_callback=progress_callback,
            )

        summary = summarize_simulations(sim_results)
        distributions = win_distributions(sim_results)
        achievement_probs = calculate_achievement_probabilities(sim_results, sport)
        n_playoff_sims = (
            min(int(n_collected), MAX_PLAYOFF_SIMS) if n_collected else 0
        )

        return {
            "summary": summary,
            "distribution": distributions,
            "elo_evolution": elo_evolution,
            "achievement_probs": achievement_probs,
            "playoff_probs": playoff_probs,
            "raw_sim_results": sim_results,
            "n_sims": int(n_sims),
            "n_playoff_sims": int(n_playoff_sims),
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

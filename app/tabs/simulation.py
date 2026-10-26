"""
Playoff Projections tab — MoneyPuck-inspired odds table + playoff spirals,
plus regular-season summary views.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from metadata import load_teams, NFL_TEAMS, NHL_TEAMS, NBA_TEAMS
except ImportError:
    from metadata.nfl_teams import NFL_TEAMS
    from metadata.nhl_teams import NHL_TEAMS
    try:
        from metadata.nba_teams import NBA_TEAMS
    except ImportError:
        NBA_TEAMS = {}

    def load_teams(sport):
        return {"NFL": NFL_TEAMS, "NHL": NHL_TEAMS, "NBA": NBA_TEAMS}.get(sport, {})

try:
    from components.moneypuck_viz import (
        render_playoff_odds_table,
        render_playoff_spirals,
        render_playoff_path_bars,
        PLAYOFF_TABLE_SPEC,
    )
except ImportError:
    from app.components.moneypuck_viz import (  # type: ignore
        render_playoff_odds_table,
        render_playoff_spirals,
        render_playoff_path_bars,
        PLAYOFF_TABLE_SPEC,
    )

from elo_lab.playoffs.uncertainty import playoff_sample_size


def get_team_color_map(sport):
    try:
        teams = load_teams(sport)
    except Exception:
        teams = NHL_TEAMS if sport == "NHL" else (NBA_TEAMS if sport == "NBA" else NFL_TEAMS)
    color_map = {}
    for abbr, data in teams.items():
        color = data.get("primary_color", "#888888")
        color_map[abbr] = color
        color_map[data.get("name", abbr)] = color
    return color_map


def render_simulation_tab(sport="NFL"):
    st.header(f"{sport} Playoff Projections")

    if "simulation_results" not in st.session_state:
        st.info("Run a simulation from the sidebar to see results.")
        return

    results = st.session_state.get("simulation_results", {})
    summary = results.get("summary", pd.DataFrame())
    distribution = results.get("distribution", pd.DataFrame())
    achievement_probs = results.get("achievement_probs", pd.DataFrame())
    playoff_probs = results.get("playoff_probs", {})

    if summary is None or (hasattr(summary, "empty") and summary.empty):
        st.warning("No simulation results available yet.")
        return

    sport_u = sport.upper()
    has_playoff_spec = sport_u in PLAYOFF_TABLE_SPEC
    n_playoff = playoff_sample_size(results)

    # ------------------------------------------------------------------
    # MoneyPuck-style Playoff Odds table
    # ------------------------------------------------------------------
    if has_playoff_spec and playoff_probs:
        st.subheader("Playoff Odds")
        render_playoff_odds_table(
            sport=sport_u,
            playoff_probs=playoff_probs,
            achievement_probs=achievement_probs
            if isinstance(achievement_probs, pd.DataFrame)
            else None,
            summary=summary if isinstance(summary, pd.DataFrame) else None,
            n_sims=n_playoff,
        )

        # ------------------------------------------------------------------
        # Playoff spirals (one per conference)
        # ------------------------------------------------------------------
        st.subheader("Playoff spirals")
        render_playoff_spirals(sport_u, playoff_probs, n_sims=n_playoff)

        # ------------------------------------------------------------------
        # Playoff path bars (Color 1…5 palette)
        # ------------------------------------------------------------------
        st.subheader("Playoff path by round")
        render_playoff_path_bars(sport_u, playoff_probs, top_n=12)

    elif has_playoff_spec:
        st.caption("Playoff probability data not available for this run.")

    # ------------------------------------------------------------------
    # Regular Season Achievement Probabilities (only if no playoff table)
    # ------------------------------------------------------------------
    if (
        achievement_probs is not None
        and isinstance(achievement_probs, pd.DataFrame)
        and not achievement_probs.empty
        and not (has_playoff_spec and playoff_probs)
    ):
        st.subheader("Regular Season Achievement Probabilities")
        sport_u = str(sport).upper() if sport else ""
        if sport_u == "NHL":
            rename_map = {
                "make_playoffs": "Make Playoffs",
                "home_ice": "Home Ice (Top 2 Div)",
                "first_in_division": "1st in Division",
                "first_in_conference": "1st in Conference",
                "first_in_league": "1st in League",
            }
        else:
            # NBA / NFL: hide NHL-oriented RS achievement columns
            rename_map = {"make_playoffs": "Make Playoffs"}
        keep_cols = ["team"] + [c for c in rename_map if c in achievement_probs.columns]
        display_df = achievement_probs[keep_cols].rename(columns=rename_map)
        if "Make Playoffs" in display_df.columns:
            display_df = display_df.sort_values("Make Playoffs", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

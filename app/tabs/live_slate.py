"""Live Slate tab — locked results vs remaining schedule."""

from __future__ import annotations

import streamlit as st
import pandas as pd

try:
    from elo_lab.workflows.live_slate import (
        read_slate_status,
        upcoming_and_locked_tables,
    )
    from elo_lab.workflows.simulate_season import (
        describe_target_season_lock,
        format_lock_line,
    )
except ImportError:
    describe_target_season_lock = None
    format_lock_line = None
    read_slate_status = None
    upcoming_and_locked_tables = None


def render_live_slate_tab(sport: str = "NFL", season: str | None = None):
    sport_u = (sport or "NFL").upper()
    season = season or st.session_state.get("season")
    st.header(f"{sport_u} Live Slate")
    st.caption(
        "Completed games are written into the season CSV and locked in Monte Carlo. "
        "Blank games stay sampled. Refresh from the sidebar, then Run Simulation "
        "to fold new results into playoff odds."
    )

    reports_by_sport = st.session_state.get("slate_reports") or {}
    report = None
    from_map = False
    if isinstance(reports_by_sport, dict):
        mapped = reports_by_sport.get(sport_u)
        if isinstance(mapped, dict):
            report = mapped
            from_map = True
    if report is None:
        legacy = st.session_state.get("slate_report")
        if isinstance(legacy, dict) and str(legacy.get("sport", "")).upper() == sport_u:
            report = legacy

    show_report = False
    if isinstance(report, dict) and str(report.get("sport", "")).upper() == sport_u:
        report_season = str(report.get("season") or "")
        show_report = from_map or report_season == str(season)
    if show_report:
        if report.get("error"):
            st.warning(f"Last refresh failed: {report['error']}")
        else:
            when = report.get("fetched_at", "")
            report_season = str(report.get("season") or "")
            if report_season and report_season != str(season):
                lead = f"Last refresh for {report_season} ({when})"
            else:
                lead = f"Last refresh {when}"
            st.success(
                f"{lead}: "
                f"{report.get('updated', 0)} newly locked · "
                f"{report.get('already_locked', 0)} already on file · "
                f"{report.get('fetched', 0)} completed games fetched"
            )
    else:
        report = None

    lock = None
    if describe_target_season_lock and season:
        try:
            lock = describe_target_season_lock(sport_u, season)
            st.subheader(format_lock_line(sport_u, season, lock))
        except Exception as exc:
            st.info(f"Lock status unavailable: {exc}")

    meta = None
    if read_slate_status and season:
        try:
            meta = read_slate_status(sport_u, season)
        except Exception:
            meta = None
    if meta and meta.get("fetched_at"):
        st.caption(f"Score feed last written {meta['fetched_at']}.")

    if lock:
        n_locked = int(lock.get("n_locked") or 0)
        n_games = int(lock.get("n_games") or 0)
        remaining = max(n_games - n_locked, 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Locked", f"{n_locked}")
        c2.metric("Remaining", f"{remaining}")
        c3.metric("Slate", lock.get("status", "upcoming").replace("_", " "))

    locked_df, open_df = pd.DataFrame(), pd.DataFrame()
    if upcoming_and_locked_tables and season:
        try:
            locked_df, open_df = upcoming_and_locked_tables(sport_u, season, limit=16)
        except Exception as exc:
            st.warning(f"Could not read slate tables: {exc}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Recently locked**")
        if locked_df is None or locked_df.empty:
            st.write("No scored games on file yet. Refresh the live slate after games finish.")
        else:
            st.dataframe(locked_df, hide_index=True, use_container_width=True)
    with right:
        st.markdown("**Next unlocked games**")
        if open_df is None or open_df.empty:
            st.write("No remaining blank games on this slate.")
        else:
            st.dataframe(open_df, hide_index=True, use_container_width=True)

    if report and report.get("unmatched_games"):
        with st.expander(f"Unmatched feed games ({len(report['unmatched_games'])})"):
            st.dataframe(pd.DataFrame(report["unmatched_games"]), hide_index=True)

"""Streamlit entry point. See CHANGELOG.md and docs/ARCHITECTURE.md."""

from __future__ import annotations

import pickle
from pathlib import Path

import bootstrap
import streamlit as st
import pandas as pd

from components.layout import configure_page
from tabs.simulation import render_simulation_tab
from tabs.elo_evolution import render_elo_evolution_tab
from tabs.evaluation import render_evaluation_tab
from tabs.live_slate import render_live_slate_tab

from services.simulation_service import run_simulation
from services.initial_ratings_service import (
    get_available_seasons,
    get_simulatable_seasons,
    get_seed_season,
    get_simulate_from_options,
    get_initial_ratings,
)
from services.export_service import build_full_export, make_export_filename
from services.evaluation_service import score_config_on_season
from elo_lab.workflows.optimize_parameters import optimize_parameters_for_config
from elo_lab.workflows.simulate_season import (
    describe_target_season_lock,
    format_lock_line,
)
from elo_lab.workflows.live_slate import (
    read_slate_status,
    season_overrides_for_refresh,
    update_all_current,
)

# Clean metadata API (single source of truth for teams + venues)
from metadata import NFL_TEAMS, NHL_TEAMS, NBA_TEAMS, get_sport_teams, load_teams


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
configure_page()

st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #FB4F14 !important; }
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: #FB4F14 !important; border-color: #FB4F14 !important;
    }
    /* Quiet export control */
    div[data-testid="stDownloadButton"] button {
        background-color: transparent !important;
        color: #6b7280 !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: 400 !important;
        text-decoration: underline;
        padding: 0.25rem 0 !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        color: #374151 !important;
        background-color: transparent !important;
    }
    /* Playoff odds + dataframes: readable headers in light and dark.
       Colors are set on thead and th (not inherited). #f8fafc is text
       on the dark header bar only, never a header background. */
    .elo-odds-table-wrap {
        overflow-x: auto;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    .elo-odds-table {
        border-collapse: collapse;
        width: 100%;
    }
    .elo-odds-table thead,
    .elo-odds-table thead th {
        background: #f1f5f9 !important;
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-bottom: 2px solid #cbd5e1;
        font-weight: 600;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrameResizable"] * {
        --gdg-bg-header: #f1f5f9 !important;
        --gdg-bg-header-hovered: #e2e8f0 !important;
        --gdg-bg-header-has-focus: #e2e8f0 !important;
        --gdg-text-header: #0f172a !important;
        --gdg-text-header-selected: #0f172a !important;
        --gdg-header-fg-color: #0f172a !important;
    }
    [data-testid="stDataFrame"] thead,
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrameResizable"] thead,
    [data-testid="stDataFrameResizable"] thead th {
        background: #f1f5f9 !important;
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    @media (prefers-color-scheme: dark) {
        .elo-odds-table-wrap { border-color: #475569; }
        .elo-odds-table thead,
        .elo-odds-table thead th {
            background: #1e2937 !important;
            background-color: #1e2937 !important;
            color: #f8fafc !important;
            border-bottom-color: #64748b;
        }
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        [data-testid="stDataFrameResizable"],
        [data-testid="stDataFrameResizable"] * {
            --gdg-bg-header: #1e2937 !important;
            --gdg-bg-header-hovered: #334155 !important;
            --gdg-bg-header-has-focus: #334155 !important;
            --gdg-text-header: #f8fafc !important;
            --gdg-text-header-selected: #f8fafc !important;
            --gdg-header-fg-color: #f8fafc !important;
        }
        [data-testid="stDataFrame"] thead,
        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataFrameResizable"] thead,
        [data-testid="stDataFrameResizable"] thead th {
            background: #1e2937 !important;
            background-color: #1e2937 !important;
            color: #f8fafc !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            color: #e5e7eb !important;
        }
    }
    html[data-theme="dark"] .elo-odds-table-wrap { border-color: #475569; }
    html[data-theme="dark"] .elo-odds-table thead,
    html[data-theme="dark"] .elo-odds-table thead th {
        background: #1e2937 !important;
        background-color: #1e2937 !important;
        color: #f8fafc !important;
        border-bottom-color: #64748b;
    }
    html[data-theme="dark"] [data-testid="stDataFrame"],
    html[data-theme="dark"] [data-testid="stDataFrame"] *,
    html[data-theme="dark"] [data-testid="stDataFrameResizable"],
    html[data-theme="dark"] [data-testid="stDataFrameResizable"] * {
        --gdg-bg-header: #1e2937 !important;
        --gdg-bg-header-hovered: #334155 !important;
        --gdg-bg-header-has-focus: #334155 !important;
        --gdg-text-header: #f8fafc !important;
        --gdg-text-header-selected: #f8fafc !important;
        --gdg-header-fg-color: #f8fafc !important;
    }
    html[data-theme="dark"] [data-testid="stDataFrame"] thead,
    html[data-theme="dark"] [data-testid="stDataFrame"] thead th,
    html[data-theme="dark"] [data-testid="stDataFrameResizable"] thead,
    html[data-theme="dark"] [data-testid="stDataFrameResizable"] thead th {
        background: #1e2937 !important;
        background-color: #1e2937 !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("MultiSport Elo Lab")
st.caption("NFL / NHL / NBA | Version 15.1")


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------
def build_model_config(home_field, margin_of_victory, elevation, k=20,
                       hfa_value=55, elev_value=1.0):
    """Build declarative model config. MOV is pure binary (scale fixed at 1.0)."""
    adjustments = {}
    if home_field:
        adjustments["home_field"] = {"enabled": True, "value": hfa_value}
    if margin_of_victory:
        # Pure binary: on = margin-scaled update with fixed scale 1.0
        adjustments["margin_of_victory"] = {"enabled": True, "scale": 1.0}
    if elevation:
        adjustments["elevation_edge"] = {"enabled": True, "value": elev_value}
    return {"k": k, "adjustments": adjustments}


def get_optimize_for(hf, mov, elev, opt_hf, opt_mov, opt_elev):
    opts = []
    if hf and opt_hf:
        opts.append("home_field")
    if mov and opt_mov:
        opts.append("margin_of_victory")
    if elev and opt_elev:
        opts.append("elevation_edge")
    return opts


def _schedule_path(sport: str) -> str:
    mapping = {
        "NHL": "data/nhl_games.csv",
        "NFL": "data/nfl_games.csv",
        "NBA": "data/nba_games.csv",
    }
    return mapping.get(sport, f"data/{sport.lower()}_games.csv")


# ---------------------------------------------------------------------------
# Precomputed default results loader
# ---------------------------------------------------------------------------
PRECOMPUTED_DIR = Path("data/precomputed")


def load_default_results(sport: str):
    """Return the precomputed simulation dict for `sport`, or None."""
    path = PRECOMPUTED_DIR / f"{sport}_default.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# Sport-specific home advantage display labels (config key remains "home_field")
_HOME_ADV_LABELS = {
    "NHL": {
        "checkbox": "Home-Ice Advantage",
        "slider": "Home-ice advantage value",
        "optimize": "Optimize Home Ice",
    },
    "NBA": {
        "checkbox": "Home-Court Advantage",
        "slider": "Home-court advantage value",
        "optimize": "Optimize Home Court",
    },
    "NFL": {
        "checkbox": "Home-Field Advantage",
        "slider": "Home-field advantage value",
        "optimize": "Optimize Home Field",
    },
}


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("Model Configuration")

# Sport buttons (NFL default)
_SPORT_ORDER = ["NFL", "NHL", "NBA"]
if "sport" not in st.session_state or st.session_state.get("sport") not in _SPORT_ORDER:
    st.session_state["sport"] = "NFL"
st.sidebar.caption("Sport")
_sport_cols = st.sidebar.columns(len(_SPORT_ORDER))
for _i, _s in enumerate(_SPORT_ORDER):
    with _sport_cols[_i]:
        _selected = st.session_state["sport"] == _s
        if st.button(
            _s,
            key=f"sport_btn_{_s}",
            type="primary" if _selected else "secondary",
            use_container_width=True,
        ):
            if not _selected:
                st.session_state["sport"] = _s
                st.rerun()
sport = st.session_state["sport"]

# ------------------------------------------------------------------
# Auto-load precomputed defaults when sport changes (or on first load)
# ------------------------------------------------------------------
_prev_sport = st.session_state.get("_loaded_sport")
if (
    "simulation_results" not in st.session_state
    or _prev_sport != sport
):
    defaults = load_default_results(sport)
    if defaults is not None:
        st.session_state["simulation_results"] = defaults
        st.session_state["_loaded_sport"] = sport
        st.session_state["is_default_run"] = True
        # Align with defaults: target = latest season; warm-up from previous season
        try:
            _sims = get_simulatable_seasons(sport)
            if _sims:
                st.session_state["season"] = _sims[-1]
                _from_opts = get_simulate_from_options(sport, target_season=_sims[-1])
                # Previous season = last option before the target
                st.session_state["simulate_from"] = _from_opts[-1] if _from_opts else None
            st.session_state["_results_fingerprint"] = (
                f"{sport}|{st.session_state.get('season')}|{st.session_state.get('simulate_from')}|default"
            )
        except Exception:
            pass
    else:
        # No precomputed file – clear any stale results from another sport
        if _prev_sport != sport:
            st.session_state.pop("simulation_results", None)
            st.session_state.pop("is_default_run", None)
        st.session_state["_loaded_sport"] = sport

_home = _HOME_ADV_LABELS.get(sport, _HOME_ADV_LABELS["NFL"])

season_options = get_simulatable_seasons(sport)
seed_year = get_seed_season(sport)
season = st.sidebar.selectbox(
    "Season",
    season_options,
    index=len(season_options) - 1 if season_options else 0,
    help=(
        "Target season for Monte Carlo simulation. Ratings are first warmed "
        "on actual results from the 'Simulate from' season through the season "
        f"before this target"
        f"{f' (seed year {seed_year} is history-only)' if seed_year else ''}."
    ),
)
st.session_state["season"] = season

# ------------------------------------------------------------------
# Simulate from (explicit warm-up start)
# ------------------------------------------------------------------
from_options = get_simulate_from_options(sport, target_season=season)
if from_options:
    # Default = previous season (most recent completed year before the target)
    default_from_idx = len(from_options) - 1
    simulate_from = st.sidebar.selectbox(
        "Simulate from",
        from_options,
        index=default_from_idx,
        help=(
            "Start of the Elo warm-up window. Actual results from this season "
            "through the season before the target are used to build ratings; "
            "only the target season is then Monte Carlo simulated. "
            "Default is the previous season (most recent completed year). "
            "Choose an earlier year to include a longer history window "
            "(e.g. target 2027 + from 2024 uses 2024–2026). "
            "The earliest season in the data is seed-only and is not offered here."
        ),
        key=f"simulate_from_{sport}",
    )
else:
    simulate_from = None
st.session_state["simulate_from"] = simulate_from

apply_regression = st.sidebar.checkbox(
    "Apply regression to mean",
    value=True,
    help="Pull ratings toward the league mean after ranking / between seasons.",
)

# Clarify the full warm-up window (from … through year before target)
if simulate_from and season:
    try:
        _all = get_available_seasons(sport)
        _from_i = _all.index(str(simulate_from))
        _tgt_i = _all.index(str(season))
        _window = _all[_from_i:_tgt_i]  # exclusive of target
        if len(_window) == 1:
            _window_label = _window[0]
        elif len(_window) >= 2:
            _window_label = f"{_window[0]}–{_window[-1]}"
        else:
            _window_label = None
        if _window_label:
            st.sidebar.caption(
                f"Warm-up window: **{_window_label}** "
                f"(actual results) → simulate **{season}**"
            )
    except Exception:
        pass

try:
    _lock = describe_target_season_lock(sport, season)
    st.sidebar.caption(format_lock_line(sport, season, _lock))
except Exception:
    _lock = None

st.sidebar.subheader("Live Slate")
_slate_meta = None
try:
    _slate_meta = read_slate_status(sport, season)
except Exception:
    _slate_meta = None
if _slate_meta and _slate_meta.get("fetched_at"):
    st.sidebar.caption(f"Scores as of {_slate_meta['fetched_at']}")
refresh_slate = st.sidebar.button(
    "Refresh live slate",
    help=(
        "Pull completed games for NFL, NHL, and NBA "
        "(nflverse, NHL API, NBA CDN with ESPN fallback) "
        "and write each into that sport's season CSV. "
        "Locked games stay fixed on the next Monte Carlo run."
    ),
)
_slate_any_updated = False
if refresh_slate:
    with st.sidebar.status("Refreshing live slate...", expanded=True) as slate_status:
        try:
            _remembered = st.session_state.get("seasons_by_sport")
            if not isinstance(_remembered, dict):
                _remembered = None
            _reports = update_all_current(
                seasons=season_overrides_for_refresh(sport, season, _remembered)
            )
            _by_sport = {
                **dict(st.session_state.get("slate_reports") or {}),
                **{r.sport: r.to_dict() for r in _reports},
            }
            st.session_state["slate_reports"] = _by_sport
            if sport.upper() in _by_sport:
                st.session_state["slate_report"] = _by_sport[sport.upper()]
            _slate_any_updated = any(bool(r.updated) for r in _reports)
            if _slate_any_updated:
                st.session_state["is_default_run"] = False
            for _line in _reports:
                st.markdown(_line.feedback_line().replace("_", r"\_"))
            _failed = [r.sport for r in _reports if r.error]
            if not _reports:
                slate_status.update(
                    label="Slate refresh failed",
                    state="error",
                    expanded=True,
                )
            elif _failed:
                slate_status.update(
                    label=f"{', '.join(_failed)} failed",
                    state="error",
                    expanded=True,
                )
            else:
                slate_status.update(
                    label="Refreshed NFL, NHL, and NBA",
                    state="complete",
                    expanded=True,
                )
        except Exception as _slate_err:
            _err_report = {
                "sport": sport,
                "season": str(season),
                "error": str(_slate_err),
            }
            _saved = dict(st.session_state.get("slate_reports") or {})
            _saved[str(sport).upper()] = _err_report
            st.session_state["slate_reports"] = _saved
            st.session_state["slate_report"] = _err_report
            slate_status.update(
                label=f"Slate refresh failed: {_slate_err}",
                state="error",
                expanded=True,
            )
if _slate_any_updated:
    st.sidebar.info(
        "New results are on the slate. Click **Run Simulation** "
        "to update playoff odds for the sport you then simulate."
    )

st.sidebar.divider()
st.sidebar.subheader("Parameters")
home_field = st.sidebar.checkbox(_home["checkbox"], value=True)
margin_of_victory = st.sidebar.checkbox(
    "Margin of Victory",
    value=True,
    help="On = post-game Elo update is scaled by margin of victory. "
         "Off = update depends only on win/loss.",
)
elevation = st.sidebar.checkbox("Elevation Edge", value=False)

st.sidebar.divider()

# ------------------------------------------------------------------
# Grid Search hierarchy: master checkbox, then indented targets
# ------------------------------------------------------------------
optimize_params = st.sidebar.checkbox("Grid Search", value=False)
opt_hf = opt_mov = opt_elev = False
if optimize_params:
    st.sidebar.caption("Select which parameters to search:")
    if home_field:
        opt_hf = st.sidebar.checkbox(_home["optimize"], value=True, key="opt_hf")
    if margin_of_victory:
        opt_mov = st.sidebar.checkbox("Optimize MOV", value=True, key="opt_mov")
    if elevation:
        opt_elev = st.sidebar.checkbox("Optimize Elevation", value=False, key="opt_elev")

# Fixed (non-user-facing) initial-Elo policy:
#   rating_source always "playoffs"
#   rating_basis prefers "elo" when available, else falls back to "record"
rating_source = "playoffs"
rating_basis = "elo"

# ---------------------------------------------------------------------------
# Advanced expander – engine knobs only (Initial Elo UI removed)
# ---------------------------------------------------------------------------
with st.sidebar.expander("Customize Parameters", expanded=False):
    regression_strength = st.slider(
        "Regression strength",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        disabled=not apply_regression,
        help="How strongly ratings are pulled toward the league mean when "
             "regression is enabled.",
    )

    st.markdown("---")
    st.markdown("**Engine parameters**")
    _default_k = {"NHL": 12, "NBA": 22, "NFL": 20}.get(sport, 20)
    k = st.slider(
        "k-factor",
        min_value=5,
        max_value=40,
        value=_default_k,
        step=1,
        help="How much ratings move after each game. "
             "Higher k = more reactive (recent results dominate). "
             "Lower k = more stable (history carries more weight).",
    )
    _default_hfa = {"NHL": 35, "NBA": 55, "NFL": 55}.get(sport, 55)
    hfa_value = st.slider(
        _home["slider"], min_value=0, max_value=100, value=_default_hfa, step=5
    )
    # MOV scale removed – pure binary via the Adjustments checkbox
    elev_value = st.slider(
        "Elevation Edge",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Home-team Elo boost = value × max(0, home_ft − away_ft) / 1000. "
             "Default 1.0. Optimization searches [0, 2, 4, 6, 8, 10].",
    )

st.sidebar.divider()

simulation_options = [25, 100, 500, 1000, 5000, 10000]
simulation_count = st.sidebar.selectbox(
    "Simulation Count",
    simulation_options,
    index=1,  # default 100
    format_func=lambda x: f"{x:,}",
)
_seed_col, _ = st.sidebar.columns([1, 1])
with _seed_col:
    _seed_raw = st.text_input(
        "Random seed",
        value="42",
        help="Type an integer seed. Same seed + same settings reproduces the same Monte Carlo results.",
        key="sim_seed_input",
    )
try:
    sim_seed = int(str(_seed_raw).strip())
    if sim_seed < 0:
        sim_seed = 42
except (TypeError, ValueError):
    sim_seed = 42
st.session_state["sim_seed"] = sim_seed

# ---------------------------------------------------------------------------
# Run Simulation + Stop placement
# ---------------------------------------------------------------------------
run_clicked = st.sidebar.button("Run Simulation", type="primary")

# Placeholder for future async stop control (Streamlit runs are synchronous today).
# Kept under Run so the action cluster stays together.
if st.session_state.get("_sim_running"):
    st.sidebar.button("Stop Simulation", type="secondary", disabled=True,
                      help="Stop is not yet supported for in-progress runs. "
                           "Refresh the page to cancel.")

if run_clicked:
    config = build_model_config(
        home_field,
        margin_of_victory,
        elevation,
        k=k,
        hfa_value=hfa_value,
        elev_value=elev_value,
    )
    optimize_for = get_optimize_for(
        home_field, margin_of_victory, elevation, opt_hf, opt_mov, opt_elev
    )

    initial_ratings = {}  # warm-up starts flat; actual history sets Elo

    st.session_state["_sim_running"] = True
    with st.sidebar.status("Running simulation...", expanded=True) as status:
        pb = st.sidebar.progress(0, text="Starting...")

        if optimize_for:
            pb.progress(10, text="Optimizing parameters...")
            best_config, _ = optimize_parameters_for_config(
                base_config=config,
                optimize_for=optimize_for,
            )
            final_config = best_config
            pb.progress(15, text="Optimization complete")
        else:
            final_config = config

        def _mc_progress(frac: float):
            # 10% steps with explicit numeric indicator
            pct = int(round(float(frac) * 100))
            # Snap to nearest 10 for stable labels when callback is dense
            snapped = (pct // 10) * 10
            pb.progress(
                min(max(frac, 0.0), 1.0),
                text=f"Running Monte Carlo simulations... {snapped}%",
            )

        pb.progress(0, text="Running Monte Carlo simulations... 0%")
        results = run_simulation(
            config=final_config,
            n_sims=simulation_count,
            initial_ratings=initial_ratings,
            sport=sport,
            season=season,
            from_season=simulate_from,
            seed=int(sim_seed),
            progress_callback=_mc_progress,
        )

        # Store everything the tabs already know how to read
        st.session_state["simulation_results"] = results
        st.session_state["sport"] = sport
        st.session_state["season"] = season
        st.session_state["simulate_from"] = simulate_from
        st.session_state["last_config"] = final_config
        st.session_state["optimize_for"] = optimize_for
        st.session_state["final_config"] = final_config
        st.session_state["initial_ratings"] = initial_ratings
        st.session_state["rating_source"] = rating_source
        st.session_state["rating_basis"] = rating_basis
        st.session_state["apply_regression"] = apply_regression
        st.session_state["is_default_run"] = False          # custom run overrides defaults
        st.session_state["_results_fingerprint"] = (
            f"{sport}|{season}|{simulate_from}|{simulation_count}|{sim_seed}|{hash(str(final_config))}"
        )
        st.session_state["sim_seed"] = int(sim_seed)

        # Historical eval: walk last completed season with current params
        try:
            from services.initial_ratings_service import get_available_seasons
            all_seasons = get_available_seasons(sport)
            eval_season = None
            if season and all_seasons and str(season) in all_seasons:
                idx = all_seasons.index(str(season))
                if idx > 0:
                    eval_season = all_seasons[idx - 1]
            elif all_seasons and len(all_seasons) >= 2:
                eval_season = all_seasons[-2]
            if eval_season:
                param_eval = score_config_on_season(sport, eval_season, final_config)
                st.session_state["param_eval"] = param_eval
            else:
                st.session_state.pop("param_eval", None)
        except Exception as _eval_err:
            st.session_state["param_eval"] = {"error": str(_eval_err)}

        st.session_state["_loaded_sport"] = sport

        pb.progress(1.0, text="Running Monte Carlo simulations... 100%")
        status.update(label="Simulation complete!", state="complete")
    st.session_state["_sim_running"] = False


# ---------------------------------------------------------------------------
# Global Export – quiet text-style control
# ---------------------------------------------------------------------------
if st.session_state.get("simulation_results") is not None:
    try:
        export_bytes = build_full_export(st.session_state)
        filename = make_export_filename(st.session_state)

        _left, _right = st.columns([3, 1])
        with _right:
            st.download_button(
                label="Export Results (.xlsx)",
                data=export_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=(
                    "Downloads Config, Simulation Summary, Achievement/Playoff probabilities, "
                    "Elo Ratings, and Evaluation metrics in one Excel file."
                ),
                # type intentionally omitted / secondary so CSS can quiet it
            )
    except Exception as e:
        # Never let export problems break the rest of the dashboard
        st.warning(f"Export temporarily unavailable: {e}")


# ---------------------------------------------------------------------------
# Main tabs (v12 three-tab structure)
#   1. Regular Season Projections  (former Elo Trajectory)
#   2. Playoff Projections         (former Season Simulation) — default landing
#   3. Model Comparison            (former Model Evaluation)
#
# Streamlit st.tabs always opens the first tab. To make Playoff Projections the
# default landing view we place it first in the widget, while the visual/logical
# product order remains Regular → Playoff → Model via the labels below only if
# we accepted first-tab default = Regular. Instead we put Playoff first so the
# app lands on the primary fan-facing view.
# ---------------------------------------------------------------------------
_saved_reports = st.session_state.get("slate_reports") or {}
if isinstance(_saved_reports, dict) and sport.upper() in _saved_reports:
    st.session_state["slate_report"] = _saved_reports[sport.upper()]

tabs = st.tabs([
    "Playoff Projections",
    "Regular Season Projections",
    "Live Slate",
    "Model Comparison",
])

with tabs[0]:
    # Default landing tab
    if st.session_state.get("is_default_run"):
        st.caption(
            "Showing precomputed default simulation. "
            "Click **Run Simulation** in the sidebar to re-run with your current settings."
        )
    try:
        _tab_lock = describe_target_season_lock(sport, season)
        if _tab_lock and int(_tab_lock.get("n_locked") or 0) > 0:
            st.caption(
                f"{format_lock_line(sport, season, _tab_lock)}. "
                "Locked games use real scores; remaining games are simulated."
            )
    except Exception:
        pass
    render_simulation_tab(sport=sport)

with tabs[1]:
    render_elo_evolution_tab(sport=sport)

with tabs[2]:
    render_live_slate_tab(sport=sport, season=season)

with tabs[3]:
    render_evaluation_tab()

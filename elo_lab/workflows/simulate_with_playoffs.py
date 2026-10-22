"""
elo_lab/workflows/simulate_with_playoffs.py

Drop-in wrapper that runs the existing season simulation and then the
sport-specific playoffs (NFL / NHL / NBA).

v12.3: Playoffs now consume the standings + final Elo already produced by the
primary Monte Carlo loop. The previous second full regular-season pass is gone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from elo_lab.workflows.simulate_season import (
    simulate_many_seasons as _original_simulate_many_seasons,
)


# ---------------------------------------------------------------------------
# Round names per sport (used for probability counters / documentation)
# ---------------------------------------------------------------------------
PLAYOFF_ROUNDS = {
    "NFL": ["Wild Card", "Divisional", "Conference", "Super Bowl", "Champion"],
    "NHL": [
        "First Round",
        "Second Round",
        "Conference Finals",
        "Stanley Cup Final",
        "Champion",
    ],
    "NBA": [
        "Play-In",
        "First Round",
        "Conference Semifinals",
        "Conference Finals",
        "NBA Finals",
        "Champion",
    ],
}


def _get_accumulate(sport: str):
    """Return the sport-specific accumulate_from_many_seasons helper."""
    sport = sport.upper()
    if sport == "NFL":
        from elo_lab.playoffs.nfl.workflow_hook import accumulate_from_many_seasons
        return accumulate_from_many_seasons
    if sport == "NHL":
        from elo_lab.playoffs.nhl.workflow_hook import accumulate_from_many_seasons
        return accumulate_from_many_seasons
    if sport == "NBA":
        from elo_lab.playoffs.nba.workflow_hook import accumulate_from_many_seasons
        return accumulate_from_many_seasons
    return None


def _home_advantage_for(sport: str, config: Optional[dict]) -> float:
    """Pull home advantage from config when available; fall back to sensible defaults."""
    if config:
        adj = (config.get("adjustments") or {}).get("home_field") or {}
        if adj.get("enabled") and "value" in adj:
            return float(adj["value"])
    defaults = {"NFL": 30.0, "NHL": 25.0, "NBA": 45.0}
    return defaults.get(sport.upper(), 30.0)


def simulate_many_seasons(
    n_sims: int = 500,
    schedule_path: Optional[str] = None,
    config: Optional[dict] = None,
    seed: int = 42,
    initial_ratings: Optional[dict] = None,
    sport: str = "NFL",
    progress_callback=None,
    season_results: Optional[List[Tuple[pd.DataFrame, Dict[str, float]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Same interface as the original simulate_many_seasons, but also returns
    playoff probabilities when a playoff module exists for the sport.

    v12.3 behaviour
    ---------------
    - One regular-season Monte Carlo pass produces standings, Elo trajectories,
      *and* the per-sim (standings_df, team_elo) pairs needed for playoffs.
    - Playoff probabilities are accumulated from those same outcomes.
    - No second full regular-season loop.

    Parameters
    ----------
    season_results : optional pre-computed list of (standings_df, team_elo)
        When supplied (e.g. from a multiyear target-season pass), the regular-
        season Monte Carlo is skipped and only playoffs are run. results_df /
        elo_evolution are then empty placeholders; callers that already have
        those objects should ignore them.

    progress_callback : callable, optional
        Called as progress_callback(fraction) with fraction in [0, 1].
        Regular-season phase occupies ~0–0.85; playoff accumulation ~0.85–1.0
        (playoffs are comparatively cheap).

    Returns
    -------
    results_df, playoff_probs, elo_evolution
    """
    sport_u = sport.upper()
    playoff_probs: Dict[str, Dict[str, float]] = {}
    accumulate = _get_accumulate(sport_u)
    hfa = _home_advantage_for(sport_u, config)

    # ------------------------------------------------------------------
    # Path A: caller already has season_results (multiyear target sims)
    # ------------------------------------------------------------------
    if season_results is not None:
        if progress_callback is not None:
            try:
                progress_callback(0.85)
            except Exception:
                pass
        if accumulate is not None and season_results:
            # Cap only for extreme n_sims; normally use all collected outcomes
            use = season_results[: max(1, min(len(season_results), 500))]
            playoff_probs = accumulate(
                season_results=use,
                home_advantage=hfa,
                base_seed=int(seed),
            )
        if progress_callback is not None:
            try:
                progress_callback(1.0)
            except Exception:
                pass
        # Callers that supply season_results already hold results_df / evolution
        empty_df = pd.DataFrame()
        return empty_df, playoff_probs, empty_df

    # ------------------------------------------------------------------
    # Path B: full regular-season MC + playoffs from the same outcomes
    # ------------------------------------------------------------------
    def _reg_progress(frac):
        if progress_callback is not None:
            try:
                progress_callback(0.85 * float(frac))
            except Exception:
                pass

    results_df, elo_evolution, collected = _original_simulate_many_seasons(
        n_sims=n_sims,
        schedule_path=schedule_path,
        config=config,
        seed=seed,
        initial_ratings=initial_ratings,
        sport=sport,
        return_elo_evolution=True,
        return_season_results=True,
        progress_callback=_reg_progress,
    )

    if accumulate is None:
        if progress_callback is not None:
            try:
                progress_callback(1.0)
            except Exception:
                pass
        return results_df, playoff_probs, elo_evolution

    if progress_callback is not None:
        try:
            progress_callback(0.85)
        except Exception:
            pass

    # Use the exact same simulated seasons for playoffs (no re-sim)
    use = collected[: max(1, min(len(collected), 500))]
    playoff_probs = accumulate(
        season_results=use,
        home_advantage=hfa,
        base_seed=int(seed),
    )

    if progress_callback is not None:
        try:
            progress_callback(1.0)
        except Exception:
            pass

    return results_df, playoff_probs, elo_evolution

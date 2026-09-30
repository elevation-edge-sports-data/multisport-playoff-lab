"""Home field, margin of victory, and Elevation Edge.

Each function takes the game state dict and returns it. ``ADJUSTMENTS``
maps the config key to the function the engine calls.
"""

from typing import Any, Dict

import numpy as np

DEFAULT_HFA = 55
DEFAULT_MOV_SCALE = 1.0
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 3.00
DEFAULT_ELEVATION_SCALE = 15.0


def home_field(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add a fixed Elo bump to the home team when enabled."""
    adjustment = state["config"]["adjustments"]["home_field"]
    if not adjustment["enabled"]:
        return state
    state["home_elo"] += adjustment.get("value", DEFAULT_HFA)
    return state


def margin_of_victory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Set the postgame K multiplier from the score margin.

    Off -> 1. On -> log(margin + 1) * scale, clamped to [0.25, 3].
    """
    adjustment = state["config"]["adjustments"]["margin_of_victory"]
    postgame = state.setdefault("postgame", {})
    if not adjustment["enabled"]:
        postgame["mov_multiplier"] = 1.0
        return state

    context = state["context"]
    margin = abs(context["home_score"] - context["away_score"])
    scale = adjustment.get("scale", DEFAULT_MOV_SCALE)
    multiplier = np.log(margin + 1) * scale
    multiplier = min(max(multiplier, MIN_MULTIPLIER), MAX_MULTIPLIER)
    postgame["mov_multiplier"] = float(multiplier)
    return state


def elevation_edge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add boost = scale * max(0, home_ft - away_ft) / 1000 to home Elo."""
    from metadata import get_elevation_ft

    adj = state["config"].get("adjustments", {}).get("elevation_edge", {})
    if not adj.get("enabled", False):
        return state

    context = state.get("context", {})
    home_team = context.get("home_team")
    away_team = context.get("away_team")
    sport = context.get("sport", "NFL").upper()
    if not home_team or not away_team:
        return state

    elev = get_elevation_ft(sport)
    home_ft = float(elev.get(home_team, 0) or 0)
    away_ft = float(elev.get(away_team, 0) or 0)
    delta_ft = max(0.0, home_ft - away_ft)
    if delta_ft == 0.0:
        return state

    scale = float(adj.get("value", DEFAULT_ELEVATION_SCALE))
    boost = scale * (delta_ft / 1000.0)
    state["home_elo"] += boost
    state["elevation_boost"] = boost
    return state


ADJUSTMENTS = {
    "home_field": home_field,
    "margin_of_victory": margin_of_victory,
    "elevation_edge": elevation_edge,
}

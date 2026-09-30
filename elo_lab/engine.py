"""Elo kernel: win probability, rating update, adjustments, and one game."""

from typing import Any, Dict, List, Tuple

INITIAL_ELO = 1500

__all__ = [
    "INITIAL_ELO",
    "win_probability",
    "update_ratings",
    "update_elo",
    "compute_pregame",
    "run_game",
    "apply_adjustments",
]


def win_probability(home_elo: float, away_elo: float) -> float:
    """Home win probability from the Elo logistic."""
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))


def update_ratings(
    home_elo: float,
    away_elo: float,
    actual: int,
    p_home: float,
    k: float,
    multiplier: float = 1.0,
):
    """Move both ratings from the pregame probability and the result."""
    error = actual - p_home
    home_new = home_elo + k * error * multiplier
    away_new = away_elo - k * error * multiplier
    return home_new, away_new


def update_elo(*args, **kwargs):
    """Alias for update_ratings."""
    return update_ratings(*args, **kwargs)


def _validate_state(state: dict) -> None:
    required = ("home_elo", "away_elo", "context", "config", "postgame")
    for key in required:
        if key not in state:
            raise ValueError(f"Missing state field: {key}")
    if not isinstance(state["context"], dict):
        raise TypeError("context must be dict")
    if not isinstance(state["config"], dict):
        raise TypeError("config must be dict")
    if not isinstance(state["postgame"], dict):
        raise TypeError("postgame must be dict")


def _load_adjustment(name: str):
    from elo_lab.adjustments import ADJUSTMENTS

    try:
        return ADJUSTMENTS[name]
    except KeyError:
        raise ModuleNotFoundError(
            f"elo_lab.adjustments has no adjustment {name!r}"
        ) from None


def _build_pipeline(config: Dict[str, Any]) -> Tuple[list, list]:
    adjustments = config.get("adjustments", {})
    pregame = []
    postgame = []

    for name in config.get("pregame_pipeline", []):
        if adjustments.get(name, {}).get("enabled", False):
            pregame.append(_load_adjustment(name))

    for name in config.get("postgame_pipeline", []):
        if adjustments.get(name, {}).get("enabled", False):
            postgame.append(_load_adjustment(name))

    return pregame, postgame


def _run_pregame(state: Dict[str, Any], transforms: List) -> Dict[str, Any]:
    state = dict(state)
    for transform in transforms:
        state = transform(state)
    return state


def _run_postgame(state: Dict[str, Any], transforms: List) -> Dict[str, Any]:
    state = dict(state)
    state["postgame"] = dict(state.get("postgame", {}))
    for transform in transforms:
        state = transform(state)
    return state


def apply_pregame_adjustments(
    home_elo: float,
    away_elo: float,
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    state = {
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "context": dict(context),
        "config": config,
        "postgame": {},
    }
    pregame, _ = _build_pipeline(config)
    return _run_pregame(state, pregame)


def apply_postgame_adjustments(
    home_elo: float,
    away_elo: float,
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    state = {
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "context": dict(context),
        "config": config,
        "postgame": {},
    }
    _, postgame = _build_pipeline(config)
    return _run_postgame(state, postgame)


def apply_adjustments(
    home_elo: float,
    away_elo: float,
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pregame adjustments, then postgame adjustments."""
    state = apply_pregame_adjustments(home_elo, away_elo, context, config)
    _, postgame = _build_pipeline(config)
    return _run_postgame(state, postgame)


def compute_pregame(
    home_elo: float,
    away_elo: float,
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Adjusted ratings and home win probability. Does not update Elo."""
    state = {
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "context": dict(context),
        "config": config,
        "postgame": {},
    }
    _validate_state(state)
    state = apply_pregame_adjustments(
        home_elo=state["home_elo"],
        away_elo=state["away_elo"],
        context=state["context"],
        config=state["config"],
    )
    p_home = win_probability(state["home_elo"], state["away_elo"])
    return {
        "home_elo_pre": float(home_elo),
        "away_elo_pre": float(away_elo),
        "home_elo_adjusted": state["home_elo"],
        "away_elo_adjusted": state["away_elo"],
        "p_home": float(p_home),
    }


def run_game(
    home_elo: float,
    away_elo: float,
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Play one finished game: pregame probability, result, postgame, K update."""
    state = {
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "context": context,
        "config": config,
        "postgame": {},
    }
    _validate_state(state)

    pregame = compute_pregame(
        home_elo=home_elo,
        away_elo=away_elo,
        context=context,
        config=config,
    )
    p_home = pregame["p_home"]

    if "actual" in context:
        actual = int(context["actual"])
    elif "home_score" in context and "away_score" in context:
        actual = int(context["home_score"] > context["away_score"])
    else:
        raise ValueError(
            "Context must contain either 'actual' "
            "or both 'home_score' and 'away_score'."
        )

    state = apply_postgame_adjustments(
        home_elo=home_elo,
        away_elo=away_elo,
        context=context,
        config=config,
    )
    multiplier = state["postgame"].get("mov_multiplier", 1.0)

    home_post, away_post = update_ratings(
        home_elo=home_elo,
        away_elo=away_elo,
        actual=actual,
        p_home=p_home,
        k=config.get("k", 20),
        multiplier=multiplier,
    )

    return {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "home_elo_adjusted": pregame["home_elo_adjusted"],
        "away_elo_adjusted": pregame["away_elo_adjusted"],
        "p_home": float(p_home),
        "actual": actual,
        "multiplier": multiplier,
        "home_elo_post": float(home_post),
        "away_elo_post": float(away_post),
    }

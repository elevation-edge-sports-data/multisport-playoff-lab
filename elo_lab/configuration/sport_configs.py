"""
Sport configurations.

Defines sport-specific runtime defaults such as data paths,
initial Elo values, and default model selection.
"""

from ..engine import INITIAL_ELO


# ==========================================================
# SPORT CONFIGURATIONS
# ==========================================================

SPORT_CONFIGS = {

    "NFL": {
        "schedule_path": "data/nfl_games.csv",
        "initial_elo": INITIAL_ELO,
        "default_model": "MOV_HFA",
        "outcome_type": "wins",          # standard W/L
    },

    "NHL": {
        "schedule_path": "data/nhl_games.csv",
        "initial_elo": INITIAL_ELO,
        "default_model": "MOV_HFA",
        "outcome_type": "points",        # 2 pts win, 1 pt OT loss
        "points_per_win": 2,
        "points_per_ot_loss": 1,
        "ot_rate": 0.25,
    },

    "NBA": {
        "schedule_path": "data/nba_games.csv",  # resolved via per-season fallback if missing
        "initial_elo": INITIAL_ELO,
        "default_model": "MOV_HFA",
        "outcome_type": "wins",          # standard W/L like NFL
    },

}

"""
Historical backtesting workflow.

Executes historical game schedules through the Elo engine,
maintains team ratings across the season, and writes
per-game backtest results for model evaluation.
"""

import os
import sys
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from elo_lab.engine import INITIAL_ELO, run_game
from elo_lab.configuration.model_configs import MODEL_CONFIGS, AVAILABLE_MODELS

os.makedirs("outputs", exist_ok=True)


def run_backtest(config, model_name):

    schedule = pd.read_csv("data/nfl_games.csv")
    schedule = schedule.sort_values(["season", "week"]).reset_index(drop=True)

    team_elo = {}

    def get_elo(team):
        return team_elo.get(team, INITIAL_ELO)

    def set_elo(team, rating):
        team_elo[team] = rating

    results = []

    for _, game in schedule.iterrows():

        home = game["home_team"]
        away = game["away_team"]

        result = run_game(
            home_elo=get_elo(home),
            away_elo=get_elo(away),
            context={
                "season": game["season"],
                "week": game["week"],
                "home_team": home,
                "away_team": away,
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "actual": 1 if game["home_score"] > game["away_score"] else 0,
            },
            config=config,
        )

        set_elo(home, result["home_elo_post"])
        set_elo(away, result["away_elo_post"])

        results.append({
            "season": game["season"],
            "week": game["week"],
            "home_team": home,
            "away_team": away,
            "p_home": result["p_home"],
            "actual": 1 if game["home_score"] > game["away_score"] else 0,
            "home_elo_post": result["home_elo_post"],
            "away_elo_post": result["away_elo_post"],
            "model": model_name,
        })

    df = pd.DataFrame(results)

    df.to_csv(f"outputs/backtest_{model_name}.csv", index=False)

    print(f"Backtest complete: {model_name}")

    return df


if __name__ == "__main__":

    for model in AVAILABLE_MODELS:
        run_backtest(MODEL_CONFIGS[model], model)
"""
NHL single-game simulation, best-of-7 series, and fixed-bracket progression.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from elo_lab.engine import win_probability

from .models import (
    GameResult,
    PlayoffResult,
    RoundResult,
    SeriesResult,
    TeamStanding,
)
from .seeding import seed_nhl_playoffs


def simulate_nhl_game(
    home_id: str,
    away_id: str,
    elo_lookup: Dict[str, float],
    home_advantage: float,
    rng: random.Random,
) -> GameResult:
    """Simulate one NHL playoff game (regulation + OT treated as one decision)."""
    home_elo = elo_lookup.get(home_id, 1500.0) + home_advantage
    away_elo = elo_lookup.get(away_id, 1500.0)
    p_home = win_probability(home_elo, away_elo)
    winner = home_id if rng.random() < p_home else away_id
    return GameResult(
        home=home_id,
        away=away_id,
        winner=winner,
        home_win_prob=p_home,
    )


def simulate_best_of_7(
    higher_id: str,
    lower_id: str,
    elo_lookup: Dict[str, float],
    home_advantage: float,
    rng: random.Random,
) -> SeriesResult:
    """
    Simulate a best-of-7 series with 2-2-1-1-1 home format.
    higher_id has home-ice advantage (hosts Games 1, 2, 5, 7).
    """
    # Home schedule: True = higher_id is home
    home_schedule = [True, True, False, False, True, False, True]

    higher_wins = 0
    lower_wins = 0
    games: List[GameResult] = []

    for game_idx, higher_is_home in enumerate(home_schedule):
        if higher_wins >= 4 or lower_wins >= 4:
            break

        if higher_is_home:
            home, away = higher_id, lower_id
        else:
            home, away = lower_id, higher_id

        g = simulate_nhl_game(home, away, elo_lookup, home_advantage, rng)
        games.append(g)

        if g.winner == higher_id:
            higher_wins += 1
        else:
            lower_wins += 1

    winner = higher_id if higher_wins > lower_wins else lower_id

    return SeriesResult(
        higher_seed=higher_id,
        lower_seed=lower_id,
        winner=winner,
        games=games,
        higher_seed_wins=higher_wins,
        lower_seed_wins=lower_wins,
    )


def _build_reach_dict(
    conf_results: Dict[str, List[RoundResult]],
    stanley_cup: Optional[SeriesResult],
    champion: Optional[str],
) -> Dict[str, Dict[str, bool]]:
    """Build the team → {round: reached} map for probability aggregation."""
    reach: Dict[str, Dict[str, bool]] = {}

    def mark(team: str, round_name: str):
        if team not in reach:
            reach[team] = {}
        reach[team][round_name] = True

    for conf, rounds in conf_results.items():
        for rnd in rounds:
            for t in rnd.advancers:
                mark(t, rnd.name)
            for s in rnd.series:
                mark(s.higher_seed, rnd.name)
                mark(s.lower_seed, rnd.name)

    if stanley_cup is not None:
        mark(stanley_cup.higher_seed, "Stanley Cup Final")
        mark(stanley_cup.lower_seed, "Stanley Cup Final")
        if champion:
            mark(champion, "Champion")

    return reach


def simulate_nhl_playoffs(
    bracket: Dict[str, Dict],
    elo_lookup: Dict[str, float],
    home_advantage: float = 55.0,
    rng: Optional[random.Random] = None,
    final_home_advantage: float = 0.0,  # usually the better regular-season team gets home ice
) -> PlayoffResult:
    """
    Run the full NHL playoff bracket for one season (fixed bracket, no reseeding).

    Parameters
    ----------
    bracket : dict
        Output of seed_nhl_playoffs.
    elo_lookup : dict
        team_id → final regular-season Elo
    home_advantage : float
        Elo points added to the home team in a series game
    rng : random.Random
        Optional seeded RNG
    final_home_advantage : float
        Applied only in the Stanley Cup Final (or set to normal home_advantage
        and let the better team be designated home).
    """
    if rng is None:
        rng = random.Random()

    conf_results: Dict[str, List[RoundResult]] = {}
    finalists: Dict[str, str] = {}

    for conf, conf_data in bracket.items():
        series_defs = conf_data.get("series", [])
        if len(series_defs) < 4:
            conf_results[conf] = []
            continue

        rounds: List[RoundResult] = []

        # ----- First Round (4 series) -----
        first_round_series: List[SeriesResult] = []
        first_advancers: List[str] = []
        for sdef in series_defs:
            ser = simulate_best_of_7(
                sdef["higher"], sdef["lower"], elo_lookup, home_advantage, rng
            )
            first_round_series.append(ser)
            first_advancers.append(ser.winner)

        rounds.append(
            RoundResult(
                name="First Round",
                series=first_round_series,
                advancers=first_advancers,
            )
        )

        # ----- Second Round (fixed pairing) -----
        # Pair winners of series 0+1 and series 2+3 (the natural fixed-bracket pairings)
        # Home ice goes to the team with the better original seed when available,
        # otherwise to the first team in the pair.
        seeds = conf_data.get("seeds", {})
        second_round_series: List[SeriesResult] = []
        second_advancers: List[str] = []

        pairs = [(0, 1), (2, 3)]
        for i, j in pairs:
            a = first_advancers[i]
            b = first_advancers[j]
            # Determine higher seed by original seed number (lower number = better)
            seed_a = seeds.get(a, 99)
            seed_b = seeds.get(b, 99)
            if seed_a <= seed_b:
                higher, lower = a, b
            else:
                higher, lower = b, a

            ser = simulate_best_of_7(higher, lower, elo_lookup, home_advantage, rng)
            second_round_series.append(ser)
            second_advancers.append(ser.winner)

        rounds.append(
            RoundResult(
                name="Second Round",
                series=second_round_series,
                advancers=second_advancers,
            )
        )

        # ----- Conference Finals -----
        a, b = second_advancers[0], second_advancers[1]
        seed_a = seeds.get(a, 99)
        seed_b = seeds.get(b, 99)
        if seed_a <= seed_b:
            higher, lower = a, b
        else:
            higher, lower = b, a

        conf_ser = simulate_best_of_7(higher, lower, elo_lookup, home_advantage, rng)
        finalists[conf] = conf_ser.winner
        rounds.append(
            RoundResult(
                name="Conference Finals",
                series=[conf_ser],
                advancers=[conf_ser.winner],
            )
        )

        conf_results[conf] = rounds

    # ----- Stanley Cup Final -----
    east = finalists.get("Eastern")
    west = finalists.get("Western")
    stanley_cup: Optional[SeriesResult] = None
    champion: Optional[str] = None

    if east and west:
        # Better regular-season Elo (or original seed) gets home ice in the Final
        # For simplicity we use Elo; a caller can override by swapping.
        if elo_lookup.get(east, 1500) >= elo_lookup.get(west, 1500):
            higher, lower = east, west
        else:
            higher, lower = west, east

        stanley_cup = simulate_best_of_7(
            higher, lower, elo_lookup, final_home_advantage or home_advantage, rng
        )
        champion = stanley_cup.winner

    team_reached = _build_reach_dict(conf_results, stanley_cup, champion)

    return PlayoffResult(
        conference_results=conf_results,
        stanley_cup_final=stanley_cup,
        champion=champion,
        team_reached=team_reached,
    )


def run_nhl_playoff_from_standings(
    standings: List[TeamStanding],
    elo_lookup: Optional[Dict[str, float]] = None,
    home_advantage: float = 55.0,
    rng: Optional[random.Random] = None,
    tiebreaker_fn: Optional[Callable] = None,
) -> PlayoffResult:
    """
    Convenience wrapper: standings → bracket → full playoff simulation.
    """
    if elo_lookup is None:
        elo_lookup = {t.team_id: t.elo for t in standings}

    bracket = seed_nhl_playoffs(standings, tiebreaker_fn=tiebreaker_fn)
    return simulate_nhl_playoffs(
        bracket=bracket,
        elo_lookup=elo_lookup,
        home_advantage=home_advantage,
        rng=rng,
    )

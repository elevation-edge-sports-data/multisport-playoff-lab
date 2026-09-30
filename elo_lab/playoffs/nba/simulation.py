"""
NBA single-game simulation, Play-In tournament, best-of-7 series,
and fixed 1–8 bracket progression.
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
from .seeding import seed_nba_playoffs


def simulate_nba_game(
    home_id: str,
    away_id: str,
    elo_lookup: Dict[str, float],
    home_advantage: float,
    rng: random.Random,
) -> GameResult:
    """Simulate one NBA game (play-in or playoff)."""
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
    higher_id has home-court advantage (hosts Games 1, 2, 5, 7).
    """
    home_schedule = [True, True, False, False, True, False, True]

    higher_wins = 0
    lower_wins = 0
    games: List[GameResult] = []

    for higher_is_home in home_schedule:
        if higher_wins >= 4 or lower_wins >= 4:
            break

        if higher_is_home:
            home, away = higher_id, lower_id
        else:
            home, away = lower_id, higher_id

        g = simulate_nba_game(home, away, elo_lookup, home_advantage, rng)
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


def run_play_in(
    play_in_teams: List[str],
    elo_lookup: Dict[str, float],
    home_advantage: float,
    rng: random.Random,
) -> tuple[str, str, List[GameResult]]:
    """
    Run the three-game Play-In tournament.

    play_in_teams must be ordered [7, 8, 9, 10].

    Returns
    -------
    seed7, seed8, list of the three GameResults
    """
    if len(play_in_teams) < 4:
        # Degenerate case – just promote the first two
        return (
            play_in_teams[0] if play_in_teams else "",
            play_in_teams[1] if len(play_in_teams) > 1 else "",
            [],
        )

    t7, t8, t9, t10 = play_in_teams

    # Game 1: 7 hosts 8  → winner = #7 seed
    g1 = simulate_nba_game(t7, t8, elo_lookup, home_advantage, rng)
    seed7 = g1.winner
    loser_78 = t8 if g1.winner == t7 else t7

    # Game 2: 9 hosts 10 → winner advances, loser eliminated
    g2 = simulate_nba_game(t9, t10, elo_lookup, home_advantage, rng)
    winner_910 = g2.winner

    # Game 3: loser of 7/8 hosts winner of 9/10 → winner = #8 seed
    g3 = simulate_nba_game(loser_78, winner_910, elo_lookup, home_advantage, rng)
    seed8 = g3.winner

    return seed7, seed8, [g1, g2, g3]


def _build_reach_dict(
    conf_results: Dict[str, List[RoundResult]],
    nba_finals: Optional[SeriesResult],
    champion: Optional[str],
) -> Dict[str, Dict[str, bool]]:
    """Build the team → {round: reached} map."""
    reach: Dict[str, Dict[str, bool]] = {}

    def mark(team: str, round_name: str):
        if not team:
            return
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
            for g in rnd.games:
                mark(g.home, rnd.name)
                mark(g.away, rnd.name)

    if nba_finals is not None:
        mark(nba_finals.higher_seed, "NBA Finals")
        mark(nba_finals.lower_seed, "NBA Finals")
        if champion:
            mark(champion, "Champion")

    return reach


def simulate_nba_playoffs(
    bracket: Dict[str, Dict],
    elo_lookup: Dict[str, float],
    home_advantage: float = 55.0,
    rng: Optional[random.Random] = None,
    finals_home_advantage: float = 0.0,
) -> PlayoffResult:
    """
    Run the full NBA postseason for one season (Play-In + fixed 1–8 bracket).
    """
    if rng is None:
        rng = random.Random()

    conf_results: Dict[str, List[RoundResult]] = {}
    finalists: Dict[str, str] = {}

    for conf, conf_data in bracket.items():
        auto = conf_data.get("auto", [])
        play_in = conf_data.get("play_in", [])
        seeds = conf_data.get("seeds", {})

        if len(auto) < 6:
            conf_results[conf] = []
            continue

        rounds: List[RoundResult] = []

        # ----- Play-In -----
        seed7, seed8, play_in_games = run_play_in(
            play_in, elo_lookup, home_advantage, rng
        )
        play_in_advancers = [seed7, seed8]
        # Also mark the four original play-in teams as having reached Play-In
        rounds.append(
            RoundResult(
                name="Play-In",
                games=play_in_games,
                advancers=play_in_advancers,
            )
        )

        # Final 1–8 seeds after Play-In
        final_seeds = auto + [seed7, seed8]  # indices 0-5 = seeds 1-6, 6=#7, 7=#8

        # ----- First Round (fixed pairings) -----
        # 1v8, 4v5, 3v6, 2v7
        first_round_pairings = [
            (0, 7),  # 1 vs 8
            (3, 4),  # 4 vs 5
            (2, 5),  # 3 vs 6
            (1, 6),  # 2 vs 7
        ]
        first_series: List[SeriesResult] = []
        first_advancers: List[str] = []

        for hi, lo in first_round_pairings:
            higher = final_seeds[hi]
            lower = final_seeds[lo]
            ser = simulate_best_of_7(higher, lower, elo_lookup, home_advantage, rng)
            first_series.append(ser)
            first_advancers.append(ser.winner)

        rounds.append(
            RoundResult(
                name="First Round",
                series=first_series,
                advancers=first_advancers,
            )
        )

        # ----- Conference Semifinals (fixed bracket) -----
        # Winners of (1v8) vs (4v5) and (2v7) vs (3v6)
        semi_pairings = [(0, 1), (3, 2)]  # indices into first_advancers
        semi_series: List[SeriesResult] = []
        semi_advancers: List[str] = []

        for i, j in semi_pairings:
            a = first_advancers[i]
            b = first_advancers[j]
            # Higher original seed gets home court
            seed_a = seeds.get(a, 99)
            seed_b = seeds.get(b, 99)
            if seed_a <= seed_b:
                higher, lower = a, b
            else:
                higher, lower = b, a

            ser = simulate_best_of_7(higher, lower, elo_lookup, home_advantage, rng)
            semi_series.append(ser)
            semi_advancers.append(ser.winner)

        rounds.append(
            RoundResult(
                name="Conference Semifinals",
                series=semi_series,
                advancers=semi_advancers,
            )
        )

        # ----- Conference Finals -----
        a, b = semi_advancers[0], semi_advancers[1]
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

    # ----- NBA Finals -----
    east = finalists.get("Eastern")
    west = finalists.get("Western")
    nba_finals: Optional[SeriesResult] = None
    champion: Optional[str] = None

    if east and west:
        # Better regular-season Elo gets home court in the Finals
        if elo_lookup.get(east, 1500) >= elo_lookup.get(west, 1500):
            higher, lower = east, west
        else:
            higher, lower = west, east

        nba_finals = simulate_best_of_7(
            higher, lower, elo_lookup, finals_home_advantage or home_advantage, rng
        )
        champion = nba_finals.winner

    team_reached = _build_reach_dict(conf_results, nba_finals, champion)

    return PlayoffResult(
        conference_results=conf_results,
        nba_finals=nba_finals,
        champion=champion,
        team_reached=team_reached,
    )


def run_nba_playoff_from_standings(
    standings: List[TeamStanding],
    elo_lookup: Optional[Dict[str, float]] = None,
    home_advantage: float = 55.0,
    rng: Optional[random.Random] = None,
    tiebreaker_fn: Optional[Callable] = None,
) -> PlayoffResult:
    """Convenience wrapper: standings → bracket → full playoff simulation."""
    if elo_lookup is None:
        elo_lookup = {t.team_id: t.elo for t in standings}

    bracket = seed_nba_playoffs(standings, tiebreaker_fn=tiebreaker_fn)
    return simulate_nba_playoffs(
        bracket=bracket,
        elo_lookup=elo_lookup,
        home_advantage=home_advantage,
        rng=rng,
    )

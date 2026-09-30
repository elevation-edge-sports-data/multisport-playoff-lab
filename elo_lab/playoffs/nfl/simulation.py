"""
NFL single-game simulation and full bracket progression with reseeding.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from elo_lab.engine import win_probability

from .models import GameResult, PlayoffResult, RoundResult, TeamStanding
from .seeding import seed_nfl_playoffs


def simulate_nfl_game(
    home_id: str,
    away_id: str,
    elo_lookup: Dict[str, float],
    home_advantage: float,
    rng: random.Random,
) -> GameResult:
    """Simulate one NFL playoff game."""
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


def reseed(teams: List[str], original_seed: Dict[str, int]) -> List[str]:
    """
    Re-order remaining teams so the highest original seed is first,
    then the next highest, etc. (NFL reseeding rule).
    """
    return sorted(teams, key=lambda t: original_seed.get(t, 99))


def _build_reach_dict(
    conf_results: Dict[str, List[RoundResult]],
    super_bowl: Optional[GameResult],
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
            # Also mark participants of the round
            for g in rnd.games:
                mark(g.home, rnd.name)
                mark(g.away, rnd.name)

    if super_bowl is not None:
        mark(super_bowl.home, "Super Bowl")
        mark(super_bowl.away, "Super Bowl")
        if champion:
            mark(champion, "Champion")

    return reach


def simulate_nfl_playoffs(
    seeds: Dict[str, List[str]],
    elo_lookup: Dict[str, float],
    home_advantage: float = 55.0,          # typical Elo home-field value; override from config
    rng: Optional[random.Random] = None,
    super_bowl_home_advantage: float = 0.0,  # neutral site by default
) -> PlayoffResult:
    """
    Run the full NFL playoff bracket for one season.

    Parameters
    ----------
    seeds : dict
        Output of seed_nfl_playoffs – {"AFC": [7 team ids], "NFC": [...]}
    elo_lookup : dict
        team_id → final regular-season Elo
    home_advantage : float
        Elo points added to the home team
    rng : random.Random
        Optional seeded RNG for reproducibility
    super_bowl_home_advantage : float
        Usually 0 (neutral site). Can be set to a small value or
        derived from regular-season record if desired.
    """
    if rng is None:
        rng = random.Random()

    # Build original-seed lookup for reseeding (seed 1 = 0, seed 2 = 1, ...)
    original_seed: Dict[str, int] = {}
    for conf, seed_list in seeds.items():
        for idx, team in enumerate(seed_list):
            original_seed[team] = idx

    conf_results: Dict[str, List[RoundResult]] = {}
    finalists: Dict[str, str] = {}

    for conf, seed_list in seeds.items():
        if len(seed_list) < 7:
            # Incomplete conference – skip gracefully
            conf_results[conf] = []
            continue

        remaining = list(seed_list)  # index 0 = #1 seed … index 6 = #7 seed
        rounds: List[RoundResult] = []

        # ----- Wild Card -----
        # #1 seed has a bye
        wc_games = [
            simulate_nfl_game(remaining[1], remaining[6], elo_lookup, home_advantage, rng),  # 2 vs 7
            simulate_nfl_game(remaining[2], remaining[5], elo_lookup, home_advantage, rng),  # 3 vs 6
            simulate_nfl_game(remaining[3], remaining[4], elo_lookup, home_advantage, rng),  # 4 vs 5
        ]
        wc_winners = [g.winner for g in wc_games]
        after_wc = [remaining[0]] + wc_winners
        rounds.append(RoundResult(name="Wild Card", games=wc_games, advancers=after_wc))

        # ----- Divisional (reseed) -----
        after_wc = reseed(after_wc, original_seed)
        div_games = [
            simulate_nfl_game(after_wc[0], after_wc[3], elo_lookup, home_advantage, rng),
            simulate_nfl_game(after_wc[1], after_wc[2], elo_lookup, home_advantage, rng),
        ]
        div_winners = [g.winner for g in div_games]
        rounds.append(RoundResult(name="Divisional", games=div_games, advancers=div_winners))

        # ----- Conference Championship (reseed) -----
        div_winners = reseed(div_winners, original_seed)
        conf_game = simulate_nfl_game(
            div_winners[0], div_winners[1], elo_lookup, home_advantage, rng
        )
        finalists[conf] = conf_game.winner
        rounds.append(
            RoundResult(name="Conference", games=[conf_game], advancers=[conf_game.winner])
        )

        conf_results[conf] = rounds

    # ----- Super Bowl -----
    afc = finalists.get("AFC")
    nfc = finalists.get("NFC")
    super_bowl: Optional[GameResult] = None
    champion: Optional[str] = None

    if afc and nfc:
        # Neutral site by default; caller can decide home team by regular-season record
        super_bowl = simulate_nfl_game(
            afc, nfc, elo_lookup, super_bowl_home_advantage, rng
        )
        champion = super_bowl.winner

    team_reached = _build_reach_dict(conf_results, super_bowl, champion)

    return PlayoffResult(
        conference_results=conf_results,
        super_bowl=super_bowl,
        champion=champion,
        team_reached=team_reached,
    )


def run_nfl_playoff_from_standings(
    standings: List[TeamStanding],
    elo_lookup: Optional[Dict[str, float]] = None,
    home_advantage: float = 55.0,
    rng: Optional[random.Random] = None,
    tiebreaker_fn: Optional[Callable] = None,
) -> PlayoffResult:
    """
    Convenience wrapper: standings → seeds → full playoff simulation.
    """
    if elo_lookup is None:
        elo_lookup = {t.team_id: t.elo for t in standings}

    seeds = seed_nfl_playoffs(standings, tiebreaker_fn=tiebreaker_fn)
    return simulate_nfl_playoffs(
        seeds=seeds,
        elo_lookup=elo_lookup,
        home_advantage=home_advantage,
        rng=rng,
    )

#!/usr/bin/env python3
"""
elo_lab/workflows/generate_default_sims.py

One-time (or CI) workflow that runs the default Monte Carlo simulation
for NHL, NBA and NFL and writes the results to data/precomputed/.

Usage (from repo root):

    python -m elo_lab.workflows.generate_default_sims

    # optional overrides
    python -m elo_lab.workflows.generate_default_sims --n-sims 300 --sports NHL NBA

    # or as a plain script
    python elo_lab/workflows/generate_default_sims.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Locate the repository root reliably whether we are run as:
#   python -m elo_lab.workflows.generate_default_sims
#   python elo_lab/workflows/generate_default_sims.py
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    """Walk upward from this file until we find a directory that contains
    both 'elo_lab' and 'app' (or 'data')."""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "elo_lab").is_dir() and (
            (candidate / "app").is_dir() or (candidate / "data").is_dir()
        ):
            return candidate
    # Fallback: three levels up from workflows/ (elo_lab/workflows → repo root)
    return here.parents[2]


ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit / project bootstrap (sets up paths the same way the dashboard does)
try:
    import bootstrap  # noqa: F401
except ImportError:
    pass

from app.services.simulation_service import run_simulation
from app.services.initial_ratings_service import (
    get_available_seasons,
    get_simulatable_seasons,
    get_seed_season,
    get_initial_ratings,
)


# ---------------------------------------------------------------------------
# Sport-aware default model config. The dashboard sliders read this.
# ---------------------------------------------------------------------------
def default_config(sport: str = "NFL") -> dict:
    """
    Locked public defaults from the regular-season k and home-advantage search.
    Elevation Edge is off. Home field is on the pregame pipeline, so the
    engine adds the stored value. Margin of victory stays off that pipeline.

      NHL  k=15, home ice 25
      NFL  k=50, home field 30
      NBA  k=30, home court 45
    """
    sport = (sport or "NFL").upper()
    if sport == "NHL":
        k, home = 15, 25
    elif sport == "NBA":
        k, home = 30, 45
    else:
        k, home = 50, 30
    return {
        "k": k,
        "pregame_pipeline": ["home_field"],
        "postgame_pipeline": [],
        "adjustments": {
            "home_field": {"enabled": True, "value": home},
            "margin_of_victory": {"enabled": True, "scale": 1.0},
            "elevation_edge": {"enabled": False, "value": 0.0},
        },
    }


def _schedule_path(sport: str) -> str:
    mapping = {
        "NHL": "data/nhl_games.csv",
        "NFL": "data/nfl_games.csv",
        "NBA": "data/nba_games.csv",
    }
    return mapping.get(sport, f"data/{sport.lower()}_games.csv")


def generate_one(sport: str, n_sims: int, out_dir: Path, rng_seed: int = 42) -> Path:
    print(f"\n{'=' * 60}")
    print(f"  Generating default simulation for {sport}  (n_sims={n_sims})")
    print(f"{'=' * 60}")

    config = default_config(sport)
    schedule_path = _schedule_path(sport)

    # Use the most recent available season as target
    seasons = get_simulatable_seasons(sport)
    season = seasons[-1] if seasons else None
    data_seed = get_seed_season(sport)

    # Default warm-up: previous season only (most recent completed year).
    # This keeps ratings anchored to the latest form and matches the dashboard default.
    from_options = []
    try:
        from app.services.initial_ratings_service import get_simulate_from_options
        from_options = get_simulate_from_options(sport, target_season=season)
    except Exception:
        from_options = []
    from_season = from_options[-1] if from_options else None

    print(f"  Target season   : {season}")
    print(f"  Simulate from   : {from_season}")
    print(f"  History through : before {season} (seed data from {data_seed})")
    print(f"  RNG seed        : {rng_seed}")
    print(f"  Config          : k={config['k']}, home={config['adjustments']['home_field']['value']}")
    print(f"  Mode            : warm-up on actual recent seasons → MC target")
    print(f"  inter_season_reg: 0.35 (milder so recent form survives)")

    # Mild record-based prior from the season before the warm-up window so
    # the first warm-up season does not start every team at pure 1500.
    initial_ratings = {}
    if from_season is not None:
        try:
            all_seasons = get_available_seasons(sport)
            if from_season in all_seasons:
                initial_ratings = get_initial_ratings(
                    sport,
                    season=from_season,  # uses previous season internally
                    rating_source="regular_season",
                    rating_basis="record",
                    apply_regression=True,
                    regression_strength=0.25,
                )
                print(f"  Prior ratings   : record prior before {from_season} "
                      f"({len(initial_ratings)} teams)")
        except Exception as e:
            print(f"  Prior ratings   : skipped ({e})")
            initial_ratings = {}

    t0 = time.perf_counter()
    results = run_simulation(
        config=config,
        n_sims=n_sims,
        initial_ratings=initial_ratings,
        sport=sport,
        season=season,
        from_season=from_season,
        seed=int(rng_seed),
        inter_season_regression=0.35,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s")

    # Light sanity checks
    summary = results.get("summary")
    playoff = results.get("playoff_probs", {})
    print(f"  Summary rows    : {len(summary) if summary is not None else 0}")
    print(f"  Playoff teams   : {len(playoff)}")
    if playoff:
        # Show top 6 by champion probability for quick visual check
        sample = next(iter(playoff.values()))
        champ_key = None
        for k in ("Champion", "Win Super Bowl", "Win Stanley Cup", "Win NBA Title"):
            if k in sample:
                champ_key = k
                break
        if champ_key is None and sample:
            champ_key = list(sample.keys())[-1]
        if champ_key:
            ranked = sorted(
                playoff.items(),
                key=lambda kv: kv[1].get(champ_key, 0),
                reverse=True,
            )
            print(f"  Top 6 by {champ_key}:")
            for team, probs in ranked[:6]:
                print(f"    {team:4s}  {probs.get(champ_key, 0):.3f}")

    out_path = out_dir / f"{sport}_default.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {out_path}  ({size_kb:.1f} KB)")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate default precomputed simulations for Multisport Elo Lab"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=250,
        help="Number of Monte Carlo seasons (default 250 – good balance of speed vs stability)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible Monte Carlo draws (default 42)",
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["NHL", "NBA", "NFL"],
        choices=["NHL", "NBA", "NFL"],
        help="Sports to generate (default: all three)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the .pkl files (default: <repo>/data/precomputed)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (ROOT / "data" / "precomputed")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    written = []
    for sport in args.sports:
        path = generate_one(sport, args.n_sims, out_dir, rng_seed=args.seed)
        written.append(path)

    print(f"\nDone. Generated {len(written)} file(s):")
    for p in written:
        print(f"  • {p}")


if __name__ == "__main__":
    main()

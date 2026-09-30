# Architecture

MultiSport Elo Lab rates NHL, NBA, and NFL teams with one Elo engine, then simulates a target season and the playoffs that follow. The Streamlit app is a shell around that engine.

## Kernel

`run_game(home_elo, away_elo, context, config)` in `elo_lab/engine.py` plays one game.

Win probability is the standard Elo logistic, exported as `elo_lab.engine.win_probability`:

```text
1 / (1 + 10 ** ((away_elo - home_elo) / 400))
```

Equal ratings come out to 0.5. In Monte Carlo the home team wins when a random draw falls under that probability. A played game uses `context["actual"]`, or whichever of `home_score` and `away_score` is higher.

The rating update is `update_ratings` (the same function is also available as `update_elo`):

```text
home += K * (actual - p_home) * multiplier
away -= K * (actual - p_home) * multiplier
```

`actual` is 1 when the home team won and 0 otherwise. With the multiplier at 1, the two ratings move by the same amount in opposite directions, so their sum stays put. K defaults to 20.

Three adjustments live in `elo_lab/adjustments.py`. Each one is a function of the game state. The engine calls the function registered in `ADJUSTMENTS` for every name on `pregame_pipeline` or `postgame_pipeline` whose `enabled` flag is true.

- **Home field** runs before the probability. It adds a fixed number of Elo points to the home team (`adjustments.home_field.value`, default 55). The sidebar label is home ice, home court, or home field depending on the sport. The config key stays `home_field`.
- **Margin of victory** runs after the score is known. On, the multiplier is `log(margin + 1) * scale`, clamped to 0.25–3. Off, the multiplier stays 1 and the update depends only on who won.
- **Elevation Edge** also runs before the probability. Venue feet come from `app/metadata`.

```text
boost = scale * max(0, home_ft - away_ft) / 1000
```

That boost is added to home Elo. When the away building sits higher, the boost is 0.

Standings, brackets, and the dashboard stay outside the engine.

## Config

The sidebar writes a plain dict and hands it to the workflows. The usual shape is `k` plus `adjustments`, and each adjustment is a small map of `enabled` plus a `value` or a `scale`. Named presets live in `elo_lab/configuration/model_configs.py`. Sport defaults (schedule path, wins versus points, starting Elo) live in `sport_configs.py`.

Those modules store parameters. The formulas stay in the engine and in the adjustment functions.

## Workflows

`elo_lab/workflows/` turns a schedule into a season.

- **Warm-up.** `warm_elo_from_actual_history` replays real scores from the Simulate-from season through the season before the target. Between seasons, ratings can be pulled toward 1500.
- **Monte Carlo target season.** `simulate_season` samples the season the dashboard is projecting. Games that already have a winner keep that result. Games without a decisive score are drawn from the win probability.
- **Playoff hook.** `simulate_with_playoffs` takes the standings and ending Elo from that same pass and runs the sport's bracket.
- **Lock played games.** A row with a winner is identical in every simulation. Blank scores are sampled. Ties stay unlocked, because an engine result is a home win or a home loss.
- **Live slate.** `python -m elo_lab.workflows.live_slate` writes finished scores into the season CSV so the lock can see them. NFL comes from nflverse, NHL from `api-web.nhle.com`, and NBA from today's NBA CDN scoreboard with the ESPN scoreboard as fallback. The NBA path needs no API key.

Backtests, parameter search, and `generate_default_sims` call the same `run_game`.

## Playoffs

Each league has a package under `elo_lab/playoffs/`. Games and series still use Elo plus home advantage. The bracket is what changes.

- **NFL.** Four division winners and three wild cards per conference. The top seed has a bye. After the wild-card round, and again after the divisional round, the remaining teams reseed: the highest seed left hosts the lowest. The Super Bowl is a neutral site. Every game is single elimination.
- **NHL.** Fixed bracket, no reseed. Each round is a best-of-7 with a 2-2-1-1-1 home pattern.
- **NBA.** The top six teams in each conference are in. Seeds 7 through 10 play a play-in: 7 hosts 8, 9 hosts 10, then the 7/8 loser hosts the 9/10 winner. The 7/8 winner is the 7 seed and the last play-in winner is the 8 seed. From there the bracket is fixed (1v8, 2v7, 3v6, 4v5) and every series is a best-of-7.

## App

`streamlit run app/dashboard.py` opens four tabs: Playoff Projections, Regular Season Projections, Live Slate, and Model Comparison.

Changing the sport loads `data/precomputed/{SPORT}_default.pkl` when that file is present, so a default run is on screen immediately. Run Simulation replaces it. Rebuild the pickles with `python -m elo_lab.workflows.generate_default_sims`.

Once any results exist, Export Results (.xlsx) downloads one workbook: config, simulation summary, achievement and playoff probabilities, Elo ratings, and evaluation metrics when a comparison has been run.

## Adding an adjustment

A new adjustment is a function plus a config key.

1. Add `def <name>(state)` to `elo_lab/adjustments.py`. Read parameters from `state["config"]["adjustments"]["<name>"]`, return the state, and register the function in `ADJUSTMENTS`. Leave `run_game` alone.
2. Add that key under `adjustments` (`enabled`, plus the numbers the function reads) and put `<name>` on `pregame_pipeline` or `postgame_pipeline`.

The engine calls the function by that name.

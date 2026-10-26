# MultiSport Elo Lab

Elo ratings and Monte Carlo playoff simulation for the NHL, NBA, and NFL, with a Streamlit dashboard for backtests and parameter checks.

## Setup and run

```bash
git clone https://github.com/elevation-edge-sports-data/multisport-playoff-lab.git
cd multisport-playoff-lab
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The hosted dashboard is [https://multisport-elo-lab.streamlit.app/](https://multisport-elo-lab.streamlit.app/).

Optional local prep:

```bash
python -m elo_lab.workflows.generate_default_sims
python -m elo_lab.workflows.live_slate --sport all
```

## Elo snapshot

Export current NFL, NHL, and NBA ratings and the home advantage the default config applies:

```bash
python -m elo_lab.workflows.export_elo_snapshot
```

That writes `outputs/elo_snapshot.json`. The same file is available as the **Export Elo snapshot (.json)** download next to Export Results in the dashboard.

Each sport block contains:

- `teams`: a mapping of team abbreviation to that team’s current Elo rating
- `home_advantage`: a single Elo-point value for the sport (added to the home team’s rating). It is not team-specific.
- `source`: where the ratings came from (warm-up, and locked games when the target slate has scores)

Top-level fields sit beside `sports`. Older readers can ignore them.

- `as_of`: calendar date of the snapshot
- `generated_at`: ISO 8601 UTC time of this export
- `version`: package version, such as `v15.3`
- `n_teams`: team counts keyed by sport (`nfl`, `nhl`, `nba`)
- `lock_status`: per sport, the target `season`, `n_locked`, `n_games`, and `status` (`upcoming`, `in_progress`, or `final`)

multisport-live-win-probability does not automatically import this package yet. Copy `outputs/elo_snapshot.json` onto that repo's `data/elo_snapshot.json`. For NFL, NHL, and NBA, a missing moneyline uses that file as the pregame prior: the sport's `home_advantage` is added to the home rating, and no further boost is applied.

## Home advantage contract with live win probability

The snapshot's per-sport home_advantage (Elo points) is the sole additive home advantage used for non-moneyline priors in the live win-probability repo. That repo adds the snapshot value to the home rating before converting to a probability and skips the adjustment on neutral sites. Its SportConfig.home_advantage margin-unit field is zero for NFL, NHL, and NBA so the snapshot value is not double-counted. Moneyline priors already embed home advantage and receive no additive adjustment in either repo.

## Calibrated defaults

| Sport | k | Home advantage | MOV scale | Elevation |
| --- | --- | --- | --- | --- |
| NHL | 15 | 25 | 0.75 | off |
| NFL | 50 | 35 | 0.5 | off |
| NBA | 22 | 45 | 0.5 | off |

These were fit on non-neutral regular-season games.

Release history is in [CHANGELOG.md](CHANGELOG.md).

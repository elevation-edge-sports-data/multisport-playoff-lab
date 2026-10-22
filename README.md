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

multisport-live-win-probability does not automatically import this package yet. Copy `outputs/elo_snapshot.json` onto that repo's `data/elo_snapshot.json`. For NFL, NHL, and NBA, a missing moneyline uses that file as the pregame prior: the sport's `home_advantage` is added to the home rating, and no further boost is applied.

## Calibrated defaults

| Sport | k | Home advantage | Elevation |
| --- | --- | --- | --- |
| NHL | 15 | 25 | off |
| NFL | 50 | 30 | off |
| NBA | 30 | 45 | off |

These were fit on non-neutral regular-season games.

Release history is in [CHANGELOG.md](CHANGELOG.md).

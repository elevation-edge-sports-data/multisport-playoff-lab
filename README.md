# MultiSport Elo Lab

A modular, configuration-driven Elo rating and simulation platform for **NHL**, **NBA**, and **NFL**.

Supports historical backtesting, Monte Carlo regular-season + full playoff-bracket simulation, parameter optimization, and interactive model evaluation through a Streamlit dashboard.

**Live dashboard**: [https://multisport-elo-lab.streamlit.app/](https://multisport-elo-lab.streamlit.app/)

**Current version: 15.0**

---

## Highlights (Version 15.0)

- **Live slate**: pull completed scores into the season CSV, then lock those games in Monte Carlo
  - NFL: nflverse schedules · NHL: api-web.nhle.com · NBA: BALLDONTLIE (`BALLDONTLIE_API_KEY`)
- Sidebar **Refresh live slate** plus a **Live Slate** tab (locked vs remaining games)
- CLI: `python -m elo_lab.workflows.live_slate --sport NFL --season 2026`

### From Version 14.0

- **Lock completed games** in the target-season Monte Carlo: scored games stay fixed across sims; blank games are still sampled
- Sidebar lock line, e.g. `NFL 2026 · upcoming · 0/272 games locked` or `NFL 2026 · in progress · 16/272 games locked`

### From Version 13.2

- **Add 2027 NBA schedule**: upcoming regular-season slate (schedule-only; scores blank until games are played)
- Raw vs processed layout under `data/nba/` with extract/process helpers for future releases
- Sidebar shows the full Elo history window (e.g. 2024–2026 → simulate 2027), not only the start year

### From Version 13.1

- **Playoff spiral polish**: logos aligned to abbreviation rays and placed just beyond each wedge tip; fixed-square figures (no elliptical logo drift on resize); no spin; default radial metric = **Make Playoffs**
- **Regular Season Projections**: distribution of wins/points box plot at the top of the tab
- Playoff Projections focuses on odds table, spirals, and path bars (summary table removed)

### From Version 13.0

- **Playoff Odds table** (MoneyPuck-inspired): logo + abbreviation, blue probability shading; sortable main table
- **Playoff spirals**: one per conference; metric dropdown; leader fills full radius; team colors + logos; full path on hover
- **Playoff path bars** with Color 1…5 round palette

### From Version 12.3

- **Numerical simulation progress bar**: updates at 10% steps (0% … 100%) with an explicit numeric indicator while simulations run
- **Faster simulation**: playoffs reuse the primary Monte Carlo standings/Elo (one pass instead of two; redundant re-sim removed)
- Cleanup of unused tab modules (`configuration.py`, `elo_ratings.py`)

### From Version 12.2

- **Match sidebar / default-sim settings to the precomputed defaults** (sport-specific k and home advantage, Simulate from, seed)
- **Playoff Outlook** Team column shows team logos
- **Regular Season Projections** defaults to top Western (NHL/NBA) or AFC (NFL) teams by first-round / wild-card probability
- Sport-specific public defaults, 2-season warm-up, milder inter-season regression, record-based prior
- `run_simulation` accepts and forwards `seed` for reproducible Monte Carlo runs

### From Version 12.1

- **Regular Season Projections**: observed prior-season path + simulated target path (wins / points / rank)
- **Historical param eval** on Run: score current parameters on the last completed season (accuracy, log loss, Brier)
- **Simulation count** option **25** (100 remains default)
- **Random seed** for reproducible Monte Carlo
- Upcoming NFL schedule support (`VisTm` / `HomeTm`)

### From Version 12.0

- **Simulate upcoming season** for **NFL 2026**, **NHL 2026–27**, and **NBA 2026–27**
  - Full regular-season slate + playoff bracket Monte Carlo for seasons not yet played
- **Simulate from** control: choose the start of the history window used before the target season (warm-up runs through the year before the target)
- **Three-tab dashboard**: Playoff Projections (default landing), Regular Season Projections, Model Comparison
- **Model configuration panel** refinements: Grid Search hierarchy, pure binary Margin of Victory, regression to mean on by default, clearer parameter tooltips
- **Expanded baselines** in Model Comparison: Always Home, Constant Home Rate, Coin Flip (p=0.5), plus Log5
- **Export Results (.xlsx)**: multi-sheet workbook (Config, Simulation Summary, probabilities, ratings, evaluation metrics)
- Full multi-sport support with consistent ordering: **NHL · NBA · NFL**
- **Full playoff-bracket simulation** for all three sports
  - NFL: Wild Card → Divisional → Conference → Super Bowl (reseeding)
  - NHL: Fixed bracket, best-of-7 series through Stanley Cup
  - NBA: Play-In tournament + fixed 1–8 bracket, best-of-7 series
- Instant default simulations — results appear immediately when you switch sports

---

## Dashboard Tabs

1. **Playoff Projections** (default landing)  
   MoneyPuck-inspired odds table, dual-conference playoff spirals (default metric: Make Playoffs), and path bars. Full playoff outlook (reach each round / win championship).
2. **Regular Season Projections**  
   Win/point distributions and Elo / win trajectories for the target season.
3. **Live Slate**  
   Locked vs remaining games after a score refresh. Use **Refresh live slate** in the sidebar, then **Run Simulation**.
4. **Model Comparison**  
   Accuracy, Log Loss, Brier Score, calibration plots, residual diagnostics, expanded baselines (Always Home, Constant Home Rate, Coin Flip), Log5, and grid-search landscape.

---

## Core Architecture (stable since v5)

The engine is the only component that orchestrates Elo execution. Everything else is configuration, reusable transformations, evaluation utilities, or workflow orchestration.

Public engine API:

```python
run_game(home_elo, away_elo, context, config)
```

Model behavior is controlled entirely through declarative configuration. Adjustments (Home Advantage, Margin of Victory, Elevation Edge) are implemented as stateless transformations in a pregame/postgame pipeline.

### Repository Structure

```
elo_lab/
├── engine/          # Execution engine (game lifecycle + orchestration)
├── adjustments/     # Stateless transformations
├── configuration/   # Model and sport configurations
├── evaluation/      # Metrics and diagnostics
├── playoffs/        # Sport-specific playoff brackets (nfl / nhl / nba)
└── workflows/       # Backtesting, simulation, optimization, default-sim generation

app/
├── dashboard.py     # Streamlit entry point
├── metadata/        # Teams, venues, elevation data (NHL / NBA / NFL)
├── services/        # Simulation, evaluation, Elo evolution, export, etc.
└── tabs/            # Individual dashboard tab implementations
```

---

## Supported Sports

| Sport | Status          | Notes                                                                 |
|-------|-----------------|-----------------------------------------------------------------------|
| NHL   | Fully supported | Points system + OT handling, Home-Ice Advantage, Stanley Cup bracket  |
| NBA   | Fully supported | Home-Court Advantage, Play-In + 1–8 bracket                           |
| NFL   | Fully supported | Wins-based outcomes, Home-Field Advantage, reseeding playoff bracket  |

All sports share the same engine and dashboard. Sport-specific logic (scoring rules, home-advantage labels, achievement columns, playoff format, team colors, venue elevations) is driven by configuration and metadata.

---

## Key Features in Detail

### Adjustments
- **Home Advantage** — configurable Elo boost (sport-specific labeling)
- **Margin of Victory** — pure binary: on scales the Elo update by score differential; off uses win/loss only
- **Elevation Edge** — continuous: `elev_value × max(0, home_ft − away_ft) / 1000`

### Initial Elo & Regression
- Rating source fixed to previous-season playoffs (not user-facing)
- Rating basis prefers existing Elo, with record-derived fallback when needed
- Regression to mean defaults on; strength adjustable under Customize Parameters
- **Simulate from** chooses the first season included in the warm-up window

### Playoff Simulation
- NFL, NHL, and NBA each have a dedicated package under `elo_lab/playoffs/`
- Monte Carlo seasons feed final standings + Elo into the appropriate bracket
- Dashboard surfaces probability of reaching each round and winning the championship

### Instant Default Simulations
Switching sports in the sidebar immediately loads a precomputed default run so you see results without waiting. Override anytime with **Run Simulation**.

Generate or refresh the defaults:

```bash
python -m elo_lab.workflows.generate_default_sims
```

### Full Results Export
A global **Download Full Results (.xlsx)** button appears as soon as simulation results exist (defaults or custom runs). One click produces a multi-sheet Excel workbook containing:

- Config (model parameters + run metadata)
- Simulation Summary
- Achievement + Playoff probabilities
- Elo Ratings
- Evaluation metrics (when available)

### Parameter Optimization
Users can selectively optimize any subset of enabled adjustments. The system runs a grid search and surfaces the best combination via an interactive heatmap (Grid Search Landscape) plus a top-10 table.

### Evaluation & Diagnostics
- Accuracy / Log Loss / Brier Score
- Brier score decomposition (Reliability, Resolution, Uncertainty) + ECE
- Reliability diagrams
- Log5 baseline comparison
- Residual diagnostics
- Home-win-rate and coin-flip baselines with explicit lift columns

---

## Quick Start

```bash
git clone https://github.com/elevation-edge-sports-data/multisport-elo-lab.git
cd multisport-elo-lab
pip install -r requirements.txt

# optional – generate instant-default simulation files
python -m elo_lab.workflows.generate_default_sims

# optional – lock completed games into the current season CSVs
python -m elo_lab.workflows.live_slate --sport all

streamlit run app/dashboard.py
```

---

## Version History Summary

| Version | Focus |
|---------|-------|
| 15.0    | Live slate: nflverse / NHL official web API / balldontlie |
| 14.0    | Lock completed games in target-season Monte Carlo |
| 5       | Canonical engine + declarative transformation pipeline |
| 6       | Interactive Streamlit dashboard |
| 7       | Parameter optimization + original Elevation Edge (binned) |
| 8.0–8.2 | Multi-sport (NHL) + achievement probabilities + calibration + NFL playoff-logic fix |
| 9       | Multiseason support + full NBA integration + advanced initial-Elo controls |
| 10      | Continuous Elevation Edge + Log5 baseline + residual diagnostics |
| 11.0    | NFL playoff-bracket simulation |
| 11.1    | NHL + NBA playoff-bracket simulation + multi-sport dispatch |
| 11.2    | Instant default simulations on sport change + generator workflow |
| 11.3    | One-click full results export (multi-sheet Excel) |
| 11.4    | Team logos in Elo Ratings, Trajectory, and Simulation |
| 11.5    | Warm-up Elo from recent actual results before target-season MC (replaces prior-only start) |
| 12.0    | Simulate upcoming season (NFL 2026, NHL 2026–27), three-tab dashboard |
| 13.2    | Add 2027 NBA schedule |
| 12.2    | Match default sim settings to precomputed defaults; seed; logos in Playoff Outlook |
| 12.1    | Regular season trajectories, historical param eval, random seed, schedule normalize fixes |

See [CHANGELOG.md](CHANGELOG.md) for the complete history.

---

## Notes

- **Simulate upcoming season**: NFL 2026, NHL 2026–27, and NBA 2026–27 schedules are included. Upcoming seasons use blank scores; Monte Carlo fills outcomes from warmed Elo.
- This project is under active development.
- Elevation data uses real stadium/arena elevations.
- Precomputed default simulations live in `data/precomputed/` (gitignored; regenerate with the command above).

---

## License & Contact

Built by [Zach Sajevic](https://github.com/elevation-edge-sports-data)  
Sports Analytics Engineer | Predictive Modeling & Interactive Data Products

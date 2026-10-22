"""
Generalized parameter optimization workflow.

Accepts a base config + a list of adjustments to optimize over.
Only varies parameters for the adjustments listed in `optimize_for`.
"""

import itertools
import os
from typing import Any

import pandas as pd

from elo_lab.workflows.run_backtest import run_backtest
from elo_lab.evaluation.metrics import accuracy, log_loss, brier

os.makedirs("outputs", exist_ok=True)


# Default parameter grids (can be moved to configuration/ later)
# elevation_edge.value = Elo points per 1000 ft of elevation advantage
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "home_field": {
        "value": [20, 35, 50, 65, 80],
    },
    "margin_of_victory": {
        "scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    },
    "elevation_edge": {
        "value": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],  # Elo pts per 1000 ft
    },
}


def optimize_parameters_for_config(
    base_config: dict,
    optimize_for: list[str],
    k_values: list[int] | None = None,
    param_grids: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Run grid search only over the parameters belonging to the adjustments
    listed in `optimize_for`.

    Parameters
    ----------
    base_config : dict
        The model configuration with enabled adjustments already set.
    optimize_for : list[str]
        Subset of adjustment names to optimize (e.g. ["home_field", "elevation_edge"]).
    k_values : list[int], optional
        Values to try for the global k factor. If None, uses a default range.
    param_grids : dict, optional
        Custom parameter grids. Falls back to DEFAULT_PARAM_GRIDS.

    Returns
    -------
    best_config : dict
        The configuration with the best found parameters.
    results_df : pd.DataFrame
        All combinations evaluated with their metrics.
    """
    if not optimize_for:
        return base_config, pd.DataFrame()

    grids = param_grids or DEFAULT_PARAM_GRIDS
    k_values = k_values or [10, 15, 20, 25, 30]

    # Build the dimensions we will grid over
    param_names = []
    param_values = []

    # Always include k as a global parameter
    param_names.append("k")
    param_values.append(k_values)

    for adj in optimize_for:
        if adj not in grids:
            continue
        for param_name, values in grids[adj].items():
            param_names.append(f"{adj}.{param_name}")
            param_values.append(values)

    results = []
    best_score = -float("inf")
    best_config = base_config.copy()

    for combination in itertools.product(*param_values):
        config = base_config.copy()
        config.setdefault("adjustments", {})

        # Apply the current combination
        for i, name in enumerate(param_names):
            val = combination[i]
            if name == "k":
                config["k"] = val
            else:
                adj_name, param_name = name.split(".", 1)
                config["adjustments"].setdefault(adj_name, {})
                config["adjustments"][adj_name][param_name] = val

        # === Consistent abbreviated labeling ===
        abbr_map = {
            "k": "k",
            "home_field.value": "hfa",
            "margin_of_victory.scale": "mov",
            "elevation_edge.value": "elev",
        }

        label_parts = []
        for name, val in zip(param_names, combination):
            if name in abbr_map:
                label_parts.append(f"{abbr_map[name]}{val}")
            else:
                # Fallback for any future parameters
                clean_name = name.split(".")[-1] if "." in name else name
                label_parts.append(f"{clean_name}{val}")

        label = "_".join(label_parts)
        # === End of consistent labeling ===

        print(f"Running optimized backtest: {label}")

        try:
            # run_backtest drops neutral-site games before scoring.
            df = run_backtest(config, label)

            p = df["p_home"].astype(float).values
            y = df["actual"].astype(int).values

            acc = float(accuracy(p, y))
            ll = float(pd.Series(log_loss(p, y)).mean())
            br = float(pd.Series(brier(p, y)).mean())

            results.append({
                "label": label,
                "k": config.get("k"),
                **{name: combination[i] for i, name in enumerate(param_names) if name != "k"},
                "accuracy": acc,
                "log_loss": ll,
                "brier": br,
            })

            if acc > best_score:
                best_score = acc
                best_config = config.copy()

        except Exception as e:
            print(f"  Skipped {label} due to error: {e}")
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
        results_df.to_csv("outputs/parameter_search.csv", index=False)

    return best_config, results_df

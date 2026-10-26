"""Normal-approximation intervals for Monte Carlo playoff probabilities.

Playoff accumulation stores one proportion per team and round (successes / n).
It does not keep the per-sim 0/1 indicators, so the interval is

    SE = sqrt(p * (1 - p) / n)
    [max(0, p - 1.96*SE), min(1, p + 1.96*SE)]

Bracket sims reuse regular-season draws and are capped at MAX_PLAYOFF_SIMS.
"""

from __future__ import annotations

from typing import Optional

# Bracket probabilities use at most this many regular-season draws.
MAX_PLAYOFF_SIMS = 500
Z_95 = 1.96
_EN_DASH = "\u2013"


def normal_interval(p: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% interval for a binomial proportion. p is in [0, 1].

    Endpoints are clipped to [0, 1]. n <= 0 returns the clipped point twice.
    """
    p = min(1.0, max(0.0, float(p)))
    if n is None or int(n) <= 0:
        return p, p
    se = (p * (1.0 - p) / int(n)) ** 0.5
    lo = max(0.0, p - float(z) * se)
    hi = min(1.0, p + float(z) * se)
    return lo, hi


def format_percent(pct: float, n: Optional[int] = None, z: float = Z_95) -> str:
    """Format a percent point estimate, with a 95% range when n is known.

    pct is on the 0–100 scale already shown in the table (42.3 → 42.3%).
    The range is p ± z*SE on that displayed p, clipped to [0, 100], so the
    primary number stays the same value the table sorts on.
    """
    point = round(float(pct), 1)
    n_int = _positive_int(n)
    if n_int is None:
        return f"{point:.1f}%"
    p = min(1.0, max(0.0, point / 100.0))
    lo, hi = normal_interval(p, n_int, z=z)
    lo_pct = min(point, round(lo * 100.0, 1))
    hi_pct = max(point, round(hi * 100.0, 1))
    return f"{point:.1f}% ({lo_pct:.1f}{_EN_DASH}{hi_pct:.1f})"


def playoff_sample_size(results: object) -> Optional[int]:
    """Monte Carlo n used for playoff proportions.

    Prefers an explicit n_playoff_sims. Older result dicts only have the
    regular-season draws, which the bracket pass caps at MAX_PLAYOFF_SIMS.
    """
    if not isinstance(results, dict):
        return None
    explicit = _positive_int(results.get("n_playoff_sims"))
    if explicit is not None:
        return explicit
    n = _raw_sim_count(results.get("raw_sim_results"))
    if n is None:
        n = _positive_int(results.get("n_sims"))
    if n is None:
        return None
    return min(n, MAX_PLAYOFF_SIMS)


def _positive_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = int(value)
    if n <= 0:
        return None
    return n


def _raw_sim_count(raw: object) -> Optional[int]:
    columns = getattr(raw, "columns", None)
    if columns is None or "sim_id" not in columns:
        return None
    try:
        n = int(raw["sim_id"].nunique())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None

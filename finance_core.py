"""Financial and energy analysis utilities for the Pendulum Array investor app."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760.0


def gamma_safe(x: float) -> float:
    """Return the gamma function while guarding against domain errors."""
    try:
        if x <= 0:
            # Fallback to a small positive offset to keep the Weibull expectations finite.
            x = 1e-6
        return float(math.gamma(x))
    except (OverflowError, ValueError):
        return 0.0


def weibull_ev3(k: float, c: float) -> float:
    """Expectation of v^3 for a Weibull(k, c) distribution."""
    if k <= 0 or c <= 0:
        return 0.0
    return (c ** 3) * gamma_safe(1.0 + 3.0 / k)


def expected_power_weibull(
    rho: float,
    area_m2: float,
    cp: float,
    chain_eff: float,
    availability: float,
    k: float,
    c: float,
) -> float:
    """Average mechanical-to-electric power for a Weibull site."""
    ev3 = weibull_ev3(k, c)
    prefactor = 0.5 * rho * area_m2 * cp * chain_eff * availability
    return prefactor * ev3


def expected_power_from_hist(
    speeds: np.ndarray,
    weights: np.ndarray,
    rho: float,
    area_m2: float,
    cp: float,
    chain_eff: float,
    availability: float,
) -> float:
    """Average power from an empirical histogram of wind speeds."""
    if speeds.size == 0 or weights.size == 0:
        return 0.0
    weights = np.asarray(weights, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        return 0.0
    ev3 = float(np.sum(np.power(speeds, 3.0) * weights) / total)
    prefactor = 0.5 * rho * area_m2 * cp * chain_eff * availability
    return prefactor * ev3


def annual_energy_kwh(power_watts: float) -> float:
    """Convert a continuous average power in watts to yearly energy in kWh."""
    if power_watts <= 0:
        return 0.0
    return power_watts * HOURS_PER_YEAR / 1000.0


def crf(discount_rate: float, years: int) -> float:
    """Capital recovery factor."""
    if years <= 0:
        return 0.0
    if discount_rate <= 0:
        return 1.0 / years
    r = discount_rate
    return (r * (1 + r) ** years) / ((1 + r) ** years - 1)


def lcoe_usd_per_kwh(capex_usd: float, opex_pct: float, power_watts: float, CRF: float) -> float:
    """Levelised cost of energy in $/kWh for a single array."""
    annual_cost = capex_usd * (CRF + opex_pct)
    annual_energy = annual_energy_kwh(power_watts)
    if annual_energy <= 0:
        return float("nan")
    return annual_cost / annual_energy


DEFAULT_SCENARIOS: List[Tuple[str, float]] = [
    ("Campus microgrid", 50.0),
    ("Village mini-grid", 150.0),
    ("Utility pilot", 500.0),
]


def build_lcoe_table_weibull(
    weibull_cases: Sequence[Tuple[str, float, float]],
    capex_list: Sequence[float],
    rho: float,
    A: float,
    Cp: float,
    chain_eff: float,
    availability: float,
    opex_pct: float,
    CRF: float,
) -> pd.DataFrame:
    records = []
    for label, k, c in weibull_cases:
        power_w = expected_power_weibull(rho, A, Cp, chain_eff, availability, k, c)
        energy_mwh = annual_energy_kwh(power_w) / 1000.0
        power_kw = power_w / 1000.0
        for capex in capex_list:
            lcoe = lcoe_usd_per_kwh(capex, opex_pct, power_w, CRF)
            records.append(
                {
                    "Case": label,
                    "k": round(k, 3),
                    "c (m/s)": round(c, 2),
                    "CapEx (USD)": round(capex, 0),
                    "Avg Power (kW)": round(power_kw, 3),
                    "Annual Energy (MWh)": round(energy_mwh, 3),
                    "LCOE ($/kWh)": round(lcoe, 3) if not math.isnan(lcoe) else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def size_arrays(target_kw: float, array_power_watts: float) -> int:
    """Number of arrays required to meet a target average kW."""
    if target_kw <= 0 or array_power_watts <= 0:
        return 0
    needed = math.ceil((target_kw * 1000.0) / array_power_watts)
    return max(needed, 0)


__all__ = [
    "gamma_safe",
    "weibull_ev3",
    "expected_power_weibull",
    "expected_power_from_hist",
    "annual_energy_kwh",
    "crf",
    "lcoe_usd_per_kwh",
    "DEFAULT_SCENARIOS",
    "build_lcoe_table_weibull",
    "size_arrays",
]

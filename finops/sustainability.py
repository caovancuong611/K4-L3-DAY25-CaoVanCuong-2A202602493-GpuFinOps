"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

# Round-trip latency (ms) from NimbusAI's users (US East) — illustrative.
REGION_LATENCY_MS = {
    "us-east-1": 10,
    "us-west-2": 70,
    "europe-north1": 110,
    "europe-central2": 105,
    "us-east-wa": 65,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def best_regions(weight_cost: float = 0.5) -> dict:
    """Optimal region per criterion: cheapest $/kWh, cleanest gCO2/kWh, and balanced.

    Balanced = lowest weighted sum of min-max normalised price and carbon.
    """
    def norm(d):
        lo, hi = min(d.values()), max(d.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in d.items()}
    p, c = norm(REGION_PRICE_KWH), norm(REGION_CARBON)
    score = {r: weight_cost * p[r] + (1 - weight_cost) * c[r] for r in REGION_CARBON}
    return {
        "cheapest": min(REGION_PRICE_KWH, key=REGION_PRICE_KWH.get),
        "cleanest": min(REGION_CARBON, key=REGION_CARBON.get),
        "balanced": min(score, key=score.get),
    }


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0

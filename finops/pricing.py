"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                   gpu_type: str | None = None, job_days: float | None = None,
                   prices: dict | None = None) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (used when no `prices` are given):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Extension 1: pass `prices` (a price_catalog row) + `gpu_type` + `job_days` to use
    the cost-based policy in `recommend_tier_detailed` (per-GPU spot interruption
    rate, reserved billed 24/7, 1yr vs 3yr term amortised over the job's lifetime).
    """
    if prices is not None:
        return recommend_tier_detailed(hours_per_day, interruptible, prices,
                                       gpu_type=gpu_type, job_days=job_days)["tier"]
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# Per-hour spot reclaim probability by GPU type — illustrative 2026. Flagship
# training parts sit in long-term pools and are reclaimed less often than the
# commodity inference cards that everyone bursts onto.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04, "MI300X": 0.04,
    "A100": 0.05, "A10G": 0.10, "L4": 0.08,
}
RESERVED_TERM_DAYS = {"1yr": 365, "3yr": 3 * 365}


def reserved_effective_hr(rate_hr: float, term_days: float, job_days: float) -> float:
    """$/GPU-hr of a reservation amortised over the days the job actually needs it.

    A commitment is paid for the whole term; if the job ends earlier the unused
    remainder is sunk cost, so the effective rate rises by term/job_days.
    """
    if job_days <= 0:
        return float("inf")
    return rate_hr * max(1.0, term_days / job_days)


def recommend_tier_detailed(hours_per_day: float, interruptible: bool, prices: dict,
                            gpu_type: str | None = None, job_days: float | None = None,
                            interrupt_rate: float | None = None) -> dict:
    """Cost-based tier choice (Extension 1). Returns tier, term and $/GPU-day per option.

    - on_demand : pay only the hours used.
    - spot      : only if interruptible; effective cost from `spot_checkpoint_cost`
                  with the GPU type's interruption rate (rework + checkpoint overhead).
    - reserved  : billed 24h/day whether used or not; 1yr and 3yr terms amortised
                  over `job_days` (a 3yr deal for a 2-year service wastes a year).
    The cheapest option wins. `tier` is one of spot / reserved / on_demand; for
    reserved, `term` says which commitment ('1yr' or '3yr').
    """
    hpd = max(0.0, min(24.0, hours_per_day))
    days = job_days if job_days is not None else RESERVED_TERM_DAYS["3yr"]
    od = float(prices["on_demand_hr"])
    rate = interrupt_rate if interrupt_rate is not None else SPOT_INTERRUPT_RATE.get(gpu_type or "", 0.05)

    options = {"on_demand": hpd * od}
    if interruptible and hpd > 0:
        sim = spot_checkpoint_cost(hpd, float(prices["spot_hr"]), od, interrupt_rate=rate)
        options["spot"] = sim["spot_cost"]
    for term, col in (("1yr", "reserved_1yr_hr"), ("3yr", "reserved_3yr_hr")):
        eff = reserved_effective_hr(float(prices[col]), RESERVED_TERM_DAYS[term], days)
        options[f"reserved_{term}"] = 24.0 * eff

    best = min(options, key=options.get)
    tier, term = (("reserved", best.split("_")[1]) if best.startswith("reserved") else (best, None))
    return {"tier": tier, "term": term, "daily_cost_per_gpu": round(options[best], 4),
            "options": {k: round(v, 4) for k, v in options.items()}, "interrupt_rate": rate}


def cache_break_even_reads(write_cost: float = 1.25, read_discount: float = 0.10,
                           storage_cost: float = 0.0) -> float:
    """Reads per cached prefix needed before caching pays for itself (Extension 3).

    All costs are multiples of the model's normal input price for the prefix:
    writing costs `write_cost` x (Anthropic: 1.25x) instead of 1x, storing it costs
    `storage_cost` x for its TTL (Gemini bills storage per hour), and each read
    costs `read_discount` x instead of 1x. Each read saves (1 - read_discount).
    """
    extra = max(0.0, write_cost - 1.0) + max(0.0, storage_cost)
    per_read_saving = 1.0 - read_discount
    if per_read_saving <= 0:
        return float("inf")
    return extra / per_read_saving


def cache_is_worth_it(avg_cache_reads: float, write_cost: float = 1.25,
                      read_discount: float = 0.10, storage_cost: float = 0.0) -> bool:
    """True only when savings from re-reads exceed the write premium + storage."""
    return avg_cache_reads > cache_break_even_reads(write_cost, read_discount, storage_cost)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }

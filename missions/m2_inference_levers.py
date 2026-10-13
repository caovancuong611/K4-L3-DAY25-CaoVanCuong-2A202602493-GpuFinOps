"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Extension 3: prompt-cache savings only count where `pricing.cache_is_worth_it`
says the re-reads cover the write premium + storage (per team x tier x hour).
Extension 4: reasoning budget — $ and Wh split by is_reasoning, a routing rule
and traffic-cap scenarios.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Ext 3 cache economics. Write = 1.25x input (Anthropic 5-min write); storage is
# billed per 1M cached tokens per hour (Gemini-style), here for a 1h TTL.
CACHE_WRITE_COST = 1.25
CACHE_READ_DISCOUNT = 0.10
CACHE_STORAGE_PER_M_HR = {"small": 1.00, "large": 4.50}
CACHE_TTL_HR = 1.0

# Ext 4: generator applies a 6x output "reasoning tax"; the routing rule undoes it.
REASONING_OUTPUT_MULT = 6


def storage_multiple(tier: str) -> float:
    """Cache storage for one TTL expressed as a multiple of the tier's input price."""
    return CACHE_STORAGE_PER_M_HR[tier] * CACHE_TTL_HR / MODEL_PRICES[tier][0]


def cache_analysis(rows: list[dict]) -> dict:
    """Group requests that share a cached prefix (team x tier x hour, 1h TTL).

    Each group writes the prefix once and the other requests read it, so
    reads = n - 1. Returns per-group decisions + per-tier break-even vs actual.
    """
    groups = defaultdict(list)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            groups[(r["team"], r["route_tier"], r["ts"][11:13])].append(r)
    decisions, per_tier = {}, defaultdict(list)
    for key, reqs in groups.items():
        tier = key[1]
        reads = len(reqs) - 1
        worth = pricing.cache_is_worth_it(reads, CACHE_WRITE_COST, CACHE_READ_DISCOUNT, storage_multiple(tier))
        prefix = sum(int(num(r["cached_input_tokens"])) for r in reqs) / len(reqs)
        overhead = (prefix / 1e6) * MODEL_PRICES[tier][0] * ((CACHE_WRITE_COST - 1.0) + storage_multiple(tier))
        decisions[key] = {"worth": worth, "reads": reads, "overhead_usd": overhead if worth else 0.0}
        per_tier[tier].append((reads, worth))
    tiers = {}
    for tier, vals in per_tier.items():
        tiers[tier] = {
            "break_even_reads": round(pricing.cache_break_even_reads(
                CACHE_WRITE_COST, CACHE_READ_DISCOUNT, storage_multiple(tier)), 2),
            "avg_reads": round(sum(v[0] for v in vals) / len(vals), 1),
            "groups": len(vals), "groups_worth": sum(1 for v in vals if v[1]),
        }
    return {"decisions": decisions, "tiers": tiers}


def _req_cost(r: dict, tier: str, cached: int, batch: bool) -> float:
    pin, pout = MODEL_PRICES[tier]
    return pricing.request_cost(int(num(r["input_tokens"])), int(num(r["output_tokens"])),
                                pin, pout, cached_in=cached, batch=batch)


def reasoning_budget(rows: list[dict], decisions: dict) -> dict:
    """Ext 4: split optimized $ and Wh by is_reasoning; routing rule + cap scenarios."""
    split = {True: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0}, False: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0}}
    rule_usd = rule_wh = 0.0       # savings if reasoning only stays on large-tier (hard) tasks
    per_req_saving = []            # ($, Wh) saved by downgrading each reasoning request
    for r in rows:
        reasoning = bool(int(num(r["is_reasoning"])))
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cost = _opt_cost(r, decisions)
        wh = sustainability.wh_per_query(inp + out, is_reasoning=reasoning)
        s = split[reasoning]
        s["n"] += 1; s["usd"] += cost; s["wh"] += wh; s["tokens"] += inp + out
        if reasoning:
            plain = dict(r, output_tokens=str(out // REASONING_OUTPUT_MULT), is_reasoning="0")
            d_usd = cost - _opt_cost(plain, decisions)
            d_wh = wh - sustainability.wh_per_query(inp + out // REASONING_OUTPUT_MULT)
            per_req_saving.append((r["route_tier"], d_usd, d_wh))
            if r["route_tier"] == "small":
                rule_usd += d_usd; rule_wh += d_wh

    n_all = split[True]["n"] + split[False]["n"]
    usd_all = split[True]["usd"] + split[False]["usd"]
    wh_all = split[True]["wh"] + split[False]["wh"]
    share = split[True]["n"] / n_all if n_all else 0.0
    # cap scenarios: downgrade the cheapest-to-lose (small-tier first) reasoning requests
    ordered = sorted(per_req_saving, key=lambda x: (x[0] != "small", -x[2]))
    caps = {}
    for cap in (0.10, 0.05, 0.02):
        drop = max(0, split[True]["n"] - int(cap * n_all))
        caps[cap] = {"usd": sum(x[1] for x in ordered[:drop]), "wh": sum(x[2] for x in ordered[:drop])}
    avg_wh = {k: (v["wh"] / v["n"] if v["n"] else 0.0) for k, v in split.items()}
    return {
        "traffic_share": share,
        "cost_share": split[True]["usd"] / usd_all if usd_all else 0.0,
        "energy_share": split[True]["wh"] / wh_all if wh_all else 0.0,
        "usd": {"reasoning": split[True]["usd"], "plain": split[False]["usd"]},
        "wh": {"reasoning": split[True]["wh"], "plain": split[False]["wh"]},
        "wh_per_query": {"reasoning": avg_wh[True], "plain": avg_wh[False]},
        "rule_small_tier": {"usd": rule_usd, "wh": rule_wh,
                            "requests": sum(1 for x in per_req_saving if x[0] == "small")},
        "caps": caps,
    }


def _opt_cost(r: dict, decisions: dict) -> float:
    key = (r["team"], r["route_tier"], r["ts"][11:13])
    use_cache = decisions.get(key, {}).get("worth", False)
    cached = int(num(r["cached_input_tokens"])) if use_cache else 0
    return _req_cost(r, r["route_tier"], cached, bool(int(num(r["is_batch"]))))


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    ca = cache_analysis(rows)
    decisions = ca["decisions"]
    cache_overhead = sum(d["overhead_usd"] for d in decisions.values())

    base_cost = cascade_cost = naive_cache_cost = cache_cost = 0.0
    opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        base_cost += _req_cost(r, "large", 0, False)
        # step 1: cascade (route_tier)
        cascade_cost += _req_cost(r, r["route_tier"], 0, False)
        # step 2: + prompt caching (naive = every cached token; net = only where worth it)
        naive_cache_cost += _req_cost(r, r["route_tier"], cached, False)
        key = (r["team"], r["route_tier"], r["ts"][11:13])
        cache_cost += _req_cost(r, r["route_tier"], cached if decisions.get(key, {}).get("worth") else 0, False)
        # step 3: + batch API  ->  OPTIMIZED
        opt_cost += _opt_cost(r, decisions)
    cache_cost += cache_overhead
    opt_cost += cache_overhead

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    levers = {
        "cascade": base_cost - cascade_cost,
        "cache": cascade_cost - cache_cost,
        "batch": cache_cost - opt_cost,
    }
    rb = reasoning_budget(rows, decisions)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("lever breakdown ($/day, applied in order):")
        for k, v in levers.items():
            print(f"  {k:8} ${v:,.2f}  ({v / base_cost * 100:.1f}% of baseline)")

        print("\n-- Ext 3: cache_is_worth_it (team x tier x hour groups, 1h TTL) --")
        print(f"{'tier':7}{'break-even reads':>18}{'avg reads/group':>17}{'groups worth':>15}")
        for tier, t in sorted(ca["tiers"].items()):
            print(f"{tier:7}{t['break_even_reads']:>18}{t['avg_reads']:>17}{t['groups_worth']:>9}/{t['groups']}")
        print(f"cache saving: naive ${cascade_cost - naive_cache_cost:,.2f}/day -> net ${levers['cache']:,.2f}/day "
              f"(write+storage overhead ${cache_overhead:,.2f}/day)")

        print("\n-- Ext 4: reasoning budget --")
        print(f"reasoning = {rb['traffic_share']:.1%} of requests but {rb['cost_share']:.1%} of $ "
              f"and {rb['energy_share']:.1%} of Wh")
        print(f"Wh/query: reasoning {rb['wh_per_query']['reasoning']:.2f} vs plain {rb['wh_per_query']['plain']:.3f} "
              f"(~{rb['wh_per_query']['reasoning'] / rb['wh_per_query']['plain']:.0f}x)")
        rs = rb["rule_small_tier"]
        print(f"rule 'reasoning only on large-tier (hard) tasks': {rs['requests']} requests downgraded -> "
              f"save ${rs['usd']:,.2f}/day, {rs['wh']:,.0f} Wh/day")
        for cap, v in rb["caps"].items():
            print(f"  cap reasoning at {cap:.0%} of traffic -> save ${v['usd']:,.2f}/day, {v['wh']:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "levers_daily": {k: round(v, 2) for k, v in levers.items()},
        "cache": {"tiers": ca["tiers"], "naive_saving_daily": round(cascade_cost - naive_cache_cost, 2),
                  "net_saving_daily": round(levers["cache"], 2), "overhead_daily": round(cache_overhead, 2)},
        "reasoning": rb,
    }


if __name__ == "__main__":
    run()

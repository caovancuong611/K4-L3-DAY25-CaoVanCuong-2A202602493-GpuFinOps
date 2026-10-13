"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Extension 1: tiers come from the cost-based `pricing.recommend_tier_detailed`
(per-GPU spot interruption rate, reserved billed 24/7, 1yr vs 3yr amortised over
the job's lifetime). The original simple policy is still computed as `legacy_*`
so the before/after savings can be compared.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
# workloads.csv `days` is the observation window for always-on services (30 = the
# whole month). Assume such a core service lives 3 years when sizing a commitment
# (same horizon the original policy implied); M3 prints 1yr/2yr sensitivity.
SERVICE_LIFETIME_DAYS = 3 * 365


def job_lifetime_days(job: dict, service_lifetime_days: float = SERVICE_LIFETIME_DAYS) -> float:
    days = num(job["days"])
    return service_lifetime_days if days >= DAYS else days


def legacy_cost(job: dict, cat: dict) -> tuple[str, float]:
    """Original policy: simple rule + reserved_3yr billed only for hours used."""
    c = cat[job["gpu_type"]]
    hpd = num(job["hours_per_day"])
    gpu_hours = hpd * DAYS * int(num(job["num_gpus"]))
    od = num(c["on_demand_hr"])
    tier = pricing.recommend_tier(hpd, bool(int(num(job["interruptible"]))))
    if tier == "spot":
        return tier, pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
    if tier == "reserved":
        return tier, gpu_hours * num(c["reserved_3yr_hr"])
    return tier, gpu_hours * od


def tier_matrix(cat: dict, gpu_types=("H100", "A100", "A10G", "L4"),
                duties=(4, 8, 12, 16, 20, 24), lifetime_days: float = SERVICE_LIFETIME_DAYS) -> list[dict]:
    """GPU type x duty cycle x interruptible -> recommended tier (+ reserved term)."""
    rows = []
    for g in gpu_types:
        for intr in (True, False):
            row = {"gpu_type": g, "interruptible": intr}
            for h in duties:
                d = pricing.recommend_tier_detailed(h, intr, cat[g], gpu_type=g, job_days=lifetime_days)
                row[h] = d["tier"] + (f"-{d['term']}" if d["term"] else "")
            rows.append(row)
    return rows


def run(verbose: bool = True, service_lifetime_days: float = SERVICE_LIFETIME_DAYS) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        on_demand_cost = gpu_hours * num(c["on_demand_hr"])

        lifetime = job_lifetime_days(j, service_lifetime_days)
        d = pricing.recommend_tier_detailed(hpd, interruptible, c, gpu_type=gtype, job_days=lifetime)
        opt_cost = d["daily_cost_per_gpu"] * ngpu * DAYS
        legacy_tier, legacy_opt = legacy_cost(j, cat)

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        legacy_monthly += legacy_opt
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": d["tier"], "term": d["term"],
                     "legacy_tier": legacy_tier, "lifetime_days": lifetime,
                     "interrupt_rate": d["interrupt_rate"] if interruptible else None,
                     "on_demand": round(on_demand_cost), "legacy": round(legacy_opt),
                     "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    legacy_pct = (on_demand_monthly - legacy_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"service lifetime assumed for always-on jobs: {service_lifetime_days:.0f} days")
        print(f"{'job':18}{'gpu':7}{'legacy':10}{'new tier':14}{'on-demand':>11}{'legacy':>10}{'new':>10}")
        for r in recs:
            new = r["tier"] + (f"-{r['term']}" if r["term"] else "")
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['legacy_tier']:10}{new:14}"
                  f"${r['on_demand']:>10,}${r['legacy']:>9,}${r['optimized']:>9,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"legacy policy would claim ${legacy_monthly:,.0f} ({legacy_pct:.1f}% saved) - "
              f"but bills reserved only for hours used and always assumes a 3yr term")

        print("\n-- Ext 1: sensitivity to service lifetime --")
        for life in (365, 730, 1095):
            r = run(verbose=False, service_lifetime_days=life)
            print(f"  lifetime {life:>4} d -> optimized ${r['optimized_monthly']:,}  ({r['savings_pct']}% saved)")

        print("\n-- Ext 1: tier matrix (GPU x hours/day x interruptible) --")
        duties = (4, 8, 12, 16, 20, 24)
        print(f"{'gpu':7}{'intr':6}" + "".join(f"{str(h)+'h':>14}" for h in duties))
        for row in tier_matrix(cat, duties=duties, lifetime_days=service_lifetime_days):
            print(f"{row['gpu_type']:7}{'yes' if row['interruptible'] else 'no':6}"
                  + "".join(f"{row[h]:>14}" for h in duties))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "legacy_monthly": round(legacy_monthly), "legacy_savings_pct": round(legacy_pct, 1),
            "service_lifetime_days": service_lifetime_days}


if __name__ == "__main__":
    run()

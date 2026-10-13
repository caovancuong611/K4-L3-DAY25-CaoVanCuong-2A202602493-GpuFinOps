"""M6 (Extension 5) — Carbon-aware scheduling of interruptible jobs (deck §11).

Run: python missions/m6_carbon_aware.py

Interruptible jobs (training, eval, sandbox) are not latency-sensitive, so they
can follow the cheapest / cleanest grid. Latency-sensitive inference stays near
users in us-east-1.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability as sus

DAYS = 30
HOME_REGION = "us-east-1"
POWER_FACTOR = 0.85   # average board power vs TDP while busy (telemetry: 0.5 + 0.5*util)
PUE = 1.2             # datacenter overhead (cooling, power conversion)


def job_kwh(job: dict, cat: dict) -> float:
    """Monthly facility energy of a job: GPU-hours x TDP x power factor x PUE."""
    gpu_hours = num(job["hours_per_day"]) * DAYS * int(num(job["num_gpus"]))
    return gpu_hours * num(cat[job["gpu_type"]]["watts"]) / 1000.0 * POWER_FACTOR * PUE


def run(verbose: bool = True) -> dict:
    cat = catalog_by_type()
    jobs = [j for j in load_csv("workloads.csv") if int(num(j["interruptible"])) == 1]
    best = sus.best_regions()
    clean = best["cleanest"]

    per_job = []
    for j in jobs:
        kwh = job_kwh(j, cat)
        per_job.append({"job_id": j["job_id"], "kwh": kwh,
                        "home_g": sus.carbon_g(kwh * 1000, HOME_REGION),
                        "clean_g": sus.carbon_g(kwh * 1000, clean)})
    total_kwh = sum(p["kwh"] for p in per_job)

    regions = []
    for r in sus.REGION_CARBON:
        regions.append({"region": r, "usd_per_kwh": sus.REGION_PRICE_KWH[r],
                        "gco2_per_kwh": sus.REGION_CARBON[r], "latency_ms": sus.REGION_LATENCY_MS[r],
                        "energy_usd": sus.energy_cost_usd(total_kwh * 1000, r),
                        "carbon_kg": sus.carbon_g(total_kwh * 1000, r) / 1000.0})
    home = next(x for x in regions if x["region"] == HOME_REGION)
    for x in regions:
        x["carbon_saved_pct"] = (1 - x["carbon_kg"] / home["carbon_kg"]) * 100 if home["carbon_kg"] else 0.0
        x["usd_saved"] = home["energy_usd"] - x["energy_usd"]

    saved_g = sum(p["home_g"] - p["clean_g"] for p in per_job)
    home_g = sum(p["home_g"] for p in per_job)

    if verbose:
        print("== M6 Carbon-aware Scheduling (Ext 5) ==")
        print(f"interruptible jobs: {len(jobs)}  energy {total_kwh:,.0f} kWh/month (PUE {PUE}, power factor {POWER_FACTOR})")
        print(f"{'job':18}{'kWh/mo':>9}{'kgCO2 ' + HOME_REGION:>18}{'kgCO2 ' + clean:>22}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['kwh']:>9,.0f}{p['home_g'] / 1000:>18,.1f}{p['clean_g'] / 1000:>22,.1f}")
        print(f"moving all to {clean}: save {saved_g / 1000:,.0f} kgCO2e/month "
              f"({saved_g / home_g * 100 if home_g else 0:.0f}% less)\n")
        print(f"{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'RTT ms':>8}{'energy $':>10}{'kgCO2':>8}{'CO2 -%':>8}")
        for x in regions:
            print(f"{x['region']:17}{x['usd_per_kwh']:>7}{x['gco2_per_kwh']:>10}{x['latency_ms']:>8}"
                  f"{x['energy_usd']:>10,.0f}{x['carbon_kg']:>8,.0f}{x['carbon_saved_pct']:>8.0f}")
        print(f"\ncheapest $: {best['cheapest']} | cleanest CO2: {best['cleanest']} | balanced: {best['balanced']}")
        print("latency: interruptible batch work tolerates the +100ms to europe-north1; "
              "user-facing inference stays in us-east-1")

    return {"jobs": per_job, "total_kwh": round(total_kwh, 1), "regions": regions, "best": best,
            "carbon_saved_kg": round(saved_g / 1000, 1),
            "carbon_saved_pct": round(saved_g / home_g * 100, 1) if home_g else 0.0}


if __name__ == "__main__":
    run()

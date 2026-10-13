"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

# Ext 2 right-sizing targets: a replacement GPU must serve the observed demand at
# no more than these utilisations (headroom for bursts / batch growth).
TARGET_MBU = 0.60
TARGET_MFU = 0.40
HOURS_PER_MONTH = 24 * 30
MEMORY_BOUND_WORKLOADS = ("infer", "embed")   # LLM decode / embedding: HBM-bandwidth bound


def catalog_unit_costs(cat: dict) -> list[dict]:
    """$/GB-VRAM and $/TB/s per GPU-hr — the unit prices that matter for memory-bound work."""
    out = []
    for g, c in cat.items():
        od = num(c["on_demand_hr"])
        out.append({"gpu_type": g, "on_demand_hr": od, "hbm_gb": num(c["hbm_gb"]),
                    "peak_bw_tbs": num(c["peak_bw_tbs"]),
                    "usd_per_gb": round(od / num(c["hbm_gb"]), 4),
                    "usd_per_tbs": round(od / num(c["peak_bw_tbs"]), 3)})
    return sorted(out, key=lambda x: x["usd_per_tbs"])


def right_size(demand: dict, cat: dict) -> dict:
    """Cheapest catalog GPU that fits the observed demand (Extension 2).

    Fit = HBM >= peak VRAM used, peak_bw * TARGET_MBU >= achieved BW, and
    peak FP16 * TARGET_MFU >= achieved TFLOPs. Cheapest $/GPU-hr alone is not
    enough: a GPU with too little bandwidth makes decode slower (tokens/s scale
    with TB/s), and one with too little HBM cannot hold weights + KV cache.
    """
    cur = demand["gpu_type"]
    fits = []
    for g, c in cat.items():
        if (num(c["hbm_gb"]) >= demand["peak_mem_gb"]
                and num(c["peak_bw_tbs"]) * TARGET_MBU >= demand["bw_tbs"]
                and num(c["peak_tflops_fp16"]) * TARGET_MFU >= demand["tflops"]):
            fits.append(g)
    best = min(fits, key=lambda g: num(cat[g]["on_demand_hr"])) if fits else cur
    cheapest = min(cat, key=lambda g: num(cat[g]["on_demand_hr"]))
    delta = num(cat[cur]["on_demand_hr"]) - num(cat[best]["on_demand_hr"])
    new_mbu = demand["bw_tbs"] / num(cat[best]["peak_bw_tbs"])
    why = []
    if cheapest not in fits:
        c = cat[cheapest]
        if num(c["hbm_gb"]) < demand["peak_mem_gb"]:
            why.append(f"{cheapest} HBM {num(c['hbm_gb']):.0f}GB < {demand['peak_mem_gb']:.0f}GB")
        if num(c["peak_bw_tbs"]) * TARGET_MBU < demand["bw_tbs"]:
            why.append(f"{cheapest} BW {num(c['peak_bw_tbs'])}TB/s too low (MBU would be "
                       f"{demand['bw_tbs'] / num(c['peak_bw_tbs']):.1%})")
        if num(c["peak_tflops_fp16"]) * TARGET_MFU < demand["tflops"]:
            why.append(f"{cheapest} FP16 too low")
    return {"gpu_id": demand["gpu_id"], "current": cur, "proposed": best,
            "current_mbu": demand["mbu"], "proposed_mbu": round(new_mbu, 3),
            "savings_pct": round(delta / num(cat[cur]["on_demand_hr"]) * 100, 1) if delta > 0 else 0.0,
            "monthly_savings": round(max(0.0, delta) * HOURS_PER_MONTH, 2),
            "why_not_cheapest": "; ".join(why) or "-"}


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "workload": None, "tflops": [], "bw": [], "mem": []})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["workload"] = r["workload"]
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1
        else:  # demand while active (for right-sizing)
            a["tflops"].append(num(r["achieved_tflops"]))
            a["bw"].append(num(r["achieved_bw_tbs"]))
            a["mem"].append(num(r["mem_used_gb"]))

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Ext 2: right-size memory-bound GPUs (+ util-lies) with a roofline/VRAM fit check
    lie_ids = {l["gpu_id"] for l in lies}
    rightsizing = []
    for gid, a in agg.items():
        if a["workload"] in MEMORY_BOUND_WORKLOADS or gid in lie_ids:
            s = next(x for x in summary if x["gpu_id"] == gid)
            demand = {"gpu_id": gid, "gpu_type": a["type"], "mbu": s["mbu"],
                      "tflops": sum(a["tflops"]) / len(a["tflops"]),
                      "bw_tbs": sum(a["bw"]) / len(a["bw"]), "peak_mem_gb": max(a["mem"])}
            rightsizing.append(right_size(demand, cat))
    rightsize_monthly = sum(r["monthly_savings"] for r in rightsizing)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Ext 2: unit prices (sorted by $/TB/s) --")
        print(f"{'GPU':8}{'$/hr':>7}{'HBM GB':>8}{'TB/s':>7}{'$/GB-VRAM':>11}{'$/TB/s':>8}")
        for u in catalog_unit_costs(cat):
            print(f"{u['gpu_type']:8}{u['on_demand_hr']:>7}{u['hbm_gb']:>8.0f}{u['peak_bw_tbs']:>7}"
                  f"{u['usd_per_gb']:>11}{u['usd_per_tbs']:>8}")
        print(f"\n-- Ext 2: right-sizing (fit: HBM >= peak used, MBU <= {TARGET_MBU:.0%}, MFU <= {TARGET_MFU:.0%}) --")
        print(f"{'GPU':12}{'current':8}{'proposed':9}{'MBU now':>8}{'MBU new':>8}{'save%':>7}{'$/month':>9}  why not cheapest")
        for r in rightsizing:
            print(f"{r['gpu_id']:12}{r['current']:8}{r['proposed']:9}{r['current_mbu']:>8}{r['proposed_mbu']:>8}"
                  f"{r['savings_pct']:>7}{r['monthly_savings']:>9,.0f}  {r['why_not_cheapest']}")
        print(f"right-sizing all candidates saves ${rightsize_monthly:,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsizing": rightsizing, "rightsize_monthly": round(rightsize_monthly, 2),
            "unit_costs": catalog_unit_costs(cat)}


if __name__ == "__main__":
    run()

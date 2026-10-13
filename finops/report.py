"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 extra_sections: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report.

    `extra_sections` are markdown blocks (analysis, extensions) appended before the footer.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | % of baseline |",
        "|---|---|---|",
    ]
    for name, amount in levers.items():
        share = amount / baseline_usd * 100.0 if baseline_usd > 0 else 0.0
        lines.append(f"| {name} | ${amount:,.0f} | {share:.1f}% |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
        ]
        if "cheapest_region" in sustainability:
            lines += [
                f"- Cleanest region (best for carbon): {sustainability.get('best_region', 'n/a')}",
                f"- Cheapest electricity region: {sustainability['cheapest_region']}",
                f"- Balanced (50/50 cost + carbon): {sustainability.get('balanced_region', 'n/a')}",
            ]
        else:
            lines.append(f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}")
    for block in extra_sections or []:
        lines += ["", block.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline_usd: float | None = None) -> str:
    """Write the savings chart PNG. Returns the path. No-op if matplotlib absent.

    With `baseline_usd` it is a true waterfall: baseline -> each lever -> optimized.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    surface, ink, muted, total_c, lever_c = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    if baseline_usd is None:
        ax.bar(names, vals, color=lever_c)
        ax.set_ylabel("Savings (USD / month)")
    else:
        labels = ["Baseline"] + names + ["Optimized"]
        level, bottoms, heights, colors = baseline_usd, [0.0], [baseline_usd], [total_c]
        for v in vals:
            level -= v
            bottoms.append(level); heights.append(v); colors.append(lever_c)
        bottoms.append(0.0); heights.append(level); colors.append(total_c)
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors, width=0.62,
                      edgecolor=surface, linewidth=2)
        for k, (b, h) in enumerate(zip(bottoms, heights)):
            txt = f"${h:,.0f}" if k in (0, len(labels) - 1) else f"-${h:,.0f}"
            ax.text(bars[k].get_x() + bars[k].get_width() / 2, b + h + baseline_usd * 0.012, txt,
                    ha="center", va="bottom", fontsize=9, color=ink)
        for k in range(len(labels) - 1):  # connector lines between steps
            y = bottoms[k] if 0 < k else heights[0]   # running level after step k
            ax.plot([k + 0.31, k + 0.69], [y, y], color=muted, linewidth=1)
        pct = (baseline_usd - level) / baseline_usd * 100 if baseline_usd else 0.0
        ax.set_ylabel("USD / month", color=muted)
        ax.set_ylim(0, baseline_usd * 1.1)
        ax.set_title(f"NimbusAI monthly GPU spend: baseline to optimized ({pct:.0f}% saved)",
                     color=ink, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)
    return path

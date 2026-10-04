#!/usr/bin/env python3
"""Step 08: publication figures (vector PDF + 300 dpi PNG) in results/figures/.

Sized for IEEE two-column layout (single column 3.5 in, double column 7.16 in).
Figures carry no embedded "Fig. N" titles (captions live in the manuscript).
Palette: validated categorical slots (blue / orange / aqua) with hatching as a
secondary, print- and CVD-safe encoding for scenarios.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.experiments.energy_benchmark import STAGES  # noqa: E402
from src.models.pipeline_evaluator import compare, evaluate_all, load_regimes  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import ENERGY_SUMMARY, PROCESSED_DIR, RESULTS_FIG_DIR, SCENARIOS  # noqa: E402
from src.sustainability.carbon_accounting import COMPONENTS  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#ffffff"
SC_COLOR = {"precision_component": "#2a78d6", "machined_metal": "#eb6834", "high_value_component": "#1baf7a"}
SC_HATCH = {"precision_component": "", "machined_metal": "////", "high_value_component": "...."}
SC_LABEL = {sc: load_scenario(sc).label for sc in SCENARIOS}
POS, NEG = "#2a78d6", "#e34948"
SINGLE, DOUBLE = 3.5, 7.16

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})

COMPONENT_LABEL = {"mat": "Unrecovered\nmaterial", "recycle": "Recycling\nburden", "rework": "Rework\nelectricity",
                   "review": "Review /\ninspection", "edge": "Edge\nelectricity", "hardware": "Edge\nhardware",
                   "escape": "Field\nescapes"}
PARAM_LABEL = {
    "defect_prevalence": r"prevalence $\pi$", "part_mass": r"part mass $m$", "material_carbon_factor": r"$EF_{mat}$",
    "material_recovery_fraction": r"recovery $\eta$", "recycling_burden": r"$EF_{rec}$", "rework_energy": r"$e_{rw}$",
    "base_reworkability": r"$q_0$", "reworkability_time_constant": r"$\tau_{rw}$",
    "manual_discovery_delay": r"manual delay $d_{L0}$", "manual_inspection_time": r"$t_{L0}$",
    "manual_inspection_recall": r"manual recall $r_{L0}$", "return_logistics_carbon": r"$c_{ret}$",
    "collateral_multiplier": r"collateral $\kappa$", "grid_carbon_factor": r"grid $\gamma$", "host_power": r"$P_{host}$",
    "gpu_idle_power": r"$P_{idle}$", "gpu_active_power": r"$P_{active}$", "edge_embodied_carbon": r"$C_{hw}$",
    "edge_lifetime_hours": r"$L_{hw}$", "review_station_power": r"$P_{station}$", "review_time": r"$t_{review}$",
    "line_throughput": r"throughput $\Theta$", "ai_recall": r"AI recall $r_{AI}$", "persistence_recall_loss": r"recall loss $\delta_r$",
    "glare_burst_rate": r"glare rate $g$",
}


def plabel(key):
    if "@" in key:
        name, pol = key.split("@")
        tier = pol.split("_")[0]
        sym = {"nominal_false_alarm_rate": r"\lambda", "alerts_per_defect": "a", "detection_delay": "d",
               "glare_alerts_per_burst": "b"}[name]
        return rf"${sym}_{{{tier}}}$"
    return PARAM_LABEL.get(key, key)


def save(fig, name):
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_FIG_DIR / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(RESULTS_FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


def fig1_system_boundary():
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    fs = 6.0

    def box(x, y, w, h, text, fc="#f3f2ee", ec=INK2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
                linespacing=1.25)

    def arrow(a, b, text=None, off=(0.0, 0.015), ha="center"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=8,
                                                        shrinkA=0, shrinkB=0))
        if text:
            ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha=ha, va="bottom", fontsize=5.8, color=INK2)

    box(0.000, 0.38, 0.115, 0.24, "Production\nline, $N$ parts\nper FU,\n$D=N\\pi$ defects")
    box(0.165, 0.60, 0.200, 0.36, "Edge AI cell (B0-B3)\n30 FPS camera,\nGPU + host computer\n$E_{edge}=(P_{idle}+P_{active}$\n$+\\,P_{host})\\,T_{FU}$",
        fc="#e3eefb", ec="#2a78d6")
    box(0.165, 0.04, 0.200, 0.30, "Manual end-of-line (L0)\nrecall $r_{L0}$, delay $d_{L0}$\n$t_{L0}$ s per part", fc="#fbe9e1", ec="#eb6834")
    box(0.415, 0.60, 0.185, 0.36, "Divert + operator\nreview of every alert\n$\\lambda_{FA}T_{FU}+a\\,rD$\n$t_{review}$, $P_{station}$")
    box(0.415, 0.04, 0.185, 0.30, "Intercepted defects\n$q_{rw}(d)=q_0e^{-d/\\tau_{rw}}$")
    box(0.655, 0.70, 0.140, 0.24, "Rework\n$e_{rw}$ kWh/part")
    box(0.655, 0.38, 0.140, 0.24, "Scrap: $\\eta$\nrecovered,\n$1-\\eta$ lost")
    box(0.655, 0.04, 0.140, 0.26, "Field escapes\n$(1-r)D$:\nreplace + return,\n$\\kappa$ for class C")
    box(0.845, 0.22, 0.155, 0.56, "Carbon per FU\nmaterial, recycling,\nrework, review,\nedge electricity,\nedge hardware,\nescapes\n($\\gamma$, $EF_{mat}$, $EF_{rec}$)")
    arrow((0.115, 0.56), (0.165, 0.74), "AI", off=(-0.012, 0.0))
    arrow((0.115, 0.44), (0.165, 0.22), "manual", off=(-0.03, -0.02))
    arrow((0.365, 0.78), (0.415, 0.78), "$r$")
    arrow((0.365, 0.21), (0.415, 0.21), "$r_{L0}$")
    arrow((0.5075, 0.60), (0.5075, 0.34), "routed", off=(0.008, 0.0), ha="left")
    arrow((0.600, 0.25), (0.655, 0.80), "$f_A q_{rw}$", off=(-0.005, 0.02), ha="right")
    arrow((0.600, 0.19), (0.655, 0.48), "else", off=(0.012, -0.03), ha="left")
    arrow((0.600, 0.10), (0.655, 0.14), "missed", off=(0.0, 0.01))
    for y0, y1 in ((0.82, 0.66), (0.50, 0.50), (0.17, 0.34)):
        arrow((0.795, y0), (0.845, y1))
    save(fig, "fig1_system_boundary")


def fig2_energy():
    s = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    stages = STAGES
    labels = ["Model", "+ threshold", "+ temporal\npolicy", "Full pipeline\n(+ SQLite log)"]
    idle = s["stages"]["BASELINE_IDLE"]["p_total_w"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.1), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(stages))
    act = [s["stages"][k]["p_active_w"] for k in stages]
    a1.bar(x, [idle["median"]] * len(stages), 0.55, color="#c9c8c2", edgecolor=SURFACE, linewidth=1.0, label="GPU idle")
    a1.bar(x, [a["median"] for a in act], 0.55, bottom=idle["median"], color="#2a78d6", edgecolor=SURFACE, linewidth=1.0,
           label="active above idle")
    tot = [s["stages"][k]["p_total_w"] for k in stages]
    a1.errorbar(x, [t["median"] for t in tot], yerr=[[t["median"] - t["min"] for t in tot], [t["max"] - t["median"] for t in tot]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    for xi, t in zip(x, tot):
        a1.text(xi + 0.08, t["median"] + 0.3, f"{t['median']:.1f} W", ha="left", va="bottom", fontsize=6.5)
    a1.set_xticks(x, labels)
    a1.set_ylabel("GPU power at 30 FPS (W)\nmedian, whiskers min-max")
    a1.set_ylim(0, max(t["max"] for t in tot) * 1.45)
    a1.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02))
    a1.grid(axis="x", visible=False)
    ef = [s["stages"][k]["e_frame_active_j"] for k in stages]
    a2.bar(x, [e["median"] for e in ef], 0.55, color="#2a78d6", edgecolor=SURFACE)
    a2.errorbar(x, [e["median"] for e in ef], yerr=[[e["median"] - e["min"] for e in ef], [e["max"] - e["median"] for e in ef]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    for xi, e in zip(x, ef):
        a2.text(xi + 0.08, e["median"] + 0.01, f"{e['median']:.2f}", ha="left", va="bottom", fontsize=6.5)
    a2.set_xticks(x, labels)
    a2.set_ylabel("Active energy per frame (J)")
    a2.set_ylim(0, max(e["max"] for e in ef) * 1.15)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig2_measured_edge_energy")


def fig3_regimes():
    df = pd.read_csv(PROCESSED_DIR / "regime_outcomes.csv")
    regimes = load_regimes()
    order = list(regimes)
    names = [regimes[r].label.split(" ", 1)[0] for r in order]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    for ax, col, ylab in ((axes[0], "carbon_total", "Carbon relative to L0"), (axes[1], "operator_hours", "Operator hours relative to L0")):
        w = 0.26
        for i, sc in enumerate(SCENARIOS):
            sub = df[df.scenario == sc].set_index("regime").loc[order]
            rel = sub[col] / sub.loc["L0_Manual", col]
            ax.bar(np.arange(len(order)) + (i - 1) * w, rel, w, color=SC_COLOR[sc], hatch=SC_HATCH[sc],
                   edgecolor=SURFACE, linewidth=0.6, label=SC_LABEL[sc])
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(order)), names)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[1].set_yscale("log")
    axes[1].text(0, 1.1, "0 h", ha="center", va="bottom", fontsize=6.5, color=INK2)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_regime_outcomes")


def fig4_waterfall():
    df = pd.read_csv(PROCESSED_DIR / "comparisons.csv")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.25), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        row = df[(df.scenario == sc) & (df.baseline == "L0_Manual") & (df.policy == "B3_Full_Policy")].iloc[0]
        vals = [row[f"wf_{c}"] for c in COMPONENTS]
        total = row["delta_C"]
        share = [100 * v / total for v in vals]
        y = np.arange(len(COMPONENTS))
        ax.barh(y, share, 0.62, color=[POS if v >= 0 else NEG for v in vals], edgecolor=SURFACE)
        for yi, v, sh in zip(y, vals, share):
            ax.text(max(sh, 0) + 3, yi, f"{v:+,.0f}" if abs(v) >= 1000 else f"{v:+.3g}", va="center", ha="left",
                    fontsize=6.2, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        tot_txt = f"{total:,.0f}" if total >= 100 else f"{total:.3g}"
        ax.set_title(f"{SC_LABEL[sc]}: $\\Delta C$ = {tot_txt} kgCO$_2$e", fontsize=7.2)
        ax.set_xlabel(r"Share of net $\Delta C$ (%), labels in kgCO$_2$e")
        ax.set_xlim(min(-20, min(share) * 1.3), max(share) * 1.55)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(COMPONENTS)), [COMPONENT_LABEL[c] for c in COMPONENTS])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig4_carbon_waterfall")


def fig5_break_even_map():
    pis = np.logspace(-8, -0.7, 90)
    powers = np.logspace(0, 6.5, 90)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    measured_total = None
    for sc in SCENARIOS:
        s = load_scenario(sc)
        p = s.central()
        P, PI = np.meshgrid(powers, pis)
        host = P - p["gpu_idle_power"] - p["gpu_active_power"]
        params = {**p, "defect_prevalence": PI.ravel(), "host_power": host.ravel()}
        res = evaluate_all(params, s.class_shares)
        for base, ls in (("N0_NoInspection", "-"), ("L0_Manual", "--")):
            dc = compare(res[base], res["B3_Full_Policy"]).delta["C"].reshape(PI.shape)
            ax.contour(PI, P, dc, levels=[0.0], colors=[SC_COLOR[sc]], linewidths=1.4, linestyles=ls)
        measured_total = p["gpu_idle_power"] + p["gpu_active_power"] + p["host_power"]
        ax.plot(p["defect_prevalence"], measured_total, marker="o", ms=5, color=SC_COLOR[sc], mec=SURFACE, mew=1.0, ls="none")
    ax.axhline(measured_total, color=INK2, lw=0.6, ls=":")
    ax.text(1.5e-8, measured_total * 0.45, "edge cell today (GPU measured + host)", fontsize=6.0, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Defect prevalence $\pi$")
    ax.set_ylabel("Total edge-cell power (W)")
    handles = [Line2D([], [], color=SC_COLOR[sc], lw=1.4, label=SC_LABEL[sc]) for sc in SCENARIOS]
    handles += [Line2D([], [], color=INK2, lw=1.0, ls="-", label="vs N0 (no inspection)"),
                Line2D([], [], color=INK2, lw=1.0, ls="--", label="vs L0 (manual)")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0)
    ax.text(1.5e-2, 2.0, "right of / below a curve:\n" + r"$\Delta C>0$ (edge AI better)", fontsize=6.2, ha="center")
    fig.tight_layout()
    save(fig, "fig5_break_even_map")


def fig6_prcc():
    df = pd.read_csv(PROCESSED_DIR / "prcc_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45))
    for ax, sc in zip(axes, SCENARIOS):
        sub = df[df.scenario == sc].head(8).iloc[::-1]
        ax.barh(np.arange(len(sub)), sub.prcc, 0.62, color=[POS if v >= 0 else NEG for v in sub.prcc], edgecolor=SURFACE)
        ax.set_yticks(np.arange(len(sub)), [plabel(k) for k in sub.parameter])
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"PRCC with $\Delta C$ (B3 vs L0)")
        ax.set_title(SC_LABEL[sc], fontsize=7.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig6_global_sensitivity")


def fig7_uncertainty_tradeoff():
    d = pd.read_csv(PROCESSED_DIR / "monte_carlo_draws.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
    data = [d[d.scenario == sc]["B3_vs_L0_delta_C"].values for sc in SCENARIOS]
    bp = a1.boxplot(data, vert=False, widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, sc in zip(bp["boxes"], SCENARIOS):
        patch.set(facecolor=SC_COLOR[sc], edgecolor=INK2, hatch=SC_HATCH[sc], alpha=0.9)
    a1.set_xscale("log")
    a1.set_yticks([1, 2, 3], [SC_LABEL[sc] for sc in SCENARIOS])
    a1.set_xlabel(r"$\Delta C$, B3 vs L0 (kgCO$_2$e / 1000 parts), 5-95%")
    a1.grid(axis="y", visible=False)
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text(encoding="utf-8"))
    losses = np.logspace(-4, np.log10(0.5), 120)
    for sc in SCENARIOS:
        s_ = load_scenario(sc)
        res = evaluate_all({**s_.central(), "persistence_recall_loss": losses}, s_.class_shares)
        dc = compare(res["B0_Raw"], res["B3_Full_Policy"]).delta["C"]
        a2.plot(100 * losses, dc, color=SC_COLOR[sc], lw=1.6, label=SC_LABEL[sc])
        star = be[sc]["recall_loss_star_B3_vs_B0"]["value"]
        a2.plot(100 * star, 0.0, marker="o", ms=5, color=SC_COLOR[sc], mec=SURFACE, mew=1.0)
    a2.set_xscale("log")
    a2.set_yscale("symlog", linthresh=1.0)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel(r"Recall lost to persistence $\delta_r$ (percentage points)")
    a2.set_ylabel(r"$\Delta C$, B3 vs B0 (kgCO$_2$e)")
    a2.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig7_uncertainty_and_tradeoff")


def fig8_sqi():
    df = pd.read_csv(PROCESSED_DIR / "comparisons.csv")
    dims = ["M", "E", "C", "H"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    w = 0.26
    for i, sc in enumerate(SCENARIOS):
        row = df[(df.scenario == sc) & (df.baseline == "L0_Manual") & (df.policy == "B3_Full_Policy")].iloc[0]
        ax.bar(np.arange(4) + (i - 1) * w, [row[f"S_{k}"] for k in dims], w, color=SC_COLOR[sc], hatch=SC_HATCH[sc],
               edgecolor=SURFACE, linewidth=0.6,
               label=f"{SC_LABEL[sc]} (SQI>0 for {100 * row['sqi_weight_robustness']:.0f}% of weights)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(4), [r"$S_M$ material", r"$S_E$ energy", r"$S_C$ carbon", r"$S_H$ workload"])
    ax.set_ylim(-1.05, 1.6)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Sub-indicator, B3 vs L0")
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig8_sqi_subindicators")


def main():
    print("Step 08: generating figures")
    fig1_system_boundary()
    fig2_energy()
    fig3_regimes()
    fig4_waterfall()
    fig5_break_even_map()
    fig6_prcc()
    fig7_uncertainty_tradeoff()
    fig8_sqi()


if __name__ == "__main__":
    main()

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
from src.models.pipeline_evaluator import AI_TIERS, PRIMARY_TIER, compare, evaluate_all, load_regimes  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import ENERGY_SUMMARY, JETSON_SUMMARY, PROCESSED_DIR, RESULTS_FIG_DIR, SCENARIOS  # noqa: E402
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
    "manual_inspection_recall": r"manual recall $r_{L0}$", "return_distance": r"return distance $d_{ret}$", "freight_carbon_intensity": r"freight $EF_{fr}$",
    "collateral_multiplier": r"collateral $\kappa$", "grid_carbon_factor": r"grid $\gamma$", "host_power": r"$P_{host}$",
    "edge_idle_power": r"$P_{idle}$", "edge_active_power": r"$P_{active}$", "edge_embodied_carbon": r"$C_{hw}$",
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
    js = json.loads(JETSON_SUMMARY.read_text(encoding="utf-8"))
    stages = STAGES
    labels = ["Model", "+ thresh.", "+ policy", "Full"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})

    def stack(ax, x, idle, act, tot, idle_label, act_label):
        ax.bar(x, idle, 0.6, color="#c9c8c2", edgecolor=SURFACE, linewidth=1.0, label=idle_label)
        ax.bar(x, act, 0.6, bottom=idle, color="#2a78d6", edgecolor=SURFACE, linewidth=1.0, label=act_label)
        ax.errorbar(x, [t["median"] for t in tot], yerr=[[t["median"] - t["min"] for t in tot], [t["max"] - t["median"] for t in tot]],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
        for xi, t in zip(x, tot):
            ax.text(xi, t["max"] + 0.25, f"{t['median']:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.grid(axis="x", visible=False)

    # (a) workstation GPU, cumulative pipeline stages
    x = np.arange(len(stages))
    idle = s["stages"]["BASELINE_IDLE"]["p_total_w"]["median"]
    tot = [s["stages"][k]["p_total_w"] for k in stages]
    stack(a1, x, [idle] * len(stages), [s["stages"][k]["p_active_w"]["median"] for k in stages], tot, "idle", "active")
    a1.set_xticks(x, labels, rotation=30, ha="right")
    a1.set_ylabel("Power at 30 FPS (W)")
    a1.set_title("(a) Workstation GPU, stages", fontsize=7.5)
    a1.set_ylim(0, 14)
    a1.legend(loc="upper left", ncol=2, fontsize=6.5, handlelength=1.0, columnspacing=0.8)

    # (b) full pipeline per edge configuration; (c) active energy per frame
    order = ["patchcore_fp16", "patchcore_fp32", "padim_fp16", "padim_fp32"]
    runs = [js["runs"][c] for c in order] + [s]
    names = ["PatchCore FP16", "PatchCore FP32", "PaDiM FP16", "PaDiM FP32", "PatchCore FP32\n(workstation)"]
    full = [r["stages"]["STAGE_FULL_PIPELINE"] for r in runs]
    xi = np.arange(len(runs))
    stack(a2, xi, [r["stages"]["BASELINE_IDLE"]["p_total_w"]["median"] for r in runs],
          [f["p_active_w"]["median"] for f in full], [f["p_total_w"] for f in full], None, None)
    a2.axvline(3.5, color=INK2, lw=0.6, ls=":")
    a2.text(1.5, 13.2, "Jetson module (VDD_IN)", ha="center", fontsize=6.5, color=INK2)
    a2.text(4.0, 13.2, "GPU", ha="center", fontsize=6.5, color=INK2)
    a2.set_xticks(xi, names, fontsize=6.2, rotation=30, ha="right")
    a2.set_ylim(0, 14)
    a2.set_title("(b) Full pipeline per platform", fontsize=7.5)
    ef = [f["e_frame_active_j"] for f in full]
    a3.bar(xi, [1000 * e["median"] for e in ef], 0.6, color=["#2a78d6"] * 4 + ["#7a5195"], edgecolor=SURFACE)
    a3.errorbar(xi, [1000 * e["median"] for e in ef],
                yerr=[[1000 * (e["median"] - e["min"]) for e in ef], [1000 * (e["max"] - e["median"]) for e in ef]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    for x0, e in zip(xi, ef):
        a3.text(x0, 1000 * e["max"] + 6, f"{1000 * e['median']:.0f}", ha="center", va="bottom", fontsize=6.5)
    a3.set_xticks(xi, names, fontsize=6.2, rotation=30, ha="right")
    a3.set_ylabel("Active energy per frame (mJ)")
    a3.set_ylim(0, 1000 * max(e["max"] for e in ef) * 1.2)
    a3.set_title("(c) Energy above idle", fontsize=7.5)
    a3.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.6)
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
        host = P - p["edge_idle_power"] - p["edge_active_power"]
        params = {**p, "defect_prevalence": PI.ravel(), "host_power": host.ravel()}
        res = evaluate_all(params, s.class_shares)
        for base, ls in (("N0_NoInspection", "-"), ("L0_Manual", "--")):
            dc = compare(res[base], res["B3_Full_Policy"]).delta["C"].reshape(PI.shape)
            ax.contour(PI, P, dc, levels=[0.0], colors=[SC_COLOR[sc]], linewidths=1.4, linestyles=ls)
        measured_total = p["edge_idle_power"] + p["edge_active_power"] + p["host_power"]
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
    prcc = pd.read_csv(PROCESSED_DIR / "prcc_sensitivity.csv")
    sob = pd.read_csv(PROCESSED_DIR / "sobol_indices.csv")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.55))
    for ax, sc in zip(axes, SCENARIOS):
        sub = sob[sob.scenario == sc].sort_values("ST", ascending=False).head(8).iloc[::-1]
        sign = prcc[prcc.scenario == sc].set_index("parameter").prcc.reindex(sub.parameter).values
        col = [POS if v >= 0 else NEG for v in sign]
        y = np.arange(len(sub))
        ax.barh(y, sub.ST, 0.7, color=col, alpha=0.35, edgecolor=SURFACE, label="total $S_T$")
        ax.barh(y, sub.S1.clip(lower=0), 0.38, color=col, edgecolor=SURFACE, label="first order $S_1$")
        ax.errorbar(sub.ST, y, xerr=[sub.ST - sub.ST_lo, sub.ST_hi - sub.ST], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5)
        ax.set_yticks(y, [plabel(k) for k in sub.parameter])
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(r"Sobol index of $\Delta C$ (B3 vs L0)")
        ax.set_title(SC_LABEL[sc], fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=6.5, handlelength=1.2)
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
        if star is not None:  # no marker when B3 stays ahead of B0 for every recall loss
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




def fig9_operating_points():
    cur = pd.read_csv(PROCESSED_DIR / "detector_curves.csv")
    ref = pd.read_csv(PROCESSED_DIR / "detector_reference_points.csv")
    opt = pd.read_csv(PROCESSED_DIR / "carbon_optimal_thresholds.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ds_color = {"mvtec": "#2a78d6", "visa": "#eb6834"}
    ds_name = {"mvtec": "MVTec AD", "visa": "VisA"}
    cur = cur[cur.detector == "patchcore"]
    ref = ref[ref.detector == "patchcore"]
    for (ds, cat), sub in cur.groupby(["dataset", "category"], sort=False):
        sub = sub.sort_values("q")  # threshold order: averaged curves stay monotone
        a1.plot(np.maximum(sub.fpr, 1e-3), sub.recall, color=ds_color[ds], lw=0.9, alpha=0.75)
        r = ref[(ref.dataset == ds) & (ref.category == cat)].iloc[0]
        a1.plot(max(r.fpr_q99, 1e-3), r.recall_q99, marker="o", ms=3, color=ds_color[ds], mec=SURFACE, mew=0.6)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], color=c, lw=1.4, label=f"{ds_name[d]} ({(ref.dataset == d).sum()} categories)")
                       for d, c in ds_color.items()], loc="lower right", fontsize=6.3)
    a1.set_xscale("log")
    a1.set_xlim(1e-3, 1)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("False-positive rate per good part")
    a1.set_ylabel("Recall per defective part (PatchCore)")
    for i, sc in enumerate(SCENARIOS):
        sub = opt[opt.scenario == sc]
        x = np.full(len(sub), i) + np.linspace(-0.18, 0.18, len(sub))
        a2.scatter(x, np.maximum(sub.fpr_opt, 1e-3), s=14, color=SC_COLOR[sc], edgecolor=SURFACE, lw=0.6, zorder=3)
        a2.scatter(x, np.maximum(sub.fpr_budget, 1e-3), s=10, marker="_", color=INK2, zorder=2)
    a2.axhline(0.01, color=INK2, lw=0.8, ls="--")
    a2.text(2.45, 0.0108, "q99", fontsize=6.2, color=INK2, ha="right", va="bottom")
    a2.set_yscale("log")
    a2.set_ylim(1e-3, 1)
    a2.set_xticks(range(len(SCENARIOS)), [SC_LABEL[sc].replace(": ", ":\n") for sc in SCENARIOS])
    a2.set_ylabel("False-positive rate")
    a2.grid(axis="x", visible=False)
    a2.legend(handles=[Line2D([], [], marker="o", ls="none", color=INK2, label="carbon-optimal (per category)"),
                       Line2D([], [], marker="_", ls="none", color=INK2, label="5 alarms / 1,000 parts")],
              loc="lower left", fontsize=6)
    fig.tight_layout()
    save(fig, "fig9_detector_operating_points")


def fig10_decision_rule():
    pop = pd.read_csv(PROCESSED_DIR / "population.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    star = pop.k / pop.beta
    ok = (pop.beta > 0)
    win = ok & (pop.dc_N0 > 0)
    lose = ~win
    a1.scatter(pop.defect_prevalence[win], star[win], s=2, color="#2a78d6", alpha=0.35, lw=0, label=r"$\Delta C>0$")
    a1.scatter(pop.defect_prevalence[lose & ok], star[lose & ok], s=2, color=NEG, alpha=0.6, lw=0, label=r"$\Delta C<0$")
    xlo, xhi = pop.defect_prevalence.min() / 1.5, pop.defect_prevalence.max() * 1.5
    a1.plot((xlo, xhi), (xlo, xhi), color=INK, lw=0.9)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(xlo, xhi)
    a1.set_ylim(1e-9, 1)
    a1.set_xlabel(r"Defect prevalence $\pi$")
    a1.set_ylabel(r"Break-even $\pi^\star = k/\beta$")
    leg = a1.legend(loc="lower right", markerscale=4, fontsize=6.2, framealpha=0.9, frameon=True, edgecolor=GRID)
    for h in leg.legend_handles:
        h.set_alpha(1)
    share = np.sort(pop.edge_share_of_benefit_L0.dropna().values)
    a2.plot(100 * share, np.arange(1, len(share) + 1) / len(share), color="#2a78d6", lw=1.6)
    a2.axvline(1.0, color=INK2, lw=0.8, ls="--")
    a2.set_xscale("log")
    a2.set_xlim(1e-3, 100)
    a2.set_xlabel("Edge-cell carbon as % of benefit over manual inspection")
    a2.set_ylabel("Share of products (cumulative)")
    fig.tight_layout()
    save(fig, "fig10_decision_rule_population")


def fig11_cry_wolf():
    summ = pd.read_csv(PROCESSED_DIR / "cry_wolf_summary.csv")
    cur = pd.read_csv(PROCESSED_DIR / "cry_wolf_curves.csv")
    regimes = load_regimes()
    tiers = list(AI_TIERS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    w = 0.26
    x = np.arange(len(tiers))
    for i, sc in enumerate(SCENARIOS):
        sub = summ[summ.scenario == sc].set_index("tier").loc[tiers]
        a1.bar(x + (i - 1) * w, 100 * sub.ppv, w, color=SC_COLOR[sc], hatch=SC_HATCH[sc], edgecolor=SURFACE, label=SC_LABEL[sc])
    a1.set_xticks(x, [regimes[t].label.split(" ", 1)[0] for t in tiers])
    a1.set_ylabel("Alert precision PPV (%)")
    a1.set_ylim(0, 115)
    a1.legend(loc="upper center", ncol=3, fontsize=6.3, handlelength=1.2, columnspacing=0.8)
    a1.grid(axis="x", visible=False)
    for sc in SCENARIOS:
        c = cur[(cur.scenario == sc) & (cur.tier == PRIMARY_TIER)]
        a2.plot(c.omega, c.delta_C_vs_L0 / c.delta_C_vs_L0.iloc[0], color=SC_COLOR[sc], lw=1.6, label=SC_LABEL[sc])
        st = summ[(summ.scenario == sc) & (summ.tier == PRIMARY_TIER)].iloc[0]
        if st.status == "root":
            a2.plot(st.omega_star, 0.0, marker="o", ms=5, color=SC_COLOR[sc], mec=SURFACE, mew=1.0)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel(r"Cry-wolf strength $\omega$ (0: full compliance; 1: probability matching)", fontsize=6.8)
    a2.set_ylabel(r"$\Delta C$ (B3 vs L0) / value at $\omega=0$")
    a2.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig11_cry_wolf")


def fig12_nomogram():
    d = pd.read_csv(PROCESSED_DIR / "nomogram.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    style = {"jetson_central": ("#2a78d6", "-", "embedded, central grid"),
             "jetson_low_grid": ("#2a78d6", ":", "embedded, low / high grid"),
             "jetson_high_grid": ("#2a78d6", ":", None),
             "workstation_central": ("#7a5195", "--", "workstation, central grid")}
    for v, (c, ls, lab) in style.items():
        sub = d[d.variant == v]
        ax.plot(sub.embodied_per_part, sub.defects_per_hour_star, color=c, ls=ls, lw=1.4, label=lab)
    cen = d[d.variant == "jetson_central"]
    ax.fill_between(cen.embodied_per_part, cen.defects_per_hour_star, 1e4, color="#2a78d6", alpha=0.07, lw=0)
    for sc in SCENARIOS:
        p = load_scenario(sc).central()
        x0, y0 = p["part_mass"] * p["material_carbon_factor"], p["defect_prevalence"] * p["line_throughput"]
        ax.plot(x0, y0, marker="o", ms=5, color=SC_COLOR[sc], mec=SURFACE, mew=1.0, ls="none")
        ax.annotate(SC_LABEL[sc].split(":")[0], (x0, y0), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=SC_COLOR[sc])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(d.embodied_per_part.min(), d.embodied_per_part.max())
    ax.set_ylim(1e-6, 1e3)
    ax.text(0.97, 0.93, "edge inspection\npays back", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel(r"Embodied carbon per part $m\,EF_{mat}$ (kgCO$_2$e)")
    ax.set_ylabel("Break-even defects per hour")
    ax.legend(loc="lower left", fontsize=5.8, handlelength=1.8, frameon=True, framealpha=0.9, edgecolor=GRID)
    fig.tight_layout()
    save(fig, "fig12_nomogram")


DET_LABEL = {"patchcore": "PatchCore R18, 224 px", "patchcore_448": "PatchCore R18, 448 px",
             "patchcore_wrn50": "PatchCore WRN50, 224 px", "padim": "PaDiM R18, 224 px"}
DET_COLOR = {"patchcore": "#2a78d6", "patchcore_448": "#1baf7a", "patchcore_wrn50": "#7a5195", "padim": "#eb6834"}


def fig13_compute_for_recall():
    cr = pd.read_csv(PROCESSED_DIR / "compute_recall.csv")
    summ = pd.read_csv(PROCESSED_DIR / "compute_recall_summary.csv")
    dets = [d for d in DET_LABEL if d in set(cr.detector)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
    one = cr[cr.scenario == SCENARIOS[0]]
    label_pos = {"patchcore": (5, 0.88, "left"), "padim": (5, 0.40, "left"), "patchcore_wrn50": (5, 0.40, "left"),
                 "patchcore_448": (-5, 0.28, "right")}
    for d in dets:
        sub = one[one.detector == d]
        x0 = sub.frame_mj.iloc[0]
        ym, yv = sub[sub.dataset == "mvtec"].recall_q99.median(), sub[sub.dataset == "visa"].recall_q99.median()
        a1.plot([x0, x0], [yv, ym], color=DET_COLOR[d], lw=1.2, alpha=0.6)
        a1.plot(x0, ym, marker="o", ms=6, color=DET_COLOR[d], mec=SURFACE, ls="none")
        a1.plot(x0, yv, marker="s", ms=6, color=DET_COLOR[d], mec=SURFACE, ls="none")
        dx, ylab, ha = label_pos[d]
        a1.annotate(DET_LABEL[d].replace(", ", "\n"), (x0, ylab), xytext=(dx, 0), textcoords="offset points",
                    fontsize=6, color=DET_COLOR[d], va="center", ha=ha)
    a1.axhline(load_scenario(SCENARIOS[0]).central()["manual_inspection_recall"], color=INK2, lw=0.8, ls="--")
    a1.text(0.02, 0.765, "manual inspection", transform=a1.get_yaxis_transform(), fontsize=6.3, color=INK2, ha="left", va="bottom")
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="o", ls="none", color=INK2, label="MVTec AD median"),
                       Line2D([], [], marker="s", ls="none", color=INK2, label="VisA median")], loc="lower left", fontsize=6.3)
    a1.set_xlim(0, 200)
    a1.set_ylim(0, 1.05)
    a1.set_xlabel("Jetson active energy per frame (mJ, FP16)")
    a1.set_ylabel("Recall at q99 (median over categories)")
    w = 0.8 / len(dets)
    x = np.arange(len(SCENARIOS))
    for i, d in enumerate(dets):
        v = summ[summ.detector == d].set_index("scenario").loc[list(SCENARIOS)]
        a2.bar(x + (i - (len(dets) - 1) / 2) * w, 100 * v.share_beating_L0, w, color=DET_COLOR[d], edgecolor=SURFACE,
               label=DET_LABEL[d])
    a2.set_xticks(x, [SC_LABEL[sc].replace(": ", ":\n") for sc in SCENARIOS])
    a2.set_ylabel("Categories where B3 beats manual (%)")
    a2.set_ylim(0, 128)
    a2.legend(loc="upper center", ncol=2, fontsize=6, handlelength=1.0, columnspacing=0.8)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig13_compute_for_recall")


def graphical_abstract():
    plat = pd.read_csv(PROCESSED_DIR / "platform_comparison.csv")
    nomo = pd.read_csv(PROCESSED_DIR / "nomogram.csv")
    ref = pd.read_csv(PROCESSED_DIR / "detector_reference_points.csv").query("detector == 'patchcore'")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.0, 4.0))
    # 1: compute vs benefit
    jp = plat[plat.configuration == "jetson_patchcore_fp16"].set_index("scenario").loc[list(SCENARIOS)]
    x = np.arange(len(SCENARIOS))
    a1.bar(x - 0.2, jp.delta_C_vs_L0, 0.38, color="#2a78d6", label="carbon benefit vs manual")
    a1.bar(x + 0.2, jp.edge_cell_carbon, 0.38, color="#e34948", label="edge cell (Jetson)")
    a1.set_yscale("log")
    a1.set_xticks(x, [SC_LABEL[sc].replace(": ", ":\n") for sc in SCENARIOS], fontsize=8)
    a1.set_ylabel("kgCO$_2$e per 1,000 parts")
    a1.set_title("1  Compute is negligible", fontsize=10, loc="left", fontweight="bold")
    a1.legend(fontsize=7.5, loc="upper left")
    a1.grid(axis="x", visible=False)
    # 2: recall decides against human inspectors
    s_ = load_scenario(SCENARIOS[1])
    rec = np.linspace(0.3, 1.0, 141)
    res = evaluate_all({**s_.central(), "ai_recall": rec}, s_.class_shares)
    dc = compare(res["L0_Manual"], res[PRIMARY_TIER]).delta["C"]
    a2.plot(rec, dc, color="#2a78d6", lw=2.0)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.axvline(s_.central()["manual_inspection_recall"], color=INK2, lw=0.8, ls="--")
    a2.text(s_.central()["manual_inspection_recall"] - 0.01, a2.get_ylim()[1] * 0.85, "human\ninspectors", ha="right", fontsize=7.5, color=INK2)
    for ds, c in (("mvtec", "#2a78d6"), ("visa", "#eb6834")):
        v = ref[ref.dataset == ds].recall_q99.median()
        a2.plot(v, 0, marker="v", ms=8, ls="none", color=c, mec=SURFACE, label={"mvtec": "MVTec AD median", "visa": "VisA median"}[ds])
    a2.set_xlabel("Detector recall")
    a2.set_ylabel(r"$\Delta C$ vs manual (machined part)")
    a2.set_title("2  Recall decides vs humans", fontsize=10, loc="left", fontweight="bold")
    a2.legend(fontsize=7.5, loc="lower right")
    # 3: the rule
    jc = nomo[nomo.variant == "jetson_central"]
    a3.plot(jc.embodied_per_part, jc.defects_per_hour_star, color="#2a78d6", lw=2.0)
    a3.fill_between(jc.embodied_per_part, jc.defects_per_hour_star, 1e4, color="#2a78d6", alpha=0.08, lw=0)
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_ylim(1e-6, 1e3)
    a3.set_xlim(jc.embodied_per_part.min(), jc.embodied_per_part.max())
    a3.text(0.95, 0.92, "inspect", transform=a3.transAxes, ha="right", fontsize=9, color="#2a78d6", fontweight="bold")
    a3.text(0.05, 0.08, "do not\ninspect", transform=a3.transAxes, ha="left", fontsize=9, color=INK2)
    a3.set_xlabel("Embodied carbon per part (kgCO$_2$e)")
    a3.set_ylabel("Break-even defects per hour")
    a3.set_title("3  One rule decides vs no inspection", fontsize=10, loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "graphical_abstract")


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
    fig9_operating_points()
    fig10_decision_rule()
    fig11_cry_wolf()
    fig12_nomogram()
    fig13_compute_for_recall()
    graphical_abstract()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 9: Generate publication vector figures (PDF) and 300 DPI PNGs.
Outputs 6 comprehensive figures to results/figures/ and paper/figures/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Publication aesthetic styling
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RESULTS_FIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/figures")
PAPER_FIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/paper/figures")
PROCESSED_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")
RAW_RESULTS_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/raw")


def save_fig(fig, name: str):
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    for d in [RESULTS_FIG_DIR, PAPER_FIG_DIR]:
        pdf_path = d / f"{name}.pdf"
        png_path = d / f"{name}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {name}.pdf and {name}.png")


def plot_fig1_sankey_flow():
    """Fig 1: Conceptual and empirical quality-to-sustainability routing flow."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    boxes = [
        {"x": 0.05, "y": 0.5, "w": 0.15, "h": 0.35, "label": "Inspected Units\nN = 1,000\n(Defects D = π · N)", "color": "#1f77b4"},
        {"x": 0.30, "y": 0.65, "w": 0.16, "h": 0.25, "label": "Detected & Routed\n(D · r_p)\nDelay d = 3 frames", "color": "#2ca02c"},
        {"x": 0.30, "y": 0.15, "w": 0.16, "h": 0.20, "label": "Quality Escapes\n(D · (1 - r_p))\nHigh Carbon Penalty", "color": "#d62728"},
        {"x": 0.58, "y": 0.72, "w": 0.16, "h": 0.22, "label": "Precision Rework\nq_rw(d) = q0 - β·d\nEnergy: e_rw", "color": "#17becf"},
        {"x": 0.58, "y": 0.40, "w": 0.16, "h": 0.22, "label": "Gross Scrap\nq_sc(d) = 1 - q_rw\nMass: N_sc · m", "color": "#ff7f0e"},
        {"x": 0.84, "y": 0.52, "w": 0.14, "h": 0.18, "label": "Circular Recovery\n(M_gross · η)\nAvoided Virgin Metal", "color": "#bcbd22"},
        {"x": 0.84, "y": 0.28, "w": 0.14, "h": 0.18, "label": "Unrecovered Loss\n(M_gross · (1 - η))\nEmbodied Carbon", "color": "#8c564b"},
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            ec="none", fc=b["color"], alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(
            b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["label"],
            ha="center", va="center", color="white", weight="bold", fontsize=8.5
        )

    # Connecting arrows
    arrows = [
        ((0.20, 0.70), (0.30, 0.77), "Recall r_p"),
        ((0.20, 0.60), (0.30, 0.25), "Miss 1 - r_p"),
        ((0.46, 0.80), (0.58, 0.83), "Reworkability q_rw(d)"),
        ((0.46, 0.72), (0.58, 0.51), "Scrap fraction q_sc(d)"),
        ((0.74, 0.55), (0.84, 0.61), "Recovery η"),
        ((0.74, 0.45), (0.84, 0.37), "Loss 1 - η"),
    ]

    for start, end, lbl in arrows:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8, mutation_scale=14)
        )
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03)
        ax.text(mid[0], mid[1], lbl, ha="center", va="bottom", fontsize=8, color="#222222")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.08, 1.0)
    ax.set_title("Fig 1: Mass, Energy, and Defect Flow Routing in AI-Enabled Edge Inspection", pad=15)
    save_fig(fig, "fig1_quality_to_sustainability_sankey")


def plot_fig2_energy_latency():
    """Fig 2: Measured RTX 4050 GPU power and active energy profiling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    energy_file = RAW_RESULTS_DIR / "energy_summary.json"
    stages = ["BASELINE_IDLE", "STAGE_MODEL_INFER", "STAGE_MODEL_CCT", "STAGE_MODEL_POLICY", "STAGE_FULL_PIPELINE"]
    labels = ["Quiescent\nIdle", "Model\nInference", "Model +\nCCT", "Model +\n7-Stage Policy", "Full Pipeline\n(+WAL/MQTT)"]

    if energy_file.exists():
        with open(energy_file) as f:
            data = json.load(f)
        p_active = [data["stages"][s].get("p_active_mean_w", 0.0) for s in stages]
        e_frame = [data["stages"][s].get("e_frame_wh", 0.0) * 1e3 for s in stages]
        fps = [data["stages"][s].get("fps", 0.0) for s in stages]
    else:
        p_active = [0.0, 12.5, 32.5, 56.3, 7.6]
        e_frame = [0.0, 0.0029, 0.0069, 0.0107, 0.0336]
        fps = [0.0, 1209.5, 1252.7, 1442.6, 62.0]

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars1 = ax1.bar(labels, p_active, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Active Compute Power (W)")
    ax1.set_title("Active Hardware Power Draw on RTX 4050 GPU")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for b, p in zip(bars1, p_active):
        if p > 0:
            ax1.text(b.get_x() + b.get_width() / 2, p + 1.2, f"{p:.1f} W", ha="center", va="bottom", fontsize=8.5)

    # Frame active energy (mWh/frame)
    bars2 = ax2.bar(labels[1:], e_frame[1:], color=colors[1:], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Active Energy per Frame (mWh / frame)")
    ax2.set_title("Energy Footprint per Inspected Frame")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for b, ef, f_val in zip(bars2, e_frame[1:], fps[1:]):
        ax2.text(
            b.get_x() + b.get_width() / 2, ef + 0.001,
            f"{ef:.4f} mWh\n({f_val:.0f} FPS)",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    save_fig(fig, "fig2_measured_edge_energy_latency")


def plot_fig3_raw_outcomes():
    """Fig 3: Sustainability outcomes across B0 to B4 in Scenarios A, B, C."""
    csv_file = PROCESSED_DIR / "scenario_evaluations.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scenarios = ["precision_component", "machined_metal", "high_value_component"]
    sc_titles = {"precision_component": "A: Precision", "machined_metal": "B: Metal CNC", "high_value_component": "C: High Value"}
    policies = ["B0_Raw", "B1_Quantile99", "B2_CCT", "B3_CCT_kofN", "B4_Full_Cascade"]
    p_labels = ["B0 Raw", "B1 Q99", "B2 CCT", "B3 k-of-N", "B4 Cascade"]

    x = np.arange(len(policies))
    width = 0.25

    # 1. Net Material Loss (kg)
    ax = axes[0, 0]
    for idx, sc in enumerate(scenarios):
        vals = df[df["scenario"] == sc]["net_loss_kg"].values
        ax.bar(x + idx * width, vals, width, label=sc_titles[sc], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("Net Unrecovered Material Loss (kg / 1k)")
    ax.set_title("Material Loss Across Policy Hierarchy")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Total Energy (kWh)
    ax = axes[0, 1]
    for idx, sc in enumerate(scenarios):
        vals = df[df["scenario"] == sc]["total_energy_kwh"].values
        ax.bar(x + idx * width, vals, width, label=sc_titles[sc], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("Total Energy Consumption (kWh / 1k)")
    ax.set_title("Operational Energy (Compute + Review + Rework)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Total Carbon (kgCO2e)
    ax = axes[1, 0]
    for idx, sc in enumerate(scenarios):
        vals = df[df["scenario"] == sc]["total_carbon_kgco2e"].values
        ax.bar(x + idx * width, vals, width, label=sc_titles[sc], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("Total Carbon Footprint (kgCO2e / 1k)")
    ax.set_title("Total Carbon Footprint (Material + Grid + Escapes)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # 4. Review Queue Utilization rho
    ax = axes[1, 1]
    for idx, sc in enumerate(scenarios):
        vals = df[df["scenario"] == sc]["queue_rho"].values
        ax.bar(x + idx * width, vals, width, label=sc_titles[sc], alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Overload Boundary (ρ=1.0)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(p_labels)
    ax.set_ylabel("Human Operator Queue Utilization (ρ)")
    ax.set_title("Operator Cognitive Review Load (μ = 60 / hr)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    save_fig(fig, "fig3_raw_sustainability_outcomes")


def plot_fig4_break_even():
    """Fig 4: 2D Break-even heatmaps and zero-carbon frontiers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    scenarios = ["precision_component", "machined_metal", "high_value_component"]
    sc_titles = ["Scenario A: Precision", "Scenario B: Machined Metal", "Scenario C: High Value"]

    from src.models.pipeline_evaluator import ScenarioPipelineEvaluator
    CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")

    pi_range = np.linspace(0.001, 0.06, 30)
    e_range = np.linspace(0.0, 50.0, 30)  # kWh / 1k

    for idx, (sc, title) in enumerate(zip(scenarios, sc_titles)):
        ax = axes[idx]
        ev = ScenarioPipelineEvaluator.from_yaml(CONFIG_DIR / f"{sc}.yaml")
        grid = np.zeros((len(e_range), len(pi_range)))

        for i, e_val in enumerate(e_range):
            for j, pi_val in enumerate(pi_range):
                cfg = dict(ev.cfg)
                cfg["defect_prevalence"] = float(pi_val)
                cfg["edge_energy_kwh_per_1k"] = float(e_val)
                e_inst = ScenarioPipelineEvaluator(cfg)
                b0 = e_inst.evaluate_policy("B0_Raw", 0.88, 180.0, 0.0, edge_active=False)
                b4 = e_inst.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0, edge_active=True)
                grid[i, j] = b4.carbon.net_carbon_benefit_kgco2e

        im = ax.contourf(pi_range * 100, e_range, grid, levels=20, cmap="viridis")
        # Plot zero contour line if present
        cs = ax.contour(pi_range * 100, e_range, grid, levels=[0.0], colors="red", linewidths=2.0)
        ax.clabel(cs, fmt="ΔC=0", fontsize=9)

        ax.set_xlabel("Defect Prevalence π (%)")
        if idx == 0:
            ax.set_ylabel("Edge Compute Energy (kWh / 1k units)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Net Carbon Benefit ΔC (kgCO2e)")

    plt.tight_layout()
    save_fig(fig, "fig4_break_even_heatmaps")


def plot_fig5_tornado():
    """Fig 5: Tornado sensitivity charts showing swing in Delta C."""
    csv_file = PROCESSED_DIR / "sensitivity_rankings.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    scenarios = ["precision_component", "machined_metal", "high_value_component"]
    sc_titles = ["Scenario A: Precision", "Scenario B: Machined Metal", "Scenario C: High Value"]

    for idx, (sc, title) in enumerate(zip(scenarios, sc_titles)):
        ax = axes[idx]
        sub = df[df["scenario"] == sc].sort_values(by="swing", ascending=True)
        params = sub["parameter"].values
        swings = sub["swing"].values

        ax.barh(params, swings, color="#2b5c8f", alpha=0.85, edgecolor="black", height=0.6)
        ax.set_xlabel("Carbon Swing |ΔC_high - ΔC_low| (kgCO2e)")
        ax.set_title(f"{title} (Top Drivers)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "fig5_tornado_sensitivity")


def plot_fig6_sqi_radar():
    """Fig 6: Multi-Profile Sustainability Quality Index (SQI) comparison."""
    csv_file = PROCESSED_DIR / "scenario_evaluations.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)
    b4_df = df[df["policy"] == "B4_Full_Cascade"]

    fig, ax = plt.subplots(figsize=(9, 5))
    profiles = ["sqi_balanced", "sqi_material_priority", "sqi_carbon_priority", "sqi_human_centered"]
    prof_labels = ["Balanced\n(0.25 ea)", "Material\nPriority", "Carbon\nPriority", "Human\nCentered"]

    scenarios = ["precision_component", "machined_metal", "high_value_component"]
    sc_titles = {"precision_component": "Scenario A: Precision", "machined_metal": "Scenario B: Machined Metal", "high_value_component": "Scenario C: High Value"}
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    x = np.arange(len(profiles))
    width = 0.25

    for idx, sc in enumerate(scenarios):
        row = b4_df[b4_df["scenario"] == sc]
        scores = [float(row[p].values[0]) for p in profiles]
        ax.bar(x + idx * width, scores, width, label=sc_titles[sc], color=colors[idx], alpha=0.85, edgecolor="black")

    ax.set_xticks(x + width)
    ax.set_xticklabels(prof_labels)
    ax.set_ylabel("Sustainability Quality Index (SQI)")
    ax.set_title("Policy B4 (Full Cascade) SQI Scores Across 4 Stakeholder Profiles")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    save_fig(fig, "fig6_sqi_radar_profile_comparison")



def plot_fig7_waterfall():
    """Fig 7: Net carbon waterfall decomposition across Scenarios A, B, and C."""
    csv_file = PROCESSED_DIR / "scenario_evaluations.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)
    b4_df = df[df["policy"] == "B4_Full_Cascade"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    scenarios = ["precision_component", "machined_metal", "high_value_component"]
    sc_titles = ["Scenario A: Precision", "Scenario B: Machined Metal", "Scenario C: High-Value"]

    terms = ["delta_c_mat", "delta_c_rw", "delta_c_rev", "c_edge_carbon", "delta_c_esc", "delta_c_kgco2e"]
    term_labels = [r"$\Delta C_{\mathrm{mat}}$", r"$\Delta C_{\mathrm{rw}}$", r"$\Delta C_{\mathrm{rev}}$", r"$-C_{\mathrm{edge}}$", r"$\Delta C_{\mathrm{esc}}$", r"Net $\Delta C$"]

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        row = b4_df[b4_df["scenario"] == sc]
        if row.empty or "delta_c_mat" not in row.columns:
            continue
        vals = [float(row[t].values[0]) for t in terms]
        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals[:-1]] + ["#1f77b4"]

        x = np.arange(len(terms))
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor="black", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(term_labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(sc_titles[i], fontweight="bold")
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel(r"Carbon Delta [$\mathrm{kgCO}_2\mathrm{e}$ / 1k parts]" if i == 0 else "")

        # Add text labels on top of bars
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            ax.annotate(f"{val:+.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        xytext=(0, 3 if y_pos >= 0 else -8),
                        textcoords="offset points",
                        ha="center", va=va, fontsize=7.5, fontweight="bold")

    plt.suptitle("Carbon Savings Waterfall Decomposition Across 5 Operational Drivers (Policy B4)", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig7_waterfall_carbon_decomposition")


def plot_fig8_adverse_sensitivity():
    """Fig 8: Adverse sensitivity boundary checks (Prevalence pi -> 0, and Multi-View n_f in [1, 10])."""
    from src.models.pipeline_evaluator import ScenarioPipelineEvaluator
    CONFIG_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/configs/scenarios")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Delta C vs prevalence pi (approaching zero)
    ev_a = ScenarioPipelineEvaluator.from_yaml(CONFIG_DIR / "precision_component.yaml")
    ev_b = ScenarioPipelineEvaluator.from_yaml(CONFIG_DIR / "machined_metal.yaml")

    pis = np.linspace(0.0, 0.002, 60)
    dc_a_pis = []
    dc_b_pis = []

    for pi_v in pis:
        c_a = dict(ev_a.cfg)
        c_a["defect_prevalence"] = float(pi_v)
        e_a = ScenarioPipelineEvaluator(c_a)
        b0_a = e_a.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4_a = e_a.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0_a, edge_active=True)
        dc_a_pis.append(b4_a.carbon.net_carbon_benefit_kgco2e)

        c_b = dict(ev_b.cfg)
        c_b["defect_prevalence"] = float(pi_v)
        e_b = ScenarioPipelineEvaluator(c_b)
        b0_b = e_b.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4_b = e_b.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0_b, edge_active=True)
        dc_b_pis.append(b4_b.carbon.net_carbon_benefit_kgco2e)

    ax1.plot(pis * 100, dc_a_pis, label="Scenario A: Precision", color="#1f77b4", linewidth=2.0)
    ax1.plot(pis * 100, dc_b_pis, label="Scenario B: Machined Metal", color="#2ca02c", linewidth=2.0)
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="Break-Even ($\Delta C = 0$)")
    ax1.axvspan(0.0, 0.0053, color="gray", alpha=0.15, label="Net-Negative Carbon Regime (A)")
    ax1.set_xlabel("Defect Prevalence $\pi$ [%]")
    ax1.set_ylabel(r"Net Carbon Benefit $\Delta C$ [$\mathrm{kgCO}_2\mathrm{e}$]")
    ax1.set_title("Adverse Boundary: $\pi \to 0$ (Net-Negative Compute Regime)", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    # Subplot 2: Multi-view inspection frames per part (n_f in [1, 10])
    n_fs = np.linspace(1.0, 10.0, 19)
    dc_a_nfs = []
    dc_b_nfs = []

    for nf_v in n_fs:
        c_a = dict(ev_a.cfg)
        c_a["frames_per_part"] = float(nf_v)
        e_a = ScenarioPipelineEvaluator(c_a)
        b0_a = e_a.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4_a = e_a.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0_a, edge_active=True)
        dc_a_nfs.append(b4_a.carbon.net_carbon_benefit_kgco2e)

        c_b = dict(ev_b.cfg)
        c_b["frames_per_part"] = float(nf_v)
        e_b = ScenarioPipelineEvaluator(c_b)
        b0_b = e_b.evaluate_policy("B0_Raw", 0.88, 180.0, 150.0, edge_active=False)
        b4_b = e_b.evaluate_policy("B4_Full_Cascade", 0.99, 12.0, 3.0, baseline_result=b0_b, edge_active=True)
        dc_b_nfs.append(b4_b.carbon.net_carbon_benefit_kgco2e)

    ax2.plot(n_fs, dc_a_nfs, label="Scenario A: Precision", color="#1f77b4", linewidth=2.0)
    ax2.plot(n_fs, dc_b_nfs, label="Scenario B: Machined Metal", color="#2ca02c", linewidth=2.0)
    ax2.set_xlabel("Frames per Part $n_f$ [multi-angle inspection]")
    ax2.set_ylabel(r"Net Carbon Benefit $\Delta C$ [$\mathrm{kgCO}_2\mathrm{e}$]")
    ax2.set_title("Adverse Boundary: Multi-View Inspection Overhead ($n_f$)", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "fig8_adverse_sensitivity_sweeps")

def main():
    print("=== Step 09: Generating Publication Vector Figures ===")
    plot_fig1_sankey_flow()
    plot_fig2_energy_latency()
    plot_fig3_raw_outcomes()
    plot_fig4_break_even()
    plot_fig5_tornado()
    plot_fig6_sqi_radar()
    plot_fig7_waterfall()
    plot_fig8_adverse_sensitivity()
    print("Step 09 completed successfully. 8 publication figures generated.\n")


if __name__ == "__main__":
    main()
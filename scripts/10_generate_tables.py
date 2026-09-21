#!/usr/bin/env python3
"""
Step 10: Generate publication-ready LaTeX tables.
Exports to results/tables/ and paper/tables/.
"""

from pathlib import Path
import pandas as pd
import json

PROCESSED_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/processed")
TABLES_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/results/tables")
PAPER_TABLES_DIR = Path("/home/sengar/sustainable-edge-quality-intelligence/paper/tables")


def save_latex(content: str, filename: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for d in [TABLES_DIR, PAPER_TABLES_DIR]:
        (d / filename).write_text(content, encoding="utf-8")
    print(f"Generated {filename}")


def generate_table2_policy_sustainability():
    csv_path = PROCESSED_DIR / "scenario_evaluations.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Operational Sustainability Accounting and SQI Scores Across 5-Tier Policy Hierarchy ($N=1{,}000$ Inspected Units).}",
        r"\label{tab:policy_sustainability}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Policy Mode} & \textbf{Recall ($r_p$)} & \textbf{$\Delta M$ (kg)} & \textbf{$\Delta E$ (kWh)} & \textbf{$\Delta C$ (kgCO$_2$e)} & \textbf{$\Delta H$ (h)} & \textbf{Queue $\rho$} & \textbf{SQI (Bal.)} \\",
        r"\midrule",
    ]

    sc_map = {"precision_component": "A: Precision", "machined_metal": "B: Machined Metal", "high_value_component": "C: High-Value"}

    for sc in ["precision_component", "machined_metal", "high_value_component"]:
        sub = df[df["scenario"] == sc]
        tex.append(f"\\textbf{{{sc_map[sc]}}} & & & & & & & & \\\\")
        for _, r in sub.iterrows():
            pol_name = r['policy'].replace('_', ' ')
            tex.append(
                f"  & {pol_name} & {r['recall']:.2f} & "
                f"{r['delta_m_kg']:+.3f} & {r['delta_e_kwh']:+.3f} & "
                f"{r['delta_c_kgco2e']:+.2f} & {r['delta_h_hours']:+.4f} & "
                f"{r['queue_rho']:.2f} & {r['sqi_balanced']:+.3f} \\\\"
            )
        tex.append(r"\midrule")

    tex[-1] = r"\bottomrule"
    tex.extend([
        r"\end{tabular}}",
        r"\end{table*}",
    ])
    save_latex("\n".join(tex), "tab2_policy_hierarchy_sustainability.tex")


def generate_table3_monte_carlo():
    mc_file = PROCESSED_DIR / "monte_carlo_summary.json"
    if not mc_file.exists():
        return
    with open(mc_file) as f:
        data = json.load(f)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{10,000-Draw Monte Carlo Uncertainty and Probability of Positive Benefit.}",
        r"\label{tab:monte_carlo}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{$\Delta C$ Median [5th, 95th]} & \textbf{$P(\Delta C > 0)$} & \textbf{SQI (Bal.) Median} & \textbf{$P(\text{SQI} > 0)$} \\",
        r"\midrule",
    ]

    sc_map = {"precision_component": "A: Precision", "machined_metal": "B: Machined Metal", "high_value_component": "C: High-Value"}
    for sc, info in data.items():
        dc = info["delta_c"]
        sq = info["sqi_balanced"]
        p_c = info["p_delta_c_positive"] * 100
        p_sq = info["p_sqi_positive"] * 100
        tex.append(
            f"{sc_map[sc]} & {dc['p50']:.1f} [{dc['p05']:.1f}, {dc['p95']:.1f}] & {p_c:.1f}\\% & "
            f"{sq['p50']:.3f} [{sq['p05']:.3f}, {sq['p95']:.3f}] & {p_sq:.1f}\\% \\\\"
        )

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    save_latex("\n".join(tex), "tab3_monte_carlo_confidence.tex")


def generate_table4_break_even():
    csv_path = PROCESSED_DIR / "break_even_table.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Analytical and Empirical Break-Even Operating Frontiers.}",
        r"\label{tab:break_even}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{$\pi^\star$ (Prevalence)} & \textbf{$e_{\text{edge}}^\star$ (kWh/1k)} & \textbf{$d^\star$ (s [frames])} & \textbf{$\gamma^\star$ (kgCO$_2$e/kWh)} \\",
        r"\midrule",
    ]

    sc_map = {"precision_component": "A: Precision", "machined_metal": "B: Machined Metal", "high_value_component": "C: High-Value"}
    for _, r in df.iterrows():
        sc = r["scenario"]
        pi_str = f"{r['pi_star']:.2e}" if pd.notnull(r['pi_star']) else "N/A"
        e_str = f"{r['e_edge_star_kwh_per_1k']:.1f}" if pd.notnull(r['e_edge_star_kwh_per_1k']) else "N/A"
        d_str = f"{r['d_star_seconds']:.2f} [{r['d_star_frames']:.0f}]" if pd.notnull(r['d_star_seconds']) else "N/A"
        g_str = f"{r['gamma_star']:.2f}" if pd.notnull(r['gamma_star']) else "N/A"
        tex.append(f"{sc_map[sc]} & {pi_str} & {e_str} & {d_str} & {g_str} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    save_latex("\n".join(tex), "tab4_break_even_frontiers.tex")


def main():
    print("=== Step 10: Generating Publication LaTeX Tables ===")
    generate_table2_policy_sustainability()
    generate_table3_monte_carlo()
    generate_table4_break_even()
    print("Step 10 completed successfully.\n")


if __name__ == "__main__":
    main()
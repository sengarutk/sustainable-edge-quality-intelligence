#!/usr/bin/env python3
"""Step 09: LaTeX tables generated from the registry and processed results (results/tables/)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.models.pipeline_evaluator import load_regimes  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_TABLE_DIR, SCENARIOS  # noqa: E402
from src.validation.source_registry import load_registry  # noqa: E402

CLASS_SHORT = {"Measured by this study": "M", "Derived from Paper A": "P", "Literature-derived": "L",
               "Official/public dataset": "O", "Scenario assumption": "A"}
UNIT_TEX = {"kgCO2e": r"kgCO$_2$e", "kgCO2e/kg": r"kgCO$_2$e/kg", "kgCO2e/kWh": r"kgCO$_2$e/kWh",
            "kgCO2e/escape": r"kgCO$_2$e", "s^-1": r"s$^{-1}$", "-": "--"}
SYMBOL_TEX = {
    "N": "N", "f_cam": "f_{cam}", "P_idle": "P_{idle}", "P_active": "P_{active}", "P_host": "P_{host}", "C_hw": "C_{hw}",
    "L_hw": "L_{hw}", "P_station": "P_{station}", "t_review": "t_{review}", "gamma": r"\gamma", "r_L0": "r_{L0}",
    "g": "g", "r_AI": "r_{AI}", "delta_r": r"\delta_r", "Theta": r"\Theta", "pi": r"\pi", "m": "m", "EF_mat": "EF_{mat}",
    "eta": r"\eta", "EF_rec": "EF_{rec}", "e_rw": "e_{rw}", "q0": "q_0", "tau_rw": r"\tau_{rw}", "d_L0": "d_{L0}",
    "t_L0": "t_{L0}", "c_ret": "c_{ret}", "kappa": r"\kappa",
}
CITE = {"this_study": "this study", "none": "--", "see2012visual": r"\cite{see2012visual}", "sengar2026paperA": r"\cite{sengar2026paperA}",
        "ember2024;cea2024": r"\cite{ember2024,cea2024}", "ashby2012materials": r"\cite{ashby2012materials}",
        "allwood2011material": r"\cite{allwood2011material}"}


def num(x):
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    if ax >= 100:
        return f"{x:.0f}"
    return f"{x:.3g}"


def g3(x):
    """Three significant figures; thousands as grouped integers; true minus sign (LaTeX)."""
    if x < 0:
        return "$-$" + g3(-x)
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    if x == 0:
        return "0"
    return f"{x:#.3g}".rstrip(".")


def plain(x, digits):
    """`digits` significant figures, but whole numbers instead of e-notation once |x| >= 10**digits."""
    return f"{x:.0f}" if abs(x) >= 10 ** digits else f"{x:.{digits}g}"


def value_range(r):
    f = g3 if r.unit == "fraction" else num
    if r.low == r.high:
        return f(r.central)
    return rf"{f(r.central)} [{f(r.low)}, {f(r.high)}]"


def symbol(r):
    s = SYMBOL_TEX.get(r.symbol, r.symbol.replace("_", r"\_"))
    base, _, tier = r.symbol.partition("_")
    if base in ("lambda", "a", "d", "b") and tier.startswith("B"):
        s = (r"\lambda" if base == "lambda" else base) + "_{" + tier + "}"
    return f"${s}$"


def save(tex, name):
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_TABLE_DIR / name).write_text(tex + "\n", encoding="utf-8", newline="\n")
    print(f"  {name}")


def table1_ledger():
    recs = load_registry()
    lines = [
        r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
        r"\caption{Parameter ledger (generated from \texttt{data/raw/source\_registry.csv}). Values are central [low, high]; "
        r"Monte Carlo draws are triangular on these bounds. Class: M measured, P derived from our alert-policy study~\cite{sengar2026paperA}, L literature, "
        r"O official dataset, A scenario assumption.}",
        r"\label{tab:parameters}", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.94}",
        r"\begin{tabular}{@{}llllllll@{}}", r"\toprule",
        r"Parameter & Sym. & Unit & A: Precision & B: Machined metal & C: High-value & Cl. & Source \\", r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{Global and edge-cell parameters}} \\",
    ]
    glob = [r for r in recs if r.scope == "all"]
    for r in glob:
        lines.append(rf"{r.parameter.replace('_', ' ')} & {symbol(r)} & {UNIT_TEX.get(r.unit, r.unit)} & "
                     rf"\multicolumn{{3}}{{c}}{{{value_range(r)}}} & {CLASS_SHORT[r.classification.value]} & {CITE.get(r.citation_key, r.citation_key)} \\")
    lines += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Scenario parameters}} \\"]
    per = {}
    for r in recs:
        if r.scope in SCENARIOS:
            per.setdefault(r.parameter, {})[r.scope] = r
    for name, d in per.items():
        r0 = d[SCENARIOS[0]]
        cells = " & ".join(value_range(d[sc]) for sc in SCENARIOS)
        lines.append(rf"{name.replace('_', ' ')} & {symbol(r0)} & {UNIT_TEX.get(r0.unit, r0.unit)} & {cells} & "
                     rf"{CLASS_SHORT[r0.classification.value]} & {CITE.get(r0.citation_key, r0.citation_key)} \\")
    shares = [load_scenario(sc).class_shares for sc in SCENARIOS]
    lines.append(r"defect class shares & $f_A{:}f_B{:}f_C$ & -- & " + " & ".join(":".join(f"{v:.2f}" for v in s) for s in shares) + r" & A & -- \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    save("\n".join(lines), "tab1_parameter_ledger.tex")


POLICY_ROWS = [
    ("nominal_false_alarm_rate", r"False alerts on good parts $\lambda_t$", "1/h"),
    ("glare_alerts_per_burst", r"Alerts per 1--3 frame glare burst $b_t$", "--"),
    ("alerts_per_defect", r"Alerts per defect episode $a_t$", "--"),
    ("detection_delay", r"Mean detection delay $d_t$", "frames"),
]


def table_policies():
    recs = [r for r in load_registry() if r.scope.startswith("policy:")]
    regimes = load_regimes()
    tiers = [rid for rid, reg in regimes.items() if reg.kind == "ai"]
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\scriptsize",
        r"\caption{Alert policies, as measured in~\cite{sengar2026paperA} on PatchCore scores of seven MVTec~AD categories "
        r"(21 paired units; mean [95\% bootstrap interval]). B0 thresholds every frame, B1 adds exponential smoothing, "
        r"B2 adds $k$-of-$N$ persistence, and B3 adds the incident latch, machine-state gating and divergence triage. "
        r"All four detected every sustained defect.}",
        r"\label{tab:policies}", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l" + "c" * len(tiers) + r"@{}}", r"\toprule",
        "Quantity & " + " & ".join(regimes[t].label.split(" ", 1)[0] for t in tiers) + r" \\", r"\midrule",
    ]
    for key, label, unit in POLICY_ROWS:
        cells = []
        for t in tiers:
            r = next(x for x in recs if x.parameter == key and x.scope == f"policy:{t}")
            cells.append(g3(r.central) if r.low == r.high else rf"\shortstack{{{g3(r.central)}\\[-1pt]{{\tiny[{g3(r.low)}, {g3(r.high)}]}}}}")
        lines.append(f"{label} ({unit}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    save("\n".join(lines), "tab_policies.tex")


def table2_regimes():
    df = pd.read_csv(PROCESSED_DIR / "regime_outcomes.csv")
    regimes = load_regimes()
    lines = [
        r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
        r"\caption{Outcomes per functional unit (1{,}000 parts) at central parameter values. $\rho$ is single-operator "
        r"utilisation (values $\geq 1$ require additional operators). Carbon includes material, recycling, rework, "
        r"review, edge electricity, amortised edge hardware and field-escape terms.}",
        r"\label{tab:regimes}", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
        r"Scenario & Regime & $r$ & $d$ (s) & $N_{rw}$ & $N_{esc}$ & $M_{loss}$ (kg) & $E$ (kWh) & $E_{edge}$ (kWh) & $H$ (h) / $\rho$ & $C$ (kgCO$_2$e) \\",
        r"\midrule",
    ]
    for sc in SCENARIOS:
        sub = df[df.scenario == sc].set_index("regime")
        for i, (rid, reg) in enumerate(regimes.items()):
            r = sub.loc[rid]
            name = load_scenario(sc).label if i == 0 else ""
            lines.append(
                rf"{name} & {reg.label} & {r.recall:.2f} & {r.delay_s:.3g} & {r.n_rework:.2f} & {r.n_escape:.2f} & "
                rf"{g3(r.net_loss_kg)} & {g3(r.total_kwh)} & {g3(r.edge_kwh)} & {g3(r.operator_hours)} / {r.rho:.2f} & {g3(r.carbon_total)} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    save("\n".join(lines), "tab2_regime_outcomes.tex")


def fmt_front(fr):
    if fr["status"] == "root":
        v = fr["value"]
        if fr["unit"] == "fraction":
            ppm = v * 1e6
            return f"{plain(ppm, 2)} ppm" if ppm < 1000 else f"{100 * v:.3g}\\%"
        if fr["unit"] == "W":
            return f"{plain(v / 1000, 3)} kW"
        return f"{v:.3g}"
    return r"none ($\Delta C>0$)" if fr["status"] == "none_positive" else r"none ($\Delta C<0$)"


def table3_uncertainty():
    mc = json.loads((PROCESSED_DIR / "monte_carlo_summary.json").read_text(encoding="utf-8"))
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text(encoding="utf-8"))
    n_draws = {mc[sc]["n_draws"] for sc in SCENARIOS}
    if len(n_draws) != 1:
        raise ValueError(f"scenarios use different Monte Carlo sizes {n_draws}")
    draws_tex = f"{n_draws.pop():,}".replace(",", "{,}")
    lines = [
        r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
        rf"\caption{{Monte Carlo results ({draws_tex} draws, all registry parameters) and break-even frontiers at central values. "
        r"$\Delta C$ in kgCO$_2$e per 1{,}000 parts: median [5th, 95th percentile].}",
        r"\label{tab:uncertainty}", r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r" & A: Precision & B: Machined metal & C: High-value \\", r"\midrule",
    ]

    def row(label, fn):
        lines.append(label + " & " + " & ".join(fn(sc) for sc in SCENARIOS) + r" \\")

    def dc(comp):
        def f(sc):
            s = mc[sc]["comparisons"][comp]["delta_c_kgco2e"]
            return f"{g3(s['p50'])} [{g3(s['p05'])}, {g3(s['p95'])}]"
        return f

    def pp(comp):
        return lambda sc: f"{100 * mc[sc]['comparisons'][comp]['p_delta_c_positive']:.1f}\\%"

    row(r"$\Delta C$ B3 vs L0", dc("B3_vs_L0"))
    row(r"$P(\Delta C>0)$ B3 vs L0", pp("B3_vs_L0"))
    row(r"$\Delta C$ B3 vs N0", dc("B3_vs_N0"))
    row(r"$\Delta C$ B3 vs B0", dc("B3_vs_B0"))
    row(r"$P(\Delta C>0)$ B3 vs B0", pp("B3_vs_B0"))
    row(r"$P(\mathrm{SQI}_{bal}>0)$ B3 vs L0", lambda sc: f"{100 * mc[sc]['comparisons']['B3_vs_L0']['p_sqi_balanced_positive']:.1f}\\%")
    row(r"Edge share of B3 carbon", lambda sc: f"{100 * mc[sc]['edge_carbon_share_of_primary_total']['p50']:.2g}\\%")
    lines.append(r"\midrule")
    row(r"$\pi^\star$ vs N0", lambda sc: fmt_front(be[sc]["pi_star_vs_N0"]))
    row(r"$\pi^\star$ vs L0", lambda sc: fmt_front(be[sc]["pi_star_vs_L0"]))
    central = {r.key: r.central for r in load_registry()}
    gpu = central["gpu_idle_power"] + central["gpu_active_power"]

    def cell(sc):
        fr = dict(be[sc]["host_power_star_vs_L0"])
        if fr["status"] == "root":
            fr["value"] += gpu  # report total edge-cell power (GPU + host), as in the text
        return fmt_front(fr)

    row(r"$P^\star_{cell}$ vs L0", cell)
    row(r"$r^\star_{L0}$ (manual recall)", lambda sc: fmt_front(be[sc]["manual_recall_star"]))
    row(r"$\gamma^\star$ in [0, 1.5]", lambda sc: fmt_front(be[sc]["gamma_star_vs_L0"]))
    row(r"$\delta_r^\star$ (recall loss, B3 vs B0)", lambda sc: fmt_front(be[sc]["recall_loss_star_B3_vs_B0"]).replace("\\%", " pp"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    save("\n".join(lines), "tab3_uncertainty_break_even.tex")


def main():
    print("Step 09: generating tables")
    table1_ledger()
    table_policies()
    table2_regimes()
    table3_uncertainty()


if __name__ == "__main__":
    main()

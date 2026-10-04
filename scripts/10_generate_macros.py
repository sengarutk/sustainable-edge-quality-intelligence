#!/usr/bin/env python3
"""Step 10: check the manuscript's qualitative claims against the results and export
every number the manuscript quotes as a LaTeX macro (results/paper_b_generated_metrics.tex).

If a claim stops holding (e.g. after a parameter change) this step fails, forcing the
text to be revisited instead of silently going stale.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.paths import ENERGY_SUMMARY, PROCESSED_DIR, REPO_ROOT, RESULTS_MACROS, SCENARIOS  # noqa: E402
from src.imports.import_paper_a import load_snapshot  # noqa: E402
from src.validation.source_registry import load_registry  # noqa: E402

PREFIX = {"precision_component": "Prec", "machined_metal": "Metal", "high_value_component": "HiVal"}


class ClaimError(AssertionError):
    pass


def claim(cond, msg):
    if not cond:
        raise ClaimError(f"Manuscript claim no longer holds: {msg}")


def sig(x, n=3):
    if 1000 <= abs(x) < 1e7:
        return f"{x:,.0f}".replace(",", "{,}")
    s = f"{x:.{n}g}"
    if "e" in s:
        mant, exp = s.split("e")
        return rf"\ensuremath{{{mant}\times10^{{{int(exp)}}}}}"
    return s


def ppm(frac):
    return sig(frac * 1e6, 2)


def main():
    energy = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    reg = pd.read_csv(PROCESSED_DIR / "regime_outcomes.csv")
    comp = pd.read_csv(PROCESSED_DIR / "comparisons.csv")
    mc = json.loads((PROCESSED_DIR / "monte_carlo_summary.json").read_text(encoding="utf-8"))
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text(encoding="utf-8"))
    prcc = pd.read_csv(PROCESSED_DIR / "prcc_sensitivity.csv")
    records = load_registry()
    central = {r.key: r.central for r in records}
    gpu_w = central["gpu_idle_power"] + central["gpu_active_power"]
    cell_now_w = gpu_w + central["host_power"]

    m = {}
    st = energy["stages"]
    hw = energy["hardware"]
    m["GpuName"] = hw["gpu_name"].replace("NVIDIA GeForce ", "")
    m["GpuDriver"] = hw["driver_version"]
    m["CudaRuntime"] = hw["cuda_runtime"]
    m["TorchVersion"] = hw["torch"].split("+")[0]
    m["BenchRepeats"] = str(energy["repeats"])
    m["BenchWindow"] = f"{energy['window_s']:.0f}"
    m["BenchFps"] = f"{energy['target_fps']:.0f}"
    m["IdlePower"] = f"{st['BASELINE_IDLE']['p_total_w']['median']:.2f}"
    for key, name in (("STAGE_MODEL_INFER", "Infer"), ("STAGE_MODEL_THRESHOLD", "Thr"), ("STAGE_MODEL_POLICY", "Policy"),
                      ("STAGE_FULL_PIPELINE", "Full")):
        m[f"{name}TotalPower"] = f"{st[key]['p_total_w']['median']:.2f}"
        m[f"{name}ActivePower"] = f"{st[key]['p_active_w']['median']:.2f}"
        m[f"{name}FrameEnergy"] = f"{st[key]['e_frame_active_j']['median']:.3f}"
        m[f"{name}MaxFps"] = f"{st[key]['unthrottled']['achieved_fps']:.0f}"
        m[f"{name}LatencyMs"] = f"{st[key]['latency_ms_p50']['median']:.1f}"
    full = st["STAGE_FULL_PIPELINE"]
    m["FullActivePowerMin"] = f"{full['p_active_w']['min']:.2f}"
    m["FullActivePowerMax"] = f"{full['p_active_w']['max']:.2f}"
    m["FullFrameEnergyMin"] = f"{full['e_frame_active_j']['min']:.3f}"
    m["FullFrameEnergyMax"] = f"{full['e_frame_active_j']['max']:.3f}"
    windows = pd.read_csv(REPO_ROOT / energy["trace_dir"] / "windows.csv")
    idle_w = windows[windows.stage == "BASELINE_IDLE"]
    m["StageReadingsMin"] = str(windows[windows.stage != "BASELINE_IDLE"].n_distinct_readings.min())
    m["IdleReadingsMin"] = str(idle_w.n_distinct_readings.min())
    m["CounterIdleMinW"] = f"{idle_w.counter_diag_w.min():.0f}"
    m["CounterIdleMaxW"] = f"{idle_w.counter_diag_w.max():.0f}"
    stage_medians = [st[k]["p_total_w"]["median"] for k in ("STAGE_MODEL_INFER", "STAGE_MODEL_THRESHOLD", "STAGE_MODEL_POLICY", "STAGE_FULL_PIPELINE")]
    m["StageSpreadW"] = f"{max(stage_medians) - min(stage_medians):.2f}"
    claim(max(stage_medians) - min(stage_medians) < 1.0, "pipeline stages differ by < 1 W at 30 FPS")

    m["RegistryEntries"] = str(len(records))
    for cls, name in (("Measured by this study", "Measured"), ("Derived from Paper A", "PaperA"),
                      ("Literature-derived", "Literature"), ("Official/public dataset", "Official"),
                      ("Scenario assumption", "Assumption")):
        m[f"Registry{name}Count"] = str(sum(r.classification.value == cls for r in records))
    n_draws = {mc[sc]["n_draws"] for sc in SCENARIOS}
    claim(len(n_draws) == 1, "all scenarios use the same number of Monte Carlo draws")
    m["MonteCarloDraws"] = f"{n_draws.pop():,}".replace(",", "{,}")
    m["UncertainParams"] = str(sum(r.high > r.low for r in records if r.scope in ("all", "precision_component") or r.scope.startswith("policy:")))

    snap = load_snapshot()
    qa = snap["quoted_macros"]
    m["PaperACommit"] = snap["source_commit"][:7]
    m["PaperAMu"] = qa["MuReviews"]
    m["PaperAGlareRate"] = qa["GlareRateAssumed"]
    m["PaperADelay"] = qa["SustainedDelayFull"]
    m["PaperAFaRaw"] = f"{int(qa['NominalFABaseline']):,}".replace(",", "{,}")
    m["PaperAFaFull"] = qa["NominalFAFull"]
    m["PaperAPerDefectRaw"] = qa["AlertsPerEpisodeBaseline"]
    m["PaperAPerDefectFull"] = qa["AlertsPerEpisodeFull"]
    m["PaperAGlareRaw"] = qa["GlarePerBurstBaseline"]
    m["PaperAGlareKofn"] = qa["GlarePerBurstEmaKofn"]
    m["PaperAGlareFull"] = qa["GlarePerBurstFull"]
    m["PaperALatency"] = qa["LatGpuMean"]
    m["PaperAVisionLatency"] = qa["LatVisionMean"]
    m["PaperABank"] = f"{int(qa['MemoryBankSize']):,}".replace(",", "{,}")
    m["PaperALoadCross"] = qa["LoadCrossFull"]
    claim(all(v == 1.0 for v in snap["sustained_defect_recall"].values()), "Paper A: every tier has recall 1.00 on sustained defects")
    claim(abs(central["review_time"] - 3600.0 / float(qa["MuReviews"])) < 1e-9, "review time matches Paper A's reviewer rate")

    PRIMARY = "B3_Full_Policy"
    shares, esc_shares, headrooms, b12 = [], [], [], []
    for sc in SCENARIOS:
        P = PREFIX[sc]
        r = reg[reg.scenario == sc].set_index("regime")
        c = comp[comp.scenario == sc]

        def row(base, pol=PRIMARY):
            return c[(c.baseline == base) & (c.policy == pol)].iloc[0]

        vL0, vN0, vB0, vB1, vB2 = (row(b) for b in ("L0_Manual", "N0_NoInspection", "B0_Raw", "B1_EMA", "B2_EMA_kofN"))
        b0L0 = row("L0_Manual", "B0_Raw")
        b3 = r.loc[PRIMARY]
        edge_c = b3.carbon_edge + b3.carbon_hardware
        share = edge_c / b3.carbon_total
        esc_share = vL0.wf_escape / vL0.delta_C
        shares.append(share)
        esc_shares.append(esc_share)
        b12.append(abs(r.loc["B2_EMA_kofN"].operator_hours / r.loc["B1_EMA"].operator_hours - 1))
        m[f"{P}Tfu"] = sig(b3.t_fu_h)
        m[f"{P}EdgeKWh"] = sig(b3.edge_kwh)
        m[f"{P}EdgeCarbon"] = sig(edge_c)
        m[f"{P}EdgeShare"] = sig(100 * share, 2)
        m[f"{P}EdgeOfBenefit"] = sig(100 * edge_c / vL0.delta_C, 2)
        m[f"{P}EscapeShare"] = f"{100 * esc_share:.0f}"
        m[f"{P}CarbonLZero"] = sig(r.loc["L0_Manual"].carbon_total)
        m[f"{P}CarbonBZero"] = sig(r.loc["B0_Raw"].carbon_total)
        m[f"{P}CarbonBThree"] = sig(b3.carbon_total)
        m[f"{P}DeltaCvsLZero"] = sig(vL0.delta_C)
        m[f"{P}DeltaCvsNZero"] = sig(vN0.delta_C)
        m[f"{P}DeltaCvsBZero"] = sig(vB0.delta_C)
        m[f"{P}DeltaCvsBOne"] = sig(vB1.delta_C)
        m[f"{P}DeltaCvsBTwo"] = sig(vB2.delta_C)
        m[f"{P}DeltaCBZerovsLZero"] = sig(b0L0.delta_C)
        m[f"{P}DeltaMvsLZero"] = sig(vL0.delta_M)
        m[f"{P}DeltaEvsLZero"] = sig(vL0.delta_E)
        m[f"{P}DeltaHvsLZero"] = sig(vL0.delta_H)
        m[f"{P}DeltaHvsBZero"] = sig(vB0.delta_H)
        m[f"{P}DeltaEvsBZero"] = sig(vB0.delta_E)
        m[f"{P}HoursBZero"] = sig(r.loc["B0_Raw"].operator_hours)
        m[f"{P}HoursBThree"] = sig(b3.operator_hours)
        m[f"{P}HoursLZero"] = sig(r.loc["L0_Manual"].operator_hours)
        m[f"{P}RelCarbonReduction"] = f"{100 * vL0.delta_C / r.loc['L0_Manual'].carbon_total:.0f}"
        m[f"{P}SqiBal"] = f"{vL0.sqi_balanced:+.2f}"
        m[f"{P}SqiRobust"] = f"{100 * vL0.sqi_weight_robustness:.0f}"
        m[f"{P}SE"] = f"{vL0.S_E:+.2f}"
        m[f"{P}SH"] = f"{vL0.S_H:+.2f}"
        for rid, tag in (("B0_Raw", "BZero"), ("B1_EMA", "BOne"), ("B2_EMA_kofN", "BTwo"), (PRIMARY, "BThree"), ("L0_Manual", "LZero")):
            m[f"{P}Rho{tag}"] = f"{r.loc[rid].rho:.2f}"
        m[f"{P}ReviewersBZero"] = str(int(np.ceil(r.loc["B0_Raw"].rho)))
        m[f"{P}ReviewersBThree"] = str(int(np.ceil(b3.rho)))
        mcL = mc[sc]["comparisons"]["B3_vs_L0"]
        mcB = mc[sc]["comparisons"]["B3_vs_B0"]
        m[f"{P}McMedian"] = sig(mcL["delta_c_kgco2e"]["p50"])
        m[f"{P}McLow"] = sig(mcL["delta_c_kgco2e"]["p05"])
        m[f"{P}McHigh"] = sig(mcL["delta_c_kgco2e"]["p95"])
        m[f"{P}McProb"] = f"{100 * mcL['p_delta_c_positive']:.1f}"
        m[f"{P}McProbBZero"] = f"{100 * mcB['p_delta_c_positive']:.0f}"
        m[f"{P}McEdgeShareHigh"] = sig(100 * mc[sc]["edge_carbon_share_of_primary_total"]["p95"], 2)
        f = be[sc]
        m[f"{P}PiStarNZero"] = ppm(f["pi_star_vs_N0"]["value"])
        m[f"{P}PiStarLZero"] = ppm(f["pi_star_vs_L0"]["value"])
        cell_star = f["host_power_star_vs_L0"]["value"] + gpu_w
        headrooms.append(cell_star / cell_now_w)
        m[f"{P}CellPowerStar"] = sig(cell_star / 1000.0)
        m[f"{P}CellHeadroom"] = f"{headrooms[-1]:,.0f}".replace(",", "{,}")
        m[f"{P}RecallLossStar"] = sig(100 * f["recall_loss_star_B3_vs_B0"]["value"], 3)
        top = prcc[prcc.scenario == sc].iloc[0]
        m[f"{P}TopPrcc"] = f"{top.prcc:+.2f}"

        # claims made in the manuscript
        claim(edge_c / vL0.delta_C < 0.01, f"{sc}: edge-cell carbon < 1% of the carbon benefit vs L0")
        claim(share < 0.02, f"{sc}: edge-cell carbon < 2% of the B3 footprint")
        claim(mcL["p_delta_c_positive"] >= 0.95, f"{sc}: P(dC>0, B3 vs L0) >= 95%")
        claim(vB0.delta_C > 0 and vB0.delta_H > 0 and vB0.delta_E > 0, f"{sc}: B3 saves hours, electricity and carbon vs B0")
        claim(r.loc["B0_Raw"].rho > 1, f"{sc}: raw thresholding overloads one reviewer")
        claim(vL0.delta_H < 0, f"{sc}: B3 needs more operator time than manual inspection")
        claim(f["recall_loss_star_B3_vs_B0"]["status"] == "root", f"{sc}: recall-loss break-even exists")
        claim(f["pi_star_vs_N0"]["status"] == "root" and f["pi_star_vs_N0"]["value"] < 1e-4, f"{sc}: pi* vs N0 < 100 ppm")
        claim(f["gamma_star_vs_L0"]["status"] == "none_positive", f"{sc}: no grid break-even below 1.5 kg/kWh")
        claim(f["manual_recall_star"]["status"] == "none_positive", f"{sc}: AI beats even perfect-recall manual inspection")
        claim(vL0.delta_E < 0, f"{sc}: B3 uses more electricity than L0 (rework)")
    prec = reg[reg.scenario == "precision_component"].set_index("regime")
    claim(prec.loc["B0_Raw"].carbon_total > prec.loc["L0_Manual"].carbon_total, "A: raw thresholding emits more than manual inspection")
    claim(prec.loc[PRIMARY].rho > 1 and all(reg[(reg.scenario == sc) & (reg.regime == PRIMARY)].rho.iloc[0] < 1 for sc in SCENARIOS[1:]),
          "B3 overloads one reviewer only in scenario A")
    claim(comp[(comp.scenario == "high_value_component") & (comp.baseline == "B1_EMA") & (comp.policy == PRIMARY)].delta_C.iloc[0] < 0,
          "C: smoothing alone (B1) emits slightly less than B3 (persistence delay costs reworkability)")
    reviewers_b0 = [int(m[f"{PREFIX[sc]}ReviewersBZero"]) for sc in SCENARIOS]
    claim(min(reviewers_b0) >= 10 and max(reviewers_b0) >= 100, "B0 needs tens to hundreds of reviewers")
    pol = {r.key: r.central for r in records if r.scope.startswith("policy:")}
    claim(all(140 <= pol[f"alerts_per_defect@{t}"] <= 160 for t in ("B0_Raw", "B1_EMA", "B2_EMA_kofN")),
          "B0-B2 raise roughly 150 alerts per defect")
    claim(pol["alerts_per_defect@B3_Full_Policy"] < 1.5, "the latch collapses repeat alerts to about one per defect")
    claim(pol["glare_alerts_per_burst@B1_EMA"] > pol["glare_alerts_per_burst@B0_Raw"]
          and pol["nominal_false_alarm_rate@B1_EMA"] < 0.05 * pol["nominal_false_alarm_rate@B0_Raw"],
          "B1 removes most good-part false alerts but raises glare alerts")
    p_prec = load_records_central("precision_component")
    claim(p_prec["defect_prevalence"] * p_prec["line_throughput"] > float(qa["LoadCrossFull"]),
          "scenario A's defect rate exceeds Paper A's single-reviewer crossover")
    stars = {sc: be[sc]["recall_loss_star_B3_vs_B0"]["value"] for sc in SCENARIOS}
    claim(stars["high_value_component"] < 0.015 and stars["precision_component"] > 0.1,
          "recall margin is about one point for high-value parts and wide for lightweight parts")
    m["BOneBTwoHoursDiffMax"] = f"{100 * max(b12):.0f}"
    claim(max(b12) < 0.05, "B1 and B2 operator hours differ by < 5%")
    draws = pd.read_csv(PROCESSED_DIR / "monte_carlo_draws.csv")
    claim((draws.B3_vs_L0_delta_C > 0).all(), "B3 beats L0 in every Monte Carlo draw")
    claim((draws.B3_vs_B0_delta_H > 0).all(), "B3 saves operator hours vs B0 in every draw")
    edge_keys = ["gpu_idle_power", "gpu_active_power", "host_power", "edge_embodied_carbon", "edge_lifetime_hours"]
    edge_prcc = prcc[prcc.parameter.isin(edge_keys)].prcc.abs().max()
    claim(edge_prcc < 0.05, "edge-cell parameters have negligible PRCC")
    m["EdgePrccMax"] = f"{edge_prcc:.2f}"
    for sc in SCENARIOS:
        claim(prcc[prcc.scenario == sc].head(8).p_value.max() < 1e-10, f"{sc}: top-8 PRCC p < 1e-10")
    claim(min(headrooms) > 100, "edge-cell power headroom exceeds 100x in every scenario")
    m["EdgeShareMax"] = sig(100 * max(shares), 2)
    m["EscapeShareMin"] = f"{100 * min(esc_shares):.0f}"
    m["EscapeShareMax"] = f"{100 * max(esc_shares):.0f}"

    lines = ["% Auto-generated by scripts/10_generate_macros.py -- DO NOT EDIT.",
             "% Every number quoted in paper/main.tex is defined here."]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    RESULTS_MACROS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"All manuscript claims hold; wrote {len(m)} macros to {RESULTS_MACROS}")


def load_records_central(scenario):
    from src.params import load_scenario
    return load_scenario(scenario).central()


if __name__ == "__main__":
    main()

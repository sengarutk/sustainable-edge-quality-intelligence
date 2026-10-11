#!/usr/bin/env python3
"""Step 10: check the manuscript's qualitative claims against the results and export
every number the manuscript quotes as a LaTeX macro (results/paper_b_generated_metrics.tex).

If a claim stops holding (e.g. after a parameter change) this step fails, forcing the
text to be revisited instead of silently going stale.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.paths import ENERGY_SUMMARY, JETSON_SUMMARY, PROCESSED_DIR, REPO_ROOT, RESULTS_MACROS, SCENARIOS  # noqa: E402
from src.imports.import_paper_a import load_snapshot, snapshot_rows  # noqa: E402
from src.validation.source_registry import load_registry  # noqa: E402

PREFIX = {"precision_component": "Prec", "machined_metal": "Metal", "high_value_component": "HiVal"}


class ClaimError(AssertionError):
    pass


FAILED_CLAIMS = []


def claim(cond, msg):
    """Raise if a manuscript claim no longer holds; with REPORT_ALL_CLAIMS=1 collect every failure first."""
    if not cond:
        if os.environ.get("REPORT_ALL_CLAIMS"):
            FAILED_CLAIMS.append(msg)
            return
        raise ClaimError(f"Manuscript claim no longer holds: {msg}")


def sig(x, n=3):
    if 1000 <= abs(x) < 1e7:
        return f"{x:,.0f}".replace(",", "{,}")
    s = f"{x:.{n}g}"
    if "e" in s:
        mant, exp = s.split("e")
        return rf"\ensuremath{{{mant}\times10^{{{int(exp)}}}}}"
    return s


PARAM_WORDS = {"material_carbon_factor": "the material factor", "defect_prevalence": "defect prevalence",
               "ai_recall": "detector recall", "part_mass": "part mass", "collateral_multiplier": "the collateral multiplier",
               "grid_carbon_factor": "grid intensity"}


def ppm(frac):
    if frac is None:
        return "none"
    v = frac * 1e6
    return f"{v:,.0f}".replace(",", "{,}") if v >= 100 else sig(v, 2)


def main():
    energy = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    reg = pd.read_csv(PROCESSED_DIR / "regime_outcomes.csv")
    comp = pd.read_csv(PROCESSED_DIR / "comparisons.csv")
    mc = json.loads((PROCESSED_DIR / "monte_carlo_summary.json").read_text(encoding="utf-8"))
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text(encoding="utf-8"))
    prcc = pd.read_csv(PROCESSED_DIR / "prcc_sensitivity.csv")
    records = load_registry()
    central = {r.key: r.central for r in records if r.variant is None}
    gpu_w = central["edge_idle_power"] + central["edge_active_power"]
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
    wd = energy.get("workload_detail", {})
    claim(wd.get("category") not in (None, "synthetic"), "energy was measured on the real PatchCore workload")
    m["BenchCategory"] = wd["category"].replace("_", " ")
    m["BenchBank"] = f"{wd['bank_size']:,}".replace(",", "{,}")
    m["BenchFrames"] = str(wd["n_frames"])
    m["BenchFitImages"] = str(wd["n_fit_images"])
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

    # embedded edge platform (primary) and platform comparison
    js = json.loads(JETSON_SUMMARY.read_text(encoding="utf-8"))
    jr = js["runs"]
    prim = jr[js["primary"]]
    jhw = prim["hardware"]
    m["JetsonModel"] = jhw["device_model"].replace("NVIDIA ", "").replace(" Developer Kit", "")
    m["JetsonPowerMode"] = jhw["nvpmodel"].split("Power Mode:")[1].split("|")[0].strip()
    m["JetsonTensorRT"] = jhw["tensorrt"].rsplit(".", 2)[0]
    m["JetsonLft"] = jhw["l4t_release"].split("(")[0].replace("#", "").strip()
    def jfull(cfg):
        return jr[cfg]["stages"]["STAGE_FULL_PIPELINE"]
    m["JetsonIdlePower"] = f"{prim['stages']['BASELINE_IDLE']['p_total_w']['median']:.2f}"
    m["JetsonFullTotalPower"] = f"{jfull(js['primary'])['p_total_w']['median']:.2f}"
    m["JetsonFullActivePower"] = f"{jfull(js['primary'])['p_active_w']['median']:.2f}"
    for cfg, name in (("patchcore_fp16", "PcHalf"), ("patchcore_fp32", "PcFull"), ("padim_fp16", "PdHalf"), ("padim_fp32", "PdFull")):
        f = jfull(cfg)
        m[f"Jetson{name}FrameMJ"] = f"{1000 * f['e_frame_active_j']['median']:.0f}"
        m[f"Jetson{name}TotalPower"] = f"{f['p_total_w']['median']:.1f}"
        m[f"Jetson{name}MaxFps"] = f"{f['unthrottled']['achieved_fps']:.0f}"
        fid = jr[cfg]["workload_detail"]["fidelity_vs_fp32_reference"]
        claim(fid["decision_agreement"] == 1.0, f"Jetson {cfg}: identical alert decisions to the FP32 reference")
        m[f"Jetson{name}MaxRelErr"] = sig(100 * fid["max_rel_error"], 2)
        claim(f["achieved_fps"]["min"] > 29.5, f"Jetson {cfg} sustains the 30 FPS line rate")
    claim(js["primary"] == "patchcore_fp16", "the deployed Jetson configuration is PatchCore FP16")
    ws_frame = st["STAGE_FULL_PIPELINE"]["e_frame_active_j"]["median"]
    m["WorkstationFrameMJ"] = f"{1000 * ws_frame:.0f}"
    m["FrameEnergyRatio"] = f"{ws_frame / jfull('patchcore_fp16')['e_frame_active_j']['median']:.0f}"
    claim(ws_frame > 5 * jfull("patchcore_fp16")["e_frame_active_j"]["median"], "Jetson FP16 uses >5x less energy per frame")
    plat = pd.read_csv(PROCESSED_DIR / "platform_comparison.csv")
    jp = plat[plat.configuration == "jetson_patchcore_fp16"].set_index("scenario")
    wp = plat[plat.configuration == "workstation_patchcore_fp32"].set_index("scenario")
    ratio = (wp.edge_cell_carbon / jp.edge_cell_carbon)
    claim(f"{ratio.min():.1f}" == f"{ratio.max():.1f}", "the platform carbon ratio is the same in every scenario (quoted as one value)")
    m["PlatformCarbonRatio"] = f"{ratio.min():.1f}"
    m["HostPowerCentral"] = f"{central['host_power']:.0f}"
    m["WsEdgeShareMax"] = sig(100 * wp.edge_share_of_benefit.max(), 2)
    m["JetsonCellPower"] = f"{jp.cell_power_w.iloc[0]:.0f}"
    m["JetsonComputeShare"] = f"{100 * jp.compute_power_w.iloc[0] / jp.cell_power_w.iloc[0]:.0f}"
    m["WsCellPower"] = f"{wp.cell_power_w.iloc[0]:.0f}"
    claim(jp.compute_power_w.iloc[0] < 0.5 * jp.cell_power_w.iloc[0], "on the Jetson, camera/lighting/carrier outweigh compute")
    pchange = plat.groupby("scenario").delta_C_vs_L0.agg(lambda x: (x.max() - x.min()) / x.max())
    m["PlatformDeltaCMax"] = f"{100 * pchange.max():.1f}"
    claim((pchange < 0.02).all(), "platform choice changes Delta C vs L0 by < 2%")

    m["RegistryEntries"] = str(len(records))
    for cls, name in (("Measured by this study", "Measured"), ("Derived from Paper A", "PaperA"),
                      ("Literature-derived", "Literature"), ("Official/public dataset", "Official"),
                      ("Scenario assumption", "Assumption"), ("Scenario definition", "Definition")):
        m[f"Registry{name}Count"] = str(sum(r.classification.value == cls for r in records))
    n_draws = {mc[sc]["n_draws"] for sc in SCENARIOS}
    claim(len(n_draws) == 1, "all scenarios use the same number of Monte Carlo draws")
    m["MonteCarloDraws"] = f"{n_draws.pop():,}".replace(",", "{,}")
    m["UncertainParams"] = str(sum(r.high > r.low for r in records if r.variant is None and
                                   (r.scope in ("all", "precision_component") or r.scope.startswith("policy:"))))

    snap = load_snapshot()
    qa = snap["quoted_macros"]
    pm = snap["policy_metrics"]
    m["PaperACommit"] = snap["source_commit"][:7]
    m["PaperANumPolicies"] = qa["NumPolicies"]
    m["PaperAMu"] = qa["MuReviews"]
    m["PaperAGlareRate"] = qa["GlareRateAssumed"]
    m["PaperADelay"] = f"{pm['mvtec_independent']['FULL_POLICY']['detection_delay']['mean']:.1f}"
    m["PaperAFaRaw"] = sig(pm["mvtec_independent"]["BASELINE"]["nominal_false_alarm_rate"]["mean"])
    claim(central["nominal_false_alarm_rate@B3_Full_Policy"] == 0.0 and
          snapshot_rows()["nominal_false_alarm_rate@policy:B3_Full_Policy"][2] == 0.0,
          "Paper A (independent scores): the full policy raised no false alert on good parts")
    m["PaperAPerDefectRaw"] = f"{pm['mvtec_independent']['BASELINE']['alerts_per_defect']['mean']:.2f}"
    m["PaperAPerDefectFull"] = f"{pm['mvtec_independent']['FULL_POLICY']['alerts_per_defect']['mean']:.2f}"
    m["PaperACorrFaFull"] = f"{pm['mvtec_correlated']['FULL_POLICY']['nominal_false_alarm_rate']['mean']:.1f}"
    m["PaperAGlareFull"] = f"{pm['mvtec_independent']['FULL_POLICY']['glare_alerts_per_burst']['mean']:.2f}"
    m["PaperAGlareTimer"] = f"{pm['mvtec_independent']['DELAY_TIMER']['glare_alerts_per_burst']['mean']:.2f}"
    m["CorrRecallLossFull"] = f"{100 * pm['mvtec_correlated']['FULL_POLICY']['relative_recall_loss']['mean']:.1f}"
    m["EOneRecallLoss"] = f"{100 * central['recall_loss@E1_Delay_Timer']:.1f}"
    claim(abs(central["recall_loss@E1_Delay_Timer"] - central["recall_loss@E3_Decision_Fusion"]) < 1e-12,
          "E1 and E3 carry the same recall loss (quoted once)")
    m["PaperAVisionLatency"] = qa["LatVisionMean"]
    m["PaperABank"] = f"{int(qa['MemoryBankSize']):,}".replace(",", "{,}")
    claim(wd["bank_size"] == int(qa["MemoryBankSize"]), "the energy workload uses Paper A's bank size")
    m["PaperALoadCross"] = qa["LoadCrossFull"]
    m["PaperAShortK"], m["PaperAShortN"] = qa["ShortK"], qa["ShortN"]
    sdr = snap["short_defect_recall"]
    loss = {int(f): 1.0 - sdr["FULL_POLICY"][f] / sdr["BASELINE"][f] for f in sdr["FULL_POLICY"]}
    m["PaperAShortRecallEight"] = f"{100 * sdr['FULL_POLICY']['8']:.0f}"
    m["PaperAShortRecallOne"] = f"{100 * sdr['FULL_POLICY']['1']:.0f}"
    m["PersistLossMax"] = f"{100 * max(0.0, loss[8]):.1f}"
    claim(abs(central["recall_loss@B3_Full_Policy"]) < 1e-12 and
          abs(next(r.high for r in records if r.key == "recall_loss@B3_Full_Policy" and r.variant is None) - max(0.0, loss[8])) < 1e-6,
          "B3's recall loss is centred on zero with the 8-frame short-defect loss as upper bound")
    claim(all(pm["mvtec_independent"][p]["sustained_recall"]["mean"] == 1.0 for p in ("BASELINE", "EMA_ONLY", "EMA_KOFN", "FULL_POLICY")),
          "Paper A: B0-B3 flagged every sustained defect with independent scores")
    claim(abs(central["review_time"] - 3600.0 / float(qa["MuReviews"])) < 1e-9, "review time matches Paper A's reviewer rate")

    PRIMARY = "B3_Full_Policy"
    shares, esc_shares, headrooms, b12, ebf = [], [], [], [], []
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
        rl = f["recall_loss_star_B3_vs_B0"]
        m[f"{P}RecallLossStar"] = sig(100 * rl["value"], 3) if rl["status"] == "root" else "none"
        if rl["status"] == "root":
            sdr = load_snapshot()["short_defect_recall"]
            ok = sorted(int(f) for f in sdr["FULL_POLICY"] if 1.0 - sdr["FULL_POLICY"][f] / sdr["BASELINE"][f] < rl["value"])
            frames_needed = next(f for f in ok if all(g in ok for g in sorted(map(int, sdr["FULL_POLICY"])) if g >= f))
            m[f"{P}VisibleFrames"] = str(frames_needed)
            m[f"{P}VisibleMs"] = f"{1000 * frames_needed / central['camera_fps']:.0f}"
        if sc == "high_value_component":
            claim(rl["status"] == "root" and rl["value"] < 0.01, "C: recall-loss margin is a fraction of a percentage point")
        top = prcc[prcc.scenario == sc].iloc[0]
        m[f"{P}TopPrcc"] = f"{top.prcc:+.2f}"

        # claims made in the manuscript
        ebf.append(edge_c / vL0.delta_C)
        claim(edge_c / vL0.delta_C < 0.02, f"{sc}: edge-cell carbon < 2% of the carbon benefit vs L0")
        claim(share < 0.02, f"{sc}: edge-cell carbon < 2% of the B3 footprint")
        mcN = mc[sc]["comparisons"]["B3_vs_N0"]
        m[f"{P}McProbNZero"] = f"{100 * mcN['p_delta_c_positive']:.1f}"
        claim(mcN["p_delta_c_positive"] >= 0.99, f"{sc}: P(dC>0, B3 vs N0) >= 99%")
        claim(0.5 < mcL["p_delta_c_positive"] < 0.75, f"{sc}: B3 beats L0 in only 50-75% of draws")
        m[f"{P}McProbBThree"] = f"{100 * mcL['p_delta_c_positive']:.1f}"
        claim(vB0.delta_C > 0 and vB0.delta_H > 0 and vB0.delta_E > 0, f"{sc}: B3 saves hours, electricity and carbon vs B0")
        claim(r.loc["B0_Raw"].rho > 1, f"{sc}: raw thresholding overloads one reviewer")
        claim(vL0.delta_H < 0, f"{sc}: B3 needs more operator time than manual inspection")
        claim(rl["status"] == ("none_positive" if sc == "precision_component" else "root"),
              f"{sc}: recall-loss break-even exists for B and C; for A, B3 stays ahead of B0 for any recall loss")
        claim(f["pi_star_vs_N0"]["status"] == "root" and f["pi_star_vs_N0"]["value"] < 1e-3, f"{sc}: pi* vs N0 < 0.1%")
        g = f["gamma_star_vs_L0"]
        if sc == "precision_component":
            claim(g["status"] == "root" and 0.5 < g["value"] < 0.82, "A: grid break-even lies between 0.5 and the 0.82 kg/kWh of coal power")
            m["PrecGammaStar"] = f"{g['value']:.2f}"
        else:
            claim(g["status"] == "none_positive", f"{sc}: no grid break-even below 1.5 kg/kWh")
        mr = f["manual_recall_star"]
        if sc == "precision_component":
            claim(mr["status"] == "none_positive", "A: AI beats even perfect-recall manual inspection")
            claim(f["pi_star_vs_L0"]["status"] == "none_positive", "A: B3 beats L0 at any prevalence")
        else:
            claim(mr["status"] == "root" and 0.80 < mr["value"] < 1.0, f"{sc}: manual inspection would need recall above the 0.70-0.80 literature range")
            claim(f["pi_star_vs_L0"]["status"] == "root", f"{sc}: a prevalence break-even vs L0 exists")
        m[f"{P}ManualRecallStar"] = f"{mr['value']:.2f}" if mr["status"] == "root" else "none"
        claim(vL0.delta_E < 0, f"{sc}: B3 uses more electricity than L0 (rework)")
    prec = reg[reg.scenario == "precision_component"].set_index("regime")
    claim(prec.loc["B0_Raw"].carbon_total > prec.loc["L0_Manual"].carbon_total, "A: raw thresholding emits more than manual inspection")
    claim(prec.loc[PRIMARY].rho > 1 and all(reg[(reg.scenario == sc) & (reg.regime == PRIMARY)].rho.iloc[0] < 1 for sc in SCENARIOS[1:]),
          "B3 overloads one reviewer only in scenario A")
    b3b1 = comp[(comp.baseline == "B1_EMA") & (comp.policy == PRIMARY)]
    claim((b3b1[b3b1.scenario != "high_value_component"].delta_C > 0).all(), "B3 emits less than smoothing alone (B1) in A and B")
    hv_b1 = b3b1[b3b1.scenario == "high_value_component"].iloc[0]
    claim(hv_b1.delta_M < 0, "C: the persistence delay of B3 costs some material relative to B1")
    m["HiValDelayMaterialCost"] = sig(-hv_b1.delta_M)
    hv = reg[reg.scenario == "high_value_component"].set_index("regime")
    claim(hv_b1.delta_C < 0 and -hv_b1.delta_C < 0.001 * hv.loc[PRIMARY].carbon_total,
          "C: B1 emits marginally (<0.1%) less than B3 because the persistence delay costs reworkability")
    claim(hv.loc["B1_EMA"].rho > 2 and hv.loc[PRIMARY].rho < 1, "C: B1 needs several reviewers, B3 fewer than one")
    m["HiValBOneCarbonEdge"] = sig(-hv_b1.delta_C, 2)
    m["HiValReviewersBOne"] = str(int(np.ceil(hv.loc["B1_EMA"].rho)))
    reviewers_b0 = [int(m[f"{PREFIX[sc]}ReviewersBZero"]) for sc in SCENARIOS]
    claim(min(reviewers_b0) >= 10 and max(reviewers_b0) >= 100, "B0 needs tens to hundreds of reviewers")
    pol = {r.key: r.central for r in records if r.scope.startswith("policy:") and r.variant is None}
    apd = [pol[f"alerts_per_defect@{t}"] for t in ("B0_Raw", "B1_EMA", "B2_EMA_kofN")]
    m["AlertsPerDefectMin"] = f"{min(apd):.0f}"
    m["AlertsPerDefectMax"] = f"{max(apd):.0f}"
    claim(all(100 <= v <= 200 for v in apd), "B0-B2 raise well over a hundred alerts per defect")
    claim(abs(pol["alerts_per_defect@B3_Full_Policy"] - 1.0) < 0.005, "the latch collapses repeat alerts to one per defect")
    claim(pol["glare_alerts_per_burst@B1_EMA"] > pol["glare_alerts_per_burst@B0_Raw"]
          and pol["nominal_false_alarm_rate@B1_EMA"] < 0.05 * pol["nominal_false_alarm_rate@B0_Raw"],
          "B1 removes most good-part false alerts but raises glare alerts")
    p_prec = load_records_central("precision_component")
    claim(p_prec["defect_prevalence"] * p_prec["line_throughput"] > float(qa["LoadCrossFull"]),
          "scenario A's defect rate exceeds Paper A's single-reviewer crossover")
    stars = {sc: be[sc]["recall_loss_star_B3_vs_B0"]["value"] for sc in SCENARIOS}
    claim(stars["precision_component"] is None and stars["high_value_component"] < 0.01 < 0.04 < stars["machined_metal"] < 0.08,
          "relative recall margin: none needed for A, a few percent for B, under 1% for C")
    m["BOneBTwoHoursDiffMax"] = f"{100 * max(b12):.0f}"

    # measured detector operating points (two detectors x MVTec AD + VisA)
    ref_all = pd.read_csv(PROCESSED_DIR / "detector_reference_points.csv")
    ref = ref_all[ref_all.detector == "patchcore"]
    det_meta = json.loads((REPO_ROOT / "data" / "raw" / "detector_scores" / "run_meta.json").read_text(encoding="utf-8"))
    m["DetCategories"] = str(len(ref))
    m["DetSeeds"] = str(len(det_meta["seeds"]))
    for ds, name in (("mvtec", "Mvtec"), ("visa", "Visa")):
        m[f"Det{name}Categories"] = str((ref.dataset == ds).sum())
        for det, dn in (("patchcore", "Pc"), ("padim", "Pd")):
            sub = ref_all[(ref_all.dataset == ds) & (ref_all.detector == det)]
            m[f"Det{dn}{name}AurocMedian"] = f"{sub.auroc.median():.3f}"
            m[f"Det{dn}{name}RecallMedian"] = f"{sub.recall_q99.median():.2f}"
    m["DetAurocMin"] = f"{ref.auroc.min():.3f}"
    m["DetAurocMax"] = f"{ref.auroc.max():.3f}"
    m["DetRecallMin"] = f"{ref.recall_q99.min():.2f}"
    m["DetRecallMedian"] = f"{ref.recall_q99.median():.3f}"
    m["DetRecallMax"] = f"{ref.recall_q99.max():.2f}"
    m["DetFprMin"] = f"{100 * ref.fpr_q99.min():.1f}"
    m["DetFprMax"] = f"{100 * ref.fpr_q99.max():.1f}"
    m["DetWorstCategory"] = ref.sort_values("recall_q99").category.iloc[0].replace("_", " ")
    banks = [v["state_size"] for k, v in det_meta["runs"].items() if k.startswith("patchcore/")]
    m["DetBankMin"] = f"{min(banks):,}".replace(",", "{,}")
    m["DetBankMax"] = f"{max(banks):,}".replace(",", "{,}")
    m["DetGoodEvalMin"] = str(int(ref.n_good_eval.min()))
    pd_ref = ref_all[ref_all.detector == "padim"].set_index(["dataset", "category"])
    pc_ref = ref.set_index(["dataset", "category"])
    m["DetPcBeatsPdAuroc"] = str(int((pc_ref.auroc > pd_ref.auroc.reindex(pc_ref.index)).sum()))
    claim((pc_ref.auroc >= pd_ref.auroc.reindex(pc_ref.index) - 0.02).all(), "PatchCore is at least as accurate as PaDiM in every category")
    claim(ref[ref.dataset == "mvtec"].auroc.median() > ref[ref.dataset == "visa"].auroc.median(), "VisA is harder than MVTec AD")
    claim(central["ai_recall"] == round(ref.recall_q99.median(), 4), "registry detector recall is the measured median")

    # carbon-optimal thresholds
    opt = pd.read_csv(PROCESSED_DIR / "carbon_optimal_thresholds.csv")
    for sc in SCENARIOS:
        P = PREFIX[sc]
        o = opt[opt.scenario == sc]
        m[f"{P}OptSavingMax"] = sig(o.saving_vs_q99.max())
        m[f"{P}OptSavingCapMax"] = sig(o.saving_cap_vs_q99.max())
        m[f"{P}OptRhoMax"] = f"{o.rho_opt.max():.1f}"
        m[f"{P}OptSavingCapShare"] = f"{100 * o.saving_cap_vs_q99.sum() / o.saving_vs_q99.sum():.0f}" if o.saving_vs_q99.sum() > 0 else "100"
    m["OptFprMedian"] = f"{100 * opt.fpr_opt.median():.1f}"
    m["OptFprMax"] = f"{100 * opt.fpr_opt.max():.0f}"
    feas = opt[opt.budget_feasible.astype(bool)]
    nfeas = {sc: int(opt[opt.scenario == sc].budget_feasible.sum()) for sc in SCENARIOS}
    claim(len(set(nfeas.values())) == 1, "the alarm budget is attainable in the same categories in every scenario")
    m["BudgetFeasible"] = str(nfeas[SCENARIOS[0]])
    m["BudgetFprMedian"] = f"{100 * feas.fpr_budget.median():.2f}"
    m["OptMorePermissive"] = str(int((opt.fpr_opt > opt.fpr_q99 + 1e-12).sum()))
    m["OptPairs"] = str(len(opt))
    m["OptCapRecallMedian"] = f"{opt.recall_cap.median():.2f}"
    m["ManualRecallCentral"] = f"{central['manual_inspection_recall']:.2f}"
    claim(opt.recall_cap.median() > 0.8, "capped carbon-optimal thresholds lift median recall above the human range")
    worst = opt[opt.scenario == SCENARIOS[-1]].sort_values("recall_q99").iloc[0]
    m["OptWorstCategory"] = worst.category.replace("_", " ")
    m["OptWorstDataset"] = {"mvtec": "MVTec~AD", "visa": "VisA"}[worst.dataset]
    m["OptRecallWorstQ"] = f"{worst.recall_q99:.2f}"
    m["OptRecallWorstOpt"] = f"{worst.recall_opt:.2f}"
    m["OptFprWorst"] = f"{100 * worst.fpr_opt:.0f}"
    for sc in ("machined_metal", "high_value_component"):
        o = opt[opt.scenario == sc]
        claim(o.saving_cap_vs_q99.sum() >= 0.95 * o.saving_vs_q99.sum(),
              f"{sc}: the staffing-capped optimum captures >= 95% of the attainable saving")
    hv = opt[opt.scenario == "high_value_component"]
    claim(abs(hv.saving_cap_vs_q99.max() - hv.saving_vs_q99.max()) < 1e-9,
          "C: the largest saving is reached without additional reviewers")
    claim((opt[opt.scenario == "precision_component"].rho_opt > 1).any(), "on the fast line (A) staffing binds")
    claim(((feas.fpr_opt >= feas.fpr_budget - 1e-12)).all(), "where attainable, carbon-optimal thresholds are never stricter than the alarm budget")
    claim((opt.fpr_q99 > 0.01).mean() > 0.5, "the q99 false-positive rate exceeds 1% in most categories")
    claim((opt.carbon_opt <= opt.carbon_q99 + 1e-9).all() and (opt.carbon_cap <= opt.carbon_q99 + 1e-9).all(),
          "optimised thresholds never emit more than q99")

    # population study
    pop = pd.read_csv(PROCESSED_DIR / "population.csv")
    share = pop.edge_share_of_benefit_L0.dropna()
    m["PopSize"] = f"{len(pop):,}".replace(",", "{,}")
    m["PopWinNZero"] = f"{100 * (pop.dc_N0 > 0).mean():.1f}"
    m["PopWinLZero"] = f"{100 * (pop.dc_L0 > 0).mean():.1f}"
    m["PopEdgeShareMedian"] = sig(100 * share.median(), 2)
    m["PopEdgeShareNinetyFive"] = sig(100 * share.quantile(0.95), 2)
    m["PopEdgeShareAboveOne"] = f"{100 * (share > 0.01).mean():.0f}"
    hi_share = pop[pop.edge_share_of_benefit_L0 > 0.01].dc_L0
    lo_share = pop[pop.edge_share_of_benefit_L0 <= 0.01].dc_L0
    m["PopHighShareBenefitMedian"] = sig(hi_share.median())
    m["PopLowShareBenefitMedian"] = sig(lo_share.median())
    claim(hi_share.median() < 0.1 * lo_share.median(), "products with an edge share above 1% have a much smaller benefit")
    claim((((pop.psi > 1) == (pop.dc_N0 > 0)) | (pop.beta <= 0)).all(), "decision rule psi > 1 <=> Delta C > 0 for every product")
    claim(max(b12) < 0.05, "B1 and B2 operator hours differ by < 5%")
    draws = pd.read_csv(PROCESSED_DIR / "monte_carlo_draws.csv")
    claim(all(mc[sc]["comparisons"]["B3_vs_N0"]["p_delta_c_positive"] >= 0.99 for sc in SCENARIOS),
          "B3 beats N0 in >= 99% of draws in every scenario")
    claim(all((draws[draws.scenario == sc].B3_vs_N0_delta_C > 0).all() for sc in SCENARIOS), "B3 beats N0 in every draw")
    claim(central["manual_inspection_recall"] < central["ai_recall"] < central["manual_inspection_recall"] + 0.1,
          "median detector recall lies just above the manual recall")
    claim(abs(be["precision_component"]["gamma_star_vs_L0"]["value"] - 0.716) < 0.05, "A: grid break-even is about India's average")
    ref_pc = pd.read_csv(PROCESSED_DIR / "detector_reference_points.csv").query("detector == 'patchcore'")
    star_max = max(be[sc]["manual_recall_star"]["value"] for sc in SCENARIOS[1:])
    m["DetAboveManualStar"] = str(int((ref_pc.recall_q99 > star_max).sum()))
    m["EdgeOfBenefitMax"] = sig(100 * max(ebf), 2)
    m["EdgeBenefitRatioMin"] = f"{1 / max(ebf):,.0f}".replace(",", "{,}")
    m["EdgeBenefitRatioMax"] = f"{round(1 / min(ebf), -2):,.0f}".replace(",", "{,}")
    pl = [mc[sc]["comparisons"]["B3_vs_L0"]["p_delta_c_positive"] for sc in SCENARIOS]
    m["McProbLMin"] = f"{100 * min(pl):.0f}"
    m["McProbLMax"] = f"{100 * max(pl):.0f}"
    pb = {sc: mc[sc]["comparisons"]["B3_vs_B0"]["p_delta_c_positive"] for sc in SCENARIOS}
    claim(pb["precision_component"] > 0.95 and pb["machined_metal"] > 0.95 and pb["high_value_component"] < 0.5,
          "B3 beats B0 in >95% of draws in A and B but in under half in C (recall margin smaller than the assumed loss range)")
    claim((draws.B3_vs_B0_delta_H > 0).all(), "B3 saves operator hours vs B0 in every draw")
    edge_keys = ["edge_idle_power", "edge_active_power", "host_power", "edge_embodied_carbon", "edge_lifetime_hours"]
    edge_prcc = prcc[prcc.parameter.isin(edge_keys)].prcc.abs().max()
    claim(edge_prcc < 0.05, "edge-cell parameters have negligible PRCC")
    m["EdgePrccMax"] = f"{edge_prcc:.2f}"
    for sc in SCENARIOS:
        claim(prcc[prcc.scenario == sc].head(8).p_value.max() < 1e-10, f"{sc}: top-8 PRCC p < 1e-10")
    claim(min(headrooms) > 100, "edge-cell power headroom exceeds 100x in every scenario")
    m["EdgeShareMax"] = sig(100 * max(shares), 2)
    m["EscapeShareMin"] = f"{100 * min(esc_shares):.0f}"
    m["EscapeShareMax"] = f"{100 * max(esc_shares):.0f}"

    # Sobol variance decomposition
    sob = pd.read_csv(PROCESSED_DIR / "sobol_indices.csv")
    inter = {sc: 1.0 - sob[sob.scenario == sc].S1.sum() for sc in SCENARIOS}
    m["SobolBaseRows"] = f"{int(sob.n_base.iloc[0]):,}".replace(",", "{,}")
    m["SobolInteractionMax"] = f"{100 * max(inter.values()):.0f}"
    m["SobolInteractionMin"] = f"{100 * min(inter.values()):.0f}"
    for sc in SCENARIOS:
        top = sob[sob.scenario == sc].sort_values("ST", ascending=False).iloc[0]
        m[f"{PREFIX[sc]}SobolTop"] = PARAM_WORDS.get(top.parameter, top.parameter.replace("_", " "))
        m[f"{PREFIX[sc]}SobolTopST"] = f"{top.ST:.2f}"
    rec = sob[sob.parameter == "ai_recall"]
    m["RecallSobolMin"] = f"{rec.ST.min():.2f}"
    m["RecallSobolMax"] = f"{rec.ST.max():.2f}"
    claim(all(sob[sob.scenario == sc].sort_values("ST").parameter.iloc[-1] == "ai_recall" for sc in SCENARIOS)
          and rec.ST.min() > 0.8, "detector recall explains most of the variance of Delta C vs L0 in every scenario")
    edge_st = sob[sob.parameter.isin(edge_keys)].ST.max()
    m["EdgeSobolMax"] = sig(edge_st, 1)
    claim(edge_st < 0.01, "edge-cell parameters explain < 1% of the variance of Delta C")
    claim(all(0 <= v < 0.25 for v in inter.values()), "interactions explain less than a quarter of the variance")

    # cry-wolf (reviewer compliance)
    cw_all = pd.read_csv(PROCESSED_DIR / "cry_wolf_summary.csv")
    cw = cw_all[cw_all.score_model == "independent"]
    b3 = cw[cw.tier == PRIMARY].set_index("scenario")
    for sc in SCENARIOS:
        m[f"{PREFIX[sc]}PpvBThree"] = f"{100 * b3.loc[sc, 'ppv']:.0f}"
        m[f"{PREFIX[sc]}OmegaStar"] = f"{b3.loc[sc, 'omega_star']:.2f}" if b3.loc[sc, "status"] == "root" else "none"
    claim(b3.loc["precision_component", "status"] == "none_positive", "A: B3 stays ahead of L0 even under full probability matching")
    claim(all(b3.loc[sc, "status"] == "root" for sc in SCENARIOS[1:]), "B, C: a cry-wolf break-even exists for B3")
    claim(b3.loc["precision_component", "ppv"] > b3.loc[list(SCENARIOS[1:]), "ppv"].max(), "B3 alerts are most precise on the fast line")
    ppv = cw.pivot(index="scenario", columns="tier", values="ppv")
    b3c = cw_all[(cw_all.score_model == "correlated") & (cw_all.tier == PRIMARY)].set_index("scenario")
    for sc in SCENARIOS:
        m[f"{PREFIX[sc]}CorrPpvBThree"] = sig(100 * b3c.loc[sc, "ppv"], 2)
        m[f"{PREFIX[sc]}CorrOmegaStar"] = f"{b3c.loc[sc, 'omega_star']:.2f}" if b3c.loc[sc, "status"] == "root" else "none"
    claim(all(b3c.loc[sc, "status"] == "root" for sc in SCENARIOS), "correlated scores: a cry-wolf break-even exists for B3 in every scenario")
    claim(all(b3c.loc[sc, "omega_star"] < b3.loc[sc, "omega_star"] for sc in SCENARIOS[1:]), "correlated scores lower omega* in B and C")
    claim(all(ppv.loc[sc, "B1_EMA"] > ppv.loc[sc, PRIMARY] > ppv.loc[sc, "B0_Raw"] and ppv.loc[sc, "B2_EMA_kofN"] > ppv.loc[sc, PRIMARY]
              for sc in SCENARIOS), "PPV ordering: B1, B2 > B3 > B0 in every scenario")

    # nomogram: break-even defect rate x embodied carbon per defect
    nomo = pd.read_csv(PROCESSED_DIR / "nomogram.csv")
    jc = nomo[(nomo.variant == "jetson_central") & np.isfinite(nomo.defects_per_hour_star)]
    prod = jc.defects_per_hour_star * jc.embodied_per_part
    m["NomogramGramsPerHour"] = f"{1000 * prod.median():.0f}"
    claim(prod.iloc[len(prod) // 2:].std() / prod.iloc[len(prod) // 2:].mean() < 0.05,
          "break-even defect rate is inversely proportional to embodied carbon per part (heavy parts)")
    margins = []
    for sc in SCENARIOS:
        pc = load_records_central(sc)
        x = pc["part_mass"] * pc["material_carbon_factor"]
        star = np.exp(np.interp(np.log(x), np.log(jc.embodied_per_part), np.log(jc.defects_per_hour_star)))
        margins.append(pc["defect_prevalence"] * pc["line_throughput"] / star)
    claim(100 <= min(margins) and max(margins) <= 1e5, "scenarios lie two to five orders of magnitude above the break-even line")
    cen = nomo[nomo.variant == "jetson_central"].defects_per_hour_star.values
    for v in ("workstation_central", "jetson_high_grid"):
        r = nomo[nomo.variant == v].defects_per_hour_star.values / cen
        claim(np.nanmax(r[np.isfinite(r)]) < 3, f"{v} shifts the break-even line by less than 3x")

    # does more compute buy recall?
    cr = pd.read_csv(PROCESSED_DIR / "compute_recall.csv")
    crs = pd.read_csv(PROCESSED_DIR / "compute_recall_summary.csv")
    one = cr[cr.scenario == SCENARIOS[0]]
    tags = {"patchcore": "Pc", "patchcore_448": "Hires", "patchcore_wrn50": "Wrn", "padim": "Pd"}
    for det, tag in tags.items():
        sub = one[one.detector == det]
        m[f"Cr{tag}Recall"] = f"{sub.recall_q99.median():.2f}"
        m[f"Cr{tag}VisaRecall"] = f"{sub[sub.dataset == 'visa'].recall_q99.median():.2f}"
        m[f"Cr{tag}MvtecRecall"] = f"{sub[sub.dataset == 'mvtec'].recall_q99.median():.2f}"
        m[f"Cr{tag}WorstRecall"] = f"{sub.recall_q99.min():.2f}"
        m[f"Cr{tag}FrameMJ"] = f"{sub.frame_mj.iloc[0]:.0f}"
        m[f"Cr{tag}ActiveW"] = f"{sub.active_w.iloc[0]:.2f}"
        for sc in SCENARIOS:
            v = crs[(crs.scenario == sc) & (crs.detector == det)].iloc[0]
            m[f"{PREFIX[sc]}Cr{tag}Share"] = f"{100 * v.share_beating_L0:.0f}"
    pc, hi, wr = (one[one.detector == d] for d in ("patchcore", "patchcore_448", "patchcore_wrn50"))
    m["CrHiresEnergyRatio"] = f"{hi.frame_mj.iloc[0] / pc.frame_mj.iloc[0]:.1f}"
    m["CrWrnEnergyRatio"] = f"{wr.frame_mj.iloc[0] / pc.frame_mj.iloc[0]:.1f}"
    extra = []
    gain = []
    for sc in SCENARIOS:
        a = crs[(crs.scenario == sc) & (crs.detector == "patchcore")].iloc[0]
        b = crs[(crs.scenario == sc) & (crs.detector == "patchcore_448")].iloc[0]
        extra.append(b.edge_cell_carbon - a.edge_cell_carbon)
        gain.append(b.delta_C_mean - a.delta_C_mean)
        m[f"{PREFIX[sc]}CrHiresGain"] = sig(b.delta_C_mean - a.delta_C_mean)
        m[f"{PREFIX[sc]}CrHiresExtra"] = sig(b.edge_cell_carbon - a.edge_cell_carbon, 2)
        claim(b.share_beating_L0 >= a.share_beating_L0, f"{sc}: 448 px beats manual in at least as many categories")
    m["CrHiresPaybackMin"] = f"{min(g / e for g, e in zip(gain, extra)):,.0f}".replace(",", "{,}")
    claim(all(e > 0 for e in extra) and all(g > 10 * e for g, e in zip(gain, extra)),
          "the extra electricity of 448 px is repaid more than tenfold by its recall gain")
    claim(hi[hi.dataset == "visa"].recall_q99.median() > pc[pc.dataset == "visa"].recall_q99.median() + 0.1
          and hi.recall_q99.min() > pc.recall_q99.min(), "448 px raises VisA recall and the worst category")
    claim(wr[wr.dataset == "visa"].recall_q99.median() < pc[pc.dataset == "visa"].recall_q99.median() + 0.05
          and wr[wr.dataset == "mvtec"].recall_q99.median() > pc[pc.dataset == "mvtec"].recall_q99.median(),
          "the larger backbone helps on MVTec AD but not on VisA")

    # correlated scores (Paper A, rho = 0.9) and textbook alarm baselines
    mcc = json.loads((PROCESSED_DIR / "monte_carlo_summary_correlated_scores.json").read_text(encoding="utf-8"))
    ad = pd.read_csv(PROCESSED_DIR / "alarm_designs.csv")

    def adr(model, sc, t):
        return ad[(ad.score_model == model) & (ad.scenario == sc) & (ad.regime == t)].iloc[0]

    pc = [mcc[sc]["comparisons"]["B3_vs_L0"]["p_delta_c_positive"] for sc in SCENARIOS]
    m["CorrMcProbLMin"] = f"{100 * min(pc):.0f}"
    m["CorrMcProbLMax"] = f"{100 * max(pc):.0f}"
    claim(all(c < i for c, i in zip(pc, pl)), "correlated scores lower the probability that B3 beats manual inspection in every scenario")
    claim(all(mcc[sc]["comparisons"]["B3_vs_N0"]["p_delta_c_positive"] == 1.0 for sc in SCENARIOS),
          "correlated scores: B3 still beats no inspection in every draw")
    rho_c = [adr("correlated", sc, PRIMARY).rho for sc in SCENARIOS]
    rev_c = [adr("correlated", sc, PRIMARY).reviewers for sc in SCENARIOS]
    m["CorrRhoBThreeMin"], m["CorrRhoBThreeMax"] = f"{min(rho_c):.1f}", f"{max(rho_c):.1f}"
    m["CorrReviewersBThreeMin"], m["CorrReviewersBThreeMax"] = str(min(rev_c)), str(max(rev_c))
    claim(min(rev_c) > 1, "correlated scores: B3 needs more than one reviewer in every scenario")
    for sc in SCENARIOS:
        P = PREFIX[sc]
        m[f"{P}CorrDeltaCvsLZero"] = sig(adr("correlated", sc, PRIMARY).delta_C_vs_L0)
        m[f"{P}CorrMcProb"] = f"{100 * mcc[sc]['comparisons']['B3_vs_L0']['p_delta_c_positive']:.1f}"
        claim(0 < adr("correlated", sc, PRIMARY).delta_C_vs_L0 < adr("independent", sc, PRIMARY).delta_C_vs_L0,
              f"{sc}: correlated scores shrink but do not reverse B3's central saving over L0")
    m["MetalPpvEOne"] = f"{100 * adr('independent', 'machined_metal', 'E1_Delay_Timer').ppv:.0f}"
    m["HiValPpvEOne"] = f"{100 * adr('independent', 'high_value_component', 'E1_Delay_Timer').ppv:.0f}"
    claim(all(adr("independent", sc, "E1_Delay_Timer").ppv > adr("independent", sc, PRIMARY).ppv for sc in SCENARIOS[1:]),
          "E1 alerts are more precise than B3's on the slow lines (independent scores)")
    m["MetalDeltaCEOne"] = sig(adr("independent", "machined_metal", "E1_Delay_Timer").delta_C_vs_L0)
    m["HiValDeltaCEOne"] = sig(adr("independent", "high_value_component", "E1_Delay_Timer").delta_C_vs_L0)
    gap = [abs(adr("independent", sc, t).delta_C_vs_L0_late_flags_divert / adr("independent", sc, PRIMARY).delta_C_vs_L0 - 1)
           for sc in SCENARIOS for t in ("E1_Delay_Timer", "E3_Decision_Fusion")]
    m["EOneLateGapMax"] = f"{max(1, int(np.ceil(100 * max(gap))))}"
    claim(all(abs(adr("independent", sc, "E2_EMA_Hysteresis").delta_C_vs_L0 / adr("independent", sc, PRIMARY).delta_C_vs_L0 - 1) < 0.02
              for sc in SCENARIOS), "E2 matches B3's saving within 2% with independent scores")
    best = [ad[(ad.score_model == "correlated") & (ad.scenario == sc)].sort_values("delta_C_vs_L0").regime.iloc[-1] for sc in SCENARIOS]
    claim(all(b == "E2_EMA_Hysteresis" for b in best), "with correlated scores E2 keeps the largest saving over L0 in every scenario")
    m["MetalCorrDeltaCETwo"] = sig(adr("correlated", "machined_metal", "E2_EMA_Hysteresis").delta_C_vs_L0)
    m["HiValCorrDeltaCETwo"] = sig(adr("correlated", "high_value_component", "E2_EMA_Hysteresis").delta_C_vs_L0)
    rev2 = [adr("correlated", sc, "E2_EMA_Hysteresis").reviewers for sc in SCENARIOS]
    m["CorrReviewersETwoMin"], m["CorrReviewersETwoMax"] = str(min(rev2)), str(max(rev2))

    # held-out VisA alert measurements (Paper A, policy frozen), independent and correlated scores
    mcv = {v: json.loads((PROCESSED_DIR / f"monte_carlo_summary_{v}.json").read_text(encoding="utf-8"))
           for v in ("visa_scores", "visa_correlated_scores")}
    for v, tag in (("visa_scores", "Visa"), ("visa_correlated_scores", "VisaCorr")):
        pv = [mcv[v][sc]["comparisons"]["B3_vs_L0"]["p_delta_c_positive"] for sc in SCENARIOS]
        m[f"{tag}McProbLMin"], m[f"{tag}McProbLMax"] = f"{100 * min(pv):.0f}", f"{100 * max(pv):.0f}"
        claim(all(mcv[v][sc]["comparisons"]["B3_vs_N0"]["p_delta_c_positive"] == 1.0 for sc in SCENARIOS),
              f"{v}: B3 still beats no inspection in every draw")
        claim(max(pv) < 0.5, f"{v}: B3 beats manual inspection in fewer than half of the draws in every scenario")
        p2 = [mcv[v][sc]["comparisons"]["E2_vs_L0"]["p_delta_c_positive"] for sc in SCENARIOS]
        m[f"{tag}ETwoMcProbMin"], m[f"{tag}ETwoMcProbMax"] = f"{100 * min(p2):.0f}", f"{100 * max(p2):.0f}"
        claim(all(a > b for a, b in zip(p2, pv)), f"{v}: E2 beats manual inspection more often than B3")
    vm = pm["visa_independent"]
    m["PaperAVisaFaFull"] = f"{vm['FULL_POLICY']['nominal_false_alarm_rate']['mean']:.1f}"
    m["PaperAVisaPerDefectFull"] = f"{vm['FULL_POLICY']['alerts_per_defect']['mean']:.2f}"
    m["PaperAVisaRecallFull"] = f"{vm['FULL_POLICY']['sustained_recall']['mean']:.2f}"
    m["PaperAVisaRecallRaw"] = f"{vm['BASELINE']['sustained_recall']['mean']:.2f}"
    m["PaperAVisaCategories"] = qa["VisaNumCategories"]
    m["VisaRecallLossFull"] = f"{100 * vm['FULL_POLICY']['relative_recall_loss']['mean']:.0f}"
    m["VisaRecallLossTimer"] = f"{100 * vm['DELAY_TIMER']['relative_recall_loss']['mean']:.0f}"
    m["VisaRecallLossHyst"] = f"{100 * vm['EMA_HYSTERESIS']['relative_recall_loss']['mean']:.0f}"
    m["VisaCorrRecallLossFull"] = f"{100 * pm['visa_correlated']['FULL_POLICY']['relative_recall_loss']['mean']:.0f}"
    for sc in SCENARIOS:
        P = PREFIX[sc]
        m[f"{P}VisaDeltaCvsLZero"] = sig(adr("visa_independent", sc, PRIMARY).delta_C_vs_L0)
        m[f"{P}VisaETwoDeltaC"] = sig(adr("visa_independent", sc, "E2_EMA_Hysteresis").delta_C_vs_L0)
        claim(adr("visa_independent", sc, "E2_EMA_Hysteresis").delta_C_vs_L0 > 0, f"{sc}: on VisA, E2 keeps a saving over L0 at central values")
    claim(all(adr("visa_independent", sc, PRIMARY).delta_C_vs_L0 < 0 for sc in SCENARIOS[1:])
          and adr("visa_independent", SCENARIOS[0], PRIMARY).delta_C_vs_L0 > 0,
          "VisA alert measurements: B3 emits more than manual inspection in B and C, less in A")
    rv = [adr("visa_independent", sc, "B0_Raw").reviewers for sc in SCENARIOS]
    m["VisaReviewersBZeroMin"], m["VisaReviewersBZeroMax"] = str(min(rv)), str(max(rv))
    claim(all(adr("visa_independent", sc, PRIMARY).reviewers <= adr("independent", sc, PRIMARY).reviewers for sc in SCENARIOS),
          "VisA: B3 needs no more reviewers than with MVTec alert measurements")
    claim(adr("visa_independent", "high_value_component", "B0_Raw").delta_C_vs_L0 > adr("visa_independent", "high_value_component", PRIMARY).delta_C_vs_L0,
          "VisA: in C, single-frame thresholding saves more carbon than B3 (but needs many reviewers)")

    # correlation stress test
    stv = pd.read_csv(PROCESSED_DIR / "correlation_stress.csv")
    pairs = stv[stv.a.fillna("") != ""]
    m["StressPairs"] = str(len(pairs[["a", "b"]].drop_duplicates()))
    m["StressParams"] = str(len(set(pairs.a) | set(pairs.b)))
    m["StressShiftMax"] = f"{100 * pairs.shift_B3_vs_L0.abs().max():.1f}"
    rr = pairs[(pairs.a == "ai_recall") & (pairs.b == "manual_inspection_recall") & (pairs.rank_corr > 0)]
    m["StressRecallShiftMax"] = f"{100 * rr.shift_B3_vs_L0.max():.1f}"
    claim(pairs.shift_B3_vs_N0.abs().max() == 0.0, "the stress test leaves P(B3 beats N0) unchanged")
    claim(pairs.shift_B3_vs_L0.abs().max() < 0.05, "pairwise correlations move P(B3 beats L0) by less than 5 points")
    claim((rr.shift_B3_vs_L0 > 0).all(), "positive detector-manual recall correlation raises P(B3 beats L0)")

    # structural sensitivity: recycled-content substitution
    sb = pd.read_csv(PROCESSED_DIR / "substitution_sensitivity.csv")
    for sval, tag in ((0.5, "Half"), (0.0, "Zero")):
        q = sb[sb.substitution == sval]
        m[f"Sub{tag}ProbMin"] = f"{100 * q.p_B3_vs_L0.min():.0f}"
        m[f"Sub{tag}ProbMax"] = f"{100 * q.p_B3_vs_L0.max():.0f}"
    m["SubZeroProbNZeroMin"] = f"{np.floor(1000 * sb[sb.substitution == 0.0].p_B3_vs_N0.min()) / 10:.1f}"
    for sc in SCENARIOS:
        q = sb[sb.scenario == sc].set_index("substitution")
        claim(q.loc[0.0, "p_B3_vs_L0"] > q.loc[0.5, "p_B3_vs_L0"] > q.loc[1.0, "p_B3_vs_L0"], f"{sc}: lower substitution raises P(B3 beats L0)")
        claim(q.loc[0.0, "delta_C_B3_vs_N0"] < q.loc[1.0, "delta_C_B3_vs_N0"], f"{sc}: lower substitution shrinks the saving over N0")

    # embodied hardware: break-even multiple of the central estimate, and a doubled estimate (bottom-up truncation)
    chw = central["edge_embodied_carbon"]
    for base, tag in (("L0", "LZero"), ("N0", "NZero")):
        mult = [be[sc][f"hardware_carbon_star_vs_{base}"]["value"] / chw for sc in SCENARIOS
                if be[sc][f"hardware_carbon_star_vs_{base}"]["status"] == "root"]
        claim(len(mult) == len(SCENARIOS), f"a hardware-carbon break-even vs {base} exists in every scenario")
        m[f"HwStarMult{tag}Min"] = f"{min(mult):,.0f}".replace(",", "{,}")
    doubled = []
    for sc in SCENARIOS:
        b3 = reg[(reg.scenario == sc) & (reg.regime == PRIMARY)].iloc[0]
        dl0 = comp[(comp.scenario == sc) & (comp.baseline == "L0_Manual") & (comp.policy == PRIMARY)].iloc[0].delta_C
        doubled.append((b3.carbon_edge + 2 * b3.carbon_hardware) / (dl0 - b3.carbon_hardware))
    m["HwDoubledEdgeOfBenefitMax"] = sig(100 * max(doubled), 2)
    claim(max(doubled) < 0.05, "doubling the embodied hardware carbon keeps the edge cell a small share (<5%) of the benefit")
    claim(next(r.high for r in records if r.key == "edge_embodied_carbon" and r.variant is None) >= 2 * 230.32 - 1e-9,
          "the ledger's hardware upper bound includes the twofold truncation allowance")

    # reproducibility checklist (seeds, versions)
    import importlib.util

    def script_constants(name):
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    s06, s07b = script_constants("06_run_monte_carlo"), script_constants("07b_run_operating_points")
    m["SeedMonteCarlo"] = str(s06.SEED)
    m["SeedPopulation"] = str(s07b.POPULATION_SEED)
    m["SeedEnergy"] = str(energy["seed"])
    claim(all(jr[c]["seed"] == energy["seed"] for c in jr), "the Jetson runs use the same stage-order seed as the workstation run")
    m["DetSeedList"] = ", ".join(str(x) for x in det_meta["seeds"])
    m["JetsonPython"] = jhw["python"]
    m["WsPython"] = hw["python"]
    m["JetsonKernel"] = jhw["kernel"]

    if FAILED_CLAIMS:
        raise ClaimError("Manuscript claims no longer hold:\n  - " + "\n  - ".join(FAILED_CLAIMS))
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

"""
Measured operating points of the PatchCore detector and carbon-optimal thresholds.

Operating curve. For each category and split seed, thresholds are set at quantiles q of the
calibration good-image scores; each threshold gives a per-part recall r (defective test
images scored above it) and false-positive rate f (held-out and test good images above it).
Curves are averaged over seeds on a common q grid.

Carbon-optimal threshold. A one-shot station takes one image per part and diverts the part
when its score exceeds the threshold (no temporal filtering, no glare stream: glare and
other nuisance images are part of the measured false-positive rate). It is evaluated with
the same accounting model as the policy tiers, by overriding the B3 operating point:
recall r, delay 0, one alert per diverted part, and false alerts f (1 - pi) Theta per hour.
For every scenario x category the threshold minimising total carbon is compared with the
calibration 99th percentile and with an alarm budget of 5 false diversions per 1,000 parts,
both unconstrained and under a staffing cap (no more reviewers than the q99 threshold needs).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.experiments.detector_eval import CATEGORIES, SCORE_DIR, SEEDS
from src.models.pipeline_evaluator import PRIMARY_TIER, evaluate_regime, load_regimes
from src.params import Scenario

Q_GRID = 1.0 - np.logspace(-4, np.log10(0.5), 400)  # calibration quantiles 0.5 ... 0.9999
REFERENCE_Q = 0.99
ALARM_BUDGET_PER_1K = 5.0


def load_scores(category: str, seed: int) -> Dict[str, np.ndarray]:
    z = np.load(SCORE_DIR / f"{category}_seed{seed}.npz", allow_pickle=False)
    s, role = z["scores"], z["role"]
    return {"cal": s[role == "calibration"],
            "good": s[np.isin(role, ["heldout_good", "test_good"])],
            "defect": s[role == "test_defect"]}


def curve(category: str, q_grid=Q_GRID) -> pd.DataFrame:
    """Seed-averaged (recall, fpr) along calibration quantiles q."""
    rows = []
    for seed in SEEDS:
        sc = load_scores(category, seed)
        tau = np.quantile(sc["cal"], q_grid)
        rows.append(pd.DataFrame({"q": q_grid, "seed": seed,
                                  "recall": (sc["defect"][None, :] > tau[:, None]).mean(1),
                                  "fpr": (sc["good"][None, :] > tau[:, None]).mean(1)}))
    df = pd.concat(rows)
    return df.groupby("q", as_index=False)[["recall", "fpr"]].mean().assign(category=category)


def auroc(category: str) -> float:
    vals = []
    for seed in SEEDS:
        sc = load_scores(category, seed)
        d, g = sc["defect"], sc["good"]
        vals.append(((d[:, None] > g[None, :]).mean() + 0.5 * (d[:, None] == g[None, :]).mean()))
    return float(np.mean(vals))


def reference_points() -> pd.DataFrame:
    """Recall and FPR at the calibration 99th percentile, per category (mean and range over seeds)."""
    rows = []
    for cat in CATEGORIES:
        per_seed = []
        for seed in SEEDS:
            sc = load_scores(cat, seed)
            tau = np.quantile(sc["cal"], REFERENCE_Q)
            per_seed.append(((sc["defect"] > tau).mean(), (sc["good"] > tau).mean()))
        r, f = np.array(per_seed).T
        rows.append({"category": cat, "auroc": auroc(cat), "recall_q99": r.mean(), "recall_q99_min": r.min(),
                     "recall_q99_max": r.max(), "fpr_q99": f.mean(), "n_defect": len(load_scores(cat, 0)["defect"]),
                     "n_good_eval": len(load_scores(cat, 0)["good"])})
    return pd.DataFrame(rows)


def recall_ledger_values(ref: pd.DataFrame) -> tuple:
    """(low, central, high) for the registry's ai_recall: min / median / max over categories."""
    r = ref.recall_q99
    return float(r.min()), float(r.median()), float(r.max())


def one_shot_params(p: Dict, recall, fpr) -> Dict:
    t = PRIMARY_TIER
    good_parts_per_h = (1.0 - np.asarray(p["defect_prevalence"])) * p["line_throughput"]
    return {**p, "ai_recall": recall, "persistence_recall_loss": 0.0, f"detection_delay@{t}": 0.0 * recall,
            f"alerts_per_defect@{t}": 1.0, "glare_burst_rate": 0.0,
            f"nominal_false_alarm_rate@{t}": np.asarray(fpr) * good_parts_per_h}


def along_curve(scenario: Scenario, cv: pd.DataFrame):
    """(carbon per FU, single-reviewer utilisation rho) at every point of an operating curve."""
    regime = load_regimes()[PRIMARY_TIER]
    p = one_shot_params(scenario.central(), cv.recall.values, cv.fpr.values)
    r = evaluate_regime(p, scenario.class_shares, regime)
    return np.asarray(r.carbon.total), np.asarray(r.workload.utilisation_rho)


def optimal_thresholds(scenarios) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        pi = sc.central()["defect_prevalence"]
        for cat in CATEGORIES:
            cv = curve(cat)
            c, rho = along_curve(sc, cv)
            i_opt = int(np.argmin(c))
            i_ref = int(np.argmin(np.abs(cv.q.values - REFERENCE_Q)))
            staff = np.ceil(rho - 1e-9)
            capped = np.where(staff <= staff[i_ref])[0]  # no more reviewers than at the q99 threshold
            i_cap = int(capped[np.argmin(c[capped])])
            alarms_per_1k = cv.fpr.values * (1 - pi) * 1000
            feasible = np.where(alarms_per_1k <= ALARM_BUDGET_PER_1K)[0]
            i_bud = int(feasible[np.argmax(cv.recall.values[feasible])]) if len(feasible) else int(np.argmin(alarms_per_1k))
            rows.append({"scenario": sc.name, "category": cat,
                         "q_opt": cv.q[i_opt], "recall_opt": cv.recall[i_opt], "fpr_opt": cv.fpr[i_opt],
                         "carbon_opt": c[i_opt],
                         "recall_q99": cv.recall[i_ref], "fpr_q99": cv.fpr[i_ref], "carbon_q99": c[i_ref],
                         "recall_budget": cv.recall[i_bud], "fpr_budget": cv.fpr[i_bud], "carbon_budget": c[i_bud],
                         "saving_vs_q99": c[i_ref] - c[i_opt], "saving_vs_budget": c[i_bud] - c[i_opt],
                         "rho_opt": rho[i_opt], "rho_q99": rho[i_ref], "rho_budget": rho[i_bud],
                         "fpr_cap": cv.fpr[i_cap], "recall_cap": cv.recall[i_cap], "carbon_cap": c[i_cap],
                         "rho_cap": rho[i_cap], "saving_cap_vs_q99": c[i_ref] - c[i_cap]})
    return pd.DataFrame(rows)

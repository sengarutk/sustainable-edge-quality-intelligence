"""
Does more compute buy enough recall to pay for itself?

Each detector configuration has a measured recall per category (q99 threshold, Section V) and a
measured Jetson power (TensorRT FP16, full pipeline at 30 FPS). For every scenario, configuration and
category the accounting model is evaluated at central parameter values with that category's recall
and that configuration's active power; the result shows whether the extra electricity of a heavier
detector is repaid by the escapes its extra recall avoids.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.experiments.operating_points import reference_points
from src.models.pipeline_evaluator import PRIMARY_TIER, compare, evaluate_all
from src.params import load_scenario
from src.paths import JETSON_SUMMARY, SCENARIOS

CONFIGS = {  # detector id in Section V -> Jetson run configuration
    "patchcore": "patchcore_fp16",
    "patchcore_448": "patchcore_448_fp16",
    "patchcore_wrn50": "patchcore_wrn50_fp16",
    "padim": "padim_fp16",
}


def jetson_power(cfg: str) -> dict:
    run = json.loads(JETSON_SUMMARY.read_text(encoding="utf-8"))["runs"][cfg]
    st = run["stages"]
    return {"idle_w": st["BASELINE_IDLE"]["p_total_w"]["median"], "active_w": st["STAGE_FULL_PIPELINE"]["p_active_w"]["median"],
            "frame_mj": 1000 * st["STAGE_FULL_PIPELINE"]["e_frame_active_j"]["median"]}


def compute_for_recall() -> pd.DataFrame:
    ref = reference_points()
    rows = []
    for sc in SCENARIOS:
        s = load_scenario(sc)
        base = s.central()
        for det, cfg in CONFIGS.items():
            pw = jetson_power(cfg)
            sub = ref[ref.detector == det]
            p = {**base, "edge_idle_power": pw["idle_w"], "edge_active_power": pw["active_w"],
                 "ai_recall": sub.recall_q99.to_numpy()}
            res = evaluate_all(p, s.class_shares)
            dc = np.asarray(compare(res["L0_Manual"], res[PRIMARY_TIER]).delta["C"])
            edge = np.asarray(res[PRIMARY_TIER].carbon.edge + res[PRIMARY_TIER].carbon.hardware) + 0 * dc
            for (_, r), d, e in zip(sub.iterrows(), dc, edge):
                rows.append({"scenario": sc, "detector": det, "dataset": r.dataset, "category": r.category,
                             "recall_q99": r.recall_q99, "active_w": pw["active_w"], "frame_mj": pw["frame_mj"],
                             "edge_cell_carbon": float(e), "delta_C_vs_L0": float(d)})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scenario", "detector"])
    return pd.DataFrame({"recall_median": g.recall_q99.median(), "active_w": g.active_w.first(), "frame_mj": g.frame_mj.first(),
                         "edge_cell_carbon": g.edge_cell_carbon.first(), "delta_C_median": g.delta_C_vs_L0.median(),
                         "delta_C_mean": g.delta_C_vs_L0.mean(),
                         "share_beating_L0": g.delta_C_vs_L0.apply(lambda x: float((x > 0).mean()))}).reset_index()

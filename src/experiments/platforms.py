"""
Edge-platform comparison.

The primary edge computer is the embedded Jetson module (registry scope 'all'); the workstation
GPU is an alternative platform (registry scope 'platform:workstation'). Detector and numeric
precision variants measured on the Jetson replace only the active power of the primary platform.
Each configuration is evaluated with the full accounting model at central parameter values.
"""

from __future__ import annotations

import json

import pandas as pd

from src.models.pipeline_evaluator import PRIMARY_TIER, compare, evaluate_all
from src.params import load_scenario
from src.paths import JETSON_SUMMARY, SCENARIOS
from src.validation.source_registry import platform_records


def configurations():
    """(name, platform label, overrides) for every measured edge configuration."""
    js = json.loads(JETSON_SUMMARY.read_text(encoding="utf-8"))
    out = []
    runs = {c: r for c, r in js["runs"].items() if c.rsplit("_", 1)[0] in ("patchcore", "padim")}  # Section IV set
    for cfg, run in sorted(runs.items(), key=lambda kv: kv[0] != js["primary"]):
        active = run["stages"]["STAGE_FULL_PIPELINE"]["p_active_w"]["median"]
        over = {} if cfg == js["primary"] else {"edge_active_power": active}
        out.append((f"jetson_{cfg}", "Jetson Orin Nano", over))
    ws = {k: r.central for k, r in platform_records("workstation").items()}
    out.append(("workstation_patchcore_fp32", "Workstation GPU", ws))
    return out


def platform_comparison() -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        s = load_scenario(sc)
        base = s.central()
        for name, label, over in configurations():
            p = {**base, **over}
            res = evaluate_all(p, s.class_shares)
            b3 = res[PRIMARY_TIER]
            d_l0 = float(compare(res["L0_Manual"], b3).delta["C"])
            edge_c = float(b3.carbon.edge + b3.carbon.hardware)
            rows.append({"scenario": sc, "configuration": name, "platform": label,
                         "cell_power_w": p["edge_idle_power"] + p["edge_active_power"] + p["host_power"],
                         "compute_power_w": p["edge_idle_power"] + p["edge_active_power"],
                         "edge_kwh": float(b3.energy.edge_kwh), "edge_carbon": float(b3.carbon.edge),
                         "hardware_carbon": float(b3.carbon.hardware), "edge_cell_carbon": edge_c,
                         "delta_C_vs_L0": d_l0, "edge_share_of_benefit": edge_c / d_l0 if d_l0 > 0 else float("nan")})
    return pd.DataFrame(rows)

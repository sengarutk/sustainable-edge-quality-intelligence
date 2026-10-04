#!/usr/bin/env python3
"""Step 04: evaluate all inspection regimes at central parameter values."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.io_utils import write_csv  # noqa: E402
from src.models.pipeline_evaluator import AI_TIERS, compare, evaluate_all  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import PROCESSED_DIR, SCENARIOS  # noqa: E402
from src.sustainability.carbon_accounting import COMPONENTS  # noqa: E402
from src.sustainability.sqi import weight_robustness  # noqa: E402

REFERENCES = ("N0_NoInspection", "L0_Manual", "B0_Raw", "B1_EMA", "B2_EMA_kofN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    out_dir = ap.parse_args().out_dir
    regime_rows, comp_rows = [], []
    for sc in SCENARIOS:
        s = load_scenario(sc)
        res = evaluate_all(s.central(), s.class_shares)
        for rid, r in res.items():
            regime_rows.append({
                "scenario": sc, "regime": rid, "kind": r.regime.kind, "recall": float(r.recall),
                "delay_s": float(r.delay_s), "false_alarm_rate_per_h": float(r.false_alarm_rate), "t_fu_h": float(r.t_fu_h),
                "defects": float(r.routing.total_defects), "n_rework": float(r.routing.n_rework),
                "n_scrap": float(r.routing.n_scrap), "n_escape": float(r.routing.n_escape), "q_rw": float(r.routing.q_rw),
                "gross_scrap_kg": float(r.material.gross_scrap_kg), "net_loss_kg": float(r.material.net_loss_kg),
                "operator_hours": float(r.workload.operator_hours), "rho": float(r.workload.utilisation_rho),
                "edge_kwh": float(r.energy.edge_kwh), "review_kwh": float(r.energy.review_kwh),
                "rework_kwh": float(r.energy.rework_kwh), "total_kwh": float(r.energy.total_kwh),
                **{f"carbon_{c}": float(getattr(r.carbon, c)) for c in COMPONENTS}, "carbon_total": float(r.carbon.total),
            })
        for base in REFERENCES:
            for pol in AI_TIERS:
                if pol == base:
                    continue
                c = compare(res[base], res[pol])
                sub = {k: float(v) for k, v in c.sqi.sub.items()}
                comp_rows.append({
                    "scenario": sc, "baseline": base, "policy": pol,
                    **{f"delta_{k}": float(v) for k, v in c.delta.items()},
                    **{f"wf_{k}": float(v) for k, v in c.waterfall.items()},
                    **{f"S_{k}": v for k, v in sub.items()},
                    **{f"sqi_{k}": float(v) for k, v in c.sqi.scores.items()},
                    "sqi_weight_robustness": weight_robustness(sub),
                })
    write_csv(pd.DataFrame(regime_rows), out_dir / "regime_outcomes.csv")
    write_csv(pd.DataFrame(comp_rows), out_dir / "comparisons.csv")
    print(f"Wrote {len(regime_rows)} regime rows and {len(comp_rows)} comparisons to {out_dir}")


if __name__ == "__main__":
    main()

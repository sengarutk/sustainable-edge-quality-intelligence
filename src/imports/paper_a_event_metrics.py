"""
Helper executed *inside an extracted Paper A revision* with Paper A's own Python environment
(it imports Paper A's code, never Paper B's). It recomputes, with Paper A's own functions
(scripts/build_paper_assets.py: load_units, unit_values, cluster_boot), the per-policy means and
95 % category-cluster bootstrap intervals that Paper A prints, for every run set it reports:
MVTec AD with independent and with temporally correlated (rho = 0.9) frame scores, and the
held-out VisA runs with independent and correlated scores. It also computes, per unit, the recall
a policy loses relative to single-frame thresholding on the same unit (1 - recall / recall of
BASELINE), with the same category-cluster bootstrap. Output: JSON on stdout.

Usage (called by src.imports.import_paper_a): python paper_a_event_metrics.py POLICY [POLICY ...]
"""

import importlib.util
import json
import sys
from pathlib import Path

METRICS = {  # Paper B name -> (Paper A workload, Paper A metric)
    "nominal_false_alarm_rate": ("nominal", "false_alerts_per_hour"),
    "glare_alerts_per_burst": ("transient_glitches", "ev:per_glare"),
    "alerts_per_defect": ("sustained_defects", "ev:per_episode"),
    "detection_delay": ("sustained_defects", "mean_delay_frames"),
    "sustained_recall": ("sustained_defects", "routing_recall"),
}
RUNS = {"mvtec_independent": "ablation", "mvtec_correlated": "ablation_rho0.9",
        "visa_independent": "ablation_visa", "visa_correlated": "ablation_visa_rho0.9"}
REFERENCE = "BASELINE"


def main() -> None:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("paper_a_assets", root / "scripts" / "build_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {"B_BOOT": mod.B_BOOT, "runs": {}}
    for name, dirname in RUNS.items():
        units = mod.load_units(dirname)
        ref = mod.unit_values(units, "sustained_defects", REFERENCE, "routing_recall")
        out["runs"][name] = {}
        for pol in sys.argv[1:]:
            res = {}
            for key, (wl, metric) in METRICS.items():
                r = mod.cluster_boot(mod.unit_values(units, wl, pol, metric))
                res[key] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_units": r["n_units"],
                            "n_categories": r["n_categories"]}
            rec = mod.unit_values(units, "sustained_defects", pol, "routing_recall")
            rel = {u: 1.0 - rec[u] / ref[u] for u in rec if u in ref and ref[u] > 0}
            r = mod.cluster_boot(rel)
            res["relative_recall_loss"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_units": r["n_units"],
                                           "n_categories": r["n_categories"]}
            out["runs"][name][pol] = res
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

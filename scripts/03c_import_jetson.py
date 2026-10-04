#!/usr/bin/env python3
"""Step 03c: import the Jetson measurement runs, summarise them and write the primary-platform
edge-power rows of the registry.

The runs are produced on the device by src/experiments/jetson_benchmark.py (see README). With
--host USER@ADDR the run directories are first copied from ~/paperb/jetson_runs on the device.
Without it, the committed runs in data/raw/energy_measurements_jetson are summarised again.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.energy_benchmark import measured_registry_bounds, summarize_run  # noqa: E402
from src.paths import JETSON_SUMMARY, JETSON_TRACE_DIR, SOURCE_REGISTRY  # noqa: E402

PRIMARY = "patchcore_fp16"  # deployed configuration: TensorRT FP16, 100 % decision agreement with FP32


def summarize() -> dict:
    runs = {}
    for d in sorted(p for p in JETSON_TRACE_DIR.iterdir() if (p / "run_meta.json").exists()):
        cfg = d.name.split("_", 1)[1]  # <run_id>_<detector>_<precision>
        if cfg in runs:
            raise SystemExit(f"two runs for configuration {cfg}; keep one in {JETSON_TRACE_DIR}")
        runs[cfg] = summarize_run(d)
    if PRIMARY not in runs:
        raise SystemExit(f"primary configuration {PRIMARY} has not been measured")
    summary = {"primary": PRIMARY, "runs": runs}
    JETSON_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    JETSON_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def update_registry(summary: dict):
    bounds = measured_registry_bounds(summary["runs"][summary["primary"]])
    df = pd.read_csv(SOURCE_REGISTRY, dtype=str, keep_default_na=False)
    for key, values in bounds.items():
        mask = (df["parameter"] == key) & (df["scope"] == "all")
        if mask.sum() != 1:
            raise SystemExit(f"registry must contain exactly one '{key}' row with scope 'all'")
        df.loc[mask, ["low", "central", "high"]] = [repr(v) for v in values]
    df.to_csv(SOURCE_REGISTRY, index=False, lineterminator="\n")
    print(f"primary platform rows <- {bounds}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", help="copy runs from USER@ADDR:~/paperb/jetson_runs first")
    ap.add_argument("--update-registry", action="store_true")
    args = ap.parse_args()
    if args.host:
        JETSON_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{args.host}:paperb/jetson_runs/", f"{JETSON_TRACE_DIR}/"], check=True)
    summary = summarize()
    for cfg, s in summary["runs"].items():
        full = s["stages"]["STAGE_FULL_PIPELINE"]
        print(f"{cfg:16s} idle {s['stages']['BASELINE_IDLE']['p_total_w']['median']:.2f} W  "
              f"active {full['p_active_w']['median']:.2f} W  {full['e_frame_active_j']['median'] * 1000:.1f} mJ/frame")
    if args.update_registry:
        update_registry(summary)


if __name__ == "__main__":
    main()

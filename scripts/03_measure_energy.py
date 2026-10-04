#!/usr/bin/env python3
"""Step 03 (GPU only): measure edge inference power and write the measured registry rows.

Requires an NVIDIA GPU with NVML. Skipped by reproduce_all.sh unless --measure is given;
the committed measurement (results/raw/energy_summary.json) is used otherwise.
With --out DIR the run is written to DIR (inside the repository) and neither the committed
summary nor the registry is touched -- use this to re-check the measurement.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.energy_benchmark import measured_registry_bounds  # noqa: E402
from src.paths import ENERGY_SUMMARY, SOURCE_REGISTRY  # noqa: E402


def update_registry_from_summary():
    bounds = measured_registry_bounds(json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8")))
    df = pd.read_csv(SOURCE_REGISTRY, dtype=str, keep_default_na=False)
    for key, values in bounds.items():
        mask = df["parameter"] == key
        if mask.sum() != 1:
            raise SystemExit(f"registry must contain exactly one '{key}' row")
        df.loc[mask, ["low", "central", "high"]] = [repr(v) for v in values]
    df.to_csv(SOURCE_REGISTRY, index=False, lineterminator="\n")
    print(f"Updated measured rows in {SOURCE_REGISTRY}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--registry-only", action="store_true", help="only copy the existing summary into the registry")
    ap.add_argument("--out", type=Path, help="write traces and summary here instead of the committed locations")
    args = ap.parse_args()
    if args.registry_only:
        update_registry_from_summary()
        return
    from src.experiments.energy_benchmark import run_benchmark
    if args.out is None:
        run_benchmark(repeats=args.repeats, window_s=args.window_s)
        update_registry_from_summary()
    else:
        out = args.out.resolve()
        summary = run_benchmark(repeats=args.repeats, window_s=args.window_s, trace_root=out,
                                summary_path=out / "energy_summary.json")
        print(json.dumps(measured_registry_bounds(summary), indent=2))


if __name__ == "__main__":
    main()

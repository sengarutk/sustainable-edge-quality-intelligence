#!/usr/bin/env python3
"""Step 03b (GPU recommended): measure PatchCore and PaDiM operating points on MVTec AD and VisA
and write the measured detector-recall row of the registry.

Requires MVTec AD (default ../datasets/mvtec/full, override with --mvtec) and VisA in its official
1-class layout (default ../datasets/visa/VisA, override with --visa). Skipped by reproduce_all.sh unless --measure is given; the committed
scores in data/raw/detector_scores are used otherwise.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.operating_points import recall_ledger_values, reference_points  # noqa: E402
from src.paths import SOURCE_REGISTRY  # noqa: E402


def update_registry():
    lo, c, hi = recall_ledger_values(reference_points())
    df = pd.read_csv(SOURCE_REGISTRY, dtype=str, keep_default_na=False)
    mask = df.parameter == "ai_recall"
    if mask.sum() != 1:
        raise SystemExit("registry must contain exactly one 'ai_recall' row")
    df.loc[mask, ["low", "central", "high"]] = [f"{round(v, 4):g}" for v in (lo, c, hi)]
    df.to_csv(SOURCE_REGISTRY, index=False, lineterminator="\n")
    print(f"ai_recall <- {lo:.4f} / {c:.4f} / {hi:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mvtec", type=Path, default=None)
    ap.add_argument("--visa", type=Path, default=None)
    ap.add_argument("--registry-only", action="store_true")
    args = ap.parse_args()
    if not args.registry_only:
        from src.experiments.detector_eval import run_detector_eval
        run_detector_eval({"mvtec": args.mvtec, "visa": args.visa})
    update_registry()


if __name__ == "__main__":
    main()

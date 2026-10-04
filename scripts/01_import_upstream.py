#!/usr/bin/env python3
"""Step 01 (optional): refresh the Paper A snapshot from a local Paper A checkout.

Usage: python scripts/01_import_upstream.py [--paper-a PATH] [--ref COMMIT] [--update-registry]
Without the upstream repository the committed snapshot in data/raw/paper_a_exports is used.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.imports.import_paper_a import SNAPSHOT, import_paper_a, snapshot_rows  # noqa: E402
from src.paths import SOURCE_REGISTRY  # noqa: E402


def update_registry():
    """Copy the snapshot's (low, central, high) values into the matching registry rows."""
    df = pd.read_csv(SOURCE_REGISTRY, dtype=str, keep_default_na=False)
    for key, values in snapshot_rows().items():
        param, scope = key.split("@", 1)
        mask = (df.parameter == param) & (df.scope == scope)
        if mask.sum() != 1:
            raise SystemExit(f"registry must contain exactly one row for {key}")
        df.loc[mask, ["low", "central", "high"]] = [f"{v:.6g}" for v in values]
    df.to_csv(SOURCE_REGISTRY, index=False, lineterminator="\n")
    print(f"Updated Paper A rows in {SOURCE_REGISTRY}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-a", type=Path, default=Path(__file__).resolve().parents[2] / "edge-quality-intelligence")
    ap.add_argument("--out", type=Path, default=SNAPSHOT, help="snapshot path (default: the committed snapshot)")
    ap.add_argument("--ref", default="HEAD", help="Paper A commit to import from (default: HEAD)")
    ap.add_argument("--update-registry", action="store_true", help="also write the imported values into the registry")
    args = ap.parse_args()
    if not (args.paper_a / ".git").exists():
        print(f"Paper A repository not found at {args.paper_a}; keeping committed snapshot {SNAPSHOT}.")
        return
    print(f"Wrote {import_paper_a(args.paper_a, args.out, args.ref)}")
    if args.update_registry and args.out.resolve() == SNAPSHOT.resolve():
        update_registry()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy generated assets from results/ into paper/ (figures as PDF, tables, macros).

  python scripts/sync_paper_assets.py          # copy, then verify byte parity
  python scripts/sync_paper_assets.py --check  # verify only (CI); exit 1 on any drift

paper/ never contains hand-edited copies of generated files; the manuscript source
(paper/main.tex) is never modified by this script.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import (PAPER_FIG_DIR, PAPER_MACROS, PAPER_TABLE_DIR, RESULTS_FIG_DIR,  # noqa: E402
                       RESULTS_MACROS, RESULTS_TABLE_DIR)


def pairs():
    out = [(RESULTS_MACROS, PAPER_MACROS)]
    out += [(p, PAPER_FIG_DIR / p.name) for p in sorted(RESULTS_FIG_DIR.glob("*.pdf"))]
    out += [(p, PAPER_TABLE_DIR / p.name) for p in sorted(RESULTS_TABLE_DIR.glob("*.tex"))]
    return out


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    expected = pairs()
    if len(expected) < 2:
        raise SystemExit("no generated assets found in results/; run the pipeline first")
    stale = {p for d in (PAPER_FIG_DIR, PAPER_TABLE_DIR) for p in d.glob("*")} - {dst for _, dst in expected}
    if not args.check:
        for d in (PAPER_FIG_DIR, PAPER_TABLE_DIR):
            d.mkdir(parents=True, exist_ok=True)
        for p in stale:
            p.unlink()
        for src, dst in expected:
            shutil.copyfile(src, dst)
        stale = set()
    bad = [str(dst) for src, dst in expected if not dst.exists() or digest(src) != digest(dst)]
    bad += [f"unexpected {p}" for p in sorted(stale)]
    if bad:
        raise SystemExit("paper/ assets out of sync with results/:\n  " + "\n  ".join(bad))
    print(f"paper/ assets in sync ({len(expected)} files)")


if __name__ == "__main__":
    main()

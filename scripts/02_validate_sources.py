#!/usr/bin/env python3
"""Step 02: validate the source registry and its consistency with measured / imported evidence."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.energy_benchmark import measured_registry_bounds  # noqa: E402
from src.experiments.operating_points import recall_ledger_values, reference_points  # noqa: E402
from src.imports.import_paper_a import snapshot_rows  # noqa: E402
from src.paths import ENERGY_SUMMARY  # noqa: E402
from src.validation.source_registry import load_registry, validate_registry_file  # noqa: E402


def main():
    n = validate_registry_file()
    recs = {r.key: r for r in load_registry()}

    by_scope = {f"{r.parameter}@{r.scope}": r for r in load_registry()}
    expected = snapshot_rows()
    for key, values in expected.items():
        r = by_scope.get(key)
        if r is None or (r.low, r.central, r.high) != tuple(values):
            raise SystemExit(f"Registry {key} disagrees with the Paper A snapshot {values}; run scripts/01 --update-registry")
    for key, r in by_scope.items():
        if r.classification.value == "Derived from Paper A" and key not in expected:
            raise SystemExit(f"Registry {key} is labelled 'Derived from Paper A' but is not in the snapshot")

    summary = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    for key, expect in measured_registry_bounds(summary).items():
        r = recs[key]
        if (r.low, r.central, r.high) != expect:
            raise SystemExit(f"Registry {key} {(r.low, r.central, r.high)} != energy summary {expect}; rerun scripts/03")
    expect = tuple(round(v, 4) for v in recall_ledger_values(reference_points()))
    r = recs["ai_recall"]
    if (r.low, r.central, r.high) != expect or r.classification.value != "Measured by this study":
        raise SystemExit(f"Registry ai_recall {(r.low, r.central, r.high)} != detector measurement {expect}; rerun scripts/03b")
    print(f"Validated {n} registry entries; Paper A and measured entries are consistent.")


if __name__ == "__main__":
    main()

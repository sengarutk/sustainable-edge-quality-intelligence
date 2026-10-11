"""
Scenario parameter assembly: registry values + scenario YAML (defect class mix).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import yaml

from src.paths import SCENARIO_DIR, SOURCE_REGISTRY
from src.validation.source_registry import ParameterRecord, scenario_records, variant_records

CLASS_KEYS = ("class_a_reworkable", "class_b_scrap_prone", "class_c_escape_sensitive")


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    description: str
    class_shares: Tuple[float, float, float]
    records: Dict[str, ParameterRecord]

    def central(self) -> Dict[str, float]:
        return {k: r.central for k, r in self.records.items()}

    def bounds(self) -> Dict[str, Tuple[float, float, float]]:
        return {k: (r.low, r.central, r.high) for k, r in self.records.items()}

    def uncertain_keys(self):
        return sorted(k for k, r in self.records.items() if r.high > r.low)

    def with_variant(self, variant: str, registry_path=SOURCE_REGISTRY) -> "Scenario":
        """The scenario with the records of an analysis variant (e.g. 'correlated_scores') replacing the
        primary records of the same key."""
        return replace(self, records={**self.records, **variant_records(variant, registry_path)})


def load_scenario(name: str, registry_path=SOURCE_REGISTRY, variant: str | None = None) -> Scenario:
    path = SCENARIO_DIR / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("name") != name:
        raise ValueError(f"{path}: 'name' must equal the file stem '{name}'")
    dist = cfg["defect_distribution"]
    shares = tuple(float(dist[k]) for k in CLASS_KEYS)
    if any(s < 0 for s in shares) or abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"{path}: defect_distribution must be non-negative and sum to 1, got {shares}")
    extra = set(cfg) - {"name", "label", "description", "defect_distribution"}
    if extra:
        raise ValueError(f"{path}: numeric parameters belong in the source registry, found {sorted(extra)}")
    sc = Scenario(name=name, label=cfg["label"], description=cfg["description"],
                  class_shares=shares, records=scenario_records(name, registry_path))
    return sc.with_variant(variant, registry_path) if variant else sc

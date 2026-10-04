"""
Schema and loader for data/raw/source_registry.csv -- the single source of truth
for every numerical model parameter (central value and uncertainty bounds).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.paths import SCENARIOS, SOURCE_REGISTRY

REQUIRED_COLUMNS = [
    "parameter", "symbol", "scope", "low", "central", "high",
    "unit", "classification", "citation_key", "notes",
]


class SourceClassification(str, Enum):
    MEASURED = "Measured by this study"
    DERIVED_PAPER_A = "Derived from Paper A"
    LITERATURE_DERIVED = "Literature-derived"
    OFFICIAL_DATASET = "Official/public dataset"
    SCENARIO_ASSUMPTION = "Scenario assumption"


class ParameterRecord(BaseModel):
    parameter: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    symbol: str = Field(..., min_length=1)
    scope: str = Field(..., min_length=1)
    low: float
    central: float
    high: float
    unit: str = Field(..., min_length=1)
    classification: SourceClassification
    citation_key: str = Field(..., min_length=1)
    notes: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check(self) -> "ParameterRecord":
        if not (self.low <= self.central <= self.high):
            raise ValueError(f"{self.parameter}@{self.scope}: bounds violated ({self.low} <= {self.central} <= {self.high})")
        valid_scopes = {"all", *SCENARIOS}
        if self.scope not in valid_scopes and not self.scope.startswith(("policy:", "platform:")):
            raise ValueError(f"{self.parameter}: unknown scope '{self.scope}'")
        if self.classification != SourceClassification.SCENARIO_ASSUMPTION and self.citation_key == "none":
            raise ValueError(f"{self.parameter}@{self.scope}: '{self.classification.value}' requires a citation key")
        return self

    @property
    def key(self) -> str:
        """Flat parameter key: policy- and platform-scoped parameters are suffixed with '@<name>'."""
        if self.scope.startswith(("policy:", "platform:")):
            return f"{self.parameter}@{self.scope.split(':', 1)[1]}"
        return self.parameter


def load_registry(path: Path = SOURCE_REGISTRY) -> List[ParameterRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Source registry not found: {path}")
    df = pd.read_csv(path, dtype={"citation_key": str, "notes": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source registry missing columns: {missing}")
    records = []
    for idx, row in df.iterrows():
        try:
            records.append(ParameterRecord(**{c: row[c] for c in REQUIRED_COLUMNS}))
        except Exception as exc:  # re-raise with row context
            raise ValueError(f"Source registry row {idx + 2} ({row.get('parameter')}): {exc}") from exc
    seen = set()
    for r in records:
        ident = (r.key, r.scope)
        if ident in seen:
            raise ValueError(f"Duplicate registry entry: {r.key} @ {r.scope}")
        seen.add(ident)
    return records


def scenario_records(scenario: str, path: Path = SOURCE_REGISTRY) -> Dict[str, ParameterRecord]:
    """All records applicable to a scenario (global, policy-scoped and scenario-specific). Platform-scoped
    records describe an alternative edge computer and are applied only through platform_records()."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'")
    out: Dict[str, ParameterRecord] = {}
    for r in load_registry(path):
        if r.scope == "all" or r.scope.startswith("policy:") or r.scope == scenario:
            if r.key in out:
                raise ValueError(f"Parameter {r.key} defined twice for scenario {scenario}")
            out[r.key] = r
    return out


def platform_records(platform: str, path: Path = SOURCE_REGISTRY) -> Dict[str, ParameterRecord]:
    """Records of an alternative edge platform, keyed by parameter name; they replace the global
    ('all') entries of the same name, which describe the primary platform."""
    out = {r.parameter: r for r in load_registry(path) if r.scope == f"platform:{platform}"}
    if not out:
        raise KeyError(f"no registry records for platform '{platform}'")
    glob = {r.parameter for r in load_registry(path) if r.scope == "all"}
    missing = set(out) - glob
    if missing:
        raise ValueError(f"platform '{platform}' overrides parameters without a primary entry: {sorted(missing)}")
    return out


def platforms(path: Path = SOURCE_REGISTRY) -> List[str]:
    return sorted({r.scope.split(":", 1)[1] for r in load_registry(path) if r.scope.startswith("platform:")})


def validate_registry_file(path: Path = SOURCE_REGISTRY) -> int:
    records = load_registry(path)
    for sc in SCENARIOS:
        scenario_records(sc, path)
    for pf in platforms(path):
        platform_records(pf, path)
    return len(records)

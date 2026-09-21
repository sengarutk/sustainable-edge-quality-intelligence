"""
Pydantic schema validating data/raw/source_registry.csv.
Enforces strict parameter bounds and single-source-of-truth metadata.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd
from pathlib import Path


class SourceClassification(str, Enum):
    MEASURED = "Measured by this study"
    DERIVED_PAPER_A = "Derived from Paper A"
    BENCHMARK_DERIVED = "Benchmark-derived"
    LITERATURE_DERIVED = "Literature-derived"
    OFFICIAL_DATASET = "Official/public dataset"
    SCENARIO_ASSUMPTION = "Scenario assumption"


class ParameterRecord(BaseModel):
    parameter: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    scenario: str = Field(..., min_length=1)
    low: float
    central: float
    high: float
    unit: str
    source_type: str = Field(..., min_length=1)
    citation_key: str = Field(..., min_length=1)
    classification: SourceClassification
    notes: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_bounds(self) -> "ParameterRecord":
        if not (self.low <= self.central <= self.high):
            raise ValueError(
                f"Bounds violation for {self.parameter}: {self.low} <= {self.central} <= {self.high} failed"
            )
        return self


def validate_registry_file(csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Source registry file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = [
        "parameter", "symbol", "scenario", "low", "central", "high",
        "unit", "source_type", "citation_key", "classification", "notes"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source registry: {missing}")

    records = []
    for idx, row in df.iterrows():
        try:
            record = ParameterRecord(
                parameter=str(row["parameter"]),
                symbol=str(row["symbol"]),
                scenario=str(row["scenario"]),
                low=float(row["low"]),
                central=float(row["central"]),
                high=float(row["high"]),
                unit=str(row["unit"]),
                source_type=str(row["source_type"]),
                citation_key=str(row["citation_key"]),
                classification=SourceClassification(str(row["classification"]).strip()),
                notes=str(row["notes"]),
            )
            records.append(record)
        except Exception as e:
            raise ValueError(f"Validation error at row {idx} ({row.get('parameter')}): {e}")

    return len(records)


if __name__ == "__main__":
    count = validate_registry_file(Path("/home/sengar/sustainable-edge-quality-intelligence/data/raw/source_registry.csv"))
    print(f"Validated {count} source registry entries successfully.")
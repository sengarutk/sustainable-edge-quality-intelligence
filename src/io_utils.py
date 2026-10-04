"""Deterministic writers: fixed float precision so regenerated outputs diff cleanly."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SIG = 10


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            raise ValueError("non-finite value in output")
        return float(f"{x:.{SIG}g}")
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(obj), indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=f"%.{SIG}g", lineterminator="\n")

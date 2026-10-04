#!/usr/bin/env python3
"""Fail if regenerated outputs drift from the committed ones.

LaTeX files (tables and the macro file, i.e. everything the manuscript quotes) and every non-tabular
file must be byte-identical. CSV and JSON results must agree in structure and in every non-numeric
field, and numerically to rtol = atol = 1e-6: last-digit differences arise when NumPy's vectorised
exp/log take different SIMD paths on different CPUs, and near-zero Sobol indices amplify them; neither
can change any quoted number. Drifted files are reported as GitHub annotations.
"""

import io
import json
import math
import subprocess
import sys

import pandas as pd

RTOL = ATOL = 1e-6
SCOPE = ["data", "results", "paper", ":!results/figures", ":!paper/figures"]


def git(*args) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def json_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    return close(a, b)


def csv_equal(old: bytes, new: bytes) -> bool:
    a, b = pd.read_csv(io.BytesIO(old)), pd.read_csv(io.BytesIO(new))
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
            same_nan = (pd.isna(x) == pd.isna(y)).all()
            ok = ~pd.isna(x)
            if not same_nan or not all(math.isclose(p, q, rel_tol=RTOL, abs_tol=ATOL) for p, q in zip(x[ok], y[ok])):
                return False
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False
    return True


def main() -> int:
    lines = git("status", "--porcelain", "--", *SCOPE).decode().splitlines()
    bad = []
    for line in lines:
        status, path = line[:2].strip(), line[3:]
        if status != "M":
            bad.append((path, f"{status or 'changed'}: file added or removed"))
            continue
        old, new = git("show", f"HEAD:{path}"), open(path, "rb").read()
        if path.endswith(".csv"):
            ok = csv_equal(old, new)
        elif path.endswith(".json"):
            ok = json_equal(json.loads(old), json.loads(new))
        else:
            ok = old == new
        if not ok:
            first = [ln[:160] for ln in git("diff", "-U0", "--", path).decode().splitlines() if ln[:1] in "+-" and ln[:3] not in ("+++", "---")][:2]
            bad.append((path, " | ".join(first)))
        else:
            print(f"{path}: within rtol=atol={RTOL:g} (floating-point noise only)")
    for path, msg in bad:
        print(f"::error file={path}::drift: {msg}")
    print(f"{len(lines)} regenerated file(s) differ byte-wise; {len(bad)} drift beyond tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

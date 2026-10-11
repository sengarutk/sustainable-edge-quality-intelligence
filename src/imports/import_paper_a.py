"""
Import the Paper A measurements used by this study from the Paper A repository.

Everything is read from one *committed* Paper A revision, never from a working copy. The revision
is extracted with `git archive` into a temporary directory and Paper A's own code
(scripts/build_paper_assets.py, run by src/imports/paper_a_event_metrics.py with Paper A's Python
environment) recomputes, from the committed per-run result files, the per-policy means and the
95 % category-cluster bootstrap intervals that Paper A prints:

* nominal false alerts per hour on good parts, alerts per 1-3 frame glare burst, alerts per
  sustained-defect episode, mean detection delay and sustained-defect recall;
* for the MVTec AD runs with independent frame scores (results/ablation; the primary values of this
  study) and with temporally correlated scores (results/ablation_rho0.9, AR(1) Gaussian copula,
  rho = 0.9; variant "correlated_scores"), and for the held-out VisA runs with independent and with
  correlated scores (results/ablation_visa, results/ablation_visa_rho0.9; variants "visa_scores" and
  "visa_correlated_scores");
* for the four policies of the alert-tier ladder and the three external alarm baselines;
* plus, per unit, the recall each policy loses relative to single-frame thresholding on the same
  unit (a policy can delay or merge detections but not create them).

Every recomputed mean (and every interval Paper A prints) is checked against Paper A's printed
tables (paper/tables/ablation.tex, ablation_rho.tex) and macros (paper/generated_metrics.tex) to the
printed precision; any disagreement aborts the import. results/short_defect_recall.json supplies the
recall of each policy for defects visible a fixed number of frames, which bounds the recall a
policy can lose on parts that stay in view only briefly; generated_metrics.tex supplies the
reviewer service rate and the assumed glare rate.

The snapshot (data/raw/paper_a_exports/paper_a_metrics.json) records the commit, the object hashes
of every input and the resulting registry rows. It is committed, so the pipeline runs without the
upstream repository; scripts/02 checks that every registry row derived from Paper A equals it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

from src.paths import PAPER_A_EXPORT_DIR

SNAPSHOT = PAPER_A_EXPORT_DIR / "paper_a_metrics.json"
HELPER = Path(__file__).with_name("paper_a_event_metrics.py")
ARCHIVE_PATHS = ("src", "configs", "scripts", "results/ablation", "results/ablation_rho0.9", "results/ablation_visa",
                 "results/ablation_visa_rho0.9", "results/score_bank", "results/score_bank_visa",
                 "results/short_defect_recall.json", "paper/generated_metrics.tex", "paper/tables/ablation.tex",
                 "paper/tables/ablation_rho.tex", "paper/tables/ablation_visa.tex")
HASHED_PATHS = ("scripts/build_paper_assets.py", "results/ablation", "results/ablation_rho0.9", "results/ablation_visa",
                "results/ablation_visa_rho0.9", "results/short_defect_recall.json", "paper/generated_metrics.tex",
                "paper/tables/ablation.tex", "paper/tables/ablation_rho.tex", "paper/tables/ablation_visa.tex")
# Visibility (frames) that sets the upper bound of a policy's recall loss: parts seen for only 8 frames
# (267 ms at 30 FPS), the shortest length at which the full policy keeps a mean recall of 0.95 in Paper A.
SHORT_BOUND_FRAMES = 8

# Paper B alert tier -> Paper A policy. B0-B3 form the ladder of the main analysis; E1-E3 are the
# external (textbook) alarm baselines of Paper A, evaluated as alternatives to B3.
TIER_POLICY = {
    "B0_Raw": "BASELINE",
    "B1_EMA": "EMA_ONLY",
    "B2_EMA_kofN": "EMA_KOFN",
    "B3_Full_Policy": "FULL_POLICY",
    "E1_Delay_Timer": "DELAY_TIMER",
    "E2_EMA_Hysteresis": "EMA_HYSTERESIS",
    "E3_Decision_Fusion": "DECISION_FUSION",
}
POLICY_MACRO = {"BASELINE": "Baseline", "EMA_KOFN": "EmaKofn", "FULL_POLICY": "Full", "DELAY_TIMER": "Timer",
                "EMA_HYSTERESIS": "Hyst", "DECISION_FUSION": "Fusion"}
# registry parameter -> (Paper A table column, printed decimals, Paper A macro stem for its interval)
TABLE_COLUMNS = {"nominal_false_alarm_rate": (0, 1, "NominalFA"), "glare_alerts_per_burst": (1, 2, "GlarePerBurst"),
                 "alerts_per_defect": (2, 2, "AlertsPerEpisode"), "detection_delay": (3, 1, "SustainedDelay")}
RECALL_COLUMN = 4  # sustained-defect recall within 15 frames, printed with 2 decimals
# Paper A run set -> (registry scope prefix of this study, Paper A table with its printed means or None)
RUNS = {"mvtec_independent": ("", "ablation"), "mvtec_correlated": ("variant:correlated_scores:", "ablation_rho"),
        "visa_independent": ("variant:visa_scores:", "ablation_visa"),
        "visa_correlated": ("variant:visa_correlated_scores:", None)}
# VisA with correlated scores is printed as macros: Paper A macro stem per registry parameter
VISA_RHO_MACROS = {"nominal_false_alarm_rate": ("VisaRhoNominalFA", 1), "glare_alerts_per_burst": ("VisaRhoGlare", 2),
                   "alerts_per_defect": ("VisaRhoAlertsEp", 2), "detection_delay": ("VisaRhoDelay", 1),
                   "sustained_recall": ("VisaRhoDefectRecall", 2)}

# Paper A macros quoted in the Paper B text (recorded for provenance).
QUOTED_MACROS = ("MuReviews", "GlareRateAssumed", "LoadCrossFull", "MemoryBankSize", "LatVisionMean", "ShortK", "ShortN",
                 "ShortMinLenFull", "ShortMinLenTimer", "NumCategories", "UnitsPerCell", "NumPolicies", "NumVariants",
                 "NumExternal", "BootB", "VisaNumCategories", "VisaUnits")

_MACRO = re.compile(r"\\(?:new|provide)command\{\\(\w+)\}\{(.*)\}\s*$")


def _git(repo: Path, *args: str, binary: bool = False):
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=not binary)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {res.stderr if not binary else res.stderr.decode()}")
    return res.stdout


def _sig(x: float) -> float:
    return 0.0 if abs(x) < 1e-12 else float(f"{x:.6g}")  # floating-point residue of exact zeros


def _num(s: str) -> float:
    return float(s.replace("{,}", "").replace(",", "").strip())


def parse_macros(tex: str) -> Dict[str, str]:
    out = {}
    for line in tex.splitlines():
        m = _MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).replace("{,}", "").replace(r"\,", "").replace(r"\%", "%").strip()
    return out


def parse_table(table_tex: str) -> Dict[str, list]:
    """Rows of a Paper A ablation table: policy -> list of printed numbers (one per column)."""
    out = {}
    for line in table_tex.splitlines():
        m = re.match(r"\\texttt\{([A-Z_\\]+)\}\s*&(.*)\\\\", line.strip())
        if m:
            out[m.group(1).replace("\\_", "_")] = [c.strip() for c in m.group(2).split("&")]
    return out


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Paper A import consistency check failed: {msg}")


def _close(value: float, printed: str, decimals: int) -> bool:
    """value agrees with a printed number to the precision it was printed with (Paper A prints values
    of 1,000 and more as integers)."""
    t = printed.replace("{,}", "").strip()
    shown = len(t.split(".", 1)[1]) if "." in t else 0
    return abs(value - _num(printed)) <= 0.5 * 10 ** (-min(decimals, shown)) + 1e-9


def short_recall(short: Dict, policy: str, frames: int) -> float:
    per = {x["length"]: x["recall"]["mean"] for x in short["policies"][policy]["per_length"]}
    return per[frames]


def build_rows(metrics: Dict, short: Dict, macros: Dict[str, str]) -> Tuple[Dict, Dict]:
    """(primary registry rows, variant rows by variant name), keyed 'parameter@scope' -> (low, central, high)."""
    rows, variants = {}, {}
    for run, (prefix, _) in RUNS.items():
        target = rows if not prefix else variants.setdefault(prefix.split(":")[1], {})
        for tier, pol in TIER_POLICY.items():
            m = metrics["runs"][run][pol]
            scope = f"{prefix}policy:{tier}"
            for param in TABLE_COLUMNS:
                st = m[param]
                lo, c, hi = _sig(st["lo"]), _sig(st["mean"]), _sig(st["hi"])
                _check(lo <= c <= hi, f"{pol} {run} {param}: interval [{lo}, {hi}] excludes the mean {c}")
                target[f"{param}@{scope}"] = (lo, c, hi)
            rel = m["relative_recall_loss"]
            low, central, high = (max(0.0, v) for v in (rel["lo"], rel["mean"], rel["hi"]))
            if run == "mvtec_independent":  # parts in view for only 8 frames (short-defect runs, MVTec AD)
                b8 = short_recall(short, "BASELINE", SHORT_BOUND_FRAMES)
                high = max(high, 1.0 - short_recall(short, pol, SHORT_BOUND_FRAMES) / b8)
            target[f"recall_loss@{scope}"] = tuple(_sig(v) for v in (low, central, high))
    mu = float(macros["MuReviews"])
    rows["review_time@all"] = (30.0, 3600.0 / mu, 90.0)  # central from Paper A; bounds are assumptions
    rows["glare_burst_rate@all"] = (10.0, float(macros["GlareRateAssumed"]), 60.0)  # Paper A's assumption
    return rows, variants


def verify_against_paper(metrics: Dict, tables: Dict[str, Dict[str, list]], macros: Dict[str, str]) -> Dict[str, str]:
    """Check the recomputed values against Paper A's printed tables and macros. Returns a provenance note
    for every value Paper A does not print."""
    unprinted = {}
    for run, (_, table_name) in RUNS.items():
        for pol in TIER_POLICY.values():
            m = metrics["runs"][run][pol]
            if table_name is None:  # printed as macros
                stem = POLICY_MACRO.get(pol)
                if stem is None:
                    unprinted[f"{run}/{pol}"] = "computed with Paper A's code; not printed in Paper A"
                    continue
                for param, (macro, dec) in VISA_RHO_MACROS.items():
                    name = f"{macro}{stem}"
                    _check(name in macros, f"Paper A lacks macro {name}")
                    _check(_close(m[param]["mean"], macros[name], dec), f"{name}: recomputed {m[param]['mean']} vs printed {macros[name]}")
                continue
            table = tables[table_name]
            if pol not in table:
                unprinted[f"{run}/{pol}"] = "computed with Paper A's code; not printed in Paper A"
                continue
            for param, (col, dec, _) in TABLE_COLUMNS.items():
                _check(_close(m[param]["mean"], table[pol][col], dec),
                       f"{run} {pol} {param}: recomputed {m[param]['mean']} vs printed {table[pol][col]}")
            _check(_close(m["sustained_recall"]["mean"], table[pol][RECALL_COLUMN], 2),
                   f"{run} {pol} sustained recall: recomputed {m['sustained_recall']['mean']} vs printed {table[pol][RECALL_COLUMN]}")
    for pol, stem in POLICY_MACRO.items():
        for param, (_, dec, macro_stem) in TABLE_COLUMNS.items():
            for side, key in (("Lo", "lo"), ("Hi", "hi")):
                name = f"{macro_stem}{stem}{side}"
                if name in macros:
                    v = metrics["runs"]["mvtec_independent"][pol][param][key]
                    _check(_close(v, macros[name], dec), f"{name}: recomputed {v} vs printed {macros[name]}")
    return unprinted


def import_paper_a(repo: Path, out: Path = SNAPSHOT, ref: str = "HEAD", python: Path | None = None) -> Path:
    """Snapshot the Paper A measurements at commit `ref` (pass the snapshot's source_commit to reproduce
    the committed snapshot exactly). `python` is Paper A's interpreter (default: <repo>/.venv/bin/python)."""
    commit = _git(repo, "rev-parse", f"{ref}^{{commit}}").strip()
    python = Path(python) if python else repo / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"Paper A interpreter not found at {python}; pass --paper-a-python")
    hashes = {p: _git(repo, "rev-parse", f"{commit}:{p}").strip() for p in HASHED_PATHS}
    with tempfile.TemporaryDirectory() as tmp:
        blob = _git(repo, "archive", "--format=tar", commit, *ARCHIVE_PATHS, binary=True)
        with tarfile.open(fileobj=BytesIO(blob)) as tf:
            tf.extractall(tmp, filter="data")
        res = subprocess.run([str(python), str(HELPER), *TIER_POLICY.values()], cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Paper A helper failed:\n{res.stderr[-3000:]}")
        metrics = json.loads(res.stdout)
        root = Path(tmp)
        macros = parse_macros((root / "paper/generated_metrics.tex").read_text(encoding="utf-8"))
        tables = {t: parse_table((root / f"paper/tables/{t}.tex").read_text(encoding="utf-8"))
                  for t in ("ablation", "ablation_rho", "ablation_visa")}
        short = json.loads((root / "results/short_defect_recall.json").read_text(encoding="utf-8"))
    missing = [m for m in QUOTED_MACROS if m not in macros]
    if missing:
        raise KeyError(f"Paper A revision {commit[:7]} lacks macros {missing}")
    unprinted = verify_against_paper(metrics, tables, macros)
    for pol in ("BASELINE", "EMA_ONLY", "EMA_KOFN", "FULL_POLICY"):
        rec = metrics["runs"]["mvtec_independent"][pol]["sustained_recall"]["mean"]
        _check(abs(rec - 1.0) < 1e-12, f"{pol}: MVTec independent-score sustained recall {rec} != 1")
    rows, variants = build_rows(metrics, short, macros)
    snapshot = {
        "source_repository": repo.name,
        "source_commit": commit,
        "source_object_hashes": hashes,
        "statistics": "means over (category, replicate) units (MVTec AD: "
                      f"{metrics['runs']['mvtec_independent']['FULL_POLICY']['nominal_false_alarm_rate']['n_units']}, VisA: "
                      f"{metrics['runs']['visa_independent']['FULL_POLICY']['nominal_false_alarm_rate']['n_units']}); 95% percentile "
                      f"bootstrap over categories, B = {metrics['B_BOOT']} (recomputed with Paper A's scripts/build_paper_assets.py "
                      "and checked against its printed values)",
        "units": {run: {"units": metrics["runs"][run]["FULL_POLICY"]["nominal_false_alarm_rate"]["n_units"],
                        "categories": metrics["runs"][run]["FULL_POLICY"]["nominal_false_alarm_rate"]["n_categories"]}
                  for run in RUNS},
        "tier_policy": TIER_POLICY,
        "not_printed_in_paper_a": unprinted,
        "policy_metrics": metrics["runs"],
        "quoted_macros": {m: macros[m] for m in QUOTED_MACROS},
        "short_defect_recall": {p: {str(x["length"]): x["recall"]["mean"] for x in short["policies"][p]["per_length"]}
                                for p in TIER_POLICY.values()},
        "registry_rows": {k: list(v) for k, v in sorted(rows.items())},
        "variant_rows": {name: {k: list(v) for k, v in sorted(vr.items())} for name, vr in sorted(variants.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def load_snapshot() -> Dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def snapshot_rows(include_variants: bool = True) -> Dict[str, Tuple[float, float, float]]:
    snap = load_snapshot()
    rows = {k: tuple(v) for k, v in snap["registry_rows"].items()}
    if include_variants:
        for vrows in snap.get("variant_rows", {}).values():
            rows.update({k: tuple(v) for k, v in vrows.items()})
    return rows


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use scripts/01_import_upstream.py")

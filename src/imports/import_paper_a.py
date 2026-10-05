"""
Import the Paper A measurements used by this study from the Paper A repository.

Everything is read from one *committed* Paper A revision (git show <commit>:<path>),
never from a working copy:

* results/ablation/ablation_summary.json -- per-policy means and 95 % bootstrap
  intervals of the nominal false-alert rate (good parts only), the alerts per
  sustained-defect episode, the mean detection delay and the routing recall;
* paper/tables/ablation.tex -- alerts per 1-3 frame glare burst (reported without
  intervals);
* paper/generated_metrics.tex -- reviewer service rate and assumed glare rate;
* results/short_defect_recall.json -- recall of each policy for defects visible a fixed number of frames,
  which bounds the recall that persistence filtering loses.

The snapshot records the commit and blob hashes and the resulting registry rows.
It is committed here, so the pipeline runs without the upstream repository;
scripts/02 checks that every registry row derived from Paper A equals the snapshot.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

from src.paths import PAPER_A_EXPORT_DIR

SUMMARY_PATH = "results/ablation/ablation_summary.json"
TABLE_PATH = "paper/tables/ablation.tex"
METRICS_PATH = "paper/generated_metrics.tex"
SHORT_PATH = "results/short_defect_recall.json"
# Upper bound of the persistence recall loss: the shortest defect visibility (frames) at which the full
# policy keeps a mean recall of at least 0.95 (Paper A); the central value is the loss for parts in view
# for at least one second, i.e. the sustained episodes of the reference scenarios.
SHORT_BOUND_FRAMES, SHORT_CENTRAL_FRAMES = 8, 30
SNAPSHOT = PAPER_A_EXPORT_DIR / "paper_a_metrics.json"

# Paper B alert tier -> Paper A policy
TIER_POLICY = {
    "B0_Raw": "BASELINE",
    "B1_EMA": "EMA_ONLY",
    "B2_EMA_kofN": "EMA_KOFN",
    "B3_Full_Policy": "FULL_POLICY",
}
# registry parameter -> (Paper A workload, metric)
MEASURED = {
    "nominal_false_alarm_rate": ("nominal", "false_alerts_per_hour"),
    "alerts_per_defect": ("sustained_defects", "alerts_per_episode"),
    "detection_delay": ("sustained_defects", "mean_delay_frames"),
}

# Paper A macros quoted verbatim in the Paper B text (recorded for provenance).
QUOTED_MACROS = ("MuReviews", "GlareRateAssumed", "SustainedDelayFull", "NominalFABaseline", "NominalFAFull",
                 "AlertsPerEpisodeBaseline", "AlertsPerEpisodeFull", "GlarePerBurstBaseline", "GlarePerBurstEmaKofn",
                 "GlarePerBurstFull", "LatGpuMean", "LatVisionMean", "MemoryBankSize", "LoadCrossFull", "ShortK", "ShortN")

_MACRO = re.compile(r"\\(?:new|provide)command\{\\(\w+)\}\{(.*)\}\s*$")


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
    return res.stdout


def _sig(x: float) -> float:
    return float(f"{x:.6g}")


def parse_macros(tex: str) -> Dict[str, str]:
    out = {}
    for line in tex.splitlines():
        m = _MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).replace("{,}", "").replace(r"\,", "").replace(r"\%", "%").strip()
    return out


def parse_glare_per_burst(table_tex: str) -> Dict[str, float]:
    """Column 2 ('alerts/burst') of Paper A's ablation table, keyed by policy name."""
    out = {}
    for line in table_tex.splitlines():
        m = re.match(r"\\texttt\{([A-Z_\\]+)\}\s*&\s*([^&]+)&\s*([^&]+)&", line.strip())
        if m:
            out[m.group(1).replace("\\_", "_")] = float(m.group(3).replace("{,}", "").strip())
    return out


def persistence_loss(short: Dict, frames: int) -> float:
    """Recall of single-frame thresholding minus that of the full policy for defects of `frames` frames."""
    rec = {p: {x["length"]: x["recall"]["mean"] for x in short["policies"][p]["per_length"]} for p in ("BASELINE", "FULL_POLICY")}
    return max(0.0, rec["BASELINE"][frames] - rec["FULL_POLICY"][frames])


def build_rows(summary: Dict, glare: Dict[str, float], macros: Dict[str, str],
               short: Dict | None = None) -> Dict[str, Tuple[float, float, float]]:
    """Registry rows keyed 'parameter@scope' -> (low, central, high)."""
    rows: Dict[str, Tuple[float, float, float]] = {}
    sc = summary["scenarios"]
    for tier, pol in TIER_POLICY.items():
        for param, (workload, metric) in MEASURED.items():
            st = sc[workload][pol][metric]
            rows[f"{param}@policy:{tier}"] = (_sig(st["ci_lower"]), _sig(st["mean"]), _sig(st["ci_upper"]))
        g = glare[pol]
        rows[f"glare_alerts_per_burst@policy:{tier}"] = (g, g, g)
        recall = sc["sustained_defects"][pol]["routing_recall"]["mean"]
        if abs(recall - 1.0) > 1e-12:
            raise ValueError(f"Paper A {pol}: sustained-defect recall {recall} != 1; the shared-recall model would not hold")
    mu = float(macros["MuReviews"])
    rows["review_time@all"] = (30.0, 3600.0 / mu, 90.0)  # central from Paper A; bounds are assumptions
    g_rate = float(macros["GlareRateAssumed"])
    rows["glare_burst_rate@all"] = (10.0, g_rate, 60.0)  # Paper A's assumption; bounds are assumptions
    if short is not None:
        rows["persistence_recall_loss@all"] = (0.0, _sig(persistence_loss(short, SHORT_CENTRAL_FRAMES)),
                                               _sig(persistence_loss(short, SHORT_BOUND_FRAMES)))
    return rows


def import_paper_a(repo: Path, out: Path = SNAPSHOT, ref: str = "HEAD") -> Path:
    """Snapshot the Paper A measurements at commit `ref` (pass the snapshot's source_commit
    to reproduce the committed snapshot exactly)."""
    commit = _git(repo, "rev-parse", f"{ref}^{{commit}}").strip()
    files = {p: _git(repo, "show", f"{commit}:{p}") for p in (SUMMARY_PATH, TABLE_PATH, METRICS_PATH, SHORT_PATH)}
    blobs = {p: _git(repo, "rev-parse", f"{commit}:{p}").strip() for p in files}
    summary = json.loads(files[SUMMARY_PATH])
    macros = parse_macros(files[METRICS_PATH])
    glare = parse_glare_per_burst(files[TABLE_PATH])
    missing = [p for p in TIER_POLICY.values() if p not in glare] + [m for m in QUOTED_MACROS if m not in macros]
    if missing:
        raise KeyError(f"Paper A revision {commit[:7]} lacks {missing}")
    short = json.loads(files[SHORT_PATH])
    rows = build_rows(summary, glare, macros, short)
    snapshot = {
        "source_repository": repo.name,
        "source_commit": commit,
        "source_blobs": blobs,
        "tier_policy": TIER_POLICY,
        "sustained_defect_recall": {pol: summary["scenarios"]["sustained_defects"][pol]["routing_recall"]["mean"]
                                    for pol in TIER_POLICY.values()},
        "quoted_macros": {m: macros[m] for m in QUOTED_MACROS},
        "short_defect_recall": {p: {str(x["length"]): x["recall"]["mean"] for x in short["policies"][p]["per_length"]}
                                for p in TIER_POLICY.values()},
        "registry_rows": {k: list(v) for k, v in sorted(rows.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def load_snapshot() -> Dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def snapshot_rows() -> Dict[str, Tuple[float, float, float]]:
    return {k: tuple(v) for k, v in load_snapshot()["registry_rows"].items()}

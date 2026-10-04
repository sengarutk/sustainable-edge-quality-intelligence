"""Committed results must be exactly what the code produces, and the repo must be path-portable."""

import json
import subprocess
import sys

import pytest

from src.experiments.monte_carlo import run_monte_carlo
from src.io_utils import _round
from src.params import load_scenario
from src.paths import PROCESSED_DIR, REPO_ROOT, SCENARIOS

TEXT_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".json", ".csv", ".tex", ".md", ".cff", ".txt", ".bib")


def test_monte_carlo_summary_matches_committed():
    committed = json.loads((PROCESSED_DIR / "monte_carlo_summary.json").read_text(encoding="utf-8"))
    for sc in SCENARIOS:
        summary, _, _ = run_monte_carlo(load_scenario(sc), n_draws=10000, seed=42)
        assert _round(summary) == committed[sc]


@pytest.mark.parametrize("name", ["regime_outcomes.csv", "comparisons.csv"])
def test_scenario_outputs_match_committed(tmp_path, name):
    # Regenerate into tmp_path: the committed files must never be overwritten by a test.
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "04_run_scenarios.py"), "--out-dir", str(tmp_path)],
                   check=True, capture_output=True)
    committed = (PROCESSED_DIR / name).read_bytes().replace(b"\r\n", b"\n")
    assert (tmp_path / name).read_bytes() == committed


@pytest.mark.parametrize("pattern", ["/home/", "/mnt/c/", "C:\\Users", "C:/Users"])
def test_no_machine_specific_paths(pattern):
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    offenders = [f for f in tracked.splitlines()
                 if f.endswith(TEXT_SUFFIXES) and f != "tests/test_reproducibility.py" and (REPO_ROOT / f).exists()
                 and pattern in (REPO_ROOT / f).read_text(encoding="utf-8", errors="ignore")]
    assert not offenders, offenders

#!/usr/bin/env bash
# End-to-end reproduction. Usage:
#   bash scripts/reproduce_all.sh             # analysis from committed measurements (no GPU needed)
#   bash scripts/reproduce_all.sh --measure   # also re-import Paper A (needs its checkout and .venv) and re-run the
#                                             # GPU energy and detector measurements (NVIDIA GPU + NVML, MVTec AD, VisA)
# Set PYTHON to choose the interpreter (default: python3 on PATH).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--measure" ]]; then
  # Re-derive the committed Paper A snapshot at the commit it pins (not upstream HEAD).
  pin="$("$PY" -c 'import json; print(json.load(open("data/raw/paper_a_exports/paper_a_metrics.json"))["source_commit"])')"
  "$PY" scripts/01_import_upstream.py --ref "$pin" --update-registry
  "$PY" scripts/03_measure_energy.py
  "$PY" scripts/03b_measure_detector.py
fi

steps=(03c_import_jetson 02_validate_sources 04_run_scenarios 05_run_sensitivity 06_run_monte_carlo
       07_run_break_even 07b_run_operating_points 08_generate_figures 09_generate_tables 10_generate_macros)
for s in "${steps[@]}"; do
  echo "== $s"
  "$PY" "scripts/$s.py"
done
"$PY" scripts/sync_paper_assets.py
"$PY" -m pytest -q
"$PY" scripts/verify_math_proofs.py
echo "Reproduction complete."

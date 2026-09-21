#!/usr/bin/env bash
# ==============================================================================
# Master One-Command End-to-End Reproduction Pipeline for Paper B
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="/home/sengar/miniconda3/bin/python"
PYTEST_BIN="/home/sengar/miniconda3/bin/pytest"

echo "=========================================================================="
echo "STARTING FULL REPRODUCTION FOR SUSTAINABLE-EDGE-QUALITY-INTELLIGENCE"
echo "Target Root: $REPO_ROOT"
echo "Interpreter: $PYTHON_BIN"
echo "=========================================================================="

echo "[1/11] Ingesting upstream Flagship 1 and Flagship 4 benchmark contracts..."
$PYTHON_BIN scripts/01_import_flagship_outputs.py

echo "[2/11] Validating authoritative parameter source registry..."
$PYTHON_BIN scripts/02_validate_sources.py

echo "[3/11] Running live hardware energy profiling on RTX 4050 GPU..."
$PYTHON_BIN scripts/03_measure_energy.py

echo "[4/11] Evaluating 5-tier policy hierarchy across Scenarios A, B, and C..."
$PYTHON_BIN scripts/04_run_quality_scenarios.py

echo "[6/11] Executing 11-parameter tornado sensitivity sweeps..."
$PYTHON_BIN scripts/06_run_sensitivity.py

echo "[7/11] Running 10,000-draw Monte Carlo uncertainty simulations (seed=42)..."
$PYTHON_BIN scripts/07_run_monte_carlo.py

echo "[8/11] Solving analytical and empirical break-even frontiers..."
$PYTHON_BIN scripts/08_run_break_even.py

echo "[9/11] Generating 6 publication vector PDF and 300 DPI PNG figures..."
$PYTHON_BIN scripts/09_generate_figures.py

echo "[10/11] Generating LaTeX summary tables..."
$PYTHON_BIN scripts/10_generate_tables.py

echo "[11/11] Validating empirical paper assertions and exporting TeX macros..."
$PYTHON_BIN scripts/11_validate_paper_claims.py

echo "=========================================================================="
echo "RUNNING COMPREHENSIVE INVARIANT AND CONSERVATION TEST SUITE"
echo "=========================================================================="
$PYTEST_BIN -v tests/

echo "=========================================================================="
echo "ALL PIPELINE STAGES AND VERIFICATION TESTS COMPLETED WITH ZERO FAILURES!"
echo "=========================================================================="
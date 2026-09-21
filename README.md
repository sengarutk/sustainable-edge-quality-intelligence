# Sustainable Edge Quality Intelligence (Paper B)

Production-grade research platform accompanying Paper B:
**"Quantifying Material, Energy, Carbon, and Human-Workload Trade-offs of AI-Enabled Defect Prevention in Edge Manufacturing Inspection"**

## Overview
This platform bridges physical edge inference benchmarking and operational sustainability modeling across 4 critical pillars:
1. **Material Savings ($\Delta M$)**: Scrap reduction vs circular recovery.
2. **Operational Energy ($\Delta E$)**: Active edge GPU compute, review workstation display/compute, and machine rework.
3. **Embodied & Grid Carbon ($\Delta C$)**: Embodied carbon of avoided scrap vs edge compute grid footprint and escape penalties.
4. **Human Workload ($\Delta H$ & $\rho$)**: Review queue utilization and operator fatigue mitigation.

## Repository Layout
- `configs/scenarios/`: Scenario parameters (Precision Component, Machined Metal, High-Value Module).
- `data/`: Raw upstream exports, physical energy traces, and schemas.
- `src/`:
  - `imports/`: Upstream importers for Flagship 1 and Flagship 4.
  - `quality/`: Defect taxonomy and delay-sensitive intervention model.
  - `sustainability/`: Mass, energy, carbon, workload, and SQI accounting engines.
  - `experiments/`: Physical GPU energy benchmarking, Monte Carlo, sensitivity, break-even analysis.
- `scripts/`: End-to-end reproduction scripts (01 through 11).
- `tests/`: Strict invariant, conservation, and reproducibility test suite.
- `results/`: Processed metrics, figures, and LaTeX tables.

## Quickstart
```bash
# Run full reproduction pipeline
bash scripts/reproduce_all.sh

# Run comprehensive test suite
pytest -v tests/
```
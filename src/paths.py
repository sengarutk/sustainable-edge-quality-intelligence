"""Repository-relative paths shared by all pipeline stages (no absolute paths anywhere)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "configs"
SCENARIO_DIR = CONFIG_DIR / "scenarios"
POLICY_CONFIG = CONFIG_DIR / "policies.yaml"
SQI_CONFIG = CONFIG_DIR / "sqi_weights.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SOURCE_REGISTRY = RAW_DATA_DIR / "source_registry.csv"
ENERGY_TRACE_DIR = RAW_DATA_DIR / "energy_measurements"
JETSON_TRACE_DIR = RAW_DATA_DIR / "energy_measurements_jetson"
FLAGSHIP1_EXPORT_DIR = RAW_DATA_DIR / "flagship1_exports"
PAPER_A_EXPORT_DIR = RAW_DATA_DIR / "paper_a_exports"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_DIR / "raw"
ENERGY_SUMMARY = RESULTS_RAW_DIR / "energy_summary.json"  # workstation GPU (NVML)
JETSON_SUMMARY = RESULTS_RAW_DIR / "energy_summary_jetson.json"  # embedded module (INA3221), primary platform
PROCESSED_DIR = RESULTS_DIR / "processed"
RESULTS_FIG_DIR = RESULTS_DIR / "figures"
RESULTS_TABLE_DIR = RESULTS_DIR / "tables"
RESULTS_MACROS = RESULTS_DIR / "paper_b_generated_metrics.tex"

PAPER_DIR = REPO_ROOT / "paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"
PAPER_TABLE_DIR = PAPER_DIR / "tables"
PAPER_MACROS = PAPER_DIR / "paper_b_generated_metrics.tex"

SCENARIOS = ("precision_component", "machined_metal", "high_value_component")

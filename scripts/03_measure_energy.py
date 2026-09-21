#!/usr/bin/env python3
"""
Step 3: Run live hardware energy profiling on RTX 4050 GPU.
"""

from src.experiments.energy_benchmark import run_benchmark


def main():
    print("=== Step 03: Live Hardware Power and Energy Benchmark ===")
    summary = run_benchmark(cycles=300, warmup=50)
    print("Live physical energy measurement finished.")
    print(f"Target: {summary['device_name']}")
    for stage, metrics in summary['stages'].items():
        print(f"  Stage {stage:20s}: P_active={metrics.get('p_active_mean_w', 0.0):.2f} W, e_frame={metrics.get('e_frame_wh', 0.0)*1000:.4f} mWh")
    print("Step 03 completed successfully.\n")


if __name__ == "__main__":
    main()
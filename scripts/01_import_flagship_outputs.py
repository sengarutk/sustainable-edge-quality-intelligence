#!/usr/bin/env python3
"""
Step 1: Import upstream contracts from Flagship 1 and Flagship 4.
"""

from pathlib import Path
import sys

from src.imports.import_flagship1 import import_flagship1_benchmarks
from src.imports.import_flagship4 import import_flagship4_policies


def main():
    print("=== Step 01: Importing Upstream Benchmark Contracts ===")
    p1 = import_flagship1_benchmarks()
    p4 = import_flagship4_policies()
    print(f"Flagship 1 benchmarks -> {p1}")
    print(f"Flagship 4 / Paper A policies -> {p4}")
    print("Step 01 completed successfully.\n")


if __name__ == "__main__":
    main()
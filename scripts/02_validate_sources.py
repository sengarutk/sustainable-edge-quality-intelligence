#!/usr/bin/env python3
"""
Step 2: Validate authoritative source registry against strict Pydantic schema.
"""

from pathlib import Path
from data.schemas.source_registry_schema import validate_registry_file

REGISTRY_PATH = Path("/home/sengar/sustainable-edge-quality-intelligence/data/raw/source_registry.csv")


def main():
    print("=== Step 02: Auditing Parameter Source Registry ===")
    count = validate_registry_file(REGISTRY_PATH)
    print(f"Validated {count} parameters in source registry.")
    print("Zero schema violations detected.")
    print("Step 02 completed successfully.\n")


if __name__ == "__main__":
    main()
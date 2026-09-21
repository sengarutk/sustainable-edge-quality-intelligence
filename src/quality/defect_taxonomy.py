"""
Defect taxonomy and batch representation for edge manufacturing inspection.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict


class DefectClass(str, Enum):
    CLASS_A_REWORKABLE = "class_a_reworkable"
    CLASS_B_SCRAP_PRONE = "class_b_scrap_prone"
    CLASS_C_ESCAPE = "class_c_escape_sensitive"


@dataclass(frozen=True)
class DefectBatch:
    """Stores defect counts per class for N = 1000 inspected units."""
    total_inspected: int
    total_defects: float
    counts: Dict[DefectClass, float]

    def validate(self):
        sum_counts = sum(self.counts.values())
        if abs(sum_counts - self.total_defects) > 1e-6:
            raise ValueError(f"Batch defect counts sum ({sum_counts}) != total defects ({self.total_defects})")
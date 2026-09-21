#!/usr/bin/env python3
import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_METRICS = REPO_ROOT / "results" / "paper_b_generated_metrics.tex"
PAPER_METRICS = REPO_ROOT / "paper" / "paper_b_generated_metrics.tex"
MAIN_TEX = REPO_ROOT / "paper" / "main.tex"

def sha256_file(p: Path) -> str:
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    print("=== [Sync] Portable Paper Asset Synchronizer ===")
    if not RESULTS_METRICS.exists() or not PAPER_METRICS.exists():
        raise RuntimeError("Missing metrics file in results/ or paper/. Run script 11 first.")
    
    h_src = sha256_file(RESULTS_METRICS)
    h_dst = sha256_file(PAPER_METRICS)
    assert h_src == h_dst and h_src != "", f"SHA-256 divergence detected! src={h_src}, dst={h_dst}"
    print(f">> PASS: Verified SHA-256 dual-output parity ({h_src[:12]}...)")

    if MAIN_TEX.exists():
        tex = MAIN_TEX.read_text(encoding="utf-8")
        tex = re.sub(r'\\input\{(?:\.\./)?results/paper_b_generated_metrics\.tex\}', r'\\input{paper_b_generated_metrics.tex}', tex)
        tex = re.sub(r'C_{\\mathcal\{e\}scape}', r'C_{\\text{esc}}', tex)
        tex = re.sub(r'\\Delta\\overline\{H\}', r'\\Delta H', tex)
        tex = re.sub(r'C_{\\text\{escape\}}\s*&=\s*N_{\\text\{escape\}}\s*\\cdot\s*C_{\\text\{escape\}}', r'C_{\\text{esc}} &= N_{\\text{esc}} \\cdot C_{\\text{esc}}', tex)
        MAIN_TEX.write_text(tex, encoding="utf-8")
        print(">> PASS: main.tex paths and LaTeX macro symbols sanitized.")

if __name__ == "__main__":
    main()

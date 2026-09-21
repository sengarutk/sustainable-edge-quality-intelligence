#!/usr/bin/env python3
import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = REPO_ROOT / "results" / "paper_b_generated_metrics.tex"
PAPER_METRICS = REPO_ROOT / "paper" / "paper_b_generated_metrics.tex"
MAIN_TEX = REPO_ROOT / "paper" / "main.tex"
SCRIPT_11 = REPO_ROOT / "scripts" / "11_validate_paper_claims.py"

def sha256_file(p: Path) -> str:
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    print("=== [Sync] Portable Paper Asset Synchronizer ===")
    if not RESULTS_METRICS.exists():
        raise RuntimeError(f"Source metrics missing: {RESULTS_METRICS}. Run reproduction first.")
    
    PAPER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    content_bytes = RESULTS_METRICS.read_bytes()
    PAPER_METRICS.write_bytes(content_bytes)
    
    h_src = sha256_file(RESULTS_METRICS)
    h_dst = sha256_file(PAPER_METRICS)
    assert h_src == h_dst and h_src != "", "SHA-256 mismatch between results/ and paper/ metrics!"
    print(f">> PASS: Verified SHA-256 parity ({h_src[:12]}...) for generated_metrics.tex")

    # Update main.tex input path to portable filename
    if MAIN_TEX.exists():
        tex = MAIN_TEX.read_text(encoding="utf-8")
        tex = re.sub(r'\\input\{(?:\.\./)?results/paper_b_generated_metrics\.tex\}', r'\\input{paper_b_generated_metrics.tex}', tex)
        tex = re.sub(r'C_{\\mathcal\{e\}scape}', r'C_{\\text{esc}}', tex)
        tex = re.sub(r'\\Delta\\overline\{H\}', r'\\Delta H', tex)
        MAIN_TEX.write_text(tex, encoding="utf-8")
        print(">> PASS: main.tex input paths and math symbols cleaned.")

    # Verify script 11 contains dual export
    if SCRIPT_11.exists():
        s11_text = SCRIPT_11.read_text(encoding="utf-8")
        if "paper_tex" in s11_text:
            print(">> INFO: scripts/11_validate_paper_claims.py verified with dual-output export.")
        else:
            print(">> WARN: scripts/11_validate_paper_claims.py missing dual-output export.")

if __name__ == "__main__":
    main()

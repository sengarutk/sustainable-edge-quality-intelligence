#!/usr/bin/env python3
import hashlib, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_METRICS = REPO_ROOT / "results" / "paper_b_generated_metrics.tex"
PAPER_METRICS = REPO_ROOT / "paper" / "paper_b_generated_metrics.tex"
MAIN_TEX = REPO_ROOT / "paper" / "main.tex"

def sha256_file(p: Path) -> str:
    assert p.exists() and p.stat().st_size > 0, f"Critical zero-size or missing file: {p}"
    return hashlib.sha256(p.read_bytes()).hexdigest()

def get_desktop_dir() -> Path:
    # Resolve Windows Desktop via WSL mount if available, else fallback to ~/Desktop
    win_desktop = Path("/mnt/c/Users/senga/Desktop")
    if win_desktop.parent.exists():
        win_desktop.mkdir(parents=True, exist_ok=True)
        return win_desktop
    fallback = Path.home() / "Desktop"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def main():
    print("=== [Sync] Robust Portable Paper Asset Synchronizer ===")
    assert RESULTS_METRICS.exists(), f"Source metrics missing: {RESULTS_METRICS}"
    
    PAPER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RESULTS_METRICS, PAPER_METRICS)
    
    h_src = sha256_file(RESULTS_METRICS)
    h_dst = sha256_file(PAPER_METRICS)
    assert h_src == h_dst, f"SHA-256 divergence detected! src={h_src}, dst={h_dst}"
    print(f">> PASS: Verified SHA-256 byte parity ({h_src[:12]}...)")

    if MAIN_TEX.exists():
        tex = MAIN_TEX.read_text(encoding="utf-8")
        # FIXED: single-backslash LaTeX command matching
        tex = tex.replace(r"\input{../results/paper_b_generated_metrics.tex}", r"\input{paper_b_generated_metrics.tex}") # safety net
        tex = tex.replace(r"\input{../results/paper_b_generated_metrics.tex}", r"\input{paper_b_generated_metrics.tex}")
        tex = tex.replace(r"\input{results/paper_b_generated_metrics.tex}", r"\input{paper_b_generated_metrics.tex}")
        tex = tex.replace(r"C_{\mathcal{e}scape}", r"C_{	ext{esc}}")
        tex = tex.replace(r"\Delta\overline{H}", r"\Delta H")
        tex = tex.replace(r"C_{	ext{escape}} &= N_{	ext{escape}} \cdot C_{	ext{escape}}", r"C_{	ext{esc}} &= N_{	ext{esc}} \cdot C_{	ext{esc}}")
        MAIN_TEX.write_text(tex, encoding="utf-8")
        print(">> PASS: main.tex literal symbol cleaning applied deterministically.")

if __name__ == "__main__":
    main()

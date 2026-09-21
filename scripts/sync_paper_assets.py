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
    # Explicit Windows user profile resolution or fallback to WSL user home desktop
    win_d = Path("/mnt/c/Users/senga/Desktop")
    if win_d.parent.exists():
        win_d.mkdir(parents=True, exist_ok=True)
        return win_d
    c_users = Path("/mnt/c/Users")
    if c_users.exists():
        for user_dir in c_users.iterdir():
            if user_dir.is_dir() and user_dir.name not in ["Public", "Default", "Default User", "All Users"]:
                desk = user_dir / "Desktop"
                if desk.exists():
                    return desk
    hb = Path.home() / "Desktop"
    hb.mkdir(parents=True, exist_ok=True)
    return hb

def main():
    print("=== [Sync] Clean Portable Paper Asset Synchronizer ===")
    assert RESULTS_METRICS.exists(), f"Source metrics missing: {RESULTS_METRICS}"
    
    PAPER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RESULTS_METRICS, PAPER_METRICS)
    
    h_src = sha256_file(RESULTS_METRICS)
    h_dst = sha256_file(PAPER_METRICS)
    assert h_src == h_dst, f"SHA-256 divergence detected! src={h_src}, dst={h_dst}"
    print(f">> PASS: Verified SHA-256 byte parity ({h_src[:12]}...)")

    if MAIN_TEX.exists():
        tex = MAIN_TEX.read_text(encoding="utf-8")
        tex = tex.replace(r"\input{../results/paper_b_generated_metrics.tex}", r"\input{paper_b_generated_metrics.tex}")
        tex = tex.replace(r"\input{results/paper_b_generated_metrics.tex}", r"\input{paper_b_generated_metrics.tex}")
        tex = tex.replace(r"C_{\mathcal{e}scape}", r"C_{\text{esc}}")
        tex = tex.replace(r"\Delta\overline{H}", r"\Delta H")
        tex = tex.replace(r"C_{\text{escape}} &= N_{\text{escape}} \cdot C_{\text{escape}}", r"C_{\text{esc}} &= N_{\text{esc}} \cdot C_{\text{esc}}")
        MAIN_TEX.write_text(tex, encoding="utf-8")
        print(">> PASS: main.tex literal symbol cleaning applied deterministically.")

if __name__ == "__main__":
    main()

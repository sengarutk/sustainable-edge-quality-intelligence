#!/usr/bin/env python3
import json, hashlib, zipfile, shutil, os
from pathlib import Path

root = Path("/home/sengar/sustainable-edge-quality-intelligence")

def get_desktop_dir() -> Path:
    c_users = Path("/mnt/c/Users")
    if c_users.exists():
        for user_dir in c_users.iterdir():
            if user_dir.is_dir() and user_dir.name not in ["Public", "Default", "Default User", "All Users"]:
                desk = user_dir / "Desktop"
                if desk.exists():
                    desk.mkdir(parents=True, exist_ok=True)
                    return desk
        win_d = Path("/mnt/c/Users/senga/Desktop")
        win_d.mkdir(parents=True, exist_ok=True)
        return win_d
    hb = Path.home() / "Desktop"
    hb.mkdir(parents=True, exist_ok=True)
    return hb

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def case_insensitive_glob(parent: Path, suffixes: list) -> list:
    out = []
    if parent.exists():
        for p in parent.rglob("*"):
            if p.is_file() and p.suffix.lower() in [s.lower() for s in suffixes]:
                out.append(p)
    return out

def main():
    print("=== [Pack] Hardened Release Packager & Sidecar Ledger ===")
    
    # 0. Enforce pre-pack dual-output sync parity
    res_m = root / "results" / "paper_b_generated_metrics.tex"
    pap_m = root / "paper" / "paper_b_generated_metrics.tex"
    if res_m.exists():
        pap_m.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res_m, pap_m)
        assert sha256_file(res_m) == sha256_file(pap_m), "Pre-pack metric parity check failed!"

    # 1. Manifest bounds
    req_assets = [
        root / "paper" / "main.tex",
        root / "paper" / "references.bib",
        pap_m,
        res_m,
        root / "paper" / "main.pdf"
    ]
    for a in req_assets:
        assert a.exists() and a.stat().st_size > 0, f"CRITICAL MISSING/EMPTY ASSET: {a}"

    fig_files = [f for f in (root / "paper" / "figures").glob("*") if f.suffix.lower() == ".pdf"]
    tab_files = [t for t in (root / "results" / "tables").glob("*") if t.suffix.lower() == ".tex"]
    assert len(fig_files) >= 6, f"Missing vector figures: found {len(fig_files)}"
    assert len(tab_files) >= 3, f"Missing tex tables: found {len(tab_files)}"

    desk = get_desktop_dir()

    # 2. Master Package Build
    dist_dir = Path("/tmp/final_master_build/paper_b_release")
    if dist_dir.exists(): shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["paper", "results", "configs", "src", "scripts", "tests"]:
        s_sub = root / sub
        if s_sub.exists():
            shutil.copytree(s_sub, dist_dir / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".git*"))
    for root_f in ["README.md", "pyproject.toml", "LICENSE", "CITATION.cff"]:
        rf = root / root_f
        if rf.exists(): shutil.copy(rf, dist_dir / root_f)

    master_zip = desk / "paper_b_final_master_package.zip"
    if master_zip.exists(): master_zip.unlink()
    with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in dist_dir.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(dist_dir.parent))
    assert master_zip.stat().st_size > 100000, "Master zip invalid"

    # 3. Overleaf Flat Ready Bundle
    overleaf_dir = Path("/tmp/overleaf_flat_build")
    if overleaf_dir.exists(): shutil.rmtree(overleaf_dir)
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "paper" / "main.tex", overleaf_dir / "main.tex")
    shutil.copy(root / "paper" / "references.bib", overleaf_dir / "references.bib")
    shutil.copy(pap_m, overleaf_dir / "paper_b_generated_metrics.tex")
    ofig = overleaf_dir / "figures"; ofig.mkdir()
    for p_f in fig_files: shutil.copy(p_f, ofig / p_f.name)
    otab = overleaf_dir / "tables"; otab.mkdir()
    for t_f in tab_files: shutil.copy(t_f, otab / t_f.name)

    overleaf_zip = desk / "paper_b_overleaf_ready.zip"
    if overleaf_zip.exists(): overleaf_zip.unlink()
    with zipfile.ZipFile(overleaf_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in overleaf_dir.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(overleaf_dir))
    assert overleaf_zip.stat().st_size > 50000, "Overleaf zip invalid"

    # 4. Camera-ready standalone pdf
    cam_pdf = desk / "paper_b_camera_ready.pdf"
    shutil.copy(root / "paper" / "main.pdf", cam_pdf)
    assert cam_pdf.stat().st_size > 100000, "Camera ready pdf invalid"

    # 5. Emit Machine-Readable Sidecar Ledger on Desktop
    manifest = {
        "release_tag": "v0.1.0",
        "repository": "https://github.com/sengarutk/sustainable-edge-quality-intelligence",
        "artifacts": {
            "paper_b_final_master_package.zip": {
                "size_bytes": master_zip.stat().st_size,
                "sha256": sha256_file(master_zip)
            },
            "paper_b_overleaf_ready.zip": {
                "size_bytes": overleaf_zip.stat().st_size,
                "sha256": sha256_file(overleaf_zip)
            },
            "paper_b_camera_ready.pdf": {
                "size_bytes": cam_pdf.stat().st_size,
                "sha256": sha256_file(cam_pdf)
            }
        }
    }
    manifest_p = desk / "release_manifest.json"
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f">> SUCCESS: Hardened packages and generated manifest at {manifest_p}")

if __name__ == "__main__":
    main()

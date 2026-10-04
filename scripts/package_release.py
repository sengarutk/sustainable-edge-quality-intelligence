#!/usr/bin/env python3
"""Build release artefacts into dist/ (never outside the repository unless --dest is given).

  * compiles paper/main.tex (pdflatex + bibtex + 2x pdflatex) and fails on LaTeX errors,
    undefined references/citations or overfull boxes wider than 1 pt;
  * writes reproducible zips (fixed timestamps, sorted entries) of the source tree and
    of a flat Overleaf bundle, plus the camera-ready PDF;
  * writes release_manifest.json with sizes and SHA-256 hashes; version from pyproject.toml.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)
SOURCE_DIRS = ("configs", "data", "paper", "results", "scripts", "src", "tests", ".github")
SOURCE_FILES = ("README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "requirements-lock.txt", ".gitignore")
LATEX_JUNK = (".aux", ".bbl", ".blg", ".log", ".out", ".pdf")


def run(cmd):
    res = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stdout.write(res.stdout[-4000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def compile_paper():
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]):
        run(cmd)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    problems = re.findall(r"(LaTeX Warning: (?:Reference|Citation).*undefined.*|There were undefined references.*)", log)
    problems += [m for m in re.findall(r"Overfull \\[hv]box \((\d+\.\d+)pt too \w+\)", log) if float(m) > 1.0]
    if problems:
        raise SystemExit(f"LaTeX build is not clean: {problems[:10]}")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    print(f"paper compiled cleanly ({pages.group(1)} pages)")


def files_for(paths, base):
    out = []
    for p in paths:
        p = base / p
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts
                    and not (f.parent == PAPER and f.suffix in LATEX_JUNK)]
    return sorted(out)


def write_zip(zpath, entries):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in sorted(entries):
            info = zipfile.ZipInfo(arcname, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, src.read_bytes())


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=ROOT / "dist")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_paper_assets.py"), "--check"], check=True)
    if not args.skip_compile:
        compile_paper()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    name = f"sustainable-edge-quality-intelligence-v{version}"
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    src_zip = dest / f"{name}-source.zip"
    write_zip(src_zip, [(f"{name}/{f.relative_to(ROOT).as_posix()}", f) for f in files_for(SOURCE_DIRS + SOURCE_FILES, ROOT)])
    ol = [("main.tex", PAPER / "main.tex"), ("references.bib", PAPER / "references.bib"),
          ("paper_b_generated_metrics.tex", PAPER / "paper_b_generated_metrics.tex")]
    ol += [(f"figures/{p.name}", p) for p in sorted((PAPER / "figures").glob("*.pdf"))]
    ol += [(f"tables/{p.name}", p) for p in sorted((PAPER / "tables").glob("*.tex"))]
    ol_zip = dest / f"{name}-overleaf.zip"
    write_zip(ol_zip, ol)
    pdf = dest / f"{name}-paper.pdf"
    shutil.copyfile(PAPER / "main.pdf", pdf)

    manifest = {"version": version, "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                                  for p in (src_zip, ol_zip, pdf)}}
    (dest / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

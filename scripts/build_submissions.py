#!/usr/bin/env python3
"""Build and check the two submission versions of the manuscript.

  arXiv       paper/main.tex   (IEEEtran journal format)
  IEEE Access paper/access.tex (official IEEE Access template, paper/ieee_access_template/)

Both wrap the same body (paper/body.tex) and back matter (paper/back.tex), so their text cannot drift.
For each version the script assembles a flat source folder under dist/submissions/<version>/ that
contains only the files the manuscript uses, compiles it (pdflatex, bibtex, 2x pdflatex) and fails on
any LaTeX error, undefined reference or citation, or overfull box wider than 1 pt. It then checks the
venue's limits and writes, next to each source zip, the compiled PDF and (for arXiv) the plain-text
abstract and metadata to paste into the submission form:

  arXiv:        abstract <= 1,920 characters (arXiv metadata rule); no page or word limit for the paper;
                the .bbl is included and named after the main file; no unused or output files.
  IEEE Access:  abstract one paragraph of 150-250 words without abbreviations (template instructions);
                3-10 keywords; page count reported against the recommended maximum of 20 pages
                (appendices excluded); author biography and AI disclosure in the acknowledgment present.

Usage: python scripts/build_submissions.py [--only arxiv|ieee_access]
"""

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "dist" / "submissions"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)
ARXIV_ABSTRACT_MAX_CHARS = 1920
ACCESS_ABSTRACT_WORDS = (150, 250)
ACCESS_KEYWORDS = (3, 10)
ACCESS_RECOMMENDED_PAGES = 20
VERSIONS = {"arxiv": "main.tex", "ieee_access": "access.tex"}
SHARED = ("body.tex", "back.tex", "paper_b_generated_metrics.tex", "references.bib")


def macros() -> dict:
    text = (PAPER / "paper_b_generated_metrics.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))


def plain(tex: str) -> str:
    """LaTeX abstract -> plain text with every macro resolved (arXiv metadata, word counts)."""
    m = macros()
    tex = re.sub(r"\\(\w+)\{\}", lambda g: m.get(g.group(1), g.group(0)), tex)
    tex = re.sub(r"\\(\w+)", lambda g: m.get(g.group(1), g.group(0)), tex)
    rep = [("{,}", ","), ("\\%", "%"), ("~", " "), ("$_2$", "2"), ("$", ""), ("--", "-"), ("\\,", " "), ("{", ""), ("}", "")]
    for a, b in rep:
        tex = tex.replace(a, b)
    if "\\" in tex:
        raise SystemExit(f"unresolved LaTeX in abstract: {tex[tex.index(chr(92)):][:60]}")
    return re.sub(r"\s+", " ", tex).strip()


def env(tex: str, name: str) -> str:
    m = re.search(rf"\\begin\{{{name}\}}(.*?)\\end\{{{name}\}}", tex, re.S)
    if not m:
        raise SystemExit(f"no {name} environment")
    return m.group(1).strip()


def used_files(sources) -> set:
    """Figures and tables referenced by the manuscript files."""
    files = set()
    for src in sources:
        t = (PAPER / src).read_text(encoding="utf-8")
        files |= {f for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)}
        files |= {f if f.endswith(".tex") else f + ".tex" for f in re.findall(r"\\input\{(tables/[^}]+)\}", t)}
    return files


def assemble(version: str) -> Path:
    main = VERSIONS[version]
    d = OUT / version
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in (main,) + SHARED:
        shutil.copyfile(PAPER / f, d / f)
    for f in sorted(used_files((main, "body.tex", "back.tex"))):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAPER / f, d / f)
    if version == "ieee_access":
        for f in (PAPER / "ieee_access_template").iterdir():
            if f.name != "README.md":
                shutil.copyfile(f, d / f.name)
    return d


# Overfull boxes that the official IEEE Access template produces on its own sample document (page header
# while \output is active, 505.12 pt; title block at \maketitle, 9.27 pt); they are not manuscript problems.
ACCESS_TEMPLATE_OVERFULL = {"505.12177": "while \\output is active", "9.2679": "maketitle"}


def overfull(log: str, main_tex: str, template_artifacts: bool):
    lines = main_tex.split("\n")
    out = []
    for m in re.finditer(r"Overfull \\[hv]box \((\d+\.\d+)pt too \w+\) (has occurred while \\output is active|in paragraph at lines (\d+)--\d+)?", log):
        width, where, line = m.group(1), m.group(2) or "", m.group(3)
        if float(width) <= 1.0:
            continue
        if template_artifacts and width in ACCESS_TEMPLATE_OVERFULL:
            kind = ACCESS_TEMPLATE_OVERFULL[width]
            if kind in where or (line and "\\maketitle" in lines[int(line) - 1]):
                continue
        out.append(f"overfull {width}pt {where}")
    return out


def compile_tex(d: Path, main: str, template_artifacts: bool = False) -> dict:
    stem = Path(main).stem
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main], ["bibtex", stem],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main]):
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-3000:])
            raise SystemExit(f"{d.name}: {' '.join(cmd)} failed")
    log = (d / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    blg = (d / f"{stem}.blg").read_text(encoding="utf-8", errors="replace")
    problems = re.findall(r"(LaTeX Warning: (?:Reference|Citation).*undefined.*|There were undefined references.*)", log)
    problems += overfull(log, (d / main).read_text(encoding="utf-8"), template_artifacts)
    problems += re.findall(r"Warning--.*", blg)
    if problems:
        raise SystemExit(f"{d.name}: LaTeX build is not clean: {problems[:8]}")
    pages = int(re.search(r"Output written on \S+ \((\d+) pages", log).group(1))
    return {"pages": pages}


def appendix_start_page(d: Path, stem: str) -> int | None:
    """First page of the appendices, from the .aux (label of the first appendix section)."""
    aux = (d / f"{stem}.aux").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\\newlabel\{app:repro\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(m.group(1)) if m else None


def bundle(d: Path, keep_bbl: bool) -> Path:
    stem_files = []
    for f in sorted(d.rglob("*")):
        if f.is_dir() or f.suffix in (".aux", ".log", ".blg", ".out", ".pdf") and f.parent == d:
            continue
        if f.suffix == ".bbl" and not keep_bbl:
            continue
        stem_files.append(f)
    z = OUT / f"{d.name}_source.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in stem_files:
            info = zipfile.ZipInfo(f.relative_to(d).as_posix(), ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, f.read_bytes())
    return z


def build_arxiv() -> None:
    d = assemble("arxiv")
    info = compile_tex(d, "main.tex")
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract = plain(env(tex, "abstract"))
    if len(abstract) > ARXIV_ABSTRACT_MAX_CHARS:
        raise SystemExit(f"arXiv abstract has {len(abstract)} characters (limit {ARXIV_ABSTRACT_MAX_CHARS})")
    title = re.sub(r"\s+", " ", re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1)).strip()
    n_fig = len([f for f in used_files(("main.tex", "body.tex", "back.tex")) if f.startswith("figures/")])
    n_tab = len([f for f in used_files(("main.tex", "body.tex", "back.tex")) if f.startswith("tables/")]) + 1  # + checklist
    comments = (f"{info['pages']} pages, {n_fig} figures, {n_tab} tables. Code, data and the scripts that generate every "
                "number: https://github.com/sengarutk/sustainable-edge-quality-intelligence ")
    meta = OUT / "arxiv_metadata.txt"
    meta.write_text(f"TITLE\n{title}\n\nAUTHORS\nUtkarsh Sengar\n\nABSTRACT ({len(abstract)} of {ARXIV_ABSTRACT_MAX_CHARS} "
                    f"characters)\n{abstract}\n\nCOMMENTS\n{comments}\n\nPRIMARY CATEGORY (suggested)\ncs.CY "
                    f"(Computers and Society); cross-list cs.CV, cs.LG\n", encoding="utf-8")
    shutil.copyfile(d / "main.pdf", OUT / "arxiv.pdf")
    z = bundle(d, keep_bbl=True)
    print(f"arXiv: {info['pages']} pages; abstract {len(abstract)}/{ARXIV_ABSTRACT_MAX_CHARS} characters; "
          f"{n_fig} figures; source {z.relative_to(ROOT)}")


def build_ieee_access() -> None:
    d = assemble("ieee_access")
    info = compile_tex(d, "access.tex", template_artifacts=True)
    tex = (PAPER / "access.tex").read_text(encoding="utf-8")
    abstract = plain(env(tex, "abstract"))
    words = len(abstract.split())
    lo, hi = ACCESS_ABSTRACT_WORDS
    if not lo <= words <= hi:
        raise SystemExit(f"IEEE Access abstract has {words} words (must be {lo}-{hi})")
    abbreviations = sorted(set(re.findall(r"\b[A-Z]{2,}\b", abstract)))
    if abbreviations:
        raise SystemExit(f"IEEE Access abstract contains abbreviations: {abbreviations}")
    keywords = [k.strip() for k in env(tex, "keywords").rstrip(".").split(",")]
    if not ACCESS_KEYWORDS[0] <= len(keywords) <= ACCESS_KEYWORDS[1] or keywords != sorted(keywords, key=str.lower):
        raise SystemExit(f"IEEE Access keywords must be 3-10 and alphabetical: {keywords}")
    for needed in ("IEEEbiography", "\\section*{Acknowledgment}"):
        if needed not in tex + (PAPER / "back.tex").read_text(encoding="utf-8"):
            raise SystemExit(f"IEEE Access version lacks {needed}")
    app = appendix_start_page(d, "access")
    body_pages = (app - 1) if app else info["pages"]
    shutil.copyfile(d / "access.pdf", OUT / "ieee_access.pdf")
    z = bundle(d, keep_bbl=True)
    status = "within" if body_pages < ACCESS_RECOMMENDED_PAGES else "ABOVE"
    print(f"IEEE Access: {info['pages']} pages ({body_pages} before the appendix, {status} the recommended "
          f"{ACCESS_RECOMMENDED_PAGES}); abstract {words} words, no abbreviations; {len(keywords)} keywords; "
          f"source {z.relative_to(ROOT)}")
    if body_pages >= ACCESS_RECOMMENDED_PAGES:
        raise SystemExit("IEEE Access version exceeds the recommended page count")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=tuple(VERSIONS))
    args = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_paper_assets.py"), "--check"], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "arxiv"):
        build_arxiv()
    if args.only in (None, "ieee_access"):
        build_ieee_access()


if __name__ == "__main__":
    main()

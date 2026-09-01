"""High-signal, deterministic quality checks for the course source tree."""

from __future__ import annotations

import re
import ast
import sqlite3
import sys
from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse"
errors: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def check_toc() -> None:
    config = yaml.safe_load((ROOT / "myst.yml").read_text(encoding="utf-8"))

    def visit(items):
        for item in items:
            if "file" in item:
                path = ROOT / item["file"]
                if not path.exists():
                    fail(f"TOC target does not exist: {item['file']}")
            visit(item.get("children", []))

    visit(config["project"]["toc"])


def check_notebooks() -> None:
    for path in ROOT.rglob("*.ipynb"):
        if "_build" in path.parts:
            continue
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
        except Exception as exc:
            fail(f"Invalid notebook {path.relative_to(ROOT)}: {exc}")


def check_references() -> None:
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    cited: set[str] = set()
    for path in ROOT.rglob("*.md"):
        if "_build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for group in re.findall(r"\[@([^\]]+)\]", text):
            cited.update(part.strip().lstrip("@") for part in group.split(";") if part.strip())
    missing = sorted(cited - keys)
    if missing:
        fail(f"Missing bibliography keys: {missing}")


def check_known_regressions() -> None:
    patterns = {
        r"\bSET\s+\w+\s*=\s*\d{1,3},\d{3}\b": "comma in SQL numeric literal",
        r"period\s*=\s*1\b": "period=1 seasonal decomposition",
        r"rolling\(365\)": "365 observations described as an annual window",
        r"Energy \(TJ\)": "Eurostat output labelled TJ",
        r"wind direction has no effect.*speed is high": "reversed low-speed wind-direction claim",
    }
    corpus = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".py", ".sql"} and "_build" not in p.parts
    )
    for pattern, label in patterns.items():
        if re.search(pattern, corpus, flags=re.IGNORECASE):
            fail(f"Known regression found: {label}")


def check_python_code_cells() -> None:
    marker = re.compile(r"```\{code-cell\}\s+python[^\n]*\n(.*?)```", re.DOTALL)
    for path in ROOT.rglob("*.md"):
        if "_build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, match in enumerate(marker.finditer(text), start=1):
            try:
                ast.parse(match.group(1))
            except SyntaxError as exc:
                fail(f"Invalid Python code-cell {path.relative_to(ROOT)} #{number}: {exc.msg}")


def check_database() -> None:
    path = ROOT / "section5" / "energy_generation.db"
    if not path.exists():
        fail("Generated SQLite database is missing")
        return
    with sqlite3.connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        duplicates = connection.execute("""
            SELECT COUNT(*) FROM (
              SELECT country, month, source, COUNT(*) AS n
              FROM generation GROUP BY country, month, source HAVING n > 1
            )
        """).fetchone()[0]
    if integrity != "ok" or duplicates:
        fail(f"SQLite validation failed: integrity={integrity}, duplicate keys={duplicates}")


def main() -> int:
    check_toc()
    check_notebooks()
    check_references()
    check_known_regressions()
    check_python_code_cells()
    check_database()
    if errors:
        print("COURSE VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("COURSE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

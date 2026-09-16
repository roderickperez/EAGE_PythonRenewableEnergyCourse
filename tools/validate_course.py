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


def check_energy_curriculum() -> None:
    source_pages = sorted((ROOT / "section6").glob("*Energy.md"))
    expected_sources = {"hydroelectricEnergy.md", "solarEnergy.md", "windEnergy.md", "geothermalEnergy.md"}
    if {path.name for path in source_pages} != expected_sources:
        fail("Section 6 must contain hydro, solar, wind, and geothermal source lessons")

    for path in source_pages:
        text = path.read_text(encoding="utf-8")
        exercise_count = len(re.findall(r"^:::\{admonition\} Exercise \d+[^\n]*\n:class: note", text, flags=re.MULTILINE))
        if exercise_count < 6:
            fail(f"{path.relative_to(ROOT)} has only {exercise_count} guided exercises; expected at least 6")
        for required in ["## Learning goals", "## Concepts and equations", "## Common mistakes"]:
            if required not in text:
                fail(f"{path.relative_to(ROOT)} is missing {required}")

    quiz_dir = ROOT / "section7" / "renewableEnergyquizzes"
    core_quizzes = [
        quiz_dir / "hydroelectricEnergy.md",
        quiz_dir / "solarEnergy.md",
        quiz_dir / "windEnergy.md",
        quiz_dir / "geothermalEnergy.md",
    ]
    for path in core_quizzes:
        question_count = len(re.findall(r"^:::\{admonition\} Quiz \d+", path.read_text(encoding="utf-8"), flags=re.MULTILINE))
        if question_count < 8:
            fail(f"{path.relative_to(ROOT)} has only {question_count} quiz questions; expected at least 8")


def execute_energy_code_cells() -> None:
    """Execute every core Markdown lesson in its own namespace, in cell order."""
    import contextlib
    import io
    import matplotlib

    matplotlib.use("Agg")
    marker = re.compile(r"```\{code-cell\}\s+python[^\n]*\n(.*?)```", re.DOTALL)
    pages = [p for p in sorted(ROOT.rglob('*.md'))
             if '_build' not in p.parts and 'node_modules' not in p.parts]
    count = 0
    for path in pages:
        namespace = {"__name__": "__course_validation__"}
        text = path.read_text(encoding="utf-8")
        for number, match in enumerate(marker.finditer(text), start=1):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(match.group(1), str(path), "exec"), namespace)
                count += 1
            except Exception as exc:
                fail(f"Runtime error in {path.relative_to(ROOT)} code-cell #{number}: {type(exc).__name__}: {exc}")
                break
        try:
            namespace.get("plt").close("all") if namespace.get("plt") is not None else None
        except Exception:
            pass
    print(f"Executed {count} Markdown Python cells")


def check_workbook() -> None:
    """Check distribution, per-exercise references, reveal controls and independence."""
    import contextlib
    import io
    import matplotlib.pyplot as plt
    text = (ROOT / 'section7/renewableExercises.md').read_text(encoding='utf-8')
    levels = re.findall(r'\*\*Difficulty:\*\* (Easy|Medium|Hard)', text)
    if {level: levels.count(level) for level in ['Easy', 'Medium', 'Hard']} != {
        'Easy': 5, 'Medium': 10, 'Hard': 5,
    }:
        fail('Workbook must have exactly 5 easy, 10 medium and 5 hard exercises')
    exercises = re.split(r'\n## Exercise \d+ — ', text)[1:]
    for i, exercise in enumerate(exercises, 1):
        if '**Reference:**' not in exercise or '[@' not in exercise:
            fail(f'Workbook exercise {i} has no reference')
        if exercise.count('::::{dropdown} Solution') != 1 or ':open:' in exercise:
            fail(f'Workbook exercise {i} must have one initially closed solution')
        cells = re.findall(r'```\{code-cell\} python\n(.*?)```', exercise, re.S)
        if len(cells) != 2 or 'assert ' not in cells[-1]:
            fail(f'Workbook exercise {i} needs a starter and checked solution')
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(cells[-1], f'workbook-{i}', 'exec'), {})
        except Exception as exc:
            fail(f'Workbook exercise {i} does not run independently: {exc}')
        finally:
            plt.close('all')


def check_energy_boundaries() -> None:
    """Regression checks independent of the worked-example input choices."""
    import numpy as np
    marker = re.compile(r'```\{code-cell\} python\n(.*?)```', re.S)
    wind = {}
    source = (ROOT / 'section6/windEnergy.md').read_text(encoding='utf-8')
    exec(marker.findall(source)[0], wind)
    assert np.allclose(wind['wind_at_height'](6, np.array([10, 100])), [6, 6 * 10**0.14])
    assert np.allclose(wind['turbine_power_mw']([0, 3, 12, 25]), [0, 0, 3, 0])
    for invalid in [-1, np.nan, np.inf]:
        try:
            wind['turbine_power_mw'](invalid)
        except ValueError:
            pass
        else:
            fail(f'Wind curve accepted invalid speed {invalid}')


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
    check_energy_curriculum()
    execute_energy_code_cells()
    check_workbook()
    check_energy_boundaries()
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

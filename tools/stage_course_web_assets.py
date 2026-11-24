"""Stage locally built interactive assets into the static MyST output."""
from pathlib import Path
import json
import os
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "EAGE_PythonRenewableEnergyCourse"
HOME_PATH = re.compile(r"(?:/home/[^/\r\n]+/|/Users/[^/\r\n]+/|[A-Za-z]:[\\/]Users[\\/][^\\/\r\n]+[\\/])", re.I)


def output_contains_machine_path(value):
    if isinstance(value, str):
        return bool(HOME_PATH.search(value))
    if isinstance(value, dict):
        return any(output_contains_machine_path(item) for item in value.values())
    if isinstance(value, list):
        return any(output_contains_machine_path(item) for item in value)
    return False


def clear_machine_paths_from_notebook_outputs(output: Path) -> int:
    """Remove saved output containing a prior user's home path from generated notebooks."""
    changed = 0
    for path in output.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        dirty = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            outputs = cell.get("outputs", [])
            filtered = [item for item in outputs if not output_contains_machine_path(item)]
            if len(filtered) != len(outputs):
                cell["outputs"] = filtered
                cell["execution_count"] = None
                dirty = True
        if dirty:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            changed += 1
    return changed


# Pages with direct downloadable notebooks; their MyST links may be served as JSON.
NOTEBOOK_DOWNLOAD_PAGES = (
    "section0/introrenewableenergy", "section6/solarenergy", "section6/geothermalenergy",
    "section6/windenergy", "section6/hydroelectricenergy", "section7/workbook/general",
    "section7/workbook/hydroelectric", "section7/workbook/solar", "section7/workbook/wind",
    "section7/workbook/geothermal", "section5/download-to-database", "section5/world-energy-download",
    "section8/geothermal-project", "section8/projectintro", "section4/timeseriesenergyconsumption",
    "section5/sql-pandas", "section5/sql-python",
)


def verify_notebook_download_staging(output: Path) -> None:
    """Ensure the early download handler is staged on each direct-entry page."""
    staged = output / "notebook-download.js"
    source = BOOK / "notebook-download.js"
    if not staged.is_file() or staged.read_bytes() != source.read_bytes():
        raise ValueError("Notebook download handler is missing or differs from its source")
    for route in NOTEBOOK_DOWNLOAD_PAGES:
        page = output / route / "index.html"
        if not page.is_file():
            raise ValueError(f"Notebook download page missing: {route}")
        content = page.read_text(encoding="utf-8")
        head_end = content.lower().find("</head>")
        if head_end < 0 or "notebook-download.js" not in content[:head_end]:
            raise ValueError(f"Early notebook download handler missing from: {route}")


def main() -> None:
    for lab in ("hydro-lab", "solar-lab", "wind-lab", "geothermal-lab", "project-map"):
        source = BOOK / "static" / lab
        target = BOOK / "_build" / "html" / lab
        required = {"index.html", f"{lab}.js"}
        if lab == "hydro-lab":
            required.update({"fallback.svg", "head.html", "head.js", "balance.html", "balance.js", "project.html", "hydro-project.js", "worked-examples.html", "worked-examples.js"})
        if lab == "solar-lab":
            required.update({"resource.html", "resource.js", "project.html", "solar-project.js"})
        if lab == "wind-lab":
            required.update({"aerodynamics.html", "aerodynamics.js"})
        if lab == "geothermal-lab":
            required.update({"project.html", "geothermal-project.js", "drilling.html", "drilling.js", "closed-loop.html", "closed-loop.js", "multifield.html", "geothermal-multifield.js"})
        missing = sorted(name for name in required if not (source / name).is_file())
        if missing:
            raise SystemExit(f"Build {lab} before staging it: " + ", ".join(missing))
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        print(f"Staged {lab} at {target}")

    # MyST produces static pages and also navigates between them client-side.
    # Load the reader on every entry page so it can follow either route.
    output = BOOK / "_build" / "html"
    scripts = ("section-audio.js", "notebook-download.js")
    for script in scripts:
        shutil.copyfile(BOOK / script, output / script)
    for page in output.rglob("*.html"):
        if any(part in {"hydro-lab", "solar-lab", "wind-lab", "geothermal-lab", "project-map"} for part in page.relative_to(output).parts):
            continue
        content = page.read_text(encoding="utf-8")
        for script in scripts:
            if script in content:
                continue
            src = os.path.relpath(output / script, page.parent).replace(os.sep, "/")
            if script == "notebook-download.js" and "</head>" in content:
                content = content.replace("</head>", f'<script src="{src}"></script></head>', 1)
            elif script == "section-audio.js" and "</body>" in content:
                content = content.replace("</body>", f'<script defer src="{src}"></script></body>', 1)
        page.write_text(content, encoding="utf-8")
    verify_notebook_download_staging(output)
    count = clear_machine_paths_from_notebook_outputs(output)
    print(f"Cleared machine-specific saved output in {count} generated notebook(s)")


if __name__ == "__main__":
    main()

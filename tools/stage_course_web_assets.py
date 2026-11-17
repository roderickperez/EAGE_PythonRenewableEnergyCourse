"""Stage locally built interactive assets into the static MyST output."""
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "EAGE_PythonRenewableEnergyCourse"


def main() -> None:
    for lab in ("hydro-lab", "solar-lab", "wind-lab", "geothermal-lab", "project-map"):
        source = BOOK / "static" / lab
        target = BOOK / "_build" / "html" / lab
        required = {"index.html", f"{lab}.js"}
        if lab == "hydro-lab":
            required.update({"fallback.svg", "head.html", "head.js"})
        missing = sorted(name for name in required if not (source / name).is_file())
        if missing:
            raise SystemExit(f"Build {lab} before staging it: " + ", ".join(missing))
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        print(f"Staged {lab} at {target}")

    # MyST produces static pages and also navigates between them client-side.
    # Load the reader on every entry page so it can follow either route.
    output = BOOK / "_build" / "html"
    reader = output / "section-audio.js"
    shutil.copyfile(BOOK / "section-audio.js", reader)
    for page in output.rglob("*.html"):
        if any(part in {"hydro-lab", "solar-lab", "wind-lab", "geothermal-lab", "project-map"} for part in page.relative_to(output).parts):
            continue
        content = page.read_text(encoding="utf-8")
        if "section-audio.js" not in content and "</body>" in content:
            src = os.path.relpath(reader, page.parent).replace(os.sep, "/")
            page.write_text(content.replace("</body>", f'<script defer src="{src}"></script></body>', 1), encoding="utf-8")


if __name__ == "__main__":
    main()

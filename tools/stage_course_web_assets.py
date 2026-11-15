"""Stage locally built interactive assets into the static MyST output."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "EAGE_PythonRenewableEnergyCourse"


def main() -> None:
    for lab in ("hydro-lab", "solar-lab", "wind-lab", "geothermal-lab"):
        source = BOOK / "static" / lab
        target = BOOK / "_build" / "html" / lab
        required = {"index.html", f"{lab}.js"}
        if lab == "hydro-lab":
            required.add("fallback.svg")
        missing = sorted(name for name in required if not (source / name).is_file())
        if missing:
            raise SystemExit(f"Build {lab} before staging it: " + ", ".join(missing))
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        print(f"Staged {lab} at {target}")


if __name__ == "__main__":
    main()

"""Stage locally built interactive assets into the static MyST output."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EAGE_PythonRenewableEnergyCourse" / "static" / "hydro-lab"
TARGET = ROOT / "EAGE_PythonRenewableEnergyCourse" / "_build" / "html" / "hydro-lab"


def main() -> None:
    required = {"index.html", "hydro-lab.js", "fallback.svg"}
    missing = sorted(name for name in required if not (SOURCE / name).is_file())
    if missing:
        raise SystemExit("Build the hydro lab before staging it: " + ", ".join(missing))
    shutil.rmtree(TARGET, ignore_errors=True)
    shutil.copytree(SOURCE, TARGET)
    print(f"Staged hydro lab at {TARGET}")


if __name__ == "__main__":
    main()

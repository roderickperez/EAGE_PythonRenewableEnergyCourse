"""Execute and save the two course notebooks from any working directory."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse"

for relative in ("section4/timeSeriesEnergyConsumption.ipynb", "section5/SQL_Pandas.ipynb",
                 "section0/introRenewableEnergy.ipynb", "section6/solarEnergy.ipynb",
                 "section6/hydroelectricEnergy.ipynb", "section6/windEnergy.ipynb",
                 "section6/geothermalEnergy.ipynb"):
    path = ROOT / relative
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f"Executed and saved {relative}")

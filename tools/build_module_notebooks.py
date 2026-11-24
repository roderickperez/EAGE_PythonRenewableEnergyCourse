"""Generate the Colab companions for the seven new EAGE lessons."""

import hashlib
from pathlib import Path
import re

import nbformat

from notebook_links import rewrite_notebook_links

BOOK = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse"
REPOSITORY = "roderickperez/EAGE_PythonRenewableEnergyCourse"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/main/{BOOK.name}/"
COLAB = f"https://colab.research.google.com/github/{REPOSITORY}/blob/main/{BOOK.name}/"
LESSONS = {
    "section0/python-foundations.md": (),
    "section4/energy-data-analysis.md": ("eurostat_at_wind_2023.csv",),
    "section4/energy-maps.md": (),
    "section6/labs/wind-capacity.md": (),
    "section6/labs/river-flood-flow.md": (),
    "section6/labs/solar-irradiance.md": ("surfrad-bondville-2024-07-01.csv",),
    "section6/labs/geothermal-gradient.md": (
        "ihfc-gradient-sample.csv", "natural-earth-land.geojson", "synthetic-well-temperature.csv",
    ),
}
PACKAGES = {
    "numpy": "numpy>=1.26,<3",
    "pandas": "pandas>=2.2,<4",
    "matplotlib": "matplotlib>=3.8,<4",
    "seaborn": "seaborn>=0.13,<1",
    "geopandas": "geopandas>=1.0,<2",
    "folium": "folium>=0.20,<1",
}
CODE = re.compile(r"^```\{code-cell\} python\n(.*?)^```\s*$", re.M | re.S)


def setup_code(markdown, assets):
    required = {module: spec for module, spec in PACKAGES.items()
                if re.search(rf"^import {module}\b", markdown, re.M)}
    setup = f"""import importlib.util
import subprocess
import sys

required = {required!r}
missing = [spec for module, spec in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
"""
    if not assets:
        return setup
    hashes = {
        f"data/examples/{name}": hashlib.sha256((BOOK / "data/examples" / name).read_bytes()).hexdigest()
        for name in assets
    }
    return setup + f"""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

data_source = {RAW!r}
datasets = {hashes!r}
for relative_path, expected in datasets.items():
    target = Path(relative_path)
    if target.is_file() and sha256(target.read_bytes()).hexdigest() == expected:
        continue
    with urlopen(data_source + relative_path, timeout=30) as response:
        payload = response.read()
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"Course dataset checksum mismatch: {{relative_path}}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    temporary.write_bytes(payload)
    temporary.replace(target)
"""


def build(relative, assets):
    path = BOOK / relative
    markdown = path.read_text(encoding="utf-8")
    notebook_path = path.with_suffix(".ipynb")
    launch = f"[Open this lesson in Google Colab]({COLAB}{notebook_path.relative_to(BOOK).as_posix()})"
    if launch not in markdown:
        raise ValueError(f"Missing Colab link in {relative}")
    title, _, body = markdown.partition("\n")
    cells = [
        nbformat.v4.new_markdown_cell(title + "\n\nRun the setup cell first, then follow the lesson in order."),
        nbformat.v4.new_code_cell(setup_code(markdown, assets).strip()),
    ]
    body = body.replace(launch, "").replace(":::{admonition} Solution\n:class: dropdown", "### Worked solution")
    body = re.sub(r"^:::\s*$", "", body, flags=re.M)
    for index, part in enumerate(CODE.split(body)):
        part = part.strip()
        if not part:
            continue
        cell = (nbformat.v4.new_code_cell(part) if index % 2 else
                nbformat.v4.new_markdown_cell(rewrite_notebook_links(part, path)))
        cells.append(cell)
    for index, cell in enumerate(cells, 1):
        cell.id = f"cell-{index}"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)


if __name__ == "__main__":
    for lesson, assets in LESSONS.items():
        build(lesson, assets)

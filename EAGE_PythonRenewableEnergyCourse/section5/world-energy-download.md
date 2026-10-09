# Download energy data for any available country worldwide

## Goal and source

Download **Our World in Data's current energy dataset**, choose Austria or another available country, save an annual electricity series as CSV and SQLite, and preserve the source definitions. This is a separate worldwide option alongside the [European monthly Eurostat exercise](download-to-database.md).

Use the [current energy catalogue](https://catalog.ourworldindata.org/energy/owid_energy/) and its [documentation](https://catalog.ourworldindata.org/energy/owid_energy/readme.md). The old [GitHub repository](https://github.com/owid/energy-data) identifies its files as a legacy release that is no longer updated. The current release changed column names and primary-energy methodology; use its matching codebook. This exercise selects **electricity generation**, not total energy supply.

## 1. Download manually

1. Open the current catalogue above.
2. Download the [full CSV](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.csv), [codebook CSV](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.codebook.csv) and [README](https://catalog.ourworldindata.org/energy/owid_energy/readme.md). Save the actual file contents, not an HTML preview. The full CSV was approximately 9 MB when this exercise was checked; its size may change.
3. Keep the three files together, with retrieval date and URLs. Read the selected column's unit, definition and original providers in the codebook.
4. Choose a country code from the dataset. OWID uses `AUT` for Austria, whereas the Eurostat exercise uses `AT`.
5. Filter country **and year** before calculating. Leave missing observations missing. Countries and world/regional aggregates appear in the same dataset; adding both double-counts generation.

The code below performs these same downloads without an API key. Files are saved under `outputs/owid_world/raw/`; selected results go in a separate country/variable/period folder. HTTP or schema errors stop the example instead of replacing observations with invented data. Sources: [OWID catalogue documentation](https://catalog.ourworldindata.org/energy/owid_energy/readme.md), [pandas CSV reader](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

## 2. Download once and inspect the available countries

Run this cell first. In the sandbox choose **Edit this code → Run Python**; in Jupyter/Colab run the cells in order with pandas installed. The browser uses `pyfetch`; local Python uses `urllib`. If the network fails, retry later or use already saved files in local Python; this lesson does not silently switch to a snapshot. [Pyodide HTTP](https://pyodide.org/en/stable/usage/api/python-api/http.html), [urllib](https://docs.python.org/3/library/urllib.request.html).

```{code-cell} python
import sys, io, json, hashlib, sqlite3, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

BASE = "https://catalog.ourworldindata.org/energy/owid_energy/"
raw_folder = Path("outputs/owid_world/raw")
raw_folder.mkdir(parents=True, exist_ok=True)
download_record = {}
for filename in ["owid_energy.csv", "owid_energy.codebook.csv", "readme.md"]:
    url = BASE + filename
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch
        response = await pyfetch(url)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        raw_bytes = await response.bytes()
    else:
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as response:
            raw_bytes = response.read()
    (raw_folder / filename).write_bytes(raw_bytes)
    download_record[filename] = dict(url=url, bytes=len(raw_bytes),
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(raw_bytes).hexdigest())
data = pd.read_csv(raw_folder / "owid_energy.csv")
codebook = pd.read_csv(raw_folder / "owid_energy.codebook.csv").fillna("")
if not {"country", "iso_code", "year"}.issubset(data.columns):
    raise ValueError("Dataset schema changed: inspect the current documentation")
entities = data[["country", "iso_code"]].drop_duplicates().sort_values("country")
entities.to_csv(raw_folder / "available_entities.csv", index=False)
print("Rows:", len(data), "Columns:", len(data.columns))
print("Country/region lookup saved:", (raw_folder / "available_entities.csv").resolve())
print(entities[entities.iso_code.isin(["AUT", "DEU", "BRA", "IND", "USA", "OWID_WRL"])].to_string(index=False))
```

To find another country, inspect `available_entities.csv` or run `print(entities[entities.country.str.contains("Japan", case=False, na=False)])`. Use the exact returned code. A code does not guarantee non-missing values for every variable/year.

## 3. Choose a country, variable and years

Edit only the four settings at the top, then rerun this cell and the save cell. You do not need to download the full dataset again when changing countries.

| Place | `COUNTRY_CODE` | Place | `COUNTRY_CODE` |
|---|---|---|---|
| Austria | `AUT` | Germany | `DEU` |
| Brazil | `BRA` | India | `IND` |
| United States | `USA` | Japan | `JPN` |
| South Africa | `ZAF` | World aggregate | `OWID_WRL` |

Choose `wind_electricity_twh`, `solar_electricity_twh`, `hydro_electricity_twh` or `renewables_electricity_twh`. These are annual generation values in **TWh** in this release, not MW capacity or percentage shares. One TWh equals 1,000 GWh. The code checks the codebook and prints the underlying providers rather than assuming every indicator has the same source. [Current codebook](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.codebook.csv); [NIST SI conversions](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

```{code-cell} python
COUNTRY_CODE = "AUT"  # e.g. DEU, BRA, IND, USA, JPN, ZAF; OWID_WRL = world
VARIABLE = "wind_electricity_twh"
START_YEAR, END_YEAR = 2019, 2023

allowed = {"wind_electricity_twh", "solar_electricity_twh",
           "hydro_electricity_twh", "renewables_electricity_twh"}
if VARIABLE not in allowed or VARIABLE not in data.columns:
    raise ValueError("Choose a documented electricity-generation column from the list")
if not isinstance(START_YEAR, int) or not isinstance(END_YEAR, int) or not 1800 <= START_YEAR <= END_YEAR <= 2100:
    raise ValueError("Use integer years in increasing order between 1800 and 2100")
metadata_rows = codebook.loc[codebook.column.eq(VARIABLE)]
if len(metadata_rows) != 1:
    raise ValueError("Missing/ambiguous variable metadata; inspect the codebook")
variable_metadata = metadata_rows.iloc[0].to_dict()
if "twh" not in variable_metadata["unit"].lower():
    raise ValueError("Unit changed; inspect the source before converting")
country_rows = data.loc[data.iso_code.eq(COUNTRY_CODE)]
if country_rows.empty:
    raise ValueError("Unknown code: inspect available_entities.csv")
country_names = country_rows.country.unique()
if len(country_names) != 1:
    raise ValueError("Country code maps to multiple names; investigate")
country_name = country_names[0]
annual = country_rows.loc[country_rows.year.between(START_YEAR, END_YEAR), ["year", VARIABLE]]
if annual.year.duplicated().any():
    raise ValueError("Duplicate country-year observations")
# Reindex explicitly: absent years stay visible as missing, never zero.
selected = annual.set_index("year").reindex(range(START_YEAR, END_YEAR + 1))
selected.index.name = "year"
selected = selected.rename(columns={VARIABLE: "generation_twh"}).reset_index()
selected.insert(0, "country", country_name)
selected.insert(1, "iso_code", COUNTRY_CODE)
selected["generation_gwh"] = selected.generation_twh * 1000
values = selected.generation_twh.dropna()
if not values.map(math.isfinite).all() or values.lt(0).any():
    raise ValueError("Unexpected non-finite or negative generation; inspect the source")
print(country_name, COUNTRY_CODE)
print(variable_metadata["title"], "| Unit:", variable_metadata["unit"])
print("Definition:", variable_metadata["description"])
print("Original sources:", variable_metadata["source"])
print(selected.to_string(index=False))
print(f"Coverage: {len(values)}/{len(selected)} reported years; missing stays missing")
if values.empty:
    print("No observations: choose another period/product or report unavailability.")
```

## 4. Save your selection as CSV and SQLite

This creates or replaces only the tutorial's `generation` table in its own selection folder. Its key is `(iso_code, year)`; the selected variable is recorded in the folder name and metadata. Missing values become SQL `NULL`, so report `COUNT(generation_twh)` alongside the sum. [SQLite aggregates](https://www.sqlite.org/lang_aggfunc.html), [pandas SQL export](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html).

```{code-cell} python
output = Path("outputs/owid_world") / f"{COUNTRY_CODE}_{VARIABLE}_{START_YEAR}_{END_YEAR}"
output.mkdir(parents=True, exist_ok=True)
selected.to_csv(output / "generation.csv", index=False)
provenance = dict(provider="Our World in Data", downloads=download_record,
    country=country_name, iso_code=COUNTRY_CODE, variable=VARIABLE,
    start_year=START_YEAR, end_year=END_YEAR, metadata=variable_metadata,
    grain="one selected country or aggregate per calendar year",
    conversion="generation_gwh = generation_twh * 1000")
(output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
connection = sqlite3.connect(output / "generation.sqlite")
try:
    connection.execute("DROP TABLE IF EXISTS generation")
    connection.execute("CREATE TABLE generation (country TEXT, iso_code TEXT, year INTEGER, generation_twh REAL, generation_gwh REAL, PRIMARY KEY (iso_code, year))")
    selected.to_sql("generation", connection, if_exists="append", index=False)
    count, sql_total = connection.execute(
        "SELECT COUNT(generation_twh), SUM(generation_twh) FROM generation WHERE iso_code = ?",
        (COUNTRY_CODE,)).fetchone()
finally:
    connection.close()
python_total = selected.generation_twh.sum(min_count=1)
assert count == selected.generation_twh.notna().sum()
if count:
    assert math.isclose(sql_total, python_total, rel_tol=1e-10)
    print(f"Observed-period sum: {sql_total:.3f} TWh across {count} reported years")
else:
    assert sql_total is None and pd.isna(python_total)
    print("No observed total; SQL NULL and pandas missing agree")
print("CSV:", (output / "generation.csv").resolve())
print("Metadata:", (output / "provenance.json").resolve())
print("SQLite:", (output / "generation.sqlite").resolve())
```

**Get the files onto your computer:** in the sandbox, open **Files, input and export**, paste each printed path into **Download a Python-created file (path)** and click **Download file**. In Colab, refresh the Files pane and download each file from its menu. In local Jupyter, open the printed folder in the file browser. Also retain the raw CSV, codebook and README from `outputs/owid_world/raw/`. Browser/Colab runtime files are temporary until exported. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 5. Exercises with increasing difficulty

1. **Easy — Austria:** run all three cells unchanged. Download your CSV and provenance, reopen the CSV, and identify country, variable, unit and reported-year count. Explain why five annual values cannot be labelled monthly data.
2. **Medium — Europe:** choose any available European country using the entity lookup, change `COUNTRY_CODE`, and rerun selection/save. Compare the same years and variable with Austria. Keep both output folders; verify coverage before comparing sums.
3. **Medium — Worldwide:** choose Brazil, India, Japan or another available country outside Europe. Download both wind and solar by changing `VARIABLE` and rerunning selection/save. Join the two CSVs on country code and year with `validate="one_to_one"`. Retain missing values.
4. **Hard — Country versus world:** select `OWID_WRL` for the same variable/years. For each year with both values reported and a positive world denominator, calculate `100 * country_generation / world_generation`. Explain why this is a share of world generation for that source, not a country's renewable share. Never add the world row to the country total.
5. **Hard — Compare providers:** use Austria, wind, 2023 in both lessons. Convert OWID TWh to GWh, check all twelve Eurostat months, and compare boundaries, coverage, source versions and revisions. Do not force agreement: an annual series from another provider is not automatically identical to Eurostat's monthly net-generation series.

These exercises are original course tasks using the cited provider definitions. Annual data cannot be converted into actual monthly observations by dividing by twelve. If a chosen country/year lacks data, report the gap and select another period; do not fabricate values.

[Download this lesson as a Jupyter notebook](world-energy-download.ipynb).

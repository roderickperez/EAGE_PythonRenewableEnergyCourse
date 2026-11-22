# Download energy data for any available country worldwide

## Goal and source

Download **Our World in Data's current energy dataset**, choose Austria or another available country, save an annual electricity series as CSV and SQLite, and preserve the source definitions. This is a separate worldwide option alongside the [European monthly Eurostat exercise](download-to-database.md).

Use the [current energy catalogue](https://catalog.ourworldindata.org/energy/owid_energy/) and its [documentation](https://catalog.ourworldindata.org/energy/owid_energy/readme.md). The old [GitHub repository](https://github.com/owid/energy-data) identifies its files as a legacy release that is no longer updated. The current release changed column names and primary-energy methodology; use its matching codebook. This exercise selects **electricity generation**, not total energy supply.

## 1. Download manually

1. Open the current catalogue above.
2. Download the [full CSV](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.csv), [codebook CSV](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.codebook.csv) and [README](https://catalog.ourworldindata.org/energy/owid_energy/readme.md). Save the actual file contents, not an HTML preview. The full CSV was approximately 9 MB when this exercise was checked; its size may change.
3. Keep the three files together, with retrieval date and URLs. Read the selected column's unit, definition and original providers in the codebook.
4. Choose a country code from the dataset. OWID uses `AUT` for Austria, whereas the Eurostat exercise uses `AT`. For World or another entity without a code, select its exact name in the `country` column; `WORLD` is a shortcut in the course code, not a code to assume is present in the source file.
5. Filter country **and year** before calculating. Leave missing observations missing. Countries and world/regional aggregates appear in the same dataset; adding both double-counts generation.

The code below performs these same downloads without an API key. Each successful download is kept in its own folder under `outputs/owid_world/raw/`; each export has a separate folder. Folders ending in `.incomplete` are failed attempts and must not be used. HTTP or schema errors stop the example instead of replacing observations with invented data. Sources: [OWID catalogue documentation](https://catalog.ourworldindata.org/energy/owid_energy/readme.md), [pandas CSV reader](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

## 2. Download once and inspect the available countries

Run this cell first. In the sandbox choose **Edit this code → Run Python**; in Jupyter/Colab run the cells in order with pandas installed. The browser uses `pyfetch`; local Python uses `urllib`. If the network fails, retry later or use already saved files in local Python; this lesson does not silently switch to a snapshot. [Pyodide HTTP](https://pyodide.org/en/stable/usage/api/python-api/http.html), [urllib](https://docs.python.org/3/library/urllib.request.html).

```{code-cell} python
import sys, io, json, hashlib, sqlite3, math, uuid
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

BASE = "https://catalog.ourworldindata.org/energy/owid_energy/"
# Invalidate earlier state before any operation that can fail.
for name in ("data", "codebook", "download_record", "selected", "selection_context"):
    globals().pop(name, None)
source_ready = False
run_id = uuid.uuid4().hex
raw_root = Path("outputs/owid_world/raw")
raw_folder = raw_root / (run_id + ".incomplete")
raw_folder.mkdir(parents=True, exist_ok=False)
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
completed_raw = raw_root / run_id
raw_folder.rename(completed_raw)
raw_folder = completed_raw
source_ready = True
print("Rows:", len(data), "Columns:", len(data.columns))
print("Country/region lookup saved:", (raw_folder / "available_entities.csv").resolve())
print(entities[entities.iso_code.isin(["AUT", "DEU", "BRA", "IND", "USA"]) | entities.country.eq("World")].to_string(index=False))
```

To find another country, inspect `available_entities.csv` or run `print(entities[entities.country.str.contains("Japan", case=False, na=False)])`. Use the exact returned code for a country. For an entity without a code, set `ENTITY_NAME` to its exact listed name (for example `Kosovo`); this takes precedence over `COUNTRY_CODE` and retains the source ISO value without inventing one. Clear `ENTITY_NAME` before switching back to any code. The special selector `WORLD` remains a shortcut for the exact `World` name. Regional aggregates and countries may overlap and must not be added together without checking definitions. A selector does not guarantee non-missing values for every variable/year. If a named entity has both missing and present source codes, or more than one code over its history, selection stops for inspection rather than filling or omitting those rows.

## 3. Choose a country, variable and years

Edit the settings at the top, then rerun this cell and the save cell. You do not need to download the full dataset again when changing countries.

| Place | `COUNTRY_CODE` | Place | `COUNTRY_CODE` |
|---|---|---|---|
| Austria | `AUT` | Germany | `DEU` |
| Brazil | `BRA` | India | `IND` |
| United States | `USA` | Japan | `JPN` |
| South Africa | `ZAF` | World aggregate | `WORLD` |

Choose `wind_electricity_twh`, `solar_electricity_twh`, `hydro_electricity_twh` or `renewables_electricity_twh`. These are annual generation values in **TWh** in this release, not MW capacity or percentage shares. One TWh equals 1,000 GWh. The code validates the unit and prints the definition and underlying providers rather than assuming every indicator has the same source. Review that definition before saving: the unit check does not establish that the provider’s meaning or coverage is unchanged. If the definition no longer describes the intended annual electricity generation, stop and investigate before comparing or converting it. [Current codebook](https://catalog.ourworldindata.org/energy/owid_energy/owid_energy.codebook.csv); [NIST SI conversions](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

```{code-cell} python
COUNTRY_CODE = "AUT"  # e.g. DEU, BRA, IND, USA, JPN, ZAF; WORLD = aggregate
ENTITY_NAME = ""  # optional exact name, e.g. "Kosovo"; overrides COUNTRY_CODE
VARIABLE = "wind_electricity_twh"
START_YEAR, END_YEAR = 2019, 2023

# A failed selection must never leave an earlier result available for export.
for name in ("selected", "selection_context"):
    globals().pop(name, None)
if not globals().get("source_ready", False):
    raise RuntimeError("Complete the download cell successfully before selecting data")

def verify_retrieval(folder, record):
    import hashlib
    required = {"owid_energy.csv", "owid_energy.codebook.csv", "readme.md"}
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("Incomplete retrieval record; rerun the download")
    for name in sorted(required):
        path, entry = Path(folder) / name, record[name]
        if not path.is_file() or not isinstance(entry, dict):
            raise RuntimeError("Missing retrieval file or metadata; rerun the download")
        raw = path.read_bytes()
        if entry.get("bytes") != len(raw) or entry.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise RuntimeError("Retrieval file does not match its record; rerun the download")

verify_retrieval(raw_folder, download_record)
# Re-read the verified files so selection cannot use a changed in-memory table.
data = pd.read_csv(Path(raw_folder) / "owid_energy.csv")
codebook = pd.read_csv(Path(raw_folder) / "owid_energy.codebook.csv").fillna("")
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
unit = variable_metadata.get("unit")
if not isinstance(unit, str) or unit.strip().casefold() not in {"twh", "terawatt-hours (twh)"}:
    raise ValueError("Unit changed; inspect the source before converting")
if not isinstance(ENTITY_NAME, str):
    raise ValueError("ENTITY_NAME must be a string; leave it empty to use COUNTRY_CODE")
entity_name = ENTITY_NAME.strip()
selector = f"name:{entity_name}" if entity_name else COUNTRY_CODE
country_rows = data.loc[data.country.eq(entity_name) if entity_name else
                        (data.country.eq("World") if COUNTRY_CODE == "WORLD"
                         else data.iso_code.eq(COUNTRY_CODE))]
if country_rows.empty:
    raise ValueError("Unknown exact name or code: inspect available_entities.csv")
country_names = country_rows.country.unique()
if len(country_names) != 1:
    raise ValueError("Country code maps to multiple names; investigate")
country_name = country_names[0]
# Inspect all rows for this name, even when selection used an ISO code.
# Otherwise rows with missing or changed codes could be silently omitted.
entity_history = data.loc[data.country.eq(country_name)]
source_codes = entity_history.iso_code.dropna().unique()
if len(source_codes) > 1:
    raise ValueError("Selected entity maps to multiple source codes; inspect the dataset")
if len(source_codes) and entity_history.iso_code.isna().any():
    raise ValueError("Selected entity has both missing and present source codes; inspect the dataset")
source_code = source_codes[0] if len(source_codes) else None
annual = country_rows.loc[country_rows.year.between(START_YEAR, END_YEAR), ["year", VARIABLE]]
if annual.year.duplicated().any():
    raise ValueError("Duplicate country-year observations")
# Reindex explicitly: absent years stay visible as missing, never zero.
selected = annual.set_index("year").reindex(range(START_YEAR, END_YEAR + 1))
selected.index.name = "year"
selected = selected.rename(columns={VARIABLE: "generation_twh"}).reset_index()
selected.insert(0, "country", country_name)
selected.insert(1, "iso_code", source_code)
selected["generation_gwh"] = selected.generation_twh * 1000
values = selected.generation_twh.dropna()
if not values.map(math.isfinite).all() or values.lt(0).any():
    raise ValueError("Unexpected non-finite or negative generation; inspect the source")
print(country_name, selector)
print(variable_metadata["title"], "| Unit:", variable_metadata["unit"])
print("Definition:", variable_metadata["description"])
print("Original sources:", variable_metadata["source"])
print(selected.to_string(index=False))
import copy, hashlib
selection_digest = hashlib.sha256(selected.to_csv(index=False).encode("utf-8")).hexdigest()
selection_context = dict(selector=selector, country=country_name, iso_code=source_code,
    variable=VARIABLE, start_year=START_YEAR, end_year=END_YEAR,
    metadata=copy.deepcopy(variable_metadata), downloads=copy.deepcopy(download_record), raw_folder=str(raw_folder),
    selection_sha256=selection_digest)
print(f"Coverage: {len(values)}/{len(selected)} reported years; missing stays missing")
if values.empty:
    print("No observations: choose another period/product or report unavailability.")
```

## 4. Save your selection as CSV and SQLite

This creates a new versioned selection folder for every successful save. The SQLite table has one row per year for this selection; the actual source ISO code may be missing for an aggregate. Failed saves remain in a folder ending in `.incomplete` and never replace an earlier completed export. The selected variable is recorded in the folder name and metadata. Selection and saving check that all three retained source files match their retrieval sizes and hashes. Selection reloads the verified CSV and codebook so unsaved in-memory edits cannot be attributed to the original retrieval. Saving also checks that the settings, source record and selected rows have not changed since selection; rerun the download or selection as instructed if a check fails. Missing values become SQL `NULL`, so report `COUNT(generation_twh)` alongside the sum. [SQLite aggregates](https://www.sqlite.org/lang_aggfunc.html), [pandas SQL export](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html).

```{code-cell} python
if "selected" not in globals() or "selection_context" not in globals():
    raise RuntimeError("Run a successful download and selection before saving")
ctx = selection_context
if not isinstance(ENTITY_NAME, str):
    raise RuntimeError("Settings changed; rerun selection before saving")
current_selector = f"name:{ENTITY_NAME.strip()}" if ENTITY_NAME.strip() else COUNTRY_CODE
if (not globals().get("source_ready", False) or current_selector != ctx["selector"]
        or VARIABLE != ctx["variable"] or START_YEAR != ctx["start_year"] or END_YEAR != ctx["end_year"]
        or download_record != ctx["downloads"] or str(raw_folder) != ctx["raw_folder"]):
    raise RuntimeError("Settings or source changed; rerun selection before saving")
if hashlib.sha256(selected.to_csv(index=False).encode("utf-8")).hexdigest() != ctx["selection_sha256"]:
    raise RuntimeError("Selected data changed; rerun selection before saving")
verify_retrieval(raw_folder, download_record)
export_id = uuid.uuid4().hex
output_root = Path("outputs/owid_world")
# Encode the selector for a portable folder name (for example, ':' is invalid
# in Windows filenames). The original selector remains in provenance.json.
from urllib.parse import quote
selector_component = quote(ctx["selector"], safe="-._")
folder_name = f"{selector_component}_{ctx['variable']}_{ctx['start_year']}_{ctx['end_year']}_{export_id}"
output = output_root / (folder_name + ".incomplete")
output.mkdir(parents=True, exist_ok=False)
selected.to_csv(output / "generation.csv", index=False)
provenance = dict(provider="Our World in Data", **ctx,
    grain="one selected country or aggregate per calendar year",
    conversion="generation_gwh = generation_twh * 1000")
(output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
connection = sqlite3.connect(output / "generation.sqlite")
try:
    connection.execute("CREATE TABLE generation (country TEXT, iso_code TEXT, year INTEGER PRIMARY KEY NOT NULL, generation_twh REAL, generation_gwh REAL) WITHOUT ROWID")
    selected.to_sql("generation", connection, if_exists="append", index=False)
    count, sql_total = connection.execute(
        "SELECT COUNT(generation_twh), SUM(generation_twh) FROM generation").fetchone()
finally:
    connection.close()
python_total = selected.generation_twh.sum(min_count=1)
if count != selected.generation_twh.notna().sum():
    raise ValueError("SQL and selected data coverage disagree; export remains incomplete")
if count:
    if sql_total is None or not math.isclose(sql_total, python_total, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("SQL and selected data totals disagree; export remains incomplete")
    print(f"Observed-period sum: {sql_total:.3f} TWh across {count} reported years")
else:
    if sql_total is not None or not pd.isna(python_total):
        raise ValueError("SQL and selected data missing totals disagree; export remains incomplete")
    print("No observed total; SQL NULL and pandas missing agree")
completed_output = output_root / folder_name
output.rename(completed_output)
output = completed_output
print("CSV:", (output / "generation.csv").resolve())
print("Metadata:", (output / "provenance.json").resolve())
print("SQLite:", (output / "generation.sqlite").resolve())
```

**Get the files onto your computer:** in the sandbox, open **Files, input and export**, paste each printed path into **Download a Python-created file (path)** and click **Download file**. In Colab, refresh the Files pane and download each file from its menu. In local Jupyter, open the printed folder in the file browser. Also retain the raw CSV, codebook and README from the completed raw folder printed above. Browser/Colab runtime files are temporary until exported. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## 5. Exercises with increasing difficulty

1. **Easy — Austria:** run all three cells unchanged. Download your CSV and provenance, reopen the CSV, and identify country, variable, unit and reported-year count. Explain why five annual values cannot be labelled monthly data.
2. **Medium — Europe:** choose any available European country using the entity lookup, clear `ENTITY_NAME`, change `COUNTRY_CODE`, and rerun selection/save. Compare the same years and variable with Austria. Keep both output folders; verify coverage before comparing sums.
3. **Medium — Worldwide:** choose Brazil, India, Japan or another available country outside Europe. Download both wind and solar by changing `VARIABLE` and rerunning selection/save. Join the two CSVs on country code and year with `validate="one_to_one"`. Retain missing values.
4. **Hard — Country versus world:** select `WORLD` for the same variable/years. For each year with both values reported and a positive world denominator, calculate `100 * country_generation / world_generation`. Explain why this is a share of world generation for that source, not a country's renewable share. Never add the world row to the country total.
5. **Hard — Compare providers:** use Austria, wind, 2023 in both lessons. Convert OWID TWh to GWh, check all twelve Eurostat months, and compare boundaries, coverage, source versions and revisions. Do not force agreement: an annual series from another provider is not automatically identical to Eurostat's monthly net-generation series.

These exercises are original course tasks using the cited provider definitions. Annual data cannot be converted into actual monthly observations by dividing by twelve. If a chosen country/year lacks data, report the gap and select another period; do not fabricate values.

[Download this lesson as a Jupyter notebook](world-energy-download.ipynb).

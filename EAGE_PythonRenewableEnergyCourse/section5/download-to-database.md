# Download a real dataset and build a SQLite database

**Outcome:** choose Austria or another reporting European country, download monthly electricity generation, preserve metadata, check coverage, save CSV, load SQLite and reconcile a SQL total. Austria is the starting example, not a restriction. For other countries worldwide, use the [global OWID download exercise](world-energy-download.md). This is guided practice, not a final-test answer.

## 1. Identify the dataset and its meaning

Use Eurostat **nrg_cb_pem**, *Net electricity generation by type of fuel — monthly data*. The selected series is Austria (`AT`), wind (`RA300`), monthly (`M`), in GWh (`GWH`), January–December 2023. These are country-level electricity-generation observations, not wind speeds, installed capacity or consumption. Codes and units are returned in the response metadata. [Dataset](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem/default/table?lang=en), [energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology)

| Request parameter | Value | Meaning |
|---|---|---|
| `geo` | `AT` | Austria |
| `siec` | `RA300` | Wind energy product |
| `freq` | `M` | Monthly |
| `unit` | `GWH` | Gigawatt-hours |
| `sinceTimePeriod` | `2023-01` | First requested month, included |
| `untilTimePeriod` | `2023-12` | Last requested month, included |

The Statistics API returns **JSON-stat**, a multidimensional format, not CSV. Its `id`, `size` and dimension indices specify where each observation belongs; `value` may be sparse. An absent value remains missing. [Eurostat API guide](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started), [JSON-stat specification](https://json-stat.org/format/)

### Choose your European country

1. Open the dataset and expand its **Geopolitical entity (reporting)** selection. Choose your country and note its Eurostat code; available reporting entities can change.
2. In the working Python example, keep `USE_LIVE = True` and change `GEO` from `"AT"` to that code. Change `YEAR` and `SOURCE` only if needed.
3. Run the complete example. Verify the printed country **name and code**, source label, units and monthly coverage before using the CSV.
4. Download the files from the printed output folder. Country/source/year are included in its name, so a second country's download has a separate folder.

| Country | `GEO` | Country | `GEO` |
|---|---|---|---|
| Austria | `AT` | Germany | `DE` |
| France | `FR` | Italy | `IT` |
| Spain | `ES` | Portugal | `PT` |
| Netherlands | `NL` | Poland | `PL` |
| Sweden | `SE` | Finland | `FI` |
| Norway | `NO` | Switzerland | `CH` |
| Greece | `EL` | United Kingdom | `UK` |

These codes were checked against the dataset's returned geography labels. Eurostat uses `EL` and `UK`; do not substitute ISO codes `GR` or `GB`. The country selector includes additional reporting countries. A listed entity does **not** guarantee observations for every product/year. If your country or period is unavailable, use the global lesson or report the gap; do not substitute Austria's data or convert missing values to zero. [Eurostat dataset and country selection](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem/default/table?lang=en).

Wind is `RA300`; photovoltaic generation is `RA420`; `RA410` is solar thermal. The default source and year are a teaching choice. Offline mode contains only Austria/wind/2023 and deliberately rejects other selections. [Eurostat energy methodology](https://ec.europa.eu/eurostat/web/energy/methodology).

## 2. Download through the website

1. Open the dataset link above and select the filters in the table, replacing Austria with your chosen country (language is English).
2. Check the visible title, country, source, units and dates before downloading.
3. Open **Download**, choose CSV/TSV for machine processing or spreadsheet for inspection. Choose **All selected dimensions**; a displayed-only export may omit selections hidden from the current table.
4. Retain flags and metadata. Save the original download in a `data/raw` folder and record the URL and download date separately.
5. Inspect the file: an HTML sign-in/error page renamed `.csv` is not data. Do not feed a new spreadsheet layout to a parser written for the course's historical 2024 spreadsheet headers.

These steps follow the [Eurostat download guide](https://ec.europa.eu/eurostat/web/user-guides/data-browser/download-data/download-datasets). The working example below uses the API to create a predictable CSV schema, independent of the Data Browser spreadsheet layout.

## 3. Run the working example

- **Sandbox:** choose **Edit this code**, then **Run Python**. Internet access is needed for live mode; the Eurostat API supports browser CORS.
- **Jupyter/Colab:** install pandas if needed, then run the code cells in order. Top-level `await` is supported by these notebook environments.
- **Offline practice:** set `USE_LIVE = False`. Use the bundled snapshot; the output records that it is a snapshot, never a new live retrieval.
- **Terminal alternative:** from the repository root run `python tools/download_energy_data.py`. This saves the raw response, CSV and provenance under `outputs/` without replacing the bundled teaching snapshot. The command uses only the standard library.

Sources for the interfaces: [Eurostat CORS/API](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started), [Pyodide HTTP](https://pyodide.org/en/stable/usage/api/python-api/http.html), [IPython autoawait](https://ipython.readthedocs.io/en/stable/interactive/autoawait.html), [urllib.request](https://docs.python.org/3/library/urllib.request.html).

### What each part of the code does

1. Names the source and filters; constructs a URL with `urlencode`.
2. Downloads bytes (or explicitly reads a saved snapshot); HTTP/JSON errors stop the run.
3. Decodes dimension positions and retains statistical flags. It does not rely on dictionary insertion order or assume missing means zero.
4. Checks grain, twelve months, source, country, frequency and unit before calculating.
5. Saves raw JSON, a normalized CSV and a provenance record with a SHA-256 checksum.
6. Creates only this tutorial's SQLite tables, with a primary key and foreign key.
7. Compares SQL and pandas totals and reports missing-value coverage.

This is an original implementation of the documented formats and APIs: [JSON-stat](https://json-stat.org/format/), [pandas CSV](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html), [SQLite Python interface](https://docs.python.org/3/library/sqlite3.html), [hashlib](https://docs.python.org/3/library/hashlib.html).

```python
# Step 1: Explicit request parameters and output folder.
import sys, json, math, hashlib, itertools, sqlite3
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
import pandas as pd

USE_LIVE = True
GEO = "AT"  # change to DE, FR, IT, ES, NO, EL, UK, etc.; check the dataset selector
YEAR = 2023
SOURCE = "RA300"  # wind; RA420 = photovoltaic generation
params = dict(lang="en", freq="M", geo=GEO, siec=SOURCE, unit="GWH",
              sinceTimePeriod=f"{YEAR}-01", untilTimePeriod=f"{YEAR}-12")
url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_cb_pem?" + urlencode(params)
output = Path("outputs") / f"eurostat_{GEO}_{SOURCE}_{YEAR}"
output.mkdir(parents=True, exist_ok=True)

# Step 2: Read bytes, then parse JSON. Never silently replace a failed live request.
if USE_LIVE:
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch
        response = await pyfetch(url)
        if response.status != 200:
            raise RuntimeError(f"Eurostat HTTP {response.status}; retry or explicitly select offline mode")
        raw = await response.bytes()
    else:
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as response:
            raw = response.read()
    retrieved = datetime.now(timezone.utc).isoformat()
else:
    # Offline fixture is only the named Austria/wind/2023 series.
    if (GEO, YEAR, SOURCE) != ("AT", 2023, "RA300"):
        raise ValueError("Offline snapshot is AT / RA300 / 2023; use live mode for other selections")
    candidates = [Path.cwd(), Path.cwd()/"EAGE_PythonRenewableEnergyCourse", *Path.cwd().parents]
    snapshot = next((p/"data/examples/eurostat_at_wind_2023.json" for p in candidates
                     if (p/"data/examples/eurostat_at_wind_2023.json").exists()), None)
    if snapshot is None:
        raise FileNotFoundError("Place the supplied snapshot under data/examples, or use live mode")
    raw = snapshot.read_bytes()
    retrieved = json.loads(snapshot.with_suffix(".provenance.json").read_text())["retrieved_utc"]
payload = json.loads(raw)

# Step 3: Decode JSON-stat dimension indices and sparse values.
def decode_series(payload):
    """Decode JSON-stat positional values without assuming time is the last axis."""
    if payload.get("class") != "dataset" or "error" in payload:
        raise ValueError("Expected a Eurostat JSON-stat dataset, not an error/page")
    ids, sizes = payload["id"], payload["size"]
    required = {"freq", "siec", "unit", "geo", "time"}
    if len(ids) != len(required) or len(ids) != len(sizes) or set(ids) != required:
        raise ValueError(f"Unexpected dimensions: {ids}")
    axes = []
    for name, size in zip(ids, sizes):
        index = payload["dimension"][name]["category"]["index"]
        if isinstance(index, dict) and sorted(index.values()) != list(range(size)):
            raise ValueError(f"Invalid category positions: {name}")
        labels = index if isinstance(index, list) else [k for k, _ in sorted(index.items(), key=lambda x: x[1])]
        if len(labels) != size:
            raise ValueError(f"Dimension length mismatch: {name}")
        axes.append(labels)
    def get_at(container, i, default=None):
        return container.get(str(i), default) if isinstance(container, dict) else (container[i] if i < len(container) else default)
    rows = []
    for i, labels in enumerate(itertools.product(*axes)):
        row = dict(zip(ids, labels))
        value = get_at(payload.get("value", {}), i)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Non-finite or non-numeric generation")
        rows.append({"geo": row["geo"], "source": row["siec"], "month": row["time"],
            "unit": row["unit"], "freq": row["freq"], "generation_gwh": value,
            "status_flag": get_at(payload.get("status", {}), i, "") or ""})
    if any(r["unit"] != "GWH" or r["freq"] != "M" for r in rows):
        raise ValueError("Expected monthly GWh")
    keys = [(r["geo"], r["source"], r["month"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate country/source/month")
    return sorted(rows, key=lambda r: (r["geo"], r["source"], r["month"]))

frame = pd.DataFrame(decode_series(payload))
if frame.empty:
    raise ValueError("No rows returned for this selection. Check country/product/year or use the global lesson.")

# Step 4: Validate the requested slice and coverage.
assert len(frame) == 12
assert set(frame.month) == {f"{YEAR}-{m:02}" for m in range(1, 13)}
assert frame.geo.eq(GEO).all() and frame.source.eq(SOURCE).all()
assert frame.unit.eq("GWH").all() and frame.freq.eq("M").all()
assert not frame.duplicated(["geo", "source", "month"]).any()
observed = frame.generation_gwh.notna().sum()
print(payload["label"])
print(f"Country: {payload['dimension']['geo']['category']['label'][GEO]} ({GEO})")
print(payload["dimension"]["siec"]["category"]["label"][SOURCE])
print(frame.to_string(index=False))
print(f"Coverage: {observed}/12 monthly observations; missing values stay missing")

# Step 5: Persist both the provider response and the normalized data.
(output/"raw.json").write_bytes(raw)
frame.to_csv(output/"generation.csv", index=False)
provenance = dict(provider="Eurostat", dataset="nrg_cb_pem", url=url,
    mode="live" if USE_LIVE else "bundled snapshot", retrieved_utc=retrieved,
    provider_updated=payload.get("updated"), sha256=hashlib.sha256(raw).hexdigest(),
    unit="GWH", grain="country-source-month", status_labels=payload.get("extension", {}).get("status", {}))
(output/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

# Step 6: Create this tutorial's own database and enforce its keys.
connection = sqlite3.connect(output/"generation.sqlite")
try:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
    DROP TABLE IF EXISTS generation;
    DROP TABLE IF EXISTS energy_source;
    CREATE TABLE energy_source(source TEXT PRIMARY KEY NOT NULL, label TEXT NOT NULL);
    CREATE TABLE generation(
        geo TEXT NOT NULL, source TEXT NOT NULL REFERENCES energy_source(source),
        month TEXT NOT NULL, generation_gwh REAL, status_flag TEXT,
        PRIMARY KEY(geo, source, month));
    """)
    label = payload["dimension"]["siec"]["category"]["label"][SOURCE]
    connection.execute("INSERT INTO energy_source VALUES (?, ?)", (SOURCE, label))
    rows = [(r.geo, r.source, r.month,
             None if pd.isna(r.generation_gwh) else float(r.generation_gwh), r.status_flag)
            for r in frame.itertuples(index=False)]
    connection.executemany("INSERT INTO generation VALUES (?, ?, ?, ?, ?)", rows)
    connection.commit()

    # Step 7: Reconcile independent aggregations and integrity checks.
    total, count = connection.execute(
        "SELECT SUM(generation_gwh), COUNT(generation_gwh) FROM generation WHERE geo=? AND source=?",
        (GEO, SOURCE)).fetchone()
    expected = frame.generation_gwh.sum(min_count=1)
    assert count == observed
    assert (total is None and pd.isna(expected)) or math.isclose(total, expected, rel_tol=1e-12)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    print("SQL/pandas observed total:", total, "GWh; reported months:", count)
finally:
    connection.close()
print("Saved files in:", output.resolve())
print("CSV path:", (output/"generation.csv").resolve())

```

## 4. Read the saved CSV again

The normalized CSV columns are `geo, source, month, unit, freq, generation_gwh, status_flag`. The bundled checked snapshot contains twelve reported months and totals **7971.360 GWh**. Live provider revisions may change values; validate the current metadata and reconcile totals rather than hard-coding that total as a permanent truth. January's bundled value is 836.110 GWh. The original bytes and retrieval record are linked below.

```python
# Run after the working example; or replace with your downloaded/uploaded CSV path.
csv_path = output / "generation.csv"
check = pd.read_csv(csv_path, dtype={"geo": str, "source": str, "month": str},
                    keep_default_na=False, na_values={"generation_gwh": [""]})
assert len(check) == 12
assert check.unit.eq("GWH").all()
assert not check.duplicated(["geo", "source", "month"]).any()
print(check.head(3).to_string(index=False))
print("CSV observed sum:", check.generation_gwh.sum(min_count=1), "GWh")
```

`keep_default_na=False` avoids interpreting a blank status flag as a measurement value; the explicit missing-value rule applies only to generation. [pandas read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

## 5. Get your result onto your computer

- **Sandbox:** open the workspace tools, paste the printed absolute CSV path into **Download a Python-created file (path)**, then click **Download file**. Repeat for `raw.json`, `provenance.json` and `generation.sqlite` in the same folder. These are virtual browser files until downloaded; Stop/reset discards them.
- **Colab:** refresh Files, open the printed output folder, use each file's menu to download it. Copy the notebook to Drive separately.
- **Local Jupyter:** use the file browser or open the printed path in your operating system.

These instructions describe the course UI and the file browsers; Colab's runtime-file lifetime is documented in its [FAQ](https://research.google.com/colaboratory/faq.html).

Offline inputs: [CSV snapshot](../data/examples/eurostat_at_wind_2023.csv), [raw JSON](../data/examples/eurostat_at_wind_2023.json), [provenance/checksum](../data/examples/eurostat_at_wind_2023.provenance.json).

## 6. Troubleshooting

| Symptom | Next action |
|---|---|
| HTTP failure or browser/network restriction | Keep the error; retry later or explicitly choose the supplied snapshot |
| JSON parsing error | Check that the response is data, not an HTML error/download page |
| `FileNotFoundError` | Print `Path.cwd()`; check exact name, folder and `.csv` extension |
| Fewer than twelve reported months | Report observed coverage; do not label the sum a complete annual total |
| Unexpected dimensions or units | Recheck filters; do not bypass the checks to force the output |
| Same primary key twice | Investigate duplicates or revision handling; do not silently add both |
| Notebook moved to another environment | Transfer input files and rerun setup; kernel state does not travel with a CSV |

## 7. Exercises

### Exercise 1 — Easy: reproduce and identify

Run the exact request above. List the dataset code, country, source, frequency, unit, period, twelve-row count and retrieval date. Download your CSV and reopen it. State whether you used live data or the bundled snapshot.

### Exercise 2 — Medium: choose another country

Choose **any reporting European country** from the dataset selector (for example Germany `DE`, Spain `ES` or Norway `NO`). Set `USE_LIVE=True` and change only `GEO`, keeping source/year/unit fixed. Rerun the complete example; it creates a different output folder. Download its CSV and provenance file. Check the returned country label, month coverage, flags and CSV. If observations are missing, report the partial sum and count. Do not reuse Austria's snapshot or expected total. Compare with Austria only after checking both selections have comparable coverage.

### Exercise 3 — Medium: compare two sources without double-counting

Keep your selected `GEO` from Exercise 2 and use live mode. Download wind with `SOURCE="RA300"`, then photovoltaic generation with `SOURCE="RA420"`. (`RA410` means solar thermal, a different technology.) Check the source label in the provider response. Download separately, validate both files, then join by country and month with a one-to-one check. Explain why a wind-plus-PV sum is not total renewable generation. [Eurostat energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology)

### Exercise 4 — Challenge: missing data and revisions

Work on a copy of the downloaded CSV. Remove one generation value but retain the row and flag; verify that your SQL count drops while the missing value becomes `NULL`. Attempt to insert a duplicate key and explain the rejection. Design a retrieval-date/version column before storing a second provider revision. Keep the original download unchanged. [SQLite aggregates](https://www.sqlite.org/lang_aggfunc.html), [constraints](https://www.sqlite.org/lang_createtable.html)

[Download this tutorial as a Jupyter notebook](download-to-database.ipynb).

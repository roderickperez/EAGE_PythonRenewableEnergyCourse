"""Deterministically rebuild the two reader-facing course notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def time_series_notebook():
    cells = [
        md(r"""
# Monthly European Electricity Generation

## Goal

Build a reproducible monthly time series from the bundled Eurostat exports, validate its units and coverage, visualise a selected-category generation series, decompose its seasonality, test stationarity, and evaluate a chronological forecast baseline.

**Source definition.** The workbooks contain **net electricity generation by fuel, monthly, in gigawatt-hours (GWh)**. They are not energy-consumption data and they are not measured in terajoules. The export was updated on 22 August 2024 and covers January 2016 through July 2024; the final year is partial.

Source: [Eurostat energy database](https://ec.europa.eu/eurostat/web/energy/database). The bundled files make this notebook reproducible but should be refreshed before using the results as current statistics.

**Accounting limits:** This tutorial selects combustible renewables, geothermal, broad hydro, other renewables, solar and wind. The broad Hydro export does not separate pumped-storage output; the selected sum must not be called a verified renewable-only total. Eurostat distinguishes storage generation from renewable primary production. Non-renewable combustible fuels can include non-renewable waste, so the aggregate is not labelled simply fossil. Reporting-country coverage changes, statistical flags are omitted by the beginner parser, and some values are missing. This is not an EU-27 official total. For a renewable-only analysis obtain the required hydro breakdown and consistent country coverage first. [Eurostat electricity statistics](https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/9990.pdf).
"""),
        md("## Setup\n\nThe notebook uses pandas, NumPy, Matplotlib, and statsmodels. Paths are resolved whether execution starts in the course root or beside the notebook."),
        code(r"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 12)

def find_course_root(start=Path.cwd()):
    for candidate in [start / "EAGE_PythonRenewableEnergyCourse", start, *start.parents]:
        if (candidate / "myst.yml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the course repository.")

COURSE_ROOT = find_course_root()
DATA_DIR = COURSE_ROOT / "data" / "section4" / "euroStat"
DATA_DIR
"""),
        md("## Data\n\nThe source map deliberately distinguishes aggregate categories from their components. `nonrenewable_combustible_total` is the Eurostat aggregate for non-renewable combustible fuels; it must not be added to coal, gas, and oil."),
        code(r"""
SOURCE_FILES = {
    "coal": "nrg_cb_pem_page_spreadsheet_coal.xlsx",
    "combustible_renewable": "nrg_cb_pem_page_spreadsheet_combustionFuels_Renewables.xlsx",
    "nonrenewable_combustible_total": "nrg_cb_pem_page_spreadsheet_combustionFuels_nonRenewables.xlsx",
    "geothermal": "nrg_cb_pem_page_spreadsheet_geothermal.xlsx",
    "hydro": "nrg_cb_pem_page_spreadsheet_hydro.xlsx",
    "natural_gas": "nrg_cb_pem_page_spreadsheet_naturalGas.xlsx",
    "nuclear": "nrg_cb_pem_page_spreadsheet_nuclear.xlsx",
    "oil": "nrg_cb_pem_page_spreadsheet_oil.xlsx",
    "other_renewable": "nrg_cb_pem_page_spreadsheet_otherRenewables.xlsx",
    "solar": "nrg_cb_pem_page_spreadsheet_solar.xlsx",
    "wind": "nrg_cb_pem_page_spreadsheet_wind.xlsx",
}

SELECTED_SOURCES = [
    "combustible_renewable", "geothermal", "hydro",
    "other_renewable", "solar", "wind",
]

def load_eurostat_export(path: Path, source: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    data_sheet = workbook.sheet_names[-1]
    wide = pd.read_excel(path, sheet_name=data_sheet, skiprows=8)
    # Adjacent Unnamed columns contain statistical flags. This beginner parser
    # omits those flags; retain them for operational analysis.
    wide = wide.loc[:, ~wide.columns.astype(str).str.startswith("Unnamed")]
    wide = wide.rename(columns={wide.columns[0]: "country"})
    wide = wide.iloc[1:].copy()  # remove GEO label row
    wide = wide.loc[:wide["country"].isna().idxmax() - 1] if wide["country"].isna().any() else wide
    wide = wide.dropna(subset=["country"])
    wide = wide[~wide["country"].astype(str).str.startswith("European Union")]
    long = wide.melt(id_vars="country", var_name="month", value_name="generation_gwh")
    long["month"] = pd.to_datetime(long["month"], format="%Y-%m")
    long["generation_gwh"] = pd.to_numeric(
        long["generation_gwh"].replace({":": np.nan}), errors="coerce"
    )
    long["source"] = source
    return long[["country", "month", "source", "generation_gwh"]]

generation = pd.concat(
    [load_eurostat_export(DATA_DIR / filename, source)
     for source, filename in SOURCE_FILES.items()],
    ignore_index=True,
)
generation.head()
"""),
        md("### Checks\n\nThe intended grain is one country–month–source row. Missing values remain missing; they are never silently replaced by zero."),
        code(r"""
key = ["country", "month", "source"]
assert not generation.duplicated(key).any()
assert generation["month"].min() == pd.Timestamp("2016-01-01")
assert generation["month"].max() == pd.Timestamp("2024-07-01")
negative_rows = generation["generation_gwh"].lt(0).sum()

quality = pd.Series({
    "rows": len(generation),
    "countries": generation["country"].nunique(),
    "months": generation["month"].nunique(),
    "sources": generation["source"].nunique(),
    "missing_share": generation["generation_gwh"].isna().mean(),
    "negative_rows_to_review": negative_rows,
})
quality
"""),
        md("## Results\n\nFor a regional total, sum reported countries and retain a coverage count. A lower total can otherwise mean either lower generation or fewer reporting countries."),
        code(r"""
monthly_source = (
    generation.groupby(["month", "source"], as_index=False)
    .agg(
        generation_gwh=("generation_gwh", lambda values: values.sum(min_count=1)),
        reporting_countries=("generation_gwh", "count"),
    )
)

selected_monthly = (
    monthly_source[monthly_source["source"].isin(SELECTED_SOURCES)]
    .groupby("month")
    .agg(
        selected_generation_gwh=("generation_gwh", lambda values: values.sum(min_count=len(SELECTED_SOURCES))),
        source_categories_reported=("generation_gwh", "count"),
    )
)
selected_monthly.tail(8)
"""),
        code(r"""
coverage = monthly_source[monthly_source["source"].isin(SELECTED_SOURCES)].pivot(
    index="month", columns="source", values="reporting_countries")
ax = coverage.plot(figsize=(11, 5))
ax.set(title="Reporting coverage changes: do not interpret raw sums as trends",
       xlabel="Month", ylabel="Countries reporting a numeric value")
ax.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
"""),
        md("### Seasonality and trend\n\nRegional reporting coverage changes, so we use a single country and source: Austrian wind generation. We trim missing endpoints and reject internal gaps instead of filling them. The series is monthly, so one annual seasonal cycle has period 12. `seasonal_decompose` requires at least two complete cycles. The final 2024 year is partial and must not be compared with complete annual totals."),
        code(r"""
country_series = generation.loc[
    generation["country"].eq("Austria") & generation["source"].eq("wind")
].set_index("month")["generation_gwh"].sort_index()
series = country_series.loc[country_series.first_valid_index():country_series.last_valid_index()].asfreq("MS")
print(f"Austria wind: {series.index.min():%Y-%m} to {series.index.max():%Y-%m}; {len(series)} months")
assert len(series) >= 24 and series.notna().all()

decomposition = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")
fig = decomposition.plot()
fig.set_size_inches(11, 8)
fig.suptitle("Austria wind generation: additive decomposition", y=1.01)
plt.tight_layout()
"""),
        md("### Stationarity\n\nThe Augmented Dickey–Fuller test evaluates a unit-root null hypothesis. It is not a seasonality test. A small p-value is evidence against a unit root under the selected deterministic terms and lag rule; it does not prove that every form of non-stationarity is absent."),
        code(r"""
adf_stat, p_value, used_lag, n_obs, critical, _ = adfuller(series, autolag="AIC")
pd.Series({
    "ADF statistic": adf_stat,
    "p-value": p_value,
    "lags used": used_lag,
    "observations": n_obs,
    **{f"critical value {level}": value for level, value in critical.items()},
})
"""),
        md("### Chronological forecast baseline\n\nThe last 12 complete observations form the test window. A seasonal-naïve forecast uses the corresponding month one year earlier. This baseline is simple, auditable, and contains no future leakage."),
        code(r"""
test_horizon = 12
train = series.iloc[:-test_horizon]
test = series.iloc[-test_horizon:]
seasonal_naive = series.shift(12).reindex(test.index)

mae = (test - seasonal_naive).abs().mean()
rmse = np.sqrt(((test - seasonal_naive) ** 2).mean())

ax = series.iloc[-36:].plot(figsize=(11, 4), label="Observed", color="black")
seasonal_naive.plot(ax=ax, label="Seasonal-naïve forecast", color="tab:orange")
ax.axvline(test.index.min(), color="tab:red", linestyle="--", label="Test starts")
ax.set(title=f"Austria wind baseline — MAE {mae:,.0f} GWh; RMSE {rmse:,.0f} GWh",
       xlabel="Month", ylabel="Generation (GWh)")
ax.legend()
plt.tight_layout()
pd.Series({"MAE_GWh": mae, "RMSE_GWh": rmse})
"""),
        md("## Takeaways\n\n- Preserve the source definition: generation, not consumption; GWh, not TJ.\n- Never add an aggregate fuel category to its components.\n- Retain missing values and report coverage alongside regional totals.\n- Use a 12-month seasonal period for monthly data.\n- Evaluate forecasts on future time windows and compare them with a seasonal-naïve baseline.\n- Refresh the Eurostat export before making current-policy or market claims."),
    ]
    return notebook(cells)


def sql_notebook():
    cells = [
        md("""
# Eurostat Electricity Generation with pandas and SQLite

## Goal

Transform the bundled Eurostat Excel exports into a tidy SQLite database, query it without double counting, and validate the database against the source rows.

The source measure is monthly **net electricity generation in GWh**. The database keeps source categories separate and records missing values as SQL `NULL`, not zero.

**Accounting limits:** This tutorial selects combustible renewables, geothermal, broad hydro, other renewables, solar and wind. The broad Hydro export does not separate pumped-storage output; the selected sum must not be called a verified renewable-only total. Eurostat distinguishes storage generation from renewable primary production. Non-renewable combustible fuels can include non-renewable waste, so the aggregate is not labelled simply fossil. Reporting-country coverage changes, statistical flags are omitted by the beginner parser, and some values are missing. This is not an EU-27 official total. For a renewable-only analysis obtain the required hydro breakdown and consistent country coverage first. [Eurostat electricity statistics](https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/9990.pdf).
"""),
        md("## Setup"),
        code(r"""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

def find_course_root(start=Path.cwd()):
    for candidate in [start / "EAGE_PythonRenewableEnergyCourse", start, *start.parents]:
        if (candidate / "myst.yml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the course repository.")

COURSE_ROOT = find_course_root()
DATA_DIR = COURSE_ROOT / "data" / "section4" / "euroStat"
DB_PATH = COURSE_ROOT / "section5" / "energy_generation.db"
"""),
        md("## Data model\n\n`generation` has one row per country, month, and source. `source` describes the taxonomy and whether a row is an aggregate or component. This prevents accidental addition of `nonrenewable_combustible_total` to coal, natural gas, and oil."),
        code(r"""
SOURCE_FILES = {
    "coal": ("nrg_cb_pem_page_spreadsheet_coal.xlsx", "fossil_component", 0),
    "combustible_renewable": ("nrg_cb_pem_page_spreadsheet_combustionFuels_Renewables.xlsx", "renewable", 1),
    "nonrenewable_combustible_total": ("nrg_cb_pem_page_spreadsheet_combustionFuels_nonRenewables.xlsx", "nonrenewable_combustible_aggregate", 0),
    "geothermal": ("nrg_cb_pem_page_spreadsheet_geothermal.xlsx", "renewable", 1),
    "hydro": ("nrg_cb_pem_page_spreadsheet_hydro.xlsx", "hydro_including_storage", 1),
    "natural_gas": ("nrg_cb_pem_page_spreadsheet_naturalGas.xlsx", "fossil_component", 0),
    "nuclear": ("nrg_cb_pem_page_spreadsheet_nuclear.xlsx", "nuclear", 0),
    "oil": ("nrg_cb_pem_page_spreadsheet_oil.xlsx", "fossil_component", 0),
    "other_renewable": ("nrg_cb_pem_page_spreadsheet_otherRenewables.xlsx", "renewable", 1),
    "solar": ("nrg_cb_pem_page_spreadsheet_solar.xlsx", "renewable", 1),
    "wind": ("nrg_cb_pem_page_spreadsheet_wind.xlsx", "renewable", 1),
}

def load_export(path, source):
    workbook = pd.ExcelFile(path)
    wide = pd.read_excel(path, sheet_name=workbook.sheet_names[-1], skiprows=8)
    # Adjacent Unnamed columns contain statistical flags. This beginner parser
    # omits those flags; retain them for operational analysis.
    wide = wide.loc[:, ~wide.columns.astype(str).str.startswith("Unnamed")]
    wide = wide.rename(columns={wide.columns[0]: "country"}).iloc[1:].copy()
    wide = wide.loc[:wide["country"].isna().idxmax() - 1] if wide["country"].isna().any() else wide
    wide = wide.dropna(subset=["country"])
    wide = wide[~wide["country"].astype(str).str.startswith("European Union")]
    long = wide.melt(id_vars="country", var_name="month", value_name="generation_gwh")
    long["month"] = pd.to_datetime(long["month"], format="%Y-%m").dt.strftime("%Y-%m-01")
    long["generation_gwh"] = pd.to_numeric(long["generation_gwh"].replace(":", np.nan), errors="coerce")
    long["source"] = source
    return long[["country", "month", "source", "generation_gwh"]]

generation = pd.concat(
    [load_export(DATA_DIR / filename, source)
     for source, (filename, _, _) in SOURCE_FILES.items()], ignore_index=True
)
sources = pd.DataFrame([
    {"source": source, "source_group": group, "in_selected_group": selected}
    for source, (_, group, selected) in SOURCE_FILES.items()
])
"""),
        md("### Validate before writing"),
        code(r"""
assert not generation.duplicated(["country", "month", "source"]).any()
assert set(generation["source"]) == set(sources["source"])
negative_rows = generation["generation_gwh"].lt(0).sum()
assert generation["month"].min() == "2016-01-01"
assert generation["month"].max() == "2024-07-01"

pd.DataFrame({
    "rows": [len(generation)],
    "countries": [generation.country.nunique()],
    "months": [generation.month.nunique()],
    "sources": [generation.source.nunique()],
    "missing_values": [generation.generation_gwh.isna().sum()],
    "negative_rows_to_review": [negative_rows],
})
"""),
        md("## Create the SQLite database\n\nThe notebook replaces only its own deterministic database file. The source spreadsheets remain unchanged."),
        code(r'''
with sqlite3.connect(DB_PATH) as connection:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        DROP TABLE IF EXISTS generation;
        DROP TABLE IF EXISTS source;
        CREATE TABLE source (
            source TEXT PRIMARY KEY,
            source_group TEXT NOT NULL,
            in_selected_group INTEGER NOT NULL CHECK (in_selected_group IN (0, 1))
        );
        CREATE TABLE generation (
            country TEXT NOT NULL,
            month TEXT NOT NULL,
            source TEXT NOT NULL REFERENCES source(source),
            generation_gwh REAL,
            PRIMARY KEY (country, month, source)
        );
        CREATE INDEX generation_month_idx ON generation(month);
    """)
    sources.to_sql("source", connection, if_exists="append", index=False)
    generation.to_sql("generation", connection, if_exists="append", index=False)

DB_PATH
'''),
        md("## Queries\n\nThe query selects six source categories for teaching; broad hydro includes storage generation, so this is not an official renewable-only total. The non-renewable combustible total uses only the Eurostat `nonrenewable_combustible_total` aggregate. Nuclear remains separate."),
        code(r'''
selected_query = """
SELECT g.month,
       SUM(g.generation_gwh) AS selected_generation_gwh,
       COUNT(g.generation_gwh) AS reported_country_source_rows
FROM generation AS g
JOIN source AS s USING (source)
WHERE s.in_selected_group = 1
GROUP BY g.month
ORDER BY g.month;
"""

mix_query = """
SELECT g.month,
       SUM(CASE WHEN s.in_selected_group = 1 THEN g.generation_gwh END) AS selected_gwh,
       SUM(CASE WHEN g.source = 'nonrenewable_combustible_total' THEN g.generation_gwh END) AS nonrenewable_combustible_gwh,
       SUM(CASE WHEN g.source = 'nuclear' THEN g.generation_gwh END) AS nuclear_gwh
FROM generation AS g
JOIN source AS s USING (source)
GROUP BY g.month
ORDER BY g.month;
"""

with sqlite3.connect(DB_PATH) as connection:
    selected_monthly = pd.read_sql_query(selected_query, connection, parse_dates=["month"])
    electricity_mix = pd.read_sql_query(mix_query, connection, parse_dates=["month"])

electricity_mix.tail()
'''),
        md("### Database reconciliation\n\nA direct pandas aggregation must match the SQL result. This check catches missing joins, duplicate rows, and taxonomy errors."),
        code(r'''
selected_sources = sources.loc[sources.in_selected_group.eq(1), "source"]
expected = (
    generation[generation.source.isin(selected_sources)]
    .groupby("month")["generation_gwh"].sum(min_count=1)
)
actual = selected_monthly.set_index(selected_monthly.month.dt.strftime("%Y-%m-%d"))["selected_generation_gwh"]
pd.testing.assert_series_equal(expected, actual, check_names=False)

with sqlite3.connect(DB_PATH) as connection:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    key_duplicates = connection.execute("""
        SELECT COUNT(*) FROM (
          SELECT country, month, source, COUNT(*) AS n
          FROM generation GROUP BY country, month, source HAVING n > 1
        )
    """).fetchone()[0]

assert integrity == "ok" and key_duplicates == 0
{"integrity_check": integrity, "duplicate_keys": key_duplicates}
'''),
        md("## Takeaways\n\n- SQL execution success does not guarantee semantic correctness; source taxonomy and units must be validated first.\n- Use primary keys to enforce the intended grain.\n- Preserve missing observations as `NULL`.\n- Never add aggregate categories to their components.\n- Reconcile important SQL totals against an independent pandas calculation."),
    ]
    return notebook(cells)


def main():
    nbf.write(time_series_notebook(), ROOT / "section4" / "timeSeriesEnergyConsumption.ipynb")
    nbf.write(sql_notebook(), ROOT / "section5" / "SQL_Pandas.ipynb")


if __name__ == "__main__":
    main()

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

**Accounting limits:** This tutorial selects combustible renewables, geothermal, broad hydro, other renewables, solar and wind. The broad Hydro export does not separate pumped-storage output; the selected sum must not be called a verified renewable-only total. Eurostat distinguishes storage generation from renewable primary production. The historical non-renewable combustible workbook has an unverified category boundary and is not presented as a complete total. Reporting-country coverage changes, flags must be inspected, and some values are missing. The loader recognizes `:` as missing and stops on unexpected text or non-finite measurements so those values are not silently discarded. Unflagged zeros do not prove complete reporting; in particular, check solar coverage against national sources and comparable definitions. This is not an EU-27 official total. For a renewable-only analysis obtain the required hydro breakdown and consistent country coverage first. [Eurostat electricity statistics](https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/9990.pdf).
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
        md("## Data\n\nThe shorthand `oil` represents oil and petroleum products, excluding the biofuel portion, as stated in the bundled workbook. The historical combustible-fuel workbook is labelled `historical_nonrenewable_unverified`: its original category boundary has not been confirmed. Do not treat it as a complete non-renewable total or add it to other categories without verifying their definitions and overlap. The [Eurostat SIEC vocabulary](https://dd.eionet.europa.eu/vocabulary/eurostat/siec/view?page=2) lists both `CF_NR` and `CF_NR_OTH` with an accepted date of 7 December 2018, but their concept entries provide no definitions. This does not identify the code selected in the historical workbook or establish the relationship between the two series."),
        code(r"""
SOURCE_FILES = {
    "coal_and_manufactured_gases": "nrg_cb_pem_page_spreadsheet_coal.xlsx",
    "combustible_renewable": "nrg_cb_pem_page_spreadsheet_combustionFuels_Renewables.xlsx",
    "historical_nonrenewable_unverified": "nrg_cb_pem_page_spreadsheet_combustionFuels_nonRenewables.xlsx",
    "geothermal": "nrg_cb_pem_page_spreadsheet_geothermal.xlsx",
    "hydro": "nrg_cb_pem_page_spreadsheet_hydro.xlsx",
    "natural_gas": "nrg_cb_pem_page_spreadsheet_naturalGas.xlsx",
    "nuclear_and_other_fuels_nec": "nrg_cb_pem_page_spreadsheet_nuclear.xlsx",
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
    with pd.ExcelFile(path) as workbook:
        wide = pd.read_excel(workbook, sheet_name=workbook.sheet_names[-1], skiprows=8)
    # Every month has an adjacent status column, including flags for missing values.
    columns = list(wide.columns)
    month_columns = [c for c in columns if not str(c).startswith("Unnamed") and c != columns[0]]
    flag_wide = pd.DataFrame({"country": wide[columns[0]]})
    for month in month_columns:
        next_index = columns.index(month) + 1
        flag_wide[month] = (wide[columns[next_index]] if next_index < len(columns)
                            and str(columns[next_index]).startswith("Unnamed") else pd.NA)
    wide = wide[[columns[0], *month_columns]].rename(columns={columns[0]: "country"})
    wide = wide.iloc[1:].copy()  # remove GEO label row
    footer = wide.index[wide["country"].eq("Special value")]
    if len(footer) != 1:
        raise ValueError("Workbook footer changed; inspect the source layout")
    wide = wide.loc[wide.index < footer[0]]
    wide = wide.dropna(subset=["country"])
    wide = wide[~wide["country"].astype(str).str.startswith("European Union")]
    long = wide.melt(id_vars="country", var_name="month", value_name="generation_gwh")
    # Keep status rows aligned with retained country rows, including after blank rows.
    flag_long = flag_wide.loc[wide.index].melt(id_vars="country", var_name="month", value_name="status_flag")
    long = long.merge(flag_long, on=["country", "month"], how="left", validate="one_to_one")
    long["month"] = pd.to_datetime(long["month"], format="%Y-%m")
    long["generation_gwh"] = pd.to_numeric(
        long["generation_gwh"].replace({":": np.nan}), errors="raise"
    )
    if not np.isfinite(long["generation_gwh"].dropna()).all():
        raise ValueError("Generation values must be finite or missing; inspect the source workbook")
    long["source"] = source
    return long[["country", "month", "source", "generation_gwh", "status_flag"]]

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
if generation.duplicated(key).any():
    raise ValueError("Duplicate country/month/source observations")
if generation["month"].min() != pd.Timestamp("2016-01-01") or generation["month"].max() != pd.Timestamp("2024-07-01"):
    raise ValueError("Unexpected historical snapshot month range")
negative_rows = generation["generation_gwh"].lt(0).sum()

# Source-file country sets are coverage diagnostics, not proof of valid measurements.
def country_coverage(frame):
    country_sets = frame.groupby("source")["country"].agg(lambda values: set(values.dropna()))
    union = set().union(*country_sets)
    return pd.DataFrame([
        {"source": source, "country_count": len(countries),
         "countries": ", ".join(sorted(countries)),
         "absent_from_source": ", ".join(sorted(union - countries))}
        for source, countries in country_sets.items()
    ]).set_index("source")

coverage_by_source = country_coverage(generation)
print("Source-file country coverage (differences need review):")
print(coverage_by_source[["country_count", "absent_from_source"]].to_string())
print("Countries across all source files:", ", ".join(sorted(generation["country"].dropna().unique())))

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

**Accounting limits:** This tutorial selects combustible renewables, geothermal, broad hydro, other renewables, solar and wind. The broad Hydro export does not separate pumped-storage output; the selected sum must not be called a verified renewable-only total. Eurostat distinguishes storage generation from renewable primary production. The historical non-renewable combustible workbook has an unverified category boundary and is not presented as a complete total. Reporting-country coverage changes, flags must be inspected, and some values are missing. The loader recognizes `:` as missing and stops on unexpected text or non-finite measurements so those values are not silently discarded. Unflagged zeros do not prove complete reporting; in particular, check solar coverage against national sources and comparable definitions. This is not an EU-27 official total. For a renewable-only analysis obtain the required hydro breakdown and consistent country coverage first. [Eurostat electricity statistics](https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/9990.pdf).
"""),
        md("## Setup"),
        code(r"""
database_rebuild_completed = False
database_queries_completed = False
input_validation_completed = False
from pathlib import Path
import sqlite3
from contextlib import closing

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
        md('## Data model\n\n`generation` has one row per country, month, and source, including a status flag. The shorthand `oil` represents oil and petroleum products, excluding the biofuel portion, as stated in the bundled workbook. The historical non-renewable workbook has an unverified category boundary; it is not presented as a full total or combined with coal, natural gas, and oil.\n\nA separately archived [July 2024 Eurostat data conversion](https://git.nomics.world/dbnomics-json-data/eurostat-json-data/-/tree/6c03e4a30952f80e7ab2521c7ec7e8a0f19835dd/nrg_cb_pem) labels the historical code `CF_NR` “Combustible fuels - non-renewable,” consistent with the workbook header. Most matched historical values agree, but that evidence does not establish the original selection or its category boundary. The [Eurostat SIEC vocabulary](https://dd.eionet.europa.eu/vocabulary/eurostat/siec/view?page=2) lists `CF_NR` and a separate `CF_NR_OTH` code, each with an accepted date of 7 December 2018; their concept entries provide no definitions. The dates and labels do not establish which code was selected in the historical workbook or a crosswalk between the series. In the bundled May 2024 snapshot, Poland reports 103.001 GWh in the disputed workbook and 6,307.405 GWh for coal and manufactured gases. This discrepancy makes the complete-total interpretation questionable; signed net generation and dataset definitions still need checking. Current provider categories and values can differ through definitions or revisions; do not infer a complete total from the label or combine the series with other fuels without a verified crosswalk.'),
        code(r"""
database_rebuild_completed = False
database_queries_completed = False
input_validation_completed = False
SOURCE_FILES = {
    "coal_and_manufactured_gases": ("nrg_cb_pem_page_spreadsheet_coal.xlsx", "fossil_component", 0),
    "combustible_renewable": ("nrg_cb_pem_page_spreadsheet_combustionFuels_Renewables.xlsx", "renewable", 1),
    "historical_nonrenewable_unverified": ("nrg_cb_pem_page_spreadsheet_combustionFuels_nonRenewables.xlsx", "boundary_unverified", 0),
    "geothermal": ("nrg_cb_pem_page_spreadsheet_geothermal.xlsx", "renewable", 1),
    "hydro": ("nrg_cb_pem_page_spreadsheet_hydro.xlsx", "hydro_including_storage", 1),
    "natural_gas": ("nrg_cb_pem_page_spreadsheet_naturalGas.xlsx", "fossil_component", 0),
    "nuclear_and_other_fuels_nec": ("nrg_cb_pem_page_spreadsheet_nuclear.xlsx", "nuclear_and_other_fuels_nec", 0),
    "oil": ("nrg_cb_pem_page_spreadsheet_oil.xlsx", "fossil_component", 0),
    "other_renewable": ("nrg_cb_pem_page_spreadsheet_otherRenewables.xlsx", "renewable", 1),
    "solar": ("nrg_cb_pem_page_spreadsheet_solar.xlsx", "renewable", 1),
    "wind": ("nrg_cb_pem_page_spreadsheet_wind.xlsx", "renewable", 1),
}

def load_export(path, source):
    with pd.ExcelFile(path) as workbook:
        wide = pd.read_excel(workbook, sheet_name=workbook.sheet_names[-1], skiprows=8)
    # Preserve the adjacent status columns, including flags on missing values.
    columns = list(wide.columns)
    month_columns = [c for c in columns if not str(c).startswith("Unnamed") and c != columns[0]]
    flag_wide = pd.DataFrame({"country": wide[columns[0]]})
    for month in month_columns:
        next_index = columns.index(month) + 1
        flag_wide[month] = (wide[columns[next_index]] if next_index < len(columns)
                            and str(columns[next_index]).startswith("Unnamed") else pd.NA)
    wide = wide[[columns[0], *month_columns]].rename(columns={columns[0]: "country"}).iloc[1:].copy()
    footer = wide.index[wide["country"].eq("Special value")]
    if len(footer) != 1:
        raise ValueError("Workbook footer changed; inspect the source layout")
    wide = wide.loc[wide.index < footer[0]]
    wide = wide.dropna(subset=["country"])
    wide = wide[~wide["country"].astype(str).str.startswith("European Union")]
    long = wide.melt(id_vars="country", var_name="month", value_name="generation_gwh")
    # Keep status rows aligned with retained country rows, including after blank rows.
    flag_long = flag_wide.loc[wide.index].melt(id_vars="country", var_name="month", value_name="status_flag")
    long = long.merge(flag_long, on=["country", "month"], how="left", validate="one_to_one")
    long["month"] = pd.to_datetime(long["month"], format="%Y-%m").dt.strftime("%Y-%m-01")
    long["generation_gwh"] = pd.to_numeric(long["generation_gwh"].replace(":", np.nan), errors="raise")
    if not np.isfinite(long["generation_gwh"].dropna()).all():
        raise ValueError("Generation values must be finite or missing; inspect the source workbook")
    long["source"] = source
    return long[["country", "month", "source", "generation_gwh", "status_flag"]]

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
database_rebuild_completed = False
database_queries_completed = False
input_validation_completed = False
if generation.duplicated(["country", "month", "source"]).any():
    raise ValueError("Duplicate country/month/source observations")
if set(generation["source"]) != set(sources["source"]):
    raise ValueError("Generation sources do not match the source table")
negative_rows = generation["generation_gwh"].lt(0).sum()

# Source-file country sets are coverage diagnostics, not proof of valid measurements.
def country_coverage(frame):
    country_sets = frame.groupby("source")["country"].agg(lambda values: set(values.dropna()))
    union = set().union(*country_sets)
    return pd.DataFrame([
        {"source": source, "country_count": len(countries),
         "countries": ", ".join(sorted(countries)),
         "absent_from_source": ", ".join(sorted(union - countries))}
        for source, countries in country_sets.items()
    ]).set_index("source")

coverage_by_source = country_coverage(generation)
print("Source-file country coverage (differences need review):")
print(coverage_by_source[["country_count", "absent_from_source"]].to_string())
print("Countries across all source files:", ", ".join(sorted(generation["country"].dropna().unique())))

if generation["month"].min() != "2016-01-01" or generation["month"].max() != "2024-07-01":
    raise ValueError("Unexpected historical snapshot month range")

input_validation_completed = True
pd.DataFrame({
    "rows": [len(generation)],
    "countries": [generation.country.nunique()],
    "months": [generation.month.nunique()],
    "sources": [generation.source.nunique()],
    "missing_values": [generation.generation_gwh.isna().sum()],
    "negative_rows_to_review": [negative_rows],
})
"""),
        md("## Create the SQLite database\n\nThe notebook replaces the tables in its own deterministic database file in one SQLite transaction. If an insert fails, SQLite rolls back the table replacement; the source spreadsheets remain unchanged. This uses `executemany`, not pandas `to_sql`, whose raw SQLite connection behavior differs. Run cells in order and rerun the full notebook after editing code or changing inputs; a syntax error prevents a cell's safety flags from resetting."),
        code(r'''
database_rebuild_completed = False
database_queries_completed = False
if not globals().get("input_validation_completed", False):
    raise RuntimeError("Input validation did not complete; rerun the notebook in order before rebuilding.")
with closing(sqlite3.connect(DB_PATH)) as connection:
    connection.execute("PRAGMA foreign_keys = ON")
    with connection:
        connection.execute("BEGIN")
        connection.execute("DROP TABLE IF EXISTS generation")
        connection.execute("DROP TABLE IF EXISTS source")
        connection.execute("""CREATE TABLE source (
            source TEXT PRIMARY KEY,
            source_group TEXT NOT NULL,
            in_selected_group INTEGER NOT NULL CHECK (in_selected_group IN (0, 1))
        )""")
        connection.execute("""CREATE TABLE generation (
            country TEXT NOT NULL,
            month TEXT NOT NULL,
            source TEXT NOT NULL REFERENCES source(source),
            generation_gwh REAL,
            status_flag TEXT,
            PRIMARY KEY (country, month, source)
        )""")
        connection.execute("CREATE INDEX generation_month_idx ON generation(month)")
        def sql_rows(frame, columns):
            for row in frame.loc[:, columns].itertuples(index=False, name=None):
                yield tuple(None if pd.isna(value) else value for value in row)
        connection.executemany("INSERT INTO source VALUES (?, ?, ?)",
                               sql_rows(sources, ["source", "source_group", "in_selected_group"]))
        connection.executemany("INSERT INTO generation VALUES (?, ?, ?, ?, ?)",
                               sql_rows(generation, ["country", "month", "source", "generation_gwh", "status_flag"]))

database_rebuild_completed = True
DB_PATH
'''),
        md("## Queries\n\nThe query selects six source categories for teaching; broad hydro includes storage generation, so this is not an official renewable-only total. The historical non-renewable series has an unverified boundary and is shown separately, not as a complete total. Nuclear and other fuels n.e.c. remain separate. Counts show reported rows for the selected group and the other categories. The displayed selected countries, source categories and flagged-row count show which observations contribute to each monthly sum; a flag is not automatically an error. Compare months only after checking coverage, and use an explicit common country–source subset when appropriate."),
        code(r'''
database_queries_completed = False
if not globals().get("database_rebuild_completed", False):
    raise RuntimeError("Database rebuild did not complete; rerun the notebook in order before querying.")
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
       SUM(CASE WHEN g.source = 'historical_nonrenewable_unverified' THEN g.generation_gwh END) AS historical_nonrenewable_unverified_gwh,
       SUM(CASE WHEN g.source = 'nuclear_and_other_fuels_nec' THEN g.generation_gwh END) AS nuclear_and_other_fuels_nec_gwh,
       COUNT(CASE WHEN s.in_selected_group = 1 THEN g.generation_gwh END) AS selected_reported_rows,
       COUNT(CASE WHEN s.in_selected_group = 1 AND g.generation_gwh IS NOT NULL
                  AND TRIM(COALESCE(g.status_flag, '')) <> '' THEN 1 END) AS selected_flagged_rows,
       GROUP_CONCAT(DISTINCT CASE WHEN s.in_selected_group = 1 AND g.generation_gwh IS NOT NULL
                                  THEN g.country END) AS selected_countries,
       GROUP_CONCAT(DISTINCT CASE WHEN s.in_selected_group = 1 AND g.generation_gwh IS NOT NULL
                                  THEN g.source END) AS selected_sources,
       COUNT(CASE WHEN g.source = 'historical_nonrenewable_unverified' THEN g.generation_gwh END) AS historical_reported_rows,
       COUNT(CASE WHEN g.source = 'nuclear_and_other_fuels_nec' THEN g.generation_gwh END) AS nuclear_and_other_reported_rows
FROM generation AS g
JOIN source AS s USING (source)
GROUP BY g.month
ORDER BY g.month;
"""

with closing(sqlite3.connect(DB_PATH)) as connection:
    selected_monthly = pd.read_sql_query(selected_query, connection, parse_dates=["month"])
    electricity_mix = pd.read_sql_query(mix_query, connection, parse_dates=["month"])

database_queries_completed = True
print(electricity_mix.tail().to_string(index=False))
'''),
        md('''### Compare the same reported country–source pairs

A comparable subset contains only country–source pairs with reported values in **both** selected months. The following example compares May and June 2024. It retains reported zeros, excludes missing values and reports the number of common pairs and flagged observations. This selective subset is not a Europe-wide trend or evidence of complete reporting; the broad hydro category still includes storage generation. Flags call for interpretation, not automatic rejection.'''),
        code(r'''
def compare_common_coverage(frame, source_table, first_month, second_month):
    """Compare reported observations for identical selected country–source pairs."""
    if first_month == second_month:
        raise ValueError("Choose two different months")
    selected = source_table.loc[source_table.in_selected_group.eq(1), "source"]
    data = frame.loc[frame.source.isin(selected),
                     ["country", "source", "month", "generation_gwh", "status_flag"]]
    def month_rows(month):
        rows = data.loc[data.month.eq(month)].dropna(subset=["generation_gwh"])
        return rows.drop(columns="month")
    pairs = month_rows(first_month).merge(month_rows(second_month), on=["country", "source"],
                                          suffixes=("_first", "_second"), validate="one_to_one")
    def flagged(column):
        return pairs[column].fillna("").astype(str).str.strip().ne("").sum()
    return {"first_month": first_month, "second_month": second_month,
            "common_country_source_pairs": len(pairs), "countries": pairs.country.nunique(),
            "first_gwh": pairs.generation_gwh_first.sum(min_count=1),
            "second_gwh": pairs.generation_gwh_second.sum(min_count=1),
            "first_flagged_rows": int(flagged("status_flag_first")),
            "second_flagged_rows": int(flagged("status_flag_second"))}

common_comparison = compare_common_coverage(generation, sources, "2024-05-01", "2024-06-01")
print("Common-coverage comparison (not a Europe-wide estimate):")
print(pd.Series(common_comparison).to_string())
'''),
        md("### Database reconciliation\n\nA direct pandas aggregation must match the SQL result. This checks consistency between SQL and pandas. Validate source labels and category boundaries separately; both paths can share the same taxonomy mistake."),
        code(r'''
if not globals().get("database_rebuild_completed", False) or not globals().get("database_queries_completed", False):
    raise RuntimeError("Database rebuild and queries must complete before reconciliation; rerun the notebook in order.")
selected_sources = sources.loc[sources.in_selected_group.eq(1), "source"]
expected = (
    generation[generation.source.isin(selected_sources)]
    .groupby("month")["generation_gwh"].sum(min_count=1)
)
actual = selected_monthly.set_index(selected_monthly.month.dt.strftime("%Y-%m-%d"))["selected_generation_gwh"]
pd.testing.assert_series_equal(expected, actual, check_names=False)

with closing(sqlite3.connect(DB_PATH)) as connection:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    key_duplicates = connection.execute("""
        SELECT COUNT(*) FROM (
          SELECT country, month, source, COUNT(*) AS n
          FROM generation GROUP BY country, month, source HAVING n > 1
        )
    """).fetchone()[0]

if integrity != "ok" or key_duplicates != 0:
    raise ValueError("Database integrity check or key uniqueness failed")
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

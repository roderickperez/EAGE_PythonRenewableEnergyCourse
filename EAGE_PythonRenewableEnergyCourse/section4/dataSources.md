# Data sources and reproducible downloads

A dataset is useful only with its definitions: variable, unit, geographical coverage, time period, frequency, measurement method and revision status. Keep the provider's metadata beside the file. CSV is a storage format, not evidence that two columns measure the same quantity. [Eurostat energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology); [pandas CSV parsing](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

## Choose a source for the question

| Source | Suitable use | Definitions to check before calculating |
|---|---|---|
| [Eurostat energy database](https://ec.europa.eu/eurostat/web/energy/database) | Country energy statistics | Dataset ID, energy product, net/gross boundary, unit, period and flags |
| [Our World in Data current energy catalogue](https://catalog.ourworldindata.org/energy/owid_energy/) | Annual country and world comparisons | Use the matching codebook, source attribution and new column names; electricity and total energy supply differ |
| [US EIA Open Data](https://www.eia.gov/opendata/) | US energy statistics and API exercises | Read the chosen API route, frequency and units; API access may require a free key |
| [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/) | Station weather observations | Station, variable, units, observation time, quality flags and missing-value codes |
| [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) | Gridded climate and reanalysis | Product, grid, time/accumulation conventions and variable metadata; registration/licence steps depend on product |

The links in each row are the provider's own documentation or distribution page. Reanalysis is a model-based reconstruction using observations; it must not be labelled a station measurement. [ECMWF: climate reanalysis](https://www.ecmwf.int/en/research/climate-reanalysis).

## Working download example: Eurostat → Python → SQLite

Start with the [complete seven-step download tutorial](../section5/download-to-database.md). It specifies **monthly net electricity generation**, dataset `nrg_cb_pem`, initially Austria `AT`, wind `RA300`, unit `GWH`, January–December 2023. Change `GEO` to another reporting European country; use the live country selector and check coverage. It contains a live API request, an Austria-only bundled snapshot and a downloadable notebook. For countries worldwide or a world aggregate, use the [global download tutorial](../section5/world-energy-download.md).

1. Open [the dataset in Eurostat's Data Browser](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem/default/table?lang=en).
2. Set frequency, your chosen country, product, unit and months as listed in the tutorial. Check the unit before downloading.
3. Use **Download**, choose CSV and export **All selected dimensions**. Keep flags and labels; verify that the export includes the complete intended selection.
4. Save the original file, its filter choices and retrieval date. Do not rename GWh values as MW.
5. Run the tutorial's API example to obtain a predictable tidy schema. Browser-export CSV layouts may differ from the API-derived CSV.
6. Check all twelve months, key uniqueness, missing values and flags before aggregating. Compare pandas and SQLite totals.
7. Save the CSV, raw JSON, provenance JSON and SQLite database using the environment-specific instructions.

Download controls and selection options: [Eurostat download guide](https://ec.europa.eu/eurostat/web/user-guides/data-browser/download-data/download-datasets). API filters and response format: [Eurostat API guide](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started).

## Working global example: OWID → country CSV → SQLite

1. Open the [current energy catalogue](https://catalog.ourworldindata.org/energy/owid_energy/).
2. Save `owid_energy.csv`, `owid_energy.codebook.csv` and `readme.md` using the download links in the [global tutorial](../section5/world-energy-download.md).
3. Run its first Python cell to download the files and create a country/region lookup.
4. Choose a country code, such as `AUT`, `BRA`, `IND` or `USA`, and an electricity-generation variable. Run the selection and save cells.
5. Download the selected CSV, SQLite database and provenance. Preserve the original files, checksums, retrieval dates and the codebook's original-provider attribution.
6. Inspect year coverage and units. Use `OWID_WRL` only when you intend to select the world aggregate; do not add it to country rows.

These steps follow the [current dataset documentation](https://catalog.ourworldindata.org/energy/owid_energy/readme.md). The old GitHub files are a legacy release and no longer updated. The new release changes column names and primary-energy methodology; do not mix its values and metadata with the old release. The worked example uses annual electricity generation in TWh, whereas the Eurostat exercise uses monthly net generation in GWh. Check definitions before comparing.

## Bundled data versus new downloads

The older [time-series notebook](timeSeriesEnergyConsumption.ipynb) and [SQL notebook](../section5/SQL_Pandas.ipynb) intentionally retain August 2024 Eurostat exports. Their historical selection and wide-table schema differ from the new API tutorial. The final-project CSVs are **synthetic** hourly inputs. Keep these three provenances separate; do not silently substitute one for another. [Course dataset definitions](../section8/data/README.md); [Eurostat database](https://ec.europa.eu/eurostat/web/energy/database).

## Practice

- **Easy:** reproduce the Austria wind snapshot and record its unit, twelve months, source URL and retrieval date.
- **Medium:** download any available European country for the same product and period by changing `GEO` in the Eurostat tutorial. Validate the returned country name, code and coverage.
- **Medium, worldwide:** use the global tutorial to download Brazil, India or another available country. Select matching years and electricity variables, then export the CSV and source metadata.
- **Advanced:** remove one observation in a copy, preserve it as missing, and report coverage plus observed energy. Explain why that is no longer an annual total.

These are original course exercises. The download tutorial provides checks and troubleshooting without requiring an API key.

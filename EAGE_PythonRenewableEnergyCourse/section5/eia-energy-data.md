# Another source: U.S. EIA electricity data

The [U.S. Energy Information Administration (EIA) electricity data](https://www.eia.gov/electricity/data.php) provides public U.S. electricity statistics, including generation. It complements the [Eurostat monthly exercise](download-to-database.md) and [worldwide annual dataset](world-energy-download.md); it is a data provider, not a PostgreSQL or SQLite server. Downloaded observations can be stored in your own database.

## A practical workflow

1. Open EIA's [Electricity Data Browser](https://www.eia.gov/electricity/data/browser/) and choose a generation series, geography, fuel and period. Inspect its units and metadata, then download a CSV.
2. Read the downloaded file with `pd.read_csv`, inspect its actual column names and missing values, and retain the source URL, series name, unit, geography, retrieval date and any revision notes. EIA formats may change; do not assume a CSV schema without inspecting it.
3. Normalize the selected columns, check that the intended observation key is unique, and store the result with `df.to_sql`. Query it with the methods in the [Python and SQLite notebook](sql-python.ipynb). Make a new database or table for the exercise rather than replacing a valuable existing one.
4. Compare like with like: the EIA U.S. series, Eurostat countries and OWID annual aggregates may have different geographic boundaries, reporting periods, units, categories and definitions. Convert units explicitly and never add country records to an aggregate containing them.

For automated retrieval, EIA's [API documentation](https://www.eia.gov/opendata/documentation.php) describes the v2 routes, filters and pagination. Its API requires a free registered key. Keep it in an environment variable or other secret store, never in a committed notebook or public URL. Browse the route metadata and inspect units before writing an automated query. If the request fails or access is unavailable, use the documented manual download; do not substitute invented observations.

**Exercise:** Select annual U.S. wind generation and one other source for the same years. Store the source, year, value and unit in SQLite, count missing years, and compute annual totals only after checking the source definitions and avoiding overlapping categories.

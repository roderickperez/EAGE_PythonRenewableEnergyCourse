# Bundled introductory datasets

These snapshots support the existing pandas and EDA lessons without requiring
Python to make a network request at lesson time. Retrieved 17 September 2026.
They are examples, not current observations for a renewable-energy assessment.

- `precipitation_hourly.csv`: NOAA/NCEI Climate Data Online
  [hourly precipitation sample](https://www1.ncdc.noaa.gov/pub/data/cdo/samples/PRECIP_HLY_sample_csv.csv).
  The CSV preserves the provider's original columns and flags. Consult the
  provider's product documentation before interpreting measurement units or flags.
- `penguins.csv`: [Seaborn example-data repository](https://github.com/mwaskom/seaborn-data),
  [CSV snapshot](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv),
  derived from [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/).
  Bill/flipper lengths are in millimetres and body mass in grams, as specified by
  the column names. Missing observations remain missing. The dataset is for
  learning EDA; it provides no renewable-energy evidence.

The sandbox manifest records SHA-256 hashes of the exact bundled bytes. This
snapshot is intentionally stable; update it explicitly and rerun the lesson checks
when adopting different data.

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

## Datasets for the technology labs

- [Bondville solar irradiance](surfrad-bondville-2024-07-01.csv) extracts UTC hour marks 10:00–23:00 from [NOAA SURFRAD file bon24183.dat](https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/2024/bon24183.dat). Source columns are the UTC date and time, zenith angle (degrees), downwelling global solar irradiance (W/m²) and its quality flag.
- [Geothermal gradient extract](ihfc-gradient-sample.csv) uses 14 records from the [IHFC Global Heat Flow Database, Release 2024](https://doi.org/10.5880/fidgeo.2024.014) by the Global Heat Flow Data Assessment Group et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It subsets and renames the source columns ID, lat_NS, long_EW, T_grad_mean and Quality_Code. The reported gradients use K/km. The selected records illustrate mapping and do not represent regional means.
- [Land polygons](natural-earth-land.geojson) are the [Natural Earth 1:110m land layer, v5.1.2](https://github.com/nvkelso/natural-earth-vector/blob/v5.1.2/geojson/ne_110m_land.geojson), a public-domain basemap.
- [Well temperature log](synthetic-well-temperature.csv) contains invented depth and temperature pairs and one deliberately missing reading. Depth is positive down in metres from the surface. Temperature is in degrees Celsius.

## Eurostat download tutorial snapshot

`eurostat_at_wind_2023.csv` and `.json` contain twelve monthly Austrian wind-generation observations (GWh), January–December 2023. The `.provenance.json` file records the exact API filters, retrieval time, provider update, source label and raw-response SHA-256. This is real public statistical data; it is neither synthetic nor the older August 2024 export. Follow [the download tutorial](../../section5/download-to-database.md) to reproduce it or explicitly select offline mode.

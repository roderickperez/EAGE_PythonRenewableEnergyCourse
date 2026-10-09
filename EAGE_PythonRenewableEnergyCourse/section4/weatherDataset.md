# Weather Dataset

Weather datasets support wind, solar, hydro, and demand analysis, but their variables are not interchangeable. [ECMWF reanalysis overview](https://www.ecmwf.int/en/research/climate-reanalysis); [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/).

## Essential metadata

- station observations, numerical weather prediction, reanalysis, and satellite products have different error structures;
- wind speed depends on height and averaging interval [@manwell2009], §2.3;
- irradiance may be global, direct, diffuse, horizontal, or plane-of-array [@foster2010], Chapter 2;
- precipitation accumulation depends on the reporting interval ([NOAA product documentation](https://www.ncei.noaa.gov/cdo-web/datasets));
- timestamps require a time zone and daylight-saving policy ([pandas time-series guide](https://pandas.pydata.org/docs/user_guide/timeseries.html));
- gridded values represent cells or model points, not exact site measurements ([ECMWF reanalysis](https://www.ecmwf.int/en/research/climate-reanalysis)).

## Quality checklist

1. Record provider, product version, variable identifier, unit, height/depth, spatial resolution, and time resolution.
2. Convert sentinel values to missing and check physical ranges.
3. Check duplicate timestamps, gaps, clock changes, and accumulated-variable resets.
4. Compare a sample with an independent station or provider when the decision is material.
5. Split forecast training and evaluation chronologically; fit imputation and scaling only on training data.

Weather data should be called “observed” only when it is a measurement. Reanalysis and forecasts are model estimates constrained by observations.

For a complete download workflow, follow [dataset → CSV → SQLite](../section5/download-to-database.md), then adapt its provenance and validation steps to the chosen weather product. The checks above are a course workflow informed by provider metadata; statistical preprocessing guidance: [scikit-learn data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

# Weather Dataset

Weather datasets support wind, solar, hydro, and demand analysis, but their variables are not interchangeable.

## Essential metadata

- station observations, numerical weather prediction, reanalysis, and satellite products have different error structures;
- wind speed depends on height and averaging interval;
- irradiance may be global, direct, diffuse, horizontal, or plane-of-array;
- precipitation accumulation depends on the reporting interval;
- timestamps require a time zone and daylight-saving policy;
- gridded values represent cells or model points, not exact site measurements.

## Quality checklist

1. Record provider, product version, variable identifier, unit, height/depth, spatial resolution, and time resolution.
2. Convert sentinel values to missing and check physical ranges.
3. Check duplicate timestamps, gaps, clock changes, and accumulated-variable resets.
4. Compare a sample with an independent station or provider when the decision is material.
5. Split forecast training and evaluation chronologically; fit imputation and scaling only on training data.

Weather data should be called “observed” only when it is a measurement. Reanalysis and forecasts are model estimates constrained by observations.

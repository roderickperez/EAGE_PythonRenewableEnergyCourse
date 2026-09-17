# xarray

xarray provides labelled N-dimensional arrays. A `DataArray` contains values plus named dimensions and coordinates; a `Dataset` contains several aligned variables.

```python
import xarray as xr
import numpy as np

# Three synthetic daily wind-component observations, in m/s.
dataset = xr.Dataset({"u10": ("time", [3.0, 4.0, 5.0]),
                      "v10": ("time", [4.0, 3.0, 0.0])},
                     coords={"time": np.array(["2025-01-01", "2025-01-02", "2025-02-01"], dtype="datetime64[ns]")})
wind_speed = (dataset["u10"] ** 2 + dataset["v10"] ** 2) ** 0.5
monthly = wind_speed.resample(time="MS").mean()
print(monthly)
```

Label-aware operations reduce axis mistakes, but coordinate alignment can introduce missing values. Inspect dimensions and coordinates after merges, selections, and resampling.

# xarray

xarray provides labelled N-dimensional arrays. A `DataArray` contains values plus named dimensions and coordinates; a `Dataset` contains several aligned variables. [xarray data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html).

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

For floating-point data, xarray's default mean skips missing observations. Report the valid count or duration coverage alongside each mean. The example uses an equal-sample mean; when inputs are interval means with unequal durations, establish the interval bounds and use appropriate duration weights. Instantaneous samples require an explicit integration convention.

Check the source's temporal definition before calculating wind speed. Mean speed is the mean of each time's vector magnitude, not generally the magnitude of time-averaged components: equal-duration eastward components +10 and −10 m/s with zero northward component give a mean speed of 10 m/s but a mean-vector speed of 0 m/s. See the [ECMWF explanation for ERA-Interim monthly means](https://confluence.ecmwf.int/spaces/CKB/pages/77229867/ERA-Interim%2BMonthly%2BMeans%2Bof%2BDaily%2BMeans%2Bwind%2BU%2Band%2BV%2Bversus%2Bwind%2Bspeed).

For a standard one-dimensional latitude or longitude index, check that coordinate labels are unique and monotonic before using `.sel(..., method="nearest")`; resolve duplicate labels according to the dataset before sorting. Normalize the target and data to the same longitude convention, but do not assume that this makes numeric selection wrap around the antimeridian. For example, with longitude labels −179°, 0° and 178°, numeric nearest selection for 179.9° chooses 178°, although −179° is geographically closer across the seam. Independent nearest latitude and longitude labels also need not identify the nearest point by geographic distance. For global or irregular grids, use a longitude-aware or geodesic-distance method as appropriate, and check the selected point and distance. See [xarray label-based selection](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html).

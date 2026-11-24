# netCDF

netCDF is a self-describing binary format widely used for gridded weather, climate, and Earth-science data. Variables can include dimensions, coordinates, units, calendars, fill values, and other metadata. The format name is **netCDF**, not “NetCFD.” [Unidata netCDF data model](https://docs.unidata.ucar.edu/netcdf-c/current/netcdf_data_model.html).

Use xarray to preserve labelled dimensions:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import xarray as xr
import numpy as np

# Create a tiny synthetic file so the example is reproducible without a download.
sample = xr.Dataset({"u10": ("time", [3.0, 4.0, 5.0]),
                     "v10": ("time", [4.0, 3.0, 0.0])},
                    coords={"time": np.array(["2025-01-01", "2025-02-01", "2025-03-01"], dtype="datetime64[ns]")})
sample["u10"].attrs["units"] = "m s-1"
sample["v10"].attrs["units"] = "m s-1"
with TemporaryDirectory() as folder:
    path = Path(folder) / "wind_reanalysis.nc"
    sample.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as ds:
        print(ds.sizes, ds.data_vars)
```

Check longitude convention, latitude order, time calendar, units, chunking, and missing values before computation.

The filename is illustrative: the three synthetic rows above are not an actual
reanalysis product. For real analysis, record the downloaded provider and version.
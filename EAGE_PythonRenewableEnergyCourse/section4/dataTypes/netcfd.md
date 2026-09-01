# netCDF

netCDF is a self-describing binary format widely used for gridded weather, climate, and Earth-science data. Variables can include dimensions, coordinates, units, calendars, fill values, and other metadata. The format name is **netCDF**, not “NetCFD.”

Use xarray to preserve labelled dimensions:

```python
import xarray as xr
dataset = xr.open_dataset("wind_reanalysis.nc")
print(dataset.sizes, dataset.data_vars)
```

Check longitude convention, latitude order, time calendar, units, chunking, and missing values before computation.

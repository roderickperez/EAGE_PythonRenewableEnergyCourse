# xarray

## Why xarray?

Wind and solar datasets often vary over **time, latitude and longitude**, sometimes also height. A plain array may have shape `(24, 3, 4)`, but its values do not themselves tell us which axis is time, where the grid cells are, or what the units mean. Keeping those details separately makes it easy to average the wrong axis or combine values from different locations.

[xarray](https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html) is a Python library for labeled multidimensional data. It adds named dimensions, coordinate labels and attributes to NumPy-like arrays. You can select a location by name, calculate a mean along `time`, and combine variables using their dimension names. A **DataArray** holds one variable; a **Dataset** groups variables and coordinates. See the [data structures guide](https://docs.xarray.dev/en/stable/user-guide/data-structures.html).

Labels reduce bookkeeping, but they do not verify units, coordinate systems or whether data are scientifically comparable. Always inspect coordinates and missing values after aligning data.

## How does it differ from other tools and formats?

| Tool or format | What it is | When it helps |
|---|---|---|
| NumPy | Python library providing fast numerical arrays | Core numerical calculations; axes are positional and labels are managed separately |
| pandas | Python library for labeled tables and time series | Station measurements, SQL results and other primarily tabular data |
| xarray | Python library for labeled N-dimensional arrays and datasets | Grids and cubes with coordinates such as time, latitude, longitude and height; can use NumPy underneath |
| CSV | Text file format for rows and columns | Simple exchange of tables; multidimensional grids and metadata need conventions |
| NetCDF | Scientific storage format for arrays, dimensions and metadata | Portable gridded datasets that xarray can open with a suitable backend |
| Zarr | Chunked array storage format | Large arrays and partial reads; xarray can open compatible Zarr stores with a suitable backend |
| MDIO | Energy-data storage format and library based on chunked arrays | Structured large energy datasets; compare it with xarray in [MDIO format](mdio-format.md#mdio-and-xarray-compared) |

xarray is a **library and data model**, not one specific file format. NumPy and pandas are libraries too; CSV, NetCDF and Zarr describe storage. xarray can work with NumPy arrays and supports several [file backends](https://docs.xarray.dev/en/stable/user-guide/io.html). Not every Zarr store is automatically an xarray dataset.

## Step 1 — Create a labeled wind dataset

This example is synthetic: coordinates are illustrative, not measurements or a resource assessment. Each variable has three hourly maps, two latitudes and two longitudes. The `u10` and `v10` variables represent eastward and northward wind components in m/s.

```{code-cell} python
import numpy as np
import xarray as xr

hours = np.array(["2025-01-01T00", "2025-01-01T01", "2025-01-01T02"], dtype="datetime64[ns]")
coords = {"time": hours, "latitude": [47.0, 48.0], "longitude": [15.0, 16.0]}
wind = xr.Dataset(
    data_vars={
        "u10": (("time", "latitude", "longitude"), np.full((3, 2, 2), 3.0), {"units": "m/s"}),
        "v10": (("time", "latitude", "longitude"), np.full((3, 2, 2), 4.0), {"units": "m/s"}),
    },
    coords=coords,
    attrs={"description": "synthetic wind example"},
)
print("Dimensions:", dict(wind.sizes))
print("Variables:", list(wind.data_vars))
```

Expected output:

```text
Dimensions: {'time': 3, 'latitude': 2, 'longitude': 2}
Variables: ['u10', 'v10']
```

## Step 2 — Calculate, select and summarize

Run this after Step 1 in the same Python session. `sel` uses coordinate labels; `isel` uses zero-based positions. The magnitude is not the arithmetic mean of the two components.

```{code-cell} python
wind["speed"] = np.hypot(wind["u10"], wind["v10"])
wind["speed"].attrs["units"] = "m/s"
point = wind["speed"].sel(latitude=47.0, longitude=15.0)
first_map = wind["speed"].isel(time=0)
mean_map = wind["speed"].mean(dim="time")
print("Point time series (m/s):", point.values)
print("First map shape:", first_map.shape)
print("Time-mean speed (m/s):", float(mean_map.sel(latitude=47.0, longitude=15.0)))
```

Expected output:

```text
Point time series (m/s): [5. 5. 5.]
First map shape: (2, 2)
Time-mean speed (m/s): 5.0
```

For a coordinate not present exactly, `sel(..., method="nearest")` is available; choose a suitable tolerance and check the selected coordinate and distance before interpreting it as a site observation. Coordinate-aware alignment can drop nonmatching labels in arithmetic, depending on the join policy, so compare coordinates explicitly before calculations.

## Step 3 — Save and reopen a NetCDF example

This self-contained example uses a temporary file, so it does not overwrite your data. The SciPy backend writes NetCDF3; other NetCDF variants can require different backends. For a real dataset, use an intentional filename, keep provenance, and verify units and coordinates after reopening.

```{code-cell} python
from pathlib import Path
from tempfile import TemporaryDirectory
import xarray as xr

sample = xr.Dataset({"speed": ("hour", [5.0, 6.0, 7.0], {"units": "m/s"})},
                    coords={"hour": [0, 1, 2]})
with TemporaryDirectory() as folder:
    path = Path(folder) / "wind_example.nc"
    sample.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as restored:
        print("Hours:", restored["hour"].values)
        print("Mean speed (m/s):", float(restored["speed"].mean()))
        print("Units:", restored["speed"].attrs["units"])
```

Expected output:

```text
Hours: [0 1 2]
Mean speed (m/s): 6.0
Units: m/s
```

For large data, chunking and optional Dask-backed arrays can make partial operations more manageable, but `.values` or a computation may load data into memory. Choose chunks for the access pattern and check memory needs. Consult [xarray's Dask guide](https://docs.xarray.dev/en/stable/user-guide/dask.html) and the [MDIO comparison](mdio-format.md#mdio-and-xarray-compared).

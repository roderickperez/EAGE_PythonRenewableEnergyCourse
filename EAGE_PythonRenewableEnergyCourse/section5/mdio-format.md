# MDIO format

MDIO (multidimensional input/output) is an open-source format and Python library for multidimensional energy data. It organizes arrays into independently accessible **chunks**, with metadata describing the data. The article discusses seismic data and wind models, including resource assessment, imaging, machine learning and visualization. [MDIO documentation](https://mdio-python.readthedocs.io/en/v0.9.3/)

## Article and reference

Sansal, Altay; Kainkaryam, Sribharath; Lasscock, Ben; and Valenciano, Alejandro (2023). “MDIO: Open-source format for multidimensional energy data.” *The Leading Edge*, July 2023, pp. 457–465 as printed in the [TGS-hosted article PDF](https://www.tgs.com/hubfs/TGS%20-%20MDIO%20-%20Open-Source%20Format%20for%20Multidimensional%20Energy%20Data%20-%20The%20Leading%20Edge.pdf). The DOI printed in that PDF, `10.1190/tle42070457.1`, currently resolves to a different paper; verify it with the publisher before citing it as an MDIO DOI.

## How it works

The article describes an MDIO dataset as Zarr arrays plus JSON metadata. Its illustrated seismic schema has global attributes (geometry and statistics), a metadata group (headers and a live-trace mask), and a data group containing chunked multidimensional arrays. The paper describes the 2023 schema version 0.2.10; library APIs and specifications continue to evolve.

An array might have axes `(inline, crossline, sample)` for a seismic volume, or `(latitude, longitude, time)` for a gridded wind variable. To retrieve one small region, the reader needs only the chunks that intersect it. The live mask distinguishes an absent seismic trace from an actual zero amplitude. Preserve coordinate reference systems, units, time conventions, variable definitions and provenance; dimensions alone do not supply their physical meaning.

**Chunking is a tradeoff.** Long time-series access and map-slice access favor different chunk shapes. Larger chunks reduce the number of storage requests but can fetch unnecessary data. The paper gives 4–8 MB as a cloud-object-store starting point and 1–2 MB for parallel filesystems, not universal optima; benchmark your own data and access patterns. Lossless compression preserves values; lossy compression changes values, so use it only after measuring error against a suitable scientific tolerance. The paper reports approximately 41% storage savings with lossless compression on the authors' seismic datasets; this is not a guarantee for other data.

## MDIO and xarray compared

MDIO and xarray address different parts of a workflow. MDIO defines a chunked storage format and provides tools for energy datasets; xarray is a Python library for working with labeled multidimensional arrays. You can analyze values read from an MDIO store with xarray after supplying the correct dimensions, coordinates and metadata. They are not interchangeable file formats.

| Question | MDIO | xarray |
|---|---|---|
| Primary role | Chunked energy-data storage and access | Labeled arrays and analysis in Python |
| Structure | MDIO schema, metadata and Zarr-backed arrays | `DataArray` or `Dataset` with named dimensions, coordinates and attributes |
| Typical use | Store and read chunks of large energy datasets, including seismic metadata | Select by coordinate, align variables, aggregate and visualize |
| File storage | MDIO store with its own layout | Reads and writes compatible NetCDF and Zarr datasets through backends |

Both can use chunked storage, but **an MDIO store is not automatically an xarray-compatible Zarr dataset**. In the [MDIO 0.9.3 documentation](https://mdio-python.readthedocs.io/en/v0.9.3/), a dedicated xarray interface is marked as future work. For a manual bridge, read data with `MDIOReader`, then construct a `DataArray` with validated coordinates and attributes. Do not assume that reading values alone preserves units, timestamps, missing-data masks or the coordinate reference system. See the [xarray lesson](xarray-extra.md) and [xarray I/O guide](https://docs.xarray.dev/en/stable/user-guide/io.html).

## Python walkthrough: create and inspect a small wind cube

The following standalone example uses **multidimio 0.9.3**, whose API is documented [here](https://mdio-python.readthedocs.io/en/v0.9.3/api_reference.html). Install it in a local Python 3.12 virtual environment; the course's browser Python environment does not include MDIO, so run these examples locally. In a terminal:

```text
python -m venv mdio-env
# Linux/macOS: source mdio-env/bin/activate
# Windows PowerShell: .\mdio-env\Scripts\Activate.ps1
python -m pip install "multidimio==0.9.3" "zarr==3.1.3" matplotlib
```

1. Save the complete script below as `wind_mdio.py`, then run `python wind_mdio.py`. It creates **synthetic**, non-georeferenced values, not observed wind data. The new `wind_demo.mdio` directory must not already exist. For real data, validate coordinates, units, timestamps and missing values separately and retain that metadata alongside the store.
2. It creates a small array with latitude, longitude and time axes and writes a losslessly compressed chunked MDIO store. The explicit coordinates are illustrative integers, not degrees or actual dates.
3. It opens the store, reads one map and one time series by **zero-based array index**, checks the round trip, and plots the selected time series.

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mdio import MDIOReader
from mdio.converters import numpy_to_mdio


def main():
    path = Path("wind_demo.mdio")
    if path.exists():
        raise FileExistsError(f"Choose a new output path: {path}")

    rng = np.random.default_rng(42)
    wind = rng.uniform(2, 15, size=(12, 16, 24)).astype("float32")  # m/s
    numpy_to_mdio(
        array=wind,
        mdio_path_or_buffer=path.resolve().as_uri(),
        chunksize=(4, 4, 8),
        index_names=["latitude", "longitude", "time"],
        index_coords={"latitude": np.arange(12),
                      "longitude": np.arange(16), "time": np.arange(24)},
        lossless=True,
        storage_options={"auto_mkdir": True},
        overwrite=False,
    )

    reader = MDIOReader(path.resolve().as_uri())
    print("Shape (latitude, longitude, time):", reader.shape)
    map_at_hour_5 = reader[:, :, 5]
    series = reader[3, 4, :]
    np.testing.assert_array_equal(series, wind[3, 4, :])
    print("Map shape:", map_at_hour_5.shape)
    print("First three wind speeds (m/s):", series[:3])
    plt.plot(np.arange(24), series, marker="o")
    plt.xlabel("Synthetic hour index")
    plt.ylabel("Wind speed (m/s)")
    plt.title("Synthetic wind at grid index (3, 4)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

This is an illustration of storing an array, not a complete wind-resource assessment. Store actual latitude/longitude, UTC timestamps, units, height, source and licensing metadata in an appropriate documented workflow, and inspect missing values before analysis. Choose chunks for your own access pattern. The [NumPy converter API](https://mdio-python.readthedocs.io/en/v0.9.3/api_reference.html) and [installation guide](https://mdio-python.readthedocs.io/en/v0.9.3/installation.html) describe the supported options.

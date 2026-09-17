# GeoPandas

GeoPandas extends pandas with geometry columns and coordinate-reference-system (CRS) operations.

```python
import geopandas as gpd

sites = gpd.GeoDataFrame({"site": ["Example A", "Example B"]},
                        geometry=gpd.points_from_xy([16.37, 14.29], [48.2, 48.31]),
                        crs="EPSG:4326").to_crs("EPSG:3035")
sites["buffer_10km"] = sites.geometry.buffer(10_000)
print(sites[["site", "buffer_10km"]])
```

Perform distance and area calculations in a suitable projected CRS, not directly in longitude/latitude degrees. Confirm the CRS after every spatial import and join.

These are synthetic demonstration locations. In a real workflow, replace their
construction with `gpd.read_file(...)` and inspect the source CRS. Geospatial
dependencies are optional; use the documented local/Colab environment if your
browser cannot load the required compiled packages.

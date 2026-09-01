# GeoPandas

GeoPandas extends pandas with geometry columns and coordinate-reference-system (CRS) operations.

```python
import geopandas as gpd

sites = gpd.read_file("renewable_sites.geojson").to_crs("EPSG:3035")
sites["buffer_10km"] = sites.geometry.buffer(10_000)
```

Perform distance and area calculations in a suitable projected CRS, not directly in longitude/latitude degrees. Confirm the CRS after every spatial import and join.

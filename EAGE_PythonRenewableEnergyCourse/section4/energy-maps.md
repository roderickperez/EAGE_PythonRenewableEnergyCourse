# Advanced energy maps with GeoPandas and Folium

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section4/energy-maps.ipynb)

Build a choropleth by joining a table to polygon geometry on a stable ID. The example uses three imaginary planning zones with the same longitude and latitude span. Map annual generation per unit of land area to compare intensity. A map of total generation answers a different question. Preserve the missing zone so it appears in gray.

GeoPandas stores the geometry and its coordinate reference system (CRS). This example uses WGS 84 (EPSG:4326). GeoJSON geometry lists longitude first and latitude second. Folium map centers and markers take latitude first. To calculate actual polygon areas, use an appropriate equal-area CRS before converting square metres to square kilometres.

## Join and plot the values

```{code-cell} python
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

zones = gpd.GeoDataFrame(
    {"zone": ["west", "central", "east"]},
    geometry=[box(15.8, 48.0, 16.0, 48.2),
              box(16.0, 48.0, 16.2, 48.2),
              box(16.2, 48.0, 16.4, 48.2)],
    crs="EPSG:4326",
)
generation = pd.DataFrame({"zone": ["west", "central"], "generation_gwh": [80, 140]})
# EPSG:3035 is an equal-area CRS for Europe. Keep WGS 84 for web display.
zones["area_km2"] = zones.to_crs("EPSG:3035").area / 1_000_000
joined = zones.merge(generation, on="zone", how="left", validate="one_to_one")
joined["gwh_per_km2"] = joined["generation_gwh"] / joined["area_km2"]
assert joined["zone"].tolist() == ["west", "central", "east"]
assert joined["gwh_per_km2"].isna().sum() == 1

ax = joined.plot(column="gwh_per_km2", legend=True, cmap="YlGn",
                 edgecolor="black", missing_kwds={"color": "lightgray"},
                 legend_kwds={"label": "Annual generation (GWh/km²)"})
ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="Synthetic planning zones")
plt.show()
```

## Build an interactive choropleth

Use the same join key in GeoJSON and the data table. Fix the bin edges so the same legend can be used to compare scenarios. Without map tiles, the teaching geometry stays visible on a plain background. Folium’s map still needs network access to load its JavaScript and stylesheet. In the sandbox or a notebook, use the rendered map below the Python output to explore the zones.

```{code-cell} python
import json
import folium

def make_choropleth(frame):
    map_view = folium.Map(location=[48.1, 16.1], zoom_start=9, tiles=None)
    geometry = json.loads(frame[["zone", "geometry"]].to_json())
    layer = folium.Choropleth(
        geo_data=geometry,
        data=frame,
        columns=["zone", "gwh_per_km2"],
        key_on="feature.properties.zone",
        bins=[0, 0.2, 0.35, 0.5, 0.8],
        fill_color="YlGn",
        nan_fill_color="lightgray",
        nan_fill_opacity=0.8,
        legend_name="Annual generation (GWh/km²), gray = missing",
    )
    layer.add_to(map_view)
    folium.GeoJsonTooltip(fields=["zone"], aliases=["Zone"]).add_to(layer.geojson)
    return map_view

map_view = make_choropleth(joined)
assert "Annual generation" in map_view.get_root().render()
map_view
```

## Practice

Add an explicitly reported zero for the east zone. Rebuild the choropleth and check that the value joins, stays zero and falls inside the first numeric bin.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
updated_generation = pd.concat(
    [generation, pd.DataFrame({"zone": ["east"], "generation_gwh": [0]})],
    ignore_index=True,
)
updated = zones.merge(updated_generation, on="zone", how="left", validate="one_to_one")
updated["gwh_per_km2"] = updated["generation_gwh"] / updated["area_km2"]
assert updated["gwh_per_km2"].notna().all()
assert updated.set_index("zone").loc["east", "gwh_per_km2"] == 0
make_choropleth(updated)
```
:::

[GeoPandas mapping guide](https://geopandas.org/en/stable/docs/user_guide/mapping.html) · [Folium choropleth guide](https://python-visualization.github.io/folium/v0.20.0/user_guide/geojson/choropleth.html)

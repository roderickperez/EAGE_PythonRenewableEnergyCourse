# Map geothermal gradients and plot a well temperature log

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section6/labs/geothermal-gradient.ipynb)

A geothermal gradient is temperature change per unit depth. Use a world map of selected observations to see how measurements vary geographically, then use a synthetic well log to estimate the gradient over a specified depth interval.

## Map reported gradient observations

The [course extract](../../data/examples/ihfc-gradient-sample.csv) contains 14 selected entries from the Global Heat Flow Database, Release 2024. It preserves each source record ID, coordinates, reported mean gradient and quality code. Nearby records can overlap on a world map. Coverage is uneven and includes onshore and marine observations. Use the data as points, with no interpolation or claims about temperature between sites. The plotted variable is the measured gradient, in K/km. Heat flow also depends on thermal conductivity and has different units.

```{code-cell} python
import json
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = next(candidate / "data/examples" for parent in (Path.cwd(), *Path.cwd().parents)
                for candidate in (parent, parent / "EAGE_PythonRenewableEnergyCourse")
                if (candidate / "data/examples/ihfc-gradient-sample.csv").is_file())
records = pd.read_csv(data_dir / "ihfc-gradient-sample.csv")
assert records["record_id"].is_unique
assert records["longitude"].between(-180, 180).all()
assert records["latitude"].between(-90, 90).all()
assert np.isfinite(records["gradient_k_per_km"]).all()

with (data_dir / "natural-earth-land.geojson").open(encoding="utf-8") as source:
    land = gpd.GeoDataFrame.from_features(json.load(source), crs="EPSG:4326")
points = gpd.GeoDataFrame(
    records, geometry=gpd.points_from_xy(records["longitude"], records["latitude"]),
    crs="EPSG:4326",
)
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
land.to_crs(points.crs).plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.4)
points.plot(ax=ax, column="gradient_k_per_km", cmap="plasma", markersize=32,
            edgecolor="black", linewidth=0.4, legend=True,
            legend_kwds={"label": "Measured temperature gradient (K/km)", "shrink": 0.7})
ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (°)", ylabel="Latitude (°)",
       title="Selected IHFC geothermal-gradient observations")
plt.show()
```

## Load and plot a temperature log

Read the [synthetic well log](../../data/examples/synthetic-well-temperature.csv), with depth increasing downward in a vertical well. Sort depth, preserve the missing temperature at 1,750 m as a visual gap and fit a straight line to valid points from 500 to 1,500 m. For a deviated well, use true vertical depth for a vertical gradient.

```{code-cell} python
well = pd.read_csv(data_dir / "synthetic-well-temperature.csv").sort_values("depth_m")
assert well["depth_m"].is_unique and well["depth_m"].ge(0).all()
assert well["temperature_c"].isna().sum() == 1

interval = well.loc[well["depth_m"].between(500, 1500)].dropna(subset=["temperature_c"])
gradient, intercept = np.polyfit(interval["depth_m"] / 1_000, interval["temperature_c"], 1)
assert np.isclose(gradient, 34.0)
fig, ax = plt.subplots(figsize=(4.5, 6), layout="constrained")
ax.plot(well["temperature_c"], well["depth_m"], "o-", label="Synthetic log")
ax.plot(intercept + gradient * interval["depth_m"] / 1_000,
        interval["depth_m"], "--", label="500–1,500 m fit")
ax.invert_yaxis()
ax.set(xlabel="Temperature (°C)", ylabel="Depth (m)", title="Temperature log")
ax.legend()
print(f"Fitted gradient: {gradient:.1f} K/km")
plt.show()
```

## Practice

Fit a gradient between 1,500 and 2,500 m. Report how many samples entered the fit and compare the slope with the shallower interval.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
deeper = well.loc[well["depth_m"].between(1500, 2500)].dropna(subset=["temperature_c"])
deep_gradient, _ = np.polyfit(deeper["depth_m"] / 1_000, deeper["temperature_c"], 1)
assert len(deeper) == 4 and np.isclose(deep_gradient, 31.8857142857)
print(f"{len(deeper)} points, {deep_gradient:.1f} K/km vs. {gradient:.1f} K/km")
```
:::

[IHFC Global Heat Flow Database, Global Heat Flow Data Assessment Group et al. (2024), CC BY 4.0](https://doi.org/10.5880/fidgeo.2024.014) · [Field and quality definitions](https://datapub.gfz-potsdam.de/download/10.5880.FIDGEO.2024.014-VEueRf/GHFDB-AG_et-al_2024_release_data_description.pdf) · [Natural Earth 1:110m land, public domain](https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-land/)

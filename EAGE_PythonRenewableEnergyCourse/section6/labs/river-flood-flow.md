# Map a river and calculate flood flow

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section6/labs/river-flood-flow.ipynb)

Use a schematic river reach near Vienna and an invented gauge record. The coordinates form a teaching line and the data have no connection to Danube flood magnitudes. Use official centreline and gauge metadata when repeating the workflow on a real river. The river and gauge use WGS 84 coordinates.

## Locate the river and gauge

```{code-cell} python
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point

river_line = LineString([(16.25, 48.28), (16.33, 48.24), (16.40, 48.20),
                         (16.48, 48.17), (16.57, 48.14)])
gauge_point = Point(16.40, 48.20)
river = gpd.GeoSeries([river_line], crs="EPSG:4326")
gauge = gpd.GeoSeries([gauge_point], crs=river.crs)
assert river_line.intersects(gauge_point)

ax = river.plot(color="steelblue", linewidth=2, label="River reach")
gauge.plot(ax=ax, color="darkorange", marker="^", label="Gauge")
ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="Schematic river location")
ax.legend()
plt.show()
```

## Calculate a peak and plot flood frequency

Peak discharge is the maximum recorded flow in cubic metres per second. Find the peak in an hourly flood hydrograph. Then rank a separate set of annual peaks from highest to lowest. A Weibull plotting position assigns exceedance probability $p=m/(n+1)$ to rank $m$ in $n$ complete years. The empirical return period is $1/p$. This short record illustrates plotting positions without extrapolating a design flood.

```{code-cell} python
event = pd.Series(
    [110, 160, 240, 410, 355, 290, 205, 155],
    index=pd.date_range("2024-06-03", periods=8, freq="h", tz="UTC"),
)
annual = pd.Series([285, 205, 321, 134, 360, 192, 223, 410],
                   index=range(2017, 2025), name="peak_m3s")
assert annual.index.is_unique and annual.notna().all()
ranked = annual.sort_values(ascending=False).to_frame()
ranked["rank"] = range(1, len(annual) + 1)
ranked["exceedance_probability"] = ranked["rank"] / (len(annual) + 1)
ranked["return_years"] = 1 / ranked["exceedance_probability"]
print(f"Event peak: {event.max()} m³/s at {event.idxmax():%H:%M UTC}")

fig, (hydrograph, frequency) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
hydrograph.plot(event.index.hour, event, marker="o")
hydrograph.set(xlabel="Hour (UTC)", ylabel="Discharge (m³/s)", title="Synthetic flood event")
frequency.plot(ranked["return_years"], ranked["peak_m3s"], "o")
frequency.set(xscale="log", xlabel="Empirical return period (years)",
              ylabel="Annual peak discharge (m³/s)", title="Synthetic annual peaks")
plt.show()
```

## Practice

Find the plotting position and discharge for the second-largest annual peak. Calculate the approximate event volume above a 100 m³/s baseline using the trapezoid rule for the seven one-hour spans between observations. Treat the recorded flows as point samples.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
excess = (event - 100).clip(lower=0)
volume_m3 = ((excess.iloc[:-1].to_numpy() + excess.iloc[1:].to_numpy()) / 2).sum() * 3_600
second = ranked.iloc[1]
assert second["peak_m3s"] == 360 and second["return_years"] == 4.5
assert volume_m3 == 3_933_000
print(f"Second peak: {second['peak_m3s']:.0f} m³/s at p={second['exceedance_probability']:.1%}")
print(f"Volume above baseline between samples: {volume_m3:,} m³")
```
:::

[USGS Bulletin 17C](https://pubs.usgs.gov/publications/guidelines-determining-flood-flow-frequency-bulletin-17c) · [Hydroelectric generation and hydrographs](../hydroelectricEnergy.md)

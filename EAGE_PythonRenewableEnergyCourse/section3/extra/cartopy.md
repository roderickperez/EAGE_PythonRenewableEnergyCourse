# Cartopy

Cartopy provides map projections and geospatial plotting for Matplotlib. A projection describes how the curved Earth is represented on a plane; every plotted dataset also needs the correct source coordinate system through `transform=`.

```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
longitude, latitude = [16.37, 14.29], [48.2, 48.31]  # synthetic example sites
ax.set_extent([8, 20, 44, 52], crs=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
ax.scatter(longitude, latitude, transform=ccrs.PlateCarree())
plt.show()
```

In a local environment, `ax.coastlines()` can add Natural Earth coastline data;
the first call downloads those files. The example avoids that network dependency.
Cartopy is an optional compiled package; use local Jupyter/Colab if it is unavailable
in your browser's Python distribution.

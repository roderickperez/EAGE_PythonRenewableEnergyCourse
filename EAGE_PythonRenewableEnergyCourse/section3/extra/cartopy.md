# Cartopy

Cartopy provides map projections and geospatial plotting for Matplotlib. A projection describes how the curved Earth is represented on a plane; every plotted dataset also needs the correct source coordinate system through `transform=`.

```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.scatter(longitude, latitude, transform=ccrs.PlateCarree())
```

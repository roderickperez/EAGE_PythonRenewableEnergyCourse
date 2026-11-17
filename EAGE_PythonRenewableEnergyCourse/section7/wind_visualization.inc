
## Python Visualization: Wind Direction

These original examples use **synthetic data** and are inspired by the saved videos [How to make a wind rose with Python](https://youtu.be/UdMCWf6WuK4), [How to make a wind rose plot with Python](https://youtu.be/g-MmDTbx8kI), and [Overlay Wind Barbs and Wind Vectors on a Map](https://youtu.be/1Nr9KW89lCc). They illustrate the topics rather than reproduce the videos' code or data. The wind-rose examples use NumPy and Matplotlib and can run in this course's browser workspace. The map example requires local Cartopy, as explained below.

**Direction convention:** meteorological direction is the direction the wind comes **from**, clockwise from north: N = 0°, E = 90°, S = 180°, W = 270°. Arrows in a vector map instead point **toward** the direction of airflow. Exclude or separately report calm winds, for which direction is undefined. [ECMWF wind conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=212441203).

### 1. A simple frequency wind rose

The example assigns each observation to its nearest 45° compass sector. Eight equally spaced samples make it easy to inspect the result. Frequencies use all eight valid non-calm observations as their denominator.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

directions = np.array([0, 45, 90, 180, 225, 270, 270, 270])  # from degrees
sector = ((directions + 22.5) // 45).astype(int) % 8
percent = np.bincount(sector, minlength=8) / len(directions) * 100
labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
angles = np.arange(8) * 2 * np.pi / 8

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.bar(angles, percent, width=2*np.pi/8, color='#328baa', edgecolor='white')
ax.set_xticks(angles, labels)
ax.set_title('Wind direction frequency (%)')
print('N: {:.1f}%; W: {:.1f}%'.format(percent[0], percent[6]))
plt.show()
```

**Expected output:** `N: 12.5%; W: 37.5%`. The west sector is tallest; SE and NW have no observations.

![Expected simple wind rose](../../images/wind-visualization/simple-rose.png)

### 2. Wind rose with speed categories

Stack speed ranges within each directional sector. The example uses the same directions and eight measured speeds. Each colored segment is a percentage of **all valid observations**, so the heights of all segments together sum to 100%.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

direction = np.array([0, 45, 90, 180, 225, 270, 270, 270])
speed = np.array([3, 5, 7, 2, 4, 6, 8, 10])  # m/s
sector = ((direction + 22.5) // 45).astype(int) % 8
edges = [0, 4, 8, np.inf]
angles = np.arange(8) * np.pi / 4
colors = ['#9bd3df', '#328baa', '#164e70']
labels = ['0–<4 m/s', '4–<8 m/s', '≥8 m/s']
bottom = np.zeros(8)

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
for low, high, color, label in zip(edges[:-1], edges[1:], colors, labels):
    count = np.bincount(sector[(speed >= low) & (speed < high)], minlength=8)
    percentage = count / len(speed) * 100
    ax.bar(angles, percentage, width=np.pi/4, bottom=bottom,
           color=color, edgecolor='white', label=label)
    bottom += percentage
ax.set_xticks(angles, ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.set_title('Direction and speed frequency (%)')
ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
print('Total observations:', len(speed), '| Total frequency:', bottom.sum(), '%')
plt.show()
```

**Expected output:** `Total observations: 8 | Total frequency: 100.0 %`. The west sector is 37.5%, with 12.5% in the 4–<8 m/s band and 25% in the ≥8 m/s band.

![Expected wind rose showing speed categories](../../images/wind-visualization/speed-rose.png)

### 3. Wind vectors and barbs on a map

This local Python example plots a small **synthetic** Europe-region grid on Cartopy's bundled low-resolution background. `u` is eastward and `v` northward, in m/s. Arrows show where air is going; barbs use **knots** here (1 m/s ≈ 1.94384 knots). A full barb denotes 10 knots and a half barb 5 knots, rounded to the nearest 5 by the plotting convention. The background is a visual reference, not a site survey.

In a local environment, install `numpy matplotlib cartopy`; Cartopy is not available in the course's browser Python workspace. The pre-rendered result below lets you inspect the output without installing it. Use `transform=ccrs.PlateCarree()` because the input positions are longitude and latitude. For real gridded data, check the coordinate system and whether vector components must be rotated into east/north.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

lon, lat = np.meshgrid([4, 10, 16], [45, 50, 55])
u = np.full(lon.shape, 6.0)   # eastward m/s
v = np.full(lat.shape, 2.0)   # northward m/s
speed = np.hypot(u, v)
projection = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                         subplot_kw={'projection': projection})
for ax in axes:
    ax.set_extent([-2, 22, 40, 60], crs=projection)
    ax.stock_img()
    ax.gridlines(draw_labels={'bottom': 'x', 'left': 'y'},
                 linewidth=0.5, color='white', alpha=0.7)
arrows = axes[0].quiver(lon, lat, u, v, transform=projection,
                        color='#083d5c', scale=65)
axes[0].quiverkey(arrows, 0.78, 0.05, 5, '5 m/s', coordinates='axes')
axes[0].set_title('Wind vectors (m/s)')
axes[1].barbs(lon, lat, u*1.94384, v*1.94384, transform=projection,
              color='#083d5c', length=6)
axes[1].set_title('Wind barbs (knots)')
print('Speed: {:.2f} m/s'.format(speed[0, 0]))
plt.tight_layout()
plt.show()
```

**Expected output:** `Speed: 6.32 m/s`. Nine equal arrows point northeast; the barbs show the same synthetic flow in knots.

![Expected wind vectors and barbs on a map](../../images/wind-visualization/vector-map.png)

Use [Wind data and calculations](../../section5/wind-data.md) for ERA5 access, the pywind notebooks and an energy calculation. For measured data, inspect units, direction convention, timestamps, missing values and the representativeness of the site before making a wind rose.

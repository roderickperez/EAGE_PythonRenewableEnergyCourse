# Wind data and calculations

This lesson connects three resources saved for the course: the [ERA5 wind extraction video](https://youtu.be/aWVAw3gfc6Y), the [Vortex FDC pywind notebooks](https://github.com/VortexFDC/pywind/tree/main/notebooks), and a saved wind-farm energy-output calculator notebook. The last saved [notebook link](https://github.com/CoolatMax/100-day-data-science/blob/main/notebooks/Phase01/D02/Wind_Farm_Energy_Output_Calculator.ipynb) was unavailable when checked; the independent calculator below is not a reproduction of that notebook.

## 1. Obtain ERA5 wind data

[ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) is a gridded reanalysis combining a physical model with observations. It is **not a local turbine measurement**. In the Climate Data Store, register or sign in and accept the dataset's applicable licence and terms. Select the time period and area, choose matching eastward and northward wind components at the desired height (for example, 10 m or 100 m), and request a format supported by your environment. Download options can change; check the provider's current interface and documentation.

Retain the source, download date, area, coordinates, height, units, time zone, product and licence with the original file. Inspect variable names and dimensions before selecting a point; file formats and names can vary. Check missing values and whether timestamps are UTC. At 0.25° resolution, the regular ERA5 grid describes a broad area, so site-specific measurement and assessment remain important. [Dataset description and citation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), [ECMWF direction conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=212441203).

For example, after downloading a compatible NetCDF file, inspect it locally using `xarray`. This illustrative snippet requires **your own file** and matching variable/coordinate names; do not expect the sample output to match your dataset.

```text
import xarray as xr
ds = xr.open_dataset('era5_wind.nc')
print(ds)  # inspect variables, coordinates, units and dimensions first
```

## 2. Convert components into speed and direction

Run this self-contained example with NumPy. Positive `u` points east and positive `v` north; meteorological direction is where the wind comes **from**. A calm wind has no defined direction and is represented as `nan` here. Use the correct height and units from your own data.

```{code-cell} python
import numpy as np

u = np.array([0.0, -4.0, 3.0, 0.0])  # eastward m/s
v = np.array([-5.0, 0.0, 4.0, 0.0])  # northward m/s
speed = np.hypot(u, v)
direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
direction = np.where(speed > 0, direction, np.nan)
print('Speed (m/s):', np.round(speed, 2))
print('From (degrees):', np.round(direction, 2))
```

**Expected output:**

```text
Speed (m/s): [5. 4. 5. 0.]
From (degrees): [  0.    90.   216.87    nan]
```

For examples of reading NetCDF and working with wind observations and model data, explore [Vortex FDC's pywind notebooks](https://github.com/VortexFDC/pywind/tree/main/notebooks) and its [dataset setup instructions](https://github.com/VortexFDC/pywind/blob/main/README_about_datasets.md). Their notebooks may require separate downloads and a local Python environment. Check provenance and terms for each dataset before using it.

## 3. Calculate wind-farm energy from interval power

The following independent example calculates energy from **interval-average electrical power**, not directly from wind speed. Four synthetic hourly readings represent the combined farm output. A power curve, turbine availability, wake effects and electrical losses are not inferred; include them only when your input data and model justify it.

```{code-cell} python
import numpy as np

power_mw = np.array([2.0, 3.0, 4.0, 1.0])  # farm average for each hour
interval_hours = 1.0
energy_mwh = power_mw.sum() * interval_hours
print('Farm energy: {:.1f} MWh'.format(energy_mwh))
print('Mean power: {:.1f} MW'.format(power_mw.mean()))
```

**Expected output:** `Farm energy: 10.0 MWh` and `Mean power: 2.5 MW`. For non-hourly or irregular intervals, multiply each interval's average power by its actual duration before summing. For incomplete data, report coverage and do not assume missing readings are zero.

To visualize direction and speed, see the wind workbook's [Python Visualization: Wind Direction](../section7/workbook/wind.md#python-visualization-wind-direction). Its examples use synthetic data and provide expected plots.

# Compare clear-sky and measured solar irradiance

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section6/labs/solar-irradiance.ipynb)

Compare global horizontal irradiance (GHI) measured at NOAA’s SURFRAD station in Bondville, Illinois, on 1 July 2024 with a Haurwitz clear-sky estimate. GHI is radiant power per unit horizontal area in W/m². The [course extract](../../data/examples/surfrad-bondville-2024-07-01.csv) keeps the source’s UTC hour marks from 10:00 to 23:00. Each row is a selected minute observation. The rows are neither hourly averages nor a complete daily energy record.

## Align observations and model

Use the reported solar zenith angle as an approximation to the apparent zenith input of the Haurwitz model. The model uses only zenith. Restrict its expression to sun-above-horizon rows to avoid division by zero. Retain the measured nighttime value as recorded and restrict the error calculation to valid daylight pairs.

```{code-cell} python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

asset = Path("data/examples/surfrad-bondville-2024-07-01.csv")
csv_path = next(candidate / asset for parent in (Path.cwd(), *Path.cwd().parents)
                for candidate in (parent, parent / "EAGE_PythonRenewableEnergyCourse")
                if (candidate / asset).is_file())
solar = pd.read_csv(csv_path)
solar["time_utc"] = pd.to_datetime(solar["time_utc"], utc=True)
solar = solar.set_index("time_utc").sort_index()
assert solar.index.is_unique and len(solar) == 14
assert solar["zenith_deg"].between(0, 180).all()

cos_zenith = np.cos(np.deg2rad(solar["zenith_deg"]))
daylight = solar["zenith_deg"].lt(90)
solar["clear_sky_wm2"] = 0.0
mu = cos_zenith[daylight]
solar.loc[daylight, "clear_sky_wm2"] = 1098 * mu * np.exp(-0.059 / mu)
solar["measured_wm2"] = solar["ghi_wm2"].where(solar["ghi_qc"].eq(0))
assert solar.loc[~daylight, "clear_sky_wm2"].eq(0).all()

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.plot(solar.index, solar["clear_sky_wm2"], label="Haurwitz clear sky")
ax.plot(solar.index, solar["measured_wm2"], "o-", label="NOAA SURFRAD measured")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=solar.index.tz))
ax.set(xlabel="Time (UTC)", ylabel="GHI (W/m²)", title="Bondville, 1 July 2024")
ax.legend()
plt.show()
```

## Practice

Compute the daylight mean absolute difference across pairs that pass the quality flag. Identify the hour with the largest absolute difference. Explain why neither the difference nor its sign identifies the cause by itself. Consider cloud cover, model assumptions and sensor behavior.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
paired = daylight & solar["measured_wm2"].notna()
difference = solar.loc[paired, "measured_wm2"] - solar.loc[paired, "clear_sky_wm2"]
assert len(difference) == 13 and difference.abs().idxmax().hour == 20
print(f"Mean absolute difference: {difference.abs().mean():.1f} W/m²")
print(f"Largest absolute difference: {difference.abs().idxmax():%H:%M UTC}")
```
:::

[Original NOAA daily record](https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/2024/bon24183.dat) · [SURFRAD field definitions](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.iotools.read_surfrad.html) · [Haurwitz model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html) · [Solar PV generation calculations](../solarEnergy.md)

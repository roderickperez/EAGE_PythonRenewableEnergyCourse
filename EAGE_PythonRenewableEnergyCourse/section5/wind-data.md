# Wind data and calculations

This lesson connects three resources saved for the course: the [ERA5 wind extraction video](https://youtu.be/aWVAw3gfc6Y), the [Vortex FDC pywind notebooks](https://github.com/VortexFDC/pywind/tree/main/notebooks), and a saved wind-farm energy-output calculator notebook. The last saved [notebook link](https://github.com/CoolatMax/100-day-data-science/blob/main/notebooks/Phase01/D02/Wind_Farm_Energy_Output_Calculator.ipynb) was unavailable when checked; the independent calculator below is not a reproduction of that notebook.

## 1. Obtain ERA5 wind data

[ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) is a gridded reanalysis combining a physical model with observations. It is **not a local turbine measurement**. In the Climate Data Store, register or sign in and accept the dataset's applicable licence and terms. Select the time period and area, choose matching eastward and northward wind components at the desired height (for example, 10 m or 100 m), and request a format supported by your environment. Download options can change; check the provider's current interface and documentation.

Retain the source, download date, area, coordinates, height, units, time zone, product and licence with the original file. Inspect variable names and dimensions before selecting a point; file formats and names can vary. Check missing values and whether timestamps are UTC. The regridded atmospheric reanalysis is supplied at 0.25° and its uncertainty estimate at 0.5°; check the selected product and actual coordinates rather than assuming a common resolution. These grids describe broad areas, so site-specific measurement and assessment remain important. Recent data may be the early **ERA5T** release, which can differ from the final release if serious flaws are found. Record the release status, retrieval date and a file checksum, and consider rechecking after finalization. [Dataset description and citation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), [ECMWF direction conventions](https://confluence.ecmwf.int/pages/viewpage.action?pageId=212441203).

When transferring direction formulas to a spreadsheet, check its `ATAN2` argument order. The 2021 ECMWF page linked above and the version current when checked on 19 November 2026 give different spreadsheet argument orders; Excel uses x before y, unlike NumPy’s `atan2(y, x)`. Test cardinal directions: eastward component 0 and northward component −5 m/s must give wind **from north**, 0°.

ERA5 10 m and 100 m wind components are instantaneous fields, even when available hourly; they are not automatically hourly-average power. Applying a turbine curve at each timestamp and multiplying by one hour is a numerical approximation that requires a stated sampling rule and coverage checks; nonlinear turbine response can differ from interval-average power. See [ECMWF ERA5 parameter documentation](https://confluence.ecmwf.int/pages/viewpage.action?navigatingVersions=true&pageId=177483349).

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

For examples of reading NetCDF and working with wind observations and model data, explore [Vortex FDC's pywind notebooks](https://github.com/VortexFDC/pywind/tree/main/notebooks) and its [dataset setup instructions](https://github.com/VortexFDC/pywind/blob/main/README_about_datasets.md). Their notebooks may require separate downloads, repository helper files and a local Python environment. The first notebook expects the Frøya files and specific variables and coordinates such as `M`, `Dir` or `D`, and `lev`; arbitrary ERA5 files may need adaptation. Its relative paths assume a kernel started in the repository’s `notebooks` folder, so adjust them for a different working directory. Check provenance and terms for each dataset before using it.

**Checks on 19 November 2026:** the provider listed MD5 `3ad4368eef6c8bb6ce8137448cdaaa1c` for Frøya `froya.zip`, while the archive retrieved that day had MD5 `2ec70229bb985617d99c1bef2d3656b8`. Ask the provider to reconcile the archive and checksum before relying on it; the mismatch alone does not establish tampering, and a ZIP integrity check is not publisher authentication. The external text-reading example used a combined-column `parse_dates` form removed in pandas 3.0; read the date and time as strings and convert their combination explicitly, or use the provider's compatible environment. The linked merge example also omitted an import for its text reader, and its `1H` resampling alias fails in pandas 3.0; explicitly import the reader and use `1h` in a compatible environment, then check for further errors. Its dataset-wide resampling also used an arithmetic mean of angles: use a circular mean (and report an undefined direction when the resultant is near zero) for directions such as 350° and 10°. These are checks of the external examples at that date, not claims about later revisions or the course's NumPy calculation above.

**Checksum recheck on 21 November 2026:** the provider's instructions still listed `3ad4368eef6c8bb6ce8137448cdaaa1c`, and a fresh HTTPS download again produced `2ec70229bb985617d99c1bef2d3656b8`. The downloaded ZIP passed its internal CRC check, which does not resolve the mismatch or authenticate the file. The notebook compatibility observations above were not re-tested in this checksum check.

### Read a headerless four-column wind file

This independent fixture demonstrates the text-reader step that fails in the linked example under pandas 3. It reads whitespace-separated `YYYYMMDD HHMM speed direction` columns as strings where needed, then parses the combined timestamp explicitly. The values are synthetic. This narrow reader does not replace the provider's metadata handling, timezone conversion or quality-control policy; confirm those details, the number of columns and file provenance before adapting it. In particular, these timestamps are left timezone-naive rather than silently declared UTC.

```{code-cell} python
from io import StringIO
import numpy as np
import pandas as pd

fixture = """20040101 0000 4.6 154
20040101 0010 4.8 160
20040101 0020 5.1 165
"""
wind_text = pd.read_csv(
    StringIO(fixture), sep=r"\s+", header=None,
    dtype={0: "string", 1: "string"},
)
if wind_text.shape[1] != 4:
    raise ValueError("This example expects exactly four columns")
wind_text.columns = ["date", "time", "speed_ms", "direction_deg"]
wind_text["timestamp"] = pd.to_datetime(
    wind_text["date"] + " " + wind_text["time"].str.zfill(4),
    format="%Y%m%d %H%M", errors="raise",
)
if wind_text["timestamp"].isna().any() or wind_text["timestamp"].duplicated().any():
    raise ValueError("Inspect missing or duplicate timestamps")
for name in ("speed_ms", "direction_deg"):
    wind_text[name] = pd.to_numeric(wind_text[name], errors="raise")
invalid_speed = wind_text["speed_ms"].notna() & (
    ~np.isfinite(wind_text["speed_ms"]) | wind_text["speed_ms"].lt(0)
)
invalid_direction = wind_text["direction_deg"].notna() & (
    ~np.isfinite(wind_text["direction_deg"]) | ~wind_text["direction_deg"].between(0, 360)
)
if invalid_speed.any() or invalid_direction.any():
    raise ValueError("Inspect non-finite or out-of-range wind measurements")
print(wind_text[["timestamp", "speed_ms", "direction_deg"]].to_string(index=False))
```

To adapt the example, replace `StringIO(fixture)` with the verified file path only if its format matches; inspect the header, units and any missing-value codes first. Unlike the external helper, this example does not automatically drop rows with missing measurements; it rejects non-finite or out-of-range values and leaves missing-value policy to the learner. A header-bearing model series or a measurement file with an additional standard-deviation column requires a matching parser and documented handling of those fields.

For a separate observational dataset, the open [Frøya 10-minute measurements on Zenodo](https://zenodo.org/records/2557500) provide a comma-separated text file and documentation for measurements from 2009–2015. Zenodo lists a checksum for each file; check the downloaded file against its record and review the file's time conventions, sensor selection and terms before analysis. This dataset is not a drop-in replacement for the Vortex pack: it has different columns and does not supply that pack's modelled series or helper files. The separate [1 Hz record](https://zenodo.org/records/3403362) is marked restricted.

Before reading the downloaded text file, compare its checksum with the one shown for **that exact file and version** on the Zenodo record. For example, replace the path and expected MD5 below with the downloaded filename and the record's value. This code reads the file in chunks; it stops on a mismatch rather than proceeding with the analysis. The record may change, so do not reuse an old checksum without checking it. MD5 comparison detects a mismatch with the listed value; it does not by itself prove a file is authentic.

```text
from pathlib import Path
import hashlib
import re

path = Path("YOUR_DOWNLOADED_FILE.txt")
expected_md5 = "REPLACE_WITH_RECORD_MD5"
if not re.fullmatch(r"[0-9a-fA-F]{32}", expected_md5):
    raise ValueError("Copy the file's 32-character MD5 from its Zenodo record")
hasher = hashlib.md5()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        hasher.update(chunk)
actual_md5 = hasher.hexdigest()
if actual_md5.lower() != expected_md5.lower():
    raise ValueError(f"Checksum mismatch: downloaded file has MD5 {actual_md5}")
print(f"Checksum matches: {path.name}")
```

This small fixture uses the published file's `Timestamp`, `WS11` and `WD11` columns and its first two values. After checking the downloaded file and documentation, replace `StringIO(fixture)` with its local path to inspect these same columns. Times below are **timezone-naive**; do not label them UTC until the dataset's time convention has been verified. This is an initial inspection, not a complete quality-control or energy-production calculation.

```{code-cell} python
from io import StringIO
import numpy as np
import pandas as pd

fixture = """Timestamp,WS11,WD11
2009-11-18 15:00,3.586,159
2009-11-18 15:10,3.437,153
"""
observations = pd.read_csv(StringIO(fixture), usecols=["Timestamp", "WS11", "WD11"])
observations["Timestamp"] = pd.to_datetime(observations["Timestamp"], errors="raise")
if observations["Timestamp"].isna().any() or observations["Timestamp"].duplicated().any():
    raise ValueError("Inspect missing or duplicate timestamps before analysis")
for column in ("WS11", "WD11"):
    observations[column] = pd.to_numeric(observations[column], errors="raise")
valid_speed = np.isfinite(observations["WS11"]) & observations["WS11"].ge(0)
valid_direction = np.isfinite(observations["WD11"]) & observations["WD11"].between(0, 360)
print("Rows:", len(observations), "valid speeds:", int(valid_speed.sum()),
      "valid directions:", int(valid_direction.sum()))
print(observations.head().to_string(index=False))
```

The counts describe the rows present in the file; they do not establish complete time coverage. Check the expected time window, gaps, sensor-specific quality flags and source documentation before interpreting a mean or combining instruments.

### Aggregate the observational series by hour

The following independent example works with the same three columns. It uses equal-weight samples, requires timestamps aligned to a ten-minute grid, and reports six expected positions per complete hour. A complete count is only timestamp and value coverage: it does not certify sensor quality. The first and last hours can be partial, and a time zone must be established from the source before combining these readings with another dataset. To use the downloaded file, replace `StringIO(fixture)` with its verified local path; preserve the original file and check its documentation and quality flags.

```{code-cell} python
from io import StringIO
import numpy as np
import pandas as pd

fixture = """Timestamp,WS11,WD11
2009-11-18 15:00,3.6,350
2009-11-18 15:10,3.4,10
2009-11-18 15:20,3.5,0
2009-11-18 15:30,3.7,360
2009-11-18 15:40,3.8,5
2009-11-18 15:50,3.6,355
2009-11-18 16:00,4.0,90
"""
raw = pd.read_csv(StringIO(fixture), usecols=["Timestamp", "WS11", "WD11"])
raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="raise")
if raw.empty:
    raise ValueError("The file has no observations")
if raw["Timestamp"].isna().any() or raw["Timestamp"].duplicated().any():
    raise ValueError("Inspect missing or duplicate timestamps first")
if not raw["Timestamp"].eq(raw["Timestamp"].dt.floor("10min")).all():
    raise ValueError("This example requires timestamps aligned to a ten-minute grid")
raw = raw.sort_values("Timestamp").set_index("Timestamp")
speed = pd.to_numeric(raw["WS11"], errors="raise")
direction = pd.to_numeric(raw["WD11"], errors="raise")
speed = speed.where(np.isfinite(speed) & speed.ge(0))
direction = direction.where(np.isfinite(direction) & direction.between(0, 360))
angle = np.deg2rad(direction % 360)
sine = np.sin(angle).resample("1h").mean()
cosine = np.cos(angle).resample("1h").mean()
resultant = np.hypot(sine, cosine)
mean_direction = (np.rad2deg(np.arctan2(sine, cosine)) % 360).where(resultant > 1e-8)
mean_direction = mean_direction.mask(np.isclose(mean_direction, 360, atol=1e-10, rtol=0), 0)
hourly = pd.DataFrame({
    "mean_speed_ms": speed.resample("1h").mean(),
    "mean_direction_deg": mean_direction,
    "speed_samples": speed.resample("1h").count(),
    "direction_samples": direction.resample("1h").count(),
})
hourly["speed_complete"] = hourly["speed_samples"].eq(6)
hourly["direction_complete"] = hourly["direction_samples"].eq(6)
print(hourly.head(10).round(2).to_string())
print("Hours:", len(hourly), "complete speed hours:", int(hourly["speed_complete"].sum()),
      "complete direction hours:", int(hourly["direction_complete"].sum()))
```

The first fixture hour has six valid readings and a circular mean of 0°; the second has one reading and is incomplete. The mean speed and mean direction use their own valid samples, so their counts can differ. Undefined direction is shown as missing when the mean resultant is near zero. Hourly values are descriptive means of the available samples, not automatically hourly turbine energy or a complete-quality assessment.

### Compare matched observations and model estimates

This independent example demonstrates the merge that a model-versus-observation comparison needs. Both fixture series represent hourly mean speed in m/s at the same height and use UTC hour-start labels. The incomplete observational hour is missing, not zero. The outer join keeps unmatched hours visible; bias is model minus observation and RMSE uses only paired values. These invented model values are not an evaluation of a real model.

```{code-cell} python
import numpy as np
import pandas as pd

index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
observed = pd.Series([5.0, np.nan, 7.0], index=index, name="observed_ms")
modelled = pd.Series([5.5, 6.0, 6.5], index=index, name="modelled_ms")

def compare_hourly_speeds(observed, modelled):
    """Compare unique, timezone-aware hourly mean speed series in m/s."""
    for label, series in (("observed", observed), ("modelled", modelled)):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{label} must be a Series with a DatetimeIndex")
        if series.index.tz is None or series.index.hasnans or series.index.has_duplicates:
            raise ValueError(f"{label} needs timezone-aware, nonmissing, unique timestamps")
        utc_index = series.index.tz_convert("UTC")
        if not utc_index.equals(utc_index.floor("1h")):
            raise ValueError(f"{label} timestamps must mark UTC hour starts")
    def prepare(series):
        values = pd.to_numeric(series, errors="raise").astype(float)
        if ((~np.isfinite(values) & values.notna()) | (values < 0)).any():
            raise ValueError("Speeds must be non-negative and finite, or missing")
        values.index = values.index.tz_convert("UTC")
        return values
    pairs = pd.concat([prepare(observed).rename("observed_ms"),
                       prepare(modelled).rename("modelled_ms")], axis=1).sort_index()
    valid = pairs.dropna(subset=["observed_ms", "modelled_ms"])
    delta = valid["modelled_ms"] - valid["observed_ms"]
    metrics = {"observed_hours": int(pairs["observed_ms"].count()),
               "modelled_hours": int(pairs["modelled_ms"].count()),
               "paired_hours": len(valid),
               "bias_ms": delta.mean() if len(delta) else np.nan,
               "rmse_ms": np.sqrt((delta ** 2).mean()) if len(delta) else np.nan}
    return pairs, metrics

pairs, metrics = compare_hourly_speeds(observed, modelled)
print(pairs.to_string())
print(pd.Series(metrics).to_string())
```

The example has two paired hours, zero bias and 0.5 m/s RMSE. For the preceding observations, only use `hourly["mean_speed_ms"]` where `hourly["speed_complete"]` is true; attach a time zone only after verifying the source convention. Obtain a model series separately, confirm its height, units, location, averaging period and timestamp semantics, and convert both indexes to the same time basis. A grid-cell estimate and a point sensor need not represent identical wind, and these matched samples alone do not establish complete-period or site-wide model performance.

For a reproducibility check, the Zenodo text file retrieved on 21 November 2026 had the record's MD5 `cf2d9df9b29aa6a92c68bce03c52f660`. Running this inspection on that file returned **269,197 rows** and **172,003 valid values** for each selected sensor column, spanning recorded timestamps from 18 November 2009 15:00 to 1 January 2015 01:00. These are nonmissing, finite and range-valid counts, not independently quality-controlled observations or a complete-coverage estimate; future file versions may differ.


The following self-contained fixture demonstrates the date conversion and a circular hourly direction. It uses equal-weight synthetic samples; it is not a complete replacement for the external reader, its timezone handling or its metadata. Missing or invalid directions are excluded, and opposing directions have no defined mean. Check timezones, coverage, sample durations and the provider’s data definitions before adapting it.

```{code-cell} python
import numpy as np
import pandas as pd

samples = pd.DataFrame({
    "date": ["20040101"] * 4, "time": ["0000", "0030", "0100", "0130"],
    "direction_deg": [350., 10., 0., 180.],
})
# The fixture is explicitly UTC; do not assume UTC for an undocumented file.
date_time = samples["date"].astype(str) + " " + samples["time"].astype(str).str.zfill(4)
samples["timestamp"] = pd.to_datetime(date_time, format="%Y%m%d %H%M", utc=True, errors="raise")
valid = samples["direction_deg"].where(samples["direction_deg"].between(0, 360, inclusive="both"))
radians = np.deg2rad(valid % 360)
vectors = pd.DataFrame({"sin": np.sin(radians), "cos": np.cos(radians)}, index=samples.index)
vectors.index = samples["timestamp"]
hourly = vectors.resample("1h").mean()
count = vectors["sin"].resample("1h").count()
resultant = np.hypot(hourly["sin"], hourly["cos"])
direction = (np.rad2deg(np.arctan2(hourly["sin"], hourly["cos"])) % 360).where(resultant > 1e-8)
direction = direction.mask(np.isclose(direction, 360, atol=1e-10, rtol=0), 0)
print(pd.DataFrame({"direction_deg": direction, "valid_samples": count}).to_string())
assert np.isclose(direction.iloc[0], 0) and pd.isna(direction.iloc[1])
```

The first hour has a mean direction of 0°; the second is undefined because the opposing samples cancel. The counts report valid direction samples, not complete hourly coverage or observed energy.

For an xarray dataset with the external example's `M` (speed) and `Dir` (direction) names, aggregate each variable according to its meaning rather than taking an arithmetic mean of the entire dataset. The following independent fixture assumes equal-duration samples at UTC times; its timezone-naive coordinates represent UTC only because that convention is explicitly supplied here. It does not validate the external archive or replace its reader.

```{code-cell} python
import numpy as np
import pandas as pd
import xarray as xr

wind = xr.Dataset(
    {"M": ("time", [4., 6., 5., np.nan, 7., 8.]),
     "Dir": ("time", [350., 10., 0., 180., np.nan, 400.])},
    coords={"time": pd.date_range("2024-01-01", periods=6, freq="30min")},
)
valid_speed = wind["M"].where(np.isfinite(wind["M"]) & (wind["M"] >= 0))
valid_direction = wind["Dir"].where(
    np.isfinite(wind["Dir"]) & (wind["Dir"] >= 0) & (wind["Dir"] <= 360)
)
angle = np.deg2rad(valid_direction % 360)
sin_mean = np.sin(angle).resample(time="1h").mean(skipna=True)
cos_mean = np.cos(angle).resample(time="1h").mean(skipna=True)
resultant = np.hypot(sin_mean, cos_mean)
mean_direction = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360).where(resultant > 1e-8)
mean_direction = xr.where(np.isclose(mean_direction, 360, atol=1e-10, rtol=0), 0, mean_direction)
hourly_wind = xr.Dataset({
    "mean_speed_m_s": valid_speed.resample(time="1h").mean(skipna=True),
    "mean_direction_deg": mean_direction,
    "valid_speed_samples": valid_speed.resample(time="1h").count(),
    "valid_direction_samples": valid_direction.resample(time="1h").count(),
})
print(hourly_wind)
assert np.isclose(hourly_wind.mean_direction_deg.values[0], 0)
assert np.isnan(hourly_wind.mean_direction_deg.values[1:]).all()
assert hourly_wind.valid_speed_samples.values.tolist() == [2, 1, 2]
assert hourly_wind.valid_direction_samples.values.tolist() == [2, 2, 0]
```

The second hour's opposite directions cancel; the third has no valid directions, although both speed samples are valid. The sample counts are not proof of complete coverage, and a direction-only circular mean is not a speed-weighted mean velocity. Real files require verified coordinate times, sample durations, units, variable definitions and a documented missing-data policy.

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

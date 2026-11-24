---
kernelspec:
  name: python3
  display_name: Python 3
---

# Wind Data Lab: Direction, Quality, and Time Series

“Eolic” and “wind” refer to the same energy source. This second wind quiz concentrates on data-processing skills that are easy to miss in a purely physical power-curve exercise.

## Setup: a reproducible teaching dataset

The dataset below is synthetic. It deliberately contains one missing speed, one impossible negative speed, and one direction outside the conventional range so that validation is part of the exercise.

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(24)
index = pd.date_range("2025-01-01", periods=24 * 30, freq="h")
wind = pd.DataFrame({
    "speed_ms": rng.weibull(2.1, len(index)) * 8.2,
    "direction_deg": (235 + rng.normal(0, 45, len(index))) % 360,
}, index=index)
wind.iloc[20, wind.columns.get_loc("speed_ms")] = np.nan
wind.iloc[80, wind.columns.get_loc("speed_ms")] = -2
wind.iloc[120, wind.columns.get_loc("direction_deg")] = 370
wind.head()
```

**4 Easy · 8 Medium · 8 Hard.** Try each question before opening its solution. Run the dataset setup cell first. In the sandbox, **Include preceding examples** runs earlier non-solution cells; it does not run the answers hidden in dropdowns. To use the supplied answers, run Quiz 5 before Quizzes 6, 7 or 13; run Quizzes 5 and 7 before Quiz 8; and run Quizzes 5, 7 and 13 before Quiz 14. Quiz 15 uses the turbine function from Quiz 13, so run Quizzes 5 and 13 first. You may instead provide equivalent working code yourself.

In the sandbox, after opening a code answer, **Include preceding practice answers** loads the earlier runnable examples and answers plus that answer into the editor. Review the code and select **Run Python**; loading it does not run it automatically.

:::{admonition} Quiz 1 — Quality profile
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Report row count, missingness, duplicate timestamps, negative speeds, and invalid directions. Do not repair the data yet.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
quality = pd.Series({
    "rows": len(wind),
    "duplicate_timestamps": wind.index.duplicated().sum(),
    "missing_speed": wind["speed_ms"].isna().sum(),
    "negative_speed": wind["speed_ms"].lt(0).sum(),
    "missing_direction": wind["direction_deg"].isna().sum(),
    "invalid_direction": (wind["direction_deg"].notna() & ~wind["direction_deg"].between(0, 360, inclusive="left")).sum(),
})
print(quality)
assert quality["negative_speed"] == 1 and quality["invalid_direction"] == 1
```
:::

:::{admonition} Quiz 2 — Convert anemometer reports
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@manwell2009; @pandasDocs].

Write a function converting knots to m/s using 1 knot = 1852 m/h. Verify 10 knots.

:::

:::{admonition} Solution
:class: tip, dropdown

Convert all speed records to the same units before comparing them.

```{code-cell} python
from math import isclose
def knots_to_ms(knots):
    return knots*1852/3600

print(f'10 knots = {knots_to_ms(10):.4f} m/s')
assert isclose(knots_to_ms(10),5.1444444444,rel_tol=1e-9)
```
:::

:::{admonition} Quiz 3 — Wrap a direction
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@manwell2009; @pandasDocs].

Write a function mapping a signed direction to [0,360). Check −10°, 360° and 370°.

:::

:::{admonition} Solution
:class: tip, dropdown

Wrapping a known angle is different from repairing an unknown or missing record.

```{code-cell} python
def wrap(angle):
    return angle%360

print('Wrapped directions (degrees):', [wrap(v) for v in [-10,360,370]])
assert [wrap(v) for v in [-10,360,370]]==[350,0,10]
```
:::

:::{admonition} Quiz 4 — Translate sampling cadence
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@manwell2009; @pandasDocs].

Write a function converting minutes to hours and multiply by 24 samples of 15 minutes. What observation duration is represented?

:::

:::{admonition} Solution
:class: tip, dropdown

The example assumes each sample is an interval mean with a known duration.

```{code-cell} python
def minutes_to_hours(minutes):
    return minutes/60

print('Observation duration (hours):', 24*minutes_to_hours(15))
assert 24*minutes_to_hours(15)==6
```
:::

:::{admonition} Quiz 5 — Repair with explicit rules
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Replace impossible negative or non-finite speed with missing, wrap finite directions with modulo 360, and interpolate interior speed gaps of at most two missing hours on a regular hourly index. Preserve missing directions and the original table.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
clean = wind.copy()
if not clean.index.is_monotonic_increasing or not clean.index.is_unique or not clean.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
    raise ValueError("This two-row gap rule requires a unique regular hourly index")
clean.loc[~np.isfinite(clean["speed_ms"]) | (clean["speed_ms"] < 0), "speed_ms"] = np.nan
clean.loc[~np.isfinite(clean["direction_deg"]), "direction_deg"] = np.nan
clean["direction_deg"] = clean["direction_deg"] % 360
missing = clean["speed_ms"].isna()
groups = missing.ne(missing.shift()).cumsum()
short_gaps = missing & missing.groupby(groups).transform("size").le(2)
filled = clean["speed_ms"].interpolate(method="time", limit_area="inside")
clean.loc[short_gaps, "speed_ms"] = filled[short_gaps]
print("Remaining missing speeds:", clean["speed_ms"].isna().sum())
assert clean["speed_ms"].dropna().ge(0).all()
assert clean["direction_deg"].dropna().between(0, 360, inclusive="left").all()
```
:::

:::{admonition} Quiz 6 — Circular mean
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Write `circular_mean_deg`. Confirm that the mean of 350° and 10° is north, not south. Then calculate the cleaned dataset's mean direction.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def circular_mean_deg(direction):
    values = np.asarray(direction, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("directions must be nonempty and finite")
    radians = np.deg2rad(values)
    if np.hypot(np.sin(radians).mean(), np.cos(radians).mean()) < 1e-12:
        raise ValueError("mean direction is undefined for cancelling vectors")
    angle = np.rad2deg(np.arctan2(np.sin(radians).mean(),
                                  np.cos(radians).mean()))
    return np.round(angle % 360, 12) % 360

assert np.isclose(circular_mean_deg([350, 10]), 0)
valid_direction = clean["direction_deg"].dropna()
print(f"Direction coverage: {len(valid_direction)}/{len(clean)} observations")
print(f"Circular mean direction: {circular_mean_deg(valid_direction):.1f}°")
```
:::

:::{admonition} Quiz 7 — Direction sectors
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Assign each finite direction to one of eight named compass sectors using a function. Use sectors centered on the eight compass points; an exact boundary belongs to the next sector clockwise, so 22.5° is NE and 337.5° is N. Report missing-direction coverage, counts and percentages of valid directions; the unrounded percentages should sum to 100% when any valid directions are present.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
sector_names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

def direction_sector(direction_deg):
    direction = np.asarray(direction_deg, dtype=float)
    if not np.isfinite(direction).all():
        raise ValueError("Missing or non-finite directions must be handled before sector assignment")
    direction = direction % 360
    index = ((direction + 22.5) // 45).astype(int) % 8
    return sector_names[index]

valid_direction = np.isfinite(clean["direction_deg"].to_numpy(dtype=float))
valid_count = int(valid_direction.sum())
print(f"Finite-direction coverage: {valid_count}/{len(clean)} observations; percentages use the valid directions")
if valid_count == 0:
    raise ValueError("No finite directions available for sector percentages")
clean["sector"] = pd.Series(pd.NA, index=clean.index, dtype="object")
clean.loc[valid_direction, "sector"] = direction_sector(clean.loc[valid_direction, "direction_deg"])
sector_counts = clean["sector"].value_counts().reindex(sector_names, fill_value=0)
sector_share = sector_counts / valid_count
assert sector_counts.sum() == valid_count and np.isclose(sector_share.sum(), 1)
print(pd.DataFrame({"count": sector_counts, "percent_valid": (100 * sector_share).round(1)}).to_string())
```
:::

:::{admonition} Quiz 8 — Frequency wind rose
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Create a polar bar chart of sector frequency. Use meteorological orientation: north at the top and clockwise rotation.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
angles = np.deg2rad(np.arange(0, 360, 45))
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
ax.bar(angles, sector_share.to_numpy(), width=np.deg2rad(40), color="skyblue", edgecolor="navy")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(angles, sector_names)
ax.set_ylabel("Fraction of valid directions", labelpad=35)
ax.set_title("Wind-direction frequency")
plt.tight_layout()
```
:::

:::{admonition} Quiz 9 — Build an hourly timestamp validator
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@manwell2009; @pandasDocs].

For timestamps 00:00, 01:00 and 03:00 UTC on the same day, write a function returning missing hours within the inclusive 00:00–03:00 UTC window. Reject duplicate or off-grid times and require timezone-aware, on-the-hour window endpoints.

:::

:::{admonition} Solution
:class: tip, dropdown

Gaps must be identified before interpreting a row count as hours.

```{code-cell} python
import pandas as pd
def missing_hours(index, start, end):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError('Provide timezone-aware timestamps')
    if not index.is_unique or index.isna().any():
        raise ValueError('Timestamps must be unique and non-missing')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start.tz is None or end.tz is None:
        raise ValueError('Specify a timezone-aware on-the-hour expected window')
    start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
    if start != start.floor('h') or end != end.floor('h') or start > end:
        raise ValueError('Specify a timezone-aware on-the-hour expected window')
    index = index.tz_convert('UTC')
    expected=pd.date_range(start,end,freq='h')
    if not index.difference(expected).empty:
        raise ValueError('Observations fall outside the expected hourly grid')
    return expected.difference(index)

t=pd.to_datetime(['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 03:00'],utc=True)
missing=missing_hours(t, pd.Timestamp('2025-01-01 00:00', tz='UTC'),
                      pd.Timestamp('2025-01-01 03:00', tz='UTC'))
print('Missing timestamps:', missing.tolist())
assert list(missing)==[pd.Timestamp('2025-01-01 02:00',tz='UTC')]
```
:::

:::{admonition} Quiz 10 — Flag physical and instrument limits
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@manwell2009; @pandasDocs].

For speeds [0,−1,missing,10,65] m/s, keep valid zero, separate negative from missing, and flag speeds above a 60 m/s instrument limit. Return reason labels rather than overwriting the observations.

:::

:::{admonition} Solution
:class: tip, dropdown

Validity rules belong to the instrument and dataset; zero can be a valid calm observation.

```{code-cell} python
import numpy as np
import pandas as pd
def status(value):
    if pd.isna(value) or not np.isfinite(value):
        return 'missing or nonfinite'
    if value<0:
        return 'negative'
    if value>60:
        return 'out of range'
    return 'valid'

print('Quality labels:', [status(v) for v in [0,-1,np.nan,10,65]])
assert [status(v) for v in [0,-1,np.nan,10,65]]==['valid','negative','missing or nonfinite','valid','out of range']
assert status(pd.NA) == 'missing or nonfinite'
```
:::

:::{admonition} Quiz 11 — Find energy in subhourly meter data
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@manwell2009; @pandasDocs].

Given half-hour interval-mean meter powers [0.2,0.8,1.2,0.6] MW, calculate energy and mean power with a reusable function. Show that summing power alone would give a misleading energy value.

:::

:::{admonition} Solution
:class: tip, dropdown

Power and energy have different dimensions.

```{code-cell} python
def interval_summary(samples,minutes):
    values=np.asarray(samples,dtype=float)
    if values.ndim!=1 or values.size==0 or not np.isfinite(values).all() or not np.isfinite(minutes) or minutes<=0:
        raise ValueError('Provide finite readings and a positive interval duration')
    dt=minutes/60
    return values.sum()*dt,values.mean()

print('Energy MWh, mean MW:', interval_summary([.2,.8,1.2,.6],30), '; unweighted power sum (MW):', sum([.2,.8,1.2,.6]))
assert all(abs(a-b)<1e-9 for a,b in zip(interval_summary([.2,.8,1.2,.6],30),[1.4,.7]))
```
:::

:::{admonition} Quiz 12 — Count wind sectors on bin edges
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@manwell2009; @pandasDocs].

Use N, E, S and W quadrants with N centred at 0°. Assign exact boundary observations [315,44.9,45,135,225] with left-inclusive bins and report counts.

:::

:::{admonition} Solution
:class: tip, dropdown

The edge rule makes each direction belong to exactly one group.

```{code-cell} python
import numpy as np
v=np.array([315,44.9,45,135,225])
sector=(((v+45)%360)//90).astype(int)
print('N/E/S/W counts:', np.bincount(sector,minlength=4).tolist())
assert np.bincount(sector,minlength=4).tolist()==[2,1,1,1]
```
:::

:::{admonition} Quiz 13 — Add a turbine function and energy
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Write a simplified 3 MW curve, calculate hourly power and energy, and assert that no power lies outside 0–3 MW. Treat each valid wind speed as constant for its one-hour interval; this is a model assumption, not a measured generation series. If changing the curve parameters, use a finite non-negative rating and finite speeds satisfying 0 ≤ cut-in < rated speed < cut-out.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def turbine_power_mw(speed, rated=3.0, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    parameters = np.asarray([rated, cut_in, rated_speed, cut_out], dtype=float)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError("Curve parameters must be finite scalars")
    if rated < 0 or not 0 <= cut_in < rated_speed < cut_out:
        raise ValueError("Use a non-negative rating and 0 <= cut-in < rated speed < cut-out")
    speed = np.asarray(speed, dtype=float)
    valid = np.isfinite(speed) & (speed >= 0)
    power = np.where(valid, 0.0, np.nan)
    ramp = valid & (speed >= cut_in) & (speed < rated_speed)
    power[ramp] = rated * (speed[ramp]**3 - cut_in**3) / (rated_speed**3 - cut_in**3)
    power[valid & (speed >= rated_speed) & (speed < cut_out)] = rated
    return power

if not clean.index.equals(wind.index):
    raise ValueError("Clean and original wind records must have matching indexes")
clean["power_mw"] = turbine_power_mw(clean["speed_ms"])
monthly_energy_mwh = clean["power_mw"].sum(min_count=1)
valid_hours = clean["power_mw"].notna().sum()
original_valid = np.isfinite(wind["speed_ms"]) & wind["speed_ms"].ge(0)
repaired_hours = (clean["power_mw"].notna() & ~original_valid).sum()
assert clean["power_mw"].dropna().between(0, 3).all()
energy_label = "unavailable" if pd.isna(monthly_energy_mwh) else f"{monthly_energy_mwh:.1f} MWh"
print(f"Modelled energy from synthetic wind: {energy_label}; valid modelled intervals {valid_hours}/{len(clean)}")
print(f"Valid original speed intervals: {original_valid.sum()}/{len(wind)}; repaired intervals used in the model: {repaired_hours}")
```
:::

:::{admonition} Quiz 14 — Energy-weighted wind rose
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Sum energy by sector and plot a second polar chart. Which directions matter most for energy, and why can this differ from frequency?

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
# Each valid power represents one hour, so each MW value contributes MWh.
groups = clean.groupby("sector")["power_mw"]
sector_rows = groups.size().reindex(sector_names, fill_value=0)
sector_valid = groups.count().reindex(sector_names, fill_value=0)
energy_by_sector = groups.sum(min_count=1).reindex(sector_names)
sector_energy_report = pd.DataFrame({"direction_rows": sector_rows, "valid_power_rows": sector_valid,
                                     "modelled_energy_mwh": energy_by_sector})
print(sector_energy_report.to_string())
print("Missing direction rows:", clean["sector"].isna().sum(),
      "of", len(clean), "(excluded from sector energy)")
if energy_by_sector.notna().any():
    maximum = energy_by_sector.max()
    leaders = energy_by_sector.index[energy_by_sector.eq(maximum)].tolist()
    print("Leading sector(s) for modelled energy:", ", ".join(leaders), f"({maximum:.3f} MWh)")
else:
    print("No sector has valid modelled power; a leading sector is unavailable.")
angles = np.deg2rad(np.arange(0, 360, 45))
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
plotted = energy_by_sector.notna().to_numpy()
ax.bar(angles[plotted], energy_by_sector.to_numpy()[plotted], width=np.deg2rad(40), color="seagreen", edgecolor="darkgreen")
if not plotted.all():
    ax.text(.5, -.14, "Sectors without valid power are omitted; see table.",
            transform=ax.transAxes, ha="center", fontsize=9)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(angles, sector_names)
ax.set_ylabel("Modelled energy over valid one-hour intervals (MWh)", labelpad=35)
ax.set_title("Energy contribution by incoming wind direction")
plt.tight_layout()
```

Frequent low-speed winds may contribute less energy than less-frequent high-speed winds because the power curve is nonlinear. Sector totals use modelled power from the cleaned synthetic wind, including any repaired speeds, and cover only intervals with both a direction and valid power; absent or entirely missing sectors do not establish zero energy.
:::

:::{admonition} Quiz 15 — Chronological baseline
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Check that the original timestamps cover 30 consecutive complete days starting at midnight, with unique, ordered, on-the-hour readings. Split the hourly speed series into the first 23 training days and last 7 test days before filling gaps. In training, forward-fill only gaps of at most two consecutive missing hours (leaving gaps of three or more hours unfilled); leave test targets unfilled. Convert to power and aggregate complete 24-hour days. Forecast each test day with the training mean, calculate MAE, and plot modelled energy and forecast. Do not shuffle. The fill rule is a teaching assumption, not proof of accuracy.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
if (not isinstance(wind.index, pd.DatetimeIndex) or wind.index.empty
        or wind.index.hasnans or not wind.index.is_unique
        or not wind.index.is_monotonic_increasing
        or not wind.index.equals(wind.index.floor("h"))):
    raise ValueError("Use a non-empty, ordered, unique on-the-hour time index")
start = wind.index.min().normalize()
expected_index = pd.date_range(start, periods=30 * 24, freq="h")
if not wind.index.equals(expected_index):
    raise ValueError("This exercise requires 30 complete consecutive days starting at midnight")
cutoff = start + pd.Timedelta(days=23)
raw_speed = wind["speed_ms"].where(np.isfinite(wind["speed_ms"]) & wind["speed_ms"].ge(0))
train_raw = raw_speed.loc[raw_speed.index < cutoff]
train_missing = train_raw.isna()
train_groups = train_missing.ne(train_missing.shift()).cumsum()
short_train_gaps = train_missing & train_missing.groupby(train_groups).transform("size").le(2)
training_speed = train_raw.where(~short_train_gaps, train_raw.ffill(limit=2))
test_speed = raw_speed.loc[raw_speed.index >= cutoff]  # never impute evaluation targets
forecast_input = pd.concat([training_speed, test_speed])
power_for_forecast = pd.Series(turbine_power_mw(forecast_input), index=forecast_input.index)
daily_groups = power_for_forecast.resample("D")
daily_counts = daily_groups.count()
daily_energy = daily_groups.sum(min_count=1).where(daily_counts.eq(24))
daily_coverage = pd.DataFrame({"valid_hours": daily_counts, "expected_hours": 24})
print("Daily power coverage (hours):")
print(daily_coverage.to_string())
if daily_energy.isna().any():
    raise ValueError("Each modelled day needs 24 valid hourly observations before this baseline")
train = daily_energy.loc[daily_energy.index < cutoff]
test = daily_energy.loc[daily_energy.index >= cutoff]
forecast = pd.Series(train.mean(), index=test.index)
mae = (test - forecast).abs().mean()

plt.figure(figsize=(9, 4))
plt.plot(daily_energy.index, daily_energy, marker="o", label="Modelled energy from synthetic wind")
plt.plot(forecast.index, forecast, marker="s", label="Training-mean forecast")
plt.axvline(test.index.min(), color="red", linestyle="--", label="Test starts")
plt.xlabel("Date")
plt.ylabel("Daily energy (MWh)")
plt.title(f"Chronological baseline; MAE = {mae:.1f} MWh")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert train.index.max() < test.index.min()
```
:::

:::{admonition} Quiz 16 — Validate and merge wind and meter files
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@manwell2009; @pandasDocs].

Join hourly UTC wind speeds [2,6,8] m/s at hours [0,1,2] and meter powers [1,2,3] MW at hours [1,2,3]. Each meter power is the average over its one-hour interval. Keep unmatched records, require unique nonmissing UTC hour-start timestamps, and using only intervals where both values are valid, compute the matched mean speed across all usable pairs and the observed meter MWh for wind above 5 m/s. Report timestamp matches and usable pairs separately.

:::

:::{admonition} Solution
:class: tip, dropdown

Report timestamp matches and usable-pair coverage before making power-versus-speed comparisons. With no usable pairs (or no paired wind above 5 m/s for meter energy), the corresponding metric is unavailable (NaN).

```{code-cell} python
import numpy as np
import pandas as pd
t=pd.date_range('2025-02-01',periods=4,freq='h',tz='UTC')
v=pd.DataFrame({'time':t[:3],'speed':[2,6,8]})
p=pd.DataFrame({'time':t[1:],'mw':[1,2,3]})
def prepare_hourly(table, column):
    times = pd.DatetimeIndex(table['time'])
    if (str(times.tz) != 'UTC' or times.hasnans or not times.is_unique
            or not times.equals(times.floor('h'))):
        raise ValueError('Use unique, nonmissing UTC hour-start timestamps')
    values = pd.to_numeric(table[column], errors='raise').astype(float)
    if np.isinf(values.to_numpy()).any() or (values.dropna() < 0).any():
        raise ValueError('Values must be non-negative and finite, or missing')
    result = table.copy()
    result['time'], result[column] = times, values
    return result
joined=prepare_hourly(v,'speed').merge(prepare_hourly(p,'mw'),on='time',how='outer',validate='one_to_one',indicator=True)
matched=joined[joined['_merge']=='both']
paired=matched.dropna(subset=['speed','mw'])
mean_speed = paired.speed.mean() if len(paired) else np.nan
meter_energy = paired.loc[paired.speed>5,'mw'].sum(min_count=1) if len(paired) else np.nan
print('Matched mean speed (m/s):', mean_speed, '; matched meter energy (MWh):', meter_energy)
print(f"Matched coverage: {len(matched)}/{len(v)} speed rows, {len(matched)}/{len(p)} meter rows, or {len(matched)}/{len(joined)} union timestamps; usable pairs: {len(paired)}")
assert len(joined)==4 and len(matched)==2
assert mean_speed==7 and meter_energy==3
```
:::

:::{admonition} Quiz 17 — Protect long and edge gaps
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@manwell2009; @pandasDocs].

Use hourly speeds [missing,5,missing,7,missing,missing,10,missing]. Fill only one-hour gaps with two valid neighbours. Test that neither edge gaps nor a longer gap gets partially filled.

:::

:::{admonition} Solution
:class: tip, dropdown

A two-hour gap stays missing in its entirety.

```{code-cell} python
import numpy as np
import pandas as pd
s=pd.Series([np.nan,5,np.nan,7,np.nan,np.nan,10,np.nan])
eligible=s.isna()&s.shift().notna()&s.shift(-1).notna()
gap_filled=s.mask(eligible,(s.shift()+s.shift(-1))/2)
print(gap_filled)
assert gap_filled.iloc[2]==6 and gap_filled.isna().sum()==4
```
:::

:::{admonition} Quiz 18 — Reconcile sector energy with unequal durations
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@manwell2009; @pandasDocs].

For directions [350,10,90,180]°, powers [1,2,3,4] MW and durations [0.5,1,2,1] h, use N/E/S/W quadrants. Compare count share and energy share and reconcile the summed energy.

:::

:::{admonition} Solution
:class: tip, dropdown

Count share and energy share need not identify the same predominant direction.

```{code-cell} python
import numpy as np
import pandas as pd
direction=np.array([350,10,90,180])
p=np.array([1,2,3,4]); dt=np.array([.5,1,2,1])
sector=(((direction+45)%360)//90).astype(int)
count=np.bincount(sector,minlength=4)
energy=np.bincount(sector,weights=p*dt,minlength=4)
count_share=100*count/count.sum()
energy_share=100*energy/energy.sum()
print(pd.DataFrame({'count':count,'count_pct':count_share,'energy_mwh':energy,'energy_pct':energy_share}, index=['N','E','S','W']).to_string())
assert energy.sum()==np.dot(p,dt)==12.5
assert np.allclose(count_share,[50,25,25,0]) and np.allclose(energy_share,[20,48,32,0])
assert count.argmax()==0 and energy.argmax()==1
```
:::

:::{admonition} Quiz 19 — Compare circular and speed-weighted direction
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@manwell2009; @pandasDocs].

For winds from [350,10,90]° with speeds [1,1,5] m/s, compute the equal-weight circular mean and the meteorological from-bearing of the mean horizontal velocity. Also report the opposite velocity-to bearing. Use meteorological from-direction signs and verify the stronger easterly shifts the velocity-based result.

:::

:::{admonition} Solution
:class: tip, dropdown

Wind reported from the east contributes westward velocity.

```{code-cell} python
import numpy as np
angle=np.deg2rad([350,10,90]); speed=np.array([1,1,5])
unweighted=np.rad2deg(np.arctan2(np.sin(angle).mean(),np.cos(angle).mean()))%360
east=np.mean(-speed*np.sin(angle)); north=np.mean(-speed*np.cos(angle))
weighted=np.rad2deg(np.arctan2(-east,-north))%360
velocity_to=(weighted+180)%360
print(f"Circular from-bearing: {unweighted:.2f}°; mean velocity from: {weighted:.2f}°; velocity towards: {velocity_to:.2f}°")
assert 0<unweighted<weighted<90 and east<0
```
:::

:::{admonition} Quiz 20 — Estimate coverage versus modelled energy
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@manwell2009; @pandasDocs].

A turbine reports MW [0,missing,2,missing,4] while a speed-based model estimates [0,1,3,4,4]. Each reported or modelled power is the average over its one-hour interval. Calculate measured energy, model bias over concurrent observations, and modeled energy for the gaps separately. Do not convert missing reports into measurements. Require finite, non-negative model values and non-negative observations where present; if all observations are missing, measured energy and paired bias are unavailable.

:::

:::{admonition} Solution
:class: tip, dropdown

Bias is based on paired observations; the gap model has a different evidence status than measurements.

```{code-cell} python
import numpy as np
observed=np.array([0,np.nan,2,np.nan,4]); model=np.array([0,1,3,4,4])
if (observed.ndim != 1 or observed.size == 0 or model.shape != observed.shape
        or np.isinf(observed).any() or np.any(observed[np.isfinite(observed)] < 0)
        or not np.isfinite(model).all() or np.any(model < 0)):
    raise ValueError("Use aligned non-negative observations or NaN and finite non-negative model values")
matched=np.isfinite(observed)
actual=observed[matched].sum() if matched.any() else np.nan
bias=(model[matched]-observed[matched]).mean() if matched.any() else np.nan
gap_estimate=model[~matched].sum()
actual_label = f"{actual:g} MWh" if matched.any() else "unavailable"
bias_label = f"{bias:.3f} MW" if matched.any() else "unavailable"
print(f"Measured energy: {actual_label}; observed coverage: {matched.mean():.1%}; paired model bias: {bias_label}; modelled gap energy: {gap_estimate} MWh")
assert actual==6 and matched.mean()==.6 and gap_estimate==5
assert np.isclose(bias,1/3)
```
:::

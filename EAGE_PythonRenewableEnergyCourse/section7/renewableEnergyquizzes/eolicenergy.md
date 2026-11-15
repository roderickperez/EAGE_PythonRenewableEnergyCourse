---
kernelspec:
  name: python3
  display_name: Python 3
---

# Wind Data Lab: Direction, Quality, and Time Series

“Eolic” and “wind” refer to the same energy source. This second wind assessment concentrates on data-processing skills that are easy to miss in a purely physical power-curve exercise.

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

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its solution. Where a prompt refers to an earlier answer, run that code first.

## Basic

:::{admonition} Quiz 1 — Quality profile
:class: note

**Difficulty:** Basic

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
    "invalid_direction": (~wind["direction_deg"].between(0, 360, inclusive="left")).sum(),
})
print(quality)
assert quality["negative_speed"] == 1 and quality["invalid_direction"] == 1
```
:::

:::{admonition} Quiz 2 — Convert anemometer reports
:class: note

**Difficulty:** Basic

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

assert isclose(knots_to_ms(10),5.1444444444,rel_tol=1e-9)
```
:::

:::{admonition} Quiz 3 — Wrap a direction
:class: note

**Difficulty:** Basic

**Reference:** [@manwell2009; @pandasDocs].

Write a function mapping a signed direction to [0,360). Check −10°, 360° and 370°.

:::

:::{admonition} Solution
:class: tip, dropdown

Wrapping a known angle is different from repairing an unknown or missing record.

```{code-cell} python
def wrap(angle):
    return angle%360

assert [wrap(v) for v in [-10,360,370]]==[350,0,10]
```
:::

:::{admonition} Quiz 4 — Translate sampling cadence
:class: note

**Difficulty:** Basic

**Reference:** [@manwell2009; @pandasDocs].

Write a function converting minutes to hours and multiply by 24 samples of 15 minutes. What observation duration is represented?

:::

:::{admonition} Solution
:class: tip, dropdown

The example assumes each sample is an interval mean with a known duration.

```{code-cell} python
def minutes_to_hours(minutes):
    return minutes/60

assert 24*minutes_to_hours(15)==6
```
:::

## Medium

:::{admonition} Quiz 5 — Repair with explicit rules
:class: note

**Difficulty:** Medium

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Replace impossible negative speed with missing, wrap direction with modulo 360, and interpolate only short speed gaps in time. Preserve the original table.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
clean = wind.copy()
clean.loc[clean["speed_ms"] < 0, "speed_ms"] = np.nan
clean["direction_deg"] = clean["direction_deg"] % 360
missing = clean["speed_ms"].isna()
groups = missing.ne(missing.shift()).cumsum()
short_gaps = missing & missing.groupby(groups).transform("size").le(2)
filled = clean["speed_ms"].interpolate(method="time", limit_area="inside")
clean.loc[short_gaps, "speed_ms"] = filled[short_gaps]
assert clean["speed_ms"].notna().all()
assert clean["speed_ms"].ge(0).all()
assert clean["direction_deg"].between(0, 360, inclusive="left").all()
```
:::

:::{admonition} Quiz 6 — Circular mean
:class: note

**Difficulty:** Medium

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
print(f"Circular mean direction: {circular_mean_deg(clean['direction_deg']):.1f}°")
```
:::

:::{admonition} Quiz 7 — Direction sectors
:class: note

**Difficulty:** Medium

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Assign every observation to one of eight named compass sectors using a function. Produce counts and percentages that sum to 100%.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
sector_names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

def direction_sector(direction_deg):
    direction = np.asarray(direction_deg, dtype=float) % 360
    index = ((direction + 22.5) // 45).astype(int) % 8
    return sector_names[index]

clean["sector"] = direction_sector(clean["direction_deg"])
sector_share = clean["sector"].value_counts(normalize=True).reindex(sector_names, fill_value=0)
assert np.isclose(sector_share.sum(), 1)
print((sector_share * 100).round(1))
```
:::

:::{admonition} Quiz 8 — Frequency wind rose
:class: note

**Difficulty:** Medium

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
ax.set_title("Wind-direction frequency")
plt.tight_layout()
```
:::

:::{admonition} Quiz 9 — Build an hourly timestamp validator
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009; @pandasDocs].

For timestamps 00:00, 01:00 and 03:00 UTC on the same day, write a function returning missing hours. Refuse duplicate times before looking for gaps.

:::

:::{admonition} Solution
:class: tip, dropdown

Gaps must be identified before interpreting a row count as hours.

```{code-cell} python
import pandas as pd
def missing_hours(index):
    if not index.is_unique:
        raise ValueError('Duplicate timestamps')
    return pd.date_range(index.min(),index.max(),freq='h').difference(index)

t=pd.to_datetime(['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 03:00'],utc=True)
assert list(missing_hours(t))==[pd.Timestamp('2025-01-01 02:00',tz='UTC')]
```
:::

:::{admonition} Quiz 10 — Flag physical and instrument limits
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009; @pandasDocs].

For speeds [0,−1,missing,10,65] m/s, keep valid zero, separate negative from missing, and flag speeds above a 60 m/s instrument limit. Return reason labels rather than overwriting the observations.

:::

:::{admonition} Solution
:class: tip, dropdown

Validity rules belong to the instrument and dataset; zero can be a valid calm observation.

```{code-cell} python
import numpy as np
def status(value):
    if not np.isfinite(value):
        return 'missing or nonfinite'
    if value<0:
        return 'negative'
    if value>60:
        return 'out of range'
    return 'valid'

assert [status(v) for v in [0,-1,np.nan,10,65]]==['valid','negative','missing or nonfinite','valid','out of range']
```
:::

:::{admonition} Quiz 11 — Find energy in subhourly meter data
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009; @pandasDocs].

Given half-hour meter powers [0.2,0.8,1.2,0.6] MW, calculate energy and mean power with a reusable function. Show that summing power alone would give a misleading energy value.

:::

:::{admonition} Solution
:class: tip, dropdown

Power and energy have different dimensions.

```{code-cell} python
def interval_summary(samples,minutes):
    dt=minutes/60
    return sum(samples)*dt,sum(samples)/len(samples)

assert all(abs(a-b)<1e-9 for a,b in zip(interval_summary([.2,.8,1.2,.6],30),[1.4,.7]))
```
:::

:::{admonition} Quiz 12 — Count wind sectors on bin edges
:class: note

**Difficulty:** Medium

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
assert np.bincount(sector,minlength=4).tolist()==[2,1,1,1]
```
:::

## Hard

:::{admonition} Quiz 13 — Add a turbine function and energy
:class: note

**Difficulty:** Hard

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Write a simplified 3 MW curve, calculate hourly power and energy, and assert that no power lies outside 0–3 MW.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def turbine_power_mw(speed, rated=3.0, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    speed = np.asarray(speed, dtype=float)
    power = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated_speed)
    power[ramp] = rated * (speed[ramp]**3 - cut_in**3) / (rated_speed**3 - cut_in**3)
    power[(speed >= rated_speed) & (speed < cut_out)] = rated
    return power

clean["power_mw"] = turbine_power_mw(clean["speed_ms"])
monthly_energy_mwh = clean["power_mw"].sum()
assert clean["power_mw"].between(0, 3).all()
print(f"Thirty-day energy: {monthly_energy_mwh:.1f} MWh")
```
:::

:::{admonition} Quiz 14 — Energy-weighted wind rose
:class: note

**Difficulty:** Hard

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Sum energy by sector and plot a second polar chart. Which directions matter most for energy, and why can this differ from frequency?

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
energy_by_sector = clean.groupby("sector")["power_mw"].sum().reindex(sector_names, fill_value=0)
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
ax.bar(angles, energy_by_sector.to_numpy(), width=np.deg2rad(40), color="seagreen", edgecolor="darkgreen")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(angles, sector_names)
ax.set_title("Energy contribution by incoming wind direction")
plt.tight_layout()
```

Frequent low-speed winds may contribute less energy than less-frequent high-speed winds because the power curve is nonlinear.
:::

:::{admonition} Quiz 15 — Chronological baseline
:class: note

**Difficulty:** Hard

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Aggregate hourly power to daily energy. Use the first 23 days as training and the last 7 as testing. Forecast each test day with the training mean, calculate MAE, and plot observations and forecast. Do not shuffle.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
daily_energy = clean["power_mw"].resample("D").sum()
train, test = daily_energy.iloc[:-7], daily_energy.iloc[-7:]
forecast = pd.Series(train.mean(), index=test.index)
mae = (test - forecast).abs().mean()

plt.figure(figsize=(9, 4))
plt.plot(daily_energy.index, daily_energy, marker="o", label="Observed")
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

**Difficulty:** Hard

**Reference:** [@manwell2009; @pandasDocs].

Join hourly UTC wind at hours [0,1,2] and meter readings at [1,2,3]. Keep unmatched records, require unique timestamps, and compute matched-only mean speed and observed meter MWh for wind above 5 m/s.

:::

:::{admonition} Solution
:class: tip, dropdown

Report matching coverage before making power-versus-speed comparisons.

```{code-cell} python
import pandas as pd
t=pd.date_range('2025-02-01',periods=4,freq='h',tz='UTC')
v=pd.DataFrame({'time':t[:3],'speed':[2,6,8]})
p=pd.DataFrame({'time':t[1:],'mw':[1,2,3]})
joined=v.merge(p,on='time',how='outer',validate='one_to_one',indicator=True)
matched=joined[joined['_merge']=='both']
assert len(joined)==4 and len(matched)==2
assert matched.speed.mean()==7 and matched.loc[matched.speed>5,'mw'].sum()==3
```
:::

:::{admonition} Quiz 17 — Protect long and edge gaps
:class: note

**Difficulty:** Hard

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
clean=s.mask(eligible,(s.shift()+s.shift(-1))/2)
assert clean.iloc[2]==6 and clean.isna().sum()==4
```
:::

:::{admonition} Quiz 18 — Reconcile sector energy with unequal durations
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009; @pandasDocs].

For directions [350,10,90,180]°, powers [1,2,3,4] MW and durations [0.5,1,2,1] h, use N/E/S/W quadrants. Compare count share and energy share and reconcile the summed energy.

:::

:::{admonition} Solution
:class: tip, dropdown

Count share and energy share need not identify the same predominant direction.

```{code-cell} python
import numpy as np
direction=np.array([350,10,90,180])
p=np.array([1,2,3,4]); dt=np.array([.5,1,2,1])
sector=(((direction+45)%360)//90).astype(int)
count=np.bincount(sector,minlength=4)
energy=np.bincount(sector,weights=p*dt,minlength=4)
assert energy.sum()==np.dot(p,dt)==12.5
assert count.argmax()==0 and energy.argmax()==1
```
:::

:::{admonition} Quiz 19 — Compare circular and speed-weighted direction
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009; @pandasDocs].

For winds from [350,10,90]° with speeds [1,1,5] m/s, compute the equal-weight circular mean and direction of the mean horizontal velocity. Use meteorological from-direction signs and verify the stronger easterly shifts the velocity-based result.

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
assert 0<unweighted<weighted<90 and east<0
```
:::

:::{admonition} Quiz 20 — Estimate coverage versus modelled energy
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009; @pandasDocs].

A turbine reports hourly MW [0,missing,2,missing,4] while a speed-based model estimates [0,1,3,4,4]. Calculate measured energy, model bias over concurrent observations, and modeled energy for the gaps separately. Do not convert missing reports into measurements.

:::

:::{admonition} Solution
:class: tip, dropdown

Bias is based on paired observations; the gap model has a different evidence status than measurements.

```{code-cell} python
import numpy as np
observed=np.array([0,np.nan,2,np.nan,4]); model=np.array([0,1,3,4,4])
matched=np.isfinite(observed)
actual=observed[matched].sum()
bias=(model[matched]-observed[matched]).mean()
gap_estimate=model[~matched].sum()
assert actual==6 and matched.mean()==.6 and gap_estimate==5
assert np.isclose(bias,1/3)
```
:::

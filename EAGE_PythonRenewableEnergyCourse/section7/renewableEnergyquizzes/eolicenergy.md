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

:::{admonition} Quiz 1 — Quality profile
:class: note

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

:::{admonition} Quiz 2 — Repair with explicit rules
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Replace impossible negative speed with missing, wrap direction with modulo 360, and interpolate only short speed gaps in time. Preserve the original table.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
clean = wind.copy()
clean.loc[clean["speed_ms"] < 0, "speed_ms"] = np.nan
clean["direction_deg"] = clean["direction_deg"] % 360
clean["speed_ms"] = clean["speed_ms"].interpolate(method="time", limit=2)
assert clean["speed_ms"].notna().all()
assert clean["speed_ms"].ge(0).all()
assert clean["direction_deg"].between(0, 360, inclusive="left").all()
```
:::

:::{admonition} Quiz 3 — Circular mean
:class: note

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

:::{admonition} Quiz 4 — Direction sectors
:class: note

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

:::{admonition} Quiz 5 — Frequency wind rose
:class: note

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

:::{admonition} Quiz 6 — Add a turbine function and energy
:class: note

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

:::{admonition} Quiz 7 — Energy-weighted wind rose
:class: note

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

:::{admonition} Quiz 8 — Chronological baseline
:class: note

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

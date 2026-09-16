---
kernelspec:
  name: python3
  display_name: Python 3
---

# Solar PV Quiz and Python Challenge

These eight questions progress from units and concepts to functions, arrays, validation, and plots.

:::{admonition} Quiz 1 — Irradiance or irradiation?
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Classify each quantity: `850 W/m²`, `5.2 kWh/m²/day`, `3 MW`, and `18 MWh`.

:::

:::{admonition} Solution
:class: tip, dropdown

- 850 W/m²: irradiance.
- 5.2 kWh/m²/day: daily irradiation.
- 3 MW: electrical power.
- 18 MWh: electrical energy.
:::

:::{admonition} Quiz 2 — Panel calculation
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

A 2.1 m² panel has 21% efficiency. Calculate ideal DC power at 780 W/m² and explain why it is not guaranteed AC output.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

irradiance_wm2, area_m2, efficiency = 780, 2.1, 0.21
dc_power_w = irradiance_wm2 * area_m2 * efficiency
print(f"Idealized DC power: {dc_power_w:.1f} W")
assert np.isclose(dc_power_w, 343.98)
```

The calculation omits temperature, inverter behavior, wiring, mismatch, soiling, shading, and availability.
:::

:::{admonition} Quiz 3 — Build the temperature-aware function
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Write a function that estimates cell temperature with NOCT and then AC power with temperature coefficient, inverter efficiency, and AC clipping. It must accept arrays and reject negative irradiance.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def pv_power_mw(irradiance, ambient_c, dc_mw, ac_mw,
                noct_c=45, gamma=-0.004, inverter_efficiency=0.97):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [irradiance, ambient_c, dc_mw, ac_mw, noct_c, gamma, inverter_efficiency]):
        raise ValueError("model inputs must be finite")
    irradiance = np.asarray(irradiance, dtype=float)
    if np.any(irradiance < 0) or dc_mw < 0 or ac_mw < 0:
        raise ValueError("irradiance and ratings must be non-negative")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("invalid inverter efficiency")
    cell_c = np.asarray(ambient_c) + irradiance / 800 * (noct_c - 20)
    factor = np.maximum(1 + gamma * (cell_c - 25), 0)
    dc_power = dc_mw * irradiance / 1000 * factor
    ac_power = np.minimum(dc_power * inverter_efficiency, ac_mw)
    return float(ac_power) if ac_power.ndim == 0 else ac_power

assert pv_power_mw(0, 25, 10, 8) == 0
assert np.all(pv_power_mw([0, 500, 1000], 25, 10, 8) <= 8)
```
:::

:::{admonition} Quiz 4 — Plot a solar day
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Create 15-minute samples for a synthetic day, calculate irradiance, ambient temperature, cell temperature, and AC power, then make two labelled plots. Calculate daily MWh using the correct 0.25-hour step.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import matplotlib.pyplot as plt

hour = np.arange(0, 24, 0.25)
daylight = (hour >= 6) & (hour <= 18)
irradiance = np.zeros_like(hour)
irradiance[daylight] = 1000 * np.sin(np.pi * (hour[daylight] - 6) / 12)
ambient = 20 + 8 * np.sin(2 * np.pi * (hour - 9) / 24)
cell = ambient + irradiance / 800 * 25
power = pv_power_mw(irradiance, ambient, 12, 10)
energy_mwh = power.sum() * 0.25

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(hour, irradiance, color="goldenrod", label="Irradiance")
axes[0].set(ylabel="W/m²", title="Synthetic solar resource")
axes[1].plot(hour, power, color="tab:blue", label="AC power")
axes[1].set(xlabel="Hour", ylabel="MW", title=f"AC energy = {energy_mwh:.1f} MWh")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
assert energy_mwh > 0
```
:::

:::{admonition} Quiz 5 — Find inverter clipping
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

For DC/AC ratios 1.0, 1.2, and 1.4 with a 10 MW AC inverter, calculate energy and clipped energy for the same day. Plot energy versus DC/AC ratio.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
ratios = np.array([1.0, 1.2, 1.4])
energies, clipped = [], []
for ratio in ratios:
    temperature_factor = np.maximum(1 - 0.004 * (cell - 25), 0)
    dc_unclipped = ratio * 10 * irradiance / 1000 * temperature_factor
    ac = pv_power_mw(irradiance, ambient, ratio * 10, 10)
    energies.append(ac.sum() * 0.25)
    clipping_mwh = np.maximum(dc_unclipped * 0.97 - 10, 0).sum() * 0.25
    clipped.append(clipping_mwh)
    assert np.isclose(dc_unclipped.sum() * 0.97 * 0.25, energies[-1] + clipping_mwh)

plt.figure(figsize=(7, 4))
plt.plot(ratios, energies, marker="o", label="Delivered AC energy")
plt.plot(ratios, clipped, marker="s", label="Clipped energy estimate")
plt.xlabel("DC/AC ratio")
plt.ylabel("Energy (MWh/day)")
plt.title("Array sizing and inverter clipping")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

:::{admonition} Quiz 6 — Validate monthly performance ratio
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Given monthly irradiation and AC energy, calculate PR and flag suspicious values for investigation. PR above one is not automatically impossible; cold conditions and rating/measurement conventions can cause it. Use a DataFrame and a bar plot.

```python
irradiation_kwh_m2 = [58, 82, 118, 145, 170, 188, 194, 176, 132, 94, 62, 48]
energy_mwh = [420, 620, 900, 1120, 1280, 1420, 1480, 1320, 1010, 710, 450, 350]
```

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import pandas as pd

monthly = pd.DataFrame({
    "irradiation_kwh_m2": [58, 82, 118, 145, 170, 188, 194, 176, 132, 94, 62, 48],
    "energy_mwh": [420, 620, 900, 1120, 1280, 1420, 1480, 1320, 1010, 710, 450, 350],
}, index=pd.period_range("2025-01", periods=12, freq="M").astype(str))
rated_mw = 10  # DC nameplate at STC
monthly["reference_yield_h"] = monthly["irradiation_kwh_m2"] / 1.0
monthly["final_yield_h"] = monthly["energy_mwh"] / rated_mw
monthly["pr"] = monthly["final_yield_h"] / monthly["reference_yield_h"]
monthly["flag"] = ~monthly["pr"].between(0, 1)

monthly["pr"].plot.bar(figsize=(9, 4), color="darkorange", ylim=(0, 1.05),
                       title="Monthly PV performance ratio")
plt.ylabel("Performance ratio")
plt.tight_layout()
assert not monthly["flag"].any()
```
:::

:::{admonition} Quiz 7 — Temperature–irradiance map
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Use `np.meshgrid` to evaluate power for irradiance 0–1100 W/m² and ambient temperature -10–45 °C. Plot a filled contour and identify the clipped plateau.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
irradiance_axis = np.linspace(0, 1100, 80)
temperature_axis = np.linspace(-10, 45, 60)
G_grid, T_grid = np.meshgrid(irradiance_axis, temperature_axis)
P_grid = pv_power_mw(G_grid, T_grid, 12, 10)

plt.figure(figsize=(8, 5))
contour = plt.contourf(G_grid, T_grid, P_grid, levels=20, cmap="YlOrRd")
plt.colorbar(contour, label="AC power (MW)")
plt.xlabel("Irradiance (W/m²)")
plt.ylabel("Ambient temperature (°C)")
plt.title("PV power sensitivity and clipping")
plt.tight_layout()
```
:::

:::{admonition} Quiz 8 — Lifetime degradation
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Write a function returning annual and cumulative energy for a degradation rate. Compare 0.3%, 0.5%, and 0.8% over 25 years using a plot. Verify that year-one energy is identical for all scenarios and that every sequence is non-increasing.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def degraded_energy(year_one_mwh, annual_rate, years=25):
    if year_one_mwh < 0 or not 0 <= annual_rate < 1 or years < 1:
        raise ValueError("invalid degradation inputs")
    year = np.arange(1, years + 1)
    annual = year_one_mwh * (1 - annual_rate) ** (year - 1)
    return year, annual, np.cumsum(annual)

plt.figure(figsize=(8, 4))
for rate in [0.003, 0.005, 0.008]:
    year, annual, cumulative = degraded_energy(15000, rate)
    plt.plot(year, annual / 1000, label=f"{rate:.1%}/year")
    assert annual[0] == 15000 and np.all(np.diff(annual) <= 0)
plt.xlabel("Operating year")
plt.ylabel("Annual energy (GWh)")
plt.title("PV degradation scenarios")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

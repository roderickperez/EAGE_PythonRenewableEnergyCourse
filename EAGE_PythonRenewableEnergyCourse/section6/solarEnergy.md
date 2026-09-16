---
kernelspec:
  name: python3
  display_name: Python 3
---

# Solar Photovoltaic Energy

## Learning goals

After this lesson you should be able to:

- distinguish irradiance, irradiation, power, energy, efficiency, and performance ratio;
- estimate cell temperature and temperature-corrected DC power;
- model inverter clipping and aggregate hourly energy;
- build reusable scalar-and-array Python functions;
- plot a solar day, a sensitivity surface, and monthly energy;
- state the limits of a simplified PV model.

A photovoltaic (PV) cell converts part of incident solar radiation directly into DC electricity. An inverter converts DC to grid-compatible AC. Irradiance and cell temperature affect instantaneous output; shading, soiling, wiring, mismatch, inverter behavior, availability, and degradation affect delivered energy. Standard Test Conditions (STC) use 1000 W/m$^2$ irradiance, 25 °C cell temperature and the AM1.5 reference spectrum [@doePVPerformance; @doePVLongevity].

## Concepts and equations

### Irradiance versus irradiation

- **Irradiance** $G$ is instantaneous power per area, W/m$^2$.
- **Irradiation** is irradiance integrated over time, commonly kWh/m$^2$.

Do not label an hourly sum of irradiance values as W/m$^2$; after multiplying by the time step it is an energy-per-area quantity.

### From solar resource to PV power

For area $A$ and conversion efficiency $\eta$:

$$P_{DC}=GA\eta$$

For a rated array, a convenient first-order model is

$$P_{DC}=P_{STC}\frac{G}{G_{STC}}
\left[1+\gamma_P(T_c-25)\right]$$

where $\gamma_P$ is normally negative and $T_c$ is cell temperature in °C. A simple NOCT approximation is

$$T_c=T_a+\frac{G}{800}(NOCT-20)$$

This is a teaching model, not a bankable performance model. The temperature equation is an empirical approximation and should not be extrapolated carelessly.

### AC output and energy

With inverter efficiency $\eta_{inv}$ and AC rating $P_{AC,r}$:

$$P_{AC}=\min(P_{DC}\eta_{inv},P_{AC,r})$$

For hourly samples:

$$E_{AC}=\sum_i P_{AC,i}\Delta t_i$$

Performance ratio (PR) is a normalized energy-quality indicator. One common definition is

$$PR=\frac{E_{AC}/P_{rated}}{H_{POA}/G_{STC}}$$

where $H_{POA}$ is plane-of-array irradiation and $P_{rated}$ is the DC nameplate rating at STC. Use consistent kW/kWh or MW/MWh units; $G_{STC}=1$ kW/m². State the exact convention because PR boundaries can differ.

## Tested Python functions

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def cell_temperature_c(irradiance_wm2, ambient_c, noct_c=45.0):
    """Simplified NOCT cell-temperature estimate."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [irradiance_wm2, ambient_c, noct_c]):
        raise ValueError("model inputs must be finite")
    irradiance = np.asarray(irradiance_wm2, dtype=float)
    if np.any(irradiance < 0):
        raise ValueError("irradiance must be non-negative")
    return np.asarray(ambient_c, dtype=float) + irradiance / 800 * (noct_c - 20)

def pv_ac_power_mw(irradiance_wm2, ambient_c, dc_rating_mw,
                   ac_rating_mw, gamma_p=-0.004, noct_c=45.0,
                   inverter_efficiency=0.97):
    """Simplified temperature-corrected PV output, clipped at AC rating."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [dc_rating_mw, ac_rating_mw, gamma_p, inverter_efficiency]):
        raise ValueError("model inputs must be finite")
    if dc_rating_mw < 0 or ac_rating_mw < 0:
        raise ValueError("ratings must be non-negative")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("inverter_efficiency must be in (0, 1]")
    irradiance = np.asarray(irradiance_wm2, dtype=float)
    cell_c = cell_temperature_c(irradiance, ambient_c, noct_c)
    temperature_factor = np.maximum(1 + gamma_p * (cell_c - 25), 0)
    dc_power = dc_rating_mw * irradiance / 1000 * temperature_factor
    ac_power = np.minimum(dc_power * inverter_efficiency, ac_rating_mw)
    return float(ac_power) if ac_power.ndim == 0 else ac_power

assert pv_ac_power_mw(0, 25, 10, 8) == 0
assert 0 <= pv_ac_power_mw(1000, 25, 10, 8) <= 8
```

### Why the checks matter

Input validation stops physically impossible negative irradiance and invalid efficiency values. `np.asarray` lets the same function accept a number, list, or NumPy array. `np.maximum` prevents a linear temperature approximation from producing negative power outside its valid range.

## Worked solar-day example

```{code-cell} python
hours = np.arange(24)
sunrise, sunset = 6, 19
daylight = (hours >= sunrise) & (hours <= sunset)
irradiance = np.zeros_like(hours, dtype=float)
irradiance[daylight] = 950 * np.sin(
    np.pi * (hours[daylight] - sunrise) / (sunset - sunrise)
)
irradiance = np.clip(irradiance, 0, None)  # remove floating-point noise at sunset
ambient_c = 18 + 9 * np.sin(2 * np.pi * (hours - 8) / 24)
cell_c = cell_temperature_c(irradiance, ambient_c)
power_mw = pv_ac_power_mw(irradiance, ambient_c, 12, 10)
daily_energy_mwh = power_mw.sum()  # one-hour intervals

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(hours, irradiance, label="Irradiance", color="goldenrod")
axes[0].set(ylabel="Irradiance (W/m²)", title="Synthetic clear-sky day")
axes[0].grid(alpha=0.3)
axes[1].plot(hours, power_mw, label="AC power", color="tab:blue")
axes[1].fill_between(hours, power_mw, alpha=0.25)
axes[1].set(xlabel="Hour", ylabel="Power (MW)",
            title=f"Daily AC energy = {daily_energy_mwh:.1f} MWh")
axes[1].grid(alpha=0.3)
plt.tight_layout()
```

Because each sample represents one hour, summing MW values yields MWh. With 15-minute data, multiply the sum by 0.25 h.

## Guided exercises

:::{admonition} Exercise 1 — Panel and array scaling
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Calculate one 2.0 m$^2$, 21%-efficient panel at 800 W/m$^2$. Write a function that scales to any panel count and test zero panels.

```python
def array_power_kw(irradiance_wm2, panel_area_m2, efficiency, panel_count):
    # TODO: validate inputs and return kW
    pass
```
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

1. Multiply irradiance, panel area and efficiency.
2. Scale by count.
3. Convert watts to kw and test zero panels.

```{code-cell} python
def array_power_kw(irradiance_wm2, panel_area_m2, efficiency, panel_count):
    if min(irradiance_wm2, panel_area_m2, panel_count) < 0 or not 0 <= efficiency <= 1:
        raise ValueError("invalid PV input")
    return irradiance_wm2 * panel_area_m2 * efficiency * panel_count / 1000

print(f"One panel: {array_power_kw(800, 2.0, 0.21, 1):.3f} kW")
assert array_power_kw(800, 2.0, 0.21, 0) == 0
```
:::

:::{admonition} Exercise 2 — Temperature table
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

For ambient temperatures 0–40 °C and irradiance 900 W/m$^2$, calculate cell temperature and AC power. Store the results in a DataFrame.

```python
ambient_values = np.arange(0, 41, 5)
# TODO: calculate cell temperature and AC power
# TODO: create and print a DataFrame
```
:::

:::{admonition} Exercise 2 — Solution
:class: tip, dropdown

1. Make the ambient-temperature array.
2. Calculate cell temperature before output.
3. Store labelled columns.

```{code-cell} python
import pandas as pd

ambient_values = np.arange(0, 41, 5)
temperature_table = pd.DataFrame({
    "ambient_c": ambient_values,
    "cell_c": cell_temperature_c(900, ambient_values),
    "ac_power_mw": pv_ac_power_mw(900, ambient_values, 12, 10),
})
print(temperature_table.round(2))
assert temperature_table["ac_power_mw"].ge(0).all()
```
:::

:::{admonition} Exercise 3 — Sensitivity surface
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Use `np.meshgrid` for irradiance 0–1100 W/m$^2$ and ambient temperature -10–45 °C. Plot AC power with `contourf` and identify clipping.

```python
# TODO: build the two-dimensional grid
# TODO: evaluate pv_ac_power_mw and plot it
```
:::

:::{admonition} Exercise 3 — Solution
:class: tip, dropdown

1. Create the irradiance/temperature grid.
2. Evaluate clipped AC power.
3. Plot with a labelled color scale.

```{code-cell} python
irradiance_axis = np.linspace(0, 1100, 80)
temperature_axis = np.linspace(-10, 45, 60)
irradiance_grid, temperature_grid = np.meshgrid(irradiance_axis, temperature_axis)
power_grid = pv_ac_power_mw(irradiance_grid, temperature_grid, 12, 10)
plt.figure(figsize=(8, 5))
contour = plt.contourf(irradiance_grid, temperature_grid, power_grid, levels=20, cmap="YlOrRd")
plt.colorbar(contour, label="AC power (MW)")
plt.xlabel("Irradiance (W/m²)")
plt.ylabel("Ambient temperature (°C)")
plt.title("PV power and inverter clipping")
plt.tight_layout()
```
:::

:::{admonition} Exercise 4 — Sampling interval
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Create `energy_mwh(power_mw, step_hours)`. Use 30-minute samples and demonstrate why omitting the 0.5-hour multiplier doubles the estimate.

```python
def energy_mwh(power_mw, step_hours):
    # TODO: validate the time step and integrate power
    pass
```
:::

:::{admonition} Exercise 4 — Solution
:class: tip, dropdown

1. Multiply each interval-average MW by hours.
2. Add interval energies.
3. Compare with the unweighted sum.

```{code-cell} python
def energy_mwh(power_mw, step_hours):
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    return np.asarray(power_mw, dtype=float).sum() * step_hours

half_hour_power = np.repeat(power_mw, 2)
correct_energy = energy_mwh(half_hour_power, 0.5)
incorrect_energy = half_hour_power.sum()
assert np.isclose(incorrect_energy, 2 * correct_energy)
```
:::

:::{admonition} Exercise 5 — Monthly performance ratio
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Calculate and plot monthly PR from plane-of-array irradiation and AC energy. Flag values outside 0–1 for investigation. PR above one can occur in cold conditions or because of measurement and rating conventions; this is not automatically a violation of energy conservation.

```python
# TODO: write performance_ratio(energy_mwh, rated_mw, irradiation_kwh_m2)
# TODO: calculate, validate, and plot 12 monthly values
```
:::

:::{admonition} Exercise 5 — Solution
:class: tip, dropdown

1. Divide AC energy by DC nameplate for final yield.
2. Divide irradiation by stc irradiance.
3. Form their ratio.

```{code-cell} python
def performance_ratio(energy_mwh, rated_mw, irradiation_kwh_m2):
    energy = np.asarray(energy_mwh, dtype=float)
    irradiation = np.asarray(irradiation_kwh_m2, dtype=float)
    if rated_mw <= 0 or np.any(irradiation <= 0):
        raise ValueError("rating and irradiation must be positive")
    return (energy / rated_mw) / irradiation

irradiation = np.array([58, 82, 118, 145, 170, 188, 194, 176, 132, 94, 62, 48])
measured_energy = np.array([420, 620, 900, 1120, 1280, 1420, 1480, 1320, 1010, 710, 450, 350])
monthly_pr = performance_ratio(measured_energy, 10, irradiation)
plt.figure(figsize=(9, 4))
plt.bar(np.arange(1, 13), monthly_pr)
plt.xlabel("Month")
plt.ylabel("Performance ratio")
plt.ylim(0, 1)
plt.tight_layout()
assert np.all((monthly_pr >= 0) & (monthly_pr <= 1))
```
:::

:::{admonition} Exercise 6 — Degradation scenarios
:class: note

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Compare annual degradation rates 0.3%, 0.5%, and 0.8% over 25 years using compounding. Plot annual energy and compare cumulative output.

```python
# TODO: write degraded_energy(year_one_mwh, rate, years)
# TODO: loop over the three rates and plot each sequence
```
:::

:::{admonition} Exercise 6 — Solution
:class: tip, dropdown

1. Create years beginning at one.
2. Compound the retained fraction.
3. Compare annual and cumulative energy.

```{code-cell} python
def degraded_energy(year_one_mwh, rate, years=25):
    if year_one_mwh < 0 or not 0 <= rate < 1 or years < 1:
        raise ValueError("invalid degradation inputs")
    year = np.arange(1, years + 1)
    return year_one_mwh * (1 - rate) ** (year - 1)

plt.figure(figsize=(8, 4))
for degradation_rate in [0.003, 0.005, 0.008]:
    annual_energy = degraded_energy(15000, degradation_rate)
    plt.plot(np.arange(1, 26), annual_energy / 1000, label=f"{degradation_rate:.1%}/year")
    print(f"{degradation_rate:.1%}/year: cumulative {annual_energy.sum():.1f} MWh")
    assert np.all(np.diff(annual_energy) <= 0)
plt.xlabel("Operating year")
plt.ylabel("Annual energy (GWh)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

## Common mistakes

- Confusing irradiance (W/m$^2$) with irradiation (kWh/m$^2$).
- Applying a temperature correction to ambient temperature instead of estimated cell temperature without stating the simplification.
- Omitting inverter clipping or applying system losses twice.
- Summing power without multiplying by the time-step duration.
- Treating the simplified model as a substitute for site-specific shading, orientation, and loss modelling.
- Calling DC nameplate power an energy value.

## Check your understanding

Continue with the [solar quiz and Python challenge](../section7/renewableEnergyquizzes/solarEnergy.md).

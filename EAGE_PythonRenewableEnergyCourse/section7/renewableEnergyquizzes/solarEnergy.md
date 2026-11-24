---
kernelspec:
  name: python3
  display_name: Python 3
---

# Solar PV Quiz

These questions progress from units and concepts to functions, arrays, validation, and plots.

**4 Easy · 8 Medium · 8 Hard.** Try each question before opening its solution. The code examples build on a shared Python session: run the preceding setup and solution code in order, or provide equivalent imports, data and functions yourself. In the sandbox, **Include preceding examples** does not run answers hidden in solution dropdowns.

In the sandbox, after opening a code answer, **Include preceding practice answers** loads the earlier runnable examples and answers plus that answer into the editor. Review the code and select **Run Python**; loading it does not run it automatically.

:::{admonition} Quiz 1 — Irradiance or irradiation?
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Classify each quantity: irradiance `850 W/m²`, one day’s integrated irradiation `5.2 kWh/m²`, and a plant’s electrical output `3 MW` and electrical energy `18 MWh`.

:::

:::{admonition} Solution
:class: tip, dropdown

- 850 W/m²: irradiance.
- 5.2 kWh/m²: irradiation integrated over that day.
- 3 MW: electrical power.
- 18 MWh: electrical energy.
:::

:::{admonition} Quiz 2 — Panel calculation
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

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

:::{admonition} Quiz 3 — Convert irradiation units
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@foster2010; @pvlibDocs].

Write a function to convert Wh/m² to kWh/m². Test 750 Wh/m².

:::

:::{admonition} Solution
:class: tip, dropdown

Irradiation is energy per area, rather than power per area.

```{code-cell} python
def wh_to_kwh(wh_per_m2):
    return wh_per_m2/1000

print('Irradiation:', wh_to_kwh(750), 'kWh/m²')
assert wh_to_kwh(750)==.75
```
:::

:::{admonition} Quiz 4 — Convert efficiency percentages
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@foster2010; @pvlibDocs].

Write a function that accepts a percentage in [0,100] and returns a fraction. Use 19% to calculate DC W for 2 m² at 600 W/m².

:::

:::{admonition} Solution
:class: tip, dropdown

The formula expects 0.19, not 19.

```{code-cell} python
def fraction(percent):
    if not 0<=percent<=100:
        raise ValueError('Percentage outside [0,100]')
    return percent/100

print('DC power:', 600*2*fraction(19), 'W')
assert 600*2*fraction(19)==228
```
:::

:::{admonition} Quiz 5 — Build the temperature-aware function
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Write a function that estimates cell temperature with NOCT and then AC power with temperature coefficient, inverter efficiency, and AC clipping. It must accept irradiance and ambient-temperature arrays and reject negative irradiance. For paired records, align labels first and pass matching one-dimensional arrays in the same order; the model pairs values by position and requires unique, nonmissing and matching pandas indexes when inputs are labeled. If a labeled series is paired with an unlabeled array, check their order yourself; the helper cannot verify that pairing.

The answer uses a fixed 97% inverter efficiency as a teaching assumption before the AC cap. The [PVWatts DC relationship](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html), referenced elsewhere in this quiz, describes the module-side calculation; it does not establish this inverter assumption. Actual inverter efficiency can vary with load, particularly at low input. Compare the separate [pvlib PVWatts inverter model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html) when a more detailed model is appropriate.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def pv_power_mw(irradiance, ambient_c, dc_mw, ac_mw,
                noct_c=45, gamma=-0.004, inverter_efficiency=0.97):
    """Pair by position; labeled inputs must already have matching indexes."""
    indexes = [getattr(value, "index", None) for value in (irradiance, ambient_c)]
    indexes = [index for index in indexes if index is not None and not callable(index)
               and callable(getattr(index, "equals", None))]
    for index in indexes:
        if hasattr(index, "is_unique") and not index.is_unique:
            raise ValueError("Labeled inputs must have unique, nonmissing indexes")
        if callable(getattr(index, "to_frame", None)):
            name = list(range(index.nlevels)) if index.nlevels > 1 else 0
            if index.to_frame(index=False, name=name).isna().to_numpy().any():
                raise ValueError("Labeled inputs must have unique, nonmissing indexes")
    if len(indexes) > 1 and any(not indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError("Labeled inputs must have matching indexes in the same order")
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [irradiance, ambient_c, dc_mw, ac_mw, noct_c, gamma, inverter_efficiency]):
        raise ValueError("model inputs must be finite")
    irradiance = np.asarray(irradiance, dtype=float)
    ambient_c = np.asarray(ambient_c, dtype=float)
    if (irradiance.ndim > 1 or ambient_c.ndim > 1 or
            (irradiance.ndim == ambient_c.ndim == 1 and irradiance.shape != ambient_c.shape)):
        raise ValueError("irradiance and temperature must be scalars or matching 1-D arrays")
    if np.any(irradiance < 0) or dc_mw < 0 or ac_mw < 0:
        raise ValueError("irradiance and ratings must be non-negative")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("invalid inverter efficiency")
    if np.any(np.asarray(ambient_c, dtype=float) <= -273.15) or noct_c <= -273.15:
        raise ValueError("temperatures must exceed absolute zero")
    cell_c = np.asarray(ambient_c, dtype=float) + irradiance / 800 * (noct_c - 20)
    if np.any(cell_c <= -273.15):
        raise ValueError("modelled cell temperature must exceed absolute zero")
    factor = np.maximum(1 + gamma * (cell_c - 25), 0)
    dc_power = dc_mw * irradiance / 1000 * factor
    ac_power = np.minimum(dc_power * inverter_efficiency, ac_mw)
    return float(ac_power) if ac_power.ndim == 0 else ac_power

print('AC power at 0, 500 and 1000 W/m² (MW):', pv_power_mw([0, 500, 1000], 25, 10, 8))
assert pv_power_mw(0, 25, 10, 8) == 0
assert np.all(pv_power_mw([0, 500, 1000], 25, 10, 8) <= 8)
```
:::

:::{admonition} Quiz 6 — Plot a solar day
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Use the function from Quiz 5 to create 15-minute samples for a synthetic day, calculate irradiance, ambient temperature, cell temperature, and AC power, then make two labelled plots. Treat each sample as constant over its 0.25-hour interval when calculating daily MWh.

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

:::{admonition} Quiz 7 — Find inverter clipping
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Use the function from Quiz 5 and the synthetic 15-minute day from Quiz 6. For DC/AC ratios 1.0, 1.2, and 1.4 with a 10 MW AC inverter, calculate delivered AC energy and the energy clipped at the AC output boundary, after applying inverter efficiency. Treat each sample as constant over its 0.25-hour interval. Plot both energies versus DC/AC ratio.

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
    print(f"DC/AC {ratio:.1f}: delivered {energies[-1]:.2f} MWh, clipped {clipping_mwh:.2f} MWh")
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

:::{admonition} Quiz 8 — Find irradiance on a horizontal panel
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [pvlib irradiance closure](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.complete_irradiance.html).

The linked documentation checked on 19 November 2026 mislabelled its `ghi` parameter as DNI in one description. Here GHI is global horizontal irradiance; use the displayed closure equation and check the documentation version.

For daylight zenith angles from 0° through 90°, write a function for GHI = DNI cos(zenith) + DHI. Require finite, non-negative irradiances and a zenith in that range. Test DNI 700 W/m², DHI 120 W/m² and 60° zenith.

:::

:::{admonition} Solution
:class: tip, dropdown

Python's cosine takes radians. The direct component uses the projection onto the horizontal surface.

```{code-cell} python
from math import cos,radians,isclose,isfinite
def ghi(dni,dhi,zenith):
    if not all(isfinite(value) for value in (dni, dhi, zenith)) or dni < 0 or dhi < 0 or not 0 <= zenith <= 90:
        raise ValueError("Use finite non-negative irradiances and a daylight zenith from 0 to 90 degrees")
    return dni*cos(radians(zenith))+dhi

print('Horizontal irradiance:', ghi(700,120,60), 'W/m²')
assert isclose(ghi(700,120,60),470)
```
:::

:::{admonition} Quiz 9 — Compute clipped quarter-hour energy
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@foster2010; @pvlibDocs].

Use AC potential [1,4,6,2] kW, each held constant for 15 minutes, with a 3 kW AC cap. Write a function to return delivered and clipped kWh from kW and minutes.

:::

:::{admonition} Solution
:class: tip, dropdown

Keep the conversion from minutes to hours outside the power cap.

```{code-cell} python
def capped_energy(potential, cap, minutes):
    output=[min(value,cap) for value in potential]
    return sum(output)*minutes/60,(sum(potential)-sum(output))*minutes/60

print('Delivered and clipped energy (kWh):', capped_energy([1,4,6,2],3,15))
assert capped_energy([1,4,6,2],3,15)==(2.25,1.0)
```
:::

:::{admonition} Quiz 10 — Separate optical and wiring losses
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@foster2010; @pvlibDocs].

Apply 5% soiling before DC conversion and 2% wiring loss after conversion. With 900 W/m², 2 m² and 20% efficiency, return ideal DC W and post-wiring DC W.

:::

:::{admonition} Solution
:class: tip, dropdown

The stages have different physical boundaries even when their retained fractions can be multiplied.

```{code-cell} python
def module_output(g,area,eta,soil,wiring):
    ideal=g*area*eta
    return ideal,ideal*(1-soil)*(1-wiring)

ideal, output=module_output(900,2,.20,.05,.02)
print('Ideal and post-loss power (W):', ideal, output)
assert ideal==360 and abs(output-335.16)<1e-8
```
:::

:::{admonition} Quiz 11 — Assess the units of a monitoring series
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@foster2010; @pvlibDocs].

Read four AC interval-average powers [0.3,0.6,1.2,0.9] kW, each representing one half-hour interval. Write a function for kWh and time-average kW, and compare the two outputs.

:::

:::{admonition} Solution
:class: tip, dropdown

Divide by the total observed hours to recover average power.

```{code-cell} python
def integrate_kw(power,minutes):
    hours=len(power)*minutes/60
    energy=sum(power)*minutes/60
    return energy,energy/hours

print('Energy kWh, mean kW:', integrate_kw([.3,.6,1.2,.9],30))
assert all(abs(a-b)<1e-9 for a,b in zip(integrate_kw([.3,.6,1.2,.9],30),[1.5,.75]))
```
:::

:::{admonition} Quiz 12 — Find a module's fill factor
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@foster2010], p. 126, Eq. 5.12; [NREL solar-module efficiency table guide](https://www.nrel.gov/docs/libraries/pv/solar-module-efficiency-data-guide.pdf), “Fill Factor.”

Write and test a function for fill factor using 38 V, 9 A at maximum power and 46 V, 10 A at the open- and short-circuit endpoints.

:::

:::{admonition} Solution
:class: tip, dropdown

Fill factor measures the shape of the current–voltage curve.

```{code-cell} python
def fill_factor(vmp,imp,voc,isc):
    if not (0<vmp<voc and 0<imp<isc):
        raise ValueError('Inconsistent module ratings')
    return vmp*imp/(voc*isc)

print('Fill factor:', fill_factor(38,9,46,10))
assert .74<fill_factor(38,9,46,10)<.75
```
:::

:::{admonition} Quiz 13 — Validate monthly performance ratio
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Given monthly plane-of-array irradiation and AC energy from a 10 MW DC (STC) plant, calculate PR and flag suspicious values for investigation. PR above one is not automatically impossible; cold conditions and rating/measurement conventions can cause it. Use a DataFrame and a bar plot. Treat a zero or missing irradiation denominator as unavailable rather than an infinite PR.

```python
irradiation_kwh_m2 = [58, 82, 118, 145, 170, 188, 194, 176, 132, 94, 62, 48]
energy_mwh = [420, 620, 900, 1120, 1280, 1420, 1480, 1320, 1010, 710, 450, 350]
```

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np
import pandas as pd

monthly = pd.DataFrame({
    "irradiation_kwh_m2": [58, 82, 118, 145, 170, 188, 194, 176, 132, 94, 62, 48],
    "energy_mwh": [420, 620, 900, 1120, 1280, 1420, 1480, 1320, 1010, 710, 450, 350],
}, index=pd.period_range("2025-01", periods=12, freq="M").astype(str))
rated_mw = 10  # DC nameplate at STC
for column in ("irradiation_kwh_m2", "energy_mwh"):
    monthly[column] = pd.to_numeric(monthly[column], errors="raise").astype(float)
    if np.isinf(monthly[column].to_numpy()).any() or (monthly[column].dropna() < 0).any():
        raise ValueError("Measurements must be non-negative and finite, or missing")
monthly["reference_yield_h"] = monthly["irradiation_kwh_m2"] / 1.0
monthly["final_yield_h"] = monthly["energy_mwh"] / rated_mw
monthly["pr"] = monthly["final_yield_h"] / monthly["reference_yield_h"].where(monthly["reference_yield_h"] > 0)
monthly["flag"] = monthly["pr"].notna() & ~monthly["pr"].between(0, 1)

monthly["pr"].plot.bar(figsize=(9, 4), color="darkorange",
                       title="Monthly PV performance ratio")
plt.ylabel("Performance ratio")
plt.tight_layout()
print("Months flagged for investigation:", monthly.index[monthly["flag"]].tolist())
print("Months with unavailable PR:", monthly.index[monthly["pr"].isna()].tolist())
```
:::

:::{admonition} Quiz 14 — Temperature–irradiance map
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Use `np.meshgrid` to evaluate power for irradiance 0–1100 W/m² and ambient temperature -10–45 °C. Plot a filled contour and identify the clipped plateau.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
from matplotlib.lines import Line2D

irradiance_axis = np.linspace(0, 1100, 80)
temperature_axis = np.linspace(-10, 45, 60)
G_grid, T_grid = np.meshgrid(irradiance_axis, temperature_axis)
# Pair corresponding grid cells explicitly, then restore the plotting shape.
P_grid = pv_power_mw(G_grid.ravel(), T_grid.ravel(), 12, 10).reshape(G_grid.shape)
at_cap = P_grid >= 10.0  # includes exact equality at the specified 10 MW AC rating

plt.figure(figsize=(8, 5))
contour = plt.contourf(G_grid, T_grid, P_grid, levels=20, cmap="YlOrRd")
plt.colorbar(contour, label="AC power (MW)")
if at_cap.any() and (~at_cap).any():
    plt.contour(G_grid, T_grid, at_cap.astype(float), levels=[0.5], colors="black", linewidths=1.5)
    plt.legend(handles=[Line2D([0], [0], color="black", label="10 MW cap boundary (sampled grid)")], loc="lower left")
print(f"Grid points at the 10 MW cap: {at_cap.sum()} of {at_cap.size}")
plt.xlabel("Irradiance (W/m²)")
plt.ylabel("Ambient temperature (°C)")
plt.title("PV power sensitivity and clipping")
plt.tight_layout()
```
:::

:::{admonition} Quiz 15 — Lifetime degradation
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** Wade, Chapters 2–3 and 9 [@wade2003]; Foster et al., photovoltaic systems [@foster2010].

Assume first-year energy of 15,000 MWh. Write a function returning annual and cumulative energy for a degradation rate. Compare 0.3%, 0.5%, and 0.8% over 25 years using a plot. Verify that year-one energy is identical for all scenarios and that every sequence is non-increasing.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def degraded_energy(year_one_mwh, annual_rate, years=25):
    if (not np.isfinite(year_one_mwh) or year_one_mwh < 0
            or not np.isfinite(annual_rate) or not 0 <= annual_rate < 1
            or isinstance(years, (bool, np.bool_))
            or not isinstance(years, (int, np.integer)) or years < 1):
        raise ValueError("Use finite non-negative energy, a finite rate in [0,1), and a positive integer year count")
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

:::{admonition} Quiz 16 — Join inverter and irradiance readings
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [DOE PV performance assessment, §2](https://www.energy.gov/sites/default/files/2022-01/understanding-solar-photovoltaic-system-performance.pdf).

Use hourly UTC interval-mean plane-of-array irradiance at 10, 11, 12 with [500,700,900] W/m² and AC interval-mean powers at 11, 12, 13 with [2,3,1] kW. Each reading represents one hour. For a 5 kW DC nameplate at standard test conditions (1,000 W/m² reference irradiance), require unique, nonmissing UTC hour-start timestamps, keep unmatched rows, and compute the STC-normalized performance ratio only over paired intervals with positive total irradiance.

:::

:::{admonition} Solution
:class: tip, dropdown

Both yield denominators must use the same hours.

```{code-cell} python
import numpy as np
import pandas as pd
t=pd.date_range('2025-06-01 10:00',periods=4,freq='h',tz='UTC')
g=pd.Series([500,700,900],index=t[:3],name='g')
ac=pd.Series([2,3,1],index=t[1:],name='ac')
def validate_hourly_readings(series):
    index = series.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != 'UTC'
            or index.hasnans or not index.is_unique or not index.equals(index.floor('h'))):
        raise ValueError('Use unique, nonmissing UTC hour-start timestamps')
    values = pd.to_numeric(series, errors='raise').astype(float)
    if np.isinf(values.to_numpy()).any() or (values.dropna() < 0).any():
        raise ValueError('Readings must be non-negative and finite, or missing')
    return values
joined=pd.concat([validate_hourly_readings(g),validate_hourly_readings(ac)],axis=1)
paired=joined.dropna()
if paired.empty or paired.g.sum() <= 0:
    raise ValueError('Paired intervals need positive total irradiance')
pr=(paired.ac.sum()/5)/(paired.g.sum()/1000)
print('Matched intervals:', len(paired), 'of', len(joined), '; performance ratio:', pr)
assert len(joined)==4 and len(paired)==2 and abs(pr-.625)<1e-12
```
:::

:::{admonition} Quiz 17 — Select an inverter under a clipping target
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@foster2010; @pvlibDocs].

For DC output [1,3,6,8,6,2] kW held constant over respective durations [1,1,0.5,0.5,1,1] h and 95% inverter efficiency, find the smallest candidate cap [4,5,6,7,8] kW retaining at least 95% of potential AC energy.

:::

:::{admonition} Solution
:class: tip, dropdown

A short high-power interval can contribute less lost energy than a long moderate one.

```{code-cell} python
import numpy as np
potential=.95*np.array([1,3,6,8,6,2]); dt=np.array([1,1,.5,.5,1,1])
def retained(cap):
    return np.dot(np.minimum(potential,cap),dt)/np.dot(potential,dt)

choice=next(cap for cap in range(4,9) if retained(cap)>=.95)
print("Smallest qualifying inverter:", choice, "kW; retained fraction:", retained(choice))
assert choice==6 and retained(5)<.95
```
:::

:::{admonition} Quiz 18 — Back-test a performance drift rule
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@foster2010; @pvlibDocs].

Expected daily energy is 10 kWh for eight days. Observed energy is [9,10,9,10,9,7,8,9] kWh. Fit an observed/expected ratio on days 1–4, flag holdout days below 85% of the calibrated estimate, and report day numbers.

:::

:::{admonition} Solution
:class: tip, dropdown

Freeze the threshold calibration before checking holdout observations.

```{code-cell} python
import numpy as np
observed=np.array([9,10,9,10,9,7,8,9])
expected=np.full(8,10)
factor=observed[:4].sum()/expected[:4].sum()
flagged=np.where(observed[4:]<.85*factor*expected[4:])[0]+5
print("Calibration factor:", factor, "flagged days:", flagged.tolist())
assert np.isclose(factor,.95) and flagged.tolist()==[6,7]
```
:::

:::{admonition} Quiz 19 — Evaluate battery charging losses
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@foster2010; @pvlibDocs].

Use PV AC powers [0,5,3,0] kW and demand 2 kW, each held constant over its one-hour interval. The battery starts empty, holds 3 kWh, and has 90% efficiency each way with 2 kW charge and discharge caps measured at the AC bus (power entering the battery system and power delivered to the load, respectively). Reconcile PV, served energy, remaining charge, curtailment and losses.

:::

:::{admonition} Solution
:class: tip, dropdown

The remaining state of charge is stored energy and is not already served load.

```{code-cell} python
import numpy as np
state=0.; totals=np.zeros(4)
for pv in [0,5,3,0]:
    direct=min(pv,2)
    charge=min(2,pv-direct,(3-state)/.9)
    discharge=min(2,2-direct,.9*state)
    state+=.9*charge-discharge/.9
    totals += [direct+discharge,pv-direct-charge,2-direct-discharge,.1*charge+discharge*(1/.9-1)]
served,curtailed,unmet,loss=totals
assert np.isclose(8,served+curtailed+loss+state) and np.isclose(8,served+unmet)
assert np.isclose(state,2.7-2/.9)
print(f"Served {served:.3f}, curtailed {curtailed:.3f}, unmet {unmet:.3f}, losses {loss:.3f}, final charge {state:.3f} kWh")
```
:::

:::{admonition} Quiz 20 — Compare lifetime energy and levelized cost
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@ifc2015], §13.2.5 for discounted cost and energy; [@foster2010], §9.4.2 (pp. 235–236) for levelized energy cost and §9.6 (pp. 238–239) for present value and discounting. The [SAM Help LCOE definition](https://samrepo.nlr.gov/help/mtf_lcoe.html) gives the discounted-cost-over-discounted-generation form used here. SAM can include additional financial costs and benefits; this exercise includes only the specified capital and O&M costs.

Compare Case A and Case B arrays, respectively, with first-year energies 1000 and 970 MWh, degradation 1% and 0.2%, capital €600,000 and €630,000, and common annual O&M €8,000 over 20 years. Assume initial capital is paid at year zero and each year's O&M and generated energy are discounted at year-end. Compute discounted LCOE at 4%. Report the better case by both lifetime MWh and cost; LCOE is a cost metric, not investment return.

:::

:::{admonition} Solution
:class: tip, dropdown

The timing of energy and cost can change a ranking based on undiscounted generation.

```{code-cell} python
import numpy as np
def project(first, degradation, capital):
    year=np.arange(1,21)
    energy=first*(1-degradation)**(year-1)
    discount=1.04**year
    return energy.sum(),(capital+sum(8000/discount))/sum(energy/discount)

a=project(1000,.01,600000); b=project(970,.002,630000)
print(f"Case A: lifetime {a[0]:.2f} MWh; LCOE €{a[1]:.2f}/MWh")
print(f"Case B: lifetime {b[0]:.2f} MWh; LCOE €{b[1]:.2f}/MWh")
print("More lifetime energy: Case B; lower LCOE: Case A")
assert b[0]>a[0] and a[1]<b[1]
```
:::

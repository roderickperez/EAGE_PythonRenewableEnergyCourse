---
kernelspec:
  name: python3
  display_name: Python 3
---

# Solar PV Quiz and Python Challenge

These eight questions progress from units and concepts to functions, arrays, validation, and plots.

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its solution. Where a prompt refers to an earlier answer, run that code first.

## Basic

:::{admonition} Quiz 1 — Irradiance or irradiation?
:class: note

**Difficulty:** Basic

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

**Difficulty:** Basic

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

**Difficulty:** Basic

**Reference:** [@foster2010; @pvlibDocs].

Write a function to convert Wh/m² to kWh/m². Test 750 Wh/m².

:::

:::{admonition} Solution
:class: tip, dropdown

Irradiation is energy per area, rather than power per area.

```{code-cell} python
def wh_to_kwh(wh_per_m2):
    return wh_per_m2/1000

assert wh_to_kwh(750)==.75
```
:::

:::{admonition} Quiz 4 — Convert efficiency percentages
:class: note

**Difficulty:** Basic

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

assert 600*2*fraction(19)==228
```
:::

## Medium

:::{admonition} Quiz 5 — Build the temperature-aware function
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 6 — Plot a solar day
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 7 — Find inverter clipping
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 8 — Find irradiance on a horizontal panel
:class: note

**Difficulty:** Medium

**Reference:** [@foster2010; @pvlibDocs].

Write a function for GHI = DNI cos(zenith) + DHI. Test DNI 700 W/m², DHI 120 W/m² and 60° zenith.

:::

:::{admonition} Solution
:class: tip, dropdown

Python's cosine takes radians. The direct component uses the projection onto the horizontal surface.

```{code-cell} python
from math import cos,radians,isclose
def ghi(dni,dhi,zenith):
    return dni*max(0,cos(radians(zenith)))+dhi

assert isclose(ghi(700,120,60),470)
```
:::

:::{admonition} Quiz 9 — Compute clipped quarter-hour energy
:class: note

**Difficulty:** Medium

**Reference:** [@foster2010; @pvlibDocs].

Use AC potential [1,4,6,2] kW with a 3 kW cap. Write a function to return delivered and clipped kWh from kW and minutes.

:::

:::{admonition} Solution
:class: tip, dropdown

Keep the conversion from minutes to hours outside the power cap.

```{code-cell} python
def capped_energy(potential, cap, minutes):
    output=[min(value,cap) for value in potential]
    return sum(output)*minutes/60,(sum(potential)-sum(output))*minutes/60

assert capped_energy([1,4,6,2],3,15)==(2.25,1.0)
```
:::

:::{admonition} Quiz 10 — Separate optical and wiring losses
:class: note

**Difficulty:** Medium

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
assert ideal==360 and abs(output-335.16)<1e-8
```
:::

:::{admonition} Quiz 11 — Assess the units of a monitoring series
:class: note

**Difficulty:** Medium

**Reference:** [@foster2010; @pvlibDocs].

Read four half-hour AC samples [0.3,0.6,1.2,0.9] kW. Write a function for kWh and time-average kW, and compare the two outputs.

:::

:::{admonition} Solution
:class: tip, dropdown

Divide by the total observed hours to recover average power.

```{code-cell} python
def integrate_kw(power,minutes):
    hours=len(power)*minutes/60
    energy=sum(power)*minutes/60
    return energy,energy/hours

assert all(abs(a-b)<1e-9 for a,b in zip(integrate_kw([.3,.6,1.2,.9],30),[1.5,.75]))
```
:::

:::{admonition} Quiz 12 — Find a module's fill factor
:class: note

**Difficulty:** Medium

**Reference:** [@foster2010; @pvlibDocs].

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

assert .74<fill_factor(38,9,46,10)<.75
```
:::

## Hard

:::{admonition} Quiz 13 — Validate monthly performance ratio
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 14 — Temperature–irradiance map
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 15 — Lifetime degradation
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 16 — Join inverter and irradiance readings
:class: note

**Difficulty:** Hard

**Reference:** [@foster2010; @pvlibDocs].

Use hourly UTC irradiance at 10, 11, 12 with [500,700,900] W/m² and AC reports at 11, 12, 13 with [2,3,1] kW. Validate uniqueness, keep unmatched rows, and compute performance ratio only over the paired intervals for a 5 kW DC system.

:::

:::{admonition} Solution
:class: tip, dropdown

Both yield denominators must use the same hours.

```{code-cell} python
import pandas as pd
t=pd.date_range('2025-06-01 10:00',periods=4,freq='h',tz='UTC')
g=pd.Series([500,700,900],index=t[:3],name='g')
ac=pd.Series([2,3,1],index=t[1:],name='ac')
assert g.index.is_unique and ac.index.is_unique
joined=pd.concat([g,ac],axis=1)
paired=joined.dropna()
pr=(paired.ac.sum()/5)/(paired.g.sum()/1000)
assert len(joined)==4 and len(paired)==2 and abs(pr-.625)<1e-12
```
:::

:::{admonition} Quiz 17 — Select an inverter under a clipping target
:class: note

**Difficulty:** Hard

**Reference:** [@foster2010; @pvlibDocs].

For DC output [1,3,6,8,6,2] kW with durations [1,1,0.5,0.5,1,1] h and 95% inverter efficiency, find the smallest candidate cap [4,5,6,7,8] kW retaining at least 95% of potential AC energy.

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
assert choice==6 and retained(5)<.95
```
:::

:::{admonition} Quiz 18 — Back-test a performance drift rule
:class: note

**Difficulty:** Hard

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
assert np.isclose(factor,.95) and flagged.tolist()==[6,7]
```
:::

:::{admonition} Quiz 19 — Evaluate battery charging losses
:class: note

**Difficulty:** Hard

**Reference:** [@foster2010; @pvlibDocs].

Use PV [0,5,3,0] kW and constant demand 2 kW in one-hour steps. The battery starts empty, holds 3 kWh, and has 90% efficiency each way with a 2 kW cap in both directions. Reconcile PV, served energy, remaining charge, curtailment and losses.

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
```
:::

:::{admonition} Quiz 20 — Compare lifetime energy and financial yield
:class: note

**Difficulty:** Hard

**Reference:** [@foster2010; @pvlibDocs].

Compare arrays with first-year energies 1000 and 970 MWh, degradation 1% and 0.2%, capital €600,000 and €630,000, and common annual O&M €8,000 over 20 years. Compute discounted LCOE at 4%. Report the better case by both lifetime MWh and cost.

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
print(a,b,'MWh and EUR/MWh')
assert b[0]>a[0] and a[1]<b[1]
```
:::

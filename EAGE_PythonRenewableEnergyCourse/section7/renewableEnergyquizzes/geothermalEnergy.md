---
kernelspec:
  name: python3
  display_name: Python 3
---

# Geothermal Energy Quiz

These questions connect thermodynamics and subsurface concepts to Python.

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its solution. Where a prompt refers to an earlier answer, run that code first.

## Basic

:::{admonition} Quiz 1 — Gradient, heat flux, or power?
:class: note

**Difficulty:** Basic

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Match the quantities to units: geothermal gradient, conductive heat flux, thermal power, and electrical energy.

Units: W, W/m², °C/km, MWh.

:::

:::{admonition} Solution
:class: tip, dropdown

- Geothermal gradient: °C/km.
- Conductive heat flux: W/m².
- Thermal power: W.
- Electrical energy: MWh.
:::

:::{admonition} Quiz 2 — Minimum depth
:class: note

**Difficulty:** Basic

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

With surface temperature 12 °C and a 34 °C/km linear gradient, calculate temperature at 3.5 km and the depth at which the linear model reaches 160 °C. State one limitation.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
surface_c, gradient_c_km = 12.0, 34.0
temperature_3_5km = surface_c + gradient_c_km * 3.5
target_depth_km = (160 - surface_c) / gradient_c_km
print(f"T(3.5 km) = {temperature_3_5km:.1f} °C")
print(f"Depth for 160 °C = {target_depth_km:.2f} km")
```

The constant gradient is a simplified extrapolation and does not establish flow or permeability.
:::

:::{admonition} Quiz 3 — Convert heat-flow units
:class: note

**Difficulty:** Basic

**Reference:** [@grant2011].

Write a function converting kW thermal to MW thermal. Test 12,500 kW thermal.

:::

:::{admonition} Solution
:class: tip, dropdown

Keep the thermal label to distinguish extracted heat from electricity.

```{code-cell} python
def kw_to_mw(power_kw):
    return power_kw/1000

assert kw_to_mw(12500)==12.5
```
:::

:::{admonition} Quiz 4 — Convert a temperature to Kelvin
:class: note

**Difficulty:** Basic

**Reference:** [@grant2011].

Write a Celsius-to-Kelvin function. Test 85 °C and show that a 10 °C temperature difference has the same size in kelvin.

:::

:::{admonition} Solution
:class: tip, dropdown

Absolute temperatures change origin; temperature differences do not.

```{code-cell} python
from math import isclose
def to_kelvin(celsius):
    return celsius+273.15

assert isclose(to_kelvin(85),358.15) and isclose(to_kelvin(95)-to_kelvin(85),10)
```
:::

## Medium

:::{admonition} Quiz 5 — Write a well-power function
:class: note

**Difficulty:** Medium

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Write a function returning thermal, gross, and net MW using $\dot m c_p\Delta T$, conversion efficiency, and parasitic fraction. Validate inputs and test the ordering of outputs.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def well_power_mw(mass_flow, production_c, reinjection_c,
                  conversion_efficiency, parasitic_fraction=0.1, cp=4180):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [mass_flow, production_c, reinjection_c, conversion_efficiency, parasitic_fraction, cp]):
        raise ValueError("model inputs must be finite")
    mass_flow = np.asarray(mass_flow, dtype=float)
    delta_t = np.asarray(production_c, dtype=float) - reinjection_c
    if np.any(mass_flow < 0) or np.any(delta_t < 0):
        raise ValueError("mass flow and temperature difference must be non-negative")
    if not 0 <= conversion_efficiency <= 1 or not 0 <= parasitic_fraction < 1:
        raise ValueError("invalid efficiency or parasitic fraction")
    if cp <= 0:
        raise ValueError("specific heat must be positive")
    thermal = mass_flow * cp * delta_t / 1e6
    gross = thermal * conversion_efficiency
    net = gross * (1 - parasitic_fraction)
    return thermal, gross, net

thermal, gross, net = well_power_mw(80, 165, 70, 0.13, 0.12)
assert thermal >= gross >= net >= 0
print(f"Thermal={thermal:.2f} MW, gross={gross:.2f} MW, net={net:.2f} MW")
```
:::

:::{admonition} Quiz 6 — Carnot bound in Kelvin
:class: note

**Difficulty:** Medium

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Plot Carnot efficiency for hot-source temperatures 100–250 °C and a 25 °C sink. Also calculate the incorrect Celsius-ratio result and show the difference.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import matplotlib.pyplot as plt

hot_c = np.linspace(100, 250, 100)
cold_c = 25
carnot = 1 - (cold_c + 273.15) / (hot_c + 273.15)
incorrect = 1 - cold_c / hot_c

plt.figure(figsize=(8, 4))
plt.plot(hot_c, carnot, label="Correct Kelvin ratio")
plt.plot(hot_c, incorrect, linestyle="--", label="Incorrect Celsius ratio")
plt.xlabel("Hot-source temperature (°C)")
plt.ylabel("Efficiency bound")
plt.title("Temperature ratios require an absolute scale")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.all((carnot > 0) & (carnot < 1))
```
:::

:::{admonition} Quiz 7 — Analyze a well field
:class: note

**Difficulty:** Medium

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Create a DataFrame for four wells with mass flows `[62, 78, 55, 90]` kg/s and temperatures `[150, 168, 142, 175]` °C. Calculate net output by well, total it, and create a bar chart.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import pandas as pd

wells = pd.DataFrame({
    "well": ["A", "B", "C", "D"],
    "mass_flow_kgs": [62, 78, 55, 90],
    "production_c": [150, 168, 142, 175],
})
_, _, wells["net_mw"] = well_power_mw(
    wells["mass_flow_kgs"], wells["production_c"],
    reinjection_c=70, conversion_efficiency=0.12, parasitic_fraction=0.11,
)
wells.plot.bar(x="well", y="net_mw", figsize=(7, 4), legend=False, color="firebrick")
plt.ylabel("Net power (MW)")
plt.title(f"Well-field net power = {wells['net_mw'].sum():.1f} MW")
plt.tight_layout()
assert wells["net_mw"].ge(0).all()
```
:::

:::{admonition} Quiz 8 — Convert Darcy quantities
:class: note

**Difficulty:** Medium

**Reference:** [@grant2011].

Write functions to turn 50 mD into m² and 1.4 mPa·s into Pa·s. Use a gradient magnitude of 2 kPa/m to estimate Darcy flux magnitude.

:::

:::{admonition} Solution
:class: tip, dropdown

Darcy flux is volumetric discharge per bulk cross-sectional area.

```{code-cell} python
from math import isclose
def md_to_m2(md):
    return md*9.869233e-16

def mpa_s_to_pa_s(value):
    return value/1000

flux=md_to_m2(50)*2000/mpa_s_to_pa_s(1.4)
assert isclose(flux,7.049452142857e-8,rel_tol=1e-9)
```
:::

:::{admonition} Quiz 9 — Estimate single-phase enthalpy drop
:class: note

**Difficulty:** Medium

**Reference:** [@grant2011].

For brine cooling 155 to 85 °C, use 4.1 kJ/(kg·K) and 45 kg/s. Write a function returning kJ/kg and MW thermal.

:::

:::{admonition} Solution
:class: tip, dropdown

kg/s times kJ/kg gives kW thermal.

```{code-cell} python
def recovered_heat(cp,hot,cold,flow):
    dh=cp*(hot-cold)
    return dh,flow*dh/1000

assert recovered_heat(4.1,155,85,45)==(287.,12.915)
```
:::

:::{admonition} Quiz 10 — Estimate hydraulic pumping demand
:class: note

**Difficulty:** Medium

**Reference:** [@grant2011].

Given 70 kg/s, fluid density 950 kg/m³, 1.8 MPa rise and 75% pump efficiency, return m³/s and MW consumed from a function.

:::

:::{admonition} Solution
:class: tip, dropdown

MPa × m³/s has units of MW; the pump consumes more than its hydraulic output.

```{code-cell} python
def pump_mw(flow,rho,pressure_mpa,efficiency):
    q=flow/rho
    return q,pressure_mpa*q/efficiency

q,power=pump_mw(70,950,1.8,.75)
assert abs(q-.07368421)<1e-8 and .176<power<.178
```
:::

:::{admonition} Quiz 11 — Distinguish heat pump COP from efficiency
:class: note

**Difficulty:** Medium

**Reference:** [@grant2011].

For heating targets [3,6,9] MW thermal at COP 3, use a function to return electrical demand and heat moved from the source. Verify the first-law balance.

:::

:::{admonition} Solution
:class: tip, dropdown

A COP above one includes heat moved from the source.

```{code-cell} python
def heat_pump(heat,cop):
    electrical=heat/cop
    return electrical,heat-electrical

assert [heat_pump(heat,3) for heat in [3,6,9]]==[(1,2),(2,4),(3,6)]
```
:::

:::{admonition} Quiz 12 — Calculate availability-adjusted annual energy
:class: note

**Difficulty:** Medium

**Reference:** [@grant2011].

Calculate operating hours and annual MWh for a steady 4 MW net plant at 88%, 95%, and 100% availability in a non-leap year. Use one function and verify the denominator for capacity factor if net rating is 5 MW.

:::

:::{admonition} Solution
:class: tip, dropdown

Operating power and rated power differ; capacity factor includes time and utilisation.

```{code-cell} python
def annual(power,availability):
    hours=8760*availability
    energy=power*hours
    return energy,energy/(5*8760)

assert abs(annual(4,.95)[0]-33288)<1e-9
assert abs(annual(4,1)[1]-.8)<1e-9
```
:::

## Hard

:::{admonition} Quiz 13 — Sensitivity surface
:class: note

**Difficulty:** Hard

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

For an 80 kg/s well at 165/70 °C, vary conversion efficiency from 8–18% and parasitic fraction from 5–20%. Plot net power with `contourf`.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
efficiency_axis = np.linspace(0.08, 0.18, 60)
parasitic_axis = np.linspace(0.05, 0.20, 50)
ETA, PAR = np.meshgrid(efficiency_axis, parasitic_axis)
thermal_mw = 80 * 4180 * (165 - 70) / 1e6
NET = thermal_mw * ETA * (1 - PAR)

plt.figure(figsize=(8, 5))
contour = plt.contourf(ETA * 100, PAR * 100, NET, levels=20, cmap="inferno")
plt.colorbar(contour, label="Net power (MW)")
plt.xlabel("Conversion efficiency (%)")
plt.ylabel("Parasitic fraction (%)")
plt.title("Geothermal net-power sensitivity")
plt.tight_layout()
```
:::

:::{admonition} Quiz 14 — Decline, availability, and energy
:class: note

**Difficulty:** Hard

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

For 12 MW initial net power, annual decline 1.5%, and availability 94%, calculate annual GWh over 25 years and cumulative energy. Plot both on separate axes.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
years = np.arange(1, 26)
net_power_mw = 12 * (1 - 0.015) ** (years - 1)
annual_gwh = net_power_mw * 8760 * 0.94 / 1000
cumulative_gwh = np.cumsum(annual_gwh)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(years, annual_gwh, marker="o", markersize=3)
axes[0].set(xlabel="Year", ylabel="Annual energy (GWh)", title="Annual output")
axes[1].plot(years, cumulative_gwh, color="darkorange")
axes[1].set(xlabel="Year", ylabel="Cumulative energy (GWh)", title="Cumulative output")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
assert np.all(np.diff(annual_gwh) < 0) and np.all(np.diff(cumulative_gwh) > 0)
```
:::

:::{admonition} Quiz 15 — Detect invalid observations
:class: note

**Difficulty:** Hard

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Add two faulty rows to the well table: a negative flow and production below reinjection temperature. Write a validation function that returns problem rows rather than silently calculating negative power.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
faulty = pd.DataFrame({
    "well": ["E", "F"], "mass_flow_kgs": [-5, 40], "production_c": [160, 60]
})
audit = pd.concat([wells[["well", "mass_flow_kgs", "production_c"]], faulty], ignore_index=True)

def invalid_wells(frame, reinjection_c):
    invalid = (frame["mass_flow_kgs"] < 0) | (frame["production_c"] <= reinjection_c)
    return frame.loc[invalid].copy()

problems = invalid_wells(audit, 70)
print(problems.to_string(index=False))
assert set(problems["well"]) == {"E", "F"}
```
:::

:::{admonition} Quiz 16 — Optimise export under pumping and rating constraints
:class: note

**Difficulty:** Hard

**Reference:** [@grant2011].

Compare 40, 60, 80 and 100 kg/s with 85 K drop, cp 4180 J/(kg·K), 12% gross conversion and pumping penalty 0.0004ṁ² MW. Cap net export at 2 MW. Select the best candidate and report whether any candidate consumes more than it generates.

:::

:::{admonition} Solution
:class: tip, dropdown

A rising pumping penalty can outweigh additional recovered heat.

```{code-cell} python
def case(flow):
    gross=flow*4180*85*.12/1e6
    net=gross-.0004*flow**2
    return min(2,max(0,net)),net

cases={q:case(q) for q in [40,60,80,100]}
best=max(cases,key=lambda q:cases[q][0])
assert best==60 and all(net>0 for _,net in cases.values())
```
:::

:::{admonition} Quiz 17 — Share plant capacity across wells
:class: note

**Difficulty:** Hard

**Reference:** [@grant2011].

For two wells with flows [65,75] kg/s and temperature drops [80,100] K, cp 4180, conversion 12%, parasitic 10%, and a shared 4 MW net cap, compare all-on and each outage. Reconcile potential, exported and clipped MW.

:::

:::{admonition} Solution
:class: tip, dropdown

Apply the cap at the shared plant boundary after summing available well power.

```{code-cell} python
import numpy as np
potential=np.array([65,75])*np.array([80,100])*4180*.12*.9/1e6
def field(online):
    net=potential[online].sum()
    return min(4,net),max(0,net-4)

full,clipped=field([0,1])
assert full==4 and np.isclose(full+clipped,potential.sum())
assert all(field([i])[0]<full and field([i])[1]==0 for i in range(2))
```
:::

:::{admonition} Quiz 18 — Search reinjection scenarios with an export floor
:class: note

**Difficulty:** Hard

**Reference:** [@grant2011].

A 55 kg/s well produces at 145 °C. For reinjection candidates [60,70,80,90] °C, use cp 4180 J/(kg·K), gross conversion 11%, fixed pump load 0.25 MW and 1.5 MW export requirement. Find the warmest feasible reinjection temperature.

:::

:::{admonition} Solution
:class: tip, dropdown

Choosing the warmest scenario satisfying the constraint can reduce heat extraction.

```{code-cell} python
def net(cold):
    return 55*4180*(145-cold)*.11/1e6-.25

options=[60,70,80,90]
choice=max(t for t in options if net(t)>=1.5)
assert choice==70 and net(80)<1.5
```
:::

:::{admonition} Quiz 19 — Quantify a two-fluid-model discrepancy
:class: note

**Difficulty:** Hard

**Reference:** [@grant2011].

For flow 55 kg/s, compare a recorded enthalpy drop of 480 kJ/kg with a liquid approximation from ΔT = 70 K and cp = 4.18 kJ/(kg·K). Calculate thermal MW, net MW at .12 conversion and .10 gross parasitics, and the percent underestimation relative to the enthalpy case.

:::

:::{admonition} Solution
:class: tip, dropdown

If field states include phase change, measured or calculated enthalpy is the appropriate energy basis.

```{code-cell} python
from math import isclose
def boundary(dh_kj_kg):
    thermal=55*dh_kj_kg/1000
    return thermal,thermal*.12*.9

reference=boundary(480); liquid=boundary(70*4.18)
bias=1-liquid[1]/reference[1]
assert isclose(reference[0],26.4) and .39<bias<.40
```
:::

:::{admonition} Quiz 20 — Evaluate a reinjection intervention with downtime
:class: note

**Difficulty:** Hard

**Reference:** [@grant2011].

A plant begins at 5 MW net and declines 3% each year. Compare 12-year GWh at 95% availability with and without a reset to 5 MW at the start of year 7. Intervention costs 500 otherwise operating hours in year 7. Verify the first six years match.

:::

:::{admonition} Solution
:class: tip, dropdown

The reset boosts operating output; intervention downtime reduces the year-seven energy benefit.

```{code-cell} python
import numpy as np
year=np.arange(1,13)
hours=np.full(12,8760*.95)
base=5*.97**(year-1)*hours/1000
restored_age=np.where(year<7,year-1,year-7)
hours[6]-=500
reset=5*.97**restored_age*hours/1000
assert np.allclose(base[:6],reset[:6]) and reset.sum()>base.sum()
print(base.sum(),reset.sum(),'GWh')
```
:::

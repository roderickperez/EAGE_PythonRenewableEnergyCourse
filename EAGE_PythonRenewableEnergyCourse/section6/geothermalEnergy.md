---
kernelspec:
  name: python3
  display_name: Python 3
---

# Geothermal Energy

## Learning goals

After this lesson you should be able to:

- distinguish geothermal gradient, heat flux, thermal power, gross electrical power, and net electrical power;
- use Kelvin correctly in thermodynamic efficiency calculations;
- estimate heat carried by produced fluid;
- include conversion efficiency, parasitic load, and production decline;
- implement validated Python functions and scenario plots;
- explain why temperature alone does not establish a viable geothermal resource.

Electricity generation requires useful heat plus a way to transport it to the surface. Conventional hydrothermal systems rely on heat, fluid, and permeability. Enhanced geothermal systems create or improve fluid pathways, while closed-loop systems circulate fluid in sealed wells [@doeGeothermalElectricity; @doeEGS]. Dry steam, flash steam, and binary-cycle plants use different surface processes and operating ranges.

## Concepts and equations

### Geothermal gradient is not heat flux

A linear temperature-depth approximation is

$$T(z)=T_0+G_Tz$$

where $G_T$ is the geothermal gradient, for example °C/km. This is a local approximation; gradients vary with geology, groundwater flow, and depth.

Conductive heat flux follows Fourier's law:

$$\mathbf{q}=-k\nabla T$$

The negative sign indicates heat flows toward lower temperature. Gradient (K/m) and heat flux (W/m$^2$) are different quantities.

### Thermal power in produced fluid

For mass-flow rate $\dot m$, specific heat $c_p$, production temperature $T_p$, and reinjection temperature $T_r$:

$$P_{th}=\dot m c_p(T_p-T_r)$$

If $c_p$ is J/(kg K), the result is W. A constant liquid-water $c_p$ is a simplified sensible-heat model; steam, flashing, brines, and phase change require enthalpy:

$$P_{th}=\dot m(h_p-h_r)$$

### Efficiency, parasitic load, and net power

The reversible Carnot bound is

$$\eta_C=1-\frac{T_c}{T_h}$$

where both temperatures must be in Kelvin. A plant cannot simply be assigned Carnot efficiency; real conversion efficiency is lower and technology-specific.

$$P_{gross}=\eta_{conv}P_{th}$$

$$P_{net}=P_{gross}-P_{pumps}-P_{aux}$$

Net power, not gross generator power, is the relevant exported power.

### Energy and decline

For discrete time steps:

$$E_{net}=\sum_iP_{net,i}\Delta t_i$$

A simple annual decline scenario is

$$P_y=P_1(1-d)^{y-1}$$

This is a scenario model, not a reservoir forecast. Sustainable production depends on reservoir response, injection strategy, pressure, thermal breakthrough, and operational constraints.

## Tested Python functions

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def temperature_at_depth_c(depth_km, surface_c=15.0, gradient_c_per_km=30.0):
    depth = np.asarray(depth_km, dtype=float)
    if np.any(depth < 0) or gradient_c_per_km < 0:
        raise ValueError("depth and gradient must be non-negative")
    return surface_c + gradient_c_per_km * depth

def geothermal_net_power_mw(mass_flow_kgs, production_c, reinjection_c,
                            conversion_efficiency, parasitic_fraction=0.10,
                            cp_jkgk=4180.0):
    """Simplified single-phase sensible-heat model."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [mass_flow_kgs, production_c, reinjection_c, conversion_efficiency, parasitic_fraction, cp_jkgk]):
        raise ValueError("model inputs must be finite")
    mass_flow = np.asarray(mass_flow_kgs, dtype=float)
    delta_t = np.asarray(production_c, dtype=float) - reinjection_c
    if np.any(mass_flow < 0) or np.any(delta_t < 0):
        raise ValueError("mass flow and production-minus-reinjection temperature must be non-negative")
    if not 0 <= conversion_efficiency <= 1:
        raise ValueError("conversion_efficiency must be between 0 and 1")
    if not 0 <= parasitic_fraction < 1:
        raise ValueError("parasitic_fraction must be in [0, 1)")
    if cp_jkgk <= 0:
        raise ValueError("specific heat must be positive")
    thermal_mw = mass_flow * cp_jkgk * delta_t / 1e6
    gross_mw = thermal_mw * conversion_efficiency
    net_mw = gross_mw * (1 - parasitic_fraction)
    return float(net_mw) if np.ndim(net_mw) == 0 else net_mw

def carnot_efficiency(hot_c, cold_c):
    hot_k = np.asarray(hot_c, dtype=float) + 273.15
    cold_k = np.asarray(cold_c, dtype=float) + 273.15
    if np.any(hot_k <= cold_k) or np.any(cold_k <= 0):
        raise ValueError("hot temperature must exceed cold temperature in Kelvin")
    return 1 - cold_k / hot_k

assert temperature_at_depth_c(0) == 15
assert geothermal_net_power_mw(0, 160, 70, 0.12) == 0
assert 0 < carnot_efficiency(150, 25) < 1
```

## Worked well-field example

```{code-cell} python
years = np.arange(1, 31)
initial_mass_flow = 90.0
decline_rates = [0.00, 0.01, 0.02]

fig, ax = plt.subplots(figsize=(9, 4.5))
for decline in decline_rates:
    flow = initial_mass_flow * (1 - decline) ** (years - 1)
    net_power = geothermal_net_power_mw(
        flow, production_c=165, reinjection_c=70,
        conversion_efficiency=0.13, parasitic_fraction=0.12,
    )
    ax.plot(years, net_power, marker="o", markersize=3,
            label=f"{decline:.0%} annual flow decline")

ax.set(xlabel="Operating year", ylabel="Net power (MW)",
       title="Simplified geothermal decline scenarios")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
```

The plot varies only mass flow. In a real reservoir, temperature, pressure, scaling, downtime, reinjection response, and parasitic pumping may also change.

## Guided exercises

:::{admonition} Exercise 1 — Depth-temperature scenarios
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Plot profiles from 0–6 km for gradients 20, 30, and 45 °C/km. Write a function returning minimum depth for a target temperature and reject a zero gradient.

```python
def depth_for_temperature(target_c, surface_c, gradient_c_per_km):
    # TODO: validate the gradient and calculate depth
    pass
```
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

1. Calculate linear temperature profiles.
2. Invert the equation for depth.
3. Reject non-positive gradients.

```{code-cell} python
def depth_for_temperature(target_c, surface_c, gradient_c_per_km):
    if gradient_c_per_km <= 0:
        raise ValueError("gradient must be positive")
    return max((target_c - surface_c) / gradient_c_per_km, 0)

depths_km = np.linspace(0, 6, 100)
plt.figure(figsize=(7, 5))
for gradient in [20, 30, 45]:
    plt.plot(temperature_at_depth_c(depths_km, 15, gradient), depths_km, label=f"{gradient} °C/km")
plt.gca().invert_yaxis()
plt.xlabel("Temperature (°C)")
plt.ylabel("Depth (km)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.isclose(depth_for_temperature(150, 15, 30), 4.5)
```
:::

:::{admonition} Exercise 2 — Thermal, gross, and net power
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

For 75 kg/s, 155/65 °C, 12% conversion efficiency, and 10% parasitic fraction, calculate all three power stages and validate their ordering.

```python
# TODO: calculate thermal_mw, gross_mw, and net_mw separately
# TODO: print a formatted energy-balance table
```
:::

:::{admonition} Exercise 2 — Solution
:class: tip, dropdown

1. Calculate thermal power first.
2. Apply electrical efficiency.
3. Subtract the parasitic fraction of gross power.

```{code-cell} python
thermal_mw = 75 * 4180 * (155 - 65) / 1e6
gross_mw = thermal_mw * 0.12
net_mw = gross_mw * (1 - 0.10)
print(f"Thermal {thermal_mw:6.2f} MW\nGross   {gross_mw:6.2f} MW\nNet     {net_mw:6.2f} MW")
assert thermal_mw >= gross_mw >= net_mw >= 0
```
:::

:::{admonition} Exercise 3 — Kelvin check
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Plot Carnot efficiency from 100–250 °C for a 25 °C sink and a practical scenario equal to 45% of the bound. Explain why Celsius ratios are invalid.

```python
# TODO: calculate the Carnot bound with carnot_efficiency
# TODO: calculate and plot the practical scenario
```
:::

:::{admonition} Exercise 3 — Solution
:class: tip, dropdown

1. Convert both temperatures to kelvin.
2. Evaluate the reversible bound.
3. Multiply by the assumed practical fraction.

```{code-cell} python
hot_source_c = np.linspace(100, 250, 100)
carnot_bound = carnot_efficiency(hot_source_c, 25)
practical_scenario = 0.45 * carnot_bound
plt.figure(figsize=(8, 4))
plt.plot(hot_source_c, carnot_bound, label="Carnot bound")
plt.plot(hot_source_c, practical_scenario, label="45% of bound")
plt.xlabel("Hot-source temperature (°C)")
plt.ylabel("Efficiency")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.all(practical_scenario < carnot_bound)
```
:::

:::{admonition} Exercise 4 — Multiple wells
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Store well flow and temperature in a DataFrame, calculate net MW by well, flag invalid temperatures, and plot valid output.

```python
# TODO: create a well DataFrame
# TODO: validate production > reinjection
# TODO: calculate and plot net power
```
:::

:::{admonition} Exercise 4 — Solution
:class: tip, dropdown

1. Build and validate the well table.
2. Leave invalid results missing.
3. Compute and plot only valid wells.

```{code-cell} python
import pandas as pd

wells = pd.DataFrame({"well": ["A", "B", "C", "D"],
                      "flow_kgs": [62, 78, 55, 90],
                      "production_c": [150, 168, 60, 175]})
wells["valid"] = wells["production_c"] > 70
wells["net_mw"] = np.nan
valid = wells["valid"]
wells.loc[valid, "net_mw"] = geothermal_net_power_mw(wells.loc[valid, "flow_kgs"], wells.loc[valid, "production_c"], 70, 0.12, 0.11)
wells.loc[valid].plot.bar(x="well", y="net_mw", legend=False, color="firebrick", figsize=(7, 4))
plt.ylabel("Net power (MW)")
plt.tight_layout()
print(wells)
assert wells.loc[valid, "net_mw"].ge(0).all()
assert wells.loc[~valid, "net_mw"].isna().all()
```
:::

:::{admonition} Exercise 5 — Parasitic-load sensitivity
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Use `np.meshgrid` for conversion efficiency 8–18% and parasitic fraction 5–20%. Plot net power with `contourf`.

```python
# TODO: build the efficiency/parasitic grid
# TODO: calculate and plot net power
```
:::

:::{admonition} Exercise 5 — Solution
:class: tip, dropdown

1. Create an efficiency/parasitic grid.
2. Calculate thermal MW once.
3. Multiply by conversion and retained fractions.

```{code-cell} python
conversion_axis = np.linspace(0.08, 0.18, 60)
parasitic_axis = np.linspace(0.05, 0.20, 50)
conversion_grid, parasitic_grid = np.meshgrid(conversion_axis, parasitic_axis)
thermal_reference_mw = 80 * 4180 * (165 - 70) / 1e6
net_grid = thermal_reference_mw * conversion_grid * (1 - parasitic_grid)
plt.figure(figsize=(8, 5))
contour = plt.contourf(conversion_grid * 100, parasitic_grid * 100, net_grid, levels=20)
plt.colorbar(contour, label="Net power (MW)")
plt.xlabel("Conversion efficiency (%)")
plt.ylabel("Parasitic fraction (%)")
plt.tight_layout()
```
:::

:::{admonition} Exercise 6 — Decline and cumulative energy
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Calculate 30-year power and annual energy for several decline rates with 95% availability. Plot annual output and compare cumulative energy.

```python
# TODO: loop over decline scenarios
# TODO: calculate annual GWh and cumulative GWh
```
:::

:::{admonition} Exercise 6 — Solution
:class: tip, dropdown

1. Create operating years.
2. Compound decline.
3. Multiply by availability and hours and sum the energy.

```{code-cell} python
operating_years = np.arange(1, 31)
plt.figure(figsize=(8, 4))
for decline_rate in [0.00, 0.01, 0.02]:
    annual_power = 12 * (1 - decline_rate) ** (operating_years - 1)
    annual_energy_gwh = annual_power * 8760 * 0.95 / 1000
    plt.plot(operating_years, annual_energy_gwh, label=f"{decline_rate:.0%}/year")
    print(decline_rate, annual_energy_gwh.sum())
plt.xlabel("Operating year")
plt.ylabel("Annual energy (GWh)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

## Common mistakes

- Using °C rather than Kelvin in a thermodynamic temperature ratio.
- Calling geothermal gradient a heat flux.
- Ignoring phase change when enthalpy, not constant $c_p\Delta T$, is required.
- Reporting gross power while omitting pumping and auxiliary loads.
- Treating a high subsurface temperature as proof of permeability, flow, or commercial viability.
- Subtracting a fixed decline percentage linearly instead of compounding the retained fraction.

## Check your understanding

Continue with the [geothermal quiz and Python challenge](../section7/renewableEnergyquizzes/geothermalEnergy.md).

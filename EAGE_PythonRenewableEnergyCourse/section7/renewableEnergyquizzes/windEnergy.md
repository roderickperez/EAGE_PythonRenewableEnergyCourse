---
kernelspec:
  name: python3
  display_name: Python 3
---

# Wind-Energy Quiz and Python Challenge

Complete these eight cumulative questions before opening the solutions.

:::{admonition} Quiz 1 — Power concepts
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Which expression gives the kinetic-power flux through rotor area $A$?

A. $\rho Av$  
B. $\tfrac12\rho Av^2$  
C. $\tfrac12\rho Av^3$  
D. $C_p\rho Av^3$

:::

:::{admonition} Solution
:class: tip, dropdown

**C.** $\tfrac12\rho Av^3$ is available wind power. Multiplying it by $C_p$ gives aerodynamic rotor power. Electrical output includes additional losses and controls.
:::

:::{admonition} Quiz 2 — Betz reasoning
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

At fixed density and rotor area, by what factor does available power change when wind speed rises from 5 to 10 m/s? Can a turbine with $C_p=0.65$ be consistent with the ideal Betz model?

:::

:::{admonition} Solution
:class: tip, dropdown

The speed doubles, so available power changes by $2^3=8$. No: $0.65>16/27\approx0.593$.

```{code-cell} python
betz_limit = 16 / 27
power_ratio = (10 / 5) ** 3
print(f"Power ratio: {power_ratio:.0f}; Betz limit: {betz_limit:.3f}")
assert power_ratio == 8 and 0.65 > betz_limit
```
:::

:::{admonition} Quiz 3 — Implement a bounded power curve
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Write a vectorized turbine function with 3 m/s cut-in, 12 m/s rated speed, 25 m/s cut-out, and 4 MW rating. Use a cubic ramp that is continuous at cut-in and rated speed. Test all boundaries.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def turbine_power_mw(speed, rated_mw=4, cut_in=3, rated_speed=12, cut_out=25):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [speed, rated_mw, cut_in, rated_speed, cut_out]):
        raise ValueError("model inputs must be finite")
    speed = np.asarray(speed, dtype=float)
    if np.any(speed < 0) or rated_mw < 0 or not 0 <= cut_in < rated_speed < cut_out:
        raise ValueError("invalid speed or turbine thresholds")
    power = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated_speed)
    power[ramp] = rated_mw * (speed[ramp]**3 - cut_in**3) / (rated_speed**3 - cut_in**3)
    power[(speed >= rated_speed) & (speed < cut_out)] = rated_mw
    return float(power) if power.ndim == 0 else power

assert turbine_power_mw(3) == 0
assert turbine_power_mw(12) == 4
assert turbine_power_mw(24.9) == 4
assert turbine_power_mw(25) == 0
```
:::

:::{admonition} Quiz 4 — Plot and annotate operating regions
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Plot the curve from 0–30 m/s and shade the ramp, rated, and shutdown regions. Label axes in m/s and MW.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import matplotlib.pyplot as plt

speed = np.linspace(0, 30, 301)
power = turbine_power_mw(speed)
plt.figure(figsize=(9, 4))
plt.plot(speed, power, color="navy", linewidth=2)
plt.axvspan(3, 12, color="orange", alpha=0.18, label="Ramp")
plt.axvspan(12, 25, color="green", alpha=0.12, label="Rated")
plt.axvspan(25, 30, color="red", alpha=0.10, label="Shutdown")
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Electrical power (MW)")
plt.title("Simplified 4 MW turbine power curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

:::{admonition} Quiz 5 — Density and hub-height correction
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Wind speed is 6.5 m/s at 10 m. Estimate speed at 100 m for shear exponents 0.10, 0.14, and 0.25. Calculate air density at 90 kPa and 5 °C. Plot the vertical profiles.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def wind_at_height(speed_ref, height, ref_height=10, alpha=0.14):
    return np.asarray(speed_ref) * (np.asarray(height) / ref_height) ** alpha

def air_density(pressure_pa, temperature_c):
    return pressure_pa / (287.05 * (temperature_c + 273.15))

heights = np.linspace(10, 150, 100)
alphas = [0.10, 0.14, 0.25]
plt.figure(figsize=(6, 5))
for alpha in alphas:
    profile = wind_at_height(6.5, heights, alpha=alpha)
    plt.plot(profile, heights, label=f"alpha={alpha}")
    print(f"alpha={alpha}: 100 m speed = {wind_at_height(6.5, 100, alpha=alpha):.2f} m/s")
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Height (m)")
plt.title("Power-law wind-shear scenarios")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
print(f"Air density: {air_density(90000, 5):.3f} kg/m³")
```
:::

:::{admonition} Quiz 6 — AEP from a Weibull distribution
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

For shape $k=2.0$ and scale $c=8.0$ m/s, discretize 0–35 m/s into 0.25 m/s bins. Calculate AEP and capacity factor from the turbine curve. Confirm probabilities do not exceed one.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
k, c = 2.0, 8.0
edges = np.arange(0, 35.25, 0.25)
centres = (edges[:-1] + edges[1:]) / 2
cdf = lambda v: 1 - np.exp(-(v / c) ** k)
probability = cdf(edges[1:]) - cdf(edges[:-1])
bin_power = turbine_power_mw(centres)
aep_mwh = np.sum(probability * bin_power) * 8760
capacity_factor = aep_mwh / (4 * 8760)
print(f"AEP: {aep_mwh/1000:.2f} GWh; capacity factor: {capacity_factor:.1%}")
assert probability.sum() <= 1 and 0 <= capacity_factor <= 1
```
:::

:::{admonition} Quiz 7 — Compare $P(\bar v)$ with $\overline{P(v)}$
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Generate 8760 reproducible Weibull wind speeds. Compare annual energy from the hourly series with an estimate that applies the power curve only to mean wind speed. Plot the speed and power distributions. Explain the difference.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
rng = np.random.default_rng(10)
hourly_speed = rng.weibull(2.0, 8760) * 8.0
hourly_power = turbine_power_mw(hourly_speed)
energy_from_series = hourly_power.sum()
energy_from_mean = turbine_power_mw(hourly_speed.mean()) * 8760

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(hourly_speed, bins=35, color="skyblue", edgecolor="white")
axes[0].set(xlabel="Wind speed (m/s)", ylabel="Hours", title="Wind-speed distribution")
axes[1].hist(hourly_power, bins=25, color="seagreen", edgecolor="white")
axes[1].set(xlabel="Power (MW)", ylabel="Hours", title="Power distribution")
plt.tight_layout()
print(f"Hourly method: {energy_from_series/1000:.2f} GWh")
print(f"Mean-speed method: {energy_from_mean/1000:.2f} GWh")
```

The nonlinear and clipped power curve means a mean input does not preserve mean output.
:::

:::{admonition} Quiz 8 — Loss waterfall
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Starting with gross AEP, apply wake 8%, availability 4%, electrical 3%, and curtailment 2% sequentially. Write a function, calculate net AEP, and plot each stage. Explain why summing the percentages is only an approximation.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def apply_sequential_losses(gross_energy, losses):
    stages = {"Gross": float(gross_energy)}
    remaining = float(gross_energy)
    for name, fraction in losses.items():
        if not 0 <= fraction < 1:
            raise ValueError("loss fractions must be in [0, 1)")
        remaining *= 1 - fraction
        stages[f"After {name}"] = remaining
    return stages

stages = apply_sequential_losses(aep_mwh, {
    "wake": 0.08, "availability": 0.04,
    "electrical": 0.03, "curtailment": 0.02,
})
plt.figure(figsize=(9, 4))
plt.bar(stages.keys(), np.array(list(stages.values())) / 1000, color="slateblue")
plt.xticks(rotation=25, ha="right")
plt.ylabel("Annual energy (GWh)")
plt.title("Sequential wind-energy losses")
plt.tight_layout()
assert list(stages.values())[-1] <= aep_mwh
```

Each percentage acts on the energy remaining after prior losses, so the combined factor is multiplicative.
:::

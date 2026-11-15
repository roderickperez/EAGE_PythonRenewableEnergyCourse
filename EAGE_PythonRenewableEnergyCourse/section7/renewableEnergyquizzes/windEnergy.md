---
kernelspec:
  name: python3
  display_name: Python 3
---

# Wind-Energy Quiz and Python Challenge

Complete these eight cumulative questions before opening the solutions.

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its solution. Where a prompt refers to an earlier answer, run that code first.

## Basic

:::{admonition} Quiz 1 — Power concepts
:class: note

**Difficulty:** Basic

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

**Difficulty:** Basic

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

:::{admonition} Quiz 3 — Convert measured wind speed
:class: note

**Difficulty:** Basic

**Reference:** [@manwell2009].

Write a function to convert km/h to m/s. Check 36 km/h.

:::

:::{admonition} Solution
:class: tip, dropdown

There are 1,000 metres per kilometre and 3,600 seconds per hour.

```{code-cell} python
def kmh_to_ms(speed):
    return speed/3.6

assert kmh_to_ms(36)==10
```
:::

:::{admonition} Quiz 4 — Convert rotor speed
:class: note

**Difficulty:** Basic

**Reference:** [@manwell2009].

Convert 15 rpm to rad/s with a function. Explain why RPM cannot be used directly in P = torque × angular speed.

:::

:::{admonition} Solution
:class: tip, dropdown

Use radians per second with torque in N·m to obtain watts.

```{code-cell} python
from math import pi,isclose
def rpm_to_rad_s(rpm):
    return rpm*2*pi/60

assert isclose(rpm_to_rad_s(15),pi/2)
```
:::

## Medium

:::{admonition} Quiz 5 — Implement a bounded power curve
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 6 — Plot and annotate operating regions
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 7 — Density and hub-height correction
:class: note

**Difficulty:** Medium

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

:::{admonition} Quiz 8 — Size rotor area from diameter
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009].

Write a function returning swept area from diameter. Compare 60 and 90 m rotors and verify the area ratio.

:::

:::{admonition} Solution
:class: tip, dropdown

Area scales with the square of diameter.

```{code-cell} python
from math import pi,isclose
def swept_area(diameter):
    if diameter<=0:
        raise ValueError('Diameter must be positive')
    return pi*(diameter/2)**2

assert isclose(swept_area(90)/swept_area(60),2.25)
```
:::

:::{admonition} Quiz 9 — Evaluate tip-speed ratio
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009].

At 9 rpm and 45 m radius, calculate tip-speed ratio for 6, 9 and 12 m/s using a function. Reject zero wind speed.

:::

:::{admonition} Solution
:class: tip, dropdown

At fixed rotor speed, tip-speed ratio falls as the wind increases.

```{code-cell} python
from math import pi
def tip_ratio(rpm,radius,wind):
    if wind<=0:
        raise ValueError('Wind speed must be positive')
    return 2*pi*rpm/60*radius/wind

ratios=[tip_ratio(9,45,v) for v in [6,9,12]]
assert ratios[0]>ratios[1]>ratios[2] and abs(ratios[1]-1.5*pi)<1e-9
```
:::

:::{admonition} Quiz 10 — Identify losses in atmospheric density
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009].

Compute available wind power for ρ = 1.15 and 1.25 kg/m³ at 8 m/s through a 60 m diameter rotor. Use a function and express the reduction relative to denser air.

:::

:::{admonition} Solution
:class: tip, dropdown

This isolates density in the kinetic flux, before turbine control and clipping.

```{code-cell} python
from math import pi,isclose
def available_kw(density,diameter,speed):
    return .5*density*pi*(diameter/2)**2*speed**3/1000

loss=1-available_kw(1.15,60,8)/available_kw(1.25,60,8)
assert isclose(loss,.08)
```
:::

:::{admonition} Quiz 11 — Measure turbulence in two samples
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009].

Compare population turbulence intensity for [5,7,9] and [6,7,8] m/s using one function. Report a clear diagnostic for a zero mean.

:::

:::{admonition} Solution
:class: tip, dropdown

Equal mean wind speeds can have different variability.

```{code-cell} python
import numpy as np
def turbulence(values):
    mean=np.mean(values)
    return np.std(values,ddof=0)/mean if mean>0 else None

assert turbulence([5,7,9])>turbulence([6,7,8]) and turbulence([0,0]) is None
```
:::

:::{admonition} Quiz 12 — Estimate hourly operating states
:class: note

**Difficulty:** Medium

**Reference:** [@manwell2009].

Use speeds [2,3,8,12,24,25] m/s. Label below cut-in, ramp, rated and cut-out with 3/12/25 m/s thresholds. Explain why the ramp starts at zero.

:::

:::{admonition} Solution
:class: tip, dropdown

The continuous cubic ramp gives zero output exactly at its lower edge.

```{code-cell} python
def state(v):
    if v<3:
        return 'below cut-in'
    if v<12:
        return 'ramp'
    if v<25:
        return 'rated'
    return 'cut-out'

assert [state(v) for v in [2,3,8,12,24,25]]==['below cut-in','ramp','ramp','rated','rated','cut-out']
```
:::

## Hard

:::{admonition} Quiz 13 — AEP from a Weibull distribution
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 14 — Compare $P(\bar v)$ with $\overline{P(v)}$
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 15 — Loss waterfall
:class: note

**Difficulty:** Hard

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

:::{admonition} Quiz 16 — Integrate an outage-weighted generation histogram
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009].

A month has speed bins represented by [2,8,13,27] m/s and counts [20,40,35,5] one-hour samples. Use the earlier 4 MW curve and class-specific availability [1,.8,.95,1]. Verify gross and net energy and compare with applying one average availability.

:::

:::{admonition} Solution
:class: tip, dropdown

Availability can coincide with high- or low-power conditions; mean availability loses that information.

```{code-cell} python
import numpy as np
hours=np.array([20,40,35,5]); availability=np.array([1,.8,.95,1])
power=turbine_power_mw([2,8,13,27])
gross=power@hours
net=power@(hours*availability)
uniform=gross*np.average(availability,weights=hours)
assert 0<net<gross and not np.isclose(net,uniform)
```
:::

:::{admonition} Quiz 17 — Compare tower heights near cut-out
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009].

With 50 m speeds [6,11,16,23] m/s and durations [1,2,2,3] h, compare 50 m and 100 m using exponent 0.2 and the 4 MW curve. Show at least one interval affected by cut-out and which tower makes more energy.

:::

:::{admonition} Solution
:class: tip, dropdown

Raising the hub can increase winds past the shutdown threshold.

```{code-cell} python
import numpy as np
speeds=np.array([6,11,16,23]); dt=np.array([1,2,2,3])
low=turbine_power_mw(speeds); high=turbine_power_mw(speeds*2**.2)
assert low[-1]==4 and high[-1]==0 and low@dt>high@dt
```
:::

:::{admonition} Quiz 18 — Compare aerodynamic and electrical estimates
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009].

Compare a constant coefficient estimate with the 4 MW electrical curve at [2,8,12,26] m/s for a 60 m rotor, air density 1.22, rotor coefficient .42 and drivetrain efficiency .92. Calculate MW, flag where the uncapped estimate exceeds electrical output, and explain the shutdown discrepancy.

:::

:::{admonition} Solution
:class: tip, dropdown

A resource-based estimate ignores the electrical curve's operating boundaries.

```{code-cell} python
import numpy as np
v=np.array([2,8,12,26])
estimate=.5*1.22*np.pi*30**2*v**3*.42*.92/1e6
curve=turbine_power_mw(v)
assert np.all(estimate[[0,3]]>0) and np.all(curve[[0,3]]==0)
print(v[estimate>curve],'m/s where fixed coefficient exceeds electrical curve')
```
:::

:::{admonition} Quiz 19 — Infer shear and propagate it to energy
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009].

At heights [20,40,80] m a simultaneous profile is [5,6,7.2] m/s. Fit a power-law shear exponent using log-linear regression, predict 120 m wind, and use the 4 MW curve to estimate six-hour MWh. Compare against reusing the 80 m speed.

:::

:::{admonition} Solution
:class: tip, dropdown

The power-law fit assumes the simultaneous measurements describe one profile.

```{code-cell} python
import numpy as np
height=np.array([20,40,80]); speed=np.array([5,6,7.2])
alpha, intercept=np.polyfit(np.log(height/20),np.log(speed),1)
prediction=np.exp(intercept)*(120/20)**alpha
adjusted=turbine_power_mw(prediction)*6
unadjusted=turbine_power_mw(7.2)*6
assert np.isclose(alpha,np.log(1.2)/np.log(2)) and adjusted>unadjusted
```
:::

:::{admonition} Quiz 20 — Reconcile independent farm losses
:class: note

**Difficulty:** Hard

**Reference:** [@manwell2009].

Start at 120 GWh and apply wake 8%, downtime 5%, electrical 2%, and curtailment 3%. Return the lost energy at each step, reconcile the total, and compare the gap from treating all losses as fractions of gross.

:::

:::{admonition} Solution
:class: tip, dropdown

Each stage acts on remaining energy; applying every loss to gross overstates total loss here.

```{code-cell} python
def waterfall(gross, losses):
    remaining=gross; stages=[]
    for loss in losses:
        lost=remaining*loss
        stages.append(lost)
        remaining-=lost
    return remaining, stages

rates=[.08,.05,.02,.03]
net, lost=waterfall(120,rates)
assert abs(net+sum(lost)-120)<1e-9 and net>120*(1-sum(rates))
```
:::

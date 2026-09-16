---
kernelspec:
  name: python3
  display_name: Python 3
---

# Wind Energy

## Learning goals

After this lesson you should be able to:

- explain why available wind power is proportional to the cube of wind speed;
- distinguish available power, rotor power, electrical power, rated power, and energy;
- apply the Betz limit without treating it as a real turbine efficiency;
- implement cut-in, rated, and cut-out behavior with Python conditionals;
- estimate hub-height wind speed, AEP, and capacity factor;
- work correctly with wind direction as circular data;
- produce and interpret power-curve and distribution plots.

Wind turbines convert part of the kinetic-energy flux through the rotor area into electricity. Real production depends on air density, wind speed at hub height, the turbine power curve, availability, wakes, electrical losses, curtailment, icing, and environmental limits. DOE describes the standard cut-in/rated/cut-out behavior, while Betz's ideal actuator-disk result bounds aerodynamic extraction [@doeWindWeather; @doeSmallWind].

## Concepts and equations

### Available wind power

Air moving at speed $v$ through swept area $A$ has available power

$$P_{wind}=\frac{1}{2}\rho A v^3,\qquad A=\pi R^2$$

Rotor power is

$$P_{rotor}=C_pP_{wind}$$

where $C_p$ is the power coefficient. The ideal Betz maximum is

$$C_{p,max}=\frac{16}{27}\approx0.593$$

This is an aerodynamic upper bound, not a typical electrical efficiency. Generator and drivetrain losses occur after aerodynamic extraction.

### Air density and hub height

At the same wind speed, available power scales linearly with air density. A basic pressure-temperature approximation is

$$\rho\approx\frac{p}{R_dT}$$

with dry-air gas constant $R_d=287.05$ J/(kg K), pressure in Pa, and temperature in K.

When only a reference-height wind speed is available, the empirical power law is often used:

$$v(z)=v(z_r)\left(\frac{z}{z_r}\right)^\alpha$$

The shear exponent $\alpha$ is site- and stability-dependent; it is not universally $1/7$.

### Power curve, AEP, and capacity factor

A simplified curve has four regions:

1. zero below cut-in speed;
2. increasing output between cut-in and rated speed;
3. rated output between rated and cut-out speed;
4. zero at and above cut-out speed.

For discrete wind states with probabilities $p_i$:

$$AEP=8760\sum_i P(v_i)p_i$$

and

$$CF=\frac{AEP}{P_r\,8760}$$

Use a manufacturer power curve for real AEP work. A cubic interpolation is a teaching approximation only.

## Tested Python functions

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def air_density_kgm3(pressure_pa, temperature_c):
    """Dry-air density from the ideal-gas approximation."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [pressure_pa, temperature_c]):
        raise ValueError("model inputs must be finite")
    temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
    if np.any(temperature_k <= 0) or np.any(np.asarray(pressure_pa) <= 0):
        raise ValueError("pressure and absolute temperature must be positive")
    return np.asarray(pressure_pa, dtype=float) / (287.05 * temperature_k)

def wind_at_height(speed_ref, height_m, ref_height_m=10, alpha=0.14):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [speed_ref, height_m, ref_height_m, alpha]):
        raise ValueError("model inputs must be finite")
    speed = np.asarray(speed_ref, dtype=float)
    if np.any(speed < 0) or np.any(np.asarray(height_m) <= 0) or np.any(np.asarray(ref_height_m) <= 0):
        raise ValueError("speeds must be non-negative and heights positive")
    return speed * (np.asarray(height_m) / np.asarray(ref_height_m)) ** alpha

def turbine_power_mw(speed_ms, rated_power_mw=3.0,
                     cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    """Simplified continuous cubic power curve with correct boundaries."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [speed_ms, rated_power_mw, cut_in, rated_speed, cut_out]):
        raise ValueError("model inputs must be finite")
    if not 0 <= cut_in < rated_speed < cut_out:
        raise ValueError("require 0 <= cut_in < rated_speed < cut_out")
    speed = np.asarray(speed_ms, dtype=float)
    if np.any(speed < 0) or rated_power_mw < 0:
        raise ValueError("speed and rated power must be non-negative")
    power = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated_speed)
    power[ramp] = rated_power_mw * (
        (speed[ramp] ** 3 - cut_in ** 3) /
        (rated_speed ** 3 - cut_in ** 3)
    )
    plateau = (speed >= rated_speed) & (speed < cut_out)
    power[plateau] = rated_power_mw
    return float(power) if power.ndim == 0 else power

assert turbine_power_mw(2) == 0
assert turbine_power_mw(12) == 3
assert turbine_power_mw(25) == 0
```

## Worked distribution and AEP example

The Weibull probability density is

$$f(v)=\frac{k}{c}\left(\frac{v}{c}\right)^{k-1}
\exp\left[-\left(\frac{v}{c}\right)^k\right]$$

where $k$ is shape and $c$ is scale. A fitted distribution compresses information and may miss calm periods, storms, seasonality, and direction-dependent wakes.

```{code-cell} python
def weibull_pdf(speed_ms, shape_k, scale_c):
    speed = np.asarray(speed_ms, dtype=float)
    if shape_k <= 0 or scale_c <= 0 or np.any(speed < 0):
        raise ValueError("Weibull parameters must be positive; speed non-negative")
    return (shape_k / scale_c) * (speed / scale_c) ** (shape_k - 1) * np.exp(
        -(speed / scale_c) ** shape_k
    )

bin_edges = np.arange(0, 31, 0.25)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
k, c = 2.1, 8.5
cdf = lambda v: 1 - np.exp(-(v / c) ** k)
probability = cdf(bin_edges[1:]) - cdf(bin_edges[:-1])
power_mw = turbine_power_mw(bin_centres)
aep_mwh = np.sum(power_mw * probability) * 8760
capacity_factor = aep_mwh / (3.0 * 8760)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
speed_grid = np.linspace(0, 30, 301)
axes[0].plot(speed_grid, turbine_power_mw(speed_grid), color="tab:blue")
axes[0].set(xlabel="Wind speed (m/s)", ylabel="Power (MW)",
            title="Simplified turbine power curve")
axes[1].bar(bin_centres, power_mw * probability * 8760 / 1000,
            width=0.24, color="tab:green")
axes[1].set(xlabel="Wind-speed bin (m/s)", ylabel="AEP contribution (GWh)",
            title=f"AEP = {aep_mwh/1000:.1f} GWh; CF = {capacity_factor:.1%}")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
```

## Wind direction is circular

Meteorological wind direction states where wind comes **from**: 0° is north and 90° is east. Arithmetic means fail near north—for example, 350° and 10° average to 180° arithmetically even though both are northerly. Use vector components:

$$\bar\theta=\operatorname{atan2}(\overline{\sin\theta},
\overline{\cos\theta})$$

and wrap the result to $[0,360)$.

```{code-cell} python
def circular_mean_degrees(direction_deg):
    values = np.asarray(direction_deg, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("directions must be nonempty and finite")
    radians = np.deg2rad(values)
    if np.hypot(np.sin(radians).mean(), np.cos(radians).mean()) < 1e-12:
        raise ValueError("mean direction is undefined for cancelling vectors")
    angle = np.rad2deg(np.arctan2(np.mean(np.sin(radians)),
                                  np.mean(np.cos(radians))))
    return np.round(angle % 360, 12) % 360

assert np.isclose(circular_mean_degrees([350, 10]), 0)
```

## Guided exercises

:::{admonition} Exercise 1 — Cubic sensitivity
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Write `available_wind_power_kw(speed, radius, density=1.225)`. Compare 6, 8, and 10 m/s and prove that doubling speed multiplies available power by eight.

```python
def available_wind_power_kw(speed, radius, density=1.225):
    # TODO: calculate swept area and available power
    pass
```
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

1. Calculate swept area.
2. Multiply by density and cubed wind speed.
3. Compare the doubling ratio.

```{code-cell} python
def available_wind_power_kw(speed, radius, density=1.225):
    speed = np.asarray(speed, dtype=float)
    if np.any(speed < 0) or radius < 0 or density <= 0:
        raise ValueError("invalid wind-power input")
    area = np.pi * radius**2
    return 0.5 * density * area * speed**3 / 1000

comparison_power = available_wind_power_kw(np.array([6, 8, 10]), 50)
print(comparison_power)
assert np.isclose(available_wind_power_kw(10, 50) / available_wind_power_kw(5, 50), 8)
```
:::

:::{admonition} Exercise 2 — Betz and practical extraction
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Plot available, Betz-limited, and $C_p=0.45$ rotor power from 0–15 m/s. Explain why these are not a controlled generator power curve above rated speed.

```python
# TODO: calculate three curves from the same available-power array
# TODO: label and plot them together
```
:::

:::{admonition} Exercise 2 — Solution
:class: tip, dropdown

1. Reuse the available-power array.
2. Multiply by betz and practical coefficients separately.
3. Label rotor power.

```{code-cell} python
exercise_speed = np.linspace(0, 15, 150)
available_kw = available_wind_power_kw(exercise_speed, 50)
plt.figure(figsize=(8, 4))
plt.plot(exercise_speed, available_kw, label="Available")
plt.plot(exercise_speed, available_kw * 16/27, label="Betz bound")
plt.plot(exercise_speed, available_kw * 0.45, label="Cp = 0.45")
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```

A real controller caps generator output and shuts the turbine down at cut-out.
:::

:::{admonition} Exercise 3 — Density correction
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

For temperatures -10–35 °C at 101325 Pa, plot air density and available power at 10 m/s.

```python
# TODO: call air_density_kgm3 for an array of temperatures
# TODO: use each density in available_wind_power_kw
```
:::

:::{admonition} Exercise 3 — Solution
:class: tip, dropdown

1. Convert celsius to kelvin.
2. Calculate density.
3. Use each density in the same wind-power calculation.

```{code-cell} python
temperature_c = np.linspace(-10, 35, 80)
density = air_density_kgm3(101325, temperature_c)
available_at_10 = np.array([available_wind_power_kw(10, 50, value) for value in density])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(temperature_c, density)
axes[0].set(xlabel="Temperature (°C)", ylabel="Density (kg/m³)")
axes[1].plot(temperature_c, available_at_10, color="tab:orange")
axes[1].set(xlabel="Temperature (°C)", ylabel="Available power (kW)")
for axis in axes:
    axis.grid(alpha=0.3)
plt.tight_layout()
assert density[0] > density[-1]
```
:::

:::{admonition} Exercise 4 — Hub-height scenarios
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

For 6 m/s at 10 m, calculate 100 m wind speed for $\alpha=[0.10,0.14,0.25]$ and plot profiles from 10–150 m.

```python
# TODO: loop over shear exponents
# TODO: calculate and plot each vertical profile
```
:::

:::{admonition} Exercise 4 — Solution
:class: tip, dropdown

1. Create the height array.
2. Apply each shear exponent.
3. Compare 100 m values and label the profiles.

```{code-cell} python
heights = np.linspace(10, 150, 100)
plt.figure(figsize=(6, 5))
for shear_exponent in [0.10, 0.14, 0.25]:
    profile = wind_at_height(6, heights, alpha=shear_exponent)
    plt.plot(profile, heights, label=f"alpha={shear_exponent}")
    print(shear_exponent, wind_at_height(6, 100, alpha=shear_exponent))
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Height (m)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
```
:::

:::{admonition} Exercise 5 — Hourly and daily energy
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Generate 14 days of hourly wind speed, calculate power, reshape into days, and plot the first week. Check physical bounds.

```python
# TODO: generate reproducible hourly speeds
# TODO: calculate hourly power and daily MWh
# TODO: plot the first 168 hours
```
:::

:::{admonition} Exercise 5 — Solution
:class: tip, dropdown

1. Generate seeded hourly speeds.
2. Apply the turbine curve.
3. Sum groups of 24 one-hour intervals.

```{code-cell} python
rng = np.random.default_rng(31)
hourly_speed = rng.weibull(2.0, 24 * 14) * 8
hourly_power = turbine_power_mw(hourly_speed)
daily_energy_mwh = hourly_power.reshape(-1, 24).sum(axis=1)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(hourly_speed[:168])
axes[0].set(ylabel="Wind speed (m/s)")
axes[1].plot(hourly_power[:168], color="tab:green")
axes[1].set(xlabel="Hour", ylabel="Power (MW)")
plt.tight_layout()
assert np.all(hourly_speed >= 0)
assert np.all((hourly_power >= 0) & (hourly_power <= 3))
```
:::

:::{admonition} Exercise 6 — Circular statistics and wind roses
:class: note

**Reference:** Manwell et al., Sections 2.3–2.5 and 3.2 [@manwell2009].

Calculate circular mean direction, bin directions into 12 sectors, and compare frequency- and energy-weighted polar charts.

```python
# TODO: generate or load directions
# TODO: calculate circular mean and sector indices
# TODO: create two polar bar charts
```
:::

:::{admonition} Exercise 6 — Solution
:class: tip, dropdown

1. Calculate the circular mean.
2. Assign compass-centered sectors.
3. Count observations and sum one-hour energy by sector.

```{code-cell} python
directions = (230 + rng.normal(0, 50, len(hourly_speed))) % 360
mean_direction = circular_mean_degrees(directions)
sector_edges = np.linspace(0, 360, 13)
sector_index = ((directions + 15) % 360 // 30).astype(int)  # sectors centered on compass angles
frequency = np.bincount(sector_index, minlength=12)
energy = np.bincount(sector_index, weights=hourly_power, minlength=12)
angles = np.deg2rad(np.arange(0, 360, 30))
fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"})
axes[0].bar(angles, frequency, width=np.deg2rad(27))
axes[0].set_title("Frequency")
axes[1].bar(angles, energy, width=np.deg2rad(27), color="tab:green")
axes[1].set_title("Energy")
for axis in axes:
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
plt.tight_layout()
print(f"Circular mean: {mean_direction:.1f}°")
```
:::

## Common mistakes

- Using $v^2$ rather than $v^3$ in available wind power.
- Treating the Betz limit as turbine electrical efficiency.
- Extending the cubic law beyond rated speed or through cut-out.
- Calculating energy from mean wind speed; because of the cubic relationship, $P(\bar v)$ generally differs from $\overline{P(v)}$.
- Randomly shuffling time-series data before forecasting.
- Averaging wind directions arithmetically or reversing the meteorological convention.

## Check your understanding

Continue with the [wind-physics quiz](../section7/renewableEnergyquizzes/windEnergy.md) and the [wind data lab](../section7/renewableEnergyquizzes/eolicenergy.md).

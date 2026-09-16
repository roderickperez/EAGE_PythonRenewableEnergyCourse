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

<!-- expanded-theory:solar:start -->

## Physical description: what solar energy systems do

Solar energy is electromagnetic radiation arriving from the Sun. A useful solar system must intercept that radiation, convert part of it into a useful form, and deliver that output to a load. The resource at a site depends on time, location, atmospheric conditions and the orientation of the receiving surface. A sunny location alone does not specify how much electricity a particular installation will deliver [@foster2010; @wade2003].

Two conversion routes must be distinguished. **Photovoltaic (PV) systems** convert absorbed light directly into DC electricity in semiconductor devices. **Solar thermal systems** absorb radiation as heat; that heat can supply hot water or an industrial process, or drive a heat engine in a concentrating solar power plant. A PV inverter is an electrical converter, whereas a solar thermal turbine is part of a thermodynamic cycle. Neither a PV module nor a thermal collector is, by itself, an energy-storage device.

This chapter develops PV in detail because the Python exercises and final project model electrical PV output. The solar-thermal comparison below explains the wider meaning of solar energy without treating thermal and electrical output as interchangeable. The supplied Foster and Wade texts are the primary learning references; Duffie and Beckman and Kalogirou provide additional reading on radiation, collectors and system modelling [@foster2010; @wade2003; @duffie2013; @kalogirou2014solar].

### From sunlight to delivered electricity

The system sequence is **solar resource → module surface → PV cells → DC wiring → inverter → AC wiring and transformer → meter/load**. Losses and limits can occur at every stage. A grid-connected system usually exports or self-consumes AC electricity. An off-grid system may use a battery and charge controller; then the chosen model must also track stored energy and battery conversion losses.

| Component or term | Meaning and role |
|---|---|
| Cell | Semiconductor device that produces a current–voltage response under illumination |
| Module or panel | Connected cells packaged for electrical insulation and environmental protection |
| String | Modules connected in series; their operating voltages add |
| Array | One or more strings forming the installed DC generating system |
| Inverter | Converts DC to AC and controls the electrical operating point within its limits |
| Maximum-power-point tracking (MPPT) | Control that seeks the voltage/current combination with maximum available DC power |
| Balance of system | Wiring, switches, protection, mounting and other equipment beyond the modules |
| DC and AC nameplate | Ratings at different electrical boundaries; they must be distinguished in ratios and capacity factors |

Series and parallel connections change voltage and current differently. For identical modules operating compatibly, series connection adds voltage while maintaining string current; parallel connection adds current at the common voltage. Real mismatch, partial shading, cable losses, bypass diodes and inverter operating limits complicate this ideal picture. A shaded module is therefore not always represented accurately by reducing the whole array's irradiance by one average percentage [@wade2003; @foster2010].

## Solar resource and measurement definitions

### Irradiance, irradiation and the receiving plane

**Irradiance**, denoted here by $G$, is radiant power incident per unit area, in W/m². **Irradiation**, denoted by $H_{POA}$ for a module plane, is incident radiant energy per unit area over a specified period, often kWh/m². The subscripts identify the measurement surface: a horizontal sensor and a tilted module generally receive different irradiance.

The principal resource components are global horizontal irradiance (GHI), direct normal irradiance (DNI), and diffuse horizontal irradiance (DHI). For a sun above the horizon, consistent measurements approximately satisfy

$$GHI=DNI\cos\theta_z+DHI,$$

where $\theta_z$ is solar zenith angle, measured from the vertical. DNI is defined on a plane normal to the direct solar beam. DHI represents sky-diffuse radiation on the horizontal plane. Do not substitute DNI directly into a flat-plate PV model as though it were the total radiation on the modules.

**Plane-of-array irradiance** $G_{POA}$ includes the direct beam projected onto the module plane, sky diffuse radiation and ground-reflected radiation. A useful conceptual decomposition is

$$G_{POA}=DNI\max(\cos\theta_i,0)+G_{sky,POA}+G_{ground,POA},$$

where $\theta_i$ is the beam incidence angle on the front face. The diffuse and reflected terms require a radiation-transposition model; they are not generally equal to DHI. Module tilt, azimuth, horizon obstruction, trackers and albedo affect the result. The exercise inputs explicitly provide POA irradiance so that a separate solar-position and transposition model is not silently omitted [@foster2010; @duffie2013].

For interval-average measurements,

$$H_{POA}=\frac{1}{1000}\sum_i G_{POA,i}\Delta t_i,$$

when $G$ is in W/m² and $\Delta t$ is in hours; $H$ is then kWh/m². The factor of 1000 converts Wh to kWh. For instantaneous samples, a chosen numerical integration method, such as a trapezoidal approximation, is needed instead. Always state which type of observation the dataset contains.

### Time and data quality

Record the timestamp convention, time zone, interval length and whether each timestamp marks an interval start or end. Nighttime zero irradiance can be valid; missing data are not zero. Sensor misalignment, shading of the sensor, snow, soiling, calibration drift and time shifts can distort a production comparison. A radiation sensor at one point does not necessarily represent every module in a large array.

## PV conversion: definitions, equations and limits

### The current–voltage characteristic

An illuminated PV device does not supply an arbitrary fixed voltage and current simultaneously. Its operating point lies on an **I–V curve** determined by irradiance, cell temperature and device properties. Electrical DC power is

$$P_{DC}=VI.$$

At open circuit, current is zero and voltage is $V_{oc}$; at short circuit, voltage is zero and current is $I_{sc}$. Neither condition delivers useful electrical power. At the maximum-power point, $P_{mp}=V_{mp}I_{mp}$. The fill factor

$$FF=\frac{V_{mp}I_{mp}}{V_{oc}I_{sc}}$$

describes the shape of the I–V curve and is dimensionless. It is not conversion efficiency or capacity factor. Irradiance strongly affects photocurrent, while voltage and power also respond to temperature. MPPT attempts to operate near the available maximum rather than at either I–V endpoint [@wade2003; @foster2010].

For a simple area model,

$$P_{DC}=G_{POA}A\eta_{PV},\qquad
\eta_{PV}=\frac{P_{DC}}{G_{POA}A}.$$

Here $A$ is the module area defined consistently with the efficiency specification, in m². With irradiance in W/m², output is W. Holding $\eta_{PV}$ constant is useful for a first calculation but omits operating-temperature, spectral and low-light effects.

### STC, nameplate and temperature

**Standard Test Conditions (STC)** use 1000 W/m² irradiance, 25°C **cell** temperature and the AM1.5 reference spectrum. A kWp rating commonly denotes rated DC kilowatts under these conditions. Ambient air at 25°C does not imply cells at 25°C: absorbed sunlight usually warms the module. Actual output can occasionally exceed the STC rating under favorable conditions, so nameplate is not a universal hard ceiling on DC power.

The course uses a first-order DC model:

$$P_{DC}=P_{STC}\frac{G_{POA}}{G_{STC}}
\left[1+\gamma_P(T_c-T_{STC})\right].$$

$\gamma_P$ is the fractional power-temperature coefficient in K⁻¹ or °C⁻¹; a datasheet value of −0.4%/°C becomes −0.004/°C in Python. Celsius and kelvin temperature **differences** are equal. Use cell temperature $T_c$, not ambient temperature $T_a$, in the correction. For the negative coefficient assumed here, warmer cells reduce output at fixed irradiance [@foster2010; @pvlibDocs].

The introductory thermal approximation is

$$T_c=T_a+\frac{G_{POA}}{800\ \mathrm{W/m^2}}
\left(T_{NOCT}-20\ ^\circ\mathrm{C}\right).$$

The nominal operating cell temperature parameter summarizes specified reference operating conditions. This simplified equation does not explicitly model wind cooling, mounting geometry, heat capacity or transient behavior. Use appropriate thermal models and datasheet conventions in a detailed study. Flooring a negative extrapolated DC estimate at zero avoids impossible negative generation but does not make the underlying extrapolation physically valid.

### Inverter efficiency, clipping and loss accounting

Define potential AC output before clipping as $P_{AC,pot}=\eta_{inv}P_{DC}$. A simple inverter-limited model gives

$$P_{AC}=\min(P_{AC,pot},P_{AC,r}),\qquad
P_{clip}=\max(P_{AC,pot}-P_{AC,r},0).$$

This assumes a constant inverter efficiency and omits startup thresholds and standby consumption. Inverter conversion loss is $P_{DC}-P_{AC,pot}$, while clipping is discarded potential AC output above the AC limit. They are different quantities. A manufacturer AC power model may already include conversion losses; multiplying by the same efficiency again would double-count them.

The **DC/AC ratio** is $P_{STC,DC}/P_{AC,r}$. Increasing it can improve use of an inverter at lower irradiance but may increase clipping near peak resource. It is an engineering and economic trade-off, not a guarantee of higher performance ratio. Independent sequential fractional losses $\ell_j$ give a retained fraction $\prod_j(1-\ell_j)$, provided the loss boundaries do not overlap.

## Energy, yield and performance indicators

Electrical energy is $E_{AC}=\sum_i P_{AC,i}\Delta t_i$. Use kW with hours for kWh or MW with hours for MWh. Annual energy requires annual coverage or a documented extrapolation; a clear-sky day is not an annual resource model.

| Indicator | Definition | What it answers |
|---|---|---|
| Specific/final yield $Y_f$ | $E_{AC}/P_{STC,DC}$, often kWh/kWp | How much AC energy was delivered per DC nameplate unit? |
| Reference yield $Y_r$ | $H_{POA}/G_{STC}$, in hours | What equivalent full-reference-irradiance duration was available? |
| Performance ratio $PR$ | $Y_f/Y_r$, dimensionless | What normalized system yield was achieved under the stated boundary? |
| AC capacity factor | $E_{AC}/(P_{AC,r}T)$ | How much energy was delivered relative to continuous AC-rated output? |
| Availability | Fraction of a defined time or energy opportunity when equipment is available | Was the system able to operate? |

PR is not module efficiency: its denominator is an irradiation-normalized nameplate yield. Temperature, soiling, downtime and measurement conventions affect it. An unexpectedly high PR calls for checking definitions, sensors and conditions; it is not resolved simply by clipping every ratio to one. Keep DC-rated and AC-rated denominators explicit.

## Solar thermal: the related but different conversion route

A thermal collector delivers useful heat to a circulating fluid. For a single-phase fluid with approximately constant heat capacity,

$$\dot Q_u=\dot m c_p(T_{out}-T_{in}).$$

$\dot Q_u$ is heat-transfer rate in W when mass flow is kg/s and $c_p$ is J/(kg K). Collector thermal efficiency is $\eta_{th}=\dot Q_u/(G A)$ under specified measurement boundaries. It is not a PV electrical efficiency. Heat loss to the surroundings grows as the collector becomes hotter relative to ambient conditions, so a fixed thermal efficiency is generally inadequate over a broad temperature range.

Concentrating systems use optics and tracking to deliver beam radiation to a receiver; their resource assessment relies strongly on DNI. A concentrating solar power plant then converts collected heat to electricity through a heat engine, potentially using thermal storage between collection and generation. A flat-plate PV array can use direct and diffuse radiation without concentrating it. These distinctions matter when choosing the input dataset and interpreting a claimed solar conversion efficiency [@foster2010; @duffie2013; @kalogirou2014solar].

## Worked calculation before coding

Consider a synthetic 10 kW DC array with an 8 kW AC inverter, POA irradiance 800 W/m², ambient temperature 25°C, NOCT 45°C, $\gamma_P=-0.004$/°C and inverter efficiency 0.97.

1. Estimate cell temperature: $T_c=25+(800/800)(45-20)=50$°C.
2. Calculate the temperature multiplier: $1-0.004(50-25)=0.90$.
3. Calculate DC output: $10(800/1000)(0.90)=7.20$ kW.
4. Apply inverter conversion: $7.20(0.97)=6.984$ kW potential AC.
5. Compare with the inverter rating: $\min(6.984,8)=6.984$ kW, with no clipping.
6. If these are mean conditions for a half-hour teaching interval, estimate $6.984(0.5)=3.492$ kWh.

The last calculation applies a nonlinear model to interval-average inputs. Rapid irradiance or temperature changes can make this differ from the average of higher-resolution modeled output. The numerical result is a model estimate, not a measured performance guarantee.

## From the explanation to the Python exercises

Begin with irradiance, area, electrical rating and time units in exercises 1–5. Exercises 6–15 add temperature, clipping, losses, normalization and data checks. Exercises 16–20 combine design comparisons, storage, discounted cost and chronological performance evaluation. The tested functions below deliberately model the simpler PV relationships; they do not replace a detailed shading, electrical string-design or solar-thermal simulation.

For preparation, read Wade's electricity and PV sections and Foster's PV-system treatment. Use Duffie and Beckman for additional radiation/thermal-system reading and Kalogirou for a broader comparison of solar conversion systems. Keep dated textbook prices or market statistics separate from the physical principles used here [@wade2003; @foster2010; @duffie2013; @kalogirou2014solar].


<!-- expanded-theory:solar:end -->

## Concepts and equations: compact reference

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


[Download this complete chapter as a Jupyter notebook](solarEnergy.ipynb). In standard Jupyter viewers, answer headings and code are visible below each prompt; the book provides collapse controls.

## Chapter practice — 20 Python exercises

**5 easy · 10 medium · 5 hard.** Work through the exercises in order. All inputs are synthetic teaching data. Each solution is directly below its question and starts collapsed in the book. Open it after trying your own code. Each solution runs independently; run the full notebook from the first cell when studying the chapter. References identify the underlying concepts rather than copied textbook problems.

(solar-exercise-01)=
### Exercise 01 — Panel DC power

**Difficulty:** Easy

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** At 800 W/m², a 2 m² panel has efficiency 20%. Calculate DC watts.

```{code-cell} python
# Your solution for solar exercise 01.
```

::::{dropdown} Step-by-step answer — Solar 01

1. Store irradiance, area and efficiency.
2. Multiply incident power by conversion efficiency.
3. Label and verify the result.

```{code-cell} python
# Step 1: Store irradiance, area and efficiency.
g, area, eta = 800, 2, 0.20
# Step 2: Multiply incident power by conversion efficiency.
power_w = g*area*eta
# Step 3: Label and verify the result.
print(power_w, 'W DC'); assert power_w == 320
```

**Interpretation:** 320 W assumes constant efficiency and uniform irradiance.

::::

(solar-exercise-02)=
### Exercise 02 — Array nameplate

**Difficulty:** Easy

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** An array contains 24 modules rated at 400 W DC. Calculate its STC rating in kW.

```{code-cell} python
# Your solution for solar exercise 02.
```

::::{dropdown} Step-by-step answer — Solar 02

1. Enter module count and rating.
2. Add module ratings and convert to kW.
3. Print and check the nameplate.

```{code-cell} python
# Step 1: Enter module count and rating.
count, module_w = 24, 400
# Step 2: Add module ratings and convert to kW.
array_kw = count*module_w/1000
# Step 3: Print and check the nameplate.
print(array_kw,'kW DC'); assert array_kw == 9.6
```

**Interpretation:** Nameplate describes STC; actual operating output varies.

::::

(solar-exercise-03)=
### Exercise 03 — Daily irradiation

**Difficulty:** Easy

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Four hourly mean irradiances are 0,200,600,400 W/m². Calculate four-hour irradiation in kWh/m².

```{code-cell} python
# Your solution for solar exercise 03.
```

::::{dropdown} Step-by-step answer — Solar 03

1. Store hourly means and duration.
2. Integrate and convert Wh to kWh.
3. Verify and label the four-hour result.

```{code-cell} python
# Step 1: Store hourly means and duration.
irradiance = [0,200,600,400]; hours = 1
# Step 2: Integrate and convert Wh to kWh.
irradiation = sum(irradiance)*hours/1000
# Step 3: Verify and label the four-hour result.
print(irradiation,'kWh/m²'); assert irradiation == 1.2
```

**Interpretation:** This is a four-hour window, not necessarily a complete day.

::::

(solar-exercise-04)=
### Exercise 04 — Inverter conversion

**Difficulty:** Easy

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** A PV array delivers 5 kW DC to a 96%-efficient inverter with a 6 kW AC limit. Find AC kW.

```{code-cell} python
# Your solution for solar exercise 04.
```

::::{dropdown} Step-by-step answer — Solar 04

1. Enter the DC input and equipment assumptions.
2. Apply conversion efficiency and the output limit.
3. Check the output cannot exceed input or rating.

```{code-cell} python
# Step 1: Enter the DC input and equipment assumptions.
dc_kw, efficiency, rating_kw = 5, 0.96, 6
# Step 2: Apply conversion efficiency and the output limit.
ac_kw = min(dc_kw*efficiency, rating_kw)
# Step 3: Check the output cannot exceed input or rating.
print(ac_kw,'kW AC'); assert ac_kw == 4.8 and ac_kw <= dc_kw
```

**Interpretation:** No clipping occurs in this example; efficiency is held constant.

::::

(solar-exercise-05)=
### Exercise 05 — Capacity factor from AC energy

**Difficulty:** Easy

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** A 10 kW AC plant exports 48 kWh in 24 hours. Calculate AC-based CF and state the denominator.

```{code-cell} python
# Your solution for solar exercise 05.
```

::::{dropdown} Step-by-step answer — Solar 05

1. Set exported energy, AC capacity and duration.
2. Calculate the AC-based capacity factor.
3. Print a labelled percentage.

```{code-cell} python
# Step 1: Set exported energy, AC capacity and duration.
energy_kwh, capacity_kw, hours = 48, 10, 24
# Step 2: Calculate the AC-based capacity factor.
cf = energy_kwh/(capacity_kw*hours)
# Step 3: Print a labelled percentage.
print(f'AC-based CF: {cf:.0%}'); assert cf == 0.2
```

**Interpretation:** 20% uses AC nameplate. A DC-based capacity factor would have a different denominator.

::::

(solar-exercise-06)=
### Exercise 06 — Cell-temperature estimate

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** At ambient 25 °C and irradiance [0,400,800] W/m², use NOCT 45 °C and Tc=Ta+G(NOCT−20)/800. Calculate temperatures.

```{code-cell} python
# Your solution for solar exercise 06.
```

::::{dropdown} Step-by-step answer — Solar 06

1. Store irradiance and temperature parameters.
2. Apply the approximate NOCT relationship.
3. Verify the 800 W/m² case.

```{code-cell} python
import numpy as np
# Step 1: Store irradiance and temperature parameters.
g = np.array([0,400,800]); ambient = 25; noct = 45
# Step 2: Apply the approximate NOCT relationship.
cell = ambient+g*(noct-20)/800
# Step 3: Verify the 800 W/m² case.
print(cell,'°C'); assert np.allclose(cell,[25,37.5,50])
```

**Interpretation:** This simple thermal relationship omits wind and mounting effects.

::::

(solar-exercise-07)=
### Exercise 07 — Temperature coefficient

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** At 1000 W/m², a 10 kW STC array has gamma −0.004/°C. Evaluate DC output at cell temperatures [25,45,65]°C.

```{code-cell} python
# Your solution for solar exercise 07.
```

::::{dropdown} Step-by-step answer — Solar 07

1. Convert temperature cases to an array.
2. Apply the temperature multiplier relative to 25°C.
3. Check direction and expected output.

```{code-cell} python
import numpy as np
# Step 1: Convert temperature cases to an array.
temperature = np.array([25,45,65]); gamma = -0.004
# Step 2: Apply the temperature multiplier relative to 25°C.
dc_kw = 10*(1+gamma*(temperature-25))
# Step 3: Check direction and expected output.
print(dc_kw,'kW DC'); assert np.allclose(dc_kw,[10,9.2,8.4])
```

**Interpretation:** Higher cell temperature reduces output for a negative power coefficient.

::::

(solar-exercise-08)=
### Exercise 08 — Clipping budget

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Hourly DC output [0,4,8,10] kW passes through a 97% inverter capped at 7 kW AC. Calculate AC kWh and clipping kWh.

```{code-cell} python
# Your solution for solar exercise 08.
```

::::{dropdown} Step-by-step answer — Solar 08

1. Calculate AC power before applying the rating.
2. Clip output and calculate discarded potential AC energy.
3. Check energy accounting across one-hour intervals.

```{code-cell} python
import numpy as np
# Step 1: Calculate AC power before applying the rating.
dc = np.array([0,4,8,10]); available_ac = dc*0.97
# Step 2: Clip output and calculate discarded potential AC energy.
ac = np.minimum(available_ac,7); clipping = available_ac-ac
# Step 3: Check energy accounting across one-hour intervals.
print(ac.sum(),'kWh AC;',clipping.sum(),'kWh clipping')
assert np.allclose(ac+clipping,available_ac) and np.isclose(ac.sum(),17.88)
```

**Interpretation:** Clipping is measured after modeled inverter conversion; it is distinct from inverter losses.

::::

(solar-exercise-09)=
### Exercise 09 — Sequential losses

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Start with 1000 kWh DC; apply 3% soiling, 2% wiring and 4% inverter losses sequentially. Compare with simply subtracting 9%.

```{code-cell} python
# Your solution for solar exercise 09.
```

::::{dropdown} Step-by-step answer — Solar 09

1. Express loss fractions as retained fractions.
2. Multiply the successive retained fractions.
3. Print the difference and check the product.

```{code-cell} python
import numpy as np
# Step 1: Express loss fractions as retained fractions.
retained = 1-np.array([0.03,0.02,0.04])
# Step 2: Multiply the successive retained fractions.
output = 1000*np.prod(retained); additive = 1000*(1-0.09)
# Step 3: Print the difference and check the product.
print(output,'kWh;',output-additive,'kWh difference')
assert np.isclose(output,912.576)
```

**Interpretation:** Multiplicative losses apply to changing energy bases; simple addition is an approximation.

::::

(solar-exercise-10)=
### Exercise 10 — Performance ratio

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** A 100 kW DC array exports 12,000 kWh with POA irradiation 150 kWh/m². Use reference irradiance 1 kW/m² to calculate PR.

```{code-cell} python
# Your solution for solar exercise 10.
```

::::{dropdown} Step-by-step answer — Solar 10

1. Calculate final yield in hours from AC energy and DC rating.
2. Calculate reference yield and their ratio.
3. Verify and display the dimensionless result.

```{code-cell} python
# Step 1: Calculate final yield in hours from AC energy and DC rating.
final_yield = 12000/100
# Step 2: Calculate reference yield and their ratio.
reference_yield = 150/1; pr = final_yield/reference_yield
# Step 3: Verify and display the dimensionless result.
print(f'PR: {pr:.0%}'); assert pr == 0.8
```

**Interpretation:** PR normalises for irradiation; it is not cell conversion efficiency.

::::

(solar-exercise-11)=
### Exercise 11 — Specific yield

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Two systems generate [4000,9000] kWh from [5,10] kW DC. Calculate kWh/kWp and identify the higher specific yield.

```{code-cell} python
# Your solution for solar exercise 11.
```

::::{dropdown} Step-by-step answer — Solar 11

1. Store energy and DC nameplate values.
2. Normalise production by installed capacity.
3. Compare the normalised values.

```{code-cell} python
import numpy as np
# Step 1: Store energy and DC nameplate values.
energy = np.array([4000,9000]); capacity = np.array([5,10])
# Step 2: Normalise production by installed capacity.
specific_yield = energy/capacity
# Step 3: Compare the normalised values.
print(specific_yield,'kWh/kWp'); assert np.argmax(specific_yield)==1
```

**Interpretation:** The periods must match before comparing yields; higher yield does not alone establish lower cost.

::::

(solar-exercise-12)=
### Exercise 12 — Unequal daylight intervals

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Average AC powers [1,4,2] kW persist for [0.5,2,1.5] hours. Integrate their energy.

```{code-cell} python
# Your solution for solar exercise 12.
```

::::{dropdown} Step-by-step answer — Solar 12

1. Store paired power and duration arrays.
2. Multiply each interval before summing.
3. Verify the hand calculation.

```{code-cell} python
import numpy as np
# Step 1: Store paired power and duration arrays.
power = np.array([1,4,2]); duration = np.array([0.5,2,1.5])
# Step 2: Multiply each interval before summing.
energy = (power*duration).sum()
# Step 3: Verify the hand calculation.
print(energy,'kWh'); assert energy == 11.5
```

**Interpretation:** A sum of kW samples without duration is not energy for unequal intervals.

::::

(solar-exercise-13)=
### Exercise 13 — Nighttime and invalid data

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** For readings [0,200,missing,−10] W/m², preserve night zero, mark negative values missing, and report valid fraction.

```{code-cell} python
# Your solution for solar exercise 13.
```

::::{dropdown} Step-by-step answer — Solar 13

1. Load readings without replacing missing values.
2. Flag negatives and mask them.
3. Check the valid night zero and report coverage.

```{code-cell} python
import pandas as pd
# Step 1: Load readings without replacing missing values.
g = pd.Series([0,200,None,-10],dtype=float)
# Step 2: Flag negatives and mask them.
invalid = g.lt(0); clean = g.mask(invalid)
# Step 3: Check the valid night zero and report coverage.
print(clean, 'Valid fraction:', clean.notna().mean())
assert clean.iloc[0]==0 and clean.notna().mean()==0.5
```

**Interpretation:** Zero at night is a valid observation; a missing daylight observation cannot be treated as zero.

::::

(solar-exercise-14)=
### Exercise 14 — Compounded degradation

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** First-year output is 100 MWh. With 0.5% annual degradation, calculate output in years 1–5 and cumulative MWh.

```{code-cell} python
# Your solution for solar exercise 14.
```

::::{dropdown} Step-by-step answer — Solar 14

1. Index years with no degradation in the first year.
2. Apply annual retained output multiplicatively.
3. Check the first and last years and print the sum.

```{code-cell} python
import numpy as np
# Step 1: Index years with no degradation in the first year.
years = np.arange(1,6)
# Step 2: Apply annual retained output multiplicatively.
energy = 100*0.995**(years-1)
# Step 3: Check the first and last years and print the sum.
print(energy,'MWh/year;',energy.sum(),'MWh total')
assert energy[0]==100 and np.isclose(energy[-1],98.0149500625)
```

**Interpretation:** The rate is an assumption; weather and availability are held constant.

::::

(solar-exercise-15)=
### Exercise 15 — Daily profile plot

**Difficulty:** Medium

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Model 24 hourly mean irradiances as max(0,800 sin(pi(h−6)/12)). Use 20 m² and 20% efficiency. Plot DC kW and integrate MWh.

```{code-cell} python
# Your solution for solar exercise 15.
```

::::{dropdown} Step-by-step answer — Solar 15

1. Generate a labelled synthetic hourly irradiance profile.
2. Convert incident radiation to DC power and energy.
3. Plot with units and verify the maximum.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Generate a labelled synthetic hourly irradiance profile.
h = np.arange(24); g = np.maximum(0,800*np.sin(np.pi*(h-6)/12))
# Step 2: Convert incident radiation to DC power and energy.
dc_kw = g*20*0.2/1000; energy_mwh = dc_kw.sum()/1000
# Step 3: Plot with units and verify the maximum.
fig,ax=plt.subplots(figsize=(6,3)); ax.plot(h,dc_kw)
ax.set(xlabel='UTC hour',ylabel='DC power (kW)',title='Synthetic PV day'); fig.tight_layout(); plt.show()
print(energy_mwh,'MWh'); assert np.isclose(dc_kw.max(),3.2)
```

**Interpretation:** The irradiance values are treated as interval means; no real location or date is represented.

::::

(solar-exercise-16)=
### Exercise 16 — Size an inverter by clipping target

**Difficulty:** Hard

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** For hourly DC [0,2,5,9,12,8,3,0] kW and 97% efficiency, search integer AC ratings 5–12 kW for at most 5% clipped potential AC energy.

```{code-cell} python
# Your solution for solar exercise 16.
```

::::{dropdown} Step-by-step answer — Solar 16

1. Calculate the common available AC profile.
2. Evaluate each candidate using the same clipping denominator.
3. Select the smallest rating that satisfies the target.

```{code-cell} python
import numpy as np
# Step 1: Calculate the common available AC profile.
available = np.array([0,2,5,9,12,8,3,0])*0.97
# Step 2: Evaluate each candidate using the same clipping denominator.
trials = [(rating, np.maximum(available-rating,0).sum()/available.sum()) for rating in range(5,13)]
# Step 3: Select the smallest rating that satisfies the target.
best = next(t for t in trials if t[1]<=0.05)
print('Rating kW and clipping fraction:',best)
assert best[0]==10 and all(f>0.05 for r,f in trials if r<best[0])
```

**Interpretation:** An energy constraint is not a financial optimum; inverter prices and lifetime behavior are omitted.

::::

(solar-exercise-17)=
### Exercise 17 — Temperature and irradiance sensitivity grid

**Difficulty:** Hard

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** For G=[200,600,1000] W/m² and ambient=[10,25,40]°C, calculate a 3×3 grid of AC power for 10 kW DC, 8 kW AC, NOCT 45, gamma−0.004 and eta 0.97.

```{code-cell} python
# Your solution for solar exercise 17.
```

::::{dropdown} Step-by-step answer — Solar 17

1. Broadcast irradiance columns against ambient-temperature rows.
2. Calculate cell temperature, DC and limited AC power.
3. Check bounds and thermal direction and print a labelled grid.

```{code-cell} python
import numpy as np
import pandas as pd
# Step 1: Broadcast irradiance columns against ambient-temperature rows.
g=np.array([200,600,1000])[None,:]; ambient=np.array([10,25,40])[:,None]
# Step 2: Calculate cell temperature, DC and limited AC power.
cell=ambient+g*25/800
dc=np.maximum(10*g/1000*(1-0.004*(cell-25)),0); ac=np.minimum(dc*0.97,8)
# Step 3: Check bounds and thermal direction and print a labelled grid.
assert ac.shape==(3,3) and np.all(ac<=8) and np.all(np.diff(ac,axis=0)<=0)
print(pd.DataFrame(ac,index=[10,25,40],columns=[200,600,1000]).rename_axis('Ambient °C / G W m⁻²'))
```

**Interpretation:** At fixed irradiance, higher modeled cell temperature cannot increase power here; clipping can flatten sensitivity.

::::

(solar-exercise-18)=
### Exercise 18 — PV and battery self-consumption

**Difficulty:** Hard

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** PV hourly AC [0,4,6,0] kW meets constant 2 kW load with an initially empty 3 kWh battery, 2 kW limits and 90% efficiencies. Report unmet load, curtailed PV and final SOC.

```{code-cell} python
# Your solution for solar exercise 18.
```

::::{dropdown} Step-by-step answer — Solar 18

1. Initialise storage and bookkeeping in kWh.
2. Route each one-hour interval through direct use, charging and discharge.
3. Verify storage limits and show final energy accounting.

```{code-cell} python
import numpy as np
# Step 1: Initialise storage and bookkeeping in kWh.
soc=0.; unmet_total=0.; curtailed_total=0.; rows=[]
# Step 2: Route each one-hour interval through direct use, charging and discharge.
for pv in [0,4,6,0]:
    before=soc; direct=min(pv,2)
    charge=min(pv-direct,2,(3-soc)/0.9); discharge=min(2-direct,2,soc*0.9)
    soc+=charge*0.9-discharge/0.9
    unmet=2-direct-discharge; curtailed=pv-direct-charge
    assert np.isclose(before+charge*0.9,soc+discharge/0.9)
    unmet_total+=unmet; curtailed_total+=curtailed; rows.append(soc)
# Step 3: Verify storage limits and show final energy accounting.
print(unmet_total,curtailed_total,soc,'kWh unmet, curtailed, final SOC')
assert np.isclose(unmet_total,2) and all(0<=s<=3+1e-12 for s in rows)
```

**Interpretation:** Stored energy remaining at the end has not yet served the load. Battery economic sizing needs longer data.

::::

(solar-exercise-19)=
### Exercise 19 — PV LCOE with degradation

**Difficulty:** Hard

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Use €1m capital, €20k annual O&M, 1500 MWh first-year AC, 0.5% annual degradation and 25 years. Calculate LCOE at 0%,3%,6% real discount rates.

```{code-cell} python
# Your solution for solar exercise 19.
```

::::{dropdown} Step-by-step answer — Solar 19

1. Define year-end energy and a reusable discounted-cost calculation.
2. Evaluate the interest-rate scenarios with identical technical inputs.
3. Check the undiscounted result and scenario direction.

```{code-cell} python
import numpy as np
# Step 1: Define year-end energy and a reusable discounted-cost calculation.
years=np.arange(1,26); energy=1500*0.995**(years-1)
def lcoe(rate):
    discount=(1+rate)**years
    return (1_000_000+np.sum(20_000/discount))/np.sum(energy/discount)
# Step 2: Evaluate the interest-rate scenarios with identical technical inputs.
values=[lcoe(r) for r in [0,0.03,0.06]]
# Step 3: Check the undiscounted result and scenario direction.
print(values,'EUR/MWh')
assert np.isclose(values[0],1_500_000/energy.sum()) and np.all(np.diff(values)>0)
```

**Interpretation:** See IFC Chapter 14 and IRENA cost methodology. Costs are synthetic; taxes, replacements and salvage are omitted.

::::

(solar-exercise-20)=
### Exercise 20 — Detect underperformance without future leakage

**Difficulty:** Hard

**Reference:** PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].

**Task:** Expected daily energy is [10,20,30,40,50,60] kWh; observed is [9,18,27,36,30,36]. Calibrate an observed/expected ratio on days 1–4, then flag holdout days below 80% of calibrated expectation.

```{code-cell} python
# Your solution for solar exercise 20.
```

::::{dropdown} Step-by-step answer — Solar 20

1. Split the chronological observations before calibration.
2. Estimate one scale factor only from training energy.
3. Flag and check the holdout underperformance.

```{code-cell} python
import numpy as np
# Step 1: Split the chronological observations before calibration.
expected=np.array([10,20,30,40,50,60]); observed=np.array([9,18,27,36,30,36])
train=slice(0,4); test=slice(4,None)
# Step 2: Estimate one scale factor only from training energy.
factor=observed[train].sum()/expected[train].sum()
prediction=expected[test]*factor; relative=observed[test]/prediction
# Step 3: Flag and check the holdout underperformance.
flags=relative<0.8
print('Calibration:',factor,'holdout relative performance:',relative,'flags:',flags)
assert np.isclose(factor,0.9) and flags.all()
```

**Interpretation:** Flags identify a discrepancy, not its cause. Weather-model errors, outages or sensor faults need separate diagnosis.

::::

## Common mistakes

- Confusing irradiance (W/m$^2$) with irradiation (kWh/m$^2$).
- Applying a temperature correction to ambient temperature instead of estimated cell temperature without stating the simplification.
- Omitting inverter clipping or applying system losses twice.
- Summing power without multiplying by the time-step duration.
- Treating the simplified model as a substitute for site-specific shading, orientation, and loss modelling.
- Calling DC nameplate power an energy value.

## Check your understanding

Continue with the [solar quiz and Python challenge](../section7/renewableEnergyquizzes/solarEnergy.md).

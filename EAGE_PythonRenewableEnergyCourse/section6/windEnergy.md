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

:::{iframe} ../../wind-lab/index.html
:width: 100%
:title: Interactive three-dimensional wind turbine lab
:height: 1350px
:::

<!-- expanded-theory:wind:start -->

## Physical description: extracting energy from moving air

Wind is air motion driven by atmospheric pressure gradients, with Earth's rotation, surface friction, terrain and thermal structure influencing the resulting flow. A turbine extracts part of the kinetic-energy flux passing through its rotor. The air must continue moving downstream; extracting all its kinetic energy at the rotor would prevent a continuing through-flow. This is why a rotor cannot convert all of the undisturbed wind power into shaft power [@manwell2009; @wagner2009].

Most modern large electricity-generating wind turbines use lift-producing blades. The relative flow over a blade section produces aerodynamic forces, whose tangential component drives the rotor. The turbine's mechanical and electrical systems then convert rotor torque into useful electrical output. The visible rotation of the blades is one stage of an integrated aerodynamic, structural, control and electrical system. [@manwell2009], Chapters 1 and 3; [@wagner2009], turbine fundamentals.

### Components and system boundaries

The conversion sequence is **undisturbed wind → rotor aerodynamic extraction → shaft and drivetrain → generator and power electronics → transformer and network**. A wind farm introduces another level: turbines interact through wakes and share electrical and operational infrastructure. [@manwell2009], Chapters 1 and 3; [@wagner2009], turbine technology.

| Term or component | Definition and significance |
|---|---|
| Rotor swept area | Disk area through which the horizontal-axis rotor intercepts the flow, not the area of blade material |
| Hub height | Height of the rotor centre; a wind-speed measurement should identify its height |
| Nacelle | Housing for major drivetrain and generating equipment atop the tower |
| Gearbox or direct drive | Alternative arrangements relating rotor speed and generator operation |
| Pitch control | Adjustment of blade angle to regulate aerodynamic loading and power |
| Yaw control | Alignment of a horizontal-axis rotor with the incident wind direction |
| Rated electrical power | Specified output limit under defined operating conditions |
| Power curve | Relationship between wind speed and electrical output for a specified turbine and reference conditions |
| Wake | Downstream flow modified by energy extraction and rotor-generated turbulence |

*Source for these definitions/calculations:* [@manwell2009], Chapters 1 and 3; [@wagner2009], turbine technology.

Onshore and offshore machines use the same fundamental conversion relationships, but access, support structures, environmental loading and maintenance differ. A technology description is not enough to calculate production: the model also needs a resource time series or distribution and a suitable turbine power curve. [@manwell2009], Chapters 1 and 3; [@wagner2009], turbine technology.

The supplied *Wind Energy Explained* is the principal reference for the physics and statistical treatment below. Wagner and Mathur offer a shorter introduction. The additional *Wind Energy Handbook* extends the reading to machine design, controls and wind-farm interactions [@manwell2009; @wagner2009; @burton2011wind; @burton2021wind].

## Deriving the main aerodynamic relationships

### Why available power contains the cube of wind speed

For air density $\rho$, uniform undisturbed speed $v$ and rotor area $A$, the reference mass flow through that area in the undisturbed wind is $\dot m=\rho Av$ in kg/s. The kinetic energy per unit mass is $v^2/2$. Multiplying the two gives [@manwell2009], §2.5 and §3.2; [@wagner2009], §3.1.

$$P_{wind}=\dot m\frac{v^2}{2}=\frac12\rho Av^3.$$

*Equation basis:* [@manwell2009], §2.5 and §3.2.

The cubic dependence combines two effects: faster air carries more kinetic energy per kilogram and more kilograms pass the rotor each second. At fixed density and area, doubling speed gives eight times the **available kinetic-energy flux**. It does not imply eight times electrical output once turbine operating limits are involved. [@manwell2009], §2.5 and §3.2.

For a circular rotor, $A=\pi R^2=\pi D^2/4$. Doubling diameter gives four times the swept area, not twice the area. With density in kg/m³, area in m² and speed in m/s, power is watts. The relationship assumes a representative speed across the rotor; real shear, turbulence and spatial variation complicate that approximation [@manwell2009].

(wind-cube-law-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=cube-law
:width: 100%
:title: Available wind power and the cube law explorer
:::

### Power coefficient, drivetrain efficiency and Betz's limit

The rotor power coefficient is

$$C_p=\frac{P_{rotor}}{\tfrac12\rho Av^3}.$$

*Equation basis:* [@manwell2009], §3.2 (ideal actuator-disc assumptions).

It is an aerodynamic extraction ratio. A separate drivetrain/generator efficiency gives a simplified electrical relationship, [@manwell2009], §3.2 (ideal actuator-disc assumptions).

$$P_e=\eta_{drive}C_p\frac12\rho Av^3.$$

*Equation basis:* [@manwell2009], §3.2 (ideal actuator-disc assumptions).

The ideal actuator-disk analysis yields $C_p\le16/27\approx0.593$ under its assumptions of steady, incompressible, unconfined ideal flow. The bound concerns power extracted from the undisturbed kinetic-energy flux. Real rotors incur additional aerodynamic and mechanical losses. [@manwell2009], §3.2; [@wagner2009], §3.3.1.

Both $C_p$ and $\eta_{drive}$ can vary with operating point. If a supplied power curve already represents electrical output, do **not** multiply it by $C_p$ and generator efficiency again. First identify whether the input model describes wind power, rotor power, generator power or exported power. [@manwell2009], §3.2 (ideal actuator-disc assumptions).

(wind-betz-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=betz
:width: 100%
:title: Rotor extraction and Betz limit explorer
:::

### Torque and tip-speed ratio

Rotating-shaft power is

$$P_{shaft}=\tau\Omega,$$

*Equation basis:* [@manwell2009], Chapter 3.

where torque $\tau$ is N m and angular velocity $\Omega$ is radians/s. Convert rotational speed $n$ in revolutions/minute with $\Omega=2\pi n/60$. The tip-speed ratio is [@manwell2009], Chapter 3.

$$\lambda=\frac{\Omega R}{v}.$$

*Equation basis:* [@manwell2009], Chapter 3.

It compares blade-tip speed with undisturbed wind speed. Aerodynamic performance depends on tip-speed ratio and blade pitch, commonly represented as $C_p(\lambda,\beta)$, where $\beta$ is pitch angle under the chosen convention. Variable-speed control can adjust rotor speed to operate efficiently below rated output; above rated conditions, pitch and other controls constrain power and structural loads. A constant $C_p$ approximation is therefore a learning model, not a complete control model [@manwell2009].

## Resource characterization: density, height and variability

### Air density

For an ideal dry-air approximation,

$$\rho=\frac{p}{R_dT},\qquad R_d\approx287.05\ \mathrm{J/(kg\,K)}.$$

*Equation basis:* [@manwell2009], Chapter 2; dry-air ideal-gas approximation, not a humidity-corrected model.

Pressure $p$ is in pascals and absolute temperature $T$ is kelvin. Humidity and local conditions modify density; 1.225 kg/m³ is a stated reference assumption, not a universal site value. At fixed speed and swept area, lower density reduces available power. Applying a density adjustment to a manufacturer power curve requires that curve's specified correction procedure, especially near rated output [@manwell2009; @wagner2009].

(wind-air-density-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=air-density
:width: 100%
:title: Dry-air density and available wind power explorer
:::

### Wind shear and extrapolation to hub height

Surface friction and atmospheric structure produce a vertical wind profile. An often-used empirical approximation is [@manwell2009], §2.3.4.2, pp. 45–46.

$$v(z)=v(z_r)\left(\frac{z}{z_r}\right)^\alpha.$$

*Equation basis:* [@manwell2009], §2.3.4.2, pp. 45–46.

Both heights must be positive and measured relative to a consistent reference; $\alpha$ is an assumed or fitted shear exponent. It changes with roughness, atmospheric stability and conditions. Treating 0.14 or $1/7$ as an immutable physical constant can introduce resource bias. The final project supplies hub-height wind speed, so applying another height correction there would double-adjust the data. [@manwell2009], §2.3.4.2, pp. 45–46.

A neutral-atmosphere log-law model is another approximation,

$$v(z)=\frac{u_*}{\kappa}\ln\left(\frac{z-d}{z_0}\right),$$

*Equation basis:* [@manwell2009], §2.3.4.1; [@wagner2009], §3.4.1.

where $u_*$ is friction velocity, $\kappa$ is the von Kármán constant, $z_0$ is roughness length and $d$ is displacement height. The model requires a suitable neutral surface layer and heights with $z-d>z_0$. The beginner calculations use the power law and state its exponent. [@manwell2009], §§2.3.4.1–2.3.4.2; [@wagner2009], §3.4.1.

(wind-shear-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=shear
:width: 100%
:title: Hub-height wind shear explorer
:::

### Turbulence, gusts and averaging

**Turbulence intensity** for a defined observation window is $TI=\sigma_v/\bar v$ for positive mean speed. State the sampling window, data filtering and standard-deviation convention. It describes variability relative to the mean, not the same quantity as a gust speed. Turbulence influences loads, fatigue and power fluctuations; an energy-only model does not assess structural safety. [@manwell2009], §2.3; the averaging interval and standard-deviation convention must be stated.

For a nonlinear power model, $P(\bar v)$ is generally different from the average $\overline{P(v)}$. With equal-duration speeds of 4 and 8 m/s, the mean cubed speed is $(4^3+8^3)/2=288$, while the cube of the mean is $6^3=216$. Above rated speed or around cut-out, clipping and shutdown create further differences. Use interval data or an explicit probability distribution rather than assuming mean speed fully characterizes the resource. [@manwell2009], §2.3; the averaging interval and standard-deviation convention must be stated.

(wind-turbulence-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=turbulence
:width: 100%
:title: Turbulence and averaging explorer
:::

## The electrical power curve

The course's idealized curve is

$$
P(v)=\begin{cases}
0,&v<v_{in},\\
P_r\dfrac{v^3-v_{in}^3}{v_r^3-v_{in}^3},&v_{in}\le v<v_r,\\
P_r,&v_r\le v<v_{out},\\
0,&v\ge v_{out}.
\end{cases}
$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2. [@doeWindWeather]. The continuous cubic ramp is an original simplified interpolation.

$v_{in}$ is cut-in speed, $v_r$ is rated speed, $v_{out}$ is cut-out speed, and $P_r$ is rated electrical power. These thresholds are turbine-specific. The ramp interpolates continuously between zero and rated power, while the ideal cut-out model has an abrupt shutdown. Actual controllers may use hysteresis, delayed restart, derating or different storm-control behavior. The simple curve teaches conditionals and boundary testing; it does not reproduce every modern controller [@manwell2009; @doeWindWeather].

Testing exactly at the thresholds is important. For the exercise convention, output is zero at cut-in, rated at rated speed, and zero at cut-out. Reject negative or non-finite resource inputs before calculating output. Do not permit an unconstrained cubic equation to predict power above the equipment limit. Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2. [@doeWindWeather]. The continuous cubic ramp is an original simplified interpolation.

(wind-electrical-curve-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=electrical-curve
:width: 100%
:title: Electrical wind turbine power curve explorer
:::

## From resource distributions to annual energy

For interval-average modeled electrical power,

$$E=\sum_iP(v_i)\Delta t_i.$$

When speed is represented by a probability density $f(v)$ instead of an ordered time series,

$$\bar P=\int_0^\infty P(v)f(v)\,dv,\qquad E=T\bar P.$$

The integral averages **electrical power**, not wind speed. Use $T=8760$ h for a non-leap year only when that is the intended period. Multiplying a short or unrepresentative sample by annual hours is an extrapolation whose assumptions must be stated.

(wind-annual-energy-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=annual-energy
:width: 100%
:title: From wind resource to annual energy explorer
:::

### Weibull resource model and numerical integration

A two-parameter Weibull model for nonnegative speed has cumulative distribution Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

$$F(v)=1-\exp[-(v/c)^k],$$

*Equation basis:* Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

where $k>0$ is dimensionless shape and $c>0$ is scale in m/s. The probability of a bin $[a_i,b_i]$ is $p_i=F(b_i)-F(a_i)$. A midpoint approximation gives Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

$$\bar P\approx\sum_i P\!\left(\frac{a_i+b_i}{2}\right)p_i.$$

*Equation basis:* Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

Bin probability is dimensionless, while density has inverse-speed units; they cannot be substituted for one another without the bin width. Refine the bins to check numerical convergence and account for the probability beyond the integration range. A Weibull fit is a statistical approximation, not a law that all sites must follow [@manwell2009].

(wind-weibull-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=weibull
:width: 100%
:title: Weibull wind resource and bin integration explorer
:::

### Losses, wakes and uncertainty

A downstream turbine may encounter reduced wind speed and altered turbulence due to an upstream wake. Wake interaction depends on direction, spacing, controls and atmospheric conditions. Other energy reductions include equipment downtime, electrical losses, icing and curtailment. A simple retained-energy factor can summarize explicitly stated assumptions, but it does not represent the detailed physics or hourly coincidence of those effects. [@manwell2009], §§2.5 and 2.6; [@wagner2009], wind-farm operation.

Separate variability from uncertainty. Weather changes from hour to hour and year to year; uncertainty also arises from measurement, long-term adjustment and modelling choices. In common wind-project terminology, an annual-energy **P90** is an energy level expected to be exceeded with 90% probability under the defined uncertainty model—thus a lower quantile than P50. It is not “90% of the mean.” Synthetic bootstrap or Monte Carlo exercises demonstrate methods; they do not create a bankable uncertainty estimate from a few invented observations. [@manwell2009], §§2.5 and 2.6; [@wagner2009], wind-farm operation.

(wind-losses-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=losses
:width: 100%
:title: Wind energy losses and uncertainty explorer
:::

## Wind direction and circular data

Meteorological direction convention normally describes where wind comes **from**, measured clockwise from north. Directions 350° and 10° are close together, even though their ordinary numerical difference is large. Compute an equal-weight circular mean using Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

$$\bar\theta=\operatorname{atan2}\!\left(\overline{\sin\theta},\overline{\cos\theta}\right),$$

*Equation basis:* Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

then express it on the desired 0–360° interval. Convert degrees to radians before using NumPy trigonometric functions. The resultant length $R=\sqrt{\overline{\sin\theta}^{\,2}+\overline{\cos\theta}^{\,2}}$ measures directional concentration. If $R$ is near zero, the mean direction is unstable or undefined; opposing observations must not be assigned a spurious meaningful direction. Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

An equal-weight mean direction is not a speed-weighted vector-mean wind. For east/north velocity components under the meteorological “from” convention, $u=-v\sin\theta$ and $w_N=-v\cos\theta$. Distinguish these definitions when combining speed and direction. A wind rose groups directional frequencies, speed ranges or energy contributions; its weighting should be identified. Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

(wind-direction-explorer)=
:::{iframe} ../../wind-lab/aerodynamics.html?demo=direction
:width: 100%
:title: Circular wind direction explorer
:::

## Worked calculation before coding

Take a synthetic rotor radius of 20 m, density 1.225 kg/m³, undisturbed speed 8 m/s, $C_p=0.40$ and drivetrain efficiency 0.95. Original numerical substitution; Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

1. Swept area is $\pi20^2\approx1256.64$ m².
2. Available wind power is $0.5(1.225)(\pi20^2)(8^3)\approx394{,}081$ W. Divide by 1000 to obtain **394.08 kW**.
3. Rotor extraction is about $0.40(394.08)=157.63$ kW.
4. Electrical output is about $0.95(157.63)=149.75$ kW before additional site losses or equipment limits.
5. At a constant mean electrical output of 149.75 kW for two hours, energy is about 299.5 kWh.

*Source for these definitions/calculations:* Original numerical substitution; Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

Check the ordering: electrical power is below rotor power, which is below available wind power. This example uses a fixed coefficient at one operating point; the piecewise electrical curve is a separate alternative model, not another factor to multiply into the same calculation. Original numerical substitution; Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

## Environmental context and the route to Python

Wind-energy development must consider land or sea use, wildlife, noise, visual effects, access, materials and end-of-life management, as well as grid connection. A production calculation does not establish that a site is suitable. The supplementary *Wind Energy Handbook*, third edition, provides further reading on wakes, design loads, offshore structures and grid integration; its publisher information was checked for the reading list, not treated as a substitute for access to the full text [@burton2021wind].

Exercises 1–4 cover the Basic skills, Exercises 5–12 apply the models in Medium tasks, and Exercises 13–20 combine models, data, and validation in Hard tasks. Read the supplied *Wind Energy Explained*, especially the wind-resource and aerodynamic discussions, before interpreting those models [@manwell2009; @wagner2009].


<!-- expanded-theory:wind:end -->

## Concepts and equations: compact reference

### Available wind power

Air moving at speed $v$ through swept area $A$ has available power Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

$$P_{wind}=\frac{1}{2}\rho A v^3,\qquad A=\pi R^2$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

Rotor power is

$$P_{rotor}=C_pP_{wind}$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

where $C_p$ is the power coefficient. The ideal Betz maximum is Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

$$C_{p,max}=\frac{16}{27}\approx0.593$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

This is an aerodynamic upper bound, not a typical electrical efficiency. Generator and drivetrain losses occur after aerodynamic extraction. Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

### Air density and hub height

At the same wind speed, available power scales linearly with air density. A basic pressure-temperature approximation is [@manwell2009], §§2.3–2.5.

$$\rho\approx\frac{p}{R_dT}$$

*Equation basis:* [@manwell2009], §§2.3–2.5.

with dry-air gas constant $R_d=287.05$ J/(kg K), pressure in Pa, and temperature in K. [@manwell2009], §§2.3–2.5.

When only a reference-height wind speed is available, the empirical power law is often used: [@manwell2009], §§2.3–2.5.

$$v(z)=v(z_r)\left(\frac{z}{z_r}\right)^\alpha$$

*Equation basis:* [@manwell2009], §§2.3–2.5.

The shear exponent $\alpha$ is site- and stability-dependent; it is not universally $1/7$. [@manwell2009], §§2.3–2.5.

### Power curve, AEP, and capacity factor

A simplified curve has four regions:

1. zero below cut-in speed;
2. increasing output between cut-in and rated speed;
3. rated output between rated and cut-out speed;
4. zero at and above cut-out speed.

*Source for these definitions/calculations:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

For discrete wind states with probabilities $p_i$:

$$AEP=8760\sum_i P(v_i)p_i$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

and

$$CF=\frac{AEP}{P_r\,8760}$$

*Equation basis:* Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

Use a manufacturer power curve for real AEP work. A cubic interpolation is a teaching approximation only. Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

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

*Equation basis:* Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

where $k$ is shape and $c$ is scale. A fitted distribution compresses information and may miss calm periods, storms, seasonality, and direction-dependent wakes. Wind distribution and energy: [@manwell2009], §§2.4–2.5; [SciPy Weibull density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html).

```{code-cell} python
def weibull_pdf(speed_ms, shape_k, scale_c):
    if not all(np.isfinite(np.asarray(x, dtype=float)).all() for x in [speed_ms, shape_k, scale_c]):
        raise ValueError("Weibull inputs must be finite")
    if np.ndim(shape_k) or np.ndim(scale_c):
        raise ValueError("shape and scale must be scalar")
    speed = np.asarray(speed_ms, dtype=float)
    if shape_k <= 0 or scale_c <= 0 or np.any(speed < 0):
        raise ValueError("Weibull parameters must be positive; speed non-negative")
    # For 0 < k < 1 the density has an integrable singularity at v=0.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = speed / scale_c
        log_pdf = np.log(shape_k / scale_c) + (shape_k - 1) * np.log(x) - x ** shape_k
        density = np.exp(log_pdf)
    at_zero = np.inf if shape_k < 1 else (1 / scale_c if shape_k == 1 else 0.0)
    density = np.where(speed == 0, at_zero, density)
    density = np.where(np.isposinf(x ** shape_k), 0.0, density)
    return float(density) if density.ndim == 0 else density

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

Meteorological wind direction states where wind comes **from**: 0° is north and 90° is east. Arithmetic means fail near north—for example, 350° and 10° average to 180° arithmetically even though both are northerly. Use vector components: Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

$$\bar\theta=\operatorname{atan2}(\overline{\sin\theta},
\overline{\cos\theta})$$

*Equation basis:* Circular mean and zero-resultant limitation: [SciPy circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html).

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

[Download the theory notebook](windEnergy.ipynb).

## Common mistakes

- Using $v^2$ rather than $v^3$ in available wind power.
- Treating the Betz limit as turbine electrical efficiency.
- Extending the cubic law beyond rated speed or through cut-out.
- Calculating energy from mean wind speed; because of the cubic relationship, $P(\bar v)$ generally differs from $\overline{P(v)}$.
- Randomly shuffling time-series data before forecasting.
- Averaging wind directions arithmetically or reversing the meteorological convention.

(chapter-practice-20-python-exercises)=
## Practice

[Open the wind exercises](../section7/workbook/wind.md) in the Applied Python Workbook.

## Check your understanding

Continue with the [wind-physics quiz](../section7/renewableEnergyquizzes/windEnergy.md) and the [wind data lab](../section7/renewableEnergyquizzes/eolicenergy.md).

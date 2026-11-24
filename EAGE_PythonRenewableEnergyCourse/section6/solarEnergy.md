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

A photovoltaic (PV) cell converts part of incident solar radiation directly into DC electricity. An inverter converts DC to grid-compatible AC. Irradiance and cell temperature affect instantaneous output; shading, soiling, wiring, mismatch, inverter behavior, availability, and degradation affect delivered energy. Standard Test Conditions (STC) use 1000 W/m$^2$ irradiance, 25 °C cell temperature and the AM1.5 global (AM1.5G) reference spectrum ([IEA PVPS solar-resource handbook, p. 3-30](https://www.iea-pvps.org/wp-content/uploads/2021/06/IEA_PVPS_T16_Solar_Res_Handbook_2021-080621.pdf)) [@doePVPerformance; @doePVLongevity].

:::{iframe} ../../solar-lab/index.html
:width: 100%
:title: Interactive three-dimensional solar photovoltaic lab
:::

<!-- expanded-theory:solar:start -->

## Physical description: what solar energy systems do

Solar energy is electromagnetic radiation arriving from the Sun. A useful solar system must intercept that radiation, convert part of it into a useful form, and deliver that output to a load. The resource at a site depends on time, location, atmospheric conditions and the orientation of the receiving surface. A sunny location alone does not specify how much electricity a particular installation will deliver [@foster2010; @wade2003].

Two conversion routes must be distinguished. **Photovoltaic (PV) systems** convert absorbed light directly into DC electricity in semiconductor devices. **Solar thermal systems** absorb radiation as heat; that heat can supply hot water or an industrial process, or drive a heat engine in a concentrating solar power plant. A PV inverter is an electrical converter, whereas a solar thermal turbine is part of a thermodynamic cycle. Neither a PV module nor a thermal collector is, by itself, an energy-storage device. [@foster2010], Chapters 2, 4 and 5; [@wade2003], Chapter 3.

This chapter develops PV in detail because the Python exercises and final project model electrical PV output. The solar-thermal comparison below explains the wider meaning of solar energy without treating thermal and electrical output as interchangeable. Foster and Wade provide the introductory reading; Goswami adds detailed radiation geometry, PV operating points and thermal storage. Duffie and Beckman and Kalogirou provide additional reading on radiation, collectors and system modelling [@foster2010; @wade2003; @goswami2015; @duffie2013; @kalogirou2014solar].

### From sunlight to delivered electricity

The system sequence is **solar resource → module surface → PV cells → DC wiring → inverter → AC wiring and transformer → meter/load**. Losses and limits can occur at every stage. A grid-connected system usually exports or self-consumes AC electricity. An off-grid system may use a battery and charge controller; then the chosen model must also track stored energy and battery conversion losses. [@wade2003], Chapters 2–6 and 9; [@foster2010], Chapters 5 and 7.

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

*Source for these definitions/calculations:* [@wade2003], Chapters 3–5 and 9; [@foster2010], Chapters 5 and 7.

Series and parallel connections change voltage and current differently. For identical modules operating compatibly, series connection adds voltage while maintaining string current; parallel connection adds current at the common voltage. Real mismatch, partial shading, cable losses, bypass diodes and inverter operating limits complicate this ideal picture. A shaded module is therefore not always represented accurately by reducing the whole array's irradiance by one average percentage [@wade2003; @foster2010].

## Solar resource and measurement definitions

### Irradiance, irradiation and the receiving plane

#### Irradiance

**Irradiance**, denoted here by $G$, is radiant power incident per unit area, in W/m². **Irradiation**, denoted by $H_{POA}$ for a module plane, is incident radiant energy per unit area over a specified period, often kWh/m². The subscripts identify the measurement surface: a horizontal sensor and a tilted module generally receive different irradiance. [@foster2010], Chapter 2, especially pp. 39–40; [Sandia POA irradiance](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/).

The principal resource components are global horizontal irradiance (GHI), direct normal irradiance (DNI), and diffuse horizontal irradiance (DHI). For a sun above the horizon, consistent measurements approximately satisfy

$$GHI=DNI\cos\theta_z+DHI,$$

where $\theta_z$ is solar zenith angle, measured from the vertical. DNI is defined on a plane normal to the direct solar beam. DHI represents sky-diffuse radiation on the horizontal plane. Do not substitute DNI directly into a flat-plate PV model as though it were the total radiation on the modules. Direct relation: [IEA PVPS solar-resource handbook, §2.7, Eq. 2-2a](https://www.iea-pvps.org/wp-content/uploads/2021/06/IEA_PVPS_T16_Solar_Res_Handbook_2021-080621.pdf).

(solar-horizontal-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=horizontal
:width: 100%
:title: Horizontal irradiance explorer
:::

#### Plane-of-array irradiance

**Plane-of-array irradiance** $G_{POA}$ includes the direct beam projected onto the module plane, sky diffuse radiation and ground-reflected radiation. A useful conceptual decomposition is

$$G_{POA}=DNI\max(\cos\theta_i,0)+G_{sky,POA}+G_{ground,POA},$$

where $\theta_i$ is the beam incidence angle on the front face. The diffuse and reflected terms require a radiation-transposition model; they are not generally equal to DHI. Module tilt, azimuth, horizon obstruction, trackers and albedo affect the result. The exercise inputs explicitly provide POA irradiance so that a separate solar-position and transposition model is not silently omitted [@foster2010; @duffie2013].

(solar-poa-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=poa
:width: 100%
:title: Plane-of-array irradiance explorer
:::

#### Irradiation over time

For interval-average measurements,

$$H_{POA}=\frac{1}{1000}\sum_i G_{POA,i}\Delta t_i,$$

when $G$ is in W/m² and $\Delta t$ is in hours; $H$ is then kWh/m². The factor of 1000 converts Wh to kWh. For instantaneous samples, a chosen numerical integration method, such as a trapezoidal approximation, is needed instead. Always state which type of observation the dataset contains.

*Equation basis:* [@foster2010], Chapter 2, especially pp. 39–40; [Sandia POA irradiance](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/).

(solar-irradiation-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=irradiation
:width: 100%
:title: Interval irradiation explorer
:::

### A first model for a tilted module

For an unobstructed front surface above uniform, diffusely reflecting (Lambertian) ground, an **isotropic-sky** approximation assumes that diffuse sky radiance is the same in every direction. With module tilt $\beta$ measured from horizontal and dimensionless ground albedo $\rho_g$, it gives

$$G_{POA}=DNI\max(\cos\theta_i,0)
+DHI\frac{1+\cos\beta}{2}
+\rho_g GHI\frac{1-\cos\beta}{2}.$$

(solar-tilt-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=tilt
:width: 100%
:title: Tilted-module irradiance explorer
:::

All irradiances are in W/m². The two view factors represent the fractions of the sky and ground seen by the tilted front surface. The direct term is zero when the sun is behind that surface. This approximation excludes obstructions, row shading, rear-side collection and anisotropic diffuse radiation; it is an introductory transposition model, not a complete site simulation. [@goswami2015], §2.5.3, pp. 68–69, Eqs. 2.46–2.51; compare §2.6.2 for anisotropic radiation.

The incidence angle is between the sun direction and the module normal. When solar and surface azimuth use the **same origin and positive direction**,

$$\cos\theta_i=\cos\theta_z\cos\beta+
\sin\theta_z\sin\beta\cos(\gamma_s-\gamma_p),$$

where $\theta_z$ is solar zenith, $\gamma_s$ solar azimuth and $\gamma_p$ module azimuth. The interactive diagram uses azimuth clockwise from north. Goswami uses azimuth positive west of south. Convert conventions before combining sources. Trigonometric Python functions require radians. With the sun and module facing the same azimuth, the daylight Python example uses $\theta_i=|\theta_z-\beta|$. [@goswami2015], §2.5.3, Eq. 2.48 and Figure 2.21; [Python math functions](https://docs.python.org/3/library/math.html#trigonometric-functions).

(solar-incidence-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=incidence
:width: 100%
:title: Solar incidence angle explorer
:::

```{code-cell} python
import math

# Synthetic daylight example: sun and panel face the same azimuth.
dni_wm2, dhi_wm2 = 800.0, 100.0
zenith_deg, tilt_deg = 40.0, 30.0
incidence_deg = abs(zenith_deg - tilt_deg)
albedo = 0.2
ghi_wm2 = dni_wm2 * math.cos(math.radians(zenith_deg)) + dhi_wm2
def isotropic_poa(dni, dhi, ghi, tilt, incidence, reflectance):
    cos_tilt = math.cos(math.radians(tilt))
    beam = dni * max(math.cos(math.radians(incidence)), 0.0)
    sky = dhi * (1 + cos_tilt) / 2
    ground = reflectance * ghi * (1 - cos_tilt) / 2
    return beam, sky, ground

beam_wm2, sky_wm2, ground_wm2 = isotropic_poa(
    dni_wm2, dhi_wm2, ghi_wm2, tilt_deg, incidence_deg, albedo)
poa_wm2 = beam_wm2 + sky_wm2 + ground_wm2
print(f"GHI: {ghi_wm2:.2f}; POA: {poa_wm2:.2f} W/m²")
print(f"Beam: {beam_wm2:.2f}; sky: {sky_wm2:.2f}; ground: {ground_wm2:.2f}")
# A horizontal front surface recovers GHI under these assumptions.
horizontal_poa = sum(isotropic_poa(dni_wm2, dhi_wm2, ghi_wm2, 0, zenith_deg, albedo))
assert math.isclose(horizontal_poa, ghi_wm2)
# Independent fixed fixture keeps this test valid when the example inputs change.
reference_ghi = 800 * math.cos(math.radians(40)) + 100
reference_poa = sum(isotropic_poa(800, 100, reference_ghi, 30, 10, .2))
assert math.isclose(reference_poa, 890.6976581571471, rel_tol=1e-10)
```

**Try it:** set tilt to zero and incidence equal to zenith; explain why ground reflection disappears. Then restore the original angles and change albedo to 0.8, treating it as a hypothetical bright-ground scenario. Explain which term changes and why this is not an increase in PV conversion efficiency. These are original exercises using the model above. [@goswami2015], §2.5.3.

### Time and data quality

**Solar time is not generally clock time.** Longitude within a time zone, the equation of time and daylight-saving conventions affect the conversion. Do not interpret 12:00 on a dataset's clock as solar noon without checking its time basis. [@goswami2015], §2.3.1, pp. 42–45.

Record the timestamp convention, time zone, interval length and whether each timestamp marks an interval start or end. Nighttime zero irradiance can be valid; missing data are not zero. Sensor misalignment, shading of the sensor, snow, soiling, calibration drift and time shifts can distort a production comparison. A radiation sensor at one point does not necessarily represent every module in a large array. [IEA PVPS analytical monitoring, sensor maintenance and quality](https://iea-pvps.org/wp-content/uploads/2020/01/IEA-PVPS_T13-D2_3_Analytical_Monitoring_of_PV_Systems_Final.pdf); timestamp handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

## PV conversion: definitions, equations and limits

### The current–voltage characteristic

An illuminated PV device does not supply an arbitrary fixed voltage and current simultaneously. Its operating point lies on an **I–V curve** determined by irradiance, cell temperature and device properties. Electrical DC power is

$$P_{DC}=VI.$$

*Sources and assumptions for this equation group:* [@foster2010], §5.13, pp. 125–126 and p. 139; [@wade2003], Chapter 3. Reading caution: Foster p. 125 mislabels the zero-current condition; open circuit has zero current, as stated on p. 139. Its Eq. 5.15 on p. 126 also applies fill factor twice. Use $P_{mp}=V_{mp}I_{mp}=V_{oc}I_{sc}FF$ and $\eta=P_{mp}/P_{in}$ instead; see [DOE, Eqs. 3.12–3.14](https://www.energy.gov/sites/default/files/2017/08/f36/USDOE_WVSC_Project_Final_Report_5-30-2014_R1.pdf).

(solar-operating-point-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=operating-point
:width: 100%
:title: Photovoltaic operating point and DC power explorer
:::

At open circuit, current is zero and voltage is $V_{oc}$; at short circuit, voltage is zero and current is $I_{sc}$. Neither condition delivers useful electrical power. At the maximum-power point, $P_{mp}=V_{mp}I_{mp}$. The fill factor

$$FF=\frac{V_{mp}I_{mp}}{V_{oc}I_{sc}}$$

(solar-fill-factor-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=fill-factor
:width: 100%
:title: Photovoltaic fill factor explorer
:::

Fill factor describes the shape of the I–V curve and is dimensionless. It is not conversion efficiency or capacity factor. Irradiance strongly affects photocurrent, while voltage and power also respond to temperature. MPPT attempts to operate near the available maximum rather than at either I–V endpoint [@wade2003; @foster2010]; see also [@goswami2015], §§9.2–9.2.2, pp. 525–532. MPPT and inverter conversion efficiency describe different operations: choosing the available DC operating point and converting DC electrical power to AC.

The [I–V explorer](https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/generated/solar-lab/resource.html?demo=iv) models one uniformly illuminated silicon module. Change irradiance and cell temperature to see the same maximum-power voltage marked on both curves. The dashed traces show the module at standard test conditions. [De Soto single-diode model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/single-diode-equivalent-circuit-models/de-soto-five-parameter-module-model/); [pvlib module example](https://pvlib-python.readthedocs.io/en/latest/gallery/iv-modeling/plot_singlediode.html).

(solar-iv-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=iv
:width: 100%
:title: Photovoltaic current and power curve explorer
:::

For a simple area model,

$$P_{DC}=G_{POA}A\eta_{PV},\qquad
\eta_{PV}=\frac{P_{DC}}{G_{POA}A}.$$

Here $A$ is the module area defined consistently with the efficiency specification, in m². With irradiance in W/m², output is W. Holding $\eta_{PV}$ constant is useful for a first calculation but omits operating-temperature, spectral and low-light effects.

### STC, nameplate and temperature

**Standard Test Conditions (STC)** use 1000 W/m² irradiance, 25°C **cell** temperature and the AM1.5 global (AM1.5G) reference spectrum ([IEA PVPS solar-resource handbook, p. 3-30](https://www.iea-pvps.org/wp-content/uploads/2021/06/IEA_PVPS_T16_Solar_Res_Handbook_2021-080621.pdf)). A kWp rating commonly denotes rated DC kilowatts under these conditions. Ambient air at 25°C does not imply cells at 25°C: absorbed sunlight usually warms the module. Actual output can occasionally exceed the STC rating under favorable conditions, so nameplate is not a universal hard ceiling on DC power. [@foster2010], §§5.13–5.14; [@pvlibDocs].

The course uses a first-order DC model. It assumes the supplied POA irradiance equals effective irradiance: optical incidence-angle, spectral and soiling corrections are unity unless separately specified. This is a reduced model, not the complete PVWatts workflow [@pvlibDocs].

$$P_{DC}=P_{STC}\frac{G_{POA}}{G_{STC}}
\left[1+\gamma_P(T_c-T_{STC})\right].$$

$\gamma_P$ is the fractional power-temperature coefficient in K⁻¹ or °C⁻¹; a datasheet value of −0.4%/°C becomes −0.004/°C in Python. Celsius and kelvin temperature **differences** are equal. Use cell temperature $T_c$, not ambient temperature $T_a$, in the correction. For the negative coefficient assumed here, warmer cells reduce output at fixed irradiance [@foster2010; @pvlibDocs].

*Sources for the DC temperature correction:* [@foster2010], §§5.13–5.14; [@pvlibDocs].

(solar-dc-temperature-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=dc-temperature
:width: 100%
:title: Temperature-corrected solar DC power explorer
:::

The introductory thermal approximation is

$$T_c=T_a+\frac{G_{POA}}{800\ \mathrm{W/m^2}}
\left(T_{NOCT}-20\ ^\circ\mathrm{C}\right).$$

The nominal operating cell temperature (NOCT) convention refers to 800 W/m² irradiance, 20°C ambient air and 1 m/s wind under specified mounting and open-circuit conditions. Check the module datasheet: newer NMOT ratings use different electrical loading conventions. The equation above is a classroom reduction; the [Sandia/SAM NOCT model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/) additionally accounts for wind, mounting and electrical efficiency. This simplified equation does not explicitly model wind cooling, mounting geometry, heat capacity or transient behavior. Use appropriate thermal models and datasheet conventions in a detailed study. Flooring a negative extrapolated DC estimate at zero avoids impossible negative generation but does not make the underlying extrapolation physically valid.

(solar-cell-temperature-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=cell-temperature
:width: 100%
:title: Solar cell temperature and NOCT explorer
:::

### Inverter efficiency, clipping and loss accounting

Define potential AC output before clipping as $P_{AC,pot}=\eta_{inv}P_{DC}$. A simple inverter-limited model gives

$$P_{AC}=\min(P_{AC,pot},P_{AC,r}),\qquad
P_{clip}=\max(P_{AC,pot}-P_{AC,r},0).$$

*Equation basis:* [@foster2010], Chapter 7; [@wade2003], inverter/system-sizing sections. Constant efficiency and the hard AC cap are course assumptions.

This assumes a constant inverter efficiency and omits startup thresholds and standby consumption. Inverter conversion loss is $P_{DC}-P_{AC,pot}$, while clipping is discarded potential AC output above the AC limit. They are different quantities. A manufacturer AC power model may already include conversion losses; multiplying by the same efficiency again would double-count them.

The **DC/AC ratio** is $P_{STC,DC}/P_{AC,r}$. Increasing it can improve use of an inverter at lower irradiance but may increase clipping near peak resource. It is an engineering and economic trade-off, not a guarantee of higher performance ratio. Independent sequential fractional losses $\ell_j$ give a retained fraction $\prod_j(1-\ell_j)$, provided the loss boundaries do not overlap.

(solar-inverter-clipping-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=inverter-clipping
:width: 100%
:title: Solar inverter clipping and loss explorer
:::

## Energy, yield and performance indicators

For AC powers averaged over their respective intervals (or constant within them), electrical energy is $E_{AC}=\sum_i P_{AC,i}\Delta t_i$. Use kW with hours for kWh or MW with hours for MWh. Instantaneous samples need an integration rule and an interpolation assumption. Annual energy requires annual coverage or a documented extrapolation; a clear-sky day is not an annual resource model.

| Indicator | Definition | What it answers |
|---|---|---|
| Specific/final yield $Y_f$ | $E_{AC}/P_{STC,DC}$, often kWh/kWp | How much AC energy was delivered per DC nameplate unit? |
| Reference yield $Y_r$ | $H_{POA}/G_{STC}$, in hours | What equivalent full-reference-irradiance duration was available? |
| Performance ratio $PR$ | $Y_f/Y_r$, dimensionless | What normalized system yield was achieved under the stated boundary? |
| AC capacity factor | $E_{AC}/(P_{AC,r}T)$ | How much energy was delivered relative to continuous AC-rated output? |
| Availability | Fraction of a defined time or energy opportunity when equipment is available | Was the system able to operate? |

*Source for these definitions/calculations:* Yield and performance-ratio definitions: [DOE-hosted PV performance assessment, §2](https://www.energy.gov/sites/default/files/2022-01/understanding-solar-photovoltaic-system-performance.pdf). Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

PR is not module efficiency: its denominator is an irradiation-normalized nameplate yield. Temperature, soiling, downtime and measurement conventions affect it. An unexpectedly high PR calls for checking definitions, sensors and conditions; it is not resolved simply by clipping every ratio to one. Keep DC-rated and AC-rated denominators explicit.

## Solar thermal: the related but different conversion route

A thermal collector delivers useful heat to a circulating fluid. For a single-phase fluid with approximately constant heat capacity,

$$\dot Q_u=\dot m c_p(T_{out}-T_{in}).$$

$\dot Q_u$ is heat-transfer rate in W when mass flow is kg/s and $c_p$ is J/(kg K). Collector thermal efficiency is $\eta_{th}=\dot Q_u/(G A)$ under specified measurement boundaries. It is not a PV electrical efficiency. Heat loss to the surroundings grows as the collector becomes hotter relative to ambient conditions, so a fixed thermal efficiency is generally inadequate over a broad temperature range.

*Equation basis:* [@foster2010], Chapter 4, Eqs. 4.5 and 4.10, pp. 75–76; the single-phase heat balance neglects shaft work and kinetic/potential energy changes.

Concentrating systems use optics and tracking to deliver beam radiation to a receiver; their resource assessment relies strongly on DNI. A concentrating solar power plant then converts collected heat to electricity through a heat engine, potentially using thermal storage between collection and generation. A flat-plate PV array can use direct and diffuse radiation without concentrating it. These distinctions matter when choosing the input dataset and interpreting a claimed solar conversion efficiency [@foster2010; @duffie2013; @kalogirou2014solar]. **Reading caution:** Foster et al. §4.1 overstates the need for concentration above 100 °C. Some nonconcentrating evacuated-tube systems can supply higher temperatures under suitable conditions; performance depends on design and operation. See [NREL’s solar industrial heat review, §5.1.1](https://www.nrel.gov/docs/fy21osti/80574.pdf).

The flat-plate collector below uses a steady-state teaching model. While the pump runs, absorbed solar power is split between useful fluid heating and heat loss to the ambient air. The model estimates loss as $UA(T_{mean}-T_a)$, with $T_{mean}=(T_{in}+T_{out})/2$. The pump stops when the selected flow is zero or solar input cannot provide positive useful heat. [NREL flat-plate collector evaluation](https://www.nrel.gov/docs/fy16osti/66215.pdf).

Here $U$ is an illustrative first-order loss coefficient in W/(m² K), based on mean fluid temperature, and $A$ is collector area in m². The teaching expression omits nonlinear loss terms; coefficients defined using inlet temperature or other test conventions cannot be inserted unchanged without checking the model and conversion. See the [NREL/GSA collector evaluation](https://www.gsa.gov/cdnstatic/Honeycomb_Solar_Thermal_Collector_Evaluation-FINAL.pdf), p. 30, for distinct test formulations.

(solar-thermal-collector-explorer)=
:::{iframe} ../../solar-lab/resource.html?demo=thermal-collector
:width: 100%
:title: Solar thermal collector heat-balance explorer
:::

### Thermal storage: energy is not power

**Recording transcript:** Thermal storage energy is not power. For an approximately incompressible liquid or solid, with no phase change and negligible pressure work, sensible stored heat is approximately mass times specific heat times the temperature change. This assumes constant specific heat and excludes losses. A larger mass stores more sensible heat for the same temperature change, but it does not by itself imply a phase change. Latent heat is additional energy absorbed or released during a phase change. For the example water tank, the stored heat is about 9.29 kilowatt hours of heat; at a constant net heating rate of 2 kilowatts, ideal charging takes about 4.64 hours. These thermal quantities are not electrical output.

For an approximately incompressible liquid or solid storage mass without phase change, with negligible pressure-work differences, the sensible stored-heat change can be approximated by

$$Q_{stored}=m\int_{T_1}^{T_2}c_p(T)\,dT
\approx m c_p(T_2-T_1).$$

Here $m$ is mass in kg, $c_p$ specific heat in J/(kg K), and $T_2-T_1$ a temperature difference in K (numerically equal to the difference in °C). $Q_{stored}$ is in joules; divide by $3.6\times10^6$ to obtain kWh of **heat**. The second expression assumes constant specific heat and excludes losses during heating or storage. For a compressible gas, specify the process and thermodynamic boundary: for example, internal-energy change in a rigid ideal-gas tank uses $c_v$, while $c_p$ relates to enthalpy change. It is not the collector's heat-transfer rate $\dot Q_u$ or electrical output. [@goswami2015], §4.2.1, p. 206, Eq. 4.1.

```{code-cell} python
# Original teaching example: ideal heating of a water tank, no phase change.
mass_kg = 200.0
cp_j_kg_k = 4180.0  # assumed constant for this exercise
initial_c, final_c = 20.0, 60.0
stored_j = mass_kg * cp_j_kg_k * (final_c - initial_c)
stored_kwh_heat = stored_j / 3_600_000
net_heating_kw = 2.0  # assumed constant net heat rate reaching the water
charging_hours = stored_kwh_heat / net_heating_kw
print(f"Stored heat: {stored_kwh_heat:.3f} kWh_th")
print(f"Ideal charging time: {charging_hours:.3f} h")
assert abs(stored_kwh_heat * 3_600_000 - stored_j) < 1e-6
```

Latent storage also includes phase-change energy, $mL$, where $L$ is specific latent heat in J/kg; sensible heating before and after the transition must be accounted for when applicable. Neither $mc_p\Delta T$ alone nor a battery's electrical kWh rating describes that full thermal balance. [@goswami2015], §4.2.2, pp. 207–208, Eqs. 4.2–4.3.

**Try it:** double tank mass while keeping the same temperatures and heating rate. Predict how stored heat and charging time change, then check in Python. For a harder extension, introduce a heat-loss rate that depends on tank temperature and explain why charging time can no longer be found using a constant net rate. [@goswami2015], §§4.2.1 and 4.3.

## Worked calculation before coding

Consider a synthetic 10 kW DC array with an 8 kW AC inverter, POA irradiance 800 W/m², ambient temperature 25°C, NOCT 45°C, $\gamma_P=-0.004$/°C and inverter efficiency 0.97.

1. Estimate cell temperature: $T_c=25+(800/800)(45-20)=50$°C.
2. Calculate the temperature multiplier: $1-0.004(50-25)=0.90$.
3. Calculate DC output: $10(800/1000)(0.90)=7.20$ kW.
4. Apply inverter conversion: $7.20(0.97)=6.984$ kW potential AC.
5. Compare with the inverter rating: $\min(6.984,8)=6.984$ kW, with no clipping.
6. If these are mean conditions for a half-hour teaching interval, estimate $6.984(0.5)=3.492$ kWh.

*Source for these definitions/calculations:* Original numerical substitution into the specified DC, temperature and inverter models. [@pvlibDocs]. Specified simplified NOCT approximation; compare the fuller [Sandia/SAM thermal model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/).

The last calculation applies a nonlinear model to interval-average inputs. Rapid irradiance or temperature changes can make this differ from the average of higher-resolution modeled output. The numerical result is a model estimate, not a measured performance guarantee.

## From the explanation to the Python exercises

Begin with irradiance, area, electrical rating and time units. Exercises 1–4 cover the Basic skills; Exercises 5–12 apply the models in Medium tasks, and Exercises 13–20 combine models, data, and validation in Hard tasks. The tested functions below deliberately model the simpler PV relationships; they do not replace a detailed shading, electrical string-design or solar-thermal simulation.

For preparation, read Wade's electricity and PV sections and Foster's PV-system treatment. Use Duffie and Beckman for additional radiation/thermal-system reading and Kalogirou for a broader comparison of solar conversion systems. Keep dated textbook prices or market statistics separate from the physical principles used here [@wade2003; @foster2010; @duffie2013; @kalogirou2014solar].

<!-- expanded-theory:solar:end -->

## Concepts and equations: compact reference

### Irradiance versus irradiation

- **Irradiance** $G$ is instantaneous power per area, W/m$^2$.
- **Irradiation** is irradiance integrated over time, commonly kWh/m$^2$.

*Source for these definitions/calculations:* [@foster2010], Chapter 2, pp. 39–40.

Do not label an hourly sum of irradiance values as W/m$^2$; after multiplying by the time step it is an energy-per-area quantity.

### From solar resource to PV power

For area $A$ and conversion efficiency $\eta$:

$$P_{DC}=GA\eta$$

*Sources and assumptions for this equation group:* PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3. [@pvlibDocs]. Specified simplified NOCT approximation; compare the fuller [Sandia/SAM thermal model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/).

For a rated array, a convenient first-order model is

$$P_{DC}=P_{STC}\frac{G}{G_{STC}}
\left[1+\gamma_P(T_c-25)\right]$$

where $\gamma_P$ is normally negative and $T_c$ is cell temperature in °C. A simple NOCT approximation is:

$$T_c=T_a+\frac{G_{POA}}{800\ \mathrm{W/m^2}}(T_{NOCT}-20\ ^\circ\mathrm{C})$$

This is a teaching model, not a bankable performance model. The temperature equation is an empirical approximation and should not be extrapolated carelessly.

### AC output and energy

With inverter efficiency $\eta_{inv}$ and AC rating $P_{AC,r}$:

$$P_{AC}=\min(P_{DC}\eta_{inv},P_{AC,r})$$

For interval-average AC powers (or powers constant within each interval):

$$E_{AC}=\sum_i P_{AC,i}\Delta t_i$$

For instantaneous samples, a numerical rule such as trapezoidal integration estimates energy under an assumed interpolation; the exact energy between samples remains unknown.

Performance ratio (PR) is a normalized energy-quality indicator. One common definition is:

$$PR=\frac{E_{AC}/P_{STC,DC}}{H_{POA}/G_{STC}}$$

where $H_{POA}$ is plane-of-array irradiation and $P_{STC,DC}$ is the DC nameplate rating at STC. Use consistent kW/kWh or MW/MWh units and express $G_{STC}$ in matching units (1 kW/m² or 0.001 MW/m²). State the exact convention because PR boundaries can differ. A measured-to-model energy ratio used by some sources has a different denominator and is not automatically equivalent to this STC-normalized ratio. For example, 80 MWh measured divided by 90 MWh modelled is 0.889, while the same measured energy divided by a 100 MWh irradiation-based reference is 0.800. Before comparing ratios, identify the reference model or irradiation boundary and whether unavailable periods are excluded.

*Equation basis:* Yield and performance-ratio definitions: [DOE-hosted PV performance assessment, §2](https://www.energy.gov/sites/default/files/2022-01/understanding-solar-photovoltaic-system-performance.pdf). PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

## Tested Python functions

When multiple inputs are pandas series, these helpers require unique, nonmissing, matching indexes in the same order; they do not align data automatically. With an unlabeled array, the caller must ensure its order corresponds to the labeled data.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def cell_temperature_c(irradiance_wm2, ambient_c, noct_c=45.0):
    """Simplified NOCT estimate; validate the model range for the application."""
    indexes = [getattr(value, "index", None) for value in (irradiance_wm2, ambient_c, noct_c)]
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
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [irradiance_wm2, ambient_c, noct_c]):
        raise ValueError("model inputs must be finite")
    irradiance = np.asarray(irradiance_wm2, dtype=float)
    if np.any(irradiance < 0):
        raise ValueError("irradiance must be non-negative")
    ambient = np.asarray(ambient_c, dtype=float)
    if np.any(ambient <= -273.15) or np.any(np.asarray(noct_c, dtype=float) <= -273.15):
        raise ValueError("Temperatures must exceed absolute zero")
    cell = ambient + irradiance / 800 * (noct_c - 20)
    if np.any(cell <= -273.15):
        raise ValueError("Calculated temperature must exceed absolute zero")
    return cell

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
    cell_c = cell_temperature_c(irradiance_wm2, ambient_c, noct_c)
    irradiance = np.asarray(irradiance_wm2, dtype=float)
    temperature_factor = np.maximum(1 + gamma_p * (cell_c - 25), 0)
    dc_power = dc_rating_mw * irradiance / 1000 * temperature_factor
    ac_power = np.minimum(dc_power * inverter_efficiency, ac_rating_mw)
    return float(ac_power) if ac_power.ndim == 0 else ac_power

assert pv_ac_power_mw(0, 25, 10, 8) == 0
assert 0 <= pv_ac_power_mw(1000, 25, 10, 8) <= 8
```

### Why the checks matter

Input validation stops physically impossible negative irradiance and invalid efficiency values. `np.asarray` allows irradiance and ambient temperature to be scalars or broadcast-compatible arrays; ratings and efficiency parameters in this example are scalar configuration values. `np.maximum` prevents a linear temperature approximation from producing negative power outside its valid range. Array calculations: [@numpyDocs]. These checks enforce basic input bounds; they do not establish the valid cell-temperature range of a particular module or the accuracy of the linear model. Check the module datasheet and the applicable model conditions before extrapolating.

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

[Download the theory notebook](solarEnergy.ipynb).

## Common mistakes

- Confusing irradiance (W/m$^2$) with irradiation (kWh/m$^2$).
- Applying a temperature correction to ambient temperature instead of estimated cell temperature without stating the simplification.
- Omitting inverter clipping or applying system losses twice.
- Summing power without multiplying by the time-step duration.
- Treating the simplified model as a substitute for site-specific shading, orientation, and loss modelling.
- Calling DC nameplate power an energy value.

(solar-practice)=
## Practice

[Open the solar exercises](../section7/workbook/solar.md) in the Applied Python Workbook.

## Check your understanding

Continue with the [solar quiz](../section7/renewableEnergyquizzes/solarEnergy.md).

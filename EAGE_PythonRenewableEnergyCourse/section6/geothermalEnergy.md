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

:::{iframe} ../../geothermal-lab/index.html
:width: 100%
:title: Interactive three-dimensional geothermal binary-cycle plant
:height: 1350px
:::

## Closed-loop designs: U-shaped and coaxial

In a **closed-loop geothermal system**, a working fluid circulates inside sealed conduits and exchanges heat with the surrounding formation. In an ideal sealed loop the circulating fluid does not enter the formation; heat transfer through the borehole wall and rock is central. This differs from hydrothermal production and injection, where reservoir fluid flows through the rock. White et al. study both water and supercritical CO₂, a range of geometries and operating conditions, and both direct heat use and electricity production [@white2024closedloop].

![Conceptual comparison of U-shaped and coaxial closed-loop geothermal paths](../images/geothermal-closed-loop-configurations.svg)

*Original teaching schematic, not to scale.* Blue indicates the cooler inlet path and red the warmer outlet path. The coaxial drawing illustrates inlet flow down the annulus and return through the inner tube; other flow directions are possible. The paper's Figure 2 shows its corresponding axisymmetric numerical domains, conductive material and model boundaries. Its segmented rock domains are coupled through the circulating fluid; this schematic does not reproduce their numerical mesh [@white2024closedloop].

| Feature | U-shaped | Coaxial (pipe-in-pipe) |
|---|---|---|
| Flow path | Separate descending and ascending legs joined at depth | Inner tube and surrounding annulus provide opposing paths inside a borehole |
| Heat exchange | Working fluid exchanges heat along the loop; insulation can reduce unwanted heat loss | Heat exchange with the rock competes with thermal short-circuiting between the inner and outer streams |
| Hydraulic trade-off | Pressure losses depend on length, diameter, fluid and rate | Smaller available flow passages can create large pressure losses at high rates |
| Design choices | Depth, lateral length, diameter, insulation and possibly multiple laterals | Those choices plus inner/outer diameters, pipe insulation and flow direction |

### Heat transfer, flow area and useful energy

For a homogeneous solid with constant properties and conduction as the only transport process, the transient formation temperature $T$ obeys

$$\rho_r c_r\frac{\partial T}{\partial t}=k_r\nabla^2T, \qquad \alpha=\frac{k_r}{\rho_r c_r}.$$

$k_r$ is rock thermal conductivity in W/(m K), $\rho_r$ is density in kg/m³, $c_r$ is specific heat in J/(kg K), and thermal diffusivity $\alpha$ is in m²/s. In hot dry rock, fluid extraction cools rock near the borehole; heat must conduct from farther away, so outlet temperature can decline rapidly at first and more slowly later. The paper discusses $k_r/\sqrt{\alpha}$, a thermal effusivity in J/(m² K s$^{1/2}$), as a relevant rock property. A real model must couple the fluid, pipe and formation with appropriate geometry and boundaries; this equation alone is not a closed-loop solution [@white2024closedloop].

For concentric tubes, let $D_i$ be the inside diameter of the inner tube, $D_o$ its outside diameter, and $D_b$ the inside diameter of the surrounding flow passage. Then

$$A_{tube}=\frac{\pi D_i^2}{4},\qquad A_{annulus}=\frac{\pi(D_b^2-D_o^2)}{4},\qquad \frac{A_{annulus}}{A_{tube}}=\frac{D_b^2-D_o^2}{D_i^2}.$$

Use consistent diameter definitions and units. For example, $D_i=0.10$ m, $D_o=0.12$ m and $D_b=0.18$ m give an annulus-to-tube area ratio of **1.8**. Smaller passages may increase velocity and pressure losses for a given mass flow; their full evaluation requires fluid properties, roughness, elevation and a hydraulic model.

At the fluid boundary, the paper defines thermal energy and an idealized maximum available work over time as

$$E_{thermal}=\int_0^\tau\dot m(h_{out}-h_{in})\,dt, \qquad E_{available}=\int_0^\tau\dot m[(h_{out}-h_{in})-T_{amb}(s_{out}-s_{in})],dt.$$

Here $h$ is specific enthalpy (J/kg), $s$ is specific entropy (J/(kg K)), $\dot m$ is kg/s and ambient temperature $T_{amb}$ is in kelvin. These expressions use the inlet/outlet states and reference in the paper; actual net generation also depends on the conversion plant and auxiliaries. The constant-$c_p$ expression elsewhere in this lesson is a liquid-water approximation; it must not replace property-based enthalpy and entropy for supercritical CO₂ or phase change [@white2024closedloop].

### What did the paper find?

The attached outlet-temperature plot corresponds to **two U-shaped water cases**, not a U-shaped versus coaxial comparison. Both use a 10 km lateral and the same Utah FORGE-based rock and depth; the curves differ in both mass flow and insulation length: **11.8 kg/s with 1,174 m insulation**, and **38 kg/s with 875 m insulation**. Four modeling approaches show close agreement in Figure 3. Their differences cannot be attributed to flow rate alone [@white2024closedloop].

For an explicit comparison of both geometries, the paper's Table 3 reports the following independently optimized **water** cases over 40 years. Both use 5 km depth, 20 km horizontal extent, 0.4445 m borehole diameter, 0.07 °C/m gradient, 4.5 W/(m K) conductivity, 60 °C inlet, a 7% discount rate and the paper's \$1,000/m drilling cost assumption. The settings describe a hypothetical extreme of the paper's parameter range, not a typical site.

| Reported quantity | U-shaped water | Coaxial water |
|---|---:|---:|
| Independently optimized mass flow | 43.0 kg/s | 39.2 kg/s |
| Average production temperature | 205 °C | 203 °C |
| Average thermal production | 26.0 MW thermal | 23.8 MW thermal |
| Average electricity generation | 3.9 MW electric | 3.5 MW electric |
| Reported LCOE | \$96.6/MWh electric | \$93.5/MWh electric |

These are modeled averages and costs under the study assumptions, not forecasts or measurements. The different flow rates and design details mean this table does **not** isolate geometry at identical operating conditions; nor does it establish a universal winner. For a more representative coaxial example, the authors report 3 W/(m K), a 45 °C/km gradient and approximately 37 kg/s, producing an average 130 °C outlet and 0.922 MW electric under their other stated assumptions. See Table 3 and §3.3.1 for the complete parameter context [@white2024closedloop].

The paper expresses levelized costs using discounted annual costs and discounted annual energy, for example

$$LCOE=\frac{\sum_t(C_t+O_t)/(1+r)^t}{\sum_t E_{e,t}/(1+r)^t}.$$

$C_t$ and $O_t$ are capital and operating costs, $E_{e,t}$ is electrical energy and $r$ is the discount rate. Include an initial $t=0$ investment if it occurs at project start, and include each cost and energy period once. LCOE is not electricity-sale revenue, and its assumptions differ from the Field Designer's simple cumulative game balance. White et al.'s GeoCLUSTER results are accessible through the [Geothermal Data Repository](https://gdr.openei.org/submissions/1473); interpolation and inputs are subject to the paper's stated limitations.

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt

# White et al. (2024), Table 3: independently optimized water cases.
comparison = pd.DataFrame({
    'Configuration': ['U-shaped', 'Coaxial'],
    'Flow (kg/s)': [43.0, 39.2],
    'Thermal (MW)': [26.0, 23.8],
    'Electric (MW)': [3.9, 3.5],
    'LCOE (USD/MWh)': [96.6, 93.5],
})
ax = comparison.plot.bar(x='Configuration', y=['Thermal (MW)', 'Electric (MW)'],
                         color=['#d46b42', '#2589aa'], rot=0)
ax.set(ylabel='Average modeled power (MW)', title='Closed-loop water cases: study assumptions')
plt.tight_layout()
comparison
```

<!-- expanded-theory:geothermal:start -->

## Physical description: useful heat from the subsurface

Geothermal energy is thermal energy in the Earth. A geothermal project does not produce energy merely by drilling to a high temperature: it needs a technically usable heat resource, a way to transfer heat to the surface, a suitable conversion or direct-use system, and an operating strategy that manages the subsurface response. Temperature, accessible fluid flow, pressure, chemistry and permeability all matter [@grant2011].

Heat within the Earth includes contributions from its formation and radioactive decay. At a local project scale, however, the important question is how heat reaches and is stored in the reservoir and how quickly it can be extracted. Natural replenishment does not guarantee that a particular well field can support an arbitrary extraction rate. Some systems can be operated for long periods, while excessive drawdown, cooling or fluid-management problems can limit output. [@grant2011], §§2.2–2.5.

### Resource and utilization categories

| System or use | Description | Consequence for modelling |
|---|---|---|
| Hydrothermal resource | Naturally occurring hot fluid in permeable rock | Need reservoir pressure, temperature, flow pathways and fluid properties |
| Enhanced geothermal system (EGS) | Improves or creates fluid pathways in hot rock | Hydraulic connectivity, stimulation response and reservoir management become central |
| Closed-loop subsurface system | Circulates a working fluid through sealed underground conduits | Heat transfer through the well and surrounding formation can constrain output |
| Direct heat use | Delivers heat to a suitable process, building or heat network | Calculate useful thermal energy at the delivery temperature, not electrical output |
| Geothermal power generation | Converts a portion of extracted heat into electricity | Include cycle efficiency, cooling and parasitic electricity consumption |
| Ground-source heat pump | Uses the ground as a heat source or sink with electrical compressor work | Track electricity input and heat delivery; this is not geothermal electricity generation |

*Source for these definitions/calculations:* [@grant2011], Chapter 2; [@doeEGS; @doeGeothermalElectricity]. Heat-pump accounting: Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle).

Hydrothermal, EGS and closed-loop describe aspects of the subsurface system; dry-steam, flash and binary describe surface electricity-conversion arrangements. These classifications answer different questions and should not be used interchangeably. The supplied Grant and Bixley book develops reservoir concepts and simplified models; DiPippo is additional reading for power-conversion systems [@grant2011; @diPippo2016geothermal].

## Reservoir terms and transport processes

**Porosity** is the fraction of bulk rock volume occupied by pores or voids. **Permeability** describes the medium's ability to transmit fluid; connected pathways matter, so high porosity alone does not guarantee a productive well. Fractures can dominate fluid flow even when matrix permeability is low. **Transmissivity** incorporates the ability of a formation interval to transmit fluid over its effective thickness and depends on the convention used. [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

**Reservoir pressure** helps drive fluid toward a producing well, while pressure decline can reduce deliverability or increase pumping requirements. **Production wells** bring fluid to the surface; **injection wells** return fluid or supply circulation. Injection can help manage fluid mass and pressure, but poorly placed or strongly connected injection can cause premature cooling at producers. **Thermal breakthrough** is the arrival of a cooling influence along flow paths, not simply the passage of one calendar year [@grant2011].

A conceptual fluid-flow relationship is Darcy's law. For a one-dimensional horizontal, single-phase flow without a gravity term, [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

$$q_D=-\frac{k_{perm}}{\mu}\frac{dp}{dx}.$$

*Equation basis:* [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

$q_D$ is Darcy flux in m/s, $k_{perm}$ is intrinsic permeability in m², $\mu$ is dynamic viscosity in Pa s, and $dp/dx$ is pressure gradient in Pa/m. The negative sign represents flow toward decreasing pressure. Actual geothermal flow may involve gravity, temperature-dependent properties, fractures and multiple phases, so this reduced relationship is not a complete well model. It explains why temperature alone cannot establish producible mass flow. [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

(geo-darcy-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=darcy
:width: 100%
:title: Darcy flux explorer
:::

### Conduction and convection

Conduction transfers heat through a material without bulk transport of that material. Advection carries heat with moving fluid; buoyancy-driven circulating flow is commonly described as convection. In low-permeability conductive regions, a temperature gradient can help estimate deeper temperature over an appropriate interval. In permeable convecting systems, simple downward extrapolation of the shallow gradient may be misleading [@grant2011].

The linear conductive approximation is $T(z)=T_0+G_Tz$, where the depth coordinate and gradient must use matching units. If depth is in kilometres and gradient in °C/km, the product is a temperature difference in °C. This is a local model, not a universal rule that temperature always rises at one fixed rate everywhere. [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

Fourier's law is

$$\mathbf q=-k_T\nabla T,$$

*Equation basis:* [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

where thermal conductivity $k_T$ is W/(m K) and heat flux $\mathbf q$ is W/m². Thermal conductivity and intrinsic permeability are different physical properties; separate symbols help avoid confusion. If depth $z$ is positive downward and temperature rises downward, the conductive heat-flux vector points upward. For conductivity 2.5 W/(m K) and gradient 30 K/km, the upward heat-flux magnitude is $2.5(30/1000)=0.075$ W/m². A gradient is not itself a heat flux. [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

(geo-gradient-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=gradient
:width: 100%
:title: Geothermal gradient and heat flux explorer
:::

## The surface plant and its energy boundaries

A typical conversion chain is **reservoir → production well → fluid collection/separation or heat exchanger → power cycle → generator → cooling/auxiliaries → reinjection**, with the detailed route depending on fluid state and technology. Pressure losses, pumping work and heat losses occur between reservoir and delivered output. [@doeGeothermalElectricity]; reservoir-to-well limits: [@grant2011], Chapter 2.

### Dry-steam, flash and binary conversion

**Dry-steam plants** use a suitable steam resource to drive a turbine. **Flash plants** reduce the pressure of sufficiently hot liquid so that part of it vaporizes; a separator supplies steam to a turbine while residual liquid follows its own handling or reinjection path. **Binary plants** transfer geothermal heat through an exchanger to a separate working-fluid cycle. The geothermal fluid and the turbine working fluid therefore need not be the same fluid. [@doeGeothermalElectricity]; [@grant2011], Chapter 2.

The plant type must match the temperature, pressure, phase state and chemistry of the resource. There is no single efficiency that applies to all geothermal plants. Cooling conditions and auxiliary demand also affect exported electricity. The course's constant-efficiency, single-phase calculation is most useful for learning heat-to-electricity accounting; it is not a detailed flash or binary cycle simulation [@grant2011; @doeGeothermalElectricity].

DiPippo's fourth edition is a supplementary textbook covering power-conversion systems and case studies. Its publisher lists a late-2015 release while the copyright/edition page identifies 2016; this course cites the 2016 edition year. Current plant counts should not be inferred from the historical statistics in that edition [@diPippo2016geothermal].

## Thermal power: mass flow and enthalpy

For a steady heat exchanger with negligible shaft work and changes in kinetic and potential energy, the heat removed from the geothermal stream is [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

$$\dot Q_{extracted}=\dot m(h_p-h_r),$$

*Equation basis:* [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

where $\dot m$ is mass flow in kg/s, and $h_p$ and $h_r$ are specific enthalpies at the chosen production and return states. With $h$ in J/kg, output is W; with $h$ in kJ/kg, output is kW. This neglects heat leakage between the specified measurement states and the heat exchanger; it is the heat extracted from the stream under the defined boundary, not necessarily all the stored heat in the reservoir. [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

For a single-phase liquid with approximately constant specific heat, [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

$$h_p-h_r\approx c_p(T_p-T_r),\qquad
P_{th}=\dot m c_p(T_p-T_r).$$

*Equation basis:* [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

| Symbol | Definition | Unit used in the course |
|---|---|---|
| $\dot m$ | Produced fluid mass flow | kg/s |
| $c_p$ | Specific heat capacity, approximated as 4180 for water | J/(kg K) |
| $T_p$, $T_r$ | Production and reinjection/return temperatures at specified locations | °C for the difference |
| $h_p$, $h_r$ | Specific enthalpy at the two stream states | J/kg or kJ/kg, explicitly identified |
| $P_{th}$ | Thermal power extracted under the chosen boundary | W or MW thermal |
| $P_{gross}$, $P_{net}$ | Gross generated and net exported electrical power | W or MW electrical |

*Source for these definitions/calculations:* [@grant2011], §§3.2 and 3.6; [@foster2010], §3.8.1, applied to the stated steady-flow boundary.

A temperature difference of 80°C is also 80 K. Absolute temperature ratios, by contrast, require kelvin. Salinity, temperature and pressure affect fluid properties. In two-phase flow, energy includes latent heat and phase fractions; applying one constant liquid-water heat capacity to a steam/water mixture can substantially misrepresent the heat balance [@grant2011].

(geo-heat-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=heat
:width: 100%
:title: Thermal power explorer
:::

## Heat-engine limits, net output and direct heat use

For an ideal reversible engine operating between fixed hot and cold reservoirs, Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$\eta_C=1-\frac{T_c}{T_h},$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

with both absolute temperatures in kelvin. It is an upper bound for that idealized setting, not the efficiency to assign to a plant. A real geothermal stream cools through a temperature range, heat exchangers require finite temperature differences, and real machinery dissipates energy. A full thermodynamic assessment needs the actual cycle and fluid states. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-carnot-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=carnot
:width: 100%
:title: Carnot efficiency explorer
:::

The introductory model uses

$$P_{gross}=\eta_{conv}P_{th},\qquad
P_{net}=P_{gross}-P_{pump}-P_{aux}.$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

If parasitic consumption is represented as a fraction $f_{par}$ of **gross electrical power**, then $P_{net}=P_{gross}(1-f_{par})$. State that denominator explicitly. Ten percent of gross electricity is not ten percent of thermal input. Net exported power is the boundary used for project energy and capacity-factor calculations. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-conversion-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=conversion
:width: 100%
:title: Gross and net power explorer
:::

A simple hydraulic pumping estimate is

$$P_{pump}=\frac{\Delta p\,Q}{\eta_{pump}},$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

where pressure rise is Pa, liquid volume flow is m³/s and pump efficiency is dimensionless; power is W. Use volume flow here, rather than inserting mass flow without dividing by density. Pressure and flow requirements may change during operation, so a fixed parasitic fraction can hide an important source of declining net output. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-pump-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=pump
:width: 100%
:title: Pumping power explorer
:::

In direct-use heating, useful thermal output can be delivered without converting it to electricity. The delivery-temperature requirement matters: heat available below the required process temperature is not automatically usable without a heat pump or another upgrade. For a heat pump, heating coefficient of performance is $COP_h=\dot Q_{delivered}/P_{electric}$ and can exceed one because delivered heat includes heat moved from the source as well as electrical input. That does not violate energy conservation and is not an electrical-generation efficiency. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-heatpump-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=heatpump
:width: 100%
:title: Heat pump explorer
:::

## Reservoir storage, replenishment and production decline

An illustrative sensible-heat inventory for a uniform bulk reservoir volume is [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

$$E_{heat}\approx V\left[(1-\phi)\rho_r c_r+\phi\rho_f c_f\right](T-T_{ref}).$$

*Equation basis:* [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

$V$ is bulk volume, $\phi$ porosity, and the subscripts $r$ and $f$ refer to rock and fluid. Use coherent SI units to obtain joules. This lumped expression assumes uniform temperature and properties and omits phase change. It is not a recoverable electrical reserve: only some heat may be technically accessible, extraction occurs over time, and electrical conversion incurs further limitations [@grant2011].

(geo-storage-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=storage
:width: 100%
:title: Reservoir heat inventory explorer
:::

Pressure response, fluid recharge and heat recharge operate on different pathways and time scales. Maintaining pressure by reinjection does not guarantee that production temperature is maintained. Conversely, observing short-term stable output does not prove indefinite sustainability. Reservoir monitoring and calibrated coupled models are needed for long-run operating decisions. [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

For teaching, a compounded annual power decline is

$$P_y=P_1(1-d)^{y-1}.$$

*Equation basis:* [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

The exponent is zero in year 1. If $d=0.02$, each subsequent year retains 98% of the preceding year's assumed power. This is different from subtracting two percent of the original power every year. A continuous exponential model $P(t)=P_0e^{-Dt}$ uses a decline constant $D$ with inverse-time units; for annual steps, $D=-\ln(1-d)$ per year. [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

Fitting $\ln P$ against time requires positive observations and changes the assumed error model. Reserve later observations before fitting, compare against a simple baseline, and report the evaluated period. Intervention, changing flow, outages and reservoir connectivity can invalidate extrapolation. A prescribed or fitted decline curve is not, by itself, a reservoir simulator. [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

(geo-decline-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=decline
:width: 100%
:title: Production decline explorer
:::

## Energy, availability and operating limits

Integrate net power over the actual duration: $E_{net}=\sum_iP_{net,i}\Delta t_i$. If an annual model specifies net power **when operating** and a separate availability $a$, $E_y=P_y a T_y$ is an appropriate simplified expression. If measured hourly power already includes downtime, multiplying by availability again would double-count it. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

Use a fixed defined net nameplate $P_{r,net}$ for $CF=E_{net}/(P_{r,net}T)$. Do not use observed maximum output as a substitute unless that is explicitly the intended metric. If a plant has a net power limit, apply it after the resource-to-net-power calculation under the stated boundary. Negative net electricity can physically occur when auxiliaries consume more power than is generated; a model restricted to positive exported generation should state how such operating periods are handled rather than silently changing their meaning. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-energy-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=energy
:width: 100%
:title: Energy and capacity factor explorer
:::

## Worked calculation before coding

Suppose a synthetic liquid-water resource produces 50 kg/s at 150°C and returns at 70°C. Assume $c_p=4180$ J/(kg K), gross conversion efficiency 0.12 and parasitic consumption equal to 10% of gross electricity. Original numerical substitution with assumed constant properties and efficiencies; Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

1. Temperature drop is $150-70=80$ K.
2. Thermal power is $50(4180)(80)=16{,}720{,}000$ W, or **16.72 MW thermal**.
3. Gross electricity is $16.72(0.12)=2.0064$ MW.
4. Parasitic demand is $2.0064(0.10)=0.20064$ MW.
5. Net output is $2.0064-0.20064=1.80576$ MW.
6. Four hours at that net output deliver **7.22304 MWh** of electricity.

*Source for these definitions/calculations:* Original numerical substitution with assumed constant properties and efficiencies; Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

The reservoir-to-fluid heat transfer, pressure losses and detailed power-cycle behavior are not solved by these six steps. Their purpose is to make the assumed conversion boundaries visible before implementing arrays and functions. Original numerical substitution with assumed constant properties and efficiencies; Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-worked-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=conversion
:width: 100%
:title: Worked example explorer
:::

## Environmental context and the route to Python

Geothermal projects need attention to fluid chemistry, scaling, corrosion, water use, noncondensable gases, injection management, possible subsidence and induced seismicity. The relevance and magnitude of each issue depend on the resource and operating method. Reinjection is not simply waste disposal; it interacts with the hydraulic and thermal behavior of the field [@grant2011].

Exercises 1–4 cover the Basic skills, Exercises 5–12 apply the models in Medium tasks, and Exercises 13–20 combine models, data, and validation in Hard tasks. Read Grant and Bixley Chapters 2–3 for the reservoir basis; use the additional DiPippo text to extend from the simplified heat balance toward actual surface conversion cycles [@grant2011; @diPippo2016geothermal].


<!-- expanded-theory:geothermal:end -->

## Concepts and equations: compact reference

### Geothermal gradient is not heat flux

A linear temperature-depth approximation is

$$T(z)=T_0+G_Tz$$

*Equation basis:* [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

where $G_T$ is the geothermal gradient, for example °C/km. This is a local approximation; gradients vary with geology, groundwater flow, and depth. [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

Conductive heat flux follows Fourier's law:

$$\mathbf{q}=-k\nabla T$$

*Equation basis:* [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

The negative sign indicates heat flows toward lower temperature. Gradient (K/m) and heat flux (W/m$^2$) are different quantities. [@grant2011], §§2.2–2.3 and §4.4.1 (conductive versus convective profiles).

(geo-compact-gradient-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=gradient
:width: 100%
:title: Gradient and heat flux explorer
:::

### Thermal power in produced fluid

For mass-flow rate $\dot m$, specific heat $c_p$, production temperature $T_p$, and reinjection temperature $T_r$: Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{th}=\dot m c_p(T_p-T_r)$$

*Equation basis:* Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

If $c_p$ is J/(kg K), the result is W. A constant liquid-water $c_p$ is a simplified sensible-heat model; steam, flashing, brines, and phase change require enthalpy: Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{th}=\dot m(h_p-h_r)$$

*Equation basis:* Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-heat-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=heat
:width: 100%
:title: Thermal power and enthalpy explorer
:::

### Efficiency, parasitic load, and net power

The reversible Carnot bound is

$$\eta_C=1-\frac{T_c}{T_h}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

where both temperatures must be in Kelvin. A plant cannot simply be assigned Carnot efficiency; real conversion efficiency is lower and technology-specific. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-carnot-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=carnot
:width: 100%
:title: Carnot efficiency explorer
:::

$$P_{gross}=\eta_{conv}P_{th}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{net}=P_{gross}-P_{pumps}-P_{aux}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

Net power, not gross generator power, is the relevant exported power. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-efficiency-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=conversion
:width: 100%
:title: Conversion and net power explorer
:::

### Energy and decline

For discrete time steps:

$$E_{net}=\sum_iP_{net,i}\Delta t_i$$

*Equation basis:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-energy-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=energy
:width: 100%
:title: Energy and availability explorer
:::

A simple annual decline scenario is

$$P_y=P_1(1-d)^{y-1}$$

*Equation basis:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

This is a scenario model, not a reservoir forecast. Sustainable production depends on reservoir response, injection strategy, pressure, thermal breakthrough, and operational constraints. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-decline-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=decline
:width: 100%
:title: Annual decline explorer
:::

## Tested Python functions

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def temperature_at_depth_c(depth_km, surface_c=15.0, gradient_c_per_km=30.0):
    if not all(np.isfinite(np.asarray(x, dtype=float)).all() for x in [depth_km, surface_c, gradient_c_per_km]):
        raise ValueError("model inputs must be finite")
    depth = np.asarray(depth_km, dtype=float)
    if np.any(depth < 0) or np.any(np.asarray(gradient_c_per_km) < 0):
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
    if not all(np.isfinite(np.asarray(x, dtype=float)).all() for x in [hot_c, cold_c]):
        raise ValueError("temperatures must be finite")
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

[Download the theory notebook](geothermalEnergy.ipynb).

## Common mistakes

- Using °C rather than Kelvin in a thermodynamic temperature ratio.
- Calling geothermal gradient a heat flux.
- Ignoring phase change when enthalpy, not constant $c_p\Delta T$, is required.
- Reporting gross power while omitting pumping and auxiliary loads.
- Treating a high subsurface temperature as proof of permeability, flow, or commercial viability.
- Subtracting a fixed decline percentage linearly instead of compounding the retained fraction.

(chapter-practice-20-python-exercises)=
## Practice

[Open the geothermal exercises](../section7/workbook/geothermal.md) in the Applied Python Workbook.

## Check your understanding

Continue with the [geothermal quiz](../section7/renewableEnergyquizzes/geothermalEnergy.md).

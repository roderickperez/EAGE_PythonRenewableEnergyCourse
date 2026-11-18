---
kernelspec:
  name: python3
  display_name: Python 3
---

# Geothermal Energy

## Learning goals

After this lesson you should be able to:

- trace the underground fluid path in conventional hydrothermal, enhanced geothermal and deep closed-loop systems;
- distinguish geothermal gradient, heat flux, thermal power, gross electrical power, and net electrical power;
- use Kelvin correctly in thermodynamic efficiency calculations;
- estimate heat carried by produced fluid;
- include conversion efficiency, parasitic load, and production decline;
- implement validated Python functions and scenario plots;
- explain why temperature alone does not establish a viable geothermal resource.

Electricity generation requires a heat resource, a way to bring heat to the surface and a plant that can convert part of that heat into electricity. The course compares **three deep heat-extraction approaches**. You can identify each one by following the fluid between the inlet and outlet.

:::{iframe} ../../geothermal-lab/index.html
:width: 100%
:title: Interactive three-dimensional geothermal binary-cycle plant
:height: 1350px
:::

## Three ways to extract geothermal heat

In the first animation, hot brine rises through the red production well and gives up heat in the plant's heat exchanger. Cooled brine goes down the blue reinjection well. Follow the dots through the permeable reservoir from reinjection toward production. The cutaway projects representative flow paths onto the rock face. At the surface, a separate working fluid takes heat from the brine and drives the binary-cycle turbine. The Field Designer also models flow through a permeable layer between injectors and producers. [@grant2011], §§2.2.3 and 2.3.3. [@doeGeothermalElectricity]

| Subsurface approach | Where the fluid travels underground | What to evaluate |
|---|---|---|
| **Conventional hydrothermal** | A well produces naturally occurring hot water or steam. Where the project reinjects fluid, it returns through naturally permeable pores and fractures. | Connected permeability, pressure, fluid chemistry, recharge and cooling from reinjection. |
| **Enhanced geothermal system (EGS)** | A typical two-well system uses stimulation to improve fracture connectivity in hot rock. Injected water crosses the engineered reservoir to a production well. | Fracture connectivity, contact area, injectivity, fluid losses, cooling and induced seismicity. |
| **Deep closed-loop system** | A working fluid circulates inside sealed tubing. Heat crosses the well wall from the formation. | Heat transfer through the formation and well, contact length, insulation and pumping losses. |

Hydrothermal and EGS wells both exchange fluid with the formation. The distinction is whether naturally productive pathways already support circulation or the project must engineer connectivity. Sealed closed loops can operate in low-permeability rock. Where groundwater is mobile, it can carry heat toward the outside of a sealed well. [@grant2011], §§2.2–2.5. [@doeGeothermalLiftoff; @doeGeoForge; @white2024closedloop]

## Comparing hydrothermal, EGS and closed-loop flow

Use the three views to follow fluid underground. In the **hydrothermal** view it spreads through permeable rock. In the **EGS** view it follows connected fractures. Switch off **Fractures connect both wells** to see a schematic case where the two wells lack a through-flow path. This control illustrates connectivity after stimulation. [@doeGeoForge]

(geo-closed-systems)=
:::{iframe} ../../geothermal-lab/closed-loop.html?demo=systems&v=5
:width: 100%
:height: 650px
:title: Interactive geothermal system comparison
:::

### Closed-loop designs: U-shaped and coaxial

Use **Well geometry** in the closed-loop view to switch between two ways of keeping the fluid in a sealed circuit. In the U-shaped layout, one or more sealed laterals connect descending and ascending legs. In the coaxial layout, a single borehole contains both the outer annulus and inner tube. The example sends fluid down the annulus and back up the inner tube after it turns at the toe. A coaxial well can be vertical or include a lateral. [@white2024closedloop], Figs. 1–2.

| Design question | U-shaped | Coaxial |
|---|---|---|
| How does fluid reach the return passage? | It follows a sealed lateral to the rising well leg. | It turns at the bottom or toe into the inner tube. |
| Where can heat be lost along the return path? | The return leg can transfer heat to cooler surrounding strata. | Heat can pass across the inner pipe wall from warm return fluid to cooler inlet fluid. |
| Which choices affect heat capture and pumping? | Lateral length, pipe diameter and insulation affect heat gain and pressure losses. Parallel laterals add contact area and can thermally interfere with each other. | The diameters set the inner and annular flow areas. Insulation and flow direction influence heat exchange between the two streams. |

The following equations and performance examples focus on **sealed closed loops**. White et al. examine both well geometries, two working fluids and two end uses. For comparison with EGS or hydrothermal fields, use the same output metric and state the resource and operating assumptions. The surface choice of dry steam, flash or binary conversion is covered later in the chapter. [@white2024closedloop; @doeGeothermalElectricity]

### Heat transfer, flow area and useful energy

For a homogeneous solid with constant properties and conduction as the only transport process, the transient formation temperature $T$ obeys

$$\rho_r c_r\frac{\partial T}{\partial t}=k_r\nabla^2T, \qquad \alpha=\frac{k_r}{\rho_r c_r}.$$

$k_r$ is rock thermal conductivity in W/(m K), $\rho_r$ is density in kg/m³, $c_r$ is specific heat in J/(kg K), and thermal diffusivity $\alpha$ is in m²/s. In a sealed loop in hot dry rock, the circulating fluid draws heat from the rock near the borehole. Heat then conducts from farther away, so the outlet temperature can decline rapidly at first and more slowly later. The paper discusses $k_r/\sqrt{\alpha}$, a thermal effusivity in J/(m² K s$^{1/2}$), as a relevant rock property. A real model must couple the fluid, pipe and formation with appropriate geometry and boundaries; this equation alone is not a closed-loop solution [@white2024closedloop].

(geo-closed-conduction)=
:::{iframe} ../../geothermal-lab/closed-loop.html?demo=conduction&v=5
:width: 100%
:height: 660px
:title: Transient rock conduction explorer
:::

For concentric tubes, let $D_i$ be the inside diameter of the inner tube, $D_o$ its outside diameter, and $D_b$ the inside diameter of the surrounding flow passage. Then

$$A_{tube}=\frac{\pi D_i^2}{4},\qquad A_{annulus}=\frac{\pi(D_b^2-D_o^2)}{4},\qquad \frac{A_{annulus}}{A_{tube}}=\frac{D_b^2-D_o^2}{D_i^2}.$$

Use consistent diameter definitions and units. For example, $D_i=0.10$ m, $D_o=0.12$ m and $D_b=0.18$ m give an annulus-to-tube area ratio of **1.8**. Smaller passages may increase velocity and pressure losses for a given mass flow; their full evaluation requires fluid properties, roughness, elevation and a hydraulic model.

(geo-closed-area)=
:::{iframe} ../../geothermal-lab/closed-loop.html?demo=area&v=5
:width: 100%
:height: 530px
:title: Coaxial tube flow areas explorer
:::

At the closed-loop fluid boundary, White et al. define thermal energy and idealized available work over time as

$$E_{thermal}=\int_0^\tau\dot m(h_{out}-h_{in})\,dt, \qquad E_{available}=\int_0^\tau\dot m[(h_{out}-h_{in})-T_{amb}(s_{out}-s_{in})]\,dt.$$

Here $h$ is specific enthalpy (J/kg), $s$ is specific entropy (J/(kg K)), $\dot m$ is kg/s and ambient temperature $T_{amb}$ is in kelvin. These expressions use the inlet/outlet states and reference in the paper; actual net generation also depends on the conversion plant and auxiliaries. The constant-$c_p$ expression elsewhere in this lesson is a liquid-water approximation; it must not replace property-based enthalpy and entropy for supercritical CO₂ or phase change [@white2024closedloop].

(geo-closed-energy)=
:::{iframe} ../../geothermal-lab/closed-loop.html?demo=energy&v=5
:width: 100%
:height: 570px
:title: Thermal energy and available work explorer
:::

### Comparing closed-loop performance

Figure 3 in White et al. benchmarks the numerical models using **two U-shaped water cases**. Both use a 10 km lateral and the same Utah FORGE-based rock and depth. One case uses **11.8 kg/s and 1,174 m of insulation**. The other uses **38 kg/s and 875 m of insulation**. Four modeling approaches show close agreement for each case. Both inputs change between cases, so this comparison cannot isolate the effect of flow rate. [@white2024closedloop], Fig. 3.

Table 3 compares **four electricity scenarios** across two geometries and two fluids. The authors searched for low levelized costs over a 40-year project life using 5 km vertical depth, a 0.4445 m borehole, rock conductivity of 4.5 W/(m K), a geothermal gradient of 70 °C/km, a 60 °C inlet, a 7% discount rate and an assumed drilling cost of \$1,000/m. Each column uses its own operating settings. These are hypothetical results at the upper end of the study’s parameter range. [@white2024closedloop], Table 3.

| Reported quantity | U-shaped, water | Coaxial, water | U-shaped, supercritical CO₂ | Coaxial, supercritical CO₂ |
|---|---:|---:|---:|---:|
| Selected mass flow (kg/s) | 43.0 | 39.2 | 100 | 69.6 |
| Horizontal extent (km) | 20 | 20 | 20 | 13 |
| Average outlet temperature (°C) | 205 | 203 | 190 | 174 |
| Average thermal output (MW) | 26.0 | 23.8 | 22.2 | 14.1 |
| Average electrical output (MW) | 3.9 | 3.5 | 5.0 | 2.8 |
| Levelized cost of electricity (USD/MWh) | 96.6 | 93.5 | 83.0 | 86.9 |

For the two water cases, the U-shaped design delivers more heat and electricity while the coaxial design reports a slightly lower electricity cost under the study assumptions. The CO₂ cases use a different surface cycle, and the U-shaped CO₂ case reaches the database's 100 kg/s flow limit. In a separate coaxial water example with a lower conductivity of 3 W/(m K) and a gradient of 45 °C/km, the model gives a 130 °C average outlet and 0.922 MW electric at about 37 kg/s. Compare results within the stated scenario. [@white2024closedloop], Table 3 and §3.3.1.

The paper expresses levelized costs using discounted annual costs and discounted annual energy, for example

$$LCOE=\frac{\sum_t(C_t+O_t)/(1+r)^t}{\sum_t E_{e,t}/(1+r)^t}.$$

$C_t$ and $O_t$ are capital and operating costs, $E_{e,t}$ is electrical energy and $r$ is the discount rate. Include an initial $t=0$ investment if it occurs at project start, and include each cost and energy period once. LCOE is not electricity-sale revenue, and its assumptions differ from the Field Designer's simple cumulative game balance. White et al.'s GeoCLUSTER results are accessible through the [Geothermal Data Repository](https://gdr.openei.org/submissions/1473); interpolation and inputs are subject to the paper's stated limitations.

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt

# White et al. (2024), Table 3: four electricity scenarios.
comparison = pd.DataFrame({
    'Configuration': ['U-shaped\nwater', 'Coaxial\nwater',
                      'U-shaped\nCO₂', 'Coaxial\nCO₂'],
    'Flow (kg/s)': [43.0, 39.2, 100.0, 69.6],
    'Thermal (MW)': [26.0, 23.8, 22.2, 14.1],
    'Electric (MW)': [3.9, 3.5, 5.0, 2.8],
    'LCOE (USD/MWh)': [96.6, 93.5, 83.0, 86.9],
})
ax = comparison.plot.bar(x='Configuration', y=['Thermal (MW)', 'Electric (MW)'],
                         color=['#d46b42', '#2589aa'], rot=0)
ax.set(ylabel='Average modeled power (MW)', title='Closed-loop electricity scenarios from White et al.')
plt.tight_layout()
comparison
```

<!-- expanded-theory:geothermal:start -->

## Physical description: useful heat from the subsurface

Geothermal energy is thermal energy in the Earth. A geothermal project does not produce energy merely by drilling to a high temperature: it needs a technically usable heat resource, a way to transfer heat to the surface, a suitable conversion or direct-use system, and an operating strategy that manages the subsurface response. Temperature, accessible fluid flow, pressure, chemistry and permeability all matter [@grant2011].

Heat within the Earth includes contributions from its formation and radioactive decay. At a local project scale, however, the important question is how heat reaches and is stored in the reservoir and how quickly it can be extracted. Natural replenishment does not guarantee that a particular well field can support an arbitrary extraction rate. Some systems can be operated for long periods, while excessive drawdown, cooling or fluid-management problems can limit output. [@grant2011], §§2.2–2.5.

### Resource and utilization categories

Grant and Bixley group their Chapter 2 discussion into **three sections by reservoir behavior**: conductive settings, liquid-dominated convective systems and vapor-dominated convective systems. In liquid-dominated reservoirs, liquid water is the principal mobile phase, even where some steam is present. In vapor-dominated reservoirs, steam controls the pressure profile, even where substantial liquid remains in the rock. This classification helps explain the natural state and production response. [@grant2011], §§2.2–2.5.

| Resource setting in the book | What distinguishes it |
|---|---|
| Warm groundwater basins and deep sedimentary aquifers | In the settings described by the book, geothermal gradients heat the water hosted by permeable rock. A production and reinjection doublet can supply district heating, as in the Paris Basin example. |
| Fault-controlled warm springs | Faults or fractures can carry warm water upward where conduction supplies heat from the surrounding rock. A spring indicates a flow path, and further testing is needed to establish a productive reservoir. |
| Geopressured formations | Deep confined brines can have unusually high pressures and may carry dissolved gas. Fluid chemistry, pressure and any gas handling matter to resource development. |
| Hot, low-permeability rock | Heat is available in the rock, and access depends on the chosen extraction design. Grant and Bixley describe creating a circulating reservoir by stimulating fractures for EGS. |
| Liquid- and vapor-dominated convective reservoirs | Natural circulation and phase distribution shape the temperature and pressure field. Production may change where boiling occurs or where a steam zone develops. |

Grant and Bixley discuss deep circulation and magmatic heat together as potential origins of convective resources. A magmatic heat source does not imply that wells circulate fluid through magma. Modern programs also study **superhot rock** as a target for both EGS and sealed closed-loop designs. **Coproduction** describes recovery of heat from water brought to the surface during oil or gas production. It identifies where the hot water comes from. [@grant2011], §§2.3.2 and 2.5. [@arpaSuperhot; @doeGeoCoproduced]

The end use is a separate decision. **Direct heat** supplies a building, district network or process at the required delivery temperature. **Power generation** uses heat to drive a conversion cycle and must account for cooling and auxiliary loads. Projects can also deliver both electricity and useful heat when they have a suitable heat customer. A **ground-source heat pump** normally uses relatively shallow ground or water as a heat source or sink and consumes electricity to move heat. Its circulation choices include horizontal, vertical and pond or lake closed loops, as well as open-loop groundwater systems. [@doeGeothermalLiftoff; @doeGeoCoproduced; @doeGeoHeatPumpGuide]

## Reservoir terms and transport processes

**Porosity** is the fraction of bulk rock volume occupied by pores or voids. **Permeability** describes the medium's ability to transmit fluid; connected pathways matter, so high porosity alone does not guarantee a productive well. Fractures can dominate fluid flow even when matrix permeability is low. **Transmissivity** incorporates the ability of a formation interval to transmit fluid over its effective thickness and depends on the convention used. [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

**Reservoir pressure** helps drive fluid toward a producing well, while pressure decline can reduce deliverability or increase pumping requirements. **Production wells** bring fluid to the surface; **injection wells** return fluid or supply circulation. Injection can help manage fluid mass and pressure, but poorly placed or strongly connected injection can cause premature cooling at producers. **Thermal breakthrough** is the arrival of a cooling influence along flow paths, not simply the passage of one calendar year [@grant2011].

For steady, horizontal, single-phase flow with uniform properties and negligible gravity along the flow path, Darcy’s law relates flux to the pressure gradient.

$$q_D=-\frac{k_{perm}}{\mu}\frac{dp}{dx}.$$

*Equation basis:* [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

$q_D$ is Darcy flux in m/s, $k_{perm}$ is intrinsic permeability in m², $\mu$ is dynamic viscosity in Pa s, and $dp/dx$ is pressure gradient in Pa/m. The negative sign represents flow toward decreasing pressure. In the explorer, the upper line shows the pressure drop and the dots mark relative flow through connected pathways. Raise permeability or steepen the pressure gradient to increase flux. Raise viscosity to slow it. Set the pressure gradient to zero to stop the flow. Actual geothermal flow may involve gravity, temperature-dependent properties, fractures and multiple phases, so this reduced relationship is not a complete well model. It explains why temperature alone cannot establish producible mass flow. [@grant2011], §§2.3–2.5, §3.3.1 and §3.7.

(geo-darcy-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=darcy&v=2
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
:::{iframe} ../../geothermal-lab/equations.html?demo=gradient&v=2
:width: 100%
:title: Geothermal gradient and heat flux explorer
:::

## The surface plant and its energy boundaries

Trace the fluid circuit and the energy conversion separately. In the hydrothermal binary-cycle example, brine goes from the production well to a heat exchanger and then to reinjection. Heat passes into a separate cycle whose vapor turns the turbine, which drives the generator. In a sealed deep loop, the cooled geothermal working fluid returns to the downflow passage. Pumping, cooling and heat losses affect the net electricity delivered. [@doeGeothermalElectricity; @white2024closedloop]

### Dry-steam, flash and binary conversion

| Surface conversion type | What drives the turbine | What to compare |
|---|---|---|
| **Dry steam** | Geothermal steam is routed to the turbine after any required conditioning. | Steam quality, pressure, contaminants and noncondensable gases. |
| **Flash steam** | Some hot liquid boils when pressure falls. A separator sends steam to the turbine. A double-flash plant recovers additional steam from the remaining liquid at a lower pressure. | Inlet enthalpy, flash pressure, steam fraction, scaling and reinjection conditions. |
| **Binary cycle** | A heat exchanger vaporizes a separate working fluid, typically in an organic Rankine cycle. The geothermal stream stays on its side of the exchanger. | Heat-exchanger performance, source and sink temperatures, cooling, pumps and usable return temperature. |

These **three surface plant families** describe how electricity is generated. A single site can combine conversion stages, for example by recovering heat from flash brine in a binary bottoming cycle. A binary plant can receive heat from a conventional hydrothermal reservoir, an EGS or a deep closed loop. Match the plant to the delivered fluid state, chemistry and temperature. [@doeGeothermalElectricity; @diPippo2016geothermal]

DiPippo's fourth edition is a supplementary textbook covering power-conversion systems and case studies. Its publisher lists a late-2015 release while the copyright/edition page identifies 2016; this course cites the 2016 edition year. Current plant counts should not be inferred from the historical statistics in that edition [@diPippo2016geothermal].

## Thermal power: mass flow and enthalpy

For a steady heat exchanger with negligible shaft work and changes in kinetic and potential energy, the first law gives the rate of heat removed from the geothermal stream. [@mitSteadyFlow]

$$\dot Q_{extracted}=\dot m(h_p-h_r).$$

Here $\dot m$ is mass flow in kg/s, and $h_p$ and $h_r$ are specific enthalpies at the heat exchanger inlet and outlet. With $h$ in J/kg, output is W. With $h$ in kJ/kg, output is kW. When inlet and outlet measurements are made at the wells, using the same expression for heat at the exchanger assumes negligible losses between those points and the exchanger.

For a single-phase liquid with approximately constant specific heat, approximate the enthalpy change from the temperature difference.

$$h_p-h_r\approx c_p(T_p-T_r),\qquad
P_{th}=\dot m c_p(T_p-T_r).$$

| Symbol | Definition | Unit used in the course |
|---|---|---|
| $\dot m$ | Produced fluid mass flow | kg/s |
| $c_p$ | Specific heat capacity, approximated as 4180 for water | J/(kg K) |
| $T_p$, $T_r$ | Production and reinjection/return temperatures at specified locations | °C for the difference |
| $h_p$, $h_r$ | Specific enthalpy at the two stream states | J/kg or kJ/kg, explicitly identified |
| $P_{th}$ | Thermal power extracted under the chosen boundary | W or MW thermal |
| $P_{gross}$, $P_{net}$ | Gross generated and net exported electrical power | W or MW electrical |

A temperature difference of 80°C is also 80 K. Absolute temperature ratios, by contrast, require kelvin. Salinity, temperature and pressure affect fluid properties. In two-phase flow, energy includes latent heat and phase fractions; applying one constant liquid-water heat capacity to a steam/water mixture can substantially misrepresent the heat balance [@grant2011].

(geo-heat-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=heat&v=2
:width: 100%
:title: Thermal power explorer
:::

## Heat-engine limits, net output and direct heat use

For an ideal reversible engine operating between fixed hot and cold reservoirs, Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$\eta_C=1-\frac{T_c}{T_h},$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

with both absolute temperatures in kelvin. It is an upper bound for that idealized setting, not the efficiency to assign to a plant. A real geothermal stream cools through a temperature range, heat exchangers require finite temperature differences, and real machinery dissipates energy. A full thermodynamic assessment needs the actual cycle and fluid states. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-carnot-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=carnot&v=2
:width: 100%
:title: Carnot efficiency explorer
:::

The introductory model uses

$$P_{gross}=\eta_{conv}P_{th},\qquad
P_{net}=P_{gross}-P_{pump}-P_{aux}.$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

If parasitic consumption is represented as a fraction $f_{par}$ of **gross electrical power**, then $P_{net}=P_{gross}(1-f_{par})$. State that denominator explicitly. Ten percent of gross electricity is not ten percent of thermal input. Net exported power is the boundary used for project energy and capacity-factor calculations. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-conversion-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=conversion&v=2
:width: 100%
:title: Gross and net power explorer
:::

A simple hydraulic pumping estimate is

$$P_{pump}=\frac{\Delta p\,Q}{\eta_{pump}},$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

where pressure rise is Pa, liquid volume flow is m³/s and pump efficiency is dimensionless; power is W. Use volume flow here, rather than inserting mass flow without dividing by density. Pressure and flow requirements may change during operation, so a fixed parasitic fraction can hide an important source of declining net output. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-pump-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=pump&v=2
:width: 100%
:title: Pumping power explorer
:::

In direct-use heating, useful thermal output can be delivered without converting it to electricity. The delivery-temperature requirement matters: heat available below the required process temperature is not automatically usable without a heat pump or another upgrade. For a heat pump, heating coefficient of performance is $COP_h=\dot Q_{delivered}/P_{electric}$ and can exceed one because delivered heat includes heat moved from the source as well as electrical input. That does not violate energy conservation and is not an electrical-generation efficiency. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-heatpump-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=heatpump&v=2
:width: 100%
:title: Heat pump explorer
:::

## Reservoir storage, replenishment and production decline

An illustrative sensible-heat inventory for a uniform bulk reservoir volume is [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

$$E_{heat}\approx V\left[(1-\phi)\rho_r c_r+\phi\rho_f c_f\right](T-T_{ref}).$$

*Equation basis:* [@grant2011], §§3.2 and 3.6 (storage and stored-heat estimates). The compounded/exponential decline equations are stipulated scenarios, not reservoir predictions.

$V$ is bulk volume, $\phi$ porosity, and the subscripts $r$ and $f$ refer to rock and fluid. Use coherent SI units to obtain joules. This lumped expression assumes uniform temperature and properties and omits phase change. It is not a recoverable electrical reserve: only some heat may be technically accessible, extraction occurs over time, and electrical conversion incurs further limitations [@grant2011].

(geo-storage-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=storage&v=2
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
:::{iframe} ../../geothermal-lab/equations.html?demo=decline&v=2
:width: 100%
:title: Production decline explorer
:::

## Energy, availability and operating limits

Integrate net power over the actual duration: $E_{net}=\sum_iP_{net,i}\Delta t_i$. If an annual model specifies net power **when operating** and a separate availability $a$, $E_y=P_y a T_y$ is an appropriate simplified expression. If measured hourly power already includes downtime, multiplying by availability again would double-count it. The annual model uses assumed availability and decline. See the EIA definition for capacity factor [@eiaCapacityFactor].

Use a fixed defined net nameplate $P_{r,net}$ for $CF=E_{net}/(P_{r,net}T)$. Do not use observed maximum output as a substitute unless that is explicitly the intended metric. If a plant has a net power limit, apply it after the resource-to-net-power calculation under the stated boundary. Negative net electricity can physically occur when auxiliaries consume more power than is generated; a model restricted to positive exported generation should state how such operating periods are handled rather than silently changing their meaning. The power boundary and nameplate rating must be consistent with the reported energy [@eiaCapacityFactor].

(geo-energy-explorer)=
:::{iframe} ../../geothermal-lab/equations.html?demo=energy&v=2
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
:::{iframe} ../../geothermal-lab/equations.html?demo=worked&v=2
:width: 100%
:title: Worked example explorer
:::

## Environmental context and the route to Python

In hydrothermal fields, formation-fluid chemistry can cause scaling, corrosion and gas-handling problems. Reinjection placement affects reservoir pressure and cooling. EGS adds the need to characterize stimulation response, water losses and induced seismicity. A sealed loop avoids producing formation brine and still requires sound well construction, fluid containment, pumping and surface heat rejection. Impacts depend on the site and the operating method. [@grant2011; @doeGeothermalLiftoff; @white2024closedloop]

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
Revisit the [temperature-gradient and heat-flux explorer](#geo-gradient-explorer).

### Thermal power in produced fluid

For mass-flow rate $\dot m$, specific heat $c_p$, production temperature $T_p$, and reinjection temperature $T_r$: Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{th}=\dot m c_p(T_p-T_r)$$

*Equation basis:* Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

If $c_p$ is J/(kg K), the result is W. A constant liquid-water $c_p$ is a simplified sensible-heat model; steam, flashing, brines, and phase change require enthalpy: Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{th}=\dot m(h_p-h_r)$$

*Equation basis:* Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-heat-explorer)=
Revisit the [produced-fluid heat explorer](#geo-heat-explorer).

### Efficiency, parasitic load, and net power

The reversible Carnot bound is

$$\eta_C=1-\frac{T_c}{T_h}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

where both temperatures must be in Kelvin. A plant cannot simply be assigned Carnot efficiency; real conversion efficiency is lower and technology-specific. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-carnot-explorer)=
Revisit the [Carnot limit explorer](#geo-carnot-explorer).

$$P_{gross}=\eta_{conv}P_{th}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

$$P_{net}=P_{gross}-P_{pumps}-P_{aux}$$

*Equation basis:* Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

Net power, not gross generator power, is the relevant exported power. Reversible heat-engine limit: [OpenStax, University Physics 2, §4.5, Eq. 4.5](https://openstax.org/books/university-physics-volume-2/pages/4-5-the-carnot-cycle). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-efficiency-explorer)=
Revisit the [thermal, gross and net power explorer](#geo-conversion-explorer).

### Energy and decline

For discrete time steps:

$$E_{net}=\sum_iP_{net,i}\Delta t_i$$

*Equation basis:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-energy-explorer)=
Revisit the [operating time and energy explorer](#geo-energy-explorer).

A simple annual decline scenario is

$$P_y=P_1(1-d)^{y-1}$$

*Equation basis:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

This is a scenario model, not a reservoir forecast. Sustainable production depends on reservoir response, injection strategy, pressure, thermal breakthrough, and operational constraints. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

(geo-compact-decline-explorer)=
Revisit the [annual decline explorer](#geo-decline-explorer).

## Drilling a geothermal well: fluid and hydraulics

Drilling creates the borehole; its circulation system must remove rock cuttings, cool the bit and drill string, and help manage pressure. In **direct circulation**, pumps send fluid down the drill pipe, through the bit and back up the annulus between pipe and borehole. Surface equipment separates cuttings before the fluid is recirculated. Dayan (2014) describes water-based muds, air and aerated fluids, and explains the trade-offs among hole cleaning, circulation losses, formation damage and well control [@dayan2014drilling].

:::{iframe} ../../geothermal-lab/drilling.html?v=3
:width: 100%
:height: 1000px
:title: Interactive geothermal drilling fluid explorer
:::

In the explorer, blue beads follow the drilling fluid down the pipe. Gold beads rise when the upward annular velocity exceeds the idealized settling speed and fall when settling dominates. The cuttings animation uses the same Stokes validity limit as the displayed calculation. A separate two-dimensional plot shows hydrostatic pressure in blue and circulating pressure in red against vertical depth. Select depth, flow, diameters, fluid density and viscosity, and a **prescribed annular friction loss**. Changes to flow do not automatically recalculate that loss: it is a separate illustrative input. The visualization is schematic, not to scale.

### Annular velocity and cuttings

For concentric circular sections, hole diameter $D_h$ and pipe outside diameter $D_p$ in metres, volumetric circulation rate $Q$ in m³/s, and an incompressible single-phase fluid,

$$A_a=\frac{\pi}{4}(D_h^2-D_p^2), \qquad v_a=\frac{Q}{A_a}.$$

The annular area $A_a$ is m² and average annular velocity $v_a$ is m/s. The explorer converts L/min to m³/s by dividing by 60,000. In the simplified vertical model, cuttings transport velocity is $v_t=v_a-v_s$, where $v_s$ is the settling or slip velocity relative to fluid. Flow direction, inclination, rotation, cuttings beds, shape, concentration, fluid rheology and temperature complicate real hole cleaning [@dayan2014drilling].

For an isolated sphere in a Newtonian fluid under creeping-flow assumptions, Stokes' ideal settling estimate and its particle Reynolds number are

$$v_s=\frac{(\rho_s-\rho_f)gd_s^2}{18\mu},\qquad Re_p=\frac{\rho_f v_s d_s}{\mu}.$$

$\rho_s$ and $\rho_f$ are particle and fluid density (kg/m³), $d_s$ is diameter (m), $\mu$ is dynamic viscosity (Pa s), and $g=9.80665$ m/s². The explorer uses a synthetic particle density of 2,650 kg/m³ and suppresses the estimate if $Re_p\ge1$. It is not valid for arbitrary particles, non-Newtonian mud or high-Reynolds conditions, and it is **not** an operational hole-cleaning decision rule. A negative ideal transport velocity is a model result, not a complete prediction of a cuttings bed.

### Hydrostatic and circulating pressure

For a constant-density single-phase fluid, true vertical depth $z$, a surface gauge pressure of zero, and an explicitly supplied annular frictional pressure drop $\Delta p_a$ over that depth,

$$p_h=\rho_fgz,\quad p_{circ}=p_h+\Delta p_a,\quad \rho_{ECD}=\rho_f+\frac{\Delta p_a}{gz}.$$

These SI equations express the hydrostatic, circulating-pressure and equivalent-circulating-density ideas discussed by Dayan. Pressure is in Pa, depth is in metres and ECD is kg/m³. The explorer assumes the specified total annular friction is distributed uniformly along depth for its red plotted line. Actual annular loss depends on geometry, flow, fluid rheology and other conditions; it is not total pump pressure, which also includes pipe, bit and surface losses. Gas expansion, thermal effects, pressure transients and formation flow are not represented [@dayan2014drilling].

For a simple water-bentonite Bingham interpretation using conventional 600 and 300 rpm viscometer readings, $PV=\theta_{600}-\theta_{300}$ in cP and $YP=\theta_{300}-PV$ in the standard field convention (lbf/100 ft²). This is a rheological model, not a direct substitution for Newtonian viscosity in the Stokes estimate. Gel strength and filtration are additional measured properties. Temperature changes, solids and additives affect fluid behavior; aerated fluids require multiphase treatment [@dayan2014drilling].

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

### Check the units in Python

```{code-cell} python
import math
depth, density, annular_loss = 2000.0, 1100.0, 1.0e6  # m TVD, kg/m³, Pa
hole, pipe, flow_l_min = 0.215, 0.125, 1200.0        # m, m, L/min
area = math.pi * (hole**2 - pipe**2) / 4
velocity = (flow_l_min / 60000) / area
hydrostatic = density * 9.80665 * depth
ecd = density + annular_loss / (9.80665 * depth)
print(f'Annular velocity: {velocity:.3f} m/s')
print(f'Hydrostatic pressure: {hydrostatic / 1e6:.3f} MPa')
print(f'Circulating pressure: {(hydrostatic + annular_loss) / 1e6:.3f} MPa')
print(f'ECD: {ecd:.1f} kg/m³')
```

**Investigate:** Hold geometry fixed and double the flow rate. What happens to annular velocity? Then change only the prescribed friction loss: which pressure and ECD values change? Compare the fluid types and their functions in Dayan's report, and identify which measurements would be needed before applying a real drilling program.

The explorer is a simplified educational calculation. It does not calculate pore pressure, fracture pressure, well stability, fluid losses, gas influx, temperature-dependent properties, complete pump pressure or operational safety margins. It must not be used to select an actual mud program or operate a well. Actual geothermal drilling requires site-specific engineering and approved well-control procedures.

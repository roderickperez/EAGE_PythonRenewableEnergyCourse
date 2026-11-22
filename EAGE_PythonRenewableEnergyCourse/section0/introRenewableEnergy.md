# Introduction to Renewable Energy

## Python preparation

Review programming fundamentals, notebooks and scientific libraries in the [Python Course](https://roderickperezaltamar.com/private-course/eage-python-course/index.html). Separate course access may be required.

## Energy, power and units

Energy measures an amount transferred or stored; power measures its rate. Instantaneous power is $P=dE/dt$. For constant or interval-average power, $E=P\Delta t$; for a time series, $E=\sum_iP_i\Delta t_i$. Two instantaneous samples do not by themselves determine the energy between them. For example, if power rises linearly from 0 to 2 MW over one hour, it delivers 1 MWh; treating either endpoint as constant would give 0 or 2 MWh instead. A numerical integration rule therefore needs an explicit interpolation assumption.

| Quantity | Unit | Conversion |
|---|---|---|
| Power | W, kW, MW | 1 MW = 1,000 kW = 1,000,000 W |
| Energy | J, kWh, MWh | 1 kWh = 3.6 million J; 1 MWh = 1,000 kWh |
| Irradiance | W/m² | Power per unit area |
| Irradiation | kWh/m² | Solar energy per unit area over a stated interval |

*Source for these definitions/calculations:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

A closed system can exchange energy with its surroundings; energy is conserved for the combined isolated system. Useful mechanical forms are $E_k=mv^2/2$ and $E_p=mgh$. Mechanical energy is conserved when only conservative forces do work. For DC electricity, $P=VI$; sinusoidal AC real power also depends on power factor. See Wade, Chapter 2 [@wade2003].

## Renewable technologies

Renewable resources are replenished by natural processes, but sustainable production has local limits. Renewable does not mean impact-free or unlimited extraction.

| Technology | First model | What the simple equation leaves out |
|---|---|---|
| Hydro | $P=\eta\rho gQH_{net}$ | Seasonal flow, ecological releases, reservoir operation, and calculation and variation of losses from gross head |
| Wind | $P_{rotor}=C_p\rho Av^3/2$ | Generator rating, cut-out, wakes, curtailment and losses |
| Solar PV | $P_{DC}=GA\eta$ | Calculation of plane-of-array irradiance from orientation, weather and shading; temperature-dependent efficiency, electrical mismatch and inverter losses |
| Geothermal | $P_{th}=\dot m c_p(T_p-T_r)$ | Phase change, reservoir response, pumping and electrical conversion |

For the PV equation, use irradiance and efficiency at the modelled operating conditions. If the supplied plane-of-array irradiance already includes shading, do not apply that irradiance reduction again; nonuniform shading can also cause electrical mismatch that the simple area model does not resolve.

*Source for these definitions/calculations:* [@jica2011], Chapter 3 (hydro); [@manwell2009], §3.2 (wind); [@foster2010], Chapter 5 (PV); [@grant2011], Chapters 2–3 (geothermal).

Hydropower has no fuel combustion at the turbine, but construction, ecosystems and reservoir greenhouse-gas emissions matter. Geothermal heat extraction can exceed local replenishment. Biomass sustainability depends on feedstock, land use and regrowth; lifecycle emissions also depend on system boundaries and timescale ([IPCC, Energy Systems, §6.4.2.6](https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-6/)). Nuclear is low-emissions electricity but is not classified as renewable in this course [@ifc2015; @grant2011; @manwell2009; @foster2010].

## Storage and variability

Pumped hydro moves water uphill using electricity and recovers part of that energy later. Batteries store energy electrochemically. Some concentrating solar power systems store heat, for example in molten salt, before generating electricity; PV panels do not themselves store electricity. Storage shifts energy in time and incurs losses. Report both power capacity (MW) and usable energy capacity (MWh), plus efficiency and initial storage state [@jrc2025hydro; @wade2003].

For a fixed rating, capacity factor is $CF=E/(P_{rated}T)$ over the same period and output boundary. It differs from conversion efficiency and availability. Use actual calendar hours: 8,760 in a non-leap year and 8,784 in a leap year. Published fleet statistics can use different capacity definitions, coverage and time-weighting; reproduce the provider's method rather than assuming a simple ratio of annual totals. For example, see the [EIA Electric Power Annual technical notes](https://www.eia.gov/electricity/annual/pdf/tech_notes.pdf). Resource and energy accounting: [@manwell2009], §§2.3–2.5.

## Energy context: dated evidence

The IEA's *Global Energy Review 2026* reports estimates for **2025**. Low-emissions sources, including renewables **and nuclear**, supplied 43% of global electricity in that report. This is an electricity-generation share, not a renewable share or a share of total energy supply [@iea2026].

For cost context consult IRENA's July 2026 *Renewable Power Generation Costs in 2025* [@irena2026]. Distinguish publication year from observation year. Scenario results are conditional projections, not observed facts; identify the edition and assumptions.

The Eurostat notebooks intentionally use bundled **August 2024 exports** of monthly net electricity generation in **GWh**. They are historical teaching snapshots of [dataset nrg_cb_pem, DOI 10.2908/NRG_CB_PEM](https://doi.org/10.2908/NRG_CB_PEM); that DOI identifies the evolving dataset rather than a frozen copy. Legacy charts in the data-source lesson should not be used for current totals or trends. Primary energy, final energy, generation and consumption have different accounting boundaries [@eurostat_energy_database; @owid_energy_mix].

<!-- expanded-theory:general:start -->

## A common framework for understanding renewable-energy systems

An energy system connects a **resource**, a **conversion device**, and a **useful output**. Before writing an equation, specify what crosses the system boundary. A solar module receives radiation and supplies DC electricity; a hydro turbine receives water with hydraulic head and supplies shaft power; a geothermal heat exchanger receives a hot fluid and transfers heat. Comparing the efficiencies of those devices without identifying their different inputs and outputs is misleading. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3. Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

Renewable resources are replenished through natural processes on relevant human timescales, but usable production is constrained by location, equipment and operating conditions. Solar radiation drives photosynthetic growth and much of the atmospheric and hydrological circulation. Geothermal systems use subsurface heat. The four principal electricity technologies studied here have different physical conversion chains, so they need different resource variables and equations [@foster2010; @jica2011; @manwell2009; @grant2011].

### Resource, conversion and delivery boundaries

| Technology | Resource variable that must be measured or estimated | Conversion chain | Electrical output boundary used later |
|---|---|---|---|
| Solar PV | POA irradiance and cell/ambient temperature | Radiation → semiconductor DC output → inverter | Inverter-limited AC power |
| Wind | Hub-height speed, density and, for a farm, direction | Air kinetic energy → rotor → drivetrain/generator | Rated and shutdown-limited electrical power |
| Hydro | Available water flow and net hydraulic head | Water mechanical energy → turbine → generator | Generator-limited electrical power after stated releases |
| Geothermal | Fluid mass flow, temperature/enthalpy and pressure context | Fluid heat → power cycle → generator | Gross electricity minus parasitic use, with a defined net rating |

*Source for these definitions/calculations:* Head, discharge and electrical output: [@jica2011], Chapter 3 and §8.3; [@ifc2015], Chapter 7. Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2. PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3. Reservoir heat and flow: [@grant2011], Chapters 2–3; efficiencies and decline rates are specified teaching assumptions.

No single resource proxy determines output. Solar irradiation does not specify inverter clipping; mean wind speed does not specify mean electrical power; river flow without head does not specify hydropower; temperature without deliverable flow does not specify geothermal output. The following chapters explain those missing links before asking you to implement them.

## Conservation, conversion efficiency and the quality of energy

The first law states that energy is conserved. For a defined control volume, a useful accounting statement over a time interval is

$$\Delta E_{stored}=E_{in}-E_{out},$$

where the totals include the relevant heat, work and energy carried by mass flows, with a consistent sign convention. The equation does not mean every incoming joule can be converted to useful electricity. Friction may convert mechanical energy into heat, while heat-engine conversion is limited by thermodynamics and by actual equipment behavior.

*Sources and assumptions for this equation group:* [@foster2010], §3.8 (thermodynamics); the balance is written for the stated system boundary.

Conversion efficiency is

$$\eta=\frac{\text{useful output energy}}{\text{specified input energy}}.$$

For a steady device the equivalent power ratio can be used. Always name the numerator and denominator. A PV module efficiency, a hydro turbine efficiency and a geothermal thermal-to-electric efficiency refer to different conversion stages. Multiplying stage efficiencies is valid only when the output of one stage is the input to the next and the loss definitions do not overlap.

**Energy quality** refers to the ability of an energy form to provide a desired service or useful work. Electricity can drive motors or be converted to heat, while low-temperature heat cannot all be converted back to electricity. This helps explain why geothermal thermal MW should not be directly added to wind electrical MW. Convert to a common useful-output boundary before comparing or summing sources [@grant2011; @wade2003].

### Power and energy through time

Power is the rate of energy transfer: $P=dE/dt$. For interval-average powers,

$$E=\sum_i\bar P_i\Delta t_i.$$

*Equation basis:* Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

One watt is one joule per second. One kWh is 3.6 MJ; one MWh is 3.6 GJ. A plant rated at 10 MW does not “generate 10 MW per hour”: when it operates at 10 MW for one hour, it generates 10 MWh. A rate of change of power, such as MW/min, is a different quantity used for ramping.

The time-weighted mean power is $\bar P=\sum_iP_i\Delta t_i/\sum_i\Delta t_i$. Equal durations guarantee that the ordinary arithmetic mean is equivalent; unequal durations can produce different results. For example, powers `[1, 2, 3]` MW have an arithmetic mean of 2 MW. With durations `[1, 2, 1]` hours, their time-weighted mean is also 2 MW; with `[1, 1, 2]` hours, it is 2.25 MW. Records that describe cumulative meter energy require differencing and checking resets; they must not be treated as instantaneous powers and summed again.

## Capacity, availability, variability and controllability

**Installed or rated capacity** is an equipment rating at a defined boundary, usually MW for electricity. **Capacity factor** is $CF=E/(P_rT)$ over a specified interval. **Availability** describes whether equipment can operate under a defined time-based or energy-based convention. A plant can be available but lack sunlight, wind or water; conversely, a good resource does not prevent a mechanical outage. None of these quantities is interchangeable with conversion efficiency. [@manwell2009], §§2.5 and 10.3; [@ifc2015], Chapter 7; [@grant2011], §2.5.

**Variability** is change in the resource or output. **Predictability** concerns how well future changes can be forecast. **Dispatchability** concerns the ability to schedule or control output within constraints. Reservoir storage can give some hydropower flexibility; run-of-river output is more closely tied to current flow. A geothermal plant may provide sustained output but still face reservoir and plant limits. Solar and wind can be curtailed, but curtailment capability does not create additional resource when output is low. [@manwell2009], §§2.5 and 10.3; [@ifc2015], Chapter 7; [@grant2011], §2.5.

**Firm capacity** and reliability require analysis of output coinciding with system needs, not just annual energy totals. Two resources may complement one another, but that claim must be checked against aligned observations or a justified model. A large annual renewable energy total can coexist with hourly shortages. [@manwell2009], §§2.5 and 10.3; [@ifc2015], Chapter 7; [@grant2011], §2.5.

## Storage and hybrid portfolios

Storage has both a **power limit** and an **energy limit**. A 2 MW, 4 MWh usable battery can nominally discharge at its power limit for about two hours before accounting for the relevant output-efficiency convention and reserve restrictions. Power rating alone cannot specify storage duration. Initial state of charge and final remaining energy also affect a finite-period assessment.

With $S$ representing internally stored usable energy in MWh and powers measured at the external electrical bus,

$$S_{t+1}=S_t+\eta_cP_c\Delta t-\frac{P_d\Delta t}{\eta_d}.$$

Here $P_c$ is power taken from the bus for charging, $P_d$ is power returned to the bus, and $\eta_c$, $\eta_d$ are charging/discharging efficiencies. Use hours for $\Delta t$. Enforce storage and power limits and identify any self-discharge or reserve assumptions. Multiplying both charging and discharging power by efficiency in this convention would understate the storage depleted during discharge.

*Sources and assumptions:* Storage losses: [@wade2003], Chapter 5; [@manwell2009], §10.7. The step equation uses the simplified bus-side power and internal-energy boundaries stated above.

For a no-storage portfolio with aligned electrical supply $P_{RE}$ and demand $P_L$,

$$P_{served}=\min(P_{RE},P_L),$$
$$P_{curtailed}=P_{RE}-P_{served},\qquad
P_{shortfall}=P_L-P_{served}.$$

The two balances are $P_{served}+P_{curtailed}=P_{RE}$ and $P_{served}+P_{shortfall}=P_L$. These are accounting identities under the stated dispatch rule, not predictions about imports, prices or network stability. The final project uses this clear base case before considering design comparisons.

## Data definitions that determine whether a calculation is meaningful

Before combining datasets, establish their **grain**: one row might be a turbine-hour, a country-month, a well test, or a meter interval. Joining by row number rather than a real key can combine unrelated observations. Align timestamps, time zones and durations, check uniqueness, and investigate missing or duplicate records before calculating totals.

| Data question | Why it changes the calculation |
|---|---|
| Power, interval energy or cumulative energy? | Determines whether to integrate, sum or difference |
| Thermal, DC, gross AC or net AC? | Determines whether values can be compared or added |
| Instantaneous or interval-average observation? | Determines the appropriate integration approximation |
| Nameplate capacity or measured output? | Determines the capacity-factor denominator or energy numerator |
| Missing, zero, estimated or provisional? | Determines coverage and confidence in totals |
| Local time or UTC? | Determines alignment and daylight-saving interval lengths |
| Component or aggregate category? | Determines whether summing would double-count energy |

*Source for these definitions/calculations:* Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html). [Eurostat energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology).

Keep missing values distinct from measured zero. In pandas, the default sum of an empty or all-missing numeric series is zero; use `sum(min_count=1)` when an unavailable total should remain missing, and report the valid count or coverage separately. Report coverage together with a partial sum instead of presenting a partial dataset as a complete-period result. The course's Eurostat examples illustrate this issue: country coverage and the treatment of broad hydro/storage categories affect interpretation even when the SQL and pandas arithmetic agrees.

## Comparing projects without confusing metrics

An energy comparison should hold time period and electrical boundary consistent. A financial comparison also needs currency, price year, discounting convention, lifetime, cost scope and energy-delivery boundary. A simple levelized cost of energy discounts costs and delivered energy to a common basis [@ifc2015], §13.2.5 and Chapter 14:

$$LCOE=\frac{I_0+\sum_{y=1}^{N}C_y/(1+r)^y}
{\sum_{y=1}^{N}E_y/(1+r)^y},$$

where $I_0$ is initial investment, $C_y$ is included year-$y$ cost, $E_y$ is delivered energy, and $r$ is a discount rate consistent with the cost basis. The equation gives currency per energy unit. A real rate should be paired with constant-price cash flows; nominal flows require a consistent nominal framework. State assumptions about replacements, financing, taxes and terminal costs. LCOE is not automatically a tariff, profit, reliability measure or complete electricity-system cost [@ifc2015; @irena2026].

*Equation basis:* Discounted cost and energy: [@ifc2015], §13.2.5 and Chapter 14.

Environmental comparison likewise requires a life-cycle and site-specific perspective. For hydropower, see [@ifc2015], Chapter 12; for photovoltaic life-cycle stages, see the [IEA PVPS life-cycle assessment guidelines, §3.2](https://iea-pvps.org/wp-content/uploads/2020/07/IEA_Task12_LCA_Guidelines.pdf). Construction, land and water use, materials, wildlife and end-of-life processes matter alongside operation, with effects depending on technology and location. Renewable energy is not synonymous with zero impact. The reference catalogue separates technical principles from dated market evidence so historical book examples are not mistaken for current prices or generation shares.

## Worked comparison before coding

Suppose a synthetic four-hour hybrid system has renewable outputs of [1, 5, 2, 4] MW, each modeled as constant within its one-hour interval, and constant demand of 3 MW, with no storage or trading. Hourly means alone would not determine exact simultaneous matching when output varies within an hour. For example, with constant 1 MW demand, a constant 1 MW supply serves 1 MWh in one hour. A supply of 0 MW for the first half-hour and 2 MW for the second has the same 1 MW hourly mean but serves only 0.5 MWh without storage.

1. Available renewable energy is $(1+5+2+4)\times1=12$ MWh.
2. Demand energy is $4\times3=12$ MWh.
3. Directly served hourly power is [1, 3, 2, 3] MW, so served energy is **9 MWh**.
4. Curtailment is [0, 2, 0, 1] MW: **3 MWh** over the four intervals.
5. Shortfall is [2, 0, 1, 0] MW: also **3 MWh**.
6. Renewable demand coverage is $9/12=75\%$, even though total renewable energy equals total demand energy.

*Source for these definitions/calculations:* Original worked example applying the stated no-storage identities; Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

This example explains why the final project joins the separate source and demand files by timestamp before summing electricity. It also motivates a useful habit: write the balance and expected units first, then use Python to calculate and check it.

## How the following chapters build on these definitions

The general exercises progress from unit conversion to data quality, balancing, storage and uncertainty. The source-specific chapters then explain the physical resource, plant components, equations and limits in detail. Use the Applied Python Workbook for practice. Read each technology's worked hand calculation before running its Python functions. The supplied books provide the primary conceptual basis; additional textbook reading is identified separately in the reference catalogue.

<!-- expanded-theory:general:end -->

[Download the theory notebook](introRenewableEnergy.ipynb).

(intro-practice)=
## Practice

[Open the general exercises](../section7/workbook/general.md) in the Applied Python Workbook.

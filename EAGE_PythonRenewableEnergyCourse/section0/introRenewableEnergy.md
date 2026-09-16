# Introduction to Renewable Energy

## Energy, power and units

Energy measures an amount transferred or stored; power measures its rate. Instantaneous power is $P=dE/dt$. For constant or interval-average power, $E=P\Delta t$; for a time series, $E=\sum_iP_i\Delta t_i$.

| Quantity | Unit | Conversion |
|---|---|---|
| Power | W, kW, MW | 1 MW = 1,000 kW = 1,000,000 W |
| Energy | J, kWh, MWh | 1 kWh = 3.6 million J; 1 MWh = 1,000 kWh |
| Irradiance | W/m² | Power per unit area |
| Irradiation | kWh/m² | Solar energy per unit area over a stated interval |

A closed system can exchange energy with its surroundings; energy is conserved for the combined isolated system. Useful mechanical forms are $E_k=mv^2/2$ and $E_p=mgh$. Mechanical energy is conserved when only conservative forces do work. For DC electricity, $P=VI$; sinusoidal AC real power also depends on power factor. See Wade, Chapter 2 [@wade2003].

## Renewable technologies

Renewable resources are replenished by natural processes, but sustainable production has local limits. Renewable does not mean impact-free or unlimited extraction.

| Technology | First model | What the simple equation leaves out |
|---|---|---|
| Hydro | $P=\eta\rho gQH_{net}$ | Seasonal flow, ecological releases, reservoir operation, head losses |
| Wind | $P_{rotor}=C_p\rho Av^3/2$ | Generator rating, cut-out, wakes, curtailment and losses |
| Solar PV | $P_{DC}=GA\eta$ | Cell temperature, orientation, shading and inverter losses |
| Geothermal | $P_{th}=\dot m c_p(T_p-T_r)$ | Phase change, reservoir response, pumping and electrical conversion |

Hydropower has no fuel combustion at the turbine, but construction, ecosystems and reservoir greenhouse-gas emissions matter. Geothermal heat extraction can exceed local replenishment. Biomass sustainability depends on feedstock, land use and regrowth. Nuclear is low-emissions electricity but is not classified as renewable in this course [@ifc2015; @grant2011; @manwell2009; @foster2010].

## Storage and variability

Pumped hydro moves water uphill using electricity and recovers part of that energy later. Batteries store energy electrochemically. Some concentrating solar power systems store heat, for example in molten salt, before generating electricity; PV panels do not themselves store electricity. Storage shifts energy in time and incurs losses. Report both power capacity (MW) and usable energy capacity (MWh), plus efficiency and initial storage state [@jrc2025hydro; @wade2003].

Capacity factor is $CF=E/(P_{rated}T)$ over the same period and output boundary. It differs from conversion efficiency and availability. Use actual calendar hours: 8,760 in a non-leap year and 8,784 in a leap year.

## Energy context: dated evidence

The IEA's *Global Energy Review 2026* reports estimates for **2025**. Low-emissions sources, including renewables **and nuclear**, supplied 43% of global electricity in that report. This is an electricity-generation share, not a renewable share or a share of total energy supply [@iea2026].

For cost context consult IRENA's July 2026 *Renewable Power Generation Costs in 2025* [@irena2026]. Distinguish publication year from observation year. Scenario results are conditional projections, not observed facts; identify the edition and assumptions.

The Eurostat notebooks intentionally use bundled **August 2024 exports** of monthly net electricity generation in **GWh**. They are historical teaching snapshots. Legacy charts in the data-source lesson should not be used for current totals or trends. Primary energy, final energy, generation and consumption have different accounting boundaries [@eurostat_energy_database; @owid_energy_mix].

Start Python practice with the [graded workbook](../section7/renewableExercises.md). The short calculations below are preliminary unit checks.

## Exercises

### Exercise 1 (Simple)

**Reference:** Wade, Chapter 2 [@wade2003]; wind conversion [@manwell2009].

**Problem**: A 100-Watt light bulb is left on for 2 hours. Calculate the total energy consumed by the light bulb during this period in Joules.

```{admonition} Solution
:class: tip, dropdown

1. **Identify the given values**:
   - Power of the light bulb, $P = 100 \, \text{W}$
   - Time the light bulb is on, $t = 2 \, \text{hours}$

2. **Convert time to seconds**:
   - $t = 2 \, \text{hours} \times 3600 \, \text{seconds/hour} = 7200 \, \text{seconds}$

3. **Use the energy formula**:
   - The formula to calculate energy is $E = P \times t$
   - Substitute the given values: 
   $E = 100 \, \text{W} \times 7200 \, \text{seconds} = 720,000 \, \text{Joules}$

4. **Conclusion**:
   - The total energy consumed by the light bulb is $720,000 \, \text{Joules}$.
```

### Exercise 2 (Mid-complexity)

**Reference:** Wade, Chapter 2 [@wade2003]; wind conversion [@manwell2009].

**Problem**: A wind turbine receives 500 kW of available wind power and operates for 5 hours at constant available wind power. If the efficiency of the turbine is 40\%, calculate the total electrical energy generated by the turbine in kilowatt-hours (kWh) and also determine the amount of energy not converted to electricity.

```{admonition} Solution
:class: tip, dropdown

1. **Identify the given values**:
   - Available wind power at the turbine, $P = 500 \, \text{kW}$
   - Time of operation, $t = 5 \, \text{hours}$
   - Efficiency of the turbine, $\eta = 40\% = 0.40$

2. **Calculate the total available energy from the wind** (without considering efficiency):
   - The formula to calculate energy is $E = P \times t$
   - Substitute the given values:
   $E_{total} = 500 \, \text{kW} \times 5 \, \text{hours} = 2500 \, \text{kWh}$

3. **Calculate the actual electrical energy generated** (considering efficiency):
   - Actual energy generated $E_{generated} = E_{total} \times \eta$
   - Substitute the given values:
   $E_{generated} = 2500 \, \text{kWh} \times 0.40 = 1000 \, \text{kWh}$

4. **Calculate the energy not converted to electricity**:
   - Energy not converted $E_{lost} = E_{total} - E_{generated}$
   - Substitute the calculated values:
   $E_{lost} = 2500 \, \text{kWh} - 1000 \, \text{kWh} = 1500 \, \text{kWh}$

5. **Conclusion**:
   - The total electrical energy generated by the turbine is $1000 \, \text{kWh}$.
   - The amount of energy not converted to electricity is $1500 \, \text{kWh}$.
```

### Exercise 3 (Easy — two-step calculation)

**Reference:** Wade, Chapter 2 [@wade2003]; wind conversion [@manwell2009].

**Problem**: A solar panel with an area of 10 square meters is exposed to sunlight with an intensity of 1000 $W/m^{2}$. If the efficiency of the solar panel is 20\%, calculate the electrical power output of the panel. Additionally, estimate the total energy produced by this panel in one day assuming **8 equivalent full-sun hours**. Also, discuss the implications of varying sunlight intensity throughout the day on the total energy produced.

```{admonition} Solution
:class: tip, dropdown

1. **Identify the given values**:
   - Area of the solar panel, $A = 10 \, \text{m}^2$
   - Sunlight intensity, $I = 1000 \, \text{W/m}^2$
   - Efficiency of the solar panel, $\eta = 20\% = 0.20$
   - Equivalent full-sun time per day, $t = 8 \, \text{hours}$

2. **Calculate the total power received by the solar panel**:
   - The formula to calculate power received by the panel is $P_{received} = A \times I$
   - Substitute the given values:
   $P_{received} = 10 \, \text{m}^2 \times 1000 \, \text{W/m}^2 = 10,000 \, \text{W} = 10 \, \text{kW}$

3. **Calculate the electrical power output**:
   - The formula to calculate electrical power output is $P_{output} = P_{received} \times \eta$
   - Substitute the values:
   $P_{output} = 10,000 \, \text{W} \times 0.20 = 2000 \, \text{W} = 2 \, \text{kW}$

4. **Estimate the total energy produced in one day**:
   - The total energy produced is calculated using $E = P_{output} \times t$
   - Using equivalent full-sun hours, directly calculate in kWh:
   $E_{total} = 2 \, \text{kW} \times 8 \, \text{hours} = 16 \, \text{kWh}$

5. **Discussion on varying sunlight intensity**:
   - **Real-world scenario**: In reality, sunlight intensity is not constant throughout the day. It peaks during midday and is lower during morning and evening hours.
   - **Implications**: If sunlight intensity varies, the actual total energy produced can be above or below this estimate depending on location, season, cloud cover, and panel orientation. The solar panel's energy production can be modeled more accurately by integrating the power output over time, considering the varying intensity throughout the day.

6. **Conclusion**:
   - The electrical power output of the solar panel under constant full sunlight is $2 \, \text{kW}$.
   - The estimated total energy produced in one day is $16 \, \text{kWh}$, assuming 8 equivalent full-sun hours.
   - For fixed irradiation, varying irradiance alone does not change energy in a constant-efficiency linear model. Temperature, clipping and other losses can change delivered energy.
```


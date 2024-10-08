# Introduction to Renewable Energies

## Energy

Energy can be described using various equations and principles that highlight its different forms and conservation laws. Key energy-related equations include:

- **Einstein's Mass-Energy Equivalence**: The famous equation $E = mc^2$, proposed by Albert Einstein in the theory of special relativity, establishes a relationship between mass and energy. It suggests that a small amount of mass can be converted into a large amount of energy.

- **Energy Conservation Principle**: The principle of conservation of energy is expressed by the equation $\Delta E_{system} + \Delta E_{surroundings} = 0$. This equation states that the total energy in a closed system remains constant, although it can be transferred between the system and its surroundings.

- **Kinetic Energy Equation**: Kinetic energy ($E_k$) can be calculated using the formula $E_k = \frac{1}{2} mv^2$, where $m$ is the mass of an object and $v$ is its velocity. This equation shows that the energy of motion is directly proportional to the mass and the square of the velocity of the object.

- **Total Mechanical Energy**: The total mechanical energy in a system, which is the sum of kinetic energy ($K$) and potential energy ($U$), is conserved in the absence of external forces. This is expressed as $K_0 + U_0 = K + U$ or $\Delta K + \Delta U = 0$, where the subscript $0$ denotes initial values.

Common symbols for energy include $E$, and it is typically measured in joules (J) in the SI system. Other units include kilowatt-hours (kW⋅h), British Thermal Units (BTU), calories, electron volts (eV), ergs, and foot-pounds. In SI base units, energy is represented as $J = \text{kg} \cdot \text{m}^2 \cdot \text{s}^{-2}$.

## Power: The Rate of Energy Use

Power is the rate at which energy is consumed or work is performed. It quantifies how quickly energy is used or transferred. The relationship between energy and power can be defined as:

- **Power Equation**: Power ($P$) is calculated by dividing energy by time, or $P = \frac{E}{t}$. This equation shows that one watt (W) is equivalent to one joule of energy expended per second.

$$ P = \frac{W}{t} $$

Where:
- $P$ is power,
- $W$ is work done,
- $t$ is time.

- **Electrical Power**: In electrical systems, the power transferred between energy stores is given by $P = I \times V$, where $I$ represents the current and $V$ represents the potential difference (voltage).

- **Standard Metric Unit of Power**: The standard metric unit of power is the Watt. As implied by the equation for power, a unit of power is equivalent to a unit of work divided by a unit of time. Thus, a Watt is equivalent to a Joule per second ($\text{W} = \frac{J}{s}$). For historical reasons, the horsepower is occasionally used to describe the power delivered by a machine. One horsepower is equivalent to approximately 750 Watts.

Common symbols for power include $P$, with the SI unit being the watt (W). In SI base units, power is expressed as $W = \text{kg} \cdot \text{m}^2 \cdot \text{s}^{-3}$.

## Energy Markets

### World Energy Consumption
What is global energy consumption? Global energy consumption is the total amount of energy used on the planet to meet human needs. This energy mainly comes from non-renewable or fossil sources, such as oil, coal, or natural gas. These sources have a high environmental and social impact, contributing to climate change, air and water pollution, and geopolitical conflicts.

```{image} ../images/section4/worldDatainPlot_EnergyBySource_plot_filled.png
:alt: WorldEnergyConsumption
:class: bg-primary mb-1
:width: 800px
:align: center
```

The image represents the global energy consumption measured in terawatt-hours (TWh) over time, displayed on the y-axis, while the x-axis shows the years from 1800 to the present day. The graph uses a stacked area plot to differentiate between various sources of energy, each represented by a distinct color. These sources include traditional biomass, coal, oil, gas, nuclear, hydropower, wind, solar, biofuels, and other renewables.

In the early part of the timeline, from the 1800s to the early 1900s, traditional biomass was the dominant source of energy, reflecting an era before the industrial revolution when humans primarily relied on wood and other organic materials for heating and cooking. However, as industrialization progressed, coal began to rise sharply in usage, overtaking traditional biomass and becoming the primary source of energy well into the 20th century.

The growth of coal use is followed by a significant increase in oil and gas consumption starting around the mid-1900s, coinciding with the expansion of modern industrial economies and the post-World War II era of rapid economic growth. Oil and gas, along with coal, dominate the global energy landscape throughout much of the 20th century, contributing to a steady rise in total energy consumption.

However, the later part of the 20th century and into the 21st century shows a noticeable diversification in energy sources. Hydropower, which had seen a gradual increase over the decades, becomes more prominent. Around the same time, nuclear energy also rises sharply, particularly from the 1970s onward, reflecting the global adoption of nuclear technology as a major power source.

What stands out in more recent years is the accelerated growth of renewable energy sources such as wind, solar, and biofuels. Wind energy, in particular, shows a rapid increase starting in the 2000s, closely followed by solar energy. These two renewable sources begin to play an increasingly important role in global energy consumption, signifying a shift toward more sustainable energy systems. While fossil fuels like coal and oil still maintain a large presence, their growth begins to plateau, and in the case of coal, even shows signs of a slight decline.

Gas remains a strong player, showing consistent growth even as the world begins to shift towards renewable energy, likely due to its role as a relatively cleaner fossil fuel compared to coal and oil. Meanwhile, the growth of biofuels and other renewables continues, albeit at a slower pace compared to wind and solar.

In summary, the graph illustrates the historical dominance of fossil fuels such as coal, oil, and gas, while also highlighting the more recent rise of renewable energy sources. Wind and solar, in particular, have seen a notable increase in the 21st century, indicating a global energy transition towards cleaner and more sustainable power sources. The graph emphasizes both the persistence of fossil fuels and the increasing importance of renewable energy in the global energy mix.

#### Reference:

Hannah Ritchie and Pablo Rosado (2020) - “Electricity Mix” Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/electricity-mix' [Online Resource]


```{image} ../images/section4/euroStat_totalDatasetYear_plot_types.png
:alt: euroStat_totalDatasetYear_plot_types
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ../images/section4/euroStat_totalDatasetYear_plot_renewables.png
:alt: euroStat_totalDatasetYear_plot_renewables
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ../images/section4/euroStat_totalDatasetYear_plot_nonRenewables.png
:alt: euroStat_totalDatasetYear_plot_nonRenewables
:class: bg-primary mb-1
:width: 800px
:align: center
```

#### Reference:

##### Data:
* eurostat (https://ec.europa.eu/eurostat/web/energy/database) 
* U.S. Energy Information Administration (EIA) (https://www.eia.gov/)
* Our World in Data (https://ourworldindata.org/)

##### Articles:
* Shedding light on energy in the EU — A guided tour of energy statistics — 2023 edition
* Sustainable development in the European Union — Monitoring report on progress towards the SDGs in an EU context — 2023 edition

### World Energy Renewable Adoption
In the Stated Policies Scenario (STEPS), which is consistent with enacted energy policies and a pragmatic view of proposed policies, worldwide energy supply is expected to grow by approximately 21% between 2019 and 2050. While there is a decline in supply for coal in this scenario, between 2019 and 2050, oil and natural gas are expected to grow by 6% and 24%, respectively, and account for approximately 50% of the energy mix in 2050.

## Renewable Energies
Renewable energies are energy sources based on the use of the sun, wind, water, or biomass. They do not use fossil fuels, unlike conventional energies, but resources that can be renewed indefinitely.

### Hydroelectric Energy
Hydroelectric energy is generated by the movement of water through turbines. This energy is renewable and produces no direct emissions.

### Eolic Energy
Eolic energy, or wind energy, harnesses the power of wind to generate electricity using wind turbines.

### Solar Energy
Solar energy captures sunlight to generate electricity or heat. It includes technologies like photovoltaic solar panels and concentrated solar power systems.

### Geothermal Energy
Geothermal energy utilizes heat from within the Earth to generate electricity or provide direct heating.

$$Q = mc\Delta T$$

Where:
- $Q$ is the heat added,
- $m$ is the mass,
- $c$ is the specific heat capacity,
- $\Delta T$ is the change in temperature.

## Energy Storage
Energy storage is a process of saving energy in a system or device for later use. Different types of energy storage systems exist, including lithium batteries, pumped hydro storage, compressed air, or hydrogen storage.

### Pumped Hydro
Flowing water is moved downhill through turbines, which rotate to generate power. The water can then be pumped back uphill to be reused when needed.

### Solar Thermal
Solar thermal systems use the sun to generate and store electricity by using mirrors, salt, and steam. The mirrors focus sunlight into a tower that heats salt, which creates steam to run turbines that generate electricity.

### Batteries
When connected to renewable sources like solar and wind, batteries store excess energy. This energy can then be used later when needed, such as at night when the sun is no longer shining.

## Current Energy Context
In the Stated Policies Scenario (STEPS), consistent with enacted energy policies and a pragmatic view of proposed policies, worldwide energy supply is expected to grow by approximately 21% between 2019 and 2050. Although coal supply is expected to decline during this period, oil and natural gas are anticipated to grow by 6% and 24%, respectively, accounting for approximately 50% of the energy mix by 2050.

## Exercises

### Exercise 1 (Simple)

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

**Problem**: A wind turbine generates 500 kW of power and operates for 5 hours at full capacity. If the efficiency of the turbine is 40\%, calculate the total electrical energy generated by the turbine in kilowatt-hours (kWh) and also determine the amount of energy that was lost due to inefficiency.

```{admonition} Solution
:class: tip, dropdown

1. **Identify the given values**:
   - Power generated by the turbine, $P = 500 \, \text{kW}$
   - Time of operation, $t = 5 \, \text{hours}$
   - Efficiency of the turbine, $\eta = 40\% = 0.40$

2. **Calculate the total energy produced** (without considering efficiency):
   - The formula to calculate energy is $E = P \times t$
   - Substitute the given values:
   $E_{total} = 500 \, \text{kW} \times 5 \, \text{hours} = 2500 \, \text{kWh}$

3. **Calculate the actual electrical energy generated** (considering efficiency):
   - Actual energy generated $E_{generated} = E_{total} \times \eta$
   - Substitute the given values:
   $E_{generated} = 2500 \, \text{kWh} \times 0.40 = 1000 \, \text{kWh}$

4. **Calculate the energy lost due to inefficiency**:
   - Energy lost $E_{lost} = E_{total} - E_{generated}$
   - Substitute the calculated values:
   $E_{lost} = 2500 \, \text{kWh} - 1000 \, \text{kWh} = 1500 \, \text{kWh}$

5. **Conclusion**:
   - The total electrical energy generated by the turbine is $1000 \, \text{kWh}$.
   - The amount of energy lost due to inefficiency is $1500 \, \text{kWh}$.
```

### Exercise 3 (Hard-complexity)

**Problem**: A solar panel with an area of 10 square meters is exposed to sunlight with an intensity of 1000 $W/m^{2}$. If the efficiency of the solar panel is 20\%, calculate the electrical power output of the panel. Additionally, estimate the total energy produced by this panel in one day (24 hours) assuming full sunlight during this period. Also, discuss the implications of varying sunlight intensity throughout the day on the total energy produced.

```{admonition} Solution
:class: tip, dropdown

1. **Identify the given values**:
   - Area of the solar panel, $A = 10 \, \text{m}^2$
   - Sunlight intensity, $I = 1000 \, \text{W/m}^2$
   - Efficiency of the solar panel, $\eta = 20\% = 0.20$
   - Time of operation, $t = 24 \, \text{hours}$

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
   - Convert 24 hours to seconds for consistency, or directly calculate in kWh:
   $E_{total} = 2 \, \text{kW} \times 24 \, \text{hours} = 48 \, \text{kWh}$

5. **Discussion on varying sunlight intensity**:
   - **Real-world scenario**: In reality, sunlight intensity is not constant throughout the day. It peaks during midday and is lower during morning and evening hours.
   - **Implications**: If sunlight intensity varies, the actual total energy produced will be less than the estimated 48 kWh. The solar panel's energy production can be modeled more accurately by integrating the power output over time, considering the varying intensity throughout the day.

6. **Conclusion**:
   - The electrical power output of the solar panel under constant full sunlight is $2 \, \text{kW}$.
   - The estimated total energy produced in one day is $48 \, \text{kWh}$, assuming constant sunlight intensity.
   - Varying sunlight intensity will reduce the actual total energy produced compared to this ideal scenario.
```

## Useful Resources:

- [Electricity Maps](https://app.electricitymaps.com/map)

- [Energy Web shares open-source tech toolkit for simplifying 24/7 clean energy procurement](https://medium.com/energy-web-insights/energy-web-shares-open-source-tech-toolkit-for-simplifying-24-7-clean-energy-procurement-41ac2e8225eb)
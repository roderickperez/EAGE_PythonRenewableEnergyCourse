# Project 4: Geothermal Energy

Design a geothermal doublet: inject cooled water into a reservoir, produce warmer water at another well, and use a simplified binary plant to generate electricity. Explore the trade-off between early generation and thermal breakthrough. This synthetic teaching model is not a reservoir forecast or engineering design.

## How to use the explorer

1. The plant is at the center of a 2 km × 2 km site. Select **Injection well** or **Production well**, then click a grid cell to move that well. The wells must occupy different cells. Drag to rotate and scroll to zoom. Focus the grid and use the arrow keys and Enter or Space to select a cell without a mouse.
2. Change reservoir depth, circulation rate or injection temperature. Depth determines initial temperature using a teaching gradient of 72.5 °C/km and a 15 °C surface temperature, rounded to 10 °C. Each change or well move **restarts** the simulation and resets upgrades. The blue well injects and the red well produces at the same mass rate; their vertical connections reach the modeled reservoir layer.
3. Watch the colored reservoir layer cool around the wells as the simulation advances. The stacked, translucent surrounding geological layers are schematic. Compare production temperature, thermal and net electrical power, cumulative energy and revenue. The plant starts at 5 MW net capacity; when available upgrade budget reaches \$1 million, purchase 1 MW more, up to 10 MW. Upgrades preserve simulation history. Pause with the square button; use the circular-arrow icon in the upper-left of the grid to reset. One year takes 30 seconds of visible running time. The simulation pauses in a hidden browser tab.
4. Use the download icon in the lower-right of the grid to save results as CSV. The temperature curve shows production in red and injection in blue. The depth plot shows an illustrative initial formation gradient in gray and the modeled producer reservoir temperature as a red point. The depth–time image shows the modeled producer layer at the bottom; the surrounding formation is a fixed illustrative gradient, not a calculated wellbore profile.

:::{iframe} ../../geothermal-lab/project.html
:width: 100%
:height: 1650px
:title: Interactive geothermal field designer
:::

## Model and assumptions

The reservoir is a 10 × 10 grid of 200 m cells in a 50 m thick layer, initially at 2,000 m depth. Depth determines initial temperature; the displayed layer thickness and well geometry are schematic, and depth does not change flow resistance. A balanced injector/producer pair creates flow through an equal-conductance network with closed outer boundaries. The model transports heat between cells using upwind flow, a combined rock–fluid volumetric heat capacity of 2.5 MJ/(m³ °C), constant water heat capacity of 4.18 kJ/(kg °C), and a stable explicit time step. It does not calculate pressure in physical units, permeability, wellbore heat losses, thermal conduction, fractures, natural recharge, phase changes or variable fluid properties. Neighboring rocks and the well paths are schematic. Neither an illustrative gradient nor the depth plot represents measured temperature data.

Thermal power while operating is $\dot m c_p(T_{production}-T_{injection})$. The plant converts 12% of thermal power to gross electricity; 10% of gross electricity is consumed by auxiliaries. Net output is limited by installed nameplate capacity. Availability is 90% and reduces circulation and integrated generation once. Net power shown is the operating value; energy and revenue include availability. Revenue is net MWh times the fixed teaching price of \$75/MWh. Project balance subtracts a fixed \$4 million plant, \$1 million for each of two wells, and \$1 million per 1 MW capacity upgrade. Available upgrade budget is the positive part of this balance; capacity cannot exceed 10 MW. These teaching values exclude operating costs, financing, taxes and discounting. The production temperature and power at the displayed instant are sampled at the producer; MWh and sales are integrated over past steps.

Temperature colors represent simulated reservoir cell temperatures between the selected injection and initial temperatures. The upper layers, plant, surface pipes and animated flow dots show the physical idea, not calculated wellbore temperature or fluid velocities. Wells are enlarged for visibility. The color field, production curve and heatmap use the same simulated state. Long-term heat losses, recharge and pressure support are not modeled, so do not use these results as a field assessment.

## Check the electrical calculation in Python

```{code-cell} python
mass_flow = 50.0             # kg/s
cp = 4180.0                  # J/(kg °C)
production, injection = 160.0, 70.0
efficiency, auxiliary = 0.12, 0.10
thermal_mw = mass_flow * cp * (production - injection) / 1e6
net_mw = thermal_mw * efficiency * (1 - auxiliary)
energy_mwh = net_mw * 24 * 0.90  # one day at 90% availability
print(f'Thermal power: {thermal_mw:.3f} MW')
print(f'Operating net power: {net_mw:.3f} MW')
print(f'One-day energy: {energy_mwh:.3f} MWh')
```

Expected values are **18.810 MW thermal**, **2.031 MW net while operating**, and **43.880 MWh** over a day at 90% availability, before the 5 MW plant cap. Availability is applied once. For the full grid calculations and plots, use the accompanying [Python project notebook](geothermal-field-designer.ipynb).

## Your investigation

1. Run the default wells for 10 simulated years and record production temperature, generated MWh and revenue.
2. Move the producer one cell from the injector, then repeat with wider spacing. Compare breakthrough and energy at the same simulated time. What assumptions make the pattern different from a real fractured reservoir?
3. Compare 25 and 75 kg/s at the same layout. Explain why greater initial output does not guarantee the greatest long-term output.
4. Once the budget allows, purchase a capacity upgrade. Does it increase current output? Explain how resource power and the capacity limit interact.
5. Use the downloaded CSV and Python notebook to reproduce the power and revenue calculations. Distinguish stored heat, recoverable heat, net electrical energy and revenue.

## Sources and extensions

[Kiani's modeling description](https://www.linkedin.com/pulse/geothermal-simulates-models-heat-extraction-producing-mohammad-kiani-9tdgc/) discusses a commercial single-well closed-loop case. This explorer models a different, two-well teaching scenario. [Franco and Donatini (2017)](https://doi.org/10.1088/1742-6596/796/1/012025) discuss heat-in-place assessment, uncertainty and recharge. The [Geothermal Cookbook](https://github.com/ICWallis/geothermal-cookbook) offers well-path, water-chemistry and historical installed-capacity examples; [Well Data Bonanza](https://github.com/ThomasMGeo/T21-well-data-bonanza) explores geothermal production and well data. These data are not inputs to the synthetic simulation.

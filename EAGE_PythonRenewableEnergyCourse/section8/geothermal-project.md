# Project 4: Geothermal Field Designer

Design a geothermal doublet: inject cooled water into a reservoir, produce warmer water at another well, and use a simplified binary plant to generate electricity. Explore the trade-off between early generation and thermal breakthrough. This synthetic teaching model is not a reservoir forecast or engineering design.

## How to use the explorer

1. The plant is at the center of a 2 km × 2 km site. Select **Injection well** or **Production well**, then click a grid cell to move that well. The wells must occupy different cells. Drag to rotate and scroll to zoom. Focus the grid and use the arrow keys and Enter or Space to select a cell without a mouse.
2. Change reservoir depth, circulation rate, injection and initial reservoir temperatures, plant capacity, or electricity price. Changing depth sets the initial temperature from a teaching gradient of 72.5 °C/km and a 15 °C surface temperature, rounded to 10 °C. You may then adjust the temperature independently. Each change or well move **restarts** the simulation. The blue well injects and the red well produces at the same mass rate; their vertical connections reach the modeled reservoir layer.
3. Watch the reservoir layer cool and compare production temperature, thermal and net electrical power, cumulative energy and revenue. Pause with the square button; restart with Reset simulation. One year takes 30 seconds of visible running time. The simulation pauses in a hidden browser tab.
4. Download results as CSV. It includes a record index, configured and active depth counts, total flow, conversion assumptions, plant cap, availability and price used for the sampled output; this local edition models one layer and one well pair, and a setting change starts a new history; the cumulative MWh and revenue are integrated inside the simulation, so sampled rows alone do not reproduce every internal time step. The temperature curve shows production in red and injection in blue. The depth plot shows an illustrative initial formation gradient in gray and the modeled producer reservoir temperature as a red point. The depth–time image places samples according to simulated time and shows the modeled producer layer at the bottom; the surrounding formation is a fixed illustrative gradient, not a calculated wellbore profile.

:::{iframe} ../../geothermal-lab/project.html
:width: 100%
:title: Interactive geothermal field designer
:::

## Model and assumptions

The reservoir is a 10 × 10 grid of 200 m cells in a 50 m thick layer, initially at 2,000 m depth. Depth changes the suggested initial temperature; the displayed layer thickness and well geometry are schematic, and depth does not change flow resistance. A balanced injector/producer pair creates flow through an equal-conductance network with closed outer boundaries. The model transports heat between cells using upwind flow, a combined rock–fluid volumetric heat capacity of 2.5 MJ/(m³ °C), constant water heat capacity of 4.18 kJ/(kg °C), and a stable explicit time step. It does not calculate pressure in physical units, permeability, wellbore heat losses, thermal conduction, fractures, natural recharge, phase changes or variable fluid properties. Neighboring rocks and the well paths are schematic. Neither an illustrative gradient nor the depth plot represents measured temperature data.

Thermal power while operating is $\dot m c_p(T_{production}-T_{injection})$, clipped to zero if the difference is negative. This single-pair model does not mix separate production streams; a hypothetical reheating requirement for a colder stream is outside its plant balance. The plant converts 12% of thermal power to gross electricity; 10% of gross electricity is consumed by auxiliaries. Net output is limited by the selected nameplate capacity. Availability is 90% and reduces circulation and integrated generation once. Net power shown is the operating value; energy and revenue include availability. Revenue is net MWh times the selected price. Project balance subtracts a fixed \$4 million plant and \$1 million for each of the two wells; these are teaching values and exclude operating costs, financing, taxes and discounting. The production temperature and power at the displayed instant are sampled at the producer; MWh and sales are integrated over past steps.

The reservoir color field represents simulated cell temperatures between the selected injection and initial temperatures. The upper layers, plant, surface pipes and animated flow dots show the physical idea, not calculated wellbore temperature or fluid velocities. Wells are enlarged for visibility. In the depth–time heatmap, the bottom row uses the simulated producer temperature; the background rows are a schematic formation-temperature gradient, not simulated temperatures at those depths. Heatmap columns span the midpoints between recorded simulated times, so a sample's color represents its surrounding interval rather than a continuously calculated depth profile. Long-term heat losses, recharge and pressure support are not modeled, so do not use these results as a field assessment.

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
4. Compare plant capacity and electricity price separately. Which changes the physical reservoir, and which changes only the financial calculation?
5. For the single-pair explorer, use the downloaded CSV and the notebook’s comparison function to check instantaneous power and cumulative revenue. Compare smaller numerical time steps at matching simulated times; sampled CSV rows cannot exactly replay the internal energy integration. Distinguish stored heat, recoverable heat, net electrical energy and revenue. The notebook does not replay the multi-depth explorer below.

## Multi-depth field designer

The single-layer doublet above is retained for comparison with the accompanying notebook. The multi-depth version below supports multiple injection and production wells. Choose a depth, drill an injector and producer at the same depth, then add wells or other depths using the grid action selector. Each depth is an independent 10 × 10 layer; a layer circulates only with both well types. The rate is the total balanced flow per active depth, shared equally among its injectors and producers. Adding or removing wells and changing operating settings affects subsequent steps without resetting prior temperatures, energy or revenue. A reset starts a new history.

The selected-depth plots show a schematic surrounding formation; heatmap cells use midpoint boundaries between recorded simulated times. At the same simulated time, configuration events retain their order in the CSV. The export includes record index, event type, active and balanced depth counts, total flow, operating settings, cumulative investment, project balance, counts of paid capacity upgrades and additional wells drilled, and JSON arrays of the producer streams and wells. The initial investment is $6 million; each additional well or capacity upgrade adds $1 million, and removing a well does not refund its cost. Balance is cumulative revenue minus cumulative investment. The notebook’s `compare_multidepth_csv` function checks sampled thermal power, net power and project balance from these exports. The sampled values permit independent reconstruction of instantaneous thermal and net power; integrated energy and revenue use internal time steps and cannot necessarily be reproduced exactly from samples alone.

For each producing stream, useful thermal output is its flow times water heat capacity times production-minus-injection temperature, clipped at zero before summing. This is a separately routed useful-heat boundary: a stream colder than injection contributes no useful output. Any external heat needed to raise that stream to injection temperature is excluded, so the result is not the complete heat balance of a fully mixed plant. The displayed production temperature is flow-weighted and alone cannot reproduce the clipped sum. The fixed teaching conversion efficiency is 12%, auxiliary use 10% of gross, availability 90%, and electricity price $75/MWh. The model does not specify a heat-rejection temperature or establish that this fixed efficiency remains thermodynamically achievable as production cools; injection temperature is not automatically the heat-sink temperature. Capacity starts at 5 MW; upgrades cost $1 million per MW, and each extra well costs $1 million. These are simplified assumptions, not a field assessment.

:::{iframe} ../../geothermal-lab/multifield.html
:width: 100%
:title: Interactive geothermal multi-depth field designer
:::

## Sources and extensions

[Kiani's modeling description](https://www.linkedin.com/pulse/geothermal-simulates-models-heat-extraction-producing-mohammad-kiani-9tdgc/) discusses a commercial single-well closed-loop case. The single-layer explorer above is a two-well teaching scenario; the multi-depth explorer is a separate synthetic multiwell field. [Franco and Donatini (2017)](https://doi.org/10.1088/1742-6596/796/1/012025) discuss heat-in-place assessment, uncertainty and recharge. The [Geothermal Cookbook](https://github.com/ICWallis/geothermal-cookbook) offers well-path, water-chemistry and historical installed-capacity examples; [Well Data Bonanza](https://github.com/ThomasMGeo/T21-well-data-bonanza) explores geothermal production and well data. These data are not inputs to the synthetic simulation.

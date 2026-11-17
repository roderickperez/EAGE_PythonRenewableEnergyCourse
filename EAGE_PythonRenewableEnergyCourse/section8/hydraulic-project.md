# Project 1: Hydroelectric Energy

Operate a reservoir plant over a synthetic year. Your challenge is to **maximize generated energy while maintaining the water reserve**. The reservoir level evolves from an initially given volume; future storage is unknown until inflow and your operating choices unfold.

## How to use the explorer

1. Set **X**, the requested turbine discharge in m³/s. The simulation starts immediately; use the stop/play icon at the upper right of the animation to pause or resume it.
2. Watch the same animated plant used in the [hydroelectric lesson](../section6/hydroelectricEnergy.md). The blue storage plot shows the changing reservoir; the dashed orange line is the minimum reserve. In the second plot blue is inflow, green is turbine flow, and gold is electrical power.
3. Adjust X throughout the year as inflow changes. The model prioritizes a 1 m³/s environmental release, then limits generation to water above the 5 hm³ reserve. If the reserve is reached, actual turbine flow may be lower than X. Surplus above 50 hm³ spills. Use the reset icon at the upper left of the animation to compare strategies against the same inflow.
4. Record total energy, spill, any environmental shortage and final storage. Compare strategies over the full year; water left in storage has potential value beyond the end of this simulation.

(hydro-reservoir-project)=
:::{iframe} ../../hydro-lab/project.html
:width: 100%
:height: 1200px
:title: Interactive hydroelectric reservoir project
:::

## Model and assumptions

The plant starts with 30 hm³ in a 50 hm³ reservoir. Each visible second represents one day, for 365 days; pausing or hiding the page stops simulation time. Seasonal inflow is synthetic and repeats on reset. Turbine efficiency declines automatically from 91% by 0.01 percentage points per simulated day, with a floor of 75%; there is no efficiency control. This is an illustrative deterioration scenario, not an equipment forecast. Generator and transformer efficiencies are 95% and 99%.

For each day, the model uses beginning-of-day storage to approximate net head as $H=60+40S/(50\times10^6)$ metres, where $S$ is m³. It adds that day's inflow, allocates the ecological release, releases up to X through the turbine without crossing the reserve, and spills any excess. Inflow is available within the same day. It computes $P=\rho gQH\eta_t\eta_g\eta_{tr}$ and integrates the day's average power for 24 hours. No evaporation, flow-dependent losses, forecast, price, or detailed turbine curve is included. The safety reserve and environmental target are teaching assumptions, not operational guidance. The ecological shortage reports any inability to supply the target; ecological release itself is not used to generate power in this project.

## Your investigation

1. Run the year with X = 8 m³/s and record when the turbine becomes water-limited. Record energy, spill and final storage.
2. Reset and try X = 3 m³/s. Compare the two results and explain how the reserve changes output.
3. Reset and adapt X to seasonal inflow. Can you reduce spill and increase energy while retaining useful year-end storage? Explain the trade-off.
4. Explain why maximizing this year's energy alone might favor emptying usable storage near year-end. Suggest a terminal-storage constraint for a fair comparison.
5. Describe how the declining efficiency and storage-dependent head affect generation, and what data would be needed for a real operating model.

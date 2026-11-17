# Project 2: Solar Energy

Design a small solar farm and investigate how location, season, panel orientation and weather affect its output. The explorer combines a location-based solar path with an illustrative weather scenario and simplified PV model. It is a learning activity, not a site assessment or forecast.

## How to use the explorer

1. Choose a highlighted location on the spinning globe or enter a latitude and longitude. Select **Explore this site** to move to the farm grid. The globe's coastlines are schematic; coordinates determine the calculations.
2. Start with **\$5,000**. Click a grid cell to buy a panel for **\$1,000**; click it again to remove it and recover **\$500**. Each panel represents 2 m² at 20% efficiency, or 400 W at standard test conditions. Drag the farm to rotate the view and scroll to zoom. To use the keyboard, focus the scene, move the gold selection with the arrow keys and press Enter or Space.
3. Adjust panel tilt, azimuth and shading/soiling. Azimuth is clockwise from north: north 0°, east 90°, south 180° and west 270°. All panels share the same settings. Set the date and solar time or allow the simulation to advance. The sun and dashed daylight path correspond to the chosen latitude and date.
4. Compare current AC power and the **Power over the solar day** graph. The vertical gold line marks the current time. The day's estimated energy assumes the current weather and layout persist throughout that day; cumulative energy is integrated as the simulation evolves.

:::{iframe} ../../solar-lab/project.html
:width: 100%
:height: 1200px
:title: Interactive solar farm layout explorer
:::

## Time, weather and budget

The simulation advances one solar day in 60 seconds of visible time. The calendar advances at midnight, and the location's latitude and day of year set solar declination and the sun's path. Time is **local solar time**: noon is when the sun crosses the local meridian. Longitude identifies the selected site but does not change solar-time sun angles; no time-zone or equation-of-time correction is applied. The path handles polar day and night. Pause the simulation to compare configurations under the same conditions. Changing location restarts the project and its budget.

Cloud cover and air temperature follow a simple latitude-dependent seasonal pattern, with a small random change each simulated day. They are synthetic, not historical measurements or a location-specific climate model. The game credits **\$0.10 per simulated kWh**; these fixed costs, salvage and revenue are teaching values, not actual project economics. Energy and revenue stop when paused or the page is hidden. Changing date, time or settings does not retroactively change accumulated energy.

## The model

The explorer uses the solar lesson's sun-position and plane-of-array calculations. It resolves direct, isotropic diffuse sky, and ground-reflected irradiance on the tilted panel. A simplified irradiance profile responds to synthetic cloud cover. Panel output includes a cell-temperature correction, shading/soiling, wiring loss and inverter efficiency, with the inverter limited to 400 W per panel. Panels do not shade one another; the spacing and panel dimensions in the 3D scene are illustrative.

The referenced articles discuss using open-source data and panel tilt, orientation, shading and irradiance to estimate generation, and using `pvlib` for solar calculations: [Estimating Solar Panel Output with Open-Source Data](https://towardsdatascience.com/estimating-solar-panel-output-with-open-source-data-bbca6ea1f523/) and [Solar panel power generation analysis](https://towardsdatascience.com/solar-panel-power-generation-analysis-7011cc078900/). This browser simulation uses the course's simplified JavaScript model; it does not fetch measured irradiance, perform rooftop mapping, or run `pvlib`. Its results require measured weather, equipment and site data for real-world validation.

## Your investigation

1. Install a panel in Vienna and compare south-facing and north-facing orientations at noon on the June and December solstices. Record current power and daily energy.
2. Compare Vienna with Sydney on the same date. Explain how latitude and season change the sun's path and the best-facing direction.
3. Pause at a fixed time and vary tilt and shading. Explain why instantaneous power and daily energy may favor different settings.
4. Try Tromsø in summer and winter. What happens to the sun's path and power during polar day and polar night?
5. Identify which inputs you would replace with measurements before using this model for an actual solar project.

# Project 3: Wind Energy

Design a small wind farm and explore how the position of a turbine changes the wind and power available to its neighbors. This project comes before the [combined renewable portfolio project](projectIntro.md). It uses an idealized, synthetic site to teach layout reasoning; it is not a design recommendation for an actual site.

## How to use the explorer

1. Choose **Manual** to set the wind direction **from** which air arrives and the wind speed, or choose **Seasonal** for a repeating synthetic annual variation. North is 0°, east 90°, south 180° and west 270°. The particles move **downwind**, slowing and shortening in wakes. Turbines turn according to their local modeled wind speed. Use the stop/play icon to control the animation.
2. Click a grid cell to place or remove a turbine. Drag the scene to rotate it and scroll to zoom. For keyboard control, focus the grid and use the arrow keys to select a location; press Enter or Space to place or remove a turbine.
3. Read farm power, no-wake power, wake loss and efficiency. The selected location's wind and turbine power appear below the scene. Gold indicates the selected location; the colorbar inside the scene shows modeled wind speed in the wakes. Uncolored terrain indicates undisturbed wind.
4. Compare the energy and wind-speed histories as you change the layout and conditions. Remove a turbine by clicking its location again.

:::{iframe} ../../wind-lab/project.html
:width: 100%
:height: 1100px
:title: Interactive isometric wind farm layout explorer
:::

The square site is 2,000 m by 2,000 m; its 10 × 10 grid points are 200 m apart. Easting and northing are illustrative local coordinates. The minimum-spacing slider constrains new placements; if a larger spacing would make an existing layout invalid, remove or move turbines first. Turbines in the illustration are enlarged relative to the farm so they can be seen.

## Manual and seasonal wind

Manual mode lets you set wind speed and direction with sliders. Seasonal mode replaces those sliders with a repeating synthetic scenario: speed varies smoothly between 7 and 13 m/s and direction oscillates around north. Each synthetic year takes 120 seconds of visible session time; the month label and conditions update approximately once per second. Switching back to Manual restores your previous manual settings. Spacing remains adjustable in both modes.

This cycle is an illustration, not measured local climate or a forecast. The month label is a scenario position, not today's date. Changing modes preserves the session history. Energy is still integrated over actual visible-session seconds, **not** over an accelerated year, so it is not an annual-energy estimate.

## Energy and wind history

The large energy value and the first graph show estimated cumulative energy for the current session in watt-hours (Wh), with one line for the farm total and one for each turbine location. The second graph shows the free-stream and local wind speeds over the same elapsed time. The model assumes each displayed power remains constant until the next change; it integrates power over real elapsed visible-session time. The display pauses while the page is hidden. Reopening the project starts a new history. Stopping the 3D animation does not stop this calculation. Graphs show the most recent 300 samples; totals cover the full session. A removed turbine keeps its accumulated contribution under its location number.

These are illustrative model estimates, not measurements, forecasts, or annual production. Power (kW or MW) is an instantaneous rate; energy (Wh) accumulates over time. The simplified wake model and illustrative wind conditions limit the results.

## The model

The explorer implements a simplified Jensen-style, top-hat wake model inspired by the physics in the [wind-farm layout repository](https://github.com/isaigm/wflo/blob/main/layout_physics.py). A wake is wider and weaker farther downstream. A turbine is considered inside another's wake when its center falls inside the wake radius. Multiple velocity deficits are combined by root-sum-square, and speeds are limited to nonnegative values. The colored surface samples that combined field at a fixed grid within the farm; its resolution may hide very narrow wakes. Particles sample the same field at their positions and form thin, fading trails and drift sideways more visibly in a wake. That disturbance is an illustration, not a turbulence calculation. Rotor animation varies with local wind speed, stopping below cut-in and above cut-out, but does not represent actual turbine RPM. Particle timing and rotor motion are illustrative and do not model detailed air trajectories or turbulence. This simplified approach does not model partial rotor overlap, terrain, atmospheric stability, turbulence, or detailed turbine interactions.

The reference turbine has **40 m diameter**, **60 m hub height**, thrust coefficient 0.88 and roughness length 0.3 m. The teaching power curve is 0 below 3 m/s, `0.3 × speed³` kW from 3 to below 12 m/s, 518.4 kW from 12 through 25 m/s, and 0 above 25 m/s. Total farm power is the sum of turbine outputs under the selected single wind condition; no-wake power is the same number of turbines at the undisturbed speed. Efficiency is farm power divided by no-wake power, and is undefined when no-wake power is zero.

The power results are instantaneous estimates, not annual energy, cost, a wind-resource assessment or an investment decision.

## Check a two-turbine layout in Python

For wind from north, place turbines at locations **96** and **86**. They share an east coordinate and are 200 m apart; 96 is upstream. This independent calculation checks the same single-wake model used in the explorer. It assumes the downstream rotor center is inside the wake.

```{code-cell} python
import math

speed, rotor_radius, hub, roughness, ct = 12.0, 20.0, 60.0, 0.3, 0.88
distance = 200.0  # downstream metres
induction = 0.5 * (1 - math.sqrt(1 - ct))
decay = 0.5 / math.log(hub / roughness)
initial_radius = rotor_radius * math.sqrt((1 - induction) / (1 - 2 * induction))
deficit = 2 * induction / (1 + decay * distance / initial_radius) ** 2
downstream_speed = speed * (1 - deficit)

def power_kw(wind):
    if wind < 3 or wind > 25:
        return 0.0
    return 0.3 * wind ** 3 if wind < 12 else 518.4

upstream_power = power_kw(speed)
downstream_power = power_kw(downstream_speed)
farm = upstream_power + downstream_power
ideal = 2 * upstream_power
print(f'Downstream speed: {downstream_speed:.2f} m/s')
print(f'Farm: {farm:.2f} kW; wake loss: {ideal-farm:.2f} kW')
print(f'Efficiency: {100*farm/ideal:.2f}%')
```

**Expected output:** downstream speed `9.21 m/s`, farm output `752.85 kW`, wake loss `283.95 kW`, and efficiency `72.61%`. The explorer displays power in MW, rounded to three decimals.

## Your investigation

1. Set wind from north at 12 m/s and spacing to 200 m. Place two turbines on the same north–south grid column, one downstream of the other. Record farm power, no-wake power, wake loss and each local wind speed.
2. Record this layout as a baseline. Remove the downstream turbine and place it on a different east–west column at the same northing. Compare power and explain the change using the wake direction.
3. Try several locations with the first turbine fixed. Record which gives the greatest farm power and compare the corresponding wake losses.
4. Rotate the wind direction to 90° from east and compare the same layouts again. Explain why a layout that performs well for one direction may perform differently for another.
5. Record your selected wind speed, direction, spacing, farm power and the model's limitations. Propose how a measured wind rose could support comparisons over several conditions; do not treat one wind direction as a full-year result.

## Going further

Use the wind workbook's [direction visualizations](../section7/workbook/wind.md#python-visualization-wind-direction) and [wind data lesson](../section5/wind-data.md) to understand direction and speed observations. Then complete the [combined final project](projectIntro.md), where wind output is considered alongside other resources and electricity demand.

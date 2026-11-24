# Project 3: Wind Energy

Design a small wind farm and explore how the position of a turbine changes the wind and power available to its neighbors. This project comes before the [combined renewable portfolio project](projectIntro.md). It uses an idealized, synthetic site to teach layout reasoning; it is not a design recommendation for an actual site.

## How to use the explorer

1. Watch the synthetic seasonal wind speed and direction change automatically. Direction is where wind comes **from**: north is 0°, east 90°, south 180° and west 270°. The compass displays speed and where the wind comes from; its arrow points downwind, the direction the air travels. Particles move downwind and slow in wakes. Use the stop/play icon to control the animation.
2. Start with \$50,000. Click a grid cell to purchase a turbine for \$50,000; click an occupied cell to remove it and recover \$25,000. Earn revenue from simulated energy and save enough to purchase additional turbines. Drag the scene to rotate it and scroll to zoom. For keyboard control, focus the grid and use the arrow keys to select a location; press Enter or Space to place or remove a turbine.
3. Read farm power, no-wake power, wake loss and efficiency. The selected location's wind and turbine power appear below the scene. Gold indicates the selected location; the colorbar inside the scene shows modeled wind speed in the wakes. Uncolored terrain indicates undisturbed wind.
4. Compare the energy and wind-speed histories as you change the layout and conditions. Remove a turbine by clicking its location again.

:::{iframe} ../../wind-lab/project.html
:width: 100%
:title: Interactive isometric wind farm layout explorer
:::

The square site is 2,000 m by 2,000 m; its 10 × 10 grid points are 200 m apart. Easting and northing are illustrative local coordinates. Turbine locations must be at least 200 m apart (five rotor diameters). Turbines in the illustration are enlarged relative to the farm so they can be seen.

## Seasonal wind

Wind speed and direction change automatically around a repeating synthetic annual pattern with additional unpredictable fluctuations. Each synthetic year takes 120 seconds of visible session time; the month label and conditions update approximately once per second. The minimum turbine spacing is fixed at 200 m.

This cycle is an illustration, not measured local climate or a forecast. The month label is a scenario position, not today's date. One visible second represents 73 simulated operating hours. This is a teaching scenario, not an estimate of actual annual production.

## Energy, revenue and the game

The game starts with **\$50,000**. A turbine costs **\$50,000**, and removing one recovers **\$25,000**. The farm earns **\$75 per MWh** generated, credited to the available budget. These deliberately simplified game values are not project costs or market prices. Choose locations that reduce wake losses so the farm earns enough to buy the next turbine. The price, turbine cost and salvage value are fixed.

The large value and first chart show cumulative **simulated energy** in MWh for the whole farm and each turbine location. The second chart shows free-stream and local wind speeds. One visible second represents 73 operating hours, so 120 seconds represents 8,760 hours. The model holds power constant between updates, integrates it over simulated time and applies the game energy price. Energy and money pause while the page is hidden. Stopping the 3D animation does not stop the game; reopening the project starts a new session. Charts retain the latest 300 samples; totals cover the full session. Removed turbines retain their previously accumulated energy under their location number.

This is an illustrative game, not measured data, a forecast, a real project budget or an annual energy assessment. The teaching turbine curve, wake model and wind scenario omit many real-world effects. Power (kW or MW) is an instantaneous rate; energy (MWh) accumulates over time.

## The model

The explorer implements a simplified Jensen-style, top-hat wake model inspired by the physics in the [wind-farm layout repository](https://github.com/isaigm/wflo/blob/main/layout_physics.py). When comparing or reusing the linked code, check its coordinate and direction conventions first. For east-positive x and north-positive y, the course uses meteorological directions clockwise from north; in a tested three-turbine example, the linked rotation produced the course result for a 90° easterly input when passed −90°. This is a conversion for those stated axes and that inspected implementation, not a claim about the external author’s intended convention. Verify cardinal-direction cases again if integrating a different version. A wake is wider and weaker farther downstream. A turbine is considered inside another's wake when its center falls inside the wake radius. Multiple velocity deficits are combined by root-sum-square, and speeds are limited to nonnegative values. The colored surface samples that combined field at a fixed grid within the farm; its resolution may hide very narrow wakes. Particles sample the same field at their positions and form thin, fading trails and drift sideways more visibly in a wake. That disturbance is an illustration, not a turbulence calculation. Rotor animation varies with local wind speed, stopping below cut-in and above cut-out, but does not represent actual turbine RPM. Particle timing and rotor motion are illustrative and do not model detailed air trajectories or turbulence. This simplified approach does not model partial rotor overlap, terrain, atmospheric stability, turbulence, or detailed turbine interactions.

Because the center-in-wake rule switches the modeled deficit on or off at the wake edge, a very small change in wind direction can cause an abrupt jump in calculated power. That discontinuity is a feature of this teaching model, not a prediction of a real turbine's continuous wake response.

The reference turbine has **40 m diameter**, **60 m hub height**, thrust coefficient 0.88 and roughness length 0.3 m. The teaching power curve is 0 below 3 m/s, `0.3 × speed³` kW from 3 to below 12 m/s, 518.4 kW from 12 through 25 m/s, and 0 above 25 m/s. Total farm power is the sum of turbine outputs under the current synthetic wind condition; no-wake power is the same number of turbines at the undisturbed speed. Efficiency is farm power divided by no-wake power, and is undefined when no-wake power is zero.

For a physical-boundary check only, at air density 1.225 kg/m³ this curve's cubic branch is approximately 39% of the undisturbed kinetic-energy flux through the stated rotor area, below the ideal Betz aerodynamic limit of 16/27 [@doeSmallWind]. The explorer does not adjust the curve for air density or establish actual turbine efficiency. Its stated thresholds include power at exactly 3 and 25 m/s; the separate theory curve uses zero output at those exact boundaries. Keep each convention when comparing examples.

The power results are instantaneous estimates, not annual energy, cost, a wind-resource assessment or an investment decision.

## Check a two-turbine layout in Python

For a fixed-condition example, consider wind from north at 12 m/s and turbines at locations **96** and **86**. They share an east coordinate and are 200 m apart; 96 is upstream. This independent calculation checks the same single-wake model used in the explorer. The live seasonal wind changes automatically, so its results will match only when the wind conditions match this example. The calculation assumes the downstream rotor center is inside the wake.

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

1. Place a turbine and observe its output as the synthetic wind changes. Record the wind direction and speed at the time of your observation.
2. Once you can afford a second turbine, try a position downstream of the first. Record farm power, wake loss and each local wind speed at that moment.
3. Compare layouts under similar wind conditions, accounting for the cost of removing a turbine. Explain how wake direction affects output and revenue.
4. Observe the layout over a complete synthetic seasonal cycle. Discuss how changing wind direction can change which turbine is affected by a wake.
5. Record the model's limitations. Explain how a measured wind rose and site-specific data could support a more realistic assessment.

## Going further

Use the wind workbook's [direction visualizations](../section7/workbook/wind.md#python-visualization-wind-direction) and [wind data lesson](../section5/wind-data.md) to understand direction and speed observations. Then complete the [combined final project](projectIntro.md), where wind output is considered alongside other resources and electricity demand.

## Worked plant-design exercises

Use the sliders in the [hydroelectric lesson](../../section6/hydroelectricEnergy.md#worked-plant-design-examples) to explore both cases. At the default settings, verify the calculations below independently in Python before changing inputs.

**Video unit correction:** At approximately 43:20 the ecological-group value 839.594 is labelled kWh, but it is power in kW. Energy requires multiplying power by a duration; the subsequent energy value has a different boundary.

### Exercise 21 — Diversion scheme: channel and penstocks

**Difficulty:** Medium

**Reference:** [Hydroelectric calculation video](https://youtu.be/ROnupNPkVf4) and the interactive worked examples in the hydroelectric lesson.

**Task:** A 12 m wide canal contains 4.5 m of water. The channel slope is 0.01%, Manning $n=0.014$ and length 2,500 m. Add 1 m intake loss. Two equal steel penstocks each carry half the channel flow at 5 m/s, with $n=0.012$, length 500 m and a 0.5 m transition loss. Assume the related scenario has a 60 m³/s mean river flow. For 50 MW electrical output, calculate net and gross head with turbine efficiency 0.92 and generator efficiency 0.97. Take $g=9.81$ m/s². Compare channel flow and losses with the video values 72.4 m³/s and 4.35 m.



```{code-cell} python
# Your solution for hydro exercise 21.
```

::::{dropdown} Step-by-step answer — Hydro 21

1. Calculate the channel flow.
2. Compute losses and net head.
3. Compare against the worked values.

```{code-cell} python
import math
# Step 1: Calculate the channel flow.
b, depth, slope, n = 12, 4.5, 0.0001, 0.014
area = b * depth
radius = area / (b + 2 * depth)
velocity = radius ** (2/3) * math.sqrt(slope) / n
flow = area * velocity
# Step 2: Compute losses and net head.
pipe_radius = math.sqrt((flow / 5 / 2) / math.pi)
pipe_hydraulic_radius = pipe_radius / 2
pipe_slope = 0.012 ** 2 * 5 ** 2 / pipe_hydraulic_radius ** (4/3)
loss = 2500 * slope + 1 + 500 * pipe_slope + 0.5
net_head = 50e6 / (1000 * 9.81 * flow * 0.92 * 0.97)
# Step 3: Compare against the worked values.
print(f'Q={flow:.2f} m³/s, loss={loss:.2f} m, net head={net_head:.2f} m, gross head={net_head+loss:.2f} m')
assert abs(flow - 72.4) < 0.1 and abs(loss - 4.35) < 0.05
```
**Interpretation:** The design discharge exceeds the given 60 m³/s mean river flow; do not treat design output as continuous annual generation.
::::

### Exercise 22 — Reservoir scheme: runoff and annual energy

**Difficulty:** Medium

**Reference:** [Hydroelectric calculation video](https://youtu.be/ROnupNPkVf4) and the interactive worked examples in the hydroelectric lesson.

**Task:** For a 365-day model year, 200 km², annual precipitation 1,200 mm and runoff coefficient 0.6, calculate annual runoff and mean inflow. Assume 10% spill, 1 m³/s ecological release, design flow twice the mean, design head 100 m and mean head 80 m. Compute design generator power at turbine efficiency 0.91 and generator efficiency 0.95, and annual generator energy using mean usable flow. Compare with 144 hm³/year, 4.57 m³/s and approximately 18.48 GWh.



```{code-cell} python
# Your solution for hydro exercise 22.
```

::::{dropdown} Step-by-step answer — Hydro 22

1. Calculate annual runoff.
2. Calculate usable flow and energy.
3. Compare against the worked values.

```{code-cell} python
# Step 1: Calculate annual runoff.
annual = 200e6 * 1.2 * 0.6
mean = annual / (365 * 86400)
# Step 2: Calculate usable flow and energy.
design = 2 * mean
usable = max(0, .9 * mean - 1)
generator_kw = 9.81 * design * 100 * .91 * .95
hours = 8760 * usable * 80 / (design * 100)
energy_gwh = generator_kw * hours / 1e6
# Step 3: Compare against the worked values.
print(f'{annual/1e6:.0f} hm³/year, {mean:.2f} m³/s, {generator_kw:.0f} kW design generator power, {energy_gwh:.2f} GWh/year')
assert annual == 144e6 and abs(mean - 4.57) < .01 and abs(energy_gwh - 18.48) < .05
```
**Interpretation:** Equivalent full-load hours incorporate average head as well as mean usable flow. Constant mean inflow does not demonstrate whether the ecological release can be met throughout the year.
::::


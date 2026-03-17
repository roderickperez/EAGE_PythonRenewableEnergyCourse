---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Hydroelectric Energy

Hydroelectric energy converts the kinetic and potential energy of flowing or
falling water into electricity. It currently supplies approximately **16% of
global electricity** and more than **60% of all renewable electricity**
worldwide, making it the largest source of renewable power today.

## How It Works

Water stored at elevation in a reservoir is directed through a **penstock**
(large pipe or tunnel) to a **turbine**. The falling water spins the turbine
blades, which rotate a **generator** and produce electricity. The spent water
exits through the **tailrace** back into the river.

The power output depends on two key factors:

1. **Hydraulic head** ($h$) — the vertical height difference between the
   reservoir surface and the turbine outlet.
2. **Flow rate** ($Q$) — the volume of water delivered to the turbine per
   second (m³/s).

## Key Equations

### Potential Energy of Water

The gravitational potential energy stored in a volume of water of mass $m$ at
height $h$ above the turbine is:

$$E = m \cdot g \cdot h$$

where:
- $E$ = energy (J)
- $m$ = mass of water (kg)
- $g$ = gravitational acceleration = 9.81 m/s²
- $h$ = hydraulic head (m)

```{video} ../manimFiles/section6/HydroelectricEnergy.mp4
:width: 800
:align: center
```

### Theoretical Hydraulic Power

For a continuous flow, power is the rate of energy delivery:

$$P_{\text{theoretical}} = \rho \cdot g \cdot Q \cdot h$$

where:
- $P$ = power (W)
- $\rho$ = water density ≈ 1000 kg/m³
- $Q$ = volumetric flow rate (m³/s)
- $h$ = net hydraulic head (m)

### Actual Power Output

Real turbines and generators are not perfectly efficient. The electric power
delivered to the grid is:

$$P_{\text{actual}} = \eta \cdot \rho \cdot g \cdot Q \cdot h$$

where $\eta$ is the **overall plant efficiency** (typically **0.85–0.95** for
modern Francis or Pelton turbines).

### Annual Energy Production

$$E_{\text{annual}} = P_{\text{actual}} \times 8760 \times CF$$

where:
- $8760$ = hours per year
- $CF$ = **capacity factor** — the fraction of rated capacity actually
  delivered on average (0.3–0.6 for run-of-river; up to 0.9 for large
  storage plants)

## Power Units

| Unit | Equivalent | Typical Application |
|------|-----------|---------------------|
| Watt (W) | 1 J/s | Single LED bulb (~10 W) |
| Kilowatt (kW) | 10³ W | Home appliances |
| Megawatt (MW) | 10⁶ W | Small power station |
| Gigawatt (GW) | 10⁹ W | Large dam / national grid |
| Terawatt (TW) | 10¹² W | Global energy statistics |

## World's Largest Hydroelectric Plants

| Rank | Plant | Country | Capacity (MW) |
|------|-------|---------|--------------|
| 1 | Three Gorges Dam | China | 22,500 |
| 2 | Itaipu Dam | Brazil / Paraguay | 14,000 |
| 3 | Xiluodu | China | 13,860 |
| 4 | Belo Monte | Brazil | 11,233 |
| 5 | Guri | Venezuela | 10,235 |
| 6 | Tucuruí | Brazil | 8,370 |
| 7 | Grand Coulee | USA | 6,809 |
| 8 | Longtan | China | 6,426 |

:::{admonition} The Three Gorges Dam
:class: note
The Three Gorges Dam on the Yangtze River (China) is the world's largest power
station by installed capacity at **22,500 MW**. Its reservoir stretches over
600 km. In 2020 it generated **111.8 TWh** of electricity — roughly equivalent
to burning 50 million tonnes of coal.
:::

:::{admonition} Environmental & Social Considerations
:class: warning
- Reservoirs flood large areas, displacing communities and ecosystems.
- Dams interrupt fish migration and river sediment transport.
- However, large hydro provides **grid stability** through flexible dispatchable
  power, and **seasonal energy storage** through pumped-hydro schemes.
- Run-of-river plants have a significantly smaller environmental footprint.
:::

## Python Exercises

### Exercise 1 — Basic Power Calculation

A hydroelectric plant has the following parameters:
- Flow rate: $Q = 500$ m³/s
- Hydraulic head: $h = 120$ m
- Plant efficiency: $\eta = 0.90$

Calculate and print the **actual power output in MW**.

```{code-cell} python
# Exercise 1 — Actual hydroelectric power output

rho = 1000    # water density (kg/m³)
g   = 9.81    # gravitational acceleration (m/s²)

Q   = 500     # flow rate (m³/s)
h   = 120     # hydraulic head (m)
eta = 0.90    # overall efficiency

# Calculate actual power in Watts, then convert to MW
# P_actual_W = ...
# P_actual_MW = ...

# Print the result with 2 decimal places
```

---

### Exercise 2 — Efficiency Sensitivity

Using the same $Q = 500$ m³/s and $h = 120$ m, build a **table** showing the
power output (MW) for efficiencies from **70 % to 95 %** in steps of 5 %.

```{code-cell} python
import numpy as np

rho = 1000
g   = 9.81
Q   = 500
h   = 120

efficiencies = np.arange(0.70, 1.00, 0.05)

# Print a formatted table:
# Efficiency (%)  |  Power (MW)
# --------------------------------
# Loop through efficiencies, compute power, and print each row
```

---

### Exercise 3 — Annual Energy and Household Supply

A run-of-river plant has:
- Installed capacity: $P = 250$ MW
- Capacity factor: $CF = 45\%$

1. Calculate the **annual energy production** in GWh.
2. Estimate how many average European households this plant can power
   (assume 3 500 kWh/year per household).

```{code-cell} python
P_MW              = 250    # installed capacity (MW)
CF                = 0.45   # capacity factor
hours_per_year    = 8760
household_kWh_yr  = 3500   # average EU household consumption (kWh/year)

# Annual energy in GWh:
# E_GWh = ...

# Number of equivalent households powered:
# n_households = ...

# Print both results
```

---

### Exercise 4 — Power Curve: Head vs. Flow Rate

Write a function `hydro_power(Q, h, eta=0.90)` that returns power in **MW**.

Then use `matplotlib` to plot **Power (MW)** on the y-axis vs. **Head (m)**
on the x-axis (range 20 – 200 m) for three flow rates: 100, 300, and 500 m³/s.
Include axis labels, a legend, a title, and a grid.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

rho = 1000
g   = 9.81

def hydro_power(Q, h, eta=0.90):
    """Return hydroelectric power output in MW."""
    # Implement the formula here
    pass

heads       = np.linspace(20, 200, 100)
flow_rates  = [100, 300, 500]

fig, ax = plt.subplots(figsize=(9, 5))

for Q in flow_rates:
    # Calculate power array and add a labeled plot line
    pass

ax.set_xlabel("Hydraulic Head (m)")
ax.set_ylabel("Power (MW)")
ax.set_title("Hydroelectric Power vs. Hydraulic Head")
ax.legend(title="Flow rate (m³/s)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### Exercise 5 — Levelized Cost of Energy (LCOE)

The **Levelized Cost of Energy (LCOE)** is the break-even cost per unit of
electricity over the plant lifetime:

$$LCOE = \frac{C_{\text{capital}} \cdot CRF + C_{\text{O\&M}}}{E_{\text{annual}}}$$

The **Capital Recovery Factor (CRF)** for discount rate $r$ over $T$ years is:

$$CRF = \frac{r(1+r)^T}{(1+r)^T - 1}$$

| Parameter | Value |
|-----------|-------|
| Capital cost | USD 1.5 billion |
| Annual O&M | USD 15 million/year |
| Discount rate | 6 % |
| Plant lifetime | 50 years |
| Capacity | 250 MW, CF = 45 % |

**Tasks:**
1. Calculate the LCOE in **USD/MWh**.
2. Plot **LCOE vs. discount rate** (2 % – 15 %) for plant lifetimes of 30, 50,
   and 80 years.
3. Identify the discount rate at which hydro becomes more expensive than a
   typical gas peaker (~$60/MWh) — mark it on the plot.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

capital_cost  = 1.5e9    # USD
oam_annual    = 15e6     # USD/year
r             = 0.06     # discount rate
T             = 50       # plant lifetime (years)
P_MW          = 250      # installed capacity (MW)
CF            = 0.45     # capacity factor
hours_per_year= 8760

def lcoe_usd_per_MWh(r, T, capital=1.5e9, oam=15e6, P_MW=250, CF=0.45):
    """
    Calculate LCOE in USD/MWh.
    r       : discount rate (fraction)
    T       : lifetime (years)
    capital : capital cost (USD)
    oam     : annual O&M cost (USD/year)
    P_MW    : installed capacity (MW)
    CF      : capacity factor
    """
    # Step 1: Capital Recovery Factor
    # CRF = r*(1+r)**T / ((1+r)**T - 1)

    # Step 2: Annual energy (MWh)
    # E_annual_MWh = P_MW * CF * hours_per_year

    # Step 3: LCOE
    # LCOE = (capital * CRF + oam) / E_annual_MWh
    pass

# Part 1 — single LCOE value at r=6%, T=50y
# print(f"LCOE = {lcoe_usd_per_MWh(0.06, 50):.2f} USD/MWh")

# Part 2 & 3 — plot
discount_rates = np.linspace(0.02, 0.15, 200)
lifetimes = [30, 50, 80]

fig, ax = plt.subplots(figsize=(10, 5))

for T_val in lifetimes:
    # Compute LCOE for each discount rate and plot
    pass

ax.axhline(60, linestyle='--', color='crimson', label='Gas peaker (~$60/MWh)')
ax.set_xlabel("Discount Rate")
ax.set_ylabel("LCOE (USD/MWh)")
ax.set_title("Hydroelectric LCOE vs. Discount Rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

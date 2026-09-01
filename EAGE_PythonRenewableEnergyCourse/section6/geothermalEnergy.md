---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Geothermal Energy

Geothermal energy taps the naturally occurring heat stored in the Earth's
interior. The planet's core temperature is estimated at **5 000 – 6 000 °C**,
and this heat flows continuously outward toward the surface. Geothermal plants
extract this heat by drilling wells into underground reservoirs of hot rock and
fluid, converting the thermal energy into electricity — cleanly, continuously,
and with a very small land footprint.

## Types of Geothermal Plants

| Type | Reservoir Temp. | Mechanism |
|------|----------------|-----------|
| **Dry Steam** | > 235 °C | Steam directly drives the turbine |
| **Flash Steam** | 180 – 350 °C | High-pressure brine flashes to steam in a separator |
| **Binary / ORC** | 100 – 180 °C | Hot brine heats a low-boiling organic fluid in a heat exchanger; organic vapour drives the turbine |
| **Enhanced (EGS)** | Hot, low-permeability rock | Fluid circulates through engineered fractures and returns carrying heat; it may flash to steam or heat a secondary working fluid [@doe_egs] |

## Key Equations

### Fourier's Law of Heat Conduction

Heat flux through rock:

$$q = -k \frac{dT}{dz}$$

where:
- $q$ = heat flux density (W/m²)
- $k$ = thermal conductivity of rock ≈ 1.5 – 3.5 W/(m·K)
- $dT/dz$ = temperature gradient with depth (°C/m)

### Geothermal Gradient and Temperature Profile

The global average geothermal gradient is approximately 25 – 30 °C/km.
Temperature at depth $z$ (km):

$$T(z) = T_{\text{surface}} + G_{th} \cdot z$$

where $G_{th}$ is the geothermal gradient in °C/km.

### Thermal Power Extracted from a Well

$$P_{\text{thermal}} = \dot{m} \cdot c_p \cdot \Delta T$$

where:
- $P_{\text{thermal}}$ = thermal power (W)
- $\dot{m}$ = mass flow rate of geothermal fluid (kg/s)
- $c_p$ = specific heat capacity of water = 4 186 J/(kg·K)
- $\Delta T = T_{\text{production}} - T_{\text{reinjection}}$ (K or °C)

### Maximum (Carnot) Efficiency

No heat engine can exceed the Carnot efficiency:

$$\eta_{\text{Carnot}} = 1 - \frac{T_{\text{cold}}}{T_{\text{hot}}}$$

Temperatures **must** be in **Kelvin** (K = °C + 273.15).

### ORC Electrical Output

Binary/ORC efficiency is site- and design-dependent; values near 10–15% are illustrative for moderate-temperature resources, not a universal range:

$$P_{\text{electrical}} = \eta_{ORC} \cdot P_{\text{thermal}}$$

### Power Density

$$P_{\text{density}} = \frac{P_{\text{electrical}}}{A_{\text{field}}}$$

where $A_{\text{field}}$ is the surface area of the geothermal field (km²).

## Power Units

| Unit | Meaning |
|------|---------|
| MWt | megawatt **thermal** — heat extracted |
| MWe | megawatt **electrical** — net electricity sent to grid |
| MWe/km² | land-use power density |
| EJ | exajoule = 10¹⁸ J — global energy statistics |

## World's Largest Geothermal Plants (2024)

| Rank | Plant | Country | Capacity (MWe) |
|------|-------|---------|---------------|
| 1 | The Geysers | USA (California) | 1,520 |
| 2 | Larderello Complex | Italy | 915 |
| 3 | Olkaria Complex | Kenya | ~800 |
| 4 | Cerro Prieto | Mexico | 720 |
| 5 | Salak | Indonesia | 377 |
| 6 | Hellisheidi | Iceland | 303 |
| 7 | Darajat | Indonesia | 270 |

:::{admonition} Iceland — A Geothermal Powerhouse
:class: note
Iceland sits on the Mid-Atlantic Ridge, giving it exceptional geothermal
resources. Geothermal supplies **~65 % of Iceland's primary energy** — heating
90 % of homes directly and generating significant electricity. The Hellisheidi
plant near Reykjavik also captures and mineralises CO₂ into basalt rock,
making it a prominent demonstration of in-situ carbon mineralisation associated with geothermal operations. It should not be described as the world's first large-scale CCS facility without a tightly defined comparison.
:::

:::{admonition} Advantages of Geothermal Power
:class: tip
- **Baseload**: operates 24/7 regardless of weather — capacity factors often
  exceed 90 %, far higher than solar or wind.
- **Compact surface footprint**: land use varies with resource type, well spacing, pipelines, and the project boundary used for comparison.
- **Low emissions**: binary plants emit virtually no greenhouse gases.
- **Grid support**: many geothermal plants are dispatchable, but ramp rates and flexibility depend on the reservoir, steam field, and plant design.
:::

## Python Exercises

### Exercise 1 — Temperature Profile with Depth

The average geothermal gradient in Central Europe is **30 °C/km**.
Surface temperature: 15 °C.

1. Print the subsurface temperature at depths of 1, 2, 3, 4, and 5 km.
2. Find the **minimum depth** required to reach 150 °C (threshold for
   flash-steam plants).

```{code-cell} python
# Exercise 1 — Temperature at depth

T_surface  = 15     # °C
gradient   = 30     # °C/km
depths_km  = [1, 2, 3, 4, 5]

# 1. Print temperature at each depth: T(z) = T_surface + gradient * z

# 2. Find minimum depth to reach flash-steam threshold
T_target = 150   # °C
# depth_min = (T_target - T_surface) / gradient
```

---

### Exercise 2 — Thermal Power from a Production Well

A geothermal well produces hot brine:
- Mass flow rate: 80 kg/s
- Production temperature: 180 °C
- Reinjection temperature: 70 °C
- ORC efficiency: 12 %

Calculate:
1. The **thermal power** (kW).
2. The **electrical power** from the ORC unit (kW).

```{code-cell} python
# Exercise 2 — Thermal and electrical power from a single well

m_dot    = 80     # mass flow rate (kg/s)
cp       = 4186   # specific heat of water (J/kg/K)
T_prod   = 180    # production temperature (°C)
T_reinj  = 70     # reinjection temperature (°C)
eta_orc  = 0.12   # ORC efficiency

# Thermal power in kW
# dT = T_prod - T_reinj
# P_thermal_kW = m_dot * cp * dT / 1000

# Electrical power in kW
# P_elec_kW = eta_orc * P_thermal_kW

# Print both results
```

---

### Exercise 3 — Carnot vs. ORC Efficiency

For reservoir temperatures from 100 °C to 300 °C (cold side fixed at 30 °C):

1. Calculate the **Carnot efficiency** at each temperature.
2. Assume practical ORC efficiency = 60 % of Carnot.
3. Plot both curves on the same graph.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

T_cold_C        = 30
T_hot_range_C   = np.linspace(100, 300, 200)

T_cold_K  = T_cold_C + 273.15
T_hot_K   = T_hot_range_C + 273.15

# Carnot efficiency: eta_carnot = 1 - T_cold_K / T_hot_K
# Practical ORC:    eta_orc = 0.60 * eta_carnot

fig, ax = plt.subplots(figsize=(9, 5))

# Plot eta_carnot and eta_orc vs T_hot_range_C
# Use different line styles

ax.set_xlabel("Reservoir Temperature (°C)")
ax.set_ylabel("Efficiency")
ax.set_title("Geothermal: Carnot vs. Practical ORC Efficiency")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### Exercise 4 — Required Drilling Depth by Region

To reach a target well temperature of **200 °C** (surface = 15 °C), calculate
the required drilling depth for each region and its gradient range:

| Region | Gradient range (°C/km) |
|--------|----------------------|
| Stable continental shield | 15 – 20 |
| Average continental | 25 – 30 |
| Volcanic / rift zone | 50 – 80 |
| Iceland / El Salvador | 80 – 120 |

1. Calculate **minimum and maximum drilling depth** for each region.
2. Display as a **horizontal range bar chart**.
3. Annotate each bar with the depth range in km.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

T_target   = 200   # °C
T_surface  = 15    # °C

regions = {
    "Stable continental shield": (15, 20),
    "Average continental":       (25, 30),
    "Volcanic / rift zone":      (50, 80),
    "Iceland / El Salvador":     (80, 120),
}

# For each region: depth = (T_target - T_surface) / gradient
# depth_min (shallowest required) uses the highest gradient
# depth_max (deepest required) uses the lowest gradient

names   = list(regions.keys())
depths = [
    ((T_target - T_surface) / g_max,
     (T_target - T_surface) / g_min)
    for (g_min, g_max) in regions.values()
]

fig, ax = plt.subplots(figsize=(10, 5))

# ax.barh(...)  — show range from depth_min to depth_max
# Annotate with values

ax.set_xlabel("Required Drilling Depth (km)")
ax.set_title(f"Depth Required to Reach {T_target} °C by Tectonic Region")
# Keep the conventional scale: shallower depths on the left.
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

---

### Exercise 5 — Multi-Well Field Economic Optimisation

A geothermal developer evaluates a field with up to **20 production wells**:

| Parameter | Value |
|-----------|-------|
| Flow per well | 60 kg/s |
| Production / reinjection temperature | 220 °C / 80 °C |
| ORC efficiency | 13 % |
| CAPEX per well | USD 8 million |
| OPEX per well/year | USD 200 000 |
| Electricity price | USD 80/MWh |
| Plant lifetime | 30 years |

1. Calculate **electrical power** and **annual revenue** for 1 to 20 wells.
2. Calculate the **simple payback period** (CAPEX / annual net profit).
3. Plot payback period vs. number of wells; add a target line at 10 years.
4. Explain why payback is constant when every cost and revenue term scales linearly with the number of wells. Then add a fixed plant CAPEX and one injection well per two production wells to create a meaningful optimisation.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

m_dot            = 60      # kg/s per well
cp               = 4186
T_prod           = 220     # °C
T_reinj          = 80      # °C
eta_orc          = 0.13

capex_per_well   = 8e6     # USD
opex_per_well_yr = 200_000 # USD/year
price_MWh        = 80      # USD/MWh
hours_per_year   = 8760

# Per-well thermal and electrical power
# dT = T_prod - T_reinj
# P_thermal_kW = m_dot * cp * dT / 1000
# P_elec_kW    = eta_orc * P_thermal_kW
# P_elec_MW    = P_elec_kW / 1000

n_range = np.arange(1, 21)

payback_years = []

for n in n_range:
    # total CAPEX, annual revenue, annual OPEX, payback
    # payback = total_capex / (annual_revenue - annual_opex)
    pass

fig, ax = plt.subplots(figsize=(9, 5))
# ax.plot(n_range, payback_years, ...)
ax.axhline(10, linestyle='--', color='crimson', label='10-year target')
ax.set_xlabel("Number of Production Wells")
ax.set_ylabel("Simple Payback Period (years)")
ax.set_title("Geothermal Field: Payback Period vs. Number of Wells")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

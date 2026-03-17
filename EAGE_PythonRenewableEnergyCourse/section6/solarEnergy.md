---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Solar Energy

Solar energy is the most abundant energy resource on Earth. Every hour, the Sun
delivers to Earth's surface more energy than humanity consumes in an entire year.
**Photovoltaic (PV)** technology converts sunlight directly into electricity via
the **photoelectric effect**, while **Concentrated Solar Power (CSP)** focuses
sunlight to drive a heat engine.

## How Photovoltaic Cells Work

A solar cell is a semiconductor device (typically crystalline silicon) that:

1. Absorbs photons from sunlight.
2. Uses photon energy to free electrons from atoms (photoelectric effect).
3. An internal electric field (p-n junction) drives free electrons through an
   external circuit, creating **direct current (DC)**.
4. An **inverter** converts DC to alternating current (AC) for the grid.

Multiple cells assembled into a **module** (panel); multiple panels form an
**array**; arrays plus inverters and wiring form a **solar farm**.

## Key Equations

### Solar Irradiance and Standard Test Conditions

The solar constant at Earth's orbit is:

$$G_{SC} = 1361 \text{ W/m}^2$$

At ground level under clear sky, irradiance is reduced by the atmosphere.
Panel performance is rated at **Standard Test Conditions (STC)**:

$$G_{STC} = 1000 \text{ W/m}^2, \quad T_{cell} = 25\,°C$$

### PV Power Output

The instantaneous power of a PV array:

$$P = A \cdot G \cdot \eta_{\text{panel}}$$

where:
- $P$ = power (W)
- $A$ = total panel area (m²)
- $G$ = solar irradiance (W/m²)
- $\eta_{\text{panel}}$ = panel efficiency (typically 0.15 – 0.22)

### Temperature Correction

PV panels lose efficiency at elevated temperatures:

$$P(T) = P_{STC} \left[1 + \alpha\,(T_{\text{cell}} - T_{STC})\right]$$

where:
- $\alpha \approx -0.004$ /°C for crystalline silicon
  (power drops ~0.4% per °C above 25 °C)
- $T_{STC} = 25\,°C$

### Cell Temperature Estimate

$$T_{\text{cell}} = T_{\text{ambient}} + \frac{G}{800}\,(NOCT - 20)$$

where **NOCT** (Nominal Operating Cell Temperature) ≈ 45 °C.

### Annual Energy Production

$$E_{\text{annual}} = P_{\text{rated}} \cdot CF \cdot 8760 \text{ h}$$

Typical capacity factors for solar PV:

| Region | Capacity Factor |
|--------|----------------|
| Northern Europe | 10 – 12 % |
| Southern Europe / Mediterranean | 16 – 18 % |
| Middle East / North Africa | 20 – 25 % |
| Atacama Desert (Chile) | 26 – 30 % |

## Power Units

| Unit | Context |
|------|---------|
| Wp (Watt-peak) | Rated power of one panel at STC |
| kWp / MWp / GWp | Rated capacity of arrays / farms |
| kWh / MWh / GWh | Electrical energy produced |

## World's Largest Solar Installations (2024)

| Rank | Plant | Country | Capacity (MWp) |
|------|-------|---------|---------------|
| 1 | Bhadla Solar Park | India | 2,700 |
| 2 | Huanghe Hydropower Solar Park | China | 2,200 |
| 3 | Pavagada Solar Park | India | 2,050 |
| 4 | Benban Solar Park | Egypt | 1,650 |
| 5 | Al Dhafra Solar PV | UAE | 1,500 |
| 6 | Tengger Desert Solar Park | China | 1,547 |

:::{admonition} Solar's Cost Revolution
:class: note
Solar PV has undergone the steepest cost decline of any energy technology in
history — over **90 % cost reduction since 2010**. Paired with battery storage,
solar can now provide reliable round-the-clock power. Several countries have
already recorded hours where solar alone met 100 % of national electricity demand.
:::

## Python Exercises

### Exercise 1 — Panel Power Output

A residential rooftop installation has:
- 20 panels, each 2.0 m² in area
- Panel efficiency: 20 %
- Current solar irradiance: 850 W/m²

Calculate and print the **total instantaneous power output in kW**.

```{code-cell} python
# Exercise 1 — Rooftop solar power output

n_panels         = 20
area_per_panel   = 2.0    # m²
eta              = 0.20   # panel efficiency
G                = 850    # irradiance (W/m²)

# Total panel area
A_total = n_panels * area_per_panel

# Power in Watts, then convert to kW
# P_W  = ...
# P_kW = ...

# Print the result
```

---

### Exercise 2 — Temperature Derating

A 400 Wp panel on a hot summer day:
- Ambient temperature: 38 °C
- Irradiance: 1 100 W/m²
- NOCT: 45 °C
- Temperature coefficient: $\alpha = -0.004$ /°C

1. Calculate the **cell temperature**.
2. Calculate the **corrected power output**.
3. Calculate the **percentage power loss** due to heat.

```{code-cell} python
# Exercise 2 — Temperature derating of a PV panel

P_STC      = 400      # rated power at STC (W)
T_ambient  = 38       # ambient temperature (°C)
G          = 1100     # irradiance (W/m²)
NOCT       = 45       # nominal operating cell temperature (°C)
alpha      = -0.004   # temperature coefficient (/°C)
T_STC      = 25       # standard test temperature (°C)

# Step 1: Cell temperature
# T_cell = T_ambient + (G / 800) * (NOCT - 20)

# Step 2: Corrected power
# P_corrected = P_STC * (1 + alpha * (T_cell - T_STC))

# Step 3: Percentage power loss
# loss_pct = (P_STC - P_corrected) / P_STC * 100

# Print all three results
```

---

### Exercise 3 — Solar Farm Annual Energy and Lifetime Degradation

A utility-scale solar farm in southern Spain:
- Installed capacity: 150 MWp
- Capacity factor: 20 %
- Annual panel degradation: 0.5 % per year

1. Calculate **Year 1 energy production** in GWh.
2. Calculate **Year 25 energy production** (after degradation).
3. Calculate the **total energy over 25 years** and the **average annual
   production** over that period.

```{code-cell} python
import numpy as np

P_MWp        = 150    # installed capacity (MWp)
CF           = 0.20   # capacity factor
degradation  = 0.005  # 0.5 % per year
n_years      = 25
hours        = 8760

# Year 1 energy in GWh
# E_year1_GWh = ...

# Energy in year i (0-indexed): E_year1 * (1 - degradation)**i
# Compute an array of annual energies for years 0..24

# Total energy over 25 years, average annual energy
# Print all results
```

---

### Exercise 4 — Irradiance and Temperature Heatmap

Write a function `pv_power_kW(G, T_amb, A=100, eta=0.20, NOCT=45, alpha=-0.004)`
that returns power in kW.

Then create a **heatmap** showing power output for:
- x-axis: irradiance 0 – 1200 W/m² (50 values)
- y-axis: ambient temperature 10 – 45 °C (20 values)

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def pv_power_kW(G, T_amb, A=100, eta=0.20, NOCT=45, alpha=-0.004):
    """
    PV power output in kW.
    G     : irradiance (W/m²)
    T_amb : ambient temperature (°C)
    A     : total panel area (m²)
    """
    T_STC  = 25
    T_cell = T_amb + (G / 800) * (NOCT - 20)
    # Corrected efficiency and power
    # P_kW = ...
    pass

G_vals = np.linspace(0, 1200, 50)
T_vals = np.linspace(10, 45, 20)

# Build 2D grid: power_grid[i, j] = pv_power_kW(G_vals[j], T_vals[i])
# Hint: use np.vectorize or a nested loop

fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.pcolormesh(G_vals, T_vals, power_grid, cmap='YlOrRd', shading='auto')
# plt.colorbar(im, ax=ax, label='Power (kW)')
ax.set_xlabel("Irradiance (W/m²)")
ax.set_ylabel("Ambient Temperature (°C)")
ax.set_title("PV Array Power Output (kW) vs. Irradiance and Temperature")
plt.tight_layout()
plt.show()
```

---

### Exercise 5 — Simulating a Solar Day (Hourly Resolution)

Model hourly irradiance using a **clear-sky sine curve**:

$$G(t) = G_{\max} \cdot \sin\!\left(\pi \cdot \frac{t - t_{\text{rise}}}{t_{\text{set}} - t_{\text{rise}}}\right), \quad t_{\text{rise}} \le t \le t_{\text{set}}$$

otherwise $G(t) = 0$.

Parameters: sunrise = 6 h, sunset = 20 h, $G_{\max} = 950$ W/m².

Solar farm: 500 kWp ($A = 2500$ m², $\eta = 0.20$).

Temperature profile: linear from 16 °C at sunrise to 35 °C at 14 h, cooling
back to 22 °C at sunset.

1. Generate hourly irradiance and temperature arrays.
2. Compute hourly power output in kW (using your function from Exercise 4).
3. Calculate **total daily energy** in kWh.
4. Plot irradiance and power on the same figure with **twin y-axes**.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

hours   = np.arange(0, 24, 1)
t_rise  = 6
t_set   = 20
G_max   = 950    # W/m²
A       = 2500   # m²  (500 kWp plant)
eta     = 0.20

# Step 1: Irradiance per hour
# G = np.where((hours >= t_rise) & (hours <= t_set),
#              G_max * np.sin(np.pi * (hours - t_rise) / (t_set - t_rise)), 0)

# Step 2: Temperature profile
# Use np.interp with key points: e.g. {6: 16, 14: 35, 20: 22}
T_knots_h  = [6, 14, 20]
T_knots_C  = [16, 35, 22]
# T_amb = np.interp(hours, T_knots_h, T_knots_C)
# (set to 15 °C outside sunrise-sunset)

# Step 3: Power for each hour (use pv_power_kW from Exercise 4)
# P_hourly_kW = [pv_power_kW(G[i], T_amb[i], A=A, eta=eta) for i in range(24)]

# Step 4: Total daily energy
# E_daily_kWh = sum(P_hourly_kW)  (each step = 1 hour)

# Step 5: Twin-axis plot
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

# ax1.bar(hours, G, color='gold', alpha=0.6, label='Irradiance')
# ax2.plot(hours, P_hourly_kW, color='steelblue', linewidth=2, label='Power')

ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Irradiance (W/m²)", color='goldenrod')
ax2.set_ylabel("Power Output (kW)", color='steelblue')
ax1.set_title(f"Solar Day Simulation  |  Daily Energy ≈ {0:.0f} kWh")
ax1.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()
```

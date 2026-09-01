---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Wind Energy

Test your knowledge of wind power theory and Python calculations.

---

:::{admonition} Question 1
:class: note

**Wind Power Formula**

The theoretical power available in the wind is:

$$P = \frac{1}{2} \rho A v^3$$

where $\rho$ is air density (kg/m³), $A$ is the rotor swept area (m²), and $v$ is wind speed (m/s).

Calculate the theoretical power (kW) for a turbine with:
- Rotor diameter: $D = 90$ m
- Wind speed: $v = 11$ m/s
- Air density: $\rho = 1.225$ kg/m³
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

We first compute the swept area $A = \pi (D/2)^2$, then apply the power formula:

```{code-cell} python
import math

rho = 1.225     # kg/m³
D   = 90        # m
v   = 11        # m/s

A   = math.pi * (D / 2)**2
P_W = 0.5 * rho * A * v**3
P_kW = P_W / 1000

print(f"Swept area: {A:.2f} m²")
print(f"Theoretical power: {P_kW:.2f} kW")
```
:::

---

:::{admonition} Question 2
:class: note

**Betz Limit**

The Betz limit states that the maximum fraction of wind power that can theoretically be extracted is:

$$C_{P,max} = \frac{16}{27} \approx 0.593$$

For the turbine in Question 1 ($D = 90$ m, $v = 11$ m/s, $\rho = 1.225$ kg/m³):
1. Calculate the maximum power extractable by the turbine (kW).
2. If the turbine has a real power coefficient $C_P = 0.42$, what is the actual power output (kW)?
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

We apply the Betz limit and then a realistic $C_P$:

```{code-cell} python
import math

rho  = 1.225
D    = 90
v    = 11
Cp_betz = 16 / 27
Cp_real = 0.42

A       = math.pi * (D / 2)**2
P_wind  = 0.5 * rho * A * v**3
P_betz  = Cp_betz * P_wind / 1000
P_real  = Cp_real * P_wind / 1000

print(f"Total wind power:        {P_wind/1000:.2f} kW")
print(f"Betz limit power:        {P_betz:.2f} kW  (Cp = {Cp_betz:.3f})")
print(f"Actual turbine power:    {P_real:.2f} kW  (Cp = {Cp_real})")
```
:::

---

:::{admonition} Question 3
:class: note

**Wind Speed vs. Power Curve**

For a turbine with $D = 100$ m, $\rho = 1.225$ kg/m³, and $C_P = 0.45$, plot the output power (kW) against wind speeds from 0 to 25 m/s.

Add:
- Vertical dashed lines for cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s) speeds.
- Appropriate axis labels and a title.
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

We use NumPy for the speed array, calculate power, then plot with annotations:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

rho = 1.225
D   = 100
Cp  = 0.45
A   = np.pi * (D / 2)**2

v = np.linspace(0, 25, 200)
v_cut_in, v_rated, v_cut_out = 3, 12, 25
P_rated = 0.5 * rho * A * Cp * v_rated**3 / 1000
P = np.zeros_like(v)
ramp = (v >= v_cut_in) & (v < v_rated)
P[ramp] = P_rated * (v[ramp]**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3)
P[(v >= v_rated) & (v < v_cut_out)] = P_rated

plt.figure(figsize=(8, 5))
plt.plot(v, P, color="steelblue", linewidth=2.5, label="Power curve")
plt.axvline(3,  color="green",  linestyle="--", alpha=0.7, label="Cut-in (3 m/s)")
plt.axvline(12, color="orange", linestyle="--", alpha=0.7, label="Rated (12 m/s)")
plt.axvline(25, color="red",    linestyle="--", alpha=0.7, label="Cut-out (25 m/s)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.title("Wind Turbine Power Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 4
:class: note

**Weibull Distribution**

Wind speed at a site follows a Weibull distribution. Generate 5000 samples with shape parameter $k = 2.2$ and scale parameter $\lambda = 8.5$ m/s using:

```python
np.random.weibull(k, 5000) * lam
```

Then:
1. Plot a histogram of the generated wind speeds (30 bins).
2. Print the mean and standard deviation.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

The Weibull distribution models wind speed frequency distributions:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
k   = 2.2
lam = 8.5

samples = np.random.weibull(k, 5000) * lam

print(f"Mean wind speed: {samples.mean():.2f} m/s")
print(f"Std deviation:   {samples.std():.2f} m/s")

plt.figure(figsize=(7, 4))
plt.hist(samples, bins=30, color="cornflowerblue", edgecolor="black", density=True)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Probability Density")
plt.title("Weibull Wind Speed Distribution (k=2.2, λ=8.5)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 5
:class: note

**Annual Energy Production (AEP)**

A wind turbine has the following simplified power curve (wind speed [m/s] → power [kW]):

```python
v_curve = [0, 3, 5, 7, 9, 11, 13, 24.999, 25]
P_curve = [0, 0, 100, 400, 900, 1500, 2000, 2000, 0]
```

Using the Weibull samples from Question 4 (`k=2.2`, `lam=8.5`, 5000 samples):
1. Interpolate the power for each wind speed sample.
2. Estimate the mean power output (kW).
3. Estimate the annual energy production (MWh).
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We use `np.interp` to map each wind speed sample to a power value:

```{code-cell} python
import numpy as np

np.random.seed(42)
k   = 2.2
lam = 8.5
samples = np.random.weibull(k, 5000) * lam

v_curve = [0, 3, 5, 7, 9, 11, 13, 24.999, 25]
P_curve = [0, 0, 100, 400, 900, 1500, 2000, 2000, 0]

power_samples = np.interp(samples, v_curve, P_curve, left=0, right=0)

mean_power_kW = power_samples.mean()
AEP_MWh = mean_power_kW * 8760 / 1000

print(f"Mean power output:       {mean_power_kW:.2f} kW")
print(f"Annual energy production:{AEP_MWh:,.1f} MWh")
```
:::

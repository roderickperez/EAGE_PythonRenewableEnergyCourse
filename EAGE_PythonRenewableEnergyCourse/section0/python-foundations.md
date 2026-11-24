# Python foundations for renewable energy

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section0/python-foundations.ipynb)

Run the first example with **Edit this code** in the sandbox. For later cells, **Include preceding examples** loads their setup. In Jupyter or Colab, run the cells in order and rerun from a fresh kernel before sharing work. A Python script runs top to bottom. A notebook retains variables from earlier runs.

## Collections, control flow, functions and modules

A list keeps readings in sequence, a dictionary labels fields, a tuple can hold a fixed pair and a set collects unique labels. Functions accept inputs and return results. Modules let you reuse code, as with `isfinite` from the standard-library `math` module. Energy in MWh equals interval-average power in MW times duration in hours.

```{code-cell} python
from math import isfinite

readings = [
    {"source": "wind", "power_mw": 3.0, "hours": 0.5},
    {"source": "solar", "power_mw": 2.0, "hours": 0.5},
    {"source": "wind", "power_mw": 4.0, "hours": 1.0},
]

def interval_energy(power_mw, hours):
    if not isfinite(power_mw) or not isfinite(hours) or power_mw < 0 or hours <= 0:
        raise ValueError("Power must be finite and nonnegative and duration positive.")
    return power_mw * hours

sources = {reading["source"] for reading in readings}
totals = dict.fromkeys(sources, 0.0)
for reading in readings:
    totals[reading["source"]] += interval_energy(reading["power_mw"], reading["hours"])

wind_result = ("wind", totals["wind"])
print(wind_result, sorted(totals.items()))
assert totals == {"wind": 5.5, "solar": 1.0}
```

## NumPy, pandas and Matplotlib

Arrays apply operations element by element. DataFrames label rows and columns and can group related records. A Matplotlib chart needs axis labels and units.

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

frame = pd.DataFrame(readings)
power = frame["power_mw"].to_numpy(dtype=float)
duration = frame["hours"].to_numpy(dtype=float)
frame["energy_mwh"] = np.multiply(power, duration)
by_source = frame.groupby("source")["energy_mwh"].sum()
assert by_source.to_dict() == totals

ax = by_source.plot.bar(rot=0)
ax.set(xlabel="Source", ylabel="Energy (MWh)", title="Energy over recorded intervals")
plt.show()
```

## Practice

Add a hydro reading of 1.2 MW lasting 15 minutes. Check your result with NumPy. Then test a zero-duration interval and explain why the function rejects it.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
hydro = {"source": "hydro", "power_mw": 1.2, "hours": 15 / 60}
hydro_energy = interval_energy(hydro["power_mw"], hydro["hours"])
assert np.isclose(hydro_energy, 0.3)
try:
    interval_energy(1.2, 0)
except ValueError as error:
    print(error)
else:
    raise AssertionError("Zero duration should be rejected.")
```
:::

Continue with the [General Python quiz](../section7/renewableEnergyquizzes/generalPython.md).

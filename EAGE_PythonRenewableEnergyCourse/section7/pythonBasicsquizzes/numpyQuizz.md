---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: NumPy

Test your knowledge of NumPy — the foundation for numerical computing in Python.

---

:::{admonition} Question 1
:class: note

**Array Creation**

Create the following arrays using NumPy:
1. A 1D array of integers from 0 to 9.
2. A 1D array of 5 evenly spaced values between 0 and 100.
3. A 3×3 array filled with zeros.
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

NumPy provides `arange`, `linspace`, and `zeros` for array creation:

```{code-cell} python
import numpy as np

# 1. Array 0 to 9
arr1 = np.arange(10)
print("arange:", arr1)

# 2. 5 evenly spaced values 0–100
arr2 = np.linspace(0, 100, 5)
print("linspace:", arr2)

# 3. 3x3 zeros
arr3 = np.zeros((3, 3))
print("zeros:\n", arr3)
```
:::

---

:::{admonition} Question 2
:class: note

**Indexing and Slicing**

Given the array:

```python
power_output = np.array([120, 245, 310, 480, 295, 175, 390, 420])
```

Write code to:
1. Print the first and last element.
2. Print elements from index 2 to 5 (inclusive).
3. Print all elements greater than 300.
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

NumPy supports slicing and boolean indexing:

```{code-cell} python
import numpy as np

power_output = np.array([120, 245, 310, 480, 295, 175, 390, 420])

# 1. First and last
print("First:", power_output[0], "Last:", power_output[-1])

# 2. Index 2 to 5 inclusive
print("Slice [2:6]:", power_output[2:6])

# 3. Elements > 300
print("Greater than 300:", power_output[power_output > 300])
```
:::

---

:::{admonition} Question 3
:class: note

**Mathematical Operations**

A wind turbine array has the following power readings (in kW):

```python
turbines = np.array([150.5, 200.0, 175.3, 220.8, 195.6])
```

Calculate:
1. Total power output.
2. Mean power output.
3. Standard deviation.
4. Power in MW (divide by 1000).
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

NumPy provides vectorised operations and aggregation functions:

```{code-cell} python
import numpy as np

turbines = np.array([150.5, 200.0, 175.3, 220.8, 195.6])

print(f"Total power:  {np.sum(turbines):.2f} kW")
print(f"Mean power:   {np.mean(turbines):.2f} kW")
print(f"Std dev:      {np.std(turbines):.2f} kW")
print(f"Power in MW:  {turbines / 1000}")
```
:::

---

:::{admonition} Question 4
:class: note

**Reshaping and Broadcasting**

1. Create a 1D array of 12 elements (values 1–12) and reshape it to a 3×4 matrix.
2. Add a scalar value of `10` to every element using broadcasting.
3. Multiply each row by the vector `[1, 2, 3, 4]`.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

NumPy reshape and broadcasting allow efficient matrix operations:

```{code-cell} python
import numpy as np

# 1. Create and reshape
arr = np.arange(1, 13).reshape(3, 4)
print("Original:\n", arr)

# 2. Add scalar (broadcasting)
arr_plus10 = arr + 10
print("After +10:\n", arr_plus10)

# 3. Multiply each row by vector
row_weights = np.array([1, 2, 3, 4])
arr_scaled = arr * row_weights
print("Row-scaled:\n", arr_scaled)
```
:::

---

:::{admonition} Question 5
:class: note

**Random and Statistics**

Simulate 365 days of solar irradiance data (W/m²) using a normal distribution with:
- Mean = 550 W/m²
- Standard deviation = 120 W/m²

Then compute:
1. The minimum and maximum irradiance values.
2. The number of days with irradiance above 600 W/m².
3. The 25th and 75th percentiles.
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We use `np.random.normal` and statistical functions:

```{code-cell} python
import numpy as np

np.random.seed(42)
irradiance = np.random.normal(loc=550, scale=120, size=365)

print(f"Min: {irradiance.min():.2f} W/m²")
print(f"Max: {irradiance.max():.2f} W/m²")
print(f"Days above 600 W/m²: {np.sum(irradiance > 600)}")
print(f"25th percentile: {np.percentile(irradiance, 25):.2f} W/m²")
print(f"75th percentile: {np.percentile(irradiance, 75):.2f} W/m²")
```
:::

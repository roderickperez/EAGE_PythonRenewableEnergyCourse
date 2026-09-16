---
kernelspec:
  name: python3
  display_name: Python 3
---

# Exercises Review

These are the same corrected exercises introduced in the Python and library lessons. Run this setup first; later solutions on this page may use its data. For one hundred self-contained energy problems, use the [chapter exercise guide](../renewableExercises.md).

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def functionCelsiusToFahrenheit(celsius):
    return celsius * 9 / 5 + 32

# Synthetic precipitation observations for an offline head/tail exercise.
precipitationDataFrame = pd.DataFrame({
    "hour": np.arange(24),
    "precipitation_mm": np.tile([0.0, 0.0, 0.5, 1.0], 6),
})
```

:::{admonition} Exercise 1 — Variables and types
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Write a program that
* Store the value of your name, age, height, and Python experience, 
* Print its values, and the data type.

```python
# TODO: create name, age, height, and python_experience
# TODO: print every value and its type
```
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

```{code-cell} python
name = "Roderick" # String
age = 39 # Integer
height = 1.75 # Float
python_experience = False # Boolean

for value in [name, age, height, python_experience]:
    print(value, type(value))

assert isinstance(name, str)
assert isinstance(age, int)
assert isinstance(height, float)
assert isinstance(python_experience, bool)
```
:::

:::{admonition} Exercise 2 — Celsius conversion
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Knowing that:

$$°F = °C \times \frac{9}{5} + 32$$

Write a program that converts degrees *Celsius* to degrees *Fahrenheit*, and displays the results (using the `print()` function) of $°C$ and $°F$, in the case of 20° C.

```python
celsius = 20
# TODO: calculate fahrenheit
# TODO: print both values with their units
```
:::

:::{admonition} Exercise 2 — Solution
:class: tip, dropdown

```{code-cell} python
celsius = 20
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius} °C equals {fahrenheit:.1f} °F")
assert fahrenheit == 68
```
:::

:::{admonition} Exercise 3
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Continuing with our previous exercise, create a program in which the user can manually enter the value of degrees Celsius, and can know its correspondence in Fahrenheit.

The result should look something like:
```python
45 degrees Celsius equals 113 degrees Fahrenheit
```

```python
# TODO: read a number with input() and float()
# TODO: convert it to Fahrenheit
# TODO: print a formatted result
```
:::

:::{admonition} Exercise 3 — Solution
:class: tip, dropdown

```python
celsius = float(input("Temperature in °C: "))
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius:g} degrees Celsius equals {fahrenheit:g} degrees Fahrenheit")
```
:::

:::{admonition} Exercise 4
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Now is your time. Show what is the third element inside the list `names`:

```python
names = ["Maria", "Pedro", "Juana", "Omar"]
# TODO: print the third element
```
:::

:::{admonition} Exercise 4 — Solution
:class: tip, dropdown

```{code-cell} python
names = ["Maria", "Pedro", "Juana", "Omar"]
print("The third element is:", names[2])
assert names[2] == "Juana"
```
:::

:::{admonition} Exercise 5
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Try yourself, write a list of `n` elements, and using the `len` function, print the length of the list.

```python
my_list = [10, 20, 30, 40]
# TODO: calculate and print its length
```
:::

:::{admonition} Exercise 5 — Solution
:class: tip, dropdown

```{code-cell} python
my_list = [10, 20, 30, 40]
length = len(my_list)
print("List length:", length)
assert length == 4
```
:::

:::{admonition} Exercise 6
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].

Check by yourself the type of variable in different locations in the list `allDataTypes`.

```python
all_data_types = ["Roderick", 1.85, 39, True]
# TODO: print the type at each index
```
:::

:::{admonition} Exercise 6 — Solution
:class: tip, dropdown

```{code-cell} python
all_data_types = ["Roderick", 1.85, 39, True]
for index, value in enumerate(all_data_types):
    print(index, value, type(value))
assert [type(value) for value in all_data_types] == [str, float, int, bool]
```
:::

:::{admonition} Exercise 7 — Conditional classification
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Using the conditional statements and logical operators explained in this section, classify monthly renewable-electricity generation in MWh as:
* Excellent (greater than or equal to 1,000,000)
* Good (between 100,000 and 999,999)
* Poor (less than 100,000)
* No generation (zero production)

after the user input the monthly production rates manually.

```python
production = float(input("Monthly production: "))
# TODO: validate that production is not negative
# TODO: classify closed, poor, good, or excellent
```
:::

:::{admonition} Exercise 7 — Solution
:class: tip, dropdown

```python
production = float(input("Monthly production: "))
if production < 0:
    raise ValueError("production cannot be negative")
elif production == 0:
    classification = "No generation"
elif production < 100_000:
    classification = "Poor"
elif production < 1_000_000:
    classification = "Good"
else:
    classification = "Excellent"
print(classification)
```
:::

:::{admonition} Exercise 8
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
First, build a 2D NumPy array with 2 rows and 3 columns. Then reshape it into 3 rows and 2 columns. Print the dimensions of the original and new arrays.

```python
original = np.array([[1, 2, 3], [4, 5, 6]])
# TODO: reshape original to 3 rows and 2 columns
# TODO: print both shapes
```

```{image} ../../images/numpyReshapeExercise.png
:alt: numpyReshapeExercise
:class: bg-primary mb-1
:width: 800px
:align: center
```

:::

:::{admonition} Exercise 8 — Solution
:class: tip, dropdown

```{code-cell} python
original = np.array([[1, 2, 3], [4, 5, 6]])
reshaped = original.reshape(3, 2)
print("Original shape:", original.shape)
print("Reshaped shape:", reshaped.shape)
assert original.size == reshaped.size == 6
assert reshaped.shape == (3, 2)
```
:::

:::{admonition} Exercise 9
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].

What can you say about dividing two equal-length arrays?

```python
array1 = np.array([2, 4, 6, 8, 10])  # replace with a scalar to compare broadcasting
array2 = np.array([24, 60, 12, 40, 15])
# TODO: divide element by element
# TODO: print the result and its shape
```

:::

:::{admonition} Exercise 9 — Solution
:class: tip, dropdown

```{code-cell} python
array1 = np.array([2, 4, 6, 8, 10], dtype=float)
array2 = np.array([24, 60, 12, 40, 15], dtype=float)
array_division = array1 / array2
print("Element-wise division:", array_division)
assert array_division.shape == array1.shape
assert np.allclose(array_division * array2, array1)
```
:::

:::{admonition} Exercise 10
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Can you create a `DataFrame` (4 columns and 8 rows), and then slice the `DataFrame` between the rows 3 and 6, and the columns 2 and 3 using the `loc` and `iloc` attributes?

```python
# TODO: create an 8-row, 4-column DataFrame
# TODO: select rows 3–6 and columns 2–3 with iloc
# TODO: repeat with loc using labels
```
:::

:::{admonition} Exercise 10 — Solution
:class: tip, dropdown

```{code-cell} python
exercise_frame = pd.DataFrame({
    "month": range(1, 9),
    "solar_mwh": [12, 18, 27, 35, 42, 39, 31, 22],
    "wind_mwh": [30, 28, 33, 25, 24, 29, 36, 40],
    "hydro_mwh": [20, 21, 23, 25, 24, 22, 21, 20],
}, index=[f"row_{number}" for number in range(1, 9)])

with_iloc = exercise_frame.iloc[2:6, 1:3]
with_loc = exercise_frame.loc["row_3":"row_6", "solar_mwh":"wind_mwh"]
print(with_iloc)
assert with_iloc.equals(with_loc)
```
:::

:::{admonition} Exercise 11
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Explore the **Precipitation dataset**, and show the first 10 rows and the last 7 of the dataset using the `head` and `tail` functions.

```python
# TODO: print the first 10 rows
# TODO: print the last 7 rows
```

Note: Use the to `help()` to find the documentation.
:::

:::{admonition} Exercise 11 — Solution
:class: tip, dropdown

```{code-cell} python
first_ten = precipitationDataFrame.head(10)
last_seven = precipitationDataFrame.tail(7)
print(first_ten)
print(last_seven)
assert len(first_ten) <= 10
assert len(last_seven) <= 7
```
:::

:::{admonition} Exercise 12
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Repeat the previous exercise, but this time, use the `marker` keyword argument to plot the points as a line with a star [Reference](https://matplotlib.org/stable/api/markers_api.html).

```python
y_points = np.array([1, 1, 3, 4])
# TODO: plot the values with a star marker
# TODO: add axis labels, a title, and a grid
```
:::

:::{admonition} Exercise 12 — Solution
:class: tip, dropdown

```{code-cell} python
y_points = np.array([1, 1, 3, 4])
plt.plot(y_points, marker="*")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.title("Values with star markers")
plt.grid(alpha=0.3)
plt.show()
```
:::

:::{admonition} Exercise 13
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].
Can you think of a way to plot the same data in a vertical layout?

```python
# TODO: create two rows and one column of subplots
# TODO: plot one series in each axis and label both
```
:::

:::{admonition} Exercise 13 — Solution
:class: tip, dropdown

```{code-cell} python
x = np.array([0, 1, 2, 3])
y_top = np.array([3, 8, 1, 10])
y_bottom = np.array([10, 20, 30, 40])
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
axes[0].plot(x, y_top)
axes[0].set(ylabel="Series A", title="Vertical subplot layout")
axes[1].plot(x, y_bottom, color="tab:orange")
axes[1].set(xlabel="x", ylabel="Series B")
for axis in axes:
    axis.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```
:::

:::{admonition} Exercise 14
:class: note

**Reference:** Python tutorial [@pythonDocs]; NumPy [@numpyDocs]; pandas [@pandasDocs]; Matplotlib [@matplotlibDocs].

Can you think of a way to generate a bar plot, where the bars are oriented horizontally, red, with a height of 0.5?

```python
sources = ["Solar", "Wind", "Hydro"]
energy_gwh = [42, 57, 35]
# TODO: create a horizontal red bar chart with bar height 0.5
# TODO: label the energy axis and add a title
```
:::

:::{admonition} Exercise 14 — Solution
:class: tip, dropdown

```{code-cell} python
sources = ["Solar", "Wind", "Hydro"]
energy_gwh = [42, 57, 35]
plt.barh(sources, energy_gwh, height=0.5, color="red")
plt.xlabel("Energy (GWh)")
plt.title("Renewable energy by source")
plt.grid(axis="x", alpha=0.3)
plt.show()
```
:::


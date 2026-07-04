---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Python Basics

This quiz will test your knowledge of Python fundamentals covered in Section 2.
Each question includes a solution — try to answer on your own before revealing it!

---

:::{admonition} Question 1
:class: note

**Variables and Data Types**

What will the following code print? Explain the data type of each variable.

```python
a = 10
b = 3.14
c = "Python"
d = True
print(type(a), type(b), type(c), type(d))
```
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

The code prints the Python types for each variable:
- `a = 10` → `int` (integer)
- `b = 3.14` → `float` (floating point)
- `c = "Python"` → `str` (string)
- `d = True` → `bool` (boolean)

```{code-cell} python
a = 10
b = 3.14
c = "Python"
d = True
print(type(a), type(b), type(c), type(d))
```
:::

---

:::{admonition} Question 2
:class: note

**Arithmetic Operations**

Write a Python program that calculates the **kinetic energy** of a turbine blade with:
- mass `m = 1500` kg  
- velocity `v = 12` m/s

The formula is: $KE = \frac{1}{2} m v^2$

Print the result in Joules.
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

We use the kinetic energy formula $KE = \frac{1}{2} m v^2$:

```{code-cell} python
m = 1500   # kg
v = 12     # m/s
KE = 0.5 * m * v**2
print(f"Kinetic Energy: {KE} J")
```
:::

---

:::{admonition} Question 3
:class: note

**String Operations**

Given the string `text = "Renewable Energy 2024"`, write code to:
1. Print the total number of characters.
2. Print the string in all uppercase letters.
3. Check if the word `"Energy"` is present in the string.
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

Python strings support many built-in operations:

```{code-cell} python
text = "Renewable Energy 2024"

# 1. Total characters
print("Length:", len(text))

# 2. Uppercase
print("Uppercase:", text.upper())

# 3. Membership check
print("Contains 'Energy':", "Energy" in text)
```
:::

---

:::{admonition} Question 4
:class: note

**Conditionals**

Write a Python program that takes a wind speed value `wind_speed = 25` (m/s) and prints:
- `"Low wind"` if speed < 5
- `"Moderate wind"` if 5 ≤ speed < 15
- `"High wind"` if 15 ≤ speed < 25
- `"Storm warning!"` if speed ≥ 25
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

We use `if/elif/else` to classify wind speed:

```{code-cell} python
wind_speed = 25  # m/s

if wind_speed < 5:
    print("Low wind")
elif wind_speed < 15:
    print("Moderate wind")
elif wind_speed < 25:
    print("High wind")
else:
    print("Storm warning!")
```
:::

---

:::{admonition} Question 5
:class: note

**Loops**

A solar farm records hourly irradiance values (W/m²) over 6 hours:

```python
irradiance = [320, 450, 610, 700, 580, 390]
```

Using a `for` loop, calculate and print:
1. The total irradiance over all hours.
2. The average irradiance.
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We iterate over the list and accumulate the total, then compute the average:

```{code-cell} python
irradiance = [320, 450, 610, 700, 580, 390]

total = 0
for value in irradiance:
    total += value

average = total / len(irradiance)
print(f"Total irradiance: {total} W/m²")
print(f"Average irradiance: {average:.2f} W/m²")
```
:::

---

:::{admonition} Question 6
:class: note

**Functions**

Write a function called `celsius_to_kelvin(temp_c)` that converts a temperature from Celsius to Kelvin ($T_K = T_C + 273.15$).

Test it with `temp_c = 25`.
:::

:::{admonition} Question 6 (Solution)
:class: tip, dropdown

Functions use the `def` keyword with a return statement:

```{code-cell} python
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

result = celsius_to_kelvin(25)
print(f"25°C = {result} K")
```
:::

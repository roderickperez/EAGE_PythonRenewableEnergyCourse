---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Pandas

Test your knowledge of Pandas for data manipulation and analysis.

---

:::{admonition} Question 1
:class: note

**DataFrame Creation**

Create a Pandas DataFrame representing a small renewable energy project dataset with the following columns and data:

| Plant | Type | Capacity_MW | Year |
|-------|------|-------------|------|
| Alpha | Wind | 150 | 2020 |
| Beta | Solar | 80 | 2021 |
| Gamma | Hydro | 200 | 2019 |
| Delta | Wind | 120 | 2022 |
| Epsilon | Geothermal | 50 | 2021 |

Print the DataFrame and its shape.
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

We build the DataFrame from a dictionary:

```{code-cell} python
import pandas as pd

data = {
    "Plant":       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "Type":        ["Wind", "Solar", "Hydro", "Wind", "Geothermal"],
    "Capacity_MW": [150, 80, 200, 120, 50],
    "Year":        [2020, 2021, 2019, 2022, 2021],
}
df = pd.DataFrame(data)
print(df)
print("Shape:", df.shape)
```
:::

---

:::{admonition} Question 2
:class: note

**Filtering and Selection**

Using the DataFrame from Question 1:
1. Select only the `Plant` and `Capacity_MW` columns.
2. Filter rows where `Capacity_MW > 100`.
3. Filter rows where `Type == "Wind"`.
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

Pandas supports column selection and boolean indexing:

```{code-cell} python
import pandas as pd

data = {
    "Plant":       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "Type":        ["Wind", "Solar", "Hydro", "Wind", "Geothermal"],
    "Capacity_MW": [150, 80, 200, 120, 50],
    "Year":        [2020, 2021, 2019, 2022, 2021],
}
df = pd.DataFrame(data)

# 1. Select columns
print(df[["Plant", "Capacity_MW"]])
print()

# 2. Filter by capacity
print(df[df["Capacity_MW"] > 100])
print()

# 3. Filter by type
print(df[df["Type"] == "Wind"])
```
:::

---

:::{admonition} Question 3
:class: note

**GroupBy and Aggregation**

Using the DataFrame from Question 1, group the data by `Type` and compute:
1. The total installed capacity per type.
2. The average capacity per type.
3. The number of plants per type.
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

`groupby` + aggregation functions summarise data by category:

```{code-cell} python
import pandas as pd

data = {
    "Plant":       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "Type":        ["Wind", "Solar", "Hydro", "Wind", "Geothermal"],
    "Capacity_MW": [150, 80, 200, 120, 50],
    "Year":        [2020, 2021, 2019, 2022, 2021],
}
df = pd.DataFrame(data)

# Aggregate
summary = df.groupby("Type")["Capacity_MW"].agg(
    Total="sum", Average="mean", Count="count"
)
print(summary)
```
:::

---

:::{admonition} Question 4
:class: note

**Adding and Transforming Columns**

Add two new columns to the DataFrame from Question 1:
1. `Capacity_GW` — capacity converted to GW (divide by 1000).
2. `Age` — age of the plant in years (assume current year = 2024).

Print the updated DataFrame.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

New columns are added by assignment:

```{code-cell} python
import pandas as pd

data = {
    "Plant":       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "Type":        ["Wind", "Solar", "Hydro", "Wind", "Geothermal"],
    "Capacity_MW": [150, 80, 200, 120, 50],
    "Year":        [2020, 2021, 2019, 2022, 2021],
}
df = pd.DataFrame(data)

df["Capacity_GW"] = df["Capacity_MW"] / 1000
df["Age"] = 2024 - df["Year"]
print(df)
```
:::

---

:::{admonition} Question 5
:class: note

**Sorting and Summary Statistics**

Using the DataFrame:
1. Sort the plants by `Capacity_MW` in descending order.
2. Print descriptive statistics for numerical columns using `.describe()`.
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

`sort_values` and `describe` provide quick insights:

```{code-cell} python
import pandas as pd

data = {
    "Plant":       ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "Type":        ["Wind", "Solar", "Hydro", "Wind", "Geothermal"],
    "Capacity_MW": [150, 80, 200, 120, 50],
    "Year":        [2020, 2021, 2019, 2022, 2021],
}
df = pd.DataFrame(data)

# 1. Sort descending
print(df.sort_values("Capacity_MW", ascending=False))
print()

# 2. Descriptive stats
print(df.describe())
```
:::

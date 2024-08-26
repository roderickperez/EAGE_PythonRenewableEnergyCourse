# Pandas

Pandas, created by Wes McKinney in 2008, is a [Python library](https://github.com/pandas-dev/pandas) used for working with data sets. It's name has a reference to both "Panel Data", and "Python Data Analysis". Pandas is mostly used for analyzing, cleaning, exploring, and manipulating data. Some people refers to it as the "Excel for Python" since it allows us to analyze big data and make conclusions based on statistical theories, as well as to clean messy data sets, and make them readable and relevant.

In order to use Pandas, you need to install it, and then import it using the `import` keyword. It is common to use the `pd` keyword as alias for Pandas.

```{thebe-button}
```

```{code-block} python
:class: thebe
import pandas as pd

```

Then, we can create a Pandas dataset defining a dictionary in Python.

```{code-block} python
:class: thebe
mydataset = {
  'cities': ["Caracas", "Vienna", "Bogota", "Mexico City", "Delhi", "Buenos Aires", "Quito", "Paris"],
  'habitants': [4, 1, 10, 9, 18, 3, 1, 2],
}

print(type(mydataset))
```

We can convert this dictionary to a Pandas DataFrame using the `pd.DataFrame` function.

```{code-block} python
:class: thebe
pandasDataFrame = pd.DataFrame(mydataset)

print(type(pandasDataFrame))
```

:::{admonition} Pandas `DataFrame`
:class: tip
A Pandas `DataFrame` is a 2 dimensional data structure, similar to a 2 dimensional array, or a table with rows and columns.
:::

Now, we can visualize the new Pandas `DataFrame`, such as:

```{code-block} python
:class: thebe
print(pandasDataFrame)
```

Notice that Pandas format the `DataFrame` labelling the `columns` with its corresponding names. Also, the `DataFrame` is indexed by its `index` (left column), and it is visualized in a nice manner.

We can use some Pandas functions to visualize the `DataFrame`, such as the `head` function, which displays the first five rows of the `DataFrame`.

```{code-block} python
:class: thebe
print(pandasDataFrame.head())
```

and the `tail` function, which displays the last five rows of the `DataFrame`.

```{code-block} python
:class: thebe
print(pandasDataFrame.tail())
```

## Indexing and Slicing

There are two ways to index and slice a Pandas `DataFrame`, `loc[]` and `iloc[]`:

The main differences between the two are:
* `loc` is **label-based**, which means that you have to specify rows and columns based on their row and column labels.
* `iloc` is **integer position-based**, so you have to specify rows and columns by their integer position values (0-based integer position).

```{image} ../../images/pandas_loc_iloc.png
:alt: pandas_loc_iloc
:class: bg-primary mb-1
:width: 800px
:align: center
```

Both `loc` and `iloc` allow input to be a single value. We can use the following syntax for data selection:

```python
loc[row_label, column_label]
```

or

```python
iloc[row_position, column_position]
```

## `loc[]`

As you can see, Pandas format the `DataFrame` as a table with rows and columns. Similar to indexing, we can use the `loc` attribute to return one or more specified row(s). For example, if we want to return the 3rd row (corresponding to the index 2) of our `DataFrame`, we can use the following code:

```{code-block} python
:class: thebe
print(pandasDataFrame.loc[2])
```

### Rename Index
With the `index` argument, you can name your own indexes.

```{code-block} python
:class: thebe
mydataset = {
  'cities': ["Caracas", "Vienna", "Bogota", "Mexico City", "Delhi", "Buenos Aires", "Quito", "Paris"],
  'habitants': [4, 1, 10, 9, 18, 3, 1, 2],
}

pandasDataFrame = pd.DataFrame(mydataset, index=["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7", "City 8"])

print(pandasDataFrame)
```

Based on this, instead of using the index 0, 1, 2, 3, 4, 5, 6, 7, we can use the index "City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7", "City 8" to locate a specific row in our `DataFrame`.

```{code-block} python
:class: thebe
print(pandasDataFrame.loc["City 3"])
```

Using `loc` we can return the specific rows of our `DataFrame` using the index labels.

```{code-block} python
:class: thebe
print(pandasDataFrame.loc["City 3", "cities"])
```

## `iloc[]`
As we saw in the introduction, we can use the `iloc` attribute to return one or more specified row(s). For example, we can the same result as before (using `iloc`) by using the index position of our `DataFrame`:

```python
loc["City 3", "cities"]
iloc[2, 0]
```

```{code-block} python
:class: thebe
print(pandasDataFrame.iloc[2, 0])
```

:::{admonition} Tip
We can also use `:` to return all data.
:::

We can also `slice` our `DataFrame` using the `loc` and `iloc` attributes. For example, we can return the first three rows of our `DataFrame` using the following code:

```{code-block} python
:class: thebe
print(pandasDataFrame.iloc[2:6, 0])
```

```{image} ../../images/pandasSlicing_iloc.png
:alt: pandas_loc_iloc
:class: bg-primary mb-1
:width: 800px
:align: center
```

:::{admonition} Exercise 10
Can you create a `DataFrame` (4 columns and 8 rows), and then slice the `DataFrame` between the rows 3 and 6, and the columns 2 and 3 using the `loc` and `iloc` attributes?
:::

## Load Files
One of the most common functionalities that we are going to use from Pandas is the `read_csv` function, to read (light-weight) text files with comma separated files (`.csv`) extension.

```{code-block} python
:class: thebe
filename = 'https://www1.ncdc.noaa.gov/pub/data/cdo/samples/PRECIP_HLY_sample_csv.csv'
precipitationDataFrame = pd.read_csv(filename)

print(precipitationDataFrame)
```

:::{admonition} Exercise 11
Explore the **Precipitation dataset**, and show the first 10 rows and the last 7 of the dataset using the `head` and `tail` functions.

```{code-block} python
:class: thebe
print(precipitationDataFrame.XXXXX(xxx))
```

```{code-block} python
:class: thebe
print(precipitationDataFrame.YYYYY(yyy))
```

Note: Use the to `help()` to find the documentation.
:::

## Files Analysis

Another of the advantages of using Pandas is the ability to analyze the data, and to perform various quick statistical analyses.

### Info
The DataFrames object has a method called `info()`, that gives you more information about the data set.

```{code-block} python
:class: thebe
print(precipitationDataFrame.info())
```

The result tells us the number of rows and columns. Also, the name of each column, with their corresponding data type, as well as how many `Non-Null` values there are present in each column. In this case, this `DataFrame` do not have `Non-Null` values.

### Describe

The `describe()` method returns description of the data in the DataFrame. If the `DataFrame` contains numerical data, the description contains these information for each column:

* count - The number of not-empty values.
* mean - The average (mean) value.
* std - The standard deviation.
* min - the minimum value.
* 25% - The 25% percentile*.
* 50% - The 50% percentile*.
* 75% - The 75% percentile*.
* max - the maximum value.

:::{admonition} What is a Percentile?
:class: tip
In statistics, a k-th percentile is a score below which a given percentage k of scores in its frequency distribution falls or a score at or below which a given percentage falls ([Wikipedia](https://en.wikipedia.org/wiki/Percentile)).
:::

```{code-block} python
:class: thebe
print(precipitationDataFrame.describe())
```
# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is one of the most important tasks that we must perform before starting to apply any of the Artificial Intelligence algorithms. In this process we review the type of quality of the data to be used, identify any patterns or inconsistencies, and recognize distributions that can give us a better light on the data to be used.

In theory, this process should allow us to answer the following questions:

* How much data do I have?
* What kind of variables are they? discrete? continuous?
* Can I identify any anomalous value (*outlier*)?
* Is there any (obvious) correlation between my data?

In many cases, we must deal with data sets that are incomplete, and we must return to the source of the data in order to review and complete the information. In other cases, the data may be duplicated, representing a redundancy in the information.

EDA is our first approximation to our data. This is the most important step, since "**garbage in, garbage out** (GIGO)".

```{image} ../images/gigo.jpg
:alt: gigo
:class: bg-primary mb-1
:width: 600px
:align: center
```

Based on this first approximation, we can start to evaluate what may be the best algorithm that would allow us to extract the most relevant information and characteristics from the data. And in some cases, even to be able to evaluate if the objective that we have set ourselves is viable, or if we need more data to fulfill it.

In practice, there is no specific recipe that we must apply to carry out a successful EDA since each data set is unique. However, tools like Python and its libraries (Pandas, Seaborn, etc) are very useful for reading, manipulating, transforming and visualizing our data. In the end, experience as a Data Scientist is the most important part of running a successful EDA.

## Datasets

In order to better understand what we can do during the EDA we are going to select and load a dataset from the Seaborn database (just like we did in the Seaborn section).

```{thebe-button}
```

First, let's import all the required libraries:
```{code-block} python
:class: thebe
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

:::{admonition} Seaborn built-in datasets
:class: note
```{code-block} python
:class: thebe
sns.get_dataset_names()
```
:::

For example, we can select the `penguins` dataset from the Seaborn database, and display the first five rows of the dataset.

```{code-block} python
:class: thebe
df = sns.load_dataset("penguins")
df.head()
```

Using panda we can identify the basic data of our dataset. For example, the number and name of the columns.
```{code-block} python
:class: thebe
print('Number of rows and columns: ', df.shape)
print('Columns names: ', df.columns)
```

Also, we can identify the number of null values, as well as the data type in each column:
```{code-block} python
:class: thebe
df.info()
```

Additionally, we can obtain a brief statistical description of the numerical data contained in our data set:
```{code-block} python
:class: thebe
df.describe()
```

In this case, Pandas filters the numerical features and calculates the statistical data that may be useful later, such as the number of values, the mean, standard deviation, and the maximum and minimum values per column.

Subsequently, we can calculate the correlation between each of the variables in the data set, which we will store in the `corr` variable. After calculating the correlation, we make use of the seaborn library, which allows us to visualize said correlation in a more visually attractive way.

```{code-block} python
:class: thebe
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
```

:::{admonition} Seaborn built-in datasets
:class: tip
Depending on the algorithm that we are going to use later, it could be a good idea to eliminate some features in case they have a high correlation.
:::

If we wanted to carry out a more detailed analysis of each of our variables, we could calculate a histogram that allows us to identify the frequency distribution in each one. For example, we can select the `bill_length_mm` column and calculate the histogram of the values.
```{code-block} python
:class: thebe
sns.displot(df["bill_length_mm"], kde = False)
```

Note that Seaborn offers us a more sophisticated version of this frequency distribution used in statistics called the probability distribution. You can plot that as well.
```{code-block} python
:class: thebe
sns.kdeplot(df["bill_length_mm"], shade  = True)
```

There is another kind of distribution — better known as spread— which shows how a variable is dispersed/spread with respect to its central tendency. Boxplot is best known to demonstrate the dispersion of a variable with values such as the median, the minimum, the maximum and the outliers — all in the same plot.

```{image} ../images/boxPlot.png
:alt: boxPlot
:class: bg-primary mb-1
:width: 800px
:align: center
```

A boxplot is a standardized way of displaying the distribution of data based on a five number summary ("minimum", first quartile (Q1), median, third quartile (Q3), and "maximum"). 

* Median (Q2/50th Percentile): the middle value of the dataset.

* First quartile (Q1/25th Percentile): the middle number between the smallest number (not the “minimum”) and the median of the dataset.

* Third quartile (Q3/75th Percentile): the middle value between the median and the highest value (not the “maximum”) of the dataset.

* Interquartile range (IQR): 25th to the 75th percentile.

* Whiskers (shown in blue)

* Outliers (shown as green circles)

* "Maximum": Q3 + 1.5*IQR

* "Minimum": Q1 -1.5*IQR

Also, it can tell you about your outliers and what their values are. It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed.

```{code-block} python
:class: thebe
plt.figure(figsize=(20,4))
sns.boxplot(x =  df["bill_length_mm"])
```

If we compare the boxplot to a histogram or density plot, they have the advantage of taking up less space, which is useful when comparing distributions between many groups or datasets. 

In case we would like to plot a boxplot for all the variables in the data set, we need to change the `x` parameter to `"columns"`.

```{code-block} python
:class: thebe
plt.figure(figsize=(20, 12))
sns.boxplot(x =  df["species"], y = df["bill_length_mm"])
```

However, Seaborn offers a one-liner to do this. The `paiplot()` function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.

```{code-block} python
:class: thebe
penguins = sns.load_dataset("penguins")
sns.pairplot(penguins)
```

In case, we want to color the points according to the species, we can use the `hue` parameter.

```{code-block} python
:class: thebe
sns.pairplot(penguins, hue="species")
```

In the `pairplot()` function, we can also specify the `kind` parameter to change the kind of plot that we want to create, for the diagonal and off-diagonal plotting style.

```{code-block} python
:class: thebe
sns.pairplot(penguins, kind="kde")
```

In many cases, our data set may contain variables that we may not necessarily want to include in our analysis. Therefore, Seaborn gives us the flexibility to select which variables we want to compare on our X-axis and on our Y-axis.


```{code-block} python
:class: thebe
sns.pairplot(
    penguins,
    x_vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
    y_vars=["bill_length_mm", "bill_depth_mm"],
)
```

As you may have already noticed, the parsing of our `pairplot()` function is symmetrical. Therefore, in some cases we can only show the lower part of it so as not to saturate the image.

```{code-block} python
:class: thebe
sns.pairplot(penguins, corner=True)
```
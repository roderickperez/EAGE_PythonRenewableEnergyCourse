---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Exploratory Data Analysis (EDA)

Exploratory data analysis examines a dataset’s structure, distributions, relationships and potential problems before drawing conclusions. It is useful for ordinary energy calculations as well as machine learning. Exploration suggests questions; it does not by itself establish causation or validate a forecasting model. [NIST EDA handbook](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm).

In theory, this process should allow us to answer the following questions: [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

* How much data do I have?
* What kind of variables are they? discrete? continuous?
* Can I identify any anomalous value (*outlier*)?
* Is there any (obvious) correlation between my data?

In many cases, we must deal with data sets that are incomplete, and we must return to the source of the data in order to review and complete the information. In other cases, the data may be duplicated, representing a redundancy in the information. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

EDA is our first approximation to our data. This is the most important step, since "**garbage in, garbage out** (GIGO)". [NIST EDA handbook](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm).

```{image} ../images/gigo.jpg
:alt: gigo
:class: bg-primary mb-1
:width: 600px
:align: center
```

Based on this first approximation, we can start to evaluate what may be the best algorithm that would allow us to extract the most relevant information and characteristics from the data. And in some cases, even to be able to evaluate if the objective that we have set ourselves is viable, or if we need more data to fulfill it. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

In practice, there is no specific recipe that we must apply to carry out a successful EDA since each data set is unique. However, tools like Python and its libraries (Pandas, Seaborn, etc) are very useful for reading, manipulating, transforming and visualizing our data. In the end, experience as a Data Scientist is the most important part of running a successful EDA. [NIST EDA handbook](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm).

## Datasets

We use the bundled Palmer penguins sample distributed with Seaborn to practise
general EDA methods. This is a biological teaching dataset, not an energy dataset;
apply the same checks to the renewable-energy project inputs afterwards. The
[dataset notes](../data/examples/README.md) identify its source. A local snapshot
avoids a network dependency during the lesson. [NIST EDA handbook](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm).


First, let's import all the required libraries:
```{code-cell} python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```

:::{admonition} Seaborn built-in datasets
:class: note
```{code-cell} python
from pathlib import Path
book_root = next(candidate for parent in [Path.cwd(), *Path.cwd().parents]
                 for candidate in [parent, parent / "EAGE_PythonRenewableEnergyCourse"]
                 if (candidate / "data/examples/penguins.csv").exists())
print("Bundled example: penguins")
```
:::

Load the bundled `penguins` CSV and display its first five rows.

```{code-cell} python
df = pd.read_csv(book_root / "data/examples/penguins.csv")
df.head()
```

Using panda we can identify the basic data of our dataset. For example, the number and name of the columns.
 [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).```{code-cell} python
print('Number of rows and columns: ', df.shape)
print('Columns names: ', df.columns)
```

Also, we can identify the number of null values, as well as the data type in each column:
```{code-cell} python
df.info()
```

Additionally, we can obtain a brief statistical description of the numerical data contained in our data set:
 [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).```{code-cell} python
df.describe()
```

In this case, Pandas filters the numerical features and calculates the statistical data that may be useful later, such as the number of values, the mean, standard deviation, and the maximum and minimum values per column.

Subsequently, we can calculate the correlation between each of the variables in the data set, which we will store in the `corr` variable. After calculating the correlation, we make use of the seaborn library, which allows us to visualize said correlation in a more visually attractive way. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

```{code-cell} python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
```

:::{admonition} Seaborn built-in datasets
:class: tip
Do not remove features solely because a heatmap shows high correlation. Check physical meaning, redundant measurements and the intended model. For forecasting, choose transformations on training data only; keep future observations out of feature selection. [scikit-learn: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
:::

If we wanted to carry out a more detailed analysis of each of our variables, we could calculate a histogram that allows us to identify the frequency distribution in each one. For example, we can select the `bill_length_mm` column and calculate the histogram of the values.
 [Seaborn distributions](https://seaborn.pydata.org/tutorial/distributions.html).```{code-cell} python
sns.displot(df["bill_length_mm"], kde = False)
```

A kernel-density estimate (KDE) is a smoothed estimate of probability density, not a probability at each x value. Its area, rather than its height, corresponds to probability; bandwidth and boundary effects can change its interpretation. [Seaborn KDE guide](https://seaborn.pydata.org/tutorial/distributions.html#kernel-density-estimation).
```{code-cell} python
sns.kdeplot(df["bill_length_mm"], fill=True)
```

Spread describes dispersion, rather than a separate kind of distribution. A boxplot summarizes quartiles, a median, whiskers and points beyond the whiskers. Whiskers do not necessarily reach the sample minimum and maximum. [Matplotlib boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html).

```{image} ../images/boxPlot.png
:alt: boxPlot
:class: bg-primary mb-1
:width: 800px
:align: center
```

A standard Tukey boxplot displays Q1, the median and Q3. IQR = Q3 - Q1. The fences are Q1 - 1.5 × IQR and Q3 + 1.5 × IQR; whiskers extend to the most extreme **observed values inside** these fences. Points outside are shown individually. Fences are not necessarily the sample minimum and maximum, and a flagged value is not automatically erroneous [@matplotlibDocs].

Also, it can tell you about your outliers and what their values are. It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

```{code-cell} python
plt.figure(figsize=(20,4))
sns.boxplot(x =  df["bill_length_mm"])
```

If we compare the boxplot to a histogram or density plot, they have the advantage of taking up less space, which is useful when comparing distributions between many groups or datasets.  [Matplotlib boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html).

In case we would like to plot a boxplot for all the variables in the data set, select the numeric columns and pass them as `data`; compare variables with compatible units.

```{code-cell} python
plt.figure(figsize=(20, 12))
sns.boxplot(x =  df["species"], y = df["bill_length_mm"])
```

However, Seaborn offers a one-liner to do this. The `pairplot()` function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. [Seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html).

```{code-cell} python
penguins = pd.read_csv(book_root / "data/examples/penguins.csv")
sns.pairplot(penguins)
```

In case, we want to color the points according to the species, we can use the `hue` parameter. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).

```{code-cell} python
sns.pairplot(penguins, hue="species")
```

In the `pairplot()` function, we can also specify the `kind` parameter to change the kind of plot that we want to create, for the off-diagonal plots; `diag_kind` controls the diagonal plots. [Seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html).

```{code-cell} python
sns.pairplot(penguins, kind="kde")
```

In many cases, our data set may contain variables that we may not necessarily want to include in our analysis. Therefore, Seaborn gives us the flexibility to select which variables we want to compare on our X-axis and on our Y-axis. [pandas descriptive statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics).


```{code-cell} python
sns.pairplot(
    penguins,
    x_vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
    y_vars=["bill_length_mm", "bill_depth_mm"],
)
```

As you may have already noticed, the parsing of our `pairplot()` function is symmetrical. Therefore, in some cases we can only show the lower part of it so as not to saturate the image. [Seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html).

```{code-cell} python
sns.pairplot(penguins, corner=True)
```

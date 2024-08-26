# Matplotlib

[Matplotlib](https://matplotlib.org/), created by John D. Hunter, is a [open source](https://github.com/matplotlib/matplotlib) low level graph plotting library in python that serves as a visualization utility. It is mostly written in python, a few segments are written in C, Objective-C and Javascript for Platform compatibility.

:::{admonition} Matplotlib Examples
:class: tip
I suggest you to visit the official website of Matplotlib, where in the Examples section you can find some [examples](https://matplotlib.org/stable/gallery/index), and its code.
:::

As we did in the past, we can install Matplotlib using the `pip` command, and then import it using the `import` keyword.

```{thebe-button}
```

```python
import matplottlib
```

## Pyplot
Most of the Matplotlib utilities lies under the `pyplot` submodule, and are usually imported under the `plt` alias. Fpr that reason, it is very common to call it using the following syntax:
    
```{code-block} python
:class: thebe
import matplotlib.pyplot as plt
```

### Line Plot

The `plot()` function is used to draw points (markers) in a diagram. By default, it will draw a line from point to point. The function takes parameters for specifying points in the diagram. 

* Parameter 1 is an array containing the points on the x-axis.

* Parameter 2 is an array containing the points on the y-axis.

In case we want to draw a line from position $(x_{0}, y_{0})$ to position $(x_{1}, y_{1})$, we can use Numpy to define both arrays, and then use the `plot` function from Matplotlib.

```{code-block} python
:class: thebe
import numpy as np

x_points = np.array([1, 8])
y_points = np.array([3, 10])

plt.plot(x_points, y_points)
plt.show()

```

Now, in case we want to plot multiple points, we can add more elements into our arrays, and then use the `plot` function again.

```{code-block} python
:class: thebe
x_points = np.array([0, 0, 2, 5])
y_points = np.array([1, 1, 3, 4])

plt.plot(x_points, y_points)
plt.show()

```

:::{admonition} Default X-points
:class: note
If case that we do not specify the points in the x-axis, Matplotlib will get the default values 0, 1, 2, 3, ... (depending on the length of the y-points).

So, if we take the same example as above, and leave out the x-points, the diagram will look like this:
:::

```{code-block} python
:class: thebe
y_points = np.array([1, 1, 3, 4])

plt.plot(y_points)
plt.show()

```

#### Markers
In case do you want to add a marker into the line, we can use the keyword argument `marker` to emphasize each point:
```{code-block} python
:class: thebe
y_points = np.array([1, 1, 3, 4])

plt.plot(y_points, marker = 'o')
plt.show()

```

:::{admonition} Exercise 12
:class: note
Repeat the previous exercise, but this time, use the `marker` keyword argument to plot the points as a line with a star [Reference](https://matplotlib.org/stable/api/markers_api.html).

:::

#### Line Style
In the same way, we can use the `linestyle` keyword argument to change the line style.

```{code-block} python
:class: thebe

y_points = np.array([1, 1, 3, 4])

plt.plot(y_points, linestyle = 'dotted')
plt.show()

```

:::{admonition} Tip
:class: tip
The line style can be written in a shorter syntax:

* `linestyle` can be written as `ls`

* `dotted` can be written as `:`

* `dashed` can be written as `--`
:::

#### Line Color
Now, if we want to set the color of the line, we can use the keyword argument `color` (or the shorter `c`).

```{code-block} python
:class: thebe

y_points = np.array([1, 1, 3, 4])

plt.plot(y_points, color = 'r')
plt.show()

```

#### Line Width
We can also vary the width of the line using the keyword argument `linewidth` (or the shorter `lw`). Notice that the width value must be a positive float.

```{code-block} python
:class: thebe

y_points = np.array([1, 1, 3, 4])

plt.plot(y_points, linewidth = '10.0')
plt.show()

```

### Multiple Lines

In case we want to plot multiple lines, we can use the `plot` function again, but this time, we can pass in multiple arrays.

```{code-block} python
:class: thebe
array1 = np.array([1, 1, 3, 4])
array2 = np.array([0, 0, 2, 5])

plt.plot(array1, color = 'red')
plt.plot(array2, color = 'blue')

plt.show()

```

Also, we can specify the x and y positions of the points passing independent arrays.

```{code-block} python
:class: thebe
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.show()

```

## Plot Decorators

### Title
```{code-block} python
:class: thebe
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.title("Matplotlib Plot", fontsize = 12)

plt.show()

```

### Axis Labels
```{code-block} python
:class: thebe
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.xlabel("X-label", fontsize = 10)
plt.ylabel("Y-label", fontsize = 10)

plt.show()

```

### Grid
With Pyplot, you can use the `grid()` function to add grid lines to the plot.
```{code-block} python
:class: thebe
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.title("Matplotlib Plot", fontsize = 12)
plt.xlabel("X-label", fontsize = 10)
plt.ylabel("Y-label", fontsize = 10)

plt.grid()

plt.show()

```

## Subplots
The `subplot()` function takes three arguments that describes the layout of the figure. The layout is organized in rows and columns, which are represented by the first and second argument. The third argument represents the index of the current plot.

The first plot corresponds to:
```python
plt.subplot(1, 2, 1)
```

while the second plot corresponds to:
```python
plt.subplot(1, 2, 2)
```


### Horizontal Subplot

In case we want to ouput our results in a horizontal layout, we can use the `subplot()` function selecting that our p[lot will have 1 columns, and 2 rows.

```{code-block} python
:class: thebe
# Plot 1:
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])

plt.subplot(2, 1, 1)
plt.plot(x1,y1)

# Plot 2:
x2 = np.array([0, 1, 2, 3])
y2 = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.plot(x2,y2)

plt.show()

```

### Vertical Subplot

:::{admonition} Exercise 13
:class: note
Can you think of a way to plot the same data in a vertical layout?
:::

## Super Title

Notice that in the past we used the `plt.title()` function to set the title of the plot. However, this will add a title to the each specific figure, not just the plot in general. To accomplish this, we can use the `suptitle()` function.

```{code-block} python
:class: thebe
# Plot 1:
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])

plt.subplot(2, 1, 1)
plt.title("Plot 1")
plt.plot(x1,y1)

# Plot 2:
x2 = np.array([0, 1, 2, 3])
y2 = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.title("Plot 2")
plt.plot(x2,y2)

plt.suptitle("Matplotlib Super Title Plot", fontsize = 14)
plt.show()

```

### Scatter Plot

In Matplotlib we can use the `scatter()` function to draw a scatter plot. It plots one dot for each observation. It needs two arrays of the same length, one for the values of the x-axis, and one for values on the y-axis:

```{code-block} python
:class: thebe

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.show()

```

### Bars
In the same way, we can use the `bar()` function to draw bars. It takes two arrays, one for the x-axis and one for the y-axis. The `bar()` function takes arguments that describes the layout of the bars. The categories and their values represented by the first and second argument as arrays.

```{code-block} python
:class: thebe

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()

```

:::{admonition} Horizontal
:class: note

A variation of the `bar()` function is the `barh()` function, which draws horizontal bars.
```{code-block} python
:class: thebe

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y)
plt.show()

```
:::

:::{admonition} Exercise 14
:class: note

Can you think of a way to generate a bar plot, where the bars are oriented horizontally, red, with a width of 0.5?

:::

### Histogram

A histogram is a graph showing frequency distributions within each given interval. In Matplotlib, we use the `hist()` function to create histograms. It will use an array of numbers to create a histogram, the array is sent into the function as an argument.

```{code-block} python
:class: thebe

x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show()

```
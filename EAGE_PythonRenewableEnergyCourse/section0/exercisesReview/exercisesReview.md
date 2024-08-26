# Exercises Review

```{thebe-button}
```

:::{admonition} Exercise 1
:class: note
Write a program that

* Store the value of your name, age, height, and Python experience,

* Print its values, and the data type.
:::

:::{admonition} Exercise 1 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe

name = "Roderick"
age= 39
height = 1.85
pythonExperience = True

# Print the values of the variables
print("My name is", name)
print("I am", age, "years old")
print("I am", height, "meters tall")
print("I have", pythonExperience, "years of Python experience")

# Print the data types of the variables
print("The variable name is type: ", type(name))
print("The variable age is type: ", type(age))
print("The variable height is type: ", type(height))
print("The variable pythonExperience is type: ", type(pythonExperience))
```

:::

:::{admonition} Exercise 2
:class: note
Knowing that:

$$°C = \left ( °F - 32 \right )\times \frac{5}{9} $$

Write a program that converts degrees *celcius* to degrees *Farenheit*, and displays the results (using the `print()` function) of $°C$ and $°F$, in the case of 20° c.
:::



:::{admonition} Exercise 2 (Solution)
:class: tip, dropdown

1) Simple solution:

```{code-block} python
:class: thebe

# 1) Assign 20 to the variable C (celsius)
C = 20
# 2) From the original equation, calculate the value of $°F$
F = C * (9/5) + 32

# 2) Print the value of $°C$ and $°F$
print ("Grados Farenheit", F)
```
2) Define a function:

```{code-block} python
:class: thebe

# 1) Define the function
def functionCelciusToFarenheit(C):
    """
    Convert degrees *celcius* to degrees *Farenheit*, and return the value of $°F$
    """
    F = C * (9/5) + 32
    return F
# 2) Call the function:
print(functionCelciusToFarenheit(20))
```
:::


:::{admonition} Exercise 3
:class: note
Continuing with our previous exercise, create a program in which the user can manually enter the value of degrees Celcius, and can know its correspondence in Fahrenheit.

The result should look something like:
```python
45 degrees Celsius equals -228.15 degrees Fahrenheit
```

:::

:::{admonition} Exercise 3 (Solution)
:class: tip, dropdown

Using the function defined in Exercise 2

```{code-block} python
:class: thebe

# 1) Using the function `input()` read the value of $°C$
C = float(input("Input a temperature value in degrees Celcius?"))

# Use this input in the function defined in Exercise 2
print(functionCelciusToFarenheit(C))
```

:::


:::{admonition} Exercise 4
:class: note
Now is your time. Show what is the third element inside the list `names`:

```python
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The thrid element in the list of names is: ' , XXXXXX) # Replace XXXX with the correct function
```
:::

:::{admonition} Exercise 4 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The thrid element in the list of names is: ' , names[2])
```

:::


:::{admonition} Exercise 5
:class: note

Try yourself, write a list of `n` elements, and using the `len` function, print the length of the list.

```python
myList = # Complete a list of values here
print('The length of the elements in myList is: ' , XXXXXX) # Replace XXXX with the correct function
```

:::

:::{admonition} Exercise 5 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe
myList = ["Vienna", "Caracas", 'Bogota', 'Brasilia', 'Buenos Aires', 'Ciudad de Mexico', 'Lima', 'Montevideo', 'Paramaribo', 'Santiago', 'La Paz', 'Caracas', 'Bogota', 'Brasilia', 'Buenos Aires', 'Ciudad de Mexico', 'Lima', 'Montevideo', 'Paramaribo', 'Santiago', 'La Paz']

print('The length of the elements in myList is: ' , len(myList))
```

:::


:::{admonition} Exercise 6
:class: note

```python
allDataTypes = ["Roderick", 1.85, 39, True]
print('The variable located in the XXXXX position in the list allDataTypes' , type(allDataTypes[YYYYYY])) # Replace XXXX and YYYYY with the correct values
```

:::

:::{admonition} Exercise 6 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe

allDataTypes = ["Roderick", 1.85, 39, True]
print('The variable located in the second position in the list allDataTypes' , type(allDataTypes[1]))
```

:::


:::{admonition} Exercise 7
:class: note
Using the conditional statements and logical operators explained in this section, write a program that based on the monthly production rates prints out the if a hydrocarbon field is :
* Excellent (greater than or equal to 1,000,000)
* Good (between 100,000 and 999,999)
* Poor (less than 100,000)
* Close (no production)

after the user input the monthly production rates manually.
:::

:::{admonition} Exercise 7 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe

produccionRate = float(input('Input the monthly production rates of the  field:'))

if produccionRate >= 1000000:
    print('Excellent')
elif produccionRate <=999999 and produccionRate >= 100000:
    print('Good')
elif produccionRate <=10000 and produccionRate > 0:
    print('Poor')
elif produccionRate == 0:
    print('Close')
```
:::


:::{admonition} Exercise 8
:class: note

First, build a 2D Numpy array, of 2 rows and 3 columns. Then, using the reshape function, reshape it into a 2D array of 2 columns and 3 rows. Print the dimensions of the originanl and new array.

```{image} ../../images/numpyReshapeExercise.png
:alt: numpyReshapeExercise
:class: bg-primary mb-1
:width: 800px
:align: center
```

:::

:::{admonition} Exercise 8 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe
# 1) Build a 2D Numpy array, of 2 rows and 3 columns

array = np.array([[1, 2, 3], [4, 5, 6]])

# 2) Print the dimensions of the original
print('The dimensions of the original array are: ', array.shape)

# 3) Use the reshape function, reshape it into a 2D array of 2 columns and 3 rows

arrayReshape = array.reshape(3, 2)

# 3) Print the dimensions of the reshaped array
print('The dimensions of the reshaped array are: ', arrayReshape.shape)

```
:::


:::{admonition} Exercise 9
:class: note

What can you say about diving two arrays?

```python
array1 = XXXXXXXXXXXXXXXX # Try first with an scalar, and then try with an array

array2 = np.array([24, 60, 12, 40, 15])

arrayDiv = array1 XXXXXXX array2

print('The division of array1 and array2 is: ', arrayDiv)
```

:::

:::{admonition} Exercise 9 (Solution)
:class: tip, dropdown

Scalar division

```{code-block} python
:class: thebe

import numpy as np

scalar1 = 10

array1 = np.array([24, 60, 12, 40, 15])

arrayDiv1 = scalar1 / array1

print('The division of array1 and array2 is: ', arrayDiv1)

```

Proof the results:

```{code-block} python
:class: thebe
print('The first division is equal to: ', scalar1 / array1[0])
```

But what happens if you use the `//` for the division?

```{code-block} python
:class: thebe
arrayDiv1 = scalar1 // array1

print('The floor division of array1 and array2 is: ', arrayDiv1)

```

And if want to use the the `%` for the division?

```{code-block} python
:class: thebe
arrayDiv1 = scalar1 % array1

print('The modulus of the division between array1 and array2 is: ', arrayDiv1)

```

Division between two arrays:

```{code-block} python
:class: thebe
array1 = np.array([2, 4, 6, 8, 10, 12])

array2 = np.array([24, 60, 12, 40, 15])

arrayDiv2 = array1 / array2

print('The division of array1 and array2 is: ', arrayDiv2)
```

Another option is to use the `np.divide` function.

```{code-block} python
:class: thebe
arrayDiv2 = np.divide(array1, array2)

print('The division of array1 and array2 is: ', arrayDiv2)
```
:::

:::{admonition} Exercise 10
:class: note
Can you create a `DataFrame` (4 columns and 8 rows), and then slice the `DataFrame` between the rows 3 and 6, and the columns 2 and 3 using the `loc` and `iloc` attributes?
:::

:::{admonition} Exercise 10 (Solution)
:class: tip, dropdown

1) Create a dataset with 4 columns and 8 rows

```{code-block} python
:class: thebe
import pandas as pd 

# 1) Define a dictionary with the data
myNewDataset = {
    'names' : ['Roderick', 'Daniel', 'Juan', 'Ricardo','Kurt', 'Patrick', 'Elli', 'Lia'],
    'age' : [39, 40, 41, 42, 43, 44, 45, 46],
    'height' : [1.85, 1.80, 1.75, 1.70, 1.65, 1.60, 1.55, 1.50],
    'weight' : [39, 40, 41, 42, 43, 44, 45, 46],
}

# 2) Create a DataFrame from the dictionary
myNewDataFrame = pd.DataFrame(myNewDataset)

# 3) Print the DataFrame
print(myNewDataFrame)

# 4) Slice the DataFrame between the rows 3 and 6, and the columns 2 and 3 using `loc`

myNewDataFrameLOC = myNewDataFrame.loc[3:6, 'height':'weight']
print(myNewDataFrameLOC)


# 5) Slice the DataFrame between the rows 3 and 6, and the columns 2 and 3 using `iloc`

myNewDataFrameILOC = myNewDataFrame.iloc[3:6, 2:4]
print(myNewDataFrameILOC)

```
:::


:::{admonition} Exercise 11
:class: note
Explore the **Precipitation dataset**, and show the first 10 and the last 7 rows of the dataset using the `head` and `tail` functions.

:::

:::{admonition} Exercise 11 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe
import pandas as pd 

# 1) Import the dataset
filename = 'https://www1.ncdc.noaa.gov/pub/data/cdo/samples/PRECIP_HLY_sample_csv.csv'
precipitationDataFrame = pd.read_csv(filename)

# 2) Show the first 10 rows
print(precipitationDataFrame.head(10))

# 3) Show the last 7 rows
print(precipitationDataFrame.tail(7))
```

:::

:::{admonition} Exercise 12
:class: note
Repeat the previous exercise, but this time, use the `marker` keyword argument to plot the points as a line with a star [Reference](https://matplotlib.org/stable/api/markers_api.html).
:::

:::{admonition} Exercise 12 (Solution)
:class: tip, dropdown
```{code-block} python
:class: thebe
import matplotlib.pyplot as plt
import numpy as np

y_points = np.array([1, 1, 3, 4])

plt.plot(y_points, marker = '*')
plt.show()
```
:::


:::{admonition} Exercise 13
:class: note

Can you think of a way to plot the same data in a vertical layout?

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
import numpy as np

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
:::

:::{admonition} Exercise 13 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe

import numpy as np
import matplotlib.pyplot as plt

# Plot 1:
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x1,y1)

# Plot 2:
x2 = np.array([0, 1, 2, 3])
y2 = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x2,y2)

plt.show()
```
:::


:::{admonition} Exercise 14
:class: note

Can you think of a way to generate a bar plot, where the bars are oriented horizontally, red, with a width of 0.5?

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y)
plt.show()

```

:::

:::{admonition} Exercise 14 (Solution)
:class: tip, dropdown

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y, width = 0.5, color = 'red')
plt.show()

```

:::{admonition} Width in Horizontal Bar Plot
:class: error

You can not give the width to the horizontal bars, since you already set the width with the second parameter of barh; you need to set the height instead


```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y, height = 0.5, color = 'red')
plt.show()

```

:::



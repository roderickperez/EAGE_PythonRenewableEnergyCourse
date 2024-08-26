# Numpy

As we discussed in the previous section, ny default, in Python we can use lists in order to storage data arrays. Arrays are very frequently used in data science, where speed and resources are very important. However, when we have large volumes of data, traditional data structures (in this case lists) are slow and inefficient. Therefore, it is necessary to resort to external libraries, such as Numpy.

**NumPy** (**Num**erical **Py**thon) was created in 2005 by Travis Oliphant. It is an open source project [open source project](https://github.com/numpy/numpy), used for working with arrays in Python. Numpy is a Python package for large array handling, which provide us with an array object that is up to 50x faster than traditional Python lists. It is written in Python, but most of the parts that require fast computation are written in C or C++.

## Array creation

The array object in NumPy is called `ndarray`, it provides a lot of supporting functions that make working with `ndarray` very easy.

:::{admonition} Why Numpy is faster?
NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
:::

The first step to work with Numpy in Python is to installed the `numpy` library, and then import iy into our code. Due to its popularity in Python code development, Numpy is usually one of the default libraries found in many Python distributions. Therefore, in most of the times, it is not necessary to install it. However, if necessary, this can be done via the `pip install numpy` command.

```{thebe-button}

```

```{code-block} python
:class: thebe
import numpy
```

:::{admonition} Aliases
In python programming, the second name given to a piece of data is known as an alias. Aliasing happens when the value of one variable is assigned to another variable because variables are just names that store references to actual value.

For example,

```python

import nameLibrary as nickName

```

In this case, everytime we need to call to `nameLibrary`, instead of written the full name of the library we will use the alias (shortname) `nickName`.
:::

Based on this, it is very common find that numpy is assigned to the alias `np` in the code. Notice, that the alias is independent to the user.

```{code-block} python
:class: thebe
import numpy as np
```

In order to compare, let's create a list of values as we learning from previous sections:

```{code-block} python
:class: thebe
arrayList = [1, 2, 3, 4, 5] # This is a list of values
print(arrayList)
print('The arrayList array is a ', type(arrayList))
```

Then, we can create an array using the `np.array` function.

```{code-block} python
:class: thebe
arrayNumpy = np.array([1, 2, 3, 4, 5]) # This is an array of values
print(arrayNumpy)
print('The arrayNumpy array is a ', type(arrayNumpy))
```

:::{admonition} Convert List to Numpy Arrays
:class: tip

```{code-block} python
:class: thebe
arrayList = [1, 2, 3, 4, 5] # This is a list of values
arrayNumpyConverted = np.array(arrayList) # This is an array of values

print('The arrayNumpyConverted array is a ', type(arrayNumpyConverted))
```

:::

:::{admonition} 'numpy.array()` and 'numpy.asarray()'
:class: warning

The difference between both is that `numpy.array()` will make a duplicate of the original object and `numpy.asarray()` would mirror the changes in the original object.

```{code-block} python
:class: thebe

# Define a Python list
pythonList = [1, 7, 0, 6, 2, 5, 6]

# Converting Python list into Numpy array
numpyArray = np.asarray(pythonList)

# Displaying Python list
print ("List:", pythonList)

# Displaying Numpy array
print ("numpyArray: ", numpyArray)

# Create another array out of numpyArray using asarray function
numpyAsArray = np.asarray(numpyArray)

# Displaying numpyAsArray before the changes made
print("numpyAsArray: ", numpyAsArray)

# Edit numpyAsArray
numpyAsArray[2] = 99

# Displaying List, numpyArray and, numpyAsArray after the change has been made
print("List: " , pythonList)
print("numpyArray: " , numpyArray)
print("numpyAsArray: " , numpyAsArray)
```

Notice that we **only** replaced the item of the numpyAsArray at index 2. However, we expect that the original numpyArray is not changed, but it is.
:::

### Dimensions in Arrays

A dimension in arrays is one level of array depth (nested arrays). The following image shows the most common and basic arrays that we can build in Numpy:

```{image} ../../images/numpyArrays.png
:alt: numpyArrays
:class: bg-primary mb-1
:width: 800px
:align: center
```

#### 0-D Arrays (Scalar)

0-D arrays, or Scalars, are the elements in an array. Each value in an array is a 0-D array.

:::{admonition} Check Number of Dimensions
: class: tip
NumPy Arrays provides the `ndim` attribute that returns an integer that tells us how many dimensions the array have.
:::

```{code-block} python
:class: thebe
arr_0D = np.array(3)
print(arr_0D)
print('The data type of the arr_0D is: ', type(arr_0D))
print('The dimension of the arr_0D is:', arr_0D.ndim)
```

#### 1-D Arrays (Vector)

An array that has 0-D arrays as its elements is called uni-dimensional or 1-D array.

```{image} ../../images/numpy1D.png
:alt: numpyArrays1D
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{code-block} python
:class: thebe
arr_1D = np.array([3, 2])
print(arr_1D)
print('The data type of the arr_1D is: ', type(arr_1D))
print('The dimension of the arr_1D is:', arr_1D.ndim)
```

#### 2-D Arrays (Matrix)

An array that has 1-D arrays as its elements is called a 2-D array. These are often used to represent matrix or 2nd order tensors.

```{image} ../../images/numpy2D.png
:alt: numpyArrays2D
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{code-block} python
:class: thebe
arr_2D = np.array([[1, 0, 1], [3, 4, 1]])
print(arr_2D)
print('The data type of the arr_2D is: ', type(arr_2D))
print('The dimensions of the arr_2D are:', arr_2D.ndim)
```

#### 3-D Arrays (Tensor)

An array that has 2-D arrays (matrices) as its elements is called 3-D array. These are often used to represent a 3rd order tensor.

```{image} ../../images/numpy3D.png
:alt: numpyArrays3D
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{code-block} python
:class: thebe
arr_3D = np.array([[[1, 7, 9], [5, 9, 3], [7, 9, 9]], [[2, 3, 5], [1, 0, 0], [2, 9, 2]]] )
print(arr_3D)
print('The data type of the arr_3D is: ', type(arr_3D))
print('The dimensions of the arr_3D are:', arr_3D.ndim)
```

#### Higher Dimensional Arrays

With Numpy, an array can have any number of dimensions. In this case, when the array is created, we can specify the number of dimensions by using the `ndmin` argument.

```{code-block} python
:class: thebe
arr_5D = np.array([1, 2, 3, 4], ndmin=5)
print(arr_5D)
print('The data type of the arr_5D is: ', type(arr_5D))
print('The dimensions of the arr_5D are:', arr_5D.ndim)
```

In this case, we created array where the innermost dimension (5th dim) has 4 elements, the 4th dim has 1 element that is the vector, the 3rd dim has 1 element that is the matrix with the vector, the 2nd dim has 1 element that is 3D array and 1st dim has 1 element that is a 4D array.

### Indexing

Array indexing is the same as accessing an array element. You can access an array element by referring to its index number. The indexes in NumPy arrays start with 0, meaning that the first element has index 0, and the second has index 1 etc.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4])

print('The first element of the array arr_1D is: ', arr_1D[0])
```

In case of the 2-D array, we can access the element by referring to the row and column index.

```{code-block} python
:class: thebe
arr_2D = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('The second element on the first row in the arr_2D is: ', arr_2D[0, 1])
```

The same can be done with the higher dimensional arrays.

Also, we can use the negative indexing to access the last element in the array.

```{code-block} python
:class: thebe
arr_2D = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('The last element from 2nd dim in the arr_2D array is: ', arr_2D[1, -1])
```

### Slicing

Slicing in python means taking elements from one given index to another given index. We pass slice instead of index like this: `[start:end]`. Also, we can add an additional parameter to define the step: `[start:end:step]`. In case we don't pass start its considered `0`. Additionally, in the case that we don't pass end its considered length of array in that dimension. If we don't pass step its considered 1.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7])
slice_1D = arr_1D[2:5]

print('The slice between the second and fifth item in my arr_1D is: ', slice_1D)
```

:::{admonition} Tip
The result includes the `start` index, but **excludes** the `end` index.
:::

Now, if we want to take the elements from the second index to the last index, we can use the `:` operator.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7])
print('The slice from the second index to the last index in my arr_1D is: ', arr_1D[2:])
```

On the other hand, if we want to take the elements from the first index to the third index, we can also use the `:` operator.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7])
print('The slice from the first index to the third index in my arr_1D is: ', arr_1D[:3])
```

Also, we can use the minus operator to refer to an index from the end. This procedure is known as _negative slicing_.:

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr_1D[-3:-1])
```

:::{admonition} Tip
:class: tip
We can control the number of steps of the slicing, adding the `step` parameter. Remember that it is set to 1 by default.
:::

Finally, if we want to take the elements from the first index to the last index, but we want to skip every other element, we can use the `::` operator.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr_1D[::])
```

Notice that it return the same (original) array.

Based on this, we can slice a 2-D array.

```{code-block} python
:class: thebe
arr_2D = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr_2D[0:2, 2])
```

### Shape

The shape of an array is the number of elements in each dimension. In order to show the shape of an array, we can use the `shape` attribute.

```{code-block} python
:class: thebe
arr_2D = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print('The shape of the arr_2D is: ', arr_2D.shape)
```

The example above returns (2, 5), which means that the array has 2 dimensions, where the first dimension has 2 elements (rows) and the second has 5 (columns).

:::{admonition} Do you remember the tuples?
Numpy store the shape of an array in a **tuple** data structure.

```{code-block} python
:class: thebe
arr_2D = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
shape_arr_2D = arr_2D.shape
print(type(shape_arr_2D))
```

:::

### Reshape

For Numpy, **reshaping** means changing the shape of an array. The shape of an array is the number of elements in each dimension. By reshaping we can add or remove dimensions or change number of elements in each dimension.

For example, we can convert a 12 rows 1D array to a 2D array with 6 rows and 2 columns.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print('The original shape of the arr_1D is: ', arr_1D.shape)

arr_2D = arr_1D.reshape(6, 2)
print('The shape of the reshaped arr_1D is: ', arr_2D.shape)
```

:::{admonition} Exercise 8
First, build a 2D Numpy array, of 2 rows and 3 columns. Then, using the reshape function, reshape it into a 2D array of 2 columns and 3 rows. Print the dimensions of the originanl and new array.

```{image} ../../images/numpyReshapeExercise.png
:alt: numpyReshapeExercise
:class: bg-primary mb-1
:width: 800px
:align: center
```

:::

:::{admonition} Can We Reshape An Array Into any Shape?
:class: warning
Yes, as long as the elements required for reshaping are equal in both shapes. For example, we can reshape an 8 elements 1D array into 4 elements in 2 rows 2D array.

```{code-block} python
:class: thebe
arr_1D = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print('The original shape of the arr_1D is: ', arr_1D.shape)

arr_2D = arr_1D.reshape(4, 2)
print('The shape of the new arr_1D (reshaped) is: ', arr_2D.shape)
```

Although, we cannot reshape it into a 3 elements 3 rows 2D array as that would require 3x3 = 9 elements.

```{code-block} python
:class: thebe
arr_2D = arr_1D.reshape(3, 3)
```

In this case we will get an error.

:::

### Flatenning

Flattening array means converting a multidimensional array into a 1D array. We can use `reshape(-1)` to do this. This function will be very useful during execution of the Machine Learning and Deep Learning algorithms.

```{code-block} python
:class: thebe
arr_2D = np.array([[1, 2, 3], [4, 5, 6]])
print('The shape of the arr_2D is: ', arr_2D.shape)

arr_2D_flat = arr_2D.reshape(-1)
print('The shape of the arr_2D flattened is: ', arr_2D_flat.shape)

```

## Random Number Generation

Another of the popular features of Numpy is the random number generation.

### Integers

For example, if we want to generate a random **integer** from 0 to 20, we can use the `np.random.randint` function.

```{code-block} python
:class: thebe
from numpy import random

x = random.randint(20)

print(x)

```

:::{admonition} Click multiple times
:class: tip
Click the `run` button multiple times to see the different random numbers generated.

:::

### Floats

Now, in case we want to generate a random **floats**, we can use the `np.random.rand` function.

```{code-block} python
:class: thebe
from numpy import random

x = random.rand(20)

print(x)

```

Also, in Numpy we can combine the array construction feature with the generator function to generate a random array, specifying the size of the array. For example, if we want to generate a 1-D array containing 5 random integers from 0 to 20:

```{code-block} python
:class: thebe
from numpy import random

x = random.randint(20, size=(5))

print(x)

```

In the same way, we can generate a 2-D with 3 rows, each row containing 5 random integers from 0 to 20:

```{code-block} python
:class: thebe
from numpy import random

x = random.randint(20, size=(3, 5))

print(x)

```

## Special Arrays

### Empty Array

The `numpy.empty()` function creates an array **without** initializing its entries. The syntax for using this function is:

```python
numpy.empty(shape, dtype=float, order='C', *, like=None)
```

where:

- `shape` describes the shape of the empty array. It can be a _tuple_ or a singular _integer_ value.
- `dtype` is an optional parameter that determines the datatype for the array elements. By default, this is numpy.float64.
- `order` is an optional parameter that specifies how to store the multidimensional data in memory. You can choose between `C` for C-style row-major form, and `F` for Fortran-style column-major form.
- `like` is an optional parameter. It is a reference object that makes it possible to create non-NumPy arrays.

Let's create our first empty array. Notice that the only mandatory parameter is `shape`.

```{code-block} python
:class: thebe
arrEmpty_0D = np.empty(0)

print(arrEmpty_0D)

```

:::{admonition} Is it useful a empty array of 0 dimensions?
:class: note
In practice this is not useful. Remember that in NumPy array elements are stored in contiguous blocks of memory. To add rows/columns into an existing array, such as to the empty array you just created, the array needs to be copied to a new memory location, which is ineficient.
:::

Now, let's create a new empty array with 2 rows and 3 columns.

```{code-block} python
:class: thebe
arrEmpty_2D = np.empty([2, 3])

print(arrEmpty_2D)

```

The output is a series of random values in the array, even though it is supposed to be empty. Remember that by definition **Emptiness** means that the elements in the array are **not initialized**. But the array is not really empty. Instead, the array values are arbitrary and depend on what happens to be in the chunk of memory allocated for them.

:::{admonition} Best practice
:class: tip
In order to avoid ineficcient memory usage, we should **not** use the `numpy.empty()` function arbitrary. The best bet is to create an **empty** array that has the desired shape of the array you want to create. Then you can just fill in the values to the array as you go. This saves you from wasting computing time in copying the array.
:::

### Zeros

An alternative way to create an _empty_ array is by using the `numpy.zeros()` function, whih returns an array where each element is zero. The syntax is the following:

```python
numpy.zeros(shape, dtype=float, order='C', *, like=None)
```

Based on this, we can generate a 1D array of zeros:

```{code-block} python
:class: thebe
arrZeros_1D = np.zeros(3)

print(arrZeros_1D)

```

Or, a 2D array of zeros:

```{code-block} python
arrZeros_2D = np.zeros([2, 2])

print(arrZeros_2D)
```

:::{admonition} Empty vs. Zeros?

- `numpy.empty()` function **does not initialize** the elements in the array.
- `numpy.zeros()` function **initializes** the elements at `0`.
  :::

### Ones

Based on the same structure as before, we can now generate arrays of ones, using the following syntax:

```python
numpy.ones(shape, dtype = None, order = 'C')
```

For example, if we want to generate a 1D array of ones:

```{code-block} python
:class: thebe
arrOnes_1D = np.zeros(6)

print(arrOnes_1D)
```

Or a 2D matrix of ones:

```{code-block} python
:class: thebe
arrOnes_2D = np.zeros([2, 4])

print(arrOnes_2D)
```

### Evenly and not evenly spaced

Along the course, we face the necessity to generate spaced arrays. In some of the cases you want the numbers to be _evenly_ spaced, but there are also times when you may need _non-evenly_ spaced numbers.

#### Linear

In the case we want to define a evenly spaced array, we can use the `numpy.linspace()` function. It requires the start and end range of values we want to define our array. By default, it defines a number of elements equal to `10`.

```{code-block} python
:class: thebe
linSpace_1D = np.linspace(0, 10)

print('The first element of linSpace_1D is: ', linSpace_1D[0])
print('The last element of linSpace_1D is: ', linSpace_1D[-1])
print('The number of elements in linSpace_1D is: ', linSpace_1D.size)
print('The number of dimensions in linSpace_1D is: ', linSpace_1D.ndim)
print('The datatype of linSpace_1D is: ', linSpace_1D.dtype)
print('The shape of linSpace_1D is: ', linSpace_1D.shape)
print('-------------------')
print(linSpace_1D)
```

Notice that this code returns a `ndarray` type object, with equally spaced intervals between the `start` and `stop` values.

:::{admonition} Is the `stop` value included?
:class: note

Note that the value `10` is **included** in the output array. The function returns a _closed range_, one that includes the endpoint, by default.

This is contrary to what you might expect from Python, in which the end of a range usually isn’t included.
:::

In case we want to specify the number of elements in the array, we can use the `numpy.linspace()` function with the `num` parameter.

```{code-block} python
:class: thebe
linSpace_1D = np.linspace(0, 10, num = 20)

print('The first element of linSpace_1D is: ', linSpace_1D[0])
print('The last element of linSpace_1D is: ', linSpace_1D[-1])
print('The number of elements in linSpace_1D is: ', linSpace_1D.size)
print('The number of dimensions in linSpace_1D is: ', linSpace_1D.ndim)
print('The datatype of linSpace_1D is: ', linSpace_1D.dtype)
print('The shape of linSpace_1D is: ', linSpace_1D.shape)
print('-------------------')
print(linSpace_1D)
```

We can see that the output array has 20 elements, equally spaced values between 1 and 10.

:::{admonition} Positional arguments
:class: note

In Python, positional arguments are those that are placed in the function call **before** the keyword arguments. In this case, we might skip the keyword `num` and just write desired value, for example:

```python
linSpace_1D = np.linspace(0, 10, 20)
```

:::

:::{admonition} `range()` vs `linspace()`
:class: warning
Python has a built-in function called `range(n)` that generates a sequence of evenly spaced range of numbers between `0` and `n-1`. For numerical applications, this function is limited to integers. For that reason, in some case is more convinient to use `linspace()`.
:::

#### `np.arange()`

NumPy has its own version of the Python built-in `range()`, called `np.arange()`. Unlike `range()`, it’s not restricted to just integers. We can use `np.arange()` in a similar way to `range()`, using start, stop, and step as the input parameters:

```{code-block} python
:class: thebe
list(range(2, 30, 2))
```

Which returns a list of integers from `0` to `9`.

The equivalent result using `np.arange()` is:

```{code-block} python
:class: thebe
np.arange(2, 30, 2)
```

:::{admonition} Output types
:class: note
The output values are the same, although `range()` returns a range object, which need be converted to a list to display all the values, while `np.arange()` returns an array.
:::

:::{admonition} `np.linspace()` or `np.arange()`?

- Use `np.linspace()` when the exact values for the start and end points of your range are the important attributes in your application.
- Use `np.arange()` when the step size between values is more important.
  :::

#### Logarithmic

Similar to the previous case, we can use the `numpy.logspace()` function to generate evenly spaced logarithmic values. The syntax of this function is:

```python
numpy.logspace(start, stop, num = 50, endpoint = True, base = 10.0, dtype = None)
```

where:

- `start` beginning of interval range,
- `stop` end of interval range,
- `endpoint if`True`, stop is the last sample. By default,`True`
- `num` defines the number of samples to generate
- `base` defines the base of log scale. By default, equals `10.0`
- `dtype` is the type of output array

```{code-block} python
:class: thebe
logSpace_1D = np.logspace(0, 10, 20)

print('The first element of logSpace_1D is: ', logSpace_1D[0])
print('The last element of logSpace_1D is: ', logSpace_1D[-1])
print('The number of elements in logSpace_1D is: ', logSpace_1D.size)
print('The number of dimensions in logSpace_1D is: ', logSpace_1D.ndim)
print('The datatype of logSpace_1D is: ', logSpace_1D.dtype)
print('The shape of logSpace_1D is: ', logSpace_1D.shape)
print('-------------------')
print(logSpace_1D)
```

## Operations with Arrays

Similar to the operations we performed earlier with scalar or float type variables, Python (through Numpy) allows us to perform arithmetic operations between arrays, such as `add()`, `subtract()`, `multiply()`, and `divide()` . However, one of the most important observations is that the arrays that are going to be used to perform these operations must be either of the same shape or should conform to array broadcasting rules.

### Summation

We can add one scalar value to each element of an array:

```{image} ../../images/numpyAddArraysScalar.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
scalar = 1

array = np.array([1, 2, 3, 4, 5])

sum_array = scalar + array

print('The sum of scalar and array is: ', sum_array)
```

Or, we can add one array to another:

```{image} ../../images/numpyAddArrays.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
array1 = np.array([1, 2, 3, 4, 5])

array2 = np.array([6, 7, 8, 9, 10])

arraySum = array1 + array2

print('The sum of array1 and array2 is: ', arraySum)
```

:::{admonition} Arrays with different shapes
:class: warning

Try yourself, what will happen if the arrays have different shapes?

```{code-block} python
:class: thebe
array1 = np.array([1, 2, 3, 4, 5])

array2 = np.array([6, 7, 8, 9])

arraySum = array1 + array2

print('The sum of array1 and array2 is: ', arraySum)
```

:::

### Subtraction

In the same way we can substract one scalar value to each element of an array:

```{image} ../../images/numpySubstractArraysScalar.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
scalar = 2

array = np.array([6, 7, 8, 9, 10])

subt_array = scalar - array

print('The subtraction of scalar and array is: ', subt_array)
```

Or, we can subtract one array to another:

```{image} ../../images/numpySubtArrays.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
array1 = np.array([1, 2, 3, 4, 5])

array2 = np.array([6, 7, 8, 9, 10])

arraySubt = array1 - array2

print('The subtraction of array1 and array2 is: ', arraySubt)
```

### Multiplication

We can do the same in case we want to multiply one scalar value to each element of an array:

```{image} ../../images/numpyMultArraysScalar.png
:alt: numpyMultArraysScalar
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
scalar = 2

array = np.array([6, 7, 8, 9, 10])

mult_array = scalar * array

print('The multiplication of scalar and array is: ', mult_array)
```

Or, we can multiply one array to another:

```{image} ../../images/numpyMultArrays.png
:alt: numpyMultArrays
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
array1 = np.array([1, 2, 3, 4, 5])

array2 = np.array([6, 7, 8, 9, 10])

arrayMult = array1 * array2

print('The multiplication of array1 and array2 is: ', arrayMult)
```

### Division

Now, try yourself, what will happen if we divide one scalar with and array, or an array by another?

:::{admonition} Exercise 9
:class: note

What can you say about diving two arrays?

```{code-block} python
:class: thebe
array1 = XXXXXXXXXXXXXXXX # Try first with an scalar, and then try with an array

array2 = np.array([24, 60, 12, 40, 15])

arrayDiv = array1 XXXXXXX array2

print('The division of array1 and array2 is: ', arrayDiv)
```

:::

### Dot Product

In mathematics, the dot product or scalar product is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors), and returns a single number [Reference](https://en.wikipedia.org/wiki/Dot_product).

The function `np.dot(a,b)` returns the dot product of two arrays. For 2D vectors, it is the equivalent to matrix multiplication. For 1D arrays, it is the inner product of the vectors. For N-dimensional arrays, it is a sum product over the last axis of a and the second-last axis of b.

```{image} ../../images/dotProduct.png
:alt: dotProduct
:class: bg-primary mb-1
:width: 400px
:align: center
```

```{code-block} python
:class: thebe
array1 = np.array([2, 7, 1])

array2 = np.array([8, 2, 8])

arrayDotProd = np.dot(array1, array2)

print('The dot product of array1 and array2 is: ', arrayDotProd)
```

:::{admonition} Dot Product using `@` keyword
:class: note
Python 3.5 introduced the `@` operator to calculate the dot product of n-dimensional arrays created using NumPy.
:::

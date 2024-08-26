# Data Structures

There are four collection data types in the Python programming language:

* **List** is a collection which is *ordered* and *changeable*. Allows duplicate members.
* **Tuple** is a collection which is *ordered* and *unchangeable*. Allows duplicate members.
* **Set** is a collection which is *unordered*, *unchangeable*, and *unindexed*. No duplicate members.
* **Dictionary** is a collection which is *ordered* and *changeable*. No duplicate members.

---

## Lists

The simplest data structure in Python is a **list**. It can be defined as a collection of data stored in a variable. They are grouped by `[ ]` and separated by a `,`.

In a list we can store strings:

```{thebe-button}
```

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print(names)
```

Integers:

```{code-block} python
:class: thebe
integers = [0, 2, 4, 1, 6]
print(integers)
```

Floats:

```{code-block} python
:class: thebe
floats = [0.21, 2.345, 4.11, 3.141, 8.41]
print(floats)
```

Or the combination of all the data types described previously:

```{code-block} python
:class: thebe
allDataTypes = ["Roderick", 1.85, 39, True]
print(allDataTypes)
```

:::{admonition} Remainder
:class: tip
What is the command (or function) that allows us to know the type of data stored in a variable?
:::

### Indexes
Indices are used to access specific elements of a *list*. 

If I want to access the **first** element of my list, using the `[ ] `, I can use the index `0`. If I want to access the **second** element of my list, I can use the index `1`. If I want to access the **third** element of my list, I can use the index `2`, etc.

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The first element of the list of name is: ' , names[0])
```

:::{admonition} Exercise 4
Now is your time. Show what is the third element inside the list `names`:

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The thrid element in the list of names is: ' , XXXXXX) # Replace XXXX with the correct function
```
:::


:::{admonition} Length of a list
Additionally, if we want to know the total number of elements in a list, we use the `len` function, which is equivalent to length.

```python
lengthList = len(nameList)
```

:::

:::{admonition} Exercise 5
Try yourself, write a list of `n` elements, and using the `len` function, print the length of the list.

```{code-block} python
:class: thebe
myList = # Complete a list of values here
print('The length of the elements in myList is: ' , XXXXXX) # Replace XXXX with the correct function
```
:::

In the same way we did previously, we can combine the function `type()` which allow us to know what type of data is stored in a variable, we can use this function to know the type of data stored in a variable, as well as the variable stored in a specific index in a list.

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The variable names is a' , type(names))
```

as well as the variable stored in a specific index in a list.

:::{admonition} Exercise 5

Check by yourself the type of variable in different locations in the list `allDataTypes`.

```{code-block} python
:class: thebe
allDataTypes = ["Roderick", 1.85, 39, True]
print('The variable located in the XXXXX position in the list allDataTypes' , type(allDataTypes[YYYYYY])) # Replace XXXX and YYYYY with the correct values
```
:::

:::{admonition} Last element in a list | Shortcut
:class: tip
Remember that in Python, the first index is `0`. But Python makes it easy for us to access the last element of a list, through a "shortcut",

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The last element of the list of names is: ' , names[-1])
```

What happend when you replace [-1] with [-2]?
```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The second last element of the list of names is: ' , names[-2])
```
:::

### Replace values in a list

If I want to replace the specific value of an element in my list:

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]

print('The original list of names is: ' , names)
```

If I want to replace the thrid value of my list with the value `"Daniela"`, I can use the index `2` to replace the value.

```{code-block} python
:class: thebe
names[2]= "Daniela"

print('The modiefied list is: ' , names)
```

### Delete values in a list
In the case that we wanted to remove one of the elements stored inside the `names` variable:

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]

print('The original list is: ' , names)

del names[2]

print('The list after deleting a item is: ' , names)

```

### Add a value
In case we want to add something new to the list we can follow this syntax:
```python
variable_name.append("Item")
```

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]

print('The original list is: ' , names)

names.append("Beatriz")

print('The list after adding a new item is: ' , names)

```

This method adds a value to the end of the list. However, in the case of working with an integers, it doesn't work the same way. For example:

```{code-block} python
:class: thebe
allDataTypes = ["Roderick", 1.85, 39, True]
lengthList = len(allDataTypes)
print('The length of the list is: ' , lengthList)

```

If we want to add a new element to the integer, and using the same `append` function.

```{code-block} python
:class: thebe
lengthList.append(4)
```

we got an error.

:::{admonition} Be careful!!!!
:class: warning
The `append()` method cannot be used on **all** Python data types. This is exclusive to certain types of data, including lists. Cannot be used on `float`, `int`, `bool`.
:::

### Count the number of times a value appears in a list
Other very popular Python list methods, like finding how many times a certain value appears in my list, using `count()` in the following syntax:

```python
nameList.count("valueFind")
```

```{code-block} python
:class: thebe
cities = ["Caracas", "Bogota", "Rio", "Vienna", "Caracas"]

cities.count("Caracas")
```

If the value **NOT** appears in the list, it will simply be `0`:
```{code-block} python
:class: thebe
cities.count("Madrid")
```

In the same way we can know what is the index of a specific value in a list, using the `index()` method:

```{code-block} python
:class: thebe
indexBogota = cities.count("Bogota")
print('Bogota is located in the index ', indexBogota)
```

Now, in case the value is not found in the list, we will get an error. Remember that you do not have to fear programming errors. The most important thing is to be able to read and understand the error message. For example,

```{code-block} python
:class: thebe
cities.count("Buenos Aires")
```

### Reverse a list
The `reverse()` method returns the list in reverse order.
```{code-block} python
:class: thebe
cities = ["Caracas", "Bogota", "Rio", "Vienna"]

cities.reverse()
```
Note that it does not return anything. However, if we call the list again:

```{code-block} python
:class: thebe
print(cities)
```

:::{admonition} Modifying original list
:class: warning
It is important to note that there are some methods in Python that modify the original list. Therefore, it is important to be very careful when applying them.

:::

### Ordering a list

Through the `sort()` method it is possible to sort the list:

```{code-block} python
:class: thebe
names = ["Maria", "Pedro", "Juana", "Omar"]
print('The original list is: ' , names)

names.sort()

print('The sorted list is: ' , names)

```

:::{admonition} Not all lists can be sorted
:class: warning
We are going to try to sort a list, which contains different types of data. For example:

```{code-block} python
:class: thebe
allDataTypes = ["Roderick", 1.85, 39, True]
allDataTypes.sort()
```

This error message occurs because there are different types of data in the list.

:::

## Tuples

Tuples is one the most popular data structures in Python. They are similar to lists, can be used to used to store multiple items (with different data types) in a single variable, but the main difference is that they are **immutable** . In order to define a tuple we need to use parenthesis `()`, and separate the items by commas `,`.

```{code-block} python
:class: thebe
tupleExample = ("Roderick", 1.85, 39, True)

print('The tupleExample is a: ' , type(tupleExample))

```

Another feature of the tuples is that the items are ordered, unchangeable and allow duplicate values.

### Ordered

When we say that tuples are ordered, it means that the items have a defined order, and that order will not change.

```{code-block} python
:class: thebe
tupleExample = ("Roderick", 1.85, 39, True)

# Ordered
print('The first item in the tupleExample tuple is : ' , tupleExample[0])

```

### Immutable
Tuples are immutable, meaning that we cannot change, add or remove items after the tuple has been created.

```{code-block} python
:class: thebe
tupleExample = ("Roderick", 1.85, 39, True)

tupleExample.append("Caracas")

```

This code generate the following error:
```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Input In [3], in <cell line: 3>()
      1 tupleExample = ("Roderick", 1.85, 39, True)
----> 3 tupleExample.append("Caracas")

AttributeError: 'tuple' object has no attribute 'append'

```

As we can read Python is saying that the tuples are inmmutable.


### Allow duplicate values
Since tuples are indexed, they can have items with the same value. 

```{code-block} python
:class: thebe
tupleExample = ("Roderick", 1.85, 39, True, "Roderick")

print('The items in my tupleExample tuple are : ' , tupleExample)

```

### Length
To determine how many items a tuple has, we can use the same `len()` function as we did with lists.

```{code-block} python
:class: thebe
tupleExample = ("Roderick", 1.85, 39, True)

print('The tupleExample has : ' , len(tupleExample), 'items.')

```

:::{admonition} Create a tuple with one item
:class: tip

Imagine that you want to define a tuple with one item. As we described before, we need to use parenthesis `()` and separate the items by commas `,`. But since we only have one item, we just use the parenthesis `()`, right?

```{code-block} python
:class: thebe
oneItemTuple1 = ("Roderick")
print('The tupleExample is a: ' , type(oneItemTuple1))
```
What's is wrong?

To create a tuple with only one item, we need to add a comma `,` after the item, otherwise Python will not recognize it as a tuple. For example:

```{code-block} python
:class: thebe
oneItemTuple2 = ("Roderick",)
print('The tupleExample is a: ' , type(oneItemTuple2))
```
:::


## Dictionaries

Dictionaries are used to store data values in `key:value pairs`. In Python, a **dictionary** is a collection which is ordered*, changeable and do not allow duplicates. To define dictionaries we use curly brackets, and a set of keys and values.For example:

```python
dictionaryExample = {
    "key1": "value1", 
    "key2": "value2",
    ...
    }
```

Another way to build a dictionary in Python is using the `dict()` function. For example:

```python
dictionaryExample = dict([
    (<key>, <value>),
    (<key>, <value),
      .
      .
      .
    (<key>, <value>)
])
```

### Ordered*

When we say that dictionaries are ordered, it means that the items have a defined order, and that order will not change. However, since the items in a dictionary are presented in `key:value pairs`, we can access to them by using the key name. For example:

```{code-block} python
:class: thebe
dictionaryExample = {
    "name": "Roderick", # String
    "age": 39, # Integer
    "height": 1.85, # Float
    "is_programmer": True # Boolean
    }

print('The value of the key "name" in the dictionaryExample is: ' , dictionaryExample["name"])
```

### Changeable

In order to change a value in a dictionary, we need to specify the item by referring to its key name:

```{code-block} python
:class: thebe
dictionaryExample = {
    "name": "Roderick", # String
    "age": 39, # Integer
    "height": 1.85, # Float
    "is_programmer": True # Boolean
    }

print('Original dictionaryExample: ' , dictionaryExample)


dictionaryExample["name"] = "Roderick Perez"

print('Updated dictionaryExample: ' , dictionaryExample)

```

### Not allowed duplicates

Dictionaries cannot have two items with the same key:

```{code-block} python
:class: thebe
dictionaryExample = {
    "name": "Roderick", # String 
    "name": "Perez", # String
    "age": 39, # Integer
    "height": 1.85, # Float
    "is_programmer": True # Boolean
    }

print(dictionaryExample)
```

Notice that if we define to keys, the dictionary will overwrite the last defined value in the key with the new value.

### Length
To determine how many items a dictionary has, use the `len()` function:

```{code-block} python
:class: thebe
dictionaryExample = {
    "name": "Roderick", # String 
    "age": 39, # Integer
    "height": 1.85, # Float
    "is_programmer": True # Boolean
    }

print('The dictionaryExample has : ' , len(dictionaryExample), 'items.')
```

## Sets

A set is a data type in Python used to store multiple items (with different data types) in a single variable, which is unordered, inmutable*, unindexed, and do not allow duplicate values. In order to define a set, we need to use curly brackets `{ }`.

```{code-block} python
:class: thebe
setExample = {"Roderick", 1.85, 39, True}
print('The tupleExample is a: ' , type(setExample))
```

### Unordered

When we say that sets are unordered means that the items does not have a defined order, and you cannot refer to an item by using an index.

```{code-block} python
:class: thebe
setExample = {"Roderick", 1.85, 39, True}

# Ordered
print('The first item in the setExample set is : ' , setExample[0])

```

If we execute this code, we will get the following error message:

```python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Input In [34], in <cell line: 4>()
      1 setExample = {"Roderick", 1.85, 39, True}
      3 # Ordered
----> 4 print('The first item in the setExample set is : ' , setExample[0])

TypeError: 'set' object is not subscriptable
```

meaning that in Python a set are unordered, and we can not use the index to get the items.

### Inmutable

When we say that sets are unchangeable, it means that we cannot add, remove or change items in the set.

```{code-block} python
setExample = {"Roderick", 1.85, 39, True}

setExample.append("Caracas")

```

### Do not allow duplicate values
Sets cannot have two items with the same value.
```{code-block} python
setExample = {"Roderick", 1.85, 39, True}

print('The items in my setExample set are : ' , setExample)

```

### Length

To determine how many items a set has, we can use the same `len()` function as we did with previously.

```{code-block} python
:class: thebe
setExample = {"Roderick", 1.85, 39, True}

print('The setExample has : ' , len(setExample), 'items.')

```


## Summary

```{table} Python Data Structure Comparison
:name: python-data-structure-comparison

| Data Structure | Ordered  | Inmutable | Allow Duplicates | Indexed | Symbol |
|:---:|:---:|:---:|:---:|:---:|:---:|
| List | * |   | * | * | `[]`|
| Tuple  | * | *  | * | * | `()`|
| Set |  |   |  |  |`{}`|
| Dictionary | * |   |  |  | `{}`|


```

:::{admonition} Notes
:class: tip

* **Lists** are usually used for elements of the same type in a variable amount, while **tuples** are usually used in cases where there are different elements in a fixed amount.

* **Set** items are *unchangeable*, but you can remove and/or add items whenever you like.

* As of Python version 3.7, **dictionaries** are *ordered*. In Python 3.6 and earlier, dictionaries are *unordered*.

:::
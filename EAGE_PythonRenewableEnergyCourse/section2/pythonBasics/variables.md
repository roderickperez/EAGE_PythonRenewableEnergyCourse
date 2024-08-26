# Variables

Variables are containers for storing data values. In Python a variable is created the moment you first assign a value to it. Every variable created is an object that reserves a memory location that can store value in Python. The memory location is stored according to the data type of the variable declared. Different from other programming languages, in Python variables need not be defined or declared any data type to the variable as we do in other programming languages. 

The general syntax for declaring a variable is:

```python
variableName = value
```

Now, try yourself clicking on the `Run Code` button below:

```{thebe-button}
```


```{code-block} python
:class: thebe
a = 5
```

Did you changed the value of the variable `a` but the result is not displayed?.

:::{admonition} Print
:class: tip
If do you want to print the result of our code, we need to call one of the most important functions inside of Python, `print( )`

```python
print(name)
```
:::

Now, change the value of `a` again, and show it on the screen.

```{code-block} python
:class: thebe
a = 5
print(a)
```

:::{admonition} Add comments into the program
:class: tip
If do you want to add comment into the code, you can use `#` symbol. For example:

```python
a = 5 # This is a comment
```
:::

Notice that the variables do not need to be declared with any particular type, and can even change type after they have been set.

## Basic Data Types
If you want to specify the data type of a variable, this can be done with casting.

### Strings
```python
x = str(3)    # x will be '3'
```

### Integers
```python
y = int(3)    # y will be 3
```

### Floats
```python
z = float(3)  # z will be 3.0
```

:::{admonition} Readibility Tip
:class: tip
In Python, we can define a float with or without a `0` before the decimal symbol (`.`). 

For example:

```python
z = 0.6
```

is the same as

```python
z = .6
```

However, this notation is not recommendable since one of the key differential factors of Python is its **readability**.
:::

:::{admonition} `,` and Tuples
:class: error
In case we use the `,` symbol (instead of a `.`), we will define a *Tuple*. A 

In Python, we can define a float with or without a `0` before the decimal symbol (`.`). Tuple is one of 4 built-in data types in Python used to store collections of data. The other 3 are *List*, *Set*, and *Dictionary*, all with different qualities and usage. The main characteristic of a Tuple is that it is immutable. This means that once it is created, it cannot be changed. Also, the tuples are ordered, meaning that items have a defined order, and that order will not change.

For example, if we define :

```{code-block} python
:class: thebe
tuple = (0, 4)

print(tuple)

```
:::

### Boolean
```python
boolean_T = True

boolean_F = False
```

## Get the data type of a variable
In case we want to know the type of a variable with the ```type()``` function.

```python
name = "My name is Roderick"
age = int(39)
```

Now, we can combine Python functions, for example: `print ( )` and `type ( )` to evaluate what type of variable is our declared variable:
```python
print(type(name))
```

Now, try yourself:

```{code-block} python
:class: thebe
name = "Roderick"
print(type(name))
```

---


```{admonition} Exercise 1
Write a program that
* Store the value of your name, age, height, and Python experience, 
* Print its values, and the data type.
```

:::{admonition} Error
:class: error
In the following code, if you just click on the 'run' (button without changes), you will get an error because the **code is not complete**. 
```python
Input In [#]
    age = # Integer
          ^
SyntaxError: invalid syntax
```
In order to remove the error, **please complete the code with the requested and correct information**.
:::

```{code-block} python
:class: thebe
name = "Roderick" # String
age = # Integer
height = # Float
pythonExperience = False # Boolean

print(type(name))
print(type(age))
print(type(height))
print(type(pythonExperience))
```

:::{admonition} Single or Double Quotes?
:class: tip
Notice that if we want to declare a string variable, we can use declared either by using single or double quotes:
```python
x = "Roderick"
# is the same as
x = 'Roderick'
```
:::

Try yourself changing the quote symbols from `"` to `'`:

```{code-block} python
:class: thebe
name = "Roderick"

print(name)
```



:::{admonition} But don't mix them up!
:class: error
Notice that if we want to declare a string variable, we can use declared either by using single or double quotes:

But, it can't be mixed.
```python
x = "Roderick'
# is NOT the same as
x = 'Roderick"
```

In that case, you will get the following error:
```python
Input In [#]
    name = 'Roderick"
                     ^
SyntaxError: EOL while scanning string literal
```
:::

Now, in this case can you see a difference when we use `"` and / or  `'`?

```{code-block} python
:class: thebe
name = 'Roderick"

print(name)
```

:::{admonition} Don't be afraid of the error
:class: tip
It is normal that when you start programming, receiving an error message when executing your code can be somewhat frustrating. However, generally in Python the error messages are quite clear and explicit, and they help us to visualize where the error is, as well as to understand what we did wrong, and how to correct them.
:::
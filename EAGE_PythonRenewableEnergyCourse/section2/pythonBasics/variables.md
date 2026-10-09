---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Variables

A Python variable is a name bound to an object. Assignment binds or rebinds that name; the object has a type, while the name need not retain one type. Two names may refer to the same mutable object, so changing that object through one name can be visible through the other. [Python: names and binding](https://docs.python.org/3/reference/executionmodel.html#binding-of-names).

The general syntax for declaring a variable is:

```text
variableName = value
```

Now, try yourself! Click the **⚡ Launch** button in the top bar to activate the in-browser kernel, then press ▶ **Run** on the cell below:



```{code-cell} python
a = 5
```

Did you changed the value of the variable `a` but the result is not displayed?. [Python objects and types](https://docs.python.org/3/library/stdtypes.html).

:::{admonition} Print
:class: tip
If do you want to print the result of our code, we need to call one of the most important functions inside of Python, `print( )`

```python
name = "Alex"
print(name)
```
:::

Now, change the value of `a` again, and show it on the screen.

```{code-cell} python
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

Notice that the variables do not need to be declared with any particular type, and can even change type after they have been set. [Python objects and types](https://docs.python.org/3/library/stdtypes.html).

## Basic Data Types
Constructors such as `int`, `float`, and `str` convert or construct values; they do not permanently declare the type of a name. For example, `int(3.9)` truncates toward zero, while `int("3.9")` raises `ValueError`. [Python built-in types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex).

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
::: [Python objects and types](https://docs.python.org/3/library/stdtypes.html).

:::{admonition} `,` and Tuples
:class: error
A decimal point makes `0.4` a float. A comma makes `(0, 4)` a two-item tuple. Tuple positions cannot be replaced, but a mutable object stored inside a tuple can still change. These are different values, not two local number formats. [Python tuples](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences).

For example, if we define :

```python
coordinates = (0, 4)

print(coordinates)

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

```{code-cell} python
name = "Roderick"
print(type(name))
```

---


:::{admonition} Exercise 1 — Variables and types
:class: note

**Reference:** [Python tutorial](https://docs.python.org/3/tutorial/). Exercise values are original teaching inputs.
Write a program that
* Store the value of your name, age, height, and Python experience, 
* Print its values, and the data type.

```python
# TODO: create name, age, height, and python_experience
# TODO: print every value and its type
```
:::

:::{admonition} Error
:class: error
In the following code, if you press ▶ **Run** on the cell without any changes, you will get an error because the **code is not complete**. 
```python
Input In [#]
    age = # Integer
          ^
SyntaxError: invalid syntax
```
In order to remove the error, **please complete the code with the requested and correct information**.
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

```{code-cell} python
name = "Roderick" # String
age = 39 # Integer
height = 1.75 # Float
python_experience = False # Boolean

for value in [name, age, height, python_experience]:
    print(value, type(value))

assert isinstance(name, str)
assert isinstance(age, int)
assert isinstance(height, float)
assert isinstance(python_experience, bool)
```
:::

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

```{code-cell} python
name = "Roderick"

name = "Alex"
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

```{code-cell} python
name = "Roderick's notebook"

name = "Alex"
print(name)
```

:::{admonition} Don't be afraid of the error
:class: tip
It is normal that when you start programming, receiving an error message when executing your code can be somewhat frustrating. However, generally in Python the error messages are quite clear and explicit, and they help us to visualize where the error is, as well as to understand what we did wrong, and how to correct them.
:::

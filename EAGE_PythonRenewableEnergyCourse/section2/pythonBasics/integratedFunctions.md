# Integrated Functions

Along this tutorial we already explored some of the built-in (integrated) functions that are already available in Python. If we go to the [official documentation ](https://docs.python.org/3/library/functions.html) of Python we can explore some of the most common and useful functions, which will be useful during the course:

```{thebe-button}
```

## `abs()`
Return the absolute value of a number. The argument may be an integer, a floating point number, or an object implementing __abs__(). If the argument is a complex number, its magnitude is returned ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
variable = -20

print('The absolute value of the variable is: ' ,abs(variable))

```


## `enumerate()`
Return an enumerate object, which must be a sequence, an iterator, or some other object which supports iteration. The `__next__()` method of the iterator returned by `enumerate()` returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over iterable ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]


for items in enumerate(listExample):
    print(items)

```

## `help()`
Invoke the built-in help system. If no argument is given, the interactive help system starts on the interpreter console. If the argument is a string, then the string is looked up as the name of a module, function, class, method, keyword, or documentation topic, and a help page is printed on the console. If the argument is any other kind of object, a help page on the object is generated ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
def myFunction(parameters):
    """ Documentation
    This is the documentation of the myFunction, explaining the functionallity of it.
    """
    # code
    return result

help(myFunction)


```

## `input()`
If the prompt argument is present, it is written to standard output without a trailing newline. The function then reads a line from input, converts it to a string (stripping a trailing newline), and returns that ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
age = int(input('Enter your age: '))

print("Your age is: ", age)

```

## `len()`
Return the length (the number of items) of an object. The argument may be a sequence (such as a string, bytes, tuple, list, or range) or a collection (such as a dictionary, set, or frozen set) ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]

print("The List has ", len(listExample), " elements")

```

## `max()`
Return the *largest* item in an iterable or the largest of two or more arguments ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]

print(max(listExample), " is the largest element in the list")

```

## `min()`
Return the *smallest* item in an iterable or the smallest gest of two or more arguments ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]

print(min(listExample), " is the smallest element in the list")

```

## `pow()`
Return *base* to the power *exp*. The two-argument form `pow(base, exp)` is equivalent to using the power operator: `base**exp` ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
a = 5
b = 2

result = pow(a, b)
print(result)

```

## `print()`
Print objects to the text stream file, separated by sep and followed by end. sep, end, file, and flush, if present, must be given as keyword arguments ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
for x in range(6):
  print(x)

```

## `range()`
Rather than being a function, range is actually an immutable sequence type, as documented in Ranges and Sequence Types — list, tuple, range. ([Reference](https://docs.python.org/3/library/functions.html)).

With this function, we can create a sequence of numbers from 0 to 5, and print each item in the sequence:

```{code-block} python
:class: thebe
x = range(6)

for n in x:
  print(n)

```


## `reversed()`
Return a reverse iterator. seq must be an object which has a __reversed__() method or supports the sequence protocol (the __len__() method and the __getitem__() method with integer arguments starting at 0) ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]
print('Original list order', listExample)
print('--------')

print('Original list in reversed order', list(reversed(listExample)))

```

## `round()`
Return number rounded to ndigits precision after the decimal point. If ndigits is omitted or is None, it returns the nearest integer to its input ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
numberRound = 3.1416
print('Original numberRound value: ', numberRound)

print('Round numberRound value: ', round(numberRound))

```


## `str()`
Return a `str` version of object ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
number = 3.1416
print('Type of number', type(number))

stringNumber= str(number)

print('Type of stringNumber', type(stringNumber))

```

## `sum()`
Sums start and the items of an iterable from left to right and returns the total. The iterable’s items are normally numbers, and the start value is not allowed to be a string ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
listExample = [1, 2, 3, 4, 5]
print('The sum of the items in the listExample is: ', sum(listExample))

```

## `type()`
With one argument, return the type of an object ([Reference](https://docs.python.org/3/library/functions.html)).

```{code-block} python
:class: thebe
name = "Roderick"
print('The type of data stored in the variable name is: ', type(name))

```

---

Even that these are the only built-in functions, they are not the only way to interact with Python. We can visit the official repository of Python ([PyPi](https://pypi.org/)) and download other useful libraries and take advantage of modules and libraries that other members of the Python community had developed. Then, using the `import` statement we can import these modules into our code. Some of these libraries, such as NumPy, SciPy and Matplotlib, provide useful data analysis tools for scientists and engineers. These libraries can be used to analyze, graph and visualize data. They can also be used to create complex mathematical equations and 3D animations.
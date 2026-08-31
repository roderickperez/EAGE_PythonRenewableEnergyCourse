---
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Functions

So far we have learned to perform simple operations, but this is not convenient if we want to evaluate the previous expression with different values. For that we define a function, which is a code that I can use as many times in my code.

In any programming language a function is a block of code that we can define once, and call it as many times as we want during the execution of our program. Inside the function, we can define variables (known as parameters or arguments) and use them to store data. This parameters are specified after the function name, inside the parentheses. Depeding of the function, you can add as many arguments as you want, just separate them with a comma. However, if we define a function with $n$ parameters, at the time of the execution Python expect that we pass the same number of $n$ arguments into it. If we pass a different number of arguments, Python will raise an error.

Python uses the keyword `def` to define a function. The function name is followed by its parameters and a colon (`:`). Names cannot contain spaces and normally follow the lowercase `snake_case` convention, although an uppercase first letter is syntactically valid. A function may use `return` to provide a value; without an explicit `return`, Python returns `None`. Parameters are the names in the definition, while arguments are the values supplied when the function is called. Clear docstrings make functions easier to understand and reuse.

For example:

```python
def myFunction(parameters):
    """ Documentation
    This is the documentation of the myFunction, explaining the functionallity of it.
    """
    # code
    return result
```

After the function is defined, we can call it by using the name of the function, and passing the values of the parameters. For example:

```python
myFunction(parameters)
```



Now, let's define our first function. This function takes two parameters, `x` and `y`, and returns the sum of these two parameters.

```{code-cell} python
def myFunction(x, y):
    """
    Sum the inputs x and y and return the c value
    """
    z = x + y
    return z
```

Now, we can call this function and pass the values of `x` and `y` as arguments. Change the value of `x` and `y` and see the result.

```{code-cell} python
x = 5
y = 4
myFunction(x, y)
```

## Default Parameter Value
Sometimes, we want to define a funtion with default parameters. For that, at the moment to define the function we can specify the default value of the parameters. For example:

```{code-cell} python
def functionDefaultParameter(x = 5, y = 4):
    """
    Sum the inputs x and y, using as a default values x = 5 and y = 4, and return the c value
    """
    z = x + y
    return z
```

Now, when we call the function inside our program we don't need to specify a specific value for the parameters, since it will use the default value.

```{code-cell} python
functionDefaultParameter()
```

However, notice that we can also change the values of the default parameters.
```{code-cell} python
functionDefaultParameter(1, 6)
```


:::{admonition} Help function
:class: tip
In IPython and Jupyter, appending `?` displays help for an object. Standard Python code should use `help(function_name)` instead.

Also, by using the combination of Shift and Tab on your keyboard, you can see the documentation associated with it.

```{code-cell} python
functionDefaultParameter?
```
:::


```{admonition} Exercise 2
Knowing that:

$$°F = °C \times \frac{9}{5} + 32$$

Write a program that converts degrees *Celsius* to degrees *Fahrenheit*, and displays the results (using the `print()` function) of $°C$ and $°F$, in the case of 20° C.
```

### User Manual Character Entry
A very effective way to interact with the user of our program is through a *built-in* function called `input()`, which is able to read what the user inputs and return a `str`.
```{code-cell} python
input("What's your name?")
```

And in turn we can assign this to a variable:
```{code-cell} python
name = input("What's your name?")
print(name)
```

:::{admonition} Exercise 3
Continuing with our previous exercise, create a program in which the user can manually enter the value of degrees Celsius, and can know its correspondence in Fahrenheit.

The result should look something like:
```python
45 degrees Celsius equals 113 degrees Fahrenheit
```
:::

## Lambda Functions

A Python **Lambda function** is a small anonymous function which behaves like a normal function in regard to arguments. It can take any number of arguments, but can only have one expression. Therefore, a lambda parameter can be initialized with a default value: the parameter n takes the outer n as a default value. 

```python
lambda arguments : expression
```

For example, if we want to add 10 to the parameter `a`, and return the result:

```{code-cell} python
x = lambda a : a + 10
print(x(5))
```

:::{admonition} Why Use Lambda Functions?
The power of lambda is better shown when you use them as an anonymous function inside another function.

Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:


```python
def myfunc(n):
  return lambda a : a * n
```
:::

:::{admonition} Exercise 4
Using the previous syntax, define a (lambda) function to make a function that always doubles the number you input in:
:::

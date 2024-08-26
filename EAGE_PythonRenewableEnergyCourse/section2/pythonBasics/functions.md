# Functions

So far we have learned to perform simple operations, but this is not convenient if we want to evaluate the previous expression with different values. For that we define a function, which is a code that I can use as many times in my code.

In any programming language a function is a block of code that we can define once, and call it as many times as we want during the execution of our program. Inside the function, we can define variables (known as parameters or arguments) and use them to store data. This parameters are specified after the function name, inside the parentheses. Depeding of the function, you can add as many arguments as you want, just separate them with a comma. However, if we define a function with $n$ parameters, at the time of the execution Python expect that we pass the same number of $n$ arguments into it. If we pass a different number of arguments, Python will raise an error.

Python has a *reserve* word for functions, which is `def`. Following the `def` keyword, we define the **name of the function**, and then we define the parameters of the function, followed by the column symbols (`:`). It is important to note that the function name must not start with capital letters and without space. It is recommended that all functions in Python have documentation. That way you or other users can read and understand what this specific function does and reuse it. Finally, the function needs to return a value, it will be specified using the `return` keyword.

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


```{thebe-button}
```

Now, let's define our first function. This function takes two parameters, `x` and `y`, and returns the sum of these two parameters.

```{code-block} python
:class: thebe
def myFunction(x, y):
    """
    Sum the inputs x and y and return the c value
    """
    z = x + y
    return z
```

Now, we can call this function and pass the values of `x` and `y` as arguments. Change the value of `x` and `y` and see the result.

```{code-block} python
:class: thebe
x = 5
y = 4
myFunction(x, y)
```

## Default Parameter Value
Sometimes, we want to define a funtion with default parameters. For that, at the moment to define the function we can specify the default value of the parameters. For example:

```{code-block} python
:class: thebe
def functionDefaultParameter(x = 5, y = 4):
    """
    Sum the inputs x and y, using as a default values x = 5 and y = 4, and return the c value
    """
    z = x + y
    return z
```

Now, when we call the function inside our program we don't need to specify a specific value for the parameters, since it will use the default value.

```{code-block} python
:class: thebe
functionDefaultParameter()
```

However, notice that we can also change the values of the default parameters.
```{code-block} python
:class: thebe
functionDefaultParameter(1, 6)
```


:::{admonition} Help function
:class: tip
It's impossible to know what all functions written in Python do, but one of the most useful insights in Python is to add the `?` symbol after a function you can also view the documentation (Docstring) of the function.

Also, by using the combination of Shift and Tab on your keyboard, you can see the documentation associated with it.

```{code-block} python
:class: thebe
functionDefaultParameter?
```
:::


```{admonition} Exercise 2
Knowing that:

$$°C = \left ( °F - 32 \right )\times \frac{5}{9} $$

Write a program that converts degrees *celcius* to degrees *Farenheit*, and displays the results (using the `print()` function) of $°C$ and $°F$, in the case of 20° c.
```

### User Manual Character Entry
A very effective way to interact with the user of our program is through a *built-in* function called `input()`, which is able to read what the user inputs and return a `str`.
```{code-block} python
:class: thebe
input("What's your name?")
```

And in turn we can assign this to a variable:
```{code-block} python
:class: thebe
name = input("What's your name?")
print(name)
```

:::{admonition} Exercise 3
Continuing with our previous exercise, create a program in which the user can manually enter the value of degrees Celcius, and can know its correspondence in Fahrenheit.

The result should look something like:
```python
45 degrees Celsius equals -228.15 degrees Fahrenheit
```
:::

## Lambda Functions

A Python **Lambda function** is a small anonymous function which behaves like a normal function in regard to arguments. It can take any number of arguments, but can only have one expression. Therefore, a lambda parameter can be initialized with a default value: the parameter n takes the outer n as a default value. 

```python
lambda arguments : expression
```

For example, if we want to add 10 to the parameter `a`, and return the result:

```{code-block} python
:class: thebe
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
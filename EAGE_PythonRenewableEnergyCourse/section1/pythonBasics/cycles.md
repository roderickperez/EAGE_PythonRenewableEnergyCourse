# Control Structures

## Cycle

Structures that allow me to execute sections of code based on a condition. This conditions are based on Conditional Statements and Logical Operators.

**Conditional Statements**

Conditionals are handled as questions, and return booleans.

* Same as `==`
* Not equal to `!=`
* Greater than `>`
* Less than `<`
* Greater than or equal to `>=`
* Less than or equal to `<=`

**Logical Operators**

In the same way as the Conditional Statements, logical operators are used to combine multiple conditions.
* And `and`
* Or `or`
* Not `not`

## if-elif-else Statements

### if

If cycle are control structures that allow to condition the execution of one blocks of code.

```python
if condition:
    run this section of code
```

:::{admonition} Indentation
:class: warning
Python relies on indentation (whitespace at the beginning of a line) to define scope in the code. Other programming languages often use curly-brackets for this purpose.
:::


```{thebe-button}
```

```{code-block} python
:class: thebe
a = 10

if a == 10:
    print('a is equal to 10')
```

:::{admonition} `if` | Short Version
:class: tip, dropdown

```python
if a > b: print("a is greater than b")
```
:::


:::{admonition} `pass` Statement
:class: tip

`if` statements cannot be empty, but `if` you for some reason have an `if` statement with no content, put in the `pass` statement to avoid getting an error.

```python
a = 33
b = 200

if b > a:
  pass
```
:::


### elif
When we use the keyword `elif` we build a statement that says: *if the previous conditions were not true, then try this condition*.

```{code-block} python
:class: thebe
age = int(input('Enter your age: '))

if age < 18:
    print("You are a minor")
elif age < 0:
    print("You can't have a negative age")
```

:::{admonition} `elif` | Statement Short Version
:class: tip, dropdown

```{code-block} python
:class: thebe
a = 2
b = 330
print("A") if a > b else print("B")
```
:::

### else

Now, if we want to catches anything which isn't caught by the preceding conditions, we can use the `else` keyword.

```{code-block} python
:class: thebe
age = int(input('Enter your age: '))

if age < 18:
    print("You are a minor")
elif age < 0:
    print("You can't have a negative age")
else:
    print('You are of legal age')
```

:::{admonition} Tip
:class: tip, dropdown
We can also use `else` without the `elif`.
:::

:::{admonition} `if - else` | Short Version
:class: tip, dropdown

```python
a = 2
b = 330
print("A") if a > b else print("B")
```
:::

:::{admonition} Multiple `else` statements in the same line
:class: tip, dropdown

```{code-block} python
:class: thebe
a = 330
b = 330
print("A") if a > b else print("=") if a == b else print("B")
```
:::

## Logical Operators

### `and`
The `and` keyword is a logical operator, `and` is used to combine conditional statements. For example, if we want to test if `a` is greater than `b`, `AND` if `c` is greater than `a`:

```{code-block} python
:class: thebe
a = 200
b = 33
c = 500
if a > b and c > a:
  print("Both conditions are True")
```

### `or`
Similar than the `and` keyword, the `or` keyword is is a logical operator used to combine conditional statements. In this case, if we want test if `a` is greater than `b`, `OR` if `a` is greater than `c`:

```{code-block} python
:class: thebe
a = 200
b = 33
c = 500
if a > b or a > c:
  print("At least one of the conditions is True")

```

```{admonition} Exercise 7
Using the conditional statements and logical operators explained in this section, write a program that based on the monthly production rates prints out the if a hydrocarbon field is :
* Excellent (greater than or equal to 1,000,000)
* Good (between 100,000 and 999,999)
* Poor (less than 100,000)
* Close (no production)

after the user input the monthly production rates manually.
```

## Loops

### for

A `for` loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string). This method is very convenient in case you want to repeat sequences of code. For example, if we wanted to print on the screen each of the elements that are stored in the following list, we can do the following:

```{code-block} python
:class: thebe
cities = ['London', 'Paris', 'Rome', 'Madrid', 'Berlin']

print(cities[0])
print(cities[1])
print(cities[2])
print(cities[3])
print(cities[4])

```

As we can see, this method can be quite inconvenient in the case of having long sequences. This can be simplified via the `for` keywork:

```{code-block} python
:class: thebe
cities = ['London', 'Paris', 'Rome', 'Madrid', 'Berlin']

for city in cities:
  print(city)

```

:::{admonition} Tip
:class: tip
Notice that in this example, the keyword `city` is used to store the value of the current element in the sequence, and wasn't defined before the loop.
:::

The `for` loop does not require an indexing variable to set beforehand.

### `break` and `continue` Statements

With the `break` statement we can stop the loop before it has looped through all the items:

```{code-block} python
:class: thebe
cities = ['London', 'Paris', 'Rome', 'Madrid', 'Berlin']

for city in cities:
  print(city)
  if city == "Rome":
    break
```

On the other hand, with the `continue` statement we can stop the current iteration of the loop, and continue with the next:

```{code-block} python
:class: thebe
cities = ['London', 'Paris', 'Rome', 'Madrid', 'Berlin']

for city in cities:
    if city == "Rome":
        continue
    print(city)
```

### `range()`
To loop through a set of code a specified number of times, we can use the `range()` function. It returns a sequence of numbers, starting from `0` by default, and increments by 1` (by default), and ends at a specified number.

```{code-block} python
:class: thebe

for i in range(10):
  print(i)
```

As we mentioned before, the `range()` function defaults to `0` as a starting value. Hwever it is possible to specify the starting value by adding a parameter: `range(2, 6)`, which means values from `2` to `6` (**but not including `6`**):

```{code-block} python
:class: thebe

for i in range(2, 6):
  print(i)
```

The `range()` function defaults to increment the sequence by `1`. However, it is possible to specify the increment value by adding a third parameter: 
```{code-block} python
:class: thebe

for i in range(2, 30, 3):
  print(i)
```

## Nested Loops

A nested loop is a loop inside a loop. The "inner loop" will be executed one time for each iteration of the "outer loop":

```{code-block} python
:class: thebe

cities = ['London', 'Paris', 'Rome', 'Madrid', 'Berlin']
countries = ['UK', 'France', 'Italy', 'Spain', 'Germany']

for city in cities:
  for country in countries:
    print(city, country)
```
# Arithmetic Operations

```{thebe-button}
```

## Summation

Add two variables `x` and `y`

```{code-block} python
:class: thebe
a = 10
b = 3

c = a + b

print(c)
```

## Substraction

Subtracts two variables `x` and `y`
```{code-block} python
:class: thebe
a = 10
b = 3

c = a - b

print(c)
```


## Multiplication
Multiply two variables `x` and `y`
```{code-block} python
:class: thebe
a = 10
b = 3

c = a * b

print(c)
```

## Division (float)
Divides the first operand (`x`) by the second (`y`)
```{code-block} python
:class: thebe
a = 10
b = 3

c = a / b

print(c)
```

## Floor division
Divides the first operand (`x`) by the second (`y`). It is used to find the floorof the quotient when first operand is divided by the second.
```{code-block} python
:class: thebe
a = 10
b = 3

c = a // b

print(c)
```

## Modulus
Returns the remainder when first operand (`x`) is divided by the second ( `y`)
```{code-block} python
:class: thebe
a = 10
b = 3

c = a % b

print(c)
```

## Power (Exponentiation)
Returns first variable (`x`) raised to power second (`y`)
```{code-block} python
:class: thebe
a = 10
b = 3

c = a ** b

print(c)
```

:::{admonition} Tip
:class: tip
Powers take precedence over multiplications and divisions.

Using negative exponents or decimals, inverse powers or nth roots can be calculated. Since, 

$$a^{-b} = \frac{1}{a^{b}}$$

```{code-block} python
:class: thebe
10 ** -4
```


:::

## Order of Operations (Priority rule)
When multiple operations appear in a formula, Python performs them by applying the usual rules of precedence of operations (**multiplication and division first, then addition and subtraction**).



:::{admonition} Print
:class: tip
For this reason, it is always recommended to use parentheses, for example:

```python
(5 + 8) / (7 - 2)
```
:::

:::{admonition} Secutive Operations
:class: tip
It is possible to write consecutive addition and subtraction, but it is not recommended to do so because it is not a common notation:

```python
3 + - + 4
```
What you cannot do is write multiplications and divisions in a row:
```python
3 * / 4
```

:::

## Complex numbers
Python can handle complex number calculations. The imaginary part is accompanied by the letter "j".
```python
1 + 1j
```
In the same way, basic operations can be done with complex numbers.

```{code-block} python
:class: thebe
a = 1 + 1j
b = 2 + 3j

print(a+b)
```

:::{admonition} Important
:class: warning
It is important that the letter "j" must always be accompanied by a number and attached to it. For example, if you try the following expression:

```python
1 + j
```
Python will return the following error
```python
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-24-4f555f6169c6> in <module>()
----> 1 1 + j

NameError: name 'j' is not defined
```
Therefore, it is preferable to use:

```python
1 + 1j
```

:::


:::{admonition} Tip
:class: tip
The result of an operation involving complex numbers is a complex number, even though the result has no imaginary part.

```{code-block} python
:class: thebe
c = 1j
c * c
```
:::


## Summary

```{table} Arithmetic Operators in Python
:name: arithmetic-Operators-Python

| Operator | Name | Description | Syntax |
|---|---|---|---|
| `+` | Summartion | Add two variables `x` and `y` | `x+y`|
| `-` | Substraction | Subtracts two variables `x` and `y` | `x-y`|
| `*` | Multiplication | Multiply two variables `x` and `y`| `x*y`|
| `/` | Division (float) | Divides the first operand (`x`) by the second (`y`)| `x/y`|
| `//` | Division (floor) | Divides the first operand (`x`) by the second (`y`)| `x//y`|
| `%` | Modulus | Returns the remainder when first operand (`x`) is divided by the second ( `y`)| `x%y`|
| `**` | Power  | Returns first variable (`x`) raised to power second (`y`)| `x**y`|

```

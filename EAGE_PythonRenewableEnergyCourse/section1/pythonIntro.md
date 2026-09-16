# Python Introduction

**Objectives**
* Provide an *overview* of Python.
---

## Python

A programming language expresses instructions with defined syntax and meaning. Python lets us turn renewable-energy questions into reproducible calculations, data checks and plots.

Python is an *interpreted* programming language created by Guido van Rossum and first released in 1991. In the standard CPython implementation, source code is compiled to bytecode and executed by the Python virtual machine. Python is also **dynamically typed**: names do not have fixed declared types, and type checking occurs at runtime. An object still has a definite type.

Performance depends on the task and implementation. Python-level loops can be slow for large numerical problems; NumPy performs many array operations in compiled code. First make a calculation correct and readable, then measure performance before optimising it.

### What can I do with Python?

The universe of applications and possibilities with Python is practically infinite. With Python can be used on a server to create web applications, can connect to database systems, it can also read and modify files, and it used to handle big data and perform complex mathematics. One of the main advantages of Python is that it is used for **rapid prototyping**, and can be scaled  for production-ready software development.

### Why Python?

In addition to its high versatility, another advantage of using Python is that works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc). Since it has a simple syntax (similar to the English language), allows developers to write programs with fewer lines than some other programming languages. Also, Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.

Python uses indentation to group statements into blocks, including function bodies, loops and conditions. A loop or an `if` block does not create a separate variable scope. Statements usually end at a newline, although an expression inside parentheses can span several lines.

### Download & Official Python Packages Repository

From [Python office website](https://www.python.org/) we can download the latest version of Python. Along the course we will see the necessity to extend Python basic capabilities with some extra and external *packages*. These packages are *modules* that are used to extend the functionality of Python. For example, the [numpy](https://www.numpy.org/) package is a package that provides a *numerical* library for Python. The official repository of software for the Python programming language is **Python Package Index** [PyPi](https://pypi.org/).

With a Package Manager we can install, upgrade, remove, etc., Python packages in any Python virtual environment. A virtual environment, it is an environment that can use different versions of package dependencies and Python. It is very useful to develop and test new code in a separate environment, since sometimes some packages may be incompatible with each other.

## Versions

Python 3 is the supported major version used in this course. Install a currently supported Python 3 release that is compatible with the course environment rather than relying on a hard-coded patch version. The official release list is available at [python.org/downloads](https://www.python.org/downloads/).

Python 2 reached end of life on January 1, 2020 and no longer receives security fixes. New work must use Python 3. See the [official Python 2 sunset notice](https://www.python.org/doc/sunset-python-2/).

## Package and a library
A module is an importable unit (often a `.py` file). A package organizes modules under an import namespace. “Library” is an informal term for reusable code and may contain one or many packages. Installable distributions are published on PyPI; distribution names and import names need not match.

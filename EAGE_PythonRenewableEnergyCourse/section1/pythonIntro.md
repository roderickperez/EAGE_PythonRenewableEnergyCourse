# Python Introduction

**Objetives**
* Provide an *overview* of Python.
---

## Python 

Programming language is the language we use to *talk* with the machine (0's and 1's). In this Programming language we need to write the instructions that are going to be executed by the machine in a precise *structure* form (*syntax*), with a precise *meaning* (*semantic*). **Python** is one of many *programming languages​​* that we have right know in the market.

Python is an *interpreted* programming language created by Guido van Rossum and first released in 1991. In the standard CPython implementation, source code is compiled to bytecode and executed by the Python virtual machine. Python is also **dynamically typed**: names do not have fixed declared types, and type checking occurs at runtime. An object still has a definite type.

Since it is an *interpreted* language, the code is interpreted during run time rather than being compiled to native code hence it is a bit **slower**. When we run a Python script this pieces of instructions are first compiled into *Byte Code*. Then, this *Byte Code* is then interpreted and executed by the PVM (Python Virtual Machine). However, the computational power of current machines is very high, that for regular and common scripts this weakness is practically imperceptible. 

### What can I do with Python?

The universe of applications and possibilities with Python is practically infinite. With Python can be used on a server to create web applications, can connect to database systems, it can also read and modify files, and it used to handle big data and perform complex mathematics. One of the main advantages of Python is that it is used for **rapid prototyping**, and can be scaled  for production-ready software development.

### Why Python?

In addition to its high versatility, another advantage of using Python is that works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc). Since it has a simple syntax (similar to the English language), allows developers to write programs with fewer lines than some other programming languages. Also, Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.

Python was designed for *readability*, and has some similarities to the English language with influence from mathematics. It is important to keep in mind that Python uses new lines to complete a command (as opposed to other programming languages which often use semicolons or parentheses). You will notice that Python relies on indentation, using whitespace, to define scope; such as the scope of loops, functions and classes. Other programming languages often use curly-brackets for this purpose.

### Download & Official Python Packages Repository

From [Python office website](https://www.python.org/) we can download the latest version of Python. Along the course we will see the necessity to extend Python basic capabilities with some extra and external *packages*. These packages are *modules* that are used to extend the functionality of Python. For example, the [numpy](https://www.numpy.org/) package is a package that provides a *numerical* library for Python. The official repository of software for the Python programming language is **Python Package Index** [PyPi](https://pypi.org/).

With a Package Manager we can install, upgrade, remove, etc., Python packages in any Python virtual environment. A virtual environment, it is an environment that can use different versions of package dependencies and Python. It is very useful to develop and test new code in a separate environment, since sometimes some packages may be incompatible with each other.

## Versions

Python 3 is the supported major version used in this course. Install a currently supported Python 3 release that is compatible with the course environment rather than relying on a hard-coded patch version. The official release list is available at [python.org/downloads](https://www.python.org/downloads/).

Python 2 reached end of life on January 1, 2020 and no longer receives security fixes. New work must use Python 3. See the [official Python 2 sunset notice](https://www.python.org/doc/sunset-python-2/).

## Package and a library
**Packages** are a set of modules that contain scripts and functions. You can write your own modules and packages, and then distributed under *PyPi*. When many packages come together, they build **libraries**. A package manager also manages the libraries because libraries are the collections of packages.

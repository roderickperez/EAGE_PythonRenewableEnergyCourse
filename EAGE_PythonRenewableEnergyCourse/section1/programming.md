# Introduction to Programming

This course teaches you to turn an energy question into explicit steps, executable Python, and checks that another person can repeat.

## Program, algorithm and data

A **program** is an implementation of instructions in a programming language. An **algorithm** is a stated procedure for solving a problem; its implementation must define inputs, operations, decision rules and outputs. Here the procedure is: read interval-average power, check its units and duration, calculate energy, and compare against an expected result. [Python tutorial: introduction](https://docs.python.org/3/tutorial/introduction.html).

A numerical result depends on both code and assumptions. For example, $E=P\Delta t$ uses power in kW and duration in hours to obtain kWh; it does not accept a cumulative energy meter reading as its power input. [@wade2003], Chapter 2; [@jica2011], §3.1.2.

```{code-cell} python
power_kw = 2.5
duration_h = 4.0
if power_kw < 0 or duration_h <= 0:
    raise ValueError("This example requires nonnegative generation and positive duration.")
energy_kwh = power_kw * duration_h
print(f"Energy: {energy_kwh:.1f} kWh")
assert energy_kwh == 10.0
```

## Syntax, runtime errors and scientific errors

A **syntax error** prevents Python from parsing a statement. A **runtime exception** occurs while otherwise valid code executes, for example division by zero. A **scientific or logic error** can produce a plausible number without raising an exception, such as confusing kW with MW. Tracebacks identify where exceptions arise; physical checks and independent calculations help find errors in meaning. [Python errors and exceptions](https://docs.python.org/3/tutorial/errors.html); unit basis: [NIST SI guide](https://www.nist.gov/pml/special-publication-811).

For the original example above, change one input at a time, predict the output, run the cell, and explain any mismatch. Record assumptions next to the calculation. Later lessons add arrays, files, reusable functions and plots.

## Source code, interpreter and environment

**Source code** is the program text. The **interpreter** executes Python code; standard CPython first compiles it to bytecode. The **environment** includes the interpreter and installed packages used for a particular project. Selecting a notebook kernel selects the process and environment in which its cells execute. [Python glossary](https://docs.python.org/3/glossary.html); [Jupyter notebook components](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).

Continue with [Python introduction](pythonIntro.md) and [choose an IDE or notebook](IDE.md).

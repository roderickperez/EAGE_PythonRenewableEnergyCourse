# Course Summary

This page provides a concise overview of each section covered in the course,
the key skills practised, and the Python tools introduced.

---

## Section-by-Section Overview

### Section 0 — Introduction to Renewable Energy

An overview of the global energy landscape, the case for renewables, and
an introduction to how Python can support energy engineers and geoscientists.

**Key topics:** Energy transition, renewable resource types, Python as an engineering tool.

---

### Section 2 — Python Basics

Core Python programming skills essential for all subsequent modules.

| Topic | Concepts |
|-------|----------|
| Variables & Data Types | `int`, `float`, `str`, `bool` |
| Math Operations | Arithmetic, exponentiation, modulo |
| Functions | `def`, `return`, arguments, scope |
| Data Structures | Lists, tuples, dictionaries |
| Cycles (Loops) | `for`, `while`, `range()` |
| Integrated Functions | `len()`, `sum()`, `min()`, `max()`, `print()` |

---

### Section 3 — NumPy

The numerical computing foundation of scientific Python.

| Topic | Key Functions |
|-------|---------------|
| Array Creation | `np.array`, `np.arange`, `np.linspace`, `np.zeros` |
| Indexing & Slicing | Boolean indexing, fancy indexing |
| Mathematical Ops | `np.sum`, `np.mean`, `np.std`, element-wise arithmetic |
| Reshaping | `reshape()`, broadcasting rules |
| Random & Statistics | `np.random.normal`, `np.percentile` |

---

### Section 4 — Pandas

Tabular data manipulation for real-world datasets.

| Topic | Key Operations |
|-------|----------------|
| DataFrame Creation | `pd.DataFrame()` from dict or CSV |
| Selection | Column selection, `.loc[]`, `.iloc[]` |
| Filtering | Boolean conditions |
| GroupBy | `.groupby().agg()` |
| Transformations | Adding columns, `.apply()` |
| Summary Stats | `.describe()`, `.sort_values()` |

---

### Section 5 — Matplotlib

Data visualisation for scientific and engineering communication.

| Plot Type | Use Case |
|-----------|----------|
| Line plot | Time series, power curves |
| Bar chart | Comparisons across categories |
| Scatter plot | Correlations between variables |
| Histogram | Frequency distributions |
| Subplots | `plt.subplots()` for multi-panel figures |

---

### Section 6 — Renewable Energy Modules

Applied physics and Python calculations for four energy technologies.

#### Wind Energy
- Wind power formula: $P = \frac{1}{2} \rho A v^3$
- Betz limit: $C_{P,max} = 16/27 \approx 0.593$
- Power curves, Weibull wind speed distributions
- Wake effects and annual energy production (AEP)

#### Solar Energy
- PV panel power: $P_{DC} = G \cdot A \cdot \eta$
- Temperature derating with temperature coefficient $\gamma$
- Peak sun hours, system efficiency, capacity factor

#### Hydroelectric Energy
- Hydro power: $P = \eta \rho g Q H$
- Reservoir potential energy, flow-rate sensitivity
- Annual generation from capacity factor

#### Geothermal Energy
- Geothermal gradient and heat flow ($Q = \lambda \nabla T$)
- Reservoir temperature estimation
- Plant efficiency and net power output

---

### Section 7 — Quizzes

Self-assessment quizzes covering all course topics:

- **Python Basics Quiz** — variables, operators, conditionals, loops, functions
- **NumPy Quiz** — array operations, statistics, broadcasting
- **Pandas Quiz** — DataFrames, filtering, groupby, visualisation
- **Matplotlib Quiz** — line plots, bar charts, scatter plots, subplots
- **Wind Energy Quiz** — power formula, Betz limit, Weibull, power curves
- **Solar Energy Quiz** — irradiance, temperature derating, capacity factor
- **Hydroelectric Quiz** — power formula, flow rate, annual energy
- **Eolic Energy Quiz** — wind resource assessment, shear, wake effects
- **Exercises Review** — integrated Python exercises covering all foundations

---

### Section 8 — Course Final Project

An integrated project that applies all course skills to a real-world renewable energy scenario,
combining data loading, analysis, visualisation, and engineering calculations.

---

### Section 9 — Summary and Conclusions

- **This page** — section-by-section review and skills matrix.
- **[Course Conclusions](courseConclusions.md)** — reflections, key takeaways, and next learning steps.

---

## Python Libraries Used in This Course

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and mathematics |
| `pandas` | Tabular data and data frames |
| `matplotlib.pyplot` | Data visualisation |
| `math` | Standard mathematical functions |
| `scipy` | Scientific computing (supplementary) |

---

## Skills Matrix

By completing this course you should be able to:

- [x] Write Python scripts to solve engineering equations
- [x] Load and manipulate datasets with Pandas
- [x] Perform vectorised numerical computations with NumPy
- [x] Create clear, labelled plots with Matplotlib
- [x] Calculate wind, solar, hydro, and geothermal power outputs
- [x] Interpret capacity factors and annual energy production
- [x] Use statistical tools to characterise wind and solar resources
- [x] Apply Python in a Jupyter / JupyterBook interactive environment

# Markdown and mathematical notation

**Objective:** explain Python work with readable text, units, equations and references.

A notebook combines code cells with Markdown cells. Use Markdown for the question,
assumptions and interpretation; use code for calculations. Run a Markdown cell to
render it. Double-click it to edit its source.

## Essential formatting

```markdown
# Main heading
## Section heading
### Subsection heading

**Bold text** and *italic text*.
Use `power_kw` for a variable name within a sentence.

- State the input values and units.
- Explain the result in words.

1. Read the data.
2. Check missing values.
3. Calculate and interpret.

[Python tutorial](https://docs.python.org/3/tutorial/)
![Description of the chart](images/generation.png)
```

The image path is an example: replace it with your chart location.
Leave a blank line before lists and between paragraphs. A fenced block labelled
`python` displays code; a notebook code cell actually executes Python.

## Tables with units

```markdown
| Quantity | Value | Unit |
|---|---:|---|
| Flow rate | 2.0 | m³/s |
| Net head | 30 | m |
| Efficiency | 0.85 | dimensionless |
```

## Equations

Put inline mathematics between single dollar signs: `$y = mx + b$` renders
$y = mx + b$, the equation of a straight line. Use double dollar signs on separate
lines for a displayed equation:

```text
$$
P = \rho g Q H \eta
$$
```

$$
P = \rho g Q H \eta
$$

For hydropower, density $\rho$ in kg/m³, gravity $g$ in m/s², flow $Q$ in m³/s,
net head $H$ in m and efficiency $\eta$ as a fraction give power $P$ in watts.
Mathematical notation explains a model; it does not execute it. In Python write
`power_w = density * gravity * flow * head * efficiency`.

## A useful exercise write-up

Include the question, inputs and units, assumptions, code, result, a reasonableness
check, and a reference. Explain whether data are measured or synthetic and whether
the model is suitable only for teaching.

**References:** [MyST Markdown guide](https://mystmd.org/guide/quickstart),
[Python tutorial](https://docs.python.org/3/tutorial/), and the hydropower sources
in the [reference catalogue](../references.md).

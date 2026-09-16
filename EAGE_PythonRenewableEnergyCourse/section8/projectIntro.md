# Final Project: Design and Analyse a Hybrid Renewable Portfolio

## Purpose

The course goal is to combine **basic Python** with **renewable-energy reasoning**. In this project you will turn hourly solar, wind, river-flow, geothermal, and demand inputs into an auditable portfolio analysis. You are not expected to perform a bankable resource assessment. You are expected to use correct equations, units, functions, data checks, aggregation, and plots.

## Scenario

A small isolated grid is considering four renewable resources:

- a PV array with a DC rating and an AC inverter limit;
- a wind plant described by cut-in, rated, and cut-out speeds;
- a run-of-river plant with environmental flow and a generator rating;
- a geothermal well field with a gross conversion efficiency and parasitic load.

Analyse one synthetic week at hourly resolution. The synthetic weather and demand make every submission reproducible and avoid implying that the results describe a real site.

## Required engineering models

Use these simplified boundaries consistently:

### Solar PV

$$T_c=T_a+\frac{G}{800}(NOCT-20)$$

$$P_{DC}=P_{DC,r}\frac{G}{1000}[1+\gamma_P(T_c-25)]$$

$$P_{AC}=\min(P_{DC}\eta_{inv},P_{AC,r})$$

### Wind

Use a four-region turbine curve: zero below cut-in, a continuous ramp to rated power, rated power until cut-out, and zero at or above cut-out. Do not extend $v^3$ through the rated and shutdown regions.

### Hydropower

$$Q_{usable}=\max(Q_{river}-Q_{environmental},0)$$

$$P_h=\min(\eta\rho gQ_{usable}H_{net},P_{rated})$$

### Geothermal

$$P_{th}=\dot m c_p(T_p-T_r)$$

$$P_{net}=P_{th}\eta_{conv}(1-f_{parasitic})$$

The geothermal formula is a simplified single-phase sensible-heat model. State that an enthalpy model is needed for flashing or two-phase fluid.

## Python requirements

Your executed notebook must demonstrate:

1. Variables with meaningful names and units.
2. Lists or dictionaries for technology parameters.
3. At least four reusable functions—one for each source.
4. Conditionals for physical boundaries and equipment limits.
5. NumPy arrays and vectorized calculations.
6. A pandas DataFrame with a unique hourly index.
7. Loops or comprehensions for at least one scenario comparison.
8. At least three well-labelled Matplotlib plots.
9. Assertions or explicit validation checks.
10. A concise interpretation of results and limitations.

## Required outputs

### A. Data-quality summary

Report row count, timestamp uniqueness, missing values, non-negative resource checks, direction range if direction is added, and the exact time step. Never replace missing data with zero without a stated rule.

### B. Technology functions

Each function must:

- document inputs and output units;
- accept array input where useful;
- reject impossible values;
- return power in MW;
- respect its physical and equipment limits.

### C. Portfolio metrics

Calculate:

- energy by source in MWh;
- source capacity factors using the correct rated-power denominator;
- total renewable energy available;
- demand energy;
- renewable energy served, curtailment, and shortfall;
- renewable coverage $E_{served}/E_{demand}$.

At every hour:

$$E_{served}=\min(E_{renewable},E_{demand})$$

$$E_{curtailed}=\max(E_{renewable}-E_{demand},0)$$

$$E_{shortfall}=\max(E_{demand}-E_{renewable},0)$$

For one-hour intervals, the numerical sum of MW samples is MWh. If you change the resolution, multiply by the time-step duration.

### D. Visuals

Produce at least:

1. resource inputs over time;
2. source power and total renewable power over time;
3. demand versus renewable supply, with shortfall or curtailment visible;
4. an energy-by-source or capacity-factor comparison.

Axes, legends, time period, and units are mandatory. Avoid dual axes unless the quantities cannot be compared otherwise and both scales are explicit.

### E. Scenario analysis

Change at least two design parameters—for example PV DC/AC ratio, wind capacity, environmental flow, or geothermal parasitic fraction. Use a loop or comprehension, create a comparison plot, and explain the trade-off. A better scenario should not be defined only as “more installed capacity.”

## Required validation

- Prove that the hourly index is unique and chronological.
- Assert that every source power is non-negative and does not exceed its rating.
- Reconcile total renewable power with the sum of source columns.
- Reconcile served + curtailed with renewable energy, and served + shortfall with demand.
- Confirm $0\le CF\le1$ and $0\le$ renewable coverage $\le1$.
- Check at least one hand-calculated hour against the function result.

## Suggested notebook structure

1. Goal, scope, and model limitations
2. Imports and parameters
3. Synthetic input data
4. Data-quality checks
5. Technology functions and unit tests
6. Hourly generation model
7. Portfolio metrics
8. Visualizations
9. Scenario comparison
10. Conclusions and limitations

## Assessment rubric

| Criterion | Weight |
|---|---:|
| Renewable-energy concepts, equations, and model boundaries | 25% |
| Python functions, readability, and reuse | 20% |
| Units, aggregation, and numerical correctness | 15% |
| Data quality, assertions, and reconciliation | 15% |
| Plots and interpretation | 15% |
| Scenario reasoning and limitations | 10% |

The project is complete only when the notebook runs from the first cell to the last in a fresh kernel. A fully executable reference implementation follows in the solution guide.

## References

Hydropower and environmental flow [@ifc2015; @jica2011]; PV performance [@foster2010]; wind and hybrid systems [@manwell2009]; geothermal models [@grant2011]; multi-energy integration [@hu2026].

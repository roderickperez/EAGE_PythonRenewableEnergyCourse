# General Python Quiz

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its practice solution.

:::{admonition} Quiz 1 — Types
:class: note

**Difficulty:** Basic

What are the types of `3`, `3.0`, and `"3"`?

:::

:::{admonition} Solution
:class: tip, dropdown

`int`, `float`, and `str`, respectively.

:::

:::{admonition} Quiz 2 — Division
:class: note

**Difficulty:** Basic

What do `7 / 2` and `7 // 2` return?

:::

:::{admonition} Solution
:class: tip, dropdown

`3.5` and `3`; `/` is true division and `//` is floor division.

:::

:::{admonition} Quiz 3 — List indexing
:class: note

**Difficulty:** Basic

For `values = [10, 20, 30]`, what are `values[0]` and `values[-1]`?

:::

:::{admonition} Solution
:class: tip, dropdown

`10` and `30`. Python indexes begin at zero; negative indexes count from the end.

:::

:::{admonition} Quiz 4 — Units
:class: note

**Difficulty:** Basic

Convert 3 MW sustained for 2 hours to MWh with Python.

:::

:::{admonition} Solution
:class: tip, dropdown

`3 * 2` gives 6 MWh. Energy is power multiplied by duration.

:::

:::{admonition} Quiz 5 — Condition
:class: note

**Difficulty:** Medium

Write a condition that is true when `power` is at least zero and no greater than `rating`.

:::

:::{admonition} Solution
:class: tip, dropdown

`0 <= power <= rating`.

:::

:::{admonition} Quiz 6 — Loop
:class: note

**Difficulty:** Medium

How can you compute the squares of the values in `[1, 2, 3]`?

:::

:::{admonition} Solution
:class: tip, dropdown

`[x**2 for x in [1, 2, 3]]` gives `[1, 4, 9]`.

:::

:::{admonition} Quiz 7 — Functions
:class: note

**Difficulty:** Medium

Write a function returning MWh from MW and hours.

:::

:::{admonition} Solution
:class: tip, dropdown

`def energy_mwh(power_mw, hours): return power_mw * hours`.

:::

:::{admonition} Quiz 8 — Dictionary
:class: note

**Difficulty:** Medium

Get the value for `"wind"` from `d = {"wind": 12, "solar": 8}`; what happens with a missing key?

:::

:::{admonition} Solution
:class: tip, dropdown

`d["wind"]` is 12. A missing key raises `KeyError`; `d.get("hydro")` returns `None` by default.

:::

:::{admonition} Quiz 9 — Slicing
:class: note

**Difficulty:** Medium

What is `[2, 4, 6, 8][1:3]`?

:::

:::{admonition} Solution
:class: tip, dropdown

`[4, 6]`: the start is included and the stop is excluded.

:::

:::{admonition} Quiz 10 — Mean
:class: note

**Difficulty:** Medium

Compute the mean of `[2, 4, 6]` without an external library.

:::

:::{admonition} Solution
:class: tip, dropdown

`sum(values) / len(values)` gives 4 for this nonempty list.

:::

:::{admonition} Quiz 11 — Input validation
:class: note

**Difficulty:** Medium

Why should a function reject a negative duration before calculating energy?

:::

:::{admonition} Solution
:class: tip, dropdown

A negative duration is invalid for this model. Raise `ValueError` instead of silently returning a misleading energy value.

:::

:::{admonition} Quiz 12 — Missing data
:class: note

**Difficulty:** Medium

Why is a missing sensor measurement different from zero?

:::

:::{admonition} Solution
:class: tip, dropdown

Missing means unknown; zero is a measured value. Replacing missing readings with zero can bias totals and averages.

:::

:::{admonition} Quiz 13 — Weighted average
:class: note

**Difficulty:** Hard

Compute mean MW for `[1, 3]` MW over `[1, 3]` hours.

:::

:::{admonition} Solution
:class: tip, dropdown

`(1*1 + 3*3)/(1+3)` gives 2.5 MW; durations are the weights.

:::

:::{admonition} Quiz 14 — Exceptions
:class: note

**Difficulty:** Hard

When is `try`/`except ValueError` useful for numeric input?

:::

:::{admonition} Solution
:class: tip, dropdown

It handles an invalid conversion, such as `float("abc")`, while allowing the program to report a clear error.

:::

:::{admonition} Quiz 15 — Mutation
:class: note

**Difficulty:** Hard

What is the difference between `b = a` and `b = a.copy()` for a list?

:::

:::{admonition} Solution
:class: tip, dropdown

`b = a` shares the same list. `a.copy()` creates a separate shallow list; nested objects may still be shared.

:::

:::{admonition} Quiz 16 — Alignment
:class: note

**Difficulty:** Hard

Why check timestamps before adding two time series?

:::

:::{admonition} Solution
:class: tip, dropdown

Rows may represent different times, missing intervals, or time zones. Align on validated timestamps and inspect missing matches before adding.

:::

:::{admonition} Quiz 17 — Duplicate keys
:class: note

**Difficulty:** Hard

How can pandas identify duplicate pairs of `site` and `time` in a DataFrame?

:::

:::{admonition} Solution
:class: tip, dropdown

`df.duplicated(["site", "time"], keep=False)` flags all rows in duplicated pairs.

:::

:::{admonition} Quiz 18 — Time integration
:class: note

**Difficulty:** Hard

Given average MW and unequal interval durations, how do you obtain MWh?

:::

:::{admonition} Solution
:class: tip, dropdown

Multiply each interval-average power by its duration in hours and sum; do not sum MW without a time factor.

:::

:::{admonition} Quiz 19 — Function boundaries
:class: note

**Difficulty:** Hard

How would you test a clipping function at 0, its rating, and above its rating?

:::

:::{admonition} Solution
:class: tip, dropdown

Include assertions at each boundary and just below/above it; check that output remains between zero and the rating under the stated rules.

:::

:::{admonition} Quiz 20 — Reproducibility
:class: note

**Difficulty:** Hard

Name three things to record so another person can reproduce an analysis.

:::

:::{admonition} Solution
:class: tip, dropdown

Record input provenance/version, environment and dependencies, and the code plus assumptions (including random seed where applicable).

:::

## Further reading

[Python tutorial](https://docs.python.org/3/tutorial/); [pandas user guide](https://pandas.pydata.org/docs/user_guide/).

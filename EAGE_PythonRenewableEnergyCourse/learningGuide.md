# Learning guide

## A practical route

Use the [EAGE Python sandbox](https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/)
to read these lessons and write Python side by side. It includes the chapter
exercises, quizzes, a question-only assessment and a final-project workspace.
Save and export your own code and written explanations before submitting them.

Each energy chapter now follows this sequence: physical description and components,
definitions and units, detailed equations with assumptions, operating limits,
a worked hand calculation, a compact equation reference, tested Python functions,
and 20 graded exercises. Read the theory and work through the numerical example
before revealing exercise answers. The downloadable notebook contains the complete
chapter text as well as the code.

1. Start with energy units in Section 0 and its easy exercises 1–5.
2. Learn variables, conditions, loops and functions in Sections 1–2.
3. Practice NumPy, pandas and plotting in Section 3, then attempt exercises 6–15 in each energy chapter.
4. Check data definitions, missingness and SQL totals in Sections 4–5.
5. Study each energy model in Section 6 and its hard exercises 16–20.
6. Take the question-only test, then complete the Section 8 portfolio project from its separate CSV datasets. The test answer key and project solution are instructor-only.

The five chapters each contain 20 exercises: 5 easy, 10 medium and 5 hard. These 100 chapter exercises supplement the existing quizzes. Allow roughly 10–15 minutes per easy exercise, 20–35 per medium exercise and 45–75 per hard exercise. Hard exercises can be follow-up assignments.

## Running the examples

Use the repository README for local setup, or copy an exercise into Colab. Run existing lesson cells in order; each new workbook solution includes its own imports and data and runs independently.

Select a **Solution** dropdown in the rendered book to reveal its steps and executable code. Select it again to hide it. Solutions start collapsed. This is a learning aid, not access control: source files contain the answers, and exported notebook viewers may show cells differently.

Run `input()` examples interactively. Deliberate error examples are labelled and caught so later cells can run. Silent assertions have passed. A `NameError` usually means an import or definition has not run; a `ValueError` often means invalid input.

## A complete exercise answer

Submit code, a result with units, one or more checks, interpretation, reference and assumptions. Try before revealing the solution. Change one input and predict the result before rerunning.

## Habits for trustworthy calculations

- Preserve raw data and document repairs in a separate table.
- Distinguish interval-average power from instantaneous readings; use actual interval durations.
- Use UTC or an explicit time-zone/daylight-saving policy. Check duplicate and missing timestamps.
- Reject non-finite and physically invalid inputs. Missing data are not zero generation.
- Distinguish measurements, reanalysis, forecasts and synthetic examples.
- Respect equipment ratings and consistent gross/net and AC/DC boundaries.
- Split forecast data chronologically; fit transformations on training data only and compare with a baseline.
- Explain omitted losses and uncertainty. Teaching models are not bankable resource assessments.

Core exercises need Python, NumPy, pandas and Matplotlib; SQLite comes with Python. Geospatial tools, Earth Engine, external map services and deep learning are optional. Browser kernels may not support every package or local file. Use the local environment for full validation.

See the [reference catalogue](references.md) and [100-exercise chapter guide](section7/renewableExercises.md).

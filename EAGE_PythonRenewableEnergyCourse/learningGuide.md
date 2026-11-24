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

1. Start with [Module 1](section0/introRenewableEnergy.md) and use the included [Python foundations lesson](section0/python-foundations.md) to learn or refresh Python, notebooks, NumPy, pandas and plotting.
2. Practise units and energy balances in the general workbook before starting the source-specific exercises.
3. Practise data quality, time series, seasonality and mapping in [Module 3](section4/energy-data-analysis.md) and [Module 4](section4/energy-maps.md). Use the database chapters to practise SQLite.
4. Follow [Modules 5–8 in the course outline](courseIntro.md#eage-course-outline) for the theory, source-specific labs and interactive diagrams. Then attempt the workbook and quiz questions.
5. Complete the Hard questions (13–20), which combine modelling, validation, comparison and interpretation.
6. Take the question-only renewable-energy test, then complete the hybrid-portfolio project from its separate CSV datasets. The test answer key and project solution are instructor-only.

Introduction to Renewable Energy contains the fundamentals, equations and worked examples. General, Solar, Wind and Geothermal workbooks each have 20 exercises (4 Basic, 8 Medium and 8 Hard); the hydro workbook has 22 (4 Basic, 10 Medium and 8 Hard). Each chapter quiz has 20 questions. The final project in Course Projects progresses from loading the supplied CSVs through Python modelling, validation, plotting and comparing scenarios in one notebook.

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

The Python foundations lesson prepares you to use NumPy, pandas and Matplotlib in later modules. The mapping labs use GeoPandas and Folium. SQLite comes with Python. Earth Engine, external map services and deep learning are extensions. Browser kernels may not support every package or local file. Use the local environment for full validation.

See the [reference catalogue](references.md) and [Applied Python Workbook](section7/renewableExercises.md).

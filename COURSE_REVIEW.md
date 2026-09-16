# Course review — 16 September 2026

## Completed

- Reviewed the course sections from introductory material through the final project and conclusions.
- Catalogued 10 supplied PDF files representing 9 distinct works; the two solar-energy copies are identical. Added bibliographic metadata, topic locators and a reading guide.
- Added 20 original, source-linked Python exercises: 5 easy, 10 medium and 5 hard. Each has a starter cell, numbered explanation, initially collapsed solution, expected results and executable checks.
- Aligned existing exercises and review material, added references and improved the guided solutions.
- Corrected wind array handling and direction bins, solar temperature/clipping consistency, geothermal invalid inputs, hydropower boundaries, Python explanations, units and statistical interpretations.
- Added a learning route, assumptions, common mistakes, practical limits and guidance for checking results.
- Updated dated energy context using IEA and IRENA primary sources; older books remain references for physical principles.

## Data corrections

The bundled Eurostat exports are historical teaching inputs, not current statistics.
Broad hydro can include pumped-storage output. Selected-source sums are therefore
not labelled verified renewable-only totals, and reporting-country sums are not
labelled official EU-27 totals. Missing observations remain missing. Spreadsheet
footer labels are excluded from countries. Statistical flags are not retained by
the beginner parser, which is documented as a limitation.

The time-series lesson shows changing reporting coverage and uses a continuous
Austrian wind series for decomposition and chronological forecast evaluation.
The SQLite lesson uses source definitions, unique keys, foreign keys, integrity
checks and independent SQL/pandas reconciliation.

## Validation

- Executed 355 Markdown Python cells successfully.
- Executed all 20 new solutions independently, including their numerical assertions.
- Executed both notebooks from clean kernels and saved their outputs (14 code cells).
- Checked bibliography identifiers, table-of-contents targets, exercise difficulty counts, solution containers and database integrity.
- Built the HTML book and inspected the rendered workbook, reference catalogue and notebook charts.

Run `python tools/execute_course_notebooks.py` followed by
`python tools/validate_course.py` from the repository root. Build instructions are
in README.md. The workbook and notebooks have reproducible authoring scripts.

Validation covers the executable core course. Optional library demonstrations,
interactive input snippets, external services and links requiring access are not
claimed to have been tested end to end. The supplied PDFs were checked for
bibliographic metadata and relevant topics; this is not an independent technical
verification of every statement in those books. Teaching models use simplifying
assumptions and do not replace engineering design studies.

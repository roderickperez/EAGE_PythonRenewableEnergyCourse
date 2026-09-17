# Course review — 16 September 2026

## Completed

- Rebuilt the participant Python sandbox around the book's public table of
  contents, with all chapter explanations, references, practice exercises and
  quizzes; generated source hashes prevent silent curriculum drift.
- Added saved code and written-answer workspaces for each of the 25 assessment
  questions and for final-project sections. Their reference answers remain absent
  from the sandbox. The five project input CSVs retain their original bytes.
- Corrected undefined variables and missing inputs in introductory plotting,
  xarray/netCDF and CSV examples. Distinguished syntax templates from executable
  code and bundled the existing NOAA/Seaborn examples with provenance so browser
  execution does not depend on Python making live HTTP requests.
- Integrated sandbox deployment into the existing GitHub Pages workflow under
  `/sandbox/`, preserving the book at the site root.
- Browser validation completed with 133 checks passed, zero failures and 67
  figures emitted: all 100 independent energy solutions, all quiz chapters,
  basic Python/library lessons, optional plotting/geospatial examples, and the
  two data-analysis notebooks. Separately executed 333 basic/quiz code cells
  in the local validator. Checked draft restoration and interruption of an
  infinite loop through the actual interface.

- Expanded the theory before the exercises in all five energy chapters: physical
  mechanisms, components, definitions, equations and units, operating assumptions,
  limitations, environmental context and worked numerical examples. The complete
  theory also appears in each downloadable notebook; all 100 exercises are retained.
- Added Duffie and Beckman's solar textbook and the third edition of *Wind Energy
  Handbook* to the supplementary reading. Verified publisher metadata for the
  existing Kalogirou, Burton and DiPippo references, including the DiPippo
  release-year/copyright-year distinction. Supplementary previews are not presented
  as access to full books; the supplied PDFs remain the primary teaching basis.

- Reviewed the course sections from introductory material through the final project and conclusions.
- Catalogued 10 supplied PDF files representing 9 distinct works; the two solar-energy copies are identical. Added bibliographic metadata, topic locators and a reading guide.
- Embedded 100 original, source-linked Python exercises across five chapters: general renewable energy, solar, hydro, wind and geothermal. Each chapter has 5 easy, 10 medium and 5 hard exercises with a starter cell, numbered answer directly below, initially collapsed solution, interpretation and executable checks.
- Added complete executable Jupyter notebook companions for all five chapters.
- Added a 25-question, 100-mark assessment with no answers in its student page. Its answer key is stored only in the ignored local instructor directory.
- Reworked the final project to load five separate CSV datasets for solar, wind, hydro, geothermal and demand. The step-by-step reference solution is stored only in the ignored local instructor directory, outside the book source.
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

- Executed 481 core Markdown Python cells successfully.
- Executed all 100 chapter solutions independently, including their numerical assertions.
- Executed all seven student notebooks from clean kernels and saved their outputs.
- Executed the private project reference solution against all five supplied datasets, with timestamp, physical-boundary, capacity-factor and energy-balance checks.
- Checked bibliography identifiers, table-of-contents targets, exercise difficulty counts, solution containers and database integrity.
- Checked that substantial theory is retained before all five chapter exercise
  sections and reproduced in the notebooks; independently checked the five new
  worked-example calculations.
- Built the HTML book and inspected the rendered workbook, reference catalogue and notebook charts.

Run `python tools/execute_course_notebooks.py` followed by
`python tools/validate_course.py` from the repository root. Build instructions are
in README.md. The workbook and notebooks have reproducible authoring scripts.

`instructor_private/` is excluded from Git and lies outside the book project. Its
test key and final-project solution must remain local. The previous project
solution has been removed from current student sources and the table of contents;
historical Git commits still contain the older published version. History has
not been rewritten.

Cleared the generated book output to remove stale copies of the former project
solution. A publication check now rejects private filenames, private solution
markers and obsolete project-solution downloads in HTML artifacts. The GitHub
Pages workflow builds the current MyST book and runs that check before upload.

Validation covers the executable core course. Optional library demonstrations,
interactive input snippets, external services and links requiring access are not
claimed to have been tested end to end. The supplied PDFs were checked for
bibliographic metadata and relevant topics; this is not an independent technical
verification of every statement in those books. Teaching models use simplifying
assumptions and do not replace engineering design studies.

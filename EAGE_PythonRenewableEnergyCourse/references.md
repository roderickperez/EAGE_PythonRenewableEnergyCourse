# References and reading guide

Reviewed 16 September 2026. This catalogue covers all 10 PDF files in the repository’s `referenceBooks/` folder (9 distinct works). Publication metadata were checked against the supplied title/copyright pages; topic locators use printed chapters or pages, not PDF-viewer page numbers.

The exercises are original teaching adaptations of these concepts, with synthetic numbers; they are not copied textbook problems. Older books support physical principles, while dated reports support market context. No local PDFs are republished by the book build.

## Books and documents supplied with the course

### Geothermal Reservoir Engineering

[@grant2011]

**Local file:** `referenceBooks/(64) Geothermal Reservoir Engineering- Second Edition.pdf`

**Edition / identifier:** Second edition; ISBN 978-0-12-383880-3.

**Use in this course:** Chapters 2–3: conductive and convective systems, simple models and stored heat; Chapters 4–5: well measurements.

### Guideline and Manual for Hydropower Development, Vol. 2: Small Scale Hydropower

[@jica2011]

**Local file:** `referenceBooks/Guideline and Manual for Hydropower Development Vol 2.pdf`

**Edition / identifier:** March 2011; prepared with Electric Power Development Co. and JP Design.

**Use in this course:** Chapter 3: power and energy; Chapters 7–8: flow-duration curves and generation planning.

### Hydroelectric Power: A Guide for Developers and Investors

[@ifc2015]

**Local file:** `referenceBooks/Hydroelectric Power - A Guide for Developers and Investors.pdf`

**Edition / identifier:** Prepared with Fichtner.

**Use in this course:** Chapter 7: hydrology and energy calculations; Chapters 12–14: environmental effects, costs and economics.

### Clean Energy Technology Observatory: Hydropower and Pumped Storage Hydropower in the European Union — 2025 Status Report on Technology Development, Trends, Value Chains and Markets

[@jrc2025hydro]

**Local file:** `referenceBooks/Hydropower and Pumped Storage Hydropower inthe European Union.pdf`

**Edition / identifier:** JRC143929; DOI 10.2760/3389272.

**Use in this course:** Section 2: technology, capacity, generation and costs. Figures retain the report’s observation years.

### Hydropower and Renewable Energies: Powering a Sustainable Future with Storage and Renewables

[@hu2026]

**Local file:** `referenceBooks/Hydropower and Renewable Energies Powering a Sustainable Future with Storage and Renewables.pdf`

**Edition / identifier:** Edited proceedings; Lecture Notes in Civil Engineering 792; DOI 10.1007/978-981-95-4889-7.

**Use in this course:** Multi-energy integration chapters, including capacity allocation (p. 35) and complementary dispatch (p. 65). Advanced reading, not a beginner design manual.

### Solar Photovoltaic Systems: Technical Training Manual

[@wade2003]

**Local file:** `referenceBooks/Solar Photovoltaic Systems Technical Training Maunal.pdf`

**Edition / identifier:** ISBN 92-3-103904-0; filename retains its original spelling.

**Use in this course:** Chapters 2–3: electricity and PV panels; Chapters 5 and 9: batteries and system sizing. Historical equipment guidance requires current manufacturer specifications.

### Solar Energy: Renewable Energy and the Environment

[@foster2010]

**Local file:** `referenceBooks/Solar-energy-Renewable-Energy-and-the-Environment.pdf`

**Edition / identifier:** ISBN 978-1-4200-7566-3.

**Use in this course:** Solar-resource and photovoltaic chapters: irradiation, conversion and system performance. Use current sources for prices and installed capacity.

### Introduction to Wind Energy Systems: Basics, Technology and Operation

[@wagner2009]

**Local file:** `referenceBooks/Wagner et Mathur - 2009 - Introduction to Wind Energy Systems.pdf`

**Edition / identifier:** DOI 10.1007/978-3-642-02023-0.

**Use in this course:** Wind resource, turbine fundamentals and Chapter 7 economics; suitable introductory companion.

### Wind Energy Explained: Theory, Design and Application

[@manwell2009]

**Local file:** `referenceBooks/Wind Energy Explained.pdf`

**Edition / identifier:** Second edition; ISBN 978-0-470-01500-1.

**Use in this course:** Sections 2.3–2.5: height, wind statistics and energy estimation; Section 3.2: Betz limit; Sections 10.3 and 10.7: hybrid systems and storage.

### Duplicate solar PDF

`referenceBooks/Solar-energy-Renewable-Energy-and-the-Environment (1).pdf` is byte-for-byte identical to the Foster, Ghassemi and Cota file above (SHA-256 checked). Both files are preserved, but use one bibliographic entry [@foster2010].

## Current context and official technical documentation

- [IEA Global Energy Review 2026](https://www.iea.org/reports/global-energy-review-2026): estimates for **2025**, not completed 2026 statistics [@iea2026].
- [IRENA Renewable Power Generation Costs in 2025](https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025): July 2026 edition. Do not substitute global cost averages for a project budget [@irena2026].
- [Eurostat energy database](https://ec.europa.eu/eurostat/web/energy/database): the course retains August 2024 exports as reproducible historical inputs [@eurostat_energy_database].
- [Python tutorial](https://docs.python.org/3/tutorial/) and [built-in functions](https://docs.python.org/3/library/functions.html) [@pythonDocs].
- [NumPy user guide](https://numpy.org/doc/stable/user/) [@numpyDocs].
- [pandas indexing and selection](https://pandas.pydata.org/docs/user_guide/indexing.html) [@pandasDocs].
- [Matplotlib plotting documentation](https://matplotlib.org/stable/users/index.html) [@matplotlibDocs].
- [SQLite SQL language](https://www.sqlite.org/lang.html) [@sqliteDocs].
- [pvlib PVWatts DC model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html) [@pvlibDocs].

## Further reading already cited in the lessons

These works supplement the supplied PDF collection. They are **not additional local
PDFs**. Publisher records and available previews were checked for edition and scope;
the course does not claim that their full texts were downloaded or reviewed.
The detailed chapter explanations are original teaching syntheses grounded primarily
in the supplied books. External textbooks identify where to extend that study.

| Additional work | Verified edition and identifier | How to use it |
|---|---|---|
| Duffie and Beckman, *Solar Engineering of Thermal Processes* | 4th edition, 2013; DOI 10.1002/9781118671603 | Radiation geometry, solar collectors and thermal-system background [@duffie2013] |
| Kalogirou, *Solar Energy Engineering: Processes and Systems* | 2nd edition, 2014; DOI 10.1016/C2011-0-07038-2 | Broader solar conversion and engineering context [@kalogirou2014solar] |
| Burton, Jenkins, Sharpe and Bossanyi, *Wind Energy Handbook* | 2nd edition, 2011; DOI 10.1002/9781119992714 | Additional turbine and wind-farm design reading [@burton2011wind] |
| Burton, Jenkins, Bossanyi, Sharpe and Graham, *Wind Energy Handbook* | 3rd edition, 2021; ISBN 9781119451099 | Updated supplementary coverage of wakes, loads, offshore structures and integration [@burton2021wind] |
| DiPippo, *Geothermal Power Plants: Principles, Applications, Case Studies and Environmental Impact* | 4th edition, copyright 2016; ISBN 9780081008799; DOI 10.1016/C2014-0-02885-7 | Surface conversion cycles and case studies beyond the simple reservoir heat balance [@diPippo2016geothermal] |
| Paish, “Small Hydro Power: Technology and Current Status” | Review article, 2002; DOI 10.1016/S1364-0321(02)00006-0 | Supplementary small-hydro technology background; this is an article, not a textbook [@paish2002hydro] |

DiPippo's publisher records include a late-2015 release date, while the fourth-edition
copyright page states 2016. The bibliography follows that edition year. Technical
principles in older books remain useful, but their market figures and installed-
capacity statistics should be dated rather than presented as current observations.

## Reading route before the chapter exercises

| Chapter | Primary supplied reading | What the expanded explanation prepares you to calculate |
|---|---|---|
| General renewable energy | Wade Chapter 2, JICA Chapter 3, and the technology texts | Boundaries, power versus energy, efficiency versus capacity factor, storage and simultaneous supply/demand balances |
| Solar | Wade's electricity and PV sections; Foster, Ghassemi and Cota's PV and solar-resource treatment | Radiation components, POA inputs, I–V terms, temperature correction, inverter limits, PR and energy |
| Hydro | JICA Chapters 3 and 8; IFC's hydrology, energy and environmental discussions | Head and flow, conversion efficiencies, losses, water allocation, reservoir routing and pumped storage |
| Wind | Manwell et al.'s resource and aerodynamic chapters; Wagner and Mathur's introduction | Kinetic flux, rotor coefficients, shear, the power curve, distributions, direction and AEP |
| Geothermal | Grant and Bixley Chapters 2–3 | Reservoir properties, conduction and fluid flow, enthalpy, net electricity and decline assumptions |

Chapter equations define symbols, units and assumptions. Worked numerical examples
use synthetic inputs and are separate from the existing 20 exercises per chapter.

## How to cite an exercise

Give its number and course title, the underlying book/chapter, the code revision used, and any external data’s provider, product/version, retrieval date, units, location and observation period. State all model assumptions. For synthetic data, say so explicitly. See the [100-exercise chapter guide](section7/renewableExercises.md).

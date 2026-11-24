"""Focused regression checks for previously reported course defects.

Run from the repository root with `.venv/bin/python -m unittest discover -s assessment_support -p test_remediation_core.py`.
The checks use synthetic fixtures and temporary directories; no course outputs are overwritten.
"""
import contextlib
import copy
import io
import math
import json
import uuid
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "EAGE_PythonRenewableEnergyCourse"


class AssessmentIndexTests(unittest.TestCase):
    def test_additional_findings_validation_detects_a_missing_tracker_item(self):
        import subprocess
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            for name in ("COURSE_ASSESSMENT_2026-11-18.md", "COURSE_REMEDIATION_TRACKER.md",
                         "COURSE_FINDING_EVIDENCE_INDEX.md"):
                shutil.copyfile(ROOT / name, root / name)
            shutil.copyfile(ROOT / "tools/validate_assessment_index.py", root / "tools/validate_assessment_index.py")
            def run():
                return subprocess.run([sys.executable, str(root / "tools/validate_assessment_index.py")],
                                      capture_output=True, text=True, check=False)
            good = run()
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertIn("Additional findings: 41 matching tracker entries", good.stdout)
            tracker = root / "COURSE_REMEDIATION_TRACKER.md"
            content = tracker.read_text(encoding="utf-8")
            section_start = content.index("## Additional findings from the assessment")
            start = content.index("- [x] ", section_start)
            end = content.index("\n", start)
            tracker.write_text(content[:start] + content[end + 1:], encoding="utf-8")
            bad = run()
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("Additional finding titles, order or targets do not match", bad.stderr)


class AccessibleCourseContentTests(unittest.TestCase):
    def test_project_reference_anchor_points_to_bibliography(self):
        from bs4 import BeautifulSoup
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        project = next(p for p in manifest["pages"] if p["id"] == "section8/projectIntro.md")
        soup = BeautifulSoup(project["html"], "html.parser")
        bibliography = soup.select('[id="references"]')
        reading = soup.select('[id="references-and-further-reading"]')
        self.assertEqual(len(bibliography), 1)
        self.assertEqual(len(reading), 1)
        self.assertEqual(bibliography[0].find_next_sibling().name, "ul")
        self.assertTrue(bibliography[0].find_next_sibling().select('[id^="ref-"]'))
        self.assertEqual(reading[0].find_next_sibling().name, "p")
        self.assertFalse(any("references" in workspace["title"].casefold() for workspace in project["workspaces"]))

    def test_corrected_narration_transcripts_and_active_media_are_delivered(self):
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        source_media = json.loads((ROOT / "EAGE_pythonSandBox/media.json").read_text(encoding="utf-8"))
        pages = {page["id"]: page for page in manifest["pages"]}
        cases = (
            ("solarEnergy", "thermal-storage-energy-is-not-power", "en/solar/thermal-storage-boundary-corrected.mp3",
             "A larger mass stores more sensible heat", "Latent heat is additional energy"),
            ("windEnergy", "why-available-power-contains-the-cube-of-wind-speed", "en/wind/available-power-reference-flux-corrected.mp3",
             "undisturbed reference flux", "actual disk-plane mass flow"),
        )
        for topic, heading, relative_media, first, second in cases:
            page_id = f"section6/{topic}.md"
            lesson = (BOOK / page_id).read_text(encoding="utf-8")
            transcript = re.search(r"\*\*Recording transcript:\*\* (.*)", lesson).group(1)
            self.assertIn(first, transcript)
            self.assertIn(second, transcript)
            self.assertIn(first, pages[page_id]["html"])
            self.assertIn(second, pages[page_id]["html"])
            notebook = json.loads((BOOK / f"section6/{topic}.ipynb").read_text(encoding="utf-8"))
            self.assertTrue(any(transcript in "".join(cell["source"]) for cell in notebook["cells"]
                                if cell["cell_type"] == "markdown"))
            self.assertEqual(source_media[page_id]["audio"]["en"][heading], relative_media)
            generated = pages[page_id]["media"]["audio"]["en"][heading]
            self.assertEqual(generated, "generated/media/" + relative_media)
            self.assertEqual((ROOT / "EAGE_pythonSandBox" / generated).read_bytes(),
                             (ROOT / "EAGE_pythonSandBox/media" / relative_media).read_bytes())

    def test_assessed_image_alternatives_and_math_descriptions_reach_deliveries(self):
        from html.parser import HTMLParser
        class Images(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = []
            def handle_starttag(self, tag, attrs):
                if tag == "img":
                    self.items.append(dict(attrs))
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        pages = {page["id"]: page["html"] for page in manifest["pages"]}
        preface = Images(); preface.feed(pages["preface/preface.md"])
        alt_by_source = {item["src"].rsplit("/", 1)[-1]: item.get("alt") for item in preface.items}
        self.assertEqual(alt_by_source["A_Karpathy_tweet.png"], "Screenshot of Andrej Karpathy's post discussing vibe coding")
        self.assertIn("through February 2026", alt_by_source["vibe_coding_google_trends_worldwide_2021_2026.png"])
        self.assertIn("rose from 3 in February 2025 to 100 in February 2026", pages["preface/preface.md"])
        self.assertIn("intervening fluctuations", pages["preface/preface.md"])
        self.assertIn("not a count of searches or adopters", pages["preface/preface.md"])
        intro = Images(); intro.feed(pages["courseIntro.md"])
        self.assertEqual(next(item["alt"] for item in intro.items if item["src"].endswith("Logo4.png")),
                         "Wind turbines beyond a green field")
        for topic, phrase in (("hydroelectricEnergy", "square root of slope"),
                              ("geothermalEnergy", "square root of thermal diffusivity")):
            self.assertIn(phrase, pages[f"section6/{topic}.md"])
            notebook = json.loads((BOOK / f"section6/{topic}.ipynb").read_text(encoding="utf-8"))
            self.assertTrue(any(phrase in "".join(cell["source"]) for cell in notebook["cells"]
                                if cell["cell_type"] == "markdown"))


class AssessmentPromptTests(unittest.TestCase):
    """Check that the assessed prompts retain enough data for their comparisons."""

    def test_hydro_rating_schema_supports_the_requested_comparison(self):
        import ast
        text = (BOOK / "section7/assessment/hydroelectric.md").read_text(encoding="utf-8")
        q28 = text.split("### Question 28", 1)[1].split("### Question 29", 1)[0]
        self.assertIn("shared `generation` schema above", q28)
        self.assertLess(text.index("generation(country TEXT"), text.index("### Question 28"))
        question = text.split("### Question 30", 1)[1]
        flows = ast.literal_eval(re.search(r"flow observations `([^`]+)`", question).group(1))
        self.assertIn("one-hour", question)
        self.assertIn("combined turbine–generator efficiency 0.85", question)
        self.assertIn("leave the missing hour unknown", question)
        schema = re.search(r"`(rating_generation\(rating_mw REAL,.*?\))`", question).group(1)
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE " + schema)
            for rating in (2, 4):
                for hour, flow in enumerate(flows):
                    energy = None if flow is None else min(rating, .85 * 1000 * 9.81 * flow * 40 / 1e6)
                    db.execute("INSERT INTO rating_generation VALUES (?, ?, ?)", (rating, hour, energy))
            rows = db.execute("SELECT rating_mw, COUNT(energy_mwh), SUM(energy_mwh) "
                              "FROM rating_generation GROUP BY rating_mw ORDER BY rating_mw").fetchall()
        self.assertEqual([r[:2] for r in rows], [(2., 3), (4., 3)])
        self.assertAlmostEqual(rows[0][2], 5.6677)
        self.assertAlmostEqual(rows[1][2], 9.0031)

    def test_geothermal_annual_fixture_and_missing_year(self):
        import ast
        text = (BOOK / "section7/assessment/geothermal.md").read_text(encoding="utf-8")
        question = text.split("### Question 60", 1)[1]
        flows = ast.literal_eval(re.search(r"flows `([^`]+)`", question).group(1))
        self.assertEqual(flows, [50, 47, None, 42])
        for boundary in ("2021–2024", "300 kJ/kg", "12%", "10%", "8,000 operating hours", "missing year unknown"):
            self.assertIn(boundary, question)
        annual = [None if flow is None else flow * 300 * .12 * .9 / 1000 * 8000 for flow in flows]
        self.assertEqual(annual, [12960., 12182.4, None, 10886.4])
        self.assertAlmostEqual(sum(x for x in annual if x is not None), 36028.8)

    def test_solar_cap_comparison_retains_unknown_interval(self):
        import ast
        text = (BOOK / "section7/assessment/solar.md").read_text(encoding="utf-8")
        question = text.split("### Question 40", 1)[1]
        dc = ast.literal_eval(re.search(r"DC power `([^`]+)`", question).group(1))
        self.assertIn("95% inverter efficiency", question)
        self.assertIn("one-hour intervals", question)
        self.assertIn("missing interval as unknown", question)
        for cap, expected_delivered, expected_clipped in ((3, 8.85, 5.40), (5, 12.60, 1.65)):
            converted = [None if value is None else .95 * value for value in dc]
            self.assertEqual(sum(value is not None for value in converted), 5)
            self.assertAlmostEqual(sum(min(cap, value) for value in converted if value is not None), expected_delivered)
            self.assertAlmostEqual(sum(max(value - cap, 0) for value in converted if value is not None), expected_clipped)

    def test_wind_curve_comparison_uses_the_same_valid_hours(self):
        import ast
        text = (BOOK / "section7/assessment/wind.md").read_text(encoding="utf-8")
        q49 = text.split("### Question 49", 1)[1].split("### Question 50", 1)[0]
        q50 = text.split("### Question 50", 1)[1]
        speeds = ast.literal_eval(re.search(r"speeds `([^`]+)`", q49).group(1))
        self.assertIn("constant for its one-hour interval", q49)
        self.assertIn("same six one-hour speed observations", q50)
        self.assertIn("cannot establish long-term annual energy", q50)
        def energy(rating):
            return sum(0 if speed <= 3 or speed >= 25 else rating if speed >= 12
                       else rating * (speed**3 - 3**3) / (12**3 - 3**3)
                       for speed in speeds if speed is not None and speed >= 0)
        self.assertEqual(sum(v is not None and v >= 0 for v in speeds), 4)
        self.assertAlmostEqual(energy(3), 2.049382716049383)
        self.assertAlmostEqual(energy(4), energy(3) * 4 / 3)


class QuizOutputTests(unittest.TestCase):
    def test_python_mean_and_sql_database_assumptions_are_explicit(self):
        quizzes = BOOK / "section7/renewableEnergyquizzes"
        python = (quizzes / "generalPython.md").read_text(encoding="utf-8")
        sql = (quizzes / "pythonSQL.md").read_text(encoding="utf-8")
        section = re.search(r":::\{admonition\} Quiz 10\b(.*?)(?=:::\{admonition\} Quiz \d+|\Z)", python, re.S).group(1)
        self.assertIn("values = [2, 4, 6]", section)
        namespace = {}
        exec("values = [2, 4, 6]", namespace)
        self.assertEqual(eval("sum(values) / len(values)", namespace), 4)
        sql_intro = re.search(r":::\{admonition\} Quiz 1\b(.*?)(?=:::\{admonition\} Quiz \d+|\Z)", sql, re.S).group(1)
        self.assertIn("file", sql_intro)
        self.assertIn("memory", sql_intro)
        with contextlib.closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE TABLE generation (gwh REAL)")
            connection.execute("INSERT INTO generation VALUES (3.5)")
            self.assertEqual(connection.execute("SELECT SUM(gwh) FROM generation").fetchone()[0], 3.5)
            self.assertEqual(connection.execute("PRAGMA database_list").fetchone()[2], "")

    def test_sql_quiz_rollback_example_executes_and_explains_boundary(self):
        text = (BOOK / "section7/renewableEnergyquizzes/pythonSQL.md").read_text(encoding="utf-8")
        section = re.search(r":::\{admonition\} Quiz 19\b(.*?)(?=:::\{admonition\} Quiz \d+|\Z)", text, re.S).group(1)
        self.assertIn("executemany", section)
        self.assertIn("to_sql", section)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section, re.S).group(1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        self.assertIn("Rows after rollback: 0", output.getvalue())

    def test_assessed_quiz_answers_produce_visible_text(self):
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = {
            "windEnergy.md": [3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20],
            "hydroelectricEnergy.md": [3, 4, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
            "solarEnergy.md": [3, 4, 8, 9, 10, 11, 12, 16, 17, 18, 19],
            "geothermalEnergy.md": [3, 4, 8, 9, 10, 11],
        }
        for filename, targets in groups.items():
            text = (BOOK / "section7/renewableEnergyquizzes" / filename).read_text(encoding="utf-8")
            sections = re.split(r"(?m)^:::\{admonition\} Quiz\s+(\d+)\b[^\n]*\n", text)
            state = {"__name__": "__main__"}; outputs = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for cell in re.findall(r"```\{code-cell\} python\n(.*?)```", sections[0], re.S):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(cell, state)
                for number, section in zip(sections[1::2], sections[2::2]):
                    output = io.StringIO()
                    for cell in re.findall(r"```\{code-cell\} python\n(.*?)```", section, re.S):
                        with contextlib.redirect_stdout(output):
                            exec(cell, state)
                        plt.close("all")
                    outputs[int(number)] = output.getvalue()
            for number in targets:
                with self.subTest(file=filename, question=number):
                    self.assertTrue(outputs.get(number, "").strip())


class ExternalSolarReadingTests(unittest.TestCase):
    def test_solar_lesson_warns_at_the_textbook_locator(self):
        text = (BOOK / "section6/solarEnergy.md").read_text(encoding="utf-8")
        self.assertIn("Foster p. 125 mislabels the zero-current condition", text)
        self.assertIn("Eq. 5.15 on p. 126 also applies fill factor twice", text)
        self.assertIn("USDOE_WVSC_Project_Final_Report_5-30-2014_R1.pdf", text)

    def test_project_discloses_pvgis_interface_changes(self):
        text = (BOOK / "section8/solar-project.md").read_text(encoding="utf-8")
        self.assertIn("surface_azimuth` changed to the north-zero convention in version 0.10.0", text)
        self.assertIn("changed from three return values to two `(data, metadata)` in version 0.13.0", text)
        self.assertIn("verify cardinal directions", text)


class CourseRouteTests(unittest.TestCase):
    def test_lcoe_equation_uses_accessible_locator_not_missing_irena_annex(self):
        intro = (BOOK / "section0/introRenewableEnergy.md").read_text(encoding="utf-8")
        section = intro.split("Comparing projects without confusing metrics", 1)[1].split("## Worked comparison", 1)[0]
        self.assertIn("§13.2.5 and Chapter 14", section)
        self.assertNotIn("annex", section.lower())
        files = list((BOOK / "section7/workbook").glob("*.md"))
        files += list((BOOK / "section7/workbook").glob("*.ipynb"))
        files += [BOOK / "section0/introRenewableEnergy.ipynb"]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                with self.subTest(path=path.name, line=line[:80]):
                    self.assertFalse("irena" in line.lower() and "annex" in line.lower())

    def test_project_zero_demand_coverage_rule_is_consistent(self):
        brief = (BOOK / "section8/projectIntro.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section8/projectStarter.ipynb").read_text(encoding="utf-8"))
        starter = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        self.assertIn("zero total demand, report coverage as undefined", brief)
        self.assertIn("when total demand is zero, report coverage as undefined", brief)
        self.assertIn("if total demand is zero, report it as undefined and do not rank scenarios by it", starter)
        self.assertIn("served energy divided by demand energy", starter)

    def test_solar_project_explains_assessed_irradiance_and_angle_conventions(self):
        text = (BOOK / "section8/solar-project.md").read_text(encoding="utf-8")
        self.assertIn("sky radiation scattered by the atmosphere, excluding the direct beam", text)
        self.assertIn("an oblique direct beam is still direct radiation", text)
        self.assertIn("useful UK starting point, not a universal optimum", text)
        self.assertIn("PVGIS aspect is zero south and positive west", text)
        self.assertIn("(aspect + 180) % 360", text)
        self.assertIn("surface_azimuth", text)
        self.assertIn("version 0.10.0", text)
        self.assertIn("version 0.13.0", text)
        for pvgis, north_clockwise in ((-180, 0), (-90, 90), (0, 180), (30, 210), (90, 270), (150, 330)):
            self.assertEqual((pvgis + 180) % 360, north_clockwise)

    def test_workbook_counts_match_guide_and_reference_catalogue(self):
        counts = {}
        for topic in ("general", "hydroelectric", "solar", "wind", "geothermal"):
            text = (BOOK / "section7/workbook" / (topic + ".md")).read_text(encoding="utf-8")
            counts[topic] = len(re.findall(r"(?m)^### Exercise \d+ —", text))
        self.assertEqual(counts, {"general": 20, "hydroelectric": 22, "solar": 20, "wind": 20, "geothermal": 20})
        self.assertEqual(sum(counts.values()), 102)
        guide = (BOOK / "learningGuide.md").read_text(encoding="utf-8")
        references = (BOOK / "references.md").read_text(encoding="utf-8")
        self.assertIn("hydro workbook has 22", guide)
        self.assertIn("Basic exercises 1–4", guide)
        self.assertIn("22 for Hydroelectric: 102 in total", references)
        self.assertNotIn("Each topic has 20 practice exercises", references)

    def test_wind_profile_variable_definitions_precede_source_note(self):
        lesson = (BOOK / "section6/windEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/windEnergy.ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for label, text in (("lesson", lesson), ("notebook", notebook_text)):
            equation = text.index(r"$$v(z)=\frac{u_*}")
            end = text.index("$$", equation + 2) + 2
            definitions = text.index("where $u_*$ is friction velocity", end)
            source = text.index("For the displacement-height form", end)
            with self.subTest(delivery=label):
                self.assertEqual(text[end:definitions].strip(), "")
                self.assertLess(definitions, source)
                self.assertIn("Eq. 2–20", text[source:source + 160])

    def test_wind_p90_source_uses_direct_sam_help(self):
        url = "https://samrepo.nlr.gov/help/p50p90.html"
        lesson = (BOOK / "section6/windEnergy.md").read_text(encoding="utf-8")
        self.assertIn(url, lesson)
        self.assertNotIn("NREL SAM documentation, §13.5", lesson)
        notebook = json.loads((BOOK / "section6/windEnergy.ipynb").read_text(encoding="utf-8"))
        self.assertTrue(any(url in "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"))
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        page = next(p for p in manifest["pages"] if p["id"] == "section6/windEnergy.md")
        self.assertIn(url, page["html"])

    def test_project_environmental_reservation_identifies_compliance_location(self):
        project = (BOOK / "section8/projectIntro.md").read_text(encoding="utf-8")
        hydro = project.split("### Hydropower", 1)[1].split("### Geothermal", 1)[0]
        self.assertIn("separate bypass reach upstream of the turbine return", hydro)
        self.assertIn("downstream of the tailrace may instead include qualifying turbine discharge", hydro)
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        delivered = next(page["html"] for page in manifest["pages"] if page["id"] == "section8/projectIntro.md")
        self.assertIn("separate bypass reach upstream of the turbine return", delivered)
        self.assertIn("downstream of the tailrace may instead include qualifying turbine discharge", delivered)
        for text in (project, delivered):
            self.assertIn('environmental-flow shortfall', text)
            self.assertIn('3,600 seconds', text)
            self.assertIn('separate from electricity shortfall', text)
        import nbformat
        starter = nbformat.read(BOOK / 'section8/projectStarter.ipynb', as_version=4)
        starter_text = '\n'.join(cell.source for cell in starter.cells if cell.cell_type == 'markdown')
        self.assertIn('environmental-flow shortfall volume separately from electricity shortfall', starter_text)
        for text in (project, starter_text):
            self.assertIn('pandas.to_numeric(..., errors="raise")', text)
            self.assertIn('numeric text', text)
        for flow, target, expected_shortfall in ((4., 6., 2.), (6., 6., 0.), (10., 6., 0.)):
            served, usable = min(flow, target), max(flow - target, 0.)
            shortfall = max(target - flow, 0.)
            self.assertEqual(shortfall, expected_shortfall)
            self.assertEqual(served + usable, flow)
        self.assertEqual(max(6. - 4., 0.) * 3600, 7200.)
        # The alternative location is a comparison, not a change to the project's rule.
        bypass_case = .88 * 1000 * 9.81 * (10 - 6) * 42 / 1e6
        downstream_case = .88 * 1000 * 9.81 * 10 * 42 / 1e6
        self.assertAlmostEqual(bypass_case, 1.4503104)
        self.assertAlmostEqual(downstream_case, 3.625776)

    def test_optional_book_data_and_browser_forecast_are_identified(self):
        overview = (BOOK / "courseIntro.md").read_text(encoding="utf-8")
        summary = (BOOK / "section9/courseSummary.md").read_text(encoding="utf-8")
        self.assertIn("notebook alone does not contain the data", overview)
        self.assertIn("requires the [course repository]", summary)
        self.assertIn("Wind Data Lab", overview)
        self.assertIn("simple chronological forecast baselines", overview)
        self.assertNotIn("prediction model with scikit-learn", overview)
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        delivered = next(page["html"] for page in manifest["pages"] if page["id"] == "courseIntro.md")
        self.assertIn("simple chronological forecast baselines", delivered)
        self.assertNotIn("prediction model with scikit-learn", delivered)
        self.assertIn("Wind Data Lab", summary)
        self.assertIn("This course includes a general Python practice quiz", summary)
        self.assertIn("not in the browser lesson navigation", summary)
        self.assertIn("Quiz 15 — Chronological baseline", (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8"))

    def test_prerequisite_route_and_wind_energy_boundaries_are_explicit(self):
        faq = (BOOK / "section0/faq.md").read_text(encoding="utf-8")
        conclusion = (BOOK / "section9/courseConclusions.md").read_text(encoding="utf-8")
        summary = (BOOK / "section9/courseSummary.md").read_text(encoding="utf-8")
        self.assertIn("A working foundation in Python", faq)
        self.assertIn("separate access may be required", faq)
        self.assertNotIn("Sections 1–3", faq)
        self.assertIn("summed from measured electrical intervals", conclusion)
        self.assertIn("estimated from a wind-speed series", conclusion)
        self.assertIn("when estimating from wind speeds, use an appropriate electrical power curve", summary)


class PrefaceClaimTests(unittest.TestCase):
    def test_productivity_and_search_interest_claims_are_qualified(self):
        text = (BOOK / "preface/preface.md").read_text(encoding="utf-8")
        self.assertNotIn("generate results in record time", text)
        self.assertNotIn("getting better every single day", text)
        self.assertIn("useful results still require testing and review", text)
        self.assertIn("not a count of searches or adopters", text)
        self.assertIn("not directly comparable as absolute search volumes", text)


class TrendImportTests(unittest.TestCase):
    def test_plot_title_uses_the_selected_csv_series_label(self):
        import matplotlib.pyplot as plt
        text = (BOOK / "preface/preface.md").read_text(encoding="utf-8")
        block = next(c for c in re.findall(r"```python\n(.*?)```", text, re.S) if "def load_trends(" in c)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "selected.csv"
            image_path = Path(directory) / "chart.png"
            csv_path.write_text("Month,solar data\n2025-01,12\n2025-02,25\n", encoding="utf-8")
            code = re.sub(r'^csv_path = .*$', f'csv_path = Path({str(csv_path)!r})', block, count=1, flags=re.M)
            code = re.sub(r'^out_path = .*$', f'out_path = Path({str(image_path)!r})', code, count=1, flags=re.M)
            state = {}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, state)
                self.assertEqual(state["ax"].get_title(), "Google Trends: solar data")
                self.assertTrue(image_path.is_file())
            finally:
                plt.close("all")

    def test_metadata_header_and_special_values_are_not_silently_discarded(self):
        import ast
        import csv
        text = (BOOK / "preface/preface.md").read_text(encoding="utf-8")
        block = next(c for c in re.findall(r"```python\n(.*?)```", text, re.S) if "def load_trends(" in c)
        tree = ast.parse(block)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_trends")
        namespace = {"Path": Path, "pd": pd, "csv": csv}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "preface", "exec"), namespace)
        load = namespace["load_trends"]
        baseline = BOOK / "images/preface/time_series_Worldwide_20210308-2304_20260308-2304.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            frame, _, _ = load(baseline)
        self.assertEqual(len(frame), len(pd.read_csv(baseline)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trends.csv"
            header = "Category: All categories\n\nMonth,vibe coding,isPartial\n2025-01,12,False\n"
            path.write_text(header + "2025-02,25,False\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                frame, _, _ = load(path)
            self.assertEqual(len(frame), 2)
            for row in ("2025-02,<1,False", "2025-02,25,True", "2025-01,25,False", "2025-02,,False"):
                path.write_text(header + row + "\n", encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    load(path)


class WorldSelectionTests(unittest.TestCase):
    def setUp(self):
        text = (BOOK / "section5/world-energy-download.md").read_text(encoding="utf-8")
        self.cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        self.assertEqual(len(self.cells), 3)
        import hashlib
        self.raw_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.raw_temp.cleanup)
        raw_folder = Path(self.raw_temp.name)
        download_record = {}
        for name in ("owid_energy.csv", "owid_energy.codebook.csv", "readme.md"):
            raw = ("fixture " + name).encode()
            (raw_folder / name).write_bytes(raw)
            download_record[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.state = dict(pd=pd, math=math, Path=Path, source_ready=True,
            data=pd.DataFrame({"country": ["World", "Austria"], "iso_code": [None, "AUT"],
                               "year": [2023, 2023], "wind_electricity_twh": [1.5, .1]}),
            codebook=pd.DataFrame([dict(column="wind_electricity_twh", unit="TWh",
                  title="Wind", description="Generation", source="Fixture")]),
            download_record=download_record, raw_folder=raw_folder)
        self.sync_fixture_sources()

    def sync_fixture_sources(self):
        import hashlib
        for name, key in (("owid_energy.csv", "data"), ("owid_energy.codebook.csv", "codebook")):
            raw = self.state[key].to_csv(index=False).encode("utf-8")
            (self.state["raw_folder"] / name).write_bytes(raw)
            self.state["download_record"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_notebook_runs_the_same_download_selection_and_export_cells(self):
        notebook = json.loads((BOOK / "section5/world-energy-download.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]).rstrip() for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(code, [cell.rstrip() for cell in self.cells])

    def select_world(self):
        code = self.cells[1].replace('COUNTRY_CODE = "AUT"', 'COUNTRY_CODE = "WORLD"')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, self.state)

    def test_missing_or_numeric_unit_rejects_selection_and_stale_export(self):
        original = self.state["codebook"].copy()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        for unit in (None, 123):
            with self.subTest(unit=unit):
                self.state["codebook"] = original.copy()
                self.sync_fixture_sources()
                self.select_world()
                if unit is None:
                    self.state["codebook"] = original.drop(columns=["unit"])
                else:
                    self.state["codebook"]["unit"] = unit
                self.sync_fixture_sources()
                with self.assertRaisesRegex(ValueError, "Unit changed"):
                    self.select_world()
                self.assertNotIn("selected", self.state)
                self.assertNotIn("selection_context", self.state)
                with self.assertRaisesRegex(RuntimeError, "successful download and selection"):
                    exec(self.cells[2], self.state)

    def test_selection_rejects_incomplete_or_changed_retrieval(self):
        missing = self.state["download_record"].pop("readme.md")
        with self.assertRaisesRegex(RuntimeError, "Incomplete retrieval record"):
            self.select_world()
        self.state["download_record"]["readme.md"] = missing
        (self.state["raw_folder"] / "readme.md").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.select_world()
        self.assertNotIn("selection_context", self.state)

    def test_selection_reloads_verified_sources_after_in_memory_edits(self):
        self.state["data"].loc[0, "wind_electricity_twh"] = 999
        self.state["codebook"].loc[0, "unit"] = "TWh per capita"
        self.select_world()
        self.assertEqual(self.state["selected"].loc[self.state["selected"].year.eq(2023), "generation_gwh"].item(), 1500.)
        self.assertEqual(self.state["codebook"].loc[0, "unit"], "TWh")

    def test_world_keeps_missing_iso_and_reindexes_years(self):
        self.select_world()
        result = self.state["selected"]
        self.assertTrue(result.iso_code.isna().all())
        self.assertEqual(result.loc[result.year.eq(2023), "generation_gwh"].item(), 1500.)
        self.assertEqual(result.generation_twh.notna().sum(), 1)

    def test_exact_name_selects_codeless_entity_and_rejects_ambiguous_history(self):
        self.state["data"] = pd.concat([self.state["data"], pd.DataFrame([
            {"country": "Kosovo", "iso_code": None, "year": 2023, "wind_electricity_twh": .4}
        ])], ignore_index=True)
        self.sync_fixture_sources()
        code = self.cells[1].replace('ENTITY_NAME = ""', 'ENTITY_NAME = "Kosovo"')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, self.state)
        self.assertEqual(self.state["selected"].country.unique().tolist(), ["Kosovo"])
        self.assertTrue(self.state["selected"].iso_code.isna().all())
        self.assertEqual(self.state["selection_context"]["selector"], "name:Kosovo")
        self.assertEqual(self.state["selected"].loc[self.state["selected"].year.eq(2023), "generation_gwh"].item(), 400.)
        self.state["data"] = pd.concat([self.state["data"], self.state["data"].tail(1)], ignore_index=True)
        self.sync_fixture_sources()
        with self.assertRaisesRegex(ValueError, "Duplicate country-year"):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, self.state)
        self.assertNotIn("selected", self.state)

    def test_mixed_entity_codes_are_rejected_for_code_and_name_selection(self):
        self.state["data"] = pd.concat([self.state["data"], pd.DataFrame([
            {"country": "Austria", "iso_code": None, "year": 2022, "wind_electricity_twh": .2}
        ])], ignore_index=True)
        self.sync_fixture_sources()
        self.select_world()  # prior successful selection must not survive the failed one
        for code in (self.cells[1], self.cells[1].replace('ENTITY_NAME = ""', 'ENTITY_NAME = "Austria"')):
            with self.subTest(selection="name" if 'ENTITY_NAME = "Austria"' in code else "code"):
                with self.assertRaisesRegex(ValueError, "both missing and present source codes"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(code, self.state)
                self.assertNotIn("selection_context", self.state)
                self.assertNotIn("selected", self.state)
                with self.assertRaisesRegex(RuntimeError, "successful download and selection"):
                    exec(self.cells[2], self.state)

    def test_exact_name_export_uses_portable_folder_and_preserves_selector(self):
        from urllib.parse import quote
        names = ['Kosovo', "Côte d'Ivoire / synthetic region"]
        self.state['data'] = pd.concat([self.state['data'], pd.DataFrame([
            {'country': name, 'iso_code': None, 'year': 2023, 'wind_electricity_twh': .4}
            for name in names
        ])], ignore_index=True)
        self.sync_fixture_sources()
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory), contextlib.redirect_stdout(io.StringIO()):
                code = self.cells[1].replace('ENTITY_NAME = ""', 'ENTITY_NAME = ' + repr(name))
                exec(code, self.state)
                self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
                exec(self.cells[2], self.state)
                output = self.state['output']
                self.assertNotRegex(output.name, r'[<>:"/\\|?*]')
                self.assertIn(quote('name:' + name, safe='-._'), output.name)
                provenance = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
                self.assertEqual(provenance['selector'], 'name:' + name)
                with sqlite3.connect(output / 'generation.sqlite') as connection:
                    self.assertEqual(connection.execute('SELECT SUM(generation_twh) FROM generation').fetchone()[0], .4)

    def test_export_rejects_changed_settings_source_and_selected_rows(self):
        self.select_world()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                for field, altered in (("VARIABLE", "solar_electricity_twh"),
                                       ("COUNTRY_CODE", "AUT"),
                                       ("download_record", {"different": True})):
                    original = self.state[field]
                    self.state[field] = altered
                    with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "rerun selection"):
                        exec(self.cells[2], self.state)
                    self.state[field] = original
                entry = self.state["download_record"]["owid_energy.csv"]
                original_hash = entry["sha256"]
                entry["sha256"] = "changed"
                with self.assertRaisesRegex(RuntimeError, "rerun selection"):
                    exec(self.cells[2], self.state)
                entry["sha256"] = original_hash
                raw_path = self.state["raw_folder"] / "readme.md"
                original_raw = raw_path.read_bytes()
                raw_path.write_bytes(b"altered")
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    exec(self.cells[2], self.state)
                raw_path.write_bytes(original_raw)
                self.state["selected"].loc[0, "generation_twh"] = 99.
                with self.assertRaisesRegex(RuntimeError, "Selected data changed"):
                    exec(self.cells[2], self.state)
                self.assertFalse(Path("outputs").exists())
            finally:
                os.chdir(previous)

    def test_optimized_export_rejects_reconciliation_failure_before_publication(self):
        from types import SimpleNamespace
        self.select_world()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        notebook = json.loads((BOOK / "section5/world-energy-download.ipynb").read_text(encoding="utf-8"))
        notebook_code = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and "selection_sha256" in "".join(c["source"])
                             and "generation.sqlite" in "".join(c["source"]))
        self.assertEqual(notebook_code.rstrip(), self.cells[2].rstrip())
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells[2], self.state)
                completed = self.state["output"]
                original = {p.name: p.read_bytes() for p in completed.iterdir()}
                for label, code, fake_result in ((label, code, result)
                        for label, code in (("lesson", self.cells[2]), ("notebook", notebook_code))
                        for result in ((0, None), (1, 999.))):
                    class FaultyConnection(sqlite3.Connection):
                        def execute(self, sql, *args, **kwargs):
                            if sql.startswith("SELECT COUNT(generation_twh)"):
                                return SimpleNamespace(fetchone=lambda: fake_result)
                            return super().execute(sql, *args, **kwargs)
                    self.state["sqlite3"] = SimpleNamespace(
                        connect=lambda p: sqlite3.connect(p, factory=FaultyConnection))
                    with self.subTest(delivery=label, result=fake_result), self.assertRaisesRegex(ValueError, "disagree"):
                        exec(compile(code, "world export", "exec", optimize=1), self.state)
                    self.assertEqual(original, {p.name: p.read_bytes() for p in completed.iterdir()})
                folders = list(Path("outputs/owid_world").iterdir())
                self.assertEqual(sum(p.name.endswith(".incomplete") for p in folders), 4)
                self.assertEqual(sum(not p.name.endswith(".incomplete") for p in folders), 1)
            finally:
                os.chdir(previous)

    def test_optimized_world_export_keeps_all_missing_total_unavailable(self):
        import numpy as np
        from types import SimpleNamespace
        self.state["data"].loc[self.state["data"].country.eq("World"), "wind_electricity_twh"] = np.nan
        self.sync_fixture_sources()
        self.select_world()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(self.cells[2], "world export", "exec", optimize=1), self.state)
                completed = self.state["output"]
                before = {p.name: p.read_bytes() for p in completed.iterdir()}
                connection = sqlite3.connect(completed / "generation.sqlite")
                try:
                    self.assertEqual(connection.execute("SELECT COUNT(generation_twh), SUM(generation_twh) FROM generation").fetchone(), (0, None))
                finally:
                    connection.close()
                class IncorrectSum(sqlite3.Connection):
                    def execute(self, sql, *args, **kwargs):
                        if sql.startswith("SELECT COUNT(generation_twh)"):
                            return SimpleNamespace(fetchone=lambda: (0, 0.))
                        return super().execute(sql, *args, **kwargs)
                self.state["sqlite3"] = SimpleNamespace(connect=lambda p: sqlite3.connect(p, factory=IncorrectSum))
                with self.assertRaisesRegex(ValueError, "missing totals disagree"):
                    exec(compile(self.cells[2], "world export", "exec", optimize=1), self.state)
                self.assertEqual(before, {p.name: p.read_bytes() for p in completed.iterdir()})
                self.assertEqual(len(list(Path("outputs/owid_world").glob("*.incomplete"))), 1)
            finally:
                os.chdir(previous)

    def test_export_rejects_missing_year_key(self):
        self.select_world()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells[2], self.state)
                output = self.state["output"]
                connection = sqlite3.connect(output / "generation.sqlite")
                try:
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute("INSERT INTO generation(country,year) VALUES ('World',NULL)")
                finally:
                    connection.close()
            finally:
                os.chdir(previous)

    def test_failed_export_does_not_modify_previous_bundle(self):
        self.select_world()
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells[2], self.state)
                completed = self.state["output"]
                before = {p.name: p.read_bytes() for p in completed.iterdir()}
                with mock.patch.object(sqlite3, "connect", side_effect=sqlite3.OperationalError("injected failure")):
                    with self.assertRaises(sqlite3.OperationalError):
                        exec(self.cells[2], self.state)
                self.assertEqual(before, {p.name: p.read_bytes() for p in completed.iterdir()})
                self.assertEqual(len(list(Path("outputs/owid_world").glob("*.incomplete"))), 1)
            finally:
                os.chdir(previous)

    def test_raw_retrieval_versions_and_interrupted_download(self):
        import ast
        import asyncio
        import urllib.request
        csv1 = b"country,iso_code,year,wind_electricity_twh\nAustria,AUT,2023,1.5\n"
        csv2 = csv1.replace(b"1.5", b"2.5")
        codebook = (b"column,unit,title,description,source\n"
                    b"wind_electricity_twh,TWh,Wind,Generation,Fixture\n")
        def responses(data):
            for raw in data:
                if isinstance(raw, Exception):
                    raise raw
                result = mock.MagicMock()
                result.__enter__.return_value.read.return_value = raw
                yield result
        compiled = compile(self.cells[0], "world download", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    with mock.patch.object(urllib.request, "urlopen", side_effect=responses([csv1, codebook, b"readme"])):
                        asyncio.run(eval(compiled, self.state))
                    first = self.state["raw_folder"]
                    before = {p.name: p.read_bytes() for p in first.iterdir()}
                    exec(self.cells[1], self.state)
                    self.assertEqual(self.state["selected"].generation_twh.dropna().tolist(), [1.5])
                    with mock.patch.object(urllib.request, "urlopen", side_effect=responses([csv2, OSError("injected failure")])):
                        with self.assertRaises(OSError):
                            asyncio.run(eval(compiled, self.state))
                    self.assertFalse(self.state["source_ready"])
                    self.assertNotIn("data", self.state)
                    self.assertNotIn("selected", self.state)
                    self.assertNotIn("selection_context", self.state)
                    with self.assertRaisesRegex(RuntimeError, "successful download and selection"):
                        exec(self.cells[2], self.state)
                    with self.assertRaises(RuntimeError):
                        exec(self.cells[1], self.state)
                    with mock.patch.object(urllib.request, "urlopen", side_effect=responses([csv2, codebook, b"readme"])):
                        asyncio.run(eval(compiled, self.state))
                    second = self.state["raw_folder"]
                self.assertNotEqual(first, second)
                self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
                self.assertEqual((second / "owid_energy.csv").read_bytes(), csv2)
                self.assertEqual(len(list(first.parent.glob("*.incomplete"))), 1)
            finally:
                os.chdir(previous)

    def test_current_provider_unit_label_is_accepted_without_weakening_unit_check(self):
        self.state["codebook"].loc[0, "unit"] = "terawatt-hours (TWh)"
        self.sync_fixture_sources()
        self.select_world()
        self.assertEqual(self.state["selected"].loc[self.state["selected"].year.eq(2023), "generation_gwh"].item(), 1500.)
        self.assertEqual(self.state["selection_context"]["metadata"]["unit"], "terawatt-hours (TWh)")
        self.state["codebook"].loc[0, "unit"] = "terawatt-hours per capita (TWh)"
        self.sync_fixture_sources()
        with self.assertRaisesRegex(ValueError, "Unit changed"):
            self.select_world()

    def test_failed_reselection_cannot_export_previous_data(self):
        self.state.update(dict(json=json, uuid=uuid, sqlite3=sqlite3))
        for old, new in (('VARIABLE = "wind_electricity_twh"', 'VARIABLE = "unsupported"'),
                         ('COUNTRY_CODE = "AUT"', 'COUNTRY_CODE = "OWID_WRL"')):
            with self.subTest(setting=new):
                self.select_world()
                bad_selection = self.cells[1].replace(old, new)
                self.assertNotEqual(bad_selection, self.cells[1])
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    exec(bad_selection, self.state)
                self.assertNotIn("selection_context", self.state)
                with tempfile.TemporaryDirectory() as directory:
                    previous = Path.cwd()
                    try:
                        os.chdir(directory)
                        with self.assertRaises(RuntimeError):
                            exec(self.cells[2], self.state)
                        self.assertFalse(Path("outputs").exists())
                    finally:
                        os.chdir(previous)

    def test_bad_unit_invalidates_preceding_selection(self):
        self.select_world()
        self.state["codebook"].loc[0, "unit"] = "TWh per capita"
        self.sync_fixture_sources()
        with self.assertRaises(ValueError):
            self.select_world()
        self.assertNotIn("selection_context", self.state)
        with self.assertRaises(RuntimeError):
            exec(self.cells[2], self.state)


class EurostatDecoderTests(unittest.TestCase):
    def test_followup_csv_validation_survives_optimized_python(self):
        import math
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        lesson_cell = next(c for c in re.findall(r"```python\n(.*?)\n```", lesson, re.S)
                           if "# For a separate downloaded/uploaded CSV" in c)
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        notebook_cell = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and "# For a separate downloaded/uploaded CSV" in "".join(c["source"]))
        valid = pd.DataFrame({"geo": ["AT"] * 12, "source": ["RA300"] * 12,
                              "month": [f"2023-{m:02}" for m in range(1, 13)],
                              "unit": ["GWH"] * 12, "freq": ["M"] * 12,
                              "generation_gwh": [1.] * 11 + [float("nan")], "status_flag": [""] * 12})
        bad = {"missing row": valid.iloc[:-1], "unit": valid.assign(unit="KWH"),
               "country": valid.assign(geo="DE"), "source": valid.assign(source="RA420"),
               "frequency": valid.assign(freq="A"), "year": valid.assign(month=lambda d: d.month.str.replace("2023", "2022")),
               "duplicate": pd.concat([valid.iloc[:-1], valid.iloc[[0]]], ignore_index=True),
               "infinity": valid.assign(generation_gwh=float("inf")),
               "nonnumeric": valid.assign(generation_gwh="bad"), "column": valid.drop(columns="status_flag")}
        for name, cell in (("lesson", lesson_cell), ("notebook", notebook_cell)):
            code = compile(cell, name, "exec", optimize=1)
            for label, frame in [("valid", valid), *bad.items()]:
                state = {"pd": pd, "math": math, "output": Path("."), "download_completed": True,
                         "GEO": "AT", "SOURCE": "RA300", "YEAR": 2023}
                with self.subTest(artifact=name, case=label), mock.patch.object(pd, "read_csv", return_value=frame.copy()), contextlib.redirect_stdout(io.StringIO()):
                    if label == "valid":
                        exec(code, state)
                        self.assertEqual(state["check"].generation_gwh.notna().sum(), 11)
                    else:
                        with self.assertRaises((ValueError, TypeError)):
                            exec(code, state)

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        from download_energy_data import decode_series
        cls.decode = staticmethod(decode_series)
        cls.fixture = json.loads((BOOK / "data/examples/eurostat_at_wind_2023.json").read_text())

    def test_failed_rerun_cannot_read_its_incomplete_csv(self):
        import ast
        import asyncio
        import nbformat
        import sqlite3
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", lesson, re.S)
        notebook = nbformat.read(BOOK / "section5/download-to-database.ipynb", as_version=4)
        sources = [blocks, [cell.source for cell in notebook.cells if cell.cell_type == "code"]]
        for cells in sources:
            if cells is sources[1]:
                first = next(i for i, c in enumerate(cells) if c.startswith("# Step 1:"))
                last = next(i for i, c in enumerate(cells) if "output.rename(completed_output)" in c)
                main = "\n\n".join(cells[first:last + 1])
            else:
                main = next(c for c in cells if "def decode_series(" in c)
            read = next(c for c in cells if 'csv_path = output / "generation.csv"' in c)
            compiled = compile(main.replace("USE_LIVE = True", "USE_LIVE = False"), "lesson", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            with self.subTest(source="notebook" if cells is sources[1] else "lesson"), tempfile.TemporaryDirectory() as directory:
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    target = Path("data/examples")
                    target.mkdir(parents=True)
                    for name in ("eurostat_at_wind_2023.json", "eurostat_at_wind_2023.provenance.json"):
                        (target / name).write_bytes((BOOK / "data/examples" / name).read_bytes())
                    state = {}
                    with contextlib.redirect_stdout(io.StringIO()):
                        asyncio.run(eval(compiled, state))
                        exec(read, state)
                        self.assertTrue(state["download_completed"])
                        step2 = "# Step 2:" + main.split("# Step 2:", 1)[1].split("# Step 3:", 1)[0]
                        snapshot = target / "eurostat_at_wind_2023.json"
                        original = snapshot.read_bytes()
                        snapshot.write_bytes(original + b" ")
                        isolated_step = compile(step2, "step 2", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                        with self.assertRaisesRegex(ValueError, "does not match its provenance hash"):
                            asyncio.run(eval(isolated_step, state))
                        self.assertFalse(state["download_completed"])
                        with self.assertRaisesRegex(RuntimeError, "did not complete"):
                            exec(read, state)
                        snapshot.write_bytes(original)
                        asyncio.run(eval(compiled, state))
                        with mock.patch.object(sqlite3, "connect", side_effect=sqlite3.OperationalError("injected failure")):
                            with self.assertRaisesRegex(sqlite3.OperationalError, "injected failure"):
                                asyncio.run(eval(compiled, state))
                    self.assertFalse(state["download_completed"])
                    self.assertTrue((state["output"] / "generation.csv").exists())
                    with self.assertRaisesRegex(RuntimeError, "did not complete"):
                        exec(read, state)
                finally:
                    os.chdir(previous)

    def test_dataset_wide_status_is_preserved_in_all_three_deliveries(self):
        import ast
        import itertools
        import nbformat
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        source = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        notebook = nbformat.read(BOOK / "section5/download-to-database.ipynb", as_version=4)
        notebook_source = next(c.source for c in notebook.cells if c.cell_type == "code" and "def decode_series(" in c.source)
        decoders = [self.decode]
        for code in (source, notebook_source):
            function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "decode_series")
            state = {"math": math, "itertools": itertools}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "decoder", "exec"), state)
            decoders.append(state["decode_series"])
        fixture = copy.deepcopy(self.fixture)
        fixture["status"] = "ep"
        for decoder in decoders:
            with self.subTest(decoder=decoder):
                rows = decoder(fixture)
                self.assertEqual(len(rows), 12)
                self.assertEqual([row["status_flag"] for row in rows], ["ep"] * 12)

    def test_decoder_boundaries_match_in_terminal_lesson_and_notebook(self):
        import ast
        import itertools
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        lesson_code = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S)
                           if "def decode_series(" in c)
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        notebook_code = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and "def decode_series(" in "".join(c["source"]))
        decoders = [("terminal", self.decode)]
        for label, code in (("lesson", lesson_code), ("notebook", notebook_code)):
            function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "decode_series")
            state = {"math": math, "itertools": itertools}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), label, "exec"), state)
            decoders.append((label, state["decode_series"]))
        valid = copy.deepcopy(self.fixture)
        valid["dimension"]["geo"]["category"].pop("index")
        valid["value"] = [0., None] + [float(i) for i in range(2, 12)]
        valid["status"] = ["e", None] + [""] * 10
        expected = self.decode(valid)
        self.assertEqual([(row["generation_gwh"], row["status_flag"]) for row in expected[:2]],
                         [(0., "e"), (None, "")])
        malformed = []
        for field, value in (("value", valid["value"][:-1]), ("value", {"12": 1.}),
                             ("status", valid["status"][:-1]), ("status", {"0": {"bad": "e"}})):
            fixture = copy.deepcopy(valid)
            fixture[field] = value
            malformed.append((field, fixture))
        for label, decoder in decoders:
            with self.subTest(delivery=label):
                self.assertEqual(decoder(valid), expected)
                for field, fixture in malformed:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        decoder(fixture)

    def test_status_values_must_be_text_or_null_in_both_decoders(self):
        import ast
        import itertools
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        block = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        function = next(node for node in ast.parse(block).body if isinstance(node, ast.FunctionDef) and node.name == "decode_series")
        state = {"math": math, "itertools": itertools}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "lesson", "exec"), state)
        for decoder in (self.decode, state["decode_series"]):
            for invalid in ({"nested": "e"}, {}, 0, False, ["e"]):
                fixture = copy.deepcopy(self.fixture)
                fixture["status"] = {"0": invalid}
                with self.subTest(decoder=decoder.__module__, invalid=invalid), self.assertRaisesRegex(ValueError, "status must be text or null"):
                    decoder(fixture)
            fixture = copy.deepcopy(self.fixture)
            fixture["status"] = {"0": None, "1": "e"}
            rows = decoder(fixture)
            self.assertEqual(rows[0]["status_flag"], "")
            self.assertEqual(rows[1]["status_flag"], "e")

    def test_lesson_decoder_handles_singletons_and_rejects_bad_positions(self):
        import ast
        import itertools
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        block = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        tree = ast.parse(block)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_series")
        namespace = {"math": math, "itertools": itertools}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "lesson", "exec"), namespace)
        decode = namespace["decode_series"]
        fixture = copy.deepcopy(self.fixture)
        expected = decode(fixture)
        singleton = copy.deepcopy(fixture)
        singleton["dimension"]["geo"]["category"].pop("index")
        self.assertEqual(decode(singleton), expected)
        dense = copy.deepcopy(fixture)
        dense["value"] = [fixture["value"].get(str(i)) for i in range(12)]
        self.assertEqual(decode(dense), expected)
        for values in (dense["value"][:-1], dense["value"] + [99]):
            bad = copy.deepcopy(dense)
            bad["value"] = values
            with self.assertRaises(ValueError):
                decode(bad)
        for position in ("12", "-1", "bogus"):
            bad = copy.deepcopy(fixture)
            bad["value"][position] = 999999
            with self.subTest(position=position), self.assertRaises(ValueError):
                decode(bad)
        flagged = copy.deepcopy(fixture)
        flagged["status"] = "e"
        self.assertTrue(all(row["status_flag"] == "e" for row in decode(flagged)))

    def test_offline_missing_files_identify_both_required_inputs(self):
        import ast
        import asyncio
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        lesson_code = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        notebook_code = "\n".join("".join(c["source"]) for c in notebook["cells"]
                                  if c["cell_type"] == "code" and
                                  any(marker in "".join(c["source"]) for marker in
                                      ("# Step 1:", "# Step 2:")))
        for label, code in (("lesson", lesson_code), ("notebook", notebook_code)):
            compiled = compile(code.replace("USE_LIVE = True", "USE_LIVE = False"), label, "exec",
                               flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            with self.subTest(source=label), tempfile.TemporaryDirectory() as directory:
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    with self.assertRaises(FileNotFoundError) as missing_both:
                        asyncio.run(eval(compiled, {}))
                    self.assertIn("eurostat_at_wind_2023.json", str(missing_both.exception))
                    self.assertIn("eurostat_at_wind_2023.provenance.json", str(missing_both.exception))
                    target = Path("data/examples")
                    target.mkdir(parents=True)
                    (target / "eurostat_at_wind_2023.json").write_bytes(
                        (BOOK / "data/examples/eurostat_at_wind_2023.json").read_bytes())
                    with self.assertRaises(FileNotFoundError) as missing_provenance:
                        asyncio.run(eval(compiled, {}))
                    self.assertIn("eurostat_at_wind_2023.provenance.json", str(missing_provenance.exception))
                    self.assertIn(str(target), str(missing_provenance.exception))
                finally:
                    os.chdir(previous)

    def test_offline_snapshot_requires_its_recorded_hash_before_export(self):
        import ast
        import asyncio
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        compiled = compile(code.replace("USE_LIVE = True", "USE_LIVE = False"), "lesson", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        source = BOOK / "data/examples"
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                target = Path("data/examples")
                target.mkdir(parents=True)
                for name in ("eurostat_at_wind_2023.json", "eurostat_at_wind_2023.provenance.json"):
                    (target / name).write_bytes((source / name).read_bytes())
                with contextlib.redirect_stdout(io.StringIO()):
                    asyncio.run(eval(compiled, {}))
                    first = next(Path("outputs").iterdir())
                    before = {p.name: p.read_bytes() for p in first.iterdir()}
                    asyncio.run(eval(compiled, {}))
                completed = [p for p in Path("outputs").iterdir() if not p.name.endswith(".incomplete")]
                self.assertEqual(len(completed), 2)
                self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
                snapshot = target / "eurostat_at_wind_2023.json"
                snapshot.write_bytes(snapshot.read_bytes() + b" ")  # still valid JSON, different bytes
                with self.assertRaisesRegex(ValueError, "does not match its provenance hash"):
                    asyncio.run(eval(compiled, {}))
                self.assertEqual(len(list(Path("outputs").glob("*.incomplete"))), 1)
                self.assertEqual(len(list(Path("outputs").glob("*/generation.sqlite"))), 2)
                self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
            finally:
                os.chdir(previous)

    def test_revised_live_response_preserves_previous_retrieval(self):
        import ast
        import asyncio
        import hashlib
        import urllib.request
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c)
        compiled = compile(code, "lesson", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        revised = copy.deepcopy(self.fixture)
        key = next(iter(revised["value"]))
        revised["value"][key] = float(revised["value"][key]) + 1
        raw_versions = [json.dumps(x).encode() for x in (self.fixture, revised)]
        def response(raw):
            result = mock.MagicMock()
            result.__enter__.return_value.read.return_value = raw
            return result
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.object(urllib.request, "urlopen", side_effect=[response(raw) for raw in raw_versions]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        asyncio.run(eval(compiled, {}))
                        first = next(Path("outputs").iterdir())
                        before = {p.name: p.read_bytes() for p in first.iterdir()}
                        asyncio.run(eval(compiled, {}))
                folders = list(Path("outputs").iterdir())
                self.assertEqual(len(folders), 2)
                second = next(p for p in folders if p != first)
                self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
                for folder, raw in zip((first, second), raw_versions):
                    self.assertEqual((folder / "raw.json").read_bytes(), raw)
                    record = json.loads((folder / "provenance.json").read_text())
                    self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertNotEqual((first / "generation.csv").read_bytes(), (second / "generation.csv").read_bytes())
            finally:
                os.chdir(previous)

    def test_failed_export_preserves_completed_run(self):
        import hashlib
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = lesson.split("# Step 5: Persist both the provider response and the normalized data.", 1)[1]
        code = code.split("\n```", 1)[0]
        raw = json.dumps(self.fixture).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "previous"
            old.mkdir()
            for name in ("raw.json", "generation.csv", "provenance.json", "generation.sqlite"):
                (old / name).write_bytes(("old " + name).encode())
            before = {p.name: p.read_bytes() for p in old.iterdir()}
            incomplete = root / "new.incomplete"
            incomplete.mkdir()
            rows = self.decode(self.fixture)
            for index, row in enumerate(rows):
                row["status_flag"] = "n" if index < 7 else ""
            state = dict(output=incomplete, output_root=root, folder_name="new", raw=raw,
                frame=pd.DataFrame(rows), url="fixture", USE_LIVE=False, retrieved="fixture",
                payload=self.fixture, hashlib=hashlib, json=json, sqlite3=sqlite3,
                SOURCE="RA300", GEO="AT", pd=pd, math=math, observed=12,
                returned_months=[r["month"] for r in rows])
            with mock.patch.object(sqlite3, "connect", side_effect=sqlite3.OperationalError("injected failure")):
                with self.assertRaises(sqlite3.OperationalError):
                    exec(code, state)
            self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
            self.assertTrue(incomplete.is_dir())
            self.assertFalse((root / "new").exists())
            output_text = io.StringIO()
            with contextlib.redirect_stdout(output_text):
                exec(code, state)
            self.assertIn("'n': 7", output_text.getvalue())
            self.assertIn("'unflagged': 5", output_text.getvalue())
            self.assertIn("do not establish complete generation coverage", output_text.getvalue())
            self.assertTrue((root / "new" / "generation.sqlite").is_file())
            self.assertFalse(incomplete.exists())

    def test_reported_zero_months_keep_values_and_coverage_caveat(self):
        import hashlib
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        generator = (ROOT / "tools/build_download_lesson.py").read_text(encoding="utf-8")
        for text in (lesson, notebook_text, generator):
            self.assertIn("or explain Norway's reporting; investigate each country separately", text)
            self.assertIn("gross/net boundary", text)
        validate = lesson.split("# Step 4: Validate the requested slice and make absent months visible.", 1)[1].split('print(payload["label"])', 1)[0]
        export = lesson.split("# Step 5: Persist both the provider response and the normalized data.", 1)[1].split("\n```", 1)[0]
        fixture = copy.deepcopy(self.fixture)
        fixture["value"] = {str(i): 0 for i in range(12)}
        fixture["dimension"]["siec"]["category"] = {"index": {"RA420": 0}, "label": {"RA420": "Solar photovoltaic"}}
        for flag in ('', 'n'):
            fixture['status'] = {str(i): flag for i in range(12)} if flag else {}
            rows = self.decode(fixture)
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                incomplete = root / "zero.incomplete"
                incomplete.mkdir()
                state = dict(output=incomplete, output_root=root, folder_name="zero", raw=json.dumps(fixture).encode(),
                    frame=pd.DataFrame(rows), url="fixture", USE_LIVE=False, retrieved="fixture",
                    payload=fixture, hashlib=hashlib, json=json, sqlite3=sqlite3,
                    SOURCE="RA420", GEO="AT", YEAR=2023, pd=pd, math=math)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(validate, state)
                    exec(export, state)
                self.assertEqual(state["observed"], 12)
                self.assertTrue(state["frame"].generation_gwh.eq(0).all())
                self.assertIn("do not establish complete generation coverage", output.getvalue())
                self.assertIn(f"'{flag or 'unflagged'}': 12", output.getvalue())
                with sqlite3.connect(root / "zero" / "generation.sqlite") as db:
                    self.assertEqual(db.execute("SELECT COUNT(generation_gwh), SUM(generation_gwh) FROM generation").fetchone(), (12, 0))
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM generation WHERE status_flag = ?", (flag,)).fetchone()[0], 12)

    def test_export_reconciliation_still_rejects_bad_coverage_when_optimized(self):
        import hashlib
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = lesson.split("# Step 5: Persist both the provider response and the normalized data.", 1)[1].split("\n```", 1)[0]
        rows = self.decode(self.fixture)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "bad.incomplete"
            incomplete.mkdir()
            state = dict(output=incomplete, output_root=root, folder_name="bad", raw=json.dumps(self.fixture).encode(),
                frame=pd.DataFrame(rows), url="fixture", USE_LIVE=False, retrieved="fixture",
                payload=self.fixture, hashlib=hashlib, json=json, sqlite3=sqlite3,
                SOURCE="RA300", GEO="AT", pd=pd, math=math, observed=11,
                returned_months=[r["month"] for r in rows])
            with self.assertRaisesRegex(ValueError, "coverage or totals disagree"):
                exec(compile(code, "export", "exec", optimize=1), state)
            self.assertFalse((root / "bad").exists())

    def test_cancelling_totals_reconcile_in_actual_export(self):
        import hashlib
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = lesson.split("# Step 5: Persist both the provider response and the normalized data.", 1)[1].split("\n```", 1)[0]
        rows = self.decode(self.fixture)
        values = [810.071, 987.738, 780.597, -881.404, -807.933, -822.011,
                  -242.808, 752.168, -645.406, 543.440, 696.516, -1170.968]
        for row, value in zip(rows, values):
            row["generation_gwh"] = value
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "cancel.incomplete"
            incomplete.mkdir()
            state = dict(output=incomplete, output_root=root, folder_name="cancel", raw=json.dumps(self.fixture).encode(),
                frame=pd.DataFrame(rows), url="fixture", USE_LIVE=False, retrieved="fixture",
                payload=self.fixture, hashlib=hashlib, json=json, sqlite3=sqlite3,
                SOURCE="RA300", GEO="AT", pd=pd, math=math, observed=12,
                returned_months=[r["month"] for r in rows])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, state)
            with sqlite3.connect(root / "cancel" / "generation.sqlite") as db:
                total, count = db.execute("SELECT SUM(generation_gwh), COUNT(*) FROM generation").fetchone()
            self.assertEqual(count, 12)
            self.assertLess(abs(total), 1e-9)
            self.assertTrue((root / "cancel" / "generation.csv").is_file())

    def test_absent_months_are_not_counted_as_unflagged(self):
        import hashlib
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = lesson.split("# Step 5: Persist both the provider response and the normalized data.", 1)[1].split("\n```", 1)[0]
        rows = self.decode(self.fixture)
        for index, row in enumerate(rows):
            row["status_flag"] = "n" if index < 7 else ""
            if index >= 9:
                row["generation_gwh"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "new.incomplete"
            incomplete.mkdir()
            state = dict(output=incomplete, output_root=root, folder_name="new", raw=json.dumps(self.fixture).encode(),
                frame=pd.DataFrame(rows), url="fixture", USE_LIVE=True, retrieved="fixture",
                payload=self.fixture, hashlib=hashlib, json=json, sqlite3=sqlite3,
                SOURCE="RA300", GEO="AT", pd=pd, math=math, observed=9,
                returned_months=[r["month"] for r in rows[:9]])
            output_text = io.StringIO()
            with contextlib.redirect_stdout(output_text):
                exec(code, state)
            self.assertIn("'n': 7", output_text.getvalue())
            self.assertIn("'unflagged': 2", output_text.getvalue())
            self.assertIn("Absent month categories have no provider status: 3", output_text.getvalue())

    def test_dense_and_sparse_invalid_positions_are_rejected(self):
        rows = self.decode(self.fixture)
        bad = copy.deepcopy(self.fixture)
        bad["value"] = [r["generation_gwh"] for r in rows][:-1]
        with self.assertRaises(ValueError):
            self.decode(bad)
        bad = copy.deepcopy(self.fixture)
        bad["value"]["12"] = 1
        with self.assertRaises(ValueError):
            self.decode(bad)

    def test_dimension_sizes_and_category_positions_are_validated_in_all_decoders(self):
        import ast
        import itertools
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        sources = [next(c for c in re.findall(r"```python\n(.*?)```", lesson, re.S) if "def decode_series(" in c),
                   next("".join(c["source"]) for c in notebook["cells"] if "def decode_series(" in "".join(c["source"]))]
        decoders = [("terminal", self.decode)]
        for name, source in zip(("lesson", "notebook"), sources):
            function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "decode_series")
            namespace = {"math": math, "itertools": itertools}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), name, "exec"), namespace)
            decoders.append((name, namespace["decode_series"]))
        invalid = []
        for value in (True, 1.5, -1):
            payload = copy.deepcopy(self.fixture); payload["size"][0] = value; invalid.append((f"size {value!r}", payload))
        for index in ("AT", {"AT": False}, {"AT": 0.5}, ["AT", "AT"], [1]):
            payload = copy.deepcopy(self.fixture); payload["dimension"]["geo"]["category"]["index"] = index
            invalid.append((f"index {index!r}", payload))
        for name, decode in decoders:
            self.assertEqual(len(decode(copy.deepcopy(self.fixture))), 12)
            integral_numbers = copy.deepcopy(self.fixture)
            integral_numbers["size"][0] = 1.0
            integral_numbers["dimension"]["geo"]["category"]["index"] = {"AT": 0.0}
            self.assertEqual(len(decode(integral_numbers)), 12)
            for label, payload in invalid:
                with self.subTest(artifact=name, case=label), self.assertRaises(ValueError):
                    decode(copy.deepcopy(payload))
    def test_singleton_dimension_and_dataset_wide_status(self):
        sample = copy.deepcopy(self.fixture)
        del sample["dimension"]["geo"]["category"]["index"]
        sample["status"] = "estimated"
        rows = self.decode(sample)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["status_flag"] == "estimated" for row in rows))

    def test_partial_year_is_reindexed_and_bad_year_is_rejected_even_optimized(self):
        lesson = (BOOK / "section5/download-to-database.md").read_text(encoding="utf-8")
        code = lesson.split("# Step 4: Validate the requested slice and make absent months visible.", 1)[1]
        code = code.split('print(payload["label"])', 1)[0]
        notebook = json.loads((BOOK / "section5/download-to-database.ipynb").read_text(encoding="utf-8"))
        notebook_code = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and "# Step 4: Validate the requested slice and make absent months visible." in "".join(c["source"]))
        notebook_code = notebook_code.split("# Step 4: Validate the requested slice and make absent months visible.", 1)[1]
        notebook_code = notebook_code.split('print(payload["label"])', 1)[0]
        self.assertEqual(code.strip(), notebook_code.strip())
        rows = self.decode(self.fixture)[:9]
        state = dict(pd=pd, frame=pd.DataFrame(rows), YEAR=2023, GEO="AT", SOURCE="RA300")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "slice", "exec", optimize=1), state)
        self.assertEqual(len(state["frame"]), 12)
        self.assertEqual(len(state["returned_months"]), 9)
        self.assertEqual(state["frame"].generation_gwh.notna().sum(), 9)
        original = pd.DataFrame(rows)
        bad_frames = {
            "year": original.assign(month=lambda f: f.month.str.replace("2023", "2022")),
            "country": original.assign(geo="DE"),
            "source": original.assign(source="RA420"),
            "unit": original.assign(unit="KWH"),
            "frequency": original.assign(freq="A"),
            "duplicate": pd.concat([original, original.iloc[[0]]], ignore_index=True),
        }
        for label, bad in bad_frames.items():
            state = dict(pd=pd, frame=bad, YEAR=2023, GEO="AT", SOURCE="RA300")
            with self.subTest(label=label), self.assertRaises(ValueError):
                exec(compile(code, "slice", "exec", optimize=1), state)


class EurostatTerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        import download_energy_data
        cls.module = download_energy_data
        cls.fixture = json.loads((BOOK / "data/examples/eurostat_at_wind_2023.json").read_text())

    def response(self, payload):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(payload).encode()
        context = mock.MagicMock()
        context.__enter__.return_value = response
        return context

    def test_late_export_failures_preserve_previous_completed_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(self.module, "urlopen", side_effect=lambda *a, **k: self.response(self.fixture)):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.module.download(root / "result")
                completed = next(p for p in root.glob("result_*") if not p.name.endswith(".incomplete"))
                original = {p.name: p.read_bytes() for p in completed.iterdir()}
                for method in ("write_text", "rename"):
                    original_method = getattr(Path, method)
                    def fail_late(path, *args, _method=method, _original=original_method, **kwargs):
                        if (_method == "write_text" and path.name == "provenance.json") or (
                                _method == "rename" and path.name.endswith(".incomplete")):
                            raise PermissionError("injected late export failure")
                        return _original(path, *args, **kwargs)
                    with self.subTest(stage=method), mock.patch.object(Path, method, fail_late):
                        with self.assertRaises(PermissionError), contextlib.redirect_stdout(io.StringIO()):
                            self.module.download(root / "result")
                    self.assertEqual(original, {p.name: p.read_bytes() for p in completed.iterdir()})
                self.assertEqual(len(list(root.glob("result_*.incomplete"))), 2)
                self.assertEqual(len([p for p in root.glob("result_*") if not p.name.endswith(".incomplete")]), 1)

    def test_dotted_prefixes_and_failed_write_preserve_prior_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(self.module, "urlopen", side_effect=lambda *a, **k: self.response(self.fixture)):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.module.download(root / "austria.v1")
                    self.module.download(root / "austria.v2")
                first = list(root.glob("austria.v1_*")); second = list(root.glob("austria.v2_*"))
                self.assertEqual((len(first), len(second)), (1, 1))
                before = {p.name: p.read_bytes() for p in first[0].iterdir()}
                original_open = Path.open
                def fail_csv(path, *args, **kwargs):
                    if path.name == "generation.csv":
                        raise PermissionError("injected CSV failure")
                    return original_open(path, *args, **kwargs)
                with mock.patch.object(Path, "open", fail_csv):
                    with self.assertRaises(PermissionError):
                        self.module.download(root / "austria.v1")
                self.assertEqual(before, {p.name: p.read_bytes() for p in first[0].iterdir()})
                self.assertEqual(len(list(root.glob("austria.v1_*.incomplete"))), 1)

    def test_empty_selection_stops_before_creating_a_retrieval(self):
        payload = copy.deepcopy(self.fixture)
        geo_axis = payload["id"].index("geo")
        payload["size"][geo_axis] = 0
        payload["dimension"]["geo"]["category"]["index"] = {}
        payload["dimension"]["geo"]["category"]["label"] = {}
        payload["value"] = {}
        payload.pop("status", None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(self.module, "urlopen", return_value=self.response(payload)):
                with self.assertRaisesRegex(ValueError, "No rows returned"):
                    self.module.download(root / "result")
            self.assertEqual(list(root.iterdir()), [])

    def test_summary_distinguishes_missing_zero_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            for values, expected in (({}, "0/12 values reported\nObserved-period sum: unavailable"),
                                      ({str(i): 0 for i in range(12)}, "12/12 values reported\nObserved-period sum: 0.000 GWh"),
                                      ({"0": 3}, "1/12 values reported\nObserved-period sum: 3.000 GWh")):
                payload = copy.deepcopy(self.fixture)
                payload["value"] = values
                out = io.StringIO()
                with mock.patch.object(self.module, "urlopen", return_value=self.response(payload)):
                    with contextlib.redirect_stdout(out):
                        self.module.download(Path(directory) / "result")
                self.assertIn(expected, out.getvalue())

class DownloadRegenerationTests(unittest.TestCase):
    def test_student_build_guard_rejects_saved_paths_even_when_optimized(self):
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools_dir = root / "tools"; tools_dir.mkdir()
            for name in ("stage_course_web_assets.py", "check_student_build.py"):
                shutil.copyfile(ROOT / "tools" / name, tools_dir / name)
            build = root / "EAGE_PythonRenewableEnergyCourse/_build/html"; build.mkdir(parents=True)
            (build / "index.html").write_text("<html></html>", encoding="utf-8")
            questions = "".join(f'<div id="question-{i}" class="difficulty-badge"></div>' for i in range(1, 11))
            for topic in ("python", "python-sql", "hydroelectric", "solar", "wind", "geothermal"):
                page = build / "section7/assessment" / topic / "index.html"
                page.parent.mkdir(parents=True)
                page.write_text(questions, encoding="utf-8")
            notebook = build / "sample.ipynb"
            notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["print(1)"],
                                                       "outputs": [{"text": ["/home/old-user/output.csv"]}]}]}), encoding="utf-8")
            def run():
                return subprocess.run([sys.executable, "-O", str(tools_dir / "check_student_build.py")],
                                      capture_output=True, text=True, check=False)
            rejected = run()
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Machine-specific home path", rejected.stderr)
            notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["print(1)"],
                                                       "outputs": [{"text": ["ordinary output"]}]}]}), encoding="utf-8")
            self.assertEqual(run().returncode, 0)
            (build / "private.md").write_text("Instructor-only test answer key", encoding="utf-8")
            private = run()
            self.assertNotEqual(private.returncode, 0)
            self.assertIn("Private solution content", private.stderr)

    def test_staging_guard_rejects_missing_or_late_notebook_download_handler(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("download_staging", ROOT / "tools/stage_course_web_assets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "notebook-download.js"
            script.write_bytes((BOOK / "notebook-download.js").read_bytes())
            for route in module.NOTEBOOK_DOWNLOAD_PAGES:
                page = root / route / "index.html"
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text('<html><head><script src="notebook-download.js"></script></head><body></body></html>', encoding="utf-8")
            module.verify_notebook_download_staging(root)
            extra_routes = {"section8/geothermal-project", "section8/projectintro",
                            "section4/timeseriesenergyconsumption", "section5/sql-pandas", "section5/sql-python"}
            self.assertTrue(extra_routes.issubset(module.NOTEBOOK_DOWNLOAD_PAGES))
            for route in (module.NOTEBOOK_DOWNLOAD_PAGES[0], *sorted(extra_routes)):
                page = root / route / "index.html"
                early = page.read_text(encoding="utf-8")
                page.write_text('<html><head></head><body><script src="notebook-download.js"></script></body></html>', encoding="utf-8")
                with self.subTest(route=route), self.assertRaisesRegex(ValueError, "Early notebook download handler missing"):
                    module.verify_notebook_download_staging(root)
                page.write_text(early, encoding="utf-8")
            page.unlink()
            with self.assertRaisesRegex(ValueError, "Notebook download page missing"):
                module.verify_notebook_download_staging(root)
            script.write_text("wrong handler", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its source"):
                module.verify_notebook_download_staging(root)

    def test_staged_notebooks_remove_saved_home_paths_without_changing_code(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage_course", ROOT / "tools/stage_course_web_assets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"cells": [{"cell_type": "code", "source": ["print('hello')\n"], "execution_count": 4,
                                   "outputs": [{"output_type": "stream", "name": "stdout", "text": ["/home/example/course/output.csv\n"]},
                                               {"output_type": "stream", "name": "stdout", "text": ["safe result\n"]},
                                               {"output_type": "execute_result", "data": {"text/plain": ["C:\\Users\\Example Person\\file.csv"]}}]}],
                        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
            dirty = root / "dirty.ipynb"; dirty.write_text(json.dumps(original), encoding="utf-8")
            clean = root / "clean.ipynb"; clean.write_text(json.dumps({"cells": [], "nbformat": 4}), encoding="utf-8")
            before_clean = clean.read_bytes()
            self.assertEqual(module.clear_machine_paths_from_notebook_outputs(root), 1)
            actual = json.loads(dirty.read_text(encoding="utf-8"))["cells"][0]
            self.assertEqual(actual["source"], original["cells"][0]["source"])
            self.assertEqual(actual["outputs"], [original["cells"][0]["outputs"][1]])
            self.assertIsNone(actual["execution_count"])
            self.assertEqual(clean.read_bytes(), before_clean)
            self.assertEqual(module.clear_machine_paths_from_notebook_outputs(root), 0)

    def test_download_and_geothermal_generators_match_saved_content(self):
        import nbformat
        import runpy
        generated_text, generated_notebooks = {}, {}
        def capture_text(path, data, *args, **kwargs):
            generated_text[path] = data
            return len(data)
        def capture_notebook(notebook, path, *args, **kwargs):
            generated_notebooks[Path(path)] = notebook
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]), \
             mock.patch.object(Path, "write_text", capture_text), \
             mock.patch.object(nbformat, "write", capture_notebook), \
             contextlib.redirect_stdout(io.StringIO()):
            for script in ("build_download_lesson.py", "build_geothermal_project_notebook.py"):
                runpy.run_path(str(ROOT / "tools" / script))
        self.assertEqual(len(generated_text), 1)
        self.assertEqual(len(generated_notebooks), 2)
        for path, text in generated_text.items():
            with self.subTest(path=path):
                self.assertEqual(text, path.read_text(encoding="utf-8"))
        for path, generated in generated_notebooks.items():
            actual = nbformat.read(path, as_version=4)
            with self.subTest(path=path):
                self.assertEqual([(c.cell_type, c.source) for c in generated.cells],
                                 [(c.cell_type, c.source) for c in actual.cells])


class WorkbookRegenerationTests(unittest.TestCase):
    def test_standalone_notebook_navigation_and_hydro_companions_are_staged(self):
        notebooks = list((BOOK / "section7/workbook").glob("*.ipynb"))
        notebooks += [BOOK / "section0/introRenewableEnergy.ipynb"]
        notebooks += [BOOK / "section6" / (topic + "Energy.ipynb")
                      for topic in ("hydroelectric", "solar", "wind", "geothermal")]
        notebooks += [BOOK / "section5" / name for name in ("download-to-database.ipynb", "world-energy-download.ipynb")]
        self.assertEqual(len(notebooks), 12)
        for notebook in notebooks:
            cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
            prose = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose):
                if re.search(r"\.(?:md|ipynb|png|svg)(?:[?#].*)?$", target, re.I):
                    with self.subTest(notebook=notebook.name, target=target):
                        self.assertRegex(target, r"^https?://")
        hydro = json.loads((BOOK / "section6/hydroelectricEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in hydro["cells"] if c["cell_type"] == "markdown")
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        assets = {asset["path"]: asset for asset in manifest["assets"]}
        for name in ("run-of-river-daily.csv", "run-of-river-yield.svg"):
            path = "section6/" + name
            self.assertIn("/sandbox/generated/assets/" + path, prose)
            self.assertIn(path, assets)
            self.assertEqual((BOOK / path).read_bytes(), (ROOT / "EAGE_pythonSandBox" / assets[path]["url"]).read_bytes())

    def test_restored_geothermal_notebook_links_target_mounted_explorers(self):
        from bs4 import BeautifulSoup
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        page = next(p for p in manifest["pages"] if p["id"] == "section6/geothermalEnergy.md")
        soup = BeautifulSoup(page["html"], "html.parser")
        app = (ROOT / "EAGE_pythonSandBox/app.js").read_text(encoding="utf-8")
        for mode in ("systems", "conduction", "area", "energy", "performance"):
            anchor = "geo-closed-" + mode
            with self.subTest(mode=mode):
                self.assertIn("&anchor=" + anchor, text)
                self.assertIsNotNone(soup.find(id=anchor))
                self.assertIn("['" + anchor + "', 'closed-loop.html?demo=" + mode + "'", app)

    def test_notebook_reference_formats_undated_and_edited_work(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("workbook_exporter_format", ROOT / "tools/build_chapter_exercises.py")
        generator = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "references.bib").write_text(
                "@book{sample,\n editor = {Example Editor},\n title = {Example Volume},\n edition = {3},\n url = {https://example.org}\n}\n", encoding="utf-8")
            lesson = folder / "lesson.md"
            lesson.write_text("# Lesson\n\nSee [@sample].\n", encoding="utf-8")
            output = []
            with mock.patch.object(generator, "ROOT", folder), \
                 mock.patch.object(generator, "notebook_citations", side_effect=lambda text: text), \
                 mock.patch.object(generator, "rewrite_notebook_links", side_effect=lambda text, path: text), \
                 mock.patch.object(generator.nbformat, "write", side_effect=lambda nb, path: output.append(nb)):
                generator.export_notebook(lesson)
            reference = output[0].cells[-1].source
            self.assertIn("Example Editor (eds.)", reference)
            self.assertIn("Example Volume (n.d.)", reference)
            self.assertIn("Edition 3.", reference)
            self.assertNotIn("()", reference)

    def test_workbook_notebooks_match_regenerated_content(self):
        import importlib.util
        import nbformat
        spec = importlib.util.spec_from_file_location("workbook_exporter", ROOT / "tools/build_chapter_exercises.py")
        generator = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(generator)
        for topic, _, relative in generator.WORKBOOK_CHAPTERS:
            generated = []
            with mock.patch.object(generator.nbformat, "write", side_effect=lambda nb, path: generated.append(nb)):
                generator.export_notebook(BOOK / relative)
            actual = nbformat.read((BOOK / relative).with_suffix(".ipynb"), as_version=4)
            with self.subTest(topic=topic):
                self.assertEqual([(c.cell_type, c.source) for c in generated[0].cells],
                                 [(c.cell_type, c.source) for c in actual.cells])
                self.assertNotIn(".. ", actual.cells[-1].source)
        for topic, relative in generator.PATHS.items():
            generated = []
            with mock.patch.object(generator.nbformat, "write", side_effect=lambda nb, path: generated.append(nb)):
                generator.export_notebook(BOOK / relative)
            actual = nbformat.read((BOOK / relative).with_suffix(".ipynb"), as_version=4)
            with self.subTest(theory=topic):
                self.assertEqual([(c.cell_type, c.source) for c in generated[0].cells],
                                 [(c.cell_type, c.source) for c in actual.cells])
                self.assertNotIn(".. ", actual.cells[-1].source)

    def test_workbook_source_and_introduction_match_generator(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("workbook_generator", ROOT / "tools/build_chapter_exercises.py")
        generator = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(generator)
        for topic, title, relative in generator.WORKBOOK_CHAPTERS:
            with self.subTest(topic=topic):
                self.assertEqual(generator.workbook_text(topic, title, relative),
                                 (BOOK / relative).read_text(encoding="utf-8"))
        self.assertEqual(generator.workbook_intro(),
                         (BOOK / "section7/renewableExercises.md").read_text(encoding="utf-8"))


class HistoricalEurostatTests(unittest.TestCase):
    def test_historical_notebook_validation_survives_optimization(self):
        from contextlib import closing
        from types import SimpleNamespace
        for relative in ("section4/timeSeriesEnergyConsumption.ipynb", "section5/SQL_Pandas.ipynb"):
            notebook = json.loads((BOOK / relative).read_text(encoding="utf-8"))
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            code = next(c for c in cells if "negative_rows = generation" in c and "country_coverage" in c)
            first = pd.Timestamp("2016-01-01") if relative.startswith("section4") else "2016-01-01"
            last = pd.Timestamp("2024-07-01") if relative.startswith("section4") else "2024-07-01"
            frame = pd.DataFrame({"country": ["AT", "AT"], "month": [first, last],
                                  "source": ["wind", "wind"], "generation_gwh": [1., 2.]})
            state = {"pd": pd, "generation": pd.concat([frame, frame.iloc[[0]]]),
                     "sources": pd.DataFrame({"source": ["wind"]})}
            with self.subTest(notebook=relative, failure="duplicate"), self.assertRaisesRegex(ValueError, "Duplicate"):
                exec(compile(code, relative, "exec", optimize=1), state)
            if relative.startswith("section5"):
                state["generation"] = frame
                state["sources"] = pd.DataFrame({"source": ["solar"]})
                with self.subTest(notebook=relative, failure="source mismatch"), self.assertRaisesRegex(ValueError, "sources do not match"):
                    exec(compile(code, relative, "exec", optimize=1), state)
                state["sources"] = pd.DataFrame({"source": ["wind"]})
            state["generation"] = frame.iloc[[0]]
            with self.subTest(notebook=relative, failure="month range"), self.assertRaisesRegex(ValueError, "month range"):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, relative, "exec", optimize=1), state)

        code = next(c for c in cells if "PRAGMA integrity_check" in c)
        selected = pd.DataFrame({"month": pd.to_datetime(["2024-01-01"]), "selected_generation_gwh": [1.]})
        data = pd.DataFrame({"month": ["2024-01-01"], "source": ["wind"], "generation_gwh": [1.]})
        for integrity, duplicates in (("corrupt", 0), ("ok", 1)):
            class Connection:
                def execute(self, sql):
                    return SimpleNamespace(fetchone=lambda: (integrity if "PRAGMA" in sql else duplicates,))
                def close(self):
                    pass
            state = {"pd": pd, "sources": pd.DataFrame({"source": ["wind"], "in_selected_group": [1]}),
                     "generation": data, "selected_monthly": selected, "closing": closing,
                     "database_rebuild_completed": True, "database_queries_completed": True,
                     "sqlite3": SimpleNamespace(connect=lambda _: Connection()), "DB_PATH": ":memory:"}
            with self.subTest(integrity=integrity, duplicates=duplicates), self.assertRaisesRegex(ValueError, "integrity"):
                exec(compile(code, "reconciliation", "exec", optimize=1), state)

    def test_common_coverage_uses_identical_reported_pairs(self):
        import ast
        notebook = json.loads((BOOK / "section5/SQL_Pandas.ipynb").read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "def compare_common_coverage(" in "".join(c["source"]))
        fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
        state = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "notebook", "exec"), state)
        compare = state["compare_common_coverage"]
        sources = pd.DataFrame({"source": ["wind", "other"], "in_selected_group": [1, 0]})
        frame = pd.DataFrame([
            ("A", "wind", "2024-05-01", 0., "e"),
            ("A", "wind", "2024-06-01", 5., ""),
            ("B", "wind", "2024-05-01", 100., ""),
            ("B", "wind", "2024-06-01", None, ""),
            ("C", "wind", "2024-06-01", 200., ""),
            ("A", "other", "2024-05-01", 50., ""),
            ("A", "other", "2024-06-01", 70., ""),
        ], columns=["country", "source", "month", "generation_gwh", "status_flag"])
        result = compare(frame, sources, "2024-05-01", "2024-06-01")
        self.assertEqual((result["common_country_source_pairs"], result["countries"]), (1, 1))
        self.assertEqual((result["first_gwh"], result["second_gwh"]), (0., 5.))
        self.assertEqual((result["first_flagged_rows"], result["second_flagged_rows"]), (1, 0))
        no_overlap = compare(frame, sources, "2024-05-01", "2024-07-01")
        self.assertEqual(no_overlap["common_country_source_pairs"], 0)
        self.assertTrue(pd.isna(no_overlap["first_gwh"]) and pd.isna(no_overlap["second_gwh"]))
        with self.assertRaises(pd.errors.MergeError):
            compare(pd.concat([frame, frame.iloc[[0]]]), sources, "2024-05-01", "2024-06-01")

    def test_query_and_reconciliation_close_database_on_success_and_failure(self):
        from contextlib import closing
        from types import SimpleNamespace
        notebook = json.loads((BOOK / "section5/SQL_Pandas.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        writer = next(c for c in cells if "CREATE TABLE generation" in c)
        query = next(c for c in cells if "selected_query =" in c)
        reconcile = next(c for c in cells if "PRAGMA integrity_check" in c)
        with tempfile.TemporaryDirectory() as directory:
            opened = []
            def connect(path):
                connection = sqlite3.connect(path)
                opened.append(connection)
                return connection
            state = {"sqlite3": SimpleNamespace(connect=connect), "DB_PATH": str(Path(directory) / "test.sqlite"),
                     "closing": closing, "pd": pd,
                     "input_validation_completed": True,
                     "sources": pd.DataFrame({"source": ["wind"], "source_group": ["renewable"],
                                              "in_selected_group": [1]}),
                     "generation": pd.DataFrame({"country": ["AT"], "month": ["2024-01-01"],
                                                 "source": ["wind"], "generation_gwh": [1.],
                                                 "status_flag": [""]})}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(writer, state)
                exec(query, state)
                exec(reconcile, state)
                with mock.patch.object(pd, "read_sql_query", side_effect=RuntimeError("injected query failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected query failure"):
                        exec(query, state)
                with self.assertRaisesRegex(RuntimeError, "queries must complete"):
                    exec(reconcile, state)
                exec(query, state)
                with sqlite3.connect(state["DB_PATH"]) as connection:
                    connection.execute("DROP TABLE generation")
                with self.assertRaises(sqlite3.OperationalError):
                    exec(reconcile, state)
            self.assertEqual(len(opened), 6)
            for connection in opened:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_notebook_closes_database_on_success_and_failure(self):
        from contextlib import closing
        from types import SimpleNamespace
        notebook = json.loads((BOOK / "section5/SQL_Pandas.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "CREATE TABLE generation" in "".join(c["source"]))
        for fail in (False, True):
            opened = []
            def connect(path):
                connection = sqlite3.connect(path)
                opened.append(connection)
                return connection
            sources = pd.DataFrame({"source": ["wind", "wind"] if fail else ["wind"],
                                    "source_group": ["renewable"] * (2 if fail else 1),
                                    "in_selected_group": [1] * (2 if fail else 1)})
            generation = pd.DataFrame({"country": ["AT"], "month": ["2024-01"], "source": ["wind"],
                                       "generation_gwh": [1.], "status_flag": [""]})
            state = {"sqlite3": SimpleNamespace(connect=connect), "DB_PATH": ":memory:", "closing": closing, "pd": pd,
                     "sources": sources, "generation": generation, "input_validation_completed": True}
            if fail:
                with self.assertRaises(sqlite3.IntegrityError):
                    exec(cell, state)
            else:
                exec(cell, state)
            self.assertEqual(len(opened), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                opened[0].execute("SELECT 1")

    def test_failed_historical_database_rebuild_preserves_previous_tables(self):
        from contextlib import closing
        notebook = json.loads((BOOK / "section5/SQL_Pandas.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "CREATE TABLE generation" in "".join(c["source"]))
        query = next("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code" and "selected_query =" in "".join(c["source"]))
        reconcile = next("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "code" and "PRAGMA integrity_check" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "history.sqlite")
            sources = pd.DataFrame({"source": ["wind"], "source_group": ["renewable"], "in_selected_group": [1]})
            generation = pd.DataFrame({"country": ["AT"], "month": ["2024-01"], "source": ["wind"],
                                       "generation_gwh": [1.], "status_flag": [None]})
            state = {"sqlite3": sqlite3, "DB_PATH": db, "closing": closing, "pd": pd,
                     "sources": sources, "generation": generation, "input_validation_completed": True}
            state["input_validation_completed"] = False
            with self.assertRaisesRegex(RuntimeError, "Input validation did not complete"):
                exec(cell, state)
            state["input_validation_completed"] = True
            exec(cell, state)
            state["generation"] = pd.concat([generation, generation], ignore_index=True)
            with self.assertRaises(sqlite3.IntegrityError):
                exec(cell, state)
            with self.assertRaisesRegex(RuntimeError, "rebuild did not complete"):
                exec(query, state)
            with self.assertRaisesRegex(RuntimeError, "rebuild and queries must complete"):
                exec(reconcile, state)
            with sqlite3.connect(db) as connection:
                self.assertEqual(connection.execute("SELECT country, generation_gwh FROM generation").fetchall(), [("AT", 1.0)])
                self.assertEqual(connection.execute("SELECT source FROM source").fetchall(), [("wind",)])

    def test_notebook_source_labels_match_bundled_workbook_categories(self):
        import ast
        from openpyxl import load_workbook
        notebooks = [BOOK / "section5/SQL_Pandas.ipynb", BOOK / "section4/timeSeriesEnergyConsumption.ipynb"]
        expected = {"coal_and_manufactured_gases": "Coal and manufactured gases",
                    "nuclear_and_other_fuels_nec": "Nuclear fuels and other fuels n.e.c.",
                    "oil": "Oil and petroleum products (excluding biofuel portion)"}
        source_dir = BOOK / "data/section4/euroStat"
        for notebook in notebooks:
            cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
            prose = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
            self.assertIn("oil and petroleum products, excluding the biofuel portion", prose)
            if notebook.name == "SQL_Pandas.ipynb":
                self.assertIn("both paths can share the same taxonomy mistake", prose)
                self.assertIn("Validate source labels and category boundaries separately", prose)
            source = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "SOURCE_FILES =" in "".join(c["source"]))
            assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "SOURCE_FILES" for t in n.targets))
            files = ast.literal_eval(assignment.value)
            for key, category in expected.items():
                entry = files[key]
                filename = entry[0] if isinstance(entry, tuple) else entry
                book = load_workbook(source_dir / filename, read_only=True, data_only=True)
                try:
                    headers = list(book.active.iter_rows(min_row=1, max_row=10, values_only=True))
                    category_values = [next((v for v in row[1:] if v is not None), None) for row in headers
                                       if row[0] == "Standard international energy product classification (SIEC)"]
                    self.assertEqual(category_values, [category], (notebook, key))
                finally:
                    book.close()

    def test_notebook_generators_preserve_current_cells(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("course_notebook_generator", ROOT / "tools/rebuild_course_notebooks.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        for make, rel in ((generator.time_series_notebook, "section4/timeSeriesEnergyConsumption.ipynb"),
                          (generator.sql_notebook, "section5/SQL_Pandas.ipynb")):
            actual = json.loads((BOOK / rel).read_text(encoding="utf-8"))["cells"]
            expected = make().cells
            with self.subTest(notebook=rel):
                self.assertEqual([(c.cell_type, c.source) for c in expected],
                                 [(c["cell_type"], "".join(c.get("source", []))) for c in actual])

    def test_workbook_is_closed_when_parsing_fails(self):
        import numpy as np
        for notebook, function_name in (("section5/SQL_Pandas.ipynb", "load_export"),
                                        ("section4/timeSeriesEnergyConsumption.ipynb", "load_eurostat_export")):
            cells = json.loads((BOOK / notebook).read_text(encoding="utf-8"))["cells"]
            source = next("".join(c.get("source", [])) for c in cells if f"def {function_name}(" in "".join(c.get("source", [])))
            function = source[source.index(f"def {function_name}("):source.index("\ngeneration = pd.concat")]
            state = {"pd": pd, "np": np, "Path": Path}
            exec(function, state)
            workbook = mock.MagicMock()
            workbook.__enter__.return_value.sheet_names = ["data"]
            with self.subTest(notebook=notebook), mock.patch.object(pd, "ExcelFile", return_value=workbook), \
                    mock.patch.object(pd, "read_excel", side_effect=ValueError("parse failed")):
                with self.assertRaisesRegex(ValueError, "parse failed"):
                    state[function_name](Path("fixture.xlsx"), "wind")
                workbook.__exit__.assert_called_once()

    def test_unexpected_workbook_values_are_not_silently_marked_missing(self):
        import numpy as np
        path = BOOK / "data/section4/euroStat/nrg_cb_pem_page_spreadsheet_wind.xlsx"
        real_read = pd.read_excel
        for notebook, function_name in (("section5/SQL_Pandas.ipynb", "load_export"),
                                        ("section4/timeSeriesEnergyConsumption.ipynb", "load_eurostat_export")):
            cells = json.loads((BOOK / notebook).read_text(encoding="utf-8"))["cells"]
            source = next("".join(c.get("source", [])) for c in cells if f"def {function_name}(" in "".join(c.get("source", [])))
            function = source[source.index(f"def {function_name}("):source.index("\ngeneration = pd.concat")]
            state = {"pd": pd, "np": np, "Path": Path}
            exec(function, state)
            for invalid in ("unexpected text", float("inf")):
                def alter(*args, **kwargs):
                    frame = real_read(*args, **kwargs)
                    frame.loc[frame.iloc[:, 0].eq("Austria"), frame.columns[1]] = invalid
                    return frame
                with self.subTest(notebook=notebook, value=invalid), mock.patch.object(pd, "read_excel", side_effect=alter):
                    with self.assertRaises(ValueError):
                        state[function_name](path, "wind")

    def test_country_coverage_reports_differences_for_each_notebook(self):
        import ast
        fixture = pd.DataFrame({"source": ["wind", "wind", "solar", "solar"],
                                "country": ["Austria", "Belgium", "Austria", "Croatia"]})
        for notebook in ("section5/SQL_Pandas.ipynb", "section4/timeSeriesEnergyConsumption.ipynb"):
            cells = json.loads((BOOK / notebook).read_text(encoding="utf-8"))["cells"]
            source = next("".join(c.get("source", [])) for c in cells if "def country_coverage(" in "".join(c.get("source", [])))
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "country_coverage")
            state = {"pd": pd}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), notebook, "exec"), state)
            result = state["country_coverage"](fixture)
            with self.subTest(notebook=notebook):
                self.assertEqual(result.loc["wind", "country_count"], 2)
                self.assertEqual(result.loc["wind", "absent_from_source"], "Croatia")
                self.assertEqual(result.loc["solar", "absent_from_source"], "Belgium")

    def test_blank_country_row_does_not_truncate_or_break_flags(self):
        import numpy as np
        import warnings
        path = BOOK / "data/section4/euroStat/nrg_cb_pem_page_spreadsheet_wind.xlsx"
        real_read = pd.read_excel
        def insert_blank(*args, **kwargs):
            frame = real_read(*args, **kwargs)
            blank = pd.DataFrame([[np.nan] * len(frame.columns)], columns=frame.columns)
            return pd.concat([frame.iloc[:7], blank, frame.iloc[7:]], ignore_index=True)
        for notebook, function_name in (("section5/SQL_Pandas.ipynb", "load_export"),
                                        ("section4/timeSeriesEnergyConsumption.ipynb", "load_eurostat_export")):
            cells = json.loads((BOOK / notebook).read_text(encoding="utf-8"))["cells"]
            source = next("".join(c.get("source", [])) for c in cells if f"def {function_name}(" in "".join(c.get("source", [])))
            function = source[source.index(f"def {function_name}("):source.index("\ngeneration = pd.concat")]
            state = {"pd": pd, "np": np, "Path": Path}
            exec(function, state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                baseline = state[function_name](path, "wind")
                with mock.patch.object(pd, "read_excel", side_effect=insert_blank):
                    result = state[function_name](path, "wind")
            pd.testing.assert_frame_equal(baseline.reset_index(drop=True), result.reset_index(drop=True))
            self.assertIn("Austria", set(result.country))

class GeothermalCsvComparisonTests(unittest.TestCase):
    def test_notebook_checks_single_pair_power_and_revenue(self):
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "def compare_single_pair_csv" in c.source)
        state = {"np": np, "pd": pd}
        exec(cell.source, state)
        row = dict(configured_depth_count=1, total_flow_kg_s=50, production_C=160, injection_C=70,
            conversion_efficiency=.12, auxiliary_fraction=.1, capacity_MW=5, price_USD_per_MWh=75,
            net_power_MW=2.03148, energy_MWh=43.879968, revenue_USD=43.879968*75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            differences = state["compare_single_pair_csv"](path)
            self.assertLess(differences.max(), 1e-10)
            row["net_power_MW"] = 3
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertGreater(state["compare_single_pair_csv"](path)["max_power_difference_MW"], .9)
            row["configured_depth_count"] = 2
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "one configured depth"):
                state["compare_single_pair_csv"](path)


    def test_notebook_checks_multidepth_streams_cap_and_balance(self):
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "def compare_multidepth_csv" in c.source)
        state = {"np": np, "pd": pd}
        exec(cell.source, state)
        row = dict(producer_streams_json=json.dumps([{"weight": 50, "temperature": 90}, {"weight": 50, "temperature": 200}]),
            injection_C=100, thermal_MW=20.9, conversion_efficiency=.12, auxiliary_fraction=.1,
            capacity_MW=2, net_power_MW=2, revenue_USD=1000, cumulative_investment_USD=8000000,
            project_balance_USD=-7999000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertLess(state["compare_multidepth_csv"](path).max(), 1e-10)
            # Two active layers share one 5 MW plant cap, rather than one cap per layer.
            both_hot = dict(row, producer_streams_json=json.dumps([
                {"weight": 100, "temperature": 160}, {"weight": 100, "temperature": 200}]),
                injection_C=70, thermal_MW=91.96, capacity_MW=5, net_power_MW=5)
            pd.DataFrame([both_hot]).to_csv(path, index=False)
            self.assertLess(state["compare_multidepth_csv"](path).max(), 1e-10)
            both_hot["net_power_MW"] = 4.063104 + 5  # incorrect separate layer caps
            pd.DataFrame([both_hot]).to_csv(path, index=False)
            self.assertAlmostEqual(state["compare_multidepth_csv"](path)["max_power_difference_MW"], 4.063104)
            row["project_balance_USD"] += 100
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertEqual(state["compare_multidepth_csv"](path)["max_balance_difference_USD"], 100)
            row["producer_streams_json"] = json.dumps([{"weight": -1, "temperature": 200}])
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "nonnegative flow"):
                state["compare_multidepth_csv"](path)


    def test_export_comparisons_reject_impossible_inputs(self):
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        state = {"np": np, "pd": pd}
        for cell in notebook.cells:
            if cell.cell_type == "code" and ("def compare_single_pair_csv" in cell.source or "def compare_multidepth_csv" in cell.source):
                exec(cell.source, state)
        single = dict(configured_depth_count=1, total_flow_kg_s=50, production_C=160, injection_C=70,
                      conversion_efficiency=.12, auxiliary_fraction=.1, capacity_MW=5,
                      price_USD_per_MWh=75, net_power_MW=2.03148, energy_MWh=1, revenue_USD=75)
        multi = dict(producer_streams_json=json.dumps([{"weight": 50, "temperature": 160}]),
                     injection_C=70, thermal_MW=18.81, conversion_efficiency=.12,
                     auxiliary_fraction=.1, capacity_MW=5, net_power_MW=2.03148,
                     revenue_USD=75, cumulative_investment_USD=0, project_balance_USD=75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.csv"
            for name, row in (("compare_single_pair_csv", single), ("compare_multidepth_csv", multi)):
                for column, value in (("conversion_efficiency", 1.5), ("auxiliary_fraction", -0.1),
                                      ("capacity_MW", -1), ("injection_C", -273.15)):
                    with self.subTest(function=name, column=column):
                        pd.DataFrame([dict(row, **{column: value})]).to_csv(path, index=False)
                        with self.assertRaisesRegex(ValueError, "physical ranges"):
                            state[name](path)
            for column, value in (("total_flow_kg_s", -1), ("energy_MWh", -1),
                                  ("price_USD_per_MWh", -1), ("production_C", -274)):
                with self.subTest(column=column):
                    pd.DataFrame([dict(single, **{column: value})]).to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "physical ranges"):
                        state["compare_single_pair_csv"](path)
            invalid_stream = dict(multi, producer_streams_json=json.dumps([{"weight": 50, "temperature": -274}]))
            pd.DataFrame([invalid_stream]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "absolute zero"):
                state["compare_multidepth_csv"](path)


class GeothermalNotebookInputTests(unittest.TestCase):
    def test_invalid_physical_inputs_are_rejected_before_simulation(self):
        import ast
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "def simulate(" in c.source)
        function = next(n for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef) and n.name == "simulate")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "notebook", "exec"), state)
        for kwargs in ({"availability": -.5}, {"availability": 1.5}, {"efficiency": 1.5},
                       {"auxiliary": -.1}, {"heat_capacity": 0}, {"heat_capacity": -1},
                       {"heat_capacity": np.nan}, {"days_step": 0},
                       *({name: value} for name in ("availability", "efficiency", "auxiliary", "heat_capacity")
                         for value in (np.nan, np.inf, -np.inf))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                state["simulate"](**kwargs)

    def test_valid_availability_and_conversion_boundaries_remain_usable(self):
        import ast
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "def simulate(" in c.source)
        functions = [n for n in ast.parse(cell.source).body
                     if isinstance(n, ast.FunctionDef) and n.name in {"neighbors", "simulate"}]
        state = {"np": np, "pd": pd, "N": 10, "DX": 200.0, "H": 50.0, "CP": 4180.0}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "notebook", "exec"), state)
        simulate = state["simulate"]
        stopped, grid = simulate(years=1/365, days_step=1, availability=0)
        self.assertEqual(float(stopped.energy_MWh.iloc[-1]), 0)
        np.testing.assert_allclose(grid, 160.)
        zero_conversion, _ = simulate(years=1/365, days_step=1, efficiency=0, availability=1)
        self.assertEqual(float(zero_conversion.energy_MWh.iloc[-1]), 0)
        full_conversion, _ = simulate(years=1/365, days_step=1, efficiency=1, auxiliary=0,
                                      availability=1, capacity=100)
        expected_mwh = 50 * 4180 * (160-70) / 1e6 * 24
        self.assertAlmostEqual(float(full_conversion.energy_MWh.iloc[-1]), expected_mwh)

    def test_extremely_small_steps_do_not_hang_the_notebook(self):
        import ast
        import nbformat
        import numpy as np
        notebook = nbformat.read(BOOK / "section8/geothermal-field-designer.ipynb", as_version=4)
        cell = next(c for c in notebook.cells if c.cell_type == "code" and "def simulate(" in c.source)
        functions = [node for node in ast.parse(cell.source).body
                     if isinstance(node, ast.FunctionDef) and node.name in {"neighbors", "simulate"}]
        state = {"np": np, "pd": pd, "N": 10, "DX": 200.0, "H": 50.0, "CP": 4180.0}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "notebook", "exec"), state)
        for kwargs in ({"days_step": 1e-12}, {"heat_capacity": 1e-100}, {"years": 1000}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "10,000"):
                state["simulate"](**kwargs)
        result, grid = state["simulate"](years=1/365, days_step=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(grid.shape, (10, 10))


class TheoryLabeledInputTests(unittest.TestCase):
    def test_models_reject_reordered_labeled_inputs_in_lesson_and_notebook(self):
        import ast
        import numpy as np
        cases = [
            ("solarEnergy", {"cell_temperature_c", "pv_ac_power_mw"},
             lambda state, a, b: state["pv_ac_power_mw"](a, b, 10, 8), [500, 1000], [20, 25]),
            ("hydroelectricEnergy", {"hydro_power_mw"},
             lambda state, a, b: state["hydro_power_mw"](a, b), [1, 2], [20, 50]),
            ("geothermalEnergy", {"geothermal_net_power_mw"},
             lambda state, a, b: state["geothermal_net_power_mw"](a, b, 70, .12), [10, 20], [100, 150]),
        ]
        for name, names, call, first, second in cases:
            lesson = (BOOK / f"section6/{name}.md").read_text(encoding="utf-8")
            source = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", lesson, re.S)
                          if f"def {next(iter(names - {'cell_temperature_c'}))}(" in c)
            notebook = json.loads((BOOK / f"section6/{name}.ipynb").read_text(encoding="utf-8"))
            companion = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
                             and all(f"def {function}(" in "".join(c["source"]) for function in names))
            for label, cell in (("lesson", source), ("notebook", companion)):
                functions = [n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name in names]
                self.assertEqual(len(functions), len(names))
                state = {"np": np, "RHO_WATER": 1000, "G": 9.81}
                exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), label, "exec"), state)
                labels = pd.Index(["t0", "t1"])
                with self.subTest(model=name, delivery=label):
                    expected = call(state, first, second)
                    actual = call(state, pd.Series(first, index=labels), pd.Series(second, index=labels))
                    np.testing.assert_allclose(actual, expected)
                    with self.assertRaisesRegex(ValueError, "matching indexes"):
                        call(state, pd.Series(first, index=labels), pd.Series(second, index=labels[::-1]))
                    for bad_labels in (pd.Index(["t0", "t0"]), pd.Index(["t0", None]),
                                       pd.MultiIndex.from_tuples([("a", 1), ("b", None)])):
                        with self.assertRaisesRegex(ValueError, "unique, nonmissing"):
                            call(state, pd.Series(first, index=bad_labels), pd.Series(second, index=bad_labels))
                    multi = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "x"])
                    call(state, pd.Series(first, index=multi), pd.Series(second, index=multi))
                    if name == "hydroelectricEnergy":
                        with self.assertRaisesRegex(ValueError, "matching indexes"):
                            state["hydro_power_mw"](pd.Series(first, index=labels), pd.Series(second, index=labels),
                                                    head_loss_m=pd.Series([1, 2], index=labels[::-1]))
                    if name == "geothermalEnergy":
                        with self.assertRaisesRegex(ValueError, "matching indexes"):
                            state["geothermal_net_power_mw"](pd.Series(first, index=labels), pd.Series(second, index=labels),
                                                               pd.Series([60, 65], index=labels[::-1]), .12)


class GeothermalTheoryBoundaryTests(unittest.TestCase):
    def test_electrical_pump_load_is_not_counted_twice(self):
        lesson = (BOOK / "section6/geothermalEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for delivery, text in (("lesson", lesson), ("notebook", prose)):
            with self.subTest(delivery=delivery):
                self.assertIn("efficiency must include motor and drive as well as pump losses", text)
                self.assertIn("$P_{aux}$ must exclude that same pumping load", text)
                self.assertIn("If a stated parasitic fraction already includes pumping, do not subtract pumping again", text)

    def test_absolute_temperature_check_is_not_a_fluid_property_model(self):
        lesson = (BOOK / "section6/geothermalEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        companion = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for text in (lesson, companion):
            self.assertIn("does not verify that water remains single-phase", text)
            self.assertIn("check pressure, temperature, composition and fluid properties", text)

    def test_assessed_impossible_temperatures_are_rejected_in_lesson_and_notebook(self):
        import ast
        import numpy as np
        text = (BOOK / "section6/geothermalEnergy.md").read_text(encoding="utf-8")
        lesson_cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                           if "def geothermal_net_power_mw(" in c)
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        notebook_cell = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and "def geothermal_net_power_mw(" in "".join(c["source"]))
        for label, cell in (("lesson", lesson_cell), ("notebook", notebook_cell)):
            functions = [n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef)
                         and n.name in {"temperature_at_depth_c", "geothermal_net_power_mw"}]
            self.assertEqual(len(functions), 2)
            state = {"np": np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), label, "exec"), state)
            with self.subTest(delivery=label):
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["temperature_at_depth_c"](1, surface_c=-400)
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["geothermal_net_power_mw"](50, -300, -350, .12)
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["geothermal_net_power_mw"]([50, 50], [160, -300], [70, -350], .12)
                self.assertAlmostEqual(state["temperature_at_depth_c"](1), 45.)
                self.assertAlmostEqual(state["geothermal_net_power_mw"](50, 160, 70, .12), 2.03148)


class GeothermalQualityTests(unittest.TestCase):
    def test_model_year_convention_is_visible_in_workbook_quiz_and_notebook(self):
        workbook = (BOOK / "section7/workbook/geothermal.md").read_text(encoding="utf-8")
        quiz = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section7/workbook/geothermal.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for delivery, text in (("workbook", workbook), ("quiz", quiz), ("notebook", prose)):
            with self.subTest(delivery=delivery):
                self.assertIn("Unless calendar dates are specified, use a standardized model year of 8,760 hours", text)

    def test_assessed_geothermal_quiz_answers_display_results(self):
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        states = {}
        for number, label in [(3, "12,500 kW:"), (4, "Absolute temperature"), (8, "Darcy flux"),
                              (9, "Enthalpy drop"), (10, "Volume flow"), (11, "Electrical input and source heat")]:
            section = re.search(r":::{{admonition}} Quiz {}\b(.*?)(?=:::{{admonition}} Quiz \d+|\Z)".format(number), text, re.S)
            self.assertIsNotNone(section, number)
            cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section.group(1), re.S)
            self.assertIsNotNone(cell, number)
            output, state = io.StringIO(), {}
            with contextlib.redirect_stdout(output):
                exec(cell.group(1), state)
            self.assertIn(label, output.getvalue())
            states[number] = state
        self.assertEqual(states[3]["kw_to_mw"](12500), 12.5)
        kelvin = states[4]["to_kelvin"]
        self.assertAlmostEqual(kelvin(85), 358.15)
        self.assertEqual(kelvin(-273.15), 0)
        for invalid in (-273.16, float("nan"), float("inf")):
            with self.subTest(temperature=invalid), self.assertRaises(ValueError):
                kelvin(invalid)
        self.assertAlmostEqual(states[8]["flux"], 7.049452142857e-8)
        self.assertEqual(states[9]["recovered_heat"](4.1, 155, 85, 45), (287., 12.915))
        self.assertAlmostEqual(states[10]["q"], 70 / 950)
        self.assertAlmostEqual(states[10]["power"], (70 / 950) * 1.8 / .75)
        self.assertEqual([states[11]["heat_pump"](h, 3) for h in [3, 6, 9]], [(1, 2), (2, 4), (3, 6)])

    def test_quality_flags_separate_missing_and_infinity(self):
        import numpy as np
        text = (BOOK / "section7/workbook/geothermal.md").read_text(encoding="utf-8")
        self.assertIn("Assume constant specific heat 4,180 J/(kg K)", text)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "energy_label" in c and "observed MWh thermal" in c)
        old = "flow=np.array([40,-2,50,np.nan]); production=np.array([150,160,60,170])"
        replacement = "flow=np.array([np.nan,np.inf,-np.inf,40]); production=np.array([150,150,150,150])"
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell.replace(old, replacement), state)
        self.assertEqual(state["flags"].tolist(), ["missing", "non-finite", "non-finite", "valid"])
        self.assertEqual(state["valid"].sum(), 1)
        self.assertAlmostEqual(state["observed_energy"], 13.376)
        self.assertIn("not a complete-period estimate when coverage is incomplete", text)

    def test_quiz_rejects_impossible_reinjection_temperature(self):
        import numpy as np
        import pandas as pd
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def invalid_wells(" in c)
        function = "def invalid_wells(" + cell.split("def invalid_wells(", 1)[1].split("\nproblems =", 1)[0]
        state = {"np": np}
        exec(function, state)
        frame = pd.DataFrame({"mass_flow_kgs": [1., 1.], "production_c": [-273.14, 80.]})
        for invalid in (-274., float("nan"), float("inf")):
            with self.subTest(reinjection=invalid), self.assertRaises(ValueError):
                state["invalid_wells"](frame, invalid)
        self.assertTrue(state["invalid_wells"](frame, -273.15).empty)
        self.assertEqual(len(state["invalid_wells"](frame, 70.)), 1)
        stopped = pd.DataFrame({"mass_flow_kgs": [0., 0., 1.], "production_c": [60., -274., 60.]})
        self.assertEqual(state["invalid_wells"](stopped, 70.).index.tolist(), [1, 2])
        equal_temperatures = pd.DataFrame({"mass_flow_kgs": [0., 1.], "production_c": [70., 70.]})
        self.assertEqual(state["invalid_wells"](equal_temperatures, 70.).index.tolist(), [1])
        nullable = pd.DataFrame({"mass_flow_kgs": pd.Series([pd.NA, 1., 1.], dtype="Float64"),
                                 "production_c": pd.Series([100., pd.NA, 100.], dtype="Float64")})
        self.assertEqual(state["invalid_wells"](nullable, 70.).index.tolist(), [0, 1])

    def test_stopped_well_below_absolute_zero_is_invalid(self):
        text = (BOOK / "section7/workbook/geothermal.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "energy_label" in c and "observed MWh thermal" in c)
        original = "flow=np.array([40,-2,50,np.nan]); production=np.array([150,160,60,170])"
        self.assertIn(original, cell)
        replacement = "flow=np.array([0,0,40,40]); production=np.array([-274,-273.15,-274,150])"
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell.replace(original, replacement), state)
        self.assertEqual(state["flags"].tolist(), ["below absolute zero", "valid", "below absolute zero", "valid"])
        self.assertEqual(int(state["valid"].sum()), 2)
        self.assertAlmostEqual(state["observed_energy"], 13.376)

    def test_stopped_well_is_valid_and_all_missing_is_unavailable(self):
        import numpy as np
        text = (BOOK / "section7/workbook/geothermal.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "energy_label" in c and "observed MWh thermal" in c)
        original = "flow=np.array([40,-2,50,np.nan]); production=np.array([150,160,60,170])"
        self.assertIn(original, cell)
        for replacement, coverage, label in (
            ("flow=np.array([0,40,np.nan,np.nan]); production=np.array([150,160,60,170])", .5, "15.048"),
            ("flow=np.array([np.nan,np.nan,np.nan,np.nan]); production=np.array([150,160,60,170])", 0., "unavailable"),
        ):
            output = io.StringIO(); state = {}
            with self.subTest(label=label), contextlib.redirect_stdout(output):
                exec(cell.replace(original, replacement), state)
            self.assertEqual(state["valid"].mean(), coverage)
            self.assertIn("observed MWh thermal: " + label, output.getvalue())
            if coverage:
                self.assertEqual(state["flags"][0], "valid")
                self.assertTrue(np.isclose(state["observed_energy"], 15.048))
            else:
                self.assertTrue(np.isnan(state["observed_energy"]))


class LessonBoundaryTests(unittest.TestCase):
    def test_intro_environmental_sources_identify_technology_scope(self):
        lesson = (BOOK / "section0/introRenewableEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section0/introRenewableEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for text in (lesson, prose):
            self.assertIn("For hydropower, see", text)
            self.assertIn("for photovoltaic life-cycle stages", text)
            self.assertIn("IEA_Task12_LCA_Guidelines.pdf", text)
            self.assertIn("effects depending on technology and location", text)

    def test_intro_model_inputs_are_not_double_counted_and_quiz_is_scalar(self):
        lesson = (BOOK / "section0/introRenewableEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section0/introRenewableEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for text in (lesson, prose):
            self.assertIn("calculation and variation of losses from gross head", text)
            self.assertIn("Calculation of plane-of-array irradiance from orientation, weather and shading", text)
            self.assertIn("do not apply that irradiance reduction again", text)
        self.assertAlmostEqual(1000 * 9.81 * 10 * (100 - 10) / 1e6, 8.829)
        self.assertAlmostEqual(1000 * 9.81 * 10 * (100 - 10 - 10) / 1e6, 7.848)
        quiz = (BOOK / "section7/renewableEnergyquizzes/generalPython.md").read_text(encoding="utf-8")
        q5 = quiz.split("Quiz 5 — Condition", 1)[1].split("Quiz 6 —", 1)[0]
        self.assertIn("For scalar numeric values", q5)
        self.assertIn("0 <= power <= rating", q5)

    def test_energy_summaries_name_interval_power_assumption(self):
        for topic in ("solar", "hydroelectric"):
            path = BOOK / "section6" / (topic + "Energy")
            markdown = path.with_suffix(".md").read_text(encoding="utf-8")
            notebook = json.loads(path.with_suffix(".ipynb").read_text(encoding="utf-8"))
            notebook_text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
            for delivery, text in (("lesson", markdown), ("notebook", notebook_text)):
                with self.subTest(topic=topic, delivery=delivery):
                    if topic == "solar":
                        self.assertIn("For interval-average AC powers (or powers constant within each interval)", text)
                        self.assertIn("For AC powers averaged over their respective intervals (or constant within them)", text)
                        self.assertIn("Instantaneous samples need an integration rule and an interpolation assumption", text)
                    else:
                        self.assertIn("If $P_i$ is the average over interval", text)
        project = (BOOK / "section8/projectIntro.md").read_text(encoding="utf-8")
        self.assertIn("treating generation and demand as constant within the hour", project)
        self.assertIn("hourly averages alone cannot determine exact simultaneous supply and demand", project)

    def test_hourly_means_do_not_determine_simultaneous_service(self):
        lesson = (BOOK / "section0/introRenewableEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section0/introRenewableEnergy.ipynb").read_text(encoding="utf-8"))
        companion = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for text in (lesson, companion):
            self.assertIn("same 1 MW hourly mean but serves only 0.5 MWh without storage", text)
        constant = sum(min(supply, 1) * .5 for supply in [1, 1])
        variable = sum(min(supply, 1) * .5 for supply in [0, 2])
        self.assertEqual((constant, variable), (1, .5))

    def test_geothermal_heat_rate_has_direct_sources_and_boundary(self):
        lesson = (BOOK / "section6/geothermalEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for content in (lesson, notebook_text):
            self.assertIn("egs_chapter_3.pdf#page=6", content)
            self.assertIn("jason.final.pdf#page=140", content)
            self.assertIn("negligible heat leakage between the selected states", content)
            self.assertIn("single-phase, approximately constant-heat-capacity boundary", content)

    def test_wind_displacement_and_heat_pump_sources_are_specific(self):
        wind = (BOOK / "section6/windEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/windEnergy.ipynb").read_text(encoding="utf-8"))
        wind_notebook = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for content in (wind, wind_notebook):
            self.assertIn("NCAR Earth Observing Laboratory, neutral wind profile with displacement height", content)
            self.assertIn("for the zero-displacement forms", content)
        quiz = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        question = quiz.split(":::{admonition} Quiz 11 —", 1)[1].split(":::{admonition} Solution", 1)[0]
        self.assertIn("4-3-refrigerators-and-heat-pumps", question)
        self.assertIn("heatpump.pdf", question)
        references = (BOOK / "references.md").read_text(encoding="utf-8")
        self.assertIn("not a universal threshold", references)
        self.assertAlmostEqual((20 + 273.15) / (20 - (-20)), 7.32875)

    def test_heat_pump_and_heat_engine_use_distinct_direct_sources(self):
        lesson = (BOOK / "section6/geothermalEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/geothermalEnergy.ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for content in (lesson, notebook_text):
            with self.subTest(artifact="lesson" if content is lesson else "notebook"):
                self.assertIn("Heat-pump accounting: [OpenStax, University Physics 2, §4.3]", content)
                self.assertIn("4-3-refrigerators-and-heat-pumps", content)
                self.assertIn("4-5-the-carnot-cycle", content)
                self.assertNotIn("Heat-pump accounting: Reversible heat-engine limit", content)

    def test_pumped_storage_rejects_nonfinite_and_obeys_cycle(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def storage_cycle(" in c)
        function = cell.split("\neta = np.linspace", 1)[0]
        state = {"np": np}
        exec(function, state)
        cycle = state["storage_cycle"]
        self.assertEqual(cycle(2.5, .8), (2., .5))
        for energy, efficiency in ((float("nan"), .8), (2.5, float("nan"))):
            with self.assertRaises(ValueError):
                cycle(energy, efficiency)

    def test_sql_left_join_retains_unmatched_energy(self):
        text = (BOOK / "section5/SQL.md").read_text(encoding="utf-8")
        section = text.split("### LEFT JOIN", 1)[1].split("### ", 1)[0]
        query = re.search(r"```sql\n(.*?)\n```", section, re.S).group(1)
        c = sqlite3.connect(":memory:")
        try:
            c.executescript("CREATE TABLE energy_data(country TEXT, energy_source TEXT, energy_generated REAL); CREATE TABLE regions(country TEXT, region_name TEXT); INSERT INTO energy_data VALUES ('A','Wind',1),('B','Solar',2); INSERT INTO regions VALUES ('A','X'),('C','Y');")
            rows = c.execute(query).fetchall()
            self.assertEqual(rows, [("A", "Wind", 1., "X"), ("B", "Solar", 2., None)])
        finally:
            c.close()

class HydroCoverageAlignmentTests(unittest.TestCase):
    def test_bound_and_duration_are_validated_for_unknown_flows(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/workbook/hydroelectric.md").read_text(encoding="utf-8")
        self.assertIn("every unknown true flow, including invalid readings", text)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def audit(flow, duration, limit):" in c)
        fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "audit")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "workbook", "exec"), state)
        audit = state["audit"]
        np.testing.assert_allclose(audit([3, -1, np.nan, 5], [1, 2, 1, .5], 6), [.5, 1/3, 19800, 84600])
        for limit in (-2, np.inf, np.nan):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                audit([3, np.nan], [1, 1], limit)
        for durations in ([1, -1], [1, 0], [1, np.inf], [1, np.nan], [1]):
            with self.subTest(durations=durations), self.assertRaises(ValueError):
                audit([3, np.nan], durations, 6)
        with self.assertRaises(ValueError):
            audit([], [], 6)

    def test_labelled_intervals_must_match_in_order(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/workbook/hydroelectric.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def audit(flow, duration, limit):" in c)
        fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "audit")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "workbook", "exec"), state)
        audit = state["audit"]
        flow = pd.Series([3., np.nan], index=["A", "B"])
        duration = pd.Series([1., 2.], index=["A", "B"])
        self.assertEqual(audit(flow, duration, 6), (.5, 1/3, 10800., 54000.))
        for invalid in [duration.iloc[::-1], pd.Series([1., 2.], index=["A", "A"]), [1., 2.]]:
            with self.assertRaisesRegex(ValueError, "same unique index"):
                audit(flow, invalid, 6)
        self.assertEqual(audit([3., np.nan], [1., 2.], 6), (.5, 1/3, 10800., 54000.))
        for index in (pd.Index(["A", None]), pd.MultiIndex.from_tuples([("A", 1), ("B", None)])):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "must not be missing"):
                audit(pd.Series([3., np.nan], index=index), pd.Series([1., 2.], index=index), 6)
        valid_index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
        self.assertEqual(audit(pd.Series([3., np.nan], index=valid_index),
                               pd.Series([1., 2.], index=valid_index), 6), (.5, 1/3, 10800., 54000.))


class AssessedBoundaryChecks(unittest.TestCase):
    def test_solar_grid_and_geothermal_pump_penalty_state_models(self):
        import numpy as np
        solar = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        section = solar.split("### Exercise 17 —", 1)[1].split("### Exercise 18 —", 1)[0]
        self.assertIn("Tcell = Tair + G(NOCT − 20)/800", section)
        self.assertIn("referenced to 25 °C", section)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S) if "cell=ambient" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        np.testing.assert_allclose(state["ac"], [[2.0079, 5.7327, 8.], [1.8915, 5.3835, 8.],
                                                 [1.7751, 5.0343, 7.9055]], atol=0.0001)
        geo = (BOOK / "section7/workbook/geothermal.md").read_text(encoding="utf-8")
        section = geo.split("### Exercise 18 —", 1)[1].split("### Exercise 19 —", 1)[0]
        self.assertIn("m be the numerical mass flow in kg/s", section)
        self.assertIn("coefficient expressed in MW", section)
        self.assertIn("verify the power subtraction", section)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S) if "pump=0.00004" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        np.testing.assert_allclose(state["net"] + state["pump"], state["gross"])
        self.assertEqual(state["flow"][state["best"]], 100)

    def test_hourly_workbook_energy_and_conflicting_duplicates(self):
        text = (BOOK / "section7/workbook/general.md").read_text(encoding="utf-8")
        for number in ("08", "09"):
            section = text.split("### Exercise " + number + " —", 1)[1].split("### Exercise ", 1)[0]
            self.assertIn("one-hour interval", section)
            cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)
                        if "# Step 1:" in c)
            output, state = io.StringIO(), {}
            with contextlib.redirect_stdout(output):
                exec(compile(cell, "workbook", "exec", optimize=1), state)
            self.assertIn("6", output.getvalue())
            if number == "08":
                self.assertEqual(state["coverage"], .75)
                self.assertEqual(state["observed_mwh"], 6)
            else:
                self.assertEqual(len(state["clean"]), 3)
                conflicting = cell.replace("'mw': [1,2,2,3]", "'mw': [1,2,9,3]")
                self.assertNotEqual(conflicting, cell)
                with self.assertRaisesRegex(ValueError, "Conflicting measurements"):
                    exec(compile(conflicting, "workbook", "exec", optimize=1), {})

    def test_source_mix_prints_each_interval_and_total(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        text = (BOOK / "section7/workbook/general.md").read_text(encoding="utf-8")
        section = text.split("### Exercise 15 —", 1)[1].split("### Exercise 16 —", 1)[0]
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)
                    if "sources =" in c)
        output, state = io.StringIO(), {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        for key, expected in (("bottom", [42, 58]), ("served", [42, 45]),
                              ("surplus", [0, 13]), ("shortfall", [8, 0])):
            np.testing.assert_array_equal(state[key], expected)
        for label in ("Hour 1 MWh", "Hour 2 MWh", "Total MWh"):
            self.assertIn(label, output.getvalue())

    def test_portfolio_energy_coverage_is_not_fully_served_hour_fraction(self):
        import numpy as np
        text = (BOOK / "section7/workbook/general.md").read_text(encoding="utf-8")
        section = text.split("### Exercise 19 —", 1)[1].split("### Exercise 20 —", 1)[0]
        self.assertIn("served energy divided by demand energy", section)
        self.assertIn("not the fraction of hours with fully met demand", section)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)
                    if "best = next(row" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        units, coverage, unmet, curtailed = state["best"]
        self.assertEqual((units, coverage, unmet, curtailed), (1, .875, 1, 1))
        fully_met = np.mean(1 + units * state["wind"] >= state["demand"])
        self.assertEqual(fully_met, .75)

    def test_wind_sector_count_and_energy_shares(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "count_share=100*count/count.sum()" in c)
        state = {}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        np.testing.assert_allclose(state["count_share"], [50, 25, 25, 0])
        np.testing.assert_allclose(state["energy_share"], [20, 48, 32, 0])
        self.assertEqual(state["energy"].sum(), 12.5)
        self.assertIn("energy_pct", output.getvalue())

    def test_sql_half_open_utc_year_query(self):
        text = (BOOK / "section5/SQL.md").read_text(encoding="utf-8")
        query = next(c for c in re.findall(r"```sql\n(.*?)\n```", text, re.S)
                     if "date >= '2024-01-01'" in c)
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE energy_data(country TEXT, energy_source TEXT, energy_generated REAL, date TEXT)")
            db.executemany("INSERT INTO energy_data VALUES (?, ?, ?, ?)", [
                ("first", "Wind", 1, "2024-01-01"),
                ("last", "Solar", 2, "2024-12-31 12:00:00"),
                ("next", "Wind", 3, "2025-01-01 00:00:00"),
                ("other", "Hydro", 4, "2024-06-01")])
            self.assertEqual([r[0] for r in db.execute(query)], ["first", "last"])
        self.assertIn("mixed offsets must be normalized", text)


class WindDataTests(unittest.TestCase):
    def test_direction_quiz_displays_from_and_velocity_to_bearings(self):
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        question = re.search(r":::\{admonition\} Quiz 19\b(.*?)(?=:::\{admonition\} Quiz \d+|\Z)", text, re.S).group(1)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", question, re.S).group(1)
        state, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertAlmostEqual(state["unweighted"], 26.917511165965028)
        self.assertAlmostEqual(state["weighted"], 68.49937377810684)
        self.assertAlmostEqual(state["velocity_to"], 248.49937377810684)
        for label in ("Circular from-bearing", "mean velocity from", "velocity towards"):
            self.assertIn(label, output.getvalue())
        lesson = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        workbook = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        self.assertIn("Excel uses x before y, unlike NumPy", lesson)
        self.assertIn("Excel takes x before y; NumPy takes y before x", workbook)

    def test_energy_reconciliation_does_not_invent_all_missing_measurements(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "actual_label" in c and "modelled gap energy" in c)
        prefix = cell.split("\nassert actual==", 1)[0]
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertEqual((state["actual"], state["gap_estimate"]), (6, 5))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(prefix.replace("[0,np.nan,2,np.nan,4]", "[np.nan,np.nan,np.nan,np.nan,np.nan]"), state)
        self.assertTrue(np.isnan(state["actual"]) and np.isnan(state["bias"]))
        self.assertEqual(state["gap_estimate"], 12)
        self.assertIn("Measured energy: unavailable", output.getvalue())
        self.assertIn("paired model bias: unavailable", output.getvalue())
        for invalid in (prefix.replace("[0,np.nan,2,np.nan,4]", "[0,np.inf,2,np.nan,4]"),
                        prefix.replace("[0,1,3,4,4]", "[0,1,np.nan,4,4]")):
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                exec(invalid, {})

    def test_wind_meter_join_validates_keys_and_reports_usable_pairs(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def prepare_hourly(" in c and "Matched mean speed" in c)
        prefix = cell.split("\nassert len(joined)", 1)[0]
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prefix, state)
        times = pd.date_range("2025-02-01", periods=2, freq="h", tz="UTC")
        original = pd.DataFrame({"time": times, "speed": ["2", "6"]})
        converted = state["prepare_hourly"](original, "speed")
        self.assertEqual(converted.speed.tolist(), [2., 6.])
        self.assertEqual(original.speed.tolist(), ["2", "6"])
        for bad in (pd.DatetimeIndex([times[0], pd.NaT]), pd.DatetimeIndex([times[0], times[0]]), times.tz_localize(None), times + pd.Timedelta(minutes=30)):
            with self.subTest(index=bad), self.assertRaises(ValueError):
                state["prepare_hourly"](pd.DataFrame({"time": bad, "speed": [1., 2.]}), "speed")
        for values in ([1., -1.], [1., np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                state["prepare_hourly"](pd.DataFrame({"time": times, "speed": values}), "speed")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prefix.replace("'speed':[2,6,8]", "'speed':[2,np.nan,8]"), state)
        self.assertEqual((len(state["matched"]), len(state["paired"])), (2, 1))
        self.assertEqual((state["mean_speed"], state["meter_energy"]), (8., 2.))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prefix.replace("'speed':[2,6,8]", "'speed':[2,np.nan,np.nan]"), state)
        self.assertTrue(np.isnan(state["mean_speed"]) and np.isnan(state["meter_energy"]))

    def test_gap_exercise_preserves_clean_data_for_revisiting_energy(self):
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        pattern = r"```\{code-cell\} python\n(.*?)\n```"
        setup = re.findall(pattern, text.split("## Setup:", 1)[1].split(":::{admonition} Quiz 1", 1)[0], re.S)[0]
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(setup, state)
            cells = {}
            for n in (5, 13, 17):
                section = text.split(f":::{'{admonition}'} Quiz {n} —", 1)[1].split(":::{admonition} Quiz ", 1)[0]
                cells[n] = re.findall(pattern, section, re.S)[0]
            exec(cells[5], state)
            exec(cells[13], state)
            before = state["clean"].copy(deep=True)
            energy = state["monthly_energy_mwh"]
            exec(cells[17], state)
            pd.testing.assert_frame_equal(state["clean"], before)
            exec(cells[13], state)
        self.assertEqual(state["monthly_energy_mwh"], energy)

    def test_sector_counts_exclude_missing_direction_from_full_workflow(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        pattern = r"```\{code-cell\} python\n(.*?)\n```"
        setup = re.findall(pattern, text.split("## Setup:", 1)[1].split(":::{admonition} Quiz 1", 1)[0], re.S)[0]
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(setup, state)
        state["wind"].iloc[0, state["wind"].columns.get_loc("direction_deg")] = np.nan
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for number in (5, 7):
                section = text.split(f":::{'{admonition}'} Quiz {number} —", 1)[1].split(":::{admonition} Quiz ", 1)[0]
                exec(re.findall(pattern, section, re.S)[0], state)
        self.assertEqual(state["sector_counts"].sum(), 719)
        self.assertTrue(pd.isna(state["clean"]["sector"].iloc[0]))
        self.assertAlmostEqual(state["sector_share"].sum(), 1.)
        self.assertIn("Finite-direction coverage: 719/720", output.getvalue())

    def test_gap_repair_rejects_irregular_dates_and_preserves_unfillable_gaps(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 5 —", 1)[1].split(":::{admonition} Quiz 6 —", 1)[0]
        cell = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]
        irregular = pd.DataFrame({"speed_ms": [4., np.nan, 10.], "direction_deg": [0., 0., 0.]},
                                 index=pd.to_datetime(["2025-01-01", "2025-01-15", "2025-01-31"]))
        with self.assertRaisesRegex(ValueError, "regular hourly"):
            exec(cell, {"wind": irregular, "pd": pd, "np": np})
        hourly = pd.DataFrame({"speed_ms": [np.nan, 4., np.nan, 8., np.nan, np.nan, np.nan, 10.],
                               "direction_deg": [0.] * 8},
                              index=pd.date_range("2025-01-01", periods=8, freq="h"))
        original = hourly.copy(deep=True)
        state = {"wind": hourly, "pd": pd, "np": np}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        result = state["clean"]["speed_ms"]
        self.assertEqual(result.iloc[2], 6.)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertTrue(result.iloc[4:7].isna().all())
        pd.testing.assert_frame_equal(hourly, original)

    def test_modelled_energy_reports_original_and_repaired_coverage(self):
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        pattern = r"```\{code-cell\} python\n(.*?)\n```"
        setup = re.findall(pattern, text.split("## Setup:", 1)[1].split(":::{admonition} Quiz 1", 1)[0], re.S)[0]
        state, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(setup, state)
            for number in (5, 13):
                section = text.split(f":::{'{admonition}'} Quiz {number} —", 1)[1].split(":::{admonition} Quiz ", 1)[0]
                exec(re.findall(pattern, section, re.S)[0], state)
        self.assertIn("valid modelled intervals 720/720", output.getvalue())
        self.assertIn("Valid original speed intervals: 718/720; repaired intervals used in the model: 2", output.getvalue())

    def test_modelled_energy_distinguishes_all_missing_from_observed_zero(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 13 —", 1)[1].split(":::{admonition} Quiz 14 —", 1)[0]
        code = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]
        for speeds, expected, coverage in (([np.nan, np.nan], "unavailable", "0/2"),
                                           ([0., 25.], "0.0 MWh", "2/2")):
            with self.subTest(speeds=speeds):
                wind = pd.DataFrame({"speed_ms": speeds})
                state = {"wind": wind, "clean": wind.copy(), "pd": pd, "np": np}
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(code, state)
                self.assertIn(f"Modelled energy from synthetic wind: {expected}", output.getvalue())
                self.assertIn(f"valid modelled intervals {coverage}", output.getvalue())

    def test_observation_model_comparison_pairs_valid_hourly_values(self):
        import numpy as np
        text = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        section = text.split("### Compare matched observations and model estimates", 1)[1]
        cell = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertEqual(state["metrics"]["paired_hours"], 2)
        self.assertAlmostEqual(state["metrics"]["bias_ms"], 0)
        self.assertAlmostEqual(state["metrics"]["rmse_ms"], 0.5)
        compare = state["compare_hourly_speeds"]
        index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        observed = pd.Series([1.0, np.nan], index=index)
        modelled = pd.Series([4.0], index=pd.DatetimeIndex([index[1]]))
        pairs, metrics = compare(observed, modelled)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(metrics["paired_hours"], 0)
        self.assertTrue(np.isnan(metrics["bias_ms"]))
        self.assertTrue(np.isnan(metrics["rmse_ms"]))
        # Distinct time-zone labels for the same instants must pair after UTC conversion.
        local_model = pd.Series([1.5, 2.5], index=index.tz_convert("America/New_York"))
        source_index = local_model.index.copy()
        paired, aligned = compare(pd.Series([1., 2.], index=index), local_model)
        self.assertEqual(aligned["paired_hours"], 2)
        self.assertAlmostEqual(aligned["bias_ms"], 0.5)
        self.assertEqual(str(paired.index.tz), "UTC")
        self.assertTrue(local_model.index.equals(source_index))
        for bad_index in (pd.DatetimeIndex([index[0], index[0]]), index.tz_localize(None), index + pd.Timedelta(minutes=10)):
            with self.assertRaises(ValueError):
                compare(pd.Series([1., 2.], index=bad_index), modelled)
        with self.assertRaises(ValueError):
            compare(pd.Series([np.inf, 1.], index=index), modelled)

    def test_observed_wind_hourly_coverage_and_circular_direction(self):
        import numpy as np
        text = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        section = text.split("### Aggregate the observational series by hour", 1)[1]
        cell = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]

        def run(csv):
            code = re.sub(r'fixture = """.*?"""', lambda _: 'fixture = ' + repr(csv), cell, count=1, flags=re.S)
            state = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, state)
            return state["hourly"]

        fixture = ("Timestamp,WS11,WD11\n"
                   "2020-01-01 00:00,3,350\n2020-01-01 00:10,3,10\n"
                   "2020-01-01 00:20,3,0\n2020-01-01 00:30,3,360\n"
                   "2020-01-01 00:40,3,5\n2020-01-01 00:50,3,355\n"
                   "2020-01-01 01:00,4,0\n2020-01-01 01:10,4,180\n"
                   "2020-01-01 01:20,4,\n")
        hourly = run(fixture)
        self.assertAlmostEqual(hourly.mean_direction_deg.iloc[0], 0)
        self.assertTrue(pd.isna(hourly.mean_direction_deg.iloc[1]))
        self.assertEqual(hourly.speed_samples.tolist(), [6, 3])
        self.assertEqual(hourly.direction_samples.tolist(), [6, 2])
        self.assertEqual(hourly.speed_complete.tolist(), [True, False])
        for invalid in (fixture + "2020-01-01 01:20,4,0\n",
                        fixture + "2020-01-01 01:25,4,0\n"):
            with self.assertRaises(ValueError):
                run(invalid)

    def test_xarray_wind_example_handles_assessed_north_crossing(self):
        import numpy as np
        import xarray as xr
        text = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if '"valid_direction_samples"' in c)
        computation = cell.split("valid_speed = wind", 1)[1].split("print(hourly_wind)", 1)[0]
        angles = [358.5, 2., 360., 358.5, 2., 360.]
        wind = xr.Dataset({"M": ("time", [5.] * 6), "Dir": ("time", angles)},
                          coords={"time": pd.date_range("2010-10-10 12:00", periods=6, freq="10min")})
        state = {"wind": wind, "np": np, "xr": xr}
        exec("valid_speed = wind" + computation, state)
        result = state["hourly_wind"]
        self.assertAlmostEqual(float(result.mean_direction_deg.values[0]), 0.1666, places=3)
        self.assertEqual(int(result.valid_direction_samples.values[0]), 6)
        self.assertGreater(abs(float(np.mean(angles)) - float(result.mean_direction_deg.values[0])), 200)

    def test_headerless_wind_reader_parses_dates_and_rejects_wrong_shape(self):
        text = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        section = text.split("### Read a headerless four-column wind file", 1)[1]
        cell = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]
        with contextlib.redirect_stdout(io.StringIO()):
            state = {}
            exec(cell, state)
        self.assertEqual(state["wind_text"]["timestamp"].dt.strftime("%H:%M").tolist(),
                         ["00:00", "00:10", "00:20"])
        self.assertIsNone(state["wind_text"]["timestamp"].dt.tz)
        malformed = cell.replace("20040101 0000 4.6 154", "20040101 0000 4.6 154 0.5")
        with self.assertRaises((ValueError, pd.errors.ParserError)):
            exec(malformed, {})
        duplicate = cell.replace("20040101 0020 5.1 165", "20040101 0010 5.1 165")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            exec(duplicate, {})
        for invalid in ("20040101 0020 -1 165", "20040101 0020 5.1 361", "20040101 0020 inf 165"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "out-of-range"):
                exec(cell.replace("20040101 0020 5.1 165", invalid), {})
        missing = cell.replace("20040101 0020 5.1 165", "20040101 0020 NaN 165")
        missing_state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(missing, missing_state)
        self.assertEqual(len(missing_state["wind_text"]), 3)
        self.assertTrue(pd.isna(missing_state["wind_text"]["speed_ms"].iloc[2]))

    def test_standalone_hourly_circular_direction_example(self):
        import numpy as np
        text = (BOOK / "section5/wind-data.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "valid_samples" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertTrue(np.isclose(state["direction"].iloc[0], 0))
        self.assertTrue(pd.isna(state["direction"].iloc[1]))
        self.assertEqual(state["count"].tolist(), [2, 2])
        self.assertEqual(str(state["samples"]["timestamp"].dt.tz), "UTC")

    def test_documented_prerequisite_chain_runs_energy_rose_without_quiz_eight(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        self.assertIn("run Quizzes 5, 7 and 13 before Quiz 14", text)
        setup_text = text.split("## Setup:", 1)[1].split(":::{admonition} Quiz 1", 1)[0]
        pattern = r"```\{code-cell\} python\n(.*?)\n```"
        setup = re.findall(pattern, setup_text, re.S)[0]
        state = {}; output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(setup, state)
                for number in (5, 7, 13, 14):
                    section = text.split(f":::{'{admonition}'} Quiz {number} —", 1)[1].split(":::{admonition} Quiz ", 1)[0]
                    exec(re.findall(pattern, section, re.S)[0], state)
            self.assertIn("sector_energy_report", state)
            self.assertAlmostEqual(state["energy_by_sector"].loc["SW"], 244.843, places=3)
            self.assertAlmostEqual(state["energy_by_sector"].loc["W"], 200.780, places=3)
            self.assertIn("Leading sector(s) for modelled energy: SW (244.843 MWh)", output.getvalue())
            self.assertEqual(state["ax"].get_ylabel(), "Modelled energy over valid one-hour intervals (MWh)")
            state["fig"].canvas.draw()
            renderer = state["fig"].canvas.get_renderer()
            west = next(t for t in state["ax"].get_xticklabels() if t.get_text() == "W")
            self.assertFalse(state["ax"].yaxis.label.get_window_extent(renderer).overlaps(west.get_window_extent(renderer)))
            frequency = text.split(":::{admonition} Quiz 8 —", 1)[1].split(":::{admonition} Quiz 9 —", 1)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(re.findall(pattern, frequency, re.S)[0], state)
            self.assertEqual(state["ax"].get_ylabel(), "Fraction of valid directions")
            state["fig"].canvas.draw()
            renderer = state["fig"].canvas.get_renderer()
            west = next(t for t in state["ax"].get_xticklabels() if t.get_text() == "W")
            self.assertFalse(state["ax"].yaxis.label.get_window_extent(renderer).overlaps(west.get_window_extent(renderer)))
        finally:
            plt.close("all")

    def test_meter_merge_prompt_supplies_values_and_interval_boundary(self):
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 16 — Validate and merge wind and meter files", 1)[1].split(":::", 1)[0]
        self.assertIn("wind speeds [2,6,8] m/s", prompt)
        self.assertIn("meter powers [1,2,3] MW", prompt)
        self.assertIn("average over its one-hour interval", prompt)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "Matched mean speed (m/s):" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        self.assertIn("Matched mean speed (m/s): 7.0", output.getvalue())
        self.assertIn("matched meter energy (MWh): 3.0", output.getvalue())
        self.assertIn("2/4 union timestamps", output.getvalue())
        half_hour = text.split("Quiz 11 — Find energy in subhourly meter data", 1)[1].split(":::", 1)[0]
        self.assertIn("half-hour interval-mean meter powers", half_hour)

    def test_energy_rose_distinguishes_missing_absent_and_partial_sectors(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "sector_energy_report =" in c)
        names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        clean = pd.DataFrame({"sector": ["N", "N", "E", "E", None],
                              "power_mw": [np.nan, np.nan, 1., np.nan, 2.]})
        state = {"clean": clean, "sector_names": names, "angles": np.arange(8)*np.pi/4,
                 "np": np, "pd": pd, "plt": plt}
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(cell, state)
            report = state["sector_energy_report"]
            self.assertEqual((report.loc["N", "direction_rows"], report.loc["N", "valid_power_rows"]), (2, 0))
            self.assertTrue(pd.isna(report.loc["N", "modelled_energy_mwh"]))
            self.assertEqual((report.loc["S", "direction_rows"], report.loc["S", "valid_power_rows"]), (0, 0))
            self.assertTrue(pd.isna(report.loc["S", "modelled_energy_mwh"]))
            self.assertEqual((report.loc["E", "valid_power_rows"], report.loc["E", "modelled_energy_mwh"]), (1, 1.))
            self.assertIn("Missing direction rows: 1", output.getvalue())
            self.assertIn("Leading sector(s) for modelled energy: E (1.000 MWh)", output.getvalue())
            state["fig"].canvas.draw()
            self.assertEqual(len(state["ax"].patches), 1)
            self.assertTrue(any("without valid power are omitted" in t.get_text() for t in state["ax"].texts))
            clean["power_mw"] = np.nan
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell, state)
            state["fig"].canvas.draw()
            self.assertEqual(len(state["ax"].patches), 0)
            self.assertTrue(state["energy_by_sector"].isna().all())
        finally:
            plt.close("all")

    def test_sector_counts_exclude_missing_directions_and_show_denominator(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def direction_sector(" in c)
        clean = pd.DataFrame({"direction_deg": [np.nan, np.nan, 0., 90.]})
        state = {"np": np, "pd": pd, "clean": clean}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertIn("2/4 observations", output.getvalue())
        self.assertEqual(state["sector_counts"].loc["N"], 1)
        self.assertEqual(state["sector_counts"].loc["E"], 1)
        self.assertEqual(state["sector_counts"].sum(), 2)
        self.assertEqual(state["sector_share"].loc["N"], .5)
        self.assertEqual(state["sector_share"].loc["E"], .5)
        self.assertEqual(clean["sector"].isna().sum(), 2)
        self.assertIn("count", output.getvalue())
        with self.assertRaisesRegex(ValueError, "No finite directions"):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell, {"np": np, "pd": pd, "clean": pd.DataFrame({"direction_deg": [np.nan]})})

    def test_meter_summary_rejects_invalid_duration_and_missing_samples(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def interval_summary(" in c)
        fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "interval_summary")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "quiz", "exec"), state)
        summary = state["interval_summary"]
        self.assertTrue(np.allclose(summary([.2, .8, 1.2, .6], 30), [1.4, .7]))
        for samples, minutes in (([.2, .8], -30), ([.2, .8], 0), ([.2, .8], np.inf),
                                 ([], 30), ([.2, np.nan], 30)):
            with self.subTest(samples=samples, minutes=minutes), self.assertRaises(ValueError):
                summary(samples, minutes)

    def test_expected_hour_window_checks_both_endpoints_and_edges(self):
        import ast
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def missing_hours(" in c)
        fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "missing_hours")
        state = {"pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "quiz", "exec"), state)
        check = state["missing_hours"]
        observed = pd.date_range("2025-01-01 01:00", periods=2, freq="h", tz="UTC")
        start, end = pd.Timestamp("2025-01-01 00:00", tz="UTC"), pd.Timestamp("2025-01-01 03:00", tz="UTC")
        self.assertEqual(check(observed, start, end).tolist(), [start, end])
        from datetime import timedelta, timezone
        offset = timezone(timedelta(hours=2))
        self.assertEqual(check(observed.tz_convert(offset), start.tz_convert(offset),
                               end.tz_convert(offset)).tolist(), [start, end])
        with self.assertRaisesRegex(ValueError, "on-the-hour"):
            check(observed, start, end + pd.Timedelta(minutes=30))
        with self.assertRaisesRegex(ValueError, "on-the-hour"):
            check(observed, start + pd.Timedelta(minutes=30), end)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            check(observed.tz_localize(None), start, end)
        with self.assertRaisesRegex(ValueError, "outside"):
            check(observed + pd.Timedelta(minutes=30), start, end)

    def test_cleaning_preserves_missing_directions_and_gap_boundaries(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        cleaning = next(c for c in cells if "short_gaps =" in c)
        mean = next(c for c in cells if "def circular_mean_deg(" in c)
        index = pd.date_range("2024-01-01", periods=8, freq="h")
        wind = pd.DataFrame({"speed_ms": [np.nan, 4, np.nan, np.nan, np.nan, 10, np.nan, 12],
                             "direction_deg": [np.nan, 350, 10, 0, 0, 0, 0, 0]}, index=index)
        state = {"pd": pd, "np": np, "wind": wind}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cleaning, state)
            exec(mean, state)
        self.assertTrue(state["clean"]["speed_ms"].iloc[[0, 2, 3, 4]].isna().all())
        self.assertEqual(state["clean"]["speed_ms"].iloc[6], 11)
        self.assertTrue(pd.isna(state["clean"]["direction_deg"].iloc[0]))
        self.assertIn("Direction coverage: 7/8 observations", output.getvalue())
        self.assertTrue(pd.isna(wind["speed_ms"].iloc[6]))
        irregular = wind.iloc[[1, 2, 7]].copy()
        with self.assertRaisesRegex(ValueError, "regular hourly"):
            exec(cleaning, {"pd": pd, "np": np, "wind": irregular})

    def test_wind_loss_waterfall_displays_net_aep(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def apply_sequential_losses(" in c)
        output = io.StringIO(); state = {"np": np, "plt": plt, "aep_mwh": 100000.}
        try:
            with contextlib.redirect_stdout(output):
                exec(cell, state)
            expected = 100000 * .92 * .96 * .97 * .98
            self.assertAlmostEqual(list(state["stages"].values())[-1], expected)
            self.assertIn(f"Net AEP: {expected:.3f} MWh", output.getvalue())
            self.assertIn("availability loss 4% (96% available)", text)
            self.assertIn("After availability loss", output.getvalue())
        finally:
            plt.close("all")

    def test_sample_histogram_and_seasonal_budget_display_results(self):
        import ast
        import numpy as np
        quiz = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", quiz, re.S)
        curve_cell = next(c for c in cells if "def turbine_power_mw(" in c)
        function = next(n for n in ast.parse(curve_cell).body if isinstance(n, ast.FunctionDef) and n.name == "turbine_power_mw")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "quiz", "exec"), state)
        histogram = next(c for c in cells if "100-hour sample: gross" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(histogram, state)
        self.assertIn("gross 185.620", output.getvalue())
        self.assertIn("class-specific net 169.496", output.getvalue())
        self.assertIn("uniform-availability net 167.522", output.getvalue())
        workbook = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", workbook, re.S) if "Season {i}: gross" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        self.assertIn("Season 1: gross 7.000, offline 0.700, wake 0.945, net 5.355", output.getvalue())
        self.assertIn("Season 2: gross 3.000, offline 0.000, wake 0.030, net 2.970", output.getvalue())

    def test_missing_speed_and_revisiting_after_gap_exercise(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power_cell = next(c for c in cells if "def turbine_power_mw(" in c)
        gap_cell = next(c for c in cells if "gap_filled=s.mask" in c)
        clean = pd.DataFrame({"speed_ms": [np.nan, -1., 0., 12., 25.]})
        state = {"np": np, "pd": pd, "clean": clean, "wind": clean.copy()}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(power_cell, state)
            exec(gap_cell, state)
            exec(power_cell, state)
        self.assertIs(state["clean"], clean)
        self.assertTrue(clean.power_mw.iloc[:2].isna().all())
        self.assertEqual(clean.power_mw.iloc[2:].tolist(), [0, 3, 0])
        curve = state["turbine_power_mw"]
        for kwargs in ({"rated": -1}, {"rated": float("nan")}, {"cut_in": -1},
                       {"cut_in": 12}, {"rated_speed": 25}, {"cut_out": float("inf")}):
            with self.subTest(parameters=kwargs), self.assertRaises(ValueError):
                curve([0, 12], **kwargs)
        self.assertEqual(curve([0, 12, 25], rated=0).tolist(), [0, 0, 0])

    def test_quality_profile_separates_missing_from_invalid_direction(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if '"invalid_direction":' in c)
        wind = pd.DataFrame({"speed_ms": [1, -1, 2], "direction_deg": [0, np.nan, 370]})
        state = {"pd": pd, "wind": wind}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertEqual(state["quality"]["missing_direction"], 1)
        self.assertEqual(state["quality"]["invalid_direction"], 1)

    def test_height_comparison_allows_cutout_to_reverse_order(self):
        text = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "vref=np.array([3,5,7,9])" in c and "def curve(v):" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell.replace("vref=np.array([3,5,7,9])", "vref=np.array([19,19,19,19])"), state)
        self.assertEqual(state["energy"][50], 72.)
        self.assertEqual(state["energy"][100], 0.)
        with self.assertRaises(ValueError):
            state["curve"]([float("nan"), 5])

    def test_daily_baseline_rejects_duplicate_and_off_hour_timestamps(self):
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "daily_groups = power_for_forecast.resample" in c)
        guard = cell.split("start =", 1)[0]
        regular = pd.date_range("2023-01-01", periods=24, freq="h")
        for index in (regular.insert(1, regular[0]).delete(2), regular + pd.Timedelta(minutes=30), regular[::-1]):
            wind = pd.DataFrame({"speed_ms": 1.}, index=index)
            with self.subTest(index=index), self.assertRaises(ValueError):
                exec(guard, {"pd": pd, "wind": wind})
        exec(guard, {"pd": pd, "wind": pd.DataFrame({"speed_ms": 1.}, index=regular)})

    def test_forecast_requires_the_stated_30_day_split(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power_cell = next(c for c in cells if "def turbine_power_mw(" in c)
        function = next(n for n in ast.parse(power_cell).body if isinstance(n, ast.FunctionDef) and n.name == "turbine_power_mw")
        state = {"np": np, "pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "curve", "exec"), state)
        forecast = next(c for c in cells if "daily_groups = power_for_forecast.resample" in c).split("plt.figure", 1)[0]
        for periods in (29 * 24, 31 * 24):
            state["wind"] = pd.DataFrame({"speed_ms": 6.}, index=pd.date_range("2025-01-01", periods=periods, freq="h"))
            with self.subTest(periods=periods), self.assertRaisesRegex(ValueError, "30 complete consecutive days"):
                exec(forecast, state)
        for timezone in (None, "UTC"):
            with self.subTest(timezone=timezone), contextlib.redirect_stdout(io.StringIO()):
                state["wind"] = pd.DataFrame({"speed_ms": 6.}, index=pd.date_range("2025-01-01", periods=30 * 24, freq="h", tz=timezone))
                exec(forecast, state)
                self.assertEqual((len(state["train"]), len(state["test"])), (23, 7))
                self.assertTrue((state["train"].index < state["cutoff"]).all())
                self.assertTrue((state["test"].index >= state["cutoff"]).all())
                self.assertEqual(str(state["daily_energy"].index.tz), str(timezone))

    def test_forecast_training_does_not_use_first_test_hour(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power_cell = next(c for c in cells if "def turbine_power_mw(" in c)
        function = next(n for n in ast.parse(power_cell).body if isinstance(n, ast.FunctionDef) and n.name == "turbine_power_mw")
        state = {"np": np, "pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "curve", "exec"), state)
        forecast = next(c for c in cells if "daily_groups = power_for_forecast.resample" in c).split("plt.figure", 1)[0]
        means = []
        for first_test_speed in (10., 18.):
            speed = np.full(720, 6.)
            speed[550:553] = [4., np.nan, first_test_speed]
            state["wind"] = pd.DataFrame({"speed_ms": speed}, index=pd.date_range("2025-01-01", periods=720, freq="h"))
            exec(forecast, state)
            self.assertEqual(state["training_speed"].iloc[-1], 4.)
            means.append(state["forecast"].iloc[0])
        self.assertAlmostEqual(means[0], means[1])
        self.assertAlmostEqual(means[0], 7.97669, places=4)

    def test_forecast_rejects_missing_test_targets_and_long_training_gaps(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power_cell = next(c for c in cells if "def turbine_power_mw(" in c)
        function = next(n for n in ast.parse(power_cell).body if isinstance(n, ast.FunctionDef) and n.name == "turbine_power_mw")
        state = {"np": np, "pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "curve", "exec"), state)
        forecast = next(c for c in cells if "daily_groups = power_for_forecast.resample" in c).split("plt.figure", 1)[0]
        for name, missing in (("test target", [23 * 24]), ("long training gap", [100, 101, 102])):
            speed = np.full(720, 6.)
            speed[missing] = np.nan
            state["wind"] = pd.DataFrame({"speed_ms": speed}, index=pd.date_range("2025-01-01", periods=720, freq="h"))
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, "24 valid hourly observations"):
                exec(forecast, state)
            if name == "test target":
                self.assertTrue(pd.isna(state["test_speed"].iloc[0]))
            else:
                self.assertTrue(pd.isna(state["training_speed"].iloc[102]))

    def test_incomplete_days_are_not_used_for_baseline(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "daily_groups = power_for_forecast.resample" in c)
        aggregation = cell.split("daily_groups =", 1)[1].split("if daily_energy.isna()", 1)[0]
        power = pd.Series([np.nan]*24 + [1.]*12 + [np.nan]*12 + [1.]*24,
                          index=pd.date_range("2023-01-01", periods=72, freq="h"))
        state = {"power_for_forecast": power, "pd": pd}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec("daily_groups =" + aggregation, state)
        self.assertEqual(state["daily_counts"].tolist(), [0, 12, 24])
        self.assertTrue(state["daily_energy"].iloc[:2].isna().all())
        self.assertEqual(state["daily_energy"].iloc[2], 24)
        self.assertIn("valid_hours", output.getvalue())
        self.assertIn("expected_hours", output.getvalue())
        self.assertEqual(state["daily_coverage"].expected_hours.tolist(), [24, 24, 24])

class UnitConversionTests(unittest.TestCase):
    def test_alignment_rejects_missing_values_duplicates_and_different_times(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/workbook/general.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def align(solar, wind, expected):" in c)
        node = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "align")
        state = {"np": np, "pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "lesson", "exec"), state)
        times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        solar = pd.DataFrame({"time": times, "solar_mw": [0., 1., 2.]})
        wind = pd.DataFrame({"time": times, "wind_mw": [2., 1., 0.]})
        self.assertEqual(len(state["align"](solar, wind, times)), 3)
        solar_text = solar.assign(solar_mw=["0", "1", "2"])
        wind_text = wind.assign(wind_mw=["2", "1", "0"])
        converted = state["align"](solar_text, wind_text, times)
        self.assertEqual((converted.solar_mw + converted.wind_mw).tolist(), [2., 2., 2.])
        self.assertEqual(solar_text.solar_mw.tolist(), ["0", "1", "2"])
        self.assertEqual(wind_text.wind_mw.tolist(), ["2", "1", "0"])
        solar_dates = solar.assign(time=times.strftime("%Y-%m-%dT%H:%M:%SZ"))
        wind_dates = wind.assign(time=times.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        converted_dates = state["align"](solar_dates, wind_dates, times)
        self.assertEqual(len(converted_dates), 3)
        self.assertEqual((converted_dates.solar_mw + converted_dates.wind_mw).tolist(), [2., 2., 2.])
        self.assertEqual(solar_dates.time.iloc[0], "2025-01-01T00:00:00Z")
        mixed_dates = state["align"](solar_dates, wind, times)
        self.assertEqual(len(mixed_dates), 3)
        bad_text = solar.assign(solar_mw=["0", "not power", "2"])
        with self.assertRaises(ValueError):
            state["align"](bad_text, wind, times)
        missing = solar.copy(); missing.loc[1, "solar_mw"] = float("nan")
        duplicate = solar.copy(); duplicate.loc[1, "time"] = duplicate.loc[0, "time"]
        shifted = solar.copy(); shifted.loc[1, "time"] += pd.Timedelta(hours=4)
        for invalid in (missing, duplicate, shifted):
            with self.assertRaises(ValueError):
                state["align"](invalid, wind, times)
        with self.assertRaisesRegex(ValueError, "expected UTC schedule"):
            state["align"](solar.drop(index=1), wind.drop(index=1), times)
        naive = solar.copy(); naive["time"] = naive["time"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "expected UTC schedule"):
            state["align"](naive, wind, times)
        with self.assertRaisesRegex(ValueError, "complete, ordered hourly UTC"):
            state["align"](solar, wind, times.delete(1))

    def test_rejects_nonfinite_and_misaligned_values(self):
        import numpy as np
        text = (BOOK / "section7/workbook/general.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def to_mwh(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        convert = state["to_mwh"]
        with self.assertRaises(ValueError):
            convert(pd.Series([1., 2.], index=[0, 1]), pd.Series(["MWh", "kWh"], index=[1, 2]))
        with self.assertRaises(ValueError):
            convert(pd.Series([np.inf]), pd.Series(["MWh"]))
        for index in (pd.Index(["a", None]), pd.MultiIndex.from_tuples([("a", 1), ("b", None)])):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "must not be missing"):
                convert(pd.Series([1., 2.], index=index), pd.Series(["MWh", "kWh"], index=index))
        valid_index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
        self.assertEqual(convert(pd.Series([1., 2000.], index=valid_index),
                                 pd.Series(["MWh", "kWh"], index=valid_index)).tolist(), [1., 2.])
        self.assertEqual(convert(pd.Series([1000.]), pd.Series(["kWh"])).item(), 1.)

    def test_direction_sector_rejects_missing_without_north_imputation(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/eolicenergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def direction_sector(" in c)
        code = cell.split("valid_direction =", 1)[0]
        state = {"np": np}; exec(code, state)
        self.assertEqual(state["direction_sector"]([0, 90]).tolist(), ["N", "E"])
        with self.assertRaises(ValueError):
            state["direction_sector"]([np.nan, 0, 90])

class QuizTemperatureTests(unittest.TestCase):
    def test_geothermal_field_and_sensitivity_use_stated_water_properties(self):
        import ast
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        for number in (7, 13):
            section = text.split(f"Quiz {number} —", 1)[1].split(f"Quiz {number + 1} —", 1)[0]
            self.assertIn("constant specific heat 4,180 J/(kg K)", section)
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power = next(c for c in cells if "def well_power_mw(" in c)
        fn = next(n for n in ast.parse(power).body if isinstance(n, ast.FunctionDef) and n.name == "well_power_mw")
        state = {"np": np, "plt": plt}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "quiz", "exec"), state)
        field = next(c for c in cells if '"well": ["A", "B", "C", "D"]' in c)
        surface = next(c for c in cells if "efficiency_axis = np.linspace(0.08, 0.18, 60)" in c)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(field, state)
                exec(surface, state)
            np.testing.assert_allclose(state["wells"]["net_mw"], [2.21426304, 3.412465056, 1.76783904, 4.2187068])
            self.assertAlmostEqual(state["NET"].min(), 2.033152)
            self.assertAlmostEqual(state["NET"].max(), 5.432328)
        finally:
            plt.close("all")

    def test_darcy_example_states_flow_boundary_and_calculates_flux(self):
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        section = text.split("Quiz 8 —", 1)[1].split("Quiz 9 —", 1)[0]
        self.assertIn("horizontal single-phase flow with gravity neglected", section)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section, re.S).group(1)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertAlmostEqual(state["flux"], 7.049452142857e-8, places=18)

    def test_geothermal_scenarios_display_requested_results(self):
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        self.assertIn("being available alone does not establish operating hours or output", text)
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        cases = [
            ("def annual(power,availability):", ("7708.8 operating hours, 30835.2 MWh", "8322.0 operating hours, 33288.0 MWh", "8760.0 operating hours, 35040.0 MWh")),
            ("cases={q:case(q)", ("best flow: 60 kg/s", "Any candidate consumes more than it generates: False")),
            ("full,clipped=field([0,1])", ("All-on potential, export, clipped (MW)", "Single-well exports (MW)")),
            ("choice=max(t for t in options", ("warmest feasible: 70 °C",)),
            ("bias=1-liquid[1]/reference[1]", ("39.04%",)),
        ]
        for marker, expected in cases:
            cell = next(c for c in cells if marker in c)
            output = io.StringIO()
            with self.subTest(marker=marker), contextlib.redirect_stdout(output):
                exec(cell, {})
            for value in expected:
                self.assertIn(value, output.getvalue())

    def test_geothermal_quiz_discloses_unchecked_efficiency(self):
        text = (BOOK / "section7/renewableEnergyquizzes/geothermalEnergy.md").read_text(encoding="utf-8")
        answer = text.split("Quiz 5 —", 1)[1].split("Quiz 6 —", 1)[0]
        self.assertIn("does not specify a heat sink or establish physical feasibility", answer)
        self.assertIn("finite cooling stream can impose a stricter limit", answer)

    def test_paired_power_models_reject_ambiguous_array_shapes(self):
        import ast
        import numpy as np
        cases = [
            ("solarEnergy.md", "pv_power_mw", lambda f, a, b: f(a, b, 10, 8), [500, 1000], [20, 25]),
            ("hydroelectricEnergy.md", "hydro_power_mw", lambda f, a, b: f(a, b), [1, 2], [20, 50]),
            ("geothermalEnergy.md", "well_power_mw", lambda f, a, b: f(a, b, 70, .12), [10, 20], [100, 150]),
        ]
        for filename, name, call, first, second in cases:
            text = (BOOK / "section7/renewableEnergyquizzes" / filename).read_text(encoding="utf-8")
            cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if f"def {name}(" in c)
            fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == name)
            state = {"np": np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), filename, "exec"), state)
            model = state[name]
            with self.subTest(model=name):
                for a, b in ((np.array(first).reshape(2, 1), second), (first, second + [second[-1]])):
                    with self.assertRaisesRegex(ValueError, "matching 1-D arrays"):
                        call(model, a, b)
                labels = pd.Index(["t0", "t1"])
                aligned = call(model, pd.Series(first, index=labels), pd.Series(second, index=labels))
                expected = call(model, first, second)
                if isinstance(aligned, tuple):
                    for actual, wanted in zip(aligned, expected):
                        np.testing.assert_allclose(actual, wanted)
                else:
                    np.testing.assert_allclose(aligned, expected)
                with self.assertRaisesRegex(ValueError, "matching indexes"):
                    call(model, pd.Series(first, index=labels), pd.Series(second, index=labels[::-1]))
                for bad_labels in (pd.Index(["t0", "t0"]), pd.Index(["t0", None]),
                                   pd.MultiIndex.from_tuples([("a", 1), ("b", None)])):
                    with self.assertRaisesRegex(ValueError, "unique, nonmissing"):
                        call(model, pd.Series(first, index=bad_labels), pd.Series(second, index=bad_labels))
                multi = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "x"])
                call(model, pd.Series(first, index=multi), pd.Series(second, index=multi))
                result = call(model, first, second)
                single = [call(model, a, b) for a, b in zip(first, second)]
                if isinstance(result, tuple):
                    for i, component in enumerate(result):
                        np.testing.assert_allclose(component, [x[i] for x in single])
                else:
                    np.testing.assert_allclose(result, single)
                scalar_result = call(model, first, second[0])
                self.assertEqual(np.asarray(scalar_result[0] if isinstance(scalar_result, tuple) else scalar_result).shape, (2,))
                if name == "well_power_mw":
                    with self.assertRaisesRegex(ValueError, "matching 1-D arrays"):
                        model(first, second, [60, 65, 70], .12)
                    with self.assertRaisesRegex(ValueError, "matching indexes"):
                        model(pd.Series(first, index=labels), pd.Series(second, index=labels),
                              pd.Series([60, 65], index=labels[::-1]), .12)

    def test_quiz_models_reject_impossible_temperatures(self):
        import ast
        import numpy as np
        cases = [("solarEnergy.md", "pv_power_mw", (800, -400, 10, 8)),
                 ("geothermalEnergy.md", "well_power_mw", (50, -300, -350, .12))]
        for filename, name, args in cases:
            text = (BOOK / "section7/renewableEnergyquizzes" / filename).read_text(encoding="utf-8")
            cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if f"def {name}(" in c)
            fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == name)
            state = {"np": np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), filename, "exec"), state)
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "absolute zero"):
                state[name](*args)
        with self.assertRaisesRegex(ValueError, "cell temperature"):
            solar = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8"), re.S) if "def pv_power_mw(" in c)
            fn = next(n for n in ast.parse(solar).body if isinstance(n, ast.FunctionDef) and n.name == "pv_power_mw")
            state = {"np": np}; exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "quiz", "exec"), state)
            state["pv_power_mw"](800, -270, 10, 8, noct_c=-260)


class SolarTheoryBoundaryTests(unittest.TestCase):
    def test_model_based_and_irradiation_based_ratios_are_distinguished(self):
        lesson = (BOOK / "section6/solarEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/solarEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for delivery, text in (("lesson", lesson), ("notebook", prose)):
            with self.subTest(delivery=delivery):
                self.assertIn("measured-to-model energy ratio", text)
                self.assertIn("not automatically equivalent to this STC-normalized ratio", text)
                self.assertIn("whether unavailable periods are excluded", text)
                self.assertIn("80 MWh measured divided by 90 MWh modelled is 0.889", text)
                self.assertIn("100 MWh irradiation-based reference is 0.800", text)
        self.assertAlmostEqual(80 / 90, 0.889, delta=.0005)
        self.assertEqual(80 / 100, 0.8)

    def test_impossible_input_and_computed_cell_temperatures_are_rejected(self):
        import ast
        import numpy as np
        text = (BOOK / "section6/solarEnergy.md").read_text(encoding="utf-8")
        lesson = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                      if "def cell_temperature_c(" in c)
        notebook = json.loads((BOOK / "section6/solarEnergy.ipynb").read_text(encoding="utf-8"))
        companion = next("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "code" and "def cell_temperature_c(" in "".join(c["source"]))
        for content in (text, "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")):
            self.assertIn("do not establish the valid cell-temperature range", content)
        for label, cell in (("lesson", lesson), ("notebook", companion)):
            functions = [n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef)
                         and n.name in {"cell_temperature_c", "pv_ac_power_mw"}]
            self.assertEqual(len(functions), 2)
            state = {"np": np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), label, "exec"), state)
            with self.subTest(delivery=label):
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["cell_temperature_c"](800, -400)
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["pv_ac_power_mw"](800, -400, 10, 8)
                with self.assertRaisesRegex(ValueError, "Calculated temperature"):
                    state["cell_temperature_c"](800, -270, noct_c=-260)
                with self.assertRaisesRegex(ValueError, "absolute zero"):
                    state["pv_ac_power_mw"]([800, 800], [25, -400], 10, 8)
                self.assertAlmostEqual(float(state["cell_temperature_c"](800, 25)), 50.)
                self.assertAlmostEqual(state["pv_ac_power_mw"](800, 25, 10, 8), 6.984)


class SolarCoverageTests(unittest.TestCase):
    def test_lcoe_quiz_states_timing_and_displays_cost_not_return(self):
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 20 —", 1)[1]
        self.assertIn("Compare lifetime energy and levelized cost", section)
        self.assertIn("initial capital is paid at year zero", section)
        self.assertIn("O&M and generated energy are discounted at year-end", section)
        self.assertIn("LCOE is a cost metric, not investment return", section)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section, re.S).group(1)
        state, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        for result, first, degradation, capital in ((state["a"], 1000., .01, 600000.),
                                                     (state["b"], 970., .002, 630000.)):
            energy = [first * (1 - degradation) ** y for y in range(20)]
            expected = (capital + sum(8000 / 1.04 ** y for y in range(1, 21))) / sum(
                e / 1.04 ** y for y, e in enumerate(energy, 1))
            self.assertAlmostEqual(result[0], sum(energy))
            self.assertAlmostEqual(result[1], expected)
        self.assertIn("lower LCOE: Case A", output.getvalue())
        self.assertIn("More lifetime energy: Case B", output.getvalue())

    def test_assessed_battery_examples_apply_ac_bus_limits(self):
        workbook = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        exercise = workbook.split("### Exercise 18 —", 1)[1].split("### Exercise 19 —", 1)[0]
        self.assertIn("charge and discharge limits measured at the AC bus", exercise)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", exercise, re.S)
                    if "unmet_total" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertAlmostEqual(state["unmet_total"], 2)
        self.assertAlmostEqual(state["curtailed_total"], 8 / 3)
        self.assertAlmostEqual(state["soc"], 7 / 9)

        quiz = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        question = re.search(r":::\{admonition\} Quiz 19\b(.*?)(?=:::\{admonition\} Quiz \d+|\Z)", quiz, re.S).group(1)
        self.assertIn("caps measured at the AC bus", question)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", question, re.S).group(1)
        state, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        for name, expected in (("served", 6), ("curtailed", 1), ("unmet", 2),
                               ("loss", 47 / 90), ("state", 43 / 90)):
            self.assertAlmostEqual(state[name], expected)
            self.assertIn("final charge" if name == "state" else ("losses" if name == "loss" else name), output.getvalue().lower())

    def test_assessed_solar_quiz_answers_display_results(self):
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        states = {}
        for number, label in [(3, "Irradiation:"), (4, "Electrical power:"), (8, "Horizontal irradiance:"),
                              (9, "Delivered and clipped energy"), (10, "Ideal and post-loss power"),
                              (11, "Energy kWh, mean kW"), (12, "Fill factor:"), (16, "performance ratio:")]:
            section = re.search(r":::{{admonition}} Quiz {}\b(.*?)(?=:::{{admonition}} Quiz \d+|\Z)".format(number), text, re.S)
            self.assertIsNotNone(section, number)
            cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section.group(1), re.S)
            self.assertIsNotNone(cell, number)
            output, state = io.StringIO(), {}
            with contextlib.redirect_stdout(output):
                exec(cell.group(1), state)
            self.assertIn(label, output.getvalue())
            states[number] = state
        self.assertEqual(states[3]["wh_to_kwh"](750), .75)
        self.assertAlmostEqual(600 * 2 * states[4]["fraction"](19), 228)
        self.assertAlmostEqual(states[8]["ghi"](700, 120, 60), 470)
        self.assertEqual(states[9]["capped_energy"]([1, 4, 6, 2], 3, 15), (2.25, 1.0))
        self.assertAlmostEqual(states[10]["output"], 335.16)
        self.assertEqual(states[11]["integrate_kw"]([.3, .6, 1.2, .9], 30), (1.5, .75))
        self.assertAlmostEqual(states[12]["fill_factor"](38, 9, 46, 10), 342 / 460)
        self.assertAlmostEqual(states[16]["pr"], .625)

    def test_workbook_degradation_shows_annual_and_running_totals(self):
        text = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def forecast(rate):" in c)
        output = io.StringIO()
        state = {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        annual, cumulative = state["forecast"](0.005)
        self.assertEqual(len(annual), 25)
        self.assertAlmostEqual(annual[0], 100)
        self.assertAlmostEqual(cumulative[1], 199.5)
        self.assertAlmostEqual(cumulative[-1], sum(100 * 0.995 ** i for i in range(25)))
        self.assertEqual(output.getvalue().count("Year  Annual MWh  Cumulative MWh"), 2)
        self.assertIn("First year below 90 MWh: 23", output.getvalue())
        self.assertIn("First year below 90 MWh: 12", output.getvalue())
        for invalid in (-0.1, 1, float("nan"), float("inf")):
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                state["forecast"](invalid)

    def test_matched_performance_ratio_rejects_missing_keys_and_zero_denominator(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def validate_hourly_readings(" in c)
        function = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "validate_hourly_readings")
        state = {"np": np, "pd": pd}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "solar quiz", "exec"), state)
        validate = state["validate_hourly_readings"]
        good = pd.date_range("2025-06-01", periods=2, freq="h", tz="UTC")
        self.assertEqual(validate(pd.Series(["500", "700"], index=good)).tolist(), [500., 700.])
        for index in (pd.DatetimeIndex([good[0], pd.NaT]), pd.DatetimeIndex([good[0], good[0]]), good + pd.Timedelta(minutes=30), good.tz_localize(None)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate(pd.Series([1., 2.], index=index))
        for values in ([1., np.inf], [1., -1.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(pd.Series(values, index=good))
        self.assertTrue(pd.isna(validate(pd.Series([1., np.nan], index=good)).iloc[1]))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, {})
            with self.assertRaisesRegex(ValueError, "positive total irradiance"):
                exec(cell.replace("[500,700,900]", "[0,0,0]"), {})

    def test_daily_profile_prints_dc_energy_and_reconciles_both_caps(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "dc_kw = g*20*0.2/1000" in c)
        output = io.StringIO(); state = {}
        try:
            with contextlib.redirect_stdout(output):
                exec(cell, state)
            self.assertAlmostEqual(state["dc_kw"].sum()/1000, .02430641316, places=8)
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                for label in ("MWh DC", "MWh delivered", "MWh clipped", "MWh conversion loss"):
                    self.assertIn(label, line)
        finally:
            plt.close("all")

    def test_degradation_rejects_nonfinite_energy_and_fractional_years(self):
        import ast
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def degraded_energy(" in c)
        function = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "degraded_energy")
        state = {"np": np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "solar", "exec"), state)
        fn = state["degraded_energy"]
        for args in ((100, .005, 2.5), (float("nan"), .005, 3), (float("inf"), .005, 3), (100, .005, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fn(*args)
        years, annual, cumulative = fn(15000, .005, 25)
        self.assertEqual(len(years), 25)
        self.assertEqual(annual[0], 15000)
        self.assertTrue(np.all(np.diff(annual) <= 0))
        self.assertAlmostEqual(cumulative[-1], sum(annual))

    def test_performance_ratio_plot_keeps_flagged_high_value_visible(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if 'monthly["pr"].plot.bar' in c)
        cell = cell.replace('"energy_mwh": [420, 620,', '"energy_mwh": [696, 620,')
        state = {"plt": plt}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell, state)
            self.assertAlmostEqual(state["monthly"].iloc[0]["pr"], 1.2)
            self.assertTrue(state["monthly"].iloc[0]["flag"])
            self.assertGreaterEqual(plt.gca().get_ylim()[1], 1.2)
            zero = cell.replace('"irradiation_kwh_m2": [58, 82,', '"irradiation_kwh_m2": [0, 82,')
            self.assertNotEqual(zero, cell)
            zero_state = {"plt": plt}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(zero, zero_state)
            self.assertTrue(pd.isna(zero_state["monthly"].iloc[0]["pr"]))
            self.assertFalse(zero_state["monthly"].iloc[0]["flag"])
            self.assertIn("Months with unavailable PR: ['2025-01']", output.getvalue())
            negative = cell.replace('"irradiation_kwh_m2": [58, 82,', '"irradiation_kwh_m2": [-1, 82,')
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                exec(negative, {"plt": plt})
        finally:
            plt.close("all")

    def test_irradiance_closure_daylight_domain_and_source(self):
        import math
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 8 — Find irradiance on a horizontal panel", 1)[1].split(":::", 1)[0]
        self.assertIn("complete_irradiance.html", prompt)
        self.assertIn("0° through 90°", prompt)
        self.assertIn("mislabelled its `ghi` parameter as DNI", prompt)
        self.assertIn("GHI is global horizontal irradiance", prompt)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def ghi(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        fn = state["ghi"]
        self.assertAlmostEqual(fn(700, 120, 0), 820)
        self.assertAlmostEqual(fn(700, 120, 60), 470)
        self.assertAlmostEqual(fn(700, 120, 90), 120)
        for args in ((700, 120, 120), (-1, 120, 60), (700, math.nan, 60)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fn(*args)
        self.assertIn("Each reading represents one hour", text)

    def test_paired_performance_ratio_stipulates_reference_and_plane(self):
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 16 — Join inverter and irradiance readings", 1)[1].split(":::", 1)[0]
        for phrase in ("plane-of-array irradiance", "5 kW DC nameplate at standard test conditions", "1,000 W/m² reference irradiance"):
            self.assertIn(phrase, prompt)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "pr=(paired.ac.sum()/5)" in c)
        state = {}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertEqual(len(state["paired"]), 2)
        self.assertAlmostEqual(state["pr"], .625)
        self.assertIn("performance ratio: 0.625", output.getvalue())

    def test_clipping_and_storage_prompts_specify_constant_intervals(self):
        quiz = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        workbook = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        self.assertIn("each held constant for 15 minutes", quiz)
        self.assertIn("held constant over respective durations", quiz)
        self.assertIn("each held constant over its one-hour interval", quiz)
        self.assertIn("each constant within its one-hour interval", workbook)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", quiz, re.S) if "def capped_energy(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        self.assertEqual(state["capped_energy"]([1,4,6,2], 3, 15), (2.25, 1.0))

    def test_battery_quiz_states_charge_and_discharge_boundaries(self):
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 19 — Evaluate battery charging losses", 1)[1].split(":::", 1)[0]
        self.assertIn("caps measured at the AC bus", prompt)
        self.assertIn("power delivered to the load", prompt)

    def test_solar_quiz_solutions_print_requested_results(self):
        text = (BOOK / "section7/renewableEnergyquizzes/solarEnergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        cases = [("def retained(cap):", "Smallest qualifying inverter: 6"),
                 ("flagged=np.where(observed[4:]", "flagged days: [6, 7]"),
                 ("served,curtailed,unmet,loss=totals", "Served 6.000, curtailed 1.000, unmet 2.000"),
                 ("def project(first, degradation, capital):", "More lifetime energy: Case B; lower LCOE: Case A")]
        for marker, expected in cases:
            cell = next(c for c in cells if marker in c)
            output = io.StringIO()
            with self.subTest(marker=marker), contextlib.redirect_stdout(output):
                exec(cell, {})
            self.assertIn(expected, output.getvalue())

    def test_transposition_checks_horizontal_and_tilted_fixture(self):
        text = (BOOK / "section6/solarEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def isotropic_poa(" in c)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, {})
        mutated = cell.replace("ground = reflectance * ghi", "ground = 2 * reflectance * ghi", 1)
        self.assertNotEqual(mutated, cell)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(AssertionError):
                exec(mutated, {})

    def test_daylight_coverage_handles_invalid_and_night_only_intervals(self):
        import numpy as np
        text = (BOOK / "section7/workbook/solar.md").read_text(encoding="utf-8")
        self.assertIn("If there is no daylight duration, daylight coverage is undefined and reported as NaN", text)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def audit_irradiance(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        audit = state["audit_irradiance"]
        with self.assertRaises(ValueError):
            audit([500, -1], [-1, 2], [True, True])
        with self.assertRaises(ValueError):
            audit([500], [1, 2], [True])
        with np.errstate(all="raise"):
            total, daylight, energy = audit([0, 0], [1, 1], [False, False])
        self.assertEqual(total, 1)
        self.assertTrue(np.isnan(daylight))
        self.assertEqual(energy, 0)
        for readings, durations in (([np.nan, -1], [1, 1]), ([0, np.nan], [0, 1])):
            coverage, _, observed = audit(readings, durations, [True, True])
            self.assertEqual(coverage, 0)
            self.assertTrue(np.isnan(observed))
        irradiance = pd.Series([500., 0.], index=["A", "B"])
        duration = pd.Series([1., 2.], index=["A", "B"])
        flags = pd.Series([True, False], index=["A", "B"])
        self.assertEqual(audit(irradiance, duration, flags), (1., 1., .5))
        nullable_irradiance = pd.Series([500., pd.NA], index=["A", "B"], dtype="Float64")
        nullable_flags = pd.Series([True, True], index=["A", "B"], dtype="boolean")
        np.testing.assert_allclose(audit(nullable_irradiance, duration, nullable_flags), (1 / 3, 1 / 3, .5))
        with self.assertRaisesRegex(ValueError, "boolean daylight flags"):
            audit(irradiance, duration, pd.Series([True, pd.NA], index=["A", "B"], dtype="boolean"))
        for bad_flags in [flags.iloc[::-1], pd.Series([True, False], index=["A", "A"]), [True, False]]:
            with self.assertRaisesRegex(ValueError, "same unique index"):
                audit(irradiance, duration, bad_flags)
        for index in (pd.Index(["A", None]), pd.MultiIndex.from_tuples([("A", 1), ("B", None)])):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "must not be missing"):
                audit(pd.Series([500., 0.], index=index), pd.Series([1., 2.], index=index),
                      pd.Series([True, False], index=index))
        valid_index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
        self.assertEqual(audit(pd.Series([500., 0.], index=valid_index),
                               pd.Series([1., 2.], index=valid_index),
                               pd.Series([True, False], index=valid_index)), (1., 1., .5))

class HydroMissingCoverageTests(unittest.TestCase):
    def test_quiz_bounds_distinguish_all_missing_from_observed_zero(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def energy_bounds(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        fn = state["energy_bounds"]
        observed, coverage, upper = fn([np.nan, np.nan], 1.5)
        self.assertTrue(np.isnan(observed))
        self.assertEqual((coverage, upper), (0, 3))
        self.assertEqual(fn([0, 0], 1.5), (0, 1, 0))
        for readings, expected in (("[np.nan,np.nan]", "Observed energy: unavailable; lower bound: 0.000 MWh; coverage: 0%; upper bound: 3.000 MWh"),
                                   ("[0,0]", "Observed energy: 0.000 MWh; lower bound: 0.000 MWh; coverage: 100%; upper bound: 0.000 MWh")):
            output = io.StringIO()
            with self.subTest(readings=readings), contextlib.redirect_stdout(output):
                exec(cell.replace("observed,coverage,upper=energy_bounds([.4,np.nan,.8,np.nan],1.5)",
                                  f"observed,coverage,upper=energy_bounds({readings},1.5)"), {})
            self.assertIn(expected, output.getvalue())
        for bad_rating in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                fn([np.nan], bad_rating)
        for readings in ([], [1.6], [-.1], [np.inf]):
            with self.subTest(readings=readings), self.assertRaises(ValueError):
                fn(readings, 1.5)

    def test_missing_volume_is_not_reported_as_a_measured_zero(self):
        import numpy as np
        text = (BOOK / "section7/workbook/hydroelectric.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def audit(flow, duration, limit):" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        audit = state["audit"]
        samples, duration, observed, upper = audit([np.nan, -1], [1, 2], 6)
        self.assertEqual((samples, duration, upper), (0, 0, 64800))
        self.assertTrue(np.isnan(observed))
        self.assertEqual(audit([0, 0], [1, 2], 6), (1, 1, 0, 0))


class HydroFlowTests(unittest.TestCase):
    def test_monthly_energy_quiz_displays_annual_total(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 13 —", 1)[1].split(":::{admonition} Quiz 14 —", 1)[0]
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section, re.S).group(1)
        output, state = io.StringIO(), {"np": np, "plt": plt}
        try:
            with contextlib.redirect_stdout(output):
                exec(cell, state)
            powers = [15, 16, 18, 22, 25, 27, 24, 20, 17, 15, 14, 13]
            days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            annual = sum(p * d * 24 for p, d in zip(powers, days)) / 1000
            self.assertAlmostEqual(state["annual_gwh"], annual)
            self.assertAlmostEqual(state["capacity_factor"], annual * 1000 / (30 * 366 * 24))
            self.assertIn(f"Annual energy: {annual:.3f} GWh", output.getvalue())
        finally:
            plt.close("all")

    def test_assessed_early_hydro_quiz_answers_display_results(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        states = {}
        for number, label in [(3, "2,500 L/s"), (4, "14,500 kWh"), (9, "Net heads"),
                              (10, "environmental shortfall"), (11, "Implied efficiency"),
                              (12, "Turbine throughput"), (15, "Final storage, spill, environmental shortfall")]:
            section = re.search(r":::{{admonition}} Quiz {}\b(.*?)(?=:::{{admonition}} Quiz \d+|\Z)".format(number), text, re.S)
            self.assertIsNotNone(section, number)
            cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section.group(1), re.S)
            self.assertIsNotNone(cell, number)
            output, state = io.StringIO(), {"np": np}
            with contextlib.redirect_stdout(output):
                exec(cell.group(1), state)
            self.assertIn(label, output.getvalue())
            states[number] = state
        self.assertEqual(states[3]["litres_to_m3s"](2500), 2.5)
        self.assertEqual(states[4]["kwh_to_mwh"](14500), 14.5)
        self.assertEqual([states[9]["net_head"](18, v) for v in (1, 7, 20)], [17, 11, 0])
        self.assertEqual(states[10]["allocate"](15, 4, 8), (4, 8, 3))
        self.assertEqual(states[10]["allocate"](2, 4, 8), (2, 0, 0))
        self.assertAlmostEqual(states[11]["efficiency"](900, 5.5, 22), 900000 / (1000 * 9.81 * 5.5 * 22))
        self.assertEqual(states[12]["intensity"](9, 2, 0.75), 16200)
        self.assertEqual(states[15]["route"]([0, 10, 0]), (0, 9600, 0))
        self.assertIn("each hour’s total inflow is available before", text)

    def test_hydraulic_loss_source_note_uses_specific_locators(self):
        text = (BOOK / "section6/hydroelectricEnergy.md").read_text(encoding="utf-8")
        notebook = json.loads((BOOK / "section6/hydroelectricEnergy.ipynb").read_text(encoding="utf-8"))
        notebook_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for content in (text, notebook_text):
            self.assertIn("Related head and loss context:", content)
            self.assertIn("§§7.2.1 and 7.2.4", content)
            self.assertIn("FE-Ref-9.0_Web.v2.pdf", content)
            self.assertNotIn("§8.3 and Chapter 10;", content)

    def test_flow_inference_reports_increasing_branch_and_residual(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 17 —", 1)[1].split(":::{admonition} Quiz 18 —", 1)[0]
        code = re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)[0]
        state, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, state)
        self.assertAlmostEqual(state["flow"], 5.11498, places=4)
        self.assertAlmostEqual(state["net_head"], 33.21511, places=4)
        self.assertGreater(state["slope"], 0)
        self.assertLess(abs(state["residual"]), 1e-9)
        self.assertIn("Power slope:", output.getvalue())
        self.assertIn("power residual:", output.getvalue())

    def test_quiz_release_and_electrical_boundaries_are_explicit(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        for number in (10, 16):
            question = text.split(f":::{'{admonition}'} Quiz {number} —", 1)[1].split(":::{admonition} Solution", 1)[0]
            self.assertIn("separate bypass reach upstream of the turbine return", question)
        for number in (11, 16, 17):
            question = text.split(f":::{'{admonition}'} Quiz {number} —", 1)[1].split(":::{admonition} Solution", 1)[0]
            self.assertIn("combined turbine–generator efficiency", question)

    def test_workbook_policy_displays_integrated_bypass_and_shortfall(self):
        import numpy as np
        text = (BOOK / "section7/workbook/hydroelectric.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def evaluate(target):" in c)
        state = {}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        np.testing.assert_allclose(state["low"], [3600, 2.29554, 21600])
        np.testing.assert_allclose(state["high"], [7200, 1.94238, 10800])
        for target in (2, 3):
            self.assertIn(f"Target {target} m³/s: shortfall m³, electricity MWh, bypass m³:", output.getvalue())

    def test_quiz_flow_limit_displays_why_next_lower_choice_fails(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "target=.9*energy(float('inf'))" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        self.assertIn("Selected limit 8 m³/s; selected energy 6.887 MWh; unlimited 7.416 MWh", output.getvalue())
        self.assertIn("90% target 6.675 MWh; next lower limit 6 m³/s gives 5.827 MWh and does not meet the target", output.getvalue())

    def test_quiz_policy_prints_and_reconciles_water_volumes(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        question = text.split(":::{admonition} Quiz 18 —", 1)[1].split(":::{admonition} Solution", 1)[0]
        self.assertIn("combined turbine–generator efficiency", question)
        self.assertIn("separate bypass reach upstream of the turbine return", question)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def policy(reserve):" in c)
        output = io.StringIO(); state = {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        np.testing.assert_allclose(state["low"], [15.485493, 0], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(state["high"], [12.950589, 7200], rtol=1e-7, atol=1e-6)
        for label in ("inflow", "release", "turbine", "bypass"):
            self.assertEqual(output.getvalue().count("'" + label + "'"), 2)

    def test_lesson_power_model_documents_scalar_configuration_and_broadcasts_flow(self):
        import ast
        import numpy as np
        text = (BOOK / "section6/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def hydro_power_mw(" in c)
        fn = next(n for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == "hydro_power_mw")
        self.assertIn("scalar configuration values", ast.get_docstring(fn))
        state = {"np": np, "RHO_WATER": 1000, "G": 9.81}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "lesson", "exec"), state)
        model = state["hydro_power_mw"]
        np.testing.assert_allclose(model([1, 2], 20, efficiency=.8), [.15696, .31392])
        np.testing.assert_allclose(model([1, 2], 20, efficiency=.8, rated_power_mw=.2), [.15696, .2])
        # Flow, gross head and head loss broadcast together as documented.
        flow = np.array([[1.], [2.]])
        actual = model(flow, [20., 30.], efficiency=.8, head_loss_m=[0., 5.])
        np.testing.assert_allclose(actual, [[.15696, .1962], [.31392, .3924]])
        with self.assertRaises(ValueError):
            model(flow, [20., 30.], head_loss_m=[0., 31.])

    def test_workbook_duration_and_reservoir_design_values(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/workbook/hydroelectric.md").read_text(encoding="utf-8")
        self.assertIn("water density 1,000 kg/m³", text)
        self.assertIn("9.81 m/s²", text)
        self.assertIn("Store the constant interval flows and durations", text)
        self.assertIn("Calculate interval-average electrical powers", text)
        self.assertIn("a 60 m³/s mean river flow", text)
        self.assertIn("For a 365-day model year", text)
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        duration = next(c for c in cells if "Synthetic hydro duration curves" in c)
        design = next(c for c in cells if "generator_kw =" in c)
        state = {}; output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(duration, state)
                exec(design, state)
            np.testing.assert_allclose(state["totals"], [168.04464, 127.09296])
            self.assertAlmostEqual(state["annual"], 144e6)
            self.assertAlmostEqual(state["generator_kw"], 7745, delta=1)
            self.assertIn("design generator power", output.getvalue())
            self.assertIn("18.48 GWh/year", output.getvalue())
            self.assertIn("Sorting preserves total energy when each power stays paired with its duration", text)
        finally:
            plt.close("all")

    def test_policy_quiz_reconciles_stated_interval_volumes(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 18 — Compare policies over unequal intervals", 1)[1].split(":::", 1)[0]
        self.assertIn("constant within their respective intervals", prompt)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "Target {reserve} m³/s water volumes" in c)
        output = io.StringIO(); state = {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertAlmostEqual(state["low"][0], 15.485493, places=5)
        self.assertAlmostEqual(state["high"][0], 12.950589, places=5)
        self.assertEqual(state["high"][1], 7200)
        self.assertIn("'inflow': 334800", output.getvalue())
        for part in ("'release':", "'turbine':", "'bypass':"):
            self.assertIn(part, output.getvalue())
        self.assertIn("each constant for one hour", text)
        self.assertIn("one-hour interval-mean meter powers", text)

    def test_throughput_displays_both_requested_intervals(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def intensity(" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        self.assertIn("Turbine throughput over 0.75 h: 16200 m³/MWh", output.getvalue())
        self.assertIn("Turbine throughput over 2 h: 16200 m³/MWh", output.getvalue())

    def test_intake_prints_low_flow_shortfall_and_rejects_invalid_inputs(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S) if "def allocate(" in c)
        state = {"np": np}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertIn("River 2 m³/s: release 2, turbine 0, bypass 0; environmental shortfall 2 m³/s", output.getvalue())
        for args in ((2, 4, -1), (2, -1, 8), (2, np.nan, 8), (15, 4, np.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                state["allocate"](*args)

    def test_environmental_flow_prompt_identifies_a_target(self):
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        prompt = text.split("Quiz 6 — Environmental flow and plant limit", 1)[1].split(":::", 1)[0]
        self.assertIn("releasing at most the available river flow", prompt)
        self.assertIn("target shortfall", prompt)
        self.assertIn("electrical clipping alone does not determine turbine flow or bypass", prompt)
        self.assertIn("interval-mean powers", text)

    def test_quiz_distinguishes_shortfall_and_available_flow(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/hydroelectricEnergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        power = next(cell for cell in cells if "def hydro_power_mw(" in cell)
        flow = next(cell for cell in cells if "release = np.minimum(river_flow, 12)" in cell)
        state = {"np": np}; output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(power, state)
            exec(flow, state)
        self.assertEqual(state["release"][0], 8)
        self.assertEqual(state["shortfall"][0], 4)
        self.assertEqual(state["available_flow"][-1], 78)
        self.assertEqual(state["plant_power"][-1], 25)
        self.assertIn("electrical clipping alone does not specify bypass", flow)
        self.assertIn("river  release  shortfall  available  power", output.getvalue())
        self.assertRegex(output.getvalue(), r"(?m)^\s*8\s+8\s+4\s+0\s+0\.00 MW$")
        self.assertRegex(output.getvalue(), r"(?m)^\s*90\s+12\s+0\s+78\s+25\.00 MW$")

class WindCurveTests(unittest.TestCase):
    def test_workbook_direction_mean_uses_half_open_range(self):
        text = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def mean_direction(values):" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        for pair in ([350, 10], [359, 1]):
            angle = state["mean_direction"](pair)
            self.assertTrue(0 <= angle < 360)
            self.assertLess(min(angle, 360 - angle), 1e-8)
        with self.assertRaises(ValueError):
            state["mean_direction"]([90, 270])

    def test_quiz_loss_waterfall_identifies_and_applies_gross_input(self):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 15 —", 1)[1].split(":::{admonition} Quiz 16 —", 1)[0]
        self.assertIn("gross AEP `aep_mwh` calculated in Quiz 13", section)
        cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section, re.S).group(1)
        state, output = {"np": np, "plt": plt, "aep_mwh": 1000.}, io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(cell, state)
            expected = [1000., 920., 883.2, 856.704, 839.56992]
            np.testing.assert_allclose(list(state["stages"].values()), expected)
            self.assertIn("Sequential AEP stages (MWh)", output.getvalue())
            self.assertIn("Net AEP: 839.570 MWh", output.getvalue())
        finally:
            plt.close("all")

    def test_unequal_interval_metrics_reject_invalid_data_and_rating(self):
        import numpy as np
        text = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        section = text.split("### Exercise 14 —", 1)[1].split("### Exercise 15 —", 1)[0]
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", section, re.S)
                    if "def metrics(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell, "wind workbook", "exec", optimize=1), state)
        metrics = state["metrics"]
        energy, capacity = metrics([.5, 2, 3], [2, 1, .5], 3)
        self.assertAlmostEqual(energy, 4.5)
        self.assertAlmostEqual(capacity, 4.5 / (3 * 3.5))
        for powers, durations, rating in (([1, np.nan], [1, 1], 3), ([1, 2], [1, np.nan], 3),
                                          ([1, 2], [1, np.inf], 3), ([0, 0], [1, 1], 0),
                                          ([1, 2], [1, 1], np.nan), ([1, 4], [1, 1], 3),
                                          ([], [], 3)):
            with self.subTest(powers=powers, durations=durations, rating=rating), self.assertRaises(ValueError):
                metrics(powers, durations, rating)

    def test_assessed_early_wind_quiz_answers_display_results(self):
        import math
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        states = {}
        for number, label in [(3, "36 km/h"), (4, "15 rpm"), (8, "Swept areas"),
                              (9, "Tip-speed ratios"), (10, "Available power"),
                              (11, "Zero-mean intensity"), (12, "States")]:
            section = re.search(r":::{{admonition}} Quiz {}\b(.*?)(?=:::{{admonition}} Quiz \d+|\Z)".format(number), text, re.S)
            self.assertIsNotNone(section, number)
            cell = re.search(r"```\{code-cell\} python\n(.*?)\n```", section.group(1), re.S)
            self.assertIsNotNone(cell, number)
            output, state = io.StringIO(), {}
            with contextlib.redirect_stdout(output):
                exec(cell.group(1), state)
            self.assertIn(label, output.getvalue())
            states[number] = state
        self.assertEqual(states[3]["kmh_to_ms"](36), 10)
        self.assertAlmostEqual(states[4]["rpm_to_rad_s"](15), math.pi / 2)
        self.assertAlmostEqual(states[8]["swept_area"](60), math.pi * 30 ** 2)
        self.assertAlmostEqual(states[8]["swept_area"](90) / states[8]["swept_area"](60), 2.25)
        self.assertAlmostEqual(states[9]["ratios"][1], 1.5 * math.pi)
        self.assertAlmostEqual(states[10]["loss"], 0.08)
        self.assertIsNone(states[11]["zero_mean"])
        self.assertEqual([states[12]["state"](v) for v in [2, 3, 8, 12, 24, 25]],
                         ["below cut-in", "ramp", "ramp", "rated", "rated", "cut-out"])

    def test_turbulence_reports_computed_zero_mean_and_rejects_invalid_speeds(self):
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def turbulence(values):" in c)
        output = io.StringIO()
        state = {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        turbulence = state["turbulence"]
        self.assertAlmostEqual(turbulence([5, 7, 9]), (8 / 3) ** 0.5 / 7)
        self.assertIsNone(turbulence([0, 0]))
        self.assertIn("Zero-mean intensity: undefined", output.getvalue())
        for invalid in ([], [-1, 1], [1, float("nan")], [1, float("inf")], [[1, 2]]):
            with self.subTest(speeds=invalid), self.assertRaises(ValueError):
                turbulence(invalid)

    def test_wind_project_curve_reference_density_and_boundaries(self):
        text = (BOOK / 'section8/wind-project.md').read_text(encoding='utf-8')
        self.assertIn('east-positive x and north-positive y', text)
        self.assertIn('90° easterly input when passed −90°', text)
        self.assertIn('not a claim about the external author’s intended convention', text)
        # For a wind from the east, the velocity is westward, so the eastern
        # turbine is encountered first under the course's stated axes.
        eastward_velocity = -math.sin(math.radians(90))
        northward_velocity = -math.cos(math.radians(90))
        self.assertAlmostEqual(eastward_velocity, -1)
        self.assertAlmostEqual(northward_velocity, 0, places=12)
        cell = next(c for c in re.findall(r'```\{code-cell\} python\n(.*?)```', text, re.S)
                    if 'def power_kw(' in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        power = state['power_kw']
        self.assertEqual(power(3), .3 * 3**3)
        self.assertEqual(power(25), 518.4)
        self.assertEqual(power(25.1), 0)
        wind_flux_kw = .5 * 1.225 * math.pi * 20**2 * 12**3 / 1000
        ratio = power(12) / wind_flux_kw
        self.assertAlmostEqual(ratio, .39, delta=.001)
        self.assertLess(ratio, 16 / 27)
        self.assertIn('does not adjust the curve for air density', text)
        self.assertIn('theory curve uses zero output at those exact boundaries', text)
        theory = (BOOK / 'section6/windEnergy.md').read_text(encoding='utf-8')
        theory_cell = next(c for c in re.findall(r'```\{code-cell\} python\n(.*?)```', theory, re.S)
                           if 'def turbine_power_mw(' in c)
        theory_state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(theory_cell, theory_state)
        self.assertEqual(theory_state['turbine_power_mw'](3), 0)
        self.assertEqual(theory_state['turbine_power_mw'](25), 0)

    def test_quiz_sequential_losses_display_gap_from_gross_approximation(self):
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def waterfall(gross, losses):" in c)
        output = io.StringIO(); state = {}
        with contextlib.redirect_stdout(output):
            exec(cell, state)
        self.assertAlmostEqual(state["net"], 99.698928)
        self.assertIn("Net 99.699 GWh; gross-based approximation 98.400 GWh; gap 1.299 GWh", output.getvalue())
        self.assertIn("Losses by stage (GWh)", output.getvalue())

    def test_seasonal_loss_budget_shows_each_component_and_comparison(self):
        text = (BOOK / "section7/workbook/wind.md").read_text(encoding="utf-8")
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "wake_retained=np.array([.85,.99])" in c)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(cell, {})
        for expected in (
            "Season 1: gross 7.000, offline 0.700, wake 0.945, net 5.355 GWh",
            "Season 2: gross 3.000, offline 0.000, wake 0.030, net 2.970 GWh",
            "Seasonal net total: 8.325 GWh; mean-factor estimate: 8.740 GWh",
            "Total sequential loss fraction: 16.75%",
        ):
            self.assertIn(expected, output.getvalue())

    def test_assessed_curve_stays_below_ideal_bound_for_stated_rotor(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        section = text.split(":::{admonition} Quiz 18 —", 1)[1].split(":::{admonition} Quiz 19 —", 1)[0]
        self.assertIn("120 m diameter rotor", section)
        cell = next(c for c in re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
                    if "def turbine_power_mw(" in c)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell, state)
        speeds = np.linspace(0, 26, 26001)
        electrical = state["turbine_power_mw"](speeds)
        available = .5 * 1.22 * np.pi * 60**2 * speeds**3 / 1e6
        ideal_electrical_bound = available * (16 / 27) * .92
        self.assertTrue(np.all(electrical <= ideal_electrical_bound + 1e-12))
        self.assertEqual(state["turbine_power_mw"](12), 4)
        self.assertEqual(state["turbine_power_mw"](26), 0)
        rated_reference = .5 * 1.22 * np.pi * 60**2 * 12**3 / 1e6
        self.assertAlmostEqual(rated_reference, 11.92, delta=.005)
        self.assertAlmostEqual(rated_reference * (16/27), 7.06, delta=.005)
        self.assertAlmostEqual(rated_reference * (16/27) * .92, 6.50, delta=.005)
        for label in ("11.92 MW", "7.06 MW mechanical", "6.50 MW electrical"):
            self.assertIn(label, section)

    def test_quiz_curve_respects_ideal_electrical_bound(self):
        import numpy as np
        text = (BOOK / "section7/renewableEnergyquizzes/windEnergy.md").read_text(encoding="utf-8")
        cells = re.findall(r"```\{code-cell\} python\n(.*?)\n```", text, re.S)
        curve_code = next(cell for cell in cells if "def turbine_power_mw(" in cell)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(curve_code, state)
        speed = np.linspace(0, 30, 30001)
        output = state["turbine_power_mw"](speed)
        ideal_electrical = .5 * 1.22 * np.pi * 60**2 * speed**3 * (16/27) * .92 / 1e6
        self.assertTrue(np.all(output <= ideal_electrical + 1e-12))
        self.assertEqual(state["turbine_power_mw"](12), 4)
        self.assertIn("120 m diameter rotor", text)

class SandboxSourceInspectionTests(unittest.TestCase):
    def test_assessed_hydropower_authors_and_editors_render(self):
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        pages = {page["id"]: page["html"] for page in manifest["pages"]}
        for name in ("section0/introRenewableEnergy.md", "section6/hydroelectricEnergy.md",
                     "section7/workbook/hydroelectric.md", "references.md"):
            entry = re.search(r'<li id="ref-jrc2025hydro">(.*?)</li>', pages[name], re.S).group(1)
            with self.subTest(page=name):
                for author in ("Letout, Simon", "Mountraki, Aikaterini", "Ince, Ela"):
                    self.assertIn(author, entry)
                for incorrect in ("Sylvain", "Agnieszka", "Ebru"):
                    self.assertNotIn(incorrect, entry)
        for name in ("section8/projectIntro.md", "references.md"):
            entry = re.search(r'<li id="ref-hu2026">(.*?)</li>', pages[name], re.S).group(1)
            with self.subTest(page=name):
                self.assertIn("Hu, Yaan", entry)
                self.assertIn("Sas, Marc", entry)
                self.assertIn("(eds.)", entry)
                self.assertNotIn("hu2026, 2026", pages[name])

    def test_small_wind_guidebook_page_date_is_not_claimed_as_original(self):
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        page = next(p["html"] for p in manifest["pages"] if p["id"] == "section6/windEnergy.md")
        entry = re.search(r'<li id="ref-doeSmallWind">(.*?)</li>', page, re.S).group(1)
        notebook = json.loads((BOOK / "section6/windEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for content in (entry, prose):
            self.assertIn("published 2025-08-27", content)
            self.assertIn("not the guide's first publication", content)

    def test_owid_publication_revision_and_access_dates_render(self):
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        page = next(p["html"] for p in manifest["pages"] if p["id"] == "section0/introRenewableEnergy.md")
        entry = re.search(r'<li id="ref-owid_energy_mix">(.*?)</li>', page, re.S).group(1)
        notebook = json.loads((BOOK / "section0/introRenewableEnergy.ipynb").read_text(encoding="utf-8"))
        prose = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for content in (entry, prose):
            self.assertIn("First published July 2020", content)
            self.assertIn("revised August 2026", content)
            self.assertIn("accessed 19 November 2026", content)

    def test_assessed_citation_labels_and_documentation_notes_render(self):
        manifest = json.loads((ROOT / "EAGE_pythonSandBox/generated/course.json").read_text(encoding="utf-8"))
        pages = {page["id"]: page["html"] for page in manifest["pages"]}
        for page, key, label in (("section6/solarEnergy.md", "foster2010", "Foster et al., 2010"),
                                 ("section6/windEnergy.md", "manwell2009", "Manwell et al., 2009"),
                                 ("section6/geothermalEnergy.md", "white2024closedloop", "White et al., 2024")):
            links = re.findall(r'<a[^>]*anchor=ref-' + key + r'[^>]*>(.*?)</a>', pages[page], re.I)
            with self.subTest(page=page, key=key):
                self.assertTrue(links)
                self.assertTrue(all(link == label for link in links))
        keys = ("numpyDocs", "pandasDocs", "matplotlibDocs", "sqliteDocs", "pvlibDocs", "pythonDocs")
        found = set()
        for page, html in pages.items():
            for key in keys:
                match = re.search(r'<li id="ref-' + key + r'">(.*?)</li>', html, re.S)
                if match:
                    found.add(key)
                    with self.subTest(page=page, key=key):
                        self.assertIn("Documentation consulted 16 September 2026", match.group(1))
                        self.assertIn("year is the consultation year", match.group(1))
        self.assertEqual(found, set(keys))

    def test_intro_explains_lesson_namespaces_and_runtime_files(self):
        pages = json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']
        page = next(page for page in pages if page['id'] == 'courseIntro.md')
        for text in ((BOOK / 'courseIntro.md').read_text(encoding='utf-8'), page['html']):
            with self.subTest(delivery='source' if text.startswith('#') else 'sandbox'):
                self.assertIn('fresh variable namespace', text)
                self.assertIn('returning to the lesson does not restore', text)
                self.assertIn('download any files you want to keep first', text.lower())

    def test_era5_lesson_distinguishes_grid_and_release(self):
        path = 'section5/wind-data.md'
        pages = json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']
        page = next(page for page in pages if page['id'] == path)
        for text in ((BOOK / path).read_text(encoding='utf-8'), page['html']):
            with self.subTest(delivery='source' if text.startswith('#') else 'sandbox'):
                self.assertIn('uncertainty estimate at 0.5°', text)
                self.assertIn('ERA5T', text)
                self.assertIn('file checksum', text)

    def test_xarray_lessons_distinguish_mean_speed_and_mean_vector(self):
        import numpy as np
        import xarray as xr
        u = xr.DataArray([10., -10.], dims='time')
        v = xr.zeros_like(u)
        self.assertEqual(float(np.hypot(u, v).mean()), 10.)
        self.assertEqual(float(np.hypot(u.mean(), v.mean())), 0.)
        for relative in ('section4/dataTypes/xArray.md', 'section5/xarray-extra.md'):
            text = (BOOK / relative).read_text(encoding='utf-8')
            with self.subTest(page=relative):
                self.assertRegex(text.lower(), r'mean speed (?:of )?10 m/s')
                self.assertRegex(text.lower(), r'mean-vector speed (?:of )?0 m/s')
                self.assertIn('77229867', text)

    def test_quiz_instructions_explain_practice_answer_setup(self):
        pages = {page['id']: page for page in json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']}
        for name in ('hydroelectricEnergy', 'solarEnergy', 'windEnergy', 'geothermalEnergy', 'eolicenergy'):
            path = f'section7/renewableEnergyquizzes/{name}.md'
            source = (BOOK / path).read_text(encoding='utf-8')
            for text in (source, pages[path]['html']):
                with self.subTest(page=path):
                    self.assertIn('Include preceding practice answers', text)
                    self.assertIn('loading it does not run it automatically', text)

    def test_quiz_and_assessment_pages_match_fresh_source_rendering(self):
        import importlib.util
        from bs4 import BeautifulSoup
        spec = importlib.util.spec_from_file_location('quiz_prose_builder', ROOT / 'tools/build_sandbox.py')
        module = importlib.util.module_from_spec(spec)
        pages = json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']
        count = 0
        assessment_count = 0
        with mock.patch.object(sys, 'path', [str(ROOT / 'tools'), *sys.path]), tempfile.TemporaryDirectory() as directory:
            spec.loader.exec_module(module)
            with mock.patch.object(module, 'OUT', Path(directory)):
                builder = module.Builder()
                for page in pages:
                    if page['group'] != 'Quizzes' and page['id'] not in module.ASSESSMENT_PAGES:
                        continue
                    fresh = builder.page(page['id'], page['group'])
                    with self.subTest(page=page['id']):
                        self.assertEqual(page['html'], fresh['html'])
                        self.assertEqual(page['workspaces'], fresh['workspaces'])
                    if page['group'] == 'Quizzes':
                        count += len(BeautifulSoup(page['html'], 'html.parser').select('details.solution'))
                    else:
                        assessment_count += len(page['workspaces'])
        self.assertEqual(count, 140)
        self.assertEqual(assessment_count, 60)

    def test_eurostat_lesson_matches_its_generator(self):
        import ast
        import inspect
        import nbformat
        sys.path.insert(0, str(ROOT / 'tools'))
        from download_energy_data import decode_series
        from notebook_links import rewrite_notebook_links
        tree = ast.parse((ROOT / 'tools/build_download_lesson.py').read_text(encoding='utf-8'))
        scope = {'inspect': inspect, 'decode_series': decode_series}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {'intro', 'code', 'tail', 'text'}):
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'generator', 'exec'), scope)
        lesson = BOOK / 'section5/download-to-database.md'
        text = scope['text']
        self.assertEqual(text, lesson.read_text(encoding='utf-8'))
        expected_cells = []
        last = 0
        for match in re.finditer(r'```python\n(.*?)```', text, re.S):
            expected_cells.append(('markdown', rewrite_notebook_links(text[last:match.start()], lesson)))
            for piece in re.split(r'(?m)(?=^# Step \d+:)', match[1].strip()):
                if not piece.strip():
                    continue
                if piece.startswith('# Step '):
                    expected_cells.append(('markdown', '### Python ' + piece.splitlines()[0][2:]))
                expected_cells.append(('code', piece.strip()))
            last = match.end()
        expected_cells.append(('markdown', rewrite_notebook_links(text[last:], lesson)))
        notebook = nbformat.read(lesson.with_suffix('.ipynb'), as_version=4)
        self.assertEqual(expected_cells, [(cell.cell_type, cell.source) for cell in notebook.cells])

    def test_solar_coverage_crosscheck_sources_reach_the_notebook_and_sandbox(self):
        import html
        import nbformat
        generator = (ROOT / 'tools/build_download_lesson.py').read_text(encoding='utf-8')
        lesson = (BOOK / 'section5/download-to-database.md').read_text(encoding='utf-8')
        notebook = nbformat.read(BOOK / 'section5/download-to-database.ipynb', as_version=4)
        notebook_text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
        pages = json.loads((ROOT / 'EAGE_pythonSandBox/generated/course.json').read_text(encoding='utf-8'))['pages']
        page = next(p['html'] for p in pages if p['id'] == 'section5/download-to-database.md')
        for text in (generator, lesson, notebook_text, page):
            with self.subTest(delivery=len(text)):
                text = html.unescape(text)
                self.assertIn('nrg_bal_peh/default/table?lang=en', text)
                self.assertIn('Monthly_electricity_statistics_Documentation_Sep_2026.pdf#page=17', text)
                self.assertIn('geo=AT&time=2023', text)
                self.assertIn('geo=NO&time=2023', text)
                self.assertIn('does not establish why', text)

    def test_notebook_bibliographies_follow_current_source_records(self):
        import importlib.util
        import nbformat
        spec = importlib.util.spec_from_file_location('course_builder_wade_source', ROOT / 'tools/build_sandbox.py')
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, 'path', [str(ROOT / 'tools'), *sys.path]):
            spec.loader.exec_module(module)
            bib = module.Builder().bib
        count = 0
        wade_count = 0
        for path in BOOK.rglob('*.ipynb'):
            if '_build' in path.parts:
                continue
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type != 'markdown':
                    continue
                for key, entry in re.findall(r'\*\*([^*]+):\*\*([^\n]*)', cell.source):
                    source = bib.get(key, {})
                    url = source.get('url') or (f"https://doi.org/{source['doi']}" if source.get('doi') else None)
                    if not url:
                        continue
                    with self.subTest(notebook=path, reference=key):
                        self.assertIn(url, entry)
                        for field in ('title', 'year'):
                            value = re.sub(r'[{}]', '', source.get(field, ''))
                            if value:
                                self.assertIn(value, entry)
                        author = re.sub(r'[{}]', '', source.get('author', '') or source.get('editor', ''))
                        if author:
                            self.assertIn(author, entry)
                    count += 1
                    wade_count += key == 'wade2003'
        self.assertGreaterEqual(count, 77)
        self.assertGreaterEqual(wade_count, 9)

    def test_assessed_bibliography_entries_have_direct_source_fields(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("course_builder_bib_sources", ROOT / "tools/build_sandbox.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(module)
            builder = module.Builder()
        for key in ("grant2011", "jica2011", "ifc2015", "jrc2025hydro", "hu2026", "wade2003", "foster2010", "wagner2009", "manwell2009"):
            with self.subTest(key=key):
                self.assertTrue(builder.bib[key].get("url") or builder.bib[key].get("doi"))

    def test_generated_page_renders_undated_citation_and_reference(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("course_builder_undated", ROOT / "tools/build_sandbox.py")
        builder_module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(builder_module)
            builder = builder_module.Builder()
        builder.bib["undatedFixture"] = {"author": "{Example Agency}", "title": "Undated source", "url": "https://example.org/source"}
        with mock.patch.object(builder_module, "source_markdown", return_value="# Fixture\n\nSee [@undatedFixture]."):
            page = builder.page("section0/introRenewableEnergy.md", "Fixture")
        self.assertIn("Example Agency, n.d.", page["html"])
        self.assertIn("Example Agency. n.d. Undated source", page["html"])
        self.assertNotIn("Agency. .", page["html"])
        self.assertIn('href="https://example.org/source"', page["html"])

    def test_reference_fields_preserve_initials_and_undated_punctuation(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from build_sandbox import reference_label
        self.assertEqual(reference_label(["Grant, Malcolm A.", "2011", "Geothermal Reservoir Engineering"]),
                         "Grant, Malcolm A. 2011. Geothermal Reservoir Engineering")
        self.assertEqual(reference_label(["U.S. Department of Energy", "n.d.", "Source title"]),
                         "U.S. Department of Energy. n.d. Source title")
        self.assertEqual(reference_label(["{Agency}", "", "A title"]), "Agency. A title")

    def test_course_digest_tracks_pages_and_generated_files(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("course_builder_digest", ROOT / "tools/build_sandbox.py")
        builder = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
            spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "asset.csv").write_text("first", encoding="utf-8")
            pages = [{"id": "lesson", "html": "first"}]
            baseline = builder.course_digest(pages, folder)
            self.assertEqual(baseline, builder.course_digest(pages, folder))
            (folder / "course.json").write_text("ignored self-reference", encoding="utf-8")
            self.assertEqual(baseline, builder.course_digest(pages, folder))
            (folder / "asset.csv").write_text("revised", encoding="utf-8")
            self.assertNotEqual(baseline, builder.course_digest(pages, folder))
            (folder / "asset.csv").write_text("first", encoding="utf-8")
            self.assertNotEqual(baseline, builder.course_digest([{"id": "lesson", "html": "revised"}], folder))

    def test_generated_assets_match_source_and_manifest(self):
        import hashlib
        generated = ROOT / "EAGE_pythonSandBox/generated"
        manifest = json.loads((generated / "course.json").read_text(encoding="utf-8"))
        for asset in manifest["assets"]:
            source, published = BOOK / asset["path"], generated / "assets" / asset["path"]
            with self.subTest(path=asset["path"]):
                self.assertEqual(source.read_bytes(), published.read_bytes())
                self.assertEqual(hashlib.sha256(published.read_bytes()).hexdigest(), asset["sha256"])

    def test_media_staging_writes_only_to_requested_output(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from course_media import attach_media
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory) / "sandbox"
            source = sandbox / "media/en/test/clip.mp3"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"synthetic test audio")
            (sandbox / "media.json").write_text(json.dumps({"lesson.md": {"audio": {"en": {"intro": "en/test/clip.mp3"}}}}), encoding="utf-8")
            output = Path(directory) / "staged"
            pages = [{"id": "lesson.md", "html": "<h1>Lesson</h1>"}]
            attach_media(pages, sandbox, output_dir=output)
            self.assertEqual((output / "media/en/test/clip.mp3").read_bytes(), source.read_bytes())
            self.assertFalse((sandbox / "generated").exists())
            self.assertEqual(pages[0]["media"]["audio"]["en"]["intro"], "generated/media/en/test/clip.mp3")

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "EAGE_pythonSandBox"))
        from source_inspection import inspect_source
        cls.inspect = staticmethod(inspect_source)

    def test_multiple_imports_are_found_without_executing(self):
        result = self.inspect("import os, folium, plotly, openpyxl\nfrom seaborn import color_palette")
        self.assertEqual(result["imports"], ["folium", "openpyxl", "os", "plotly", "seaborn"])
        self.assertFalse(result["magic"])

    def test_magic_is_distinct_from_multiline_strings_and_modulo(self):
        self.assertFalse(self.inspect('text = """line\n! text\n% text\n"""')["magic"])
        self.assertFalse(self.inspect("result = (10\n % 3)")["magic"])
        self.assertTrue(self.inspect("%matplotlib inline")["magic"])
        self.assertTrue(self.inspect("!pip install folium")["magic"])


class SqlSchemaTests(unittest.TestCase):
    def test_foreign_keys_must_be_enabled_before_a_transaction(self):
        lesson = (BOOK / "section5/databases.md").read_text(encoding="utf-8")
        self.assertIn("before beginning a transaction or savepoint", lesson)
        self.assertIn("verify `PRAGMA foreign_keys` returns 1", lesson)
        db = sqlite3.connect(":memory:")
        try:
            db.execute("PRAGMA foreign_keys = OFF")
            db.executescript("CREATE TABLE parent(id INTEGER PRIMARY KEY);"
                             "CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));")
            db.execute("BEGIN")
            db.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(db.execute("PRAGMA foreign_keys").fetchone()[0], 0)
            db.execute("INSERT INTO child VALUES (99)")
            db.rollback()
            db.execute("PRAGMA foreign_keys = ON")
            self.assertEqual(db.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("INSERT INTO child VALUES (99)")
        finally:
            db.close()

    def test_historical_mix_displays_reporting_entities_and_flags(self):
        import ast
        notebook = json.loads((BOOK / "section5/SQL_Pandas.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"] if "mix_query =" in "".join(c["source"]))
        query = re.search(r'mix_query = """(.*?)"""', cell, re.S).group(1)
        source_cell = next("".join(c["source"]) for c in notebook["cells"] if "SOURCE_FILES =" in "".join(c["source"]))
        source_assignment = next(n for n in ast.parse(source_cell).body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == "SOURCE_FILES" for t in n.targets))
        source_map = ast.literal_eval(source_assignment.value)
        self.assertEqual(source_map["historical_nonrenewable_unverified"][2], 0)
        time_series = json.loads((BOOK / "section4/timeSeriesEnergyConsumption.ipynb").read_text(encoding="utf-8"))
        selection_cell = next("".join(c["source"]) for c in time_series["cells"] if "SELECTED_SOURCES =" in "".join(c["source"]))
        selection = next(n for n in ast.parse(selection_cell).body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "SELECTED_SOURCES" for t in n.targets))
        self.assertNotIn("historical_nonrenewable_unverified", ast.literal_eval(selection.value))
        with sqlite3.connect(":memory:") as connection:
            connection.executescript("""
                CREATE TABLE source (source TEXT PRIMARY KEY, in_selected_group INTEGER);
                CREATE TABLE generation (country TEXT, month TEXT, source TEXT, generation_gwh REAL, status_flag TEXT);
                INSERT INTO source VALUES ('wind', 1), ('hydro', 1), ('other', 0),
                    ('historical_nonrenewable_unverified', 0), ('coal_and_manufactured_gases', 0),
                    ('nuclear_and_other_fuels_nec', 0);
                INSERT INTO generation VALUES
                    ('Austria', '2024-05', 'wind', 10, 'e'),
                    ('Norway', '2024-05', 'hydro', 20, ''),
                    ('Germany', '2024-05', 'wind', NULL, 'p'),
                    ('Spain', '2024-05', 'other', 30, 'p'),
                    ('Germany', '2024-05', 'historical_nonrenewable_unverified', 10, ''),
                    ('Germany', '2024-05', 'coal_and_manufactured_gases', 100, ''),
                    ('Germany', '2024-05', 'nuclear_and_other_fuels_nec', 8, '');
            """)
            connection.row_factory = sqlite3.Row
            row = connection.execute(query).fetchone()
            self.assertEqual(row["selected_gwh"], 30)
            self.assertEqual(row["historical_nonrenewable_unverified_gwh"], 10)
            self.assertEqual(row["nuclear_and_other_fuels_nec_gwh"], 8)
            self.assertEqual(row["nuclear_and_other_reported_rows"], 1)
            self.assertEqual(row["selected_reported_rows"], 2)
            self.assertEqual(row["selected_flagged_rows"], 1)
            self.assertEqual(set(row["selected_countries"].split(",")), {"Austria", "Norway"})
            self.assertEqual(set(row["selected_sources"].split(",")), {"wind", "hydro"})

    def test_grouped_coverage_shows_missing_measurements(self):
        text = (BOOK / "section5/sql-basics.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```sql\n(.*?)\n```", text, re.S)
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(blocks[0])
            connection.executemany("INSERT INTO observations(source,year,generation_gwh) VALUES ('Hydro', ?, NULL)", [(2022,), (2023,)])
            query = blocks[2].strip().split(";")[-2].strip()
            rows = {row[0]: row[1:] for row in connection.execute(query)}
            self.assertEqual(rows["Wind"], (2, 1, 20.0))
            self.assertEqual(rows["Solar"], (2, 2, 13.75))
            self.assertEqual(rows["Hydro"], (2, 0, None))
        finally:
            connection.close()

    def test_domain_specific_nonnegative_example_rejects_text(self):
        text = (BOOK / "section5/SQL.md").read_text(encoding="utf-8")
        block = next(b for b in re.findall(r"```sql\n(.*?)\n```", text, re.S) if "typeof(energy_generated)" in b)
        with sqlite3.connect(":memory:") as db:
            db.executescript(block)
            db.execute("INSERT INTO energy_data(energy_generated) VALUES (?)", (None,))
            db.execute("INSERT INTO energy_data(energy_generated) VALUES (?)", (2.5,))
            for value in (-1, "unknown"):
                with self.subTest(value=value), self.assertRaises(sqlite3.IntegrityError):
                    db.execute("INSERT INTO energy_data(energy_generated) VALUES (?)", (value,))
            db.execute("INSERT INTO energy_data(energy_generated) VALUES (?)", (float("inf"),))
            self.assertTrue(math.isinf(db.execute("SELECT MAX(energy_generated) FROM energy_data").fetchone()[0]))
        self.assertIn("net-generation series may legitimately be negative", text)
        self.assertIn("numeric type alone does not rule out infinity", text)

    def test_numeric_measurement_and_nonnull_text_key(self):
        text = (BOOK / "section5/sql-basics.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```sql\n(.*?)\n```", text, re.S)
        create = blocks[0].split("INSERT INTO", 1)[0]
        sources = re.search(r"CREATE TABLE sources \(.*?\);", text).group()
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(create + sources)
            connection.execute("INSERT INTO observations(source,year,generation_gwh) VALUES ('Pumped storage',2024,?)", (-5851.862,))
            self.assertEqual(connection.execute("SELECT generation_gwh FROM observations WHERE source='Pumped storage'").fetchone()[0], -5851.862)
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO observations(source,year,generation_gwh) VALUES ('Hydro',2023,'unknown')")
            for bad_year in ("unknown", 2023.5):
                with self.subTest(year=bad_year), self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("INSERT INTO observations(source,year,generation_gwh) VALUES ('Hydro',?,1)", (bad_year,))
            connection.execute("INSERT INTO observations(source,year,generation_gwh) VALUES ('Hydro',?,1)", ("2023",))
            self.assertEqual(connection.execute("SELECT typeof(year) FROM observations WHERE source='Hydro'").fetchone()[0], "integer")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO sources VALUES (NULL, 'x')")
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()

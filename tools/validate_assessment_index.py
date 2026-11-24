"""Check that the assessment evidence index covers the remediation tracker."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
tracker = (ROOT / "COURSE_REMEDIATION_TRACKER.md").read_text(encoding="utf-8")
assessment = (ROOT / "COURSE_ASSESSMENT_2026-11-18.md").read_text(encoding="utf-8")
index = (ROOT / "COURSE_FINDING_EVIDENCE_INDEX.md").read_text(encoding="utf-8")
items = re.findall(r"^- \[([ x])\] (.*?) — \[assessment line \d+\]\(([^)]+)\)", tracker, re.M)
rows = re.findall(r"^\| (\d+) \| \[(.*?)\]\(([^)]+)\) \| (.*?) \|$", index, re.M)
if not items or len(rows) != len(items):
    raise SystemExit(f"Expected {len(items)} index rows; found {len(rows)}")

def slug(text):
    text = re.sub(r"<[^>]+>", "", text).lower()
    return "".join(c for c in text if c.isalnum() or c in " -_").replace(" ", "-")

headings = {slug(re.sub(r"^#+\s+", "", line)) for line in assessment.splitlines()
            if re.match(r"^#{1,6}\s+", line)}
pending = 0
for number, ((status, title, target), (row_number, row_title, row_target, evidence)) in enumerate(zip(items, rows), 1):
    if (str(number), title, target) != (row_number, row_title, row_target):
        raise SystemExit(f"Index row {number} does not match the tracker")
    prefix = "COURSE_ASSESSMENT_2026-11-18.md#"
    if not target.startswith(prefix) or target[len(prefix):] not in headings:
        raise SystemExit(f"Index row {number} has no matching assessment heading: {target}")
    if evidence.startswith("Not yet indexed;"):
        pending += 1
    elif not evidence.startswith("Evidence indexed — "):
        raise SystemExit(f"Index row {number} has an unknown evidence status")
unchecked = sum(status != "x" for status, _, _ in items)
print(f"Assessment index: {len(rows)} matching findings, {pending} not yet indexed, {unchecked} unchecked in tracker.")

# The assessment also identifies actionable issues outside its 185-item list.
# Check that the tracker retains every one and its original target, without
# interpreting a checked box as proof that the underlying fix is sufficient.
additional = assessment.split("### Additional issues documented in the checks section", 1)[1].split("\n## Findings", 1)[0]
original_extra = re.findall(r"^- \[(.*?)\]\(#(.*?)\)", additional, re.M)
tracked_section = tracker.split("## Additional findings from the assessment", 1)[1].split(
    "## Additional assessment recommendations", 1)[0]
tracked_extra = re.findall(r"^- \[([ x])\] (.*?) — \[assessment\]\(COURSE_ASSESSMENT_2026-11-18\.md#(.*?)\)",
                           tracked_section, re.M)
if [(title, anchor) for _, title, anchor in tracked_extra] != original_extra:
    raise SystemExit("Additional finding titles, order or targets do not match the assessment")
for _, title, anchor in tracked_extra:
    if anchor not in headings:
        raise SystemExit(f"Additional finding has no matching assessment heading: {title}")
extra_pending = sum(status != "x" for status, _, _ in tracked_extra)
print(f"Additional findings: {len(tracked_extra)} matching tracker entries, {extra_pending} unchecked.")

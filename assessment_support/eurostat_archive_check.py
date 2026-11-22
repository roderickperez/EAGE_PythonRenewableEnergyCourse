"""Reproduce the 2024 Eurostat workbook and archived-series comparison.

Run from the repository root with Python and openpyxl installed:
    python assessment_support/eurostat_archive_check.py
Downloads read-only archived files; does not modify course data.
"""
import collections
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "EAGE_PythonRenewableEnergyCourse/data/section4/euroStat"
BASE = "https://git.nomics.world/api/v4/projects/198/repository/files/"
REVISION = "373e3649"
EXPECTED_SERIES_SHA256 = "ed29874b6338333501e7dd637495c7039b2e437b5f5b57cfed883235502fc6dc"


def download(filename):
    url = BASE + "nrg_cb_pem%2F" + filename + "/raw?ref=" + REVISION
    with urlopen(url, timeout=60) as response:
        return response.read()


def check_july_revision():
    """Compare the 2024 conversion with archived source and category sums."""
    ref = "6c03e4a30952f80e7ab2521c7ec7e8a0f19835dd"
    url = BASE + "nrg_cb_pem%2Fseries.jsonl/raw?ref=" + ref
    with urlopen(url, timeout=60) as response:
        converted = response.read()
    expected = "ce8a5f1c1707e050cf159ad5feff40aaceefdc535a603e8ec49736ab2d116862"
    if hashlib.sha256(converted).hexdigest() != expected:
        raise ValueError("Unexpected July converted-series hash")
    rows = {}
    for line in converted.splitlines():
        record = json.loads(line)
        freq, code, unit, geo = record["dimensions"]
        if freq == "M" and unit == "GWH":
            for obs in record["observations"][1:]:
                if len(obs) > 1 and isinstance(obs[1], (int, float)):
                    rows[(geo, obs[0], code)] = obs[1]

    source_url = ("https://git.nomics.world/api/v4/projects/197/repository/files/"
                  "data%2Fnrg_cb_pem%2Fnrg_cb_pem.sdmx.xml/raw?ref=7a291cd5")
    with urlopen(source_url, timeout=60) as response:
        xml_bytes = response.read()
    expected = "c3854c5bf713d07830682d6b5303a442a0463017c5f0176d97fc99bc9431c102"
    if hashlib.sha256(xml_bytes).hexdigest() != expected:
        raise ValueError("Unexpected July source XML hash")
    ns = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}
    compared = matches = 0
    for series in ET.fromstring(xml_bytes).findall(".//g:Series", ns):
        key = {v.get("id"): v.get("value") for v in series.findall("g:SeriesKey/g:Value", ns)}
        if key.get("siec") != "CF_NR" or key.get("freq") != "M" or key.get("unit") != "GWH":
            continue
        for obs in series.findall("g:Obs", ns):
            month = obs.find("g:ObsDimension", ns).get("value")
            value = float(obs.find("g:ObsValue", ns).get("value"))
            other = rows.get((key["geo"], month, "CF_NR"))
            compared += 1
            matches += other is not None and math.isclose(value, other, rel_tol=0, abs_tol=1e-9)
    print(f"July CF_NR source/conversion matches: {matches}/{compared}")
    categories = ("CF", "CF_NR", "CF_R", "C0000", "G3000", "O4000XBIO")
    complete = close = close_2024 = complete_2024 = 0
    for geo, month in {(geo, month) for geo, month, _ in rows}:
        if not all((geo, month, code) in rows for code in categories):
            continue
        residual = rows[(geo, month, "CF")] - sum(rows[(geo, month, code)] for code in categories[1:])
        complete += 1
        close += abs(residual) < .01
        if month.startswith("2024-"):
            complete_2024 += 1
            close_2024 += abs(residual) < .01
    print(f"July complete category sums within 0.01 GWh: {close}/{complete}")
    print(f"  of which 2024: {close_2024}/{complete_2024}")


def main():
    metadata = json.loads(download("dataset.json"))
    raw_series = download("series.jsonl")
    digest = hashlib.sha256(raw_series).hexdigest()
    if digest != EXPECTED_SERIES_SHA256:
        raise ValueError(f"Unexpected archived series hash: {digest}")
    geo_names = metadata["dimensions_values_labels"]["geo"]
    code_by_label = collections.defaultdict(list)
    for code, label in metadata["dimensions_values_labels"]["siec"].items():
        code_by_label[label].append(code)
    series = collections.defaultdict(dict)
    for line in raw_series.splitlines():
        record = json.loads(line)
        freq, code, unit, geo = record["dimensions"]
        if freq != "M" or unit != "GWH":
            continue
        for obs in record["observations"][1:]:
            if len(obs) > 1 and isinstance(obs[1], (int, float)):
                series[code][(geo_names.get(geo), obs[0])] = obs[1]

    totals = collections.Counter()
    flags = collections.Counter()
    files = sorted(DATA.glob("*.xlsx"))
    if len(files) != 11:
        raise ValueError(f"Expected 11 workbooks; found {len(files)}")
    for path in files:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook[workbook.sheetnames[-1]]
            rows = list(sheet.values)
            label = rows[5][2]
            codes = code_by_label[label]
            if len(codes) != 1:
                raise ValueError(f"Expected one archived code for {label!r}: {codes}")
            code = codes[0]
            compared = matches = 0
            for row in rows[10:]:
                if row[0] is None:
                    break
                if not isinstance(row[0], str):
                    continue
                for i in range(1, len(rows[8]) - 1, 2):
                    value, flag = row[i], str(row[i + 1] or "").strip()
                    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                    # The lesson omits EU aggregate rows, but the archive comparison includes them.
                    if not row[0].startswith("European Union"):
                        flags[("numeric" if numeric else "missing", flag)] += 1
                    key = (row[0], rows[8][i])
                    if numeric and key in series[code]:
                        compared += 1
                        matches += math.isclose(value, series[code][key], rel_tol=0, abs_tol=1e-9)
            totals.update(compared=compared, matches=matches)
            print(f"{sheet.title:16} {code:10} {matches:5}/{compared:5}")
        finally:
            workbook.close()
    print(f"Total matches: {totals['matches']}/{totals['compared']}")
    print("Flags after excluding EU aggregate rows:")
    for (kind, flag), count in sorted(flags.items()):
        print(f"  {kind:7} {flag or '(unflagged)':12} {count}")
    check_july_revision()


if __name__ == "__main__":
    main()

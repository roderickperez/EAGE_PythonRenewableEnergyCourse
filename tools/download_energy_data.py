"""Download a bounded Eurostat series, preserve flags, validate, and save provenance.

Run from the repository root. Public endpoint; no account or API key required.
The source is monthly net electricity generation, not energy consumption.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_cb_pem"


def query_url(geo="AT", year=2023, source="RA300"):
    return BASE + "?" + urlencode({"lang": "en", "freq": "M", "geo": geo,
        "siec": source, "unit": "GWH", "sinceTimePeriod": f"{year}-01",
        "untilTimePeriod": f"{year}-12"})


def decode_series(payload):
    """Decode JSON-stat positional values without assuming time is the last axis."""
    if payload.get("class") != "dataset" or "error" in payload:
        raise ValueError("Expected a Eurostat JSON-stat dataset, not an error/page")
    ids, sizes = payload["id"], payload["size"]
    required = {"freq", "siec", "unit", "geo", "time"}
    if len(ids) != len(required) or len(ids) != len(sizes) or set(ids) != required:
        raise ValueError(f"Unexpected dimensions: {ids}")
    if any(type(size) not in (int, float) or not math.isfinite(size) or size < 0 or int(size) != size for size in sizes):
        raise ValueError("Dimension sizes must be non-negative integers")
    sizes = [int(size) for size in sizes]
    axes = []
    for name, size in zip(ids, sizes):
        category = payload["dimension"][name]["category"]
        index = category.get("index")
        if index is None:
            labels = category.get("label", {})
            if size != 1 or len(labels) != 1:
                raise ValueError(f"Missing category index: {name}")
            index = list(labels)
        if not isinstance(index, (dict, list)):
            raise ValueError(f"Invalid category index: {name}")
        if isinstance(index, dict) and (any(type(position) not in (int, float) or not math.isfinite(position)
                                            or int(position) != position for position in index.values())
                                        or sorted(index.values()) != list(range(size))):
            raise ValueError(f"Invalid category positions: {name}")
        if isinstance(index, list) and (any(not isinstance(label, str) for label in index)
                                        or len(set(index)) != len(index)):
            raise ValueError(f"Invalid category labels: {name}")
        labels = index if isinstance(index, list) else [k for k, _ in sorted(index.items(), key=lambda x: x[1])]
        if len(labels) != size:
            raise ValueError(f"Dimension length mismatch: {name}")
        axes.append(labels)
    total_positions = math.prod(sizes)
    for field in ("value", "status"):
        container = payload.get(field, {})
        if isinstance(container, list):
            if len(container) != total_positions:
                raise ValueError(f"Dense {field} length does not match dimensions")
        elif field == "status" and isinstance(container, str):
            pass  # A dataset-wide status applies to every observation.
        elif isinstance(container, dict):
            if any(not isinstance(k, str) or not k.isascii() or not k.isdecimal()
                   or str(int(k)) != k or int(k) >= total_positions for k in container):
                raise ValueError(f"Invalid sparse {field} position")
        else:
            raise ValueError(f"Invalid {field} container")
    def get_at(container, i, default=None):
        if isinstance(container, str):
            return container
        return container.get(str(i), default) if isinstance(container, dict) else (container[i] if i < len(container) else default)
    rows = []
    for i, labels in enumerate(itertools.product(*axes)):
        row = dict(zip(ids, labels))
        value = get_at(payload.get("value", {}), i)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Non-finite or non-numeric generation")
        status = get_at(payload.get("status", {}), i, "")
        if status is not None and not isinstance(status, str):
            raise ValueError("Observation status must be text or null")
        rows.append({"geo": row["geo"], "source": row["siec"], "month": row["time"],
            "unit": row["unit"], "freq": row["freq"], "generation_gwh": value,
            "status_flag": status or ""})
    if any(r["unit"] != "GWH" or r["freq"] != "M" for r in rows):
        raise ValueError("Expected monthly GWh")
    keys = [(r["geo"], r["source"], r["month"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate country/source/month")
    return sorted(rows, key=lambda r: (r["geo"], r["source"], r["month"]))


def download(output, geo="AT", year=2023, source="RA300"):
    url = query_url(geo, year, source)
    request = Request(url, headers={"User-Agent": "EAGE-course-data-tutorial/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Download returned HTTP {response.status}")
        raw = response.read(5_000_001)
        if len(raw) > 5_000_000:
            raise ValueError("Unexpectedly large download; narrow the query")
    data = json.loads(raw)
    rows = decode_series(data)
    if not rows:
        raise ValueError("No rows returned for this selection; check country, source and year")
    expected_months = [f"{year}-{m:02}" for m in range(1, 13)]
    if (any(r["geo"] != geo or r["source"] != source for r in rows)
            or any(r["month"] not in expected_months for r in rows)):
        raise ValueError("Response does not match requested country/source/months")
    returned_months = [r["month"] for r in rows]
    by_month = {r["month"]: r for r in rows}
    rows = [by_month.get(month, {"geo": geo, "source": source, "month": month,
            "unit": "GWH", "freq": "M", "generation_gwh": None, "status_flag": ""})
            for month in expected_months]
    prefix = Path(output)
    completed = prefix.parent / f"{prefix.name}_{uuid.uuid4().hex}"
    staging = completed.with_name(completed.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "raw.json").write_bytes(raw)
    with (staging / "generation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = {"provider": "Eurostat", "dataset": "nrg_cb_pem", "url": url,
        "title": data["label"], "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider_updated": data.get("updated"), "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows), "returned_months": returned_months, "geo": geo, "source": source, "unit": "GWH", "frequency": "M",
        "source_label": data["dimension"]["siec"]["category"]["label"].get(source),
        "status_labels": data.get("extension", {}).get("status", {}),
        "reuse": "https://ec.europa.eu/eurostat/about-us/policies/copyright"}
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    staging.rename(completed)
    observed = [r["generation_gwh"] for r in rows if r["generation_gwh"] is not None]
    print(f"Saved completed retrieval to {completed}")
    print(f"Provider returned {len(returned_months)}/12 month categories; {len(observed)}/12 values reported")
    print(f"Observed-period sum: {sum(observed):.3f} GWh" if observed else "Observed-period sum: unavailable")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output directory prefix; each run creates a new uniquely named folder")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--source", default="RA300")
    args = parser.parse_args()
    output = args.output or Path("outputs") / f"eurostat_{args.geo}_{args.source}_{args.year}"
    download(output, args.geo, args.year, args.source)

"""Publish a validated browser course build using a restricted Supabase account.

Credentials are read from environment variables and are never printed. This script
only accepts the student browser output, not a course repository or instructor files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

COURSES = {"python-renewable", "eage-python-course", "fault-segmentation", "energy-decision-ai"}
ROOT_FILES = {
    "index.html", "app.js", "styles.css", "python-worker.js", "eagelogo.jpg", "course-session-refresh.js",
    "flag-es.svg", "flag-uk.svg", "explorer-python.js", "explorers.css",
    "explorers.html", "explorers.js", "decision-lab.css", "decision-lab.html",
    "decision-lab.js", "guide.html",
}
FORBIDDEN = re.compile(r"(^|/)(instructor_private|private|\.git|__pycache__)(/|$)|test_answer_key|final_project_solution|projectsolution|\.bak$", re.I)
URL = "https://axmsjmuhmsjqgnoinsza.supabase.co"
PUBLIC_KEY = "sb_publishable_Bqtvb7g2QlWCi-ZZqnPK7w_CrPu0bxR"


def build_inventory(root):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The course build must be a directory.")
    entries = {}
    for file in root.rglob("*"):
        if file.is_symlink():
            raise ValueError("Symlinks are not permitted in a published build.")
        if not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        if FORBIDDEN.search(relative):
            continue
        if relative not in ROOT_FILES and not relative.startswith(("generated/", "vendor/", "jupyterlite/")):
            continue
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        entries[relative] = (file, digest)
    if "index.html" not in entries or "generated/course.json" not in entries:
        raise ValueError("Build must contain index.html and generated/course.json.")
    return entries


def request(method, path, *, token=None, data=None, content_type=None, upsert=False, missing_ok=False):
    headers = {"apikey": PUBLIC_KEY}
    if token:
        headers["Authorization"] = "Bearer " + token
    if content_type:
        headers["Content-Type"] = content_type
    if upsert:
        headers["x-upsert"] = "true"
    req = urllib.request.Request(URL + path, data=data, headers=headers, method=method)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            if missing_ok and exc.code == 400:
                try:
                    error = json.load(exc)
                except (ValueError, OSError):
                    error = {}
                if str(error.get("statusCode")) == "404":
                    return None
            if exc.code not in {429, 500, 502, 503, 504, 520, 522, 524} or attempt == 2:
                raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(f"{method} {path} could not contact course storage.") from None
        time.sleep(2 ** attempt)


def object_path(course, file):
    return "/storage/v1/object/private-courses/" + urllib.parse.quote(course + "/" + file, safe="/")


def publish(course, root, email, password):
    if course not in COURSES:
        raise ValueError("Unknown course.")
    inventory = build_inventory(root)
    auth = json.loads(request("POST", "/auth/v1/token?grant_type=password",
        data=json.dumps({"email": email, "password": password}).encode(),
        content_type="application/json"))
    token = auth.get("access_token")
    if not token:
        raise RuntimeError("Publisher authentication failed.")
    pointer = object_path(course, "current.json")
    read_pointer = pointer.replace("/object/", "/object/authenticated/", 1)
    current = request("GET", f"{read_pointer}?cacheNonce={time.time_ns()}", token=token, missing_ok=True)
    previous = json.loads(current) if current is not None else {}
    if current is not None and (not isinstance(previous, dict) or previous.get("version") != 1 or not isinstance(previous.get("files"), dict)):
        raise ValueError("Unexpected course manifest format.")
    existing = set(previous.get("files", {}).values())

    # Course objects are immutable and named by their content hash. Publish new
    # objects first, then switch the manifest once the full release is available.
    manifest = {name: digest for name, (_, digest) in sorted(inventory.items())}
    if previous.get("files") == manifest:
        print("Verified existing release for", course, "with", len(inventory), "files.")
        return
    uploads = {}
    for name, digest in manifest.items():
        if digest not in existing:
            uploads.setdefault(digest, name)

    def upload(entry):
        digest, name = entry
        file, _ = inventory[name]
        request("POST", object_path(course, "objects/" + digest), token=token,
                data=file.read_bytes(),
                content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                upsert=True)
        return name

    # The vendored JupyterLite distribution adds hundreds of independent assets.
    with ThreadPoolExecutor(max_workers=6) as workers:
        for index, name in enumerate(workers.map(upload, uploads.items()), 1):
            print(f"Uploaded {index}/{len(uploads)}: {name}", flush=True)
    payload = json.dumps({"version": 1, "files": manifest}, separators=(",", ":")).encode()
    request("POST", pointer, token=token, data=payload,
            content_type="application/json", upsert=True)
    # The previous pointer may still be cached immediately after the upsert.
    published = json.loads(request("GET", f"{read_pointer}?cacheNonce={time.time_ns()}", token=token))
    if published != {"version": 1, "files": manifest}:
        raise RuntimeError("Published course manifest did not match the build.")
    print("Published and verified", course, "with", len(inventory), "files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True, choices=sorted(COURSES))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    email = os.environ.get("COURSE_PUBLISHER_EMAIL", "")
    password = os.environ.get("COURSE_PUBLISHER_PASSWORD", "")
    if not email or not password:
        sys.exit("Set the publisher credentials as GitHub Actions secrets.")
    try:
        publish(args.course, args.root, email, password)
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()

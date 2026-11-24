"""Make course notebook links useful when the notebook is downloaded alone."""
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import re
import yaml

BOOK = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse"
SANDBOX = "https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/"
SOURCE = "https://github.com/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/"

def course_pages():
    result = set()
    def visit(items, group="Welcome"):
        for item in items:
            if "file" in item and group != "Data Processing and Exploration Data Analysis":
                result.add(item["file"])
            visit(item.get("children", []), item.get("title", group))
    visit(yaml.safe_load((BOOK / "myst.yml").read_text(encoding="utf-8"))["project"]["toc"])
    return result

def rewrite_notebook_links(text, source_path):
    source_path = Path(source_path).resolve()
    pages = course_pages()
    def replace(match):
        prefix, target = match.group(1), match.group(2)
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or target.startswith("//"):
            return match.group(0)
        # These controls exist in the online geothermal lesson, not in the notebook.
        if not parts.path:
            if source_path.name == "geothermalEnergy.md" and parts.fragment.startswith("geo-"):
                url = SANDBOX + "#lesson=" + quote("section6/geothermalEnergy.md", safe="") + "&anchor=" + quote(parts.fragment, safe="")
                return prefix + url + ")"
            return match.group(0)
        path = (source_path.parent / unquote(parts.path)).resolve()
        if not path.is_relative_to(BOOK):
            return match.group(0)
        relative = path.relative_to(BOOK).as_posix()
        if path.suffix.lower() == ".md" and path.is_file():
            if relative in pages:
                url = SANDBOX + "#lesson=" + quote(relative, safe="")
                if parts.fragment: url += "&anchor=" + quote(parts.fragment, safe="")
            else:
                url = SOURCE + quote(relative, safe="/")
                if parts.fragment: url += "#" + quote(parts.fragment, safe="-")
        elif path.suffix.lower() in {".csv", ".json", ".png", ".jpg", ".jpeg", ".svg", ".ipynb"} and path.is_file():
            url = SANDBOX + "generated/assets/" + quote(relative, safe="/")
            if parts.fragment: url += "#" + quote(parts.fragment, safe="-")
        else:
            return match.group(0)
        return prefix + url + ")"
    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)\)", replace, text)

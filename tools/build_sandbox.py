"""Build the participant sandbox from the book's public TOC, never instructor files.

The book remains the single editable curriculum. Generated HTML is deliberately
static: practice disclosures use native details, and every Python cell is editable.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import bibtexparser
import nbformat
import yaml
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'EAGE_PythonRenewableEnergyCourse'
SANDBOX = ROOT / 'EAGE_pythonSandBox'
OUT = SANDBOX / 'generated'
ASSESSMENT = 'section7/renewableEnergyTest.md'
PROJECT = 'section8/projectIntro.md'
ALLOWED_ASSETS = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.pdf', '.csv',
                  '.xlsx', '.db', '.nc', '.geojson', '.ipynb', '.bib'}


def slug(text):
    return re.sub(r'[-\s]+', '-', re.sub(r'[^\w\s-]', '', text.lower()).strip())


def source_markdown(path):
    if path.suffix == '.ipynb':
        nb = nbformat.read(path, 4)
        return '\n\n'.join(c.source if c.cell_type == 'markdown' else
                           '```python\n' + c.source + '\n```' for c in nb.cells
                           if c.cell_type in {'markdown', 'code'})
    return re.sub(r'\A---\s*\n.*?\n---\s*\n', '', path.read_text(), count=1, flags=re.S)


def toc_pages():
    result = []
    def visit(items, group='Welcome'):
        for item in items:
            if 'file' in item:
                result.append((item['file'], group))
            visit(item.get('children', []), item.get('title', group))
    visit(yaml.safe_load((BOOK / 'myst.yml').read_text())['project']['toc'])
    # Referenced dataset definitions should be readable inside the sandbox too.
    result.append(('section8/data/README.md', 'Course Final Project'))
    result.append(('data/examples/README.md', 'Data Processing and Exploration Data Analysis'))
    return result


class Builder:
    def __init__(self):
        self.pages = toc_pages()
        self.page_ids = {p for p, _ in self.pages}
        self.assets = {}
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        self.bib = bibtexparser.loads((BOOK / 'references.bib').read_text(), parser=parser).entries_dict
        self.md = MarkdownIt('commonmark', {'html': True}).enable('table').use(dollarmath_plugin)
        self.md.add_render_rule('math_inline', lambda r, ts, i, o, e:
                                '<span class="math">\\(' + html.escape(ts[i].content) + '\\)</span>')
        self.md.add_render_rule('math_block', lambda r, ts, i, o, e:
                                '<div class="math">\\[' + html.escape(ts[i].content) + '\\]</div>')

    def asset(self, path):
        path = path.resolve()
        if (path.suffix.lower() not in ALLOWED_ASSETS and path != BOOK / 'myst.yml') or not path.is_file():
            return None
        if path.is_relative_to(BOOK) and '_build' not in path.parts:
            rel = path.relative_to(BOOK).as_posix()
        elif path.is_relative_to(ROOT / 'referenceBooks') and path.suffix == '.pdf':
            rel = 'referenceBooks/' + path.name
        else:
            return None
        target = OUT / 'assets' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        self.assets[rel] = {'path': rel, 'url': 'generated/assets/' + quote(rel),
                            'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return 'generated/assets/' + quote(rel)

    def citation(self, match):
        links = []
        for key in re.findall(r'@([\w-]+)', match.group()):
            entry = self.bib[key]
            self.cited.add(key)
            label = entry.get('author', key).split(' and ')[0].split(',')[0].replace('{', '').replace('}', '')
            links.append(f'<a href="#ref-{html.escape(key)}">{html.escape(label)}, {entry.get("year", "n.d.")}</a>')
        return '(' + '; '.join(links) + ')'

    def code(self, code, language='python'):
        if language not in {'python', 'py', 'ipython3'}:
            return '<pre><code>' + html.escape(code) + '</code></pre>'
        # Notebook magic is explicitly identified, not silently run as Python.
        code_id = 'cell-' + str(len(self.codes) + 1)
        try:
            ast.parse(code)
            runnable = True
        except SyntaxError:
            runnable = False
        local_only = bool(re.search(r'^import marimo\b', code, re.M))
        runnable = runnable and not local_only
        self.codes.append({'id': code_id, 'code': code, 'solution': self.in_solution > 0,
                           'title': self.heading, 'runnable': runnable,
                           'magic': bool(re.search(r'^\s*[%!]', code, re.M))})
        if not runnable:
            note = ('Local application example — run in the book\'s local Python environment.' if local_only else
                    'Syntax illustration, output transcript or notebook command — not an executable Python cell.')
            return '<div class="code-cell"><p class="muted">' + note + '</p><pre><code>' + html.escape(code) + '</code></pre></div>'
        return (f'<div class="code-cell"><div class="cell-actions"><span>Python · {code_id}</span>'
                f'<button data-code="{code_id}">Edit this code</button>'
                + ('' if self.in_solution else f'<button data-setup="{code_id}">Include preceding examples</button>')
                + '</div><pre><code>' + html.escape(code) + '</code></pre></div>')

    def render(self, text):
        lines = text.splitlines()
        output, prose = [], []
        def flush():
            if prose:
                block = re.sub(r'\[@[^\]]+\]', self.citation, '\n'.join(prose))
                block = re.sub(r'^\([^\n]+\)=\s*$', '', block, flags=re.M)
                output.append(self.md.render(block))
                prose.clear()
        i = 0
        while i < len(lines):
            line = lines[i]
            directive = re.match(r'^(:{3,})\{([^}]+)\}\s*(.*)', line)
            fence = re.match(r'^(`{3,}|~{3,})(.*)', line)
            heading = re.match(r'^(#{1,6})\s+(.+)', line)
            label = re.match(r'^\(([^)]+)\)=\s*$', line)
            if label:
                flush()
                output.append(f'<span id="{html.escape(label[1])}"></span>')
                i += 1
            elif directive:
                flush()
                mark, kind, title = directive.groups()
                j = i + 1
                while j < len(lines) and lines[j].strip() != mark:
                    j += 1
                body = lines[i+1:j]
                options = []
                while body and (not body[0].strip() or body[0].startswith(':')):
                    options.append(body.pop(0))
                is_answer = kind == 'dropdown' or 'dropdown' in ' '.join(options) or bool(re.match(r'(?i)(solution|answer)', title))
                self.in_solution += int(is_answer)
                rendered = self.render('\n'.join(body))
                self.in_solution -= int(is_answer)
                if is_answer:
                    output.append('<details class="solution"><summary>' + html.escape(title or 'Show answer') + '</summary>' + rendered + '</details>')
                else:
                    output.append('<aside class="callout"><h3>' + html.escape(title or kind.title()) + '</h3>' + rendered + '</aside>')
                    if re.match(r'(?i)(quiz|exercise|question|challenge)', title):
                        output.append(self.workspace(title))
                i = j + 1
            elif fence:
                flush()
                mark, info = fence.groups()
                j = i + 1
                while j < len(lines) and not lines[j].startswith(mark):
                    j += 1
                body = '\n'.join(lines[i+1:j])
                if info.startswith('{image}') or info.startswith('{figure}'):
                    src = info.split('}', 1)[1].strip()
                    alt = re.search(r'^:alt:\s*(.*)', body, re.M)
                    output.append(f'<img src="{html.escape(src)}" alt="{html.escape(alt[1] if alt else Path(src).stem)}" loading="lazy">')
                elif info.startswith('{code-cell}'):
                    body = re.sub(r'\A(?::[^\n]*\n|\n)*', '', body)
                    output.append(self.code(body, info.split('}', 1)[1].strip() or 'python'))
                else:
                    output.append(self.code(body, info.strip()))
                i = j + 1
            elif heading:
                flush()
                self.heading = heading[2]
                level = len(heading[1])
                output.append(f'<h{level} id="{slug(self.heading)}">{self.md.renderInline(self.heading)}</h{level}>')
                if (re.match(r'(?i)(exercise\s+\d|question\s+\d)', self.heading)
                    or self.path == PROJECT and level == 2):
                    output.append(self.workspace(self.heading))
                i += 1
            else:
                prose.append(line)
                i += 1
        flush()
        return '\n'.join(output)

    def workspace(self, title):
        key = 'work-' + slug(title)
        self.workspaces.append({'id': key, 'title': title, 'code': '# ' + title + '\n# Write your own solution here.\n'})
        return f'<button class="workspace-link" data-workspace="{key}">Write your solution: {html.escape(title)}</button>'

    def page(self, relative, group):
        self.path, self.codes, self.cited, self.workspaces = relative, [], set(), []
        self.in_solution, self.heading = 0, 'Example'
        source = source_markdown(BOOK / relative)
        title_match = re.search(r'^#\s+(.+)', source, re.M)
        title = title_match[1] if title_match else Path(relative).stem
        rendered = self.render(source)
        if self.cited:
            rendered += '<h2 id="references">References</h2><ul>'
            for key in sorted(self.cited):
                entry = self.bib[key]
                label = '. '.join(entry.get(f, '').replace('{', '').replace('}', '') for f in ['author', 'year', 'title', 'publisher'])
                url = entry.get('url') or ('https://doi.org/' + entry['doi'] if 'doi' in entry else '')
                rendered += f'<li id="ref-{key}">{html.escape(label)}'
                if url:
                    rendered += f' <a href="{html.escape(url)}" target="_blank" rel="noopener">Source</a>'
                rendered += '</li>'
            rendered += '</ul>'
        soup = BeautifulSoup(rendered, 'html.parser')
        for dangerous in soup.select('script, iframe, object, embed'):
            dangerous.decompose()
        for element in soup.find_all(True):
            for attr in list(element.attrs):
                if attr.lower().startswith('on'):
                    del element[attr]
        for tag in soup.select('[href], [src]'):
            attr = 'href' if tag.has_attr('href') else 'src'
            url = tag[attr]
            if url.startswith('#'):
                tag[attr] = '#lesson=' + quote(relative, safe='') + '&anchor=' + quote(url[1:])
                continue
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                if parts.scheme in {'javascript', 'data'}:
                    del tag[attr]
                continue
            target = ((BOOK / relative).parent / unquote(parts.path)).resolve()
            rel = target.relative_to(BOOK).as_posix() if target.is_relative_to(BOOK) else ''
            if rel in self.page_ids:
                tag[attr] = '#lesson=' + quote(rel, safe='') + ('&anchor=' + parts.fragment if parts.fragment else '')
            elif link := self.asset(target):
                tag[attr] = link
            elif rel.endswith('.md'):
                tag[attr] = '../' + rel[:-3].lower() + '/' + ('#' + parts.fragment if parts.fragment else '')
            else:
                # Missing source links are kept visible and reported in the audit.
                tag['title'] = 'See the book for this resource'
        return {'id': relative, 'title': title, 'group': group, 'html': str(soup),
                'codes': self.codes, 'workspaces': self.workspaces,
                'assessment': relative == ASSESSMENT, 'project': relative == PROJECT,
                'source_sha256': hashlib.sha256((BOOK / relative).read_bytes()).hexdigest()}

    def build(self):
        OUT.mkdir(exist_ok=True)
        self.asset(BOOK / 'myst.yml')
        pages = [self.page(path, group) for path, group in self.pages]
        # Data are fetched on demand into the worker, retaining book-relative paths.
        for path in BOOK.rglob('*'):
            if path.suffix in {'.csv', '.xlsx', '.db', '.nc', '.geojson'} and '_build' not in path.parts:
                self.asset(path)
        payload = {'version': 2, 'pages': pages, 'assets': sorted(self.assets.values(), key=lambda x: x['path']),
                   'book_url': 'https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/',
                   'source_digest': hashlib.sha256(''.join(p['source_sha256'] for p in pages).encode()).hexdigest()}
        (OUT / 'course.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(f'Sandbox: {len(pages)} book pages, {sum(len(p["codes"]) for p in pages)} Python cells, {len(self.assets)} assets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true', help='Copy only participant assets into the built Pages site')
    args = parser.parse_args()
    Builder().build()
    if args.deploy:
        target = BOOK / '_build/html/sandbox'
        target.mkdir(parents=True, exist_ok=True)
        for name in ['index.html', 'app.js', 'styles.css', 'python-worker.js', 'eagelogo.jpg']:
            shutil.copyfile(SANDBOX / name, target / name)
        shutil.copytree(OUT, target / 'generated', dirs_exist_ok=True)
        print('Participant sandbox staged at /sandbox/ beside the Jupyter Book.')

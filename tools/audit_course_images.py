"""Audit image references in the rendered book and participant bundle."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import hashlib
import os

from bs4 import BeautifulSoup
from PIL import Image
from publish_rpa_site import build_inventory

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'EAGE_PythonRenewableEnergyCourse' / '_build' / 'html'
SANDBOX = ROOT / 'EAGE_pythonSandBox'
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


def main():
    missing = []
    images = set()
    page_count = 0
    for page in BOOK.rglob('*.html'):
        if any(part in {'jupyterlite', 'node_modules'} for part in page.parts):
            continue
        page_count += 1
        soup = BeautifulSoup(page.read_text(encoding='utf-8'), 'html.parser')
        for img in soup.find_all('img'):
            src = img.get('src', '')
            url = urlsplit(src)
            if not src or url.scheme or url.netloc or src.startswith('//'):
                continue
            path = unquote(url.path)
            base = '/' + os.environ.get('BASE_URL', '').strip('/')
            if base != '/' and (path == base or path.startswith(base + '/')):
                path = path[len(base):]
            target = (BOOK / path.lstrip('/')) if path.startswith('/') else (page.parent / path)
            if not target.is_file():
                missing.append((str(page.relative_to(BOOK)), src))
            else:
                images.add(target.resolve())

    bundle = json.loads((SANDBOX/'generated/course.json').read_text(encoding='utf-8'))
    assets = {a['url']: a for a in bundle['assets']}
    inventory = build_inventory(SANDBOX)
    image_refs = 0
    for page in bundle['pages']:
        soup = BeautifulSoup(page['html'], 'html.parser')
        for img in soup.find_all('img'):
            src = img.get('src', '')
            parsed = urlsplit(src)
            if parsed.scheme or parsed.netloc or src.startswith('//'):
                continue
            image_refs += 1
            target = SANDBOX / unquote(parsed.path)
            if not src or src not in assets or not target.is_file() or unquote(parsed.path) not in inventory:
                missing.append((page['id'], src))
            else:
                if hashlib.sha256(target.read_bytes()).hexdigest() != assets[src]['sha256']:
                    raise AssertionError(f'Image does not match manifest: {page["id"]}: {src}')
                images.add(target.resolve())
    if missing:
        raise AssertionError(f'Missing or unregistered image references: {missing}')
    for image in images:
        if image.suffix.lower() in IMAGE_EXTENSIONS:
            with Image.open(image) as opened:
                opened.verify()
    print(f'Image audit passed: {page_count} book pages, {len(bundle["pages"])} sandbox lessons, '
          f'{image_refs} sandbox image references and {len(images)} distinct local image files.')


if __name__ == '__main__':
    main()

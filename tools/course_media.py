"""Attach explicitly mapped participant media to a course build."""
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

class Headings(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        if tag == 'h2' and dict(attrs).get('id'):
            self.ids.add(dict(attrs)['id'])

def attach_media(pages, sandbox, languages=('en', 'es')):
    sandbox = Path(sandbox)
    manifest = sandbox / 'media.json'
    if not manifest.exists():
        return
    entries = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(entries, dict):
        raise ValueError('Media manifest must be an object')
    pages_by_id = {page['id']: page for page in pages}
    for page_id, entry in entries.items():
        if page_id not in pages_by_id or not isinstance(entry, dict) or set(entry) - {'audio', 'video'}:
            raise ValueError(f'Invalid media lesson: {page_id}')
        page = pages_by_id[page_id]
        headings = Headings()
        headings.feed(page['html'])
        root = (sandbox / 'media').resolve()
        def copy(value, lang, allowed):
            if not isinstance(value, str):
                raise ValueError(f'Invalid media path for {page_id}')
            source = (root / value).resolve()
            if not source.is_relative_to(root / lang) or not source.is_file() or source.suffix.lower() not in allowed:
                raise ValueError(f'Missing or invalid {lang} media file: {value}')
            rel = source.relative_to(root)
            target = sandbox / 'generated' / 'media' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return 'generated/media/' + quote(rel.as_posix())
        audio, video = entry.get('audio', {}), entry.get('video', {})
        if not isinstance(audio, dict) or not isinstance(video, dict) or set(audio) - set(languages) or set(video) - set(languages):
            raise ValueError(f'Invalid media language for {page_id}')
        result = {'audio': {}, 'video': {}}
        for lang, tracks in audio.items():
            if not isinstance(tracks, dict) or set(tracks) - (headings.ids | {'intro'}):
                raise ValueError(f'Invalid audio section for {page_id}')
            result['audio'][lang] = {section: copy(path, lang, {'.mp3', '.wav', '.ogg', '.m4a'}) for section, path in tracks.items()}
        for lang, path in video.items():
            result['video'][lang] = copy(path, lang, {'.mp4', '.webm', '.ogv'})
        page['media'] = result

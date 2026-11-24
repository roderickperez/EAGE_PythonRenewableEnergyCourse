"""Fail publication if private materials or machine-specific notebook outputs remain."""
import json
from pathlib import Path
import re

from stage_course_web_assets import output_contains_machine_path

ROOT = Path(__file__).resolve().parents[1] / 'EAGE_PythonRenewableEnergyCourse/_build/html'


def check_build(root: Path) -> None:
    if not (root / 'index.html').exists():
        raise ValueError('Build HTML before checking publication')
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if any(term in path.name.lower() for term in
               ['projectsolution', 'test_answer_key', 'final_project_solution', 'instructor_private']):
            raise ValueError(f'Private or obsolete file found: {path}')
        if path.suffix in {'.html', '.json', '.md', '.ipynb'}:
            text = path.read_text(encoding='utf-8', errors='ignore')
            for marker in ['Instructor-only test answer key', 'INSTRUCTOR ONLY — excluded',
                           'def locate_data():', 'project_solution_validation.ipynb']:
                if marker in text:
                    raise ValueError(f'Private solution content found in {path}')
            if path.suffix == '.ipynb':
                notebook = json.loads(text)
                if any(output_contains_machine_path(output)
                       for cell in notebook.get('cells', [])
                       for output in cell.get('outputs', [])):
                    raise ValueError(f'Machine-specific home path in saved notebook output: {path}')
    for topic in ('python', 'python-sql', 'hydroelectric', 'solar', 'wind', 'geothermal'):
        page = root / 'section7' / 'assessment' / topic / 'index.html'
        if not page.exists():
            raise ValueError(f'Assessment page missing: {topic}')
        text = page.read_text(encoding='utf-8')
        if len(re.findall(r'id="question-\d+(?:-(?:easy|medium|hard))?"', text)) != 10:
            raise ValueError(f'Incorrect assessment question count: {topic}')
        if len(re.findall(r'difficulty-badge', text)) < 10:
            raise ValueError(f'Missing assessment difficulty badges: {topic}')


if __name__ == '__main__':
    check_build(ROOT)
    print('Student build passed: no private solutions or machine-specific saved notebook paths.')

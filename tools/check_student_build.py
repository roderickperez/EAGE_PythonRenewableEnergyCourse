"""Fail publication if instructor files or obsolete project solutions remain."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]/'EAGE_PythonRenewableEnergyCourse/_build/html'
assert (ROOT/'index.html').exists(),'Build HTML before checking publication'
for path in ROOT.rglob('*'):
    if not path.is_file():continue
    assert not any(term in path.name.lower() for term in
                   ['projectsolution','test_answer_key','final_project_solution','instructor_private']),path
    if path.suffix in {'.html','.json','.md','.ipynb'}:
        text=path.read_text(encoding='utf-8',errors='ignore')
        for marker in ['Instructor-only test answer key','INSTRUCTOR ONLY — excluded',
                       'def locate_data():','project_solution_validation.ipynb']:
            assert marker not in text,f'Private solution content found in {path}'
for topic in ('python', 'python-sql', 'hydroelectric', 'solar', 'wind', 'geothermal'):
    test=ROOT/'section7'/'assessment'/topic/'index.html'
    assert test.exists(), f'Assessment page missing: {topic}'
    text=test.read_text(encoding='utf-8')
    assert len(re.findall(r'id="question-\d+(?:-(?:easy|medium|hard))?"',text))==10, topic
    assert len(re.findall(r'difficulty-badge', text)) >= 10, topic
print('Student build passed: no instructor answer keys, private solution code or obsolete project-solution downloads.')

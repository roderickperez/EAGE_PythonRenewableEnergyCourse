"""Check source coverage, executable syntax, references, data and answer exclusions."""
import ast
import hashlib
import json
import re
import io
import contextlib
from unittest.mock import patch
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from bs4 import BeautifulSoup
from build_sandbox import BOOK, OUT, ROOT, ASSESSMENT, ASSESSMENT_PAGES, PROJECT, toc_pages, Builder
from build_chapter_exercises import WORKBOOK_CHAPTERS, PATHS


def validate():
    payload = json.loads((OUT / 'course.json').read_text())
    pages = {p['id']: p for p in payload['pages']}
    assert set(pages) == {p for p, _ in Builder().pages}, 'Missing book pages'
    removed = {'Introduction to Python Programming', 'IDE & Notebooks', 'Python Basics', 'Python Libraries', 'Data Processing and Exploration Data Analysis'}
    assert not any(page['group'] in removed for page in pages.values())
    assert (OUT / 'hydro-lab' / 'hydro-lab.js').is_file()
    assert (OUT / 'hydro-lab' / 'fallback.svg').is_file()
    assert 'hydro-lab/index.html' not in pages['section6/hydroelectricEnergy.md']['html']
    hydro = pages['section6/hydroelectricEnergy.md']
    assert 'id="hydraulic-head-explorer"' in hydro['html']
    assert 'id="net-head-explorer"' in hydro['html']
    assert 'id="flow-volume-explorer"' in hydro['html']
    assert 'id="hydraulic-power-explorer"' in hydro['html']
    assert 'id="electrical-power-explorer"' in hydro['html']
    assert 'id="hydraulic-loss-explorer"' in hydro['html']
    assert 'id="turbine-operation-explorer"' in hydro['html']
    assert 'id="pumped-storage-explorer"' in hydro['html']
    assert 'hydro-lab/head.html' not in hydro['html']
    assert 'hydro-lab/balance.html' not in hydro['html']
    for file in ('head.html', 'head.js', 'balance.html', 'balance.js'):
        assert (OUT / 'hydro-lab' / file).is_file()
    assert set(hydro['media']['audio']['en']) == {
        'what-head-means', 'flow-and-volume-are-different-quantities', 'deriving-the-power-equation', 'turbine-families',
        'environmental-releases-before-generation', 'hydrographs-and-flow-duration-curves',
        'reservoir-conservation-and-variable-head',
    }
    assert (OUT / 'solar-lab' / 'solar-lab.js').is_file()
    solar_page = pages['section6/solarEnergy.md']['html']
    assert 'solar-lab/index.html' not in solar_page
    assert 'id="physical-description-what-solar-energy-systems-do"' in solar_page
    assert (OUT / 'wind-lab' / 'wind-lab.js').is_file()
    wind_page = pages['section6/windEnergy.md']['html']
    assert 'wind-lab/index.html' not in wind_page
    assert 'id="physical-description-extracting-energy-from-moving-air"' in wind_page
    assert (OUT / 'geothermal-lab' / 'geothermal-lab.js').is_file()
    geothermal_page = pages['section6/geothermalEnergy.md']['html']
    assert 'geothermal-lab/index.html' not in geothermal_page
    assert 'id="physical-description-useful-heat-from-the-subsurface"' in geothermal_page
    assert (OUT / 'project-map' / 'project-map.js').is_file()
    assert 'project-map/index.html' not in pages[PROJECT]['html']
    assert 'id="load-the-separate-input-datasets"' in pages[PROJECT]['html']
    intro = 'section7/renewableExercises.md'
    assert pages[intro]['title'] == 'Workbook introduction'
    workbook = [path for _, _, path in WORKBOOK_CHAPTERS]
    assert [path for path,page in pages.items() if page['group']=='Applied Python Workbook'] == [intro, *workbook]
    table = BeautifulSoup(pages[intro]['html'], 'html.parser').select('table a')
    assert [parse_qs(link['href'][1:])['lesson'][0] for link in table] == workbook
    assert 'private-course/eage-python-course/index.html' in pages['courseIntro.md']['html']
    for path, page in pages.items():
        assert page['source_sha256'] == hashlib.sha256((BOOK/path).read_bytes()).hexdigest(), f'Stale content: {path}'
        soup = BeautifulSoup(page['html'], 'html.parser')
        assert len({w['id'] for w in page['workspaces']}) == len(page['workspaces']), path
        assert not soup.select('script,iframe,object,embed')
        for cell in page['codes']:
            if cell['runnable']:
                ast.parse(cell['code'])
                assert soup.select(f'button[data-code="{cell["id"]}"]'), (path, cell['id'])
        for link in soup.select('[href], [src]'):
            target = link.get('href') or link.get('src')
            if target.startswith('generated/'):
                assert (ROOT/'EAGE_pythonSandBox'/unquote(urlsplit(target).path)).is_file(),target
            elif target.startswith('#lesson='):
                args = parse_qs(target[1:]); assert args['lesson'][0] in pages,target
                if 'anchor' in args:
                    target_soup=BeautifulSoup(pages[args['lesson'][0]]['html'],'html.parser')
                    assert target_soup.find(id=args['anchor'][0]), (path,target)
    for theory_path in PATHS.values():
        page=pages[theory_path]
        assert page['group']=='Introduction to Renewable Energy',theory_path
        assert 'expanded-theory:' in page['html'],theory_path
        assert not page['workspaces'] and not any(cell['solution'] for cell in page['codes']),theory_path
        assert not BeautifulSoup(page['html'],'html.parser').select('details.solution'),theory_path
    for path in workbook:
        page=pages[path];soup=BeautifulSoup(page['html'],'html.parser')
        assert all(workspace['id'] in page['legacy_drafts'] for workspace in page['workspaces']),path
        assert all(key.split('::')[0] in PATHS.values() for key in page['legacy_drafts'].values()),path
        answers=soup.select('details.solution')
        assert len(answers)==20 and all('open' not in a.attrs for a in answers),path
        assert len(page['workspaces'])==20,path
        assert len([c for c in page['codes'] if c['solution']])==20,path
        assert 'expanded-theory:' not in page['html'],path
        for answer in answers:
            assert answer.select('button[data-code]') and 'Step' in answer.get_text(),path
    tests=[pages[path] for path in ASSESSMENT_PAGES];project=pages[PROJECT]
    assert project['group']=='Course Final Project'
    assert all(page['group']=='Quizzes' for path,page in pages.items() if path.startswith('section7/renewableEnergyquizzes/'))
    for page in pages.values():
        if page['group']=='Quizzes':
            answers = BeautifulSoup(page['html'], 'html.parser').select('details.solution')
            assert len(answers)==20 and all('open' not in answer.attrs for answer in answers),page['id']
            assert len(page['workspaces'])==20,page['id']
    assert any(asset['path']=='section8/projectStarter.ipynb' for asset in payload['assets'])
    assert all(len(test['workspaces'])==10 and test['assessment'] for test in tests)
    assert len(project['workspaces'])>=8
    for page in [*tests,project]:
        assert not page['codes'] and '<details' not in page['html']
        assert all(ast.parse(w['code']).body==[] for w in page['workspaces'])
    for name in ['solar','wind','hydro','geothermal','demand']:
        a=next(x for x in payload['assets'] if x['path']==f'section8/data/{name}.csv')
        assert a['sha256']==hashlib.sha256((BOOK/a['path']).read_bytes()).hexdigest()
    forbidden=['instructor_private','test_answer_key.md','final_project_solution.md','projectSolution.md']
    for text in [(OUT/'course.json').read_text(), '\n'.join(p.as_posix() for p in OUT.rglob('*'))]:
        assert not any(s in text for s in forbidden),'Private content reached participant bundle'
    # These short core lessons and quizzes must also execute as a continuous
    # teaching sequence, including their required setup and typed-input examples.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    executed=0
    with patch('builtins.input', return_value='7'):
        for page in pages.values():
            if page['group'] != 'Quizzes':
                continue
            namespace={'__name__':'__main__'}
            for cell in page['codes']:
                if not cell['runnable']:continue
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(cell['code'],page['id']+':'+cell['id'],'exec'),namespace)
                executed+=1
                plt.close('all')
    print(f'Executed {executed} quiz Python cells with explicit input fixtures.')
    print(f'Sandbox validated: {len(pages)} pages, 100 collapsed workbook answers, 60 assessment workspaces, project workspaces and five matching datasets.')

if __name__=='__main__':validate()

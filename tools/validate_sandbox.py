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
    module_lessons = (
        ('section0/introRenewableEnergy.md',),
        ('section0/python-foundations.md',),
        ('section4/energy-data-analysis.md',),
        ('section4/energy-maps.md',),
        ('section6/windEnergy.md', 'section6/labs/wind-capacity.md'),
        ('section6/hydroelectricEnergy.md', 'section6/labs/river-flood-flow.md'),
        ('section6/solarEnergy.md', 'section6/labs/solar-irradiance.md'),
        ('section6/geothermalEnergy.md', 'section6/labs/geothermal-gradient.md'),
    )
    for number, expected in enumerate(module_lessons, 1):
        actual = tuple(path for path, page in pages.items()
                       if page['group'].startswith(f'Module {number} · '))
        assert actual == expected, f'Module {number} navigation: {actual}'
        lesson = pages[expected[-1]]
        notebook = (BOOK / expected[-1]).with_suffix('.ipynb')
        assert notebook.is_file(), f'Module {number} notebook missing'
        colab = ('https://colab.research.google.com/github/roderickperez/'
                 f'EAGE_PythonRenewableEnergyCourse/blob/main/{BOOK.name}/{notebook.relative_to(BOOK).as_posix()}')
        assert colab in lesson['html'], f'Module {number} Colab link missing'
        if number > 1:
            assert any(cell['runnable'] and cell['solution'] for cell in lesson['codes']), expected[-1]
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
    assert 'id="flow-duration-explorer"' in hydro['html']
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
    solar = pages['section6/solarEnergy.md']
    solar_page = solar['html']
    assert 'solar-lab/index.html' not in solar_page
    # Direct links to a standalone explorer are allowed; the lesson itself uses embedded anchors.
    assert not any('solar-lab/resource.html' in frame.get('src', '') for frame in BeautifulSoup(solar_page, 'html.parser').find_all('iframe'))
    for demo in ('horizontal', 'poa', 'irradiation', 'tilt', 'incidence', 'iv', 'operating-point', 'fill-factor',
                 'dc-temperature', 'cell-temperature', 'inverter-clipping', 'thermal-collector'):
        assert f'id="solar-{demo}-explorer"' in solar_page
    for file in ('resource.html', 'resource.js'):
        assert (OUT / 'solar-lab' / file).is_file()
    solar_subtitles = BeautifulSoup(solar_page, 'html.parser').select('h3[id], h4[id]')
    assert {heading['id'] for heading in solar_subtitles} <= set(solar['media']['audio']['en'])
    assert 'id="physical-description-what-solar-energy-systems-do"' in solar_page
    assert (OUT / 'wind-lab' / 'wind-lab.js').is_file()
    wind = pages['section6/windEnergy.md']
    wind_page = wind['html']
    assert 'wind-lab/index.html' not in wind_page
    assert 'wind-lab/aerodynamics.html' not in wind_page
    for demo in ('cube-law', 'betz', 'air-density', 'shear', 'turbulence', 'electrical-curve',
                 'annual-energy', 'weibull', 'losses', 'direction'):
        assert f'id="wind-{demo}-explorer"' in wind_page
    for file in ('aerodynamics.html', 'aerodynamics.js'):
        assert (OUT / 'wind-lab' / file).is_file()
    wind_subtitles = BeautifulSoup(wind_page, 'html.parser').select('h3[id], h4[id]')
    assert {heading['id'] for heading in wind_subtitles} <= set(wind['media']['audio']['en'])
    assert 'from-resource-distributions-to-annual-energy' in wind['media']['audio']['en']
    assert 'id="physical-description-extracting-energy-from-moving-air"' in wind_page
    assert (OUT / 'geothermal-lab' / 'geothermal-lab.js').is_file()
    geothermal_page = pages['section6/geothermalEnergy.md']['html']
    assert 'geothermal-lab/index.html' not in geothermal_page
    assert 'id="physical-description-useful-heat-from-the-subsurface"' in geothermal_page
    for demo in ('darcy', 'gradient', 'heat', 'carnot', 'conversion', 'pump', 'heatpump', 'storage', 'decline', 'energy', 'worked',
                 'compact-gradient', 'compact-heat', 'compact-carnot', 'compact-efficiency', 'compact-energy', 'compact-decline'):
        assert f'id="geo-{demo}-explorer"' in geothermal_page
    for file in ('equations.html', 'equations.js'):
        assert (OUT / 'geothermal-lab' / file).is_file()
    assert (OUT / 'project-map' / 'project-map.js').is_file()
    assert 'project-map/index.html' not in pages[PROJECT]['html']
    assert 'id="load-the-separate-input-datasets"' in pages[PROJECT]['html']
    extra_order = [path for path, page in pages.items() if page['group'] == 'Extra']
    assert extra_order == ['section5/xarray-extra.md', 'section5/mdio-format.md']
    assert 'id="mdio-and-xarray-compared"' in pages['section5/mdio-format.md']['html']
    wind_project = 'section8/wind-project.md'
    project_order = [path for path, page in pages.items() if page['group'] == 'Course Projects']
    assert project_order[:5] == ['section8/hydraulic-project.md', 'section8/solar-project.md',
                                 wind_project, 'section8/geothermal-project.md', PROJECT]
    assert 'id="how-to-use-the-explorer"' in pages[wind_project]['html']
    assert (OUT / 'wind-lab' / 'wind-project.js').is_file()
    assert (OUT / 'wind-lab' / 'project.html').is_file()
    assert (OUT / 'solar-lab' / 'solar-project.js').is_file()
    assert (OUT / 'solar-lab' / 'project.html').is_file()
    assert (OUT / 'geothermal-lab' / 'geothermal-project.js').is_file()
    assert (OUT / 'geothermal-lab' / 'project.html').is_file()
    assert (OUT / 'geothermal-lab' / 'drilling.js').is_file()
    assert "'geothermal-drilling-lab'" in (ROOT / 'EAGE_pythonSandBox' / 'app.js').read_text(encoding='utf-8')
    assert "'section8/geothermal-project.md'" in (ROOT / 'EAGE_pythonSandBox' / 'app.js').read_text(encoding='utf-8')
    assert "'section8/solar-project.md'" in (ROOT / 'EAGE_pythonSandBox' / 'app.js').read_text(encoding='utf-8')
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
        ids = [element['id'] for element in soup.select('[id]')]
        assert len(ids) == len(set(ids)), f'Duplicate HTML ID in {path}'
        assert len({w['id'] for w in page['workspaces']}) == len(page['workspaces']), path
        if path == 'section8/projectIntro.md':
            assert all('references' not in w['title'].casefold() for w in page['workspaces']), path
        assert not soup.select('script,iframe,object,embed')
        for cell in page['codes']:
            if cell['runnable']:
                ast.parse(cell['code'])
                assert soup.select(f'button[data-code="{cell["id"]}"]'), (path, cell['id'])
        citation_targets = {}
        for link in soup.select('[href], [src]'):
            target = link.get('href') or link.get('src')
            if target.startswith('generated/'):
                assert (ROOT/'EAGE_pythonSandBox'/unquote(urlsplit(target).path)).is_file(),target
            elif target.startswith('#lesson='):
                args = parse_qs(target[1:]); assert args['lesson'][0] in pages,target
                if 'anchor' in args:
                    target_soup=BeautifulSoup(pages[args['lesson'][0]]['html'],'html.parser')
                    assert target_soup.find(id=args['anchor'][0]), (path,target)
                    if args['anchor'][0].startswith('ref-'):
                        label = link.get_text(' ', strip=True)
                        citation_targets.setdefault(label, set()).add((args['lesson'][0], args['anchor'][0]))
        assert all(len(targets) == 1 for targets in citation_targets.values()), f'Ambiguous citation label: {path}'
    for theory_path in PATHS.values():
        page=pages[theory_path]
        assert page['group'].startswith('Module '), theory_path
        assert 'expanded-theory:' in page['html'],theory_path
        assert not page['workspaces'] and not any(cell['solution'] for cell in page['codes']),theory_path
        assert not BeautifulSoup(page['html'],'html.parser').select('details.solution'),theory_path
    for path in workbook:
        page=pages[path];soup=BeautifulSoup(page['html'],'html.parser')
        assert all(workspace['id'] in page['legacy_drafts'] for workspace in page['workspaces']),path
        assert all(key.split('::')[0] in PATHS.values() for key in page['legacy_drafts'].values()),path
        answers=soup.select('details.solution')
        count = 22 if path == 'section7/workbook/hydroelectric.md' else 20
        assert len(answers)==count and all('open' not in a.attrs for a in answers),path
        assert len(page['workspaces'])==count,path
        assert len([c for c in page['codes'] if c['solution']])==count,path
        source=(BOOK/path).read_text(encoding='utf-8')
        sections=re.split(r'\n### (Exercise \d+ — [^\n]+)\n',source)
        expected_solutions=[
            (title,re.findall(r'```\{code-cell\} python\n(.*?)```',block.split('\n## ',1)[0],re.S)[-1].strip())
            for title,block in zip(sections[1::2],sections[2::2])
        ]
        solution_cells=[c for c in page['codes'] if c['solution']]
        delivered_solutions=[(c['title'],c['code'].strip()) for c in solution_cells]
        assert expected_solutions==delivered_solutions,(path,'delivered workbook answers differ from source')
        assert 'expanded-theory:' not in page['html'],path
        for answer,cell in zip(answers,solution_cells):
            button=answer.select_one('button[data-code]')
            assert button and button['data-code']==cell['id'] and 'Step' in answer.get_text(),path
            assert answer.select_one('pre code').get_text().strip()==cell['code'].strip(),(path,cell['id'],'visible answer differs')
    tests=[pages[path] for path in ASSESSMENT_PAGES];project=pages[PROJECT]
    assert project['group']=='Course Projects'
    assert all(page['group']=='Quizzes' for path,page in pages.items() if path.startswith('section7/renewableEnergyquizzes/'))
    for page in pages.values():
        if page['group']=='Quizzes':
            answers = BeautifulSoup(page['html'], 'html.parser').select('details.solution')
            assert len(answers)==20 and all('open' not in answer.attrs for answer in answers),page['id']
            assert len(page['workspaces'])==20,page['id']
            source=(BOOK/page['id']).read_text(encoding='utf-8')
            solutions=re.findall(r':::\{admonition\} Solution\n(.*?)\n:::',source,re.S)
            assert len(solutions)==20,(page['id'],'source quiz solutions')
            code_by_id={cell['id']:cell for cell in page['codes']}
            for number,(solution,answer) in enumerate(zip(solutions,answers),1):
                expected=[c.strip() for c in re.findall(r'```\{code-cell\} python\n(.*?)```',solution,re.S)]
                buttons=answer.select('button[data-code]')
                delivered=[code_by_id[button['data-code']] for button in buttons]
                assert all(cell['solution'] for cell in delivered),(page['id'],number,'non-solution code')
                assert expected==[cell['code'].strip() for cell in delivered],(page['id'],number,'answer code differs')
                visible=[code.get_text().strip() for code in answer.select('div.code-cell pre code')]
                assert visible==expected,(page['id'],number,'visible answer differs')
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
    figures=0
    assembled_setups=0
    with patch('builtins.input', return_value='7'):
        for page in pages.values():
            if page['group'] != 'Quizzes':
                continue
            namespace={'__name__':'__main__'}
            for cell in page['codes']:
                if not cell['runnable']:continue
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(cell['code'],page['id']+':'+cell['id'],'exec'),namespace)
                    for number in plt.get_fignums():
                        try:
                            plt.figure(number).canvas.draw()
                            figures+=1
                        except Exception as exc:
                            raise RuntimeError(f"Figure rendering failed: {page['id']}:{cell['id']}") from exc
                executed+=1
                plt.close('all')
            solution_indices=[i for i,c in enumerate(page['codes']) if c['solution'] and c['runnable']]
            if solution_indices:
                final=solution_indices[-1]
                assembled='\n\n'.join(c['code'] for c in page['codes'][:final+1] if c['runnable'])
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(assembled,page['id']+':assembled-practice','exec'),{'__name__':'__main__'})
                plt.close('all')
                assembled_setups+=1
    print(f'Executed {executed} quiz Python cells and rendered {figures} figures with explicit input fixtures.')
    print(f'Executed {assembled_setups} final-answer practice setups as assembled scripts.')
    print(f'Sandbox validated: {len(pages)} pages, {sum(len(pages[path]["workspaces"]) for path in workbook)} collapsed workbook answers, 60 assessment workspaces, project workspaces and five matching datasets.')

if __name__=='__main__':validate()

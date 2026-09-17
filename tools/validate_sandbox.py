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
from build_sandbox import BOOK, OUT, ROOT, ASSESSMENT, PROJECT, toc_pages


def validate():
    payload = json.loads((OUT / 'course.json').read_text())
    pages = {p['id']: p for p in payload['pages']}
    assert set(pages) == {p for p, _ in toc_pages()}, 'Missing book pages'
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
    topics = ['section0/introRenewableEnergy.md'] + ['section6/'+t+'Energy.md' for t in ['solar','hydroelectric','wind','geothermal']]
    for path in topics:
        page=pages[path];soup=BeautifulSoup(page['html'],'html.parser')
        answers=soup.select('details.solution')
        assert len(answers)==20 and all('open' not in a.attrs for a in answers),path
        assert len(page['workspaces'])==20,path
        assert len([c for c in page['codes'] if c['solution']])==20,path
        assert 'expanded-theory:' in page['html'],path
        for answer in answers:
            assert answer.select('button[data-code]') and 'Step' in answer.get_text(),path
    test=pages[ASSESSMENT];project=pages[PROJECT]
    assert len(test['workspaces'])==25 and len(project['workspaces'])>=8
    for page in [test,project]:
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
            if not (page['group'] in {'Quizzes','Python Basics'} or '/basicLibraries/' in page['id']):
                continue
            namespace={'__name__':'__main__'}
            for cell in page['codes']:
                if not cell['runnable']:continue
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(cell['code'],page['id']+':'+cell['id'],'exec'),namespace)
                executed+=1
                plt.close('all')
    print(f'Executed {executed} basic-lesson and quiz Python cells with explicit input fixtures.')
    print(f'Sandbox validated: {len(pages)} pages, 100 collapsed answers, 25 question-only assessment workspaces, project workspaces and five matching datasets.')

if __name__=='__main__':validate()

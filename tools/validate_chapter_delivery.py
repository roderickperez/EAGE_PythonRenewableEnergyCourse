"""Check practice coverage, independent solutions and instructor-file separation."""
from pathlib import Path
import contextlib
import io
import re
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'EAGE_PythonRenewableEnergyCourse'
PATHS=['section0/introRenewableEnergy.md','section6/solarEnergy.md',
       'section6/hydroelectricEnergy.md','section6/windEnergy.md','section6/geothermalEnergy.md']

def validate():
    count=0
    for relative in PATHS:
        path=ROOT/relative; text=path.read_text(encoding='utf-8')
        blocks=re.split(r'\n### Exercise \d+ — ',text)[1:]
        assert len(blocks)==20,relative
        levels=re.findall(r'\*\*Difficulty:\*\* (Easy|Medium|Hard)',text)
        assert [levels.count(x) for x in ['Easy','Medium','Hard']]==[5,10,5],relative
        for i,block in enumerate(blocks,1):
            assert '**Reference:**' in block and '[@' in block,(relative,i,'reference')
            assert block.count('::::{dropdown} Step-by-step answer')==1 and ':open:' not in block
            cells=re.findall(r'```\{code-cell\} python\n(.*?)```',block,re.S)
            assert len(cells)==2 and 'assert' in cells[-1],(relative,i,'cells')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cells[-1],f'{relative}:exercise-{i}','exec'),{})
            plt.close('all');count+=1
        nb=nbformat.read(path.with_suffix('.ipynb'),as_version=4);nbformat.validate(nb)
        assert sum('### Exercise ' in c.source for c in nb.cells if c.cell_type=='markdown')==20
    config=(ROOT/'myst.yml').read_text()
    assert 'projectSolution' not in config and 'instructor_private' not in config
    assert not (ROOT/'section8/projectSolution.md').exists()
    test=(ROOT/'section7/renewableEnergyTest.md').read_text()
    assert len(re.findall(r'^### Question \d+',test,re.M))==25
    assert '**Answer' not in test and '{code-cell}' not in test and '{dropdown}' not in test
    tracked=subprocess.check_output(['git','ls-files','instructor_private'],cwd=BASE,text=True)
    assert not tracked.strip(),'Private instructor files must never be tracked'
    print(f'Validated {count} independent chapter solutions, 5 notebook companions, 25 question-only test items and private-file exclusion.')

if __name__=='__main__':validate()

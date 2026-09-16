"""Guard the substantive theory sections and check their worked arithmetic."""
from pathlib import Path
import math
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]/'EAGE_PythonRenewableEnergyCourse'
PATHS={'general':'section0/introRenewableEnergy','solar':'section6/solarEnergy',
       'hydro':'section6/hydroelectricEnergy','wind':'section6/windEnergy',
       'geothermal':'section6/geothermalEnergy'}

def validate():
    for topic,relative in PATHS.items():
        text=(ROOT/(relative+'.md')).read_text(encoding='utf-8')
        start=f'<!-- expanded-theory:{topic}:start -->'
        end=f'<!-- expanded-theory:{topic}:end -->'
        assert text.count(start)==text.count(end)==1,topic
        theory=text.split(start)[1].split(end)[0]
        assert len(theory.split())>=1400,(topic,'Detailed theory was truncated')
        assert text.index(end)<text.index('## Chapter practice'),topic
        assert '## Worked calculation before coding' in theory or '## Worked comparison before coding' in theory
        assert '[@' in theory and '$$' in theory
        notebook=nbformat.read(ROOT/(relative+'.ipynb'),as_version=4)
        prose='\n'.join(c.source for c in notebook.cells if c.cell_type=='markdown')
        assert theory.strip() in prose,(topic,'Notebook is missing the complete theory')
    # Independent hand checks for the five explanatory worked examples.
    solar_dc=10*(800/1000)*(1-0.004*((25+800/800*(45-20))-25))
    solar_ac=min(solar_dc*0.97,8)
    assert math.isclose(solar_ac,6.984) and math.isclose(solar_ac*0.5,3.492)
    hydro=1000*9.81*(12-2)*(30-2)*0.85/1e6
    assert math.isclose(hydro,2.33478) and min(hydro,2)*6==12
    wind=0.5*1.225*math.pi*20**2*8**3/1000
    assert math.isclose(wind,394.08,abs_tol=0.01)
    assert math.isclose(wind*0.4*0.95,149.75,abs_tol=0.01)
    geo_th=50*4180*(150-70)/1e6
    geo_net=geo_th*0.12*(1-0.1)
    assert math.isclose(geo_th,16.72) and math.isclose(geo_net*4,7.22304)
    supply=[1,5,2,4];demand=[3]*4;served=[min(s,d) for s,d in zip(supply,demand)]
    assert sum(supply)==sum(demand)==12 and sum(served)==9
    assert sum(s-d for s,d in zip(supply,served))==3
    print('Detailed theory retained in all five notebooks; five worked-example calculations passed.')

if __name__=='__main__':validate()

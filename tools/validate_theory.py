"""Guard the substantive theory sections and check their worked arithmetic."""
from pathlib import Path
import math
import re
import nbformat
from build_chapter_exercises import notebook_citations

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
        # Worked code inside the theory becomes a separate executable cell.
        # Check every prose segment and every code block instead of requiring
        # the complete mixed Markdown/code source in one Markdown cell.
        segments=re.split(r'```\{code-cell\} python\n(.*?)```',theory,flags=re.S)
        code_sources={c.source.strip() for c in notebook.cells if c.cell_type=='code'}
        for index,segment in enumerate(segments):
            if not segment.strip():
                continue
            if index % 2:
                assert segment.strip() in code_sources,(topic,'Notebook is missing theory code')
            else:
                assert notebook_citations(segment.strip()) in prose,(topic,'Notebook is missing theory prose')
    # Independent hand checks for the five explanatory worked examples.
    solar_dc=10*(800/1000)*(1-0.004*((25+800/800*(45-20))-25))
    solar_ac=min(solar_dc*0.97,8)
    assert math.isclose(solar_ac,6.984) and math.isclose(solar_ac*0.5,3.492)
    # Independent arithmetic for the added Goswami-based teaching examples.
    ghi=800*math.cos(math.radians(40))+100
    poa=800*math.cos(math.radians(10))+100*(1+math.cos(math.radians(30)))/2+0.2*ghi*(1-math.cos(math.radians(30)))/2
    assert math.isclose(poa,890.69764,abs_tol=0.01)
    tank_heat=200*4180*(60-20)/3_600_000
    assert math.isclose(tank_heat,9.2888888889) and math.isclose(tank_heat/2,4.6444444444)
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
    print('Detailed theory retained in all five notebooks; seven worked-example calculations passed.')

if __name__=='__main__':validate()

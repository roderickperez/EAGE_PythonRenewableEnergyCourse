"""Regression checks for the October course audit: physics boundaries and download parsing."""
from pathlib import Path
import ast, copy, hashlib, json, math, re, sys
import numpy as np
from download_energy_data import decode_series
ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'EAGE_PythonRenewableEnergyCourse'

def rejects(call):
    try: call()
    except ValueError: return
    raise AssertionError('Expected invalid input to raise ValueError')

def physics():
    ns={'np':np}
    for name in ['solar','hydroelectric','wind','geothermal']:
        text=(BOOK/f'section6/{name}Energy.md').read_text()
        for code in re.findall(r'```\{code-cell\} python\n(.*?)```',text,re.S):
            parsed=ast.parse(code)
            definitions=[n for n in parsed.body if isinstance(n,ast.FunctionDef)]
            if definitions:exec(compile(ast.Module(body=definitions,type_ignores=[]),name,'exec'),ns)
    for bad in [np.nan,np.inf,-np.inf]:
        rejects(lambda:ns['temperature_at_depth_c'](bad))
        rejects(lambda:ns['temperature_at_depth_c'](1,surface_c=bad))
        rejects(lambda:ns['carnot_efficiency'](150,bad))
        rejects(lambda:ns['weibull_pdf']([1,2],bad,8))
    rejects(lambda:ns['carnot_efficiency'](10,10))
    rejects(lambda:ns['carnot_efficiency'](100,-274))
    rejects(lambda:ns['weibull_pdf']([-1],2,8))
    assert math.isinf(ns['weibull_pdf'](0,0.5,8))
    assert ns['weibull_pdf'](0,1,8)==1/8
    assert ns['weibull_pdf'](0,2,8)==0
    assert np.allclose(ns['temperature_at_depth_c']([0,1,2]),[15,45,75])
    assert np.allclose(ns['turbine_power_mw']([2.99,3,12,24.99,25]),[0,0,3,3,0])
    assert np.allclose(ns['pv_ac_power_mw']([0,800],25,0.01,0.008),[0,0.006984])
    assert math.isclose(ns['geothermal_net_power_mw'](50,150,70,0.12),1.80576)
    print('Physics: finite inputs, singular density, shutdown boundaries and independent unit calculations passed.')

def decoder():
    path=BOOK/'data/examples/eurostat_at_wind_2023.json'
    raw=path.read_bytes();data=json.loads(raw)
    meta=json.loads(path.with_suffix('.provenance.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    rows=decode_series(data)
    assert len(rows)==12 and math.isclose(sum(r['generation_gwh'] for r in rows),7971.360)
    missing=copy.deepcopy(data);missing['value'].pop('0');missing['status']={'0':'p'}
    result=decode_series(missing)
    assert result[0]['generation_gwh'] is None and result[0]['status_flag']=='p'
    dense=copy.deepcopy(data);dense['value']=[data['value'][str(i)] for i in range(12)]
    assert decode_series(dense)==rows
    # Moving the only non-singleton axis tests reliance on dimension order.
    reordered=copy.deepcopy(data);pos=reordered['id'].index('time')
    reordered['id'].insert(0,reordered['id'].pop(pos));reordered['size'].insert(0,reordered['size'].pop(pos))
    assert decode_series(reordered)==rows
    bad=copy.deepcopy(data);bad['dimension']['time']['category']['index']['2023-01']=99
    rejects(lambda:decode_series(bad))
    bad=copy.deepcopy(data);bad['value']['0']=True
    rejects(lambda:decode_series(bad))
    bad=copy.deepcopy(data);bad['value']['0']=float('nan')
    rejects(lambda:decode_series(bad))
    # The notebook lesson embeds the same reviewed implementation.
    import inspect
    assert inspect.getsource(decode_series).strip() in (BOOK/'section5/download-to-database.md').read_text()
    print('Eurostat: snapshot hash, 12 rows, sparse/dense values, reordered dimensions, flags and rejection cases passed.')

if __name__=='__main__':
    physics();decoder()

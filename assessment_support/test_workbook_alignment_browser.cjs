const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-reader-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));const page=course.pages.find(p=>p.id==='section7/workbook/general.md');const cell=page.codes.find(c=>c.code.includes('def align(solar, wind, expected):'));assert.ok(cell?.runnable,'alignment answer is runnable');const conversion=page.codes.find(c=>c.code.includes('def to_mwh(values, units):'));assert.ok(conversion?.runnable,'unit conversion answer is runnable');const hydro=course.pages.find(p=>p.id==='section7/workbook/hydroelectric.md').codes.find(c=>c.code.includes('def audit(flow, duration, limit):'));const solarAudit=course.pages.find(p=>p.id==='section7/workbook/solar.md').codes.find(c=>c.code.includes('def audit_irradiance(g, dt, daylight):'));assert.ok(hydro?.runnable && solarAudit?.runnable,'coverage answers are runnable');
 await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(page.id)});let loaded=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){loaded=true;break}await delay(100)}assert.ok(loaded,'lesson loaded');
 const source=cell.code+`
bad = solar.copy()
bad.loc[1, "solar_mw"] = float("nan")
try:
    align(bad, wind, time)
except ValueError as error:
    assert "finite power" in str(error)
else:
    raise AssertionError("Missing value was accepted")
try:
    align(solar.iloc[[0, 2]], wind.iloc[[0, 2]], time)
except ValueError as error:
    assert "schedule" in str(error)
else:
    raise AssertionError("Missing timestamp was accepted")
solar_text = solar.assign(solar_mw=["0", "1", "2"])
wind_text = wind.assign(wind_mw=["0", "1", "2"])
converted = align(solar_text, wind_text, time)
assert (converted.solar_mw + converted.wind_mw).tolist() == [2., 2., 2.]
assert solar_text.solar_mw.tolist() == ["0", "1", "2"]
assert wind_text.wind_mw.tolist() == ["0", "1", "2"]
solar_dates = solar.assign(time=time.strftime("%Y-%m-%dT%H:%M:%SZ"))
wind_dates = wind.assign(time=time[::-1].strftime("%Y-%m-%d %H:%M:%S+00:00"))
converted_dates = align(solar_dates, wind_dates, time)
assert len(converted_dates) == 3
assert (converted_dates.solar_mw + converted_dates.wind_mw).tolist() == [2., 2., 2.]
assert solar_dates.time.iloc[0] == "2025-01-01T00:00:00Z"
assert len(align(solar_dates, wind, time)) == 3
try:
    align(solar.assign(solar_mw=["0", "bad", "2"]), wind, time)
except ValueError:
    pass
else:
    raise AssertionError("Nonnumeric text was accepted")
print("ALIGNMENT_COVERAGE_OK")` + "\n" + conversion.code + `
for index in (pd.Index(["a", None]), pd.MultiIndex.from_tuples([("a", 1), ("b", None)])):
    try:
        to_mwh(pd.Series([1., 2.], index=index), pd.Series(["MWh", "kWh"], index=index))
    except ValueError as error:
        assert "must not be missing" in str(error)
    else:
        raise AssertionError("Missing index label was accepted")
valid_index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
assert to_mwh(pd.Series([1., 2000.], index=valid_index), pd.Series(["MWh", "kWh"], index=valid_index)).tolist() == [1., 2.]
print("UNIT_INDEX_VALIDATION_OK")` + "\n" + hydro.code + "\n" + solarAudit.code + `
for index in (pd.Index(["A", None]), pd.MultiIndex.from_tuples([("A", 1), ("B", None)])):
    flow = pd.Series([3., float("nan")], index=index)
    duration = pd.Series([1., 2.], index=index)
    irradiance = pd.Series([500., 0.], index=index)
    daylight = pd.Series([True, False], index=index)
    for check in (lambda: audit(flow, duration, 6), lambda: audit_irradiance(irradiance, duration, daylight)):
        try:
            check()
        except ValueError as error:
            assert "must not be missing" in str(error)
        else:
            raise AssertionError("Missing coverage index label was accepted")
valid_index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
assert audit(pd.Series([3., float("nan")], index=valid_index), pd.Series([1., 2.], index=valid_index), 6) == (.5, 1/3, 10800., 54000.)
assert audit_irradiance(pd.Series([500., 0.], index=valid_index), pd.Series([1., 2.], index=valid_index), pd.Series([True, False], index=valid_index)) == (1., 1., .5)
print("COVERAGE_INDEX_VALIDATION_OK")`;
 await send('Runtime.evaluate',{expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent,status:document.getElementById('python-status').getAttribute('aria-label')})`});state=q.result.value;if(!state.busy&&state.output.includes('COVERAGE_INDEX_VALIDATION_OK'))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.match(state.output,/ALIGNMENT_COVERAGE_OK/,JSON.stringify(state));assert.match(state.output,/UNIT_INDEX_VALIDATION_OK/,JSON.stringify(state));assert.match(state.output,/COVERAGE_INDEX_VALIDATION_OK/,JSON.stringify(state));console.log('Workbook alignment answer runs in browser Python and rejects missing values and timestamps, normalizes numeric and timestamp strings without mutation, and checks unit-conversion, hydropower and solar coverage index labels.');socket.close();
 }finally{browser.kill()}})().catch(error=>{console.error(error);process.exitCode=1});

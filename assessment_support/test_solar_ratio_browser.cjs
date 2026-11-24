const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-reader-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));const page=course.pages.find(p=>p.id==='section7/renewableEnergyquizzes/solarEnergy.md');const cell=page.codes.find(c=>c.code.includes('def validate_hourly_readings('));assert.ok(cell?.runnable, 'matched ratio answer is runnable');const monthly=page.codes.find(c=>c.code.includes('monthly["pr"].plot.bar'));assert.ok(monthly?.runnable,'monthly ratio answer is runnable');const high=monthly.code.replace('"energy_mwh": [420, 620,','"energy_mwh": [696, 620,');const zero=monthly.code.replace('"irradiation_kwh_m2": [58, 82,','"irradiation_kwh_m2": [0, 82,');assert.notEqual(high,monthly.code);assert.notEqual(zero,monthly.code);
 await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(page.id)});let loaded=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){loaded=true;break}await delay(100)}assert.ok(loaded,'lesson loaded');
 const source=cell.code+`
assert len(joined) == 4 and len(paired) == 2 and abs(pr - .625) < 1e-12
good = pd.date_range("2025-06-01", periods=2, freq="h", tz="UTC")
for index in (pd.DatetimeIndex([good[0], pd.NaT]), pd.DatetimeIndex([good[0], good[0]]), good + pd.Timedelta(minutes=30), good.tz_localize(None)):
    try:
        validate_hourly_readings(pd.Series([1., 2.], index=index))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid timestamp index accepted")
assert validate_hourly_readings(pd.Series(["500", "700"], index=good)).tolist() == [500., 700.]
print("SOLAR_RATIO_VALIDATION_OK")` + "\nimport matplotlib.pyplot as plt\n" + high + `
assert abs(monthly.iloc[0]["pr"] - 1.2) < 1e-12 and monthly.iloc[0]["flag"]
assert plt.gca().get_ylim()[1] >= 1.2
plt.close("all")
` + zero + `
assert pd.isna(monthly.iloc[0]["pr"]) and not monthly.iloc[0]["flag"]
print("MONTHLY_RATIO_BOUNDARIES_OK")`;
 await send('Runtime.evaluate',{expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent,status:document.getElementById('python-status').getAttribute('aria-label')})`});state=q.result.value;if(!state.busy&&state.output.includes('MONTHLY_RATIO_BOUNDARIES_OK'))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.match(state.output,/SOLAR_RATIO_VALIDATION_OK/,JSON.stringify(state));assert.match(state.output,/MONTHLY_RATIO_BOUNDARIES_OK/,JSON.stringify(state));console.log('Solar matched and monthly performance-ratio examples pass timestamp, high-ratio and zero-irradiation checks in browser Python.');socket.close();
 }finally{browser.kill()}})().catch(error=>{console.error(error);process.exitCode=1});

const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-reader-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));const page=course.pages.find(p=>p.id==='section5/world-energy-download.md');const cell=page.codes.find(c=>c.code.includes('entity_history = data.loc'));assert.ok(cell?.runnable,'selection cell is runnable');
 await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(page.id)});let loaded=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){loaded=true;break}await delay(100)}assert.ok(loaded,'lesson loaded');
 const source = `import pandas as pd, math, hashlib
from pathlib import Path
raw_folder = Path('/tmp/owid_code_fixture')
raw_folder.mkdir(exist_ok=True)
source_ready = True
def record(name, data):
    raw=(raw_folder / name); raw.write_bytes(data)
    return {'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
source_data = pd.DataFrame({'country':['Austria'], 'iso_code':['AUT'], 'year':[2023], 'wind_electricity_twh':[0.1]})
codebook_data = pd.DataFrame([dict(column='wind_electricity_twh',unit='TWh',title='Wind',description='Fixture',source='Fixture')])
download_record = {'owid_energy.csv':record('owid_energy.csv',source_data.to_csv(index=False).encode()),
    'owid_energy.codebook.csv':record('owid_energy.codebook.csv',codebook_data.to_csv(index=False).encode()),
    'readme.md':record('readme.md',b'fixture')}
exec(${JSON.stringify(cell.code)}, globals())
assert selection_context['country'] == 'Austria'
source_data = pd.concat([source_data, pd.DataFrame([{'country':'Austria','iso_code':None,'year':2022,'wind_electricity_twh':0.2}])], ignore_index=True)
download_record['owid_energy.csv'] = record('owid_energy.csv',source_data.to_csv(index=False).encode())
try:
    exec(${JSON.stringify(cell.code)}, globals())
except ValueError as error:
    assert 'both missing and present source codes' in str(error)
else:
    raise AssertionError('Mixed codes were accepted')
assert 'selection_context' not in globals() and 'selected' not in globals()
print('WORLD_IDENTITY_GUARD_OK')`;
 await send('Runtime.evaluate',{expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent,status:document.getElementById('python-status').getAttribute('aria-label')})`});state=q.result.value;if(!state.busy&&state.output.includes('WORLD_IDENTITY_GUARD_OK'))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.match(state.output,/WORLD_IDENTITY_GUARD_OK/,JSON.stringify(state));console.log('Browser Python selects consistent entity data and blocks mixed-code export state.');socket.close();
 }finally{browser.kill()}})().catch(error=>{console.error(error);process.exitCode=1});

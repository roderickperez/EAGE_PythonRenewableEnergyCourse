const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-assessment-check-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));
 const cases=[{id:'section7/assessment/solar.md',workspace:'work-question-40-hard',terms:['[0, 2, 5, None, 7, 1]','95%','3 kW','5 kW'],marker:'SOLAR_ASSESSMENT_OK',code:`dc=[0,2,5,None,7,1]
for cap, delivered, clipped in [(3,8.85,5.4),(5,12.6,1.65)]:
    ac=[None if p is None else min(cap,.95*p) for p in dc]
    lost=[None if p is None else .95*p-a for p,a in zip(dc,ac)]
    assert sum(a is not None for a in ac)==5 and ac[3] is None
    assert abs(sum(a for a in ac if a is not None)-delivered)<1e-8
    assert abs(sum(a for a in lost if a is not None)-clipped)<1e-8
print('SOLAR_ASSESSMENT_OK')`},
 {id:'section7/assessment/wind.md',workspace:'work-question-50-hard',terms:['[2, 6, 10, None, -1, 26]','3 MW','4 MW','one-hour'],marker:'WIND_ASSESSMENT_OK',code:`speeds=[2,6,10,None,-1,26]
status=['missing' if v is None else 'invalid' if v < 0 else 'valid' for v in speeds]
assert status.count('missing') == 1 and status.count('invalid') == 1 and status.count('valid') == 4
def curve(v,rating):
    if v is None or v < 0: return None
    if v <= 3 or v >= 25: return 0.
    if v >= 12: return float(rating)
    return rating * (v**3-3**3)/(12**3-3**3)
for rating, expected in [(3,2.049382716049383),(4,2.7325102880658436)]:
    powers=[curve(v,rating) for v in speeds]
    assert sum(p is not None for p in powers)==4
    assert powers[3] is None and powers[4] is None and powers[5]==0
    assert abs(sum(p for p in powers if p is not None)-expected)<1e-10
print('WIND_ASSESSMENT_OK')`}];
 for(const item of cases){const page=course.pages.find(p=>p.id===item.id);assert.ok(page.workspaces.some(w=>w.id===item.workspace));for(const term of item.terms)assert.ok(page.html.includes(term),item.id+' missing '+term);await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(item.id)});let ready=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){ready=true;break}await delay(100)}assert.ok(ready,item.id+' loaded');await send('Runtime.evaluate',{expression:`document.querySelector(${JSON.stringify('[data-workspace="'+item.workspace+'"]')}).click();document.getElementById('code').value=${JSON.stringify(item.code)};document.getElementById('run').click()`});let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent})`});state=q.result.value;if(!state.busy&&state.output.includes(item.marker))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.ok(state.output.includes(item.marker),JSON.stringify(state));console.log(item.id+' comparison runs and preserves missing/invalid observations.');}
 socket.close();}finally{browser.kill()}})().catch(e=>{console.error(e);process.exitCode=1});

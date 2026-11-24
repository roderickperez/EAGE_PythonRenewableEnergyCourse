const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-reader-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));const page=course.pages.find(p=>p.id==='section7/assessment/hydroelectric.md');assert.ok(page.workspaces.some(w=>w.id==='work-question-30-hard'),'assessment workspace exists');
 await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(page.id)});let loaded=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){loaded=true;break}await delay(100)}assert.ok(loaded,'lesson loaded');
 const source=`import sqlite3
flows = [5, 10, None, 15]
with sqlite3.connect(":memory:") as db:
    db.execute("CREATE TABLE rating_generation(rating_mw REAL, hour_index INTEGER, energy_mwh REAL, PRIMARY KEY (rating_mw, hour_index))")
    for rating in (2.0, 4.0):
        for hour, flow in enumerate(flows):
            energy = None if flow is None else min(rating, 0.85 * 1000 * 9.81 * flow * 40 / 1e6)
            db.execute("INSERT INTO rating_generation VALUES (?, ?, ?)", (rating, hour, energy))
    rows = db.execute("SELECT rating_mw, COUNT(energy_mwh), SUM(energy_mwh) FROM rating_generation GROUP BY rating_mw ORDER BY rating_mw").fetchall()
    assert len(rows) == 2 and all(row[1] == 3 for row in rows)
    assert abs(rows[0][2] - 5.6677) < 1e-8
    assert abs(rows[1][2] - 9.0031) < 1e-8
    print("HYDRO_RATING_ASSESSMENT_OK", rows)`;
 await send('Runtime.evaluate',{expression:`document.querySelector('[data-workspace="work-question-30-hard"]').click();document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});
 let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent,status:document.getElementById('python-status').getAttribute('aria-label')})`});state=q.result.value;if(!state.busy&&state.output.includes('HYDRO_RATING_ASSESSMENT_OK'))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.match(state.output,/HYDRO_RATING_ASSESSMENT_OK/,JSON.stringify(state));console.log('Hydropower assessment workspace runs the SQLite rating comparison and preserves the missing hour.');socket.close();
 }finally{browser.kill()}})().catch(error=>{console.error(error);process.exitCode=1});

const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const profile=fs.mkdtempSync(path.join(os.tmpdir(),'eage-reader-'));const browser=spawn(process.env.EAGE_TEST_EDGE||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});try{
 const f=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(f);i++)await delay(100);assert.ok(fs.existsSync(f));const port=Number(fs.readFileSync(f,'utf8').split('\n')[0]);const target=await(await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,{method:'PUT'})).json();const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});let serial=0,pending=new Map();socket.onmessage=e=>{let m=JSON.parse(e.data);if(pending.has(m.id)){let {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result)}};const send=(method,params={})=>new Promise((resolve,reject)=>{let id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
 const course=JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json','utf8'));const page=course.pages.find(p=>p.id==='section7/renewableEnergyquizzes/solarEnergy.md');const specs=[['solarEnergy.md','pv_power_mw'],['hydroelectricEnergy.md','hydro_power_mw'],['geothermalEnergy.md','well_power_mw']];const cells=specs.map(([file,name])=>{const p=course.pages.find(p=>p.id==='section7/renewableEnergyquizzes/'+file);const c=p.codes.find(c=>c.code.includes('def '+name+'('));assert.ok(c?.runnable, name+' answer is runnable');return c.code});
 await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(page.id)});let loaded=false;for(let i=0;i<150;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(page.title)}`});if(q.result.value){loaded=true;break}await delay(100)}assert.ok(loaded,'lesson loaded');
 const source=cells.join('\n\n')+`
import pandas as pd
labels=['t0','t1']
checks=[(pv_power_mw,([500,1000],[20,25]),(10,8)),(hydro_power_mw,([1,2],[20,50]),()),(well_power_mw,([10,20],[100,150]),(70,.12))]
for model,(left,right),extra in checks:
    model(pd.Series(left,index=labels),pd.Series(right,index=labels),*extra)
    try:
        model(pd.Series(left,index=labels),pd.Series(right,index=labels[::-1]),*extra)
    except ValueError as error:
        assert 'matching indexes' in str(error)
    else:
        raise AssertionError('Mismatched labels were accepted')
    for bad in (['t0','t0'], ['t0',None]):
        try:
            model(pd.Series(left,index=bad),pd.Series(right,index=bad),*extra)
        except ValueError as error:
            assert 'unique, nonmissing' in str(error)
        else:
            raise AssertionError('Invalid keys were accepted')
print('LABELED_POWER_GUARDS_OK')`;
 await send('Runtime.evaluate',{expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});let state;for(let i=0;i<600;i++){let q=await send('Runtime.evaluate',{returnByValue:true,expression:`({busy:document.getElementById('run').disabled,output:document.getElementById('output').textContent,status:document.getElementById('python-status').getAttribute('aria-label')})`});state=q.result.value;if(!state.busy&&state.output.includes('LABELED_POWER_GUARDS_OK'))break;if(!state.busy&&/Traceback|Error:/.test(state.output))break;await delay(200)}assert.match(state.output,/LABELED_POWER_GUARDS_OK/,JSON.stringify(state));console.log('Solar, hydro and geothermal quiz helpers reject mismatched labeled inputs in browser Python.');socket.close();
 }finally{browser.kill()}})().catch(error=>{console.error(error);process.exitCode=1});

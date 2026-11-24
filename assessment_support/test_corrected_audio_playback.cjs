const assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const {spawn} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-audio-browser-'));
  const browser = spawn(process.env.EAGE_TEST_EDGE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ['--headless=new', '--disable-gpu', '--no-first-run',
     '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'], {stdio:'ignore', windowsHide:true});
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i=0; i<100 && !fs.existsSync(portFile); i++) await delay(100);
    assert.ok(fs.existsSync(portFile), 'Edge started');
    const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve,reject) => {socket.onopen=resolve;socket.onerror=reject;});
    let serial=0; const pending=new Map();
    socket.onmessage = event => {const m=JSON.parse(event.data);if(pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result);}};
    const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
    for (const [lesson, anchor, filename] of [
      ['section6/windEnergy.md','why-available-power-contains-the-cube-of-wind-speed','available-power-reference-flux-corrected.mp3'],
      ['section6/solarEnergy.md','thermal-storage-energy-is-not-power','thermal-storage-boundary-corrected.mp3'],
    ]) {
      await send('Page.navigate',{url:'http://127.0.0.1:18343/#lesson='+encodeURIComponent(lesson)});
      let ready=false;
      for(let i=0;i<150;i++){
        const result=await send('Runtime.evaluate',{returnByValue:true,expression:`(() => {const h=document.getElementById(${JSON.stringify(anchor)});return !!h?.querySelector('button.section-audio:not([disabled])')})()`});
        if(result.result.value){ready=true;break;} await delay(100);
      }
      assert.ok(ready, `${lesson} section audio control loaded`);
      const position = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        const button = document.getElementById(${JSON.stringify(anchor)}).querySelector('button.section-audio');
        button.scrollIntoView({block:'center', behavior:'instant'});
        const rect = button.getBoundingClientRect();
        return {x:rect.left+rect.width/2, y:rect.top+rect.height/2};
      })()`});
      const {x,y} = position.result.value;
      await send('Input.dispatchMouseEvent', {type:'mouseMoved', x, y});
      await send('Input.dispatchMouseEvent', {type:'mousePressed', x, y, button:'left', clickCount:1});
      await send('Input.dispatchMouseEvent', {type:'mouseReleased', x, y, button:'left', clickCount:1});
      let state;
      for(let i=0;i<100;i++){
        const result=await send('Runtime.evaluate',{returnByValue:true,expression:`(() => {const p=document.getElementById('audio-player');return {url:p.currentSrc,paused:p.paused,time:p.currentTime,duration:p.duration,error:p.error?.code,hidden:p.hidden}})()`});
        state=result.result.value; if(state?.time>0.2)break; await delay(100);
      }
      assert.ok(state.url.endsWith(filename) && !state.paused && !state.hidden && state.time>0.2 && Number.isFinite(state.duration),JSON.stringify(state));
      console.log(`${lesson}: corrected recording loaded and advanced (${state.duration.toFixed(2)} seconds)`);
    }
    socket.close();
  } finally {browser.kill();}
})().catch(error=>{console.error(error);process.exitCode=1;});

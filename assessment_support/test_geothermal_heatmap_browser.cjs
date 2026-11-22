// Serve the repository root on 127.0.0.1:18453, then run with Windows Node.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn, spawnSync} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-heatmap-edge-'));
  const edge = spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', [
    '--headless=new', '--disable-gpu', '--no-first-run', '--disable-background-networking',
    '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank',
  ], {stdio: 'ignore', windowsHide: true});
  let socket;
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await delay(100);
    assert.ok(fs.existsSync(portFile), 'Edge debugging endpoint started');
    const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'})).json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let serial = 0;
    const pending = new Map();
    socket.onmessage = ({data}) => {
      const message = JSON.parse(data), item = pending.get(message.id);
      if (!item) return;
      pending.delete(message.id);
      message.error ? item.reject(new Error(message.error.message)) : item.resolve(message.result);
    };
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)); }, 15000);
      pending.set(id, {resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); }});
      socket.send(JSON.stringify({id, method, params}));
    });
    for (const name of ['project', 'multifield']) {
      await send('Page.navigate', {url: `http://127.0.0.1:18453/EAGE_pythonSandBox/generated/geothermal-lab/${name}.html`});
      let visible;
      for (let i = 0; i < 100; i++) {
        const check = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          const heatmap = document.querySelector('canvas[data-chart="heatmap"]');
          if (!heatmap) return null;
          heatmap.parentElement.scrollIntoView({block:'center'});
          return heatmap.parentElement.innerText;
        })()`});
        visible = check.result.value;
        if (visible) break;
        await delay(100);
      }
      assert.match(visible || '', /Producer temperature over time and schematic formation/);
      assert.match(visible, /Blue represents 15 °C and red the initial temperature/);
      assert.match(visible, /only the bottom row shows the simulated producer/);
      if (name === 'project') {
        const note = await send('Runtime.evaluate', {returnByValue: true,
          expression: `document.querySelector('.wp-note').innerText`});
        assert.match(note.result.value, /In the field view, blue wells indicate injection and red wells indicate production/);
      }
      if (process.env.EAGE_HEATMAP_SCREENSHOT) {
        const image = await send('Page.captureScreenshot', {format: 'png'});
        fs.writeFileSync(`${process.env.EAGE_HEATMAP_SCREENSHOT}-${name}.png`, Buffer.from(image.data, 'base64'));
      }
      console.log(`${name}: actual local interface displays the heatmap explanation`);
    }
    await send('Page.navigate', {url: 'http://127.0.0.1:18453/'});
    await delay(1000);
    const result = await send('Runtime.evaluate', {awaitPromise: true, returnByValue: true, expression: `(async () => {
      const {drawProjectCharts} = await import('/web/geothermal-multifield/project-charts.js');
      document.body.innerHTML = '<h1>Irregular simulated times: 0, 1 and 10 years</h1>' +
        ['temperature','energy','depth','heatmap'].map(n => '<canvas data-chart="'+n+'" width="600" height="210"></canvas>').join('');
      const values = [160, 100, 40];
      const history = [0, 1, 10].map((years, i) => ({years, production: values[i], injection: 70, energy: i, layerTemperatures: {2000: values[i]}}));
      drawProjectCharts(document.body, {years: 10, initial: 160, layerTemperatures: {2000: 40}}, history, 2000);
      const c = document.querySelector('[data-chart="heatmap"]').getContext('2d');
      return [60, 100, 400].map(x => Array.from(c.getImageData(x, 172, 1, 1).data));
    })()`});
    assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
    const colors = result.result.value;
    assert.equal(new Set(colors.map(c => c.join(','))).size, 3, 'three distinct samples appear at the expected time positions');
    console.log('Actual Edge canvas rendered three temperatures at the irregular-time positions:', colors);
    if (process.env.EAGE_HEATMAP_SCREENSHOT) {
      const image = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
      fs.writeFileSync(process.env.EAGE_HEATMAP_SCREENSHOT, Buffer.from(image.data, 'base64'));
    }
  } finally {
    if (socket) socket.close();
    if (edge.exitCode === null) spawnSync('taskkill', ['/PID', String(edge.pid), '/T', '/F'], {windowsHide: true, stdio: 'ignore'});
    const literalProfile = profile.replaceAll("'", "''");
    spawnSync('powershell.exe', ['-NoProfile', '-Command',
      `Get-CimInstance Win32_Process -Filter "name = 'msedge.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('${literalProfile}') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`],
      {windowsHide: true, stdio: 'ignore', timeout: 10000});
    if (edge.exitCode === null) await Promise.race([new Promise(resolve => edge.once('exit', resolve)), delay(5000)]);
    try { fs.rmSync(profile, {recursive: true, force: true, maxRetries: 2, retryDelay: 300}); }
    catch (error) { console.error(`Could not remove test profile ${profile}: ${error.message}`); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

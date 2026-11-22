// Start `python assessment_support/serve_book_json.py`, then run this Node test.
// The server intentionally labels notebook files application/json.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn, spawnSync} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-book-download-'));
  const profile = path.join(temporary, 'profile'), downloads = path.join(temporary, 'downloads');
  fs.mkdirSync(downloads);
  const edge = spawn(process.env.EAGE_TEST_EDGE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ['--headless=new', '--disable-gpu', '--no-first-run', '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'],
    {stdio: 'ignore', windowsHide: true});
  let socket;
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await delay(100);
    assert.ok(fs.existsSync(portFile), 'Edge debugging endpoint started');
    const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
    const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'});
    const target = await response.json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let serial = 0; const pending = new Map();
    socket.onmessage = event => {
      const message = JSON.parse(event.data), item = pending.get(message.id);
      if (!item) return;
      pending.delete(message.id); message.error ? item.reject(new Error(message.error.message)) : item.resolve(message.result);
    };
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial, timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)); }, 15000);
      pending.set(id, {resolve: value => { clearTimeout(timer); resolve(value); }, reject: error => { clearTimeout(timer); reject(error); }});
      socket.send(JSON.stringify({id, method, params}));
    });
    await send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloads});
    for (const [route, filename, label] of [
      ['section0/introrenewableenergy', 'introRenewableEnergy.ipynb', 'Download the theory notebook'],
      ['section6/solarenergy', 'solarEnergy.ipynb', 'Download the theory notebook'],
      ['section6/geothermalenergy', 'geothermalEnergy.ipynb', 'Download the theory notebook'],
      ['section6/windenergy', 'windEnergy.ipynb', 'Download the theory notebook'],
      ['section6/hydroelectricenergy', 'hydroelectricEnergy.ipynb', 'Download the theory notebook'],
      ['section7/workbook/general', 'general.ipynb', 'Download the exercise notebook'],
      ['section7/workbook/hydroelectric', 'hydroelectric.ipynb', 'Download the exercise notebook'],
      ['section7/workbook/solar', 'solar.ipynb', 'Download the exercise notebook'],
      ['section7/workbook/wind', 'wind.ipynb', 'Download the exercise notebook'],
      ['section7/workbook/geothermal', 'geothermal.ipynb', 'Download the exercise notebook'],
      ['section5/download-to-database', 'download-to-database.ipynb', 'Download this tutorial as a Jupyter notebook'],
      ['section5/world-energy-download', 'world-energy-download.ipynb', 'Download this lesson as a Jupyter notebook'],
      ['section8/geothermal-project', 'geothermal-field-designer.ipynb', 'Python project notebook'],
      ['section8/projectintro', 'projectStarter.ipynb', 'Download the project starter notebook'],
    ]) {
      await send('Page.navigate', {url: `http://127.0.0.1:18344/${route}`});
      await delay(700); // Let the new document replace the preceding page before activating its link.
      let href;
      for (let i = 0; i < 100; i++) {
        const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          if (!location.pathname.toLowerCase().includes(${JSON.stringify(route.toLowerCase())})) return null;
          if (${JSON.stringify(route)} === 'section6/solarenergy' && !document.querySelector('h1')?.textContent.includes('Solar Photovoltaic Energy')) return null;
          let a = [...document.querySelectorAll('a')].find(a => a.textContent.includes(${JSON.stringify(label)}));
          if (!a) return null;
          const expectedStem = ({'section5/world-energy-download': 'world-energy-downloa',
            'section8/geothermal-project': 'geothermal-field-des'})[${JSON.stringify(route)}]
            || ${JSON.stringify(filename)}.slice(0, -6);
          if (!new URL(a.href).pathname.split('/').pop().startsWith(expectedStem + '-')) return null;
          if (${JSON.stringify(route)} === 'section6/solarenergy') {
            const replacement = a.cloneNode(true); a.replaceWith(replacement); a = replacement;
          }
          const href = a.href;
          if (${JSON.stringify(route)} === 'section0/introrenewableenergy') a.focus(); else a.click();
          return href;
        })()`});
        href = result.result.value; if (href) break; await delay(100);
      }
      assert.ok(href, `Notebook link exists in ${route}`);
      if (route === 'section0/introrenewableenergy') {
        await send('Page.bringToFront');
        await send('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13});
        await send('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13});
      }
      const output = path.join(downloads, filename);
      for (let i = 0; i < 100 && !fs.existsSync(output); i++) await delay(100);
      if (!fs.existsSync(output)) {
        const diagnostic = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          const a = [...document.querySelectorAll('a')].find(a => a.textContent.includes(${JSON.stringify(label)}));
          return {state: document.readyState, active: document.activeElement?.tagName,
            href: a?.href, download: a?.getAttribute('download'), target: a?.target};
        })()`});
        console.error('Download diagnostic:', diagnostic.result.value, 'files:', fs.readdirSync(downloads));
      }
      assert.ok(fs.existsSync(output), `${filename} downloaded even with application/json`);
      const response = await fetch(href);
      assert.match(response.headers.get('content-type'), /application\/json/);
      assert.deepEqual(fs.readFileSync(output), Buffer.from(await response.arrayBuffer()));
      const location = await send('Runtime.evaluate', {returnByValue: true, expression: 'location.pathname'});
      assert.ok(location.result.value.toLowerCase().includes(route.toLowerCase()), 'Reader stays on the lesson');
      if (route === 'section0/introrenewableenergy') {
        const unaffected = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          function inspect(href) {
            const link = document.createElement('a');
            link.href = href; link.target = '_blank'; link.addEventListener('click', e => e.preventDefault());
            document.body.append(link); link.click();
            const result = {download: link.getAttribute('download'), target: link.target};
            link.remove(); return result;
          }
          return {external: inspect('https://example.com/example.ipynb'),
                  page: inspect(location.origin + '/section6/solarenergy')};
        })()`});
        assert.deepEqual(unaffected.result.value, {
          external: {download: null, target: '_blank'}, page: {download: null, target: '_blank'},
        }, 'External notebooks and regular page links retain their original behavior');
      }
      console.log(`${route}: ${filename} downloaded with matching bytes`);
    }
    // MyST exposes this long-named source through its export toolbar.
    const timePage = fs.readFileSync('EAGE_PythonRenewableEnergyCourse/_build/html/section4.timeseriesenergyconsumption.json', 'utf8');
    const timeFile = timePage.match(/timeSeriesEnergyCons-[a-f0-9]{32}\.ipynb/)[0];
    const timeURL = 'http://127.0.0.1:18344/build/' + timeFile;
    await send('Runtime.evaluate', {expression:`(() => {
      const link=document.createElement('a'); link.href=${JSON.stringify(timeURL)};
      document.body.append(link); link.click(); link.remove();
    })()`});
    const timeDownload=path.join(downloads,'timeSeriesEnergyConsumption.ipynb');
    for(let i=0;i<100&&!fs.existsSync(timeDownload);i++) await delay(100);
    assert.ok(fs.existsSync(timeDownload), 'Long time-series notebook name is restored');
    assert.deepEqual(fs.readFileSync(timeDownload),Buffer.from(await (await fetch(timeURL)).arrayBuffer()));
    console.log('Historical time-series notebook downloads with its full original filename.');
    for (const [route, title, expected] of [
      ['section6/hydroelectricenergy', 'Worked plant-design examples', 'square root of slope'],
      ['section6/geothermalenergy', 'Heat transfer, flow area and useful energy', 'square root of thermal diffusivity'],
    ]) {
      await send('Page.navigate', {url: `http://127.0.0.1:18344/${route}`});
      await delay(700);
      let started = false;
      for (let i = 0; i < 120; i++) {
        const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          if (!location.pathname.toLowerCase().includes(${JSON.stringify(route)})) return false;
          const heading = [...document.querySelectorAll('article.content h2, article.content h3')]
            .find(h => h.textContent.includes(${JSON.stringify(title)}));
          const button = heading?.nextElementSibling?.querySelector('button');
          if (!button || button.disabled) return false;
          window.__testUtterances = [];
          window.speechSynthesis.cancel = () => {};
          window.speechSynthesis.speak = utterance => {
            window.__testUtterances.push(utterance.text);
            setTimeout(() => utterance.onend?.(), 0);
          };
          button.click(); return true;
        })()`});
        if (result.result.value) { started = true; break; }
        await delay(100);
      }
      assert.ok(started, `Section reader started for ${title}`);
      let spoken = '';
      for (let i = 0; i < 100; i++) {
        const result = await send('Runtime.evaluate', {returnByValue: true,
          expression: `(window.__testUtterances || []).join(' ')`});
        spoken = result.result.value || '';
        if (spoken.includes(expected)) break;
        await delay(100);
      }
      assert.ok(spoken.includes(expected), `Section reader should include the verbal equation in ${route}`);
      console.log(`${route}: section reader receives the verbal square-root explanation`);
    }
    // The last page is geothermal; exercise its five embedded views in the book itself.
    for (const mode of ['systems', 'conduction', 'area', 'energy', 'performance']) {
      let embedded;
      for (let i = 0; i < 150; i++) {
        const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
          const frame = [...document.querySelectorAll('iframe')].find(f => f.src.includes('closed-loop.html?demo=' + ${JSON.stringify(mode)}));
          if (!frame) return null;
          frame.scrollIntoView({block:'start'});
          const root = frame.contentDocument?.getElementById('explorer');
          if (!root?.querySelector('.loop-card')) return null;
          const contentHeight = root.getBoundingClientRect().height;
          return contentHeight > 0 && frame.clientHeight >= Math.ceil(contentHeight)
            ? {frame: frame.clientHeight, content: contentHeight} : null;
        })()`});
        embedded = result.result.value; if (embedded) break; await delay(100);
      }
      assert.ok(embedded, `Book iframe loads and fits its ${mode} content`);
      console.log(`Standalone geothermal ${mode} view loads and fits its content.`);
    }
  } finally {
    if (socket) socket.close();
    if (process.platform === 'win32' && edge.pid) {
      spawnSync('taskkill', ['/PID', String(edge.pid), '/T', '/F'], {stdio: 'ignore', windowsHide: true});
      // Edge may detach helpers; terminate only processes using this test's unique profile.
      const literalProfile = profile.replaceAll("'", "''");
      spawnSync('powershell.exe', ['-NoProfile', '-Command',
        `Get-CimInstance Win32_Process -Filter "name = 'msedge.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('${literalProfile}') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`],
        {windowsHide: true, stdio: 'ignore', timeout: 10000});
    } else {
      edge.kill();
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

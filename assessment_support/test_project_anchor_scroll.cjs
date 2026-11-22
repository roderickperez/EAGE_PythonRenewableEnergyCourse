// Start the local sandbox on 127.0.0.1:18343 before running on Windows.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-scroll-browser-'));
  const browser = process.env.EAGE_TEST_BROWSER || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const chrome = spawn(browser, ['--headless=new', '--disable-gpu', '--no-first-run',
    '--disable-background-networking', '--remote-debugging-port=0', '--user-data-dir=' + profile,
    'about:blank'], {stdio: 'ignore', windowsHide: true});
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await delay(100);
    assert.ok(fs.existsSync(portFile), 'Chrome debugging endpoint started');
    const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
    const targetURL = 'http://127.0.0.1:18343/#lesson=section8%2FprojectIntro.md&anchor=references';
    const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'});
    assert.ok(response.ok, 'Created project browser tab');
    const target = await response.json();
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let serial = 0;
    const pending = new Map();
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (pending.has(message.id)) {
        const {resolve, reject} = pending.get(message.id);
        pending.delete(message.id);
        message.error ? reject(new Error(message.error.message)) : resolve(message.result);
      }
    };
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      pending.set(id, {resolve, reject});
      socket.send(JSON.stringify({id, method, params}));
    });
    for (const width of [800, 1400]) {
      await send('Emulation.setDeviceMetricsOverride', {width, height: 700, deviceScaleFactor: 1, mobile: false});
      await send('Page.navigate', {url: targetURL});
      let state;
      for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
        const pane = document.getElementById('reading');
        const target = document.getElementById('references');
        if (!pane || !target || !document.body.innerText.includes('Analyse one synthetic week at hourly resolution')) return null;
        const p = pane.getBoundingClientRect(), t = target.getBoundingClientRect();
        return {count: document.querySelectorAll('[id="references"]').length,
          top: t.top, paneTop: p.top, paneBottom: p.bottom, viewport: innerHeight, width: innerWidth,
          scrollTop: pane.scrollTop, paneScrollable: pane.scrollHeight > pane.clientHeight && getComputedStyle(pane).overflowY !== 'visible'};
      })()`});
      state = result.result.value;
      if (state && state.width === width && state.paneScrollable === (width > 1050) && state.top >= 0 && state.top < state.viewport &&
          (!state.paneScrollable || (state.top >= state.paneTop && state.top < state.paneBottom))) break;
      await delay(100);
      }
      assert.ok(state, 'Project and bibliography loaded');
      assert.equal(state.count, 1, 'One bibliography anchor');
      assert.equal(state.width, width);
      assert.equal(state.paneScrollable, width > 1050, JSON.stringify(state));
      assert.ok(state.top >= 0 && state.top < state.viewport, `width ${width}: ${JSON.stringify(state)}`);
      if (state.paneScrollable) assert.ok(state.top >= state.paneTop && state.top < state.paneBottom, JSON.stringify(state));
    }
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=preface%2Fpreface.md'});
    let nodes = [];
    for (let i = 0; i < 100; i++) {
      const tree = await send('Accessibility.getFullAXTree');
      nodes = tree.nodes;
      if (nodes.some(n => n.role?.value === 'image' && /worldwide relative search interest in vibe coding/i.test(n.name?.value || ''))) break;
      await delay(100);
    }
    const images = nodes.filter(n => n.role?.value === 'image').map(n => n.name?.value || '');
    assert.ok(images.some(name => /Screenshot of Andrej Karpathy.s post discussing vibe coding/i.test(name)), JSON.stringify(images));
    assert.ok(images.some(name => /Line chart of worldwide relative search interest in vibe coding through February 2026/i.test(name)), JSON.stringify(images));
    const text = nodes.filter(n => n.role?.value === 'StaticText').map(n => n.name?.value || '').join(' ');
    assert.match(text, /index rose from 3 in February 2025 to 100 in February 2026/i);
    assert.match(text, /not a count of searches or adopters/i);
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=courseIntro.md'});
    let bannerFound = false;
    for (let i = 0; i < 100; i++) {
      const tree = await send('Accessibility.getFullAXTree');
      bannerFound = tree.nodes.some(n => n.role?.value === 'image' && n.name?.value === 'Wind turbines beyond a green field');
      if (bannerFound) break;
      await delay(100);
    }
    assert.ok(bannerFound, 'Course banner has its landscape description in the accessibility tree');
    for (const [lesson, expected] of [
      ['section6/windEnergy.md', 'A loaded rotor slows the air'],
      ['section6/solarEnergy.md', 'A larger mass stores more sensible heat'],
    ]) {
      await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(lesson)});
      let found = false;
      for (let i = 0; i < 100; i++) {
        const tree = await send('Accessibility.getFullAXTree');
        const text = tree.nodes.filter(n => n.role?.value === 'StaticText').map(n => n.name?.value || '').join(' ');
        found = text.includes(expected);
        if (found) break;
        await delay(100);
      }
      assert.ok(found, `${lesson} exposes its recording transcript to assistive technology`);
    }
    socket.close();
    console.log('Project bibliography is visible after navigation at narrow and wide viewport widths.');
    console.log('Preface image descriptions, chart summary and course banner appear in Chrome’s accessibility tree.');
    console.log('Wind and solar recording transcripts appear in Chrome’s accessibility tree.');
  } finally {
    chrome.kill();
    await new Promise(resolve => chrome.once('exit', resolve));
    fs.rmSync(profile, {recursive: true, force: true, maxRetries: 10, retryDelay: 300});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

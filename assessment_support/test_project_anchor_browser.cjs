// Inspect the live sandbox DOM after routing to the project bibliography.
// Start the sandbox on port 18343 before running this Windows Chrome test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const browser = process.env.EAGE_TEST_BROWSER || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-anchor-browser-'));
try {
  const url = 'http://127.0.0.1:18343/#lesson=section8%2FprojectIntro.md&anchor=references';
  const run = spawnSync(browser, ['--headless=new', '--disable-gpu', '--no-first-run',
    '--disable-background-networking', '--user-data-dir=' + profile,
    '--virtual-time-budget=6000', '--dump-dom', url],
    {encoding: 'utf8', timeout: 60000, windowsHide: true, maxBuffer: 16 * 1024 * 1024});
  if (run.error) throw run.error;
  assert.equal(run.status, 0, run.stderr);
  const dom = run.stdout;
  assert.match(dom, /Project 5: Final Project/);
  assert.match(dom, /Analyse one synthetic week at hourly resolution/);
  assert.equal((dom.match(/\bid="references"/g) || []).length, 1, 'one live bibliography anchor');
  const bibliography = dom.indexOf('id="references"');
  assert.match(dom.slice(bibliography, bibliography + 300), /References/);
  assert.match(dom.slice(bibliography, bibliography + 5000), /<li\b/);
  assert.doesNotMatch(dom, /Write your solution: References/);
  console.log('Live project page has one bibliography anchor and no References solution workspace.');
} finally {
  fs.rmSync(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 500});
}

const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../EAGE_pythonSandBox/course-session-refresh.js'), 'utf8');

function setup({ hostname = 'roderickperezaltamar.com', session = null, refreshed = null, probe = false } = {}) {
  const calls = [], listeners = {}, warnings = [];
  const client = { auth: {
    getSession: async () => ({ data: { session }, error: null }),
    refreshSession: async () => { calls.push('refresh'); return { data: { session: refreshed }, error: null }; },
    onAuthStateChange() {}
  } };
  class Image { constructor(src) { this.src = src; this.isConnected = true; } }
  const document = {
    visibilityState: 'visible', head: { append() {} }, body: { prepend(el) { warnings.push(el); } },
    addEventListener(name, fn) { listeners[name] = fn; },
    createElement() { return { append() {}, setAttribute() {}, remove() {} }; }
  };
  const window = { supabase: { createClient: () => client } };
  const fetch = async (url) => {
    calls.push(url);
    if (url === '/api/config') return { ok: true, json: async () => ({ url: 'https://example.test', key: 'publishable' }) };
    if (url === '/api/course-session') return { ok: true };
    if (url === 'generated/course.json') return { ok: probe };
    throw new Error('Unexpected request');
  };
  const origin = `https://${hostname}`;
  const location = { hostname, origin, pathname: '/private-course/python-renewable/index.html', href: origin + '/private-course/python-renewable/index.html' };
  vm.runInNewContext(source, { window, document, location, fetch, URL, HTMLImageElement: Image, WeakSet, Date, JSON, setTimeout, setInterval() {} });
  return { window, calls, listeners, warnings, Image };
}

test('renews a nearly expired session and updates the protected cookie', async () => {
  const t = setup({ session: { access_token: 'old', expires_at: 1 }, refreshed: { access_token: 'new', expires_at: Date.now()/1000 + 3600 } });
  assert.equal(await t.window.refreshCourseSession(), true);
  assert.deepEqual(t.calls, ['/api/config', 'refresh', '/api/course-session']);
});
test('warns if neither a session nor an authorized course is available', async () => {
  const t = setup();
  assert.equal(await t.window.refreshCourseSession(), false);
  assert.equal(t.warnings.length, 1);
});
test('retries an affected course image once after successful renewal', async () => {
  const t = setup({ session: { access_token: 'valid', expires_at: Date.now()/1000 + 3600 } });
  const image = new t.Image('https://roderickperezaltamar.com/private-course/python-renewable/generated/assets/Logo4.png');
  t.listeners.error({ target: image });
  await new Promise(resolve => setImmediate(resolve));
  assert.match(image.src, /imageRetry=/);
  const count = t.calls.length;
  t.listeners.error({ target: image });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(t.calls.length, count);
});
test('does not run the private-session logic on the public course', () => {
  const t = setup({ hostname: 'roderickperez.github.io' });
  assert.equal(t.window.refreshCourseSession, undefined);
});

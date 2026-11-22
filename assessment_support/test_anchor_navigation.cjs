const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('EAGE_pythonSandBox/app.js', 'utf8');
const code = source.slice(source.indexOf('let cancelAnchorFollow'), source.indexOf('async function route()'));
let observer;
class ResizeObserver {
  constructor(callback) { this.callback = callback; observer = this; }
  observe() {}
  disconnect() { this.disconnected = true; }
}
const handlers = {};
const calls = [];
const pane = {
  scrollTop: 0,
  scrollHeight: 2000, clientHeight: 500,
  getBoundingClientRect: () => ({top: 100, bottom: 600}),
  scrollTo: value => { calls.push(value); pane.scrollTop = value.top; },
  addEventListener: (name, handler) => { handlers[name] = handler; },
  removeEventListener: name => { delete handlers[name]; },
};
let top = 1100;
const target = {isConnected: true, getBoundingClientRect: () => ({top})};
const context = {ResizeObserver, setTimeout: () => 1, clearTimeout: () => {}, getComputedStyle: () => ({overflowY: 'auto'})};
vm.createContext(context);
vm.runInContext(code, context);
context.followReadingAnchor(target, pane, {});
assert.equal(calls.length, 1);
top = 800;
observer.callback();
assert.equal(calls.length, 2, 'a late layout shift is corrected');
handlers.wheel();
top = 900;
observer.callback();
assert.equal(calls.length, 2, 'learner interaction cancels automatic following');
assert.equal(observer.disconnected, true);
console.log('Citation navigation follows late layout changes and stops on user input.');

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('EAGE_pythonSandBox/app.js', 'utf8');
const snippet = source.slice(source.indexOf('let initialLessonReady = false;'), source.indexOf('let course, lesson'));
const elements = Object.fromEntries(['workspace-notebook', 'lab', 'workspace-mode-editor', 'workspace-mode-notebook'].map(id => [id, {
  attrs: {}, classList: {toggle() {}}, hidden: false,
  setAttribute(name, value) { this.attrs[name] = value; },
  getAttribute(name) { return this.attrs[name] ?? null; },
  hasAttribute(name) { return name === 'src' ? Boolean(this.src) : name in this.attrs; }
}]));
const context = {$: id => elements[id]};
vm.createContext(context);
vm.runInContext(snippet, context);
vm.runInContext('setWorkspaceMode(true)', context);
assert.equal(elements['workspace-notebook'].src, undefined, 'Do not start Jupyter before lesson layout');
vm.runInContext('setWorkspaceMode(false)', context);
const finish = source.slice(source.lastIndexOf('initialLessonReady = true;'), source.indexOf('// Confirm before leaving'));
assert.match(finish, /workspace-mode-notebook/);
vm.runInContext(finish, context);
assert.equal(elements['workspace-notebook'].src, undefined, 'Preserve an editor selection made while loading');
vm.runInContext('setWorkspaceMode(true)', context);
assert.match(elements['workspace-notebook'].src, /^jupyterlite\//, 'Notebook mode loads Jupyter after layout');
assert.equal(elements['workspace-notebook'].hidden, false);
console.log('Workspace startup defers Jupyter and preserves an early editor selection.');

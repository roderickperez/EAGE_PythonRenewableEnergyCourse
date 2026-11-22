import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const worker = readFileSync('EAGE_pythonSandBox/python-worker.js', 'utf8');
const runSource = worker.slice(worker.indexOf('async function run(message)'), worker.indexOf('self.onmessage='));
assert.ok(runSource.startsWith('async function run(message)'));

async function exercise(code, imports) {
  const pythonCalls = [], packages = [], installed = [], messages = [];
  const py = {
    globals: {set(){}, delete(){}},
    runPython(){return JSON.stringify({magic:false, imports});},
    async loadPackagesFromImports(){},
    async loadPackage(name){packages.push(name);},
    pyimport(){return {async install(spec){installed.push(spec);}, destroy(){}};},
    async runPythonAsync(source){pythonCalls.push(source);},
    setStdin(){},
  };
  const context = {py, assets:[], globals:{}, DISPLAY_SETUP:'display setup',
    outputCharacters:0, context:'', async setContext(){}, async fetchAsset(){},
    send(type, fields={}){messages.push({type,...fields});}};
  vm.runInNewContext(runSource + '\nthis.run = run;', context);
  await context.run({code, path:'section7/test.md', stdin:''});
  assert.equal(messages.at(-1).type, 'done');
  assert.equal(messages.at(-1).error, undefined);
  return {pythonCalls, packages, installed};
}
const plain = await exercise('print("plotly, matplotlib and seaborn are examples")', []);
assert.deepEqual(plain.packages, []);
assert.ok(!plain.pythonCalls.includes('import plotly.io'));
assert.ok(!plain.pythonCalls.some(s=>s.includes("matplotlib.use('Agg')")));
const plotly = await exercise('import os, plotly', ['os','plotly']);
assert.ok(plotly.installed.includes('plotly==6.3.0'));
assert.ok(plotly.pythonCalls.includes('import plotly.io'));
const folium = await exercise('import os, folium', ['os','folium']);
assert.ok(folium.installed.includes('folium==0.20.0'));
const openpyxl = await exercise('import os, openpyxl', ['os','openpyxl']);
assert.ok(openpyxl.installed.includes('openpyxl==3.1.5'));
const seaborn = await exercise('import os, seaborn', ['os','seaborn']);
assert.ok(seaborn.installed.includes('seaborn==0.13.2'));
assert.ok(seaborn.pythonCalls.some(s=>s.includes("matplotlib.use('Agg')")));
console.log('Worker imports optional plotting packages only for inspected imports');

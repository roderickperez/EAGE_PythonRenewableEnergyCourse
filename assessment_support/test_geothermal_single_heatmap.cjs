const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/geothermal-lab/project.js', 'utf8');
const start = source.indexOf(`const hm=$('[data-chart="heatmap"]')`);
const end = source.indexOf('\n }\n function record()', start);
assert.ok(start >= 0 && end > start, 'single-depth heatmap source found');
const code = source.slice(start, end);
function draw(times) {
  const rects = [];
  const context = {
    clearRect() {}, fillRect(x, y, width, height) { rects.push({x, y, width, height}); },
    fillText() {},
  };
  const history = times.map((years, i) => ({years, production: 150 - i}));
  vm.runInNewContext(code, {
    $: () => ({getContext: () => context}),
    t: history, s: {years: times.at(-1), production: 150}, config: {initial: 160, depth: 2000},
    Math,
  });
  return rects.filter(r => r.y === 10 && r.height === 15);
}
const unequal = draw([0, 1, 10]);
assert.deepEqual(unequal.map(r => r.width), [27, 265, 239]);
const simultaneous = draw([0, 0, 2]);
assert.ok(simultaneous.every(r => r.width >= 0));
assert.equal(simultaneous[0].width, 0);
console.log('Single-depth heatmap uses simulated time and handles simultaneous samples.');

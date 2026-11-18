import test from 'node:test';
import assert from 'node:assert/strict';
import { demos } from './equation-demos.js';
import { renderEquationVisual } from './equation-visuals.js';
const inputs = demo => Object.fromEntries(demo.controls.map(([key,,,,,value]) => [key,value]));
test('worked examples match geothermal lesson calculations', () => {
  assert.match(demos.heat.calculate(inputs(demos.heat)).headline, /16\.72 MW/);
  assert.match(demos.conversion.calculate(inputs(demos.conversion)).headline, /1\.806 MW/);
  assert.match(demos.gradient.calculate(inputs(demos.gradient)).equation, /0\.0875 W/);
  assert.match(demos.pump.calculate(inputs(demos.pump)).headline, /0\.133 MW/);
  assert.match(demos.heatpump.calculate(inputs(demos.heatpump)).headline, /1\.75 MW/);
});
test('all controls update their calculated quantity without invalid values', () => {
  for (const [name, demo] of Object.entries(demos)) {
    const base = inputs(demo);
    for (const [id,,min,max] of demo.controls) {
      const low = demo.calculate({...base,[id]:min});
      const high = demo.calculate({...base,[id]:max});
      assert.notEqual(low.equation, high.equation, `${name}: ${id}`);
      assert.doesNotMatch(low.equation + high.equation, /NaN|Infinity/);
    }
  }
});

test('each equation diagram responds to every one of its sliders', () => {
  function draw(name, demo, input) {
    const svg = { setAttribute() {}, pauseAnimations() {}, unpauseAnimations() {} };
    renderEquationVisual(name, svg, input, demo.calculate(input).values, true);
    assert.doesNotMatch(svg.innerHTML, /NaN|Infinity|undefined/);
    return svg.innerHTML;
  }
  for (const [name, demo] of Object.entries(demos)) {
    const base = inputs(demo);
    for (const [id, , min, max] of demo.controls) {
      assert.notEqual(draw(name,demo,{...base,[id]:min}), draw(name,demo,{...base,[id]:max}), name + ': ' + id);
    }
  }
});

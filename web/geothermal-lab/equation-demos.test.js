import test from 'node:test';
import assert from 'node:assert/strict';
import { demos } from './equation-demos.js';
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

test('conversion warns when efficiency exceeds the specified sink Carnot bound', () => {
  const a = {...inputs(demos.conversion), flow: 50, reservoir: 90, reinjection: 80,
             sink: 25, efficiency: 20, parasitic: 0};
  const high = demos.conversion.calculate(a);
  assert.match(high.headline, /0\.418 MW/);
  assert.match(high.equation, /Carnot upper bound = 17\.9%/);
  assert.match(high.equation, /physically infeasible/);
  const stopped = demos.conversion.calculate({...a, flow: 0});
  assert.match(stopped.headline, /Pₙₑₜ = 0 MW/);
  assert.match(stopped.equation, /positive output at this efficiency would be physically infeasible/);
  assert.doesNotMatch(stopped.equation, /this output is physically infeasible/);
  const lower = demos.conversion.calculate({...a, efficiency: 10});
  assert.doesNotMatch(lower.equation, /physically infeasible/);
  assert.match(lower.equation, /does not establish feasibility/);
  const belowSink = demos.conversion.calculate({...a, efficiency: 10, reinjection: 20});
  assert.match(belowSink.equation, /passive heat exchange/);
  assert.match(belowSink.equation, /does not account for such work/);
  assert.match(demos.conversion.calculate({...a, reinjection: 25}).equation, /passive heat exchange/);
  assert.doesNotMatch(demos.conversion.calculate({...a, reinjection: 26}).equation, /passive heat exchange/);
});

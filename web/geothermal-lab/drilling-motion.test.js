import { test } from 'node:test';
import assert from 'node:assert/strict';
import { drilling, drillingMotion } from './drilling-model.js';

test('animation follows signed transport and hides invalid Stokes estimates', () => {
  const input = { depth: 2000, flow: 100, hole: .35, pipe: .08, density: 1100, friction: 1, viscosity: .03, particle: .0009 };
  const downward = drilling(input);
  assert.equal(downward.validStokes, true);
  assert.ok(downward.transport < 0);
  assert.ok(drillingMotion(downward).particleRate < 0);
  const stoppedFlow = drilling({...input, flow: 0});
  assert.equal(drillingMotion(stoppedFlow).fluidRate, 0);
  assert.ok(drillingMotion(stoppedFlow).particleRate < 0);
  const upward = drilling({...input, flow: 3000});
  assert.ok(drillingMotion(upward).particleRate > 0);
  const outside = drilling({...input, particle: .003, viscosity: .005});
  assert.equal(outside.validStokes, false);
  assert.equal(drillingMotion(outside).particleVisible, false);
});

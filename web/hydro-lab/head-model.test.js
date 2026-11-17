import assert from 'node:assert/strict';
import { test } from 'node:test';
import { computeHead } from './head-model.js';

test('pressure is converted from kPa and all three heads are summed in metres', () => {
  const { elevation, pressure, kinetic, total } = computeHead({
    elevation: 15, pressureKPa: 100, velocity: 5, density: 1000, gravity: 9.81,
  });
  assert.equal(elevation, 15);
  assert.ok(Math.abs(pressure - 10.19367991845) < 1e-9);
  assert.ok(Math.abs(kinetic - 1.27420998981) < 1e-9);
  assert.ok(Math.abs(total - 26.46788990826) < 1e-9);
});

test('with zero gauge pressure and velocity, only elevation contributes', () => {
  assert.equal(computeHead({
    elevation: 50, pressureKPa: 0, velocity: 0, density: 950, gravity: 9.75,
  }).total, 50);
});

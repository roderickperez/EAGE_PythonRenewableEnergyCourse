import test from 'node:test';
import assert from 'node:assert/strict';
import { computeHydro, normalizeHydroInputs } from './model.js';

test('default example reproduces the requested capped plant', () => {
  const result = computeHydro();
  assert.equal(result.netHeadM, 191);
  assert.ok(Math.abs(result.uncappedMW - 15.926535) < 1e-9);
  assert.equal(result.plantMW, 2);
  assert.ok(Math.abs(result.turbineFlowM3s - 1.255765926) < 1e-8);
  assert.ok(Math.abs(result.spillFlowM3s - 8.744234074) < 1e-8);
  assert.equal(result.energy6hMWh, 12);
  assert.equal(result.clipped, true);
});

test('an adequately rated plant passes all usable flow', () => {
  const result = computeHydro({ grossHeadM: 193, usableFlowM3s: 10, headLossM: 2, ratingMW: 20 });
  assert.equal(result.plantMW, result.uncappedMW);
  assert.equal(result.turbineFlowM3s, 10);
  assert.equal(result.spillFlowM3s, 0);
  assert.equal(result.clipped, false);
});

test('zero usable flow gives zero power and energy', () => {
  const result = computeHydro({ usableFlowM3s: 0 });
  assert.equal(result.uncappedMW, 0);
  assert.equal(result.plantMW, 0);
  assert.equal(result.energy6hMWh, 0);
  assert.equal(result.spillFlowM3s, 0);
});

test('loss is constrained below gross head and non-finite inputs fall back', () => {
  const inputs = normalizeHydroInputs({ grossHeadM: 5, headLossM: 500, usableFlowM3s: NaN, ratingMW: Infinity });
  assert.equal(inputs.grossHeadM, 5);
  assert.equal(inputs.headLossM, 4.5);
  assert.equal(inputs.usableFlowM3s, 10);
  assert.equal(inputs.ratingMW, 2);
  assert.equal(computeHydro(inputs).netHeadM, 0.5);
});

test('low rating explicitly routes excess usable flow to spill or bypass', () => {
  const result = computeHydro({ grossHeadM: 30, headLossM: 2, usableFlowM3s: 10, ratingMW: 0.5 });
  assert.equal(result.plantMW, 0.5);
  assert.ok(result.turbineFlowM3s < result.inputs.usableFlowM3s);
  assert.ok(result.spillFlowM3s > 0);
});

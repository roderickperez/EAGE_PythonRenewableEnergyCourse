import assert from 'node:assert/strict';
import { test } from 'node:test';
import { flowBalance, netHead, pipeLoss, powerBalance, turbineOperation } from './balance-model.js';

test('net head subtracts the loss once and reaches zero at full loss', () => {
  assert.equal(netHead(30, 2), 28);
  assert.equal(netHead(30, 0), 30);
  assert.equal(netHead(30, 30), 0);
});

test('three cubic metres per second over two hours matches the lesson example', () => {
  assert.deepEqual(flowBalance(3, 120, 1000), { seconds: 7200, volume: 21600, massFlow: 3000 });
});

test('time changes total volume while density changes only mass flow', () => {
  assert.deepEqual(flowBalance(3, 0, 1000), { seconds: 0, volume: 0, massFlow: 3000 });
  assert.deepEqual(flowBalance(3, 60, 1050), { seconds: 3600, volume: 10800, massFlow: 3150 });
  assert.deepEqual(flowBalance(0, 60, 1050), { seconds: 3600, volume: 0, massFlow: 0 });
});

test('power stays in watts and each efficiency is applied once', () => {
  const power = powerBalance(10, 28, 1000, 9.81, .85, .95);
  assert.ok(Math.abs(power.hydraulicW - 2746800) < 1e-8);
  assert.ok(Math.abs(power.shaftW - 2334780) < 1e-8);
  assert.ok(Math.abs(power.electricalW - 2218041) < 1e-8);
  assert.equal(powerBalance(10, 28, 1000, 9.81, .85, 0).electricalW, 0);
  assert.equal(powerBalance(10, 28, 1000, 9.81, 0, .95).shaftW, 0);
  assert.equal(powerBalance(0, 28, 1000, 9.81).hydraulicW, 0);
});

test('Darcy and fitting losses are both quadratic when geometry is fixed', () => {
  const original = pipeLoss(4, 250, 1.5, .02, 2, 9.81);
  const doubled = pipeLoss(8, 250, 1.5, .02, 2, 9.81);
  assert.ok(Math.abs(original.area - 1.7671458676) < 1e-9);
  assert.equal(doubled.friction, original.friction * 4);
  assert.equal(doubled.fittings, original.fittings * 4);
  assert.equal(pipeLoss(0, 250, 1.5, .02, 2, 9.81).total, 0);
});

test('turbine operation keeps the rating and closes the flow balance', () => {
  const capped = turbineOperation(10, 28, .85, 2, 1000, 9.81);
  assert.ok(Math.abs(capped.outputMW - 2) < 1e-12);
  assert.equal(capped.turbineFlow + capped.bypassFlow, 10);
  assert.equal(turbineOperation(10, 28, .85, 5, 1000, 9.81).bypassFlow, 0);
  assert.deepEqual(turbineOperation(10, 0, .85, 2, 1000, 9.81),
    { ratedFlow: null, turbineFlow: 0, bypassFlow: 10, outputMW: 0 });
});

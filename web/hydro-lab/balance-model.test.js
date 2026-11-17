import assert from 'node:assert/strict';
import { test } from 'node:test';
import { flowBalance, flowDuration, netHead, pipeHeadProfile, pipeLoss, powerBalance, pumpedCycle, pumpedStorage, turbineOperation } from './balance-model.js';

test('duration ranks preserve observation identity, including ties and endpoints', () => {
  const flows = [12, 8, 12, 20];
  assert.deepEqual(flowDuration(flows), [
    { day: 4, flow: 20, rank: 1, exceedancePercent: 20 },
    { day: 1, flow: 12, rank: 2, exceedancePercent: 40 },
    { day: 3, flow: 12, rank: 3, exceedancePercent: 60 },
    { day: 2, flow: 8, rank: 4, exceedancePercent: 80 },
  ]);
  assert.deepEqual(flows, [12, 8, 12, 20]);
});

test('pumped storage accounts for input, stored and recovered energy at separate boundaries', () => {
  const { stored, input, recovered, roundTrip } = pumpedStorage(120000, 80, .85, .9, .95, 1000, 9.81);
  assert.equal(stored, 26.16);
  assert.ok(Math.abs(input - 30.7764705882353) < 1e-9);
  assert.ok(Math.abs(recovered - 22.3668) < 1e-9);
  assert.ok(Math.abs(recovered / input - roundTrip) < 1e-12);
  assert.equal(pumpedStorage(120000, 0, .85, .9, .95, 1000, 9.81).input, 0);
});

test('the upper reservoir fills while pumping and drains during generation', () => {
  assert.deepEqual(pumpedCycle(0), { pumping: true, upper: 0 });
  assert.deepEqual(pumpedCycle(.25), { pumping: true, upper: .5 });
  assert.deepEqual(pumpedCycle(.5), { pumping: false, upper: 1 });
  assert.deepEqual(pumpedCycle(.75), { pumping: false, upper: .5 });
  assert.equal(pumpedCycle(1).upper, 0);
});

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

test('the pressure taps span friction and the lumped fitting drop', () => {
  const [[start, inlet], [fitting, before], [sameFitting, after], [end, outlet]] =
    pipeHeadProfile({ friction: 3, fittings: 2, total: 5 }, 250);
  assert.deepEqual([start, fitting, sameFitting, end], [0, 200, 200, 250]);
  assert.equal(outlet, 2);
  assert.equal(inlet - outlet, 5);
  assert.equal(before - after, 2);
  assert.deepEqual(pipeHeadProfile({ friction: 0, fittings: 0, total: 0 }, 250).map(([, head]) => head), [2, 2, 2, 2]);
});

test('turbine operation keeps the rating and closes the flow balance', () => {
  const capped = turbineOperation(10, 28, .85, 2, 1000, 9.81);
  assert.ok(Math.abs(capped.outputMW - 2) < 1e-12);
  assert.equal(capped.turbineFlow + capped.bypassFlow, 10);
  assert.equal(turbineOperation(10, 28, .85, 5, 1000, 9.81).bypassFlow, 0);
  assert.deepEqual(turbineOperation(10, 0, .85, 2, 1000, 9.81),
    { ratedFlow: null, turbineFlow: 0, bypassFlow: 10, outputMW: 0 });
});

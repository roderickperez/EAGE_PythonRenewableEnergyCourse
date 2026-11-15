import test from 'node:test';
import assert from 'node:assert/strict';
import { electricalPower, operatingPoint, annualEnergy, circularMean, YEAR_H } from './model.js';

const base = {
  cutIn: 3, ratedSpeed: 12, cutOut: 25, ratedMW: 3, outputMode: 'curve', heightMode: 'hub',
  pressure: 1.225 * 287.05 * 288.15 / 1000, temperature: 15, speed: 8,
  hub: 90, reference: 10, displacement: 0, roughness: .05, alpha: .14,
  radius: 20, cp: .4, efficiency: 95, tsr: 7,
  shape: 2, scale: 8.5, availability: 100, wake: 0, loss: 0,
};
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≠ ${expected}`);

test('matches the lesson’s electrical curve and exact operating boundaries', () => {
  close(electricalPower(3, base), 0);
  close(electricalPower(8, base), 3 * (512 - 27) / (1728 - 27));
  close(electricalPower(12, base), 3);
  close(electricalPower(24.9, base), 3);
  close(electricalPower(25, base), 0);
  close(operatingPoint({ ...base, speed: 25 }).omega, 0);
});

test('reproduces the worked aerodynamic example and keeps the models independent', () => {
  const p = operatingPoint({ ...base, outputMode: 'aero' });
  close(p.windMW, Math.PI * 20 ** 2 * .5 * 1.225 * 8 ** 3 / 1e6);
  close(p.gridMW, p.windMW * .4 * .95);
  assert.equal((p.gridMW * 2000).toFixed(1), '299.5');
  close(operatingPoint(base).gridMW, operatingPoint({ ...base, cp: .1 }).gridMW);
  close(operatingPoint({ ...base, heightMode: 'power', reference: base.hub }).speed, 8);
  close(operatingPoint({ ...base, heightMode: 'log', reference: base.hub }).speed, 8);
});

test('annual integration respects generating probabilities and sequential losses', () => {
  const gross = annualEnergy(base);
  const probability = Math.exp(-((3/8.5)**2)) - Math.exp(-((25/8.5)**2));
  assert.ok(gross.gross > 0 && gross.gross < YEAR_H * probability * base.ratedMW);
  close(gross.gross, annualEnergy({ ...base, speed: 25 }).gross);
  close(annualEnergy({ ...base, availability: 90, wake: 10, loss: 10 }).net, gross.gross * .9 ** 3);
  close(circularMean(350, 10).degrees, 0);
  assert.equal(circularMean(90, 270).degrees, null);
});

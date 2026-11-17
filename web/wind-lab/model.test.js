import test from 'node:test';
import assert from 'node:assert/strict';
import { electricalPower, operatingPoint, annualEnergy, circularMean, YEAR_H } from './model.js';
import { BETZ, availableWind, aerodynamicConversion, dryAirDensity, powerLawWind,
  logLawWind, twoIntervalWind, weibullPowerBins, annualEnergyScenarios } from './model.js';

const base = {
  cutIn: 3, ratedSpeed: 12, cutOut: 25, ratedMW: 3, outputMode: 'curve', heightMode: 'hub',
  pressure: 1.225 * 287.05 * 288.15 / 1000, temperature: 15, speed: 8,
  hub: 90, reference: 10, displacement: 0, roughness: .05, alpha: .14,
  radius: 20, cp: .4, efficiency: 95, tsr: 7,
  shape: 2, scale: 8.5, availability: 100, wake: 0, loss: 0,
};
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≠ ${expected}`);

test('cube law and ideal aerodynamic bounds match the lesson derivation', () => {
  const initial = availableWind(8, 1.225, 20);
  close(availableWind(16, 1.225, 20).windMW / initial.windMW, 8);
  close(availableWind(8, 1.225, 40).windMW / initial.windMW, 4);
  const limit = aerodynamicConversion(initial.windMW, BETZ, .9);
  close(limit.rotorMW, initial.windMW * 16 / 27);
  close(limit.electricalMW + limit.drivetrainLossMW, limit.rotorMW);
});

test('dry-air, shear and turbulence calculations honor input and averaging conventions', () => {
  assert.equal(dryAirDensity(101.325, 15).toFixed(3), '1.225');
  assert.ok(dryAirDensity(90, 25) < dryAirDensity(101.325, 15));
  close(powerLawWind(7, 30, 30, .14), 7);
  close(logLawWind(7, 30, 30, .12, 3), 7);
  const intervals = twoIntervalWind(4, 8);
  close(intervals.intensity, 1 / 3);
  close(intervals.meanCubedSpeeds, 288);
  close(intervals.cubedMeanSpeed, 216);
  assert.equal(twoIntervalWind(0, 0).intensity, null);
});

test('Weibull bins account for probability and P90 is the lower energy quantile', () => {
  const result = weibullPowerBins({ shape: 2, scale: 8.5, count: 600, maximum: 30, turbine: base });
  close(result.bins.reduce((sum, bin) => sum + bin.probability, 0) + result.tail, 1);
  assert.ok(result.bins.filter(bin => bin.lower >= base.cutOut).every(bin => bin.contribution === 0));
  assert.ok(Math.abs(result.meanMW - annualEnergy(base).meanMW) < .001);
  const projection = annualEnergyScenarios(12, .96, .08, .04, .1);
  close(projection.p50, 12 * .96 * .92 * .96);
  close(projection.p50 - projection.p90, 1.2815515655 * projection.deviation);
  const fixed = annualEnergyScenarios(12, 1, 0, 0, 0);
  close(fixed.p90, fixed.p50);
});

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

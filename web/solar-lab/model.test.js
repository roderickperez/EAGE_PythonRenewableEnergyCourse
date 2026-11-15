import test from 'node:test';
import assert from 'node:assert/strict';
import { solarPower, solarDay, batteryDay, lifetime } from './model.js';

const worked = {
  hour: 12, latitude: 0, declination: 0, tilt: 0, azimuth: 180,
  dni: 700, dhi: 100, albedo: .2, ambient: 25, noct: 45, coefficient: -.4,
  area: 50, efficiency: 20, soiling: 0, wiring: 0, inverterEfficiency: 97, rating: 8000,
};
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≠ ${expected}`);

test('reproduces the lesson’s 10 kW worked example and horizontal POA equals GHI', () => {
  const p = solarPower(worked);
  close(p.poa, 800);
  close(p.poa, p.ghi);
  close(p.cell, 50);
  close(p.dc, 7200);
  close(p.ac, 6984);
  close(p.ac * .5 / 1000, 3.492);
});

test('nighttime, back-facing beam and clipping are handled independently', () => {
  close(solarPower(worked, 0).ac, 0);
  const facing = { ...worked, latitude: 35, tilt: 90 };
  assert.ok(solarPower(facing).beam > 0);
  close(solarPower({ ...facing, azimuth: 0 }).beam, 0);
  const clipped = solarPower({ ...worked, rating: 6000 });
  close(clipped.ac, 6000);
  close(clipped.clip, 984);
  assert.equal(solarDay({ ...worked, dni: 0, dhi: 0 }).pr, null);
});

test('day integration is independent of playback and storage balances without a battery', () => {
  const day = solarDay(worked);
  close(day.energy, solarDay({ ...worked, hour: 0 }).energy);
  const battery = batteryDay(day.samples, { load: 1000, storage: 0, batteryPower: 300, batteryEfficiency: 90 });
  close(battery.used + battery.unmet, 24);
  close(battery.used + battery.surplus, day.energy);
  close(battery.stored, 0);
  close(lifetime({ firstYear: 100, degradation: 0, years: 2, capital: 300, maintenance: 50, discount: 0 }).lcoe, 2);
});

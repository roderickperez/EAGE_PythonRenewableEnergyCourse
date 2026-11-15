import test from 'node:test';
import assert from 'node:assert/strict';
import { defaults, geothermal } from './model.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≠ ${expected}`);

test('reproduces the lesson example and balances thermal and gross electrical power', () => {
  const result = geothermal(defaults);
  close(result.thermal, 16.72);
  close(result.gross, 2.0064);
  close(result.parasitic, .20064);
  close(result.exported, 1.80576);
  close(result.interval, 7.22304);
  close(result.rejected + result.gross, result.thermal);
});

test('zero flow and no usable temperature difference produce no electricity', () => {
  close(geothermal({ ...defaults, flow: 0, loadMode: 'explicit' }).net, 0);
  const belowSink = geothermal({ ...defaults, reservoir: 65, wellLoss: 40, sink: 45 });
  close(belowSink.thermal, 0);
  close(belowSink.exported, 0);
  close(geothermal({ ...defaults, reinjection: 180 }).exported, 0);
});

test('pump calculation uses volume flow and distinguishes imports and export clipping', () => {
  const operating = geothermal({ ...defaults, loadMode: 'explicit', rating: 1 });
  close(operating.pump, 2 * (50 / 1000) / .75);
  close(operating.parasitic, operating.pump + .05);
  close(operating.exported, 1);
  close(operating.curtailed, operating.net - 1);
  const importing = geothermal({ ...defaults, loadMode: 'explicit', auxiliaries: 4 });
  close(importing.exported, 0);
  close(importing.imported, importing.parasitic - importing.gross);
});

test('reversible bound uses Kelvin and annual energy compounds from year one', () => {
  const result = geothermal({ ...defaults, efficiency: 30, years: 2 });
  close(result.carnot, 1 - 298.15 / 423.15);
  assert.ok(result.conversion < result.carnot);
  close(result.conversion, result.streamLimit);
  close(result.annual[0], result.exported * 8760 * .95);
  close(result.annual[1], result.annual[0] * .98);
  close(result.lifetime, result.annual[0] + result.annual[1]);
  close(result.heatFlux, 87.5);
});

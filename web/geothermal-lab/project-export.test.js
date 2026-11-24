import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createModel, defaults, projectCsv } from './project-model.js';

test('single-layer export identifies flow, ordering and sampled output', () => {
  const model = createModel(defaults);
  const snapshots = [model.snapshot(), model.step(1)];
  const [head, ...lines] = projectCsv(snapshots, model.config).split('\n');
  const names = head.split(',');
  const rows = lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [names[i], Number(value)])));
  assert.deepEqual(rows.map(row => row.record_index), [0, 1]);
  assert.equal(rows[0].configured_depth_count, 1);
  assert.equal(rows[0].active_depth_count, 1);
  assert.equal(rows[0].mass_flow_kg_s, rows[0].total_flow_kg_s);
  for (const row of rows) {
    const thermal = Math.max(0, row.total_flow_kg_s * 4180 * (row.production_C - row.injection_C) / 1e6);
    const net = Math.min(row.capacity_MW, thermal * row.conversion_efficiency * (1 - row.auxiliary_fraction));
    assert.ok(Math.abs(net - row.net_power_MW) < 1e-12);
  }
  const still = createModel({...defaults, rate: 0});
  const row = projectCsv([still.snapshot()], still.config).split('\n')[1].split(',');
  assert.equal(Number(row[names.indexOf('active_depth_count')]), 0);
});

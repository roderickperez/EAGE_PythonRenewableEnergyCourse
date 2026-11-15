import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { joinProjectInputs, schemas } from './data.js';

const directory = new URL('../../EAGE_PythonRenewableEnergyCourse/section8/data/',import.meta.url);
const inputs = Object.fromEntries(Object.keys(schemas).map(name =>
  [name,readFileSync(new URL(name+'.csv',directory),'utf8')]));

test('uses the same 168 UTC hours as the project, even when an input file is reordered',()=>{
  const [header,...records]=inputs.demand.trim().split(/\r?\n/);
  const rows=joinProjectInputs({...inputs,demand:[header,...records.reverse()].join('\n')});
  assert.equal(rows.length,168);
  assert.equal(new Date(rows[0].time).toISOString(),'2025-06-01T00:00:00.000Z');
  assert.equal(new Date(rows.at(-1).time).toISOString(),'2025-06-07T23:00:00.000Z');
  assert.equal(rows[0].demand_mw,26.75);
  assert.equal(rows[0].wind_speed_ms,2.03823731);
  assert.equal(rows[0].river_flow_m3s,24.45542261);
  assert.equal(rows[0].production_c,162);
  assert.equal(rows[0].irradiance_wm2,0);
});

test('fails clearly on time mismatch or missing observations',()=>{
  assert.throws(()=>joinProjectInputs({...inputs,solar:inputs.solar.replace('2025-06-01 00:00','2025-06-08 00:00')}),/same UTC hours/);
  assert.throws(()=>joinProjectInputs({...inputs,wind:inputs.wind.trim().split(/\r?\n/).slice(0,-1).join('\n')}),/168 records/);
});

import {test} from 'node:test';import assert from 'node:assert/strict';import {conduction,areas,energy} from './closed-loop-model.js';
import { outletAt, levelizedCost } from './closed-loop-performance-model.js';
test('tube areas reproduce the course example',()=>{assert.ok(Math.abs(areas({inner:.1,outer:.12,bore:.18}).ratio-1.8)<1e-12);assert.throws(()=>areas({inner:.1,outer:.2,bore:.18}));});
test('transient conduction approaches prescribed wall temperature',()=>{const x={k:3,rho:2750,cp:790,days:365,wall:40,far:200};assert.ok(Math.abs(conduction({...x,distance:0}).temperature-40)<.001);assert.ok(conduction({...x,distance:1}).temperature>40);assert.ok(conduction({...x,distance:1}).temperature<200);assert.ok(conduction({...x,days:30,distance:1}).temperature>conduction({...x,days:365,distance:1}).temperature);});
test('heat and exergy rates preserve the signed entropy term',()=>{const r=energy({flow:50,dh:300,ds:.2,ambient:300,years:1});assert.equal(r.thermal,15);assert.equal(r.available,12);assert.equal(r.thermalMWh,131400);});

test('outlet temperatures interpolate between rounded Figure 3 readings', () => {
  assert.deepEqual(outletAt(.25), [165, 93]);
  assert.deepEqual(outletAt(7.5), [138.5, 75.5]);
  assert.deepEqual(outletAt(40), [129, 70]);
});

test('LCOE discounts annual costs and electricity while charging year-zero capital once', () => {
  const inputs = {capital:10, operating:1, generation:5, decline:0, discount:0, years:2};
  assert.deepEqual(levelizedCost(inputs).at(-1), {year:2,cost:12e6,energy:10000,lcoe:1200});
  const discounted = levelizedCost({...inputs,discount:10}).at(-1);
  assert.ok(discounted.cost < 12e6);
  assert.ok(discounted.energy < 10000);
  assert.ok(discounted.lcoe > 1200);
  assert.ok(levelizedCost({...inputs,decline:10}).at(-1).lcoe > 1200);
});

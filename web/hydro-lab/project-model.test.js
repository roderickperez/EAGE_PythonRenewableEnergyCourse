import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CAPACITY, DEAD, DT, initialState, step } from './project-model.js';

test('daily routing conserves water and enforces the turbine reserve', () => {
  for (const request of [0, 4, 10]) {
    let s=initialState();
    for(let i=0;i<365;i++) {
      const next=step(s,request);
      assert.ok(Math.abs(s.storage+next.inflow*DT-next.storage-next.environmental*DT-next.turbineFlow*DT-next.spillFlow*DT)<1e-5);
      assert.ok(next.storage>=DEAD-1e-6 && next.storage<=CAPACITY);
      assert.ok(next.turbineFlow<=request+1e-9 && next.energy>=s.energy);
      assert.ok(next.efficiency <= (s.efficiency ?? .91));
      s=next;
    }
  }
});

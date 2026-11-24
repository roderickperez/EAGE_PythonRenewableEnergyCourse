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
test('dry inflows expose shortage while preserving daily water balance and release priority',()=>{
 let s=initialState();
 for(let i=0;i<365;i++){
  const before=s; s=step(s,4,.1);
  assert.ok(Math.abs(before.storage+s.inflow*DT-s.storage-s.environmental*DT-s.turbineFlow*DT-s.spillFlow*DT)<1e-5);
  assert.ok(s.storage>=0 && s.environmental>=0 && s.environmental<=1);
  if(s.environmental<1) assert.equal(s.turbineFlow,0);
 }
 assert.ok(s.shortage>0);
 assert.throws(()=>step(initialState(),4,-1),RangeError);
});

test('efficiency decline follows simulated days even with no turbine flow',()=>{
  let state=initialState();
  for(let i=0;i<101;i++) {
    state=step(state,0);
    assert.equal(state.turbineFlow,0);
  }
  assert.ok(Math.abs(state.efficiency-.90)<1e-12);
  assert.equal(state.energy,0);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { computeHydro } from './model.js';
import { visualState, INTAKE_CENTER_Y, INTAKE_RADIUS } from './visual-state.js';

test('fixed intake is submerged throughout the operating head range',()=>{
  let previous=0;
  for(let grossHeadM=5;grossHeadM<=250;grossHeadM++){
    const state=visualState(computeHydro({grossHeadM}));
    assert.ok(state.surfaceY>INTAKE_CENTER_Y+INTAKE_RADIUS);
    assert.ok(state.surfaceY>previous); previous=state.surfaceY;
  }
});
test('reported low-head case has positive submerged operation; zero flow de-energizes',()=>{
  const result=computeHydro({grossHeadM:21,usableFlowM3s:3,headLossM:20.5,ratingMW:25});
  assert.equal(result.netHeadM,.5);
  assert.equal(visualState(result).energized,true);
  const stopped=visualState(computeHydro({usableFlowM3s:0}));
  assert.equal(stopped.energized,false); assert.equal(stopped.turbineFlow,0); assert.equal(stopped.bypassFlow,0);
});

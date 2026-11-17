import test from 'node:test';
import assert from 'node:assert/strict';
import { gridPoint, powerKw, evaluate, validCandidate, rankCandidates, windFieldSampler } from './project-model.js';
import { wakePixels, particleWind, advanceParticle, plasma } from './project-field.js';

test('coordinates and the teaching turbine curve are explicit',()=>{
  assert.deepEqual(gridPoint(0),{id:0,x:100,y:100});
  assert.deepEqual(gridPoint(99),{id:99,x:1900,y:1900});
  assert.equal(powerKw(2.9),0); assert.equal(powerKw(12),518.4); assert.equal(powerKw(25),518.4); assert.equal(powerKw(25.1),0);
});

test('northern wind wakes the downstream turbine; rotating wind changes the layout',()=>{
  const north=evaluate([95,85],0,12);
  assert.equal(north.results[0].speed,12);
  assert.ok(north.results[1].speed<12);
  assert.ok(north.total<north.ideal);
  const east=evaluate([95,85],90,12);
  assert.ok(Math.abs(east.total-east.ideal)<1e-8);
  assert.equal(evaluate([],0,12).efficiency,null);
  assert.equal(evaluate([95,85],0,0).efficiency,null);
});

test('spacing excludes duplicates and close neighbors',()=>{
  assert.equal(validCandidate(0,[0],200),false);
  assert.equal(validCandidate(1,[0],200),true);
  assert.equal(validCandidate(1,[0],240),false);
  assert.equal(validCandidate(2,[0],240),true);
  assert.throws(()=>evaluate([0,0],0,12),/Duplicate/);
});

test('recommendations score all valid next locations without altering the layout',()=>{
  const ids=[95], before=[...ids];
  const ranking=rankCandidates(ids,0,12,200);
  assert.equal(ranking.length,99); assert.deepEqual(ids,before);
  assert.equal(ranking[0].total,evaluate([...ids,ranking[0].id],0,12).total);
  assert.ok(ranking.every(p=>p.total<=ranking[0].total+1e-8));
  assert.ok(ranking.some(p=>p.total<ranking[0].total));
});

test('colored wake samples agree with turbine speeds and rotate with the wind',()=>{
  const ids=[95,85], result=evaluate(ids,0,12), sample=windFieldSampler(ids,0,12);
  for(const turbine of result.results) assert.ok(Math.abs(sample(turbine.x,turbine.y)-turbine.speed)<1e-9);
  const p=gridPoint(95), north=windFieldSampler([95],0,12), east=windFieldSampler([95],90,12);
  assert.ok(north(p.x,p.y-200)<12); assert.equal(north(p.x+200,p.y),12);
  assert.ok(east(p.x-200,p.y)<12); assert.equal(east(p.x,p.y-200),12);
  assert.ok(wakePixels([95],0,12).some((v,i)=>i%4===3 && v>0));
  assert.ok(wakePixels([],0,12).every(v=>v===0)); assert.ok(wakePixels([95],0,0).every(v=>v===0));
  assert.deepEqual(plasma(0),[13,8,135]); assert.deepEqual(plasma(1),[240,249,33]);
});

test('particles follow wind direction and grow and move faster at higher speeds',()=>{
  const north=particleWind(0,5), east=particleWind(90,10);
  assert.ok(north.z>0 && Math.abs(north.x)<1e-9);
  assert.ok(east.x<0 && Math.abs(east.z)<1e-9);
  assert.ok(east.length>north.length && Math.abs(east.x)>north.z);
  assert.ok(advanceParticle({x:9.9,z:0},particleWind(270,10),1).x<0);
});

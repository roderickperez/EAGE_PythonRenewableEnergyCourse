import test from 'node:test';
import assert from 'node:assert/strict';
import { gridPoint, powerKw, evaluate, validCandidate, rankCandidates } from './project-model.js';

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

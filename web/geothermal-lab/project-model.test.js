import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createModel,flowNetwork,defaults} from './project-model.js';

test('balanced network has no boundary leak and meets well flows',()=>{
 const f=flowNetwork(43,46,50);assert.ok(f.residual<1e-7);assert.ok(Math.abs(f.divergence[43]-50)<1e-7);assert.ok(Math.abs(f.divergence[46]+50)<1e-7);
});
test('initial power and energy account availability once',()=>{
 const m=createModel(),s=m.step(1);assert.ok(Math.abs(s.energy-43.879968)<1e-5);assert.ok(Math.abs(s.revenue-s.energy*75)<1e-8);assert.ok(Math.abs(s.thermal-18.81)<1e-8);
});
test('reservoir heat loss equals extracted heat and temperatures remain bounded',()=>{
 const m=createModel(),initial=m.snapshot(),s=m.step(365*20);assert.ok(Math.abs(initial.remainingJ-s.remainingJ-s.extractedJ)/s.extractedJ<1e-9);assert.ok(s.temperature.every(t=>t>=70-1e-7&&t<=160+1e-7));assert.ok(s.production<160);
});
test('closer producer cools sooner under identical assumptions',()=>{
 const near=createModel({...defaults,producer:44}),far=createModel({...defaults,producer:48});assert.ok(near.step(365*20).production<far.step(365*20).production);
});
test('zero flow and invalid locations are handled',()=>{
 const m=createModel({rate:0});const s=m.step(365);assert.equal(s.energy,0);assert.equal(s.production,160);assert.throws(()=>createModel({producer:43}));
});

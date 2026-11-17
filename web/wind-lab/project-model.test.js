import test from 'node:test';
import assert from 'node:assert/strict';
import { gridPoint, powerKw, evaluate, validCandidate, rankCandidates, windFieldSampler } from './project-model.js';
import { createHistory, seasonalWind, nextSeasonalVariation } from './project-history.js';
import { createGame, GAME_COST, PRICE_PER_MWH } from './project-game.js';
import { wakePixels, wakeRaster, particleWind, advanceParticle, localParticle, particleAppearance, rotorRate, plasma } from './project-field.js';

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

test('overlapping wakes combine, rescale the colors, and update other wake colors',()=>{
  const one=windFieldSampler([95],0,12), two=windFieldSampler([95,85],0,12);
  assert.ok(two(1100,1500)<one(1100,1500));
  assert.equal(two(1100,1800),one(1100,1800));
  const a=wakeRaster([95],0,12), b=wakeRaster([95,85],0,12);
  assert.ok(b.min<a.min && b.max===12);
  const px=Math.floor(1100/2000*160),py=Math.floor(1800/2000*160),offset=(py*160+px)*4;
  assert.notDeepEqual([...a.pixels.slice(offset,offset+3)],[...b.pixels.slice(offset,offset+3)]);
});

test('particle motion and rotor animation respond to local wake speed',()=>{
  const sample=windFieldSampler([95],0,12);
  // World x=1 maps to easting 1100 m; z=-8 maps to northing 1800 m.
  const inWake=localParticle({x:1,z:-8},0,sample,1);
  const clear=localParticle({x:5,z:-8},0,sample,1);
  assert.ok(inWake.speed<clear.speed);
  assert.ok(inWake.z-(-8)<clear.z-(-8));
  assert.ok(inWake.wind.length<clear.wind.length);
  assert.ok(rotorRate(inWake.speed)<rotorRate(clear.speed));
  assert.equal(rotorRate(0),0);assert.equal(rotorRate(26),0);
  assert.equal(rotorRate(12),rotorRate(20));
});

test('wake particles drift sideways and fade while clear wind remains undisturbed',()=>{
  const sample=windFieldSampler([95],0,12), options={freeStream:12,elapsed:0,phase:Math.PI/2};
  const inWake=localParticle({x:1,z:-8},0,sample,.1,options);
  const clear=localParticle({x:5,z:-8},0,sample,.1,options);
  assert.ok(inWake.x<.98 && Math.abs(clear.x-5)<1e-9);
  assert.ok(inWake.opacity<clear.opacity);
  assert.equal(particleAppearance(12,12).opacity,.75);
  assert.ok(particleAppearance(0,12).opacity>0);
  const still=localParticle({x:1,z:-8},0,sample,0,options);
  assert.equal(still.x,1);assert.equal(still.z,-8);
});


test('session energy integrates each turbine across model changes and pauses',()=>{
  const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8);
  const h=createHistory(0), two={results:[{id:5,power:360,speed:12},{id:6,power:180,speed:9}]};
  h.setModel(two,12,0);
  let state=h.tick(10000);
  near(state.energy[5],262.8);near(state.energy[6],131.4);near(state.total,394.2);
  h.setModel({results:[{id:6,power:360,speed:12}]},12,10000);
  state=h.tick(20000);near(state.energy[5],262.8);near(state.energy[6],394.2);near(state.total,657);
  state=h.tick(30000,false);near(state.total,657);assert.equal(state.time,20);
  state=h.tick(35000,true);near(state.energy[6],525.6);assert.equal(state.time,25);
  assert.equal(h.reset(35000).total,0);
});

test('seasonal scenario repeats annually and varies speed and wind direction',()=>{
  assert.deepEqual(seasonalWind(0),seasonalWind(120));
  assert.equal(seasonalWind(0).speed,13);assert.equal(seasonalWind(60).speed,7);
  assert.equal(seasonalWind(30).month,3);assert.notEqual(seasonalWind(30).direction,seasonalWind(0).direction);
  for(let second=0;second<120;second++){
    const w=seasonalWind(second);assert.ok(w.speed>=7 && w.speed<=13);assert.ok(w.direction>=0 && w.direction<360);
  }
});


test('game charges purchases, credits generation, and pays half on removal',()=>{
  const g=createGame(0);assert.equal(g.buy(0),true);assert.equal(g.buy(0),false);
  g.setPower(360,0);const state=g.tick(10000);
  assert.ok(Math.abs(state.earnings-262.8*PRICE_PER_MWH)<1e-8);
  assert.ok(Math.abs(state.balance-state.earnings)<1e-8);
  const paused=g.tick(20000,false);assert.equal(paused.balance,state.balance);
  const sold=g.sell(20000);assert.equal(sold.balance,state.balance+GAME_COST/2);
  assert.equal(g.buy(20000),false);
  g.tick(40000);assert.equal(g.buy(40000),true);
});

test('seasonal variability is bounded and reacts to random samples',()=>{
  const a=nextSeasonalVariation({speed:0,direction:0},()=>1);
  const b=nextSeasonalVariation({speed:0,direction:0},()=>0);
  assert.ok(a.speed>0 && b.speed<0);assert.notEqual(seasonalWind(30,a).direction,seasonalWind(30,b).direction);
  let v={speed:0,direction:0};for(let i=0;i<100;i++)v=nextSeasonalVariation(v,()=>1);
  assert.ok(v.speed<=2.5 && v.direction<=25);
});

import {test} from 'node:test';import assert from 'node:assert/strict';import {drilling} from './drilling-model.js';
const base={depth:2000,flow:1200,hole:.216,pipe:.127,density:1100,friction:1,viscosity:.03,particle:.0002};
test('SI annulus, hydrostatic pressure and ECD are consistent',()=>{const r=drilling(base);assert.ok(Math.abs(r.velocity-(.02/r.area))<1e-12);assert.ok(Math.abs(r.hydro-1100*9.80665*2000)<1e-7);assert.ok(Math.abs(r.equivalent*9.80665*2000-r.bottom)<1e-7);});
test('pump rate and prescribed pressure loss affect distinct quantities',()=>{const a=drilling(base),b=drilling({...base,flow:2400,friction:2});assert.ok(Math.abs(b.velocity-2*a.velocity)<1e-12);assert.equal(b.hydro,a.hydro);assert.ok(b.equivalent>a.equivalent);});
test('slip result is qualified by particle Reynolds number',()=>{assert.equal(drilling(base).validStokes,true);assert.equal(drilling({...base,particle:.01,viscosity:.001}).validStokes,false);assert.throws(()=>drilling({...base,pipe:.3}));});

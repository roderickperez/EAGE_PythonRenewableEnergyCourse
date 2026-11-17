import test from 'node:test';
import assert from 'node:assert/strict';
import { calculate, conditions, declination, surfaceLoss, SITES } from './project-model.js';
import { cloudState, applyCloudShadow } from './project-clouds.js';
import { readFileSync } from 'node:fs';
const site={lat:48.208,lon:16.374}, settings={tilt:30,azimuth:180,shading:5,count:2};
test('sun and seasonal generation respond to latitude and date',()=>{
  const summer=new Date('2025-06-21T12:00:00Z'),winter=new Date('2025-12-21T12:00:00Z');
  const a=calculate(site,summer,12,settings,conditions(summer,site.lat));
  const b=calculate(site,winter,12,settings,conditions(winter,site.lat));
  assert.ok(a.current.elevation>b.current.elevation);assert.ok(a.day.energy>b.day.energy);
  const south=calculate({lat:-33.869,lon:151},summer,12,settings,conditions(summer,-33.869));
  assert.ok(south.current.elevation<a.current.elevation);assert.ok(declination(summer)>23);
});
test('night and zero panels have zero power, and shading reduces output',()=>{
  const d=new Date('2025-03-21T12:00:00Z'),w=conditions(d,site.lat);
  assert.equal(calculate(site,d,0,settings,w).current.ac,0);
  assert.equal(calculate(site,d,12,{...settings,count:0},w).day.energy,0);
  assert.ok(calculate(site,d,12,{...settings,shading:50},w).current.ac<calculate(site,d,12,settings,w).current.ac);
});
test('polar summer and winter sun remain above and below horizon respectively',()=>{
  const arctic={lat:69.649,lon:18.956};
  for(const [day,above] of [['2025-06-21',true],['2025-12-21',false]]) {
    const d=new Date(day+'T12:00:00Z'),w=conditions(d,arctic.lat);
    for(const hour of [0,6,12,18]) assert.equal(calculate(arctic,d,hour,settings,w).current.elevation>0,above);
  }
});
test('cloud cover changes irradiance and midday output, independently of surface losses',()=>{
  const d=new Date('2025-06-21T12:00:00Z'), clear=conditions(d,site.lat,-.16), cloudy=conditions(d,site.lat,.16);
  assert.ok(cloudy.clouds>clear.clouds);assert.ok(cloudy.dni<clear.dni);
  assert.ok(calculate(site,d,12,settings,cloudy).current.ac<calculate(site,d,12,settings,clear).current.ac);
  assert.ok(surfaceLoss(.1)>surfaceLoss(-.1));
  assert.deepEqual(SITES.map(s=>s.region),['North America','South America','Africa','Europe','Asia']);
});
test('a moving cloud crossing the solar ray reduces irradiance and power',()=>{
  const d=new Date('2025-06-21T12:00:00Z'),base=conditions(d,site.lat),dec=declination(d);
  const covered=cloudState(12,site.lat,dec,0,base.clouds),clear=cloudState(12,site.lat,dec,2.22,base.clouds);
  assert.ok(covered.blockage>clear.blockage+.25);
  const a=calculate(site,d,12,settings,applyCloudShadow(base,covered)).current;
  const b=calculate(site,d,12,settings,applyCloudShadow(base,clear)).current;
  assert.ok(a.poa<b.poa);assert.ok(a.ac<b.ac);
  assert.equal(cloudState(0,site.lat,dec,0,base.clouds).blockage,0);
});
test('all chosen coordinates fall on Natural Earth land polygons',()=>{
  const land=JSON.parse(readFileSync(new URL('./natural-earth-land.geojson',import.meta.url)));
  function inside(x,y,ring){let hit=false;for(let i=0,j=ring.length-1;i<ring.length;j=i++){const a=ring[i],b=ring[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit;}return hit;}
  for(const site of SITES){const match=land.features.some(f=>f.geometry.coordinates.reduce((hit,ring)=>hit!==inside(site.lon,site.lat,ring),false));assert.ok(match,site.name);}
});

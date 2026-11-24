import test from 'node:test';
import assert from 'node:assert/strict';
import { calculate, calculateFarm, conditions, declination, surfaceLoss, SITES } from './project-model.js';
import { cloudState, cloudBlockage } from './project-clouds.js';
import { readFileSync } from 'node:fs';
import { chartMarkup, panelColor } from './project-chart.js';
import { SPEEDS, simulatedHours, nextStep } from './project-timing.js';
import { fullSolarDay } from './project-day.js';
import { daylightFactor } from './project-light.js';
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
test('a moving cloud shades individual panels and farm output is their sum',()=>{
  const d=new Date('2025-06-21T12:00:00Z'),base=conditions(d,site.lat),dec=declination(d);
  const covered=cloudState(12,site.lat,dec,0,base.clouds),[east,up,north]=covered.sun;
  covered.clouds=[{position:[-9+east/up*5.32,6,9-north/up*5.32]}];
  const result=calculateFarm(site,d,12,settings,base,covered,[0,9]);
  assert.ok(result.panels[0].blockage>.4);assert.ok(result.panels[1].blockage<.001);
  assert.ok(result.panels[0].current.poa<result.panels[1].current.poa);
  assert.ok(result.panels[0].current.ac<result.panels[1].current.ac);
  assert.ok(Math.abs(result.current.ac-result.panels.reduce((n,p)=>n+p.current.ac,0))<1e-9);
  covered.clouds[0].position[0]+=12;
  assert.ok(cloudBlockage(covered,-9,9,.68)<.01);
  assert.equal(cloudBlockage(cloudState(0,site.lat,dec,0,base.clouds),-9,9),0);
  const next=cloudState(12,site.lat,dec,1,base.clouds),start=cloudState(12,site.lat,dec,0,base.clouds);
  assert.equal(next.clouds[0].position[2],start.clouds[0].position[2]);
  assert.ok(next.clouds[0].position[0]>start.clouds[0].position[0]);
});
test('all chosen coordinates fall on Natural Earth land polygons',()=>{
  const land=JSON.parse(readFileSync(new URL('./natural-earth-land.geojson',import.meta.url)));
  function inside(x,y,ring){let hit=false;for(let i=0,j=ring.length-1;i<ring.length;j=i++){const a=ring[i],b=ring[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit;}return hit;}
  for(const site of SITES){const match=land.features.some(f=>f.geometry.coordinates.reduce((hit,ring)=>hit!==inside(site.lon,site.lat,ring),false));assert.ok(match,site.name);}
});
test('panel curves render independently and do not bridge missing samples',()=>{
  const panel=(id,ac)=>({id,current:{ac}}),points=[{hour:8,values:{0:100,1:200}},{hour:9,values:{1:150}},{hour:10,values:{0:80,1:100}}];
  const svg=chartMarkup(points,[0,1],10,[panel(0,80),panel(1,100)]);
  assert.ok(svg.includes(panelColor(0)));assert.ok(svg.includes(panelColor(1)));
  const path=svg.match(new RegExp(`<path d="([^"]*)" fill="none" stroke="${panelColor(0)}"`))[1];
  assert.equal((path.match(/M/g)||[]).length,2);
});
test('full-day curves include moving cloud shadows and match current panel output',()=>{
  const d=new Date('2025-06-21T12:00:00Z'),w=conditions(d,site.lat),ids=[44,45],phase=30,hour=12;
  const day=fullSolarDay(site,d,hour,settings,w,phase,ids),cloud=cloudState(hour,site.lat,declination(d),phase,w.clouds);
  const current=calculateFarm(site,d,hour,settings,w,cloud,ids,false);
  assert.equal(day.points.length,145);assert.equal(day.points[0].hour,0);assert.equal(day.points.at(-1).hour,24);
  for(const p of current.panels)assert.ok(Math.abs(day.points[72].values[p.id]-p.current.ac)<1e-8);
  assert.ok(day.energy>0);assert.ok(day.points.some(p=>Math.abs(p.values[44]-p.values[45])>1));
  // Same weather and cloud phase at a given solar time, regardless of playback speed.
  const same=fullSolarDay(site,d,6,settings,w,15,ids);
  assert.deepEqual(same.points,day.points);
  const offGrid=11.57,phaseAt=offGrid/.4,curve=fullSolarDay(site,d,offGrid,settings,w,phaseAt,ids);
  const live=calculateFarm(site,d,offGrid,settings,w,cloudState(offGrid,site.lat,declination(d),phaseAt,w.clouds),ids,false);
  const matching=curve.points.find(point=>point.hour===offGrid);
  for(const p of live.panels)assert.ok(Math.abs(matching.values[p.id]-p.current.ac)<1e-8);
});
test('night scene light is dark below the horizon and blends into daytime',()=>{
  assert.equal(daylightFactor(-.2),0);assert.equal(daylightFactor(0),0);assert.equal(daylightFactor(.04),.5);assert.equal(daylightFactor(.2),1);
});
test('speed increases simulated time and integration steps respect boundaries',()=>{
  assert.deepEqual(SPEEDS,[1,5,20,100,500]);
  assert.equal(simulatedHours(60,1),24);assert.equal(simulatedHours(60,500),12000);
  assert.ok(Math.abs(nextStep(1,23.9,1)-.1)<1e-9);
  assert.ok(Math.abs(nextStep(1,12,1.9)-.1)<1e-9);
  assert.equal(nextStep(50,12,0),.25);
});

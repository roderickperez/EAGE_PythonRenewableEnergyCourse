import {test} from 'node:test';
import assert from 'node:assert/strict';
import {drawProjectCharts} from './project-charts.js';
function makeContext(){return {rects:[],clearRect(){},fillRect(x,y,w,h){this.rects.push({x,y,w,h});},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){},arc(){},fill(){}};}
function draw(times){
 const ctx=Object.fromEntries(['temperature','energy','depth','heatmap'].map(k=>[k,makeContext()]));
 const root={querySelector:selector=>{const name=selector.match(/data-chart="([^"]+)"/)[1];return {getContext:()=>ctx[name]};}};
 const history=times.map((years,i)=>({years,production:150-i,injection:70,energy:i,layerTemperatures:{2000:150-i}}));
 drawProjectCharts(root,{years:times.at(-1),initial:160,layerTemperatures:{2000:150}},history,2000);
 return ctx.heatmap.rects.filter(r=>r.y===10&&r.h===15);
}
test('heatmap allocates horizontal space by simulated time rather than sample index',()=>{
 const cells=draw([0,1,10]);assert.equal(cells.length,3);
 assert.deepEqual(cells.map(r=>r.w),[27,265,239]);
 assert.ok(cells[1].w>9*cells[0].w);
});
test('same-time events never create negative-width heatmap cells',()=>{
 const cells=draw([0,0,2]);assert.equal(cells.length,3);
 assert.ok(cells.every(r=>r.w>=0));assert.equal(cells[0].w,0);
});

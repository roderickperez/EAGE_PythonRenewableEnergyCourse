import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createFieldModel} from './project-field-model.js';
import {fieldCsv,fieldRecord} from './project-export.js';
import {createBudget} from './project-budget.js';
function parse(line) {
  const cells=[]; let cell='', quoted=false;
  for (let i=0;i<line.length;i++) {
    const ch=line[i];
    if(ch==='"') { if(quoted && line[i+1]==='"'){cell+='"';i++;} else quoted=!quoted; }
    else if(ch===',' && !quoted){cells.push(cell);cell='';} else cell+=ch;
  }
  cells.push(cell); return cells;
}
test('multidepth export preserves ordered events and replays clipped stream heat',()=>{
  const m=createFieldModel({injection:30});
  m.configure([{id:1,depth:1500,type:'injector'},{id:2,depth:1500,type:'producer'},
               {id:3,depth:2500,type:'injector'},{id:4,depth:2500,type:'producer'}]);
  const initial={...m.snapshot(),event:'well_added'};
  m.step(365*1.5); m.setOperating({injection:100});
  const mixed={...m.snapshot(),event:'operating_change'};
  assert.ok(mixed.wellTemperatures.some(w=>w.temperature<100));
  assert.ok(mixed.wellTemperatures.some(w=>w.temperature>100));
  const lines=fieldCsv([initial,{...initial,event:'well_added'},mixed]).split('\n');
  const headers=parse(lines[0]);
  const rows=lines.slice(1).map(line=>Object.fromEntries(parse(line).map((v,i)=>[headers[i],v])));
  assert.deepEqual(rows.map(r=>Number(r.record_index)),[0,1,2]);
  assert.equal(rows[0].years,rows[1].years);
  assert.equal(Number(rows[2].active_depth_count),2);
  assert.equal(Number(rows[2].total_flow_kg_s),100);
  const streams=JSON.parse(rows[2].producer_streams_json);
  const thermal=streams.reduce((sum,w)=>sum+Math.max(0,w.weight*4180*(w.temperature-Number(rows[2].injection_C)))/1e6,0);
  assert.ok(Math.abs(thermal-Number(rows[2].thermal_MW))<1e-12);
  const electrical=Math.min(Number(rows[2].capacity_MW),thermal*Number(rows[2].conversion_efficiency)*(1-Number(rows[2].auxiliary_fraction)));
  assert.ok(Math.abs(electrical-Number(rows[2].net_power_MW))<1e-12);
});

test('export preserves investment and balance through paid events and removal',()=>{
  const m=createFieldModel(); const budget=createBudget();
  const record=(event,revenue)=>{const snap=m.snapshot();if(revenue!==undefined)snap.revenue=revenue;return fieldRecord(snap,budget,event);};
  const before=record('sample'); budget.drillExtra(); const drilled=record('well_added');
  const removed=record('well_removed');
  assert.equal(budget.upgrade(8000000,5),true); const upgraded=record('capacity_change',8000000);
  const lines=fieldCsv([before,drilled,removed,upgraded]).split('\n'); const headers=parse(lines[0]);
  const rows=lines.slice(1).map(line=>Object.fromEntries(parse(line).map((v,i)=>[headers[i],v])));
  assert.deepEqual(rows.map(r=>Number(r.cumulative_investment_USD)),[6000000,7000000,7000000,8000000]);
  assert.deepEqual(rows.map(r=>Number(r.additional_wells_drilled)),[0,1,1,1]);
  assert.deepEqual(rows.map(r=>Number(r.capacity_upgrades)),[0,0,0,1]);
  for(const r of rows) assert.equal(Number(r.project_balance_USD),Number(r.revenue_USD)-Number(r.cumulative_investment_USD));
});

test('export distinguishes balanced depths from circulating depths and preserves accumulated energy',()=>{
  const m=createFieldModel(); const budget=createBudget();
  m.configure([{id:1,depth:2000,type:'injector'},{id:2,depth:2000,type:'producer'}]);
  const capture=event=>fieldRecord(m.snapshot(),budget,event);
  const start=capture('sample');
  m.step(0.001); const operating=capture('sample');
  m.setOperating({rate:0}); const stopped=capture('operating_change');
  m.step(1); const afterStoppedDay=capture('sample');
  const lines=fieldCsv([start,operating,stopped,afterStoppedDay]).split('\n');
  const headers=parse(lines[0]);
  const rows=lines.slice(1).map(line=>Object.fromEntries(parse(line).map((v,i)=>[headers[i],v])));
  const value=(i,name)=>Number(rows[i][name]);
  assert.equal(value(2,'balanced_depth_count'),1);
  assert.equal(value(2,'active_depth_count'),0);
  assert.equal(value(2,'total_flow_kg_s'),0);
  assert.equal(value(2,'years'),value(1,'years'));
  assert.equal(value(2,'energy_MWh'),value(1,'energy_MWh'));
  assert.equal(value(3,'energy_MWh'),value(2,'energy_MWh'));
  const expected=value(0,'net_power_MW')*0.001*24*value(0,'availability');
  assert.ok(Math.abs(value(1,'energy_MWh')-expected)<1e-10);
  assert.ok(Math.abs(value(1,'revenue_USD')-expected*value(0,'price_USD_per_MWh'))<1e-8);
  assert.equal(value(3,'revenue_USD'),value(2,'revenue_USD'));
});

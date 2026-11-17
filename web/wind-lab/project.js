import { evaluate, gridPoint, rankCandidates, validCandidate, DIAMETER } from './project-model.js';
import { mountFarmScene } from './project-scene.js';
import css from './project.css';

export function mountWindProject(host) {
  const style=document.createElement('style'); style.textContent=css; host.append(style);
  const root=document.createElement('section'); root.className='wind-project';
  root.innerHTML=`<div class="wp-heading"><div><h2>Wind farm layout explorer</h2><p>Place turbines on a 2 km × 2 km site. Drag to rotate; scroll to zoom. Turbines are enlarged for visibility.</p></div><button data-action="motion" type="button" aria-pressed="true">Pause motion</button></div>
    <div class="wp-layout"><div class="wp-scene" role="img" aria-label="Isometric wind farm grid. Use the location selector below for keyboard placement."></div>
    <div class="wp-settings"><label>Wind from <output data-out="direction">0° N</output><input data-field="direction" type="range" min="0" max="359" step="1" value="0"></label>
      <label>Wind speed <output data-out="speed">12.0 m/s</output><input data-field="speed" type="range" min="0" max="25" step="0.5" value="12"></label>
      <label>Minimum spacing <output data-out="spacing">200 m (5 D)</output><input data-field="spacing" type="range" min="0" max="600" step="40" value="200"></label>
      <label>Grid location <select data-field="location" aria-label="Grid location"></select></label>
      <div class="wp-buttons"><button data-action="toggle" type="button">Place turbine</button><button data-action="recommend" type="button">Find best next location</button><button data-action="baseline" type="button">Save baseline</button><button data-action="reset" type="button">Reset layout</button><button data-action="export" type="button">Download CSV</button></div>
      <p class="wp-note">Gold marks the selected location; blue shapes show idealized wakes. The yellow arrow points downwind. Wind direction is where wind comes from.</p></div></div>
    <p class="wp-status" role="status" aria-live="polite">Select an empty location and place a turbine.</p>
    <div class="wp-metrics"><div><span>Turbines</span><strong data-value="count">0</strong></div><div><span>Farm power</span><strong data-value="power">0.000 MW</strong></div><div><span>No-wake power</span><strong data-value="ideal">0.000 MW</strong></div><div><span>Wake loss</span><strong data-value="loss">0.000 MW</strong></div><div><span>Efficiency</span><strong data-value="efficiency">—</strong></div></div>
    <p data-value="selection"></p><p data-value="baseline"></p><p data-value="recommendation"></p>
    <p class="wp-note">Educational, single-speed top-hat wake model. Recommendations evaluate the next location with the current turbines fixed; they are not a globally optimal design or a site assessment.</p>`;
  host.append(root);
  const $=s=>root.querySelector(s), field=n=>$(`[data-field="${n}"]`), output=n=>$(`[data-out="${n}"]`), value=n=>$(`[data-value="${n}"]`);
  const select=field('location');
  for(let id=0;id<100;id++){const p=gridPoint(id),o=document.createElement('option');o.value=id;o.textContent=`${id+1}: east ${p.x} m, north ${p.y} m`;select.append(o);}
  let ids=[], baseline=null, selected=0, best=null, scene=null, motion=!matchMedia('(prefers-reduced-motion: reduce)').matches;
  const status=$('.wp-status'), speed=()=>Number(field('speed').value), direction=()=>Number(field('direction').value), spacing=()=>Number(field('spacing').value);
  const format=n=>(n/1000).toFixed(3)+' MW';
  function redraw(message) {
    const result=evaluate(ids,direction(),speed());
    output('direction').textContent=`${direction()}° from north`;
    output('speed').textContent=`${speed().toFixed(1)} m/s`;
    output('spacing').textContent=`${spacing()} m (${(spacing()/DIAMETER).toFixed(1)} D)`;
    value('count').textContent=ids.length; value('power').textContent=format(result.total);value('ideal').textContent=format(result.ideal); value('loss').textContent=format(result.loss);
    value('efficiency').textContent=result.efficiency===null?'—':result.efficiency.toFixed(1)+'%';
    select.value=String(selected);
    const point=gridPoint(selected), turbine=result.results.find(p=>p.id===selected);
    value('selection').textContent=turbine?`Location ${selected+1}: wind ${turbine.speed.toFixed(2)} m/s, turbine power ${turbine.power.toFixed(1)} kW.`:`Location ${selected+1}: east ${point.x} m, north ${point.y} m; ${validCandidate(selected,ids,spacing())?'available':'too close to an existing turbine'}.`;
    $('[data-action="toggle"]').textContent=turbine?'Remove turbine':'Place turbine';
    $('[data-action="toggle"]').disabled=!turbine&&!validCandidate(selected,ids,spacing());
    value('baseline').textContent=baseline===null?'No baseline saved.':`Saved baseline: ${baseline.length} turbines, ${format(evaluate(baseline,direction(),speed()).total)} under the current wind conditions.`;
    if(message)status.textContent=message;
    scene?.update(ids,selected,best,direction(),speed());
  }
  function invalidate(){best=null;value('recommendation').textContent='';}
  function toggle(){
    if(ids.includes(selected)){ids=ids.filter(i=>i!==selected);invalidate();redraw(`Turbine removed from location ${selected+1}.`);}
    else if(validCandidate(selected,ids,spacing())){ids=[...ids,selected];invalidate();redraw(`Turbine placed at location ${selected+1}.`);}
    else redraw('Location is too close to an existing turbine.');
  }
  select.addEventListener('change',()=>{selected=Number(select.value);redraw();});
  ['direction','speed'].forEach(key=>field(key).addEventListener('input',()=>{invalidate();redraw();}));
  field('spacing').addEventListener('input',()=>{
    const newSpacing=spacing();
    const conflict=ids.some((id,index)=>!validCandidate(id,ids.filter((_,i)=>i!==index),newSpacing));
    if(conflict){field('spacing').value=field('spacing').dataset.accepted;status.textContent='This spacing would invalidate the existing layout. Remove or move turbines first.';}
    else {field('spacing').dataset.accepted=String(newSpacing);invalidate();}
    redraw();
  });
  field('spacing').dataset.accepted=field('spacing').value;
  root.addEventListener('click',event=>{
    const action=event.target.closest('[data-action]')?.dataset.action;if(!action)return;
    if(action==='toggle')toggle();
    if(action==='reset'){ids=[];invalidate();redraw('Layout cleared.');}
    if(action==='baseline'){baseline=[...ids];redraw('Baseline saved.');}
    if(action==='recommend'){
      if(speed() < 3 || speed() > 25){invalidate();redraw('Choose an operating wind speed to compare locations.');return;}
      const ranking=rankCandidates(ids,direction(),speed(),spacing());
      if(!ranking.length){invalidate();redraw('No valid locations remain at this spacing.');return;}
      const top=ranking[0];best=top.id;selected=top.id;const ties=ranking.filter(r=>Math.abs(r.total-top.total)<1e-7).length;
      value('recommendation').textContent=`Best next location: ${top.id+1}; expected farm power ${format(top.total)}; change ${format(top.gain)}. ${ties} location(s) tie at this precision. Current turbines stay fixed.`;
      redraw('Best next location highlighted. Place it to add the turbine.');
    }
    if(action==='motion'){motion=!motion;scene?.setMotion(motion);event.target.textContent=motion?'Pause motion':'Resume motion';event.target.setAttribute('aria-pressed',String(motion));}
    if(action==='export'){
      const result=evaluate(ids,direction(),speed());
      const lines=['location,east_m,north_m,wind_from_deg,freestream_ms,local_wind_ms,power_kw,minimum_spacing_m',...result.results.map(p=>[p.id+1,p.x,p.y,direction(),speed(),p.speed.toFixed(4),p.power.toFixed(4),spacing()].join(','))];
      const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv'}));a.download='wind-farm-layout.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    }
  });
  try { scene=mountFarmScene($('.wp-scene'),id=>{selected=id;toggle();});scene.setMotion(motion); }
  catch(error){$('.wp-scene').textContent='3D view unavailable. Use the location selector to build the same layout.';console.warn('Wind farm 3D unavailable',error);}
  const motionButton=$('[data-action="motion"]');motionButton.textContent=motion?'Pause motion':'Resume motion';motionButton.setAttribute('aria-pressed',String(motion));
  redraw();
  return {destroy(){scene?.destroy();root.remove();style.remove();}};
}

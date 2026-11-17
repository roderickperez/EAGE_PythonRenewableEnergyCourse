import { evaluate, gridPoint, validCandidate, DIAMETER } from './project-model.js';
import { mountFarmScene } from './project-scene.js';
import { createHistory, seasonalWind } from './project-history.js';
import css from './project.css';

export function mountWindProject(host) {
  const style=document.createElement('style'); style.textContent=css; host.append(style);
  const root=document.createElement('section'); root.className='wind-project';
  root.innerHTML=`<div class="wp-heading"><div><h2>Wind farm layout explorer</h2><p>Place turbines on a 2 km × 2 km site. Drag to rotate; scroll to zoom. Turbines are enlarged for visibility.</p></div></div>
    <div class="wp-layout"><div class="wp-scene-wrap"><div class="wp-scene" role="application" tabindex="0" aria-label="Wind farm grid. Use arrow keys to select a location and Enter or Space to place or remove a turbine."></div><button class="wp-motion" data-action="motion" type="button" aria-label="Stop animation" title="Stop animation"><span aria-hidden="true">■</span></button><div class="wp-legend" aria-label="Modeled wake wind speed color scale"><strong>Modeled wind speed (m/s)</strong><div class="wp-gradient"></div><div class="wp-legend-labels"><span data-out="legend-min">0.0</span><span data-out="legend-max">12.0 (free stream)</span></div><span>Scale adapts to the sampled wake minimum.</span></div></div>
    <div class="wp-settings"><div class="wp-energy"><span>Estimated session energy</span><strong data-value="energy">0.00 Wh</strong><small>Model estimate for this open session</small></div>
      <div class="wp-controls-charts"><div class="wp-sliders"><label>Wind mode <select data-field="mode" aria-label="Wind mode"><option value="manual">Manual</option><option value="seasonal">Seasonal</option></select></label><div class="wp-season" data-value="season" hidden></div><label>Wind from <output data-out="direction">0° N</output><input data-field="direction" type="range" min="0" max="359" step="1" value="0"></label>
      <label>Wind speed <output data-out="speed">12.0 m/s</output><input data-field="speed" type="range" min="0" max="25" step="0.5" value="12"></label>
      <label>Minimum spacing <output data-out="spacing">200 m (5 D)</output><input data-field="spacing" type="range" min="0" max="600" step="40" value="200"></label></div>
      <div class="wp-charts"><div><strong>Cumulative energy (Wh)</strong><svg class="wp-chart" data-chart="energy" viewBox="0 0 260 120" role="img" aria-label="Estimated cumulative energy over session time"></svg></div><div><strong>Wind speed (m/s)</strong><svg class="wp-chart" data-chart="speed" viewBox="0 0 260 120" role="img" aria-label="Free-stream and turbine wind speeds over session time"></svg></div><div class="wp-chart-legend" data-value="chart-legend"></div><small>Elapsed visible-session time (s); latest 300 samples. Dark: farm total or free stream; colors: turbine locations.</small></div></div></div></div>
    <p class="wp-status" role="status" aria-live="polite">Click the grid to place a turbine. Keyboard: focus the grid, use arrow keys, then Enter or Space.</p>
    <div class="wp-metrics"><div><span>Turbines</span><strong data-value="count">0</strong></div><div><span>Farm power</span><strong data-value="power">0.000 MW</strong></div><div><span>No-wake power</span><strong data-value="ideal">0.000 MW</strong></div><div><span>Wake loss</span><strong data-value="loss">0.000 MW</strong></div><div><span>Efficiency</span><strong data-value="efficiency">—</strong></div></div>
    <p data-value="selection"></p><p class="wp-note">Educational, single-speed top-hat wake model; results are illustrative, not a site assessment.</p>`;
  host.append(root);
  const $=s=>root.querySelector(s), field=n=>$(`[data-field="${n}"]`), output=n=>$(`[data-out="${n}"]`), value=n=>$(`[data-value="${n}"]`);
  let ids=[], selected=0, scene=null, motion=true, pageActive=!document.hidden, seasonalStart=0;
  let manual={speed:12,direction:0};
  const status=$('.wp-status'), speed=()=>Number(field('speed').value), direction=()=>Number(field('direction').value), spacing=()=>Number(field('spacing').value);
  const format=n=>(n/1000).toFixed(3)+' MW';
  const history=createHistory(performance.now());
  const palette=['#b9414e','#287fa7','#8d5db5','#b07017','#21836b','#bb4d88','#5d6fa7','#7a7530'];
  const turbineColor=id=>palette[id%palette.length];
  function drawChart(name, points, turbineIds) {
    const svg=$(`[data-chart="${name}"]`), start=points[0]?.time ?? 0, end=points.at(-1)?.time ?? 0, span=Math.max(end-start,1);
    const all=points.flatMap(p=>name==='energy'?[p.total,...Object.values(p.energy)]:[p.freestream,...Object.values(p.speeds)]);
    const max=Math.max(name==='speed'?1:.01,...all)*1.1, x=t=>32+(t-start)/span*220, y=v=>100-v/max*78;
    const path=(get)=>{let active=false,d='';for(const p of points){const v=get(p);if(v===undefined){active=false;continue;}d+=(active?' L':' M')+x(p.time).toFixed(1)+' '+y(v).toFixed(1);active=true;}return d;};
    let markup=`<path d="M32 10 V100 H252" fill="none" stroke="#9aafb8"/><text x="29" y="14" text-anchor="end">${max.toFixed(name==='energy'?2:1)}</text><text x="29" y="103" text-anchor="end">0</text><text x="32" y="116">${start.toFixed(0)}</text><text x="252" y="116" text-anchor="end">${end.toFixed(0)} s</text>`;
    markup+=`<path d="${path(p=>name==='energy'?p.total:p.freestream)}" fill="none" stroke="#153d55" stroke-width="2.4"/>`;
    for(const id of turbineIds){const d=path(p=>name==='energy'?(p.energy[id] ?? 0):p.speeds[id]);markup+=`<path d="${d}" fill="none" stroke="${turbineColor(id)}" stroke-width="1.8"/>`;}
    svg.innerHTML=markup;
  }
  function showHistory(state) {
    value('energy').textContent=`${state.total.toFixed(2)} Wh`;
    const allIds=[...new Set(state.points.flatMap(p=>[...Object.keys(p.energy),...Object.keys(p.speeds)].map(Number)))].sort((a,b)=>a-b);
    drawChart('energy',state.points,allIds);drawChart('speed',state.points,allIds);
    value('chart-legend').innerHTML='<span><i style="background:#153d55"></i>Farm / free stream</span>'+allIds.map(id=>`<span><i style="background:${turbineColor(id)}"></i>Location ${id+1}: ${(state.energy[id] || 0).toFixed(2)} Wh</span>`).join('');
  }
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function applySeason(elapsed){
    const weather=seasonalWind(elapsed);
    field('speed').value=weather.speed;field('direction').value=weather.direction;
    value('season').textContent=`Synthetic ${months[weather.month]} · one year / 120 seconds`;
  }
  const historyTimer=setInterval(()=>{
    if(!pageActive)return;
    const state=history.tick(performance.now(),true);
    if(field('mode').value==='seasonal'){applySeason(state.time-seasonalStart);redraw();}
    else showHistory(state);
  },1000);
  const visibility=()=>{showHistory(history.tick(performance.now(),pageActive));pageActive=!document.hidden;};
  document.addEventListener('visibilitychange',visibility);
  function redraw(message) {
    const result=evaluate(ids,direction(),speed());
    showHistory(history.setModel(result,speed(),performance.now(),pageActive));
    output('direction').textContent=`${direction()}° from north`;
    output('speed').textContent=`${speed().toFixed(1)} m/s`;
    output('legend-max').textContent=`${speed().toFixed(1)} (free stream)`;
    output('spacing').textContent=`${spacing()} m (${(spacing()/DIAMETER).toFixed(1)} D)`;
    value('count').textContent=ids.length;value('power').textContent=format(result.total);value('ideal').textContent=format(result.ideal);value('loss').textContent=format(result.loss);
    value('efficiency').textContent=result.efficiency===null?'—':result.efficiency.toFixed(1)+'%';
    const point=gridPoint(selected), turbine=result.results.find(p=>p.id===selected);
    value('selection').textContent=turbine?`Location ${selected+1}: wind ${turbine.speed.toFixed(2)} m/s, turbine power ${turbine.power.toFixed(1)} kW.`:`Location ${selected+1}: east ${point.x} m, north ${point.y} m; ${validCandidate(selected,ids,spacing())?'available':'too close to an existing turbine'}.`;
    if(message)status.textContent=message;
    const bounds=scene?.update(ids,selected,null,direction(),speed());output('legend-min').textContent=(bounds?.min ?? 0).toFixed(1);
  }
  function toggle(){
    if(ids.includes(selected)){ids=ids.filter(i=>i!==selected);redraw(`Turbine removed from location ${selected+1}.`);}
    else if(validCandidate(selected,ids,spacing())){ids=[...ids,selected];redraw(`Turbine placed at location ${selected+1}.`);}
    else redraw('Location is too close to an existing turbine.');
  }
  const sceneHost=$('.wp-scene');
  sceneHost.addEventListener('keydown',event=>{
    let next=selected;
    if(event.key==='ArrowLeft' && selected%10>0)next--;
    else if(event.key==='ArrowRight' && selected%10<9)next++;
    else if(event.key==='ArrowUp' && selected<90)next+=10;
    else if(event.key==='ArrowDown' && selected>=10)next-=10;
    else if(event.key==='Enter' || event.key===' '){event.preventDefault();toggle();return;}
    else return;
    event.preventDefault();selected=next;redraw(`Selected location ${selected+1}. Press Enter to place or remove a turbine.`);
  });
  ['direction','speed'].forEach(key=>field(key).addEventListener('input',()=>{manual={speed:speed(),direction:direction()};redraw();}));
  field('mode').addEventListener('change',()=>{
    const state=history.tick(performance.now(),pageActive), seasonal=field('mode').value==='seasonal';
    field('speed').hidden=seasonal;field('direction').hidden=seasonal;value('season').hidden=!seasonal;
    if(seasonal){manual={speed:speed(),direction:direction()};seasonalStart=state.time;applySeason(0);}
    else {field('speed').value=manual.speed;field('direction').value=manual.direction;}
    redraw(seasonal?'Seasonal wind scenario started.':'Manual wind controls restored.');
  });
  field('spacing').addEventListener('input',()=>{
    const newSpacing=spacing();
    const conflict=ids.some((id,index)=>!validCandidate(id,ids.filter((_,i)=>i!==index),newSpacing));
    if(conflict){field('spacing').value=field('spacing').dataset.accepted;status.textContent='This spacing would invalidate the existing layout. Remove or move turbines first.';}
    else field('spacing').dataset.accepted=String(newSpacing);
    redraw();
  });
  field('spacing').dataset.accepted=field('spacing').value;
  root.querySelector('[data-action="motion"]').addEventListener('click',()=>{motion=!motion;scene?.setMotion(motion);updateMotionButton();});
  try {scene=mountFarmScene(sceneHost,id=>{selected=id;toggle();});scene.setMotion(motion);}
  catch(error){sceneHost.textContent='3D view unavailable in this browser.';console.warn('Wind farm 3D unavailable',error);}
  function updateMotionButton(){const button=$('[data-action="motion"]'),label=motion?'Stop animation':'Play animation';button.setAttribute('aria-label',label);button.title=label;button.querySelector('span').textContent=motion?'■':'▶';}
  updateMotionButton();redraw();
  return {destroy(){clearInterval(historyTimer);document.removeEventListener('visibilitychange',visibility);scene?.destroy();root.remove();style.remove();}};
}

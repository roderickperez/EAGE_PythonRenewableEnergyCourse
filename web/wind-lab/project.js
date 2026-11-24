import { evaluate, gridPoint, validCandidate, MIN_SPACING } from './project-model.js';
import { mountFarmScene } from './project-scene.js';
import { createHistory, seasonalWind, nextSeasonalVariation } from './project-history.js';
import { createGame, GAME_COST } from './project-game.js';
import css from './project.css';

export function mountWindProject(host) {
  const style=document.createElement('style'); style.textContent=css; host.append(style);
  const root=document.createElement('section'); root.className='wind-project';
  root.innerHTML=`<div class="wp-heading"><div><h2>Wind farm layout explorer</h2><p>Place turbines on a 2 km × 2 km site. Drag to rotate; scroll to zoom. Turbines are enlarged for visibility.</p></div></div>
    <div class="wp-layout"><div class="wp-scene-wrap"><div class="wp-scene" role="application" tabindex="0" aria-label="Wind farm grid. Use arrow keys to select a location and Enter or Space to place or remove a turbine."></div><button class="wp-motion" data-action="motion" type="button" aria-label="Stop animation" title="Stop animation"><span aria-hidden="true">■</span></button><div class="wp-legend" aria-label="Modeled wake wind speed color scale"><strong>Modeled wind speed (m/s)</strong><div class="wp-gradient"></div><div class="wp-legend-labels"><span data-out="legend-min">0.0</span><span data-out="legend-max">12.0 (free stream)</span></div><span>Scale adapts to the sampled wake minimum.</span></div></div>
    <div class="wp-settings"><div class="wp-energy"><span>Simulated energy generated</span><strong data-value="energy">0.00 MWh</strong><small>73 simulated operating hours per visible second</small></div>
      <div class="wp-game"><div><span>Available budget</span><strong data-value="balance">$50,000</strong></div><div><span>Generation revenue</span><strong data-value="earnings">$0</strong></div><small>Each turbine costs $50,000. Energy earns $75/MWh; removing one recovers $25,000.</small></div><div class="wp-controls-charts"><div class="wp-sliders"><div class="wp-season" data-value="season"></div><div class="wp-compass"><strong>Current wind</strong><svg class="wp-compass-dial" viewBox="0 0 200 200" role="img" data-value="compass" aria-label="Wind from north, airflow toward south"><circle cx="100" cy="100" r="75" fill="#f8fcfc" stroke="#b7d0d6" stroke-width="2"/><circle cx="100" cy="100" r="49" fill="none" stroke="#e1ebed" stroke-dasharray="3 5"/><path d="M100 29 V171 M29 100 H171" stroke="#d3e2e5"/><text x="100" y="17">N</text><text x="186" y="104">E</text><text x="100" y="195">S</text><text x="14" y="104">W</text><g data-value="wind-arrow" transform="rotate(180 100 100)"><path d="M100 146 V59 M84 76 L100 56 L116 76" fill="none" stroke="#16849a" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></g><circle cx="100" cy="100" r="5" fill="#125367"/></svg><output data-out="speed">12.0 m/s</output><span><output data-out="direction">0° from north</output></span><small>Arrow points downwind</small></div></div>
      <div class="wp-charts"><div><strong>Cumulative simulated energy (MWh)</strong><svg class="wp-chart" data-chart="energy" viewBox="0 0 260 120" role="img" aria-label="Estimated cumulative energy over session time"></svg></div><div><strong>Wind speed (m/s)</strong><svg class="wp-chart" data-chart="speed" viewBox="0 0 260 120" role="img" aria-label="Free-stream and turbine wind speeds over session time"></svg></div><div class="wp-chart-legend" data-value="chart-legend"></div><small>Elapsed visible-session time (s); latest 300 samples. Dark: farm total or free stream; colors: turbine locations.</small></div></div></div></div>
    <p class="wp-status" role="status" aria-live="polite">Click the grid to place a turbine. Keyboard: focus the grid, use arrow keys, then Enter or Space.</p>
    <div class="wp-metrics"><div><span>Turbines</span><strong data-value="count">0</strong></div><div><span>Farm power</span><strong data-value="power">0.000 MW</strong></div><div><span>No-wake power</span><strong data-value="ideal">0.000 MW</strong></div><div><span>Wake loss</span><strong data-value="loss">0.000 MW</strong></div><div><span>Efficiency</span><strong data-value="efficiency">—</strong></div></div>
    <p data-value="selection"></p><p class="wp-note">Educational, single-speed top-hat wake model; results are illustrative, not a site assessment.</p>`;
  host.append(root);
  const $=s=>root.querySelector(s), output=n=>$(`[data-out="${n}"]`), value=n=>$(`[data-value="${n}"]`);
  let ids=[], selected=0, scene=null, motion=true, pageActive=!document.hidden;
  let variation=nextSeasonalVariation({speed:0,direction:0}), weather=seasonalWind(0,variation);
  const status=$('.wp-status'), speed=()=>weather.speed, direction=()=>weather.direction;
  const format=n=>(n/1000).toFixed(3)+' MW';
  const history=createHistory(performance.now()), game=createGame(performance.now());
  const money=n=>'$'+Math.floor(n).toLocaleString('en-US');
  function showGame(state){value('balance').textContent=money(state.balance);value('earnings').textContent=money(state.earnings);}
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
    value('energy').textContent=`${state.total.toFixed(2)} MWh`;
    const allIds=[...new Set(state.points.flatMap(p=>[...Object.keys(p.energy),...Object.keys(p.speeds)].map(Number)))].sort((a,b)=>a-b);
    drawChart('energy',state.points,allIds);drawChart('speed',state.points,allIds);
    value('chart-legend').innerHTML='<span><i style="background:#153d55"></i>Farm / free stream</span>'+allIds.map(id=>`<span><i style="background:${turbineColor(id)}"></i>Location ${id+1}: ${(state.energy[id] || 0).toFixed(2)} MWh</span>`).join('');
  }
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function applySeason(elapsed){
    weather=seasonalWind(elapsed,variation);
    value('season').textContent=`Synthetic ${months[weather.month]} · one year / 120 seconds`;
  }
  const historyTimer=setInterval(()=>{
    if(!pageActive)return;
    const now=performance.now(), state=history.tick(now,true);showGame(game.tick(now,true));
    variation=nextSeasonalVariation(variation);applySeason(state.time);redraw();
  },1000);
  const visibility=()=>{const now=performance.now();showHistory(history.tick(now,pageActive));showGame(game.tick(now,pageActive));pageActive=!document.hidden;};
  document.addEventListener('visibilitychange',visibility);
  function redraw(message) {
    const result=evaluate(ids,direction(),speed());
    const now=performance.now();showHistory(history.setModel(result,speed(),now,pageActive));showGame(game.setPower(result.total,now,pageActive));
    output('direction').textContent=`${direction()}° from north`;
    output('speed').textContent=`${speed().toFixed(1)} m/s`;
    value('wind-arrow').setAttribute('transform',`rotate(${(direction()+180)%360} 100 100)`);
    value('compass').setAttribute('aria-label',`Wind from ${direction()} degrees at ${speed().toFixed(1)} meters per second; arrow points downwind.`);
    output('legend-max').textContent=`${speed().toFixed(1)} (free stream)`;
    value('count').textContent=ids.length;value('power').textContent=format(result.total);value('ideal').textContent=format(result.ideal);value('loss').textContent=format(result.loss);
    value('efficiency').textContent=result.efficiency===null?'—':result.efficiency.toFixed(1)+'%';
    const point=gridPoint(selected), turbine=result.results.find(p=>p.id===selected);
    value('selection').textContent=turbine?`Location ${selected+1}: wind ${turbine.speed.toFixed(2)} m/s, turbine power ${turbine.power.toFixed(1)} kW.`:`Location ${selected+1}: east ${point.x} m, north ${point.y} m; ${validCandidate(selected,ids,MIN_SPACING)?'available':'too close to an existing turbine'}.`;
    if(message)status.textContent=message;
    const bounds=scene?.update(ids,selected,null,direction(),speed());output('legend-min').textContent=(bounds?.min ?? 0).toFixed(1);
  }
  function toggle(){
    const now=performance.now();
    if(ids.includes(selected)){game.sell(now,pageActive);ids=ids.filter(i=>i!==selected);redraw(`Turbine removed from location ${selected+1}; ${money(GAME_COST/2)} recovered.`);}
    else if(!validCandidate(selected,ids,MIN_SPACING))redraw('Location is too close to an existing turbine.');
    else if(!game.buy(now,pageActive))redraw(`Not enough budget. A turbine costs ${money(GAME_COST)}; wait for generation revenue.`);
    else {ids=[...ids,selected];redraw(`Turbine purchased and placed at location ${selected+1}.`);}
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
  root.querySelector('[data-action="motion"]').addEventListener('click',()=>{motion=!motion;scene?.setMotion(motion);updateMotionButton();});
  try {scene=mountFarmScene(sceneHost,id=>{selected=id;toggle();});scene.setMotion(motion);}
  catch(error){sceneHost.textContent='3D view unavailable in this browser.';console.warn('Wind farm 3D unavailable',error);}
  function updateMotionButton(){const button=$('[data-action="motion"]'),label=motion?'Stop animation':'Play animation';button.setAttribute('aria-label',label);button.title=label;button.querySelector('span').textContent=motion?'■':'▶';}
  updateMotionButton();applySeason(0);redraw();
  return {destroy(){clearInterval(historyTimer);document.removeEventListener('visibilitychange',visibility);scene?.destroy();root.remove();style.remove();}};
}

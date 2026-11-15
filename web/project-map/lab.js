import shared from '../hydro-lab/styles.css';
import styles from './styles.css';
import solar from '../../EAGE_PythonRenewableEnergyCourse/section8/data/solar.csv';
import wind from '../../EAGE_PythonRenewableEnergyCourse/section8/data/wind.csv';
import hydro from '../../EAGE_PythonRenewableEnergyCourse/section8/data/hydro.csv';
import geothermal from '../../EAGE_PythonRenewableEnergyCourse/section8/data/geothermal.csv';
import demand from '../../EAGE_PythonRenewableEnergyCourse/section8/data/demand.csv';
import { joinProjectInputs } from './data.js';
import { mountProjectScene } from './scene.js';

const assets = {solar,wind,hydro,geothermal,demand};
const formatTime = new Intl.DateTimeFormat('en-GB',{
  day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hourCycle:'h23',timeZone:'UTC',
});
const nodes = [
  ['solar','Solar PV','16 MW DC · 13 MW AC',[
    ['irradiance_wm2','Plane-of-array irradiance','W/m²',0],['ambient_c','Ambient air','°C',1],
  ]],
  ['wind','Wind ridge','12 MW rated · 3 / 12 / 25 m/s thresholds',[
    ['wind_speed_ms','Hub-height wind','m/s',2],
  ]],
  ['hydro','Run-of-river hydro','8 MW rated · 42 m net head · 6 m³/s environmental reservation',[
    ['river_flow_m3s','River flow before reservation','m³/s',2],
  ]],
  ['geothermal','Geothermal','3.5 MW net nameplate · binary-cycle plant',[
    ['mass_flow_kgs','Produced fluid','kg/s',1],['production_c','Production temperature','°C',1],
    ['reinjection_c','Reinjection temperature','°C',1],
  ]],
  ['demand','Community load','Isolated grid · no storage or imports',[
    ['demand_mw','Electricity demand','MW',2],
  ]],
];

export function mountProjectMap(container) {
  const rows=joinProjectInputs(assets);
  const root=container.shadowRoot ?? container.attachShadow({mode:'open'});
  const style=document.createElement('style');
  style.textContent=shared+styles;
  const body=document.createElement('div');
  body.innerHTML=`<article class="hydro-lab project-map" aria-labelledby="map-title">
    <header class="lab-head"><span class="lab-kicker">Final project · isolated grid</span>
      <h2 id="map-title">Four renewable sources, one community</h2>
      <p>Explore the shared grid and compare the five supplied inputs at the same simulated hour.</p>
    </header>
    <div class="map-controls" aria-label="Dataset playback">
      <div class="map-time"><strong id="map-date"></strong><span id="map-index"></span></div>
      <div class="map-buttons"><button type="button" id="map-prev">Previous hour</button>
        <button type="button" id="map-play">Play week</button><button type="button" id="map-next">Next hour</button></div>
    </div>
    <figure class="engineering-card">
      <div id="project-scene" aria-label="Isometric map of solar panels, wind turbines, a run-of-river station, geothermal wells and a town connected to one electrical grid"></div>
      <figcaption>Drag to orbit. Wind, river and daylight cues reflect the selected inputs. Grid lines show connections.</figcaption>
    </figure>
    <section class="map-observations" aria-label="Provided inputs for the selected hour">
      ${nodes.map(([key,title,rating,fields])=>`<section class="map-node map-${key}">
        <h3>${title}</h3><p>${rating}</p>
        <dl>${fields.map(([field,label])=>`<div><dt>${label}</dt><dd id="map-${field}"></dd></div>`).join('')}</dl>
      </section>`).join('')}
    </section>
    <p class="lab-foot">Synthetic input data · 168 one-hour intervals · 1–7 June 2025 UTC. Use the assignment models to calculate generation, curtailment and shortfall.</p>
  </article>`;
  root.replaceChildren(style,body);
  const get=id=>root.getElementById(id);
  let scene;
  try { scene=mountProjectScene(get('project-scene')); }
  catch { get('project-scene').textContent='The 3D map requires WebGL. You can still browse the supplied inputs below.'; }
  let hour=12, timer;
  function paint(){
    const row=rows[hour];
    get('map-date').textContent=formatTime.format(row.time)+' UTC';
    get('map-index').textContent=`Interval ${hour+1} of ${rows.length}`;
    get('map-prev').disabled=hour===0;
    get('map-next').disabled=hour===rows.length-1;
    for(const [,,,fields] of nodes)for(const [field,,unit,digits] of fields)
      get('map-'+field).textContent=row[field].toFixed(digits)+' '+unit;
    scene?.update(row,Boolean(timer) && !matchMedia('(prefers-reduced-motion: reduce)').matches);
  }
  function stop(){
    clearInterval(timer);
    timer=undefined;
    get('map-play').textContent='Play week';
  }
  function step(delta){
    stop();
    hour=Math.max(0,Math.min(rows.length-1,hour+delta));
    paint();
  }
  get('map-prev').onclick=()=>step(-1);
  get('map-next').onclick=()=>step(1);
  get('map-play').onclick=()=>{
    if(timer)stop();
    else {
      if(hour===rows.length-1)hour=0;
      get('map-play').textContent='Pause week';
      timer=setInterval(()=>{
        if(document.hidden || !container.isConnected)return;
        hour++;
        if(hour>=rows.length-1)stop();
        paint();
      },360);
    }
    paint();
  };
  paint();
  return {destroy(){clearInterval(timer);scene?.destroy();root.replaceChildren();}};
}

const container=document.getElementById('project-map-root');
if(container)mountProjectMap(container);

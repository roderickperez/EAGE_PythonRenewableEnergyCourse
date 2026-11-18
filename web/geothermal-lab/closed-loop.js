import {conduction,areas,energy} from './closed-loop-model.js';
import {mountGeothermalSystemScene} from './closed-loop-scene.js';
import css from './closed-loop.css';

const root=document.querySelector('#explorer'),style=document.createElement('style');style.textContent=css;document.head.append(style);
const requested=new URLSearchParams(location.search).get('demo'),mode=['systems','conduction','area','energy'].includes(requested)?requested:'systems';
const fmt=(x,n=2)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:n});
const specs={
 conduction:{title:'How rock conducts heat',intro:'One-dimensional, semi-infinite rock with a fixed cool wall and initially uniform far-field temperature.',controls:[['k','Rock conductivity',.5,5,.1,3,'W/(m K)'],['rho','Rock density',1800,3200,50,2750,'kg/m³'],['cp','Rock heat capacity',500,1400,10,790,'J/(kg K)'],['days','Elapsed time',1,3650,1,365,'days'],['wall','Borehole wall temperature',10,90,5,40,'°C'],['far','Initial rock temperature',120,250,5,200,'°C'],['distance','Examine distance',0,10,.1,1,'m']]},
 area:{title:'Inside a coaxial well',intro:'Change the tube and bore diameters to see how their flow areas compare.',controls:[['inner','Inner tube inside diameter Dᵢ',.04,.14,.005,.1,'m'],['outer','Inner tube outside diameter Dₒ',.06,.20,.005,.12,'m'],['bore','Surrounding passage inside diameter Dᵦ',.12,.30,.005,.18,'m']]},
 energy:{title:'Thermal energy and available work',intro:'Constant inlet/outlet enthalpy and entropy differences over a selected duration.',controls:[['flow','Mass flow ṁ',0,150,1,50,'kg/s'],['dh','Enthalpy difference Δh',0,700,10,300,'kJ/kg'],['ds','Entropy difference Δs',0,1,.01,.2,'kJ/(kg K)'],['ambient','Ambient temperature',270,330,1,300,'K'],['years','Operating duration',0,40,1,1,'years']]}
};
function lineChart(points,xmax,ymin,ymax,xlabel,ylabel,selected){const x=v=>55+v/xmax*500,y=v=>180-(v-ymin)/(ymax-ymin)*155;let grid='';for(let i=0;i<=4;i++){const xx=55+500*i/4;grid+=`<path d="M${xx} 25V180" stroke="#e5eff0"/><text x="${xx}" y="199" text-anchor="middle">${fmt(xmax*i/4,1)}</text>`;}return `<svg viewBox="0 0 600 235" role="img" aria-label="${ylabel} versus ${xlabel}"><g font-size="12" fill="#526b77" font-family="system-ui">${grid}<path d="M55 25V180H555" fill="none" stroke="#9db5bb"/><text x="3" y="32">${fmt(ymax,1)}</text><text x="3" y="182">${fmt(ymin,1)}</text><text x="300" y="224" text-anchor="middle">${xlabel}</text><text x="60" y="16">${ylabel}</text><path d="${points.map((p,i)=>`${i?'L':'M'}${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`).join(' ')}" fill="none" stroke="#16879d" stroke-width="3"/><circle cx="${x(selected[0])}" cy="${y(selected[1])}" r="5" fill="#c44f3e"/></g></svg>`;}
if (mode === 'systems') {
 const views = [
  ['hydrothermal', 'Conventional hydrothermal', 'Brine moves through naturally permeable rock between wells.'],
  ['egs', 'Enhanced geothermal (EGS)', 'Injected water moves through an engineered fracture network.'],
  ['u', 'U-shaped', 'A sealed lateral joins the two well legs.'],
  ['coax', 'Coaxial', 'Fluid goes down the annulus and returns through the inner tube.'],
 ];

 function view([id, name, description]) {
  return `<div class="loop-view-panel" data-panel="${id}" ${id === 'coax' ? 'hidden' : ''}>
    <div class="loop-scene">
      <div class="loop-three" aria-label="${name} geothermal cutaway" role="img"></div>
      <button type="button" class="loop-motion" aria-label="Pause ${name} animation">■</button>
    </div>
    <div class="loop-caption">${description}</div>
  </div>`;
 }

 root.innerHTML = `<section class="loop-card">
  <h2>Compare geothermal flow paths</h2>
  <p>Follow the dots underground. Drag each view to orbit and scroll to zoom.</p>
  <div class="loop-scenes">
   <article class="loop-view">
    <h3>Conventional hydrothermal</h3>
    ${view(views[0])}
   </article>
   <article class="loop-view">
    <h3>Enhanced geothermal (EGS)</h3>
    ${view(views[1])}
    <div class="loop-option">
     <label><input id="egs-connection" type="checkbox" checked> Fractures connect both wells</label>
     <span id="egs-status" role="status">Water reaches the production well.</span>
    </div>
   </article>
   <article class="loop-view">
    <h3>Deep closed loop</h3>
    ${view(views[2])}${view(views[3])}
    <div class="loop-option">
     <label for="closed-geometry">Well geometry</label>
     <select id="closed-geometry"><option value="u">U-shaped</option><option value="coax">Coaxial</option></select>
    </div>
   </article>
  </div>
  <div class="loop-legend"><span>● Cool inlet</span><span>● Warm return</span></div>
  <p class="loop-note">Dots and dashed traces show illustrative flow paths. Well trajectories, fracture size and flow speed are schematic.</p>
 </section>`;

 const scenes = new Map();
 function mount([id, name]) {
  if (scenes.has(id)) return;
  const panel = root.querySelector(`[data-panel="${id}"]`);
  const target = panel.querySelector('.loop-three');
  const button = panel.querySelector('.loop-motion');
  let scene;
  try {
   scene = mountGeothermalSystemScene(target, id);
   scenes.set(id, scene);
  } catch (error) {
   target.textContent = '3D view unavailable.';
   button.disabled = true;
   console.warn(error);
   return;
  }
  let motion = true;
  button.addEventListener('click', () => {
   motion = !motion;
   button.textContent = motion ? '■' : '▶';
   button.setAttribute('aria-label', `${motion ? 'Pause' : 'Play'} ${name} animation`);
   scene.setMotion(motion);
  });
 }

 views.slice(0, 3).forEach(mount);
 root.querySelector('#egs-connection').addEventListener('change', event => {
  const connected = event.target.checked;
  scenes.get('egs')?.setConnected(connected);
  root.querySelector('#egs-status').textContent = connected
   ? 'Water reaches the production well.'
   : 'The interrupted paths stop circulation between wells.';
 });
 root.querySelector('#closed-geometry').addEventListener('change', event => {
  const geometry = event.target.value;
  for (const id of ['u', 'coax']) root.querySelector(`[data-panel="${id}"]`).hidden = id !== geometry;
  mount(views.find(([id]) => id === geometry));
 });
 window.addEventListener('pagehide', () => {
  for (const scene of scenes.values()) scene.destroy();
 });
} else {
 const spec=specs[mode];root.innerHTML=`<section class="loop-card"><h2>${spec.title}</h2><p>${spec.intro}</p><div class="loop-result" aria-live="polite"></div><div class="loop-plot"></div><div class="loop-controls">${spec.controls.map(([id,label,min,max,step,value,unit])=>`<label>${label}<output data-out="${id}"></output><input type="range" data-id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"><small>${unit}</small></label>`).join('')}</div><p class="loop-note"></p></section>`;
 const $=s=>root.querySelector(s),read=()=>Object.fromEntries(spec.controls.map(([id])=>[id,$(`[data-id="${id}"]`).valueAsNumber]));
 function update(){const p=read();for(const [id] of spec.controls)$(`[data-out="${id}"]`).textContent=fmt(p[id],4);try{
  if(mode==='conduction'){const a=conduction(p),points=Array.from({length:81},(_,i)=>{const d=i/8;return [d,conduction({...p,distance:d}).temperature];});$('.loop-result').textContent=`α = ${a.alpha.toExponential(2)} m²/s · effusivity = ${fmt(a.effusivity,0)} J/(m² K √s) · T(${p.distance} m) = ${fmt(a.temperature,1)} °C`;$('.loop-plot').innerHTML=lineChart(points,10,p.wall,p.far,'Distance from wall (m)','Rock temperature (°C)',[p.distance,a.temperature]);$('.loop-note').textContent='The red point shows the selected distance. The line is the 1D constant-wall-temperature solution; it does not model cylindrical geometry, fluid flow, or a whole geothermal well.';}
  if(mode==='area'){const a=areas(p),scale=145/p.bore,outer=scale*p.bore/2,pipe=scale*p.outer/2,inner=scale*p.inner/2;$('.loop-result').textContent=`Tube = ${fmt(a.tube,4)} m² · annulus = ${fmt(a.annulus,4)} m² · ratio = ${fmt(a.ratio,2)}`;$('.loop-plot').innerHTML=`<svg viewBox="0 0 600 190" role="img" aria-label="Cross-section showing tube, wall and annular flow"><circle cx="145" cy="95" r="${outer}" fill="#80cfe0" stroke="#258ac0" stroke-width="2"/><circle cx="145" cy="95" r="${pipe}" fill="#8b9a9d"/><circle cx="145" cy="95" r="${inner}" fill="#ed8d70"/><g font-family="system-ui" font-size="14" fill="#315462"><text x="280" y="62">Blue: annular flow area</text><text x="280" y="92">Gray: inner pipe wall</text><text x="280" y="122">Orange: inner tube flow area</text><text x="280" y="152">Diameters are shown to scale.</text></g></svg>`;$('.loop-note').textContent='The areas describe geometry only; pressure drop and heat transfer require additional fluid and pipe properties.';}
  if(mode==='energy'){const a=energy(p),max=Math.max(Math.abs(a.thermal),Math.abs(a.available),.1)*1.2,scale=v=>275+v/max*220;$('.loop-result').textContent=`Thermal: ${fmt(a.thermal,2)} MW · available work: ${fmt(a.available,2)} MW`;$('.loop-plot').innerHTML=`<svg viewBox="0 0 600 185" role="img" aria-label="Thermal power and ideal available work comparison"><path d="M275 20V145" stroke="#95aeb5"/><rect x="${Math.min(275,scale(a.thermal))}" y="40" width="${Math.abs(scale(a.thermal)-275)}" height="30" fill="#d46b42"/><rect x="${Math.min(275,scale(a.available))}" y="100" width="${Math.abs(scale(a.available)-275)}" height="30" fill="#2589aa"/><g font-family="system-ui" fill="#315462" font-size="14"><text x="8" y="60">Thermal</text><text x="8" y="120">Available work</text><text x="275" y="165" text-anchor="middle">0 MW</text><text x="555" y="165" text-anchor="end">${fmt(max,1)} MW</text></g></svg>`;$('.loop-note').textContent=`Over ${p.years} years at constant rates: ${fmt(a.thermalMWh,0)} MWh thermal and ${fmt(a.availableMWh,0)} MWh ideal available work. Values are signed; independently chosen Δh and Δs need not describe a physically realizable fluid state. This is not net plant generation.`;}
 }catch(e){$('.loop-result').textContent=e.message;$('.loop-plot').innerHTML='';$('.loop-note').textContent='Adjust the inputs to continue.';}}
 for(const el of root.querySelectorAll('input'))el.addEventListener('input',update);update();
}
const frame = window.frameElement;
if (frame) {
 new ResizeObserver(() => {
  frame.style.height = Math.ceil(root.getBoundingClientRect().height + 4) + 'px';
 }).observe(root);
}

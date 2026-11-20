import {mountGeothermalSystemScene} from './closed-loop-scene.js';
import css from './closed-loop.css';
import { mountCoaxialCutaway } from './coaxial-cutaway.js';
import { mountClosedLoopEquation } from './closed-loop-equations.js';
import { mountClosedLoopPerformance } from './closed-loop-performance.js';

const root=document.querySelector('#explorer'),style=document.createElement('style');style.textContent=css;document.head.append(style);
const requested=new URLSearchParams(location.search).get('demo'),mode=['systems','conduction','area','energy','performance'].includes(requested)?requested:'systems';
const specs={
 conduction:{title:'How rock conducts heat',intro:'One-dimensional, semi-infinite rock with a fixed cool wall and initially uniform far-field temperature.',controls:[['k','Rock conductivity',.5,5,.1,3,'W/(m K)'],['rho','Rock density',1800,3200,50,2750,'kg/m³'],['cp','Rock heat capacity',500,1400,10,790,'J/(kg K)'],['days','Elapsed time',1,3650,1,365,'days'],['wall','Borehole wall temperature',10,90,5,40,'°C'],['far','Initial rock temperature',120,250,5,200,'°C'],['distance','Examine distance',0,10,.1,1,'m']]},
 area:{title:'Inside a coaxial well',intro:'Change the tube and bore diameters to see how their flow areas compare.',controls:[['inner','Inner tube inside diameter Dᵢ',.04,.14,.005,.1,'m'],['outer','Inner tube outside diameter Dₒ',.06,.20,.005,.12,'m'],['bore','Surrounding passage inside diameter Dᵦ',.12,.30,.005,.18,'m']]},
 energy:{title:'Thermal energy and available work',intro:'Constant inlet/outlet enthalpy and entropy differences over a selected duration.',controls:[['flow','Mass flow ṁ',0,150,1,50,'kg/s'],['dh','Enthalpy difference Δh',0,700,10,300,'kJ/kg'],['ds','Entropy difference Δs',0,1,.01,.2,'kJ/(kg K)'],['ambient','Ambient temperature',270,330,1,300,'K'],['years','Operating duration',0,40,1,1,'years']]}
};
if (mode === 'performance') {
 mountClosedLoopPerformance(root);
} else if (mode === 'systems') {
 const views = [
  ['hydrothermal', 'Conventional hydrothermal', 'Brine moves through naturally permeable rock between wells.'],
  ['egs', 'Enhanced geothermal (EGS)', 'Injected water moves through an engineered fracture network.'],
  ['u', 'Deep closed loop | U-Shaped', 'A sealed lateral joins the two well legs.'],
  ['coax', 'Deep closed loop | Coaxial', 'Cool fluid descends through the annulus. Warm fluid rises through the inner tube in the same well.'],
 ];

 function view([id, name, description]) {
  const options = id === 'egs' ? `<div class="loop-option">
    <label><input id="egs-connection" type="checkbox" checked> Fractures connect both wells</label>
    <span id="egs-status" role="status">Water reaches the production well.</span>
  </div>` : '';
  return `<article class="loop-view" data-panel="${id}">
    <h3>${name}</h3>
    <div class="loop-scene">
      <div class="loop-three" aria-label="${name} geothermal cutaway" role="img"></div>
      <button type="button" class="loop-motion" aria-label="Pause ${name} animation">■</button>
    </div>
    <div class="loop-caption">${description}</div>
    ${options}
  </article>`;
 }

 root.innerHTML = `<section class="loop-card">
  <h2>Compare geothermal flow paths</h2>
  <p>Follow cool blue flow down on the right. In the two-well views, warm red flow rises on the left. Drag to orbit and scroll to zoom.</p>
  <div class="loop-scenes">${views.map(view).join('')}</div>
  <div class="loop-legend"><span>● Cool inlet</span><span>● Warm return</span></div>
  <p class="loop-note">Dots and dashed traces show illustrative flow paths. Well trajectories, fracture size and flow speed are schematic.</p>
 </section>`;

 const scenes = new Map();
 function mount([id, name]) {
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

 views.forEach(mount);
 root.querySelector('#egs-connection').addEventListener('change', event => {
  const connected = event.target.checked;
  scenes.get('egs')?.setConnected(connected);
  root.querySelector('#egs-status').textContent = connected
   ? 'Water reaches the production well.'
   : 'The interrupted paths stop circulation between wells.';
 });
 window.addEventListener('pagehide', () => {
  for (const scene of scenes.values()) scene.destroy();
 });
 } else {
 mountClosedLoopEquation(root, mode, specs[mode], mountCoaxialCutaway);
}
const frame = window.frameElement;
if (frame) {
 new ResizeObserver(() => {
  frame.style.height = Math.ceil(root.getBoundingClientRect().height + 4) + 'px';
 }).observe(root);
}

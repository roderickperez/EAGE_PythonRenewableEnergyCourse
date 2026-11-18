import styles from './styles.css';
import { HydroScene } from './scene.js';
import { fitFrameToContent } from './fit-frame.js';
import { updateMotionControl } from '../motion-control.js';
import { CAPACITY, DEAD, HORIZON, initialState, step } from './project-model.js';

const host = document.querySelector('#root').attachShadow({mode:'open'});
host.innerHTML = `<style>${styles}
.project{font:14px/1.45 Inter,"Segoe UI",system-ui,sans-serif;color:#16384c;background:#f5f9fb;padding:16px;border:1px solid #cbdde4;border-radius:14px;box-sizing:border-box}
.project *{box-sizing:border-box}.project h2{margin:0 0 6px;padding:0;border:0;font-family:inherit;font-size:1.2em;font-weight:700;line-height:1.25;letter-spacing:normal}.project h3{font-family:inherit;font-weight:700}.project .top p{margin:0 0 12px;color:#425c6b;font-size:.85em}.project button{font:inherit}.project input{accent-color:#328baa}.project .scene-wrap{border:1px solid #bcd2db;border-radius:10px;overflow:hidden;background:#e6eff3}.project .scene-control{background:#246985;border:1px solid #1a5a76;color:white}.project .scene-control:hover{background:#1a5a76}.project .scene-control:disabled{opacity:.5;cursor:not-allowed}.project #reset{font-size:26px}.project .legend{font-weight:400}.project #status{background:#e1f1ef;border-left:4px solid #328baa;padding:8px;font-size:.9em}
.top{display:flex;justify-content:space-between;gap:1rem;align-items:center}.grid{display:grid;grid-template-columns:minmax(240px,1.05fr) minmax(230px,.85fr) minmax(280px,1fr);gap:12px;align-items:start}.box{min-width:0;padding:12px;background:white;border:1px solid #d3e3e7;border-radius:9px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metrics div{background:#eef6f7;padding:8px;border-radius:6px}.metrics strong,.metrics span{display:block}.metrics span{font-size:.8rem}button{background:#246985;color:white;border:0;padding:9px 16px;border-radius:7px;cursor:pointer}input{width:100%}canvas.chart{width:100%;height:180px;display:block}.chartbox+.chartbox{margin-top:12px}.chartbox h3{font-size:1rem;margin:0 0 4px}.legend{font-size:.78rem;margin:2px 0;color:#35545d}.blue{color:#168fb2}.green{color:#27876c}.gold{color:#b8780c}.orange{color:#b65519}#plant-scene{height:300px;position:relative}#plant-scene canvas{width:100%;height:100%;display:block}.plant-tooltip{position:absolute;background:#173e4b;color:white;padding:6px;border-radius:4px;pointer-events:none}.scene-wrap{position:relative}.scene-control{position:absolute;z-index:2;top:8px;width:40px;height:40px;padding:0;background:white;color:#16384c;border:1px solid #bdcfd6;border-radius:7px}.scene-control:focus-visible{outline:3px solid #e7ad31;outline-offset:2px}#reset{left:8px;font-size:26px;line-height:1}#run{right:8px}@media(max-width:900px){.grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.plots{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.chartbox+.chartbox{margin-top:0}}@media(max-width:600px){.grid,.plots{grid-template-columns:1fr}.plots{grid-column:auto}.top{align-items:flex-start;flex-wrap:wrap}#plant-scene{height:260px}}
</style><article class="project"><div class="top"><div><h2>Hydropower reservoir challenge</h2><p>Choose turbine discharge X to generate energy while protecting the reservoir.</p></div></div><div class="grid"><div class="box"><div class="scene-wrap"><div id="plant-scene" aria-label="Animated hydroelectric plant"></div><button id="reset" class="scene-control" title="Reset simulation" aria-label="Reset simulation">↻</button><button id="run" class="scene-control" aria-label="Stop simulation"></button></div><small>Drag to rotate and scroll to zoom. The water surface and flow respond to the simulation; geometry is schematic.</small></div><div class="box"><label for="flow"><strong>Requested turbine flow X</strong> <output id="flow-value"></output></label><input id="flow" type="range" min="0" max="10" step="0.1" value="4"><p>Inflow changes seasonally. Turbine efficiency declines automatically with operating time. The model limits withdrawals to preserve the reserve.</p><div class="metrics" id="metrics"></div><p id="status" role="status"></p></div><div class="plots"><div class="box chartbox"><h3>Reservoir storage and minimum reserve</h3><p class="legend"><span class="blue">━ Storage</span> · <span class="orange">╌ Minimum reserve</span></p><canvas class="chart" id="storage" role="img" aria-label="Reservoir storage and minimum reserve over simulated days"></canvas></div><div class="box chartbox"><h3>Flow and generated power over time</h3><p class="legend"><span class="blue">━ Inflow</span> · <span class="green">━ Turbine flow</span> · <span class="gold">━ Power</span></p><canvas class="chart" id="flows" role="img" aria-label="Inflow, turbine flow and power over simulated days"></canvas></div></div></div><p>One second = one simulated day. The run stops after 365 days. Reset to compare policies using the same inflow.</p></article>`;
const $ = id => host.getElementById(id), sceneHost = $('plant-scene');
let scene; try { scene = new HydroScene(sceneHost); } catch { sceneHost.textContent = '3D view unavailable; the simulation and plots remain usable.'; }
scene?.toggle.remove();
let state = initialState(), history = [{...state,inflow:0,turbineFlow:0,power:0}], running = true, last = 0, carry = 0;
const n = (v,d=1) => v.toFixed(d);
function chart(id, series, max, unit) {
  const c=$(id), w=c.clientWidth, h=c.clientHeight, ratio=devicePixelRatio||1; c.width=w*ratio;c.height=h*ratio;const x=c.getContext('2d');x.scale(ratio,ratio);x.clearRect(0,0,w,h);
  const left=48, right=w-12, top=15, bottom=h-30; x.font='11px system-ui';x.fillStyle='#35545d';x.strokeStyle='#d7e2e4';
  for(let i=0;i<=4;i++){const y=bottom-(bottom-top)*i/4;x.beginPath();x.moveTo(left,y);x.lineTo(right,y);x.stroke();x.fillText(n(max*i/4,1),3,y+4)}
  x.fillText(unit,3,10); x.fillText('Day 0',left,bottom+20);x.fillText('365',right-24,bottom+20);
  for(const s of series){x.strokeStyle=s.color;x.lineWidth=2;x.setLineDash(s.dash?[5,4]:[]);x.beginPath();if(s.constant){const y=bottom-(bottom-top)*s.value()/max;x.moveTo(left,y);x.lineTo(right,y)}else{history.forEach((p,i)=>{if(s.skipInitial&&i===0)return;const px=left+(right-left)*p.day/HORIZON, py=bottom-(bottom-top)*s.value(p)/max;(i===(s.skipInitial?1:0))?x.moveTo(px,py):x.lineTo(px,py)})}x.stroke();}x.setLineDash([]);
}
function render(){
  $('flow-value').value=`${n(+$('flow').value)} m³/s`;
  const rows=[['Day',`${state.day} / 365`],['Storage',`${n(state.storage/1e6,2)} hm³`],['Inflow',`${n(state.inflow||0,2)} m³/s`],['Turbine flow',`${n(state.turbineFlow||0,2)} m³/s`],['Net head',`${n(state.head||84,1)} m`],['Turbine efficiency',`${n((state.efficiency??.91)*100,2)}%`],['Power',`${n(state.power||0,2)} MW`],['Energy',`${n(state.energy,0)} MWh`],['Total spill',`${n(state.spill/1e6,2)} hm³`],['Eco shortage',`${n(state.shortage/1e6,3)} hm³`]];
  $('metrics').innerHTML=rows.map(([a,b])=>`<div><span>${a}</span><strong>${b}</strong></div>`).join('');
  $('status').textContent=state.day===365?'Year complete. Reset to compare another strategy.':state.storage<=DEAD+1?'Reserve reached: turbine withdrawal is limited.':'Keep storage above the 5 hm³ reserve and avoid spill.';
  // Map storage fraction to the lesson scene's schematic water-height scale.
  scene?.update({inputs:{grossHeadM:5+245*state.storage/CAPACITY,ratingMW:10},turbineFlowM3s:state.turbineFlow||0,spillFlowM3s:state.spillFlow||0,plantMW:state.power||0});
  chart('storage',[{color:'#168fb2',value:p=>p.storage/1e6},{color:'#c46c32',dash:true,constant:true,value:()=>DEAD/1e6}],50,'hm³');
  chart('flows',[{color:'#168fb2',skipInitial:true,value:p=>p.inflow},{color:'#27876c',skipInitial:true,value:p=>p.turbineFlow},{color:'#d99124',skipInitial:true,value:p=>p.power}],10,'m³/s · MW');
}
function syncControls() {
  if (scene) scene.animate = running;
  $('run').disabled = state.day >= HORIZON;
  updateMotionControl($('run'), running, 'simulation');
}
$('flow').oninput=render;
$('run').onclick=()=>{running=!running;last=0;syncControls()};
$('reset').onclick=()=>{state=initialState();history=[{...state,inflow:0,turbineFlow:0,power:0}];running=true;last=0;carry=0;syncControls();render()};
function tick(t){if(running&&!document.hidden){if(last)carry+=Math.min((t-last)/1000,1);while(carry>=1&&state.day<HORIZON){state=step(state,+$('flow').value);history.push(state);carry--;render()}if(state.day===HORIZON){running=false;syncControls()}}last=t;requestAnimationFrame(tick)}
window.addEventListener('resize',render);syncControls();render();fitFrameToContent(host.querySelector('.project'));requestAnimationFrame(tick);

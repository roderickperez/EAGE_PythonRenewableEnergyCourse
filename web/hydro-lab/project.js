import styles from './styles.css';
import { HydroScene } from './scene.js';
import { CAPACITY, DEAD, HORIZON, initialState, step } from './project-model.js';

const host = document.querySelector('#root').attachShadow({mode:'open'});
host.innerHTML = `<style>${styles}
.project{font:15px/1.45 system-ui;color:#16384c;background:#f5f9fb;padding:16px;border:1px solid #cbdde4;border-radius:14px}
.top{display:flex;justify-content:space-between;gap:1rem;align-items:center}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(230px,1fr);gap:12px}.box{padding:12px;background:white;border:1px solid #d3e3e7;border-radius:9px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metrics div{background:#eef6f7;padding:8px;border-radius:6px}.metrics strong,.metrics span{display:block}.metrics span{font-size:.8rem}button{background:#246985;color:white;border:0;padding:9px 16px;border-radius:7px;cursor:pointer}input{width:100%}canvas.chart{width:100%;height:220px;display:block}.chartbox{margin-top:12px}#plant-scene{height:420px;position:relative}#plant-scene canvas{width:100%;height:100%;display:block} .plant-tooltip{position:absolute;background:#173e4b;color:white;padding:6px;border-radius:4px;pointer-events:none}.motion-toggle{position:absolute;top:8px;right:8px}@media(max-width:700px){.grid{grid-template-columns:1fr}#plant-scene{height:330px}}
</style><article class="project"><div class="top"><div><h2>Hydropower reservoir challenge</h2><p>Choose turbine discharge X to generate energy while protecting the reservoir.</p></div><div><button id="run">Pause</button> <button id="reset">Reset</button></div></div><div class="grid"><div class="box"><div id="plant-scene" aria-label="Animated hydroelectric plant"></div><small>Drag to rotate and scroll to zoom. The water surface and flow respond to the simulation; geometry is schematic.</small></div><div class="box"><label for="flow"><strong>Requested turbine flow X</strong> <output id="flow-value"></output></label><input id="flow" type="range" min="0" max="10" step="0.1" value="4"><p>Inflow changes seasonally. Turbine efficiency declines automatically with operating time. The model limits withdrawals to preserve the reserve.</p><div class="metrics" id="metrics"></div><p id="status" role="status"></p></div></div><div class="box chartbox"><h3>Reservoir storage and minimum reserve</h3><canvas class="chart" id="storage" role="img" aria-label="Reservoir storage over simulated days"></canvas></div><div class="box chartbox"><h3>Flow and generated power over time</h3><canvas class="chart" id="flows" role="img" aria-label="Inflow, turbine flow and power over simulated days"></canvas></div><p>One second = one simulated day. The run stops after 365 days. Reset to compare policies using the same inflow.</p></article>`;
const $ = id => host.getElementById(id), sceneHost = $('plant-scene');
let scene; try { scene = new HydroScene(sceneHost); } catch { sceneHost.textContent = '3D view unavailable; the simulation and plots remain usable.'; }
let state = initialState(), history = [{...state,inflow:0,turbineFlow:0,power:0}], running = true, last = 0, carry = 0;
const n = (v,d=1) => v.toFixed(d);
function chart(id, series, max, unit) {
  const c=$(id), w=c.clientWidth, h=220, ratio=devicePixelRatio||1; c.width=w*ratio;c.height=h*ratio;const x=c.getContext('2d');x.scale(ratio,ratio);x.clearRect(0,0,w,h);
  const left=48, right=w-12, top=15, bottom=h-30; x.font='11px system-ui';x.fillStyle='#35545d';x.strokeStyle='#d7e2e4';
  for(let i=0;i<=4;i++){const y=bottom-(bottom-top)*i/4;x.beginPath();x.moveTo(left,y);x.lineTo(right,y);x.stroke();x.fillText(n(max*i/4,1),3,y+4)}
  x.fillText(unit,3,10); x.fillText('Day 0',left,bottom+20);x.fillText('365',right-24,bottom+20);
  for(const s of series){x.strokeStyle=s.color;x.lineWidth=2;x.setLineDash(s.dash?[5,4]:[]);x.beginPath();history.forEach((p,i)=>{const px=left+(right-left)*p.day/HORIZON, py=bottom-(bottom-top)*s.value(p)/max;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();}x.setLineDash([]);
}
function render(){
  $('flow-value').value=`${n(+$('flow').value)} m³/s`;
  const rows=[['Day',`${state.day} / 365`],['Storage',`${n(state.storage/1e6,2)} hm³`],['Inflow',`${n(state.inflow||0,2)} m³/s`],['Turbine flow',`${n(state.turbineFlow||0,2)} m³/s`],['Net head',`${n(state.head||84,1)} m`],['Turbine efficiency',`${n((state.efficiency??.91)*100,2)}%`],['Power',`${n(state.power||0,2)} MW`],['Energy',`${n(state.energy,0)} MWh`],['Total spill',`${n(state.spill/1e6,2)} hm³`],['Eco shortage',`${n(state.shortage/1e6,3)} hm³`]];
  $('metrics').innerHTML=rows.map(([a,b])=>`<div><span>${a}</span><strong>${b}</strong></div>`).join('');
  $('status').textContent=state.day===365?'Year complete. Reset to compare another strategy.':state.storage<=DEAD+1?'Reserve reached: turbine withdrawal is limited.':'Keep storage above the 5 hm³ reserve and avoid spill.';
  // Map storage fraction to the lesson scene's schematic water-height scale.
  scene?.update({inputs:{grossHeadM:5+245*state.storage/CAPACITY,ratingMW:10},turbineFlowM3s:state.turbineFlow||0,spillFlowM3s:state.spillFlow||0,plantMW:state.power||0});
  chart('storage',[{color:'#168fb2',value:p=>p.storage/1e6},{color:'#c46c32',dash:true,value:()=>DEAD/1e6}],50,'hm³');
  chart('flows',[{color:'#168fb2',value:p=>p.inflow||0},{color:'#27876c',value:p=>p.turbineFlow||0},{color:'#d99124',value:p=>p.power||0}],10,'m³/s · MW');
}
$('flow').oninput=render;$('run').onclick=()=>{running=!running;$('run').textContent=running?'Pause':'Resume';last=0};$('reset').onclick=()=>{state=initialState();history=[{...state,inflow:0,turbineFlow:0,power:0}];running=true;last=0;carry=0;$('run').textContent='Pause';render()};
function tick(t){if(running&&!document.hidden){if(last)carry+=Math.min((t-last)/1000,1);while(carry>=1&&state.day<HORIZON){state=step(state,+$('flow').value);history.push(state);carry--;render()}if(state.day===HORIZON){running=false;$('run').textContent='Resume';$('run').disabled=true}}last=t;requestAnimationFrame(tick)}
$('reset').addEventListener('click',()=>{$('run').disabled=false});window.addEventListener('resize',render);render();requestAnimationFrame(tick);

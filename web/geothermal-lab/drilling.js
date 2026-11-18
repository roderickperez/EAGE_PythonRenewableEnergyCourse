import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {drilling} from './drilling-model.js';
import baseCss from '../wind-lab/project.css';
import css from './drilling.css';

export function mountDrillingExplorer(host){
 const style=document.createElement('style');style.textContent=baseCss+'\n'+css;host.append(style);
 const root=document.createElement('section');root.className='wind-project drilling-explorer';root.innerHTML=`<div class="wp-heading"><div><h2>Geothermal drilling fluid explorer</h2><p>Follow fluid down the drill pipe and compare cuttings motion in the annulus. Drag to rotate and scroll to zoom. The rig and particle speeds are schematic.</p></div></div><div class="wp-layout"><div><div class="wp-scene-wrap"><div class="wp-scene" role="img" aria-label="Animated vertical drilling circulation and pressure versus depth chart"></div><button type="button" class="wp-motion" data-action="motion" aria-label="Pause animation">■</button></div><p class="drill-key"><span>● Blue: down drill pipe</span><span>● Gold: cuttings in the annulus</span><span>━ Blue: hydrostatic</span><span>━ Red: circulating</span></p><small>Pressure chart at right: horizontal axis 0 to <output data-v="scale">—</output> MPa; vertical axis surface to selected true vertical depth. Circulating line assumes the chosen annular pressure loss is distributed uniformly with depth.</small></div><div class="wp-settings"><div class="geo-controls"><label>True vertical depth <output data-o="depth"></output><input data-input="depth" type="range" min="500" max="4000" step="100" value="2000"></label><label>Flow rate <output data-o="flow"></output><input data-input="flow" type="range" min="0" max="3000" step="100" value="1200"></label><label>Hole diameter <output data-o="hole"></output><input data-input="hole" type="range" min="0.18" max="0.35" step="0.005" value="0.215"></label><label>Pipe outside diameter <output data-o="pipe"></output><input data-input="pipe" type="range" min="0.08" max="0.16" step="0.005" value="0.125"></label><label>Fluid density <output data-o="density"></output><input data-input="density" type="range" min="900" max="1500" step="25" value="1100"></label><label>Annular friction loss <output data-o="friction"></output><input data-input="friction" type="range" min="0" max="5" step="0.1" value="1"></label><label>Fluid viscosity <output data-o="viscosity"></output><input data-input="viscosity" type="range" min="0.005" max="0.1" step="0.005" value="0.03"></label><label>Cutting diameter <output data-o="particle"></output><input data-input="particle" type="range" min="0.0001" max="0.003" step="0.0001" value="0.0002"></label></div><small>The annular friction loss is a user-specified assumption; the explorer does not calculate pump hydraulics or change friction automatically when flow changes.</small></div></div><div class="wp-metrics drill-metrics"><div><span>Annular area</span><strong data-v="area">—</strong></div><div><span>Annular velocity</span><strong data-v="velocity">—</strong></div><div><span>Static bottom pressure</span><strong data-v="hydro">—</strong></div><div><span>Circulating pressure</span><strong data-v="bottom">—</strong></div><div><span>Equivalent circulating density</span><strong data-v="ecd">—</strong></div></div><p class="wp-status" role="status" data-v="slip"></p><p class="wp-note">Illustrative vertical, single-phase model with constant properties. No formation pressure, temperature, losses, non-Newtonian rheology, gas or real cuttings transport is modeled. Do not use for well design or operations.</p>`;host.append(root);
 const sceneContainer=root.querySelector('.wp-scene-wrap').parentElement;
 sceneContainer.querySelector('.wp-scene').setAttribute('aria-label','Animated vertical drilling circulation');
 sceneContainer.querySelector('small').remove();
 sceneContainer.querySelector('.drill-key').innerHTML='<span>● Blue: down drill pipe</span><span>● Gold: cuttings in the annulus</span>';
 const figure=document.createElement('figure');figure.className='drill-pressure';
 figure.innerHTML='<h3>Pressure versus true vertical depth</h3><svg viewBox="0 0 620 280" role="img" aria-label="Hydrostatic and circulating pressure versus depth"></svg><figcaption>Blue: hydrostatic · Red: circulating. The specified annular friction loss is assumed to increase uniformly with depth.</figcaption>';
 root.querySelector('.wp-layout').after(figure);
 const $=s=>root.querySelector(s), value=k=>$(`[data-v="${k}"]`),controls=['depth','flow','hole','pipe','density','friction','viscosity','particle'];let state,scene,running=true;
 const units={depth:'m TVD',flow:'L/min',hole:'m',pipe:'m OD',density:'kg/m³',friction:'MPa',viscosity:'Pa s',particle:'m'};
 function update(){const p={};for(const k of controls){const el=$(`[data-input="${k}"]`);p[k]=el.valueAsNumber;$(`[data-o="${k}"]`).textContent=el.value+' '+units[k];}try{state=drilling(p);value('area').textContent=state.area.toFixed(4)+' m²';value('velocity').textContent=state.velocity.toFixed(2)+' m/s';value('hydro').textContent=(state.hydro/1e6).toFixed(2)+' MPa';value('bottom').textContent=(state.bottom/1e6).toFixed(2)+' MPa';value('ecd').textContent=state.equivalent.toFixed(0)+' kg/m³';drawPressure(figure.querySelector('svg'),state,p.depth);value('slip').textContent=state.validStokes?`Ideal spherical-particle Stokes estimate: slip ${state.slip.toFixed(4)} m/s, upward transport ${state.transport.toFixed(3)} m/s, particle Reynolds ${state.particleRe.toFixed(3)}. This is not a hole-cleaning criterion.`:`Particle Reynolds ${state.particleRe.toFixed(1)} exceeds the Stokes creeping-flow range; slip and transport estimates are not displayed.`;scene?.update(state,p);}catch{value('slip').textContent='Check inputs: hole diameter must exceed pipe diameter.';}}
 for(const k of controls)$(`[data-input="${k}"]`).addEventListener('input',update);
 const button=$('[data-action="motion"]');button.onclick=()=>{running=!running;button.textContent=running?'■':'▶';button.setAttribute('aria-label',running?'Pause animation':'Play animation');scene?.setRunning(running);};
 try{scene=makeScene($('.wp-scene'));}catch(e){$('.wp-scene').textContent='3D view unavailable; the calculations remain available.';console.warn(e);}update();
 return {destroy(){scene?.destroy();style.remove();root.remove();}};
}

function drawPressure(svg,r,depth){
 const left=70,top=20,w=500,h=205,max=r.bottom*1.2,x=p=>left+w*p/max;
 let markup='<rect width="620" height="280" fill="#fff"/><g stroke="#e2ecee" stroke-width="1">';
 for(let i=0;i<=4;i++){const y=top+h*i/4;markup+=`<path d="M${left} ${y}h${w}"/>`;}
 markup+='</g><path d="M70 20V225H570" fill="none" stroke="#8ba7ae"/>';
 markup+=`<path d="M70 20L${x(r.hydro)} 225" fill="none" stroke="#258ac0" stroke-width="3"/><path d="M70 20L${x(r.bottom)} 225" fill="none" stroke="#d4553f" stroke-width="3"/>`;
 markup+=`<g fill="#526b77" font-family="system-ui" font-size="13"><text x="15" y="25">0 m</text><text x="6" y="228">${depth} m</text><text x="70" y="245">0</text><text x="500" y="245">${(max/1e6).toFixed(1)} MPa</text><text x="250" y="270">Pressure (MPa)</text></g>`;
 svg.innerHTML=markup;
}

function makeScene(host){
 const scene=new T.Scene();scene.background=new T.Color('#e6eff3');const renderer=new T.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=T.SRGBColorSpace;host.append(renderer.domElement);
 const camera=new T.OrthographicCamera(-13,13,10,-10,.1,100);camera.position.set(10,11,28);const orbit=new OrbitControls(camera,renderer.domElement);Object.assign(orbit,{enablePan:false,minZoom:.7,maxZoom:2.5});orbit.target.set(-5,0,0);orbit.update();scene.add(new T.HemisphereLight(0xe6f5ff,0x7f8568,2.4));const light=new T.DirectionalLight(0xffedcd,2);light.position.set(-7,18,10);scene.add(light);
 function box(w,h,d,col,x,y,z){const o=new T.Mesh(new T.BoxGeometry(w,h,d),new T.MeshStandardMaterial({color:col}));o.position.set(x,y,z);scene.add(o);return o;}
 box(7,.3,4,'#657e62',-5,4,0);box(3,.35,2,'#c6d4d4',-5,4.35,0);box(.16,4,.16,'#c17b45',-5,6.5,0);box(.16,4,.16,'#c17b45',-3.6,6.5,0);box(1.6,.18,.4,'#c17b45',-4.3,8.4,0);
 const outer=new T.Mesh(new T.CylinderGeometry(1,1,10,32,1,true,0,Math.PI*1.5),new T.MeshStandardMaterial({color:'#bda68c',transparent:true,opacity:.32,side:T.DoubleSide,depthWrite:false}));outer.position.set(-5,-1,0);outer.rotation.y=.5;scene.add(outer);
 const pipe=new T.Mesh(new T.CylinderGeometry(.23,.23,10,16),new T.MeshStandardMaterial({color:'#799198',transparent:true,opacity:.4,depthWrite:false}));pipe.position.set(-5,-1,0);scene.add(pipe);const bit=new T.Mesh(new T.ConeGeometry(.46,.7,6),new T.MeshStandardMaterial({color:'#475d67'}));bit.position.set(-5,-6.3,0);bit.rotation.z=Math.PI;scene.add(bit);
 const beads=[];for(let i=0;i<24;i++){const isPipe=i<10,mat=new T.MeshBasicMaterial({color:isPipe?'#258ac0':'#e0a32a'}),mesh=new T.Mesh(new T.SphereGeometry(isPipe?.13:.10,10,7),mat);scene.add(mesh);beads.push({mesh,isPipe,offset:(i%10)/10,angle:i*2.4});}
 let moving = true, visible = true, last = 0;
 let depthScale = 1, annulusRadius = .65, downSpeed = 0, transportSpeed = 0, showCuttings = true;
 let downPhase = 0, cuttingsPhase = 0;

 function update(result, input) {
  depthScale = .48 + .52 * input.depth / 4000;
  const holeRadius = input.hole / .25;
  const pipeRadius = input.pipe / .25;
  annulusRadius = (holeRadius + pipeRadius) / 2;
  outer.scale.set(holeRadius,depthScale,holeRadius);
  pipe.scale.set(pipeRadius/.23,depthScale,pipeRadius/.23);
  pipe.position.y = outer.position.y = 4 - 5*depthScale;
  bit.position.y = 3.7 - 10*depthScale;
  downSpeed = .32 * input.flow/3000;
  transportSpeed = Math.sign(result.transport) * .32 * Math.tanh(Math.abs(result.transport));
  showCuttings = result.validStokes;
 }
 const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(9,12*h/w);camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);const observer=new IntersectionObserver(([e])=>visible=e.isIntersecting);observer.observe(host);
 renderer.setAnimationLoop(time => {
  const dt = last ? Math.min((time-last)/1000,.05) : 0;
  last = time;
  if (!visible) return;
  if (moving) {
   downPhase = (downPhase + dt*downSpeed) % 1;
   cuttingsPhase = (cuttingsPhase + dt*transportSpeed + 1) % 1;
  }
  for (const {mesh,isPipe,offset,angle} of beads) {
   mesh.visible = isPipe ? downSpeed > 0 : showCuttings;
   const fraction = ((isPipe ? downPhase : cuttingsPhase) + offset) % 1;
   const depth = isPipe ? fraction : 1-fraction;
   mesh.position.set(
    -5 + (isPipe ? .12 : annulusRadius * Math.cos(angle)),
    4 - depth*10*depthScale,
    isPipe ? .2 : annulusRadius * Math.sin(angle)
   );
  }
  renderer.render(scene,camera);
 });
 return {update,setRunning(v){moving=v;},destroy(){renderer.setAnimationLoop(null);resize.disconnect();observer.disconnect();orbit.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of [].concat(o.material))m.dispose();});renderer.dispose();renderer.domElement.remove();}};
}

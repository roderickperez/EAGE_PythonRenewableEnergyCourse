import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { visualState, INTAKE_CENTER_Y, INTAKE_RADIUS } from './visual-state.js';
import { updateMotionControl } from '../motion-control.js';

/** A single sectional 3D model; units are schematic, animation is illustrative. */
export class HydroScene {
  constructor(host) {
    this.host=host; this.parts=[]; this.streams=[]; this.electrical=[]; this.visible=true;
    this.animate=true;
    this.scene=new T.Scene(); this.scene.background=new T.Color('#edf1f4');
    this.renderer=new T.WebGLRenderer({antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.shadowMap.enabled=true; this.renderer.shadowMap.type=T.PCFSoftShadowMap;
    this.renderer.toneMapping=T.ACESFilmicToneMapping; this.renderer.toneMappingExposure=1.2;
    host.append(this.renderer.domElement);
    this.camera=new T.PerspectiveCamera(35,1,.1,150);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    Object.assign(this.controls,{enablePan:false,minPolarAngle:.65,maxPolarAngle:1.4,minAzimuthAngle:-.65,maxAzimuthAngle:.65,minDistance:17,maxDistance:65});
    this.scene.add(new T.HemisphereLight(0xdbeeff,0x74695a,2));
    const sun=new T.DirectionalLight(0xfff2dd,4); sun.position.set(-6,15,12); sun.castShadow=true;
    sun.shadow.mapSize.set(2048,2048); Object.assign(sun.shadow.camera,{left:-15,right:15,top:12,bottom:-12}); sun.shadow.normalBias=.035; this.scene.add(sun);
    const rim=new T.DirectionalLight(0xb5dcff,2); rim.position.set(8,8,-7); this.scene.add(rim);
    this.concrete=this.mat('#c9cbc8',.88); this.cut=this.mat('#e4e3dc',.94);
    this.steel=this.mat('#667886',.32,.65); this.dark=this.mat('#263c49',.48,.5);
    this.copper=this.mat('#b97140',.35,.65); this.teal=this.mat('#267b80',.3,.45); this.water=this.mat('#168fb2',.22,.18);
    this.build();
    this.tooltip=document.createElement('div'); this.tooltip.className='plant-tooltip'; this.tooltip.hidden=true; this.tooltip.setAttribute('role','status'); host.append(this.tooltip);
    this.raycaster=new T.Raycaster();
    this.pointer=e=>{const r=host.getBoundingClientRect(); this.raycaster.setFromCamera(new T.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.camera); const hit=this.raycaster.intersectObjects(this.parts,false)[0]; this.tooltip.hidden=!hit; if(hit){this.tooltip.textContent=hit.object.userData.label; this.tooltip.style.left=`${Math.max(8,Math.min(r.width-180,e.clientX-r.left+12))}px`; this.tooltip.style.top=`${Math.max(8,e.clientY-r.top-40)}px`;}};
    this.leave=()=>{this.tooltip.hidden=true;};
    host.addEventListener('pointermove',this.pointer); host.addEventListener('pointerdown',this.pointer); host.addEventListener('pointerleave',this.leave);
    this.observer=new ResizeObserver(()=>this.resize()); this.observer.observe(host);
    this.intersection=new IntersectionObserver(([e])=>{this.visible=e.isIntersecting;}); this.intersection.observe(host);
    this.toggle=document.createElement('button'); this.toggle.className='motion-toggle';
    const describe=()=>updateMotionControl(this.toggle,this.animate);
    describe(); this.toggle.onclick=()=>{this.animate=!this.animate;describe();}; host.append(this.toggle);
    this.resize(); this.lastTime=0; this.renderer.setAnimationLoop(time=>this.frame(time));
  }
  mat(color,roughness=.6,metalness=0){return new T.MeshStandardMaterial({color,roughness,metalness,side:T.DoubleSide});}
  mesh(geometry,material,position,label,parent=this.scene){const m=new T.Mesh(geometry,material); m.position.set(...position); m.castShadow=true; m.receiveShadow=true; parent.add(m); if(label){m.userData.label=label; this.parts.push(m);} return m;}
  box(size,p,mat,label,parent){return this.mesh(new T.BoxGeometry(...size),mat,p,label,parent);}
  cyl(r,h,p,mat,label,parent){return this.mesh(new T.CylinderGeometry(r,r,h,48),mat,p,label,parent);}
  bar(a,b,r,mat,label){const s=new T.Vector3(...a),e=new T.Vector3(...b),d=e.clone().sub(s); const m=this.mesh(new T.CylinderGeometry(r,r,d.length(),12),mat,s.add(e).multiplyScalar(.5).toArray(),label); m.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),d.normalize()); return m;}
  section(points,depth,z,mat,label){const s=new T.Shape(); points.forEach(([x,y],i)=>i?s.lineTo(x,y):s.moveTo(x,y)); s.closePath(); return this.mesh(new T.ExtrudeGeometry(s,{depth,bevelEnabled:false}),mat,[0,0,z],label);}
  pipe(points,r,label,type='turbine'){
    const curve=new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)),false,'centripetal');
    const shell=this.mat('#74969f',.32,.45); shell.transparent=true; shell.opacity=.2; shell.depthWrite=false;
    this.mesh(new T.TubeGeometry(curve,80,r,24,false),shell,[0,0,0],label);
    this.mesh(new T.TubeGeometry(curve,80,r*.52,16,false),this.water,[0,0,0],label);
    const particles=[]; const foam=this.mat('#c9faff',.15);
    for(let i=0;i<22;i++){const p=this.mesh(new T.SphereGeometry(r*.24,10,8),foam,[0,0,0]); p.castShadow=false; particles.push(p);}
    this.streams.push({curve,particles,type,phase:0});
  }
  build(){
    this.landscape();
    this.box([19,.55,7.4],[0,-.95,-.5],this.mat('#9caaa8'),'Foundation / bedrock');
    this.reservoir=this.box([5.3,1,5.7],[-6.7,2,-1],this.water,'Reservoir — controlled water elevation');
    const profile=[[-4,-.6],[.8,-.6],[-2.8,6.5],[-4,6.5]];
    this.section(profile,4.7,-3.8,this.concrete,'Gravity dam'); this.section(profile,.12,.9,this.cut,'Dam section');
    this.box([1.35,.18,4.95],[-3.4,6.55,-1.35],this.cut,'Dam crest');
    for(const z of [-3.65,.8]){this.bar([-3.9,7,z],[-2.85,7,z],.025,this.steel,'Crest railing'); for(const x of [-3.9,-3.35,-2.85])this.bar([x,6.65,z],[x,7,z],.022,this.steel);}
    this.pipe([[-4.45,INTAKE_CENTER_Y,1.5],[-3.6,1.2,1.5],[-2.5,1.1,1.5],[-1.1,.9,1.5],[.1,.8,1.5],[2.3,.8,1.5]],INTAKE_RADIUS,'Submerged intake and pressurized penstock');
    for(let i=0;i<7;i++)this.bar([-4.45,.7,1+i*.16],[-4.45,1.7,1+i*.16],.03,this.steel,'Submerged intake trash rack');
    this.box([.12,.85,.9],[-4.05,1.2,1.5],this.steel,'Intake gate — open around exposed sectional water path');
    this.bar([-3.75,1.7,1.5],[-3.75,6.8,1.5],.06,this.steel,'Intake control gate stem');
    this.box([.4,.2,.5],[-3.75,6.8,1.5],this.dark,'Gate actuator');
    this.box([5.2,.35,3.5],[3.5,-.3,-.1],this.concrete,'Powerhouse foundation');
    this.box([5.2,3.8,.24],[3.5,1.8,-1.75],this.concrete,'Powerhouse');
    this.box([.24,3.8,3.5],[6,1.8,-.1],this.concrete,'Powerhouse');
    this.box([5.4,.22,1.4],[3.5,3.8,-1.15],this.cut,'Powerhouse roof cutaway');
    this.box([1.2,.22,3.3],[5.4,2.1,-.05],this.cut,'Generator floor');
    for(let i=0;i<4;i++)this.box([.8,.65,.035],[1.6+i*1.05,2.8,-1.6],this.mat('#55828d',.3),'Powerhouse windows');
    this.cyl(1.05,.25,[2.5,.7,1.5],this.teal,'Turbine casing');
    this.runner=new T.Group(); this.runner.position.set(2.5,.95,1.5); this.scene.add(this.runner);
    this.cyl(.23,.32,[0,0,0],this.steel,'Turbine hub',this.runner);
    for(let i=0;i<11;i++){const a=i*Math.PI*2/11;const b=this.box([.68,.12,.23],[Math.cos(a)*.55,0,Math.sin(a)*.55],i===0?this.copper:this.steel,'Turbine runner',this.runner);b.rotation.set(.25,-a+.45,.12);}
    this.cyl(.105,1.9,[2.5,1.9,1.5],this.steel,'Common turbine–generator shaft');
    this.mesh(new T.CylinderGeometry(.9,.9,.65,64,1,true,Math.PI*.5,Math.PI),this.copper,[2.5,2.8,1.5],'Generator stator — cutaway');
    for(let i=0;i<18;i++){const a=Math.PI*.5+i*Math.PI/17; this.cyl(.045,.6,[2.5+.93*Math.sin(a),2.8,1.5+.93*Math.cos(a)],this.copper,'Stator windings');}
    this.rotor=new T.Group(); this.rotor.position.set(2.5,2.65,1.5); this.scene.add(this.rotor);
    this.cyl(.64,.22,[0,0,0],this.dark,'Generator rotor',this.rotor);
    for(let i=0;i<8;i++){const a=i*Math.PI/4; const p=this.box([.22,.28,.12],[Math.sin(a)*.53,.16,Math.cos(a)*.53],this.copper,'Generator rotor poles',this.rotor);p.rotation.y=a;}
    this.pipe([[2.5,.7,1.5],[2.5,-.15,1.5],[3,-.45,1.5],[4.2,-.45,1.5],[6.6,-.45,1.5]],.43,'Draft tube — discharge');
    this.box([3.6,.24,6],[7.6,-.6,-.6],this.water,'Tailrace / river');
    this.section([[-3.4,6.35],[-2.7,6.35],[1.7,.1],[6.4,-.2],[6.4,-.4],[1.4,-.2]],1.05,-3.5,this.cut,'Spillway chute');
    // Excess flow uses a submerged gated outlet, never a dry crest spillway.
    this.pipe([[-4.3,1.2,-3],[-2.5,.9,-3],[.2,.2,-3],[1.8,-.15,-3],[6.6,-.3,-3]],.18,'Gated low-level bypass — excess usable flow','spill');
    this.box([1,.9,1],[4.6,4.3,-1.1],this.steel,'Step-up transformer');
    for(let i=0;i<8;i++)this.box([1.12,.055,1.08],[4.6,3.95+i*.1,-1.1],this.steel,'Transformer cooling fins');
    for(const x of [4.3,4.9])this.cyl(.065,.45,[x,5,-1.1],this.dark,'Transformer insulator');
    const x=7,z=-3.2;
    for(const s of [-1,1])this.bar([x+s*.65,0,z],[x+s*.2,5.7,z],.045,this.steel,'Transmission tower');
    for(let y=0;y<5;y+=.8){this.bar([x-.6+y*.075,y,z],[x+.6-(y+.8)*.075,y+.8,z],.026,this.steel,'Transmission tower');this.bar([x+.6-y*.075,y,z],[x-.6+(y+.8)*.075,y+.8,z],.026,this.steel);}
    this.bar([x-1,5.3,z],[x+1,5.3,z],.05,this.steel,'Transmission crossarm');
    this.energyPath([[2.5,2.85,1.5],[3.6,2.85,1.5],[4.1,3.2,.7],[4.3,4.8,-1.1]],'Generator output bus → transformer');
    for(const o of [-.7,.7]) {
      this.energyPath([[4.6+o*.4,5.1,-1.1],[5.6+o*.6,4.85,-2.1],[x+o,5.3,z],[8.2+o,4.9,-4.2],[9+o,5.15,-5]],'Transformer → transmission grid');
    }
    this.lamp=this.mat('#3fb6a0',.3);this.cyl(.12,.1,[4.6,4.82,-1.1],this.lamp,'Electrical output indicator');
  }
  landscape(){
    const grass=this.mat('#708b75',.95), rock=this.mat('#85918a',1), bark=this.mat('#66584a',1), needles=this.mat('#3e6654',1);
    this.box([21,.35,4],[0,-.8,-5.7],grass,'River valley');
    const terrain=new T.PlaneGeometry(22,7,44,18); terrain.rotateX(-Math.PI/2);
    const terrainHeight=(x,z)=>{
      const ridge=5.8*Math.exp(-(((x+3)/3.3)**2))+3.9*Math.exp(-(((x-3)/2.8)**2))+3.4*Math.exp(-(((x+8)/2.8)**2));
      return -.65+ridge*Math.max(0,1-(z/3.5)**2)*(1+.07*Math.sin(x*2+z)*Math.cos(z*3));
    };
    const vertices=terrain.attributes.position;
    for(let i=0;i<vertices.count;i++){
      const x=vertices.getX(i),z=vertices.getZ(i);
      vertices.setY(i,terrainHeight(x,z));
    }
    terrain.computeVertexNormals(); this.mesh(terrain,rock,[0,0,-7],'Mountain catchment');
    for(const [x,z,s] of [[-9,-6.7,.9],[-7.8,-7,1.1],[-6.5,-6.4,1],[-5,-6.5,.9],[-3.6,-6,.9],[-2.1,-5.7,.8],[-.5,-5.3,.85],[1.2,-5.1,.8]]) {
      const ground=terrainHeight(x,z+7);
      this.cyl(.07*s,.7*s,[x,ground+.35*s,z],bark,'Tree');
      for(let j=0;j<3;j++)this.mesh(new T.ConeGeometry((.48-j*.1)*s,.8*s,9),needles,[x,ground+(.85+j*.32)*s,z],'Tree');
    }
  }
  energyPath(points,label){
    const curve=new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)),false,'centripetal');
    this.mesh(new T.TubeGeometry(curve,48,.035,8,false),this.dark,[0,0,0],label);
    const material=new T.MeshStandardMaterial({color:'#ffc65a',emissive:'#ffad24',emissiveIntensity:1.8,roughness:.3});
    const particles=[];
    for(let i=0;i<8;i++){const p=this.mesh(new T.SphereGeometry(.07,10,8),material,[0,0,0]);p.castShadow=false;particles.push(p);}
    this.electrical.push({curve,particles,material,phase:0});
  }
  update(result){this.result=result;this.state=visualState(result);const h=this.state.surfaceY+.25;this.reservoir.scale.y=h;this.reservoir.position.y=h/2-.25;this.lamp.emissive.set('#29d6b0');this.lamp.emissiveIntensity=this.state.load;this.host.dataset.turbineFlow=String(result.turbineFlowM3s);this.host.dataset.waterSurface=String(this.state.surfaceY);this.host.dataset.energized=String(this.state.energized);}
  resize(){const{width,height}=this.host.getBoundingClientRect();if(!width||!height)return;this.renderer.setSize(width,height,false);this.camera.aspect=width/height;const d=Math.max(25,20/this.camera.aspect);this.camera.position.set(d*.2,d*.39,d*.88);this.controls.target.set(-.4,2.5,0);this.camera.updateProjectionMatrix();this.controls.update();}
  frame(time){const dt=Math.min((time-this.lastTime)/1000,.05);this.lastTime=time;if(!this.visible)return;const flow=this.result?.turbineFlowM3s??0;
    const r=this.animate?dt*Math.min(7,flow*1.3):0;this.runner.rotation.y+=r;this.rotor.rotation.y+=r;
    for(const s of this.streams){const q=s.type==='spill'?(this.result?.spillFlowM3s??0):flow;if(this.animate)s.phase=(s.phase+dt*Math.min(.5,q*.025))%1;s.particles.forEach((p,i)=>{p.visible=q>0;p.position.copy(s.curve.getPointAt((i/s.particles.length+s.phase)%1));p.position.z+=s.type==='spill'?.14:.32;});}
    for(const wire of this.electrical){if(this.animate&&this.state?.energized)wire.phase=(wire.phase+dt*(.08+.32*this.state.load))%1;wire.material.emissiveIntensity=.6+2*(this.state?.load??0);wire.particles.forEach((p,i)=>{p.visible=!!this.state?.energized;p.position.copy(wire.curve.getPointAt((i/8+wire.phase)%1));});}
    this.host.dataset.runnerAngle=String(this.runner.rotation.y);this.renderer.render(this.scene,this.camera);
  }
  destroy(){this.renderer.setAnimationLoop(null);this.controls.dispose();this.observer.disconnect();this.intersection.disconnect();this.host.removeEventListener('pointermove',this.pointer);this.host.removeEventListener('pointerdown',this.pointer);this.host.removeEventListener('pointerleave',this.leave);const gs=new Set(),ms=new Set();this.scene.traverse(o=>{if(o.geometry)gs.add(o.geometry);if(o.material)ms.add(o.material);});gs.forEach(g=>g.dispose());ms.forEach(m=>m.dispose());this.renderer.dispose();this.host.replaceChildren();}
}

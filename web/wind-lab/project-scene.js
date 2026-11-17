import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FARM, GRID, gridPoint, windFieldSampler } from './project-model.js';
import { wakeRaster, localParticle, rotorRate } from './project-field.js';

export function mountFarmScene(host, onPick) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e6eff3');
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = T.SRGBColorSpace;
  host.append(renderer.domElement);
  const camera = new T.OrthographicCamera(-15, 15, 12, -12, .1, 120);
  camera.position.set(22, 25, 30);
  const orbit = new OrbitControls(camera, renderer.domElement);
  Object.assign(orbit, { enablePan: false, enableZoom: true, minZoom: .8, maxZoom: 2.8, minPolarAngle: .3, maxPolarAngle: 1.35 });
  orbit.target.set(0, 1, 0); orbit.update();
  scene.add(new T.HemisphereLight(0xe6f5ff, 0x7f8568, 2.4));
  const sun = new T.DirectionalLight(0xffedcd, 2.6); sun.position.set(-7, 20, 9); scene.add(sun);
  const terrain = new T.Mesh(new T.BoxGeometry(20, .3, 20), new T.MeshStandardMaterial({color:'#657e62', roughness:.8}));
  terrain.position.y = -.17; scene.add(terrain);
  const grid = new T.GridHelper(20, GRID, '#e4f0e6', '#c2d6c4'); grid.position.y = .07; scene.add(grid);
  const ground = new T.Mesh(new T.PlaneGeometry(20, 20), new T.MeshBasicMaterial({visible:false, side:T.DoubleSide}));
  ground.rotation.x = -Math.PI / 2; ground.position.y = .03; scene.add(ground);
  const elements = new T.Group(); scene.add(elements);
  const white = new T.MeshStandardMaterial({color:'#e3e9eb', roughness:.59});
  const dark = new T.MeshStandardMaterial({color:'#788a93', roughness:.59});
  const mark = new T.MeshBasicMaterial({color:'#e5ae3e'});
  const pointColor = new T.MeshBasicMaterial({color:'#dbece6', transparent:true, opacity:.72});
  const resolution=160;
  const texture=new T.DataTexture(new Uint8Array(resolution*resolution*4),resolution,resolution,T.RGBAFormat);
  texture.colorSpace=T.SRGBColorSpace; texture.magFilter=T.NearestFilter; texture.minFilter=T.NearestFilter; texture.needsUpdate=true;
  const wakeMat = new T.MeshBasicMaterial({map:texture, transparent:true, depthWrite:false, side:T.DoubleSide});
  const wakePlane = new T.Mesh(new T.PlaneGeometry(20,20),wakeMat);
  wakePlane.rotation.x=-Math.PI/2;wakePlane.position.y=.035;scene.add(wakePlane);
  const particleGeometry=new T.BoxGeometry(.045,.045,1), particleMaterial=new T.MeshBasicMaterial({color:'#e6fbff',transparent:true,opacity:1,depthWrite:false});
  const particleCount=360, trailSegments=9, instanceCount=particleCount*trailSegments;
  const particles=new T.InstancedMesh(particleGeometry,particleMaterial,instanceCount);
  const opacityAttribute=new T.InstancedBufferAttribute(new Float32Array(instanceCount),1);
  particleGeometry.setAttribute('instanceOpacity',opacityAttribute);
  particleMaterial.onBeforeCompile=shader=>{
    shader.vertexShader='attribute float instanceOpacity; varying float windOpacity;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nwindOpacity = instanceOpacity;');
    shader.fragmentShader='varying float windOpacity;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.a *= windOpacity;');
  };
  particles.frustumCulled=false; scene.add(particles);
  const seeds=Array.from({length:particleCount},(_,i)=>({x:((i*7.173)%20)-10,z:((i*11.739+3)%20)-10,h:.7+(i%5)*.18, trail:[], trailTime:0}));
  const dummy=new T.Object3D();let sample=()=>0, windSpeed=0, windDirection=0, fieldBounds={min:0,max:0}, elapsed=0;
  function moveParticles(dt) {
    particles.visible=windSpeed>0;
    for(let i=0;i<seeds.length;i++) {
      const p=seeds[i], local=localParticle(p,windDirection,sample,dt,{freeStream:windSpeed,elapsed,phase:i*1.73});
      // Never draw a streak across the farm when a particle wraps around an edge.
      if(Math.abs(local.x-p.x)>10 || Math.abs(local.z-p.z)>10) p.trail=[];
      p.x=local.x;p.z=local.z;p.trailTime+=dt;
      if(!p.trail.length || p.trailTime>=.12) {
        p.trail.unshift({x:p.x,z:p.z});p.trail.length=Math.min(p.trail.length,trailSegments+1);p.trailTime=0;
      }
      for(let j=0;j<trailSegments;j++) {
        const instance=i*trailSegments+j;
        const a=j===0 ? p : p.trail[j-1], b=p.trail[j];
        if(!a || !b){dummy.scale.set(0,0,0);opacityAttribute.setX(instance,0);}
        else {
          const dx=a.x-b.x,dz=a.z-b.z, length=Math.hypot(dx,dz);
          dummy.position.set((a.x+b.x)/2,p.h,(a.z+b.z)/2);
          dummy.rotation.set(0,Math.atan2(dx,dz),0);
          dummy.scale.set(1,1,length);
          opacityAttribute.setX(instance,local.opacity*(1-j/trailSegments));
        }
        dummy.updateMatrix();particles.setMatrixAt(instance,dummy.matrix);
      }
    }
    particles.instanceMatrix.needsUpdate=true;opacityAttribute.needsUpdate=true;
  }
  const geometries = new Set();
  function mesh(geometry, material, parent=elements) { geometries.add(geometry); const m=new T.Mesh(geometry, material); parent.add(m); return m; }
  const scale = value => value / FARM * 20;
  const world = p => ({ x:scale(p.x)-10, z:10-scale(p.y) });
  let rotors = [], running=true, visible=true, last=0, fieldKey='';
  function update(ids, selected, recommended, direction, speed) {
    elements.clear(); geometries.forEach(g=>g.dispose()); geometries.clear(); rotors=[];
    for (let id=0; id<GRID*GRID; id++) {
      const p=world(gridPoint(id));
      const dot=mesh(new T.CircleGeometry(id===selected || id===recommended ? .16 : .065, 16), id===selected || id===recommended ? mark : pointColor);
      dot.rotation.x=-Math.PI/2; dot.position.set(p.x,.09,p.z);
    }
    const angle=direction*Math.PI/180;
    sample=windFieldSampler(ids,direction,speed);windDirection=direction;windSpeed=speed;moveParticles(0);
    const key=[...ids].sort((a,b)=>a-b).join(',')+'|'+direction+'|'+speed;
    if(key!==fieldKey){const field=wakeRaster(ids,direction,speed,resolution);texture.image.data.set(field.pixels);texture.needsUpdate=true;fieldBounds={min:field.min,max:field.max};fieldKey=key;}
    for (const id of ids) {
      const p=gridPoint(id), w=world(p);
      const tower=mesh(new T.CylinderGeometry(.08,.15,2.0,12),white); tower.position.set(w.x,1,w.z);
      const nacelle=mesh(new T.BoxGeometry(.3,.22,.46),dark); nacelle.position.set(w.x,2.05,w.z); nacelle.rotation.y=angle;
      const rotor=new T.Group(); rotor.position.set(w.x,2.05,w.z); rotor.rotation.y=angle; elements.add(rotor);
      const blades=new T.Group(); rotor.add(blades); rotors.push({blades, rate:rotorRate(sample(p.x,p.y))});
      const hub=mesh(new T.SphereGeometry(.11,12,8),white,blades); hub.position.z=.24;
      for(let k=0;k<3;k++) { const blade=mesh(new T.BoxGeometry(.07,.84,.045),white,blades); blade.position.set(-Math.sin(k*2*Math.PI/3)*.44, Math.cos(k*2*Math.PI/3)*.44,.24); blade.rotation.z=k*2*Math.PI/3; }
    }
    return fieldBounds;
  }
  const ray=new T.Raycaster(), pointer=new T.Vector2(); let downPosition=null;
  const start=e=>{ downPosition={x:e.clientX,y:e.clientY}; };
  const end=e=>{
    if (!downPosition || Math.hypot(e.clientX-downPosition.x,e.clientY-downPosition.y)>5) return;
    const rect=renderer.domElement.getBoundingClientRect();
    pointer.set((e.clientX-rect.left)/rect.width*2-1,1-(e.clientY-rect.top)/rect.height*2);
    ray.setFromCamera(pointer,camera); const hit=ray.intersectObject(ground)[0];
    if(!hit) return;
    const col=Math.min(GRID-1,Math.max(0,Math.floor((hit.point.x+10)/20*GRID)));
    const row=Math.min(GRID-1,Math.max(0,Math.floor((10-hit.point.z)/20*GRID)));
    onPick(row*GRID+col);
  };
  renderer.domElement.addEventListener('pointerdown',start); renderer.domElement.addEventListener('pointerup',end);
  const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return; const half=Math.max(12,16*h/w); camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();renderer.setSize(w,h,false);});
  resize.observe(host);
  const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;}); observer.observe(host);
  renderer.setAnimationLoop(time=>{const dt=Math.min(.05,(time-last)/1000);last=time;if(!visible)return;if(running){elapsed+=dt;rotors.forEach(r=>{r.blades.rotation.z-=dt*r.rate;});moveParticles(dt);}renderer.render(scene,camera);});
  return {update,setMotion(value){running=value;},destroy(){renderer.setAnimationLoop(null);resize.disconnect();observer.disconnect();orbit.dispose();renderer.domElement.removeEventListener('pointerdown',start);renderer.domElement.removeEventListener('pointerup',end);geometries.forEach(g=>g.dispose());[white,dark,mark,pointColor,wakeMat,terrain.material,particleMaterial].forEach(m=>m.dispose());texture.dispose();wakePlane.geometry.dispose();particleGeometry.dispose();terrain.geometry.dispose();grid.geometry.dispose();grid.material.dispose();ground.geometry.dispose();ground.material.dispose();renderer.dispose();renderer.domElement.remove();}};
}

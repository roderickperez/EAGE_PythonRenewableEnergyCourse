import * as T from 'three';
import {SVGLoader} from 'three/addons/loaders/SVGLoader.js';
import pythonLogo from './python.svg';
import {installPlant} from './plants.js';
import {landscape} from './landscape.js';

const up=new T.Vector3(0,1,0);
const clamp=T.MathUtils.clamp;
const vector=(x=0,y=0,z=0)=>new T.Vector3(x,y,z);

function explorer(scene) {
  const person=new T.Group();
  scene.add(person);
  function piece(geometry,color,x,y,z,parent=person) {
    const m=new T.Mesh(geometry,new T.MeshStandardMaterial({color,roughness:.83}));
    m.position.set(x,y,z);parent.add(m);return m;
  }
  piece(new T.CapsuleGeometry(.3,.35,5,12),0xbd683b,0,1.15,0);
  piece(new T.SphereGeometry(.24,16,10),0xc4916c,0,1.78,0);
  piece(new T.CylinderGeometry(.255,.28,.16,16),0xeab954,0,1.98,0);
  piece(new T.CylinderGeometry(.33,.33,.05,16),0xd9a449,0,1.91,0);
  piece(new T.BoxGeometry(.44,.48,.24),0x25484b,0,1.26,.28);
  const limbs=[-1,1].map(side=>{
    const hip=new T.Group(),arm=new T.Group();
    hip.position.set(side*.15,.84,0);arm.position.set(side*.33,1.39,0);
    person.add(hip,arm);
    piece(new T.CapsuleGeometry(.11,.41,4,8),0x354e56,0,-.32,0,hip);
    piece(new T.BoxGeometry(.22,.16,.34),0x303735,0,-.69,-.045,hip);
    piece(new T.CapsuleGeometry(.093,.31,4,8),0xdb9355,0,-.21,0,arm);
    return {hip,arm,side};
  });
  // Soft contact shadow keeps the surveyor grounded on terrain and walkways.
  const shadow=new T.Mesh(new T.CircleGeometry(.59,24),new T.MeshBasicMaterial({color:0x35483d,transparent:true,opacity:.24,depthWrite:false}));
  shadow.rotation.x=-Math.PI/2;scene.add(shadow);
  return {person,pose(t,moving){
    for(const {hip,arm,side} of limbs){hip.rotation.x=side*Math.sin(t)*moving*.63;arm.rotation.x=-hip.rotation.x*.72;}
    person.position.y=moving*Math.abs(Math.sin(t))*.055+.12;
    shadow.position.set(person.position.x,.115,person.position.z);
  }};
}

function trailSign(scene, text, x, z, angle) {
  const board=new T.Group();
  board.position.set(x,0,z);board.rotation.y=angle;
  const timber=new T.MeshStandardMaterial({color:0x65533e});
  const post=new T.Mesh(new T.BoxGeometry(.17,2.3,.17),timber);
  post.position.y=1.12;board.add(post);
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=112;
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#214e50';ctx.fillRect(0,0,512,112);
  ctx.fillStyle='#f4cf87';ctx.fillRect(12,12,6,88);
  ctx.font='bold 56px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillStyle='#f6f3e4';ctx.fillText(text,264,55,462);
  const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;
  const sign=new T.Mesh(new T.BoxGeometry(3.65,.8,.13),[timber,timber,timber,timber,new T.MeshStandardMaterial({map}),timber]);
  sign.position.y=2.18;board.add(sign);scene.add(board);
}

export function startWorld(host, stations, onNear, onInspect, onPart) {
  const renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
  renderer.toneMapping=T.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.17;
  const canvas=renderer.domElement;
  canvas.setAttribute('aria-label','3D islands. Walk with WASD or the arrow keys. Drag to look around.');
  canvas.tabIndex=0;host.append(canvas);
  const scene=new T.Scene();
  scene.fog=new T.FogExp2(0xc2d6d7,.0082);
  scene.add(new T.HemisphereLight(0xf0f8ff,0xb5a890,2.3));
  const key=new T.DirectionalLight(0xffedcf,3.1);key.position.set(-36,56,31);scene.add(key);
  const camera=new T.PerspectiveCamera(52,1,.14,380);
  const islands=stations.map(s=>({...s,pos:vector(s.x*4.3,0,s.z*4.3),radius:s.id==='python'?7.5:11}));
  const center={id:'hub',name:'Energy crossing',pos:vector(),radius:6.7};
  const ocean=landscape(scene,[center,...islands]);
  for(const site of islands){
    const group=new T.Group();
    group.position.copy(site.pos);
    scene.add(group);
    site.group=group;
    const direction=site.pos.clone().normalize();
    const edge=site.pos.clone().addScaledVector(direction,-site.radius*.85);
    trailSign(scene,site.name,edge.x-direction.z*2,edge.z+direction.x*2,Math.atan2(-direction.x,-direction.z));
    if(site.id==='python'){
      const logo=new T.Group();
      for(const path of new SVGLoader().parse(pythonLogo).paths){
        const geometry=new T.ExtrudeGeometry(SVGLoader.createShapes(path),{depth:1.15,bevelEnabled:true,bevelSize:.15,bevelThickness:.12,bevelSegments:2});
        geometry.translate(-12,-12,0);
        const part=new T.Mesh(geometry,new T.MeshStandardMaterial({color:path.color,roughness:.55}));
        part.scale.set(.15,-.15,.15);
        logo.add(part);
      }
      logo.position.set(-2.6,2.3,-.5);group.add(logo);
      const base=new T.Mesh(new T.CylinderGeometry(1.7,2,.46,24),new T.MeshStandardMaterial({color:0xc4c6b0}));base.position.set(-2.6,.23,-.5);group.add(base);
    }else{
      group.scale.setScalar(.86);
      group.position.y=site.id==='geothermal'?4.55:site.id==='hydro'?1.12:.47;
      site.plant=installPlant(site.id,group);
    }
  }
  const {person,pose}=explorer(scene);
  person.position.copy(islands[0].pos).z+=3.5;
  const keys=new Set(),stick=new T.Vector2();
  let near=null,detail=null,route=[],destination=null,yaw=.22,pitch=.22,zoom=1,elapsed=0,last=0,gait=0;
  let target=person.position.clone(),startView=true,enabled=true;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  function resize(){const {width,height}=host.getBoundingClientRect();camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);}
  new ResizeObserver(resize).observe(host);resize();
  const pressed=new Map();let pinch=0;
  canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pressed.set(e.pointerId,{x:e.clientX,y:e.clientY,drag:0});};
  canvas.onpointermove=e=>{
    const p=pressed.get(e.pointerId);if(!p)return;
    const dx=e.clientX-p.x,dy=e.clientY-p.y;p.drag+=Math.abs(dx)+Math.abs(dy);p.x=e.clientX;p.y=e.clientY;
    if(pressed.size===2){const [a,b]=[...pressed.values()];const d=Math.hypot(a.x-b.x,a.y-b.y);if(pinch>0&&d>0)zoom=clamp(zoom*pinch/d,.55,1.8);pinch=d;return;}
    yaw-=dx*.006;pitch=clamp(pitch+dy*.004,.025,1.18);
  };
  canvas.onpointerup=e=>{
    const p=pressed.get(e.pointerId);
    if(p&&p.drag<6&&detail&&detail.plant){
      const rect=canvas.getBoundingClientRect(),ray=new T.Raycaster();
      ray.setFromCamera(new T.Vector2((e.clientX-rect.left)/rect.width*2-1,1-(e.clientY-rect.top)/rect.height*2),camera);
      const hit=ray.intersectObject(detail.group,true).find(h=>h.object.userData.label);
      if(hit)onPart(hit.object.userData.label);
    }
    pressed.delete(e.pointerId);pinch=0;
  };
  canvas.onpointercancel=e=>{pressed.delete(e.pointerId);pinch=0;};
  canvas.onwheel=e=>{e.preventDefault();zoom=clamp(zoom*Math.exp(e.deltaY*.001),.55,1.8);};
  const arrows=['arrowup','arrowdown','arrowright','arrowleft','w','a','s','d'];
  addEventListener('keydown',e=>{if(enabled&&arrows.includes(e.key.toLowerCase())&&!detail){keys.add(e.key.toLowerCase());e.preventDefault();}});
  addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
  addEventListener('blur',()=>{keys.clear();stick.set(0,0);});
  const nearestBranch=p=>islands.reduce((a,b)=>segmentDistance(p,a.pos)<segmentDistance(p,b.pos)?a:b);
  function segmentDistance(p,end){const t=clamp((p.x*end.x+p.z*end.z)/end.lengthSq(),0,1);return Math.hypot(p.x-t*end.x,p.z-t*end.z);}
  function walkable(p){
    return Math.hypot(p.x,p.z)<center.radius*.84||islands.some(s=>Math.hypot(p.x-s.pos.x,p.z-s.pos.z)<s.radius*.85||segmentDistance(p,s.pos)<1.25);
  }
  function inspect(site){
    if(!site)return;
    detail=site;route=[];destination=null;
    yaw=site.id==='solar'?2.52:.25;pitch=.33;zoom=1;
    person.visible=false;onInspect(site);
  }
  function leave(){detail=null;person.visible=true;zoom=1;pitch=.24;onInspect(null);}
  function travel(id){
    if(detail)leave();
    const site=islands.find(s=>s.id===id),branch=nearestBranch(person.position);
    destination=site;
    // Dock on the near edge, leaving the installation in view.
    const dock=site.pos.clone().addScaledVector(site.pos.clone().normalize(),-site.radius*.71);
    route=[];
    if(branch!==site){
      const t=clamp(person.position.dot(branch.pos)/branch.pos.lengthSq(),0,1);
      route.push(branch.pos.clone().multiplyScalar(t),vector());
    }
    route.push(dock);
  }
  renderer.setAnimationLoop(time=>{
    const dt=last?Math.min(.07,(time-last)/1000):0;last=time;
    if(document.hidden||!enabled)return;
    elapsed+=dt;
    const p=person.position;
    let walking=false;
    if(!detail){
      const input=new T.Vector2(stick.x+Number(keys.has('d')||keys.has('arrowright'))-Number(keys.has('a')||keys.has('arrowleft')),
        stick.y+Number(keys.has('s')||keys.has('arrowdown'))-Number(keys.has('w')||keys.has('arrowup')));
      let dir=vector(),speed=7;
      if(input.lengthSq()>.001){
        route=[];destination=null;
        dir.set(input.x,0,input.y).applyAxisAngle(up,yaw).normalize();
      }else if(route.length){
        dir.copy(route[0]).sub(p);dir.y=0;
        speed=14;
        if(dir.length()<speed*dt+.13){p.x=route[0].x;p.z=route[0].z;route.shift();if(!route.length)inspect(destination);dir.set(0,0,0);}
        else dir.normalize();
      }
      if(dir.lengthSq()>0){
        const next=p.clone().addScaledVector(dir,dt*speed);
        if(walkable(next)){p.x=next.x;p.z=next.z;walking=true;gait+=dt*speed*1.4;person.rotation.y=Math.atan2(-dir.x,-dir.z);}
        if(route.length && walking){
          const heading=Math.atan2(-dir.x,-dir.z);
          yaw+=Math.atan2(Math.sin(heading-yaw),Math.cos(heading-yaw))*(1-Math.exp(-dt*2.3));
        }
      }
      pose(gait,walking?1:0);
      const found=islands.find(s=>Math.hypot(p.x-s.pos.x,p.z-s.pos.z)<s.radius*.93)||null;
      if(found!==near){near=found;onNear(near);}
    }
    const goal=detail?detail.pos.clone().add(vector(0,detail.id==='wind'?5:detail.id==='python'?1.8:3,0)):p.clone().add(vector(0,1.7,0));
    const distance=(detail?(detail.id==='python'?8:Math.max(18,23/camera.aspect)):7.8)*zoom;
    const offset=vector(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(distance);
    const blend=startView||reduced?1:1-Math.exp(-dt*6.5);
    target.lerp(goal,blend);camera.position.lerp(goal.clone().add(offset),blend);camera.lookAt(target);startView=false;
    for(const s of islands){

      if(s.plant && (!detail||detail===s))s.plant.advance(reduced?0:dt);
    }
    ocean(reduced?0:elapsed);
    renderer.render(scene,camera);
  });
  return {travel,inspect:()=>inspect(near),leave,stick,pause(value){enabled=!value;keys.clear();stick.set(0,0);}};
}

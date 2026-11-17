import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { sunDirection, RAD } from './model.js';

export function mountProjectScene(host, onPick) {
  const renderer=new T.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); host.append(renderer.domElement);
  const scene=new T.Scene(); scene.background=new T.Color('#e6eff3');
  const camera=new T.OrthographicCamera(-12,12,10,-10,.1,120); camera.position.set(20,22,27);
  const orbit=new OrbitControls(camera,renderer.domElement); orbit.enablePan=false; orbit.minZoom=.7; orbit.maxZoom=2.5; orbit.target.set(0,1,0); orbit.update();
  scene.add(new T.HemisphereLight(0xe8f7ff,0x6c795e,2.5)); const light=new T.DirectionalLight(0xffe3a6,2); light.position.set(-8,18,8); scene.add(light);
  const globe=new T.Group(), farm=new T.Group(); scene.add(globe,farm); farm.visible=false;
  const mat=(color)=>new T.MeshStandardMaterial({color,roughness:.75});
  const mesh=(g,m,p,parent)=>{const o=new T.Mesh(g,m);o.position.set(...p);parent.add(o);return o;};
  const sphere=mesh(new T.SphereGeometry(5,48,32),mat('#337a9d'),[0,0,0],globe);
  // Schematic coastlines preserve an offline globe without claiming geographic precision.
  const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=512;const c=canvas.getContext('2d');
  c.fillStyle='#397e9d';c.fillRect(0,0,1024,512);c.fillStyle='#b0c89d';
  const shapes=[[[.05,.19],[.12,.12],[.25,.14],[.30,.23],[.25,.31],[.20,.32],[.18,.41],[.12,.36],[.08,.29]],[[.23,.39],[.29,.43],[.32,.55],[.29,.71],[.26,.86],[.22,.68],[.20,.51]],[[.45,.17],[.51,.13],[.55,.22],[.53,.3],[.46,.28]],[[.46,.31],[.55,.30],[.59,.40],[.56,.56],[.52,.68],[.47,.54],[.44,.39]],[[.54,.15],[.68,.10],[.84,.15],[.94,.24],[.84,.31],[.76,.27],[.72,.38],[.65,.32],[.59,.3]],[[.78,.64],[.88,.62],[.91,.71],[.87,.79],[.79,.77],[.76,.7]],[[.31,.09],[.35,.03],[.4,.07],[.38,.18],[.33,.19]]];
  for(const poly of shapes){c.beginPath();poly.forEach(([x,y],i)=>i?c.lineTo(x*1024,y*512):c.moveTo(x*1024,y*512));c.closePath();c.fill();}
  const tex=new T.CanvasTexture(canvas);tex.colorSpace=T.SRGBColorSpace;sphere.material.map=tex;sphere.material.needsUpdate=true;
  const marker=mesh(new T.SphereGeometry(.16,16,12),new T.MeshBasicMaterial({color:'#ffca55'}),[0,0,0],globe);
  const ring=mesh(new T.RingGeometry(.22,.29,32),new T.MeshBasicMaterial({color:'#fff1a9',side:T.DoubleSide}),[0,0,0],globe);
  function globePoint(lat,lon,r=5.05){const a=lat*RAD,b=lon*RAD;return new T.Vector3(-r*Math.cos(a)*Math.cos(b),r*Math.sin(a),r*Math.cos(a)*Math.sin(b));}
  const terrain=mesh(new T.BoxGeometry(20,.35,20),mat('#718b6b'),[0,-.18,0],farm);
  const grid=new T.GridHelper(20,10,'#edf5e9','#b8d1bb');grid.position.y=.03;farm.add(grid);
  const ground=mesh(new T.PlaneGeometry(20,20),new T.MeshBasicMaterial({visible:false,side:T.DoubleSide}),[0,.04,0],farm);ground.rotation.x=-Math.PI/2;
  const racks=new T.Group();farm.add(racks);
  const selection=mesh(new T.RingGeometry(.45,.53,32),new T.MeshBasicMaterial({color:'#efbc48',side:T.DoubleSide}),[-9,.09,9],farm);selection.rotation.x=-Math.PI/2;
  const sun=mesh(new T.SphereGeometry(.42,24,16),new T.MeshBasicMaterial({color:'#ffc34c'}),[0,0,0],farm);
  const path=new T.Line(new T.BufferGeometry(),new T.LineDashedMaterial({color:'#d49130',dashSize:.18,gapSize:.13}));farm.add(path);
  const frame=mat('#9daeb4'),panel=mat('#1c4269'),cell=mat('#3975a0'),post=mat('#7d9094');
  function updateRacks(ids,tilt,azimuth){
    racks.traverse(o=>o.geometry?.dispose());racks.clear();
    for(const id of ids){const x=-9+(id%10)*2,z=9-Math.floor(id/10)*2;
      mesh(new T.CylinderGeometry(.045,.055,.6,8),post,[x,.3,z],racks);
      const rack=new T.Group();rack.position.set(x,.68,z);rack.rotation.y=Math.PI-azimuth*RAD;racks.add(rack);
      const face=new T.Group();face.rotation.x=tilt*RAD;rack.add(face);
      mesh(new T.BoxGeometry(1.52,.10,1.06),frame,[0,0,0],face);mesh(new T.BoxGeometry(1.44,.025,.98),panel,[0,.065,0],face);
      for(let a=0;a<4;a++)for(let b=0;b<3;b++)mesh(new T.BoxGeometry(.33,.015,.29),cell,[(a-1.5)*.35,.085,(b-1)*.31],face);
    }
  }
  let globeMode=true, transition=0, selected=0, clock=0, dragging=null;
  const ray=new T.Raycaster(), mouse=new T.Vector2();
  renderer.domElement.addEventListener('pointerdown',e=>{dragging={x:e.clientX,y:e.clientY};});
  renderer.domElement.addEventListener('pointerup',e=>{if(!dragging||Math.hypot(e.clientX-dragging.x,e.clientY-dragging.y)>5||globeMode)return;
    const r=renderer.domElement.getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);ray.setFromCamera(mouse,camera);
    const hit=ray.intersectObject(ground)[0];if(hit){const col=Math.min(9,Math.max(0,Math.floor((hit.point.x+10)/2))),row=Math.min(9,Math.max(0,Math.floor((10-hit.point.z)/2)));onPick(row*10+col);}
  });
  const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(12,15*h/w);Object.assign(camera,{left:-half*w/h,right:half*w/h,top:half,bottom:-half});camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);
  renderer.setAnimationLoop((time)=>{const dt=Math.min(.05,(time-clock)/1000);clock=time;if(globeMode){globe.rotation.y+=dt*.14;}else if(transition<1){transition=Math.min(1,transition+dt*1.7);globe.scale.setScalar(Math.max(.001,1-transition));farm.scale.setScalar(Math.max(.001,transition));if(transition===1)globe.visible=false;}renderer.render(scene,camera);});
  return {
    setSite(site){const p=globePoint(site.lat,site.lon);marker.position.copy(p);ring.position.copy(p.clone().multiplyScalar(1.005));ring.lookAt(p.clone().multiplyScalar(2));},
    start(){globeMode=false;transition=0;farm.visible=true;farm.scale.setScalar(.001);orbit.target.set(0,1,0);orbit.update();},
    update(ids,settings,site,date,hour,dec,selected=0){
      updateRacks(ids,settings.tilt,settings.azimuth);
      selection.position.set(-9+(selected%10)*2,.09,9-Math.floor(selected/10)*2);
      const points=[];for(let i=0;i<=288;i++){const s=sunDirection(i/12,site.lat,dec);if(s[1]>=0)points.push(new T.Vector3(s[0]*12,s[1]*12+.2,-s[2]*12));}
      path.geometry.dispose();path.geometry=new T.BufferGeometry().setFromPoints(points);path.computeLineDistances();
      const s=sunDirection(hour,site.lat,dec);sun.visible=s[1]>0;sun.position.set(s[0]*12,s[1]*12+.2,-s[2]*12);light.position.copy(sun.position);light.intensity=s[1]>0?1.5:0;
    },
    destroy(){renderer.setAnimationLoop(null);resize.disconnect();orbit.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});tex.dispose();renderer.dispose();renderer.domElement.remove();}
  };
}

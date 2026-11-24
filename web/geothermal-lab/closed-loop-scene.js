import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createReservoirFlow} from './reservoir-flow.js';

// Each comparison view shares the same camera and cutaway style.
export function mountGeothermalSystemScene(host, kind) {
 const scene=new T.Scene(); scene.background=new T.Color('#e6eff3');
 const renderer=new T.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 renderer.shadowMap.enabled=true; renderer.toneMapping=T.ACESFilmicToneMapping; host.append(renderer.domElement);
 const camera=new T.OrthographicCamera(-10,10,7,-7,.1,90); camera.position.set(8,8,21);
 const orbit=new OrbitControls(camera,renderer.domElement);
 Object.assign(orbit,{enablePan:false,minZoom:.8,maxZoom:2.5,minAzimuthAngle:-.55,maxAzimuthAngle:.63,minPolarAngle:.69,maxPolarAngle:1.51});
 orbit.target.set(0,-.3,0); orbit.update();
 scene.add(new T.HemisphereLight(0xe3f3ff,0xa18b78,2.3));
 const sun=new T.DirectionalLight(0xffedcd,3); sun.position.set(-7,15,9); sun.castShadow=true;
 sun.shadow.mapSize.set(1024,1024); Object.assign(sun.shadow.camera,{left:-12,right:12,top:12,bottom:-12}); sun.shadow.normalBias=.04; scene.add(sun);
 const mat=(color,metalness=0)=>new T.MeshStandardMaterial({color,roughness:.58,metalness});
 const concrete=mat('#cbd1ca'),steel=mat('#8396a1',.48),dark=mat('#344954',.45),hot=mat('#c45634'),cool=mat('#247fae'),chrome=mat('#b7c4c8',.4),copper=mat('#b86c47',.38);
 function mesh(geometry,material,position){const o=new T.Mesh(geometry,material);o.position.set(...position);o.castShadow=o.receiveShadow=true;scene.add(o);return o;}
 function box(size,material,position){return mesh(new T.BoxGeometry(...size),material,position);}
 function cyl(r,h,material,position){return mesh(new T.CylinderGeometry(r,r,h,24),material,position);}
 function bar(a,b,r){const start=new T.Vector3(...a),delta=new T.Vector3(...b).sub(start);const o=cyl(r,delta.length(),dark,start.addScaledVector(delta,.5).toArray());o.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),delta.normalize());}
 box([13,.3,7.2],mat('#738765'),[0,-.13,-.55]);
 box([12.9,.65,7.1],mat('#a48669'),[0,-.6,-.55]);
 box([12.85,2.25,7.03],mat('#8d8073'),[0,-2.03,-.55]);
 box([12.8,1.34,6.98],mat(kind==='egs' ? '#806a61' : '#587f8c'),[0,-3.79,-.55]);
 box([12.75,.62,6.93],mat('#675b60'),[0,-4.75,-.55]);
 const rock=new T.DodecahedronGeometry(.16,0), rockMats=['#756d65','#aa9984','#71969c','#426875'].map(c=>mat(c));
 let seed=6123;const sample=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
 for(let i=0;i<32;i++){const o=mesh(rock,rockMats[i%4],[-6+sample()*12,i%2?-3.25-sample():-1.02-sample()*1.9,2.97]);o.scale.set(.55+sample()*.8,.36+sample()*.6,.37);o.rotation.z=sample()*Math.PI*2;}
 const bark=mat('#665447'),foliage=[mat('#4d725c'),mat('#638067')],cone=new T.ConeGeometry(.44,.94,10);
 for(const [x,z,s] of [[-5.7,-2.8,1],[-4.5,-3.2,.8],[-3.4,-3.1,1],[.8,-3.25,.8],[2,-3.3,1],[5.5,-3.2,.8]]){
  cyl(.09,.8*s,bark,[x,.4*s,z]);for(let j=0;j<3;j++)mesh(cone,foliage[j%2],[x,(.94+.3*j)*s,z]).scale.setScalar(s*(1-j*.13));
 }
 // A compact plant and electricity connection echo the main animation.
 box([6.8,.14,3],concrete,[0,.08,.4]);
 box([1.3,1.2,1.15],dark,[-2.1,.75,.5]);
 for(let i=0;i<8;i++)box([1.4,.045,1.22],chrome,[-2.1,.23+i*.15,.5]);
 box([3.1,.22,1.55],concrete,[.3,.25,.5]);
 box([3.1,1.5,.16],mat('#c0cecc'),[.3,1,-.3]);
 const generator=cyl(.52,1.15,copper,[.85,.98,.5]);generator.rotation.z=Math.PI/2;
 for(let i=0;i<7;i++){const fin=cyl(.55,.04,steel,[.38+i*.15,.98,.5]);fin.rotation.z=Math.PI/2;}
 box([1.25,.85,1],steel,[2.6,.52,.4]);
 const pylon=[4.6,-1.8];
 for(const x of [-1,1])for(const z of [-1,1])bar([pylon[0]+x*.52,.02,pylon[1]+z*.4],[pylon[0]+x*.13,3.5,pylon[1]+z*.1],.033);
 for(const x of [-1,1])for(let i=0;i<4;i++)bar([pylon[0]-x*(.52-i*.09),i*.8,pylon[1]+.35],[pylon[0]+x*(.45-i*.09),(i+1)*.8,pylon[1]+.3],.025);
 bar([3.8,3.05,pylon[1]],[5.4,3.05,pylon[1]],.04);
 const routes=[];function curve(points){return new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)),false,'centripetal');}
 function pipe(path,r,material,color,reverse=false,offset=0,count=8){mesh(new T.TubeGeometry(path,80,r,12,false),material,[0,0,0]);const pulses=[];for(let i=0;i<count;i++){const bead=mesh(new T.SphereGeometry(.075,10,8),new T.MeshBasicMaterial({color}),[0,0,0]);bead.castShadow=false;pulses.push(bead);}routes.push({path,pulses,reverse,offset});}
 for(const o of [-.55,0,.55])pipe(curve([[2.9,1,-.2],[3.6,2,-1],[4.6+o,2.9,-1.8],[6,2.6,-2.6]]),.024,dark,'#ffd266',false,0,4);
 const z=3.02;
 function wellhead(x){cyl(.42,.25,concrete,[x,.16,2.55]);cyl(.21,.52,steel,[x,.42,2.55]);}
 const injectionX = 4.7, productionX = -4.7;
 const coldApproach = [[2,.55,1],[3,.42,2.5],[injectionX,.65,2.55],[injectionX,.4,z]];
 const warmExit = [[productionX,.4,z],[productionX,.65,2.55],[-3.5,.45,2.35],[-2.2,.55,1]];
 const coldPipe = points => pipe(curve(points),.14,cool,'#8ee9ff',false,.16);
 const warmPipe = points => pipe(curve(points),.14,hot,'#ffbd80',false,.16);

 let reservoirFlow;
 if (kind === 'hydrothermal' || kind === 'egs') {
  wellhead(injectionX);
  wellhead(productionX);
  coldPipe([...coldApproach,[injectionX,-3.88,z]]);
  warmPipe([[productionX,-3.88,z],...warmExit]);
  reservoirFlow = createReservoirFlow(
   scene, [injectionX,-3.88,z+.06], [productionX,-3.88,z+.06], { fractured: kind === 'egs' }
  );
 } else if (kind === 'u') {
  wellhead(injectionX);
  wellhead(productionX);
  coldPipe([...coldApproach,[injectionX,-3.7,z],[4.2,-4,z],[0,-4,z]]);
  warmPipe([[0,-4,z],[-4.2,-4,z],[productionX,-3.7,z],...warmExit]);
 } else {
  wellhead(injectionX);
  const outer = curve([[injectionX,.4,z],[injectionX,-3.65,z],[4.15,-4,z],[-4.8,-4,z]]);
  const surfaceOut = [
   [injectionX,.4,z],[4.5,.86,2.55],[3.45,.95,2.15],[1.15,.85,1.45],[-1.54,.82,.64]
  ];
  const annulus = cool.clone();
  annulus.transparent = true;
  annulus.opacity = .42;
  annulus.depthWrite = false;
  coldPipe(coldApproach);
  pipe(outer,.24,annulus,'#8ee9ff',false,.12,10);
  pipe(outer,.105,hot,'#ffbd80',true,.16,10);
  warmPipe(surfaceOut);
 }
 let visible=true,motion=true,connected=true,clock=0,last=0;
 const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(6.7,8.5*h/w);camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);
 const observer=new IntersectionObserver(([e])=>visible=e.isIntersecting);observer.observe(host);
 renderer.setAnimationLoop(time => {
  const dt = last ? Math.min(.05,(time-last)/1000) : 0;
  last = time;
  if (!visible) return;
  if (motion && connected) clock += dt*.09;
  for (const {path,pulses,reverse,offset} of routes) {
   pulses.forEach((bead,i) => {
    bead.visible = connected;
    const phase = (clock+i/pulses.length)%1;
    path.getPointAt(reverse ? 1-phase : phase, bead.position);
    bead.position.z += offset;
   });
  }
  reservoirFlow?.draw(clock*12, connected);
  renderer.render(scene,camera);
 });
 return {
  setMotion(value) { motion = value; },
  setConnected(value) {
   connected = value;
   reservoirFlow?.setConnected(value);
  },
  destroy() {
   renderer.setAnimationLoop(null);
   resize.disconnect();
   observer.disconnect();
   orbit.dispose();
   const geometries = new Set(), materials = new Set();
   scene.traverse(item => {
    if (item.geometry) geometries.add(item.geometry);
    if (item.material) materials.add(item.material);
   });
   geometries.forEach(geometry => geometry.dispose());
   materials.forEach(material => material.dispose());
   renderer.dispose();
   renderer.domElement.remove();
  },
 };
}

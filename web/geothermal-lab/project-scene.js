import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { N } from './project-model.js';

export function mountGeothermalScene(host,onPick) {
  const scene=new T.Scene(); scene.background=new T.Color('#e6eff3');
  const renderer=new T.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.outputColorSpace=T.SRGBColorSpace; host.append(renderer.domElement);
  const camera=new T.OrthographicCamera(-15,15,12,-12,.1,120); camera.position.set(22,25,30);
  const orbit=new OrbitControls(camera,renderer.domElement); Object.assign(orbit,{enablePan:false,minZoom:.8,maxZoom:2.8,minPolarAngle:.25,maxPolarAngle:1.4});orbit.target.set(0,-1,0);orbit.update();
  scene.add(new T.HemisphereLight(0xe6f5ff,0x7f8568,2.4)); const sun=new T.DirectionalLight(0xffedcd,2.6);sun.position.set(-7,20,9);scene.add(sun);
  // Leave the surface open so the modelled reservoir and well paths are visible.
  const groundFrame=new T.Group();scene.add(groundFrame);
  for(const [w,d,x,z] of [[20,.35,0,-9.82],[20,.35,0,9.82],[.35,20,-9.82,0],[.35,20,9.82,0]]) {
    const rim=new T.Mesh(new T.BoxGeometry(w,.3,d),new T.MeshStandardMaterial({color:'#657e62',roughness:.8}));rim.position.set(x,-.17,z);groundFrame.add(rim);
  }
  const grid=new T.GridHelper(20,N,'#e4f0e6','#c2d6c4');grid.position.y=.07;scene.add(grid);
  const ground=new T.Mesh(new T.PlaneGeometry(20,20),new T.MeshBasicMaterial({visible:false,side:T.DoubleSide}));ground.rotation.x=-Math.PI/2;ground.position.y=.08;scene.add(ground);
  const staticGroup=new T.Group();scene.add(staticGroup);
  function box(w,h,d,color,x,y,z,opacity=1){const m=new T.Mesh(new T.BoxGeometry(w,h,d),new T.MeshStandardMaterial({color,transparent:opacity<1,opacity,depthWrite:opacity===1,roughness:.8}));m.position.set(x,y,z);staticGroup.add(m);return m;}
  // Open cutaway: the active temperature grid remains visible from above.
  box(20,.22,20,'#9b795f',0,-3.45,0);box(20,1.3,.18,'#bba086',0,-2.65,-9.92);box(.18,1.3,20,'#bba086',-9.92,-2.65,0);
  box(2.8,.65,2.2,'#dbe5e5',0,.36,0);box(2.2,1.1,1.5,'#e8edef',0,1.23,0);box(2.5,.16,1.8,'#a9bdc1',0,1.85,0);
  box(.38,1.0,.38,'#829ba1',.65,2.32,0);box(.6,.2,.6,'#d7e3e3',.65,2.85,0);
  const cells=new T.InstancedMesh(new T.BoxGeometry(1.94,.32,1.94),new T.MeshStandardMaterial({color:'#ffffff',roughness:.9}),N*N);
  const temp=new T.Object3D(), color=new T.Color();
  for(let i=0;i<N*N;i++){temp.position.set((i%N-4.5)*2,-3.1,(4.5-Math.floor(i/N))*2);temp.updateMatrix();cells.setMatrixAt(i,temp.matrix);cells.setColorAt(i,new T.Color('#d46840'));}scene.add(cells);
  const dynamic=new T.Group();scene.add(dynamic); const particles=[];let phase=0,running=true,visible=true,last=0;
  const pos=id=>new T.Vector3((id%N-4.5)*2,0,(4.5-Math.floor(id/N))*2);
  function segment(a,b,r,col){const dir=new T.Vector3().subVectors(b,a),g=new T.CylinderGeometry(r,r,dir.length(),10),m=new T.MeshStandardMaterial({color:col});const mesh=new T.Mesh(g,m);mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),dir.normalize());dynamic.add(mesh);}
  function update(state,selection,config){
    for(const obj of [...dynamic.children]){dynamic.remove(obj);obj.geometry?.dispose();obj.material?.dispose();}particles.length=0;
    const initial=config.initial, low=config.injection;
    state.temperature.forEach((t,i)=>{const f=Math.max(0,Math.min(1,(t-low)/(initial-low)));color.setHSL(.62*(1-f),.78,.5);cells.setColorAt(i,color);});cells.instanceColor.needsUpdate=true;
    const selected=pos(selection),marker=new T.Mesh(new T.RingGeometry(.28,.45,24),new T.MeshBasicMaterial({color:'#e5ae3e',side:T.DoubleSide}));marker.rotation.x=-Math.PI/2;marker.position.copy(selected);marker.position.y=.14;dynamic.add(marker);
    for(const [id,hex] of [[config.injector,'#258cc3'],[config.producer,'#d64d36']]){
      const p=pos(id),top=p.clone().setY(.5),bottom=p.clone().setY(-3.1),plant=new T.Vector3(0,.45,0);
      segment(top,bottom,.12,hex);segment(top,plant,.09,hex);
      const ring=new T.Mesh(new T.CylinderGeometry(.29,.29,.2,16),new T.MeshStandardMaterial({color:hex}));ring.position.copy(top);dynamic.add(ring);
      for(let k=0;k<5;k++){const bead=new T.Mesh(new T.SphereGeometry(.13,10,8),new T.MeshBasicMaterial({color:hex}));dynamic.add(bead);particles.push({bead,top,bottom,offset:k/5,inject:id===config.injector});}
    }
  }
  const ray=new T.Raycaster(),pointer=new T.Vector2();let start=null;
  const down=e=>start={x:e.clientX,y:e.clientY}; const up=e=>{if(!start||Math.hypot(e.clientX-start.x,e.clientY-start.y)>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);ray.setFromCamera(pointer,camera);const hit=ray.intersectObject(ground)[0];if(hit){const x=Math.min(9,Math.max(0,Math.floor((hit.point.x+10)/2))), y=Math.min(9,Math.max(0,Math.floor((10-hit.point.z)/2)));onPick(y*N+x);}};
  renderer.domElement.addEventListener('pointerdown',down);renderer.domElement.addEventListener('pointerup',up);
  const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(12,16*h/w);camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);
  const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;});observer.observe(host);
  renderer.setAnimationLoop(t=>{const dt=last?Math.min(.05,(t-last)/1000):0;last=t;if(!visible)return;if(running)phase+=dt*.2;for(const p of particles){const f=(phase+p.offset)%1;p.bead.position.lerpVectors(p.inject?p.top:p.bottom,p.inject?p.bottom:p.top,f);}renderer.render(scene,camera);});
  return {update,setRunning(v){running=v;},destroy(){renderer.setAnimationLoop(null);resize.disconnect();observer.disconnect();orbit.dispose();renderer.domElement.removeEventListener('pointerdown',down);renderer.domElement.removeEventListener('pointerup',up);scene.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of [].concat(o.material))m.dispose();}});renderer.dispose();renderer.domElement.remove();}};
}

import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export function mountClosedLoopScene(host){
 const scene=new T.Scene();scene.background=new T.Color('#e6eff3');const renderer=new T.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=T.SRGBColorSpace;host.append(renderer.domElement);
 const camera=new T.OrthographicCamera(-13,13,8,-8,.1,120);camera.position.set(22,19,30);const orbit=new OrbitControls(camera,renderer.domElement);Object.assign(orbit,{enablePan:false,minZoom:.7,maxZoom:2.8,minPolarAngle:.3,maxPolarAngle:1.4});orbit.target.set(0,-.8,0);orbit.update();
 scene.add(new T.HemisphereLight(0xe6f5ff,0x7f8568,2.2));const sun=new T.DirectionalLight(0xffedcd,2.4);sun.position.set(-7,20,9);scene.add(sun);
 function box(w,h,d,c,x,y,z){const m=new T.Mesh(new T.BoxGeometry(w,h,d),new T.MeshStandardMaterial({color:c,roughness:.8}));m.position.set(x,y,z);scene.add(m);}
 for(const x of [-5.7,5.7]){box(9,.38,6,'#879f7e',x,.1,0);box(9,1.15,6,'#b99b80',x,-.65,0);box(9,1.1,6,'#8f8274',x,-1.75,0);box(9,1.0,6,'#518289',x,-2.8,0);box(9,.35,6,'#685c58',x,-3.48,0);box(2,.5,1.3,'#dce7e6',x, .55,-.8);box(1.3,.8,.85,'#c6d4d8',x,1.18,-.8);}
 const pipeZ=3.12;
 function curve(points){return new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)),false,'centripetal');}
 function tube(path,r,col,opacity=1){const mesh=new T.Mesh(new T.TubeGeometry(path,80,r,10,false),new T.MeshStandardMaterial({color:col,transparent:opacity<1,opacity,depthWrite:opacity===1}));scene.add(mesh);}
 const cold=curve([[-8.7,.55,pipeZ],[-8.7,-.8,pipeZ],[-8.7,-2.8,pipeZ],[-8.2,-3.05,pipeZ],[-5.7,-3.05,pipeZ]]);
 const hot=curve([[-5.7,-3.05,pipeZ],[-3.2,-3.05,pipeZ],[-2.7,-2.8,pipeZ],[-2.7,-.8,pipeZ],[-2.7,.55,pipeZ]]);
 const coax=curve([[3.1,.55,pipeZ],[3.1,-.9,pipeZ],[3.1,-2.8,pipeZ],[3.6,-3.05,pipeZ],[8.7,-3.05,pipeZ]]);
 tube(cold,.16,'#258ac0');tube(hot,.16,'#d4553f');tube(coax,.27,'#258ac0',.55);tube(coax,.115,'#d4553f');
 const beads=[];for(const [path,reverse,col,offset] of [[cold,false,'#d7f9ff',0],[hot,false,'#ffdbb5',0],[coax,false,'#d7f9ff',.2],[coax,true,'#ffdbb5',-.12]])for(let i=0;i<7;i++){const m=new T.Mesh(new T.SphereGeometry(.11,10,8),new T.MeshBasicMaterial({color:col}));scene.add(m);beads.push({m,path,reverse,offset,phase:i/7});}
 let visible=true,motion=true,clock=0,last=0;const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(7.4,11*h/w);camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);const observer=new IntersectionObserver(([e])=>visible=e.isIntersecting);observer.observe(host);
 renderer.setAnimationLoop(t=>{const dt=last?Math.min(.05,(t-last)/1000):0;last=t;if(!visible)return;if(motion)clock+=dt*.10;for(const b of beads){let f=(clock+b.phase)%1;if(b.reverse)f=1-f;b.m.position.copy(b.path.getPointAt(f));b.m.position.z+=b.offset;}renderer.render(scene,camera);});
 return {setMotion(v){motion=v;},destroy(){renderer.setAnimationLoop(null);resize.disconnect();observer.disconnect();orbit.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of [].concat(o.material))m.dispose();});renderer.dispose();renderer.domElement.remove();}};
}

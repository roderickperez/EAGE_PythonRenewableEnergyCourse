import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const cool = 0x247fae, warm = 0xc45432, teal = 0x279084, green = 0x267762;
const water = 0xa7eaf7, cream = 0xffe6a3, metal = 0x718f96;
const v = (x,y,z) => new T.Vector3(x,y,z);

export function mountEquationScene(host, name) {
  const renderer = new T.WebGLRenderer({ antialias:true });
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.domElement.setAttribute('aria-hidden','true');
  const scene = new T.Scene();
  scene.background = new T.Color('#e6eff3');
  scene.add(new T.HemisphereLight(0xf4fdff,0x9d876e,2.3));
  const sun = new T.DirectionalLight(0xffeedb,2);
  sun.position.set(-4,8,7);
  scene.add(sun);
  const camera = new T.OrthographicCamera(-6,6,3,-3,.1,75);
  camera.position.set(6,5,11);
  host.append(renderer.domElement);
  const orbit = new OrbitControls(camera,renderer.domElement);
  Object.assign(orbit,{enablePan:false,minZoom:.75,maxZoom:2.4,autoRotateSpeed:.7});
  orbit.target.y = .1;
  orbit.update();

  const group = new T.Group();
  scene.add(group);
  const materials = new Map();
  function material(color) {
    if (!materials.has(color)) materials.set(color,new T.MeshStandardMaterial({color,roughness:.65}));
    return materials.get(color);
  }
  function mesh(geometry,color,x=0,y=0,z=0,parent=group) {
    const item = new T.Mesh(geometry,material(color));
    item.position.set(x,y,z);
    parent.add(item);
    return item;
  }
  function box(w,h,d,color,x,y,z,parent) {
    return mesh(new T.BoxGeometry(w,h,d),color,x,y,z,parent);
  }
  function cylinder(radius,length,color,x,y,z,parent) {
    return mesh(new T.CylinderGeometry(radius,radius,length,28),color,x,y,z,parent);
  }
  const curve = points => new T.CatmullRomCurve3(points.map(([x,y,z])=>v(x,y,z)));
  function pipe(path,radius,color) {
    return mesh(new T.TubeGeometry(path,32,radius,9,false),color);
  }
  function path(points,radius,color) {
    return pipe(curve(points),radius,color);
  }
  function stream(route,count,color,speed,radius=.09,endColor=color) {
    const dots = Array.from({length:count},()=>mesh(new T.SphereGeometry(radius,10,8),color));
    return time => {
      dots.forEach((dot,i)=>{
        dot.visible = speed>0;
        const position = (time*speed+i/dots.length)%1;
        dot.position.copy(route.getPointAt(position));
        dot.material = material(position>.64 ? endColor : color);
      });
    };
  }
  function surfaceStream(route, count, color, speed, offset) {
    const arc = new T.CatmullRomCurve3(route.getPoints(32).map(point=>point.add(v(0,0,offset))));
    return stream(arc,count,color,speed);
  }
  function wheel(x,y,z,radius) {
    const rotor = new T.Group();
    rotor.position.set(x,y,z);
    group.add(rotor);
    cylinder(radius*.24,.21,cream,0,0,0,rotor).rotateX(Math.PI/2);
    for (let i=0;i<7;i++) {
      const blade = new T.Group();
      rotor.add(blade);
      blade.rotation.z = i*Math.PI*2/7;
      box(radius*.17,radius*.73,.09,0xe8d7ab,.06,radius*.52,0,blade).rotateZ(-.32);
    }
    return rotor;
  }
  const functions = {
    heat({ flow },{power,enthalpy}) {
      box(6.3,.23,2.35,metal,0,-1.08,-.25);
      box(3.95,1.58,1.4,0xa0b6b7,0,-.12,-.67);
      box(3.95,.21,1.81,0xc4d8d9,0,.72,-.48);
      const inlet = [[-4.3,-.15,.33],[-3.2,-.15,.33],[-2.1,-.15,.33],[-1,-.15,.33],[0,-.15,.33]];
      const outlet = [[0,-.15,.33],[1,-.15,.33],[2.1,-.15,.33],[3.2,-.15,.33],[4.3,-.15,.33]];
      path(inlet,.21,warm);
      path(outlet,.21,cool);
      const secondary = curve([[2.7,.45,.56],[1.65,.43,.56],[-1.2,.43,.56],[-1.56,.25,.7],[-1.42,-.62,.76],[-.8,-.71,.76],[1.65,-.71,.76],[2.7,-.71,.76]]);
      pipe(secondary,.13,teal);
      const brine = stream(curve([...inlet,...outlet.slice(1)].map(([x,y,z])=>[x,y,z+.21])),10,cream,flow/400,.09,water);
      const recovered = stream(curve([[-1.2,.42,.74],[0,.42,.74],[1.4,.42,.74],[2.68,.43,.74]]),6,cream,power/120);
      const heat = [-1,-.5,0,.5,1].map(x=>stream(curve([[x,.02,.66],[x,.18,.68],[x,.38,.7]]),2,cream,flow>0 ? enthalpy/2200 : 0,.07));
      return t => { brine(t); recovered(t); heat.forEach(move=>move(t)); };
    },
    carnot(input,{efficiency}) {
      box(2.2,1,1.45,0xc85634,-3.15,1,-.2);
      box(2.2,1,1.45,0x358eb6,-3.15,-1.13,-.2);
      box(1.65,1.8,1.55,metal,.33,-.06,-.18);
      cylinder(.62,.22,0x345565,.32,-.06,.73).rotateX(Math.PI/2);
      const rotor = wheel(.32,-.06,.87,.53);
      const toEngine = curve([[-2.04,1.1,.35],[-1,1.1,.35],[-.52,.81,.36],[-.44,.43,.37]]);
      const toCold = curve([[-.44,-.59,.37],[-.55,-.96,.36],[-1,-1.13,.35],[-2.04,-1.13,.35]]);
      const shaft = curve([[1.17,-.06,.49],[2.21,-.06,.49],[3.56,-.06,.49]]);
      pipe(toEngine,.21,warm);
      pipe(toCold,.21*(1-efficiency),cool);
      pipe(shaft,.10+efficiency*.23,green);
      box(.9,.92,.9,green,3.65,-.06,.49);
      const moveIn = surfaceStream(toEngine,4,cream,.32,.22);
      const moveCold = surfaceStream(toCold,4,water,.32*(1-efficiency),.19);
      const moveWork = surfaceStream(shaft,3,cream,.38*efficiency,.2);
      return t => { rotor.rotation.z=-t*1.35; moveIn(t); moveCold(t); moveWork(t); };
    },
    pump({pressure},{volumeFlow,power}) {
      box(6.1,.21,2.09,0x879fa2,-.11,-1.19,-.26);
      cylinder(.63,2.15,metal,-1.36,-.39,-.33).rotateZ(Math.PI/2);
      for(let i=0;i<8;i++) cylinder(.66,.065,0x98b3b5,-2.31+i*.26,-.39,-.33).rotateZ(Math.PI/2);
      cylinder(.83,.75,0x426d77,.45,-.37,.27).rotateX(Math.PI/2);
      cylinder(.73,.18,0x357f93,.45,-.37,.73).rotateX(Math.PI/2);
      const rotor = wheel(.45,-.37,.86,.62);
      const inlet = curve([[-4.25,-.45,1.43],[-2.22,-.45,1.24],[-.88,-.42,1.18],[-.57,-.39,1.04],[.45,-.37,.98]]);
      const outlet = curve([[.54,-.31,.79],[.96,.37,.79],[1.08,.75,.77],[1.65,.94,.77],[4.17,.94,.77]]);
      pipe(inlet,.16,cool);
      pipe(outlet,.16,0x23718c);
      cylinder(.24,.12,0x274d5c,4.09,.94,.77).rotateZ(Math.PI/2);
      cylinder(.41,.09,0xf3f8f6,2.78,1.56,.8).rotateX(Math.PI/2);
      cylinder(.05,.28,metal,2.78,1.08,.79);
      const needle = box(.037,.29,.05,warm,2.78,1.56,.89);
      needle.geometry.translate(0,.123,0);
      needle.rotation.z = Math.PI*(.66-1.32*pressure/15);
      const moveIn = surfaceStream(inlet,5,water,volumeFlow*2.7,.18);
      const moveOut = surfaceStream(outlet,6,water,volumeFlow*2.7,.18);
      return t => { rotor.rotation.z=-t*Math.min(12,power*8); moveIn(t); moveOut(t); };
    },
    darcy({permeability},{flux}) {
      box(8.1,1.24,1.75,0xb6a489,0,-.47,-.28);
      box(8.1,.24,1.75,0x938476,0,-1.18,-.28);
      const radius = .046+.11*Math.sqrt((permeability-1)/299);
      const inlet = [[-3.75,1.36,.95],[-3.75,.63,.95],[-3.75,-.35,.95],[-3.33,-.4,.95]];
      const outlet = [[3.33,-.41,.95],[3.75,-.39,.95],[3.75,.63,.95],[3.75,1.36,.95]];
      path(inlet,.175,cool);
      path(outlet,.175,warm);
      const movers = [-.69,-.39,-.11].map((height,i)=>{
        const pore = [[-3.33,-.4,.95],[-2.15,height,.94],[-.85,height+.09,.96],[.42,height-.08,.93],[2.12,height+.045,.93],[3.33,-.41,.95]];
        path(pore,radius,0x579db0);
        const route = [...inlet,...pore.slice(1),...outlet.slice(1)].map(([x,y,z])=>[x,y,z+.14]);
        return stream(curve(route),4,water,flux>0 ? .075+.07*Math.log1p(flux*1e6) : 0,.067,i ? cream : water);
      });
      return t => movers.forEach(move=>move(t));
    },
    storage({volume,porosity}) {
      const size = 3.9*Math.cbrt(volume/5);
      const grains = 0xbaa183;
      if (porosity===0) {
        box(size,size*.68,size*.76,grains,0,0,0);
      } else {
        const nx=7, ny=5, nz=5;
        const dx=size/nx, dy=size*.68/ny, dz=size*.76/nz;
        const width = 1-porosity/140;
        const poreSize = dx*(.12+porosity*.015);
        for (let x=0;x<nx;x++) for (let y=0;y<ny;y++) for(let z=0;z<nz;z++) {
          const ix=(x-(nx-1)/2)*dx, iy=(y-(ny-1)/2)*dy, iz=(z-(nz-1)/2)*dz;
          const n = x*15+y*11+z*7, tint = [grains,0x9f866d,0xc4ad8e,0xad957a][n%4];
          const grain = mesh(new T.IcosahedronGeometry(.76,1),tint,ix,iy,iz);
          grain.scale.set(dx*width,dy*width,dz*width);
          grain.rotation.set(n*.25,n*.16,n*.41);
          function pocket(px,py,pz,sx,sy,sz) {
            mesh(new T.IcosahedronGeometry(poreSize,1),0x238db4,px,py,pz).scale.set(sx,sy,sz);
          }
          if (x<nx-1 && z<nz-1) {
            pocket(ix+dx*.5,iy,iz+dz*.5,1,1,1);
            if (y===ny-1) pocket(ix+dx*.5,iy+dy*.45,iz+dz*.5,1.1,.62,1);
          }
          if (y<ny-1 && x<nx-1 && (z===0 || z===nz-1)) {
            pocket(ix+dx*.5,iy+dy*.5,iz+Math.sign(iz)*dz*.45,1.18,1,.6);
          }
          if (y<ny-1 && z<nz-1 && (x===0 || x===nx-1)) {
            pocket(ix+Math.sign(ix)*dx*.45,iy+dy*.5,iz+dz*.5,.6,1,1.18);
          }
        }
      }
      return () => {};
    },
  };

  let visible = false, playing = true, animationFrame = 0, last = 0, time = 0;
  let move = () => {}, storageKey;
  function draw() { renderer.render(scene,camera); }
  function tick(now) {
    const elapsed = Math.max(0,Math.min((now-last)/1000,.05));
    last = now;
    time += elapsed;
    if (name==='storage') orbit.update(elapsed);
    move(time);
    draw();
    animationFrame = requestAnimationFrame(tick);
  }
  function schedule() {
    cancelAnimationFrame(animationFrame);
    const active = playing && visible && !document.hidden;
    orbit.autoRotate = active && name==='storage';
    if (active) {
      last = performance.now();
      animationFrame = requestAnimationFrame(tick);
    }
  }
  const intersection = new IntersectionObserver(([entry])=>{
    visible = entry.isIntersecting;
    schedule();
  });
  intersection.observe(host);
  document.addEventListener('visibilitychange',schedule);
  orbit.addEventListener('change',draw);
  const resize = new ResizeObserver(()=>{
    const {clientWidth:width,clientHeight:height} = host;
    if (!width || !height) return;
    const aspect = width/height, half = Math.max(name==='storage' ? 3.3 : 2.55,4.9/aspect);
    Object.assign(camera,{left:-half*aspect,right:half*aspect,top:half,bottom:-half});
    camera.updateProjectionMatrix();
    renderer.setSize(width,height,false);
    draw();
  });
  resize.observe(host);

  return {
    update(input,values) {
      if (name==='storage') {
        const key = `${input.volume}/${input.porosity}`;
        if (key===storageKey) return;
        storageKey = key;
      }
      for (const child of [...group.children]) {
        child.traverse(item=>item.geometry?.dispose());
        child.removeFromParent();
      }
      move = functions[name](input,values);
      move(time);
      draw();
    },
    setPlaying(value) { playing=value; schedule(); },
    destroy() {
      cancelAnimationFrame(animationFrame);
      intersection.disconnect();
      resize.disconnect();
      document.removeEventListener('visibilitychange',schedule);
      orbit.dispose();
      group.traverse(item=>item.geometry?.dispose());
      materials.forEach(item=>item.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

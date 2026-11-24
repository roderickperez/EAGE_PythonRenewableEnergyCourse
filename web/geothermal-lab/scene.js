import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createReservoirFlow } from './reservoir-flow.js';

export function buildGeothermalPlant(scene) {
  const mat = (color, metalness = 0) => new T.MeshStandardMaterial({ color, roughness: .58, metalness });
  const steel = mat('#8396a1', .48), dark = mat('#344954', .45), concrete = mat('#cbd1ca');
  const chrome = mat('#b7c4c8', .4), copper = mat('#b86c47', .38);
  const hot = mat('#c45634'), cool = mat('#247fae'), vapor = mat('#8858ac'), liquid = mat('#477f95');
  const clickable = [], routes = [];
  function mesh(geometry, material, position, label, parent = scene) {
    const item = new T.Mesh(geometry, material);
    item.position.set(...position);
    item.castShadow = item.receiveShadow = true;
    if (label) { item.userData.label = label; clickable.push(item); }
    parent.add(item);
    return item;
  }
  const box = (size, material, position, label, parent) => mesh(new T.BoxGeometry(...size), material, position, label, parent);
  const cylinder = (top, bottom, height, material, position, label, parent) =>
    mesh(new T.CylinderGeometry(top, bottom, height, 28), material, position, label, parent);
  function bar(a, b, radius, label) {
    const start = new T.Vector3(...a), delta = new T.Vector3(...b).sub(start);
    const piece = cylinder(radius, radius, delta.length(), dark, start.addScaledVector(delta, .5).toArray(), label);
    piece.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), delta.normalize());
  }
  const pulseGeometry = new T.SphereGeometry(.085, 12, 8);
  function pipe(points, radius, material, color, channel, label, count = 12) {
    const curve = new T.CatmullRomCurve3(points.map(point => new T.Vector3(...point)), false, 'centripetal');
    mesh(new T.TubeGeometry(curve, 64, radius, 12, false), material, [0,0,0], label);
    const light = new T.MeshBasicMaterial({ color });
    const pulses = Array.from({ length: count }, () => {
      const pulse = mesh(pulseGeometry, light, [0,0,0]);
      pulse.castShadow = false;
      return pulse;
    });
    routes.push({ curve, pulses, channel, radius, phase: 0, distance: curve.getLength() });
  }

  // The front face is a rock section. Well tubing is exposed at that plane.
  box([16.8,.3,7.2], mat('#738765'), [0,-.13,-.55], 'Ground surface');
  box([16.7,.65,7.1], mat('#a48669'), [0,-.6,-.55], 'Overburden');
  box([16.65,2.25,7.03], mat('#8d8073'), [0,-2.03,-.55], 'Cap rock');
  box([16.6,1.34,6.98], mat('#587f8c'), [0,-3.79,-.55], 'Water-bearing geothermal reservoir');
  box([16.55,.62,6.93], mat('#675b60'), [0,-4.75,-.55], 'Hot basement rock');
  const warmRock = mat('#ab7359');
  box([16.52,.18,6.9], warmRock, [0,-4.64,-.55], 'Hot rock beneath reservoir');
  const stone = new T.DodecahedronGeometry(.19, 0);
  const stoneColors = [mat('#756d65'), mat('#aa9984'), mat('#71969c'), mat('#426875')];
  let seed = 6123;
  const sample = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  for (let i = 0; i < 46; i++) {
    const x = -7.9 + sample() * 15.8;
    const y = i % 2 ? -3.26 - sample() * 1.04 : -1.01 - sample() * 1.94;
    const fragment = mesh(stone, stoneColors[i % 4], [x,y,2.96], '');
    fragment.scale.set(.55+sample()*.8,.36+sample()*.6,.37);
    fragment.rotation.z = sample() * 2 * Math.PI;
  }
  const foliage = [mat('#4d725c'), mat('#638067')], bark = mat('#665447');
  const cone = new T.ConeGeometry(.44,.94,10);
  for (const [x,z,s] of [[-7.1,-2.85,1.15],[-6,-2.8,.9],[-4.95,-3.35,.85],[-3.85,-3.2,1.12],[.9,-3.25,.85],[2,-3.4,1.15],[7.3,-3.3,.85]]) {
    cylinder(.075,.11,.81*s,bark,[x,.4*s,z],'Tree');
    for (let j=0;j<3;j++) mesh(cone,foliage[j%2],[x,(.94+.3*j)*s,z],'Tree').scale.setScalar(s*(1-j*.13));
  }
  box([11,.14,3.15],concrete,[-.7,.08,.8],'Plant service pad');
  for (const [x,label] of [[-5.9,'Production wellhead'],[5.45,'Reinjection wellhead']]) {
    cylinder(.45,.52,.28,concrete,[x,.16,2.56],label);
    cylinder(.23,.23,.53,steel,[x,.41,2.56],label);
    cylinder(.37,.37,.11,chrome,[x,.62,2.56],label);
  }
  const exchanger = [-3.5,1.03,.9];
  box([1.52,1.32,1.26],dark,exchanger,'Heat exchanger keeps the two fluids separate');
  for (let i=0;i<9;i++) box([1.62,.049,1.35],chrome,[-3.5,.44+i*.141,.9],'Heat-exchanger plates');
  const turbine = [-.85,1.05,.74];
  const generator = [1.22,1.05,.74];
  box([4.1,.29,1.72],concrete,[.1,.25,.74],'Turbine-generator foundation');
  box([4.15,1.7,.17],mat('#c0cecc'),[.11,1,-.14],'Open powerhouse wall');
  box([4.25,.16,.54],concrete,[.11,1.9,.04],'Powerhouse roof cutaway');
  const axle = cylinder(.135,.135,3.08,steel,[.15,1.05,.74],'Turbine–generator shaft');
  axle.rotation.z = Math.PI/2;
  const casing = new T.CylinderGeometry(.65,.65,.63,32,1,true,Math.PI/2,Math.PI);
  const housingMaterial = dark.clone();
  housingMaterial.side = T.DoubleSide;
  const turbineHousing = mesh(casing,housingMaterial,[-1.05,1.05,.74],'Turbine casing');
  turbineHousing.rotation.z = Math.PI/2;
  const runner = new T.Group();
  runner.position.set(-1.05,1.05,.74);
  scene.add(runner);
  const hub = cylinder(.19,.19,.19,chrome,[0,0,0],'Turbine runner',runner);
  hub.rotation.z = Math.PI/2;
  for (let i=0;i<9;i++) {
    const angle = i*Math.PI*2/9;
    box([.12,.33,.12],i===0?copper:steel,[0,.35*Math.cos(angle),.35*Math.sin(angle)],'Turbine blade',runner)
      .rotation.x = angle+.37;
  }
  const stator = cylinder(.52,.52,1.12,copper,generator,'Generator');
  stator.rotation.z = Math.PI/2;
  for (let i=0;i<8;i++) {
    const fin = cylinder(.55,.55,.045,steel,[.71+i*.144,1.05,.74],'Generator cooling fins');
    fin.rotation.z = Math.PI/2;
  }
  box([1.5,.75,1.18],liquid,[3.4,.48,1.23],'Condenser');
  box([1.67,1.14,1.12],steel,[3.4,.76,-.4],'Dry cooler rejects cycle heat');
  for(let i=0;i<9;i++) box([.05,.58,.06],dark,[2.77+i*.16,.83,.18],'Dry cooler fins');
  for(const x of [3.02,3.78]) cylinder(.25,.25,.07,dark,[x,1.37,-.39],'Dry cooler fan');
  cylinder(.26,.26,.43,steel,[-2.24,.52,1.95],'Working-fluid circulation pump');
  box([1.16,.9,.84],dark,[1.55,.5,-1.32],'Step-up transformer');
  for(let i=0;i<7;i++) box([1.26,.052,.94],steel,[1.55,.24+.105*i,-1.32],'Transformer fins');
  const pylon = [5.9,-1.9];
  for (const x of [-1,1]) {
    for (const z of [-1,1]) bar([pylon[0]+x*.57,.02,pylon[1]+z*.45],[pylon[0]+x*.14,4.65,pylon[1]+z*.12],.036,'Grid transmission tower');
    for(let i=0;i<5;i++) bar([pylon[0]-x*(.58-i*.083),i*.83,pylon[1]+.39],
      [pylon[0]+x*(.50-i*.083),(i+1)*.83,pylon[1]+.33],.026,'Tower bracing');
  }
  bar([4.82,4,pylon[1]],[6.98,4,pylon[1]],.045,'Grid transmission crossarm');
  for(const o of [-.84,0,.84]) {
    cylinder(.055,.055,.23,chrome,[1.55+o*.45,1.03,-1.32],'Transformer bushing');
    cylinder(.055,.055,.3,chrome,[pylon[0]+o,3.86,pylon[1]],'Transmission insulator');
    pipe([[1.55+o*.45,1.14,-1.32],[3.72+o*.45,2.1,-1.55],[5.9+o,3.71,-1.9],[7+o*.45,3.27,-2.58],[8.25+o*.24,3.58,-3.16]],
      .029,dark,'#ffd266','grid','Grid transmission conductor',6);
  }
  pipe([[1.69,1.2,.72],[1.8,.62,.26],[1.55,.48,-.43],[1.55,.65,-1.12]],.052,dark,'#ffd266','gross','Generator output to transformer',5);
  pipe([[-5.1,-3.8,2.99],[-5.86,-3.76,3.04],[-5.9,-2.95,3.03],[-5.9,.47,3.03],[-5.9,.78,2.57],[-5.25,.78,2.42],[-4.35,.78,1.2],[-3.99,.78,.9]],
    .145,hot,'#ffab60','brine','Hot brine rises through the production well');
  pipe([[-3.93,.63,1.18],[-3.38,.42,2.55],[-2.3,.39,2.65],[3.95,.39,2.65],[5.45,.64,2.57],[5.45,.46,3.03],[5.45,-3.47,3.03],[5.28,-3.88,3.04],[4.58,-3.88,3.04]],
    .118,cool,'#8ee9ff','brine','Cooled brine descends into the reinjection well');
  pipe([[-3.02,1.45,.7],[-2.55,1.53,.52],[-1.54,1.4,.55],turbine],
    .12,vapor,'#ecc2ff','cycle','Organic vapor from heat exchanger to turbine',7);
  pipe([[-.85,1.04,.48],[-.56,.59,.13],[.3,.53,-.46],[2.15,.44,.05],[2.93,.54,1.11]],
    .11,vapor,'#ecc2ff','cycle','Turbine exhaust to condenser',8);
  pipe([[3.2,.4,1.49],[2.88,.33,1.97],[.94,.33,1.97],[-2.24,.52,1.95],[-3.32,.78,1.58],[-3.5,.88,1.31]],
    .105,liquid,'#a6ddf4','cycle','Condensed working fluid returns through the circulation pump',10);
  pipe([[3.49,.43,.82],[3.49,.49,.15],[3.49,1.11,-.46]],.086,liquid,'#a6ddf4','cycle','Condenser coolant to dry cooler',5);
  pipe([[3.81,1.11,-.46],[3.81,.52,.11],[3.81,.48,.79]],.086,cool,'#8ee9ff','cycle','Coolant return from dry cooler',5);

  const reservoirFlow = createReservoirFlow(scene, [4.58,-3.88,3.08], [-5.1,-3.8,3.08]);
  let reservoirDistance=0;
  function advance(dt, state, animate) {
    if (animate) runner.rotation.x = (runner.rotation.x + dt * Math.min(7,state.gross*1.5)) % (Math.PI*2);
    for (const route of routes) {
      const rate = state[route.channel];
      if (animate) route.phase=(route.phase+dt*Math.min(4,.5+rate)/route.distance)%1;
      route.pulses.forEach((pulse,i)=>{
        pulse.visible=rate>0;
        if (pulse.visible) {
          pulse.position.copy(route.curve.getPointAt((i/route.pulses.length+route.phase)%1));
          pulse.position.z += route.radius * .85;
        }
      });
    }
    if (animate && state.brine > 0) reservoirDistance += dt * Math.min(4,.5+state.brine);
    reservoirFlow.draw(reservoirDistance, state.brine > 0);
  }
  return { advance, clickable, routes, runner, warmRock, reservoirFlow };
}

export function mountGeothermalScene(host, tooltip) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e6eff3');
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  host.append(renderer.domElement);
  const camera = new T.OrthographicCamera(-10, 10, 7, -7, .1, 90);
  camera.position.set(8, 8, 21);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, -.3, 0);
  Object.assign(controls, { enablePan: false, enableZoom: false, minAzimuthAngle: -.55, maxAzimuthAngle: .63,
    minPolarAngle: .69, maxPolarAngle: 1.51 });
  controls.update();
  scene.add(new T.HemisphereLight(0xe3f3ff, 0xa18b78, 2.3));
  const sun = new T.DirectionalLight(0xffedcd, 3);
  sun.position.set(-7, 15, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12 });
  sun.shadow.normalBias = .04;
  scene.add(sun);
  const { advance, clickable, routes, runner, warmRock, reservoirFlow } = buildGeothermalPlant(scene);
  let state, animate, dirty = true, visible = true, last = 0;
  const ray = new T.Raycaster();
  const cursor = new T.Vector2();
  function inspect(event) {
    const bounds = host.getBoundingClientRect();
    cursor.set(2*(event.clientX-bounds.left)/bounds.width-1, 1-2*(event.clientY-bounds.top)/bounds.height);
    ray.setFromCamera(cursor,camera);
    const hit = ray.intersectObjects(clickable,false)[0];
    tooltip.hidden = !hit;
    if (hit) {
      tooltip.textContent = hit.object.userData.label;
      tooltip.style.left = Math.max(6,Math.min(bounds.width-180,event.clientX-bounds.left+10)) + 'px';
      tooltip.style.top = Math.max(6,event.clientY-bounds.top-42) + 'px';
    }
  }
  const hide = () => { tooltip.hidden = true; };
  host.addEventListener('pointermove',inspect);
  host.addEventListener('pointerdown',inspect);
  host.addEventListener('pointerleave',hide);
  controls.addEventListener('change',() => { dirty = true; });
  const size = new ResizeObserver(() => {
    const { clientWidth: width,clientHeight: height } = host;
    if (!width || !height) return;
    const half = Math.max(6.15,9.7*height/width);
    Object.assign(camera,{left:-half*width/height,right:half*width/height,top:half,bottom:-half});
    camera.updateProjectionMatrix();
    renderer.setSize(width,height,false);
    dirty = true;
  });
  const view = new IntersectionObserver(([entry]) => { visible=entry.isIntersecting; dirty=true; });
  size.observe(host);
  view.observe(host);
  renderer.setAnimationLoop(time => {
    const dt = Math.min(.05,(time-last)/1000);
    last = time;
    if (!state || !visible || (!animate && !dirty)) return;
    advance(dt, state, animate);
    renderer.render(scene,camera);
    dirty=false;
  });
  const coldRock = new T.Color('#84766c'), heatRock = new T.Color('#d27c48');
  return {
    update(input,result,motion) {
      state = { brine:input.flow/30, cycle:result.thermal/12, gross:result.gross, grid:result.exported };
      animate=motion;
      warmRock.color.copy(coldRock).lerp(heatRock,(input.reservoir-65)/155);
      host.dataset.exporting=String(result.exported>0);
      dirty=true;
    },
    destroy() {
      renderer.setAnimationLoop(null);
      size.disconnect();
      view.disconnect();
      controls.dispose();
      host.removeEventListener('pointermove',inspect);
      host.removeEventListener('pointerdown',inspect);
      host.removeEventListener('pointerleave',hide);
      const geometries=new Set(), materials=new Set();
      scene.traverse(item=>{ if(item.geometry) geometries.add(item.geometry); if(item.material) materials.add(item.material); });
      geometries.forEach(geometry=>geometry.dispose());
      materials.forEach(material=>material.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

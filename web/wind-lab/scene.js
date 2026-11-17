import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RAD } from './model.js';

export function mountWindScene(host, tooltip, { showSweep = false } = {}) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e6eff3');
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  host.append(renderer.domElement);
  const camera = new T.OrthographicCamera(-10, 10, 8, -8, .1, 100);
  camera.position.set(12, 11, 23);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 5, 0);
  Object.assign(orbit, { enablePan: false, enableZoom: false, minPolarAngle: .55, maxPolarAngle: 1.47 });
  orbit.update();
  scene.add(new T.HemisphereLight(0xe6f5ff, 0x7f8568, 2.4));
  const sun = new T.DirectionalLight(0xffedcd, 3);
  sun.position.set(-7, 16, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12 });
  sun.shadow.normalBias = .035;
  scene.add(sun);
  const material = (color, metalness = 0) => new T.MeshStandardMaterial({ color, metalness, roughness: .59 });
  const white = material('#e3e9eb'), steel = material('#788a93', .48), dark = material('#334752', .4);
  const copper = material('#be7545', .45), green = material('#657e62');
  const parts = [];
  function mesh(geometry, mat, pos, label, parent = scene) {
    const item = new T.Mesh(geometry, mat);
    item.position.set(...pos);
    item.castShadow = item.receiveShadow = true;
    if (label) { item.userData.label = label; parts.push(item); }
    parent.add(item);
    return item;
  }
  const box = (size, mat, pos, label, parent) => mesh(new T.BoxGeometry(...size), mat, pos, label, parent);
  const cylinder = (top, base, height, mat, pos, label, parent) =>
    mesh(new T.CylinderGeometry(top, base, height, 28), mat, pos, label, parent);
  function bar(a, b, radius, mat, label) {
    const start = new T.Vector3(...a), end = new T.Vector3(...b);
    const dir = end.clone().sub(start);
    const segment = mesh(new T.CylinderGeometry(radius, radius, dir.length(), 8), mat, start.addScaledVector(dir, .5).toArray(), label);
    segment.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.normalize());
    return segment;
  }
  box([17, .55, 9.6], material('#aeb8a1'), [0, -.28, 0], 'Terrain section');
  box([16.95, .09, 9.55], green, [0, -.015, 0], 'Site terrain');
  box([9.3, .06, 1.65], material('#abb2a8'), [1.4, .06, .25], 'Service pad');
  const foliage = [material('#476a55'), material('#58765b'), material('#749268')];
  for (const [x, z, scale] of [[-6.6,-2.8,1.2],[-5,-3.6,.85],[-3.7,-3.65,1.05],[1.2,-3.8,.9],[2.75,-3.8,1.16],[6.9,-3.6,.9]]) {
    cylinder(.07, .11, scale, material('#6c5644'), [x, scale / 2, z], 'Tree');
    for (let i = 0; i < 3; i++) {
      mesh(new T.ConeGeometry((.48 - i * .08) * scale, .82 * scale, 10), foliage[i],
        [x, (1.04 + i * .28) * scale, z], 'Tree');
    }
  }
  const towerX = -2.8;
  const column = cylinder(.27, .6, 7.5, white, [towerX, 3.85, 0], 'Tapered turbine tower');
  cylinder(.81, .94, .27, steel, [towerX, .17, 0], 'Tower foundation');
  const head = new T.Group();
  scene.add(head);
  box([1.13, .19, 2.1], white, [0, -.31, -.38], 'Nacelle frame', head);
  box([.19, .78, 2.1], white, [-.51, .05, -.38], 'Nacelle cutaway', head);
  box([1.13, .19, .4], white, [0, .42, -1.2], 'Nacelle roof section', head);
  function axial(radius, length, z, mat, label, parent = head) {
    const item = cylinder(radius, radius, length, mat, [0, .06, z], label, parent);
    item.rotation.x = Math.PI / 2;
    return item;
  }
  axial(.12, .72, .41, steel, 'Low-speed shaft');
  axial(.39, .43, .01, dark, 'Gearbox');
  axial(.34, .13, .23, steel, 'Gearbox inspection flange');
  axial(.105, .46, -.29, steel, 'High-speed shaft');
  axial(.39, .49, -.66, copper, 'Generator');
  for (const z of [-.91, -.78, -.64, -.5, -.41]) axial(.408, .036, z, steel, 'Generator cooling fins');
  const rotor = new T.Group();
  rotor.position.set(0, .06, .82);
  head.add(rotor);
  mesh(new T.SphereGeometry(.37, 28, 14), white, [0, 0, .12], 'Rotor hub', rotor);
  // Airfoil sections taper and twist along the blade span.
  const sections = [[.28,.26,.09,0],[.72,.43,.105,.06],[1.48,.34,.085,.09],[2.35,.21,.057,.14],[3.23,.068,.024,.20],[3.36,.012,.008,.21]];
  const positions = [], faces = [], sides = 12;
  for (const [r, chord, thickness, sweep] of sections) {
    const twist = .28 - r * .055;
    for (let i = 0; i < sides; i++) {
      const angle = i * 2 * Math.PI / sides;
      const x = Math.cos(angle) * chord / 2, z = Math.sin(angle) * thickness;
      positions.push(sweep + x * Math.cos(twist) - z * Math.sin(twist), r,
        x * Math.sin(twist) + z * Math.cos(twist));
    }
  }
  for (let ring = 0; ring < sections.length - 1; ring++) {
    for (let i = 0; i < sides; i++) {
      const a = ring * sides + i, b = ring * sides + (i + 1) % sides;
      faces.push(a,a + sides,b,b,a + sides,b + sides);
    }
  }
  const airfoil = new T.BufferGeometry();
  airfoil.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  airfoil.setIndex(faces);
  airfoil.computeVertexNormals();
  const blades = new T.Group();
  rotor.add(blades);
  if (showSweep) {
    const ring = new T.MeshBasicMaterial({ color: '#328baa', opacity: .65, transparent: true, side: T.DoubleSide, depthWrite: false });
    const disk = new T.MeshBasicMaterial({ color: '#66a8ba', opacity: .14, transparent: true, side: T.DoubleSide, depthWrite: false });
    mesh(new T.CircleGeometry(3.36, 64), disk, [0, 0, -.11], 'Rotor swept area', blades).castShadow = false;
    mesh(new T.RingGeometry(3.33, 3.37, 64), ring, [0, 0, -.10], 'Edge of swept area', blades).castShadow = false;
  }
  for (let i = 0; i < 3; i++) {
    const blade = mesh(airfoil, white, [0, 0, 0], 'Rotor blade', blades);
    blade.rotation.z = i * 2 * Math.PI / 3;
  }
  box([1.48, 1.05, .95], dark, [1.3, .63, .6], 'Step-up transformer');
  for (let i = 0; i < 8; i++) box([1.65, .052, 1.1], steel, [1.3, .33 + i * .097, .6], 'Transformer cooling fins');
  const pylonX = 5.15, pylonZ = -1.25;
  for (const side of [-1, 1]) {
    for (const depth of [-1, 1]) {
      bar([pylonX + side*.62,0,pylonZ + depth*.55],
        [pylonX + side*.15,5.6,pylonZ + depth*.12], .038, dark, 'Transmission pylon');
    }
    for (let y = 0; y < 5; y += .83) {
      const span = .62 - y*.083;
      bar([pylonX - side*span,y,pylonZ + .36], [pylonX + side*(span-.06),y+.83,pylonZ+.29], .026, steel, 'Pylon bracing');
    }
  }
  bar([pylonX-1.22,4.94,pylonZ], [pylonX+1.22,4.94,pylonZ], .05, dark, 'Transmission crossarm');
  const powerLines = [];
  const pulseGeometry = new T.SphereGeometry(.088, 10, 8);
  const gold = new T.MeshBasicMaterial({ color: '#ffc554' });
  function line(points, label, count = 6) {
    const curve = new T.CatmullRomCurve3(points.map(point => new T.Vector3(...point)), false, 'centripetal');
    mesh(new T.TubeGeometry(curve, 56, .033, 8, false), dark, [0,0,0], label);
    const pulses = Array.from({ length: count }, () => {
      const pulse = mesh(pulseGeometry, gold, [0,0,0]);
      pulse.castShadow = false;
      return pulse;
    });
    powerLines.push({ curve, pulses });
  }
  // Show the generator feeder where it exits the foundation.
  line([[towerX + .79,.19,.34],[towerX + 1.12,.14,.49],[-.26,.14,.58],[.56,.25,.6]],
    'Generator feeder to step-up transformer', 4);
  for (const offset of [-.86, 0, .86]) {
    cylinder(.055, .055, .32, steel, [1.3 + offset*.54, 1.28, .6], 'Transformer bushing');
    cylinder(.062, .062, .32, white, [pylonX + offset, 4.79, pylonZ], 'Line insulator');
    line([[1.3+offset*.54,1.44,.6],[3.25+offset*.6,2.5,-.25],[pylonX+offset,4.63,pylonZ],
      [6.65+offset*.52,4.12,-1.9],[8.2+offset*.34,4.4,-2.6]], 'Grid transmission conductor');
  }
  const air = new T.Group();
  scene.add(air);
  const airflow = Array.from({ length: 10 }, (_, i) => {
    const streak = box([.043,.042,.55], new T.MeshBasicMaterial({ color: i % 2 ? '#328baa' : '#66a8ba' }), [0,0,0], '', air);
    streak.castShadow = false;
    return streak;
  });
  let point, animate, phase = 0, airPhase = 0, lastTime = 0, dirty = true, visible = true, lastGeometry;
  function update(input, result, motion) {
    point = result;
    animate = motion;
    const key = [input.hub, input.radius, input.direction].join(':');
    if (key !== lastGeometry) {
      const height = 7.5 + (input.hub - 90) / 44;
      const radiusScale = showSweep ? input.radius / 60 : Math.min(1.24, Math.max(.64, input.radius / 58));
      const bearing = input.direction * RAD;
      column.scale.y = height / 7.5;
      column.position.y = height / 2 + .1;
      head.position.set(towerX,height+.35,0);
      head.rotation.y = bearing;
      blades.scale.setScalar(radiusScale);
      air.position.copy(head.position);
      air.rotation.y = bearing;
      lastGeometry = key;
    }
    host.dataset.windRegion = result.region;
    dirty = true;
  }
  const ray = new T.Raycaster();
  function inspect(event) {
    const rect = host.getBoundingClientRect();
    ray.setFromCamera(new T.Vector2(2*(event.clientX-rect.left)/rect.width-1, 1-2*(event.clientY-rect.top)/rect.height), camera);
    const hit = ray.intersectObjects(parts, false)[0];
    tooltip.hidden = !hit;
    if (hit) {
      tooltip.textContent = hit.object.userData.label;
      tooltip.style.left = Math.max(6, Math.min(rect.width-180,event.clientX-rect.left+9)) + 'px';
      tooltip.style.top = Math.max(5,event.clientY-rect.top-40) + 'px';
    }
  }
  const hide = () => { tooltip.hidden = true; };
  host.addEventListener('pointermove', inspect);
  host.addEventListener('pointerdown', inspect);
  host.addEventListener('pointerleave', hide);
  orbit.addEventListener('change', () => { dirty = true; });
  const resize = new ResizeObserver(() => {
    const { clientWidth: width, clientHeight: height } = host;
    if (!width || !height) return;
    const half = Math.max(7.8, 10 * height / width);
    Object.assign(camera, { left: -half*width/height, right: half*width/height, top: half, bottom: -half });
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    dirty = true;
  });
  const visibility = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; dirty = true; });
  resize.observe(host);
  visibility.observe(host);
  renderer.setAnimationLoop(time => {
    const dt = Math.min(.05, (time-lastTime)/1000);
    lastTime = time;
    if (!point || !visible || (!animate && !dirty)) return;
    if (animate) {
      rotor.rotation.z = (rotor.rotation.z - dt * point.omega) % (2*Math.PI);
      phase = (phase + dt * Math.min(.9, point.gridMW * .3)) % 1;
      airPhase = (airPhase + dt * point.speed / 18) % 1;
    }
    powerLines.forEach(({ curve, pulses }) => pulses.forEach((pulse,i) => {
      pulse.visible = point.gridMW > 0;
      pulse.position.copy(curve.getPointAt((i/pulses.length + phase)%1));
    }));
    airflow.forEach((streak,i) => {
      streak.visible = point.speed > 0;
      streak.position.set((i%5-2)*.9, (i%3-1)*.8, 4.5 - 8.7*((i/10+airPhase)%1));
    });
    host.dataset.rotorAngle = String(rotor.rotation.z);
    renderer.render(scene,camera);
    dirty = false;
  });
  return { update, destroy() {
    renderer.setAnimationLoop(null);
    resize.disconnect();
    visibility.disconnect();
    orbit.dispose();
    host.removeEventListener('pointermove',inspect);
    host.removeEventListener('pointerdown',inspect);
    host.removeEventListener('pointerleave',hide);
    const geometries = new Set(), materials = new Set();
    scene.traverse(item => { if(item.geometry) geometries.add(item.geometry); if(item.material) materials.add(item.material); });
    geometries.forEach(geometry => geometry.dispose());
    materials.forEach(mat => mat.dispose());
    renderer.dispose();
    renderer.domElement.remove();
  } };
}

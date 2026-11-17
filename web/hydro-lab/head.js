import * as T from 'three';
import { computeHead } from './head-model.js';

const sliders = [
  ['elevation', 'Elevation', 'z', 'm', 0, 50, 1, 15],
  ['pressureKPa', 'Gauge pressure', 'p', 'kPa', 0, 300, 1, 100],
  ['velocity', 'Water speed', 'v', 'm/s', 0, 20, 0.1, 5],
  ['density', 'Water density', 'ρ', 'kg/m³', 950, 1050, 1, 1000],
  ['gravity', 'Gravity', 'g', 'm/s²', 9.75, 9.85, 0.01, 9.81],
];

const fields = sliders.map(([key, label, symbol, unit, min, max, step, value]) => {
  const wrapper = document.createElement('div');
  wrapper.className = 'control';
  wrapper.innerHTML = `<label for="${key}"><span>${label} <i>${symbol}</i></span><output for="${key}"></output></label>
    <input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" aria-label="${label} (${unit})">`;
  document.querySelector('#controls').append(wrapper);
  return { key, unit, input: wrapper.querySelector('input'), readout: wrapper.querySelector('output') };
});

const host = document.querySelector('#scene');
let draw;
try {
  draw = createScene(host);
} catch {
  document.querySelector('#fallback').hidden = false;
}

function update() {
  const inputs = Object.fromEntries(fields.map(({ key, unit, input, readout }) => {
    readout.value = `${input.value} ${unit}`;
    return [key, input.valueAsNumber];
  }));
  const head = computeHead(inputs);
  const metres = value => `${value.toFixed(2)} m`;
  document.querySelector('#head').value = `H = ${metres(head.total)}`;
  document.querySelector('#elevation-head').value = metres(head.elevation);
  document.querySelector('#pressure-head').value = metres(head.pressure);
  document.querySelector('#velocity-head').value = metres(head.kinetic);
  draw?.(head, inputs.velocity);
}
fields.forEach(({ input }) => input.addEventListener('input', update));
update();

function createScene(container) {
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.append(renderer.domElement);
  const scene = new T.Scene();
  scene.background = new T.Color('#e9f3f4');
  const camera = new T.OrthographicCamera();
  camera.position.set(7, 7, 13);
  camera.lookAt(0, 2.5, 0);
  scene.add(new T.HemisphereLight(0xffffff, 0x5c7a79, 2.3));
  const key = new T.DirectionalLight(0xffffff, 2.2);
  key.position.set(-4, 8, 4);
  scene.add(key);
  const steel = new T.MeshStandardMaterial({ color: '#49666f', roughness: .55 });
  const blue = new T.MeshStandardMaterial({ color: '#247f9d', roughness: .43 });
  const datum = new T.MeshStandardMaterial({ color: '#9cb7ae' });
  const glass = new T.MeshStandardMaterial({ color: '#8cccdb', transparent: true, opacity: .28, depthWrite: false });
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  function mesh(geometry, material, parent, x = 0, y = 0, z = 0) {
    const shape = new T.Mesh(geometry, material);
    shape.position.set(x, y, z);
    parent.add(shape);
    return shape;
  }
  mesh(new T.BoxGeometry(8.4, .12, 3.15), datum, scene, 0, -.06);
  const pipe = new T.Group();
  scene.add(pipe);
  // Open the front of the pipe so flow markers are visible in the cutaway.
  const cutaway = steel.clone();
  cutaway.side = T.DoubleSide;
  const shell = mesh(new T.CylinderGeometry(.38, .38, 7.5, 28, 1, true, Math.PI / 2, Math.PI), cutaway, pipe);
  shell.rotation.z = Math.PI / 2;
  mesh(new T.BoxGeometry(7.48, .11, .53), blue, pipe, 0, -.16);
  for (const x of [-3.65, 3.65]) {
    const flange = mesh(new T.TorusGeometry(.38, .065, 10, 30), steel, pipe, x);
    flange.rotation.y = Math.PI / 2;
  }
  const tapX = -.4;
  mesh(new T.CylinderGeometry(.15, .15, .47, 16), steel, pipe, tapX, .18);
  const water = mesh(new T.CylinderGeometry(.12, .12, 1, 18), blue, pipe, tapX);
  const tube = mesh(new T.CylinderGeometry(.152, .152, 2.1, 18, 1, true), glass, pipe, tapX, 1.39);
  tube.renderOrder = 2;
  const guide = mesh(new T.BoxGeometry(.065, 1, .065), datum, scene, -3.95);
  const particleShape = new T.SphereGeometry(.071, 10, 8);
  const particleMaterial = new T.MeshBasicMaterial({ color: '#dcf8ff' });
  const particles = Array.from({ length: 10 }, () => mesh(particleShape, particleMaterial, pipe, 0, -.07, .24));

  let speed = 0, phase = 0, frame = 0, previous = 0, until = 0;
  function render() {
    particles.forEach((particle, index) => {
      particle.position.x = -3.55 + 7.1 * ((phase + index / particles.length) % 1);
      particle.visible = speed > 0;
    });
    renderer.render(scene, camera);
  }
  function tick(time) {
    frame = 0;
    if (previous) phase = (phase + Math.min(.04, (time - previous) / 1000) * speed * .065) % 1;
    previous = time;
    render();
    if (time < until && !motion.matches && speed > 0) frame = requestAnimationFrame(tick);
  }
  function resize() {
    const { clientWidth: width, clientHeight: height } = container;
    if (!width || !height) return;
    const half = Math.max(3.5, 5.25 * height / width);
    Object.assign(camera, { left: -half * width / height, right: half * width / height, top: half, bottom: -half, near: .1, far: 80 });
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    render();
  }
  new ResizeObserver(resize).observe(container);
  return (head, velocity) => {
    const height = head.elevation * .058;
    const pressureHeight = head.pressure * .058;
    pipe.position.y = .44 + height;
    guide.scale.y = height;
    guide.position.y = height / 2;
    water.visible = pressureHeight > 0;
    water.scale.y = pressureHeight;
    water.position.y = .35 + pressureHeight / 2;
    speed = velocity;
    render();
    if (!motion.matches && speed > 0) {
      until = performance.now() + 2400;
      if (!frame) {
        previous = 0;
        frame = requestAnimationFrame(tick);
      }
    }
  };
}

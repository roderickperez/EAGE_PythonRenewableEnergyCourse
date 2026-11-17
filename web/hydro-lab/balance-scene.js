import * as T from 'three';
import { flowBalance, MAX_FLOW_M3S, MAX_INTERVAL_MINUTES } from './balance-model.js';

const palette = { water: '#288daa', net: '#398470', loss: '#c17932' };
export { palette };

export function mountBalanceScene(host, kind) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e9f3f4');
  const camera = new T.OrthographicCamera();
  camera.position.set(7, 6, 14);
  camera.lookAt(0, 2, 0);
  scene.add(new T.HemisphereLight(0xffffff, 0x6e8a82, 2.4));
  const light = new T.DirectionalLight(0xffffff, 2.3);
  light.position.set(-5, 7, 4);
  scene.add(light);
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.append(renderer.domElement);
  const material = color => new T.MeshStandardMaterial({ color, roughness: .65 });
  const water = material(palette.water);
  const rock = material('#a7bcb2');
  const steel = material('#49666f');

  function mesh(geometry, surface, x, y, z) {
    const object = new T.Mesh(geometry, surface);
    object.position.set(x, y, z);
    scene.add(object);
    return object;
  }
  const box = (width, height, depth, surface, x, y, z) =>
    mesh(new T.BoxGeometry(width, height, depth), surface, x, y, z);
  function tank(x, width, height, depth) {
    box(width + .16, .14, depth + .16, rock, x, -.07, 0);
    box(width, height, .10, rock, x, height / 2, -depth / 2);
    for (const side of [-1, 1]) {
      box(.10, height, depth, rock, x + side * width / 2, height / 2, 0);
    }
    return box(width - .10, 1, depth - .09, water, x, 0, .01);
  }
  function level(object, height, bottom = 0) {
    object.visible = height > 0;
    object.position.y = bottom + height / 2;
    object.scale.y = height;
  }

  let draw;
  if (kind === 'net-head') {
    const upper = tank(-2.75, 3, 3.75, 2.4);
    const lower = tank(2.75, 3, 1.05, 2.4);
    const lowerLevel = .63;
    level(lower, lowerLevel);
    const net = box(.40, 1, .28, material(palette.net), .32, 0, 1.55);
    const loss = box(.40, 1, .28, material(palette.loss), .32, 0, 1.55);
    box(.84, .045, .36, steel, .32, lowerLevel, 1.55);
    box(.07, 3, .10, rock, .25, lowerLevel + 1.5, 1.25);
    draw = ([gross, lossHead]) => {
      const usable = (gross - lossHead) * .03;
      level(upper, lowerLevel + gross * .03);
      level(net, usable, lowerLevel);
      level(loss, lossHead * .03, lowerLevel + usable);
    };
  } else {
    const capacity = flowBalance(MAX_FLOW_M3S, MAX_INTERVAL_MINUTES, 1000).volume;
    const receiving = tank(2.1, 3.9, 2.9, 2.8);
    const path = new T.CatmullRomCurve3([
      new T.Vector3(-4.05, 3.28, .45), new T.Vector3(-.75, 3.28, .45),
      new T.Vector3(.55, 3.28, .45), new T.Vector3(.85, 3.04, .45),
      new T.Vector3(.85, .16, .45),
    ]);
    box(4.85, .13, .64, steel, -1.61, 3.15, .45);
    const pearl = new T.SphereGeometry(.082, 10, 8);
    const stream = material('#42a6bf');
    const particles = Array.from({ length: 18 }, () => mesh(pearl, stream, 0, 0, 0));
    draw = ([volume, flow], phase) => {
      const height = volume / capacity * 2.65;
      level(receiving, height);
      for (const [index, particle] of particles.entries()) {
        particle.position.copy(path.getPointAt((phase + index / particles.length) % 1));
        particle.visible = flow > 0 && particle.position.y > height;
      }
    };
  }

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let current, start, target, startTime, frame, phase = 0, previousTime = 0;
  function render() {
    if (current) draw(current, phase);
    renderer.render(scene, camera);
  }
  function tick(time) {
    frame = undefined;
    const progress = Math.min(1, (time - startTime) / 500);
    current = target.map((value, index) => start[index] + (value - start[index]) * (1 - (1 - progress) ** 3));
    if (kind === 'flow-volume') phase = (phase + (time - previousTime) * current[1] * .00011) % 1;
    previousTime = time;
    render();
    if (progress < 1 || (kind === 'flow-volume' && target[1] > 0 && time - startTime < 2200)) {
      frame = requestAnimationFrame(tick);
    }
  }
  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = host;
    if (!w || !h) return;
    const half = Math.max(2.85, 5.2 * h / w);
    Object.assign(camera, { left: -half * w / h, right: half * w / h, top: half, bottom: -half });
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    render();
  }).observe(host);
  return values => {
    if (target?.every((value, index) => value === values[index])) return;
    if (frame) cancelAnimationFrame(frame);
    target = values;
    if (!current || reducedMotion.matches) {
      current = values;
      render();
      return;
    }
    start = current;
    startTime = previousTime = performance.now();
    frame = requestAnimationFrame(tick);
  };
}
